"""
Two-model SPARC BTFR comparison panel using the modular JAX pipeline.

This driver mirrors btfr_pipeline.py but evaluates two (alpha, scatter, x, nu)
model points and assembles a 2x2 comparison panel:

    [0, 0] BTFR, model 1            [0, 1] BTFR, model 2
    [1, 0] SHMR, both models        [1, 1] per-galaxy delta log-likelihood

Run, for example, with:
    mpiexec -n 4 python btfr_panel.py
"""

import gc
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from numpy.lib import recfunctions as rfn
from mpi4py import MPI
import matplotlib.pyplot as plt
from matplotlib import rcParams

from BAM import AbundanceMatch, proxies
from btfr.utils import *

from btfr.forward_model import (
    compute_simulated_diagnostics,
    get_loglike_split,
    mpi_weighted_average,
)


@dataclass
class ModelPointConfig:
    """One (alpha, scatter, x, nu) BTFR model point."""

    alpha_proxy: float = 0.5
    scatter: float = 0.1
    x: float = 0.0
    nu: float = 0.0

    def __post_init__(self):
        if self.scatter < 0:
            raise ValueError("scatter must be non-negative")
        if not 0.0 <= self.x < 1.0:
            raise ValueError("x must satisfy 0 <= x < 1")

    @property
    def alpha(self) -> float:
        return float(np.tan(self.alpha_proxy))

    def tag(self) -> str:
        return (
            f"alpha_{self.alpha:.3f}_scatter_{self.scatter:.3f}_"
            f"x_{self.x:.3f}_nu_{self.nu:.3f}"
        )

    def label(self) -> str:
        return (
            rf"$\alpha={self.alpha:.2f}$, $\sigma={self.scatter:.2f}$, "
            rf"x={self.x:.2f}, $\nu={self.nu:.2f}$"
        )


@dataclass
class PanelConfig:
    """Configuration for the two-model BTFR comparison panel."""

    n_am_reals: int = 100
    n_stellar_reals: int = 1000

    model_1: ModelPointConfig = field(default_factory=ModelPointConfig)
    model_2: ModelPointConfig = field(default_factory=ModelPointConfig)

    base_seed: int = 42
    vmax_shift_mode: bool = False

    use_tex: bool = True
    output_dir: str = "."
    plot_filename: Optional[str] = None

    # Save per-model result dictionaries (model_1_results.npy /
    # model_2_results.npy) in the format expected by paper_btfr_panel.py.
    save_results: bool = True
    results_dir: Optional[str] = None  # defaults to output_dir/btfr_panel_results

    def __post_init__(self):
        if self.n_am_reals <= 0 or self.n_stellar_reals <= 0:
            raise ValueError("Numbers of realizations must be positive")
        if self.results_dir is None:
            self.results_dir = str(Path(self.output_dir) / "btfr_panel_results")


class BtfrPanelPipeline:
    """Evaluate two BTFR model points and produce the comparison panel."""

    def __init__(self, config: PanelConfig):
        self.config = config
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Master PRNG key lives with the driver; per-model keys are split
        # from it identically on every rank, and per-rank streams are
        # derived inside the likelihood module via fold_in.
        self.master_key = random.PRNGKey(config.base_seed)

        self.contra_grids, self.grid_axes = load_emulators()

    # ------------------------------------------------------------------
    # Shared building blocks (identical to the grid / pipeline drivers)
    # ------------------------------------------------------------------

    def _select_halos(self, halo_catalog: np.ndarray, x: float) -> np.ndarray:
        """Apply the same x-dependent halo selection as the grid scripts."""
        slope, intercept = get_x_cutoff_fit(halo_catalog, x)
        halos_selected = rfn.append_fields(
            halo_catalog.copy(),
            "select",
            np.ones(halo_catalog.shape[0], dtype=float),
            usemask=False,
        )

        halo_log_mvir = np.log10(halos_selected["Mvir"])
        halo_log_vmax = np.log10(halos_selected["vmax"])
        cutoff = slope * halo_log_mvir + intercept
        halos_selected["select"][halo_log_vmax > cutoff] = np.nan
        return halos_selected

    def _setup_contra_interpolator(self, nu: float):
        """Construct the same JAX contra interpolator used by the grids."""
        if nu == 0.0:
            return None
        if nu not in self.contra_grids:
            # Raise on ALL ranks so the communicator can't deadlock with
            # rank 0 dead and the others waiting in a collective call.
            raise ValueError(
                f"Contra grid for nu={nu} not found. "
                f"Available grids: {list(self.contra_grids.keys())}"
            )

        grid_jax = jnp.asarray(self.contra_grids[nu], dtype=jnp.float32)
        grid_axes = self.grid_axes

        def contra_interpolator(points):
            return jax_contra_interpolator(grid_jax, points, grid_axes)

        return contra_interpolator

    @staticmethod
    def _extract_galaxy_data(
            sparc_galaxy_names, galaxy_properties: Dict,
            bulge_luminosities: Dict) -> Dict[str, np.ndarray]:
        """Extract intrinsic galaxy properties for the shared forward model."""
        galaxy_data = {
            "L36": np.asarray([
                galaxy_properties[name]["Total Luminosity at [3.6]"]
                for name in sparc_galaxy_names
            ]),
            "L36_err": np.asarray([
                galaxy_properties[name]["Luminosity Error"]
                for name in sparc_galaxy_names
            ]),
            "Reff": np.asarray([
                galaxy_properties[name]["Effective Radius at [3.6]"]
                for name in sparc_galaxy_names
            ]),
            "MH1": np.asarray([
                galaxy_properties[name]["Total HI mass"]
                for name in sparc_galaxy_names
            ]),
            "d": np.asarray([
                galaxy_properties[name]["Distance"]
                for name in sparc_galaxy_names
            ]),
            "d_err": np.asarray([
                galaxy_properties[name]["Distance Error"]
                for name in sparc_galaxy_names
            ]),
            "Lbulge": np.asarray([
                bulge_luminosities[name] for name in sparc_galaxy_names
            ]),
        }
        galaxy_data["MH1_err"] = 0.1 * galaxy_data["MH1"]
        return galaxy_data

    @staticmethod
    def _preprocess_data_for_jax(galaxy_data, mass_model_data, halo_catalog):
        galaxy_data_jax = {
            key: jnp.asarray(value, dtype=jnp.float32)
            for key, value in galaxy_data.items()
        }
        mass_model_data_jax = {
            key: jnp.asarray(value, dtype=jnp.float32)
            for key, value in mass_model_data.items()
        }
        halo_catalog_data = {
            field_name: jnp.asarray(halo_catalog[field_name])
            for field_name in halo_catalog.dtype.names
        }
        return galaxy_data_jax, mass_model_data_jax, halo_catalog_data

    def _velocity_shift(self, local_log_vmax, log_vobs) -> float:
        """Reproduce the legacy global mean-log-velocity shift using MPI."""
        if not self.config.vmax_shift_mode:
            return 0.0

        local_sum = float(np.nansum(local_log_vmax))
        local_count = int(np.count_nonzero(np.isfinite(local_log_vmax)))
        global_sum = self.comm.allreduce(local_sum, op=MPI.SUM)
        global_count = self.comm.allreduce(local_count, op=MPI.SUM)
        if global_count == 0:
            raise RuntimeError("No finite simulated velocities are available")

        mean_model = global_sum / global_count
        mean_observed = float(np.mean(log_vobs))
        return mean_observed - mean_model

    def _gather_predictions(self, local_predictions):
        """Gather variable-sized local sample blocks safely onto rank 0."""
        gathered = self.comm.gather(local_predictions, root=0)
        if self.rank != 0:
            return None

        return {
            name: np.concatenate([block[name] for block in gathered], axis=1)
            for name in local_predictions
        }

    @staticmethod
    def _summary_statistics(global_predictions, velocity_shift):
        with np.errstate(divide="ignore", invalid="ignore"):
            log_vmax = np.log10(global_predictions["Vmax"]) + velocity_shift
            log_mbar = np.log10(global_predictions["Mbar"])

        return {
            "log_vmax_samples": log_vmax,
            "log_mbar_samples": log_mbar,
            "log_vmax_mean": np.nanmean(log_vmax, axis=1),
            "log_vmax_std": np.nanstd(log_vmax, axis=1),
            "log_mbar_mean": np.nanmean(log_mbar, axis=1),
            "log_mbar_std": np.nanstd(log_mbar, axis=1),
        }

    # ------------------------------------------------------------------
    # Panel-specific pieces
    # ------------------------------------------------------------------

    def _shmr_catalog(self, model: ModelPointConfig, abundance_match,
                      halos_selected):
        """
        Draw one scattered abundance-matching catalog for the SHMR panel
        (rank 0 only; this is a plotting diagnostic, not part of the
        likelihood).
        """
        if self.rank != 0:
            return None, None

        theta = {"alpha": model.alpha, "scatter": model.scatter}
        deconv = abundance_match.deconvoluted_catalogs(theta, halos_selected)
        mask, catalog_Mstar = abundance_match.add_scatter(
            deconv, cut_range=(3, 12), return_catalog=True
        )
        halo_proxy = np.log10(halos_selected["Mvir"])[mask]
        return halo_proxy, catalog_Mstar

    def _evaluate_model(self, model: ModelPointConfig, abundance_match,
                        galaxy_data, mass_model_data, halo_catalog, observed):
        """
        Run the forward model for one model point.

        Returns a results dict on every rank; the summary / SHMR entries are
        only populated on rank 0, while the (per-galaxy) log-likelihoods are
        identical on all ranks thanks to the Allreduce inside
        mpi_weighted_average.
        """
        halos_selected = self._select_halos(halo_catalog, model.x)
        galaxy_data_jax, mass_model_data_jax, halo_catalog_data = (
            self._preprocess_data_for_jax(
                galaxy_data, mass_model_data, halos_selected
            )
        )
        contra_interpolator = self._setup_contra_interpolator(model.nu)

        self.master_key, model_key = random.split(self.master_key)
        local_predictions = compute_simulated_diagnostics(
            model_key,
            model.alpha,
            model.scatter,
            model.nu,
            abundance_match,
            contra_interpolator,
            galaxy_data_jax,
            mass_model_data_jax,
            halos_selected,
            halo_catalog_data,
            self.config.n_am_reals,
            self.config.n_stellar_reals,
            self.size,
            self.rank,
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            local_log_vmax = np.log10(local_predictions["Vmax"])
        velocity_shift = self._velocity_shift(
            local_log_vmax, observed["log_vobs"]
        )
        local_log_vmax = local_log_vmax + velocity_shift

        # Per-galaxy log-likelihoods with the same count-weighted MPI
        # reduction used by evaluate_likelihoods. We need the individual
        # values here (not just their sum) for the delta-loglike panel.
        log_avg_local, counts_local = get_loglike_split(
            local_log_vmax, observed["log_vobs"], observed["log_vobs_err"]
        )
        individual_log_likelihoods = mpi_weighted_average(
            log_avg_local, counts_local, self.comm
        )
        log_likelihood = float(np.sum(individual_log_likelihoods))

        global_predictions = self._gather_predictions(local_predictions)
        shmr_halo_proxy, shmr_Mstar = self._shmr_catalog(
            model, abundance_match, halos_selected
        )

        # Free the heavy per-model objects before the next model point.
        del halos_selected, halo_catalog_data, local_predictions
        gc.collect()
        jax.clear_caches()

        results = {
            "log_likelihood": log_likelihood,
            "individual_log_likelihoods": individual_log_likelihoods,
            "velocity_shift": velocity_shift,
            "summary": None,
            "shmr_halo_proxy": shmr_halo_proxy,
            "shmr_Mstar": shmr_Mstar,
        }
        if self.rank == 0:
            results["summary"] = self._summary_statistics(
                global_predictions, velocity_shift
            )
        return results

    def _print_computation_info(self):
        if self.rank != 0:
            return
        print(f"Number of processes: {self.size}")
        print(f"Number of AM realizations: {self.config.n_am_reals}")
        print(f"Number of stellar realizations: {self.config.n_stellar_reals}")
        for index, model in enumerate(
                (self.config.model_1, self.config.model_2), start=1):
            print(
                f"Model {index}: alpha_proxy={model.alpha_proxy:.6g}, "
                f"alpha={model.alpha:.6g}, scatter={model.scatter:.6g}, "
                f"x={model.x:.6g}, nu={model.nu:.6g}"
            )
        print(
            f"Vmax shift mode: "
            f"{'ON' if self.config.vmax_shift_mode else 'OFF'}"
        )

    def _save_model_results(self, index: int, model: ModelPointConfig,
                            results, observed) -> Optional[Path]:
        """
        Save one model's outputs as model_{index}_results.npy in the exact
        dictionary format consumed by paper_btfr_panel.py (rank 0 only).
        """
        if self.rank != 0 or not self.config.save_results:
            return None

        summary = results["summary"]
        V_mock = summary["log_vmax_mean"]
        V_mock_err = summary["log_vmax_std"]

        # Legacy normalised residuals, as in the original panel script.
        sigmas = np.sqrt(V_mock_err**2 + observed["log_vobs_err"]**2)
        residuals = (observed["log_vobs"] - V_mock) / sigmas

        model_results = {
            "V_mock": V_mock,
            "V_mock_err": V_mock_err,
            "M_mock": summary["log_mbar_mean"],
            "M_mock_err": summary["log_mbar_std"],
            "V_obs": observed["log_vobs"],
            "V_obs_err": observed["log_vobs_err"],
            "M_obs": observed["log_mobs"],
            "M_obs_err": observed["log_mobs_err"],
            "halo_proxy": results["shmr_halo_proxy"],
            "catalog": results["shmr_Mstar"],
            "residuals": residuals,
            "log_likelihood": results["log_likelihood"],
            "log_likelihoods": results["individual_log_likelihoods"],
            "params": {
                "alpha": model.alpha,
                "alpha_proxy": model.alpha_proxy,
                "scatter": model.scatter,
                "x": model.x,
                "nu": model.nu,
            },
            # Extra provenance (ignored by the paper plotter).
            "velocity_shift": results["velocity_shift"],
            "vmax_shift_mode": self.config.vmax_shift_mode,
            "n_am_reals": self.config.n_am_reals,
            "n_stellar_reals": self.config.n_stellar_reals,
            "base_seed": self.config.base_seed,
        }

        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / f"model_{index}_results.npy"
        np.save(output_path, model_results, allow_pickle=True)
        return output_path

    def _plot_path(self) -> Path:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.config.plot_filename is not None:
            return output_dir / self.config.plot_filename

        base_name = (
            f"btfr_panel_{self.config.model_1.tag()}"
            f"__vs__{self.config.model_2.tag()}"
        )
        if self.config.vmax_shift_mode:
            base_name += "_vmaxshift"
        return output_dir / f"{base_name}.png"

    def _plot_panel(self, results_1, results_2, observed) -> Path:
        """Assemble the 2x2 comparison panel (rank 0 only)."""
        rcParams["font.family"] = "serif"
        rcParams["font.serif"] = ["Computer Modern"]
        rcParams["text.usetex"] = self.config.use_tex

        fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

        # BTFR panels, one per model (utils.btfr_plot draws onto the
        # provided axis and leaves saving to us).
        for ax, model, results in (
                (axs[0, 0], self.config.model_1, results_1),
                (axs[0, 1], self.config.model_2, results_2)):
            summary = results["summary"]
            btfr_plot(
                alpha=model.alpha,
                sigma=model.scatter,
                x=model.x,
                nu=model.nu,
                vmaxshift=self.config.vmax_shift_mode,
                loglike=results["log_likelihood"],
                xsim=summary["log_vmax_mean"],
                ysim=summary["log_mbar_mean"],
                xsimerr=summary["log_vmax_std"],
                ysimerr=summary["log_mbar_std"],
                xobs=observed["log_vobs"],
                yobs=observed["log_mobs"],
                xobserr=observed["log_vobs_err"],
                yobserr=observed["log_mobs_err"],
                plotname=None,
                ax=ax,
            )

        # SHMR panel: both scattered AM catalogs on the same axis.
        for model, results, color in (
                (self.config.model_1, results_1, "purple"),
                (self.config.model_2, results_2, "orange")):
            plot_SHMR_with_contours_quantile(
                axs[1, 0],
                results["shmr_halo_proxy"],
                results["shmr_Mstar"],
                color=color,
                label=model.label(),
            )
        axs[1, 0].set_xlabel(r"$\log_{10}(M_h \, [\mathrm{M}_\odot])$",
                             fontsize=12)
        axs[1, 0].set_ylabel(r"$\log_{10}(M_* \, [\mathrm{M}_\odot])$",
                             fontsize=12)
        axs[1, 0].set_title("Stellar-to-Halo Mass Relations", fontsize=16)
        axs[1, 0].set_xlim([10, 15])
        axs[1, 0].set_ylim([7, 12])
        axs[1, 0].legend()

        # Per-galaxy delta log-likelihood (model 2 - model 1) against the
        # model-1 mean baryonic mass.
        delta_loglike = (results_2["individual_log_likelihoods"]
                         - results_1["individual_log_likelihoods"])
        delta_loglike_correlation_plot(
            results_1["summary"]["log_mbar_mean"],
            delta_loglike,
            r"$\log_{10}(M_{\rm bar} \, [\mathrm{M}_\odot])$",
            plotname=None,
            ax=axs[1, 1],
        )
        axs[1, 1].set_title(
            r"$\Delta \log \mathcal{L}$ (model 2 $-$ model 1) vs "
            r"$M_{\rm bar}$",
            fontsize=16,
        )
        axs[1, 1].legend()

        plt.tight_layout()
        plot_path = self._plot_path()
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
        return plot_path

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    def run(self):
        """Run both model points and produce the comparison panel."""
        try:
            (
                sparc_galaxy_names,
                bulge_luminosities,
                galaxy_properties,
                mass_model_data,
                sparc_btfr_data,
                stellar_mass_bins,
                stellar_mass_function,
                halo_catalog,
            ) = load_data()

            mass_model_data = vectorize_mass_model_table(
                sparc_galaxy_names, mass_model_data
            )
            galaxy_data = self._extract_galaxy_data(
                sparc_galaxy_names, galaxy_properties, bulge_luminosities
            )

            vobs = np.asarray(
                sparc_btfr_data["Vmax"], dtype=np.float64
            ).reshape(-1)
            vobs_err = np.asarray(
                sparc_btfr_data["e_Vmax"], dtype=np.float64
            ).reshape(-1)
            observed = {
                "log_vobs": np.log10(vobs),
                "log_vobs_err": vobs_err / (vobs * np.log(10.0)),
                "log_mobs": np.asarray(
                    sparc_btfr_data["log(Mb)"], dtype=np.float64
                ).reshape(-1),
                "log_mobs_err": np.asarray(
                    sparc_btfr_data["e_log(Mb)"], dtype=np.float64
                ).reshape(-1),
            }

            proxy = proxies["mvir_proxy"](use_cache=False)
            abundance_match = AbundanceMatch(
                stellar_mass_bins[10:],
                stellar_mass_function[10:],
                halo_proxy=proxy,
                ext_range=(3.0, 12.0),
                boxsize=140,
                faint_end_first=True,
                scatter_mult=1,
                faint_end_slope=-0.42,
            )

            self._print_computation_info()

            results_1 = self._evaluate_model(
                self.config.model_1, abundance_match, galaxy_data,
                mass_model_data, halo_catalog, observed
            )
            results_2 = self._evaluate_model(
                self.config.model_2, abundance_match, galaxy_data,
                mass_model_data, halo_catalog, observed
            )

            if self.rank != 0:
                return None

            print("\nTwo-model abundance-matching panel pipeline finished")
            for index, results in enumerate((results_1, results_2), start=1):
                print(f"Model {index} log likelihood: "
                      f"{results['log_likelihood']:.8f}")
                if self.config.vmax_shift_mode:
                    print(f"Model {index} applied log10(Vmax) shift: "
                          f"{results['velocity_shift']:.8f}")

            # Save per-model dictionaries for paper_btfr_panel.py.
            results_paths = [
                self._save_model_results(1, self.config.model_1,
                                         results_1, observed),
                self._save_model_results(2, self.config.model_2,
                                         results_2, observed),
            ]
            for path in results_paths:
                if path is not None:
                    print(f"Model results saved to: {path}")

            plot_path = self._plot_panel(results_1, results_2, observed)
            print(f"BTFR panel plot saved to: {plot_path}")

            gc.collect()
            jax.clear_caches()
            return {
                "model_1": results_1,
                "model_2": results_2,
                "results_paths": results_paths,
                "plot_path": plot_path,
            }

        except Exception as exc:
            if self.rank == 0:
                print(f"Fatal error in BTFR panel pipeline: {exc}")
                traceback.print_exc()
            raise


def main():
    config = PanelConfig(
        n_am_reals=100,
        n_stellar_reals=1000,
        model_1=ModelPointConfig(
            alpha_proxy=0.5,
            scatter=0.1,
            x=0.0,
            nu=float(np.linspace(-3.0, 3.0, 20)[11]),
        ),
        model_2=ModelPointConfig(
            alpha_proxy=0.5,
            scatter=0.1,
            x=0.0,
            nu=float(np.linspace(-3.0, 3.0, 20)[15]),
        ),
        base_seed=42,
        vmax_shift_mode=False,
        use_tex=True,
        output_dir="/Users/fedorboreiko/Documents/Oxford/btfr_z/plots",
        save_results=True,
        # Matches the hard-coded results_dir in paper_btfr_panel.py.
        results_dir="/Users/fedorboreiko/Documents/Oxford/btfr_z/btfr_panel_results",
    )

    pipeline = BtfrPanelPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()