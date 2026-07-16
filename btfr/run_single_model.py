##### mpiexec -n 4 python btfr/btfr_contra.py

"""
Single-model SPARC BTFR prediction and plotting pipeline.

This driver mirrors the modular JAX likelihood-grid scripts but evaluates a
single (alpha, scatter, x, nu) model point and retains the diagnostics needed
for the BTFR and optional per-galaxy plots.

Run with:
    mpiexec -n 4 python btfr_pipeline.py
"""

import gc
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from numpy.lib import recfunctions as rfn
from mpi4py import MPI
from matplotlib import rcParams

from BAM import AbundanceMatch, proxies
from utils import *
from btfr.forward_model import (
    compute_simulated_diagnostics,
    evaluate_likelihoods,
)


@dataclass
class SingleModelConfig:
    """Configuration for one BTFR model prediction."""

    n_am_reals: int = 100
    n_stellar_reals: int = 1000

    alpha_proxy: float = 0.5
    scatter: float = 0.1
    x: float = 0.0
    nu: float = float(np.linspace(-3.0, 3.0, 20)[11])

    base_seed: int = 42
    vmax_shift_mode: bool = False

    plot_individual_histograms: bool = False
    save_samples: bool = False
    use_tex: bool = True
    output_dir: str = "."
    plot_filename: Optional[str] = None

    def __post_init__(self):
        if self.n_am_reals <= 0 or self.n_stellar_reals <= 0:
            raise ValueError("Numbers of realizations must be positive")
        if self.scatter < 0:
            raise ValueError("scatter must be non-negative")
        if not 0.0 <= self.x < 1.0:
            raise ValueError("x must satisfy 0 <= x < 1")

    @property
    def alpha(self) -> float:
        return float(np.tan(self.alpha_proxy))


class SingleModelBtfrPipeline:
    """Generate and plot the prediction for one BTFR model point."""

    def __init__(self, config: SingleModelConfig):
        self.config = config
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.master_key = random.PRNGKey(config.base_seed)

        self.contra_grids, self.grid_axes = load_emulators()

    def _select_halos(self, halo_catalog: np.ndarray) -> np.ndarray:
        """Apply the same x-dependent halo selection as the grid scripts."""
        slope, intercept = get_x_cutoff_fit(halo_catalog, self.config.x)
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

    def _setup_contra_interpolator(self):
        """Construct the same JAX contra interpolator used by the grids."""
        nu = self.config.nu
        if nu == 0.0:
            return None
        if nu not in self.contra_grids:
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
            field: jnp.asarray(halo_catalog[field])
            for field in halo_catalog.dtype.names
        }
        return galaxy_data_jax, mass_model_data_jax, halo_catalog_data

    def _print_computation_info(self):
        if self.rank != 0:
            return
        print(f"Number of processes: {self.size}")
        print(f"Number of AM realizations: {self.config.n_am_reals}")
        print(f"Number of stellar realizations: {self.config.n_stellar_reals}")
        print(
            "Model parameters: "
            f"alpha_proxy={self.config.alpha_proxy:.6g}, "
            f"alpha={self.config.alpha:.6g}, "
            f"scatter={self.config.scatter:.6g}, "
            f"x={self.config.x:.6g}, nu={self.config.nu:.6g}"
        )
        print(
            f"Vmax shift mode: "
            f"{'ON' if self.config.vmax_shift_mode else 'OFF'}"
        )

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
            log_vdm = np.log10(global_predictions["Vdm_max"])
            log_vbar = np.log10(global_predictions["Vbar_max"])

        return {
            "log_vmax_samples": log_vmax,
            "log_mbar_samples": log_mbar,
            "log_vdm_samples": log_vdm,
            "log_vbar_samples": log_vbar,
            "log_vmax_mean": np.nanmean(log_vmax, axis=1),
            "log_vmax_std": np.nanstd(log_vmax, axis=1),
            "log_mbar_mean": np.nanmean(log_mbar, axis=1),
            "log_mbar_std": np.nanstd(log_mbar, axis=1),
        }

    def _model_tag(self) -> str:
        return (
            f"alpha_{self.config.alpha:.3f}_"
            f"scatter_{self.config.scatter:.3f}_"
            f"x_{self.config.x:.3f}_nu_{self.config.nu:.3f}"
        )

    def _plot_results(
            self, sparc_galaxy_names, summary, observed, log_likelihood):
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        rcParams["font.family"] = "serif"
        rcParams["font.serif"] = ["Computer Modern"]
        rcParams["text.usetex"] = self.config.use_tex

        if self.config.plot_filename is None:
            plot_path = output_dir / f"btfr_corrected_{self._model_tag()}.png"
        else:
            plot_path = output_dir / self.config.plot_filename

        if self.config.plot_individual_histograms:
            hist_dir = output_dir / f"RTstats_{self._model_tag()}"
            hist_dir.mkdir(parents=True, exist_ok=True)
            for galaxy_index, galaxy_name in enumerate(sparc_galaxy_names):
                vels_hist(
                    summary["log_vmax_samples"][galaxy_index],
                    summary["log_vdm_samples"][galaxy_index],
                    summary["log_vbar_samples"][galaxy_index],
                    summary["log_vmax_mean"][galaxy_index],
                    summary["log_vmax_std"][galaxy_index],
                    observed["log_vobs"][galaxy_index],
                    observed["log_vobs_err"][galaxy_index],
                    summary["log_mbar_mean"][galaxy_index],
                    10,
                    10,
                    str(hist_dir / f"RTstats_{galaxy_index:03d}_{galaxy_name}.png"),
                )

        btfr_plot(
            alpha=self.config.alpha,
            sigma=self.config.scatter,
            x=self.config.x,
            nu=self.config.nu,
            vmaxshift=self.config.vmax_shift_mode,
            loglike=log_likelihood,
            xsim=summary["log_vmax_mean"],
            ysim=summary["log_mbar_mean"],
            xsimerr=summary["log_vmax_std"],
            ysimerr=summary["log_mbar_std"],
            xobs=observed["log_vobs"],
            yobs=observed["log_mobs"],
            xobserr=observed["log_vobs_err"],
            yobserr=observed["log_mobs_err"],
            plotname=str(plot_path),
            ax=None,
        )
        return plot_path

    def _save_samples(self, global_predictions, velocity_shift):
        if not self.config.save_samples:
            return None
        output_path = Path(self.config.output_dir) / f"samples_{self._model_tag()}.npz"
        np.savez_compressed(
            output_path,
            velocity_shift=velocity_shift,
            **global_predictions,
        )
        return output_path

    def run(self):
        """Run the single-model forward calculation and make the BTFR plot."""
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

            vobs = np.asarray(sparc_btfr_data["Vmax"], dtype=np.float64).reshape(-1)
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

            halos_selected = self._select_halos(halo_catalog)
            galaxy_data_jax, mass_model_data_jax, halo_catalog_data = (
                self._preprocess_data_for_jax(
                    galaxy_data, mass_model_data, halos_selected
                )
            )

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
            contra_interpolator = self._setup_contra_interpolator()

            self._print_computation_info()
            self.master_key, model_key = random.split(self.master_key)
            local_predictions = compute_simulated_diagnostics(
                model_key,
                self.config.alpha,
                self.config.scatter,
                self.config.nu,
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

            log_likelihood = float(evaluate_likelihoods(
                local_log_vmax,
                observed["log_vobs"],
                observed["log_vobs_err"],
                self.comm,
            )[0])

            global_predictions = self._gather_predictions(local_predictions)
            if self.rank != 0:
                return None

            summary = self._summary_statistics(
                global_predictions, velocity_shift
            )
            total_samples = self.config.n_am_reals * self.config.n_stellar_reals
            retained_percent = 100.0 * np.count_nonzero(
                np.isfinite(summary["log_vmax_samples"]), axis=1
            ) / total_samples

            print("\nSingle-model abundance-matching pipeline finished")
            print(f"Log likelihood: {log_likelihood:.8f}")
            if self.config.vmax_shift_mode:
                print(f"Applied log10(Vmax) shift: {velocity_shift:.8f}")
            print(
                "Percentage of samples retained after halo/emulator selection "
                "(per galaxy):"
            )
            print(np.round(retained_percent, 2).tolist())

            plot_path = self._plot_results(
                sparc_galaxy_names, summary, observed, log_likelihood
            )
            samples_path = self._save_samples(
                global_predictions, velocity_shift
            )
            print(f"BTFR plot saved to: {plot_path}")
            if samples_path is not None:
                print(f"Samples saved to: {samples_path}")

            gc.collect()
            jax.clear_caches()
            return {
                "log_likelihood": log_likelihood,
                "velocity_shift": velocity_shift,
                "retained_percent": retained_percent,
                "plot_path": plot_path,
                "samples_path": samples_path,
                "summary": summary,
            }

        except Exception as exc:
            if self.rank == 0:
                print(f"Fatal error in single-model BTFR pipeline: {exc}")
                traceback.print_exc()
            raise


def main():
    config = SingleModelConfig(
        n_am_reals=100,
        n_stellar_reals=1000,
        alpha_proxy=0.0,
        scatter=0.19,
        x=0.84,
        nu=0.430, #float(np.linspace(-3.0, 3.0, 20)[10]),
        base_seed=42,
        vmax_shift_mode=False,
        plot_individual_histograms=False,
        save_samples=False,
        use_tex=True,
        output_dir="/Users/fedorboreiko/Documents/Oxford/btfr_z/plots",
    )

    pipeline = SingleModelBtfrPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
