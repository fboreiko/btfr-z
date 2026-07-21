"""
SPARC BTFR likelihood grid using the modular JAX pipeline.
"""

import gc
import sys
import traceback
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import numpy as np
from numpy.lib import recfunctions as rfn

import jax
import jax.numpy as jnp
from jax import random

from mpi4py import MPI
from tqdm import tqdm

from BAM import AbundanceMatch, proxies
from btfr.utils import *
from btfr.forward_model import compute_likelihood

@dataclass
class SparcGridConfig:
    """Configuration for either the selection or no-selection BTFR grid."""

    # True: run the four-parameter selection model and chunk over x.
    # False: run the three-parameter baseline model with x fixed to 0,
    #        no x chunking, and a 3D (alpha, scatter, nu) output grid.
    selection_model: bool = True

    n_am_reals: int = 100

    # Target number of *accepted* (post-selection) stellar samples per AM
    # realization. Raw draws are scaled as target / (1 - x) so the expected
    # post-selection sample count is constant across the x grid.
    n_postselection_target: int = 500

    # Memory/JIT cap on raw stellar draws per forward-model call. When
    # target / (1 - x) exceeds this, the shortfall is made up by running
    # proportionally more AM realizations instead.
    n_stellar_max_per_call: int = 5000

    # Grid axes: (min, max) range and number of nodes per dimension.
    alpha_proxy_range: Tuple[float, float] = (-np.pi / 2 + 0.01, np.pi / 2 - 0.01)
    n_alpha: int = 20
    scatter_range: Tuple[float, float] = (0.01, 0.8)
    n_scatter: int = 20
    x_range: Tuple[float, float] = (0.5, 0.99)
    n_x: int = 29
    nu_range: Tuple[float, float] = (-0.8, 1.6)
    n_nu: int = 25

    # Chunking parameters (split the x dimension across independent jobs)
    x_chunk_size: int = 5   # number of x values to process in this job
    x_chunk_start: int = 0  # starting x index for this job (0-based)

    base_seed: int = 42

    vmax_shift_mode: bool = False  # see note in compute_likelihood_grid()
    output_dir: str = "."

    def __post_init__(self):
        """Validate configuration."""
        if self.n_am_reals <= 0 or self.n_postselection_target <= 0:
            raise ValueError("Number of realizations must be positive")
        if self.n_stellar_max_per_call <= 0:
            raise ValueError("n_stellar_max_per_call must be positive")
        for name in ("n_alpha", "n_scatter", "n_nu"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

        # x-grid and chunk settings are irrelevant for the baseline model.
        if self.selection_model:
            if self.n_x <= 0:
                raise ValueError("n_x must be positive")
            if self.x_chunk_size <= 0:
                raise ValueError("X chunk size must be positive")
            if self.x_chunk_start < 0 or self.x_chunk_start >= self.n_x:
                raise ValueError(f"X chunk start must be between 0 and {self.n_x - 1}")
            if not (0.0 <= self.x_range[0] < self.x_range[1] < 1.0):
                raise ValueError("x_range must satisfy 0 <= x_min < x_max < 1")

class SparcLikelihoodGrid:
    """Computes the BTFR likelihood grid against the observed SPARC sample."""

    def __init__(self, config: SparcGridConfig):
        self.config = config
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.base_key = random.PRNGKey(config.base_seed)

        # Initialize grid parameters
        self._setup_grid_parameters()

        # Setup x value chunking
        self._setup_x_chunking()

        # Load contra emulator grids
        self.contra_grids, self.grid_axes = load_emulators()

        # Fail fast (on all ranks) if any grid nu lacks a trained contra grid
        self._validate_contra_grids()

    def _setup_grid_parameters(self):
        """Initialize per-dimension grid parameter arrays."""
        cfg = self.config
        self.alpha_proxy_values = np.linspace(*cfg.alpha_proxy_range, cfg.n_alpha)
        self.alpha_values = np.tan(self.alpha_proxy_values)
        self.scatter_values = np.linspace(*cfg.scatter_range, cfg.n_scatter)
        self.nu_values = np.linspace(*cfg.nu_range, cfg.n_nu)

        if cfg.selection_model:
            self.x_values = np.linspace(*cfg.x_range, cfg.n_x)

            # Raw stellar draws needed per AM realization for a constant expected
            # post-selection count: n_raw(x) = target / (1 - x).
            n_raw_needed = np.ceil(
                cfg.n_postselection_target / (1.0 - self.x_values)
            ).astype(int)

            # Cap raw draws per call; make up the shortfall with extra AM
            # realizations so the total budget n_am * n_stellar is preserved.
            self.n_stellar_range = np.minimum(
                n_raw_needed, cfg.n_stellar_max_per_call
            )
            self.n_am_range = np.ceil(
                cfg.n_am_reals * n_raw_needed / self.n_stellar_range
            ).astype(int)
        else:
            # Baseline model
            self.x_values = np.array([0.0], dtype=float)
            self.n_stellar_range = np.array(
                [cfg.n_postselection_target], dtype=int
            )
            self.n_am_range = np.array([cfg.n_am_reals], dtype=int)

        # Every rank must receive at least one AM realization, otherwise the
        # empty-stack path in compute_simulated_velocities fails.
        self.n_am_range = np.maximum(self.n_am_range, self.size)

        self.n_stellar_reals_postselection = int(cfg.n_postselection_target)

    def _setup_x_chunking(self):
        """Setup x chunking for selection runs; disable it for baseline runs."""
        if not self.config.selection_model:
            self.x_chunk_indices = range(1)
            self.x_values_chunk = self.x_values
            if self.rank == 0:
                print("Baseline model: x is fixed to 0; x chunking is disabled.")
            return

        x_end = min(
            self.config.x_chunk_start + self.config.x_chunk_size,
            len(self.x_values)
        )
        self.x_chunk_indices = range(self.config.x_chunk_start, x_end)
        self.x_values_chunk = self.x_values[self.x_chunk_indices]

        if self.rank == 0:
            print(
                f"Processing x values chunk: indices "
                f"{self.config.x_chunk_start} to {x_end - 1}"
            )
            print(f"X values in chunk: {self.x_values_chunk}")
            print(f"Total x values in full grid: {len(self.x_values)}")

    def _validate_contra_grids(self):
        """Check every non-zero grid nu has a trained contra grid up front."""
        missing = [
            nu for nu in self.nu_values
            if nu != 0.0 and nu not in self.contra_grids
        ]
        if missing:
            raise ValueError(
                f"Contra grids missing for nu values: {missing}. "
                f"Available grids: {sorted(self.contra_grids.keys())}. "
                f"Train grids for the exact linspace used here, or key the "
                f"grid dict on values rounded to a fixed precision."
            )

    def _global_grid_key(self, global_i_x: int, i_nu: int,
                         i_alpha: int, i_scatter: int):
        """Chunk-layout-invariant PRNG key for one grid point."""
        cfg = self.config
        flat_index = (
            ((global_i_x * cfg.n_nu + i_nu) * cfg.n_alpha + i_alpha)
            * cfg.n_scatter + i_scatter
        )
        return random.fold_in(self.base_key, flat_index)

    def _select_halos(self, halo_catalog: np.ndarray, x: float) -> np.ndarray:
        """Attach the selection mask, or keep every halo in baseline mode."""
        if not self.config.selection_model:
            halos_selected = halo_catalog.copy()
            if 'select' in halos_selected.dtype.names:
                halos_selected['select'] = 1.0
            else:
                halos_selected = rfn.append_fields(
                    halos_selected, 'select',
                    np.ones(halo_catalog.shape[0], dtype=float),
                    usemask=False
                )
            return halos_selected

        try:
            slope, intercept = get_x_cutoff_fit(halo_catalog, x)
            halos_selected = halo_catalog.copy()
            halos_selected = rfn.append_fields(
                halos_selected, 'select',
                np.ones(halo_catalog.shape[0], dtype=float),
                usemask=False
            )

            halo_log_Mvir = np.log10(halos_selected['Mvir'])
            halo_log_vmax = np.log10(halos_selected['vmax'])
            predicted_log_vmax = slope * halo_log_Mvir + intercept
            halos_selected['select'][halo_log_vmax > predicted_log_vmax] = np.nan

            return halos_selected
        except Exception as e:
            if self.rank == 0:
                print(f"Error in halo selection for x={x}: {e}")
            raise

    def _setup_contra_interpolator(self, nu: float) -> Optional[callable]:
        """Setup contra interpolator for a given nu value."""
        if nu == 0.0:
            return None

        if nu not in self.contra_grids:
            # Raise on ALL ranks so the communicator can't deadlock with
            # rank 0 dead and the others waiting in a collective call.
            raise ValueError(
                f"Contra grid for nu={nu} not found. "
                f"Available grids: {list(self.contra_grids.keys())}"
            )

        grid_jax = jnp.array(self.contra_grids[nu], dtype=jnp.float32)
        grid_axes = self.grid_axes

        def contra_interpolator(points):
            return jax_contra_interpolator(grid_jax, points, grid_axes)

        return contra_interpolator

    def _extract_galaxy_data(self, sparc_galaxy_names: list, galaxy_properties: Dict,
                             bulge_luminosities: Dict, btfr_data) -> Dict[str, np.ndarray]:
        """Extract galaxy data into arrays for vectorized calculations."""
        return {
            'L36': np.array([galaxy_properties[galaxy]['Total Luminosity at [3.6]']
                             for galaxy in sparc_galaxy_names]),
            'L36_err': np.array([galaxy_properties[galaxy]['Luminosity Error']
                                 for galaxy in sparc_galaxy_names]),
            'Reff': np.array([galaxy_properties[galaxy]['Effective Radius at [3.6]']
                              for galaxy in sparc_galaxy_names]),
            'MH1': np.array([galaxy_properties[galaxy]['Total HI mass']
                             for galaxy in sparc_galaxy_names]),
            'd': np.array([galaxy_properties[galaxy]['Distance']
                           for galaxy in sparc_galaxy_names]),
            'd_err': np.array([galaxy_properties[galaxy]['Distance Error']
                               for galaxy in sparc_galaxy_names]),
            'Lbulge': np.array([bulge_luminosities[galaxy]
                                for galaxy in sparc_galaxy_names]),
            'Vobs': np.array([btfr_data['Vmax']]),
            'Vobs_err': np.array([btfr_data['e_Vmax']])
        }

    def _preprocess_data_for_jax(self, galaxy_data, mass_model_catalog, halo_catalog):
        """Convert data to JAX-compatible arrays."""
        galaxy_data_jax = {
            key: jnp.array(value, dtype=jnp.float32)
            for key, value in galaxy_data.items()
        }

        mass_model_catalog_jax = {
            key: jnp.array(value, dtype=jnp.float32)
            for key, value in mass_model_catalog.items()
        }

        halo_catalog_data = {}
        for field in halo_catalog.dtype.names:
            halo_catalog_data[field] = jnp.array(halo_catalog[field])

        return galaxy_data_jax, mass_model_catalog_jax, halo_catalog_data

    def _generate_output_filename(self) -> str:
        """Generate an unambiguous output filename for the chosen model."""
        cfg = self.config

        if cfg.selection_model:
            x_chunk_end = min(
                cfg.x_chunk_start + cfg.x_chunk_size - 1,
                len(self.x_values) - 1
            )
            base_name = (
                f'likelihood_grid_'
                f'{cfg.n_am_reals}am_{self.n_stellar_reals_postselection}postsel_'
                f'shape_{cfg.n_alpha}x{cfg.n_scatter}x{cfg.n_x}x{cfg.n_nu}_'
                f'alphaproxy_{np.min(self.alpha_proxy_values):.3f}_{np.max(self.alpha_proxy_values):.3f}_'
                f'scatter_{np.min(self.scatter_values):.3f}_{np.max(self.scatter_values):.3f}_'
                f'x_{np.min(self.x_values):.3f}_{np.max(self.x_values):.3f}_'
                f'xchunk_{cfg.x_chunk_start}to{x_chunk_end}_'
                f'nu_{np.min(self.nu_values):.3f}_{np.max(self.nu_values):.3f}'
            )
        else:
            base_name = (
                f'likelihood_grid_baseline_x0_'
                f'{cfg.n_am_reals}am_{self.n_stellar_reals_postselection}stellar_'
                f'shape_{cfg.n_alpha}x{cfg.n_scatter}x{cfg.n_nu}_'
                f'alphaproxy_{np.min(self.alpha_proxy_values):.3f}_{np.max(self.alpha_proxy_values):.3f}_'
                f'scatter_{np.min(self.scatter_values):.3f}_{np.max(self.scatter_values):.3f}_'
                f'nu_{np.min(self.nu_values):.3f}_{np.max(self.nu_values):.3f}'
            )

        if cfg.vmax_shift_mode:
            base_name += '_vmaxshift'

        return f"{base_name}.npy"

    def save_results(self, likelihood_grid: np.ndarray):
        """Save results to file."""
        if self.rank == 0:
            output_path = Path(self.config.output_dir) / self._generate_output_filename()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, likelihood_grid)
            print(f"Likelihood grid saved to: {output_path}")
            print("SPARC likelihood grid computation complete!")

    def _print_computation_info(self):
        """Print information about the computation setup."""
        cfg = self.config
        model_name = "selection" if cfg.selection_model else "baseline (x = 0)"

        print(f"Model: {model_name}")
        print(f"Number of processes: {self.size}")
        print(f"\nBase number of AM realizations: {cfg.n_am_reals}")

        if cfg.selection_model:
            print(
                "Target post-selection stellar samples per AM realization: "
                f"{self.n_stellar_reals_postselection}"
            )
            print(
                "Raw stellar draws per call across x grid: "
                f"{np.min(self.n_stellar_range)} to {np.max(self.n_stellar_range)} "
                f"(cap {cfg.n_stellar_max_per_call})"
            )
            print(
                "Effective AM realizations across x grid: "
                f"{np.min(self.n_am_range)} to {np.max(self.n_am_range)}"
            )
            print(
                "\nSaved chunk shape (alpha, scatter, x, nu): "
                f"({cfg.n_alpha}, {cfg.n_scatter}, "
                f"{len(self.x_values_chunk)}, {cfg.n_nu})"
            )
        else:
            print(
                "Stellar samples per AM realization: "
                f"{self.n_stellar_reals_postselection}"
            )
            print(f"Effective AM realizations: {int(self.n_am_range[0])}")
            print(
                "\nSaved grid shape (alpha, scatter, nu): "
                f"({cfg.n_alpha}, {cfg.n_scatter}, {cfg.n_nu})"
            )

        print(
            f"Running alpha proxies from {np.min(self.alpha_proxy_values):.3f} "
            f"to {np.max(self.alpha_proxy_values):.3f},"
        )
        print(
            f"scatters from {np.min(self.scatter_values):.3f} "
            f"to {np.max(self.scatter_values):.3f},"
        )

        if cfg.selection_model:
            print(
                f"x values from {np.min(self.x_values_chunk):.3f} "
                f"to {np.max(self.x_values_chunk):.3f} "
                f"(chunk {cfg.x_chunk_start} to "
                f"{min(cfg.x_chunk_start + cfg.x_chunk_size - 1, len(self.x_values) - 1)}),"
            )
        else:
            print("x fixed to 0.000 (no halo selection),")

        print(
            f"and nu values from {np.min(self.nu_values):.3f} "
            f"to {np.max(self.nu_values):.3f}."
        )
        print(f"Vmax shift mode: {'ON' if cfg.vmax_shift_mode else 'OFF'}\n")

    def compute_likelihood_grid(self) -> Optional[np.ndarray]:
        """Main computation loop for the SPARC likelihood grid."""
        try:
            # Load data via the utils package
            (sparc_galaxy_names, bulge_luminosities, galaxy_properties,
             mass_model_data, sparc_btfr_data, stellar_mass_bins,
             stellar_mass_function, halo_catalog) = load_data()

            # Vectorise (pad) the mass model table
            mass_model_data = vectorize_mass_model_table(sparc_galaxy_names, mass_model_data)

            # Extract galaxy data
            galaxy_data = self._extract_galaxy_data(sparc_galaxy_names, galaxy_properties,
                                                    bulge_luminosities, sparc_btfr_data)

            # Add HI mass error and log-transformed observations
            galaxy_data['MH1_err'] = galaxy_data['MH1'] * 0.1
            galaxy_data['log_Vobs'] = np.log10(galaxy_data['Vobs'])
            galaxy_data['log_Vobs_err'] = (1 / np.log(10)) * (
                galaxy_data['Vobs_err'] / galaxy_data['Vobs']
            )

            log_V_targets = galaxy_data.pop('log_Vobs')          # shape (1, n_galaxies)
            log_V_targets_err = galaxy_data.pop('log_Vobs_err')  # shape (1, n_galaxies)
            galaxy_data.pop('Vobs')
            galaxy_data.pop('Vobs_err')

            if self.rank == 0:
                print("Data collected")

            # Create abundance matching object
            proxy = proxies["mvir_proxy"](use_cache=False)
            abundance_match = AbundanceMatch(
                stellar_mass_bins[10:], stellar_mass_function[10:],
                halo_proxy=proxy, ext_range=(3.0, 12.0),
                boxsize=140, faint_end_first=True, scatter_mult=1, faint_end_slope=-0.42
            )

            gc.collect()
            jax.clear_caches()

            if self.rank == 0:
                self._print_computation_info()
                total_calculations = (len(self.alpha_values) * len(self.scatter_values) *
                                      len(self.x_values_chunk) * len(self.nu_values))
                pbar = tqdm(total=total_calculations, desc="Grid Points Evaluated",
                            position=0, leave=True)

            if self.config.selection_model:
                # One 4D chunk: (alpha, scatter, x_in_chunk, nu).
                likelihood_grid = np.full(
                    (len(self.alpha_values), len(self.scatter_values),
                     len(self.x_values_chunk), len(self.nu_values)),
                    np.nan
                )
            else:
                # Baseline is intrinsically three-dimensional because x is fixed.
                likelihood_grid = np.full(
                    (len(self.alpha_values), len(self.scatter_values),
                     len(self.nu_values)),
                    np.nan
                )

            # Main computation loops - only over x values in the current chunk
            for local_i_x, global_i_x in enumerate(self.x_chunk_indices):
                x = self.x_values[global_i_x]
                halos_selected = self._select_halos(halo_catalog, x)
                n_stellar = int(self.n_stellar_range[global_i_x])
                n_am = int(self.n_am_range[global_i_x])
                _, _, halos_selected_data = self._preprocess_data_for_jax(
                    galaxy_data, mass_model_data, halos_selected
                )

                for i_nu, nu in enumerate(self.nu_values):
                    contra_interpolator = self._setup_contra_interpolator(nu)

                    for i_alpha, alpha in enumerate(self.alpha_values):
                        for i_scatter, scatter in enumerate(self.scatter_values):
                            try:
                                likelihood_key = self._global_grid_key(
                                    global_i_x, i_nu, i_alpha, i_scatter
                                )

                                likelihoods = compute_likelihood(
                                    likelihood_key, alpha, scatter, nu, abundance_match, contra_interpolator,
                                    galaxy_data, mass_model_data, halos_selected,
                                    halos_selected_data, log_V_targets, log_V_targets_err,
                                    n_am, n_stellar,
                                    self.size, self.rank, self.comm
                                )

                                if likelihoods is not None:
                                    # Single observed dataset -> length-1 array.
                                    if self.config.selection_model:
                                        likelihood_grid[
                                            i_alpha, i_scatter, local_i_x, i_nu
                                        ] = likelihoods[0]
                                    else:
                                        likelihood_grid[
                                            i_alpha, i_scatter, i_nu
                                        ] = likelihoods[0]
                                    pbar.update(1)

                            except Exception as e:
                                if self.rank == 0:
                                    print(f"Error computing likelihood for parameters "
                                          f"alpha={alpha:.3f}, scatter={scatter:.3f}, "
                                          f"x={x:.3f}, nu={nu:.3f}:")
                                    print(f"Error message: {e}")
                                    print("Full traceback:")
                                    traceback.print_exc()
                                    print("-" * 80)
                                continue

                    gc.collect()
                    jax.clear_caches()

                # Delete heavy objects to free memory
                del halos_selected, halos_selected_data
                gc.collect()
                jax.clear_caches()

            if self.rank == 0:
                pbar.close()
                return likelihood_grid
            else:
                return None

        except Exception as e:
            if self.rank == 0:
                print(f"Fatal error in computation: {e}")
            raise


def main():
    """Main function."""

    config = SparcGridConfig(
        selection_model=True,

        n_am_reals=100,
        n_postselection_target=100,
        n_stellar_max_per_call=5000,

        alpha_proxy_range=(-np.pi / 2 + 0.01, np.pi / 2 - 0.01),
        n_alpha=15,
        scatter_range=(0.01, 0.8),
        n_scatter=15,
        nu_range=(-0.8, 1.6),
        n_nu=25,

        # Used only when selection_model=True; ignored in baseline mode.
        x_range=(0.5, 0.96),
        n_x=20,
        x_chunk_size=5,
        x_chunk_start=0,

        vmax_shift_mode=False,
        output_dir="/Users/fedorboreiko/Documents/Oxford/btfr_z/btfr"
    )

    grid_runner = SparcLikelihoodGrid(config)
    likelihood_grid = grid_runner.compute_likelihood_grid()

    if likelihood_grid is not None:
        grid_runner.save_results(likelihood_grid)


if __name__ == "__main__":
    main()