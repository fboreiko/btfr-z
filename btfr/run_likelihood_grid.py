"""
SPARC BTFR likelihood grid using the modular JAX pipeline.

Run with e.g.:  mpiexec -n 4 python btfr_grid_jax.py
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
    """Configuration for the SPARC BTFR likelihood grid."""

    n_am_reals: int = 5
    n_stellar_reals: int = 100

    # Grid search parameters
    alpha_proxy_range: Tuple[float, float] = (-np.pi / 2, np.pi / 2)
    scatter_range: Tuple[float, float] = (0.01, 1.0)
    x_range: Tuple[float, float] = (0.0, 0.95)
    nu_range: Tuple[float, float] = (-3.0, 3.0)
    grid_size: int = 20

    # Chunking parameters (split the x dimension across independent jobs)
    x_chunk_size: int = 5   # number of x values to process in this job
    x_chunk_start: int = 0  # starting x index for this job (0-based)

    # Randomness: fixed base seed -> fully reproducible, while fold_in on
    # (rank, x, nu, alpha, scatter) guarantees independent streams across
    # grid points and across separately-run chunks.
    base_seed: int = 42

    vmax_shift_mode: bool = False  # see note in compute_likelihood_grid()
    output_dir: str = "."

    def __post_init__(self):
        """Validate configuration and set derived parameters."""
        if self.n_am_reals <= 0 or self.n_stellar_reals <= 0:
            raise ValueError("Number of realizations must be positive")
        if self.grid_size <= 0:
            raise ValueError("Grid size must be positive")
        if self.x_chunk_size <= 0:
            raise ValueError("X chunk size must be positive")
        if self.x_chunk_start < 0 or self.x_chunk_start >= self.grid_size:
            raise ValueError(f"X chunk start must be between 0 and {self.grid_size - 1}")

class SparcLikelihoodGrid:
    """Computes the BTFR likelihood grid against the observed SPARC sample."""

    def __init__(self, config: SparcGridConfig):
        self.config = config
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Master PRNG key lives with the driver (same pattern as MockTruthTester).
        self.master_key = random.PRNGKey(config.base_seed)

        # Initialize grid parameters
        self._setup_grid_parameters()

        # Setup x value chunking
        self._setup_x_chunking()

        # Load contra emulator grids
        self.contra_grids, self.grid_axes = load_emulators()

    def _setup_grid_parameters(self):
        """Initialize grid parameter arrays."""
        self.alpha_proxy_values = np.linspace(
            self.config.alpha_proxy_range[0],
            self.config.alpha_proxy_range[1],
            self.config.grid_size
        )
        self.alpha_values = np.tan(self.alpha_proxy_values)
        self.scatter_values = np.linspace(
            self.config.scatter_range[0],
            self.config.scatter_range[1],
            self.config.grid_size
        )
        self.x_values = np.linspace(
            self.config.x_range[0],
            self.config.x_range[1],
            self.config.grid_size
        )
        self.nu_values = np.linspace(
            self.config.nu_range[0],
            self.config.nu_range[1],
            self.config.grid_size
        )

        # Scale stellar realizations with x so the post-selection sample size
        # stays constant across the x grid
        target_postselection = self.config.n_stellar_reals * (1.0 - np.max(self.x_values))
        self.n_stellar_range = np.ceil(
            target_postselection / (1.0 - self.x_values)
        ).astype(int)
        self.n_stellar_reals_postselection = int(round(target_postselection))

    def _setup_x_chunking(self):
        """Setup x value chunking parameters."""
        x_end = min(self.config.x_chunk_start + self.config.x_chunk_size, len(self.x_values))
        self.x_chunk_indices = range(self.config.x_chunk_start, x_end)
        self.x_values_chunk = self.x_values[self.x_chunk_indices]
        self.n_stellar_range_chunk = self.n_stellar_range[self.x_chunk_indices]

        if self.rank == 0:
            print(f"Processing x values chunk: indices {self.config.x_chunk_start} to {x_end - 1}")
            print(f"X values in chunk: {self.x_values_chunk}")
            print(f"Total x values in full grid: {len(self.x_values)}")

    def _select_halos(self, halo_catalog: np.ndarray, x: float) -> np.ndarray:
        """Select halos based on the cutoff fit for a given x value."""
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
        """Generate output filename based on configuration (incl. chunk range)."""
        x_chunk_end = min(self.config.x_chunk_start + self.config.x_chunk_size - 1,
                          len(self.x_values) - 1)

        base_name = (
            f'likelihood_grid_'
            f'{self.config.n_am_reals}am_{self.n_stellar_reals_postselection}stellar_'
            f'alphaproxy_{np.min(self.alpha_proxy_values):.3f}_{np.max(self.alpha_proxy_values):.3f}_'
            f'scatter_{np.min(self.scatter_values):.3f}_{np.max(self.scatter_values):.3f}_'
            f'x_{np.min(self.x_values):.3f}_{np.max(self.x_values):.3f}_'
            f'xchunk_{self.config.x_chunk_start}to{x_chunk_end}_'
            f'nu_{np.min(self.nu_values):.3f}_{np.max(self.nu_values):.3f}'
        )

        if self.config.vmax_shift_mode:
            base_name += '_vmaxshift'

        return f"{base_name}.npy"

    def save_results(self, likelihood_grid: np.ndarray):
        """Save results to file."""
        if self.rank == 0:
            output_path = Path(self.config.output_dir) / self._generate_output_filename()
            np.save(output_path, likelihood_grid)
            print(f"Likelihood grid saved to: {output_path}")
            print("SPARC likelihood grid computation complete!")

    def _print_computation_info(self):
        """Print information about the computation setup."""
        print(f"Number of processes: {self.size}")
        print(f"\nNumber of AM realizations: {self.config.n_am_reals}")
        print(f"Number of stellar mass realizations (after selection): "
              f"{self.n_stellar_reals_postselection}")
        print(f"\nRunning a grid of alpha proxies from {np.min(self.alpha_proxy_values):.3f} "
              f"to {np.max(self.alpha_proxy_values):.3f},")
        print(f"scatters from {np.min(self.scatter_values):.3f} to {np.max(self.scatter_values):.3f},")
        print(f"x values from {np.min(self.x_values_chunk):.3f} to {np.max(self.x_values_chunk):.3f} "
              f"(chunk {self.config.x_chunk_start} to "
              f"{min(self.config.x_chunk_start + self.config.x_chunk_size - 1, len(self.x_values) - 1)}),")
        print(f"and nu values from {np.min(self.nu_values):.3f} to {np.max(self.nu_values):.3f}.")
        print(f"Vmax shift mode: {'ON' if self.config.vmax_shift_mode else 'OFF'}\n")

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

            # Likelihood grid for this x chunk only (legacy 4D shape)
            likelihood_grid = np.full(
                (len(self.alpha_values), len(self.scatter_values),
                 len(self.x_values_chunk), len(self.nu_values)),
                np.nan
            )

            # Main computation loops - only over x values in the current chunk
            for local_i_x, global_i_x in enumerate(self.x_chunk_indices):
                x = self.x_values[global_i_x]
                halos_selected = self._select_halos(halo_catalog, x)
                n_stellar = self.n_stellar_range[global_i_x]
                _, _, halos_selected_data = self._preprocess_data_for_jax(
                    galaxy_data, mass_model_data, halos_selected
                )

                for i_nu, nu in enumerate(self.nu_values):
                    contra_interpolator = self._setup_contra_interpolator(nu)

                    for i_alpha, alpha in enumerate(self.alpha_values):
                        for i_scatter, scatter in enumerate(self.scatter_values):
                            try:
                                self.master_key, likelihood_key = random.split(self.master_key)

                                likelihoods = compute_likelihood(
                                    likelihood_key, alpha, scatter, nu, abundance_match, contra_interpolator,
                                    galaxy_data, mass_model_data, halos_selected,
                                    halos_selected_data, log_V_targets, log_V_targets_err, 
                                    self.config.n_am_reals, n_stellar,
                                    self.size, self.rank, self.comm
                                )

                                if likelihoods is not None:
                                    # Single observed dataset -> length-1 array
                                    likelihood_grid[i_alpha, i_scatter, local_i_x, i_nu] = \
                                        likelihoods[0]
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
    # Configure parameters here - edit as needed
    config = SparcGridConfig(
        n_am_reals=100,
        n_stellar_reals=10000,
        grid_size=20,

        # Chunking parameters - process first 5 x values (indices 0-4).
        # Change x_chunk_start to 5, 10, 15 for the other jobs.
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