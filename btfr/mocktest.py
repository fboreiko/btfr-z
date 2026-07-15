"""
Mock truth data testing script using modular structure.
Generates mock true velocities and tests likelihood recovery.
"""

import numpy as np
from numpy.lib import recfunctions as rfn
import jax
import gc
import jax.numpy as jnp
from jax import random
import sys
import traceback
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from mpi4py import MPI
from tqdm import tqdm
from BAM import AbundanceMatch, proxies
from utils import *
from likelihood_computation_grid_jax import compute_likelihood, compute_AM_realization

@dataclass
class MockTruthConfig:
    """Configuration for mock truth testing parameters."""
    num_truths: int = 50
    n_am_reals: int = 100
    n_stellar_reals: int = 1000
    
    # True parameter values
    true_alpha_proxy: float = -0.5
    true_scatter: float = 0.1
    true_x: float = 0.5
    true_nu_index: int = 12  # Index in nu_range for true_nu
    
    # Grid search parameters
    alpha_proxy_range: Tuple[float, float] = (-np.pi / 2, np.pi / 2)
    scatter_range: Tuple[float, float] = (0.01, 1.0)
    x_range: Tuple[float, float] = (0.01, 0.95)
    nu_range: Tuple[float, float] = (-3.0, 3.0)
    grid_size: int = 20
    
    # Chunking parameters
    x_chunk_size: int = 5  # Number of x values to process at once
    x_chunk_start: int = 0  # Starting index for x values (0-based)
    
    # Mock dataset management
    create_mock_datasets: bool = True  # True to create and save, False to load existing
    mock_datasets_file: str = "mock_datasets.npy"  # File to save/load mock datasets
    
    vmax_shift_mode: bool = False
    output_dir: str = "."

    def __post_init__(self):
        """Validate configuration and set derived parameters."""
        if self.num_truths <= 0 or self.n_am_reals <= 0 or self.n_stellar_reals <= 0:
            raise ValueError("Number of realizations must be positive")
        if self.grid_size <= 0:
            raise ValueError("Grid size must be positive")
        if self.x_chunk_size <= 0:
            raise ValueError("X chunk size must be positive")
        if self.x_chunk_start < 0 or self.x_chunk_start >= self.grid_size:
            raise ValueError(f"X chunk start must be between 0 and {self.grid_size-1}")
        
        # Calculate true nu value
        nu_values = np.linspace(self.nu_range[0], self.nu_range[1], self.grid_size)
        self.true_nu = nu_values[self.true_nu_index]
        self.true_alpha = np.tan(self.true_alpha_proxy)

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print('Number of processes:', size)

class MockTruthTester:
    """Main class for testing mock truth data recovery."""
    
    def __init__(self, config: MockTruthConfig):
        self.config = config
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        # Initialize grid parameters
        self._setup_grid_parameters()
        
        # Setup x value chunking
        self._setup_x_chunking()

        # Setup contra grids
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
        
        # Calculate stellar mass realizations per x value
        n_stellar_reals_minx = int(
            self.config.n_stellar_reals * 
            (1 - np.max(self.x_values)) / (1 - np.min(self.x_values))
        )
        self.n_stellar_range = np.floor(
            np.linspace(n_stellar_reals_minx, self.config.n_stellar_reals, self.config.grid_size)
        ).astype(int)
        self.n_stellar_reals_postselection = int(n_stellar_reals_minx * (1 - np.min(self.x_values)))
    
    def _setup_x_chunking(self):
        """Setup x value chunking parameters."""
        # Calculate the actual chunk of x values to process
        x_end = min(self.config.x_chunk_start + self.config.x_chunk_size, len(self.x_values))
        self.x_chunk_indices = range(self.config.x_chunk_start, x_end)
        self.x_values_chunk = self.x_values[self.x_chunk_indices]
        self.n_stellar_range_chunk = self.n_stellar_range[self.x_chunk_indices]
        
        if self.rank == 0:
            print(f"Processing x values chunk: indices {self.config.x_chunk_start} to {x_end-1}")
            print(f"X values in chunk: {self.x_values_chunk}")
            print(f"Total x values in full grid: {len(self.x_values)}")
    
    def save_mock_datasets(self, mock_vels_ensemble: np.ndarray, mock_vels_err_ensemble: np.ndarray):
        """Save mock datasets to file."""
        if self.rank == 0:
            mock_data = {
                'mock_vels_ensemble': mock_vels_ensemble,
                'mock_vels_err_ensemble': mock_vels_err_ensemble,
                'config_params': {
                    'num_truths': self.config.num_truths,
                    'true_alpha_proxy': self.config.true_alpha_proxy,
                    'true_scatter': self.config.true_scatter,
                    'true_x': self.config.true_x,
                    'true_nu': self.config.true_nu
                }
            }
            output_path = Path(self.config.output_dir) / self.config.mock_datasets_file
            np.save(output_path, mock_data)
            print(f"Mock datasets saved to: {output_path}")
    
    def load_mock_datasets(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load mock datasets from file."""
        if self.rank == 0:
            mock_file_path = Path(self.config.output_dir) / self.config.mock_datasets_file
            if not mock_file_path.exists():
                raise FileNotFoundError(f"Mock datasets file not found: {mock_file_path}")
            
            mock_data = np.load(mock_file_path, allow_pickle=True).item()
            print(f"Mock datasets loaded from: {mock_file_path}")
            
            # Validate consistency with current config
            config_params = mock_data['config_params']
            if (config_params['num_truths'] != self.config.num_truths or
                abs(config_params['true_alpha_proxy'] - self.config.true_alpha_proxy) > 1e-6 or
                abs(config_params['true_scatter'] - self.config.true_scatter) > 1e-6 or
                abs(config_params['true_x'] - self.config.true_x) > 1e-6 or
                abs(config_params['true_nu'] - self.config.true_nu) > 1e-6):
                print("Warning: Loaded mock datasets were created with different parameters!")
                print(f"Loaded params: {config_params}")
                print(f"Current params: num_truths={self.config.num_truths}, "
                      f"true_alpha_proxy={self.config.true_alpha_proxy}, "
                      f"true_scatter={self.config.true_scatter}, "
                      f"true_x={self.config.true_x}, true_nu={self.config.true_nu}")
            
            return mock_data['mock_vels_ensemble'], mock_data['mock_vels_err_ensemble']
        else:
            return None, None

    def _select_halos(self, halo_catalog: np.ndarray, x: float) -> np.ndarray:
        """Select halos based on the cutoff fit for x values."""
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

    def _setup_contra_interpolator(self, nu: float, contra_grids: Dict, grid_axes) -> Optional[callable]:
        """Setup contra interpolator for given nu value."""
        if nu == 0.0:
            return None
            
        if nu not in contra_grids:
            if self.rank == 0:
                raise ValueError(f"Contra grid for nu={nu} not found. Available grids: {list(contra_grids.keys())}")
            return None
            
        contra_grid = contra_grids[nu]
        grid_jax = jnp.array(contra_grid, dtype=jnp.float32)
        
        def contra_interpolator(points):
            return jax_contra_interpolator(grid_jax, points, grid_axes)
        
        return contra_interpolator

    def _extract_galaxy_data(self, sparc_galaxy_names: list, galaxy_properties: Dict, 
                           bulge_luminosities: Dict, btfr_data: Dict) -> Dict[str, np.ndarray]:
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
        """
        Preprocess data to convert to JAX-compatible arrays.
        
        Args:
            galaxy_data: Galaxy sample data
            mass_model_catalog: Mass model catalog (should already be vectorized)
            halo_catalog: Halo catalog
            
        Returns:
            Processed data structures optimized for JAX
        """

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

    def _generate_mock_truth_data(self, abundance_match, galaxy_data, mass_model_data, 
                                  halo_catalog) -> Tuple[np.ndarray, np.ndarray]:
        """Generate mock truth velocity data."""
        mock_vels_ensemble = np.empty((self.config.num_truths, galaxy_data['L36'].shape[0]))
        mock_vels_err_ensemble = np.empty((self.config.num_truths, galaxy_data['L36'].shape[0]))

        key = random.PRNGKey(42 + self.rank)

        if self.rank == 0:
            print(f"Generating {self.config.num_truths} mock truth datasets...")

            # Select halos for true parameters
            halos_selected_true = self._select_halos(halo_catalog, self.config.true_x)

            # Preprocess data for jax
            galaxy_data, mass_model_data, halos_selected_data = self._preprocess_data_for_jax(
                galaxy_data, mass_model_data, halos_selected_true
            )

            print("Data preprocessed for JAX.")

            # Setup contra interpolator for true nu
            contra_interpolator_true = self._setup_contra_interpolator(
                self.config.true_nu, self.contra_grids, self.grid_axes
            )
            
            for i in tqdm(range(self.config.num_truths), desc="Mock truths"):
                key, subkey = random.split(key)

                theta_true = {"alpha": self.config.true_alpha, "scatter": self.config.true_scatter}
                deconv = abundance_match.deconvoluted_catalogs(theta_true, halos_selected_true)

                mock_vels_output = compute_AM_realization(
                    subkey, self.config.true_nu, abundance_match, deconv, contra_interpolator_true, 
                    galaxy_data, mass_model_data, halos_selected_data, n_reals=100
                )

                # Pick any non-nan element from each row
                mock_vels = np.array([
                    row[np.where(~np.isnan(row))[0][0]] if np.any(~np.isnan(row)) else np.nan
                    for row in mock_vels_output
                ])

                # Check for nans
                if np.isnan(mock_vels).any():
                    print(f"Warning: NaN values found in mock truth {i}, stopping generation")
                    break

                mock_vels_err = mock_vels * galaxy_data['Vobs_err'] / galaxy_data['Vobs']
                mock_vels = mock_vels + np.random.normal(loc=0, scale=mock_vels_err)

                mock_vels_ensemble[i] = mock_vels.flatten()
                mock_vels_err_ensemble[i] = mock_vels_err.flatten()

            return mock_vels_ensemble, mock_vels_err_ensemble
        else:
            return None, None

    def _generate_output_filename(self) -> str:
        """Generate output filename based on configuration."""
        x_chunk_end = min(self.config.x_chunk_start + self.config.x_chunk_size - 1, len(self.x_values) - 1)
        
        base_name = (
            f'likelihood_grid_{self.config.num_truths}mocktruths_'
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


    def compute_likelihood_grid(self) -> Optional[np.ndarray]:
        """Main computation loop for mock truth likelihood grid."""
        try:
            # Load data
            (sparc_galaxy_names, bulge_luminosities, galaxy_properties, 
             mass_model_data, sparc_btfr_data, stellar_mass_bins, 
             stellar_mass_function, halo_catalog) = load_data()
            
            # Vectorise (pad) the mass model table
            mass_model_data = vectorize_mass_model_table(sparc_galaxy_names, mass_model_data)

            # Extract galaxy data
            galaxy_data = self._extract_galaxy_data(sparc_galaxy_names, galaxy_properties, 
                                                    bulge_luminosities, sparc_btfr_data)
        
            # Add HI mass error and log btfr data
            galaxy_data['MH1_err'] = galaxy_data['MH1'] * 0.1
            galaxy_data['log_Vobs'] = np.log10(galaxy_data['Vobs'])
            galaxy_data['log_Vobs_err'] = (1 / np.log(10)) * (galaxy_data['Vobs_err'] / galaxy_data['Vobs'])

            if self.rank == 0:
                print("Data collected")

            # Create abundance matching object
            proxy = proxies["mvir_proxy"](use_cache=False)
            abundance_match = AbundanceMatch(
                stellar_mass_bins[10:], stellar_mass_function[10:], 
                halo_proxy=proxy, ext_range=(3.0, 12.0),
                boxsize=140, faint_end_first=True, scatter_mult=1, faint_end_slope=-0.42
            )

            # Generate or load mock truth data
            if self.config.create_mock_datasets:
                if self.rank == 0:
                    print("Creating new mock datasets...")
                mock_vels_ensemble, mock_vels_err_ensemble = self._generate_mock_truth_data(
                    abundance_match, galaxy_data, mass_model_data, halo_catalog
                )
                if self.rank == 0 and mock_vels_ensemble is not None:
                    self.save_mock_datasets(mock_vels_ensemble, mock_vels_err_ensemble)
            else:
                if self.rank == 0:
                    print("Loading existing mock datasets...")
                mock_vels_ensemble, mock_vels_err_ensemble = self.load_mock_datasets()

            # Broadcast mock data to all processes
            mock_vels_ensemble = self.comm.bcast(mock_vels_ensemble, root=0)
            mock_vels_err_ensemble = self.comm.bcast(mock_vels_err_ensemble, root=0)

            galaxy_data['log_Vmocks'] = np.log10(mock_vels_ensemble)
            galaxy_data['log_Vmocks_err'] = (1 / np.log(10)) * (mock_vels_err_ensemble / mock_vels_ensemble)

            num_truths_generated = mock_vels_ensemble.shape[0]

            if self.rank == 0:
                print("Mock vels")
                print(mock_vels_ensemble)

            gc.collect()
            jax.clear_caches()

            if self.rank == 0:
                self._print_computation_info(num_truths_generated)
                total_calculations = len(self.alpha_values) * len(self.scatter_values) * len(self.x_values_chunk) * len(self.nu_values)
                pbar = tqdm(total=total_calculations, desc="Grid Points Evaluated", position=0, leave=True)

            # Initialize likelihood grid - only for the x chunk we're processing
            likelihood_grid = np.full(
                (num_truths_generated, len(self.alpha_values), len(self.scatter_values), 
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
                    contra_interpolator = self._setup_contra_interpolator(nu, self.contra_grids, self.grid_axes)
                    
                    if nu != 0.0 and contra_interpolator is None:
                        break
                        
                    for i_alpha, alpha in enumerate(self.alpha_values):
                        for i_scatter, scatter in enumerate(self.scatter_values):
                            try:
                        
                                likelihoods = compute_likelihood(
                                    alpha, scatter, nu, abundance_match, contra_interpolator,
                                    galaxy_data, mass_model_data, halos_selected, 
                                    halos_selected_data, self.config.n_am_reals, n_stellar, 
                                    self.size, self.rank, self.comm
                                )

                                if likelihoods is not None:
                                    likelihood_grid[:, i_alpha, i_scatter, local_i_x, i_nu] = likelihoods
                                    pbar.update(1)
                                        
                            except Exception as e:
                                if self.rank == 0:
                                    print(f"Error computing likelihood for parameters "
                                          f"alpha={alpha:.3f}, scatter={scatter:.3f}, x={x:.3f}, nu={nu:.3f}:")
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

    def _print_computation_info(self, num_truths: int):
        """Print information about the computation setup."""
        print(f"Number of processes: {self.size}")
        print(f"\nNumber of mock truths generated: {num_truths}")
        print(f"Number of AM realizations: {self.config.n_am_reals}")
        print(f"Number of stellar mass realizations (after selection): {self.n_stellar_reals_postselection}")
        print(f"True alpha proxy: {self.config.true_alpha_proxy}, true scatter: {self.config.true_scatter}, true nu: {self.config.true_nu}")
        print(f"\nRunning a grid of alpha proxies from {np.min(self.alpha_proxy_values):.3f} to {np.max(self.alpha_proxy_values):.3f},")
        print(f"scatters from {np.min(self.scatter_values):.3f} to {np.max(self.scatter_values):.3f},")
        print(f"x values from {np.min(self.x_values_chunk):.3f} to {np.max(self.x_values_chunk):.3f} "
              f"(chunk {self.config.x_chunk_start} to {min(self.config.x_chunk_start + self.config.x_chunk_size - 1, len(self.x_values) - 1)}),")
        print(f"and nu values from {np.min(self.nu_values):.3f} to {np.max(self.nu_values):.3f}.")
        print(f"Vmax shift mode: {'ON' if self.config.vmax_shift_mode else 'OFF'}\n")

    def save_results(self, likelihood_grid: np.ndarray):
        """Save results to file."""
        if self.rank == 0:
            output_path = Path(self.config.output_dir) / self._generate_output_filename()
            np.save(output_path, likelihood_grid)
            print(f"Likelihood grid saved to: {output_path}")
            print("Mock truth testing complete!")


def main():
    
    """Main function."""
    # Configure parameters here - edit as needed
    config = MockTruthConfig(
        num_truths=50,
        n_am_reals=1,
        n_stellar_reals=1000,
        grid_size=20,
        
        # Chunking parameters - process first 5 x values (indices 0-4)
        x_chunk_size=5,
        x_chunk_start=0,  # Change this to process different chunks (0, 5, 10, 15)
        
        # Mock dataset management
        create_mock_datasets=False,  # Set to False to load existing mock datasets
        mock_datasets_file="mock_datasets.npy",
        
        vmax_shift_mode=False,
        output_dir="/Users/fedorboreiko/Documents/Oxford/btfr_z/btfr"
    )
    
    tester = MockTruthTester(config)
    likelihood_grid = tester.compute_likelihood_grid()
    
    if likelihood_grid is not None:
        tester.save_results(likelihood_grid)


if __name__ == "__main__":
    main()