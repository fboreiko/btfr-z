"""
Main script for computing the model parameter likelihood grid.
"""


import numpy as np
from numpy.lib import recfunctions as rfn
import jax.numpy as jnp
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from mpi4py import MPI
from tqdm import tqdm
from BAM import AbundanceMatch, proxies
from utils.halo_selection_utils import get_x_cutoff_fit
from utils.interpolation_utils import jax_contra_interpolator
from utils.likelihood_computation import compute_likelihood
from utils.data_loader import load_data, load_emulators

@dataclass
class GridConfig:
    """Configuration for grid search parameters."""
    n_am_reals: int = 100
    n_stellar_reals: int = 500
    alpha_proxy_range: Tuple[float, float] = (-np.pi / 2, np.pi / 2)
    scatter_range: Tuple[float, float] = (0.01, 1.0)
    x_range: Tuple[float, float] = (0.0, 0.95)
    nu_range: Tuple[float, float] = (-3.0, 3.0)
    grid_size: int = 20
    vmax_shift_mode: bool = False
    output_dir: str = "."

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.n_am_reals <= 0 or self.n_stellar_reals <= 0:
            raise ValueError("Number of realizations must be positive")
        if self.grid_size <= 0:
            raise ValueError("Grid size must be positive")


class LikelihoodGridComputer:
    """Main class for computing likelihood grids."""
    
    def __init__(self, config: GridConfig):
        self.config = config
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        # Initialize grid parameters
        self._setup_grid_parameters()
        
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

    def select_halos(self, halo_catalog: np.ndarray, x: float) -> np.ndarray:
        """
        Select halos based on the cutoff fit for x values.
        
        Args:
            halo_catalog: Halo catalog containing halo properties
            x: Cutoff value for halo selection
            
        Returns:
            Halo catalog with selection field
        """
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
        grid_jax = jnp.array(contra_grid, dtype=jnp.float64)
        
        def contra_interpolator(points):
            return jax_contra_interpolator(grid_jax, points, grid_axes)
        
        return contra_interpolator

    def _extract_galaxy_data(self, sparc_galaxy_names: list, galaxy_properties: Dict, 
                           bulge_luminosities: Dict) -> Dict[str, np.ndarray]:
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
            'Lbulge': np.array([bulge_luminosities[galaxy] for galaxy in sparc_galaxy_names])
        }

    def _generate_output_filename(self) -> str:
        """Generate output filename based on configuration."""
        base_name = (
            f'likelihood_grid_{self.config.n_am_reals}am_'
            f'{self.n_stellar_reals_postselection}stellar_'
            f'alphaproxy_{np.min(self.alpha_proxy_values):.3f}_{np.max(self.alpha_proxy_values):.3f}_'
            f'scatter_{np.min(self.scatter_values):.3f}_{np.max(self.scatter_values):.3f}_'
            f'x_{np.min(self.x_values):.3f}_{np.max(self.x_values):.3f}_'
            f'nu_{np.min(self.nu_values):.3f}_{np.max(self.nu_values):.3f}'
        )
        
        if self.config.vmax_shift_mode:
            base_name += '_vmaxshift'
        
        return f"{base_name}.npy"

    def compute_grid(self) -> Optional[np.ndarray]:
        """Main computation loop for likelihood grid."""
        try:
            # Load data
            (sparc_galaxy_names, bulge_luminosities, galaxy_properties, 
             mass_model_data, sparc_btfr_data, stellar_mass_bins, 
             stellar_mass_function, halo_catalog) = load_data()

            # Load emulators
            contra_grids, grid_axes = load_emulators()

            # Extract galaxy data
            galaxy_data = self._extract_galaxy_data(sparc_galaxy_names, galaxy_properties, bulge_luminosities)
            # Add HI mass error
            galaxy_data['MH1_err'] = galaxy_data['MH1'] * 0.1

            # Create abundance matching object
            proxy = proxies["mvir_proxy"](use_cache=False)
            abundance_match = AbundanceMatch(
                stellar_mass_bins[10:], stellar_mass_function[10:], 
                halo_proxy=proxy, ext_range=(3.0, 12.0),
                boxsize=140, faint_end_first=True, scatter_mult=1, faint_end_slope=-0.42
            )

            if self.rank == 0:
                self._print_computation_info()
                total_calculations = len(self.alpha_values) * len(self.scatter_values) * len(self.x_values) * len(self.nu_values)
                pbar = tqdm(total=total_calculations, desc="Grid Points Evaluated", position=0, leave=True)

            # Initialize likelihood grid
            likelihood_grid = np.full(
                (len(self.alpha_values), len(self.scatter_values), len(self.x_values), len(self.nu_values)), 
                np.nan
            )

            # Main computation loops
            for i_x, x in enumerate(self.x_values):
                halos_selected = self.select_halos(halo_catalog, x)
                n_stellar = self.n_stellar_range[i_x]

                for i_nu, nu in enumerate(self.nu_values):
                    contra_interpolator = self._setup_contra_interpolator(nu, contra_grids, grid_axes)
                    
                    if nu != 0.0 and contra_interpolator is None:
                        continue  # Skip this nu value if interpolator setup failed

                    for i_alpha, alpha in enumerate(self.alpha_values):
                        for i_scatter, scatter in enumerate(self.scatter_values):
                            try:
                                likelihood = compute_likelihood(
                                    alpha, scatter, nu, abundance_match, contra_interpolator,
                                    sparc_galaxy_names, mass_model_data, sparc_btfr_data,
                                    halos_selected, galaxy_data['Lbulge'], galaxy_data['L36'], 
                                    galaxy_data['L36_err'], galaxy_data['Reff'], galaxy_data['MH1'], 
                                    galaxy_data['MH1_err'], galaxy_data['d'], galaxy_data['d_err'],
                                    self.config.vmax_shift_mode, n_stellar, self.size, self.rank, self.comm,
                                    return_simulated_data=False
                                )

                                if likelihood is not None:
                                    likelihood_grid[i_alpha, i_scatter, i_x, i_nu] = likelihood
                                    if self.rank == 0:
                                        pbar.update(1)
                                        
                            except Exception as e:
                                if self.rank == 0:
                                    print(f"Error computing likelihood for parameters "
                                          f"alpha={alpha:.3f}, scatter={scatter:.3f}, x={x:.3f}, nu={nu:.3f}: {e}")
                                continue

                del halos_selected

            if self.rank == 0:
                pbar.close()
                return likelihood_grid
            else:
                return None
                
        except Exception as e:
            if self.rank == 0:
                print(f"Fatal error in computation: {e}")
            raise

    def _print_computation_info(self):
        """Print information about the computation setup."""
        print(f"Number of processes: {self.size}")
        print(f"\nRunning a grid of alpha proxies from {np.min(self.alpha_proxy_values):.3f} to {np.max(self.alpha_proxy_values):.3f},")
        print(f"scatters from {np.min(self.scatter_values):.3f} to {np.max(self.scatter_values):.3f},")
        print(f"x values from {np.min(self.x_values):.3f} to {np.max(self.x_values):.3f},")
        print(f"and nu values from {np.min(self.nu_values):.3f} to {np.max(self.nu_values):.3f}.")
        print(f"\nNumber of AM realizations: {self.config.n_am_reals}")
        print(f"Number of stellar mass realizations (after selection): {self.n_stellar_reals_postselection}")
        print(f"Vmax shift mode: {'ON' if self.config.vmax_shift_mode else 'OFF'}\n")

    def save_results(self, likelihood_grid: np.ndarray):
        """Save results to file."""
        if self.rank == 0:
            output_path = Path(self.config.output_dir) / self._generate_output_filename()
            np.save(output_path, likelihood_grid)
            print(f"Likelihood grid saved to: {output_path}")
            print("Computation complete!")


def main():
    """Main function."""
    # Configure parameters here - edit as needed
    config = GridConfig(
        n_am_reals=10,
        n_stellar_reals=500,
        grid_size=20,
        vmax_shift_mode=False,
        output_dir="."
    )
    
    computer = LikelihoodGridComputer(config)
    likelihood_grid = computer.compute_grid()
    
    if likelihood_grid is not None:
        computer.save_results(likelihood_grid)


if __name__ == "__main__":
    main()