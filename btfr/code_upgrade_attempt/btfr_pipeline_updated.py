###mpiexec -n 4 python btfr/btfr_pipeline_updated.py

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
from utils import *
from btfr.code_upgrade_attempt.likelihood_computation import compute_likelihood
import time

@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    n_am_reals: int = 100
    n_stellar_reals: int = 500
    alpha_proxy_value: float = 0.5
    scatter_value: float = 0.1
    x_value: float = 0.0
    nu_value: float = 0.5
    vmax_shift_mode: bool = False
    return_all_data_mode: bool = False
    plot_hists: bool = False
    output_dir: str = "."
        
class ForwardModelPipeline:
    """Class to handle the forward modeling pipeline for the BTFR analysis."""

    def __init__(self, config: ModelConfig, comm: MPI.Comm, rank: int, size: int):
        self.config = config
        self.comm = comm
        self.rank = rank
        self.size = size
        
        self.alpha = np.tan(config.alpha_proxy_value)

        # Calculate the actual number of stellar mass reasisations 
        # needed to get the n_stellar_reals on the outcome
        self.n_stellar_reals = int(config.n_stellar_reals / (1 - self.config.x_value))

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
    
    def run(self):
        """Run the forward modeling pipeline."""
        try:
            # Load data
            (sparc_galaxy_names, bulge_luminosities, galaxy_properties, 
             mass_model_data, sparc_btfr_data, stellar_mass_bins, 
             stellar_mass_function, halo_catalog) = load_data()
            
            # Vectorise (pad) the mass model table
            #mass_model_data = vectorize_mass_model_table(sparc_galaxy_names, mass_model_data)

            # Load emulator
            contra_grids, grid_axes = load_emulators()

            # Extract galaxy data
            galaxy_data = self._extract_galaxy_data(sparc_galaxy_names, galaxy_properties, 
                                                    bulge_luminosities)
            
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
                
            halos_selected = self.select_halos(halo_catalog, self.config.x_value)

            contra_interpolator = self._setup_contra_interpolator(self.config.nu_value, contra_grids, grid_axes)
                    
            if self.config.nu_value != 0.0 and contra_interpolator is None:
                raise ValueError("Contra interpolator not found for the specified nu value.")
            
            if self.rank == 0:
                start_time = time.time()
            
            results = compute_likelihood(
                self.alpha, self.config.scatter_value, self.config.nu_value, 
                abundance_match, contra_interpolator, sparc_galaxy_names, 
                mass_model_data, sparc_btfr_data, halos_selected, 
                galaxy_data['Lbulge'], galaxy_data['L36'], galaxy_data['L36_err'], 
                galaxy_data['Reff'], galaxy_data['MH1'], galaxy_data['MH1_err'], 
                galaxy_data['d'], galaxy_data['d_err'], self.config.vmax_shift_mode, 
                self.config.n_am_reals, self.n_stellar_reals, self.size, self.rank, 
                self.comm, return_all_data=self.config.return_all_data_mode) #, 
            
            if self.rank == 0:
                print("\nModel run successfully.")
                print(f"Elapsed time: {time.time() - start_time:.2f} seconds")

                if self.config.return_all_data_mode:
                    print(f"Log likelihood: {results[0]:.3f}")
                    self._plot_results(results, sparc_btfr_data)
                else:
                    log_likelihood = results
                    print(f"Log likelihood: {log_likelihood:.3f}")
        
        except Exception as e:
            if self.rank == 0:
                print(f"Fatal error in computation: {e}")
            raise

    def _print_computation_info(self):
        """Print information about the computation setup."""
        print(f"Number of processes: {self.size}")
        print(f"\nRunning a forward model with the following parameters:")
        print(f"  - Number of AM reals: {self.config.n_am_reals}")
        print(f"  - Number of stellar reals: {self.config.n_stellar_reals}")
        print(f"  - Number of adjusted stellar reals: {self.n_stellar_reals}")
        print(f"  - Alpha proxy value: {self.config.alpha_proxy_value}")
        print(f"  - Scatter value: {self.config.scatter_value}")
        print(f"  - X value: {self.config.x_value}")
        print(f"  - Nu value: {self.config.nu_value}")
        if self.config.vmax_shift_mode:
            print("  - Vmax shift mode.")
        else:
            print("  - Usual mode.")

    def _plot_results(self, results: tuple, sparc_catalog: np.ndarray):
        """Plot the results of the forward model."""
        # Process observed data
        V_obs_unlogged = np.array(sparc_catalog['Vmax'])
        V_obs_err_unlogged = np.array(sparc_catalog['e_Vmax'])
        V_obs = np.log10(V_obs_unlogged)
        V_obs_err = V_obs_err_unlogged / (V_obs_unlogged * np.log(10))

        M_obs = np.array(sparc_catalog['log(Mb)'])
        M_obs_err = np.array(sparc_catalog['e_log(Mb)'])

        # Unpack results
        log_likelihood, inidividual_log_likelihoods, V_sims, Vdm_sims, \
        Mstellar_sims, Mgas_sims, Mbar_sims, Mstar_am, Mvir_am = results

        # Print the percentages of haloes surviving the selection
        original_length = self.config.n_am_reals * self.n_stellar_reals
        filtered_lengths = [np.count_nonzero(~np.isnan(row)) for row in V_sims]
        percentages = [round((length / original_length) * 100, 2) for length in filtered_lengths]

        if self.rank == 0:
            print(f"\nPercentage of haloes surviving selection: {percentages}")

        # Prepare data for plotting
        V_sim = np.array([np.nanmean(row) for row in V_sims])
        V_sim_err = np.array([np.nanstd(row) for row in V_sims])
        Mbar_sim = np.array([np.nanmean(row) for row in Mbar_sims])
        Mbar_sim_err = np.array([np.nanstd(row) for row in Mbar_sims])

        #Save the simulated data
        if self.rank == 0:
            output_data = np.rec.fromarrays(
                [V_sim, V_sim_err, Mbar_sim, Mbar_sim_err]
            )
            output_path = Path(self.config.output_dir) / f"BTFR_nu_{self.config.nu_value:.3f}.csv"
            np.savetxt(output_path, output_data, delimiter=',', 
                    header="V_sim,V_sim_err,Mbar_sim,Mbar_sim_err", fmt="%s")

        if self.config.plot_hists:
            for galnum in range(len(sparc_catalog)):

                expore_hist_name = (
                    f'/Users/fedorboreiko/Documents/Oxford/btfr_z/'
                    f'plots/hists/hist_galaxy_{galnum}_'
                    f'alphaproxy_{self.config.alpha_proxy_value:.3f}_'
                    f'scatter_{self.config.scatter_value:.3f}_'
                    f'x_{self.config.x_value:.3f}_'
                    f'nu_{self.config.nu_value:.3f}.png'
                )

                vels_hist(
                    V_sims[galnum], Vdm_sims[galnum],
                    V_sim[galnum], V_sim_err[galnum],
                    V_obs[galnum], V_obs_err[galnum],
                    Mbar_sim[galnum], inidividual_log_likelihoods[galnum], 
                    expore_hist_name
                    )
            
        btfr_plot_name = self._generate_btfr_filename()

        btfr_plot(
            self.alpha, self.config.scatter_value,
            self.config.x_value, self.config.nu_value,
            self.config.vmax_shift_mode,
            log_likelihood, V_sim, Mbar_sim, 
            V_sim_err, Mbar_sim_err, V_obs, 
            M_obs, V_obs_err, M_obs_err, 
            plotname=btfr_plot_name
            )

    def _generate_btfr_filename(self) -> str:
            """Generate output filename based on configuration."""
            base_name = (
                '/Users/fedorboreiko/Documents/Oxford/btfr_z/'
                f'plots/btfr_{self.config.n_am_reals}am_'
                f'{self.config.n_stellar_reals}stellar_'
                f'alphaproxy_{self.config.alpha_proxy_value:.3f}_'
                f'scatter_{self.config.scatter_value:.3f}_'
                f'x_{self.config.x_value:.3f}_'
                f'nu_{self.config.nu_value:.3f}'
            )
            
            if self.config.vmax_shift_mode:
                base_name += '_vmaxshift'
            
            return f"{base_name}.png"

def main():
    """Main function with proper MPI initialization and finalization."""
    # Initialize MPI
    try:
        # Check if MPI is already initialized (some MPI implementations do this automatically)
        if not MPI.Is_initialized():
            MPI.Init()
        
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        if rank == 0:
            print(f"MPI initialized with {size} processes")
        
        # Configure parameters here - edit as needed
        config = ModelConfig(
            n_am_reals=100,
            n_stellar_reals=1000,
            alpha_proxy_value=0.5,
            scatter_value=0.1,
            x_value=0.0,
            nu_value=np.linspace(-3.0, 3.0, 20)[15],
            vmax_shift_mode=True,
            return_all_data_mode=True,
            plot_hists=False,
            output_dir="/Users/fedorboreiko/Documents/Oxford/btfr_z/btfr"
        )

        # Create and run the model
        model = ForwardModelPipeline(config, comm, rank, size)
        model.run()
        
    except Exception as e:
        if rank == 0:
            print(f"Error in main: {e}")
        raise
    finally:
        # Finalize MPI
        if MPI.Is_initialized() and not MPI.Is_finalized():
            MPI.Finalize()

if __name__ == "__main__":
    main()