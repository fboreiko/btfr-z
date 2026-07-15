##### mpiexec -n 4 python btfr/btfr_contra.py

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
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set the font to Computer Modern (LaTeX default) and enable LaTeX rendering
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern']
rcParams['text.usetex'] = True

@staticmethod
def array_mean_std(twodarray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the mean and standard deviation of a 2D array along the first axis.
    
    Args:
        twodarray (np.ndarray): Input 2D array.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Mean and standard deviation of the input array.
    """
    means = np.array([np.nanmean(row) for row in twodarray])
    stds = np.array([np.nanstd(row) for row in twodarray])
    return means, stds

@dataclass
class ModelConfig:
    """Configuration for a single model run."""
    n_am_reals: int = 100
    n_stellar_reals: int = 500
    alpha_proxy_value: float = 0.5
    scatter_value: float = 0.1
    x_value: float = 0.5
    nu_value: float = 0.5
    vmax_shift_mode: bool = False

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
            
            # Load emulator
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

            halos_selected = self.select_halos(halo_catalog, self.config.x_value)

            contra_interpolator = self._setup_contra_interpolator(self.config.nu_value, contra_grids, grid_axes)
                    
            if self.config.nu_value != 0.0 and contra_interpolator is None:
                raise ValueError("Contra interpolator not found for the specified nu value.")
            
            results = compute_likelihood(
                self.alpha, self.config.scatter_value, self.config.nu_value, 
                abundance_match, contra_interpolator, sparc_galaxy_names, 
                mass_model_data, sparc_btfr_data, halos_selected, 
                galaxy_data['Lbulge'], galaxy_data['L36'], galaxy_data['L36_err'], 
                galaxy_data['Reff'], galaxy_data['MH1'], galaxy_data['MH1_err'], 
                galaxy_data['d'], galaxy_data['d_err'], self.config.vmax_shift_mode, 
                self.config.n_am_reals, self.n_stellar_reals, self.size, self.rank, 
                self.comm, return_all_data=True
                )
            
            if self.rank == 0:
                print("\nModel run successfully.")
            
            if results is not None:
                print(f"Log likelihood: {results[0]:.3f}")

                # Process observed data
                V_obs_unlogged = np.array(sparc_btfr_data['Vmax'])
                V_obs_err_unlogged = np.array(sparc_btfr_data['e_Vmax'])
                V_obs = np.log10(V_obs_unlogged)
                V_obs_err = V_obs_err_unlogged / (V_obs_unlogged * np.log(10))

                Mbar_obs = np.array(sparc_btfr_data['log(Mb)'])
                Mbar_obs_err = np.array(sparc_btfr_data['e_log(Mb)'])

                # Unpack results
                log_likelihood, inidividual_log_likelihoods, V_sims, Vdm_sims, Mstellar_sims, \
                    Mgas_sims, Mbar_sims = results

                # Print the percentages of haloes surviving the selection
                original_length = self.config.n_am_reals * self.n_stellar_reals
                filtered_lengths = [np.count_nonzero(~np.isnan(row)) for row in V_sims]
                percentages = [round((length / original_length) * 100, 2) for length in filtered_lengths]

                if self.rank == 0:
                    print(f"\nPercentage of haloes surviving selection: {percentages}")

                # Prepare data for plotting
                V_sim, V_sim_err = array_mean_std(V_sims)
                Vdm_sim, Vdm_sim_err = array_mean_std(Vdm_sims)
                Mbar_sim, Mbar_sim_err = array_mean_std(Mbar_sims)

                # Calculate residuals
                sigmas = np.sqrt(V_sim_err**2 + V_obs_err**2)
                residuals = (V_obs - V_sim) / sigmas

                btfr_plot_name = self._generate_btfr_filename()

                btfr_plot(
                    self.alpha, self.config.scatter_value,
                    self.config.x_value, self.config.nu_value,
                    self.config.vmax_shift_mode,
                    log_likelihood, V_sim, Mbar_sim, 
                    V_sim_err, Mbar_sim_err, V_obs, 
                    Mbar_obs, V_obs_err, Mbar_obs_err, 
                    plotname=btfr_plot_name
                    )
                
                return galaxy_data['Reff'], Mstellar_sims, Mgas_sims, V_sim, V_sim_err, Mbar_sim, \
                    Mbar_sim_err, V_obs, V_obs_err, Mbar_obs, Mbar_obs_err, residuals, log_likelihood, \
                    inidividual_log_likelihoods
            
            return None
        
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
    """Main function."""
    # Initialize MPI
    try:
        # Check if MPI is already initialized (some MPI implementations do this automatically)
        if not MPI.Is_initialized():
            MPI.Init()
        
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Configure parameters here - edit as needed
        config_1 = ModelConfig(
            n_am_reals=100,
            n_stellar_reals=100,
            alpha_proxy_value=-0.62,
            scatter_value=0.01,
            x_value=0.36,
            nu_value=np.linspace(-3.0, 3.0, 20)[0],
            vmax_shift_mode=False
        )

        model_1 = ForwardModelPipeline(config_1, comm, rank, size)
        model_1_output = model_1.run()

        config_2 = ModelConfig(
            n_am_reals=100,
            n_stellar_reals=100,
            alpha_proxy_value=-0.62,
            scatter_value=0.01,
            x_value=0.36,
            nu_value=np.linspace(-3.0, 3.0, 20)[13],
            vmax_shift_mode=False,
        )

        model_2 = ForwardModelPipeline(config_2, comm, rank, size)
        model_2_output = model_2.run()

        if rank == 0:   

            Reff, Mstellar_sims, Mgas_sims = model_1_output[:3]

            delta_loglike = model_2_output[-1] - model_1_output[-1]

            # Save the delta log likelihood to a file
            delta_loglike_filename = '/Users/fedorboreiko/Documents/Oxford/btfr_z/btfr/delta_loglike.txt'
            np.savetxt(delta_loglike_filename, delta_loglike, fmt='%.6f')

            # Compute global average stellar surface mass density
            Sigma_star = Mstellar_sims / (np.pi * Reff[:, np.newaxis]**2)
            Sigma_star_mean, Sigma_star_std = array_mean_std(Sigma_star)

            # Plotting the delta log likelihood correlation with log10(Sigma_star)
            delta_loglike_correlation_plot(
                Sigma_star_mean, delta_loglike,
                r'$\log_{10}(\Sigma_{\star})$ [M$_{\odot}$ kpc$^{-2}$]', 
                plotname='/Users/fedorboreiko/Documents/Oxford/btfr_z/plots/delta_loglike_vs_sigma_star.png'
            )

            # Plotting the delta log likelihood correlation with log10(Mgas)
            Mgas_mean, Mgas_std = array_mean_std(Mgas_sims)
            delta_loglike_correlation_plot(
                Mgas_mean, delta_loglike, 
                r'$\log_{10}(M_{\rm gas})$ [M$_{\odot}$]', 
                plotname='/Users/fedorboreiko/Documents/Oxford/btfr_z/plots/delta_loglike_vs_mgas.png'
            )

            # Plotting the delta log likelihood correlation with log10(Mstellar)
            Mstellar_mean, Mstellar_std = array_mean_std(Mstellar_sims)
            delta_loglike_correlation_plot(
                Mstellar_mean, delta_loglike, 
                r'$\log_{10}(M_{\star})$ [M$_{\odot}$]', 
                plotname='/Users/fedorboreiko/Documents/Oxford/btfr_z/plots/delta_loglike_vs_mstellar.png'
            )

            # Plotting the delta log likelihood correlation with Reff
            delta_loglike_correlation_plot(
                Reff, delta_loglike, 
                r'$R_{\rm eff}$ [kpc]', 
                plotname='/Users/fedorboreiko/Documents/Oxford/btfr_z/plots/delta_loglike_vs_reff.png'
            )

            """create_btfr_panel_plot(config_1, model_1_output, config_2, model_2_output,
                '/Users/fedorboreiko/Documents/Oxford/btfr_z/plots/btfr_panel_plot.png'
            )"""

            """# Make the 3d colormap of log10(Sigma_star_mean) vs Reff vs log10(Mgas) and color it by delta_loglike
            import plotly.graph_objects as go

            fig = go.Figure()

            fig.add_trace(go.Scatter3d(
                x=np.log10(Sigma_star_mean),
                y=np.log10(Mgas_mean),
                z=Reff,
                mode='markers',
                marker=dict(
                    size=5,
                    color=delta_loglike,  # Color by delta_loglike
                    colorscale='Viridis',  # Choose a colorscale
                    opacity=0.8,
                    colorbar=dict(title=r'$\Delta \log \mathcal{L}$')
                )
            ))

            fig.update_layout(
                scene=dict(
                    xaxis_title='log10(Sigma_star)',
                    yaxis_title='log10(M_gas)',
                    zaxis_title='R_eff'
                ),
                title='3D Scatter Plot: Log10(Sigma_star_mean) vs Log10(Mgas_mean) vs Reff',
                margin=dict(l=0, r=0, b=0, t=40)
            )

            fig.show()"""

            

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