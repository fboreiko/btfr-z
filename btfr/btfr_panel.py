import sys
import os
sys.path.append('/Users/fedorboreiko/Documents/Oxford/btfr_z')

import numpy as np
import numpy.lib.recfunctions as rfn
import pandas as pd
from btfr.utils.massfuncs import get_GSMF_ELPETRO
from BAM import AbundanceMatch, proxies
from btfr.utils.plotting_utils import btfr_plot
from btfr.code_legacy.btfr_utils import get_x_cutoff_fit, nfw_circular_velocity_contra, nfw_circular_velocity, get_loglike
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import binned_statistic
import pickle
import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
from functools import partial

jax.config.update("jax_enable_x64", True)

# Set the font to Computer Modern (LaTeX default) and enable LaTeX rendering
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern']
rcParams['text.usetex'] = True

N_AM_REALS = 100
N_STELLAR_REALS = 1000

SCATTER_1 = 0.1
ALPHA_PROXY_1 = 0.5
X_1 = 0.0
NU_1 = np.linspace(-3.0, 3.0, 20)[11]

SCATTER_2 = 0.1
ALPHA_PROXY_2 = 0.5
X_2 = 0.0
NU_2 = np.linspace(-3.0, 3.0, 20)[15]

VMAXSHIFT = False

ALPHA_1 = np.tan(ALPHA_PROXY_1)
ALPHA_2 = np.tan(ALPHA_PROXY_2)

M2L_DISK_MEAN = 0.5
M2L_DISK_ERROR = 0.2 # dex

M2L_BULGE_MEAN = 0.7
M2L_BULGE_ERROR = 0.2 # dex

# Add configuration for data paths
DATA_DIR = os.environ.get('BTFR_DATA_DIR', '/Users/fedorboreiko/Documents/Oxford/Personal_codes/Codebase')
HALOS_FILE = os.path.join(DATA_DIR, 'halos_z_0p00.npy')
CONTRA_EMULATOR_DIR = os.path.join(DATA_DIR, 'contra_emulators')
CONTRA_GRIDS_FILE = os.path.join(CONTRA_EMULATOR_DIR, 'grids_fullrange.pkl')

# Pre-load data
bulge_lumins = pd.read_csv('Tabular_data/Bulge_lum_table.csv')
mass_models = pd.read_csv('Tabular_data/Mass_models_table.csv')
galaxy_sample = pd.read_csv('Tabular_data/Gal_sample_table.csv')
sparc_btfr = pd.read_csv('Tabular_data/sparc_btfr.csv')

sparc_galaxy_list = sparc_btfr['Name']

bulge_lumins = bulge_lumins[bulge_lumins['Galaxy'].isin(sparc_galaxy_list)]
mass_models = mass_models[mass_models['ID'].isin(sparc_galaxy_list)]
galaxy_sample = galaxy_sample[galaxy_sample['Galaxy'].isin(sparc_galaxy_list)]

@partial(jax.jit, static_argnames=("order",))
def jax_contra_interpolator(grid, positions, grid_axes, order=1):
    """
    A thin wrapper that converts the contra grid and query points into JAX arrays,
    performs interpolation, and applies a validity mask to handle out-of-bounds points
    by setting the log(mhi) value to nans.
    """
    indices = []
    valid_mask = jnp.ones(positions.shape[0], dtype=bool)  # Start with all points valid
    for i in range(positions.shape[1]):
        axis = grid_axes[i]
        grid_min = axis[0]
        grid_max = axis[-1]
        n_points = axis.shape[0]
        index_coord = (positions[:, i] - grid_min) / (grid_max - grid_min) * (n_points - 1)
        valid_mask &= (positions[:, i] >= grid_min) & (positions[:, i] <= grid_max)
        indices.append(index_coord)

    coords = jnp.stack(indices, axis=0)  # shape: (dimensions, n_points)
    interpolated_values = map_coordinates(grid, coords, order=order, mode='constant', cval=0)
    return jnp.where(valid_mask, interpolated_values, jnp.nan)

def load_data(galaxy):
    #Loads the data for a given galaxy.

    L_bulge = bulge_lumins[bulge_lumins['Galaxy'] == galaxy]['Lbul'].values[0]
    selected_rows = mass_models[mass_models['ID'] == galaxy]

    rads = np.asarray(selected_rows['R'])  # kpc
    V_gas = np.asarray(selected_rows['Vgas'])
    V_disk = np.asarray(selected_rows['Vdisk'])
    V_bul = np.asarray(selected_rows['Vbul'])

    galaxy_data = galaxy_sample[galaxy_sample['Galaxy'] == galaxy]

    MH1_mean = galaxy_data['Total HI mass'].values[0] # 1e9 M_sun
    MH1_error = MH1_mean * 0.1 # 1e9 M_sun
    L_36_mean = galaxy_data['Total Luminosity at [3.6]'].values[0] # 1e9 L_sun
    L_36_error = galaxy_data['Luminosity Error'].values[0] # 1e9 L_sun
    Eff_radii = galaxy_data['Effective Radius at [3.6]'].values[0] # kpc
    dist = galaxy_sample['Distance'].values[0] # Mpc
    dist_err = galaxy_sample['Distance Error'].values[0] # Mpc

    return L_bulge, rads, V_gas, V_disk, V_bul, MH1_mean, MH1_error, L_36_mean, L_36_error, Eff_radii, dist, dist_err


def forward_model_btfr(alpha, scatter, x, nu, vmaxshift=False):

    print(f'\nRunning forward model with alpha={alpha}, scatter={scatter}, x={x}, nu={nu}...')

    # Load the N.Adams SMF data
    log_stellar_masses, SMF_data, _ = get_GSMF_ELPETRO(plotting=False)

    # Load the Uchuu halos 
    halos = np.load(HALOS_FILE)

    slope, intercept = get_x_cutoff_fit(halos, x)
    halos_selected = halos.copy()
    halos_selected = rfn.append_fields(halos_selected, 'select', np.ones(halos.shape[0], dtype=int), usemask=False)
    halo_log_Mvir = np.log10(halos_selected['Mvir'])
    halo_log_vmax = np.log10(halos_selected['vmax'])
    predicted_log_vmax = slope * halo_log_Mvir + intercept
    halos_selected['select'][halo_log_vmax > predicted_log_vmax] = np.nan

    proxy = proxies["mvir_proxy"](use_cache=False)
    abundance_match = AbundanceMatch(log_stellar_masses[10:], SMF_data[10:], halo_proxy=proxy, ext_range=(3.0, 12.0),
                                        boxsize=140, faint_end_first=True, scatter_mult=1, faint_end_slope=-0.42)
    
    theta = {"alpha": alpha, "scatter": scatter}  # AM model parameters
    deconv = abundance_match.deconvoluted_catalogs(theta, halos_selected)

    # Instead of loading a pre-trained interpolator, load grids and then build the jax-based callable.
    try:
        with open(CONTRA_GRIDS_FILE, "rb") as f:
            contra_grids = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Contra emulator grids were not found at {CONTRA_GRIDS_FILE}. Please run the grid-generation script first.")
        sys.exit(1)

    # Define grid axes matching the interpolation grid-generation stage.
    N_SAMPLES = 50  # must match the grid generation
    log_c_grid  = np.linspace(0, 3.9, N_SAMPLES)
    log_fb_grid = np.linspace(-3.6, -0.03, N_SAMPLES)
    log_rb_grid = np.linspace(-3, -1, N_SAMPLES)
    log_rf_grid = np.linspace(-4.8, 0.3, N_SAMPLES)
    grid_axes = [log_c_grid, log_fb_grid, log_rb_grid, log_rf_grid]

    if nu != 0.0:
        if nu in contra_grids:
            # Create a contra_interpolator that uses the JAX-based interpolation.
            contra_grid = contra_grids[nu]
            grid_jax = jnp.array(contra_grid, dtype=jnp.float64)
            def contra_interpolator(points):
                return jax_contra_interpolator(grid_jax, points, grid_axes)
        else:
            print(f"Contra emulator for nu={nu} is not available. Please verify that the grid-generation script includes nu={nu}.")
            sys.exit(1)
    else:
        contra_interpolator = None

    vels_global = np.empty((len(sparc_galaxy_list), N_AM_REALS, N_STELLAR_REALS))
    masses_global = np.empty((len(sparc_galaxy_list), N_AM_REALS, N_STELLAR_REALS))

    print('\nAbundance Matching Pipeline started...')

    for i in tqdm(range(N_AM_REALS), desc="AM Realizations"):

        mask, catalog_sc = abundance_match.add_scatter(deconv, cut_range=(3, 12), return_catalog=True)

        halo_proxy = np.log10(halos_selected['Mvir'])[mask]

        # Sort catalog_sc and get the sorting indices
        sorted_indices = np.argsort(catalog_sc)
        catalog_sc_sorted = catalog_sc[sorted_indices]

        for j, galaxy in enumerate(sparc_galaxy_list):

            L_bulge, rads, V_gas, V_disk, V_bul, MH1_mean, MH1_error, L_36_mean, L_36_error, Eff_rads, dist, dist_err = load_data(galaxy)

            dist_samples = np.random.normal(dist, dist_err, N_STELLAR_REALS)

            L_36_samples = np.random.normal(L_36_mean, L_36_error, N_STELLAR_REALS)
            
            log_M2L_disk_samples = np.random.normal(np.log10(M2L_DISK_MEAN), M2L_DISK_ERROR, N_STELLAR_REALS)
            log_M2L_bulge_samples = np.random.normal(np.log10(M2L_BULGE_MEAN), M2L_BULGE_ERROR, N_STELLAR_REALS)

            M2L_disk_samples = 10**log_M2L_disk_samples # M_sun / L_sun
            M2L_bulge_samples = 10**log_M2L_bulge_samples # M_sun / L_sun

            MH1_samples = np.random.normal(MH1_mean, MH1_error, N_STELLAR_REALS) * 1e9 # M_sun

            M_star_samples = np.abs((L_36_samples - L_bulge) * M2L_disk_samples + L_bulge * M2L_bulge_samples) * (dist / dist_samples)**2 * 1e9 * 0.7 # M_sun / h

            log_M_star_samples = np.log10(M_star_samples)

            # Find the indices of the closest matching halos in the sorted catalog_sc
            indices_sorted = np.searchsorted(catalog_sc_sorted, log_M_star_samples)

            # Clip indices to handle boundary conditions
            indices_sorted = np.clip(indices_sorted, 0, len(catalog_sc_sorted) - 1)

            # Map the sorted indices back to the original indices
            indices = sorted_indices[indices_sorted]

            matched_halos = halos_selected[indices]

            Mvir = matched_halos['Mvir'] / 0.7
            Rvir = matched_halos['Rvir'] / 0.7
            rs = matched_halos['rs'] / 0.7

            M_baryon = M_star_samples / 0.7 + 1.33 * MH1_samples

            if nu == 0.0:
                V_dm = nfw_circular_velocity(rads, Mvir, Rvir, rs)
            
            else:
                # the case of halo contraction/expansion, use contra emulator
                V_dm = nfw_circular_velocity_contra(rads, Eff_rads, Rvir, rs, Mvir, M_baryon, contra_interpolator)

            V_dm_max = np.max(V_dm, axis=1)

            V_max = np.sqrt(np.max(
                                    V_gas * np.abs(V_gas) +
                                    M2L_disk_samples[:, np.newaxis] * V_disk * np.abs(V_disk) +
                                    M2L_bulge_samples[:, np.newaxis] * V_bul * np.abs(V_bul) +
                                    V_dm * np.abs(V_dm),
                                    axis=1
                                )) # km/s

            V_max[V_dm_max == 0] = 0

            selection_mask = matched_halos['select']
            V_max *= selection_mask
            M_baryon *= selection_mask

            vels_global[j, i, :] = V_max 
            masses_global[j, i, :] = M_baryon

    print('Abundance Matching Pipeline finished!')
    print('\nCalculating log likelihood...')

    def filter_zeros(arr):
        #Filters out zeros from a 2D array row-wise, returning a list of arrays.
        return [row[row != np.nan] for row in arr]

    V_mocks_unlogged = filter_zeros(vels_global.reshape(len(galaxy_sample), -1))

    original_length = N_AM_REALS * N_STELLAR_REALS
    filtered_lengths = [len(row) for row in V_mocks_unlogged]
    print("Percentage of original array length retained after filtering out-of-bound points (per galaxy):")
    print([round((length / original_length) * 100, 2) for length in filtered_lengths])

    V_mocks = [np.log10(row) for row in V_mocks_unlogged]

    V_mock = np.array([np.mean(row) for row in V_mocks if len(row) > 0])
    V_mock_err = np.array([np.std(row) for row in V_mocks if len(row) > 0])

    V_obs_unlogged = np.array(sparc_btfr['Vmax'])
    V_obs_err_unlogged = np.array(sparc_btfr['e_Vmax'])

    V_obs = np.log10(V_obs_unlogged)
    V_obs_err = V_obs_err_unlogged / (V_obs_unlogged * np.log(10))

    if vmaxshift:
        shift = np.mean(V_obs) - np.mean(V_mock)
        V_mocks = [row + shift for row in V_mocks]
        V_mock += shift # Vmax shift

    sigmas = np.sqrt(V_mock_err**2 + V_obs_err**2)

    residuals = (V_obs - V_mock) / sigmas

    log_likelihood, log_likelihoods = get_loglike(V_mocks, V_obs, V_obs_err)

    print(f'Log likelihood: {log_likelihood}')

    M_mocks_unlogged = filter_zeros(masses_global.reshape(len(galaxy_sample), -1))
    M_mocks = [np.log10(row) for row in M_mocks_unlogged]

    M_mock = np.array([np.mean(row) for row in M_mocks])
    M_mock_err = np.array([np.std(row) for row in M_mocks])

    M_obs = np.array(sparc_btfr['log(Mb)'])
    M_obs_err = np.array(sparc_btfr['e_log(Mb)'])

    return V_mock, V_mock_err, M_mock, M_mock_err, V_obs, V_obs_err, M_obs, M_obs_err, halo_proxy, catalog_sc, residuals, log_likelihood, log_likelihoods


def btfr_plot(xmock, ymock, xmockerr, ymockerr, xobs, yobs, xobserr, yobserr, ax):
    xlabel = r'$\log_{10} \left( \frac{M_{\rm bar}}{{\rm \, M_{\odot}}} \right)$'
    ylabel = r'$\log_{10} \left( \frac{V_{\rm max}}{{\rm km \, s^{-1}}} \right)$'

    ax.errorbar(xmock, ymock, xerr=xmockerr, yerr=ymockerr, fmt='o',
                color='blue', markersize=3, alpha=0.7, capsize=2, elinewidth=0.4, 
                capthick=0.4, ecolor='black', label='Mock data')
    ax.errorbar(xobs, yobs, xerr=xobserr, yerr=yobserr, fmt='o',
                color='red', markersize=3, alpha=0.7, capsize=2, elinewidth=0.4, 
                capthick=0.4, ecolor='black', label='Obs data')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlim([7, 12])
    ax.set_ylim([1, 3])
    ax.legend()


def plot_SHMR_with_contours(ax, halo_proxy, stellar_mass, color, label):
    # Bin data
    bin_means, bin_edges, binnumber = binned_statistic(halo_proxy, stellar_mass, statistic='mean', bins=30)
    bin_std, _, _ = binned_statistic(halo_proxy, stellar_mass, statistic='std', bins=30)
    
    # Bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Plot mean and 1-sigma contours
    ax.plot(bin_centers, bin_means, color=color, label=label)
    ax.fill_between(bin_centers, bin_means - bin_std, bin_means + bin_std, color=color, alpha=0.3)


def plot_SHMR_with_contours_quantile(ax, halo_proxy, stellar_mass, color, label):

    # Define the quantiles you want to calculate
    quantiles = [0.025, 0.16, 0.50, 0.84, 0.975]
    
    # Create a dictionary to hold the quantile results
    bin_quantiles = {}

    # Loop over each quantile and calculate it separately
    for q in quantiles:
        # Define a custom function for the current quantile
        def quantile_func(x):
            return np.quantile(x, q)
        
        # Bin data using the current quantile function
        bin_result, bin_edges, binnumber = binned_statistic(stellar_mass, halo_proxy, statistic=quantile_func, bins=30)
        
        # Store the result in the dictionary
        bin_quantiles[q] = bin_result

    # Bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Extract specific quantiles from the dictionary
    q2p5 = bin_quantiles[0.025]
    q16 = bin_quantiles[0.16]
    q50 = bin_quantiles[0.50]
    q84 = bin_quantiles[0.84]
    q97p5 = bin_quantiles[0.975]
    
    # Plot the median (50th percentile)
    ax.plot(q50, bin_centers, color=color, label=label)
    
    # Fill between the quantiles to show non-Gaussian spread
    ax.fill_betweenx(bin_centers, q16, q84, color=color, alpha=0.3)  # 68% interval
    ax.fill_betweenx(bin_centers, q2p5, q97p5, color=color, alpha=0.15)  # 95% interval


def scatter_residuals_vs_mocks(ax, Mmocks, residuals, color, label):
    ax.scatter(Mmocks, residuals, color=color, s=10, alpha=0.7, label=label)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel(r'$\log_{10}(M_{\rm bar})$', fontsize=12)
    ax.set_ylabel('Residuals', fontsize=12)
    ax.set_xlim([7, 12])
    ax.set_title('Residuals vs Mbar', fontsize=16)


def loglikehoods_vs_mocks(ax, Mmocks, log_likelihoods, color):
    ax.scatter(Mmocks, log_likelihoods, color=color, s=10, alpha=0.7)
    ax.set_xlabel(r'$\log_{10}(M_{\rm bar})$', fontsize=12)
    ax.set_ylabel('Delta log likelihood', fontsize=12)
    ax.set_xlim([7, 12])
    ax.set_title('Delta log likelihoods vs Mbar', fontsize=16)


# Create the figure and axes for the panel
fig, axs = plt.subplots(3, 2, figsize=(14, 15))

# Plot for alpha1, scatter1, nu1
V_mock_1, V_mock_err_1, M_mock_1, M_mock_err_1, V_obs_1, V_obs_err_1, M_obs_1, M_obs_err_1, halo_proxy_1, catalog_1, residuals_1, log_likelihood_1, log_likelihoods_1 = forward_model_btfr(alpha=ALPHA_1, scatter=SCATTER_1, x=X_1, nu=NU_1, vmaxshift=VMAXSHIFT)
btfr_plot(M_mock_1, V_mock_1, M_mock_err_1, V_mock_err_1, M_obs_1, V_obs_1, M_obs_err_1, V_obs_err_1, axs[0, 0])
if VMAXSHIFT:
    axs[0, 0].set_title(fr'BTFR with $\alpha={ALPHA_1:.2f}$, $\sigma={SCATTER_1}$, x={X_1}, $\nu={NU_1:.2f}$, VS, Loglike {log_likelihood_1:.2f}', fontsize=16)
else:
    axs[0, 0].set_title(fr'BTFR with $\alpha={ALPHA_1:.2f}$, $\sigma={SCATTER_1}$, x={X_1}, $\nu={NU_1:.2f}$, Loglike {log_likelihood_1:.2f}', fontsize=16)

# Plot for alpha2, scatter2, nu2
V_mock_2, V_mock_err_2, M_mock_2, M_mock_err_2, V_obs_2, V_obs_err_2, M_obs_2, M_obs_err_2, halo_proxy_2, catalog_2, residuals_2, log_likelihood_2, log_likelihoods_2 = forward_model_btfr(alpha=ALPHA_2, scatter=SCATTER_2, x=X_2, nu=NU_2, vmaxshift=VMAXSHIFT)
btfr_plot(M_mock_2, V_mock_2, M_mock_err_2, V_mock_err_2, M_obs_2, V_obs_2, M_obs_err_2, V_obs_err_2, axs[0, 1])
if VMAXSHIFT:
    axs[0, 1].set_title(fr'BTFR with $\alpha={ALPHA_2:.2f}$, $\sigma={SCATTER_2}$, x={X_2}, $\nu={NU_2:.2f}$, VS, Loglike {log_likelihood_2:.2f}', fontsize=16)
else:
    axs[0, 1].set_title(fr'BTFR with $\alpha={ALPHA_2:.2f}$, $\sigma={SCATTER_2}$, x={X_2}, $\nu={NU_2:.2f}$, Loglike {log_likelihood_2:.2f}', fontsize=16)

# SHMR Plot
plot_SHMR_with_contours_quantile(axs[1, 0], halo_proxy_1, catalog_1, color='purple', label=fr'$\alpha={ALPHA_1:.2f}$, $\sigma={SCATTER_1}$, x={X_1}, $\nu={NU_1:.2f}$')
plot_SHMR_with_contours_quantile(axs[1, 0], halo_proxy_2, catalog_2, color='orange', label=fr'$\alpha={ALPHA_2:.2f}$, $\sigma={SCATTER_2}$, x={X_2}, $\nu={NU_2:.2f}$')
axs[1, 0].set_ylabel(r'$\log_{10}(M_*/M_h)$', fontsize=12) 
axs[1, 0].set_xlabel(r'$\log_{10}(M_h (M_\odot))$', fontsize=12) 
axs[1, 0].set_title("Stellar-to-Halo Mass Relations", fontsize=16)
axs[1, 0].set_xlim([10, 15])
axs[1, 0].set_ylim([7, 12])
axs[1, 0].legend()

# Plot Residuals vs Mmocks
scatter_residuals_vs_mocks(axs[1, 1], M_mock_1, residuals_1, color='purple', label=fr'$\alpha={ALPHA_1:.2f}$, $\sigma={SCATTER_1}$, x={X_1}, $\nu={NU_1:.2f}$')
scatter_residuals_vs_mocks(axs[1, 1], M_mock_2, residuals_2, color='orange', label=fr'$\alpha={ALPHA_2:.2f}$, $\sigma={SCATTER_2}$, x={X_2}, $\nu={NU_2:.2f}$')

# Plot Loglikes vs Mocks
delta_loglike = log_likelihoods_2 - log_likelihoods_1
loglikehoods_vs_mocks(axs[2, 0], M_mock_1, delta_loglike, color='red')

plt.tight_layout()
if VMAXSHIFT:
    plt.savefig('/Users/fedorboreiko/Documents/Oxford/btfr_z/plots/btfr_panel_vmaxshift.png', dpi=300)
else:
    plt.savefig('/Users/fedorboreiko/Documents/Oxford/btfr_z/plots/btfr_panel.png', dpi=300)
    plt.show()