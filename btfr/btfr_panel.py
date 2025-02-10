import sys
sys.path.append('/Users/fedorboreiko/Documents/Oxford/btfr_z')

import numpy as np
import pandas as pd
from btfr.massfuncs import get_GSMF_ELPETRO
from BAM import AbundanceMatch, proxies
from btfr.btfr_plotting import btfr_plot
from btfr.btfr_utils import nfw_circular_velocity_from_mhi, nfw_circular_velocity, get_loglike
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import binned_statistic
import pickle

# Set the font to Computer Modern (LaTeX default) and enable LaTeX rendering
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern']
rcParams['text.usetex'] = True

N_AM_REALS = 10
N_STELLAR_REALS = 500

SCATTER_1 = 0.01
ALPHA_1 = -10
NU_1 = 0.0
SCATTER_2 = 0.01
ALPHA_2 = -100
NU_2 = 0.0
VMAXSHIFT = False

M2L_DISK_MEAN = 0.5
M2L_DISK_ERROR = 0.2 # dex

M2L_BULGE_MEAN = 0.7
M2L_BULGE_ERROR = 0.2 # dex

# Pre-load data
bulge_lumins = pd.read_csv('Tabular_data/Bulge_lum_table.csv')
mass_models = pd.read_csv('Tabular_data/Mass_models_table.csv')
galaxy_sample = pd.read_csv('Tabular_data/Gal_sample_table.csv')
sparc_btfr = pd.read_csv('Tabular_data/sparc_btfr.csv')

sparc_galaxy_list = sparc_btfr['Name']

bulge_lumins = bulge_lumins[bulge_lumins['Galaxy'].isin(sparc_galaxy_list)]
mass_models = mass_models[mass_models['ID'].isin(sparc_galaxy_list)]
galaxy_sample = galaxy_sample[galaxy_sample['Galaxy'].isin(sparc_galaxy_list)]

# Wrapper class around original interpolator 
class SafeInterpolator:
    def __init__(self, interpolator):
        self.interpolator = interpolator
        self.bounds = np.array([[0.01, 3],    #c range. Full range is [0, 3.9]
                                [-3.3, -0.3], #fb range. Full range is [-3.6, -0.03]
                                [-2.9, -1.3], #rb range. Full range is [-3, -1]
                                [-4.8, 0.3]]) #rf range. Full range is [-4.8, 0.3]
    def __call__(self, points):
        results = np.zeros(points.shape[0])  # Initialize results with zeros matching the number of input points
        valid_mask = np.all(
            (points >= self.bounds[:, 0]) & (points <= self.bounds[:, 1]), axis=1
        )
        valid_points = points[valid_mask]

        if valid_points.size > 0:
            valid_results = self.interpolator(valid_points)
            results[valid_mask] = valid_results

        return results
    

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


def forward_model_btfr(alpha, scatter, nu, vmaxshift=False):

    print(f'\nRunning forward model with alpha={alpha}, scatter={scatter}, nu={nu}...')

    # Load the N.Adams SMF data
    log_stellar_masses, SMF_data, _ = get_GSMF_ELPETRO(plotting=False)

    # Load the Uchuu halos 
    halos = np.load("/Users/fedorboreiko/Documents/Oxford/Personal_codes/Codebase/halos_z_0p00.npy")

    proxy = proxies["mvir_proxy"]()

    abundance_match = AbundanceMatch(log_stellar_masses[10:], SMF_data[10:], halo_proxy=proxy, ext_range=(3.0, 12.0),
                                        boxsize=140, faint_end_first=True, scatter_mult=1, faint_end_slope=-0.42)
    
    theta = {"alpha": alpha, "scatter": scatter}  # Will be tuned?
    deconv = abundance_match.deconvoluted_catalogs(theta, halos)

    # Load contra interpolators
    with open("/Users/fedorboreiko/Documents/Oxford/Personal_codes/Codebase/contra_emulators/contra_interpolators.pkl", "rb") as f:
        interpolators = pickle.load(f)

    if nu in interpolators:
        interpolator = SafeInterpolator(interpolators[nu])
    else:
        raise ValueError(f"Interpolator for nu={nu} not found.")

    vels_global = np.empty((len(sparc_galaxy_list), N_AM_REALS, N_STELLAR_REALS))
    masses_global = np.empty((len(sparc_galaxy_list), N_AM_REALS, N_STELLAR_REALS))

    print('\nAbundance Matching Pipeline started...')

    for i in tqdm(range(N_AM_REALS), desc="AM Realizations"):

        mask, catalog_sc = abundance_match.add_scatter(deconv, cut_range=(3, 12), return_catalog=True)

        halo_proxy = np.log10(halos['Mvir'])[mask]

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

            matched_halos = halos[indices]

            Mvir = matched_halos['Mvir'] / 0.7
            Rvir = matched_halos['Rvir'] / 0.7
            rs = matched_halos['rs'] / 0.7

            M_baryon = M_star_samples / 0.7 + 1.33 * MH1_samples

            if nu == 0.0:
                V_dm = nfw_circular_velocity(rads, Mvir, Rvir, rs)
            
            else:
                rb = Eff_rads / 1.67835
                c = Rvir / rs
                fb = M_baryon / (Mvir + M_baryon)
                rb_uless = rb / Rvir
                rads_uless = rads[np.newaxis, :] / Rvir[:, np.newaxis]

                # extend c, fb, rb, ri to match the shape of ri
                logc_extended = np.repeat(np.log10(c)[:, np.newaxis], rads_uless.shape[1], axis=1)
                logfb_extended = np.repeat(np.log10(fb)[:, np.newaxis], rads_uless.shape[1], axis=1)
                logrb_extended = np.repeat(np.log10(rb_uless)[:, np.newaxis], rads_uless.shape[1], axis=1)

                # flatten the grids and data
                points = np.vstack((logc_extended.ravel(), logfb_extended.ravel(), logrb_extended.ravel(), np.log10(rads_uless).ravel())).T

                # emulate the data
                logmhi = interpolator(points)
                logmhi = logmhi.reshape(rads_uless.shape)
                logmhi[logmhi == 0] = -np.inf

                mhi = 10**logmhi

                V_dm = nfw_circular_velocity_from_mhi(rads, mhi, fb, Mvir)

            V_dm_max = np.max(V_dm, axis=1)

            V_max = np.sqrt(np.max(
                                    V_gas * np.abs(V_gas) +
                                    M2L_disk_samples[:, np.newaxis] * V_disk * np.abs(V_disk) +
                                    M2L_bulge_samples[:, np.newaxis] * V_bul * np.abs(V_bul) +
                                    V_dm * np.abs(V_dm),
                                    axis=1
                                )) # km/s

            V_max[V_dm_max == 0] = 0

            vels_global[j, i, :] = V_max 
            masses_global[j, i, :] = M_baryon

    print('Abundance Matching Pipeline finished!')
    print('\nCalculating log likelihood...')

    V_mocks_unlogged = [row[row != 0] for row in vels_global.reshape(len(galaxy_sample), -1)]

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

    M_mocks = np.log10(masses_global.reshape(len(sparc_galaxy_list), -1))

    M_mock = np.mean(M_mocks, axis=1)
    M_mock_err = np.std(M_mocks, axis=1)

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
V_mock_1, V_mock_err_1, M_mock_1, M_mock_err_1, V_obs_1, V_obs_err_1, M_obs_1, M_obs_err_1, halo_proxy_1, catalog_1, residuals_1, log_likelihood_1, log_likelihoods_1 = forward_model_btfr(alpha=ALPHA_1, scatter=SCATTER_1, nu=NU_1, vmaxshift=VMAXSHIFT)
btfr_plot(M_mock_1, V_mock_1, M_mock_err_1, V_mock_err_1, M_obs_1, V_obs_1, M_obs_err_1, V_obs_err_1, axs[0, 0])
if VMAXSHIFT:
    axs[0, 0].set_title(fr'BTFR with $\alpha={ALPHA_1}$, $\sigma={SCATTER_1}$, $\nu={NU_1}$, Vmax shift, Loglike {log_likelihood_1:.2f}', fontsize=16)
else:
    axs[0, 0].set_title(fr'BTFR with $\alpha={ALPHA_1}$, $\sigma={SCATTER_1}$, $\nu={NU_1}$, Loglike {log_likelihood_1:.2f}', fontsize=16)

# Plot for alpha2, scatter2, nu2
V_mock_2, V_mock_err_2, M_mock_2, M_mock_err_2, V_obs_2, V_obs_err_2, M_obs_2, M_obs_err_2, halo_proxy_2, catalog_2, residuals_2, log_likelihood_2, log_likelihoods_2 = forward_model_btfr(alpha=ALPHA_2, scatter=SCATTER_2, nu=NU_2, vmaxshift=VMAXSHIFT)
btfr_plot(M_mock_2, V_mock_2, M_mock_err_2, V_mock_err_2, M_obs_2, V_obs_2, M_obs_err_2, V_obs_err_2, axs[0, 1])
if VMAXSHIFT:
    axs[0, 1].set_title(fr'BTFR with $\alpha={ALPHA_2}$, $\sigma={SCATTER_2}$, $\nu={NU_2}$, Vmax shift, Loglike {log_likelihood_2:.2f}', fontsize=16)
else:
    axs[0, 1].set_title(fr'BTFR with $\alpha={ALPHA_2}$, $\sigma={SCATTER_2}$, $\nu={NU_2}$, Loglike {log_likelihood_2:.2f}', fontsize=16)

# SHMR Plot
plot_SHMR_with_contours_quantile(axs[1, 0], halo_proxy_1, catalog_1, color='purple', label=fr'$\alpha={ALPHA_1}$, $\sigma={SCATTER_1}$, $\nu={NU_1}$')
plot_SHMR_with_contours_quantile(axs[1, 0], halo_proxy_2, catalog_2, color='orange', label=fr'$\alpha={ALPHA_2}$, $\sigma={SCATTER_2}$, $\nu={NU_2}$')
axs[1, 0].set_ylabel(r'$\log_{10}(M_*/M_h)$', fontsize=12) 
axs[1, 0].set_xlabel(r'$\log_{10}(M_h (M_\odot))$', fontsize=12) 
axs[1, 0].set_title("Stellar-to-Halo Mass Relations", fontsize=16)
axs[1, 0].set_xlim([10, 15])
axs[1, 0].set_ylim([7, 12])
axs[1, 0].legend()

# Plot Residuals vs Mmocks
scatter_residuals_vs_mocks(axs[1, 1], M_mock_1, residuals_1, color='purple', label=fr'$\alpha={ALPHA_1}$, $\sigma={SCATTER_1}$, $\nu={NU_1}$')
scatter_residuals_vs_mocks(axs[1, 1], M_mock_2, residuals_2, color='orange', label=fr'$\alpha={ALPHA_2}$, $\sigma={SCATTER_2}$, $\nu={NU_2}$')

# Plot Loglikes vs Mocks
delta_loglike = log_likelihoods_2 - log_likelihoods_1
loglikehoods_vs_mocks(axs[2, 0], M_mock_1, delta_loglike, color='red')

plt.tight_layout()
if VMAXSHIFT:
    plt.savefig('/Users/fedorboreiko/Documents/Oxford/btfr_z/plots/btfr_panel_vmaxshift.png', dpi=300)
else:
    plt.savefig('/Users/fedorboreiko/Documents/Oxford/btfr_z/plots/btfr_panel.png', dpi=300)