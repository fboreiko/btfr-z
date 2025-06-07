import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from scipy.stats import binned_statistic

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern']
rcParams['text.usetex'] = True

xlabel = r'$\log_{10} \left( \frac{V_{\rm max}}{{\rm km \, s^{-1}}} \right)$'
ylabel = r'$\log_{10} \left( \frac{M_{\rm bar}}{{\rm \, M_{\odot}}} \right)$'

def btfr_plot(alpha, sigma, x, nu, loglike, xsim, ysim, xsimerr, ysimerr, xobs, yobs, xobserr, yobserr, plotname):
    fig, ax = plt.subplots()
    ax.errorbar(xsim, ysim, xerr=xsimerr, yerr=ysimerr, fmt='o',
            color='blue', markersize=3.5, capsize=2, elinewidth=0.5, 
            capthick=0.5, ecolor='black', label='Mock data')
    ax.errorbar(xobs, yobs, xerr=xobserr, yerr=yobserr, fmt='o',
            color='red', markersize=3.5, capsize=2, elinewidth=0.5, 
            capthick=0.5, ecolor='black', label='Obs data')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(r'BTFR, Log Likelihood: {:.3f}, $\alpha$: {:.3f}, $\sigma$: {:.3f}, $x$: {:.3f}, $\nu$: {:.3f}'.format(loglike, alpha, sigma, x, nu))
    ax.set_xlim([1, 3])
    ax.set_ylim([7, 12])
    ax.legend()
    plt.savefig(plotname, dpi=300)

def vels_hist(vel_sims, veldm_sims, vel_sim_mean, vel_sim_std, vel_obs_mean, vel_obs_err, mbar, indiv_loglike, output_filename):
    # Filter out NaN values from the simulation data
    vel_sims_clean = vel_sims[~np.isnan(vel_sims)]
    veldm_sims_clean = veldm_sims[~np.isnan(veldm_sims)]
    
    # Gaussian curve for the mock data
    x = np.linspace(vel_sim_mean - 5*vel_sim_std, vel_sim_mean + 5*vel_sim_std, 200)
    y = (1 / (vel_sim_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - vel_sim_mean) / vel_sim_std)**2)
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # histogram for mock velocity data (without scaling) - using cleaned data
    n, bins, patches = ax.hist(vel_sims_clean, bins=100, color='red', alpha=0.5, density=False, label='Mock Velocity')
    
    # histograms for NFW velocities (without scaling) - using cleaned data
    veldm_n, veldm_bins = np.histogram(veldm_sims_clean, bins=100, density=False)
    
    # calculating scaling factors to match the peak of the simulated velocity data
    veldm_scale_factor = max(n) / max(veldm_n) if max(veldm_n) > 0 else 1
    
    # plotting the scaled NFW velocity histogram - using cleaned data
    ax.hist(veldm_sims_clean, bins=100, color='blue', alpha=0.4, 
            weights=np.ones_like(veldm_sims_clean) * veldm_scale_factor, label='DM velocity')
    
    # plotting the normalized Gaussian curve
    ax.plot(x, y, color='red', alpha=0.7, label='Gaussian Fit to Mock Data')
    
    # plotting mean and standard deviation lines for the data
    ax.axvline(vel_sim_mean, color='red', linestyle='--', label='Mean')
    ax.axvline(vel_sim_mean + vel_sim_std, color='red', linestyle=':', label='1 sigma')
    ax.axvline(vel_sim_mean - vel_sim_std, color='red', linestyle=':')
    
    # plotting vertical lines for the observed mean and observed error
    ax.axvline(vel_obs_mean, color='orange', linestyle='--', label='Observed Mean')
    ax.axvline(vel_obs_mean + vel_obs_err, color='orange', linestyle=':', label='Observed +1 sigma')
    ax.axvline(vel_obs_mean - vel_obs_err, color='orange', linestyle=':')
    
    ax.set_xlabel('Velocity')  # Note: xlabel variable wasn't defined in original code
    ax.set_ylabel('Normalized Frequency')
    ax.set_title(f'Galaxy of mass {mbar:.4g}, Log Likelihood: {indiv_loglike:.3g}')
    ax.legend()
    
    plt.savefig(output_filename)
    plt.close()


def scatter_plot(x, y, xlabel, ylabel, title, output_filename):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.scatter(x, y, s=0.01, color='red', alpha=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.savefig(output_filename)
    plt.close()


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


def mstellar_hists_comparison(mstellar_sims_1, m_stellar_sims_2, x_1, x_2, plotname):
        fig, ax = plt.subplots()
        # Filter out NaN values from the simulation data
        mstellar_sims_1 = mstellar_sims_1[~np.isnan(mstellar_sims_1)]
        m_stellar_sims_2 = m_stellar_sims_2[~np.isnan(m_stellar_sims_2)]

        # Plot histogram for mstellar_sims_1
        n1, bins1, patches1 = ax.hist(mstellar_sims_1, bins=100, color='blue', alpha=0.5, density=True, label=f'Model 1, x = ({x_1})')

        # Plot histogram for m_stellar_sims_2
        n2, bins2 = np.histogram(m_stellar_sims_2, bins=100, density=False)

        scale_factor = max(n1) / max(n2)
        n2_scaled = n2 * scale_factor
        ax.hist(bins2[:-1], bins=bins2, weights=n2_scaled, color='red', alpha=0.5, label= f'Model 2, x = ({x_2})')

        # Set labels and title
        ax.set_xlabel(r'$\log_{10}(M_{\rm stellar})$', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Comparison of Stellar Mass Distributions', fontsize=16)
        ax.legend()
        plt.savefig(plotname, dpi=300)
        plt.close(fig)
