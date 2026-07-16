import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from scipy.stats import binned_statistic

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern']
rcParams['text.usetex'] = True

xlabel = r'$\log_{10} \left( V_{\rm max} \, \left[ \mathrm{km \, s^{-1}} \right] \right)$'
ylabel = r'$\log_{10} \left( M_{\rm bar} \, \left[ \mathrm{M}_{\odot} \right] \right)$'

def btfr_plot(alpha, sigma, x, nu, vmaxshift, loglike, xsim, ysim, xsimerr, ysimerr, 
              xobs, yobs, xobserr, yobserr, plotname=None, ax=None):
    # Determine if we need to create our own figure
    created_figure = ax is None
    
    if created_figure:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.errorbar(xsim, ysim, xerr=xsimerr, yerr=ysimerr, fmt='o',
                color='blue', markersize=3, alpha=0.7, capsize=2, elinewidth=0.4, 
                capthick=0.4, ecolor='black', label='Mock data')
    ax.errorbar(xobs, yobs, xerr=xobserr, yerr=yobserr, fmt='o',
                color='red', markersize=3, alpha=0.7, capsize=2, elinewidth=0.4, 
                capthick=0.4, ecolor='black', label='Obs data')
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlim([1, 3])
    ax.set_ylim([7, 12])
    ax.legend()

    title = rf'BTFR: $\alpha={alpha:.2f}$, $\sigma={sigma:.2f}$, x = {x:.2f}, $\nu={nu:.2f}$, $\log \mathcal{{L}}={loglike:.2f}$'
    if vmaxshift:
        title += ' (Vmax shift)'
    ax.set_title(title, fontsize=14)

    # Handle saving and displaying only if we created the figure
    if created_figure:
        if plotname is not None:
            fig.savefig(plotname, dpi=300)
            print(f"Plot saved to {plotname}")
        else:
            # If no filename provided, show the plot
            plt.show()
        plt.close(fig)  # Clean up the figure
    # If ax was provided (panel plot), do nothing - parent function handles saving

def vels_hist(vel_sims, veldm_sims, vel_sim_mean, vel_sim_std, vel_obs_mean, vel_obs_err, mbar, indiv_loglike, output_filename):
    # Filter out NaN values from the simulation data
    vel_sims_clean = vel_sims[~np.isnan(vel_sims)]
    veldm_sims_clean = veldm_sims[~np.isnan(veldm_sims)]
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # histogram for mock velocity data (without scaling) - using cleaned data
    n, bins, patches = ax.hist(vel_sims_clean, bins=100, color='red', alpha=0.5, density=False, label=r'Predicted $V_{\rm max}$')
    
    # Gaussian curve for the observed data
    x = np.linspace(vel_obs_mean - 5*vel_obs_err, vel_obs_mean + 5*vel_obs_err, 200)
    # Calculate the normalized Gaussian (probability density)
    gaussian_pdf = (1 / (vel_obs_err * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - vel_obs_mean) / vel_obs_err)**2)
    
    # Scale the Gaussian to match the histogram height
    # Use the maximum histogram count as the scaling reference
    max_hist_count = max(n)
    gaussian_scale_factor = max_hist_count / max(gaussian_pdf)
    y_scaled = gaussian_pdf * gaussian_scale_factor
    
    # plotting the scaled Gaussian curve
    ax.plot(x, y_scaled, color='orange', linewidth=2, alpha=0.8, label=r'$V_{\rm obs}$ Distribution (Gaussian)')
    
    # plotting vertical lines for the observed mean and observed error
    ax.axvline(vel_obs_mean, color='orange', linestyle='--', linewidth=2, label=r'$V_{\rm obs}$')
    
    ax.set_xlabel(r'$\log_{10} \left( V \, \left[ \mathrm{km \, s^{-1}} \right] \right)$', fontsize=16)
    ax.set_ylabel('Count', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_title(r'Likelihood Evaluation: Galaxy Mass $%.4g \, \mathrm{M}_{\odot}^{10}$' % mbar + r', Log Likelihood: $%.3g$' % indiv_loglike)
    ax.legend(fontsize=16)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()


def scatter_plot(x, y, xlabel, ylabel, title=None, color='red', output_filename=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    ax.scatter(x, y, s=10, color=color, alpha=0.7)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    if title is not None:
        ax.set_title(title, fontsize=14)
    
    # Save the plot if a filename is provided and no axes object is passed
    if output_filename is not None and ax is None:
        plt.savefig(output_filename, dpi=300)
        plt.close(fig)


def plot_SHMR_with_contours(ax, halo_proxy, stellar_mass, color, label):
    # Bin data
    bin_means, bin_edges, binnumber = binned_statistic(np.asarray(halo_proxy).flatten(), np.asarray(stellar_mass).flatten(), statistic='mean', bins=30)
    bin_std, _, _ = binned_statistic(np.asarray(halo_proxy).flatten(), np.asarray(stellar_mass).flatten(), statistic='std', bins=30)
    
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

def delta_loglike_correlation_plot(quantity, delta_loglike, quantity_label, plotname=None, ax=None):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        
    ax.scatter(quantity, delta_loglike, s=10, alpha=0.7, color='blue')
    mean_delta_loglike = np.mean(delta_loglike)
    ax.axhline(mean_delta_loglike, color='red', linestyle='--', label='Mean $\Delta \log \mathcal{L}$')
    ax.set_xlabel(quantity_label, fontsize=16)
    ax.set_ylabel(r'$\Delta \log \mathcal{L}$', fontsize=16)

    if plotname is not None:
        plt.savefig(plotname, dpi=300)
        plt.close(fig)


def create_btfr_panel_plot(model_1_config, model_1_output, model_2_config, model_2_output, save_path=None):
    """
    Create a comprehensive BTFR panel plot from two model outputs.

    Parameters:
    -----------
    model_1_config : tuple
        Configuration parameters for the first model (alpha, scatter, nu).
    model_1_output : tuple
        Output data from the first model (V_mock, V_mock_err, M_mock, M_mock_err, 
        V_obs, V_obs_err, M_obs, M_obs_err, halo_proxy, catalog, residuals, 
        log_likelihood, log_likelihoods).
    model_2_config : tuple
        Configuration parameters for the second model (alpha, scatter, nu).
    model_2_output : tuple
        Output data from the second model (V_mock, V_mock_err, M_mock, M_mock_err, 
        V_obs, V_obs_err, M_obs, M_obs_err, halo_proxy, catalog, residuals, 
        log_likelihood, log_likelihoods).
    save_path : str, optional
        Path to save the figure. If None, uses default naming.

    Returns:
    --------
    fig, axs : matplotlib figure and axes objects
    """
    
    # Create the figure and axes for the panel
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # Unpack model_1_output
    _, _, _, V_sim_1, V_sim_err_1, M_bar_sim_1, M_bar_sim_err_1, \
    V_obs, V_obs_err, M_bar_obs, M_bar_obs_err, \
    AM_catalog_Mstar_1, AM_catalog_Mvir_1, residuals_1, \
    log_likelihood_1, inidividual_log_likelihoods_1 = model_1_output
    
    # Unpack model_2_output
    _, _, _, V_sim_2, V_sim_err_2, M_bar_sim_2, M_bar_sim_err_2, \
    _, _, _, _, \
    AM_catalog_Mstar_2, AM_catalog_Mvir_2, residuals_2, \
    log_likelihood_2, inidividual_log_likelihoods_2 = model_2_output
    
    # Plot BTFR for first parameter set
    btfr_plot(np.tan(model_1_config.alpha_proxy_value),
                model_1_config.scatter_value,
                model_1_config.x_value,
                model_1_config.nu_value,
                model_1_config.vmax_shift_mode,
                log_likelihood_1,
                V_sim_1, M_bar_sim_1, V_sim_err_1, M_bar_sim_err_1,
                V_obs, M_bar_obs, V_obs_err, M_bar_obs_err,
                plotname=None, ax=axs[0, 0])
    
    # Plot BTFR for second parameter set
    btfr_plot(np.tan(model_2_config.alpha_proxy_value),
                model_2_config.scatter_value,
                model_2_config.x_value,
                model_2_config.nu_value,
                model_2_config.vmax_shift_mode,
                log_likelihood_2,
                V_sim_2, M_bar_sim_2, V_sim_err_2, M_bar_sim_err_2,
                V_obs, M_bar_obs, V_obs_err, M_bar_obs_err,
                plotname=None, ax=axs[0, 1])
    
    # SHMR Plot for first model
    plot_SHMR_with_contours(axs[1, 0], AM_catalog_Mvir_1, AM_catalog_Mstar_1, 
                            color='purple', 
                            label=rf'$\alpha={model_1_config.alpha_proxy_value:.2f}$, $\sigma={model_1_config.scatter_value:.2f}$')
    # SHMR Plot for second model
    plot_SHMR_with_contours(axs[1, 0], AM_catalog_Mvir_2, AM_catalog_Mstar_2, 
                            color='orange', 
                            label=rf'$\alpha={model_2_config.alpha_proxy_value:.2f}$, $\sigma={model_2_config.scatter_value:.2f}$')
    
    axs[1, 0].set_ylabel(r'$\log_{10}(M_*/M_h)$', fontsize=12)
    axs[1, 0].set_xlabel(r'$\log_{10}(M_h (M_\odot))$', fontsize=12)
    axs[1, 0].set_title("Stellar-to-Halo Mass Relations", fontsize=16)
    axs[1, 0].set_xlim([10, 15])
    axs[1, 0].set_ylim([7, 12])
    axs[1, 0].legend()
    
    """# Plot Residuals vs Mmocks
    scatter_plot(M_bar_sim_1, residuals_1, 
                r'$\log_{10}(M_{\rm bar})$', 'Residuals', 
                title=None, color='red',
                output_filename=None, ax=axs[1, 1])
    scatter_plot(M_bar_sim_2, residuals_2,
                r'$\log_{10}(M_{\rm bar})$', 'Residuals', 
                title=None, color='blue', 
                output_filename=None, ax=axs[1, 1])"""
    
    # Plot Delta log-likelihoods vs Mocks
    delta_loglike = inidividual_log_likelihoods_2 - inidividual_log_likelihoods_1
    delta_loglike_correlation_plot(M_bar_sim_1, delta_loglike, 
                                   r'$\log_{10}(M_{\rm bar})$', 
                                   plotname=None, ax=axs[1, 1])
    
    # You can add more plots to axs[2, 1] if needed
    axs[1, 1].axis('off')  # Turn off empty subplot
    
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = "btfr_panel_plot.png"
    
    plt.savefig(save_path, dpi=300)
