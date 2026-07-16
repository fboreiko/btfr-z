import sys
import os
sys.path.append('/Users/fedorboreiko/Documents/Oxford/btfr_z')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import binned_statistic

# Set the font to Computer Modern (LaTeX default) and enable LaTeX rendering
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern']
rcParams['text.usetex'] = True
rcParams['font.size'] = 14
rcParams['axes.linewidth'] = 1.5
rcParams['xtick.major.width'] = 1.5
rcParams['ytick.major.width'] = 1.5
rcParams['xtick.major.size'] = 6
rcParams['ytick.major.size'] = 6
rcParams['xtick.minor.size'] = 3
rcParams['ytick.minor.size'] = 3
rcParams['legend.frameon'] = True
rcParams['legend.framealpha'] = 0.9
rcParams['legend.fontsize'] = 11


def btfr_plot(xmock, ymock, xmockerr, ymockerr, xobs, yobs, xobserr, yobserr, ax, model_label):
    xlabel = r'$\log(M_{\rm bar}/M_{\odot})$'
    ylabel = r'$\log(V_{\rm max}/{\rm km\,s^{-1}})$'

    ax.errorbar(xobs, yobs, xerr=xobserr, yerr=yobserr, fmt='s',
                color='blue', markersize=4, alpha=0.8, capsize=2, 
                elinewidth=0.4, capthick=0.4, ecolor='grey', label='SPARC BTFR', zorder=2)
    
    ax.errorbar(xmock, ymock, xerr=xmockerr, yerr=ymockerr, fmt='o',
                color='red', markersize=4, alpha=0.8, capsize=2, 
                elinewidth=0.4, capthick=0.4, ecolor='grey', label='Predicted BTFR', zorder=3)
    
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xlim([7.5, 11.7])
    ax.set_ylim([1.1, 2.75])
    ax.tick_params(labelsize=14)
    #ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.legend(loc='upper left', fontsize=14, framealpha=0.95)
    
    # Add model label
    ax.text(0.95, 0.05, model_label, transform=ax.transAxes, 
            fontsize=14, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def plot_SHMR_with_contours_quantile(ax, halo_proxy, stellar_mass, color, label, linestyle='-'):
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
        bin_result, bin_edges, binnumber = binned_statistic(
            stellar_mass, halo_proxy, statistic=quantile_func, bins=30
        )
        
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
    ax.plot(q50, bin_centers, color=color, label=label, linewidth=2.5, linestyle=linestyle, zorder=3)
    
    # Fill between the quantiles to show non-Gaussian spread
    ax.fill_betweenx(bin_centers, q16, q84, color=color, alpha=0.25, zorder=1)
    ax.fill_betweenx(bin_centers, q2p5, q97p5, color=color, alpha=0.12, zorder=0)


def scatter_residuals_vs_mocks(ax, Mmocks, residuals, color, label, marker='o'):
    ax.scatter(Mmocks, residuals, color=color, s=40, alpha=0.7, 
               label=label, marker=marker, edgecolors='black', linewidth=0.5, zorder=3)
    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, zorder=2)
    ax.set_xlabel(r'$\log(M_{\rm bar}/M_{\odot})$', fontsize=16)
    ax.set_ylabel(r'$(V_{\rm obs} - V_{\rm mock})/\sigma$', fontsize=16)
    ax.set_xlim([7.5, 11.7])
    ax.set_ylim([-3.5, 3.5])
    ax.tick_params(labelsize=14)
    #ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)


def loglikehoods_vs_mocks(ax, Mmocks, log_likelihoods, color):
    ax.scatter(Mmocks, log_likelihoods, color=color, s=40, alpha=0.7, 
               edgecolors='black', linewidth=0.5, zorder=3)
    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, zorder=2)
    mean_delta = np.nanmean(log_likelihoods)
    ax.axhline(mean_delta, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_delta:.2f}', zorder=1)
    ax.set_xlabel(r'$\log(M_{\rm bar}/M_{\odot})$', fontsize=16)
    ax.set_ylabel(r'$\Delta \ln \mathcal{L}$', fontsize=16)
    ax.set_xlim([7.5, 11.7])
    ax.tick_params(labelsize=14)
    #ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    #ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    # Add text indicating which model is favored
    favor_text = 'Model 2 favored' if mean_delta > 0 else 'Model 1 favored'


def main():
    # Load saved results
    results_dir = '/Users/fedorboreiko/Documents/Oxford/btfr_z/btfr_panel_results'
    
    print('Loading Model 1 results...')
    model_1_data = np.load(
        os.path.join(results_dir, 'model_1_results.npy'), 
        allow_pickle=True
    ).item()
    
    print('Loading Model 2 results...')
    model_2_data = np.load(
        os.path.join(results_dir, 'model_2_results.npy'), 
        allow_pickle=True
    ).item()
    
    # Extract data from Model 1
    V_mock_1 = model_1_data['V_mock']
    V_mock_err_1 = model_1_data['V_mock_err']
    M_mock_1 = model_1_data['M_mock']
    M_mock_err_1 = model_1_data['M_mock_err']
    V_obs_1 = model_1_data['V_obs']
    V_obs_err_1 = model_1_data['V_obs_err']
    M_obs_1 = model_1_data['M_obs']
    M_obs_err_1 = model_1_data['M_obs_err']
    halo_proxy_1 = model_1_data['halo_proxy']
    catalog_1 = model_1_data['catalog']
    residuals_1 = model_1_data['residuals']
    log_likelihood_1 = model_1_data['log_likelihood']
    log_likelihoods_1 = model_1_data['log_likelihoods']
    params_1 = model_1_data['params']
    params_1["nu"] = -1.12
    
    # Extract data from Model 2
    V_mock_2 = model_2_data['V_mock']
    V_mock_err_2 = model_2_data['V_mock_err']
    M_mock_2 = model_2_data['M_mock']
    M_mock_err_2 = model_2_data['M_mock_err']
    V_obs_2 = model_2_data['V_obs']
    V_obs_err_2 = model_2_data['V_obs_err']
    M_obs_2 = model_2_data['M_obs']
    M_obs_err_2 = model_2_data['M_obs_err']
    halo_proxy_2 = model_2_data['halo_proxy']
    catalog_2 = model_2_data['catalog']
    residuals_2 = model_2_data['residuals']
    log_likelihood_2 = model_2_data['log_likelihood']
    log_likelihoods_2 = model_2_data['log_likelihoods']
    params_2 = model_2_data['params']
    
    # Create the figure and axes for the panel (2x2)
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # ===== Plot BTFR for Model 1 (top left) =====
    model_1_label = (fr'$\alpha={params_1["alpha"]:.2f}$, $\sigma_{{\mathrm{{SHAM}}}}={params_1["scatter"]}$, '
                     fr'$x={params_1["x"]}$, ' #+ '\n' +
                     fr'$\nu={params_1["nu"]:.2f}$, $\ln\mathcal{{L}}={log_likelihood_1:.1f}$')
    btfr_plot(M_mock_1, V_mock_1, M_mock_err_1, V_mock_err_1, 
              M_obs_1, V_obs_1, M_obs_err_1, V_obs_err_1, axs[0, 0], model_1_label)
    
    # ===== Plot BTFR for Model 2 (top right) =====
    model_2_label = (fr'$\alpha={params_2["alpha"]:.2f}$, $\sigma_{{\mathrm{{SHAM}}}}={params_2["scatter"]}$, '
                     fr'$x={params_2["x"]}$, ' #+ '\n' +
                     fr'$\nu={params_2["nu"]:.2f}$, $\ln\mathcal{{L}}={log_likelihood_2:.1f}$')
    btfr_plot(M_mock_2, V_mock_2, M_mock_err_2, V_mock_err_2, 
              M_obs_2, V_obs_2, M_obs_err_2, V_obs_err_2, axs[0, 1], model_2_label)
    
    # ===== SHMR Plot (bottom left) =====
    shmr_label_1 = rf'$\alpha={params_1["alpha"]:.2f}$, $\sigma_{{\mathrm{{SHAM}}}}={params_1["scatter"]}$'
    shmr_label_2 = rf'$\alpha={params_2["alpha"]:.2f}$, $\sigma_{{\mathrm{{SHAM}}}}={params_2["scatter"]}$'
    
    plot_SHMR_with_contours_quantile(axs[1, 0], halo_proxy_1, catalog_1, 
                                      color='darkviolet', label=shmr_label_1, linestyle='-')
    plot_SHMR_with_contours_quantile(axs[1, 0], halo_proxy_2, catalog_2, 
                                      color='darkorange', label=shmr_label_2, linestyle='-')
    
    axs[1, 0].set_ylabel(r'$\log(M_{\star}/M_{\odot})$', fontsize=16) 
    axs[1, 0].set_xlabel(r'$\log(M_{\mathrm{vir}}/M_{\odot})$', fontsize=16) 
    axs[1, 0].set_xlim([10.5, 14.5])
    axs[1, 0].set_ylim([7.5, 11.5])
    axs[1, 0].tick_params(labelsize=14)
    #axs[1, 0].grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    axs[1, 0].legend(loc='lower right', fontsize=14, framealpha=0.95)
    
    # ===== Delta Log Likelihood Plot (bottom right) =====
    delta_loglike = log_likelihoods_2 - log_likelihoods_1
    loglikehoods_vs_mocks(axs[1, 1], M_mock_1, delta_loglike, color='teal')
    
    # ===== Save Figure =====
    plt.tight_layout()
    plots_dir = '/Users/fedorboreiko/Documents/Oxford/btfr_z/plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    output_file = 'btfr_panel_paper.png'
    output_path = os.path.join(plots_dir, output_file)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'\nPlot saved to {output_path}')
    
    # Also save as PDF for publication
    output_pdf = os.path.join(plots_dir, 'btfr_panel_paper.pdf')
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f'PDF saved to {output_pdf}')


if __name__ == '__main__':
    main()
