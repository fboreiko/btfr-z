#!/usr/bin/env python3
"""
Plotting script for editing corner plot design using pre-computed MCMC chains.
This script focuses only on plot generation and styling for easy customization.

Author: Generated for btfr-z project
Date: September 2025
"""

import numpy as np
import corner
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pickle
import argparse
import os
from matplotlib.gridspec import GridSpec

# Set font to Computer Modern (LaTeX default) and enable LaTeX rendering
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern']
rcParams['text.usetex'] = True

def load_chains(filename):
    """Load MCMC chains and metadata from a pickle file."""
    with open(filename, 'rb') as f:
        chains_data = pickle.load(f)
    print(f"Chains loaded from {filename}")
    return chains_data

def create_styled_corner_plot(chains_data, output_filename, style='default'):
    """Create an overlaid corner plot with all mock datasets on one plot."""
    
    # Style configurations
    styles = {
        'default': {
            'figsize': (12, 12),
            'colors': ['#1f77b4', '#ff7f0e', '#2ca02c'],  # blue, orange, green
            'title_fontsize': 16,
            'label_fontsize': 14,
            'truth_color': 'red',
            'truth_marker': 'o',
            'truth_markersize': 4,
            'levels': (0.68, 0.95),
            'smooth': 1.0,
            'alpha': 0.7
        },
        'publication': {
            'figsize': (11, 11),
            'colors': ['#000080', '#FF6B35', '#228B22'],  # Navy, Orange-red, Forest green
            'title_fontsize': 14,
            'label_fontsize': 14,
            'truth_color': 'red',
            'truth_marker': 's',  # square markers
            'truth_markersize': 4,
            'levels': (0.68, 0.95),
            'smooth': 1.5,
            'alpha': 0.2
        }
    }

    
    style_config = styles.get(style, styles['default'])
    
    # Parameter labels - keep tan^(-1)(alpha) as requested
    labels = [r"$\tan^{-1}(\alpha)$", r"$\sigma$", "x", r"$\nu$"]
    
    truth_values = chains_data['truth_values']
    
    # Use the same sigma levels as posterior_plotter.py
    sigma2d_levels = 1 - np.exp(-0.5 * np.array([1, 2, 3])**2)
    
    print("Creating overlaid corner plot with all mock datasets...")
    
    mock_indices = list(chains_data['samples'].keys())
    
    # Create an empty figure to build upon
    fig = plt.figure(figsize=style_config['figsize'])
    
    # Loop through all datasets to plot them one by one
    for i, mock_idx in enumerate(mock_indices):
        samples = chains_data['samples'][mock_idx]
        print(f"  Plotting Mock Test {mock_idx + 1}")
        
        # Overlay on existing figure
        fig = corner.corner(
            samples,
            labels=labels,
            truths=truth_values if i == 0 else None, # Only plot truths for the first dataset
            truth_color=style_config['truth_color'],
            truth_kwargs={
                'marker': style_config['truth_marker'], 
                'markersize': style_config['truth_markersize'], 
                'linewidth': 1
            },
            show_titles=False,
            title_kwargs={"fontsize": style_config['label_fontsize']},
            label_kwargs={"fontsize": style_config['label_fontsize']},
            color=style_config['colors'][i],
            smooth=style_config['smooth'],
            plot_datapoints=False,
            fill_contours=False,
            plot_density=False,  # disable underlying pixel-like density so only contour lines remain
            levels=sigma2d_levels,
            bins=30,
            contour_kwargs={'linewidths': 1.0},
            alpha=style_config['alpha'],
            fig=fig # Use the same figure for all plots
        )
    
    # Increase tick label size (can be moved into the loop for efficiency, but this is fine)
    for ax in fig.get_axes():
        ax.tick_params(labelsize=style_config['label_fontsize'])
    
    # Create legend
    """from matplotlib.lines import Line2D
    legend_elements = []
    for i, mock_idx in enumerate(mock_indices):
        legend_elements.append(Line2D([0], [0], color=style_config['colors'][i], 
                                    linewidth=2, label=f'Mock Test {mock_idx + 1}'))
    
    # Add truth value to legend if it exists
    if truth_values is not None:
        legend_elements.append(Line2D([0], [0], marker=style_config['truth_marker'], 
                                    color=style_config['truth_color'], linewidth=0,
                                    markersize=style_config['truth_markersize'],
                                    label='True Values'))
    
    # Place legend in upper right corner
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98),
              fontsize=style_config['label_fontsize'] - 1)"""
    
    # Save the plot
    fig.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Overlaid corner plot saved to: {output_filename}")
    
    plt.close(fig)

def print_chain_info(chains_data):
    """Print information about the loaded chains."""
    print("\n--- Chain Information ---")
    print(f"Number of mock tests: {len(chains_data['samples'])}")
    print(f"Selected mock indices: {chains_data['selected_mocks']}")
    print(f"Truth values: {chains_data['truth_values']}")
    
    for grid_idx, samples in chains_data['samples'].items():
        print(f"\nMock Test {grid_idx + 1}:")
        print(f"  Sample shape: {samples.shape}")
        print(f"  Effective sample size: {len(samples)}")
        
        # Calculate some basic statistics
        q16, q50, q84 = np.percentile(samples, [16, 50, 84], axis=0)
        sigma = (q84 - q16) / 2
        
        param_names = [r'$\alpha$', r'$\sigma$', r'$x$', r'$(t_0-t_{1/2})$']
        for j, name in enumerate(param_names):
            print(f"  {name}: {q50[j]:.3f} ± {sigma[j]:.3f}")

def main():
    # Load chains
    try:
        chains_data = load_chains("/Users/fedorboreiko/Documents/Oxford/btfr_z/plots/paper_corner_plots/mcmc_chains_paper.pkl")
    except FileNotFoundError:
        print("Error: The pickle file was not found. Please provide the correct path.")
        return

    print_chain_info(chains_data)
    
    # Create output directory
    output_dir = "/Users/fedorboreiko/Documents/Oxford/btfr_z/plots/paper_corner_plots/plot.png"
    
    # Generate styled plot
    create_styled_corner_plot(chains_data, output_dir, "publication")

    # Print summary
    print_chain_info(chains_data)

if __name__ == '__main__':
    main()