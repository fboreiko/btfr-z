#!/usr/bin/env python3
"""
MCMC Chain Plotter - Load saved MCMC chains and create corner plots
Rewritten to load chains from /Users/fedorboreiko/Documents/Oxford/btfr_z/btfr/samples

Author: Generated for btfr-z project
Date: September 2025
"""

import numpy as np
import corner
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.image as mpimg
import os

# Set font to Computer Modern (LaTeX default) and enable LaTeX rendering
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern']
rcParams['text.usetex'] = True

# Configuration for reporting one-sided upper limits on skewed/bounded posteriors
# Options: '95' for 95% UL (q=0.95), '2sigma' for Gaussian 2σ UL (q≈0.97725), '97.5' for q=0.975
ONE_SIDED_UL_METHOD = '2sigma'

def load_mcmc_data(chain_name, samples_dir="/Users/fedorboreiko/Documents/Oxford/btfr_z/btfr/samples"):
    """
    Load MCMC chain data from the samples directory.
    
    Parameters:
        chain_name (str): Name identifier for the chain (e.g., '4param_vmaxshift')
        samples_dir (str): Directory containing the chain files
        
    Returns:
        dict: Dictionary containing flat_samples, full_chain, and log_prob arrays
    """
    files = {
        'flat_samples': f'flat_samples_{chain_name}.npy',
        'full_chain': f'full_chain_{chain_name}.npy', 
        'log_prob': f'log_prob_{chain_name}.npy'
    }
    
    data = {}
    for key, filename in files.items():
        filepath = os.path.join(samples_dir, filename)
        if os.path.exists(filepath):
            data[key] = np.load(filepath)
            print(f"Loaded {key}: {data[key].shape}")
        else:
            print(f"Warning: {filepath} not found")
            data[key] = None
    
    return data

def get_parameter_labels(chain_name):
    """
    Get appropriate parameter labels based on chain name.
    
    Parameters:
        chain_name (str): Name identifier for the chain
        
    Returns:
        list: List of LaTeX-formatted parameter labels
    """
    if '3param' in chain_name:
        return [r"$\tan^{-1}(\alpha)$", r"$\sigma$", r"$\nu$"]
    elif '4param' in chain_name:
        return [r"$\tan^{-1}(\alpha)$", r"$\sigma$", "x", r"$\nu$"]
    else:
        # Default to 4 parameters
        return [r"$\tan^{-1}(\alpha)$", r"$\sigma$", "x", r"$\nu$"]

def create_corner_plot(samples, labels, output_path, truth_values=None, smooth=1.0):
    """
    Create and save a corner plot.
    
    Parameters:
        samples (np.ndarray): MCMC samples array
        labels (list): Parameter labels
        output_path (str): Output file path
        truth_values (list, optional): True parameter values to mark on plot
        title (str, optional): Plot title
    """
    print(f"Creating corner plot with {samples.shape[0]} samples and {samples.shape[1]} parameters")
    
    sigma2d_levels = 1 - np.exp(-0.5 * np.array([1, 2, 3])**2)

    fig = corner.corner(
        samples, 
        labels=labels,
        truths=truth_values,
        truth_color='red' if truth_values else None,
        truth_kwargs={'marker': 'o', 'markersize': 4, 'linewidth': 2} if truth_values else None,
        show_titles=False,
        label_kwargs={"fontsize": 14},
        # title_kwargs={"fontsize": 14},  # no titles shown
        smooth=smooth,
        plot_datapoints=False,
        fill_contours=True,
        levels=sigma2d_levels,
        bins=30,
        contour_kwargs={'linewidths': 0.8},  # Decreased contour linewidth
        #contourf_kwargs={'alpha': 0.6}       # Optional: adjust fill transparency
        max_n_ticks=4
    )
    
    # Increase tick label size
    for ax in fig.get_axes():
        ax.tick_params(labelsize=12)
        for tick in ax.get_xticklabels():
            tick.set_rotation(35)
            tick.set_horizontalalignment('right')

    # Save the plot
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Corner plot saved to: {output_path}")
    plt.close(fig)

def stack_images(image_paths, output_path, orientation='vertical', base_size=8):
    """
    Stack multiple images into a single figure and save.

    Parameters:
        image_paths (list[str]): Image file paths in order.
        output_path (str): Path to save the stacked image.
        orientation (str): 'vertical' or 'horizontal'.
        base_size (int|float): Base dimension to scale figure size.
    """
    imgs = [mpimg.imread(p) for p in image_paths if os.path.exists(p)]
    if not imgs:
        print("Warning: No images found to stack.")
        return

    n = len(imgs)
    if orientation == 'horizontal':
        figsize = (base_size * n, base_size)
        fig, axes = plt.subplots(1, n, figsize=figsize)
        if n == 1:
            axes = [axes]
        for ax, img in zip(axes, imgs):
            ax.imshow(img)
            ax.axis('off')
    else:
        figsize = (base_size, base_size * n)
        fig, axes = plt.subplots(n, 1, figsize=figsize)
        if n == 1:
            axes = [axes]
        for ax, img in zip(axes, imgs):
            ax.imshow(img)
            ax.axis('off')
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Stacked image saved to: {output_path}")
    plt.close(fig)

def create_stacked_corner_plots(chain_names, output_dir, truth_values_map=None, orientation='vertical'):
    """
    Create individual corner plots for specified chains and a combined stacked image.

    Parameters:
        chain_names (list[str]): Names of chains to plot in order (top to bottom).
        output_dir (str): Directory to save plots.
        truth_values_map (dict|None): Optional mapping chain_name -> truth_values list.
    """
    os.makedirs(output_dir, exist_ok=True)
    individual_paths = []

    available_chains = list_available_chains()
    for chain_name in chain_names:
        if chain_name not in available_chains:
            print(f"Warning: Chain '{chain_name}' not found. Skipping.")
            continue
        data = load_mcmc_data(chain_name)
        if data['flat_samples'] is None:
            print(f"Warning: No flat samples for '{chain_name}'. Skipping.")
            continue

        labels = get_parameter_labels(chain_name)
        truths = None
        if truth_values_map and chain_name in truth_values_map:
            tv = truth_values_map[chain_name]
            if tv is not None and len(tv) == data['flat_samples'].shape[1]:
                truths = tv
            else:
                print(f"Note: Provided truth values for '{chain_name}' don't match dimension; ignoring.")
        corner_output = os.path.join(output_dir, f"corner_plot_{chain_name}_edited.png")
        smooth_val = 1.5 if 'vmaxshift' in chain_name else 1.0
        create_corner_plot(data['flat_samples'], labels, corner_output, truth_values=truths, smooth=smooth_val)
        individual_paths.append(corner_output)

    if individual_paths:
        suffix = "_H" if orientation == 'horizontal' else "_V"
        stacked_path = os.path.join(output_dir, "corner_plot_stacked_" + "_".join([os.path.basename(p).replace('corner_plot_','').replace('_edited.png','') for p in individual_paths]) + suffix + "_edited.png")
        base = 6
        stack_images(individual_paths, stacked_path, orientation=orientation, base_size=base)

def create_trace_plot(full_chain, labels, output_path):
    """
    Create and save trace plots.
    
    Parameters:
        full_chain (np.ndarray): Full MCMC chain array (n_iter, nwalkers, ndim)
        labels (list): Parameter labels
        output_path (str): Output file path
        title (str, optional): Plot title
    """
    print(f"Creating trace plots with shape: {full_chain.shape}")
    
    ndim = full_chain.shape[2]
    nwalkers = full_chain.shape[1]
    
    fig, axes = plt.subplots(ndim, 1, figsize=(10, 2*ndim), sharex=True)
    if ndim == 1:
        axes = [axes]  # Make sure axes is always a list
    
    iters = np.arange(full_chain.shape[0])
    
    for i in range(ndim):
        for w in range(nwalkers):
            axes[i].plot(iters, full_chain[:, w, i], alpha=0.3, linewidth=0.5)
        axes[i].set_ylabel(labels[i], fontsize=14)
        axes[i].tick_params(labelsize=12)
    
    axes[-1].set_xlabel("Step number", fontsize=14)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Trace plots saved to: {output_path}")
    plt.close(fig)

def print_parameter_summary(samples, labels):
    """
    Print statistical summary of parameters.
    
    Parameters:
        samples (np.ndarray): MCMC samples
        labels (list): Parameter labels
    """
    print("\n--- Parameter Summary ---")
    q16, q50, q84 = np.percentile(samples, [16, 50, 84], axis=0)
    q2p5, q97p5 = np.percentile(samples, [2.5, 97.5], axis=0)
    q68 = np.percentile(samples, 68, axis=0)
    q95 = np.percentile(samples, 95, axis=0)
    
    sigma_plus = q84 - q50
    sigma_minus = q50 - q16
    two_sigma_plus = q97p5 - q50
    two_sigma_minus = q50 - q2p5

    for i, label in enumerate(labels):
        if label in [r"$\tan^{-1}(\alpha)$", r"$\sigma$"]:
            print(f"{label}: 1 sigma UL = {q68[i]:.4f}, 2 sigma UL = {q95[i]:.4f}")
        else:
            print(f"{label}: median: {q50[i]:.4f}, 1 sigma level: +{sigma_plus[i]:.4f} -{sigma_minus[i]:.4f}, 2 sigma level: +{two_sigma_plus[i]:.4f} -{two_sigma_minus[i]:.4f}")

def list_available_chains(samples_dir="/Users/fedorboreiko/Documents/Oxford/btfr_z/btfr/samples"):
    """
    List all available MCMC chains in the samples directory.
    
    Parameters:
        samples_dir (str): Directory containing the chain files
    """
    print("Available MCMC chains:")
    
    # Look for flat_samples files to identify available chains
    files = os.listdir(samples_dir)
    chain_names = []
    
    for file in files:
        if file.startswith('flat_samples_') and file.endswith('.npy'):
            chain_name = file.replace('flat_samples_', '').replace('.npy', '')
            chain_names.append(chain_name)
    
    for i, name in enumerate(sorted(chain_names)):
        print(f"  {i+1}. {name}")
    
    return sorted(chain_names)

def main():
    # Configuration - edit these variables to customize behavior
    chain_name = "3param"  # Single-chain quick look
    stacked_chain_names = ["3param", "3param_vmaxshift"]  # Order of appearance
    stack_orientation = 'vertical'  # 'vertical' or 'horizontal'
    output_dir = "/Users/fedorboreiko/Documents/Oxford/btfr_z/plots"
    truth_values = None  # Set to [alpha, sigma, x, nu] if you want to show truth values
    # truth_values = [-0.5, 0.1, 0.5, 0.0]  # Example truth values
    
    print("MCMC Chain Plotter")
    print("==================")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # List available chains
    available_chains = list_available_chains()
    
    # Check if requested chain exists
    if chain_name not in available_chains:
        print(f"Error: Chain '{chain_name}' not found.")
        print("Available chains:")
        for name in available_chains:
            print(f"  {name}")
        print(f"\nTo use a different chain, edit the 'chain_name' variable in main()")
        return
    
    print(f"\nProcessing chain: {chain_name}")
    print("="*50)
    
    # Load data
    data = load_mcmc_data(chain_name)
    
    if data['flat_samples'] is None:
        print(f"Error: No flat samples found for {chain_name}")
        return
    
    # Get labels
    labels = get_parameter_labels(chain_name)
    
    # Create output paths
    corner_output = os.path.join(output_dir, f"corner_plot_{chain_name}_edited.png")
    trace_output = os.path.join(output_dir, f"trace_plots_{chain_name}_edited.png")
    
    # Create corner plot
    create_corner_plot(
        data['flat_samples'], 
        labels, 
        corner_output, 
        truth_values=truth_values,
        smooth=1.2 if 'vmaxshift' in chain_name else 1.0
    )
    
    # Create trace plots if full chain is available
    if data['full_chain'] is not None:
        create_trace_plot(
            data['full_chain'], 
            labels, 
            trace_output
        )
    
    # Print parameter summary
    print_parameter_summary(data['flat_samples'], labels)
    
    # Also create stacked corner plot for requested pair if available
    if stacked_chain_names and len(stacked_chain_names) >= 2:
        print("\nCreating stacked corner plot for:", ", ".join(stacked_chain_names), f"({stack_orientation})")
        create_stacked_corner_plots(stacked_chain_names, output_dir, truth_values_map=None, orientation=stack_orientation)

    print(f"\nPlots saved to: {output_dir}")
    print("To change chains: edit 'chain_name' or 'stacked_chain_names' in main().")

if __name__ == '__main__':
    main()



