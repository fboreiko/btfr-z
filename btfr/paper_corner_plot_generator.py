#!/usr/bin/env python3
"""
Script to generate posterior corner plots from 3 randomly chosen mock tests
for paper publication. Now saves individual posterior plots (one per mock)
instead of a combined panel. Includes options to save/load MCMC chains for easy plot editing.

Author: Generated for btfr-z project
Date: September 2025
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import emcee
import corner
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import pickle
import random

# Set font to Computer Modern (LaTeX default) and enable LaTeX rendering
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern']
rcParams['text.usetex'] = True
rcParams['figure.figsize'] = (11, 11)  # Default size for single corner plots

def print_nan_coordinates(grid):
    """Print the coordinates of NaN values in a 3D array."""
    nan_indices = np.argwhere(np.isnan(grid))
    if nan_indices.size > 0:
        print("Found NaNs at the following coordinates:")
        return nan_indices
    else:
        print("No NaNs found in the grid.")
        return None

def average_neighbors(arr, index):
    """Compute the average of the neighbors of the cell at 'index' in a 3D or 4D array."""
    neighbor_values = []
    dims = len(index)
    
    if dims not in [3, 4]:
        raise ValueError("This function only supports 3D or 4D arrays.")
    
    ranges = [range(-1, 2) for _ in range(dims)]
    
    for offsets in np.ndindex(*[len(r) for r in ranges]):
        if all(offset == 0 for offset in offsets):
            continue
        
        neighbor_index = tuple(index[d] + offsets[d] for d in range(dims))
        
        if all(0 <= neighbor_index[d] < arr.shape[d] for d in range(dims)):
            neighbor_val = arr[neighbor_index]
            if not np.isnan(neighbor_val):
                neighbor_values.append(neighbor_val)
    
    return np.mean(neighbor_values) if neighbor_values else np.nan

def run_mcmc_for_grid(grid_index, likelihood_grid, alpha_proxy_range, scatter_range, x_range, nu_range):
    """Run MCMC sampling for a single likelihood grid."""
    print(f'Processing grid {grid_index + 1}/50')

    # Handle NaN values
    nan_indxs = print_nan_coordinates(likelihood_grid)
    if nan_indxs is not None:
        nan_indxs = [tuple(idx) for idx in nan_indxs]
        for idx in nan_indxs:
            if np.isnan(likelihood_grid[idx]):
                avg_val = average_neighbors(likelihood_grid, idx)
                print(f"Replacing NaN at index {idx} with average of neighbors: {avg_val}")
                likelihood_grid[idx] = avg_val

    # Interpolate the likelihood grid
    log_likelihood_interp = RegularGridInterpolator(
        (alpha_proxy_range, scatter_range, x_range, nu_range), 
        likelihood_grid,
        bounds_error=False,
        fill_value=-np.inf
    )

    def log_posterior(params):
        lp = log_likelihood_interp(params)[0]
        if not all([
            alpha_proxy_range[0] <= params[0] <= alpha_proxy_range[-1],
            scatter_range[0] <= params[1] <= scatter_range[-1],
            x_range[0] <= params[2] <= x_range[-1],
            nu_range[0] <= params[3] <= nu_range[-1],
        ]):
            return -np.inf
        return lp

    # MCMC parameters
    ndim, nwalkers = 4, 100
    warmup_steps, nsteps, check_interval = 5000, 100000, 100

    # Initialize walkers
    initial_pos = np.column_stack([
        np.random.uniform(r[0], r[-1], size=nwalkers)
        for r in (alpha_proxy_range, scatter_range, x_range, nu_range)
    ])

    # Set up the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)

    # Run warm-up phase
    print(f"Running warm-up phase for grid {grid_index + 1}...")
    sampler.run_mcmc(initial_pos, warmup_steps, progress=True)
    sampler.reset()

    # Main sampling with convergence check
    old_tau = np.full(ndim, np.inf)
    print(f"Running main sampling with convergence checks for grid {grid_index + 1}...")
    for _ in range(nsteps // check_interval):
        sampler.run_mcmc(None, check_interval, progress=True)
        try:
            tau = sampler.get_autocorr_time(tol=0)
        except emcee.autocorr.AutocorrError:
            print("...still too few samples to estimate τ; continuing.")
            continue

        iteration = sampler.iteration
        criterion1 = np.all(tau * 100 < iteration)
        criterion2 = np.all(np.abs(old_tau - tau) / tau < 0.005)
        print(f"Iteration={iteration}, τ={tau.round(1)}")
        if criterion1 and criterion2:
            print(f"✔ Chains have likely converged for grid {grid_index + 1}.")
            break
        old_tau = tau.copy()

    # Get samples
    samples = sampler.get_chain(flat=True)
    return samples, sampler

def save_chains(chains_data, filename):
    """Save MCMC chains and metadata to a pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(chains_data, f)
    print(f"Chains saved to {filename}")

def load_chains(filename):
    """Load MCMC chains and metadata from a pickle file."""
    with open(filename, 'rb') as f:
        chains_data = pickle.load(f)
    print(f"Chains loaded from {filename}")
    return chains_data

def create_individual_posterior_plots(chains_data, output_dir, truth_values=None):
    """Create and save individual posterior corner plots for each selected mock.

    The plotting routine mirrors the style used by posterior_plotter: use
    2D Gaussian-equivalent sigma levels for contours, no scatter datapoints,
    line contours (no filled regions), and consistent labeling.
    """

    # Plot in terms of alpha (not arctan(alpha))
    labels = [r"$\\alpha$", r"$\\sigma$", r"x", r"$\\nu$"]

    # Colors for different mock tests, aligned with selection order
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # blue, orange, green

    # 2D sigma-equivalent levels as in posterior_plotter
    sigma2d_levels = 1 - np.exp(-0.5 * np.array([1, 2, 3])**2)

    # Determine plotting order from selected_mocks for consistent color mapping
    mock_indices = list(chains_data.get('selected_mocks', chains_data['samples'].keys()))

    for i, grid_idx in enumerate(mock_indices):
        if grid_idx not in chains_data['samples']:
            print(f"Warning: samples for mock index {grid_idx} not found; skipping.")
            continue

        samples = chains_data['samples'][grid_idx]
        # Transform the first parameter from arctan(alpha) -> alpha
        samples_alpha = samples.copy()
        samples_alpha[:, 0] = np.tan(samples_alpha[:, 0])

        # Transform truth values if provided
        truths_alpha = None
        if truth_values is not None:
            truths_alpha = list(truth_values)
            truths_alpha[0] = np.tan(truths_alpha[0])

        # Build robust ranges to avoid tan() extremes
        alpha_low, alpha_high = np.percentile(samples_alpha[:, 0], [0.5, 99.5])
        plot_ranges = [(alpha_low, alpha_high), None, None, None]
        color = colors[i % len(colors)]

        fig = plt.figure(figsize=(11, 11))
        corner.corner(
            samples_alpha,
            labels=labels,
            truths=truths_alpha,
            truth_color='red',
            truth_kwargs={'marker': 'o', 'markersize': 4, 'linewidth': 1},
            show_titles=True,
            title_kwargs={"fontsize": 14},
            label_kwargs={"fontsize": 14},
            color=color,
            smooth=1.5,
            plot_datapoints=False,
            fill_contours=True,
            plot_density=False,
            levels=sigma2d_levels,
            bins=30,
            contour_kwargs={'linewidths': 1.0},
            alpha=0.7,
            range=plot_ranges,
            fig=fig
        )

        for ax in fig.get_axes():
            ax.tick_params(labelsize=12)

        outfile = os.path.join(output_dir, f"posterior_mock_{int(grid_idx)+1}.png")
        fig.savefig(outfile, dpi=300, bbox_inches='tight')
        print(f"Saved posterior plot for Mock {int(grid_idx)+1} to: {outfile}")
        plt.close(fig)

def main():
    # Defaults (parser removed as requested)
    load_chains_path = None  # e.g., set to a pickle path to load existing chains
    save_chains_filename = 'mcmc_chains_paper.pkl'
    seed = 42
    specified_mock_indices = None  # e.g., [3, 7, 12]

    # Set random seed for reproducible results
    random.seed(seed)
    np.random.seed(seed)
    
    # Define true values for parameters
    truth_alpha_proxy = -0.5
    truth_scatter = 0.1
    truth_x = 0.5
    truth_nu = np.linspace(-3.0, 3.0, 20)[12]
    truth_values = [truth_alpha_proxy, truth_scatter, truth_x, truth_nu]
    
    # Define parameter ranges
    alpha_proxy_range = np.linspace(-np.pi / 2, np.pi / 2, 20)
    scatter_range = np.linspace(0.01, 1.0, 20)
    x_range = np.linspace(0.01, 0.95, 20)
    nu_range = np.linspace(-3.0, 3.0, 20)
    
    # Create output directory
    output_dir = "/Users/fedorboreiko/Documents/Oxford/btfr_z/plots/paper_corner_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    if load_chains_path:
        # Load existing chains
        chains_data = load_chains(load_chains_path)
    else:
        # Generate new chains
        print("Generating new MCMC chains...")
        
        # Load likelihood grids
        file_path = '/Users/fedorboreiko/Documents/Oxford/btfr_z/btfr/likelihood_grid_50mocktruths_90am_49stellar_alphaproxy_-1.571_1.571_scatter_0.010_1.000_x_0.010_0.950_nu_-3.000_3.000.npy'
        likelihood_grids = np.load(file_path)
        print(f"Loaded likelihood grids with shape: {likelihood_grids.shape}")
        
        # Select 3 mock tests
        if specified_mock_indices:
            if len(specified_mock_indices) != 3:
                raise ValueError("Must specify exactly 3 mock indices")
            if not all(0 <= idx < 50 for idx in specified_mock_indices):
                raise ValueError("Mock indices must be between 0 and 49")
            selected_mocks = specified_mock_indices
        else:
            selected_mocks = random.sample(range(50), 3)
        
        print(f"Selected mock tests: {selected_mocks}")
        
        # Run MCMC for selected mocks
        chains_data = {
            'samples': {},
            'selected_mocks': selected_mocks,
            'truth_values': truth_values,
            'parameter_ranges': {
                'alpha_proxy': alpha_proxy_range,
                'scatter': scatter_range,
                'x': x_range,
                'nu': nu_range
            }
        }
        
        for grid_idx in selected_mocks:
            print(f"\n--- Processing mock test {grid_idx + 1} ---")
            samples, sampler = run_mcmc_for_grid(
                grid_idx, 
                likelihood_grids[grid_idx, :, :, :, :],
                alpha_proxy_range, scatter_range, x_range, nu_range
            )
            chains_data['samples'][grid_idx] = samples
        
        # Save chains if requested
        if save_chains_filename:
            save_path = os.path.join(output_dir, save_chains_filename)
            save_chains(chains_data, save_path)
    
    # Create and save individual posterior plots
    create_individual_posterior_plots(chains_data, output_dir, chains_data['truth_values'])
    
    # Print summary statistics
    print("\n--- Summary Statistics ---")
    for grid_idx, samples in chains_data['samples'].items():
        q16, q50, q84 = np.percentile(samples, [16, 50, 84], axis=0)
        # Convert alpha stats from arctan(alpha) -> alpha via monotonic transform
        a16, a50, a84 = np.tan([q16[0], q50[0], q84[0]])
        sigma_alpha = (a84 - a16) / 2
        # Other parameters unchanged
        sigma_other = (q84 - q16) / 2
        print(f"Mock Test {grid_idx + 1}:")
        print(f"  α: {a50:.3f} ± {sigma_alpha:.3f}")
        print(f"  σ: {q50[1]:.3f} ± {sigma_other[1]:.3f}")
        print(f"  x: {q50[2]:.3f} ± {sigma_other[2]:.3f}")
        print(f"  ν: {q50[3]:.3f} ± {sigma_other[3]:.3f}")

if __name__ == '__main__':
    main()