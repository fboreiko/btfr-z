#!/usr/bin/env python3
"""
Overlay Posterior Plotter - Compare two MCMC chains in one corner plot
Structured to mirror posterior_plotter.py design and UX.

Author: Generated for btfr-z project
Date: September 2025
"""

import os
import numpy as np
import corner
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch

# Matplotlib style: Computer Modern + LaTeX
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern']
rcParams['text.usetex'] = True


def load_mcmc_data(chain_name, samples_dir="/Users/fedorboreiko/Documents/Oxford/btfr_z/btfr/samples"):
    """Load MCMC arrays (flat_samples, full_chain, log_prob) for a chain."""
    files = {
        'flat_samples': f'flat_samples_{chain_name}.npy',
        'full_chain': f'full_chain_{chain_name}.npy',
        'log_prob': f'log_prob_{chain_name}.npy',
    }
    data = {}
    for key, fname in files.items():
        path = os.path.join(samples_dir, fname)
        if os.path.exists(path):
            arr = np.load(path)
            data[key] = arr
            print(f"Loaded {key} for '{chain_name}': {arr.shape}")
        else:
            print(f"Warning: {path} not found for '{chain_name}'")
            data[key] = None
    return data


def list_available_chains(samples_dir="/Users/fedorboreiko/Documents/Oxford/btfr_z/btfr/samples"):
    """Return a sorted list of chain name stems discovered in samples_dir."""
    if not os.path.isdir(samples_dir):
        print(f"Samples directory not found: {samples_dir}")
        return []
    names = []
    for f in os.listdir(samples_dir):
        if f.startswith('flat_samples_') and f.endswith('.npy'):
            names.append(f[len('flat_samples_'):-4])
    names = sorted(set(names))
    print("Available chains:")
    for i, n in enumerate(names, 1):
        print(f"  {i}. {n}")
    return names


def get_parameter_labels(chain_name):
    """Choose LaTeX labels based on chain dimensionality encoded in name."""
    if '3param' in chain_name:
        return [r"$\tan^{-1}(\alpha)$", r"$\sigma$", r"$\nu$"]
    # Default to 4 params if unclear
    return [r"$\tan^{-1}(\alpha)$", r"$\sigma$", "x", r"$\nu$"]


def compute_combined_ranges(samples_a, samples_b, pad=0.02):
    """Compute per-dimension (min,max) ranges covering both sets, with padding."""
    assert samples_a.shape[1] == samples_b.shape[1]
    ndim = samples_a.shape[1]
    ranges = []
    for i in range(ndim):
        lo = min(samples_a[:, i].min(), samples_b[:, i].min())
        hi = max(samples_a[:, i].max(), samples_b[:, i].max())
        span = hi - lo
        if span == 0:
            lo -= 1e-3
            hi += 1e-3
        else:
            lo -= pad * span
            hi += pad * span
        ranges.append((lo, hi))
    return ranges


def create_overlay_corner_plot(samples_a,
                               samples_b,
                               labels,
                               output_path,
                               label_a="Model A",
                               label_b="Model B",
                               color_a="#0d47a1",
                               color_b="#d62728"):
    """Create a corner plot overlaying two posterior samples using corner.py."""
    print(f"Creating overlay corner: A={samples_a.shape}, B={samples_b.shape}")
    if samples_a.shape[1] != samples_b.shape[1]:
        raise ValueError("Samples must have the same number of parameters to overlay.")

    # 1, 2, 3-sigma levels in 2D
    sigma2d_levels = 1 - np.exp(-0.5 * np.array([1, 2, 3])**2)
    shared_range = compute_combined_ranges(samples_a, samples_b)

    # First dataset
    fig = corner.corner(
        samples_a,
        labels=labels,
        color=color_a,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=False,
        label_kwargs={"fontsize": 14},
        title_kwargs={"fontsize": 14},
        smooth=1.5,
        plot_datapoints=False,
        fill_contours=True,
        levels=sigma2d_levels,
        bins=30,
        range=shared_range,
        contour_kwargs={'linewidths': 0.8},
    )

    # Second dataset overlays on same axes
    corner.corner(
        samples_b,
        labels=labels,
        color=color_b,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=False,  # keep titles from first
        smooth=1.5,
        plot_datapoints=False,
        fill_contours=True,
        levels=sigma2d_levels,
        bins=30,
        range=shared_range,
        contour_kwargs={'linewidths': 0.8},
        fig=fig,
    )

    # Ticks
    for ax in fig.get_axes():
        ax.tick_params(labelsize=14)

    # Legend
    legend_handles = [
        Patch(facecolor=color_a, edgecolor='black', alpha=0.6, label=label_a),
        Patch(facecolor=color_b, edgecolor='black', alpha=0.6, label=label_b),
    ]
    fig.legend(handles=legend_handles, frameon=False, fontsize=14, loc='upper right', bbox_to_anchor=(1, 1))

    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Overlay corner saved to: {output_path}")
    plt.close(fig)


def create_overlay_trace_plot(full_chain_a,
                              full_chain_b,
                              labels,
                              output_path,
                              color_a="#0d47a1",
                              color_b="#d62728",
                              alpha=0.15):
    """Create overlay trace plots for two chains if both are available."""
    print(f"Creating overlay traces: A={None if full_chain_a is None else full_chain_a.shape}, B={None if full_chain_b is None else full_chain_b.shape}")
    if full_chain_a is None or full_chain_b is None:
        print("Trace plot skipped: one or both full chains are missing.")
        return
    if full_chain_a.shape[2] != full_chain_b.shape[2]:
        raise ValueError("Full chains must have the same ndim for overlay traces.")

    ndim = full_chain_a.shape[2]
    fig, axes = plt.subplots(ndim, 1, figsize=(10, 2*ndim), sharex=True)
    if ndim == 1:
        axes = [axes]
    iters_a = np.arange(full_chain_a.shape[0])
    iters_b = np.arange(full_chain_b.shape[0])

    for i in range(ndim):
        # Plot A
        for w in range(full_chain_a.shape[1]):
            axes[i].plot(iters_a, full_chain_a[:, w, i], color=color_a, alpha=alpha, linewidth=0.5)
        # Plot B
        for w in range(full_chain_b.shape[1]):
            axes[i].plot(iters_b, full_chain_b[:, w, i], color=color_b, alpha=alpha, linewidth=0.5)
        axes[i].set_ylabel(labels[i], fontsize=14)
        axes[i].tick_params(labelsize=12)
    axes[-1].set_xlabel("Step number", fontsize=14)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Overlay trace plots saved to: {output_path}")
    plt.close(fig)


def print_parameter_summary(samples, labels, header):
    """Print 16-50-84 percentiles for a sample set with a header."""
    if samples is None:
        print(f"\n--- {header} ---\nNo samples available.")
        return
    q16, q50, q84 = np.percentile(samples, [16, 50, 84], axis=0)
    s_plus = q84 - q50
    s_minus = q50 - q16
    print(f"\n--- {header} ---")
    for i, lab in enumerate(labels):
        print(f"{lab}: {q50[i]:.4f} +{s_plus[i]:.4f} -{s_minus[i]:.4f}")


def main():
    # Configuration
    chain_name_a = "4param"                # e.g., '3param'
    chain_name_b = "4param_vmaxshift"      # e.g., '3param_vmaxshift'
    legend_a = "Regular Model"
    legend_b = "Mean V Shift Model"
    output_dir = "/Users/fedorboreiko/Documents/Oxford/btfr_z/plots"
    downsample_step = 30  # set to None to disable
    plot_traces = False   # set True to also create overlay trace plots

    print("Overlay Posterior Plotter")
    print("=========================")

    os.makedirs(output_dir, exist_ok=True)
    available = list_available_chains()

    missing = [c for c in (chain_name_a, chain_name_b) if c not in available]
    if missing:
        print(f"Error: Missing chains: {', '.join(missing)}")
        print("Edit 'chain_name_a' and 'chain_name_b' in main() to choose from above list.")
        return

    print(f"\nComparing chains: '{chain_name_a}' vs '{chain_name_b}'")
    print("="*50)

    data_a = load_mcmc_data(chain_name_a)
    data_b = load_mcmc_data(chain_name_b)

    if data_a['flat_samples'] is None or data_b['flat_samples'] is None:
        print("Error: Both chains must have flat_samples to plot.")
        return

    samples_a = data_a['flat_samples']
    samples_b = data_b['flat_samples']
    if downsample_step and downsample_step > 1:
        samples_a = samples_a[::downsample_step]
        samples_b = samples_b[::downsample_step]
        print(f"Downsampled to: A={samples_a.shape}, B={samples_b.shape}")

    if samples_a.shape[1] != samples_b.shape[1]:
        print("Error: Chains have different dimensionality; cannot overlay.")
        return

    labels = get_parameter_labels(chain_name_a)

    corner_out = os.path.join(output_dir, f"corner_overlay_{chain_name_a}_vs_{chain_name_b}_edited.png")
    create_overlay_corner_plot(
        samples_a,
        samples_b,
        labels,
        corner_out,
        label_a=legend_a,
        label_b=legend_b,
    )

    if plot_traces:
        trace_out = os.path.join(output_dir, f"trace_overlay_{chain_name_a}_vs_{chain_name_b}_edited.png")
        create_overlay_trace_plot(data_a['full_chain'], data_b['full_chain'], labels, trace_out)

    print_parameter_summary(samples_a, labels, header=f"{legend_a} ({chain_name_a})")
    print_parameter_summary(samples_b, labels, header=f"{legend_b} ({chain_name_b})")

    print(f"\nPlots saved to: {output_dir}")
    print("To compare different chains, edit 'chain_name_a' and 'chain_name_b' in main() and run again.")


if __name__ == '__main__':
    main()
