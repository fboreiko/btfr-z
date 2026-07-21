#!/usr/bin/env python3
"""
MCMC Chain Plotter -- load saved MCMC chains and create corner / trace plots.

All final plots are saved as PDF (vector graphics, journal-ready).
Edit the CONFIG block below
"""

import os

import numpy as np
import corner
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams

# ---------------------------------------------------------------------------
# Style: Computer Modern (LaTeX default) with LaTeX rendering
# ---------------------------------------------------------------------------
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Computer Modern"]
rcParams["text.usetex"] = True

# ---------------------------------------------------------------------------
# CONFIG -- edit these, then run the script
# ---------------------------------------------------------------------------
SAMPLES_DIR = "/Users/fedorboreiko/Documents/Oxford/btfr_z/btfr/samples"
OUTPUT_DIR = "/Users/fedorboreiko/Documents/Oxford/btfr_z/plots"

CHAIN_NAME = "4param_selection"                                 # single-chain quick look
STACKED_CHAIN_NAMES = ["3param_baseline", "4param_selection"]  # order of appearance
STACK_ORIENTATION = "horizontal"                         # 'vertical' or 'horizontal'
TRUTH_VALUES = None      # e.g. [-0.5, 0.1, 0.5, 0.0] to mark truths; None to disable

# One-sided upper limits on skewed/bounded posteriors:
# '95' -> q=0.95, '2sigma' -> Gaussian 2-sigma (q~0.97725), '97.5' -> q=0.975
ONE_SIDED_UL_METHOD = "2sigma"

# Quantile and print label for each UL method.  For a ONE-SIDED Gaussian-
# equivalent limit the quantile is Phi(n), the normal CDF at n sigma:
# Phi(1) = 0.841345, Phi(2) = 0.977250 (NOT the two-sided 68% / 95%).
UL_METHODS = {
    "95":     (0.95,     "95% UL"),
    "2sigma": (0.977250, "2 sigma UL"),
    "97.5":   (0.975,    "97.5% UL"),
}
Q_1SIGMA_ONE_SIDED = 0.841345  # Phi(1); the one-sided 1-sigma UL quantile

PARAM_LABELS_3 = [r"$\tan^{-1}(\alpha)$", r"$\sigma$", r"$\nu$"]
PARAM_LABELS_4 = [r"$\tan^{-1}(\alpha)$", r"$\sigma$", r"$x$", r"$\nu$"]

UL_ONLY_LABELS_BY_CHAIN = {
    "3param_baseline":  {r"$\tan^{-1}(\alpha)$", r"$\sigma$"},
    "4param_selection": {r"$\sigma$"},
}
DEFAULT_UL_ONLY_LABELS = {r"$\tan^{-1}(\alpha)$", r"$\sigma$"}

CORNER_COLOR = "#2f2fa8"
FILL_COLORS = [(1, 1, 1, 0), "#cacae8", "#9696d2", "#2f2fa8"]

# ---- design knobs: adjust freely and re-run ----
CORNER_SMOOTH = 1.4
CONTOUR_LINEWIDTH = 1.2
LABEL_FONTSIZE = 15
TICK_FONTSIZE = 12
TICK_ROTATION = 35

MATCH_STACKED_SIZES = True


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_mcmc_data(chain_name, samples_dir=SAMPLES_DIR):
    """Load flat_samples / full_chain / log_prob arrays for one chain.

    Returns a dict with those three keys; missing files map to None.
    """
    filenames = {
        "flat_samples": f"flat_samples_{chain_name}.npy",
        "full_chain": f"full_chain_{chain_name}.npy",
        "log_prob": f"log_prob_{chain_name}.npy",
    }

    data = {}
    for key, filename in filenames.items():
        filepath = os.path.join(samples_dir, filename)
        if os.path.exists(filepath):
            data[key] = np.load(filepath)
            print(f"Loaded {key}: {data[key].shape}")
        else:
            print(f"Warning: {filepath} not found")
            data[key] = None
    return data


def list_available_chains(samples_dir=SAMPLES_DIR):
    """List chain names found in the samples directory (via flat_samples_*.npy)."""
    chain_names = sorted(
        f[len("flat_samples_"):-len(".npy")]
        for f in os.listdir(samples_dir)
        if f.startswith("flat_samples_") and f.endswith(".npy")
    )
    print("Available MCMC chains:")
    for i, name in enumerate(chain_names, start=1):
        print(f"  {i}. {name}")
    return chain_names


def get_parameter_labels(chain_name):
    """LaTeX parameter labels appropriate for a chain (3- or 4-parameter)."""
    if "3param" in chain_name:
        return PARAM_LABELS_3
    return PARAM_LABELS_4  # '4param' and default


def get_ul_only_labels(chain_name):
    """Return the set of labels reported as one-sided ULs for this chain.

    Falls back to DEFAULT_UL_ONLY_LABELS for any chain not explicitly listed
    in UL_ONLY_LABELS_BY_CHAIN.
    """
    return UL_ONLY_LABELS_BY_CHAIN.get(chain_name, DEFAULT_UL_ONLY_LABELS)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def create_corner_plot(samples, labels, output_paths, truth_values=None,
                       smooth=None, linewidth=None, label_fontsize=None,
                       tick_fontsize=None, scale=1.0):
    """Create a corner plot and save it to every path in `output_paths`.

    Parameters:
        samples (np.ndarray): flat MCMC samples, shape (nsamples, ndim)
        labels (list): parameter labels
        output_paths (str | list[str]): one or more paths to save the same
            figure to (e.g. the final .pdf plus a temporary .png used only
            for stacking)
        truth_values (list, optional): true parameter values to mark
        smooth / linewidth / label_fontsize / tick_fontsize: design
            overrides; default to the CONFIG values when None
        scale (float): multiplier applied to font sizes and linewidths
            (used to equalise apparent sizes in the stacked figure)
    """
    if isinstance(output_paths, str):
        output_paths = [output_paths]
    smooth = CORNER_SMOOTH if smooth is None else smooth
    linewidth = (CONTOUR_LINEWIDTH if linewidth is None else linewidth) * scale
    label_fontsize = (LABEL_FONTSIZE if label_fontsize is None
                      else label_fontsize) * scale
    tick_fontsize = (TICK_FONTSIZE if tick_fontsize is None
                     else tick_fontsize) * scale

    print(f"Creating corner plot with {samples.shape[0]} samples "
          f"and {samples.shape[1]} parameters"
          + (f" (scale x{scale:.2f})" if scale != 1.0 else ""))

    sigma2d_levels = 1 - np.exp(-0.5 * np.array([1, 2, 3]) ** 2)

    fig = corner.corner(
        samples,
        labels=labels,
        color=CORNER_COLOR,
        truths=truth_values,
        truth_color="red" if truth_values else None,
        truth_kwargs={"marker": "o", "markersize": 4, "linewidth": 2}
        if truth_values else None,
        show_titles=False,
        label_kwargs={"fontsize": label_fontsize},
        smooth=smooth,
        plot_datapoints=False,
        fill_contours=True,
        levels=sigma2d_levels,
        bins=30,
        contour_kwargs={"linewidths": linewidth},
        contourf_kwargs={"colors": FILL_COLORS},
        hist_kwargs={"linewidth": linewidth},
        max_n_ticks=4,
    )

    for ax in fig.get_axes():
        ax.tick_params(labelsize=tick_fontsize)
        for tick in ax.get_xticklabels():
            tick.set_rotation(TICK_ROTATION)
            tick.set_horizontalalignment("right")

    for path in output_paths:
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Corner plot saved to: {path}")
    plt.close(fig)


def create_trace_plot(full_chain, labels, output_path):
    """Create trace plots (one panel per parameter) and save to `output_path`.

    Parameters:
        full_chain (np.ndarray): full MCMC chain, shape (n_iter, nwalkers, ndim)
        labels (list): parameter labels
        output_path (str): output file path (.pdf)
    """
    print(f"Creating trace plots with shape: {full_chain.shape}")

    n_iter, nwalkers, ndim = full_chain.shape
    fig, axes = plt.subplots(ndim, 1, figsize=(10, 2 * ndim), sharex=True)
    if ndim == 1:
        axes = [axes]

    iters = np.arange(n_iter)
    for i in range(ndim):
        for w in range(nwalkers):
            axes[i].plot(iters, full_chain[:, w, i], alpha=0.3, linewidth=0.5)
        axes[i].set_ylabel(labels[i], fontsize=14)
        axes[i].tick_params(labelsize=12)
    axes[-1].set_xlabel("Step number", fontsize=14)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Trace plots saved to: {output_path}")
    plt.close(fig)


def stack_images(image_paths, output_path, orientation="vertical", base_size=8):
    """Stack raster images into a single figure and save it (as PDF here).

    Note: the inputs must be raster images (PNG); matplotlib cannot read
    PDFs back in, which is why stacking uses temporary PNG renders.
    """
    imgs = [mpimg.imread(p) for p in image_paths if os.path.exists(p)]
    if not imgs:
        print("Warning: No images found to stack.")
        return

    n = len(imgs)
    if orientation == "horizontal":
        fig, axes = plt.subplots(1, n, figsize=(base_size * n, base_size))
    else:
        fig, axes = plt.subplots(n, 1, figsize=(base_size, base_size * n))
    if n == 1:
        axes = [axes]

    for ax, img in zip(np.ravel(axes), imgs):
        ax.imshow(img)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Stacked image saved to: {output_path}")
    plt.close(fig)


def create_stacked_corner_plots(chain_names, output_dir, truth_values_map=None,
                                orientation="vertical"):
    """Create individual corner plots (PDF) plus one combined stacked PDF.

    Parameters:
        chain_names (list[str]): chains to plot, in order of appearance
        output_dir (str): directory to save plots
        truth_values_map (dict | None): optional chain_name -> truth values
        orientation (str): 'vertical' or 'horizontal'
    """
    os.makedirs(output_dir, exist_ok=True)

    # load everything first so we know each chain's dimensionality
    available_chains = list_available_chains()
    to_plot = []
    for chain_name in chain_names:
        if chain_name not in available_chains:
            print(f"Warning: Chain '{chain_name}' not found. Skipping.")
            continue
        data = load_mcmc_data(chain_name)
        if data["flat_samples"] is None:
            print(f"Warning: No flat samples for '{chain_name}'. Skipping.")
            continue
        to_plot.append((chain_name, data))

    if not to_plot:
        return
    min_ndim = min(data["flat_samples"].shape[1] for _, data in to_plot)

    stack_pngs, stacked_tags = [], []
    for chain_name, data in to_plot:
        ndim = data["flat_samples"].shape[1]
        # An ndim-panel corner shown at the same height as a min_ndim-panel
        # one shrinks its text by min_ndim/ndim; pre-scale to compensate.
        scale = ndim / min_ndim if MATCH_STACKED_SIZES else 1.0

        labels = get_parameter_labels(chain_name)
        truths = None
        if truth_values_map and chain_name in truth_values_map:
            tv = truth_values_map[chain_name]
            if tv is not None and len(tv) == ndim:
                truths = tv
            else:
                print(f"Note: Provided truth values for '{chain_name}' "
                      f"don't match dimension; ignoring.")

        pdf_path = os.path.join(output_dir, f"corner_plot_{chain_name}.pdf")
        tmp_png = os.path.join(output_dir, f".tmp_corner_{chain_name}.png")
        create_corner_plot(
            data["flat_samples"], labels, [pdf_path, tmp_png],
            truth_values=truths,
            scale=scale,
        )
        stack_pngs.append(tmp_png)
        stacked_tags.append(chain_name)

    if stack_pngs:
        suffix = "_H" if orientation == "horizontal" else "_V"
        stacked_path = os.path.join(
            output_dir,
            "corner_plot_stacked_" + "_".join(stacked_tags) + suffix + ".pdf",
        )
        stack_images(stack_pngs, stacked_path, orientation=orientation, base_size=6)

    for p in stack_pngs:  # clean up temporary rasters
        try:
            os.remove(p)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def print_parameter_summary(samples, labels, ul_method=None, ul_only_labels=None):
    """Print medians with 1/2-sigma intervals, or upper limits for UL-only params.

    For UL-only parameters, the 1-sigma UL is the one-sided Phi(1) = 84.13%
    quantile, and the second limit is set by `ul_method` (default:
    ONE_SIDED_UL_METHOD from the config block).

    Parameters:
        samples (np.ndarray): flat MCMC samples, shape (nsamples, ndim)
        labels (list): parameter labels
        ul_method (str | None): which UL quantile convention to use
        ul_only_labels (set | None): which labels to report as one-sided ULs.
            When None, falls back to DEFAULT_UL_ONLY_LABELS.  Pass the output
            of get_ul_only_labels(chain_name) to get per-chain behaviour, so
            that e.g. tan^-1(alpha) is a UL in the baseline model but a
            two-sided constraint in the selection model.
    """
    ul_method = ul_method or ONE_SIDED_UL_METHOD
    if ul_method not in UL_METHODS:
        raise ValueError(f"Unknown UL method '{ul_method}'; "
                         f"choose from {sorted(UL_METHODS)}")
    q_ul, ul_label = UL_METHODS[ul_method]

    ul_only_labels = (DEFAULT_UL_ONLY_LABELS if ul_only_labels is None
                      else ul_only_labels)

    print("\n--- Parameter Summary ---")
    print(f"(one-sided ULs: 1 sigma = {100 * Q_1SIGMA_ONE_SIDED:.2f}th pct, "
          f"{ul_label} = {100 * q_ul:.2f}th pct)")
    q2p5, q16, q50, q84, q97p5 = np.percentile(
        samples, [2.5, 16, 50, 84, 97.5], axis=0
    )
    ul_1sig, ul_2sig = np.percentile(
        samples, [100 * Q_1SIGMA_ONE_SIDED, 100 * q_ul], axis=0
    )

    for i, label in enumerate(labels):
        if label in ul_only_labels:
            print(f"{label}: 1 sigma UL = {ul_1sig[i]:.4f}, "
                  f"{ul_label} = {ul_2sig[i]:.4f}")
        else:
            print(f"{label}: median: {q50[i]:.4f}, "
                  f"1 sigma level: +{q84[i] - q50[i]:.4f} -{q50[i] - q16[i]:.4f}, "
                  f"2 sigma level: +{q97p5[i] - q50[i]:.4f} -{q50[i] - q2p5[i]:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("MCMC Chain Plotter")
    print("==================")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    available_chains = list_available_chains()

    if CHAIN_NAME not in available_chains:
        print(f"Error: Chain '{CHAIN_NAME}' not found.")
        print("To use a different chain, edit CHAIN_NAME at the top of the script.")
        return

    print(f"\nProcessing chain: {CHAIN_NAME}")
    print("=" * 50)

    data = load_mcmc_data(CHAIN_NAME)
    if data["flat_samples"] is None:
        print(f"Error: No flat samples found for {CHAIN_NAME}")
        return

    labels = get_parameter_labels(CHAIN_NAME)

    create_corner_plot(
        data["flat_samples"],
        labels,
        os.path.join(OUTPUT_DIR, f"corner_plot_{CHAIN_NAME}.pdf"),
        truth_values=TRUTH_VALUES,
    )

    if data["full_chain"] is not None:
        create_trace_plot(
            data["full_chain"],
            labels,
            os.path.join(OUTPUT_DIR, f"trace_plots_{CHAIN_NAME}.pdf"),
        )

    print_parameter_summary(
        data["flat_samples"], labels,
        ul_only_labels=get_ul_only_labels(CHAIN_NAME),
    )

    if STACKED_CHAIN_NAMES and len(STACKED_CHAIN_NAMES) >= 2:
        print("\nCreating stacked corner plot for:",
              ", ".join(STACKED_CHAIN_NAMES), f"({STACK_ORIENTATION})")
        create_stacked_corner_plots(
            STACKED_CHAIN_NAMES, OUTPUT_DIR,
            truth_values_map=None, orientation=STACK_ORIENTATION,
        )

    print(f"\nPlots saved to: {OUTPUT_DIR}")
    print("To change chains: edit CHAIN_NAME or STACKED_CHAIN_NAMES at the top.")


if __name__ == "__main__":
    main()
