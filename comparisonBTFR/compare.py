#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Compare SHAM posteriors with BTFR posteriors in a 4-panel contour plot.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.ndimage import gaussian_filter
from pathlib import Path
from SHAM_posteriors.load_posteriors import SHAMPosteriors

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path("/Users/fedorboreiko/Documents/Oxford/btfr_z/plots")
OUTPUT_DIR.mkdir(exist_ok=True)

# Match style from posterior_plotter.py
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern']
rcParams['text.usetex'] = True

# Colors from posterior_plotter.py
SPARC_COLOR = "#2424A4F1"  # deep blue (SPARC)
CLUSTERING_COLOR = "#D80000EE"  # red (Clustering)


def compute_density_contours(x, y, bins=50, weights=None, range=None, smooth=1.0):
    """
    Compute 2D histogram and density levels for contour plotting.

    Returns:
        X, Y: meshgrid arrays
        density: 2D density array
        levels: contour levels for 1, 2, 3 sigma
    """
    # Create 2D histogram (with optional weights)
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=weights, density=True, range=range)

    # Get bin centers
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])
    X, Y = np.meshgrid(xcenters, ycenters)

    # Transpose H to match meshgrid orientation
    density = H.T

    # Apply Gaussian smoothing
    if smooth > 0:
        density = gaussian_filter(density, sigma=smooth)

    # Compute contour levels for 2D GAUSSIAN
    sigma_fracs = 1 - np.exp(-0.5 * np.array([1, 2, 3])**2)

    # Sort density values and find levels
    sorted_density = np.sort(density.flatten())[::-1]
    cumsum = np.cumsum(sorted_density)
    cumsum /= cumsum[-1]

    levels = []
    for frac in sigma_fracs:
        idx = np.searchsorted(cumsum, frac)
        if idx < len(sorted_density):
            levels.append(sorted_density[idx])
        else:
            levels.append(sorted_density[-1])

    return X, Y, density, sorted(levels)


def plot_contours(ax, x, y, color, label=None, bins=50, filled=True, weights=None, hist_range=None, smooth=1.0):
    """Plot 2D contours on given axes."""
    X, Y, density, levels = compute_density_contours(x, y, bins=bins, weights=weights, range=hist_range, smooth=smooth)

    # Filter out invalid levels (must be strictly less than max)
    density_max = density.max()
    valid_levels = [l for l in levels if l < density_max]

    if filled and valid_levels:
        # Filled contours - fill bands between levels (not cumulative)
        # Levels are sorted low to high, so first band is outermost (3σ), last is innermost (1σ)
        alphas = [0.2, 0.45, 0.75]
        # Add max to levels for the innermost region
        all_levels = valid_levels + [density_max * 1.01]
        for i in range(len(all_levels) - 1):
            if i < len(alphas):
                ax.contourf(X, Y, density, levels=[all_levels[i], all_levels[i+1]],
                           colors=[color], alpha=alphas[i])

    # Contour lines
    if valid_levels:
        ax.contour(X, Y, density, levels=valid_levels, colors=[color], linewidths=1.2)

    # Add label via proxy artist
    if label:
        ax.plot([], [], color=color, linewidth=2, label=label)


def compute_combined_posterior(btfr_arctan_alpha, btfr_sigma,
                                sham_arctan_alpha, sham_scatter, sham_weights,
                                bins=50, theta_range=(-1.5, 1.5), sigma_range=(0, 0.8)):
    """
    Compute combined posterior L_BTFR × L_SHAM with flat prior on arctan(alpha).

    Returns grid, densities, and evidence ratio (tension statistic).
    """
    hist_range = [theta_range, sigma_range]

    # Estimate BTFR density (flat prior on arctan(alpha), no weights needed)
    H_btfr, xedges, yedges = np.histogram2d(
        btfr_arctan_alpha, btfr_sigma, bins=bins, range=hist_range, density=True
    )

    # Estimate SHAM density (reweighted to flat prior on arctan(alpha))
    H_sham, _, _ = np.histogram2d(
        sham_arctan_alpha, sham_scatter, bins=bins, range=hist_range,
        weights=sham_weights, density=True
    )

    # Apply smoothing
    H_btfr = gaussian_filter(H_btfr, sigma=0.7)
    H_sham = gaussian_filter(H_sham, sigma=0.7)

    # Grid
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])
    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]

    # Combined posterior (product of densities, which are proportional to likelihoods with flat prior)
    H_combined = H_btfr * H_sham

    # Normalize combined posterior
    H_combined_norm = H_combined / (H_combined.sum() * dx * dy) if H_combined.sum() > 0 else H_combined

    # Overlap integral R = ∫ p_BTFR × p_SHAM dθ dσ
    R = np.sum(H_btfr * H_sham) * dx * dy

    # Prior volume
    V = (theta_range[1] - theta_range[0]) * (sigma_range[1] - sigma_range[0])

    # Bayes factor K = Z_joint / (Z_BTFR × Z_SHAM) = R × V
    # With flat prior 1/V: Z_BTFR = Z_SHAM = 1/V, Z_joint = R/V
    # So K = (R/V) / (1/V × 1/V) = R × V
    K = R * V

    log_R = np.log10(R) if R > 0 else -np.inf
    log_K = np.log10(K) if K > 0 else -np.inf

    return xcenters, ycenters, H_btfr.T, H_sham.T, H_combined_norm.T, R, log_R, K, log_K


def compute_combined_posterior_alpha(btfr_alpha, btfr_sigma, btfr_weights,
                                      sham_alpha, sham_scatter,
                                      bins=50, alpha_range=(-3, 3), sigma_range=(0, 0.8)):
    """
    Compute combined posterior L_BTFR × L_SHAM with flat prior on alpha.

    Returns grid, densities, and evidence ratio (tension statistic).
    """
    hist_range = [alpha_range, sigma_range]

    # Estimate BTFR density (reweighted to flat prior on alpha)
    H_btfr, xedges, yedges = np.histogram2d(
        btfr_alpha, btfr_sigma, bins=bins, range=hist_range,
        weights=btfr_weights, density=True
    )

    # Estimate SHAM density (flat prior on alpha, no weights needed)
    H_sham, _, _ = np.histogram2d(
        sham_alpha, sham_scatter, bins=bins, range=hist_range, density=True
    )

    # Apply smoothing
    H_btfr = gaussian_filter(H_btfr, sigma=0.7)
    H_sham = gaussian_filter(H_sham, sigma=0.7)

    # Grid
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])
    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]

    # Combined posterior (product of densities)
    H_combined = H_btfr * H_sham

    # Normalize combined posterior
    H_combined_norm = H_combined / (H_combined.sum() * dx * dy) if H_combined.sum() > 0 else H_combined

    # Overlap integral R = ∫ p_BTFR × p_SHAM dα dσ
    R = np.sum(H_btfr * H_sham) * dx * dy

    # Prior volume
    V = (alpha_range[1] - alpha_range[0]) * (sigma_range[1] - sigma_range[0])

    # Bayes factor K = Z_joint / (Z_BTFR × Z_SHAM) = R × V
    K = R * V

    log_R = np.log10(R) if R > 0 else -np.inf
    log_K = np.log10(K) if K > 0 else -np.inf

    return xcenters, ycenters, H_btfr.T, H_sham.T, H_combined_norm.T, R, log_R, K, log_K


def main():
    # Configuration options
    PLOT_COMBINED_CONTOURS = False  # Set to True to show combined posterior contours

    # Load SHAM posteriors
    loader = SHAMPosteriors()
    all_bins = loader.load_all_bins('NSA_ELPETRO', 'ELPETRO_LOG_MASS', 'K')

    # Load BTFR posterior (flat_samples_4param_selection.npy)
    # Columns: 0=arctan(alpha), 1=sigma, 2=x, 3=nu
    btfr_samples = np.load(
        '/Users/fedorboreiko/Documents/Oxford/btfr_z/btfr/samples/flat_samples_4param_selection.npy'
    )
    btfr_arctan_alpha = btfr_samples[:, 0]  # arctan(alpha)
    btfr_alpha = np.tan(btfr_arctan_alpha)  # alpha
    btfr_sigma = btfr_samples[:, 1]  # sigma

    # ==================== PLOT 1: arctan(alpha) space ====================
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    fig.subplots_adjust(wspace=0)

    for i, (samples, metadata) in enumerate(all_bins):
        ax = axes[i]

        if samples is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            continue

        # Extract SHAM samples and convert to arctan(alpha)
        sham_alpha = samples['alpha']
        sham_arctan_alpha = np.arctan(sham_alpha)
        sham_scatter = samples['scatter']

        # Jacobian weight: convert from flat prior on alpha to flat prior on arctan(alpha)
        # d(arctan(alpha))/d(alpha) = 1/(1+alpha^2), so weight = 1/(1+alpha^2)
        sham_weights = 1.0 / (1.0 + sham_alpha**2)

        # Plot BTFR posterior (same in each panel) - no weights needed
        plot_contours(ax, btfr_arctan_alpha, btfr_sigma, SPARC_COLOR,
                     label='SPARC (this paper)', bins=40)

        # Plot SHAM posterior for this bin (with Jacobian weights)
        plot_contours(ax, sham_arctan_alpha, sham_scatter, CLUSTERING_COLOR,
                     label='Clustering (ST21)', bins=40, weights=sham_weights)

        # Labels
        ax.set_xlabel(r'$\tan^{-1}(\alpha)$', fontsize=16)
        if i == 0:
            ax.set_ylabel(r'$\sigma_{\rm SHAM}$', fontsize=16)
        ax.tick_params(labelsize=12)

        # Title with mass range if available
        if metadata and 'cut_range' in metadata:
            lo, hi = metadata['cut_range']
            ax.set_title(f'${lo:.1f} < \\log M_*/M_\\odot < {hi:.1f}$', fontsize=14)

        # Legend (only on first panel)
        if i == 0:
            ax.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'compare_posteriors_arctan.pdf'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to: {output_path}")
    plt.close(fig)

    # ==================== PLOT 2: alpha space ====================
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    fig.subplots_adjust(wspace=0)

    # Jacobian weight for BTFR: convert from flat prior on arctan(alpha) to flat prior on alpha
    # d(alpha)/d(arctan(alpha)) = sec^2(arctan(alpha)) = 1 + alpha^2
    btfr_weights = 1.0 + btfr_alpha**2

    # View limits for alpha plot
    alpha_xlim = (-3, 3)
    sigma_ylim = (0, 1.0)
    # Histogram range (wider than view to capture data, but not extreme tails)
    hist_range = [(-5, 5), (0, 1.0)]

    for i, (samples, metadata) in enumerate(all_bins):
        ax = axes[i]

        if samples is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            continue

        # Extract SHAM samples (already in alpha space)
        sham_alpha = samples['alpha']
        sham_scatter = samples['scatter']

        # Plot SHAM posterior for this bin first (underneath) - no weights needed
        plot_contours(ax, sham_alpha, sham_scatter, CLUSTERING_COLOR,
                     label='Clustering (ST21)', bins=40, hist_range=hist_range)

        # Plot BTFR posterior on top (with Jacobian weights)
        plot_contours(ax, btfr_alpha, btfr_sigma, SPARC_COLOR,
                     label='SPARC (this paper)', bins=40, weights=btfr_weights, hist_range=hist_range)

        # Labels and limits (zoom to view region)
        ax.set_xlabel(r'$\alpha$', fontsize=16)
        if i == 0:
            ax.set_ylabel(r'$\sigma_{\rm SHAM}$', fontsize=16)
        ax.tick_params(labelsize=12)
        ax.set_xlim(alpha_xlim)
        ax.set_ylim(sigma_ylim)

        # Title with mass range if available
        if metadata and 'cut_range' in metadata:
            lo, hi = metadata['cut_range']
            ax.set_title(f'${lo:.1f} < \\log M_*/M_\\odot < {hi:.1f}$', fontsize=14)

        # Legend (only on first panel)
        if i == 0:
            ax.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'compare_posteriors_alpha.pdf'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to: {output_path}")
    plt.close(fig)

    # ==================== PLOT 3: Combined posterior ====================
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    fig.subplots_adjust(wspace=0)

    COMBINED_COLOR = "#E69F00"  # orange (colorblind-friendly)

    print("\n=== Tension Analysis (arctan(alpha) space) ===")
    print("K = Bayes factor = Z_joint / (Z_BTFR × Z_SHAM)")
    print("Jeffreys' scale: |log10(K)| < 0.5 weak, 0.5-1 substantial, 1-2 strong, >2 decisive\n")

    for i, (samples, metadata) in enumerate(all_bins):
        ax = axes[i]

        if samples is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            continue

        # Extract SHAM samples and convert to arctan(alpha)
        sham_alpha = samples['alpha']
        sham_arctan_alpha = np.arctan(sham_alpha)
        sham_scatter = samples['scatter']
        sham_weights = 1.0 / (1.0 + sham_alpha**2)

        # Compute combined posterior and tension
        # Prior: flat on arctan(alpha) in (-pi/2, pi/2), flat on sigma in (0.01, 1)
        xc, yc, H_btfr, H_sham, H_combined, R, log_R, K, log_K = compute_combined_posterior(
            btfr_arctan_alpha, btfr_sigma,
            sham_arctan_alpha, sham_scatter, sham_weights,
            bins=50, theta_range=(-np.pi/2, np.pi/2), sigma_range=(0.01, 1.0)
        )

        # Print tension results
        bin_label = f"Bin {i+1}"
        if metadata and 'cut_range' in metadata:
            lo, hi = metadata['cut_range']
            bin_label += f" ({lo:.1f} < log M*/Msun < {hi:.1f})"
        print(f"{bin_label}: K = {K:.4e}, log10(K) = {log_K:.2f}")

        # Compute contour levels for combined posterior
        X, Y = np.meshgrid(xc, yc)
        sigma_fracs = 1 - np.exp(-0.5 * np.array([1, 2, 3])**2)
        sorted_density = np.sort(H_combined.flatten())[::-1]
        cumsum = np.cumsum(sorted_density)
        if cumsum[-1] > 0:
            cumsum /= cumsum[-1]
        levels = []
        for frac in sigma_fracs:
            idx = np.searchsorted(cumsum, frac)
            if idx < len(sorted_density):
                levels.append(sorted_density[idx])
            else:
                levels.append(sorted_density[-1])
        levels = sorted([l for l in levels if l < H_combined.max()])

        # Plot individual posteriors (faint)
        plot_contours(ax, btfr_arctan_alpha, btfr_sigma, SPARC_COLOR,
                     label='SPARC (this paper)', bins=40)
        plot_contours(ax, sham_arctan_alpha, sham_scatter, CLUSTERING_COLOR,
                     label='Clustering (ST21)', bins=40, weights=sham_weights)

        # Plot combined posterior (optional)
        if PLOT_COMBINED_CONTOURS and levels:
            ax.contour(X, Y, H_combined, levels=levels, colors=[COMBINED_COLOR], linewidths=1.5)
            if i == 0:
                ax.plot([], [], color=COMBINED_COLOR, linewidth=2, label='Combined')

        # Labels and limits
        ax.set_xlabel(r'$\tan^{-1}(\alpha)$', fontsize=16)
        if i == 0:
            ax.set_ylabel(r'$\sigma_{\rm SHAM}$', fontsize=16)
        ax.tick_params(labelsize=12)

        # Title with mass range and tension
        if metadata and 'cut_range' in metadata:
            lo, hi = metadata['cut_range']
            ax.set_title(f'${lo:.1f} < \\log M_*/M_\\odot < {hi:.1f}$\n$\\log_{{10}} K_{{\\mathrm{{joint}}}} = {log_K:.2f}$', fontsize=12)

        # Legend (only on first panel)
        if i == 0:
            ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'compare_posteriors_combined.pdf'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved combined posterior plot to: {output_path}")
    plt.close(fig)

    # ==================== PLOT 4: Combined posterior in alpha space ====================
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    fig.subplots_adjust(wspace=0)

    # BTFR weights for alpha space (flat prior on arctan -> flat prior on alpha)
    btfr_weights = 1.0 + btfr_alpha**2

    # Histogram range for alpha space
    alpha_range = (-3, 3)
    sigma_range_alpha = (0, 1.0)

    print("\n=== Tension Analysis (alpha space) ===")
    print("K = Bayes factor = Z_joint / (Z_BTFR × Z_SHAM)")
    print("Jeffreys' scale: |log10(K)| < 0.5 weak, 0.5-1 substantial, 1-2 strong, >2 decisive\n")

    for i, (samples, metadata) in enumerate(all_bins):
        ax = axes[i]

        if samples is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            continue

        # Extract SHAM samples (already in alpha space)
        sham_alpha = samples['alpha']
        sham_scatter = samples['scatter']

        # Compute combined posterior and tension in alpha space
        xc, yc, H_btfr, H_sham, H_combined, R, log_R, K, log_K = compute_combined_posterior_alpha(
            btfr_alpha, btfr_sigma, btfr_weights,
            sham_alpha, sham_scatter,
            bins=50, alpha_range=alpha_range, sigma_range=sigma_range_alpha
        )

        # Print tension results
        bin_label = f"Bin {i+1}"
        if metadata and 'cut_range' in metadata:
            lo, hi = metadata['cut_range']
            bin_label += f" ({lo:.1f} < log M*/Msun < {hi:.1f})"
        print(f"{bin_label}: K = {K:.4e}, log10(K) = {log_K:.2f}")

        # Compute contour levels for combined posterior
        X, Y = np.meshgrid(xc, yc)
        sigma_fracs = 1 - np.exp(-0.5 * np.array([1, 2, 3])**2)
        sorted_density = np.sort(H_combined.flatten())[::-1]
        cumsum = np.cumsum(sorted_density)
        if cumsum[-1] > 0:
            cumsum /= cumsum[-1]
        levels = []
        for frac in sigma_fracs:
            idx = np.searchsorted(cumsum, frac)
            if idx < len(sorted_density):
                levels.append(sorted_density[idx])
            else:
                levels.append(sorted_density[-1])
        levels = sorted([l for l in levels if l < H_combined.max()])

        # Plot individual posteriors
        hist_range = [alpha_range, sigma_range_alpha]
        plot_contours(ax, btfr_alpha, btfr_sigma, SPARC_COLOR,
                     label='SPARC (this paper)', bins=40, weights=btfr_weights, hist_range=hist_range)
        plot_contours(ax, sham_alpha, sham_scatter, CLUSTERING_COLOR,
                     label='Clustering (ST21)', bins=40, hist_range=hist_range)

        # Plot combined posterior (optional)
        if PLOT_COMBINED_CONTOURS and levels:
            ax.contour(X, Y, H_combined, levels=levels, colors=[COMBINED_COLOR], linewidths=1.5)
            if i == 0:
                ax.plot([], [], color=COMBINED_COLOR, linewidth=2, label='Combined')

        # Labels and limits
        ax.set_xlabel(r'$\alpha$', fontsize=16)
        if i == 0:
            ax.set_ylabel(r'$\sigma_{\rm SHAM}$', fontsize=16)
        ax.tick_params(labelsize=12)
        ax.set_xlim(alpha_range)
        ax.set_ylim(sigma_range_alpha)

        # Title with mass range and tension
        if metadata and 'cut_range' in metadata:
            lo, hi = metadata['cut_range']
            ax.set_title(f'${lo:.1f} < \\log M_*/M_\\odot < {hi:.1f}$\n$\\log_{{10}} K_{{\\mathrm{{joint}}}} = {log_K:.2f}$', fontsize=12)

        # Legend (only on first panel)
        if i == 0:
            ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'compare_posteriors_combined_alpha.pdf'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved combined posterior (alpha space) plot to: {output_path}")
    plt.close(fig)


    # ==================== PLOT 5: Prior flat on arctan(alpha), plotted in alpha space ====================
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    fig.subplots_adjust(wspace=0)

    print("\n=== Tension Analysis (flat prior on arctan(alpha), plotted in alpha space) ===")
    print("K = Bayes factor = Z_joint / (Z_BTFR × Z_SHAM)")
    print("Jeffreys' scale: |log10(K)| < 0.5 weak, 0.5-1 substantial, 1-2 strong, >2 decisive\n")

    # View limits for alpha (display range)
    alpha_xlim = (-3, 3)
    sigma_ylim = (0.01, 1.0)
    # Different histogram ranges for each dataset
    # BTFR: wide range (-50, 5) to capture 99.95% of mass (long negative tail)
    # SHAM: narrow range (-1, 3) since all samples are in [0, 2]
    hist_range_btfr = [(-50, 5), (0.01, 1.0)]
    hist_range_sham = [(-1, 3), (0.01, 1.0)]
    bins_btfr = 200  # More bins for BTFR to get good resolution in central region
    bins_sham = 60   # Fewer bins needed for narrow SHAM range

    for i, (samples, metadata) in enumerate(all_bins):
        ax = axes[i]

        if samples is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            continue

        # Extract SHAM samples
        sham_alpha = samples['alpha']
        sham_scatter = samples['scatter']

        # Jacobian weight for SHAM: flat prior on alpha -> flat prior on arctan(alpha)
        sham_weights = 1.0 / (1.0 + sham_alpha**2)

        # Compute tension with flat prior on arctan(alpha)
        sham_arctan_alpha = np.arctan(sham_alpha)
        xc, yc, H_btfr, H_sham, H_combined, R, log_R, K, log_K = compute_combined_posterior(
            btfr_arctan_alpha, btfr_sigma,
            sham_arctan_alpha, sham_scatter, sham_weights,
            bins=50, theta_range=(-np.pi/2, np.pi/2), sigma_range=(0.01, 1.0)
        )

        # Print tension results
        bin_label = f"Bin {i+1}"
        if metadata and 'cut_range' in metadata:
            lo, hi = metadata['cut_range']
            bin_label += f" ({lo:.1f} < log M*/Msun < {hi:.1f})"
        print(f"{bin_label}: K = {K:.4e}, log10(K) = {log_K:.2f}")

        # Plot in ALPHA space (but with flat prior on arctan(alpha))
        # BTFR: no weights needed (flat prior on arctan(alpha) is native)
        # Use wide hist_range to capture full posterior, then restrict view with xlim
        # Higher smoothing (1.8) to reduce jaggedness
        plot_contours(ax, btfr_alpha, btfr_sigma, SPARC_COLOR,
                     label='SPARC (this paper)', bins=bins_btfr, hist_range=hist_range_btfr, smooth=1.8)

        # SHAM: weighted by 1/(1+alpha²) to convert to flat prior on arctan(alpha)
        # Use narrower hist_range since SHAM samples are all in [0, 2]
        plot_contours(ax, sham_alpha, sham_scatter, CLUSTERING_COLOR,
                     label='Clustering (S21)', bins=bins_sham, weights=sham_weights, hist_range=hist_range_sham)

        # Labels and limits (enlarged text)
        ax.set_xlabel(r'$\alpha$', fontsize=18)
        if i == 0:
            ax.set_ylabel(r'$\sigma_{\rm SHAM}$ [dex]', fontsize=18)
        ax.tick_params(labelsize=14)
        ax.set_xlim(alpha_xlim)
        ax.set_ylim(sigma_ylim)

        # Title with mass range and tension (log instead of log10)
        if metadata and 'cut_range' in metadata:
            lo, hi = metadata['cut_range']
            ax.set_title(f'${lo:.1f} < \\log M_*/M_\\odot < {hi:.1f}$\n$\\log K_{{\\mathrm{{joint}}}} = {log_K:.2f}$', fontsize=14)

        # Legend (only on first panel)
        if i == 0:
            ax.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'compare_posteriors_arctan_prior_alpha_plot.pdf'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to: {output_path}")
    plt.close(fig)

    # ==================== PLOT 6: Single panel (Bin 4 only), 1σ and 2σ contours ====================
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    # Get Bin 4 data (index 3)
    samples, metadata = all_bins[3]
    sham_alpha = samples['alpha']
    sham_scatter = samples['scatter']
    sham_weights = 1.0 / (1.0 + sham_alpha**2)

    # Compute 1σ and 2σ levels
    sigma_1_frac = 1 - np.exp(-0.5 * 1**2)  # ~39.3%
    sigma_2_frac = 1 - np.exp(-0.5 * 2**2)  # ~86.5%

    # BTFR contours (1σ and 2σ)
    X_btfr, Y_btfr, density_btfr, _ = compute_density_contours(
        btfr_alpha, btfr_sigma, bins=bins_btfr, weights=None, range=hist_range_btfr, smooth=1.4
    )
    sorted_d = np.sort(density_btfr.flatten())[::-1]
    cumsum = np.cumsum(sorted_d)
    cumsum /= cumsum[-1]
    idx_1 = np.searchsorted(cumsum, sigma_1_frac)
    idx_2 = np.searchsorted(cumsum, sigma_2_frac)
    level_1sig_btfr = sorted_d[idx_1] if idx_1 < len(sorted_d) else sorted_d[-1]
    level_2sig_btfr = sorted_d[idx_2] if idx_2 < len(sorted_d) else sorted_d[-1]

    # Plot 2σ then 1σ (outer to inner) - levels must be increasing
    btfr_levels = sorted([level_2sig_btfr, level_1sig_btfr])
    ax.contourf(X_btfr, Y_btfr, density_btfr, levels=[btfr_levels[0], btfr_levels[1]],
                colors=[SPARC_COLOR], alpha=0.3)
    ax.contourf(X_btfr, Y_btfr, density_btfr, levels=[btfr_levels[1], density_btfr.max() * 1.01],
                colors=[SPARC_COLOR], alpha=0.6)
    ax.contour(X_btfr, Y_btfr, density_btfr, levels=btfr_levels, colors=[SPARC_COLOR], linewidths=1.5)
    ax.plot([], [], color=SPARC_COLOR, linewidth=2, label='Dynamics (this paper)')

    # SHAM contours (1σ and 2σ, with smoothing)
    X_sham, Y_sham, density_sham, _ = compute_density_contours(
        sham_alpha, sham_scatter, bins=bins_sham, weights=sham_weights, range=hist_range_sham, smooth=2.0
    )
    sorted_d = np.sort(density_sham.flatten())[::-1]
    cumsum = np.cumsum(sorted_d)
    cumsum /= cumsum[-1]
    idx_1 = np.searchsorted(cumsum, sigma_1_frac)
    idx_2 = np.searchsorted(cumsum, sigma_2_frac)
    level_1sig_sham = sorted_d[idx_1] if idx_1 < len(sorted_d) else sorted_d[-1]
    level_2sig_sham = sorted_d[idx_2] if idx_2 < len(sorted_d) else sorted_d[-1]

    # Plot 2σ then 1σ (outer to inner) - levels must be increasing
    sham_levels = sorted([level_2sig_sham, level_1sig_sham])
    ax.contourf(X_sham, Y_sham, density_sham, levels=[sham_levels[0], sham_levels[1]],
                colors=[CLUSTERING_COLOR], alpha=0.3)
    ax.contourf(X_sham, Y_sham, density_sham, levels=[sham_levels[1], density_sham.max() * 1.01],
                colors=[CLUSTERING_COLOR], alpha=0.6)
    ax.contour(X_sham, Y_sham, density_sham, levels=sham_levels, colors=[CLUSTERING_COLOR], linewidths=1.5)
    ax.plot([], [], color=CLUSTERING_COLOR, linewidth=2, label='Clustering (S21)')

    # Labels and limits
    ax.set_xlabel(r'$\alpha$', fontsize=18)
    ax.set_ylabel(r'$\sigma_{\rm SHAM}$ [dex]', fontsize=18)
    ax.tick_params(labelsize=14)
    ax.set_xlim(alpha_xlim)
    ax.set_ylim(sigma_ylim)
    ax.legend(loc='upper left', fontsize=14, frameon=False)

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'compare_posteriors_bin4_2sigma.pdf'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved single panel plot to: {output_path}")
    plt.close(fig)


if __name__ == '__main__':
    main()
