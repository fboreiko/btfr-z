import numpy as np

def get_x_cutoff_fit(halos, x):
    """
    Fits a linear relation between log10(Mvir) and the x-th percentile of log10(vmax).
    
    Parameters:
    halos (numpy structured array): The input halos catalog.
    x (float): The fraction representing the upper percentile cutoff (e.g., x=0.1 for top 10%).
    
    Returns:
    tuple: (slope, intercept) of the best-fit line.
    """
    log_vmax = np.log10(halos['vmax'])
    log_Mvir = np.log10(halos['Mvir'])

    window_width = 0.1  # dex
    step_size = 0.01  # dex
    percentile = 100 * (1 - x)  # Convert fraction to percentile

    max_mvir = np.max(log_Mvir)
    min_mvir = np.min(log_Mvir)
    window_right = max_mvir
    window_left = window_right - window_width

    mvir_bins = []
    vmax_percentiles = []

    while window_left >= min_mvir:
        # Select halos within the window range
        mask = (log_Mvir >= window_left) & (log_Mvir <= window_right)
        if np.sum(mask) > 0:
            vmax_percentile = np.percentile(log_vmax[mask], percentile)
            mvir_bins.append((window_left + window_right) / 2)
            vmax_percentiles.append(vmax_percentile)

        # Move the window to the left
        window_right -= step_size
        window_left -= step_size

    mvir_bins = np.array(mvir_bins)
    vmax_percentiles = np.array(vmax_percentiles)

    valid_range_mask = (mvir_bins >= 10.2) & (mvir_bins <= 13.5)
    mvir_valid = mvir_bins[valid_range_mask]
    vmax_valid = vmax_percentiles[valid_range_mask]

    slope, intercept = np.polyfit(mvir_valid, vmax_valid, 1)

    return slope, intercept