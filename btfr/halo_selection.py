import numpy as np
import numpy.lib.recfunctions as rfn  # Used to append a field to the structured array

def get_x_cutoff_fit(halos, x=0.1):
    """
    Fits a linear relation between log10(Mvir) and the x-th percentile of log10(vmax).
    
    Parameters:
    halos (numpy structured array): The input halos catalog.
    x (float): The fraction representing the upper percentile cutoff (e.g., x=0.1 for top 10%).
    
    Returns:
    tuple: (slope, intercept) of the best-fit line.
    """
    # Compute log values
    log_vmax = np.log10(halos['vmax'])
    log_Mvir = np.log10(halos['Mvir'])

    # Define moving window width and step size
    window_width = 0.1  # dex
    step_size = 0.01  # dex
    percentile = 100 * (1 - x)  # Convert fraction to percentile

    # Set up the moving window for binning
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

    # Convert lists to numpy arrays
    mvir_bins = np.array(mvir_bins)
    vmax_percentiles = np.array(vmax_percentiles)

    # Select valid points in the range 10.2 to 13.5 dex
    valid_range_mask = (mvir_bins >= 10.2) & (mvir_bins <= 13.5)
    mvir_valid = mvir_bins[valid_range_mask]
    vmax_valid = vmax_percentiles[valid_range_mask]

    # Fit a straight line (linear regression) to the valid points
    slope, intercept = np.polyfit(mvir_valid, vmax_valid, 1)
    
    return slope, intercept


# ----------------------------
# Main script execution
# ----------------------------

# Define the file path
file_path = "/Users/fedorboreiko/Documents/Oxford/Personal_codes/Codebase/halos_z_0p00.npy"

# Load the halos dataset
halos = np.load(file_path)

# Set x value (fraction of highest vmax halos to consider)
x = 0.9

# Get the fitted slope and intercept
slope, intercept = get_x_cutoff_fit(halos, x)
print(f"Fitted line: log10(vmax) = {slope:.3f} * log10(Mvir) + {intercept:.3f}")

# Create a copy of halos with an extra field "select"
halos_selected = rfn.append_fields(halos, 'select', np.ones(halos.shape[0], dtype=int), usemask=False)

# Compute predicted log10(vmax) for each halo using the fitted line
halo_log_Mvir = np.log10(halos_selected['Mvir'])
predicted_log_vmax = slope * halo_log_Mvir + intercept

# Compute actual log10(vmax) for each halo
halo_log_vmax = np.log10(halos_selected['vmax'])

# Update "select" field: if log10(vmax) > predicted value, set select to 0
halos_selected['select'][halo_log_vmax > predicted_log_vmax] = 0

# Print summary statistics
num_selected = np.sum(halos_selected['select'] == 1)
num_not_selected = np.sum(halos_selected['select'] == 0)
print(f"Number of halos selected: {num_selected}")
print(f"Number of halos not selected: {num_not_selected}")


