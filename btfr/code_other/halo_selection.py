import numpy as np
import numpy.lib.recfunctions as rfn  # Used to append a field to the structured array
from utils import *
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set up matplotlib parameters for LaTeX-like rendering
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern']
rcParams['text.usetex'] = True

# Define the file path
file_path = "/Users/fedorboreiko/Documents/Oxford/Personal_codes/Codebase/halos_z_0p00.npy"

colors = ["#F37B933A", "blue", "orange"]

# Load the halos dataset
halos = np.load(file_path)

# Set x value (fraction of highest vmax halos to consider)
x = 0.5

# Get the fitted slope and intercept for x=0.5
slope_05, intercept_05 = get_x_cutoff_fit(halos, 0.5)
print(f"Fitted line (x=0.5): log10(vmax) = {slope_05:.3f} * log10(Mvir) + {intercept_05:.3f}")

# Get the fitted slope and intercept for x=0.0
slope_00, intercept_00 = get_x_cutoff_fit(halos, 0.0)
print(f"Fitted line (x=0.0): log10(vmax) = {slope_00:.3f} * log10(Mvir) + {intercept_00:.3f}")

# Use x=0.5 for selection
slope, intercept = slope_05, intercept_05

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

# plot log10 vmax vs log10 Mvir with the red line indicating the fitted relation
plt.figure(figsize=(8, 6))
#scatter only randomly chosen 10 percent of data
halo_log_Mvir_plot = halo_log_Mvir
halo_log_vmax_plot = halo_log_vmax

# Compute predicted values for both thresholds
predicted_log_vmax_05 = slope_05 * halo_log_Mvir + intercept_05
predicted_log_vmax_00 = slope_00 * halo_log_Mvir + intercept_00

plt.scatter(halo_log_Mvir_plot, halo_log_vmax_plot, s=5, alpha=0.05, color=colors[0], rasterized=True)
plt.plot(halo_log_Mvir, predicted_log_vmax_00, color=colors[2], label=r'$x=0.0$', linewidth=2)
plt.plot(halo_log_Mvir, predicted_log_vmax_05, color=colors[1], label=r'$x=0.5$', linewidth=2)
plt.xlabel(r'$\log \left( M_{\mathrm{vir}} \, \left[ \mathrm{M}_{\odot} \right] \right)$', fontsize=18)
plt.ylabel(r'$\log \left( V_{\mathrm{max, halo}} \, \left[ \mathrm{km \, s^{-1}} \right] \right)$', fontsize=18)
plt.xlim([10, 15])
plt.ylim([1.45, 3.25])
plt.tick_params(axis='both', which='major', labelsize=18)
#plt.title(r'Halo Selection: $V_{\mathrm{max}}$ vs $M_{\mathrm{vir}}$', fontsize=16)
plt.legend(fontsize=18)
plt.tight_layout()
plt.savefig(f'/Users/fedorboreiko/Documents/Oxford/btfr_z/plots/halo_selection_plot_x_{x}.pdf', dpi=300, bbox_inches='tight')




