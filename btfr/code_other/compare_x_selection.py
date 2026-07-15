import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set up matplotlib parameters for LaTeX-like rendering
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern']
rcParams['text.usetex'] = True

# Define the base path for the saved data
base_path = '/Users/fedorboreiko/Documents/Oxford/btfr_z/plots/'

X_1 = 0.0
X_2 = 0.5

# Load data for x=0.0 (no selection)
V_mocks_mode_x00 = np.load(base_path + f'vels_hist_args_V_mocks_mode_galnum91_x{X_1}.npy')
V_obs_x00 = np.load(base_path + f'vels_hist_args_V_obs_galnum91_x{X_1}.npy')
V_obs_err_x00 = np.load(base_path + f'vels_hist_args_V_obs_err_galnum91_x{X_1}.npy')
M_mock_x00 = np.load(base_path + f'vels_hist_args_M_mock_galnum91_x{X_1}.npy')
indiv_loglike_x00 = np.load(base_path + f'vels_hist_args_indiv_log_likelihoods_galnum91_x{X_1}.npy')

# Load data for x=0.5 (with selection)
V_mocks_mode_x05 = np.load(base_path + f'vels_hist_args_V_mocks_mode_galnum91_x{X_2}.npy')
V_obs_x05 = np.load(base_path + f'vels_hist_args_V_obs_galnum91_x{X_2}.npy')
V_obs_err_x05 = np.load(base_path + f'vels_hist_args_V_obs_err_galnum91_x{X_2}.npy')
M_mock_x05 = np.load(base_path + f'vels_hist_args_M_mock_galnum91_x{X_2}.npy')
indiv_loglike_x05 = np.load(base_path + f'vels_hist_args_indiv_log_likelihoods_galnum91_x{X_2}.npy')
# Print summary statistics
print(f'x=0.0 (No selection):')
print(f'  Galaxy number: 91')
print(f'  Mass: {M_mock_x00:.4g} M_sun^10')
print(f'  Log Likelihood: {indiv_loglike_x00:.3g}')
print(f'  Observed V: {V_obs_x00:.4f}, Error: {V_obs_err_x00:.4f}')
print(f'  Number of mock samples: {len(V_mocks_mode_x00)}')
print()
print(f'x=0.5 (With selection):')
print(f'  Galaxy number: 91')
print(f'  Mass: {M_mock_x05:.4g} M_sun^10')
print(f'  Log Likelihood: {indiv_loglike_x05:.3g}')
print(f'  Observed V: {V_obs_x05:.4f}, Error: {V_obs_err_x05:.4f}')
print(f'  Number of mock samples: {len(V_mocks_mode_x05)}')

# Filter out NaN values
vel_sims_clean_x00 = V_mocks_mode_x00[~np.isnan(V_mocks_mode_x00)]
vel_sims_clean_x05 = V_mocks_mode_x05[~np.isnan(V_mocks_mode_x05)]

print(f'\nAfter filtering NaNs:')
print(f'  x=0.0: {len(vel_sims_clean_x00)} samples')
print(f'  x=0.5: {len(vel_sims_clean_x05)} samples')

# Create the figure
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

colors = ["red", "orange", "blue"]

# Plot histograms for both cases
n_x05, bins_x05, patches_x05 = ax.hist(vel_sims_clean_x05, bins=50, color=colors[2], alpha=0.5, 
                                        density=False, label=rf'Predicted $v$ ($x={X_2}$)')
n_x00, bins_x00, patches_x00 = ax.hist(vel_sims_clean_x00, bins=50, color=colors[1], alpha=0.5, 
                                        density=False, label=rf'Predicted $v$ ($x={X_1}$)')

# Gaussian curve for the observed data (should be the same for both)
x = np.linspace(V_obs_x00 - 5*V_obs_err_x00, V_obs_x00 + 5*V_obs_err_x00, 200)
gaussian_pdf = (1 / (V_obs_err_x00 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - V_obs_x00) / V_obs_err_x00)**2)

# Scale the Gaussian to match the histogram height (use max from x=0.0 case as reference)
max_hist_count = max(max(n_x00), max(n_x05))
gaussian_scale_factor = max_hist_count / max(gaussian_pdf)
y_scaled = gaussian_pdf * gaussian_scale_factor

# Plot the scaled Gaussian curve
ax.plot(x, y_scaled, color=colors[0], linewidth=2.0, alpha=0.6)

# Plot vertical line for the observed mean
ax.axvline(V_obs_x00, color=colors[0], linestyle='--', linewidth=2.0, alpha=0.6, label=r'SPARC $v$')

# Labels and formatting
ax.set_xlabel(r'$v \equiv \log \left( V \, \left[ \mathrm{km \, s^{-1}} \right] \right)$', fontsize=18)
ax.set_ylabel('Count', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=18)
ax.legend(fontsize=16, loc='upper left')

print(f'\nLog Likelihoods:')
print(fr'  $\mathrm{{x}}={X_1}: {indiv_loglike_x00:.3f}')
print(fr'  $\mathrm{{x}}={X_2}: {indiv_loglike_x05:.3f}')
print(fr'  Difference ($\mathrm{{x}}={X_2} - \mathrm{{x}}={X_1}$): {indiv_loglike_x05 - indiv_loglike_x00:.3f}')

plt.tight_layout()
plt.savefig(base_path + f'velocity_comparison_x{X_1}_vs_x{X_2}_galaxy91.pdf', dpi=300, bbox_inches='tight')
print(f'\nPlot saved to: {base_path}velocity_comparison_x{X_1}_vs_x{X_2}_galaxy91.pdf')