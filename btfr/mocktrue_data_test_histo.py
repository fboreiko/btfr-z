import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib import rcParams

# Set font to Computer Modern (LaTeX default) and enable LaTeX rendering
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern']
rcParams['text.usetex'] = True

# True values for the parameters (update as needed)
true_alpha = -0.5
true_scatter = 0.1
true_x = 0.5
true_nu = np.linspace(-3.0, 3.0, 20)[12]

# Load the best-fit parameters along with uncertainties
# The file now contains 8 columns: Alpha_median, Alpha_sigma, Scatter_median, Scatter_sigma, x_median, x_sigma, nu_median, nu_sigma
data = np.loadtxt("/Users/fedorboreiko/Documents/Oxford/btfr_z/btfr/best_fit_parameters.txt")

# Separate inferred medians and posterior uncertainties
inferred_alpha, alpha_unc = data[:, 0], data[:, 1]
inferred_scatter, scatter_unc = data[:, 2], data[:, 3]
inferred_x, x_unc = data[:, 4], data[:, 5]
inferred_nu, nu_unc = data[:, 6], data[:, 7]

# Compute normalized (weighted) residuals:
# (Inferred median - True value) divided by the posterior uncertainty
resid_alpha = (true_alpha - inferred_alpha) / alpha_unc
resid_scatter = (true_scatter - inferred_scatter) / scatter_unc
resid_x = (true_x - inferred_x) / x_unc
resid_nu = (true_nu - inferred_nu) / nu_unc

# Define parameters for plotting
parameters = [r"\tan^{-1}(\alpha)", r"\sigma", "x", r"\nu"]
residuals = [resid_alpha, resid_scatter, resid_x, resid_nu]

# Create histograms with the standard Gaussian overlay
plt.figure(figsize=(15, 4))
for i, (param, resid) in enumerate(zip(parameters, residuals)):
    plt.subplot(1, 4, i + 1)
    # Plot histogram of weighted residuals
    plt.hist(resid, bins=10, density=True, alpha=0.6, color="blue", label="Histogram")
    
    # Overlay standard Gaussian curve: mean=0, std=1
    x = np.linspace(min(resid)-0.5, max(resid)+0.5, 1000)
    standard_gaussian = norm.pdf(x, loc=0, scale=1)
    plt.plot(x, standard_gaussian, "r--", label="Standard Normal")

    # Build math expression without nesting $: param strings are bare math tokens
    core = r"\mathrm{x}" if param == "x" else param
    plt.xlabel(r"$z(" + core + r")$", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #plt.ylabel("Density")
    #plt.legend()

plt.tight_layout()
plt.savefig("plots/weighted_residuals_histogram.png", dpi=300)