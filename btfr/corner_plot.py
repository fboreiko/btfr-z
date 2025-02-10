import numpy as np
from scipy.interpolate import RegularGridInterpolator
import emcee
import corner
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set font to Computer Modern (LaTeX default) and enable LaTeX rendering
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern']
rcParams['text.usetex'] = True

# Load the 3D array of log likelihood values
file_path = 'likelihood_grid_filepath.npy'
likelihood_grid = np.load(file_path)

# Define the parameter ranges for alpha, scatter, and nu
alpha_proxy_range = np.linspace(-np.pi / 2, 0, 31)
scatter_range = np.linspace(0.01, 1, 31)
nu_range = np.round(np.linspace(-3.0, 3.0, 31), 1)

print('Interpolation started')

# Interpolate the likelihood grid
log_likelihood_interp = RegularGridInterpolator(
    (alpha_proxy_range, scatter_range, nu_range), 
    likelihood_grid,
    bounds_error=False,
    fill_value=-np.inf  # log-likelihood should be very low outside bounds
)

print('Interpolation complete')

# Define the log-posterior function
def log_posterior(params):
    alpha, scatter, nu = params

    # Interpolate the log-likelihood
    log_likelihood = log_likelihood_interp([alpha, scatter, nu])[0]
    
    # Add priors (if any). Here we assume uniform priors within parameter ranges
    if not (alpha_proxy_range[0] <= alpha <= alpha_proxy_range[-1] and
            scatter_range[0] <= scatter <= scatter_range[-1] and
            nu_range[0] <= nu <= nu_range[-1]):
        return -np.inf  # return very low log-posterior outside bounds

    return log_likelihood  # + log(prior), if you have priors

# Initialize MCMC with emcee
ndim = 3  # Number of parameters
nwalkers = 50  # Number of MCMC walkers
nsteps = 5000  # Maximum number of MCMC steps
warmup_steps = 200  # Number of warm-up steps
check_interval = 100  # Interval for checking convergence

# Initial positions of walkers
initial_pos = [
    [np.random.uniform(alpha_proxy_range[0], alpha_proxy_range[-1]), np.random.uniform(scatter_range[0], scatter_range[-1]), np.random.uniform(nu_range[0], nu_range[-1])]
    for _ in range(nwalkers)
]

# Set up the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)

# Run warm-up (burn-in) phase
print("Running warm-up phase...")
sampler.run_mcmc(initial_pos, warmup_steps, progress=True)

# Reset the sampler to discard the warm-up samples
sampler.reset()

# Run the main MCMC sampling with convergence checks
print("Running main sampling phase with convergence checks...")
for step in range(0, nsteps, check_interval):
    sampler.run_mcmc(None, check_interval, progress=True)
    
    # Check the autocorrelation time
    try:
        tau = sampler.get_autocorr_time(tol=0)  # tol=0 for a strict check
        print(f"Estimated autocorrelation time: {tau}")
        
        # Check if we have sufficient samples for convergence
        if np.all(tau * 50 < sampler.iteration):
            print("Chains have likely converged.")
            break
    except emcee.autocorr.AutocorrError:
        print("Not enough samples to estimate autocorrelation time. Continuing sampling...")

# Get samples
samples = sampler.get_chain(flat=True)

# Labels for the parameters (you can adjust these based on your parameter names)
labels = [r"$\tan^{-1}(\alpha)$", r"$\sigma$", r"$\nu$"]

# Create a corner plot
fig = corner.corner(samples, labels=labels, truths=[None, None, None],
                    quantiles=[0.16, 0.5, 0.84], show_titles=True,
                    title_kwargs={"fontsize": 12})

plt.savefig("/Users/fedorboreiko/Documents/Oxford/btfr_z/plots/corner_plot.png", dpi=300)
