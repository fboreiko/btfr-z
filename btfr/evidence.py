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

# ---- Helper functions for grid handling ----

def print_nan_coordinates(grid):
    """
    Print the coordinates of NaN values in a 3D array.
    """
    nan_indices = np.argwhere(np.isnan(grid))
    if nan_indices.size > 0:
        print("Found NaNs at the following coordinates:")
        return nan_indices
    else:
        print("No NaNs found in the grid.")
        return None

def average_neighbors(arr, index):
    """
    Compute the average of the neighbors of the cell at 'index' in a 4D array.
    Only valid (non-NaN) neighbors within the bounds of the array are used.
    """
    i, j, k, l = index
    neighbor_values = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            for dk in [-1, 0, 1]:
                for dl in [-1, 0, 1]:
                    if di == dj == dk == dl == 0:
                        continue
                    ni, nj, nk, nl = i + di, j + dj, k + dk, l + dl
                    if (0 <= ni < arr.shape[0] and 0 <= nj < arr.shape[1] and 
                        0 <= nk < arr.shape[2] and 0 <= nl < arr.shape[3]):
                        neighbor_val = arr[ni, nj, nk, nl]
                        if not np.isnan(neighbor_val):
                            neighbor_values.append(neighbor_val)
    return np.mean(neighbor_values) if neighbor_values else np.nan

# ---- Load and interpolate the likelihood grid ----

file_path = '/Users/fedorboreiko/Documents/Oxford/btfr_z/btfr/likelihood_grid_100am_1000stellar_alphaproxy_-1.5707963267948966_0.0_scatter_0.01_1.0_nu_-3.0_3.0.npy'
likelihood_grid = np.load(file_path)
nan_indxs = print_nan_coordinates(likelihood_grid)

# Define parameter ranges
alpha_proxy_range = np.linspace(-np.pi / 2, 0, 31)
scatter_range = np.linspace(0.01, 1.0, 31)
#x_range = np.linspace(0.01, 0.95, 20)
nu_range = np.linspace(-3.0, 3.0, 31)

print('Interpolation started') #x_range
log_likelihood_interp = RegularGridInterpolator(
    (alpha_proxy_range, scatter_range, nu_range), 
    likelihood_grid,
    bounds_error=False,
    fill_value=-np.inf  # log-likelihood should be very low outside bounds
)
print('Interpolation complete')

# Define the log-posterior function
def log_posterior_fn(params):
    alpha_proxy, scatter, nu = params
    log_likelihood = log_likelihood_interp([alpha_proxy, scatter, nu])[0]
    # Assuming uniform priors within the parameter ranges: #x_range[0] <= x <= x_range[-1] and
    if not (alpha_proxy_range[0] <= alpha_proxy <= alpha_proxy_range[-1] and
            scatter_range[0] <= scatter <= scatter_range[-1] and
            nu_range[0] <= nu <= nu_range[-1]):
        return -np.inf
    return log_likelihood

# ---- Set up and run MCMC sampling with emcee ----

ndim = 3           # Number of parameters
nwalkers = 100      # Number of MCMC walkers
nsteps = 2000       # Maximum number of MCMC steps for the main sampling phase
warmup_steps = 1000 # Burn-in steps
check_interval = 100

# Initial positions for walkers #np.random.uniform(x_range[0], x_range[-1]),
initial_pos = [
    [np.random.uniform(alpha_proxy_range[0], alpha_proxy_range[-1]),
     np.random.uniform(scatter_range[0], scatter_range[-1]),
     np.random.uniform(nu_range[0], nu_range[-1])]
    for _ in range(nwalkers)
]

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior_fn)

# Run warm-up (burn-in)
print("Running warm-up phase...")
sampler.run_mcmc(initial_pos, warmup_steps, progress=True)
sampler.reset()  # Discard burn-in samples

# Run the main sampling phase with convergence checks
print("Running main sampling phase with convergence checks...")
for step in range(0, nsteps, check_interval):
    sampler.run_mcmc(None, check_interval, progress=True)
    try:
        tau = sampler.get_autocorr_time(tol=0)
        print(f"Estimated autocorrelation time: {tau}")
        if np.all(tau * 50 < sampler.iteration):
            print("Chains have likely converged.")
            break
    except emcee.autocorr.AutocorrError:
        print("Not enough samples to estimate autocorrelation time. Continuing sampling...")

# Get the chain samples and log-posterior values without flattening.
# The harmonic evidence function expects samples of shape (nchains, nsamples, ndim)
samples = sampler.get_chain()            # Shape: (nwalkers, nsteps, ndim)
log_posterior_values = sampler.get_log_prob()  # Shape: (nwalkers, nsteps)

# Plot the corner plot for visualization
labels = [r"$\tan^{-1}(\alpha)$", r"$\sigma$", "x", r"$\nu$"]
fig = corner.corner(samples.reshape((-1, ndim)), labels=labels,
                    quantiles=[0.16, 0.5, 0.84], show_titles=True,
                    title_kwargs={"fontsize": 12})
plt.savefig("/Users/fedorboreiko/Documents/Oxford/btfr_z/plots/corner_plot_3_params.png", dpi=300)
plt.show()

# Check the number of samples with x > 0.9
x_samples = samples[:, :, 2]
x_fraction = np.mean(x_samples > 0.9)


# ---- Evaluate the Evidence using the harmonic_evidence function ----

# Make sure the 'harmonic' package is installed.
# The harmonic_evidence function is assumed to be defined or imported beforehand.
def harmonic_evidence(samples, log_posterior, temperature=0.8, epochs_num=20,
                      return_flow_samples=True, verbose=True):
    """
    Calculate the evidence using the `harmonic` package.
    """
    try:
        import harmonic as hm
    except ImportError:
        raise ImportError("The `harmonic` package is required to calculate the evidence.") from None

    if samples.ndim != 3:
        raise ValueError("Samples must be a 3-dimensional array of shape `(nchains, nsamples, ndim)`.")
    if log_posterior.ndim != 2 or log_posterior.shape[:2] != samples.shape[:2]:
        raise ValueError("Log posterior must be a 2-dimensional array of shape `(nchains, nsamples)`.")

    ndim = samples.shape[-1]
    chains = hm.Chains(ndim)
    chains.add_chains_3d(samples, log_posterior)
    chains_train, chains_infer = hm.utils.split_data(chains, training_proportion=0.5)

    model = hm.model.RQSplineModel(ndim, standardize=True, temperature=temperature)
    model.fit(chains_train.samples, epochs=epochs_num, verbose=verbose)

    ev = hm.Evidence(chains_infer.nchains, model)
    ev.add_chains(chains_infer)
    ln_inv_evidence = ev.ln_evidence_inv
    err_ln_inv_evidence = ev.compute_ln_inv_evidence_errors()

    if return_flow_samples:
        samples_reshaped = samples.reshape((-1, ndim))
        samp_num = samples_reshaped.shape[0]
        flow_samples = model.sample(samp_num)
        return ln_inv_evidence, err_ln_inv_evidence, flow_samples

    return ln_inv_evidence, err_ln_inv_evidence

# Now compute the evidence
try:
    ln_inv_evidence, err_ln_inv_evidence, flow_samples = harmonic_evidence(
        samples, log_posterior_values,
        temperature=0.8,
        epochs_num=20,
        return_flow_samples=True,
        verbose=True
    )
    print("Log inverse evidence:", ln_inv_evidence)
    print("Error on log inverse evidence:", err_ln_inv_evidence)
except ImportError as e:
    print(e)