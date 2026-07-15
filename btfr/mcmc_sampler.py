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

def print_nan_coordinates(grid):
    """
    Print the coordinates of NaN values in a 3D array.

    Parameters:
        grid (np.ndarray): The 3D array to check for NaN values.
    """
    # Find the indices where the grid is NaN
    nan_indices = np.argwhere(np.isnan(grid))
    
    if nan_indices.size > 0:
        print("Found NaNs at the following coordinates:")
        return nan_indices
    else:
        print("No NaNs found in the grid.")
        return None



def average_neighbors(arr, index):
    """
    Compute the average of the neighbors of the cell at 'index' in a 3D or 4D array.
    Only valid (non-NaN) neighbors within the bounds of the array are used.
    
    Parameters:
        arr (np.ndarray): 3D or 4D array.
        index (tuple): A tuple indicating the index of the cell.
    
    Returns:
        float: The average value of the neighbors. If no valid neighbor is found,
               returns np.nan.
    """
    neighbor_values = []
    dims = len(index)
    
    if dims not in [3, 4]:
        raise ValueError("This function only supports 3D or 4D arrays.")
    
    # Generate ranges for neighbors based on the number of dimensions
    ranges = [range(-1, 2) for _ in range(dims)]
    
    # Iterate over all possible neighbor offsets
    for offsets in np.ndindex(*[len(r) for r in ranges]):
        # Skip the center cell itself
        if all(offset == 0 for offset in offsets):
            continue
        
        # Compute neighbor indices
        neighbor_index = tuple(index[d] + offsets[d] for d in range(dims))
        
        # Check if the neighbor index is within bounds
        if all(0 <= neighbor_index[d] < arr.shape[d] for d in range(dims)):
            neighbor_val = arr[neighbor_index]
            # Only add non-NaN values
            if not np.isnan(neighbor_val):
                neighbor_values.append(neighbor_val)
    
    if neighbor_values:
        return np.mean(neighbor_values)
    else:
        return np.nan
    

# Load and clean your likelihood grid (as before)
file_path = '/Users/fedorboreiko/Documents/Oxford/btfr_z/btfr/likelihood_grid_90am_50stellar_alphaproxy_-1.5707963267948966_1.5707963267948966_scatter_0.01_1.0_x_0.01_0.95_nu_-3.0_3.0_vmaxshift_proper.npy'
likelihood_grid = np.load(file_path)

nan_indxs = print_nan_coordinates(likelihood_grid)

# convert nan_indxs to a list of tuples
if nan_indxs is not None:
    nan_indxs = [tuple(idx) for idx in nan_indxs]
    print(nan_indxs)

    for idx in nan_indxs:
        if np.isnan(likelihood_grid[idx]):
            avg_val = average_neighbors(likelihood_grid, idx)
            print(f"Replacing NaN at index {idx} with average of neighbors: {avg_val}")
            likelihood_grid[idx] = avg_val
        else:
            print(f"Index {idx} is not NaN.")

# Define your grids
alpha_proxy_range = np.linspace(-np.pi / 2, np.pi / 2, 20)
scatter_range     = np.linspace(0.01, 1.0, 20)
x_range         = np.linspace(0.01, 0.95, 20)
nu_range          = np.linspace(-3.0, 3.0, 20)

log_likelihood_interp = RegularGridInterpolator(
    (alpha_proxy_range, scatter_range, x_range, nu_range),
    likelihood_grid,
    bounds_error=False,
    fill_value=-np.inf
)

def log_posterior(params):
    lp = log_likelihood_interp(params)[0]
    # uniform priors
    if not all([
        alpha_proxy_range[0] <= params[0] <= alpha_proxy_range[-1],
        scatter_range[0]     <= params[1] <= scatter_range[-1],
        x_range[0]          <= params[2] <= x_range[-1],
        nu_range[0]          <= params[3] <= nu_range[-1],
    ]):
        return -np.inf
    return lp

ndim, nwalkers = 4, 100
warmup_steps, nsteps, check_interval = 5000, 100000, 100

# initialize
initial_pos = np.column_stack([
    np.random.uniform(r[0], r[-1], size=nwalkers)
    for r in (alpha_proxy_range, scatter_range, x_range, nu_range)
])

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)

# 1) Warm-up
print("Running warm-up phase…")
sampler.run_mcmc(initial_pos, warmup_steps, progress=True)
sampler.reset()  # discard burn-in

# 2) Main sampling with convergence check
old_tau = np.full(ndim, np.inf)
print("Running main sampling with convergence checks…")
for _ in range(nsteps // check_interval):
    sampler.run_mcmc(None, check_interval, progress=True)
    try:
        tau = sampler.get_autocorr_time(tol=0)
    except emcee.autocorr.AutocorrError:
        print("…still too few samples to estimate τ; continuing.")
        continue

    iteration = sampler.iteration  # total samples per walker so far
    # convergence criteria
    criterion1 = np.all(tau * 100 < iteration)
    criterion2 = np.all(np.abs(old_tau - tau) / tau < 0.005)
    print(f"Iteration={iteration}, τ={tau.round(1)}")
    if criterion1 and criterion2:
        print("✔ Chains have likely converged.")
        break
    old_tau = tau.copy()

# 3) Extract and plot
samples = sampler.get_chain(flat=True)
labels  = [r"$\tan^{-1}(\alpha)$", r"$\sigma$", "x", r"$\nu$"]

# Corner plot
fig = corner.corner(
    samples, 
    labels=labels, 
    quantiles=[0.16, 0.5, 0.84], 
    show_titles=True, 
    label_kwargs={"fontsize": 14},
    title_kwargs={"fontsize": 14}
)
# increase tick label size
for ax in fig.get_axes():
    ax.tick_params(labelsize=14)  # Increase tick label size
fig.savefig("/Users/fedorboreiko/Documents/Oxford/btfr_z/plots/corner_plot_4param_vmaxshift.png", dpi=300)

# Trace plots
chain = sampler.get_chain()  # shape = (n_iter, nwalkers, ndim)
fig, axes = plt.subplots(ndim, 1, figsize=(8, 2*ndim), sharex=True)
iters = np.arange(chain.shape[0])
for i in range(ndim):
    for w in range(nwalkers):
        axes[i].plot(iters, chain[:, w, i], alpha=0.3)
    axes[i].set_ylabel(labels[i])
axes[-1].set_xlabel("Step number")
fig.tight_layout()
fig.savefig("/Users/fedorboreiko/Documents/Oxford/btfr_z/plots/trace_plots_4param_vmaxshift.png", dpi=300)

# Save the full chain and log probabilities
np.save("/Users/fedorboreiko/Documents/Oxford/btfr_z/btfr/samples/full_chain_4param_vmaxshift.npy", sampler.get_chain())
np.save("/Users/fedorboreiko/Documents/Oxford/btfr_z/btfr/samples/log_prob_4param_vmaxshift.npy", sampler.get_log_prob())
np.save("/Users/fedorboreiko/Documents/Oxford/btfr_z/btfr/samples/flat_samples_4param_vmaxshift.npy", samples)

