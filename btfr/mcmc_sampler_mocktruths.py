import numpy as np
from scipy.interpolate import RegularGridInterpolator
import emcee
import corner
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpi4py import MPI
import os

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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

# Define true values for parameters (update these with your actual truth values)
truth_alpha_proxy = -0.5
truth_scatter = 0.1  # example true value for scatter
truth_x = 0.5
truth_nu = np.linspace(-3.0, 3.0, 20)[12]

# Only load data on rank 0 to avoid multiple file reads
if rank == 0:
    # Load the 4D array of log-likelihood values (36x31x31x31)
    file_path = '/Users/fedorboreiko/Documents/Oxford/btfr_z/btfr/likelihood_grid_50mocktruths_90am_49stellar_alphaproxy_-1.571_1.571_scatter_0.010_1.000_x_0.010_0.950_nu_-3.000_3.000.npy'
    likelihood_grids = np.load(file_path)
    print(f"Loaded likelihood grids with shape: {likelihood_grids.shape}")
    print(f"Running on {size} MPI processes")
    
    # Create output directory if it doesn't exist
    os.makedirs("/Users/fedorboreiko/Documents/Oxford/btfr_z/plots/corner_plot_mocktruths", exist_ok=True)
else:
    likelihood_grids = None

# Broadcast the likelihood grids to all processes
likelihood_grids = comm.bcast(likelihood_grids, root=0)

# Define the parameter ranges for alpha, scatter, and nu
alpha_proxy_range = np.linspace(-np.pi / 2, np.pi / 2, 20)
scatter_range = np.linspace(0.01, 1.0, 20)
x_range = np.linspace(0.01, 0.95, 20)
nu_range = np.linspace(-3.0, 3.0, 20)

# Function to calculate the best-fit parameters for a single likelihood grid
def process_single_grid(grid_index, likelihood_grid):
    print(f'Rank {rank}: Processing grid {grid_index + 1}/50')

    nan_indxs = print_nan_coordinates(likelihood_grid)

    # convert nan_indxs to a list of tuples
    if nan_indxs is not None:
        nan_indxs = [tuple(idx) for idx in nan_indxs]

        for idx in nan_indxs:
            if np.isnan(likelihood_grid[idx]):
                avg_val = average_neighbors(likelihood_grid, idx)
                print(f"Rank {rank}: Replacing NaN at index {idx} with average of neighbors: {avg_val}")
                likelihood_grid[idx] = avg_val
            else:
                print(f"Rank {rank}: Index {idx} is not NaN.")

    # Interpolate the likelihood grid
    log_likelihood_interp = RegularGridInterpolator(
        (alpha_proxy_range, scatter_range, x_range, nu_range), 
        likelihood_grid,
        bounds_error=False,
        fill_value=-np.inf  # log-likelihood should be very low outside bounds
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

    # Set up the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)

    # 1) Run warm-up (burn-in) phase
    print(f"Rank {rank}: Running warm-up phase for grid {grid_index + 1}...")
    sampler.run_mcmc(initial_pos, warmup_steps, progress=True)
    sampler.reset()  # discard burn-in

    # 2) Main sampling with convergence check
    old_tau = np.full(ndim, np.inf)
    print(f"Rank {rank}: Running main sampling with convergence checks for grid {grid_index + 1}...")
    for _ in range(nsteps // check_interval):
        sampler.run_mcmc(None, check_interval, progress=True)
        try:
            tau = sampler.get_autocorr_time(tol=0)
        except emcee.autocorr.AutocorrError:
            print(f"Rank {rank}: ...still too few samples to estimate τ; continuing.")
            continue

        iteration = sampler.iteration  # total samples per walker so far
        # convergence criteria
        criterion1 = np.all(tau * 100 < iteration)
        criterion2 = np.all(np.abs(old_tau - tau) / tau < 0.005)
        print(f"Rank {rank}: Iteration={iteration}, τ={tau.round(1)}")
        if criterion1 and criterion2:
            print(f"Rank {rank}: ✔ Chains have likely converged for grid {grid_index + 1}.")
            break
        old_tau = tau.copy()

    # Get samples and compute quantiles for best-fit and uncertainties
    samples = sampler.get_chain(flat=True)
    q16, q50, q84 = np.percentile(samples, [16, 50, 84], axis=0)
    best_fit = q50            # median values
    sigma = (q84 - q16) / 2   # symmetric uncertainties

    # Output the results for this grid
    print(f"Rank {rank}: Best-fit parameters for grid {grid_index + 1}:")
    print(f"Rank {rank}: Alpha: {best_fit[0]:.3f} ± {sigma[0]:.3f}, Scatter: {best_fit[1]:.3f} ± {sigma[1]:.3f}, \
            x: {best_fit[2]:.3f} ± {sigma[2]:.3f}, Nu: {best_fit[3]:.3f} ± {sigma[3]:.3f}")

    # Create and save a corner plot with truth values
    labels  = [r"$\tan^{-1}(\alpha)$", r"$\sigma$", "x", r"$\nu$"]
    fig = corner.corner(samples, labels=labels,
                        truths=[truth_alpha_proxy, truth_scatter, truth_x, truth_nu],
                        truth_color='red',
                        truth_kwargs={'marker': 'o', 'markersize': 3},
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True,
                        label_kwargs={"fontsize": 14},
                        title_kwargs={"fontsize": 14})
    
    for ax in fig.get_axes():
        ax.tick_params(labelsize=14)
    plt.savefig(f"/Users/fedorboreiko/Documents/Oxford/btfr_z/plots/corner_plot_mocktruths/corner_plot_4param_mocktrue_grid_{grid_index + 1}.png", dpi=300)
    plt.close(fig)

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
    fig.savefig(f"/Users/fedorboreiko/Documents/Oxford/btfr_z/plots/corner_plot_mocktruths/trace_plots_4param_mocktrue_grid_{grid_index + 1}.png", dpi=300)

    # Return best-fit parameters and uncertainties as:
    return np.array([
        best_fit[0], sigma[0],
        best_fit[1], sigma[1],
        best_fit[2], sigma[2],
        best_fit[3], sigma[3]
    ])

# Determine which grids this rank should process
num_grids = likelihood_grids.shape[0]
grids_per_rank = num_grids // size
remainder = num_grids % size

# Calculate start and end indices for this rank
if rank < remainder:
    start_idx = rank * (grids_per_rank + 1)
    end_idx = start_idx + grids_per_rank + 1
else:
    start_idx = rank * grids_per_rank + remainder
    end_idx = start_idx + grids_per_rank

# Process assigned grids
local_results = []
my_grid_indices = list(range(start_idx, end_idx))

if rank == 0:
    print(f"Total grids to process: {num_grids}")
    print(f"Each rank processing approximately {grids_per_rank} grids")

print(f"Rank {rank}: Processing grids {my_grid_indices}")

for grid_idx in my_grid_indices:
    result = process_single_grid(grid_idx, likelihood_grids[grid_idx, :, :, :, :])
    local_results.append(result)

# Gather all results on rank 0
all_results = comm.gather(local_results, root=0)

# Only rank 0 processes and saves the final results
if rank == 0:
    # Flatten the gathered results and sort by original grid index
    final_results = []
    for rank_idx, rank_results in enumerate(all_results):
        if rank_idx < remainder:
            start = rank_idx * (grids_per_rank + 1)
        else:
            start = rank_idx * grids_per_rank + remainder
        
        for i, result in enumerate(rank_results):
            final_results.append((start + i, result))
    
    # Sort by grid index to maintain original order
    final_results.sort(key=lambda x: x[0])
    
    # Extract just the results (remove grid indices)
    final_results_array = np.array([result for _, result in final_results])
    
    # Save the best-fit results and uncertainties for all grids
    header = "Alpha_median Alpha_sigma Scatter_median Scatter_sigma x_median x_sigma Nu_median Nu_sigma"
    np.savetxt("best_fit_parameters.txt", final_results_array, header=header, fmt="%.6f")
    
    print("All results saved to best_fit_parameters.txt")

# Finalize MPI
MPI.Finalize()