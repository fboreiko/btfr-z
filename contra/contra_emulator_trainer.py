from mpi4py import MPI
import numpy as np
import pickle, os, sys
from scipy.interpolate import RegularGridInterpolator
from contra_unvect import do_contra
from scipy.optimize import root_scalar
from scipy.signal import find_peaks
from tqdm import tqdm

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Parameters
N_SAMPLES = 50
nu_values = np.linspace(-3.0, 3.0, 20)
n_groups = len(nu_values)
# Total processes must equal (number of nu values) * 4.
if size != n_groups * 5:
    if rank == 0:
        print("Error: Total MPI processes must equal number of nu values * 5")
    sys.exit(1)

# Determine group id and local rank within the group
group_id = rank // 5
local_rank = rank % 5

# Create a subcommunicator for each group (each nu value)
subcomm = comm.Split(color=group_id, key=local_rank)

# Assign the nu value for this group
nu = nu_values[group_id]

# Define parameter arrays (shared within each group)
log_rf_values = np.linspace(-4.8, 0.3, N_SAMPLES, dtype=np.float128)
c_values  = np.logspace(0, 3.9, N_SAMPLES, dtype=np.float128)
fb_values = np.logspace(-3.6, -0.03, N_SAMPLES, dtype=np.float128)
rb_values = np.logspace(-3, -1, N_SAMPLES, dtype=np.float128)

log_c_values  = np.log10(c_values)
log_fb_values = np.log10(fb_values)
log_rb_values = np.log10(rb_values)

# Define the function to compute log_rf_direct
def compute_log_rf_direct(log_ri, c, fb, rb, nu):
    ri = 10**log_ri
    rf, _ = do_contra(
        np.array([ri], dtype=np.float128), c, fb, rb, A=1.6, w=0.8
    )
    # Compute the transformed rf_direct value as given
    rf = (rf / ri)**nu * ri
    log_rf = np.log10(rf)
    return log_rf

# Define the objective function for root finding
def objective_function(log_ri_value, c, fb, rb, nu, log_rf_value):
    return compute_log_rf_direct(log_ri_value, c, fb, rb, nu) - log_rf_value

# Function to compute the value for a given flat index.
def compute_value_for_flat_index(flat_index, nu):
    # Convert flat index to 4D indices
    i, j, k, l = np.unravel_index(flat_index, (N_SAMPLES, N_SAMPLES, N_SAMPLES, N_SAMPLES))
    c = c_values[i]
    fb = fb_values[j]
    rb = rb_values[k]
    log_rf = log_rf_values[l]
    try:
        # Primary attempt using an initial bracket
        result = root_scalar(
            objective_function,
            args=(c, fb, rb, nu, log_rf),
            bracket=[-9, 1],
            method="brentq",
            xtol=1e-10,
            maxiter=10000
        )
        log_ri = result.root
        ri = 10**log_ri
        _, mhi = do_contra(np.array([ri], dtype=np.float128), c, fb, rb, A=1.6, w=0.8)
        value = np.log10(mhi[0])
    except ValueError:
        # Fallback: search with a probe array
        ri_probe_array = np.logspace(-7, 1, 30, dtype=np.float128)
        rf_probe_array, _ = do_contra(ri_probe_array, c, fb, rb, A=1.6, w=0.8)
        rf_probe_array = (rf_probe_array / ri_probe_array)**(nu) * ri_probe_array
        peaks, _ = find_peaks(rf_probe_array) 
        if len(peaks) > 0 and log_rf <= np.log10(rf_probe_array[peaks[0]]):
            upper_bound = np.log10(ri_probe_array[peaks[0]])
        else:
            upper_bound = 5  # fallback default if none are found
        try:
            result = root_scalar(
                objective_function,
                args=(c, fb, rb, nu, log_rf),
                bracket=[-7, upper_bound],
                method="brentq",
                xtol=1e-10,
                maxiter=10000
            )
            if not result.converged:
                value = np.nan
            else:
                log_ri = result.root
                ri = 10**log_ri
                _, mhi = do_contra(np.array([ri], dtype=np.float128), c, fb, rb, A=1.6, w=0.8)
                value = np.log10(mhi[0])
        except ValueError as e:
            print(f"Group {group_id}, local_rank {local_rank}: Root finding failed for nu={nu}, c={c}, fb={fb}, rb={rb}, log_rf={log_rf}")
            print(e)
            value = np.nan
    return (flat_index, value)

# Each group will compute the full N_SAMPLES**4 grid.
total_iterations = N_SAMPLES**4
chunk_size = total_iterations // 5  # each of the 4 processes in the group gets roughly equal share
if local_rank < 4:
    my_start = local_rank * chunk_size
    my_end = my_start + chunk_size
else:
    my_start = local_rank * chunk_size
    my_end = total_iterations  # last process takes any remainder

# Each process computes its assigned iterations.
my_results = []
for flat_index in range(my_start, my_end):
    my_results.append(compute_value_for_flat_index(flat_index, nu))

# Gather computed data from all 5 processes in this group to the group leader (local_rank==0)
group_results = subcomm.gather(my_results, root=0)

# Only the group leader assembles the full array and trains the interpolator.
group_interpolator = None
if subcomm.Get_rank() == 0:
    # Allocate full 4D grid array for the computed log(mhi) values.
    log_mhi_values = np.empty((N_SAMPLES, N_SAMPLES, N_SAMPLES, N_SAMPLES))
    # group_results is a list of lists from the 4 processes.
    for proc_results in group_results:
        for flat_index, value in proc_results:
            i, j, k, l = np.unravel_index(flat_index, (N_SAMPLES, N_SAMPLES, N_SAMPLES, N_SAMPLES))
            log_mhi_values[i, j, k, l] = value

    # Train the interpolator using the computed data.
    interpolator = RegularGridInterpolator(
        (log_c_values, log_fb_values, log_rb_values, log_rf_values),
        log_mhi_values,
        bounds_error=False, fill_value=None
    )
    print(f"Group {group_id}: Interpolator training for nu = {nu} finished.")
    group_interpolator = {nu: interpolator}

# Now, let every process send the group's interpolator (only group leader has one) to global rank 0.
# Non-leaders send None.
all_interpolators = comm.gather(group_interpolator, root=0)

# Global rank 0 collects and saves the interpolators.
if rank == 0:
    final_interpolators = {}
    for interp_dict in all_interpolators:
        if interp_dict is not None:
            final_interpolators.update(interp_dict)
    output_dir = "contra_emulators"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, "contra_interpolators_fullrange.pkl")
    with open(output_filename, "wb") as f:
        pickle.dump(final_interpolators, f)
    print("Interpolators saved to", output_filename)

