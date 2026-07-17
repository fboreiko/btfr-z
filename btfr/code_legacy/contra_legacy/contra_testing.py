from mpi4py import MPI
import numpy as np
import pickle
import pandas as pd
from btfr.code_legacy.contra_legacy.contra_unvect import do_contra
from scipy.optimize import root_scalar

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Parameters
nu_values = np.linspace(-1.0, 1.5, 40)
#nu_values = np.round(nu_values, 1)
num_samples = 100  # Number of test points per emulator

# Define parameter ranges
log_c_range = [0.5, 2.2]
log_fb_range = [-2.5, -1]
log_rb_range = [-2.5, -1.6]
log_rf_range = [-4, 0]

with open("/Users/fedorboreiko/Documents/Oxford/Personal_codes/Codebase/contra_emulators/grids_fullrange.pkl", "rb") as f:
    interpolators = pickle.load(f)

# Distribute emulators among available MPI processes
if rank < len(nu_values):
    nu = nu_values[rank]  # Assign one emulator per process

    contra_interpolator = interpolators[nu]

    # Generate random test points
    log_c_values = np.random.uniform(log_c_range[0], log_c_range[1], num_samples)
    log_fb_values = np.random.uniform(log_fb_range[0], log_fb_range[1], num_samples)
    log_rb_values = np.random.uniform(log_rb_range[0], log_rb_range[1], num_samples)
    log_rf_values = np.random.uniform(log_rf_range[0], log_rf_range[1], num_samples)

    # Storage for errors
    absolute_errors = []
    relative_errors = []
    percentage_errors = []

    # Function to compute log_rf_direct
    def compute_log_rf_direct(log_ri_direct, c, fb, rb):
        ri_direct = 10**log_ri_direct
        rf_direct, _ = do_contra(
            np.array([ri_direct]), np.array([c]), np.array([fb]), np.array([rb]), A=1.6, w=0.8
        )
        rf_direct = (rf_direct / ri_direct)**nu * ri_direct
        log_rf_direct = np.log10(rf_direct)
        return log_rf_direct

    # Objective function for root finding
    def objective_function(log_ri_direct, c, fb, rb, log_rf_value):
        return compute_log_rf_direct(log_ri_direct, c, fb, rb) - log_rf_value

    # Iterate over test points
    for i in range(num_samples):
        log_c_value = log_c_values[i]
        log_fb_value = log_fb_values[i]
        log_rb_value = log_rb_values[i]
        log_rf_value = log_rf_values[i]

        # Convert log-scaled values to linear scale
        c = 10**log_c_value
        fb = 10**log_fb_value
        rb = 10**log_rb_value

        # Emulated mhi
        points = np.array([[log_c_value, log_fb_value, log_rb_value, log_rf_value]])
        log_mhi_emulated = contra_interpolator(points)

        # Ensure the interpolator produced a valid value
        if log_mhi_emulated[0] == 0:
            continue  # Skip invalid points

        # Solve for log_ri_direct
        result = root_scalar(objective_function, args=(c, fb, rb, log_rf_value), bracket=[-7, 1], method="brentq")

        if not result.converged:
            continue  # Skip points where root finding fails

        log_ri_direct = result.root
        ri_direct = 10**log_ri_direct

        # Compute direct mhi
        rf_direct, mhi_direct = do_contra(
            np.array([ri_direct]), np.array([c]), np.array([fb]), np.array([rb]), A=1.6, w=0.8
        )
        rf_direct = (rf_direct / ri_direct)**nu * ri_direct
        log_mhi_direct = np.log10(mhi_direct)

        # Compute errors
        abs_error = abs(log_mhi_emulated[0] - log_mhi_direct[0])
        rel_error = abs_error / abs(log_mhi_direct[0])
        percent_error = abs(10**log_mhi_emulated[0] - mhi_direct[0]) / abs(mhi_direct[0]) * 100

        absolute_errors.append(abs_error)
        relative_errors.append(rel_error)
        percentage_errors.append(percent_error)

    # Compute error statistics for this process
    mean_abs_error = np.mean(absolute_errors) if absolute_errors else 0
    mean_rel_error = np.mean(relative_errors) if relative_errors else 0
    mean_percentage_error = np.mean(percentage_errors) if percentage_errors else 0

    # Send results to rank 0
    results = np.array([nu, mean_abs_error, mean_rel_error, mean_percentage_error])
    comm.send(results, dest=0, tag=rank)

# Rank 0 gathers and aggregates results into a Pandas DataFrame
if rank == 0:
    all_results = []
    for src in range(min(size, len(nu_values))):  # Only expect results from valid ranks
        results = comm.recv(source=src, tag=src)
        all_results.append(results)

    # Convert to Pandas DataFrame
    df_results = pd.DataFrame(all_results, columns=["nu", "Mean Abs Error", "Mean Rel Error", "Mean % Error"])

    # Save results to CSV file
    df_results.to_csv("/Users/fedorboreiko/Documents/Oxford/btfr_z/contra/emulator_results.csv", index=False)

    # Print summary
    print("\nFinal Results Saved to emulator_results.csv")
    print(df_results)





