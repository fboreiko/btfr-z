from mpi4py import MPI
import numpy as np
import pickle
import os
from scipy.interpolate import RegularGridInterpolator
from contra_unvect import do_contra
from scipy.optimize import root_scalar
import sys

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Function to compute log_rf_direct
def compute_log_rf_direct(log_ri, c, fb, rb, nu):
    ri = 10**log_ri
    rf, _ = do_contra(
        np.array([ri]), np.array([c]), np.array([fb]), np.array([rb]), A=1.6, w=0.8
    )
    # Compute the transformed rf_direct value as given
    rf = (rf / ri)**nu * ri
    log_rf = np.log10(rf)
    return log_rf

# Objective function for root finding
def objective_function(log_ri_value, c, fb, rb, nu, log_rf_value):
    return compute_log_rf_direct(log_ri_value, c, fb, rb, nu) - log_rf_value

# Parameters
N_SAMPLES = 50
nu_values = np.round(np.linspace(-3.0, 3.0, 31), 2)

log_rf_values = np.linspace(-4.8, 0.3, N_SAMPLES)

c_values  = np.logspace(0.01, 3, N_SAMPLES)
fb_values = np.logspace(-3.3, -0.3, N_SAMPLES)
rb_values = np.logspace(-2.9, -1.3, N_SAMPLES)

log_c_values  = np.log10(c_values)
log_fb_values = np.log10(fb_values)
log_rb_values = np.log10(rb_values)

interpolators = {}

# Distribute emulators among available MPI processes
if rank < len(nu_values):
    nu = nu_values[rank]  # Assign one emulator per process
    # Preallocate a 4D array to store log(mhi) values.
    log_mhi_values = np.empty((N_SAMPLES, N_SAMPLES, N_SAMPLES, N_SAMPLES))
    
    print(f'Process {rank}: Dataspawn for nu = {nu} started')
    
    # Loop over the 4D grid
    for i, c in enumerate(c_values):
        for j, fb in enumerate(fb_values):
            for k, rb in enumerate(rb_values):
                for l, log_rf in enumerate(log_rf_values):
                    try:
                        result = root_scalar(
                            objective_function,
                            args=(c, fb, rb, nu, log_rf),
                            bracket=[-7, 1],
                            method="brentq"
                        )
                        if not result.converged:
                            log_mhi_values[i, j, k, l] = np.nan
                        else:
                            log_ri = result.root
                            ri = 10**log_ri
                            # Compute the second output from do_contra (mhi)
                            _, mhi = do_contra(
                                np.array([ri]), np.array([c]), np.array([fb]), np.array([rb]),
                                A=1.6, w=0.8
                            )
                            log_mhi_values[i, j, k, l] = np.log10(mhi[0])
                    except ValueError as e:
                        print(f"ValueError encountered at log_c={np.log10(c)}, log_fb={np.log10(fb)}, log_rb={np.log10(rb)}, log_rf={log_rf}: {e}")
                        sys.exit(1)      

    print(f'Process {rank}: Training for nu = {nu} started')
    
    # Create the interpolator using the computed 4D grid
    interpolator = RegularGridInterpolator(
        (log_c_values, log_fb_values, log_rb_values, log_rf_values), log_mhi_values,
        bounds_error=False, fill_value=None
    )
    interpolators[nu] = interpolator
    print(f'Process {rank}: Training for nu = {nu} finished')

# Gather interpolators
all_interpolators = comm.gather(interpolators, root=0)

if rank == 0:
    final_interpolators = {}
    for partial_interpolators in all_interpolators:
        final_interpolators.update(partial_interpolators)
   
    output_dir = "/Users/fedorboreiko/Documents/Oxford/Personal_codes/Codebase/contra_emulators"
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = os.path.join(output_dir, "contra_interpolators.pkl")
    with open(output_filename, "wb") as f:
        pickle.dump(final_interpolators, f)
    
    print("Interpolators saved to", output_filename)
