import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt

G = 4.3009e-6  # kpc (km/s)^2 / M_sun
H = 0.07 # km/s/kpc

def get_x_cutoff_fit(halos, x):
    """
    Fits a linear relation between log10(Mvir) and the x-th percentile of log10(vmax).
    
    Parameters:
    halos (numpy structured array): The input halos catalog.
    x (float): The fraction representing the upper percentile cutoff (e.g., x=0.1 for top 10%).
    
    Returns:
    tuple: (slope, intercept) of the best-fit line.
    """
    log_vmax = np.log10(halos['vmax'])
    log_Mvir = np.log10(halos['Mvir'])

    window_width = 0.1  # dex
    step_size = 0.01  # dex
    percentile = 100 * (1 - x)  # Convert fraction to percentile

    max_mvir = np.max(log_Mvir)
    min_mvir = np.min(log_Mvir)
    window_right = max_mvir
    window_left = window_right - window_width

    mvir_bins = []
    vmax_percentiles = []

    while window_left >= min_mvir:
        # Select halos within the window range
        mask = (log_Mvir >= window_left) & (log_Mvir <= window_right)
        if np.sum(mask) > 0:
            vmax_percentile = np.percentile(log_vmax[mask], percentile)
            mvir_bins.append((window_left + window_right) / 2)
            vmax_percentiles.append(vmax_percentile)

        # Move the window to the left
        window_right -= step_size
        window_left -= step_size

    mvir_bins = np.array(mvir_bins)
    vmax_percentiles = np.array(vmax_percentiles)

    valid_range_mask = (mvir_bins >= 10.2) & (mvir_bins <= 13.5)
    mvir_valid = mvir_bins[valid_range_mask]
    vmax_valid = vmax_percentiles[valid_range_mask]

    slope, intercept = np.polyfit(mvir_valid, vmax_valid, 1)

    return slope, intercept

def nfw_circular_velocity(r, M_vir, R_vir, r_s):
    """
    Inputs: r in kpc (1D array, length e.g. 15)
            M_vir in M_sun (1D array, length e.g. 1000)
            R_vir in kpc (1D array, length e.g. 1000)
            r_s in kpc (1D array, length e.g. 1000)
    Outputs: V_circ in km/s (2D array of shape (len(M_vir), len(r)))
"""
    x = np.divide(r, R_vir[:, np.newaxis])
    c = np.divide(R_vir[:, np.newaxis], r_s[:, np.newaxis])
    vc = np.sqrt((G * M_vir[:, np.newaxis] / r) * 
                            (np.log(1 + c * x) - c * x / (1 + c * x)) / 
                            (np.log(1 + c) - c / (1 + c)))
    return vc


def nfw_circular_velocity_from_mhi(r, mhi, fb, Mvir):
    """
    Inputs: 
        r in kpc (1D array, length e.g. 15)
        mhi in unitless (2D array of shape (len(Mvir), len(r)))
        fb in unitless (1D array, length e.g. 1000)
        Mvir in M_sun (1D array, length e.g. 1000)
    Outputs: 
        V_circ in km/s (2D array of shape (len(Mvir), len(r)))
    """
    if np.all(fb != 1):
        M_enclosed = mhi / (1 - fb[:, None]) * Mvir[:, None]
        v_c = np.sqrt(G * M_enclosed / r)
    else:
        # Initialize v_c with NaNs so that values corresponding to fb==1 remain NaN.
        v_c = np.full((len(Mvir), len(r)), np.nan)
        valid = fb != 1
        M_enclosed = mhi[valid] / (1 - fb[valid, None]) * Mvir[valid, None]
        v_c[valid] = np.sqrt(G * M_enclosed / r)
    return v_c


def nfw_circular_velocity_contra(rads, Eff_rad, Rvir, rs, Mvir, M_baryon, contra_emulator):
    """
    Compute the contracted DM circular velocity.
    Uses the contra_emulator (RegularGrid or jax-based interpolator) to emulate log(mhi) values.
    
    Inputs:
      rads in kpc (1D array, length e.g. 15)
      Eff_rad in kpc (a number)
      Rvir in kpc (1D array, length e.g. 1000)
      rs in kpc (1D array, length e.g. 1000)
      Mvir in M_sun (1D array, length e.g. 1000)
      M_baryon in M_sun (1D array, length e.g. 1000)
      contra_emulator: callable that takes points of shape (n,4) and returns interpolated log(mhi)
      
    Outputs:
      vc : array of DM circular velocities in km/s (2D array of shape (len(Mvir), len(rads)))
    """
    
    rb = Eff_rad / 1.67835
    c = Rvir / rs
    fb= M_baryon/ (Mvir + M_baryon)
    rb_uless = rb / Rvir
    rads_uless = rads[np.newaxis, :] / Rvir[:, np.newaxis]
    
    logc_extended  = np.repeat(np.log10(c)[:, np.newaxis], rads_uless.shape[1], axis=1)
    logfb_extended = np.repeat(np.log10(fb)[:, np.newaxis], rads_uless.shape[1], axis=1)
    logrb_extended = np.repeat(np.log10(rb_uless)[:, np.newaxis], rads_uless.shape[1], axis=1)
    
    # Build the points array: each row is [log(c), log(fb), log(rb), log(r)]
    points = np.vstack((
        logc_extended.ravel(),
        logfb_extended.ravel(),
        logrb_extended.ravel(),
        np.log10(rads_uless).ravel()
    )).T
    
    # Call the emulator (our jax-based interpolator)
    logmhi = np.array(contra_emulator(points)) # out-of-bounds points return nan
    logmhi = logmhi.reshape(rads_uless.shape)
    mhi = 10**logmhi
    vc = nfw_circular_velocity_from_mhi(rads, mhi, fb, Mvir)
    return vc


def get_loglike(V_mocks, V_obs, V_obs_err):
    """
    Computes the log likelihood, adapted for V_mocks as a list of arrays.
    
    Parameters:
        V_mocks (list of np.ndarray): A list where each element is an array of mock velocities for a galaxy.
        V_obs (np.ndarray): Observed velocities (log-transformed).
        V_obs_err (np.ndarray): Observational errors (log-transformed).

    Returns:
        float: The computed log likelihood.
    """
    # Initialize log likelihood
    log_likelihood = 0.0
    log_likelihoods = np.zeros(len(V_obs))
    # Loop over each observed velocity and corresponding mocks
    for i in range(len(V_obs)):
        # Extract the mock velocities for the i-th observed velocity
        V_max_samples = V_mocks[i]  # This is now an individual array
        # Calculate the likelihood for each sample
        likelihoods = np.exp(-0.5 * ((V_obs[i] - V_max_samples) / V_obs_err[i])**2) / (np.sqrt(2 * np.pi) * V_obs_err[i])
        # Average likelihood across all samples (approximating the integral)
        avg_likelihood = np.nanmean(likelihoods)
        # Update the log likelihood
        log_likelihood += np.log(avg_likelihood)
        log_likelihoods[i] = np.log(avg_likelihood)
    return log_likelihood, log_likelihoods


def update_progress(comm, rank, size, local_progress, total_work):
    all_progress = np.zeros(size, dtype=int)
    all_progress[rank] = local_progress
    comm.Allreduce(MPI.IN_PLACE, all_progress, op=MPI.SUM)
    return int(np.sum(all_progress) / total_work)


def get_Rvir(Mvir):
    """
    Inputs: Mvir in M_sun
    Outputs: Rvir in kpc
    """
    rho_crit = 3 * H**2 / (8 * np.pi * G)  # M_sun / kpc^3
    Rvir = np.power(3 * Mvir / (4 * 102.34925 * np.pi * rho_crit), 1/3) # kpc
    return Rvir