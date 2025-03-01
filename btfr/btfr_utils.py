import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt

G = 4.3009e-6  # kpc (km/s)^2 / M_sun
H = 0.07 # km/s/kpc


def nfw_circular_velocity(r, M_vir, R_vir, r_s):
    """
    Inputs: r in kpc
            M_vir in M_sun
            R_vir in kpc
            r_s in kpc
    Outputs: V_circ in km/s
    """
    # Initialize the circular velocity array with zeros
    vc = np.zeros((M_vir.shape[0], r.shape[0]))

    # Compute circular velocity
    x = np.divide(r, R_vir[:, np.newaxis])
    c = np.divide(R_vir[:, np.newaxis], r_s[:, np.newaxis])
    vc = np.sqrt((G * M_vir[:, np.newaxis] / r) * 
                                (np.log(1 + c * x) - c * x / (1 + c * x)) / 
                                (np.log(1 + c) - c / (1 + c)))
    return vc


def nfw_circular_velocity_from_mhi(r, mhi, fb, Mvir):
    """
    Inputs: r in kpc
            mhi in unitless
            fb in unitless
            Mvir in M_sun
    Outputs: V_circ in km/s
    """
    M_enclosed = mhi / (1 - fb)[:, np.newaxis] * Mvir[:, np.newaxis]
    v_c = np.sqrt(G * M_enclosed / r)
    return v_c


def nfw_circular_velocity_contra(rads, Eff_rad, Rvir, rs, Mvir, M_baryon, contra_emulator):
    """
    Compute the contracted DM circular velocity.
    Uses the contra_emulator (RegularGrid or jax-based interpolator) to emulate log(mhi) values.
    
    Inputs:
      rads     : array of radii in kpc
      Eff_rad  : effective radius in kpc
      Rvir     : virial radius in kpc
      rs       : scale radius in kpc
      Mvir     : halo virial mass in M_sun
      M_baryon : baryonic mass in M_sun
      contra_emulator: callable that takes points of shape (n,4) and returns interpolated log(mhi)
    Outputs:
      vc       : array of DM circular velocities in km/s
    """
    vc = np.zeros((Mvir.shape[0], rads.shape[0]))
    rb = Eff_rad / 1.67835
    c = Rvir / rs
    fb = M_baryon / (Mvir + M_baryon)
    rb_uless = rb / Rvir
    rads_uless = rads[np.newaxis, :] / Rvir[:, np.newaxis]
    
    logc_extended = np.repeat(np.log10(c)[:, np.newaxis], rads_uless.shape[1], axis=1)
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
    logmhi = np.array(contra_emulator(points))
    logmhi = logmhi.reshape(rads_uless.shape)
    logmhi[logmhi == 0] = -np.inf  # Treat points outside bounds as -inf
    
    mhi = 10**logmhi
    vc = nfw_circular_velocity_from_mhi(rads, mhi, fb, Mvir) # returns an array of DM circular velocities in km/s, in cases of mhi == 0 returns array of zeros
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
        avg_likelihood = np.mean(likelihoods)
        # Update the log likelihood
        log_likelihood += np.log(avg_likelihood)
        log_likelihoods[i] = np.log(avg_likelihood)
    return log_likelihood, log_likelihoods


def update_progress(comm, rank, size, local_progress, total_work):
    all_progress = np.zeros(size, dtype=int)
    all_progress[rank] = local_progress
    comm.Allreduce(MPI.IN_PLACE, all_progress, op=MPI.SUM)
    return int(np.sum(all_progress) / total_work)


'''def get_Rvir(Mvir):
    """
    Inputs: Mvir in M_sun
    Outputs: Rvir in kpc
    """
    rho_crit = 3 * H**2 / (8 * np.pi * G)  # M_sun / kpc^3
    Rvir = np.power(3 * Mvir / (4 * 102.34925 * np.pi * rho_crit), 1/3) # kpc
    return Rvir'''