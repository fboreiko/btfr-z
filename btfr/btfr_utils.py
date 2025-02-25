import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt

G = 4.3009e-6  # kpc (km/s)^2 / M_sun
H = 0.07 # km/s/kpc

def halo_selection(matched_halos, x):
    """
    For each row of a 2D structured array, remove (set to zero) the halos
    with the highest 'vmax' values corresponding to a fraction x of the total
    number of halos in that row.
    
    Parameters
    ----------
    matched_halos : np.ndarray
        A 2D structured array (e.g. shape (153, 1000)) with at least a field 'vmax'.
    x : float
        A fraction (between 0 and 1) representing the fraction of halos to remove 
        in each row, starting from the highest 'vmax' values.
        
    Returns
    -------
    new_halos : np.ndarray
        A structured array of the same shape as matched_halos, but with halos 
        (i.e., array entries) removed (set to zeros) as indicated by the elimination mask.
    eliminate_mask : np.ndarray (bool)
        A boolean mask of the same shape as matched_halos, where True indicates 
        that the halo was removed.
    """
    
    if matched_halos.ndim == 1:
        ncols = matched_halos.shape[0]
        eliminate_mask = np.zeros(ncols, dtype=bool)

        n_remove = int(np.floor(x * ncols))
        
        if n_remove > 0:
            sorted_indices = np.argsort(matched_halos['vmax'])[::-1]
            remove_indices = sorted_indices[:n_remove]
            eliminate_mask[remove_indices] = True

    else:
        nrows, ncols = matched_halos.shape
        eliminate_mask = np.zeros((nrows, ncols), dtype=bool)

        for i in range(nrows):
            n_remove = int(np.floor(x * ncols))
            
            if n_remove > 0:
                row = matched_halos[i]
                sorted_indices = np.argsort(row['vmax'])[::-1]
                remove_indices = sorted_indices[:n_remove]
                eliminate_mask[i, remove_indices] = True
    
    new_halos = matched_halos.copy()
    new_halos[eliminate_mask] = np.zeros(1, dtype=matched_halos.dtype)
    return new_halos, eliminate_mask


def get_Rvir(Mvir):

    """
    Inputs: Mvir in M_sun
    Outputs: Rvir in kpc
    """

    rho_crit = 3 * H**2 / (8 * np.pi * G)  # M_sun / kpc^3
    Rvir = np.power(3 * Mvir / (4 * 102.34925 * np.pi * rho_crit), 1/3) # kpc

    return Rvir

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

    # Identify non-zero elements
    non_zero_mask = (R_vir != 0) & (r_s != 0) & (M_vir != 0)

    # Compute circular velocity only for non-zero elements
    if np.any(non_zero_mask):
        x = np.divide(r, R_vir[non_zero_mask][:, np.newaxis])
        c = np.divide(R_vir[non_zero_mask][:, np.newaxis], r_s[non_zero_mask][:, np.newaxis])
        vc[non_zero_mask] = np.sqrt((G * M_vir[non_zero_mask][:, np.newaxis] / r) * 
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
    Inputs: rads in kpc
            Eff_rads in kpc
            Rvir in kpc
            rs in kpc
            Mvir in M_sun
            M_baryon in M_sun
            contra_emulator: emulator object
    Outputs: V_circ in km/s
    """

    # Initialize the circular velocity array with zeros
    vc = np.zeros((Mvir.shape[0], rads.shape[0]))

    # Identify non-zero elements
    non_zero_mask = (Rvir != 0) & (rs != 0) & (Mvir != 0) & (M_baryon != 0)

    # Compute circular velocity only for non-zero elements
    if np.any(non_zero_mask):
        rb = Eff_rad / 1.67835
        c = Rvir[non_zero_mask] / rs[non_zero_mask]
        fb = M_baryon[non_zero_mask] / (Mvir[non_zero_mask] + M_baryon[non_zero_mask])
        rb_uless = rb / Rvir[non_zero_mask]
        rads_uless = rads[np.newaxis, :] / Rvir[non_zero_mask][:, np.newaxis]

        # extend c, fb, rb, ri to match the shape of ri
        logc_extended = np.repeat(np.log10(c)[:, np.newaxis], rads_uless.shape[1], axis=1)
        logfb_extended = np.repeat(np.log10(fb)[:, np.newaxis], rads_uless.shape[1], axis=1)
        logrb_extended = np.repeat(np.log10(rb_uless)[:, np.newaxis], rads_uless.shape[1], axis=1)

        # flatten the grids and data
        points = np.vstack((logc_extended.ravel(), logfb_extended.ravel(), logrb_extended.ravel(), np.log10(rads_uless).ravel())).T

        # emulate the data
        logmhi = contra_emulator(points)
        logmhi = logmhi.reshape(rads_uless.shape)
        logmhi[logmhi == 0] = -np.inf # handle the case where mock datapoint is outside the emulator bounds

        mhi = 10**logmhi # mhi == 0 for points outside the emulator bounds

        vc[non_zero_mask] = nfw_circular_velocity_from_mhi(rads, mhi, fb, Mvir[non_zero_mask]) # returns an array of DM circular velocities in km/s, in cases of mhi == 0 returns array of zeros

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