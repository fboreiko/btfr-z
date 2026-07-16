import numpy as np
from mpi4py import MPI

def get_loglike(V_sims, V_obs, V_obs_err):
    """
    Computes the Bayesian log likelihood, adapted for V_mocks as a 2D array.

    Parameters:
        V_sims (np.ndarray): Simulated circular velocities with shape (num_galaxies, num_samples).
        V_obs (np.ndarray): Observed velocities (log-transformed).
        V_obs_err (np.ndarray): Observational errors (log-transformed).

    Returns:
        tuple: 
            - float: The computed total log likelihood.
            - np.ndarray: Log likelihoods for individual observations.
    """
    # Vectorized likelihood calculation
    # Broadcast V_obs and V_obs_err to match the shape of V_sims
    V_obs_expanded = V_obs[:, np.newaxis]  # Shape: (num_galaxies, 1)
    V_obs_err_expanded = V_obs_err[:, np.newaxis]  # Shape: (num_galaxies, 1)
    
    # Calculate likelihoods for all samples at once
    likelihoods = np.exp(-0.5 * ((V_obs_expanded - V_sims) / V_obs_err_expanded)**2) / (
        np.sqrt(2 * np.pi) * V_obs_err_expanded
    )
    
    # Average likelihood across all samples for each galaxy
    avg_likelihoods = np.nanmean(likelihoods, axis=1)
    
    # Compute log likelihoods
    log_likelihoods = np.log(avg_likelihoods)
    
    # Total log likelihood is the sum of individual log likelihoods
    log_likelihood = np.sum(log_likelihoods)
    
    return log_likelihood, log_likelihoods


def get_loglike_vect(V_mocks_mode, V_obs, V_obs_err):
    """
    Vectorized computation of the log likelihood for multiple observed velocities.

    Parameters:
        V_mocks_mode (np.ndarray): Flattened mock velocities (shape: [num_galaxies, num_samples]).
        V_obs (np.ndarray): Observed velocities (shape: [num_truths, num_galaxies]).
        V_obs_err (np.ndarray): Observational errors (shape: [num_truths, num_galaxies]).

    Returns:
        np.ndarray: Log likelihoods for each truth (shape: [num_truths]).
    """
    # Compute the likelihoods for all truths and galaxies in a vectorized manner
    likelihoods = np.exp(-0.5 * ((V_obs[:, :, None] - V_mocks_mode[None, :, :]) / V_obs_err[:, :, None])**2) / (
        np.sqrt(2 * np.pi) * V_obs_err[:, :, None]
    )
    
    # Average likelihoods over the mock samples (axis=2)
    avg_likelihoods = np.nanmean(likelihoods, axis=2)
    
    # Compute the log likelihoods for each truth (sum over galaxies, axis=1)
    log_likelihoods = np.sum(np.log(avg_likelihoods), axis=1)
    
    return log_likelihoods


def get_loglike_split(V_sims, V_obs, V_obs_err):
    """
    Computes the Bayesian log likelihood, adapted for V_sims from one process and
    V_mocks as a 2D array.

    Parameters:
        V_sims (np.ndarray): Simulated circular velocities with shape (num_galaxies, num_samples).
        V_obs (np.ndarray): Observed velocities (log-transformed).
        V_obs_err (np.ndarray): Observational errors (log-transformed).

    Returns:
        tuple: 
            - np.ndarray: Averaged likelihoods for individual galaxies 
                          over the samples from this process.
            - np.ndarray: Number of non-nan values per galaxy
    """
    # Vectorized likelihood calculation
    # Broadcast V_obs and V_obs_err to match the shape of V_sims
    V_obs_expanded = V_obs[:, np.newaxis]  # Shape: (num_galaxies, 1)
    V_obs_err_expanded = V_obs_err[:, np.newaxis]  # Shape: (num_galaxies, 1)
    
    # Calculate likelihoods for all samples at once
    likelihoods = np.exp(-0.5 * ((V_obs_expanded - V_sims) / V_obs_err_expanded)**2) / (
        np.sqrt(2 * np.pi) * V_obs_err_expanded
    )
    
    # Average likelihood across all samples for each galaxy
    avg_likelihoods = np.nanmean(likelihoods, axis=1)

    # Count non-nan values per galaxy
    non_nan_counts = np.sum(~np.isnan(likelihoods), axis=1)

    return avg_likelihoods, non_nan_counts


def update_progress(comm, rank, size, local_progress, total_work):
    """
    Updates the progress of MPI processes.

    Parameters:
        comm (MPI.Comm): MPI communicator.
        rank (int): Rank of the current process.
        size (int): Total number of processes.
        local_progress (int): Progress of the current process.
        total_work (int): Total amount of work.

    Returns:
        int: Percentage of total progress completed.
    """
    all_progress = np.zeros(size, dtype=int)
    all_progress[rank] = local_progress
    comm.Allreduce(MPI.IN_PLACE, all_progress, op=MPI.SUM)
    return int(np.sum(all_progress) / total_work)