import numpy as np
from mpi4py import MPI

def get_loglike(V_sims, V_obs, V_obs_err):
    """
    Computes the Bayesian log likelihood, adapted for V_mocks as a list of arrays.

    Parameters:
        V_sims (list of np.ndarray): A list where each element is an array of simulated circular velocities for a galaxy.
        V_obs (np.ndarray): Observed velocities (log-transformed).
        V_obs_err (np.ndarray): Observational errors (log-transformed).

    Returns:
        tuple: 
            - float: The computed total log likelihood.
            - np.ndarray: Log likelihoods for individual observations.
    """
    # Initialize log likelihood
    log_likelihood = 0.0
    log_likelihoods = np.zeros(len(V_obs))
    # Loop over each observed velocity and corresponding mocks
    for i in range(len(V_obs)):
        # Extract the mock velocities for the i-th observed velocity
        V_sim_samples = V_sims[i]  # This is now an individual array
        # Calculate the likelihood for each sample
        likelihoods = np.exp(-0.5 * ((V_obs[i] - V_sim_samples) / V_obs_err[i])**2) / (np.sqrt(2 * np.pi) * V_obs_err[i])
        # Average likelihood across all samples (approximating the integral)
        avg_likelihood = np.nanmean(likelihoods)
        # Update the log likelihood
        log_likelihood += np.log(avg_likelihood)
        log_likelihoods[i] = np.log(avg_likelihood)
    return log_likelihood, log_likelihoods


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