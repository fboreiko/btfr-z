# mpiexec -n 4 python training_data_getter.py

"""
Simulates training data [Mstar, Mvir, Concentration] for the emulator. 
Units are Mstar: log10(Msun/h), Mvir: log10(Msun/h), Concentration: unitless.
"""

import numpy as np
from mpi4py import MPI
from btfr.massfuncs import get_GSMF_ELPETRO
from BAM import AbundanceMatch, proxies
from btfr.btfr_utils import update_progress
from tqdm import tqdm

N_AM_REALS = 1000

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print('Number of processes:', size)

def downsample(data):

    # Define the bin edges (adjust these as necessary based on your data)
    bin_edges = np.linspace(min(data[0, :]), max(data[0, :]), num=50)

    # Digitize the stellar masses into bins
    stellar_masses = data[0, :]
    bins = np.digitize(stellar_masses, bin_edges)

    # Determine the number of entries in the bin corresponding to a stellar mass of 11.5
    mass_target = 11.8
    target_bin_index = np.digitize(mass_target, bin_edges)

    # Find the number of entries in the target bin
    n_target_bin = np.sum(bins == target_bin_index)

    # Initialize a list to hold the downsampled indices
    downsampled_indices = []

    # For each bin, randomly select the same number of entries as in the target bin
    for bin_index in range(1, target_bin_index + 1):
        # Find indices of entries in this bin
        bin_indices = np.where(bins == bin_index)[0]
        
        if len(bin_indices) > 0:
            # If there are more entries than the target, downsample
            if len(bin_indices) > n_target_bin:
                selected_indices = np.random.choice(bin_indices, n_target_bin, replace=False)
            else:
                selected_indices = bin_indices
            
            downsampled_indices.append(selected_indices)

    # Flatten the list of downsampled indices
    downsampled_indices = np.concatenate(downsampled_indices)

    # Downsample the training_data using these indices
    data_downsampled = data[:, downsampled_indices]

    return data_downsampled

if __name__ == "__main__":

    if rank == 0:

        # Load the N.Adams SMF data
        log_stellar_masses, SMF_data, _ = get_GSMF_ELPETRO(plotting=False)

        # Load the Uchuu halos 
        halos = np.load("/Users/fedorboreiko/Documents/Oxford/Personal_codes/Codebase/halos_z_0p00.npy")

        proxy = proxies["mvir_proxy"]()

        abundance_match = AbundanceMatch(log_stellar_masses[10:], SMF_data[10:], halo_proxy=proxy, ext_range=(3.0, 12.0),
                                                boxsize=140, faint_end_first=True, scatter_mult=1, faint_end_slope=-0.42)

        theta = {"alpha": 1.2, "scatter": 0.2}  # Will be tuned?
        deconv = abundance_match.deconvoluted_catalogs(theta, halos)

    else:

        abundance_match = None
        deconv = None
        halos = None

    abundance_match = comm.bcast(abundance_match, root=0)
    deconv = comm.bcast(deconv, root=0)
    halos = comm.bcast(halos, root=0)

    realisations_per_process = np.array_split(np.arange(N_AM_REALS), size)
    local_realisations = realisations_per_process[rank]

    local_training_data = np.empty((3, len(local_realisations), 336))

    total_work = len(local_realisations) * size

    if rank == 0:
        pbar = tqdm(total=100, desc="Progress", position=0, leave=True)

    for i, realization in enumerate(local_realisations):

        training_data_iteration = np.empty((3, len(halos)))

        mask, catalog_sc = abundance_match.add_scatter(deconv, cut_range=(4, 12), return_catalog=True)

        sorted_indices = np.argsort(catalog_sc)

        training_data_iteration[0, :] = catalog_sc[sorted_indices]
        training_data_iteration[1, :] = np.log10(halos['Mvir'][sorted_indices])
        training_data_iteration[2, :] = np.divide(halos['Rvir'][sorted_indices],
                                                    halos['rs'][sorted_indices],
                                                    out=np.zeros_like(halos['Rvir'][sorted_indices]),
                                                    where=halos['rs'][sorted_indices] != 0)
        
        local_training_data[:, i, :] = downsample(training_data_iteration)
        
        # Update progress
        local_progress = i + 1
        global_progress = update_progress(comm, rank, size, local_progress, total_work)

        if rank == 0:
            pbar.n = global_progress
            pbar.refresh()

    if rank == 0:
        pbar.close()

    training_data = None

    if rank == 0:

        training_data = np.empty((3, N_AM_REALS * 336))

    comm.Gather(local_training_data, training_data, root=0)

    if rank == 0:

        training_data = training_data.reshape((3, N_AM_REALS * 336))

        sorted_indices = np.argsort(training_data[0, :])

        training_data = training_data[:, sorted_indices]

        np.save("training_data.npy", training_data)

        print(training_data.shape)