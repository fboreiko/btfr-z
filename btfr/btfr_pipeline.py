##### mpiexec -n 4 python btfr/btfr_contra.py

import sys
sys.path.append('/Users/fedorboreiko/Documents/Oxford/btfr_z')

from mpi4py import MPI
import numpy as np
import pandas as pd
from btfr.massfuncs import get_GSMF_ELPETRO
from BAM import AbundanceMatch, proxies
from btfr.btfr_plotting import btfr_plot, explore_hist
from btfr.btfr_utils import halo_selection, nfw_circular_velocity_contra, nfw_circular_velocity, get_loglike
from matplotlib import rcParams
from tqdm import tqdm
import pickle

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern']
rcParams['text.usetex'] = True

N_AM_REALS = 10
N_STELLAR_REALS = 1000

NU = -1.0
ALPHA = -10.0
SCATTER = 0.01
X = 0.2
HALO_SELECTION = True

M2L_DISK_MEAN = 0.5
M2L_DISK_ERROR = 0.2

M2L_BULGE_MEAN = 0.7
M2L_BULGE_ERROR = 0.2

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print('Number of processes:', size)

# Wrapper class around original interpolator, handles out-of-bounds inputs by returning zeros in these cases
class SafeInterpolator:
    def __init__(self, interpolator):
        self.interpolator = interpolator
        # These bounds should be the same as the ones used to train the interpolator, adress the contra_emulator_trainer.py
        self.bounds = np.array([[0.01, 3],    #c range. Full range is [0, 3.9]
                                [-3.3, -0.3], #fb range. Full range is [-3.6, -0.03]
                                [-2.9, -1.3], #rb range. Full range is [-3, -1]
                                [-4.8, 0.3]]) #rf range. Full range is [-4.8, 0.3]
    def __call__(self, points):
        results = np.zeros(points.shape[0])
        valid_mask = np.all(
            (points >= self.bounds[:, 0]) & (points <= self.bounds[:, 1]), axis=1
        )
        valid_points = points[valid_mask]

        if valid_points.size > 0:
            valid_results = self.interpolator(valid_points)
            results[valid_mask] = valid_results

        return results

def compute_AM_realization(abundance_match, deconv, sparc_galaxy_list, mass_models, halos, 
                           L_bulges, L_36_means, L_36_errors, Eff_rads, MH1_means, MH1_errors, 
                           dists, dists_err, emulator):
    
    # Add scatter to the deconvoluted catalog, and return the catalog of stellar masses matched to halos from the halo catalog
    mask, catalog_sc = abundance_match.add_scatter(deconv, cut_range=(3, 12), return_catalog=True)

    # Sort the catalog and extract the sorted indices
    sorted_indices = np.argsort(catalog_sc)
    catalog_sc_sorted = catalog_sc[sorted_indices]

    # Generate N_STELLAR_REALS of mock stellar masses
    dist_samples = np.random.normal(loc=dists[:, np.newaxis], scale=dists_err[:, np.newaxis], size=(len(galaxy_sample), N_STELLAR_REALS)) # Mpc
    L_36_samples = np.random.normal(loc=L_36_means[:, np.newaxis], scale=L_36_errors[:, np.newaxis], size=(len(sparc_galaxy_list), N_STELLAR_REALS)) # 1e9 L_sun

    log_M2L_disk_samples = np.random.normal(loc=np.log10(M2L_DISK_MEAN), scale=M2L_DISK_ERROR, size=(len(sparc_galaxy_list), N_STELLAR_REALS)) 
    log_M2L_bulge_samples = np.random.normal(loc=np.log10(M2L_BULGE_MEAN), scale=M2L_BULGE_ERROR, size=(len(sparc_galaxy_list), N_STELLAR_REALS)) 

    M2L_disk_samples = 10**log_M2L_disk_samples # M_sun / L_sun
    M2L_bulge_samples = 10**log_M2L_bulge_samples # M_sun / L_sun

    MH1_samples = np.random.normal(loc=MH1_means[:, np.newaxis], scale=MH1_errors[:, np.newaxis], size=(len(sparc_galaxy_list), N_STELLAR_REALS)) * 1e9 # M_sun

    M_star_samples = np.abs((L_36_samples - L_bulges[:, np.newaxis]) * M2L_disk_samples + L_bulges[:, np.newaxis] * M2L_bulge_samples) * (dists[:, np.newaxis] / dist_samples)**2 * 1e9 * 0.7 # M_sun / h
    log_M_star_samples = np.log10(M_star_samples)

    # Match the stellar masses to halos
    indices_sorted = np.searchsorted(catalog_sc_sorted, log_M_star_samples.flatten())
    indices_sorted = np.clip(indices_sorted, 0, len(catalog_sc_sorted) - 1)
    indices = sorted_indices[indices_sorted].reshape(len(sparc_galaxy_list), N_STELLAR_REALS)

    matched_halos = halos[indices]

    # Halo selection
    if HALO_SELECTION:
        matched_halos, elliminate_masks = halo_selection(matched_halos, X)

        M_star_samples[elliminate_masks] = 0
        MH1_samples[elliminate_masks] = 0
        M2L_disk_samples[elliminate_masks] = 0
        M2L_bulge_samples[elliminate_masks] = 0

    # Simulate the rotation curves
    vels = np.empty((len(sparc_galaxy_list), N_STELLAR_REALS))
    masses = np.empty((len(sparc_galaxy_list), N_STELLAR_REALS))
    dm_vels = np.empty((len(sparc_galaxy_list), N_STELLAR_REALS)) # these three are not strictly necessary, but are kept for visualisation purposes
    bar_vels = np.empty((len(sparc_galaxy_list), N_STELLAR_REALS))

    for j, galaxy in enumerate(sparc_galaxy_list):
        
        # Extract the galaxy's data
        selected_rows = mass_models[mass_models['ID'] == galaxy]
        rads = np.asarray(selected_rows['R']) # kpc
        V_gas = np.asarray(selected_rows['Vgas'])
        V_disk = np.asarray(selected_rows['Vdisk'])
        V_bul = np.asarray(selected_rows['Vbul'])

        M_baryon = M_star_samples[j] / 0.7 + 1.33 * MH1_samples[j]

        Mvir = matched_halos[j]['Mvir'] / 0.7
        Rvir = matched_halos[j]['Rvir'] / 0.7
        rs = matched_halos[j]['rs'] / 0.7

        if NU == 0.0:
            # the case of no halo contraction/expansion, use basic NFW profile
            V_dm = nfw_circular_velocity(rads, Mvir, Rvir, rs)
        
        else:
            # the case of halo contraction/expansion, use contra emulator
            V_dm = nfw_circular_velocity_contra(rads, Eff_rads[j], Rvir, rs, Mvir, M_baryon, emulator)
 
        # Calculate the maximum circular velocity of the galaxy's total rotation curve
        V_max = np.sqrt(np.max(
                                V_gas * np.abs(V_gas) +
                                M2L_disk_samples[j][:, np.newaxis] * V_disk * np.abs(V_disk) +
                                M2L_bulge_samples[j][:, np.newaxis] * V_bul * np.abs(V_bul) +
                                V_dm * np.abs(V_dm),
                                axis=1
                            )) # km/s
        
        # handle cases where V_dm is an array of zeros
        V_dm_max = np.max(V_dm, axis=1)
        V_max[V_dm_max == 0] = 0

        vels[j, :] = V_max
        masses[j, :] = M_baryon

        # To visualise baryonic and dark matter contributions to the rotation curve:
        V_bar_max = np.sqrt(np.max(
                                    V_gas * np.abs(V_gas) +
                                    M2L_disk_samples[j][:, np.newaxis] * V_disk * np.abs(V_disk) +
                                    M2L_bulge_samples[j][:, np.newaxis] * V_bul * np.abs(V_bul),
                                    axis=1
                                )) # km/s
        
        V_bar_max[V_dm_max == 0] = 0

        dm_vels[j, :] = V_dm_max
        bar_vels[j, :] = V_bar_max

    return vels, masses, dm_vels, bar_vels, 

if rank == 0:

    # Pre-load data
    bulge_lumins = pd.read_csv('Tabular_data/Bulge_lum_table.csv')
    mass_models = pd.read_csv('Tabular_data/Mass_models_table.csv')
    galaxy_sample = pd.read_csv('Tabular_data/Gal_sample_table.csv')
    sparc_btfr = pd.read_csv('Tabular_data/sparc_btfr.csv')

    sparc_galaxy_list = sparc_btfr['Name']

    bulge_lumins = bulge_lumins[bulge_lumins['Galaxy'].isin(sparc_galaxy_list)]
    mass_models = mass_models[mass_models['ID'].isin(sparc_galaxy_list)]
    galaxy_sample = galaxy_sample[galaxy_sample['Galaxy'].isin(sparc_galaxy_list)]
    sparc_btfr = sparc_btfr[sparc_btfr['Name'].isin(sparc_galaxy_list)]

    bulge_lumins_dict = bulge_lumins.set_index('Galaxy')['Lbul'].to_dict()
    galaxy_data_dict = galaxy_sample.set_index('Galaxy').to_dict('index')

    L_36_means = np.array([galaxy_data_dict[galaxy]['Total Luminosity at [3.6]'] for galaxy in sparc_galaxy_list]) # 1e9 L_sun
    L_36_errors = np.array([galaxy_data_dict[galaxy]['Luminosity Error'] for galaxy in sparc_galaxy_list]) # 1e9 L_sun
    Eff_radii = np.array([galaxy_data_dict[galaxy]['Effective Radius at [3.6]'] for galaxy in sparc_galaxy_list]) # kpc
    MH1_means = np.array([galaxy_data_dict[galaxy]['Total HI mass'] for galaxy in sparc_galaxy_list]) # 1e9 M_sun
    MH1_errors = MH1_means * 0.1 # 1e9 M_sun
    dists = np.array([galaxy_data_dict[galaxy]['Distance'] for galaxy in sparc_galaxy_list]) # Mpc
    dists_err = np.array([galaxy_data_dict[galaxy]['Distance Error'] for galaxy in sparc_galaxy_list]) # Mpc
    L_bulges = np.array([bulge_lumins_dict[galaxy] for galaxy in sparc_galaxy_list]) # 1e9 L_sun

    # Load GSMF data and halo catalog
    log_stellar_masses, SMF_data, _ = get_GSMF_ELPETRO(plotting=False)
    halos = np.load("/Users/fedorboreiko/Documents/Oxford/Personal_codes/Codebase/halos_z_0p00.npy")

    # Create abundance matching (AM) object
    proxy = proxies["mvir_proxy"]()
    abundance_match = AbundanceMatch(log_stellar_masses[10:], SMF_data[10:], halo_proxy=proxy, ext_range=(3.0, 12.0),
                                     boxsize=140, faint_end_first=True, scatter_mult=1, faint_end_slope=-0.42)

    theta = {"alpha": ALPHA, "scatter": SCATTER}  # AM model parameters
    # Deconvolute the AM catalog
    deconv = abundance_match.deconvoluted_catalogs(theta, halos)

    # Load contra interpolator
    with open("/Users/fedorboreiko/Documents/Oxford/Personal_codes/Codebase/contra_emulators/contra_interpolators.pkl", "rb") as f:
        interpolators = pickle.load(f)

    if NU in interpolators:
        contra_interpolator = SafeInterpolator(interpolators[NU])
    else:
        print(f"Interpolator for nu={NU} not found.")
        sys.exit(1)

else:

    sparc_galaxy_list = None
    mass_models = None
    L_36_means = None
    L_36_errors = None
    Eff_radii = None
    MH1_means = None
    MH1_errors = None
    dists = None
    dists_err = None
    L_bulges = None
    abundance_match = None
    deconv = None
    halos = None
    contra_interpolator = None

# Broadcast data to all processes
sparc_galaxy_list = comm.bcast(sparc_galaxy_list, root=0)
mass_models = comm.bcast(mass_models, root=0)
L_36_means = comm.bcast(L_36_means, root=0)
L_36_errors = comm.bcast(L_36_errors, root=0)
Eff_radii = comm.bcast(Eff_radii, root=0)
MH1_means = comm.bcast(MH1_means, root=0)
MH1_errors = comm.bcast(MH1_errors, root=0)
dists = comm.bcast(dists, root=0)
dists_err = comm.bcast(dists_err, root=0)
L_bulges = comm.bcast(L_bulges, root=0)
abundance_match = comm.bcast(abundance_match, root=0)
deconv = comm.bcast(deconv, root=0)
halos = comm.bcast(halos, root=0)
contra_interpolator = comm.bcast(contra_interpolator, root=0)

# Split the realizations across processes
realizations_per_process = np.array_split(np.arange(N_AM_REALS), size)
local_realizations = realizations_per_process[rank]

# Compute local results
local_vels = np.empty((len(local_realizations), len(sparc_galaxy_list), N_STELLAR_REALS))
local_masses = np.empty((len(local_realizations), len(sparc_galaxy_list), N_STELLAR_REALS))
local_dm_vels = np.empty((len(local_realizations), len(sparc_galaxy_list), N_STELLAR_REALS))
local_bar_vels = np.empty((len(local_realizations), len(sparc_galaxy_list), N_STELLAR_REALS))

if rank == 0:
    pbar = tqdm(total=len(local_realizations), desc="Progress", position=0, leave=True)

for i, realization in enumerate(local_realizations):

    vels, masses, dm_vels, bar_vels = compute_AM_realization(abundance_match, deconv, sparc_galaxy_list, mass_models, halos,
                                       L_bulges, L_36_means, L_36_errors, Eff_radii, MH1_means, MH1_errors, dists, dists_err,
                                       contra_interpolator)
    local_vels[i, :, :] = vels
    local_masses[i, :, :] = masses
    local_dm_vels[i, :, :] = dm_vels
    local_bar_vels[i, :, :] = bar_vels

    if rank == 0:
        pbar.update(1)

if rank == 0:
    pbar.close()

# Gather results from all processes
global_vels = None
global_masses = None
global_dm_vels = None
global_bar_vels = None

if rank == 0:

    global_vels = np.empty((N_AM_REALS, len(sparc_galaxy_list), N_STELLAR_REALS))
    global_masses = np.empty((N_AM_REALS, len(sparc_galaxy_list), N_STELLAR_REALS))
    global_dm_vels = np.empty((N_AM_REALS, len(sparc_galaxy_list), N_STELLAR_REALS))
    global_bar_vels = np.empty((N_AM_REALS, len(sparc_galaxy_list), N_STELLAR_REALS))

comm.Gather(local_vels, global_vels, root=0)
comm.Gather(local_masses, global_masses, root=0)
comm.Gather(local_dm_vels, global_dm_vels, root=0)
comm.Gather(local_bar_vels, global_bar_vels, root=0)

# Calculate log likelihood
if rank == 0:

    print('\nAbundance Matching Pipeline finished!')
    print('Calculating log likelihood...')

    global_vels_reshaped = np.transpose(global_vels, (1, 0, 2))
    global_masses_reshaped = np.transpose(global_masses, (1, 0, 2))
    global_dm_vels_reshaped = np.transpose(global_dm_vels, (1, 0, 2))
    global_bar_vels_reshaped = np.transpose(global_bar_vels, (1, 0, 2))

    def filter_zeros(arr):
        #Filters out zeros from a 2D array row-wise, returning a list of arrays.
        return [row[row != 0] for row in arr]

    V_mocks_unlogged = filter_zeros(global_vels_reshaped.reshape(len(galaxy_sample), -1))
    V_mocks = [np.log10(row) for row in V_mocks_unlogged]

    V_dm_mocks_unlogged = filter_zeros(global_dm_vels_reshaped.reshape(len(galaxy_sample), -1))
    V_bar_mocks_unlogged = filter_zeros(global_bar_vels_reshaped.reshape(len(galaxy_sample), -1))
    V_dm_mocks = [np.log10(row) for row in V_dm_mocks_unlogged]
    V_bar_mocks = [np.log10(row) for row in V_bar_mocks_unlogged]

    # Print the percentages
    original_length = N_AM_REALS * N_STELLAR_REALS
    filtered_lengths = [len(row) for row in V_mocks_unlogged]
    print("Percentage of original array length retained after filtering out-of-bound points (per galaxy):")
    print([round((length / original_length) * 100, 2) for length in filtered_lengths])

    V_mock = np.array([np.mean(row) for row in V_mocks])
    V_mock_err = np.array([np.std(row) for row in V_mocks])

    # Observed data
    V_obs_unlogged = np.array(sparc_btfr['Vmax'])
    V_obs_err_unlogged = np.array(sparc_btfr['e_Vmax'])

    V_obs = np.log10(V_obs_unlogged)
    V_obs_err = V_obs_err_unlogged / (V_obs_unlogged * np.log(10))

    log_likelihood, _ = get_loglike(V_mocks, V_obs, V_obs_err)

    print(f'Log likelihood: {log_likelihood}')

    # Plot the BTFR
    M_mocks_unlogged = filter_zeros(global_masses_reshaped.reshape(len(galaxy_sample), -1))
    M_mocks = [np.log10(row) for row in M_mocks_unlogged]

    M_mock = np.array([np.mean(row) for row in M_mocks])
    M_mock_err = np.array([np.std(row) for row in M_mocks])

    M_obs = np.array(sparc_btfr['log(Mb)'])
    M_obs_err = np.array(sparc_btfr['e_log(Mb)'])

    # Plot individual galaxy rotation curve statistics
    indiv_hists = False
    if indiv_hists:
        for galnum in range(len(sparc_galaxy_list)):

            explore_hist(V_mocks[galnum], V_dm_mocks[galnum], V_bar_mocks[galnum],
                        V_mock[galnum], V_mock_err[galnum],
                        V_obs[galnum], V_obs_err[galnum],
                        M_mock[galnum], 10, 10, 
                        f'/Users/fedorboreiko/Documents/Oxford/btfr_z/plots/RTstats_alpha_{ALPHA}_scatter_{SCATTER}_nu_{NU}/RTstats_galaxy_{galnum}_alpha_{ALPHA}_scatter_{SCATTER}_nu_{NU}.png')

    btfr_plot(V_mock, M_mock, V_mock_err, M_mock_err, V_obs, M_obs, V_obs_err, M_obs_err, plotname=f'/Users/fedorboreiko/Documents/Oxford/btfr_z/plots/btfr_alpha_{ALPHA}_scatter_{SCATTER}_nu_{NU}.png')
