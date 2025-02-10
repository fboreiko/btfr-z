##!/mnt/users/boreikof/fedirfiles/oxford_env/bin/python

import sys
sys.path.append('/Users/fedorboreiko/Documents/Oxford/btfr_z')

from mpi4py import MPI
import numpy as np
import pandas as pd
from massfuncs import get_GSMF_ELPETRO
from BAM import AbundanceMatch, proxies
from btfr.btfr_utils import nfw_circular_velocity, nfw_circular_velocity_contra, get_loglike
from tqdm import tqdm
import os
import pickle
import re

N_AM_REALS = 10
N_STELLAR_REALS = 100

M2L_DISK_MEAN = 0.5
M2L_DISK_ERROR = 0.2 # dex

M2L_BULGE_MEAN = 0.7
M2L_BULGE_ERROR = 0.2 # dex

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
    

def load_data():

    # Loading and preparing the data. Could be modified to load NSA stellar masses from backup_sparcx.csv, then cut
    # off the galaxies from SPARC that are not in the NSA data. Here, we load the data and cut it to match SPARC 
    # sample only. It was found that Sersic masses match SPARC masses within 0.2 dex, so we can continue using 
    # SPARC while adding more scatter to M2L ratios.

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
    
    return sparc_galaxy_list, bulge_lumins_dict, galaxy_data_dict, mass_models, sparc_btfr


def load_emulators(directory_path):

    # Load all emulators from the given directory

    emulators = {}
    
    # Regex to capture 'nu' value from filename pattern
    filename_pattern = re.compile(r'contra_emulator_nu_(-?\d+\.?\d*)\.pkl')

    for filename in os.listdir(directory_path):

        match = filename_pattern.match(filename)

        if match:

            nu = float(match.group(1)) # Extract the 'nu' number as an integer
            file_path = os.path.join(directory_path, filename)

            with open(file_path, 'rb') as f:

                emulator = pickle.load(f)
                emulators[nu] = emulator  # Store each emulator with 'nu' as the key

    return emulators


def compute_AM_realization(AM_object, deconv, emulator, nu_value, galaxy_sample, mass_model_catalog, halo_catalog, Luminosities_bulge, 
                           Luminosities_36, Luminosity_36_errs, Eff_rads, MH1_means, MH1_errors, distances, distances_err,
                           n_stellar_reals=N_STELLAR_REALS):

    mask, catalog_sc = AM_object.add_scatter(deconv, cut_range=(3, 12), return_catalog=True)

    sorted_indices = np.argsort(catalog_sc)
    catalog_sc_sorted = catalog_sc[sorted_indices]

    dist_samples = np.random.normal(loc=distances[:, np.newaxis], scale=distances_err[:, np.newaxis], size=(len(galaxy_sample), n_stellar_reals))
    L_36_samples = np.random.normal(loc=Luminosities_36[:, np.newaxis], scale=Luminosity_36_errs[:, np.newaxis], size=(len(galaxy_sample), n_stellar_reals)) # 1e9 L_sun

    log_M2L_disk_samples = np.random.normal(loc=np.log10(M2L_DISK_MEAN), scale=M2L_DISK_ERROR, size=(len(galaxy_sample), n_stellar_reals))
    log_M2L_bulge_samples = np.random.normal(loc=np.log10(M2L_BULGE_MEAN), scale=M2L_BULGE_ERROR, size=(len(galaxy_sample), n_stellar_reals))

    M2L_disk_samples = 10**log_M2L_disk_samples # M_sun / L_sun
    M2L_bulge_samples = 10**log_M2L_bulge_samples # M_sun / L_sun

    MH1_samples = np.random.normal(loc=MH1_means[:, np.newaxis], scale=MH1_errors[:, np.newaxis], size=(len(galaxy_sample), n_stellar_reals)) * 1e9 # M_sun

    M_star_samples = np.abs((L_36_samples - Luminosities_bulge[:, np.newaxis]) * M2L_disk_samples + Luminosities_bulge[:, np.newaxis] * M2L_bulge_samples) * (distances[:, np.newaxis] / dist_samples)**2 * 1e9 * 0.7 # M_sun / h
    log_M_star_samples = np.log10(M_star_samples)

    indices_sorted = np.searchsorted(catalog_sc_sorted, log_M_star_samples.flatten())
    indices_sorted = np.clip(indices_sorted, 0, len(catalog_sc_sorted) - 1)
    indices = sorted_indices[indices_sorted].reshape(len(galaxy_sample), n_stellar_reals)

    matched_halos = halo_catalog[indices]

    vels = np.empty((len(galaxy_sample), n_stellar_reals))

    for j, galaxy in enumerate(galaxy_sample):

        selected_rows = mass_model_catalog[mass_model_catalog['ID'] == galaxy]
        rads = np.asarray(selected_rows['R']) # kpc
        V_gas = np.asarray(selected_rows['Vgas'])
        V_disk = np.asarray(selected_rows['Vdisk'])
        V_bul = np.asarray(selected_rows['Vbul'])

        M_baryon = M_star_samples[j] / 0.7 + 1.33 * MH1_samples[j]

        Mvir = matched_halos[j]['Mvir'] / 0.7
        Rvir = matched_halos[j]['Rvir'] / 0.7
        rs = matched_halos[j]['rs'] / 0.7

        if nu_value == 0.0:
            # the case of no halo contraction/expansion, use basic NFW profile
            V_dm = nfw_circular_velocity(rads, Mvir, Rvir, rs)
        
        else:
            # the case of halo contraction/expansion, use contra emulator
            V_dm = nfw_circular_velocity_contra(rads, Eff_rads[j], Rvir, rs, Mvir, M_baryon, emulator)

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

    return vels


def compute_likelihood(alpha_value, scatter_value, AM_object, emulator, nu_value, galaxy_sample, mass_model_catalog, vels_obs_ensemble, vels_obs_err_ensemble, halo_catalog, 
                       Luminosity_bulge, Luminosity_36, Luminosity_36_errs, Eff_radii, MH1_means, MH1_errors, distances, distances_err,
                       Vmax_shift_mode=False):

    theta = {"alpha": alpha_value, "scatter": scatter_value}
    deconv = AM_object.deconvoluted_catalogs(theta, halo_catalog)

    # Split the realizations across processes
    realizations_per_process = np.array_split(np.arange(N_AM_REALS), size)
    local_realizations = realizations_per_process[rank]

    # Compute local results
    local_vels = np.empty((len(local_realizations), len(galaxy_sample), N_STELLAR_REALS))

    for i, realization in enumerate(local_realizations):

        vels = compute_AM_realization(AM_object, deconv, emulator, nu_value, galaxy_sample, mass_model_catalog, halo_catalog,
                                      Luminosity_bulge, Luminosity_36, Luminosity_36_errs, Eff_radii, MH1_means, 
                                      MH1_errors, distances, distances_err)

        local_vels[i, :, :] = vels

    # Gather results from all processes
    gathered_vels = None

    if rank == 0:
        gathered_vels = np.empty((N_AM_REALS, len(galaxy_sample), N_STELLAR_REALS))

    comm.Gather(local_vels, gathered_vels, root=0)

    if rank == 0:
        
        gathered_vels_reshaped = np.transpose(gathered_vels, (1, 0, 2))

        V_mocks_unlogged = [row[row != 0] for row in gathered_vels_reshaped.reshape(len(galaxy_sample), -1)]

        V_mocks = [np.log10(row) for row in V_mocks_unlogged]

        V_obs_ensemble = np.log10(vels_obs_ensemble)
        V_obs_err_ensemble = vels_obs_err_ensemble / (V_obs_ensemble * np.log(10))

        log_likelihoods = np.empty((V_obs_ensemble.shape[0]))

        for i in range(V_obs_ensemble.shape[0]):

            log_likelihood, _ = get_loglike(V_mocks, V_obs_ensemble[i], V_obs_err_ensemble[i])
            log_likelihoods[i] = log_likelihood

        return log_likelihoods

    return None


if __name__ == "__main__":

    galaxy_list, bulge_lumins_dict, galaxy_data_dict, mass_models, sparc_btfr = load_data()

    # Load contra interpolators
    with open("/Users/fedorboreiko/Documents/Oxford/Personal_codes/Codebase/contra_emulators/contra_interpolators.pkl", "rb") as f:
        interpolators = pickle.load(f)

    # Extracting the data for the galaxies in the SPARC sample in the form of 2d array for vectorized calculations.
    L_36_means = np.array([galaxy_data_dict[galaxy]['Total Luminosity at [3.6]'] for galaxy in galaxy_list]) # 1e9 L_sun
    L_36_errors = np.array([galaxy_data_dict[galaxy]['Luminosity Error'] for galaxy in galaxy_list]) # 1e9 L_sun
    Eff_radii = np.array([galaxy_data_dict[galaxy]['Effective Radius at [3.6]'] for galaxy in galaxy_list]) # kpc
    MH1_means = np.array([galaxy_data_dict[galaxy]['Total HI mass'] for galaxy in galaxy_list]) # 1e9 M_sun
    MH1_errors = MH1_means * 0.1 # 1e9 M_sun
    dists = np.array([galaxy_data_dict[galaxy]['Distance'] for galaxy in galaxy_list]) # Mpc
    dists_err = np.array([galaxy_data_dict[galaxy]['Distance Error'] for galaxy in galaxy_list]) # Mpc
    L_bulges = np.array([bulge_lumins_dict[galaxy] for galaxy in galaxy_list]) # 1e9 L_sun

    log_stellar_masses, SMF_data, _ = get_GSMF_ELPETRO(plotting=False)
    halos = np.load("/Users/fedorboreiko/Documents/Oxford/Personal_codes/Codebase/halos_z_0p00.npy")

    proxy = proxies["mvir_proxy"]()
    abundance_match = AbundanceMatch(log_stellar_masses[10:], SMF_data[10:], halo_proxy=proxy, ext_range=(3.0, 12.0),
                                     boxsize=140, faint_end_first=True, scatter_mult=1, faint_end_slope=-0.42)

    # Run one realisation to obtain a true mock velocities

    NUM_TRUTHS = 1
    true_alpha_proxy = -0.4
    true_alpha = np.tan(true_alpha_proxy)
    true_scatter = 0.1
    true_nu = -1.2
    Vmax_shift_mode = False

    if rank == 0:

        mock_vels_ensemble = []
        mock_vels_err_ensemble = []

        for i in range(NUM_TRUTHS):

            theta_true = {"alpha": true_alpha, "scatter": true_scatter}
            deconv = abundance_match.deconvoluted_catalogs(theta_true, halos)

            if true_nu in interpolators:
                true_interpolator = SafeInterpolator(interpolators[true_nu])
            else:
                raise ValueError(f"Interpolator for nu={true_nu} not found.")

            mock_vels_output = compute_AM_realization(abundance_match, deconv, true_interpolator, true_nu, galaxy_list, 
                                                      mass_models, halos, L_bulges, L_36_means, L_36_errors, Eff_radii, 
                                                      MH1_means, MH1_errors, dists, dists_err, n_stellar_reals=1000)
            
            # Compute the mean over only the nonzero entries for each row
            mock_vels = np.array([
                np.mean(row[row != 0]) if np.any(row != 0) else 0
                for row in mock_vels_output
            ])
            
            mock_vels = mock_vels.flatten()
            mock_vels_err = mock_vels * np.array(sparc_btfr['e_Vmax']) / np.array(sparc_btfr['Vmax'])
            mock_vels = mock_vels + np.random.normal(loc=0, scale=mock_vels_err)

            mock_vels_ensemble.append(mock_vels)
            mock_vels_err_ensemble.append(mock_vels_err)

        mock_vels_ensemble = np.array(mock_vels_ensemble)
        mock_vels_err_ensemble = np.array(mock_vels_err_ensemble)

    else: 

        mock_vels_ensemble = None
        mock_vels_err_ensemble = None

    mock_vels_ensemble = comm.bcast(mock_vels_ensemble, root=0)
    mock_vels_err_ensemble = comm.bcast(mock_vels_err_ensemble, root=0)
    
    NUM_TRUTHS = mock_vels_err_ensemble.shape[0]
    
    # Define ranges for alpha, scatter, nu and the mode (Vmax shift or not)
    alpha_proxy_range = np.linspace(-np.pi / 2, 0, 31)
    alpha_range = np.tan(alpha_proxy_range)
    scatter_range = np.linspace(0.01, 1, 31)
    nu_range = np.linspace(-3.0, 3.0, 31)
    nu_range = np.round(nu_range, 1)

    if rank == 0:

        print(f"Running a grid of alpha proxies from {np.min(alpha_proxy_range)} to {np.max(alpha_proxy_range)},")
        print(f"scatters from {np.min(scatter_range)} to {np.max(scatter_range)},")
        print(f"and nu values from {np.min(nu_range)} to {np.max(nu_range)}.")
        
        if Vmax_shift_mode:
            print('Vmax shift')
        else:
            print('Usual mode\n')

    # Create a grid to store the likelihood values for the five true_vels ensembles
    likelihood_grids = np.empty((NUM_TRUTHS, len(alpha_range), len(scatter_range), len(nu_range)))

    # Initialize the progress bar
    if rank == 0:
        total_calculations = len(alpha_range) * len(scatter_range) * len(nu_range)
        pbar = tqdm(total=total_calculations, desc="Grid Points Evaluated", position=0, leave=True)

    # Compute likelihood for each combination of alpha and scatter
    for i, alpha in enumerate(alpha_range):

        for j, scatter in enumerate(scatter_range):

            for k, nu in enumerate(nu_range):

                if nu in interpolators:
                    interpolator = SafeInterpolator(interpolators[nu])
                else:
                    raise ValueError(f"Interpolator for nu={nu} not found.")

                likelihoods = compute_likelihood(alpha, scatter, abundance_match, interpolator, nu, galaxy_list, mass_models, mock_vels_ensemble, 
                                                mock_vels_err_ensemble, halos, L_bulges, L_36_means, L_36_errors, Eff_radii, MH1_means, MH1_errors, 
                                                dists, dists_err, Vmax_shift_mode)

                if likelihoods is not None:
                    likelihood_grids[:, i, j, k] = likelihoods
                    pbar.update(1)

    if rank == 0:
        pbar.close()

        if Vmax_shift_mode:
            np.save(f'likelihood_grid_{NUM_TRUTHS}mocktest_{N_AM_REALS}am_{N_STELLAR_REALS}stellar_alphaproxy_{np.min(alpha_proxy_range)}_{np.max(alpha_proxy_range)}_scatter_{np.min(scatter_range)}_{np.max(scatter_range)}_nu_{np.min(nu_range)}_{np.max(nu_range)}_vmaxshift.npy', likelihood_grids)
        else:
            np.save(f'likelihood_grid_{NUM_TRUTHS}mocktest_{N_AM_REALS}am_{N_STELLAR_REALS}stellar_alphaproxy_{np.min(alpha_proxy_range)}_{np.max(alpha_proxy_range)}_scatter_{np.min(scatter_range)}_{np.max(scatter_range)}_nu_{np.min(nu_range)}_{np.max(nu_range)}.npy', likelihood_grids)

        print("Likelihood grid computation complete!")