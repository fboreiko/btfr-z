from mpi4py import MPI
import numpy as np
from numpy.lib import recfunctions as rfn
import pandas as pd
from utils import *
from BAM import AbundanceMatch, proxies
from memory_profiler import profile
from tqdm import tqdm
import os
import pickle
import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
from functools import partial
import sys

#jax.config.update("jax_enable_x64", True)

NUM_TRUTHS = 50

N_AM_REALS = 100
N_STELLAR_REALS = 1000

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

@partial(jax.jit, static_argnames=("order",))
def jax_contra_interpolator(grid, positions, grid_axes, order=1):
    """
    A thin wrapper that converts the contra grid and query points into JAX arrays,
    performs interpolation, and applies a validity mask to handle out-of-bounds points
    by setting the log(mhi) value to nans.
    """
    indices = []
    valid_mask = jnp.ones(positions.shape[0], dtype=bool)  #start with all points valid
    for i in range(positions.shape[1]):
        axis = grid_axes[i]
        grid_min = axis[0]
        grid_max = axis[-1]
        n_points = axis.shape[0]
        index_coord = (positions[:, i] - grid_min) / (grid_max - grid_min) * (n_points - 1)
        valid_mask &= (positions[:, i] >= grid_min) & (positions[:, i] <= grid_max)
        indices.append(index_coord)

    coords = jnp.stack(indices, axis=0)  # shape: (dimensions, n_points)
    interpolated_values = map_coordinates(grid, coords, order=order, mode='constant', cval=0)
    return jnp.where(valid_mask, interpolated_values, jnp.nan)


def load_data():
    """
    Loading and preparing the data. Could be modified to load NSA stellar masses from backup_sparcx.csv, then cut
    off the galaxies from SPARC that are not in the NSA data. Here, we load the data and cut it to match SPARC 
    sample only. It was found that Sersic masses match SPARC masses within 0.2 dex, so we can continue using 
    SPARC while adding more scatter to M2L ratios.
    """
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


def compute_AM_realization(abundance_match, deconv, nu_value, emulator, galaxy_sample, mass_model_catalog, halo_catalog, 
                           Luminosities_bulge, Luminosities_36, Luminosity_36_errs, Eff_rads, MH1_means, MH1_errors, 
                           distances, distances_err, n_stellar_reals):

    # Add scatter to the deconvoluted catalog, and return the catalog of stellar masses matched to halos from the halo catalog
    mask, catalog_sc = abundance_match.add_scatter(deconv, cut_range=(3, 12), return_catalog=True)

    # Sort the catalog and extract the sorted indices
    sorted_indices = np.argsort(catalog_sc)
    catalog_sc_sorted = catalog_sc[sorted_indices]

    # Generate N_STELLAR_REALS of mock stellar masses
    dist_samples = np.random.normal(loc=distances[:, np.newaxis], scale=distances_err[:, np.newaxis], size=(len(galaxy_sample), n_stellar_reals)).astype(np.float32) # Mpc
    L_36_samples = np.random.normal(loc=Luminosities_36[:, np.newaxis], scale=Luminosity_36_errs[:, np.newaxis], size=(len(galaxy_sample), n_stellar_reals)).astype(np.float32) # 1e9 L_sun

    log_M2L_disk_samples = np.random.normal(loc=np.log10(M2L_DISK_MEAN), scale=M2L_DISK_ERROR, size=(len(galaxy_sample), n_stellar_reals)).astype(np.float32)
    log_M2L_bulge_samples = np.random.normal(loc=np.log10(M2L_BULGE_MEAN), scale=M2L_BULGE_ERROR, size=(len(galaxy_sample), n_stellar_reals)).astype(np.float32)

    M2L_disk_samples = (10**log_M2L_disk_samples).astype(np.float32) # M_sun / L_sun
    M2L_bulge_samples = (10**log_M2L_bulge_samples).astype(np.float32) # M_sun / L_sun

    MH1_samples = (np.random.normal(loc=MH1_means[:, np.newaxis], scale=MH1_errors[:, np.newaxis], size=(len(galaxy_sample), n_stellar_reals)) * 1e9).astype(np.float32) # M_sun

    M_star_samples = (np.abs((L_36_samples - Luminosities_bulge[:, np.newaxis]) * M2L_disk_samples + Luminosities_bulge[:, np.newaxis] * M2L_bulge_samples) * (distances[:, np.newaxis] / dist_samples)**2 * 1e9 * 0.7).astype(np.float32) # M_sun / h
    log_M_star_samples = np.log10(M_star_samples).astype(np.float32)

    # Match the stellar masses to halos
    indices_sorted = np.searchsorted(catalog_sc_sorted, log_M_star_samples.flatten())
    indices_sorted = np.clip(indices_sorted, 0, len(catalog_sc_sorted) - 1)
    indices = sorted_indices[indices_sorted].reshape(len(galaxy_sample), n_stellar_reals)

    matched_halos = halo_catalog[indices]

    # Simulate the rotation curves
    vels = np.empty((len(galaxy_sample), n_stellar_reals), dtype=np.float32)

    for j, galaxy in enumerate(galaxy_sample):
        # Extract the galaxy's data
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
        
        # Calculate the maximum circular velocity of the galaxy's total rotation curve
        V_max = np.sqrt(np.max(
                                V_gas * np.abs(V_gas) +
                                M2L_disk_samples[j][:, np.newaxis] * V_disk * np.abs(V_disk) +
                                M2L_bulge_samples[j][:, np.newaxis] * V_bul * np.abs(V_bul) +
                                V_dm * np.abs(V_dm),
                                axis=1
                            )).astype(np.float32) # km/s

        vels[j, :] = V_max

    selection_mask = matched_halos['select']
    vels *= selection_mask

    return vels


def compute_likelihood(alpha_value, scatter_value, nu_value, abundance_match, emulator, galaxy_sample, mass_model_catalog, 
                       mock_vels, mock_vels_err, halo_catalog, Luminosity_bulge, Luminosity_36, Luminosity_36_errs, Eff_radii, MH1_means, 
                       MH1_errors, distances, distances_err, Vmax_shift_mode, n_stellar_reals):
    
    theta = {"alpha": alpha_value, "scatter": scatter_value}
    deconv = abundance_match.deconvoluted_catalogs(theta, halo_catalog)

    # Split the realizations across processes
    realizations_per_process = np.array_split(np.arange(N_AM_REALS), size)
    local_realizations = realizations_per_process[rank]

    # Compute local results
    local_vels = np.empty((len(local_realizations), len(galaxy_sample), n_stellar_reals), dtype=np.float32)

    for i, realization in enumerate(local_realizations):

        vels = compute_AM_realization(abundance_match, deconv, nu_value, emulator, galaxy_sample, mass_model_catalog, 
                                      halo_catalog,Luminosity_bulge, Luminosity_36, Luminosity_36_errs, Eff_radii, 
                                      MH1_means, MH1_errors, distances, distances_err, n_stellar_reals)

        local_vels[i, :, :] = vels

    # Gather results from all processes
    gathered_vels = None

    if rank == 0:
        gathered_vels = np.empty((N_AM_REALS, len(galaxy_sample), n_stellar_reals), dtype=np.float32)

    comm.Gather(local_vels, gathered_vels, root=0)

    # Calculate log likelihood
    if rank == 0:
    
        V_mocks = np.log10(np.transpose(gathered_vels, (1, 0, 2)))
        V_mocks_flat = V_mocks.reshape(len(galaxy_sample), -1)

        V_obs = np.log10(mock_vels)
        V_obs_err = mock_vels_err / (mock_vels * np.log(10))

        # Mean velocity shift mode
        if Vmax_shift_mode:
            mean_V_obs = np.mean(V_obs)
            mean_V_mocks = np.nanmean(V_mocks_flat)
            shift = mean_V_obs - mean_V_mocks
            V_mocks_mode = V_mocks_flat + shift
        else:
            V_mocks_mode = V_mocks_flat

        log_likelihoods = get_loglike_vect(V_mocks_mode, V_obs, V_obs_err)
    
        return log_likelihoods 

    return None


if __name__ == "__main__":

    galaxy_list, bulge_lumins_dict, galaxy_data_dict, mass_models, sparc_btfr = load_data()
   
    # Load contra interpolators
    try:
        with open("/Users/fedorboreiko/Documents/Oxford/Personal_codes/Codebase/contra_emulators/grids_fullrange.pkl", "rb") as f:
            contra_grids = pickle.load(f)
    except FileNotFoundError:
        print("Error: Contra emulator grids were not found. Please run the grid-generation script first.")
        sys.exit(1)

    # Define grid axes matching the interpolation grid-generation stage.
    N_SAMPLES = 50  # must match the grid generation
    log_c_grid  = np.linspace(0, 3.9, N_SAMPLES)
    log_fb_grid = np.linspace(-3.6, -0.03, N_SAMPLES)
    log_rb_grid = np.linspace(-3, -1, N_SAMPLES)
    log_rf_grid = np.linspace(-4.8, 0.3, N_SAMPLES)
    grid_axes = [log_c_grid, log_fb_grid, log_rb_grid, log_rf_grid]

    # Extracting the data for the galaxies in the SPARC sample in the form of 2d array for vectorized calculations.
    L_36_means = np.array([galaxy_data_dict[galaxy]['Total Luminosity at [3.6]'] for galaxy in galaxy_list]) # 1e9 L_sun
    L_36_errors = np.array([galaxy_data_dict[galaxy]['Luminosity Error'] for galaxy in galaxy_list]) # 1e9 L_sun
    Eff_radii = np.array([galaxy_data_dict[galaxy]['Effective Radius at [3.6]'] for galaxy in galaxy_list]) # kpc
    MH1_means = np.array([galaxy_data_dict[galaxy]['Total HI mass'] for galaxy in galaxy_list]) # 1e9 M_sun
    MH1_errors = MH1_means * 0.1 # 1e9 M_sun
    dists = np.array([galaxy_data_dict[galaxy]['Distance'] for galaxy in galaxy_list]) # Mpc
    dists_err = np.array([galaxy_data_dict[galaxy]['Distance Error'] for galaxy in galaxy_list]) # Mpc
    L_bulges = np.array([bulge_lumins_dict[galaxy] for galaxy in galaxy_list]) # 1e9 L_sun

    # Load GSMF data and halo catalog
    log_stellar_masses, SMF_data, _ = get_GSMF_ELPETRO(plotting=False)
    halos = np.load("/Users/fedorboreiko/Documents/Oxford/Personal_codes/Codebase/halos_z_0p00.npy")

    # Create abundance matching (AM) object
    proxy = proxies["mvir_proxy"](use_cache=False)
    abundance_match = AbundanceMatch(log_stellar_masses[10:], SMF_data[10:], halo_proxy=proxy, ext_range=(3.0, 12.0),
                                    boxsize=140, faint_end_first=True, scatter_mult=1, faint_end_slope=-0.42)

    # Run one realisation to obtain a true mock velocities
    true_alpha_proxy = -0.5
    true_alpha = np.tan(true_alpha_proxy)
    true_scatter = 0.1
    true_x = 0.5
    true_nu = np.linspace(-3.0, 3.0, 20)[12]
    Vmax_shift_mode = False

    if rank == 0:
        
        mock_vels_ensemble = []
        mock_vels_err_ensemble = []

        # Implement halo selection on the halo catalog
        slope, intercept = get_x_cutoff_fit(halos, true_x)
        halos_selected = halos.copy()
        halos_selected = rfn.append_fields(halos_selected, 'select', np.ones(halos.shape[0], dtype=float), usemask=False)
        halo_log_Mvir = np.log10(halos_selected['Mvir'])
        halo_log_vmax = np.log10(halos_selected['vmax'])
        predicted_log_vmax = slope * halo_log_Mvir + intercept
        halos_selected['select'][halo_log_vmax > predicted_log_vmax] = np.nan

        # Create a contra_interpolator that uses the JAX-based interpolation.
        true_contra_grid = contra_grids[true_nu]
        grid_jax = jnp.array(true_contra_grid, dtype=jnp.float32)
        def contra_interpolator(points):
            return jax_contra_interpolator(grid_jax, points, grid_axes)

        for i in range(NUM_TRUTHS):

            theta_true = {"alpha": true_alpha, "scatter": true_scatter}
            deconv = abundance_match.deconvoluted_catalogs(theta_true, halos_selected)

            mock_vels_output = compute_AM_realization(abundance_match, deconv, true_nu, contra_interpolator, galaxy_list, 
                                mass_models, halos_selected, L_bulges, L_36_means, L_36_errors, Eff_radii, 
                                MH1_means, MH1_errors, dists, dists_err, n_stellar_reals=100)

            # Pick any non-nan element from each row
            mock_vels = np.array([
                row[np.where(~np.isnan(row))[0][0]] if np.any(~np.isnan(row)) else np.nan
                for row in mock_vels_output
            ])

            # check for nans
            if np.isnan(mock_vels).any():
                break

            mock_vels_err = mock_vels * np.array(sparc_btfr['e_Vmax']) / np.array(sparc_btfr['Vmax'])
            mock_vels = mock_vels + np.random.normal(loc=0, scale=mock_vels_err)

            mock_vels_ensemble.append(mock_vels)
            mock_vels_err_ensemble.append(mock_vels_err)

        mock_vels_ensemble = np.array(mock_vels_ensemble).astype(np.float32)
        mock_vels_err_ensemble = np.array(mock_vels_err_ensemble).astype(np.float32)

    else: 

        mock_vels_ensemble = None
        mock_vels_err_ensemble = None

    mock_vels_ensemble = comm.bcast(mock_vels_ensemble, root=0)
    mock_vels_err_ensemble = comm.bcast(mock_vels_err_ensemble, root=0)
    
    num_truths_corrected = mock_vels_ensemble.shape[0]
    
    # Define ranges for alpha, scatter, nu and the mode (Vmax shift or not)
    alpha_proxy_range = np.linspace(-np.pi / 2, np.pi / 2, 20)
    alpha_range = np.tan(alpha_proxy_range)
    scatter_range = np.linspace(0.01, 1, 20)
    x_range = np.linspace(0.01, 0.95, 20)
    nu_range = np.linspace(-3.0, 3.0, 20)

    N_STELLAR_REALS_MINX = int(N_STELLAR_REALS * (1 - np.max(x_range)) / (1 - np.min(x_range)))
    n_stellar_range = np.floor(np.linspace(N_STELLAR_REALS_MINX, N_STELLAR_REALS, len(x_range))).astype(int)
    N_STELLAR_REALS_SELECT = int(N_STELLAR_REALS_MINX * (1 - np.min(x_range)))

    if rank == 0:
        print(f"\nNumber of mocks corrected: {num_truths_corrected}")
        print(f'Number of AM realizations: {N_AM_REALS}, number of stellar realizations: {N_STELLAR_REALS_SELECT}')
        print(f"True alpha proxy: {true_alpha_proxy}, true scatter: {true_scatter}, true nu: {true_nu}")

        print(f"\nRunning a grid of alpha proxies from {np.min(alpha_proxy_range)} to {np.max(alpha_proxy_range)},")
        print(f"scatters from {np.min(scatter_range)} to {np.max(scatter_range)},")
        print(f"x values from {np.min(x_range)} to {np.max(x_range)},")
        print(f"and nu values from {np.min(nu_range)} to {np.max(nu_range)}.")
        
        if Vmax_shift_mode:
            print('Vmax shift\n')
        else:
            print('Usual mode\n')

    # Create a grid to store the likelihood values for the five true_vels ensembles
    likelihood_grids = np.empty((num_truths_corrected, len(alpha_range), len(scatter_range), len(x_range), len(nu_range)), dtype=np.float32)

    # Initialize the progress bar
    if rank == 0:
        total_calculations = len(alpha_range) * len(scatter_range) * len(x_range) * len(nu_range)
        pbar = tqdm(total=total_calculations, desc="Grid Points Evaluated", position=0, leave=True)

    for i_x, x in enumerate(x_range):

        # Implement halo selection on the halo catalog
        slope, intercept = get_x_cutoff_fit(halos, x)
        halos_selected = halos.copy()
        halos_selected = rfn.append_fields(halos_selected, 'select', np.ones(halos.shape[0], dtype=float), usemask=False)
        halo_log_Mvir = np.log10(halos_selected['Mvir'])
        halo_log_vmax = np.log10(halos_selected['vmax'])
        predicted_log_vmax = slope * halo_log_Mvir + intercept
        halos_selected['select'][halo_log_vmax > predicted_log_vmax] = np.nan

        n_stellar = n_stellar_range[i_x] # in order to keep the number of matched halos constant

        for i_nu, nu in enumerate(nu_range):

            if rank == 0:
                print(f"Starting nu parameter {nu:.3f} ({i_nu+1}/{len(nu_range)})")

            if nu != 0.0:
                if nu in contra_grids:
                    # Create a contra_interpolator that uses the JAX-based interpolation.
                    contra_grid = contra_grids[nu]
                    grid_jax = jnp.array(contra_grid, dtype=jnp.float32)
                    def contra_interpolator(points):
                        return jax_contra_interpolator(grid_jax, points, grid_axes)
                else:
                    print(f"Contra emulator for nu={nu} is not available. Please verify that the grid-generation script includes nu={nu}.")
                    sys.exit(1)
            else:
                contra_interpolator = None

            # Compute likelihood for each combination of alpha, scatter, and nu
            for i_alpha, alpha in enumerate(alpha_range):

                for i_scatter, scatter in enumerate(scatter_range):

                    likelihoods = compute_likelihood(alpha, scatter, nu, abundance_match, contra_interpolator, galaxy_list, mass_models, 
                                                     mock_vels_ensemble, mock_vels_err_ensemble, halos_selected, L_bulges, L_36_means, L_36_errors, Eff_radii, MH1_means, MH1_errors, 
                                                     dists, dists_err, Vmax_shift_mode, n_stellar)

                    #likelihoods is a five element array, each element corresponding to a true_vels ensemble
                    if likelihoods is not None:
                        likelihood_grids[:, i_alpha, i_scatter, i_x, i_nu] = likelihoods
                        pbar.update(1)

    if rank == 0:
        pbar.close()

        if Vmax_shift_mode:
            np.save(f'likelihood_grid_{num_truths_corrected}mocktruths_{N_AM_REALS}am_{N_STELLAR_REALS_SELECT}stellar_alphaproxy_{np.min(alpha_proxy_range)}_{np.max(alpha_proxy_range)}_scatter_{np.min(scatter_range)}_{np.max(scatter_range)}_x_{np.min(x_range)}_{np.max(x_range)}_nu_{np.min(nu_range)}_{np.max(nu_range)}_vmaxshift.npy', likelihood_grids)
        else:
            np.save(f'likelihood_grid_{num_truths_corrected}mocktruths_{N_AM_REALS}am_{N_STELLAR_REALS_SELECT}stellar_alphaproxy_{np.min(alpha_proxy_range)}_{np.max(alpha_proxy_range)}_scatter_{np.min(scatter_range)}_{np.max(scatter_range)}_x_{np.min(x_range)}_{np.max(x_range)}_nu_{np.min(nu_range)}_{np.max(nu_range)}.npy', likelihood_grids)

        print("Likelihood grid computation complete!")