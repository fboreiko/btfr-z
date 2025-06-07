##### mpiexec -n 4 python btfr/btfr_contra.py

import sys
sys.path.append('/Users/fedorboreiko/Documents/Oxford/btfr_z')

from mpi4py import MPI
import numpy as np
from numpy.lib import recfunctions as rfn
import pandas as pd
from btfr.utils.massfuncs import get_GSMF_ELPETRO
from BAM import AbundanceMatch, proxies
from btfr.utils.plotting_utils import btfr_plot, explore_hist
from btfr.btfr_utils import get_x_cutoff_fit, nfw_circular_velocity_contra, nfw_circular_velocity, get_loglike
from matplotlib import rcParams
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import time
import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
from functools import partial

jax.config.update("jax_enable_x64", True)

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern']
rcParams['text.usetex'] = True

N_AM_REALS = 10
N_STELLAR_REALS = 1000

ALPHA = -2
SCATTER = 0.01
X = 0.6
NU = np.linspace(-3.0, 3.0, 20)[10]

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
    print(f'Model parameters: alpha = {ALPHA}, scatter = {SCATTER}, x = {X}, nu = {NU}')

@partial(jax.jit, static_argnames=("order",))
def jax_contra_interpolator(grid, positions, grid_axes, order=1):
    """
    A thin wrapper that converts the contra grid and query points into JAX arrays,
    performs interpolation, and applies a validity mask to handle out-of-bounds points
    by setting the log(mhi) value to nans.
    """
    indices = []
    valid_mask = jnp.ones(positions.shape[0], dtype=bool)
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

        vels[j, :] = V_max
        masses[j, :] = M_baryon

        # To visualise baryonic and dark matter contributions to the rotation curve:
        V_bar_max = np.sqrt(np.max(
                                    V_gas * np.abs(V_gas) +
                                    M2L_disk_samples[j][:, np.newaxis] * V_disk * np.abs(V_disk) +
                                    M2L_bulge_samples[j][:, np.newaxis] * V_bul * np.abs(V_bul),
                                    axis=1
                                )) # km/s
        
        dm_vels[j, :] = np.max(V_dm, axis=1)
        bar_vels[j, :] = V_bar_max
    
    selection_mask = matched_halos['select']
    vels *= selection_mask
    masses *= selection_mask
    dm_vels *= selection_mask
    bar_vels *= selection_mask

    return vels, masses, dm_vels, bar_vels

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

    # Implement halo selection on the halo catalog
    slope, intercept = get_x_cutoff_fit(halos, X)
    halos_selected = halos.copy()
    halos_selected = rfn.append_fields(halos_selected, 'select', np.ones(halos.shape[0], dtype=float), usemask=False)
    halo_log_Mvir = np.log10(halos_selected['Mvir'])
    halo_log_vmax = np.log10(halos_selected['vmax'])
    predicted_log_vmax = slope * halo_log_Mvir + intercept
    halos_selected['select'][halo_log_vmax > predicted_log_vmax] = np.nan

    # Create abundance matching (AM) object
    proxy = proxies["mvir_proxy"](use_cache=False)
    abundance_match = AbundanceMatch(log_stellar_masses[10:], SMF_data[10:], halo_proxy=proxy, ext_range=(3.0, 12.0),
                                     boxsize=140, faint_end_first=True, scatter_mult=1, faint_end_slope=-0.42)

    theta = {"alpha": ALPHA, "scatter": SCATTER}  # AM model parameters
    # Deconvolute the AM catalog
    deconv = abundance_match.deconvoluted_catalogs(theta, halos_selected)

    # Instead of loading a pre-trained interpolator, load grids and then build the jax-based callable.
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

    if NU != 0.0:
        if NU in contra_grids:
            # Create a contra_interpolator that uses the JAX-based interpolation.
            contra_grid = contra_grids[NU]
            grid_jax = jnp.array(contra_grid, dtype=jnp.float64)
            def contra_interpolator(points):
                return jax_contra_interpolator(grid_jax, points, grid_axes)
        else:
            print(f"Contra emulator for nu={NU} is not available. Please verify that the grid-generation script includes nu={NU}.")
            sys.exit(1)
    else:
        contra_interpolator = None

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
    halos_selected = None
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
halos_selected = comm.bcast(halos_selected, root=0)
contra_interpolator = comm.bcast(contra_interpolator, root=0)

log_c_grid  = np.linspace(0, 3.9, N_SAMPLES)
log_fb_grid = np.linspace(-3.6, -0.03, N_SAMPLES)
log_rb_grid = np.linspace(-3, -1, N_SAMPLES)
log_rf_grid = np.linspace(-4.8, 0.3, N_SAMPLES)

bounds = [[log_c_grid[0], log_c_grid[-1]], [log_fb_grid[0], log_fb_grid[-1]],
            [log_rb_grid[0], log_rb_grid[-1]], [log_rf_grid[0], log_rf_grid[-1]]]

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

    vels, masses, dm_vels, bar_vels = compute_AM_realization(abundance_match, deconv, sparc_galaxy_list, mass_models, halos_selected, 
                                                             L_bulges, L_36_means, L_36_errors, Eff_radii, MH1_means, MH1_errors, 
                                                             dists, dists_err, contra_interpolator)
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

    V_mocks = np.log10(global_vels_reshaped.reshape(len(galaxy_sample), -1))

    V_dm_mocks = np.log10(global_dm_vels_reshaped.reshape(len(galaxy_sample), -1))
    V_bar_mocks = np.log10(global_bar_vels_reshaped.reshape(len(galaxy_sample), -1))

    # Print the percentages
    original_length = N_AM_REALS * N_STELLAR_REALS
    filtered_lengths = [np.count_nonzero(~np.isnan(row)) for row in V_mocks]
    print("Percentage of original array length retained after filtering out-of-bound points (per galaxy):")
    print([round((length / original_length) * 100, 2) for length in filtered_lengths])

    V_mock = np.array([np.nanmean(row) for row in V_mocks])
    V_mock_err = np.array([np.nanstd(row) for row in V_mocks])

    # Observed data
    V_obs_unlogged = np.array(sparc_btfr['Vmax'])
    V_obs_err_unlogged = np.array(sparc_btfr['e_Vmax'])

    V_obs = np.log10(V_obs_unlogged)
    V_obs_err = V_obs_err_unlogged / (V_obs_unlogged * np.log(10))

    log_likelihood, _ = get_loglike(V_mocks, V_obs, V_obs_err)

    print(f'Log likelihood: {log_likelihood}')

    # Plot the BTFR
    M_mocks = np.log10(global_masses_reshaped.reshape(len(galaxy_sample), -1))

    M_mock = np.array([np.nanmean(row) for row in M_mocks])
    M_mock_err = np.array([np.nanstd(row) for row in M_mocks])

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
                        f'/Users/fedorboreiko/Documents/Oxford/btfr_z/plots/RTstats_alpha_{ALPHA}_scatter_{SCATTER}_x{X}_nu_{NU}/RTstats_galaxy_{galnum}_alpha_{ALPHA}_scatter_{SCATTER}_nu_{NU:.3f}.png')

    btfr_plot(log_likelihood, V_mock, M_mock, V_mock_err, M_mock_err, V_obs, M_obs, V_obs_err, M_obs_err, plotname=f'/Users/fedorboreiko/Documents/Oxford/btfr_z/plots/btfr_alpha_{ALPHA}_scatter_{SCATTER}_x_{X}_nu_{NU:.3f}.png')
