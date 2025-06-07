##### mpiexec -n 4 python contra_emulator_dataspawn.py

from mpi4py import MPI
import numpy as np
import pandas as pd
from btfr.utils.massfuncs import get_GSMF_ELPETRO
from BAM import AbundanceMatch, proxies
from btfr.btfr_utils import update_progress
from tqdm import tqdm

N_AM_REALS = 100
N_STELLAR_REALS = 1000

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

def compute_realization(abundance_match, deconv, sparc_galaxy_list, mass_models, halos, 
                        L_bulges, L_36_means, L_36_errors, Eff_rads, MH1_means, MH1_errors):

    mask, catalog_sc = abundance_match.add_scatter(deconv, cut_range=(3, 12), return_catalog=True)

    sorted_indices = np.argsort(catalog_sc)
    catalog_sc_sorted = catalog_sc[sorted_indices]

    L_36_samples = np.random.normal(loc=L_36_means[:, np.newaxis], scale=L_36_errors[:, np.newaxis], size=(len(sparc_galaxy_list), N_STELLAR_REALS)) # L_sun

    log_M2L_disk_samples = np.random.normal(loc=np.log10(M2L_DISK_MEAN), scale=M2L_DISK_ERROR, size=(len(sparc_galaxy_list), N_STELLAR_REALS)) 
    log_M2L_bulge_samples = np.random.normal(loc=np.log10(M2L_BULGE_MEAN), scale=M2L_BULGE_ERROR, size=(len(sparc_galaxy_list), N_STELLAR_REALS)) 

    M2L_disk_samples = 10**log_M2L_disk_samples # M_sun / L_sun
    M2L_bulge_samples = 10**log_M2L_bulge_samples # M_sun / L_sun

    MH1_samples = np.random.normal(loc=MH1_means[:, np.newaxis], scale=MH1_errors[:, np.newaxis], size=(len(sparc_galaxy_list), N_STELLAR_REALS)) * 1e9 # M_sun

    M_star_samples = ((L_36_samples - L_bulges[:, np.newaxis]) * M2L_disk_samples + L_bulges[:, np.newaxis] * M2L_bulge_samples) * 1e9 * 0.7 # M_sun / h
    log_M_star_samples = np.log10(M_star_samples)

    indices_sorted = np.searchsorted(catalog_sc_sorted, log_M_star_samples.flatten())
    indices_sorted = np.clip(indices_sorted, 0, len(catalog_sc_sorted) - 1)
    indices = sorted_indices[indices_sorted].reshape(len(sparc_galaxy_list), N_STELLAR_REALS)

    matched_halos = halos[indices]

    baryonic_fractions = np.empty((len(sparc_galaxy_list), N_STELLAR_REALS))
    baryonic_scale_radii = np.empty((len(sparc_galaxy_list), N_STELLAR_REALS))
    concentrations = np.empty((len(sparc_galaxy_list), N_STELLAR_REALS))
    initial_radii = np.empty((len(sparc_galaxy_list), N_STELLAR_REALS))

    for j, galaxy in enumerate(sparc_galaxy_list):

        Mvirs = matched_halos[j]['Mvir'] / 0.7 # M_sun
        Rvirs = matched_halos[j]['Rvir'] / 0.7 # kpc
        rs = matched_halos[j]['rs'] / 0.7 # kpc

        rbs = Eff_rads[j] / 1.67835 * np.ones((N_STELLAR_REALS,)) # kpc
        Mbars = M_star_samples[j] / 0.7 + 1.33 * MH1_samples[j] # M_sun

        ris = np.linspace(0.05, 110, 100) # kpc

        cs = Rvirs / rs
        fbs = Mbars / (Mvirs + Mbars)
        rbs = rbs / Rvirs
        ris = ris[np.newaxis, :] / Rvirs[:, np.newaxis]

        baryonic_fractions[j, :] = fbs
        baryonic_scale_radii[j, :] = rbs # in units of Rvir
        concentrations[j, :] = cs
        initial_radii[j, :] = np.max(ris) # in units of Rvir

    return baryonic_fractions, baryonic_scale_radii, concentrations, initial_radii

# Pre-load data
if rank == 0:

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
    L_bulges = np.array([bulge_lumins_dict[galaxy] for galaxy in sparc_galaxy_list]) # 1e9 L_sun

    log_stellar_masses, SMF_data, _ = get_GSMF_ELPETRO(plotting=False)
    halos = np.load("halos_z_0p00.npy")

    proxy = proxies["mvir_proxy"]()
    abundance_match = AbundanceMatch(log_stellar_masses[10:], SMF_data[10:], halo_proxy=proxy, ext_range=(3.0, 12.0),
                                     boxsize=140, faint_end_first=True, scatter_mult=1, faint_end_slope=-0.42)

    theta = {"alpha": 1.2, "scatter": 0.2}  # Will be tuned?
    deconv = abundance_match.deconvoluted_catalogs(theta, halos)

else:

    sparc_galaxy_list = None
    mass_models = None
    L_36_means = None
    L_36_errors = None
    Eff_radii = None
    MH1_means = None
    MH1_errors = None
    L_bulges = None
    abundance_match = None
    deconv = None
    halos = None

# Broadcast data to all processes
sparc_galaxy_list = comm.bcast(sparc_galaxy_list, root=0)
mass_models = comm.bcast(mass_models, root=0)
L_36_means = comm.bcast(L_36_means, root=0)
L_36_errors = comm.bcast(L_36_errors, root=0)
Eff_radii = comm.bcast(Eff_radii, root=0)
MH1_means = comm.bcast(MH1_means, root=0)
MH1_errors = comm.bcast(MH1_errors, root=0)
L_bulges = comm.bcast(L_bulges, root=0)
abundance_match = comm.bcast(abundance_match, root=0)
deconv = comm.bcast(deconv, root=0)
halos = comm.bcast(halos, root=0)

# Split the realizations across processes
realizations_per_process = np.array_split(np.arange(N_AM_REALS), size)
local_realizations = realizations_per_process[rank]

# Compute local results
local_fb = np.empty((len(local_realizations), len(sparc_galaxy_list), N_STELLAR_REALS))
local_rb = np.empty((len(local_realizations), len(sparc_galaxy_list), N_STELLAR_REALS))
local_c = np.empty((len(local_realizations), len(sparc_galaxy_list), N_STELLAR_REALS))
local_ri = np.empty((len(local_realizations), len(sparc_galaxy_list), N_STELLAR_REALS))

total_work = len(local_realizations) * size

if rank == 0:
    pbar = tqdm(total=100, desc="Progress", position=0, leave=True)

for i, realization in enumerate(local_realizations):

    fb, rb, c, ri = compute_realization(abundance_match, deconv, sparc_galaxy_list, mass_models, halos, 
                                 L_bulges, L_36_means, L_36_errors, Eff_radii, MH1_means, MH1_errors)
    
    # Save results to a file per realization
    np.save(f"contra_emulator/fb_rank_{rank}_realization_{i}.npy", fb)
    np.save(f"contra_emulator/rb_rank_{rank}_realization_{i}.npy", rb)
    np.save(f"contra_emulator/c_rank_{rank}_realization_{i}.npy", c)
    np.save(f"contra_emulator/ri_rank_{rank}_realization_{i}.npy", ri)

    # Update progress
    local_progress = i + 1
    global_progress = update_progress(comm, rank, size, local_progress, total_work)

    if rank == 0:
        pbar.n = global_progress
        pbar.refresh()

if rank == 0:
    pbar.close()
    print("Computation completed!")

