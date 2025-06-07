import numpy as np
from typing import Tuple, Optional, Union
from utils.rotation_curve_utils import nfw_circular_velocity, nfw_circular_velocity_contra
from utils.likelihood_utils import get_loglike


def _generate_galaxy_property_samples(galaxy_sample, L36, L36_errs, MH1, MH1_err, d, d_err, n_reals):
    """
    Generate random samples for galaxy properties.
    
    Args:
        galaxy_sample: Galaxy sample data
        L36, L36_errs: Disk luminosities and errors
        MH1, MH1_err: HI masses and errors
        d, d_err: Distances and errors
        n_reals: Number of stellar realizations
        
    Returns:
        dict: Dictionary containing all sampled properties
    """
    n_galaxies = len(galaxy_sample)
    
    samples = {
        'd': np.random.normal(
            loc=d[:, np.newaxis], 
            scale=d_err[:, np.newaxis], 
            size=(n_galaxies, n_reals)
        ),
        'L36': np.random.normal(
            loc=L36[:, np.newaxis], 
            scale=L36_errs[:, np.newaxis], 
            size=(n_galaxies, n_reals)
        ),
        'M2L_disk': 10**np.random.normal(
            loc=np.log10(0.5), 
            scale=0.2, 
            size=(n_galaxies, n_reals)
        ),
        'M2L_bulge': 10**np.random.normal(
            loc=np.log10(0.7), 
            scale=0.2, 
            size=(n_galaxies, n_reals)
        ),
        'MH1': np.random.normal(
            loc=MH1[:, np.newaxis], 
            scale=MH1_err[:, np.newaxis], 
            size=(n_galaxies, n_reals)
        ) * 1e9
    }
    
    return samples


def _compute_stellar_masses(samples, Lbulge, d):
    """
    Compute stellar masses from sampled properties.
    
    Args:
        samples: Dictionary of sampled galaxy properties
        Lbulge: Bulge luminosities
        d: Distances
        
    Returns:
        tuple: (stellar_masses, log_stellar_masses)
    """
    Mstellar = np.abs(
        (samples['L36'] - Lbulge[:, np.newaxis]) * samples['M2L_disk'] + 
        Lbulge[:, np.newaxis] * samples['M2L_bulge']
    ) * (d[:, np.newaxis] / samples['d'])**2 * 1e9 * 0.7
    
    log_Mstellar = np.log10(Mstellar)
    
    return Mstellar, log_Mstellar


def _match_halos_to_galaxies(log_Mstellar, catalog_sc_sorted, sorted_indices, halo_catalog):
    """
    Match halos to galaxies based on stellar masses using abundance matching.
    
    Args:
        log_Mstellar: Log stellar masses
        catalog_sc_sorted: Sorted catalog with scatter
        sorted_indices: Indices for sorting
        halo_catalog: Halo catalog
        
    Returns:
        np.ndarray: Matched halos
    """
    indices_sorted = np.searchsorted(catalog_sc_sorted, log_Mstellar.flatten())
    indices_sorted = np.clip(indices_sorted, 0, len(catalog_sc_sorted) - 1)
    indices = sorted_indices[indices_sorted].reshape(log_Mstellar.shape)
    
    return halo_catalog[indices]


def _compute_rotation_curve_for_galaxy(galaxy, galaxy_idx, mass_model_catalog, samples, 
                                     Mstellar, matched_halos, Reff, nu, emulator,
                                     return_all_data: bool = False):
    """
    Compute rotation curve for a single galaxy.
    
    Args:
        galaxy: Galaxy identifier
        galaxy_idx: Galaxy index
        mass_model_catalog: Mass model catalog
        samples: Dictionary of sampled properties
        Mstellar: Stellar masses
        matched_halos: Matched halos
        Reff: Effective radii
        nu: Contraction/expansion parameter
        emulator: Emulator function
        
    Returns:
        np.ndarray: Maximum circular velocities, optionally with baryon masses
    """
    # Get mass model data for this galaxy
    selected_rows = mass_model_catalog[mass_model_catalog['ID'] == galaxy]
    r = np.asarray(selected_rows['R'])
    Vgas = np.asarray(selected_rows['Vgas'])
    Vdisk = np.asarray(selected_rows['Vdisk'])
    Vbul = np.asarray(selected_rows['Vbul'])
    
    # Compute baryon mass and halo properties
    Mbar = Mstellar[galaxy_idx] / 0.7 + 1.33 * samples['MH1'][galaxy_idx]
    Mvir = matched_halos[galaxy_idx]['Mvir'] / 0.7
    Rvir = matched_halos[galaxy_idx]['Rvir'] / 0.7
    rs = matched_halos[galaxy_idx]['rs'] / 0.7
    
    # Compute dark matter circular velocity
    if nu == 0.0:
        Vdm = nfw_circular_velocity(r, Mvir, Rvir, rs)
    else:
        Vdm = nfw_circular_velocity_contra(r, Reff[galaxy_idx], Rvir, rs, Mvir, Mbar, emulator)
    
    # Calculate maximum circular velocity
    Vmax = np.sqrt(np.max(
        Vgas * np.abs(Vgas) +
        samples['M2L_disk'][galaxy_idx][:, np.newaxis] * Vdisk * np.abs(Vdisk) +
        samples['M2L_bulge'][galaxy_idx][:, np.newaxis] * Vbul * np.abs(Vbul) +
        Vdm * np.abs(Vdm),
        axis=1
    ))

    if return_all_data:
        Vdm_max = np.sqrt(np.max(Vdm * np.abs(Vdm), axis=1))

    if return_all_data:
        return Vmax, Vdm_max, Mbar
    else:
        return Vmax


def compute_AM_realization(nu, abundance_match, deconv, emulator, galaxy_sample, mass_model_catalog, 
                          halo_catalog, Lbulge, L36, L36_errs, Reff, MH1, MH1_err, d, d_err, n_reals,
                          return_all_data: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, 
                                                                                    np.ndarray, np.ndarray]]:
    """
    Simulates galaxy properties and rotation curves based on SPARC data and abundance matching.

    Steps:
    1. Abundance matching: Adds scatter to the deconvoluted catalog and matches stellar masses to halos.
    2. Rotation curve simulation: Uses NFW profile and halo contraction/expansion parametrized by `nu`.
    3. Halo selection: Filters halos based on selection criteria.

    Args:
        nu (float): Parameter for halo contraction/expansion.
        abundance_match (AbundanceMatch): Object for abundance matching.
        deconv (np.ndarray): Deconvoluted catalog.
        emulator (Callable): Emulator for rotation curve simulation.
        galaxy_sample (pd.DataFrame): Galaxy sample data.
        mass_model_catalog (pd.DataFrame): Mass model catalog.
        halo_catalog (np.ndarray): Halo catalog.
        Lbulge (np.ndarray): Bulge luminosities.
        L36 (np.ndarray): Disk luminosities.
        L36_errs (np.ndarray): Errors in disk luminosities.
        Reff (np.ndarray): Effective radii.
        MH1 (np.ndarray): Mean HI masses.
        MH1_err (np.ndarray): Errors in HI masses.
        d (np.ndarray): Distances to galaxies.
        d_err (np.ndarray): Errors in distances.
        n_reals (int): Number of stellar realizations.

    Returns:
        np.ndarray or tuple: 
            - If return_stellar_masses=False: Simulated maximum circular velocities
            - If return_stellar_masses=True: (velocities, baryon_masses)
    """
    # Prepare abundance matching catalog
    mask, catalog_sc = abundance_match.add_scatter(deconv, cut_range=(3, 12), return_catalog=True)
    sorted_indices = np.argsort(catalog_sc)
    catalog_sc_sorted = catalog_sc[sorted_indices]

    # Generate samples for galaxy properties
    samples = _generate_galaxy_property_samples(
        galaxy_sample, L36, L36_errs, MH1, MH1_err, d, d_err, n_reals
    )

    # Compute stellar masses
    Mstellar, log_Mstellar = _compute_stellar_masses(samples, Lbulge, d)

    # Match halos to galaxies
    matched_halos = _match_halos_to_galaxies(
        log_Mstellar, catalog_sc_sorted, sorted_indices, halo_catalog
    )

    # Initialize output arrays
    vc = np.empty((len(galaxy_sample), n_reals))
    if return_all_data:
        Vdm_max = np.empty((len(galaxy_sample), n_reals))
        Mbar = np.empty((len(galaxy_sample), n_reals))

    # Simulate rotation curves for each galaxy
    for j, galaxy in enumerate(galaxy_sample):
        results = _compute_rotation_curve_for_galaxy(
                galaxy, j, mass_model_catalog, samples, Mstellar, matched_halos, 
                Reff, nu, emulator, return_all_data=return_all_data)
        
        if return_all_data:
            vc[j, :], Vdm_max[j, :], Mbar[j, :] = results
        else:
            vc[j, :] = results

    # Apply selection mask to filter halos
    selection_mask = matched_halos['select']
    vc *= selection_mask

    if return_all_data:
        Mstellar = Mstellar * selection_mask
        Mbar *= selection_mask
        return vc, Vdm_max, Mstellar, Mbar
    else:
        return vc


def compute_likelihood(alpha, scatter, nu, abundance_match, emulator, galaxy_sample, mass_model_catalog, 
                      sparc_catalog, halo_catalog, Lbulge, L36, L36_err, Reff, MH1, 
                      MH1_err, d, d_err, Vmax_shift_mode, n_am_reals, n_stellar_reals, size, rank, comm,
                      return_all_data: bool = False) -> Optional[Union[float, Tuple[float, np.ndarray, np.ndarray, 
                                                                                    np.ndarray, np.ndarray]]]:
    """
    Computes the Gaussian likelihood of a model based on galaxy and halo data.

    Args:
        alpha (float): Alpha parameter for likelihood computation.
        scatter (float): Scatter parameter for likelihood computation.
        nu (float): Parameter for halo contraction/expansion.
        abundance_match (AbundanceMatch): Object for abundance matching.
        emulator (Callable): Emulator for rotation curve simulation.
        galaxy_sample (pd.DataFrame): Galaxy sample data.
        mass_model_catalog (pd.DataFrame): Mass model catalog.
        sparc_catalog (pd.DataFrame): SPARC catalog data.
        halo_catalog (np.ndarray): Halo catalog.
        Lbulge (np.ndarray): Bulge luminosities.
        L36 (np.ndarray): Disk luminosities.
        L36_err (np.ndarray): Errors in disk luminosities.
        Reff (np.ndarray): Effective radii.
        MH1 (np.ndarray): Mean HI masses.
        MH1_err (np.ndarray): Errors in HI masses.
        d (np.ndarray): Distances to galaxies.
        d_err (np.ndarray): Errors in distances.
        Vmax_shift_mode (str): Mode for shifting Vmax values.
        n_am_reals (int): Number of abundance matching realizations.
        n_stellar_reals (int): Number of stellar realizations.
        size (int): Number of processes in parallel computation.
        rank (int): Rank of the current process.
        comm (MPI.Comm): MPI communicator for parallel computation.

    Returns:
        Optional[float or tuple]: 
            - On rank 0: likelihood value, optionally with baryon masses and velocities if requested
            - On other ranks: None
    """
    # Generate deconvoluted catalog based on model parameters
    theta = {"alpha": alpha, "scatter": scatter}
    deconv = abundance_match.deconvoluted_catalogs(theta, halo_catalog)

    # Split AM realizations across processes
    realizations_per_process = np.array_split(np.arange(n_am_reals), size)
    local_realizations = realizations_per_process[rank]

    # Compute local results
    local_results = []
    for realization in local_realizations:
        result = compute_AM_realization(
            nu, abundance_match, deconv, emulator, galaxy_sample, mass_model_catalog, 
            halo_catalog, Lbulge, L36, L36_err, Reff, MH1, MH1_err, d, d_err, n_stellar_reals,
            return_all_data=return_all_data
        )
        local_results.append(result)

    # Prepare data for gathering
    if return_all_data:
        local_vc = np.array([result[0] for result in local_results])
        local_vdm = np.array([result[1] for result in local_results])
        local_Mstellar = np.array([result[2] for result in local_results])
        local_Mbar = np.array([result[3] for result in local_results])
    else:
        local_vc = np.array(local_results)
        local_vdm = None
        local_Mstellar = None
        local_Mbar = None

    # Gather results from all processes
    gathered_vc = None
    gathered_vdm = None
    gathered_Mstellar = None
    gathered_Mbar = None
    
    if rank == 0:
        gathered_vc = np.empty((n_am_reals, len(galaxy_sample), n_stellar_reals))
        if return_all_data:
            gathered_vdm = np.empty((n_am_reals, len(galaxy_sample), n_stellar_reals))
            gathered_Mstellar = np.empty((n_am_reals, len(galaxy_sample), n_stellar_reals))
            gathered_Mbar = np.empty((n_am_reals, len(galaxy_sample), n_stellar_reals))

    comm.Gather(local_vc, gathered_vc, root=0)
    if return_all_data:
        comm.Gather(local_vdm, gathered_vdm, root=0)
        comm.Gather(local_Mstellar, gathered_Mstellar, root=0)
        comm.Gather(local_Mbar, gathered_Mbar, root=0)

    # Compute log likelihood on rank 0
    if rank == 0:
        # Process velocity data
        V_sims = np.log10(np.transpose(gathered_vc, (1, 0, 2)))
        V_sims_flat = V_sims.reshape(len(galaxy_sample), -1)

        # Process observed data
        V_obs_unlogged = np.array(sparc_catalog['Vmax'])
        V_obs_err_unlogged = np.array(sparc_catalog['e_Vmax'])
        V_obs = np.log10(V_obs_unlogged)
        V_obs_err = V_obs_err_unlogged / (V_obs_unlogged * np.log(10))

        # Apply velocity shift mode if specified
        if Vmax_shift_mode:
            mean_V_obs = np.mean(V_obs)
            mean_V_sims = np.nanmean(V_sims_flat)
            shift = mean_V_obs - mean_V_sims
            V_sims_mode = V_sims_flat + shift
        else:
            V_sims_mode = V_sims_flat

        # Compute log likelihood
        log_likelihood, inidividual_log_likelihoods = get_loglike(V_sims_mode, V_obs, V_obs_err)
        
        if return_all_data:
            # Process stellar mass data
            Vdm_sims = np.log10(np.transpose(gathered_vdm, (1, 0, 2)))
            Vdm_sims_flat = Vdm_sims.reshape(len(galaxy_sample), -1)
            Mstellar_sims = np.log10(np.transpose(gathered_Mstellar, (1, 0, 2)))
            Mstellar_sims_flat = Mstellar_sims.reshape(len(galaxy_sample), -1)
            Mbar_sims = np.log10(np.transpose(gathered_Mbar, (1, 0, 2)))
            Mbar_sims_flat = Mbar_sims.reshape(len(galaxy_sample), -1)
            return log_likelihood, inidividual_log_likelihoods, V_sims_flat, Vdm_sims_flat, Mstellar_sims_flat, Mbar_sims_flat
        else:
            return log_likelihood

    return None