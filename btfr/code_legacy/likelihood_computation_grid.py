import numpy as np
from utils import *
from mpi4py import MPI

G = 4.30091e-6


def mpi_weighted_average(avg_likelihoods_local, non_nan_counts_local, comm):
    """
    Compute global weighted average using MPI.Allreduce with JAX arrays.
    
    Args:
        avg_likelihoods_local (jnp.ndarray): Local averaged likelihoods, shape (175,)
        non_nan_counts_local (jnp.ndarray): Local non-NaN counts, shape (175,)
        comm (MPI.Comm): MPI communicator
        
    Returns:
        jnp.ndarray: Global weighted averaged likelihoods, shape (175,)
    """
    # Compute weighted sum locally using JAX
    weighted_local = avg_likelihoods_local * non_nan_counts_local
    
    # Convert to NumPy for MPI operations (MPI4PY requires NumPy arrays)
    weighted_local_np = np.array(weighted_local)
    counts_local_np = np.array(non_nan_counts_local)
    
    # Initialize arrays for global sums
    weighted_global_np = np.zeros_like(weighted_local_np)
    counts_global_np = np.zeros_like(counts_local_np)
    
    # Sum weighted values and counts across all processes
    comm.Allreduce(weighted_local_np, weighted_global_np, op=MPI.SUM)
    comm.Allreduce(counts_local_np, counts_global_np, op=MPI.SUM)
    
    # Convert back to JAX arrays for final computation
    weighted_global = np.array(weighted_global_np)
    counts_global = np.array(counts_global_np)
    
    # Compute global weighted average using JAX operations
    # Handle division by zero by setting result to NaN where counts are zero
    avg_likelihoods_global = np.where(
        counts_global != 0,
        weighted_global / counts_global,
        np.nan
    )
    
    return avg_likelihoods_global


def _generate_galaxy_property_samples(L36, L36_errs, MH1, MH1_err, d, d_err, n_reals):
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
    n_galaxies = len(L36)
    
    samples = {
        'd': np.random.normal(
            loc=d[:, np.newaxis], 
            scale=d_err[:, np.newaxis], 
            size=(n_galaxies, n_reals) # Mpc
        ),
        'L36': np.random.normal(
            loc=L36[:, np.newaxis], 
            scale=L36_errs[:, np.newaxis], 
            size=(n_galaxies, n_reals) # 1e9 L_sun
        ),
        'M2L_disk': 10**np.random.normal(
            loc=np.log10(0.5), 
            scale=0.2, 
            size=(n_galaxies, n_reals) # M_sun / L_sun
        ),
        'M2L_bulge': 10**np.random.normal(
            loc=np.log10(0.7), 
            scale=0.2, 
            size=(n_galaxies, n_reals) # M_sun / L_sun
        ),
        'MH1': np.random.normal(
            loc=MH1[:, np.newaxis], 
            scale=MH1_err[:, np.newaxis], 
            size=(n_galaxies, n_reals)
        ) * 1e9 # M_sun
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
    ) * (d[:, np.newaxis] / samples['d'])**2 * 1e9 * 0.7 # M_sun / h
    
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


def nfw_circular_velocity_vect(r, Mvir, Rvir, rs):
    """
    Calculate the circular velocity for a galaxy using the NFW profile.
    
    Args:
        r (np.ndarray): Radii at which to calculate the velocity (kpc). Shape: (n_galaxies, n_radii)
        Mvir (np.ndarray): Virial mass of the halo (M_sun). Shape: (n_galaxies, n_stellar_reals)
        Rvir (np.ndarray): Virial radius of the halo (kpc). Shape: (n_galaxies, n_stellar_reals)
        rs (np.ndarray): Scale radius of the halo (kpc). Shape: (n_galaxies, n_stellar_reals)
        
    Returns:
        np.ndarray: Circular velocities at the given radii (km/s). Shape: (n_galaxies, n_stellar_reals, n_radii)
    """

    rads_expanded = r[:, np.newaxis, :]  # Shape: (n_galaxies, 1, n_radii)
    Mvir_expanded = Mvir[:, :, np.newaxis]  # Shape: (n_galaxies, n_stellar_reals, 1)
    Rvir_expanded = Rvir[:, :, np.newaxis]  # Shape: (n_galaxies, n_stellar_reals, 1)
    rs_expanded = rs[:, :, np.newaxis]  # Shape: (n_galaxies, n_stellar_reals, 1)

    # Calculate NFW profile components
    x = rads_expanded / Rvir_expanded
    c = Rvir_expanded / rs_expanded

    # Calculate circular velocity
    vc = np.sqrt((G * Mvir_expanded / rads_expanded) *
                 (np.log(1 + c * x) - c * x / (1 + c * x)) /
                 (np.log(1 + c) - c / (1 + c)))
    return vc


def nfw_circular_velocity_from_mhi_vect(r, mhi, fb, Mvir):
    """
    Calculate circular velocity using enclosed dark matter mass (NumPy version).
    
    Args:
        r (np.ndarray): Radii at which to calculate the velocity (kpc), shape (n_galaxies, n_radii)
        mhi (np.ndarray): Dark matter mass fraction (unitless), shape (n_galaxies, n_stellar_reals, n_radii)
        fb (np.ndarray): Baryon fraction (unitless), shape (n_galaxies, n_stellar_reals)
        Mvir (np.ndarray): Virial mass of the halo (M_sun), shape (n_galaxies, n_stellar_reals).

    Returns:
        np.ndarray: Circular velocities at the given radii (km/s), shape (n_galaxies, n_stellar_reals, n_radii).
    """
    n_galaxies, n_stellar_reals, n_radii = mhi.shape

    rads_expanded = r[:, np.newaxis, :]  # Shape: (n_galaxies, 1, n_radii)
    fb_expanded = fb[:, :, np.newaxis]  # Shape: (n_galaxies, n_stellar_reals, 1)
    Mvir_expanded = Mvir[:, :, np.newaxis]  # Shape: (n_galaxies, n_stellar_reals, 1)
    
    # check for fb = 1 cases
    valid_mask = fb_expanded != 1

    denominator  = np.where(valid_mask, 1 - fb_expanded, 1)
    M_enclosed = np.where(valid_mask, mhi / denominator * Mvir_expanded, np.nan)

    vc = np.where(valid_mask, np.sqrt(G * M_enclosed / rads_expanded), np.nan)

    return vc


def nfw_circular_velocity_contra_vect(r, Reff, Rvir, rs, Mvir, Mbar, emulator):
    """
    Calculate the circular velocity using NFW profile + halo contraction/expansion (NumPy version).
    
    Args:
        r (np.ndarray): Radii at which to calculate the velocity (kpc). Shape: (n_galaxies, n_radii)
        Reff (float): Effective radius of the galaxy (kpc). (1D array of length n_galaxies).
        Rvir (np.ndarray): Virial radius of the halo (kpc). Shape: (n_galaxies, n_stellar_reals)
        rs (np.ndarray): Scale radius of the halo (kpc). Shape: (n_galaxies, n_stellar_reals)
        Mvir (np.ndarray): Virial mass of the halo (M_sun). Shape: (n_galaxies, n_stellar_reals)
        Mbar (np.ndarray): Baryon mass of the galaxy (M_sun). Shape: (n_galaxies, n_stellar_reals)
        emulator: Emulator function for interpolating log(mhi).
        
    Returns:
        np.ndarray: Circular velocities at the given radii (km/s). Shape: (n_galaxies, n_stellar_reals, n_radii)
    """
    rb = Reff[:, np.newaxis] / 1.67835
    c = Rvir / rs
    fb = Mbar / (Mvir + Mbar)
    rb_uless = rb / Rvir

    rads_expanded = r[:, np.newaxis, :]  # Shape: (n_galaxies, 1, n_radii)
    Rvir_expanded = Rvir[:, :, np.newaxis]  # Shape: (n_galaxies, n_stellar_reals, 1)

    rads_uless = rads_expanded / Rvir_expanded

    c_expanded = c[:, :, np.newaxis]  # Shape: (n_galaxies, n_stellar_reals, 1)
    fb_expanded = fb[:, :, np.newaxis]  # Shape: (n_galaxies, n_stellar_reals, 1)
    rb_uless_expanded = rb_uless[:, :, np.newaxis]  # Shape: (n_galaxies, n_stellar_reals, 1)

    # Prepare arrays for emulator input
    logc_extended = np.broadcast_to(np.log10(c_expanded), rads_uless.shape)
    logfb_extended = np.broadcast_to(np.log10(fb_expanded), rads_uless.shape)
    logrb_extended = np.broadcast_to(np.log10(rb_uless_expanded), rads_uless.shape)
    lograds_extended = np.log10(rads_uless)

    # Build the points array for emulator
    points = np.stack([
        logc_extended.ravel(),
        logfb_extended.ravel(),
        logrb_extended.ravel(),
        lograds_extended.ravel()
    ]).T
    
    # Call the emulator
    logmhi = emulator(points)
    logmhi = logmhi.reshape(rads_uless.shape)
    mhi = 10**logmhi
    
    vc = nfw_circular_velocity_from_mhi_vect(r, mhi, fb, Mvir)
    return vc


def _compute_rotation_curve_for_galaxy_vect(mass_model_catalog, samples, 
                                     Mstellar, matched_halos, Reff, nu, emulator):
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
    r = mass_model_catalog['R']  # kpc
    Vgas = mass_model_catalog['Vgas']  # km/s
    Vdisk = mass_model_catalog['Vdisk']  # km/s
    Vbul = mass_model_catalog['Vbul']  # km/s
    
    # Compute baryon mass and halo properties
    Mbar = Mstellar / 0.7 + 1.33 * samples['MH1']  # M_sun
    Mvir = matched_halos['Mvir'] / 0.7  
    Rvir = matched_halos['Rvir'] / 0.7  
    rs = matched_halos['rs'] / 0.7 
    
    # Compute dark matter circular velocity
    if nu == 0.0:
        Vdm = nfw_circular_velocity_vect(r, Mvir, Rvir, rs)
    else:
        Vdm = nfw_circular_velocity_contra_vect(r, Reff, Rvir, rs, Mvir, Mbar, emulator)
    
    Vgas_sq = Vgas[:, np.newaxis, :] * np.abs(Vgas[:, np.newaxis, :])  # shape: (n_galaxies, 1, n_radii)

    Vdisk_sq = (samples['M2L_disk'][:, :, np.newaxis] * Vdisk[:, np.newaxis, :] *
                 np.abs(Vdisk[:, np.newaxis, :]))  # shape: (n_galaxies, n_stellar_reals, n_radii)

    Vbul_sq = (samples['M2L_bulge'][:, :, np.newaxis] * Vbul[:, np.newaxis, :] *
                 np.abs(Vbul[:, np.newaxis, :]))  # shape: (n_galaxies, n_stellar_reals, n_radii)

    Vdm_sq = Vdm * np.abs(Vdm)  # shape: (n_galaxies, n_stellar_reals, n_radii)

    Vtot_sq = Vgas_sq + Vdisk_sq + Vbul_sq + Vdm_sq

    Vmax = np.sqrt(np.nanmax(Vtot_sq, axis=2))  # shape: (n_galaxies, n_stellar_reals)

    return Vmax


def _compute_rotation_curve_for_galaxy(galaxy_idx, mass_model_catalog, samples, 
                                     Mstellar, matched_halos, Reff, nu, emulator):
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
    r = np.asarray(mass_model_catalog['R'][galaxy_idx, :]) # kpc
    Vgas = np.asarray(mass_model_catalog['Vgas'][galaxy_idx, :])
    Vdisk = np.asarray(mass_model_catalog['Vdisk'][galaxy_idx, :])
    Vbul = np.asarray(mass_model_catalog['Vbul'][galaxy_idx, :])
    
    # Compute baryon mass and halo properties
    Mbar = Mstellar[galaxy_idx] / 0.7 + 1.33 * samples['MH1'][galaxy_idx] # M_sun
    Mvir = matched_halos[galaxy_idx]['Mvir'] / 0.7
    Rvir = matched_halos[galaxy_idx]['Rvir'] / 0.7
    rs = matched_halos[galaxy_idx]['rs'] / 0.7
    
    # Compute dark matter circular velocity
    if nu == 0.0:
        Vdm = nfw_circular_velocity(r, Mvir, Rvir, rs)
    else:
        Vdm = nfw_circular_velocity_contra(r, Reff[galaxy_idx], Rvir, rs, Mvir, Mbar, emulator)

    # Calculate maximum circular velocity
    Vmax = np.sqrt(np.nanmax(
        Vgas * np.abs(Vgas) +
        samples['M2L_disk'][galaxy_idx][:, np.newaxis] * Vdisk * np.abs(Vdisk) +
        samples['M2L_bulge'][galaxy_idx][:, np.newaxis] * Vbul * np.abs(Vbul) +
        Vdm * np.abs(Vdm),
        axis=1 
    )) # km/s

    return Vmax


def compute_AM_realization(nu, abundance_match, deconv, emulator, galaxy_data, mass_model_catalog, 
                          halo_catalog, n_reals):
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
        galaxy_data (Dict[str, np.ndarray]): Dictionary containing galaxy data arrays.
        mass_model_catalog (pd.DataFrame): Mass model catalog.
        halo_catalog (np.ndarray): Halo catalog.
        n_reals (int): Number of stellar realizations.

    Returns:
        np.ndarray or tuple: 
            - If return_stellar_masses=False: Simulated maximum circular velocities
            - If return_stellar_masses=True: returns additional data for plotting and analysis
    """
    # Prepare abundance matching catalog
    mask_matched, catalog_Mstar_matched = abundance_match.add_scatter(deconv, cut_range=(3, 12), return_catalog=True)
    sorted_indices = np.argsort(catalog_Mstar_matched)
    catalog_Mstar_sorted = catalog_Mstar_matched[sorted_indices]

    # Generate samples for galaxy properties
    samples = _generate_galaxy_property_samples(
        galaxy_data['L36'], galaxy_data['L36_err'], galaxy_data['MH1'], 
        galaxy_data['MH1_err'], galaxy_data['d'], galaxy_data['d_err'], n_reals
    )

    # Compute stellar masses, in Msun/h
    Mstellar, log_Mstellar = _compute_stellar_masses(samples, galaxy_data['Lbulge'], galaxy_data['d'])

    # Match halos to galaxies, masses in Msun/h
    matched_halos = _match_halos_to_galaxies(
        log_Mstellar, catalog_Mstar_sorted, sorted_indices, halo_catalog
    )

    # Initialize output arrays
    vc = np.empty((len(galaxy_data['L36']), n_reals))

    # mass_model_data is assumed to be in the correct format already
    '''vc = _compute_rotation_curve_for_galaxy(
        mass_model_catalog, samples, Mstellar, matched_halos, 
        galaxy_data['Reff'], nu, emulator
    )'''

    # Simulate rotation curves for each galaxy
    for j in range(len(galaxy_data['L36'])):
        results = _compute_rotation_curve_for_galaxy(
                j, mass_model_catalog, samples, Mstellar, matched_halos, 
                galaxy_data['Reff'], nu, emulator
                )
    
        vc[j, :] = results

    # Apply selection mask to filter halos
    selection_mask = matched_halos['select']
    vc *= selection_mask

    return vc


def get_loglike_split(V_sims, V_obs, V_obs_err):
    """
    JAX-optimized version of get_loglike_split.
    
    Args:
        V_sims: Simulated velocities (n_galaxies, n_samples)
        V_obs: Observed velocities (n_galaxies,)
        V_obs_err: Observational errors (n_galaxies,)
        
    Returns:
        tuple: (avg_likelihoods, non_nan_counts)
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


def compute_likelihood(alpha, scatter, nu, abundance_match, emulator, galaxy_data, mass_model_catalog, 
                      halo_catalog, n_am_reals, n_stellar_reals, size, rank, comm):
    """
    Computes the Gaussian likelihood of a model based on galaxy and halo data.

    Args:
        alpha (float): SHAM halo proxy model parameter.
        scatter (float): SHAM scatter model parameter.
        nu (float): Model parameter for halo contraction/expansion.
        abundance_match (AbundanceMatch): Object for abundance matching.
        emulator (Callable): Emulator for adiabatic contraction effect on rotation curves.
        galaxy_data (Dict[str, np.ndarray]): Dictionary containing galaxy data arrays.
        mass_model_catalog (pd.DataFrame): Mass model catalog.
        halo_catalog (np.ndarray): Halo catalog.
        n_am_reals (int): Number of abundance matching realizations.
        n_stellar_reals (int): Number of stellar realizations.
        size (int): Number of processes in parallel computation.
        rank (int): Rank of the current process.
        comm (MPI.Comm): MPI communicator for parallel computation.

    Returns:
        Optional[float or tuple]: 
            - On rank 0: likelihood value, optionally with additional data if requested
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
            nu, abundance_match, deconv, emulator, galaxy_data, mass_model_catalog, 
            halo_catalog, n_stellar_reals
        )
        local_results.append(result)

    # Prepare data for gathering
    local_vc = np.array([result for result in local_results])

    V_sims_local = np.log10(np.transpose(local_vc, (1, 0, 2)))
    V_sims_local = V_sims_local.reshape(len(galaxy_data['L36']), -1)

    # if Vmax_shift_mode: not available 
    # mean_V_obs = jnp.mean(V_obs_jax)
    # mean_V_sims = jnp.nanmean(V_sims_local_jax)
    # shift = mean_V_obs - mean_V_sims
    # V_sims_mode = V_sims_local_jax + shift

    avg_likelihoods_local, non_nan_counts_local = get_loglike_split(
        V_sims_local, galaxy_data['log_Vobs'], galaxy_data['log_Vobs_err']
    )

    # Compute global weighted average using MPI with JAX arrays
    avg_likelihoods_global = mpi_weighted_average(
        avg_likelihoods_local, non_nan_counts_local, comm
    )

    # Compute log likelihood using JAX operations
    individual_log_likelihoods = np.log(avg_likelihoods_global)
    log_likelihood = np.sum(individual_log_likelihoods)
    
    return float(log_likelihood)