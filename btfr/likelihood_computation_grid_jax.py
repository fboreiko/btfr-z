import jax
import jax.numpy as jnp
from jax import random, jit
from functools import partial
import numpy as np
from mpi4py import MPI

# Enable 32-bit precision for better performance and memory usage
jax.config.update("jax_enable_x64", True)

# Constants
G = 4.30091e-6  # Gravitational constant in kpc * (km/s)^2 / M_sun


def mpi_weighted_average(log_avg_likelihoods_local, non_nan_counts_local, comm):
    """
    Compute global weighted average using MPI.Allreduce.
    
    Args:
        log_avg_likelihoods_local (np.ndarray): Local averaged log-likelihoods, shape (n_galaxies,)
        non_nan_counts_local (np.ndarray): Local non-NaN counts, shape (n_galaxies,)
        comm (MPI.Comm): MPI communicator
        
    Returns:
        np.ndarray: Global weighted averaged log-likelihoods, shape (n_galaxies,)
    """

    avg_likelihoods_local = np.exp(log_avg_likelihoods_local)

    weighted_local = avg_likelihoods_local * non_nan_counts_local
    
    # Initialize arrays for global sums
    weighted_global = np.zeros_like(weighted_local)
    counts_global = np.zeros_like(non_nan_counts_local)

    # Sum weighted values and counts across all processes
    comm.Allreduce(weighted_local, weighted_global, op=MPI.SUM)
    comm.Allreduce(non_nan_counts_local, counts_global, op=MPI.SUM)

    # Compute global weighted average using JAX operations
    # Handle division by zero by setting result to NaN where counts are zero
    avg_likelihoods_global = np.where(
        counts_global != 0,
        weighted_global / counts_global,
        np.nan
    )

    log_avg_likelihoods_global = np.log(avg_likelihoods_global)

    return log_avg_likelihoods_global

@partial(jit, static_argnames=('n_reals',))
def _generate_galaxy_property_samples_jax(key, L36, L36_errs, MH1, MH1_err, d, d_err, n_reals):
    """
    Generate random samples for galaxy properties using JAX.
    
    Args:
        key: JAX random key
        L36, L36_errs: Disk luminosities and errors (n_galaxies,)
        MH1, MH1_err: HI masses and errors (n_galaxies,)
        d, d_err: Distances and errors (n_galaxies,)
        n_reals: Number of stellar realizations
        
    Returns:
        dict: Dictionary containing all sampled properties
    """
    n_galaxies = len(L36)
    
    # Split random keys for different sampling operations
    keys = random.split(key, 5)
    
    samples = {
        'd': random.normal(
            keys[0], shape=(n_galaxies, n_reals)
        ) * d_err[:, jnp.newaxis] + d[:, jnp.newaxis],
        
        'L36': random.normal(
            keys[1], shape=(n_galaxies, n_reals)
        ) * L36_errs[:, jnp.newaxis] + L36[:, jnp.newaxis],
        
        'M2L_disk': 10**(random.normal(
            keys[2], shape=(n_galaxies, n_reals)
        ) * 0.2 + jnp.log10(0.5)),
        
        'M2L_bulge': 10**(random.normal(
            keys[3], shape=(n_galaxies, n_reals)
        ) * 0.2 + jnp.log10(0.7)),
        
        'MH1': (random.normal(
            keys[4], shape=(n_galaxies, n_reals)
        ) * MH1_err[:, jnp.newaxis] + MH1[:, jnp.newaxis]) * 1e9
    }
    
    return samples


@jit
def _compute_stellar_masses_jax(samples, Lbulge, d):
    """
    Compute stellar masses from sampled properties using JAX.
    
    Args:
        samples: Dictionary of sampled galaxy properties
        Lbulge: Bulge luminosities (n_galaxies,)
        d: Distances (n_galaxies,)
        
    Returns:
        tuple: (stellar_masses, log_stellar_masses)
    """
    Mstellar = jnp.abs(
        (samples['L36'] - Lbulge[:, jnp.newaxis]) * samples['M2L_disk'] + 
        Lbulge[:, jnp.newaxis] * samples['M2L_bulge']
    ) * (d[:, jnp.newaxis] / samples['d'])**2 * 1e9 * 0.7  # M_sun / h
    
    log_Mstellar = jnp.log10(Mstellar)
    
    return Mstellar, log_Mstellar


@jit
def _match_halos_to_galaxies_jax(log_Mstellar, catalog_sc_sorted, sorted_indices):
    """
    Match halos to galaxies based on stellar masses using abundance matching.
    
    Args:
        log_Mstellar: Log stellar masses (n_galaxies, n_reals)
        catalog_sc_sorted: Sorted catalog with scatter (n_halos,)
        sorted_indices: Indices for sorting (n_halos,)
        
    Returns:
        jnp.ndarray: Indices of matched halos
    """
    # Flatten for searchsorted, then reshape
    log_Mstellar_flat = log_Mstellar.flatten()
    indices_sorted = jnp.searchsorted(catalog_sc_sorted, log_Mstellar_flat)
    indices_sorted = jnp.clip(indices_sorted, 0, len(catalog_sc_sorted) - 1)
    indices = sorted_indices[indices_sorted].reshape(log_Mstellar.shape)
    
    return indices


@jit
def nfw_circular_velocity_jax(r, Mvir, Rvir, rs):
    """
    Calculate the circular velocity for a galaxy using the NFW profile (JAX version).
    
    Args:
        r (jnp.ndarray): Radii at which to calculate the velocity (kpc). Shape: (n_galaxies, n_radii)
        Mvir (jnp.ndarray): Virial mass of the halo (M_sun). Shape: (n_galaxies, n_stellar_reals)
        Rvir (jnp.ndarray): Virial radius of the halo (kpc). Shape: (n_galaxies, n_stellar_reals)
        rs (jnp.ndarray): Scale radius of the halo (kpc). Shape: (n_galaxies, n_stellar_reals)
        
    Returns:
        jnp.ndarray: Circular velocities at the given radii (km/s). Shape: (n_galaxies, n_stellar_reals, n_radii)
    """

    rads_expanded = r[:, jnp.newaxis, :]  # Shape: (n_galaxies, 1, n_radii)
    Mvir_expanded = Mvir[:, :, jnp.newaxis]  # Shape: (n_galaxies, n_stellar_reals, 1)
    Rvir_expanded = Rvir[:, :, jnp.newaxis]  # Shape: (n_galaxies, n_stellar_reals, 1)
    rs_expanded = rs[:, :, jnp.newaxis]  # Shape: (n_galaxies, n_stellar_reals, 1)

    # Calculate NFW profile components
    x = rads_expanded / Rvir_expanded
    c = Rvir_expanded / rs_expanded

    # Calculate circular velocity
    vc = jnp.sqrt((G * Mvir_expanded / rads_expanded) *
                 (jnp.log(1 + c * x) - c * x / (1 + c * x)) /
                 (jnp.log(1 + c) - c / (1 + c)))
    return vc


@jit
def nfw_circular_velocity_from_mhi_jax(r, mhi, fb, Mvir):
    """
    Calculate circular velocity using enclosed dark matter mass (JAX version).
    
    Args:
        r (jnp.ndarray): Radii at which to calculate the velocity (kpc), shape (n_galaxies, n_radii)
        mhi (jnp.ndarray): Dark matter mass fraction (unitless), shape (n_galaxies, n_stellar_reals, n_radii)
        fb (jnp.ndarray): Baryon fraction (unitless), shape (n_galaxies, n_stellar_reals)
        Mvir (jnp.ndarray): Virial mass of the halo (M_sun), shape (n_galaxies, n_stellar_reals).

    Returns:
        jnp.ndarray: Circular velocities at the given radii (km/s), shape (n_galaxies, n_stellar_reals, n_radii).
    """
    n_galaxies, n_stellar_reals, n_radii = mhi.shape

    rads_expanded = r[:, jnp.newaxis, :]  # Shape: (n_galaxies, 1, n_radii)
    fb_expanded = fb[:, :, jnp.newaxis]  # Shape: (n_galaxies, n_stellar_reals, 1)
    Mvir_expanded = Mvir[:, :, jnp.newaxis]  # Shape: (n_galaxies, n_stellar_reals, 1)
    
    # check for fb = 1 cases
    valid_mask = fb_expanded != 1

    denominator  = jnp.where(valid_mask, 1 - fb_expanded, 1)
    M_enclosed = jnp.where(valid_mask, mhi / denominator * Mvir_expanded, jnp.nan)

    vc = jnp.where(valid_mask, jnp.sqrt(G * M_enclosed / rads_expanded), jnp.nan)

    return vc


@partial(jit, static_argnames=('emulator',))
def nfw_circular_velocity_contra_jax(r, Reff, Rvir, rs, Mvir, Mbar, emulator):
    """
    Calculate the circular velocity using NFW profile + halo contraction/expansion (JAX version).
    
    Args:
        r (jnp.ndarray): Radii at which to calculate the velocity (kpc). Shape: (n_galaxies, n_radii)
        Reff (float): Effective radius of the galaxy (kpc). (1D array of length n_galaxies).
        Rvir (jnp.ndarray): Virial radius of the halo (kpc). Shape: (n_galaxies, n_stellar_reals)
        rs (jnp.ndarray): Scale radius of the halo (kpc). Shape: (n_galaxies, n_stellar_reals)
        Mvir (jnp.ndarray): Virial mass of the halo (M_sun). Shape: (n_galaxies, n_stellar_reals)
        Mbar (jnp.ndarray): Baryon mass of the galaxy (M_sun). Shape: (n_galaxies, n_stellar_reals)
        emulator: Emulator function for interpolating log(mhi).
        
    Returns:
        jnp.ndarray: Circular velocities at the given radii (km/s). Shape: (n_galaxies, n_stellar_reals, n_radii)
    """
    rb = Reff[:, jnp.newaxis] / 1.67835
    c = Rvir / rs
    fb = Mbar / (Mvir + Mbar)
    rb_uless = rb / Rvir

    rads_expanded = r[:, jnp.newaxis, :]  # Shape: (n_galaxies, 1, n_radii)
    Rvir_expanded = Rvir[:, :, jnp.newaxis]  # Shape: (n_galaxies, n_stellar_reals, 1)

    rads_uless = rads_expanded / Rvir_expanded

    c_expanded = c[:, :, jnp.newaxis]  # Shape: (n_galaxies, n_stellar_reals, 1)
    fb_expanded = fb[:, :, jnp.newaxis]  # Shape: (n_galaxies, n_stellar_reals, 1)
    rb_uless_expanded = rb_uless[:, :, jnp.newaxis]  # Shape: (n_galaxies, n_stellar_reals, 1)

    # Prepare arrays for emulator input
    logc_extended = jnp.broadcast_to(jnp.log10(c_expanded), rads_uless.shape)
    logfb_extended = jnp.broadcast_to(jnp.log10(fb_expanded), rads_uless.shape)
    logrb_extended = jnp.broadcast_to(jnp.log10(rb_uless_expanded), rads_uless.shape)
    lograds_extended = jnp.log10(rads_uless)

    # Build the points array for emulator
    points = jnp.stack([
        logc_extended.ravel(),
        logfb_extended.ravel(),
        logrb_extended.ravel(),
        lograds_extended.ravel()
    ]).T
    
    # Call the emulator
    logmhi = emulator(points)
    logmhi = logmhi.reshape(rads_uless.shape)
    mhi = 10**logmhi
    
    vc = nfw_circular_velocity_from_mhi_jax(r, mhi, fb, Mvir)
    return vc


# JIT this function with static arguments for better performance
@partial(jit, static_argnames=('nu', 'emulator'))
def _compute_rotation_curve_for_galaxy_jax(mass_model_data, samples, 
                                         Mstellar, halo_data, Reff, nu, emulator):
    """
    Compute rotation curve for a single galaxy (JAX version).
    
    Args:
        mass_model_data: Mass model data for all galaxies
        samples: Dictionary of sampled properties
        Mstellar: Stellar masses
        halo_data: Halo data for this galaxy
        Reff: Effective radius
        nu: Contraction/expansion parameter
        emulator: Emulator function
        
    Returns:
        jnp.ndarray: Maximum circular velocities
    """
    r = mass_model_data['R']  # kpc
    Vgas = mass_model_data['Vgas']  # km/s
    Vdisk = mass_model_data['Vdisk']  # km/s
    Vbul = mass_model_data['Vbul']  # km/s

    # Compute baryon mass and halo properties
    Mbar = Mstellar / 0.7 + 1.33 * samples['MH1']  # M_sun
    Mvir = halo_data['Mvir'] / 0.7  
    Rvir = halo_data['Rvir'] / 0.7  
    rs = halo_data['rs'] / 0.7 

    # Compute dark matter circular velocity using JAX conditional
    Vdm = jax.lax.cond(
        nu == 0.0,
        lambda: nfw_circular_velocity_jax(r, Mvir, Rvir, rs),
        lambda: nfw_circular_velocity_contra_jax(r, Reff, Rvir, rs, Mvir, Mbar, emulator)
    )
    
    # Calculate maximum circular velocity
    Vgas_sq = Vgas[:, jnp.newaxis, :] * jnp.abs(Vgas[:, jnp.newaxis, :])  # shape: (n_galaxies, 1, n_radii)

    Vdisk_sq = (samples['M2L_disk'][:, :, jnp.newaxis] * Vdisk[:, jnp.newaxis, :] *
                 jnp.abs(Vdisk[:, jnp.newaxis, :]))  # shape: (n_galaxies, n_stellar_reals, n_radii)

    Vbul_sq = (samples['M2L_bulge'][:, :, jnp.newaxis] * Vbul[:, jnp.newaxis, :] *
                 jnp.abs(Vbul[:, jnp.newaxis, :]))  # shape: (n_galaxies, n_stellar_reals, n_radii)
    
    Vdm_sq = Vdm * jnp.abs(Vdm)  # shape: (n_galaxies, n_stellar_reals, n_radii)

    Vtot_sq = Vgas_sq + Vdisk_sq + Vbul_sq + Vdm_sq

    Vmax = jnp.sqrt(jnp.nanmax(Vtot_sq, axis=2))  # shape: (n_galaxies, n_stellar_reals)

    return Vmax


def compute_AM_realization(key, nu, abundance_match, deconv, emulator, galaxy_data, mass_model_data, 
                           halo_catalog_data, n_reals):
    """
    JAX-optimized version of compute_AM_realization.
    
    Args:
        key: JAX random key
        nu: Parameter for halo contraction/expansion
        deconv: Deconvoluted catalog
        abundance_match: Abundance matching object
        emulator: Emulator function for rotation curve simulation
        mass_model_data: Pre-processed mass model data in vectorized format
                        Dictionary with keys 'R', 'Vgas', 'Vdisk', 'Vbul' 
                        Each value has shape (n_galaxies, n_radii)
        halo_catalog: Original halo catalog
        halo_catalog_data: Pre-processed halo catalog arrays
        Lbulge, L36, L36_errs: Luminosity data
        Reff: Effective radii
        MH1, MH1_err: HI mass data
        d, d_err: Distance data
        n_reals: Number of stellar realizations
        
    Returns:
        jnp.ndarray: Simulated maximum circular velocities
    """
    # Add scatter to the deconvoluted catalog, and return the catalog of stellar masses matched to halos
    mask_matched, catalog_Mstar_matched = abundance_match.add_scatter(deconv, cut_range=(3, 12), return_catalog=True)
    sorted_indices = np.argsort(catalog_Mstar_matched)
    catalog_Mstar_sorted = catalog_Mstar_matched[sorted_indices]
    
    # Generate samples for galaxy properties
    samples = _generate_galaxy_property_samples_jax(
        key, galaxy_data['L36'], galaxy_data['L36_err'], galaxy_data['MH1'], 
        galaxy_data['MH1_err'], galaxy_data['d'], galaxy_data['d_err'], n_reals
    )
    
    # Compute stellar masses
    Mstellar, log_Mstellar = _compute_stellar_masses_jax(samples, galaxy_data['Lbulge'], galaxy_data['d'])
    
    # Match halos to galaxies
    halo_indices = _match_halos_to_galaxies_jax(
        log_Mstellar, jnp.array(catalog_Mstar_sorted), jnp.array(sorted_indices)
    )
    
    # Extract matched halo properties
    keys = ['Mvir', 'Rvir', 'rs', 'select']
    matched_halo_data = {}
    for key_name in keys:
        matched_halo_data[key_name] = halo_catalog_data[key_name][halo_indices]
    
    # Compute rotation curves for ALL galaxies simultaneously (VECTORIZED!)
    # mass_model_data is assumed to be in the correct format already
    vc = _compute_rotation_curve_for_galaxy_jax(
        mass_model_data, samples, Mstellar, matched_halo_data, 
        galaxy_data['Reff'], nu, emulator
    )
    
    # Apply selection mask
    selection_mask = matched_halo_data['select']
    vc = vc * selection_mask
    
    return vc


def get_loglike_split(V_sims, V_mock, V_mock_err):
    """
    NumPy version of get_loglike_split for a single truth.
    
    Args:
        V_sims: Simulated velocities (n_galaxies, n_samples)
        V_mock: Mock observed velocities for one truth (n_galaxies,)
        V_mock_err: Mock observational errors for one truth (n_galaxies,)

    Returns:
        tuple: (avg_likelihoods, non_nan_counts) both with shape (n_galaxies,)
    """
    
    # V_sims: (n_galaxies, n_samples)
    # V_mock: (n_galaxies,) 
    # V_mock_err: (n_galaxies,)
    
    # Expand dimensions for broadcasting
    V_mock_expanded = V_mock[:, np.newaxis]  # Shape: (n_galaxies, 1)
    V_mock_err_expanded = V_mock_err[:, np.newaxis]  # Shape: (n_galaxies, 1)
    
    # Calculate likelihoods for all samples for this single truth
    #likelihoods = np.exp(-0.5 * ((V_mock_expanded - V_sims) / V_mock_err_expanded)**2) / (
    #    np.sqrt(2 * np.pi) * V_mock_err_expanded
    #)  # Shape: (n_galaxies, n_samples)

    # Average likelihood across all samples for each galaxy
    #avg_likelihoods = np.nanmean(likelihoods, axis=1)  # Shape: (n_galaxies,)

    # Count non-nan values per galaxy
    #non_nan_counts = np.sum(~np.isnan(likelihoods), axis=1)  # Shape: (n_galaxies,)

    log_likelihoods = -0.5 * ((V_mock_expanded - V_sims) / V_mock_err_expanded)**2 - np.log(np.sqrt(2 * np.pi) * V_mock_err_expanded)

    valid_mask = ~np.isnan(log_likelihoods)
    non_nan_counts = np.sum(valid_mask, axis=1)

    log_avg_likelihoods = np.full(V_sims.shape[0], -np.inf)

    for i in range(V_sims.shape[0]):
        valid_log_likelihoods = log_likelihoods[i, valid_mask[i, :]]
        if len(valid_log_likelihoods) > 0:
            log_avg_likelihoods[i] = jax.scipy.special.logsumexp(valid_log_likelihoods) - np.log(len(valid_log_likelihoods))
        else:
            log_avg_likelihoods[i] = -np.inf

    return log_avg_likelihoods, non_nan_counts


# Main computation function that can be called from the existing workflow
def compute_likelihood(alpha, scatter, nu, abundance_match, emulator, galaxy_data, mass_model_catalog, 
                       halo_catalog, halo_catalog_data, n_am_reals, n_stellar_reals, size, rank, comm):
    """
    JAX-optimized version of compute_likelihood with improved memory management.
    
    This function maintains the same interface as the original but uses JAX for computation.
    Returns array of log likelihood values for each mock truth (shape: num_truths,).
    """
    # Initialize JAX random key
    key = random.PRNGKey(42 + rank)  # Different seed per process
    
    # Generate deconvoluted catalog
    theta = {"alpha": alpha, "scatter": scatter}
    deconv = abundance_match.deconvoluted_catalogs(theta, halo_catalog)
    
    # Split realizations across processes
    local_start = rank * n_am_reals // size
    local_end = (rank + 1) * n_am_reals // size if rank < size - 1 else n_am_reals
    local_realizations = jnp.arange(local_start, local_end)
    
    # Compute local results using JAX with memory optimization
    local_results = []
    n_galaxies = len(galaxy_data['L36'])
    
    for i in range(len(local_realizations)):
        key, subkey = random.split(key)
        
        result = compute_AM_realization(
            subkey, nu, abundance_match, deconv, emulator, galaxy_data, mass_model_catalog, 
            halo_catalog_data, n_stellar_reals
        )
        
        # Convert to regular numpy to save memory and avoid JAX overhead
        local_results.append(result)

    # Prepare data for gathering using NumPy operations
    local_vc_numpy = np.array([np.array(result) for result in local_results])
    
    # Reshape and process using NumPy
    V_sims_local_numpy = np.log10(np.transpose(local_vc_numpy, (1, 0, 2)))
    V_sims_local_numpy = V_sims_local_numpy.reshape(len(galaxy_data['L36']), -1)

    # if Vmax_shift_mode: not available 
    # mean_V_obs = np.mean(V_obs_numpy)
    # mean_V_sims = np.nanmean(V_sims_local_numpy)
    # shift = mean_V_obs - mean_V_sims
    # V_sims_mode = V_sims_local_numpy + shift

    # V_mocks and V_mocks_err now have shape (num_truths, n_galaxies)
    num_truths = galaxy_data['log_Vmocks'].shape[0]
    log_likelihoods = []
    
    for truth_idx in range(num_truths):

        # Call the per-truth function
        log_avg_likelihoods_local, non_nan_counts_local = get_loglike_split(
            V_sims_local_numpy, galaxy_data['log_Vmocks'][truth_idx], galaxy_data['log_Vmocks_err'][truth_idx]
        )

        # Compute global weighted average using MPI for this specific mock truth
        individual_log_likelihoods = mpi_weighted_average(
            log_avg_likelihoods_local, non_nan_counts_local, comm
        )  # Shape: (n_galaxies,)

        # Compute log likelihood for this specific mock truth
        log_likelihood = np.sum(individual_log_likelihoods)  # Scalar
        log_likelihoods.append(float(log_likelihood))  # Convert to Python float

    # Clean up large arrays
    del local_results, local_vc_numpy, V_sims_local_numpy
    
    if rank == 0:
        return np.array(log_likelihoods)  # Shape: (num_truths,)
    else:
        return None