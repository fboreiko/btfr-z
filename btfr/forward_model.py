import jax
import jax.numpy as jnp
from jax import random, jit
from functools import partial
import numpy as np
from mpi4py import MPI
 
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

    L36 and MH1 samples remain referenced to the catalogue's fiducial
    distance. Their common distance rescaling is applied later using
    q = D_draw / D_fid, so that all distance-dependent quantities use the
    same Monte Carlo draw.

    Args:
        key: JAX random key
        L36, L36_errs: 3.6-micron luminosities and errors (n_galaxies,)
        MH1, MH1_err: HI masses and errors (n_galaxies,)
        d, d_err: Fiducial distances and errors (n_galaxies,)
        n_reals: Number of stellar realizations

    Returns:
        dict: Dictionary containing all sampled properties
    """
    n_galaxies = len(L36)

    # Split random keys for different sampling operations
    keys = random.split(key, 5)

    sampled_d = (
        random.normal(keys[0], shape=(n_galaxies, n_reals))
        * d_err[:, jnp.newaxis]
        + d[:, jnp.newaxis]
    )
    # Gaussian distance draws are physically required to be positive.
    sampled_d = jnp.where(sampled_d > 0, sampled_d, jnp.nan)

    samples = {
        'd': sampled_d,

        'L36': random.normal(
            keys[1], shape=(n_galaxies, n_reals)
        ) * L36_errs[:, jnp.newaxis] + L36[:, jnp.newaxis],

        'M2L_disk': 10**(random.normal(
            keys[2], shape=(n_galaxies, n_reals)
        ) * 0.2 + jnp.log10(0.5)),

        'M2L_bulge': 10**(random.normal(
            keys[3], shape=(n_galaxies, n_reals)
        ) * 0.2 + jnp.log10(0.7)),

        # Stored in M_sun at the fiducial catalogue distance. The q^2
        # rescaling is applied when Mbar and the gas velocity are computed.
        'MH1': (random.normal(
            keys[4], shape=(n_galaxies, n_reals)
        ) * MH1_err[:, jnp.newaxis] + MH1[:, jnp.newaxis]) * 1e9
    }

    return samples
 
 
@jit
def _compute_stellar_masses_jax(samples, Lbulge, d):
    """
    Compute stellar masses from sampled properties using JAX.

    For each realisation, luminosity-derived masses scale as
    q^2 = (D_draw / D_fid)^2.

    Args:
        samples: Dictionary of sampled galaxy properties
        Lbulge: Bulge luminosities at the fiducial distance (n_galaxies,)
        d: Fiducial distances (n_galaxies,)

    Returns:
        tuple: (stellar_masses, log_stellar_masses)
    """
    distance_ratio = samples['d'] / d[:, jnp.newaxis]

    Mstellar = jnp.abs(
        (samples['L36'] - Lbulge[:, jnp.newaxis]) * samples['M2L_disk'] +
        Lbulge[:, jnp.newaxis] * samples['M2L_bulge']
    ) * distance_ratio**2 * 1e9 * 0.7  # M_sun / h

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
    Calculate the circular velocity for an NFW halo (JAX version).

    Args:
        r (jnp.ndarray): Realisation-dependent physical radii in kpc,
            shape (n_galaxies, n_stellar_reals, n_radii). A legacy
            (n_galaxies, n_radii) array is also accepted and broadcast.
        Mvir (jnp.ndarray): Virial mass of the halo (M_sun),
            shape (n_galaxies, n_stellar_reals).
        Rvir (jnp.ndarray): Virial radius of the halo (kpc),
            shape (n_galaxies, n_stellar_reals).
        rs (jnp.ndarray): NFW scale radius of the halo (kpc),
            shape (n_galaxies, n_stellar_reals).

    Returns:
        jnp.ndarray: Circular velocities, shape
            (n_galaxies, n_stellar_reals, n_radii).
    """
    rads_expanded = r[:, jnp.newaxis, :] if r.ndim == 2 else r
    Mvir_expanded = Mvir[:, :, jnp.newaxis]
    Rvir_expanded = Rvir[:, :, jnp.newaxis]
    rs_expanded = rs[:, :, jnp.newaxis]

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
        r (jnp.ndarray): Realisation-dependent physical radii in kpc,
            shape (n_galaxies, n_stellar_reals, n_radii). A legacy
            (n_galaxies, n_radii) array is also accepted and broadcast.
        mhi (jnp.ndarray): Dark matter mass fraction (unitless), shape
            (n_galaxies, n_stellar_reals, n_radii).
        fb (jnp.ndarray): Baryon fraction (unitless), shape
            (n_galaxies, n_stellar_reals).
        Mvir (jnp.ndarray): Virial mass of the halo (M_sun), shape
            (n_galaxies, n_stellar_reals).

    Returns:
        jnp.ndarray: Circular velocities, shape
            (n_galaxies, n_stellar_reals, n_radii).
    """
    rads_expanded = r[:, jnp.newaxis, :] if r.ndim == 2 else r
    fb_expanded = fb[:, :, jnp.newaxis]
    Mvir_expanded = Mvir[:, :, jnp.newaxis]

    # Check for fb = 1 cases.
    valid_mask = fb_expanded != 1

    denominator = jnp.where(valid_mask, 1 - fb_expanded, 1)
    M_enclosed = jnp.where(valid_mask, mhi / denominator * Mvir_expanded, jnp.nan)

    vc = jnp.where(valid_mask, jnp.sqrt(G * M_enclosed / rads_expanded), jnp.nan)

    return vc
 
 
@partial(jit, static_argnames=('emulator',))
def nfw_circular_velocity_contra_jax(r, Reff, Rvir, rs, Mvir, Mbar, emulator):
    """
    Calculate the circular velocity using NFW plus halo response.

    Args:
        r (jnp.ndarray): Realisation-dependent physical radii in kpc,
            shape (n_galaxies, n_stellar_reals, n_radii). A legacy
            (n_galaxies, n_radii) array is also accepted and broadcast.
        Reff (jnp.ndarray): Realisation-dependent effective radii in kpc,
            shape (n_galaxies, n_stellar_reals). A legacy one-dimensional
            array is also accepted and broadcast.
        Rvir (jnp.ndarray): Virial radius of the halo (kpc), shape
            (n_galaxies, n_stellar_reals).
        rs (jnp.ndarray): NFW scale radius of the halo (kpc), shape
            (n_galaxies, n_stellar_reals).
        Mvir (jnp.ndarray): Virial mass of the halo (M_sun), shape
            (n_galaxies, n_stellar_reals).
        Mbar (jnp.ndarray): Baryon mass of the galaxy (M_sun), shape
            (n_galaxies, n_stellar_reals).
        emulator: Emulator function for interpolating log(mhi).

    Returns:
        jnp.ndarray: Circular velocities, shape
            (n_galaxies, n_stellar_reals, n_radii).
    """
    rads_expanded = r[:, jnp.newaxis, :] if r.ndim == 2 else r
    Reff_expanded = Reff[:, jnp.newaxis] if Reff.ndim == 1 else Reff

    rb = Reff_expanded / 1.67835
    c = Rvir / rs
    fb = Mbar / (Mvir + Mbar)
    rb_uless = rb / Rvir

    Rvir_expanded = Rvir[:, :, jnp.newaxis]
    rads_uless = rads_expanded / Rvir_expanded

    c_expanded = c[:, :, jnp.newaxis]
    fb_expanded = fb[:, :, jnp.newaxis]
    rb_uless_expanded = rb_uless[:, :, jnp.newaxis]

    # Prepare arrays for emulator input.
    logc_extended = jnp.broadcast_to(jnp.log10(c_expanded), rads_uless.shape)
    logfb_extended = jnp.broadcast_to(jnp.log10(fb_expanded), rads_uless.shape)
    logrb_extended = jnp.broadcast_to(jnp.log10(rb_uless_expanded), rads_uless.shape)
    lograds_extended = jnp.log10(rads_uless)

    points = jnp.stack([
        logc_extended.ravel(),
        logfb_extended.ravel(),
        logrb_extended.ravel(),
        lograds_extended.ravel()
    ]).T

    logmhi = emulator(points)
    logmhi = logmhi.reshape(rads_uless.shape)
    mhi = 10**logmhi

    vc = nfw_circular_velocity_from_mhi_jax(rads_expanded, mhi, fb, Mvir)
    return vc
 
 
@partial(jit, static_argnames=('nu', 'emulator'))
def _compute_rotation_curve_diagnostics_jax(
        mass_model_data, samples, Mstellar, halo_data,
        L36, Lbulge, MH1, d, Reff, nu, emulator):
    """
    Compute total and component rotation-curve diagnostics for all galaxies.

    All distance-dependent quantities use the same Monte Carlo distance draw.
    The returned arrays have shape (n_galaxies, n_stellar_reals).

    Returns:
        tuple:
            Vmax: maximum total circular velocity (km/s)
            Mbar: total baryonic mass (M_sun)
            Vdm_max: maximum dark-matter circular velocity (km/s)
            Vbar_max: maximum baryonic circular velocity (km/s)
    """
    r_fid = mass_model_data['R']
    Vgas = mass_model_data['Vgas']
    Vdisk = mass_model_data['Vdisk']
    Vbul = mass_model_data['Vbul']

    distance_ratio = samples['d'] / d[:, jnp.newaxis]
    q_expanded = distance_ratio[:, :, jnp.newaxis]

    # Use the same distance draw for all physical radii and size parameters.
    r = r_fid[:, jnp.newaxis, :] * q_expanded
    Reff_draw = Reff[:, jnp.newaxis] * distance_ratio

    # The sampled HI mass is referenced to the fiducial distance, so apply q^2.
    MH1_draw = samples['MH1'] * distance_ratio**2

    Mbar = Mstellar / 0.7 + 1.33 * MH1_draw
    Mvir = halo_data['Mvir'] / 0.7
    Rvir = halo_data['Rvir'] / 0.7
    rs = halo_data['rs'] / 0.7

    # ``nu`` and ``emulator`` are static JIT arguments, so use a Python
    # branch. ``lax.cond`` would trace the emulator branch even for nu == 0
    # and fail when emulator is None.
    if nu == 0.0:
        Vdm = nfw_circular_velocity_jax(r, Mvir, Rvir, rs)
    else:
        Vdm = nfw_circular_velocity_contra_jax(
            r, Reff_draw, Rvir, rs, Mvir, Mbar, emulator
        )

    # Scale each baryonic template by (M_draw / M_fid) / q in V^2.
    disk_lum_fid = L36 - Lbulge
    disk_lum_draw = samples['L36'] - Lbulge[:, jnp.newaxis]
    disk_lum_ratio = jnp.where(
        disk_lum_fid[:, jnp.newaxis] > 0,
        disk_lum_draw / disk_lum_fid[:, jnp.newaxis],
        1.0
    )
    disk_lum_ratio = jnp.maximum(disk_lum_ratio, 0.0)

    MH1_fid = MH1[:, jnp.newaxis] * 1e9
    gas_mass_ratio = jnp.where(
        MH1_fid > 0,
        MH1_draw / MH1_fid,
        1.0
    )
    gas_mass_ratio = jnp.maximum(gas_mass_ratio, 0.0)

    gas_vsq_factor = gas_mass_ratio / distance_ratio
    disk_vsq_factor = (
        distance_ratio * disk_lum_ratio * samples['M2L_disk']
    )
    bulge_vsq_factor = distance_ratio * samples['M2L_bulge']

    Vgas_template_sq = Vgas[:, jnp.newaxis, :] * jnp.abs(Vgas[:, jnp.newaxis, :])
    Vdisk_template_sq = Vdisk[:, jnp.newaxis, :] * jnp.abs(Vdisk[:, jnp.newaxis, :])
    Vbul_template_sq = Vbul[:, jnp.newaxis, :] * jnp.abs(Vbul[:, jnp.newaxis, :])

    Vgas_sq = gas_vsq_factor[:, :, jnp.newaxis] * Vgas_template_sq
    Vdisk_sq = disk_vsq_factor[:, :, jnp.newaxis] * Vdisk_template_sq
    Vbul_sq = bulge_vsq_factor[:, :, jnp.newaxis] * Vbul_template_sq
    Vdm_sq = Vdm * jnp.abs(Vdm)

    Vbar_sq = Vgas_sq + Vdisk_sq + Vbul_sq
    Vtot_sq = Vbar_sq + Vdm_sq

    Vmax = jnp.sqrt(jnp.nanmax(Vtot_sq, axis=2))
    Vbar_max = jnp.sqrt(jnp.nanmax(Vbar_sq, axis=2))
    Vdm_max = jnp.nanmax(Vdm, axis=2)

    return Vmax, Mbar, Vdm_max, Vbar_max


@partial(jit, static_argnames=('nu', 'emulator'))
def _compute_rotation_curve_for_galaxy_jax(
        mass_model_data, samples, Mstellar, halo_data,
        L36, Lbulge, MH1, d, Reff, nu, emulator):
    """Velocity-only wrapper used by the likelihood-grid hot path."""
    Vmax, _, _, _ = _compute_rotation_curve_diagnostics_jax(
        mass_model_data, samples, Mstellar, halo_data,
        L36, Lbulge, MH1, d, Reff, nu, emulator
    )
    return Vmax

def _prepare_am_realization_inputs(key, abundance_match, deconv, galaxy_data,
                                   halo_catalog_data, n_reals):
    """Sample galaxy properties and match every draw to a halo."""
    _, catalog_Mstar_matched = abundance_match.add_scatter(
        deconv, cut_range=(3, 12), return_catalog=True
    )
    sorted_indices = np.argsort(catalog_Mstar_matched)
    catalog_Mstar_sorted = catalog_Mstar_matched[sorted_indices]

    samples = _generate_galaxy_property_samples_jax(
        key, galaxy_data['L36'], galaxy_data['L36_err'], galaxy_data['MH1'],
        galaxy_data['MH1_err'], galaxy_data['d'], galaxy_data['d_err'], n_reals
    )

    Mstellar, log_Mstellar = _compute_stellar_masses_jax(
        samples, galaxy_data['Lbulge'], galaxy_data['d']
    )

    halo_indices = _match_halos_to_galaxies_jax(
        log_Mstellar, jnp.asarray(catalog_Mstar_sorted),
        jnp.asarray(sorted_indices)
    )

    matched_halo_data = {
        key_name: halo_catalog_data[key_name][halo_indices]
        for key_name in ('Mvir', 'Rvir', 'rs', 'select')
    }

    return samples, Mstellar, matched_halo_data


def compute_AM_realization(key, nu, abundance_match, deconv, emulator, galaxy_data,
                           mass_model_data, halo_catalog_data, n_reals):
    """
    Compute one abundance-matching realization and return total Vmax only.

    This signature and return type are intentionally unchanged so existing
    likelihood-grid and mock-truth drivers continue to work.
    """
    samples, Mstellar, matched_halo_data = _prepare_am_realization_inputs(
        key, abundance_match, deconv, galaxy_data, halo_catalog_data, n_reals
    )

    Vmax = _compute_rotation_curve_for_galaxy_jax(
        mass_model_data, samples, Mstellar, matched_halo_data,
        galaxy_data['L36'], galaxy_data['Lbulge'], galaxy_data['MH1'],
        galaxy_data['d'], galaxy_data['Reff'], nu, emulator
    )

    return Vmax * matched_halo_data['select']


def compute_AM_realization_diagnostics(
        key, nu, abundance_match, deconv, emulator, galaxy_data,
        mass_model_data, halo_catalog_data, n_reals):
    """
    Compute one abundance-matching realization with plotting diagnostics.

    Returns:
        tuple of JAX arrays (Vmax, Mbar, Vdm_max, Vbar_max), each with shape
        (n_galaxies, n_reals). The halo-selection mask is applied to every
        returned quantity.
    """
    samples, Mstellar, matched_halo_data = _prepare_am_realization_inputs(
        key, abundance_match, deconv, galaxy_data, halo_catalog_data, n_reals
    )

    outputs = _compute_rotation_curve_diagnostics_jax(
        mass_model_data, samples, Mstellar, matched_halo_data,
        galaxy_data['L36'], galaxy_data['Lbulge'], galaxy_data['MH1'],
        galaxy_data['d'], galaxy_data['Reff'], nu, emulator
    )

    selection_mask = matched_halo_data['select']
    return tuple(output * selection_mask for output in outputs)

def get_loglike_split(V_sims, V_target, V_target_err):
    """
    Per-galaxy sample-averaged log-likelihood for a single target dataset,
    computed in log space via logsumexp for numerical stability.
 
    Args:
        V_sims: Simulated log-velocities (n_galaxies, n_samples)
        V_target: Target log-velocities for one dataset (n_galaxies,)
        V_target_err: Target log-velocity errors for one dataset (n_galaxies,)
 
    Returns:
        tuple: (log_avg_likelihoods, non_nan_counts) both with shape (n_galaxies,)
    """
    # Expand dimensions for broadcasting
    V_target_expanded = V_target[:, np.newaxis]  # Shape: (n_galaxies, 1)
    V_target_err_expanded = V_target_err[:, np.newaxis]  # Shape: (n_galaxies, 1)
 
    log_likelihoods = (-0.5 * ((V_target_expanded - V_sims) / V_target_err_expanded)**2
                       - np.log(np.sqrt(2 * np.pi) * V_target_err_expanded))
 
    valid_mask = ~np.isnan(log_likelihoods)
    non_nan_counts = np.sum(valid_mask, axis=1)
 
    log_avg_likelihoods = np.full(V_sims.shape[0], -np.inf)
 
    for i in range(V_sims.shape[0]):
        valid_log_likelihoods = log_likelihoods[i, valid_mask[i, :]]
        if len(valid_log_likelihoods) > 0:
            log_avg_likelihoods[i] = (jax.scipy.special.logsumexp(valid_log_likelihoods)
                                      - np.log(len(valid_log_likelihoods)))
        else:
            log_avg_likelihoods[i] = -np.inf
 
    return log_avg_likelihoods, non_nan_counts
 
 
def compute_simulated_velocities(key, alpha, scatter, nu, abundance_match, emulator,
                                 galaxy_data, mass_model_catalog, halo_catalog,
                                 halo_catalog_data, n_am_reals, n_stellar_reals,
                                 size, rank):
    """
    Run the forward model for one (alpha, scatter, nu) grid point and return
    this rank's local simulated log10(Vmax) samples.
 
    Args:
        key: JAX PRNG key for this grid point (per-rank stream is derived
             internally via fold_in, so all ranks may pass the same key).
        alpha, scatter, nu: Model parameters.
        abundance_match: Abundance matching object.
        emulator: Contra emulator callable (or None for nu == 0).
        galaxy_data: Intrinsic galaxy properties only (see compute_AM_realization).
        mass_model_catalog: Vectorized mass model data.
        halo_catalog: Structured halo catalog (for deconvolution).
        halo_catalog_data: JAX-preprocessed halo catalog arrays.
        n_am_reals: Total number of AM realizations (split across ranks).
        n_stellar_reals: Stellar realizations per AM realization.
        size, rank: MPI layout.
 
    Returns:
        np.ndarray: Local simulated log10 velocities,
                    shape (n_galaxies, n_local_am_reals * n_stellar_reals).
    """
    # Derive an independent stream per rank from the grid-point key
    key = random.fold_in(key, rank)
 
    # Generate deconvoluted catalog
    theta = {"alpha": alpha, "scatter": scatter}
    deconv = abundance_match.deconvoluted_catalogs(theta, halo_catalog)
 
    # Split realizations across processes
    local_start = rank * n_am_reals // size
    local_end = (rank + 1) * n_am_reals // size if rank < size - 1 else n_am_reals
    n_local = local_end - local_start
 
    local_results = []
 
    for _ in range(n_local):
        key, subkey = random.split(key)
 
        result = compute_AM_realization(
            subkey, nu, abundance_match, deconv, emulator, galaxy_data,
            mass_model_catalog, halo_catalog_data, n_stellar_reals
        )
 
        local_results.append(result)
 
    # Stack, log, and flatten the (AM realization, stellar realization) axes
    local_vc_numpy = np.array([np.array(result) for result in local_results])
    V_sims_local = np.log10(np.transpose(local_vc_numpy, (1, 0, 2)))
    V_sims_local = V_sims_local.reshape(V_sims_local.shape[0], -1)
 
    return V_sims_local
 

def compute_simulated_diagnostics(
        key, alpha, scatter, nu, abundance_match, emulator, galaxy_data,
        mass_model_catalog, halo_catalog, halo_catalog_data,
        n_am_reals, n_stellar_reals, size, rank):
    """
    Run one model point and return this rank's local plotting diagnostics.

    Unlike ``compute_simulated_velocities``, values are returned in linear
    units and include baryonic masses and component maxima. The sample axis
    combines local AM realizations and stellar-property realizations.

    Returns:
        dict with keys ``Vmax``, ``Mbar``, ``Vdm_max`` and ``Vbar_max``.
        Every value has shape (n_galaxies, n_local_samples).
    """
    key = random.fold_in(key, rank)

    theta = {"alpha": alpha, "scatter": scatter}
    deconv = abundance_match.deconvoluted_catalogs(theta, halo_catalog)

    local_start = rank * n_am_reals // size
    local_end = ((rank + 1) * n_am_reals // size
                 if rank < size - 1 else n_am_reals)
    n_local = local_end - local_start
    n_galaxies = int(galaxy_data['L36'].shape[0])

    names = ('Vmax', 'Mbar', 'Vdm_max', 'Vbar_max')
    if n_local == 0:
        return {
            name: np.empty((n_galaxies, 0), dtype=np.float64)
            for name in names
        }

    local_results = []
    for _ in range(n_local):
        key, subkey = random.split(key)
        local_results.append(
            compute_AM_realization_diagnostics(
                subkey, nu, abundance_match, deconv, emulator, galaxy_data,
                mass_model_catalog, halo_catalog_data, n_stellar_reals
            )
        )

    diagnostics = {}
    for output_index, name in enumerate(names):
        stacked = np.stack(
            [np.asarray(result[output_index]) for result in local_results],
            axis=0
        )
        diagnostics[name] = np.transpose(stacked, (1, 0, 2)).reshape(
            n_galaxies, -1
        )

    return diagnostics

def evaluate_likelihoods(V_sims_local, log_V_targets, log_V_targets_err, comm):
    """
    Evaluate the log-likelihood of one or more target datasets against a set
    of local simulated velocities, combining across MPI ranks with a
    count-weighted average.
 
    The targets are explicit arguments: pass the observed SPARC data (a
    single dataset) or a mock-truth ensemble (many datasets). This function
    is agnostic to which it is.
 
    Args:
        V_sims_local (np.ndarray): Local simulated log-velocities,
            shape (n_galaxies, n_local_samples).
        log_V_targets (np.ndarray): Target log-velocities,
            shape (n_galaxies,) for a single dataset or
            (num_targets, n_galaxies) for an ensemble.
        log_V_targets_err (np.ndarray): Target log-velocity errors,
            same shape as log_V_targets.
        comm (MPI.Comm): MPI communicator.
 
    Returns:
        np.ndarray: Log-likelihoods, shape (num_targets,). All ranks return
            the same values (the reduction uses Allreduce).
    """
    log_V_targets = np.atleast_2d(np.asarray(log_V_targets, dtype=np.float64))
    log_V_targets_err = np.atleast_2d(np.asarray(log_V_targets_err, dtype=np.float64))
 
    if log_V_targets.shape != log_V_targets_err.shape:
        raise ValueError(
            f"Target and error arrays must have the same shape, got "
            f"{log_V_targets.shape} and {log_V_targets_err.shape}"
        )
    if log_V_targets.shape[1] != V_sims_local.shape[0]:
        raise ValueError(
            f"Targets have {log_V_targets.shape[1]} galaxies but simulations "
            f"have {V_sims_local.shape[0]}"
        )
 
    num_targets = log_V_targets.shape[0]
    log_likelihoods = np.empty(num_targets)
 
    for target_idx in range(num_targets):
        # Per-galaxy averaged likelihood over this rank's samples
        log_avg_likelihoods_local, non_nan_counts_local = get_loglike_split(
            V_sims_local, log_V_targets[target_idx], log_V_targets_err[target_idx]
        )
 
        # Combine across ranks with a count-weighted average
        individual_log_likelihoods = mpi_weighted_average(
            log_avg_likelihoods_local, non_nan_counts_local, comm
        )  # Shape: (n_galaxies,)
 
        # Total log-likelihood for this target: sum over galaxies
        log_likelihoods[target_idx] = float(np.sum(individual_log_likelihoods))
 
    return log_likelihoods
 
 
def compute_likelihood(key, alpha, scatter, nu, abundance_match, emulator, galaxy_data,
                       mass_model_catalog, halo_catalog, halo_catalog_data,
                       log_V_targets, log_V_targets_err,
                       n_am_reals, n_stellar_reals, size, rank, comm):
    """
    Forward model + likelihood evaluation for one grid point.
 
    Thin composition of compute_simulated_velocities and evaluate_likelihoods.
    Target velocities (observed data or mock ensemble) are explicit arguments;
    galaxy_data carries intrinsic galaxy properties only.
 
    Args:
        log_V_targets, log_V_targets_err: shape (n_galaxies,) or
            (num_targets, n_galaxies). See evaluate_likelihoods.
        key: JAX PRNG key for this grid point (same key on all ranks).
        (remaining arguments as in compute_simulated_velocities)
 
    Returns:
        np.ndarray of shape (num_targets,) on rank 0, None on other ranks.
    """
    V_sims_local = compute_simulated_velocities(
        key, alpha, scatter, nu, abundance_match, emulator,
        galaxy_data, mass_model_catalog, halo_catalog, halo_catalog_data,
        n_am_reals, n_stellar_reals, size, rank
    )
 
    log_likelihoods = evaluate_likelihoods(
        V_sims_local, log_V_targets, log_V_targets_err, comm
    )
 
    # Clean up the large simulation array
    del V_sims_local
 
    if rank == 0:
        return log_likelihoods
    else:
        return None