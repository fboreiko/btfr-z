import numpy as np

G = 4.30091e-6  # Gravitational constant in kpc * (km/s)^2 / M_sun

def nfw_circular_velocity(r, Mvir, Rvir, rs):
    """
    Calculate the circular velocity for a galaxy using the NFW profile.

    Args:
        r (np.ndarray): Radii at which to calculate the velocity (kpc).
        Mvir (float): Virial mass of the halo (M_sun).
        Rvir (float): Virial radius of the halo (kpc).
        rs (float): Scale radius of the halo (kpc).

    Returns:
        np.ndarray: Circular velocities at the given radii (km/s).
    """
    x = r / Rvir
    c = Rvir / rs # concentration
    # Calculate circular velocity
    vc = np.sqrt((G * Mvir[:, np.newaxis] / r) * 
                (np.log(1 + c * x) - c * x / (1 + c * x)) / 
                (np.log(1 + c) - c / (1 + c)))
    return vc


def nfw_circular_velocity_from_mhi(r, mh, fb, Mvir):
    """
    Calculate circular velocity using enclosed dark matter mass.

    Args:
        r (np.ndarray): Radii at which to calculate the velocity (kpc).
        mh (np.ndarray): Dark matter mass fraction (unitless).
        fb (np.ndarray): Baryon fraction (unitless).
        Mvir (np.ndarray): Virial mass of the halo (M_sun).

    Returns:
        np.ndarray: Circular velocities at the given radii (km/s).
    """
    if np.all(fb != 1):
        Menc = mh / (1 - fb[:, None]) * Mvir[:, None] # Enclosed mass
        vc = np.sqrt(G * Menc / r) # Circular velocity
    else:
        # Initialize vc with NaNs so that values corresponding to fb==1 remain NaN.
        vc = np.full((len(Mvir), len(r)), np.nan)
        valid = fb != 1
        Menc = mh[valid] / (1 - fb[valid, None]) * Mvir[valid, None]
        vc[valid] = np.sqrt(G * Menc / r)
    return vc


def nfw_circular_velocity_contra(r, Reff, Rvir, rs, Mvir, Mbar, emulator):
    """
    Calculate the circular velocity for a galaxy using the NFW profile + halo contraction/expansion.
    Uses an emulator to interpolate the log(mhi) values based on the input parameters.
    This function assumes that the emulator is a callable that takes points of shape (n, 4) and returns interpolated log(mhi).

    Args:
        r (np.ndarray): Radii at which to calculate the velocity (kpc).
        Reff (float): Effective radius of the galaxy (kpc).
        Rvir (np.ndarray): Virial radius of the halo (kpc).
        rs (np.ndarray): Scale radius of the halo (kpc).
        Mvir (np.ndarray): Virial mass of the halo (M_sun).
        Mbar (np.ndarray): Baryon mass of the galaxy (M_sun).
        emulator (callable): Emulator for interpolating log(mhi).

    Returns:
        np.ndarray: Circular velocities at the given radii (km/s).
    """
    rb = Reff / 1.67835
    c = Rvir / rs
    fb= Mbar/ (Mvir + Mbar)
    rb_uless = rb / Rvir
    rads_uless = r[np.newaxis, :] / Rvir[:, np.newaxis]
    
    logc_extended  = np.repeat(np.log10(c)[:, np.newaxis], rads_uless.shape[1], axis=1)
    logfb_extended = np.repeat(np.log10(fb)[:, np.newaxis], rads_uless.shape[1], axis=1)
    logrb_extended = np.repeat(np.log10(rb_uless)[:, np.newaxis], rads_uless.shape[1], axis=1)
    
    # Build the points array: each row is [log(c), log(fb), log(rb), log(r)]
    points = np.vstack((
        logc_extended.ravel(),
        logfb_extended.ravel(),
        logrb_extended.ravel(),
        np.log10(rads_uless).ravel()
    )).T
    
    # Call the emulator (jax-based interpolator) to interpolate initial dark matter mass fraction
    logmhi = np.array(emulator(points)) # out-of-bounds points return nan
    logmhi = logmhi.reshape(rads_uless.shape)
    mhi = 10**logmhi
    vc = nfw_circular_velocity_from_mhi(r, mhi, fb, Mvir)
    return vc