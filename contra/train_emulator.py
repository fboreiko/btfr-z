"""
MPI trainer for the adiabatic-contraction emulator grids.

For every halo-response value nu, tabulates log10 m_dm(rf,true) on a
regular (log c, log fb, log rb, log rf) grid, where rf,true is the
nu-transformed final radius (Eqs. 15-16 of the paper), and pickles a
{nu: RegularGridInterpolator} dictionary.

Method
-----------------------------------------------------------
For each (c, fb, rb) the contraction equation is solved ONCE on a
dense grid of initial radii (vectorised, bracketed Newton with an
rtsafe safeguard in contra_core.contract). Since Gamma(ri) is
nu-independent, that single forward curve is reused for ALL nu values;
each nu costs only the transform rf = Gamma**nu * ri and a 1-D
interpolation onto the target log_rf grid.

Run with any number of MPI ranks (or serially without MPI):
    mpirun -n <nproc> python train_emulator.py
"""

import os
import pickle
 
import numpy as np
 
from contra_core import contract
 
 
N_SAMPLES = 30
# Focus the likelihood-grid computation on the region where the
# posterior resides.
#nu_values = np.linspace(-0.8, 1.6, 25) #selection model
#nu_values = np.linspace(-3.0, 0.0, 31) #baseline model
nu_values = np.array([-1.43, 0.48]) #peaks of posteriors

 
log_rf_values = np.linspace(-4.8, 0.3, N_SAMPLES)
log_c_values = np.linspace(0.0, 3.9, N_SAMPLES)
log_fb_values = np.linspace(-3.6, -0.03, N_SAMPLES)
log_rb_values = np.linspace(-3, -1, N_SAMPLES)

"""
Ideally, subsequent emulator training should be done over the grids:
N_SAMPLES = 50
log_c_values  = np.linspace( 0.0,  4.0, N_SAMPLES)   # concentration
log_fb_values = np.linspace(-4.5, -0.6, N_SAMPLES)   # fb = Mbar/(Mvir+Mbar)
log_rb_values = np.linspace(-4.5, -0.6, N_SAMPLES)   # rb/Rvir
log_rf_values = np.linspace(-5.0,  0.6, N_SAMPLES)   # rf/Rvir
"""
 
c_values = 10.0 ** log_c_values
fb_values = 10.0 ** log_fb_values
rb_values = 10.0 ** log_rb_values
 
# Axis order matches the position columns fed to the jax interpolator.
GRID_AXES = [log_c_values, log_fb_values, log_rb_values, log_rf_values]
 
# Dense initial-radius grid for the forward solve. Wide enough that the
# nu-transformed log_rf covers the target range for all nu.
LOG_RI_DENSE = np.linspace(-13.0, 2.0, 1200)
RI_DENSE = 10.0 ** LOG_RI_DENSE
 
OUTPUT_DIR = "/Users/fedorboreiko/Documents/Oxford/Personal_codes/Codebase/contra_emulators"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "grids_fullrange_finer.pkl")
 
 
def log_mhi_curves(c, fb, rb):
    """Forward solve for one (c, fb, rb): returns (log Gamma, log ri,
    log mhi) on the dense radius grid. nu-independent."""
    rf, mhi = contract(RI_DENSE, c, fb, rb)
    return np.log10(rf / RI_DENSE), LOG_RI_DENSE, np.log10(mhi)
 
 
def invert_for_nu(log_gamma, log_ri, log_mhi, nu,
                  method="mass_conserving"):
    """Given the nu-independent forward curves, return log mhi at the
    target log_rf grid for one nu.
 
    method="first_branch"
        Restrict to the first monotonically increasing branch of
        rf,true(ri) (the innermost-shell solution). Reproduces the
        original pipeline. Whenever shells formally cross (strong
        expansion at high fb), targets beyond the branch are NaN.
 
    method="mass_conserving"  (default)
        Enclosed DM mass at radius R counts EVERY shell whose final
        radius lies inside R:  m(R) = integral of dm_hi over
        {ri : rf,true(ri) <= R}. Identical to first_branch when
        rf,true(ri) is monotonic (the case throughout the shrunk
        parameter box for nu >= -1), and remains well defined under
        shell crossing, where a single-branch root is not.
    """
    log_rf = nu * log_gamma + log_ri          # log of Gamma**nu * ri
 
    if method == "first_branch":
        drops = np.flatnonzero(np.diff(log_rf) <= 0.0)
        end = drops[0] + 1 if drops.size else log_rf.size
        return np.interp(log_rf_values, log_rf[:end], log_mhi[:end],
                         left=np.nan, right=np.nan)
 
    # mass_conserving: m(R) = sum of shell masses with rf_true <= R
    mhi = 10.0 ** log_mhi
    d_mhi = np.diff(mhi)
    log_rf_mid = 0.5 * (log_rf[1:] + log_rf[:-1])
    inside = log_rf_mid[None, :] <= log_rf_values[:, None]
    m = inside @ d_mhi
    m += np.where(log_rf[0] <= log_rf_values, mhi[0], 0.0)  # innermost
    with np.errstate(divide="ignore"):
        out = np.log10(m)
    return np.where(np.isfinite(out), out, np.nan)
 
 
def fill_nan_along_rf(grid):
    """Replace NaNs by interpolation/nearest value along the log_rf
    axis, so the interpolator is not poisoned near unreachable nodes.
    Returns the filled grid and the NaN count."""
    n_bad = int(np.isnan(grid).sum())
    if n_bad:
        flat = grid.reshape(-1, grid.shape[-1])
        idx = np.arange(grid.shape[-1])
        floor = np.nanmin(grid)
        for row in flat:
            good = ~np.isnan(row)
            if not good.any():
                row[:] = floor       # zero enclosed mass everywhere
            elif not good.all():
                row[~good] = np.interp(idx[~good], idx[good], row[good])
    return grid, n_bad
 
 
def main():
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank, size = comm.Get_rank(), comm.Get_size()
    except (ImportError, RuntimeError):
        comm, rank, size = None, 0, 1
 
    n_combo = N_SAMPLES ** 3
    local = np.zeros((len(nu_values), N_SAMPLES, N_SAMPLES, N_SAMPLES,
                      N_SAMPLES))
 
    for flat in range(rank, n_combo, size):
        i, j, k = np.unravel_index(flat, (N_SAMPLES,) * 3)
        curves = log_mhi_curves(c_values[i], fb_values[j], rb_values[k])
        for a, nu in enumerate(nu_values):
            local[a, i, j, k] = invert_for_nu(*curves, nu)
 
    if comm is not None and size > 1:
        from mpi4py import MPI
        full = np.zeros_like(local) if rank == 0 else None
        comm.Reduce(local, full, op=MPI.SUM, root=0)
    else:
        full = local
 
    if rank == 0:
        full, n_bad = fill_nan_along_rf(full)
        grids = {float(nu): full[a] for a, nu in enumerate(nu_values)}
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(OUTPUT_FILE, "wb") as f:
            pickle.dump(grids, f)
        print(f"Grids saved to {OUTPUT_FILE} "
              f"({n_bad} unreachable grid nodes filled along log_rf)")
 
 
def load_grids(path=OUTPUT_FILE):
    with open(path, "rb") as f:
        return pickle.load(f)
 
 
def get_grid(grids, nu, atol=1e-8):
    keys = np.array(sorted(grids))
    key = keys[np.argmin(np.abs(keys - nu))]
    if abs(key - nu) > atol:
        raise KeyError(f"No grid trained at nu={nu}; nearest is "
                       f"{key} (offset {abs(key - nu):.3g})")
    return grids[key]
 
 
if __name__ == "__main__":
    main()

