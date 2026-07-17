"""
Validate the trained emulator grids against direct calculation
(replaces contra_testing.py).

For each trained nu, draws random (log c, log fb, log rb, log rf)
points, computes log mhi both from the emulator and from a direct
forward solve + inversion, and reports error statistics.
"""

import numpy as np
import pandas as pd
 
import train_emulator as te
import sys
sys.path.append("/Users/fedorboreiko/Documents/Oxford/btfr_z/btfr")
from utils.emulator import jax_contra_interpolator
 
NUM_SAMPLES = 1000
RNG_SEED = 0
 
# Test ranges (log10), spanning the training box
LOG_C_RANGE = (1, 3)
LOG_FB_RANGE = (-2.5, -1.0)
LOG_RB_RANGE = (-2.5, -1.5)
LOG_RF_RANGE = (-4, 0)
 
 
def direct_log_mhi(nu, c, fb, rb, log_rf):
    """Direct calculation via the dense forward curve."""
    curves = te.log_mhi_curves(c, fb, rb)
    log_gamma, log_ri, log_mhi = curves
    log_rf_curve = nu * log_gamma + log_ri
    drops = np.flatnonzero(np.diff(log_rf_curve) <= 0.0)
    end = drops[0] + 1 if drops.size else log_rf_curve.size
    return np.interp(log_rf, log_rf_curve[:end], log_mhi[:end],
                     left=np.nan, right=np.nan)
 
 
def validate_one_nu(nu, grid, rng):
    log_c = rng.uniform(*LOG_C_RANGE, NUM_SAMPLES)
    log_fb = rng.uniform(*LOG_FB_RANGE, NUM_SAMPLES)
    log_rb = rng.uniform(*LOG_RB_RANGE, NUM_SAMPLES)
    log_rf = rng.uniform(*LOG_RF_RANGE, NUM_SAMPLES)
 
    positions = np.column_stack([log_c, log_fb, log_rb, log_rf])
    emulated = np.asarray(
        jax_contra_interpolator(grid, positions, te.GRID_AXES))
    direct = np.array([
        direct_log_mhi(nu, 10**lc, 10**lf, 10**lr, lrf)
        for lc, lf, lr, lrf in zip(log_c, log_fb, log_rb, log_rf)])
 
    ok = np.isfinite(emulated) & np.isfinite(direct)
    abs_err = np.abs(emulated[ok] - direct[ok])
    pct_err = np.abs(10**emulated[ok] / 10**direct[ok] - 1) * 100
    return {
        "nu": nu,
        "n_valid": int(ok.sum()),
        "mean_abs_err_dex": abs_err.mean(),
        "max_abs_err_dex": abs_err.max(),
        "mean_pct_err": pct_err.mean(),
    }
 
 
def main():
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank, size = comm.Get_rank(), comm.Get_size()
    except (ImportError, RuntimeError):
        comm, rank, size = None, 0, 1
 
    grids = te.load_grids()
    nus = np.array(sorted(grids))
 
    rows = []
    for i in range(rank, len(nus), size):
        rng = np.random.default_rng(RNG_SEED + i)
        rows.append(validate_one_nu(nus[i], grids[nus[i]], rng))
 
    if comm is not None and size > 1:
        rows = [r for chunk in comm.gather(rows, root=0) or []
                for r in chunk] if rank == 0 else None
 
    if rank == 0:
        df = pd.DataFrame(rows).sort_values("nu")
        df.to_csv("emulator_validation.csv", index=False)
        print(df.to_string(index=False))
 
 
if __name__ == "__main__":
    main()

