"""
Core physics for the modified adiabatic contraction model of
Gnedin et al. (2011, arXiv:1108.5736), generalised to allow halo
expansion via the nu-parameterisation of Dutton et al. (2007)

All radii are expressed in units of Rvir and all masses in units of
Mtot(Rvir).
"""

import numpy as np

# Orbit-averaging power law  r_bar = A * r0 * (r/r0)**w  (Gnedin+11, Eq. 4)
R0 = 0.03
A_DEFAULT = 1.6
W_DEFAULT = 0.8


def _nfw_m(u):
    """log(1+u) - u/(1+u), evaluated stably.

    For small u the two terms cancel to u**2/2, so direct evaluation
    loses all precision below u ~ 1e-8 (relative error ~ eps/u). Use
    the Taylor series  u^2/2 - 2u^3/3 + 3u^4/4 - 4u^5/5  for u < 1e-3
    (truncation error < 1e-15 relative there).
    """
    u = np.asarray(u, dtype=np.float64)
    small = u < 1e-3
    us = np.where(small, u, 0.0)
    series = us**2 * (0.5 + us * (-2.0/3.0 + us * (0.75 - 0.8 * us)))
    ub = np.where(small, 1.0, u)
    direct = np.log1p(ub) - ub / (1.0 + ub)
    return np.where(small, series, direct)


def dm_mass_fraction(x, fb, c):
    """NFW dark-matter mass fraction, m_dm(x) = (1 - fb) M(x)/M(1).

    Paper Eqs. (11)-(12).
    """
    return (1.0 - fb) * _nfw_m(c * np.asarray(x)) / _nfw_m(c)


def _exp_disc_m(u):
    """1 - (1+u)*exp(-u), evaluated stably.

    Direct evaluation cancels 1-vs-1 (signal u**2/2, noise ~ machine
    eps), so it is pure noise for u < 1e-8. Use the Taylor series
    u^2/2 - u^3/3 + u^4/8 - u^5/30  for u < 1e-3.
    """
    u = np.asarray(u, dtype=np.float64)
    small = u < 1e-3
    us = np.where(small, u, 0.0)
    series = us**2 * (0.5 + us * (-1.0/3.0 + us * (0.125 - us / 30.0)))
    ub = np.where(small, 1.0, u)
    direct = 1.0 - (1.0 + ub) * np.exp(-ub)
    return np.where(small, series, direct)


def bar_mass_fraction(x, fb, rb):
    """Exponential-disc baryonic mass fraction and its derivative w.r.t. x.

    Paper Eqs. (10) and (13), normalised so that m_b(1) = fb.
    """
    x = np.asarray(x)
    denom = _exp_disc_m(1.0 / rb)
    mb = fb * _exp_disc_m(x / rb) / denom
    dmb = fb * (x / rb**2) * np.exp(-x / rb) / denom
    return mb, dmb


def orbit_average(r, A=A_DEFAULT, w=W_DEFAULT):
    """Orbit-averaged radius r_bar(r) and its derivative d r_bar / d r."""
    f = R0 * A * (r / R0) ** w
    d = A * w * (r / R0) ** (w - 1.0)
    return f, d


def contract(ri, c, fb, rb, A=A_DEFAULT, w=W_DEFAULT,
             rtol=1e-12, maxiter=200):
    """Solve the contraction equation (paper Eq. 14) for all radii at once.

    The residual f(r) = r*(m_dm,i(y_bar_i) + m_b(y_bar(r))) - g is strictly
    increasing in r, so the root is unique and can be bracketed
    analytically:

        f(r) <= r*(m_dm + m_b_max) - g  =>  f < 0 for r < g/(m_dm + m_b_max)
        f(r) >= r*m_dm - g              =>  f > 0 for r = g/m_dm

    A safeguarded Newton iteration (falling back to bisection whenever a
    Newton step leaves the bracket) is therefore guaranteed to converge.

    Parameters
    ----------
    ri : array_like
        Initial radii in units of Rvir.
    c, fb, rb : float
        Concentration, baryon fraction Mb/(Mvir+Mb), and baryonic
        scale length rb/Rvir.

    Returns
    -------
    rf : ndarray
        Final radii of each Lagrangian shell for standard adiabatic
        contraction (i.e. Gamma(ri) * ri with nu = 1).
    mhi : ndarray
        DM mass fraction enclosed at ri. Because DM mass is conserved
        within each (non-crossing) shell, this is also the DM mass
        enclosed at the shell's final radius.
    """
    ri = np.atleast_1d(np.asarray(ri, dtype=np.float64))
    ri_av, _ = orbit_average(ri, A, w)
    mhi_av = dm_mass_fraction(ri_av, fb, c)
    g = mhi_av * ri / (1.0 - fb)          # LHS of Eq. (14)

    # Analytic bracket [lo, hi] with f(lo) <= 0 <= f(hi).
    mb_max = fb / (1.0 - (1.0 + 1.0 / rb) * np.exp(-1.0 / rb))
    lo = g / (mhi_av + mb_max)
    hi = g / mhi_av
    rf = 0.5 * (lo + hi)

    step_old = hi - lo
    active = np.ones(ri.shape, dtype=bool)
    for _ in range(maxiter):
        r_act = rf[active]
        x, dy = orbit_average(r_act, A, w)
        mb, dmb = bar_mass_fraction(x, fb, rb)
        f = r_act * (mhi_av[active] + mb) - g[active]
        df = mhi_av[active] + mb + r_act * dmb * dy

        # Shrink the bracket using the sign of f (f is increasing in r).
        lo_act, hi_act = lo[active], hi[active]
        lo_act = np.where(f < 0, r_act, lo_act)
        hi_act = np.where(f >= 0, r_act, hi_act)

        # rtsafe safeguard (Numerical Recipes): take the Newton step
        # only if it stays inside the bracket AND is at least halving
        # the previous step size; otherwise bisect. Plain
        # inside-the-bracket Newton can cycle between two interior
        # points when f has a sharp kink (e.g. the baryonic mass
        # saturating near r ~ rb), shrinking the bracket only
        # arithmetically; this safeguard guarantees geometric
        # convergence.
        newton = r_act - f / df
        slow = np.abs(2.0 * f) > np.abs(step_old[active] * df)
        outside = (newton <= lo_act) | (newton >= hi_act)
        bisect = outside | slow
        r_new = np.where(bisect, 0.5 * (lo_act + hi_act), newton)

        rf[active] = r_new
        lo[active], hi[active] = lo_act, hi_act
        step_old[active] = np.where(bisect, 0.5 * (hi_act - lo_act),
                                    np.abs(r_new - r_act))

        converged = (np.abs(r_new - r_act) <= rtol * r_new) | \
                    (hi_act - lo_act <= rtol * r_new)
        idx = np.flatnonzero(active)
        active[idx[converged]] = False
        if not active.any():
            break
    else:
        raise RuntimeError(
            f"contract() did not converge for {active.sum()} radii "
            f"(c={c:g}, fb={fb:g}, rb={rb:g})")

    mhi = dm_mass_fraction(ri, fb, c)
    return rf, mhi


def contract_nu(ri, c, fb, rb, nu, A=A_DEFAULT, w=W_DEFAULT):
    """Final radii under the generalised response rf_true = Gamma**nu * ri
    (paper Eqs. 15-16), together with the enclosed DM mass fraction."""
    rf, mhi = contract(ri, c, fb, rb, A=A, w=w)
    return (rf / ri) ** nu * ri, mhi


def do_contra(ri, c, fb, rb, A=A_DEFAULT, w=W_DEFAULT):
    c = float(np.asarray(c).ravel()[0])
    fb = float(np.asarray(fb).ravel()[0])
    rb = float(np.asarray(rb).ravel()[0])
    return contract(ri, c, fb, rb, A=A, w=w)
