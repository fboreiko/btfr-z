import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set the font to Computer Modern (LaTeX default) and enable LaTeX rendering
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern']
rcParams['text.usetex'] = True

r0 = 0.03  # Rvir

def dm_mass_fraction(x, fb, c):

    """
    Inputs: x in unitless (rads/R_vir)
            fb in unitless (M_bar/M_vir)
            c in unitless (R_vir/r_s)
    Outputs: mdm in unitless
    NFW
    """

    mdm = (1 - fb) * (np.log(1 + c*x) - c*x / (1 + c*x)) / (np.log(1 + c) - c / (1 + c))

    return mdm


def bar_mass_fraction(x, fb, rb):

    """
    Inputs: x in unitless (rads/R_vir)
            fb in unitless (M_bar/M_vir)
            rb in unitless (r_b/R_vir)
    Outputs: f in unitless
             d in unitless
    """

    f = fb * (1 - (1 + x / rb) * np.exp(-x / rb)) / (1 - 2 * np.exp(-1 / rb))
    d = fb * x / rb**2 * np.exp(-x / rb) / (1 - 2 * np.exp(-1 / rb))

    return f, d


def tracer_density(x, rb):

    """
    Inputs: x in unitless (rads/R_vir)
            rb in unitless (r_b/R_vir)
    Outputs: rho in unitless
    NFW
    """

    rho = 1 / (x * (rb + x)**2)

    return rho


def y(r, A, w):
    
        """
        Inputs: r in unitless (rads/R_vir)
                A in unitless
                w in unitless
        Outputs: f in unitless (rads/R_vir)
                 d in unitless
        """
    
        try:
            f = r0 * A * (r / r0) ** w
            d = A * w * (r / r0) ** (w - 1)
        except Exception as e:
            print(f"Exception encountered: {e}")
            print(f"Values at the time of exception - r: {r}, r0: {r0}, A: {A}, w: {w}")
            raise  # re-raise the exception after logging
        if np.isnan(f).any() or np.isnan(d).any():
            print(f"NaN encountered - r: {r}, r0: {r0}, A: {A}, w: {w}")
            print(f"Intermediate results - (r / r0): {r / r0}, (r / r0) ** w: {(r / r0) ** w}, (r / r0) ** (w - 1): {(r / r0) ** (w - 1)}")
            exit()

        return f, d


def funcd(r, fb, rb, mhi, g, A, w):

    x, dy = y(r, A, w) #x is unitless, dy is unitless
    mbx, dmb = bar_mass_fraction(x, fb, rb) #mbx is unitless, dmb is unitless
    f = r * (mhi + mbx) - g
    df = mhi + mbx + r * dmb * dy

    return f, df


def find_root(r, fb, rb, mhi, g, A, w):

    # Define a wrapper for the root finding
    def wrapped_funcd(r):
        f, df = funcd(r, fb, rb, mhi, g, A, w)
        return f, df

    # Use root_scalar to find the root with Newton's method
    result = root_scalar(
        lambda r: wrapped_funcd(r)[0],
        fprime=lambda r: wrapped_funcd(r)[1],
        x0=r,
        method='newton',
        xtol=1e-10,  #-5
        maxiter=10000 #1000
    )

    if result.converged:
        return result.root
    else:
        raise ValueError("Root finding did not converge")


def do_contra(ri, c, fb, rb, A, w):

    # Logarithms of initial radii and dark matter masses
    mhi = dm_mass_fraction(ri, fb, c)

    # Remap initial mass distributions on the average orbital radii
    rav, _ = y(ri, A, w)
    mhi_av = dm_mass_fraction(rav, fb, c)
    g = mhi_av * ri / (1 - fb)

    # Find the roots of the function f(r) = 0
    rf = np.zeros(len(ri))

    for i in range(len(ri)):
        rad_f = find_root(ri[i], fb, rb, mhi_av[i], g[i], A, w)
        rf[i] = rad_f

    return rf, mhi
