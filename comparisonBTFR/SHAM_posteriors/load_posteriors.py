"""
SHAM Posteriors Loader
======================

This script loads posterior samples from Subhalo Abundance Matching (SHAM) fits.

Data Structure
--------------
The posteriors are organized by survey/catalog:
    - NYU:        NYU Value-Added Galaxy Catalog
    - NSA_SERSIC: NASA-Sloan Atlas with Sersic photometry
    - NSA_ELPETRO: NASA-Sloan Atlas with Elliptical Petrosian photometry
    - matched:    Matched catalog (HI-optical)

Each catalog has fits using different galaxy properties:
    - ABSMAG / *_ABSMAG: Absolute r-band magnitude (Mr)
    - LOG_MASS / *_LOG_MASS: Stellar mass (log10(M*/Msun))
    - LOG_MB / LOG_MH: Baryonic/HI mass (for matched catalog)

The suffix (N1-N4, K1-K4, etc.) indicates the luminosity/mass bin:
    - 1 = brightest/most massive bin
    - 4 = faintest/least massive bin

Files
-----
- surrogate_samples.npy: MCMC posterior samples with fields:
    - 'alpha': Power-law slope of the SHAM relation
    - 'scatter': Scatter in the abundance matching (sigma_AM)

- CF_*.p: Pickled dictionary with correlation function data, including:
    - 'cut_range': The luminosity/mass range for this bin
    - 'attr': The galaxy property used (e.g., 'ABSMAG', 'LOG_MASS')

Usage
-----
>>> from load_posteriors import SHAMPosteriors
>>> loader = SHAMPosteriors()
>>> samples, metadata = loader.load('NYU', 'ABSMAG', 'N', 1)
>>> print(samples['alpha'].shape)  # MCMC chain samples
>>> print(metadata['cut_range'])   # Magnitude/mass bin edges
"""

import os
import numpy as np
import joblib
from pathlib import Path


class SHAMPosteriors:
    """
    Loader class for SHAM posterior samples.

    Parameters
    ----------
    base_path : str, optional
        Path to the SHAM_posteriors directory.
        Default: directory containing this script.

    Examples
    --------
    >>> loader = SHAMPosteriors()

    # Load NYU VAGC absolute magnitude fit, bin 1 (brightest)
    >>> samples, meta = loader.load('NYU', 'ABSMAG', 'N', 1)

    # Load NSA Elpetro stellar mass fit, bin 3
    >>> samples, meta = loader.load('NSA_ELPETRO', 'ELPETRO_LOG_MASS', 'K', 3)

    # Load all bins for a given catalog/property
    >>> all_samples = loader.load_all_bins('NYU', 'ABSMAG', 'N')

    # List available catalogs
    >>> print(loader.list_catalogs())
    """

    # Parameter names in the samples
    PARAM_NAMES = ['alpha', 'scatter']

    # Mapping of catalog to available attributes
    CATALOG_ATTRS = {
        'NYU': {
            'N': 'ABSMAG',      # Mr absolute magnitude
            'K': 'LOG_MASS',    # Stellar mass
        },
        'NSA_SERSIC': {
            'N': 'SERSIC_ABSMAG',
            'K': 'SERSIC_LOG_MASS',
        },
        'NSA_ELPETRO': {
            'N': 'ELPETRO_ABSMAG',
            'K': 'ELPETRO_LOG_MASS',
        },
        'matched': {
            'N': 'ELPETRO_ABSMAG',
            'K': 'ELPETRO_LOG_MASS',
            'B': 'LOG_MB',      # Baryonic mass
            'H': 'LOG_MH',      # HI mass
        },
    }

    def __init__(self, base_path=None):
        if base_path is None:
            base_path = Path(__file__).parent
        self.base_path = Path(base_path)

    def load(self, catalog, attr, sub_id, bin_index):
        """
        Load posterior samples for a specific catalog/property/bin.

        Parameters
        ----------
        catalog : str
            Catalog name: 'NYU', 'NSA_SERSIC', 'NSA_ELPETRO', or 'matched'
        attr : str
            Galaxy property: 'ABSMAG', 'LOG_MASS', 'SERSIC_ABSMAG', etc.
        sub_id : str
            Subset identifier: 'N' (magnitude), 'K' (mass), 'B'/'H' (matched)
        bin_index : int
            Bin number (1-4), where 1 is brightest/most massive

        Returns
        -------
        samples : numpy.ndarray
            Structured array with 'alpha' and 'scatter' fields
        metadata : dict
            Dictionary containing 'cut_range', 'attr', and other info
        """
        # Build paths
        samples_dir = self.base_path / catalog / f"{attr}_{sub_id}{bin_index}"
        samples_path = samples_dir / "surrogate_samples.npy"
        cf_path = self.base_path / catalog / f"CF_{attr}_{sub_id}{bin_index}.p"

        # Load samples
        if not samples_path.exists():
            raise FileNotFoundError(f"Samples file not found: {samples_path}")
        samples = np.load(samples_path)

        # Load metadata
        metadata = {}
        if cf_path.exists():
            metadata = joblib.load(cf_path)
        else:
            print(f"Warning: CF metadata file not found: {cf_path}")

        return samples, metadata

    def load_all_bins(self, catalog, attr, sub_id, n_bins=4):
        """
        Load all bins for a given catalog/property combination.

        Parameters
        ----------
        catalog : str
            Catalog name
        attr : str
            Galaxy property
        sub_id : str
            Subset identifier
        n_bins : int, optional
            Number of bins (default: 4)

        Returns
        -------
        list of tuples
            List of (samples, metadata) for each bin
        """
        results = []
        for i in range(1, n_bins + 1):
            try:
                samples, meta = self.load(catalog, attr, sub_id, i)
                results.append((samples, meta))
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                results.append((None, None))
        return results

    def get_samples_array(self, catalog, attr, sub_id, bin_index, params=None):
        """
        Get samples as a 2D numpy array (n_samples, n_params).

        Parameters
        ----------
        catalog, attr, sub_id, bin_index : see load()
        params : list of str, optional
            Parameter names to extract. Default: ['alpha', 'scatter']

        Returns
        -------
        numpy.ndarray
            Shape (n_samples, n_params)
        """
        if params is None:
            params = self.PARAM_NAMES

        samples, _ = self.load(catalog, attr, sub_id, bin_index)
        return np.column_stack([samples[p] for p in params])

    def list_catalogs(self):
        """List available catalog directories."""
        return [d.name for d in self.base_path.iterdir()
                if d.is_dir() and d.name in self.CATALOG_ATTRS]

    def list_available(self, catalog):
        """List available property/bin combinations for a catalog."""
        catalog_path = self.base_path / catalog
        if not catalog_path.exists():
            return []

        available = []
        for subdir in catalog_path.iterdir():
            if subdir.is_dir() and (subdir / "surrogate_samples.npy").exists():
                available.append(subdir.name)
        return sorted(available)


def get_legend_label(cut_range, attr):
    """
    Format a legend label for a bin's magnitude/mass range.

    Parameters
    ----------
    cut_range : list
        [min, max] of the property range
    attr : str
        Property name (e.g., 'ABSMAG', 'LOG_MASS')

    Returns
    -------
    str
        LaTeX-formatted label
    """
    lo, hi = cut_range

    # Handle edge cases
    if lo < -25:
        lo = -30.0
    if hi > 15.0:
        hi = 15.0

    # Format property name
    if 'ABSMAG' in attr:
        prop = 'M_r'
    elif 'LOG_MASS' in attr:
        prop = r'M_{* / \odot}'
    elif 'LOG_MH' in attr:
        prop = r'M_{\mathrm{HI}}'
    elif 'LOG_MB' in attr:
        prop = r'M_{\mathrm{bar}}'
    else:
        prop = attr

    return rf"${lo:.1f} < {prop} < {hi:.1f}$"


# Example usage
if __name__ == "__main__":
    loader = SHAMPosteriors()

    print("Available catalogs:", loader.list_catalogs())
    print()

    # Load an example
    for catalog in ['NYU', 'NSA_ELPETRO', 'NSA_SERSIC']:
        print(f"\n{catalog}:")
        print("  Available:", loader.list_available(catalog))

        # Try loading first available
        available = loader.list_available(catalog)
        if available:
            # Parse the first available entry
            name = available[0]
            # Extract attr, sub_id, bin from name like "ABSMAG_N1"
            parts = name.rsplit('_', 1)
            attr = parts[0]
            sub_id = parts[1][0]
            bin_idx = int(parts[1][1])

            samples, meta = loader.load(catalog, attr, sub_id, bin_idx)
            print(f"  Loaded {name}:")
            print(f"    - Samples shape: {samples.shape}")
            print(f"    - Parameters: {samples.dtype.names}")
            print(f"    - Alpha: mean={samples['alpha'].mean():.3f}, "
                  f"std={samples['alpha'].std():.3f}")
            print(f"    - Scatter: mean={samples['scatter'].mean():.3f}, "
                  f"std={samples['scatter'].std():.3f}")
            if 'cut_range' in meta:
                print(f"    - Cut range: {meta['cut_range']}")
