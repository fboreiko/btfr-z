"""
Utils package for BTFR analysis.

This package contains utility functions for data loading, mass functions,
plotting, halo selection, interpolation, likelihood calculations, and 
rotation curve modeling.
"""

# Data loading utilities
from .data_loader import (
    load_csv_data,
    load_gsmf_and_halos,
    load_data,
    load_emulators,
    vectorize_mass_model_table
)

# Mass function utilities
from .mass_functions import (
    GSMF,
    get_GSMF_Adams,
    get_GSMF_Bernardi,
    get_GSMF_GAMA,
    get_GSMF_ELPETRO,
    get_HMF
)

# Plotting utilities
from .plotting_utils import (
    btfr_plot,
    vels_hist,
    plot_SHMR_with_contours_quantile,
    mstellar_hists_comparison,
    delta_loglike_correlation_plot,
)

# Halo selection utilities
from .halo_selection_utils import (
    get_x_cutoff_fit
)

# Interpolation utilities
from .emulator import (
    jax_contra_interpolator
)

# Make all functions available at package level
__all__ = [
    # Data loading
    'load_csv_data',
    'load_gsmf_and_halos', 
    'load_data',
    'load_emulators',
    'vectorize_mass_model_table',
    
    # Mass functions
    'GSMF',
    'get_GSMF_Adams',
    'get_GSMF_Bernardi',
    'get_GSMF_GAMA',
    'get_GSMF_ELPETRO',
    'get_HMF',
    
    # Plotting
    'btfr_plot',
    'vels_hist',
    'plot_SHMR_with_contours_quantile',
    'mstellar_hists_comparison',
    'delta_loglike_correlation_plot',
    
    # Halo selection
    'get_x_cutoff_fit',
    
    # Interpolation
    'jax_contra_interpolator',

]
