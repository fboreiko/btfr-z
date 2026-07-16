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
from .massfuncs import (
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
    scatter_plot,
    plot_SHMR_with_contours,
    plot_SHMR_with_contours_quantile,
    mstellar_hists_comparison,
    delta_loglike_correlation_plot,
    create_btfr_panel_plot
)

# Halo selection utilities
from .halo_selection_utils import (
    get_x_cutoff_fit
)

# Interpolation utilities
from .interpolation_utils import (
    jax_contra_interpolator
)

# Likelihood utilities
from .likelihood_utils import (
    get_loglike,
    get_loglike_vect,
    get_loglike_split,
    update_progress
)

# Rotation curve utilities
from .rotation_curve_utils import (
    nfw_circular_velocity,
    nfw_circular_velocity_from_mhi,
    nfw_circular_velocity_contra
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
    'scatter_plot',
    'plot_SHMR_with_contours',
    'plot_SHMR_with_contours_quantile',
    'mstellar_hists_comparison',
    'delta_loglike_correlation_plot',
    'create_btfr_panel_plot',
    
    # Halo selection
    'get_x_cutoff_fit',
    
    # Interpolation
    'jax_contra_interpolator',
    
    # Likelihood
    'get_loglike',
    'get_loglike_vect',
    'get_loglike_split',
    'update_progress',
    
    # Rotation curves
    'nfw_circular_velocity',
    'nfw_circular_velocity_from_mhi',
    'nfw_circular_velocity_contra'
]
