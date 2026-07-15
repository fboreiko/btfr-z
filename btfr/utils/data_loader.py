import pandas as pd
import numpy as np
import pickle
import sys
from .massfuncs import get_GSMF_ELPETRO


def load_csv_data(file_path, filter_column=None, filter_values=None):
    """
    Load a CSV file and optionally filter rows based on a column's values.

    Args:
        file_path (str): Path to the CSV file.
        filter_column (str, optional): Column name to filter rows.
        filter_values (list, optional): Values to filter rows by.

    Returns:
        pd.DataFrame: Loaded and optionally filtered DataFrame.
    """
    data = pd.read_csv(file_path)
    if filter_column and filter_values:
        data = data[data[filter_column].isin(filter_values)]
    return data


def load_gsmf_and_halos():
    """
    Load GSMF data and halo catalog.

    Returns:
        tuple: Contains the following:
            - stellar_mass_bins (np.ndarray): Logarithmic stellar mass bins.
            - stellar_mass_function (np.ndarray): Stellar mass function data.
            - halo_catalog (np.ndarray): Halo catalog data.
    """
    stellar_mass_bins, stellar_mass_function, _ = get_GSMF_ELPETRO(plotting=False)
    halo_catalog = np.load("/Users/fedorboreiko/Documents/Oxford/Personal_codes/Codebase/halos_z_0p00.npy")
    return stellar_mass_bins, stellar_mass_function, halo_catalog


def load_data():
    """
    Load and prepare the data for the SPARC sample.

    Returns:
        tuple: Contains the following:
            - sparc_galaxy_names (list): List of galaxy names in the SPARC sample.
            - bulge_luminosities (dict): Dictionary mapping galaxy names to bulge luminosities.
            - galaxy_properties (dict): Dictionary mapping galaxy names to their properties.
            - mass_model_data (pd.DataFrame): DataFrame containing mass model data.
            - sparc_btfr_data (pd.DataFrame): DataFrame containing SPARC BTFR data.
            - stellar_mass_bins (np.ndarray): Logarithmic stellar mass bins.
            - stellar_mass_function (np.ndarray): Stellar mass function data.
            - halo_catalog (np.ndarray): Halo catalog data.
    """
    sparc_btfr_table = load_csv_data('Tabular_data/sparc_btfr.csv')
    sparc_galaxy_names = sparc_btfr_table['Name'].tolist()

    bulge_luminosity_table = load_csv_data('Tabular_data/Bulge_lum_table.csv', 'Galaxy', sparc_galaxy_names)
    mass_model_table = load_csv_data('Tabular_data/Mass_models_table.csv', 'ID', sparc_galaxy_names)
    galaxy_sample_table = load_csv_data('Tabular_data/Gal_sample_table.csv', 'Galaxy', sparc_galaxy_names)

    bulge_luminosities = bulge_luminosity_table.set_index('Galaxy')['Lbul'].to_dict()
    galaxy_properties = galaxy_sample_table.set_index('Galaxy').to_dict('index')

    stellar_mass_bins, stellar_mass_function, halo_catalog = load_gsmf_and_halos()

    return (
        sparc_galaxy_names,
        bulge_luminosities,
        galaxy_properties,
        mass_model_table,
        sparc_btfr_table,
        stellar_mass_bins,
        stellar_mass_function,
        halo_catalog,
    )


def vectorize_mass_model_table(sparc_galaxy_names, mass_model_table):
    """
    Vectorize the mass model table for efficient processing.

    Args:
        sparc_galaxy_names (list): List of galaxy names in the SPARC sample.
        mass_model_table (pd.DataFrame): The original mass model table.

    Returns:
        dict: A dictionary containing the vectorized mass model data with keys:
              'R', 'Vgas', 'Vdisk', 'Vbul' - each containing 2D arrays of shape 
              (n_galaxies, max_entries) padded with NaNs where needed.
    """
    # Determine the maximum number of radial entries across all galaxies
    entries_per_galaxy = mass_model_table.groupby('ID').size()
    max_entries = entries_per_galaxy.max()
    
    # Initialize arrays to store the vectorized data
    columns = ['R', 'Vgas', 'Vdisk', 'Vbul']
    mass_models_vectorized = {}
    
    for column in columns:
        mass_models_vectorized[column] = np.full((len(sparc_galaxy_names), max_entries), np.nan)

    # Fill the vectorized arrays with actual data, padding with NaNs where needed
    for i, galaxy in enumerate(sparc_galaxy_names):
        galaxy_data = mass_model_table[mass_model_table['ID'] == galaxy]
        n_entries = len(galaxy_data)
        
        if n_entries > 0:
            for column in columns:
                mass_models_vectorized[column][i, :n_entries] = galaxy_data[column].values
    
    return mass_models_vectorized


def load_emulators():
    """
    Load the halo contraction emulators.

    Returns:
        tuple: Contains the following:
            - contra_grids (dict): Dictionary of emulators for halo contraction.
            - grid_axes (list): List of grid axes used for interpolation.
    """
    try:
        with open("/Users/fedorboreiko/Documents/Oxford/Personal_codes/Codebase/contra_emulators/grids_fullrange.pkl", "rb") as f:
            contra_grids = pickle.load(f)
    except FileNotFoundError:
        print("Error: Contra emulator grids were not found. Please run the grid-generation script first.")
        sys.exit(1)

    # Define grid axes matching the interpolation grid-generation stage.
    N_SAMPLES = 50  # must match the number of samples used in the grid generation
    log_c_grid = np.linspace(0, 3.9, N_SAMPLES)
    log_fb_grid = np.linspace(-3.6, -0.03, N_SAMPLES)
    log_rb_grid = np.linspace(-3, -1, N_SAMPLES)
    log_rf_grid = np.linspace(-4.8, 0.3, N_SAMPLES)
    grid_axes = [log_c_grid, log_fb_grid, log_rb_grid, log_rf_grid]

    return contra_grids, grid_axes
