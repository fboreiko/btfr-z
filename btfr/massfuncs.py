from colossus.cosmology import cosmology
from colossus.lss import mass_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

os.chdir("/Users/fedorboreiko/Documents/Oxford/glamdring")

def GSMF(M_star, alp_1, alp_2, phi_1, phi_2, char_M_star):
    delta_M = M_star - char_M_star
    GSMF = np.log(10) * np.exp(-1 * np.power(10, delta_M)) * np.power(10, delta_M) * \
    ((phi_1 * np.power(10, delta_M * alp_1) + phi_2 * np.power(10, delta_M*alp_2)))

    return GSMF

def get_GSMF_Adams(redshift=0, plotting=True):
    # obtains SMF from N.Adams paper for a given redshift 

    log_char_m_star = 10.88
    alpha_1 = -0.74
    log_phi_1 = -2.71
    alpha_2 = -1.62
    log_phi_2 = -3.54

    phi_1 = 10**log_phi_1 * 2.5
    phi_2 = 10**log_phi_2 * 2.5

    # Calculate the GSMF
    log_m_stars = np.linspace(6.5, 12, 20000)

    phi = GSMF(log_m_stars, alpha_1, alpha_2, phi_1, phi_2, log_char_m_star)

    # Plot the GSMF
    if plotting:

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(log_m_stars, phi, label=f'SExtractor+IRAC', color='red')
        ax.set_yscale('log')
        ax.set_xlabel('log(M/M_sun)')
        ax.set_ylabel('Phi dex^-1 Mpc^-3')
        ax.set_title('Galaxy Stellar Mass Function (GSMF)')
        ax.legend()

        plt.savefig(f'plots/GSMF_z_{redshift}.png')
        plt.show()

    m_stars = 10**log_m_stars

    return log_m_stars, phi

def get_GSMF_Bernardi(plotting=True):

    # Define the column names for Bernardi data
    column_names = ['Mr', 'phi_full', 'err_full', 
                    'phi_type1', 'err_type1',
                    'phi_type2', 'err_type2',
                    'phi_type3', 'err_type3',
                    'phi_type4', 'err_type4']

    # Read the data into a pandas DataFrame
    data = pd.read_csv('Tabular_data/MsF_cmodel.dat', delim_whitespace=True, names=column_names)

    phi_data = 10**data['phi_full']

    stellar_masses = data['Mr']

    # Plot the GSMF
    if plotting:

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plotting the full sample
        ax.plot(stellar_masses, data['phi_full'], 'o-')

        ax.set_xlabel('log(Mr [mag])')
        ax.set_ylabel('log(Phi) dex^-1 Mpc^-3')
        #ax.set_yscale('log')
        ax.set_title('Galaxy Stellar Mass Function (GSMF)')
        #ax.invert_xaxis()
        ax.legend()

        plt.savefig(f'plots/GSMF_Bernardi_z_0.png')
        plt.show()

    return np.array(stellar_masses), np.array(phi_data)

def get_GSMF_GAMA(mode='bolometric', plotting=False):

    log_rel_m_stars_bol = 10.78
    log_rel_m_stars_bol_err = 0.01
    log_rel_m_stars_opt = 10.76
    log_rel_m_stars_opt_err = 0.01

    alpha_1_bol = -0.62
    alpha_1_bol_err = 0.03
    alpha_1_opt = -0.55
    alpha_1_opt_err = 0.04

    normalisation_1_bol = 2.93 * 10**(-3)
    normalisation_1_bol_err = 0.4 * 10**(-3)
    normalisation_1_opt = 3.10 * 10**(-3)
    normalisation_1_opt_err = 0.42 * 10**(-3)

    alpha_2_bol = -1.50
    alpha_2_bol_err = 0.01
    alpha_2_opt = -1.49
    alpha_2_opt_err = 0.02

    normalisation_2_bol = 0.63 * 10**(-3)
    normalisation_2_bol_err = 0.1 * 10**(-3)
    normalisation_2_opt = 0.75 * 10**(-3)
    normalisation_2_opt_err = 0.12 * 10**(-3)

    # Calculate the GSMF
    rel_stellar_masses = np.linspace(6.5, 12, 20000)
    stellar_masses = 10**rel_stellar_masses

    if mode == 'bolometric':

        GSMF_data = GSMF(rel_stellar_masses, alpha_1_bol, alpha_2_bol, normalisation_1_bol, normalisation_2_bol, log_rel_m_stars_bol)

    elif mode == 'optical':

        GSMF_data = GSMF(rel_stellar_masses, alpha_1_opt, alpha_2_opt, normalisation_1_opt, normalisation_2_opt, log_rel_m_stars_opt)

    else:

        raise ValueError("Invalid mode")
    
    # Plot the GSMF
    if plotting:

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(rel_stellar_masses, GSMF_data, label=f'GAMA, {mode}, z=0', color='purple')
        ax.set_yscale('log')
        ax.set_xlabel('log(M_h/M_sun)')
        ax.set_ylabel('Phi dex^-1 Mpc^-3')
        ax.set_title('GSMF Comparison')
        ax.legend()

        plt.savefig(f'plots/GSMF_GAMA_{mode}.png')
    
    return rel_stellar_masses, GSMF_data


def get_GSMF_ELPETRO (plotting=False):

    ELPETRO_SMF = np.load("/Users/fedorboreiko/Documents/Oxford/btfr_z/SMF_NSA_ELPETRO_Z015_h07.npy")

    log_stellar_masses = ELPETRO_SMF["proxy"]
    GSMF_data = ELPETRO_SMF["phi"] * 0.7**3
    GSMF_err = ELPETRO_SMF["err"] * 0.7**3

    if plotting:

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.errorbar(ELPETRO_SMF["proxy"], ELPETRO_SMF["phi"]* 0.7**3, ELPETRO_SMF["err"]* 0.7**3, label='SMF_NSA_ELPETRO_Z015_h07', color='cyan')
        ax.set_yscale('log')
        ax.set_xlabel('log(M_h/M_sun)')
        ax.set_ylabel('Phi dex^-1 Mpc^-3')
        ax.set_title('GSMF Comparison')
        ax.legend()

        plt.savefig(f'plots/GSMF_ELPETRO.png')

    return log_stellar_masses, GSMF_data, GSMF_err


def get_HMF(redshift, plotting=True, start_value=10, end_value=16.3):
    # obtains HMF from colossus

    cosmology.setCosmology('WMAP9')

    # Define the halo mass range
    rel_halo_mass = np.linspace(start_value, end_value, 20000)  # change the start value to check robustness
    halo_masses = 10**rel_halo_mass * 0.7
    HMF_data = mass_function.massFunction(halo_masses, redshift, mdef = 'fof', model = 'watson13', q_out= 'dndlnM') * np.log(10) / 0.7**3

    # Plot the HMF
    if plotting:

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(rel_halo_mass, np.log10(HMF_data), label='HMF, watson13', color='red')
        ax.set_xlabel('log(M_h/M_sun)')
        ax.set_ylabel('Phi dex^-1 Mpc^-3')
        ax.set_title('Halo Mass Function (HMF)')
        ax.legend()

        plt.savefig(f'plots/HMF_z_{redshift}.png')
        plt.show()

    return rel_halo_mass, HMF_data