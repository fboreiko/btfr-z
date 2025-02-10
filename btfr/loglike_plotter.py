import numpy as np
from scipy.interpolate import RegularGridInterpolator
import emcee
import corner
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set font to Computer Modern (LaTeX default) and enable LaTeX rendering
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern']
rcParams['text.usetex'] = True

# Load the 3D array of log likelihood values
file_path = '/Users/fedorboreiko/Documents/Oxford/btfr_z/btfr/likelihood_grid_50am_1000stellar_alphaproxy_-0.4_-0.4_scatter_0.1_0.1_x_0.0_0.9_nu_-1.0_-1.0.npy'
likelihood_grid = np.load(file_path)

x_range = np.linspace(0.0, 0.9, 100)

lines_range = np.linspace(0.0, 0.9, 10)

plt.plot(x_range, likelihood_grid[0, 0, :, 0])
plt.vlines(lines_range, ymin=0, ymax=200, color='red', linestyle='--')
plt.xlabel(r'$x$')
plt.ylabel(r'$\log \mathcal{L}$')

plt.savefig('loglike_plot.png', dpi=300)



