import matplotlib.pyplot as plt
import numpy as np

xlabel = r'$\log_{10} \left( \frac{V_{\rm max}}{{\rm km \, s^{-1}}} \right)$'
ylabel = r'$\log_{10} \left( \frac{M_{\rm bar}}{{\rm \, M_{\odot}}} \right)$'


def btfr_plot(xmock, ymock, xmockerr, ymockerr, xobs, yobs, xobserr, yobserr, plotname):

    fig, ax = plt.subplots()
    ax.errorbar(xmock, ymock, xerr=xmockerr, yerr=ymockerr, fmt='o',
            color='blue', markersize=3.5, capsize=2, elinewidth=0.5, 
            capthick=0.5, ecolor='black', label='Mock data')
    ax.errorbar(xobs, yobs, xerr=xobserr, yerr=yobserr, fmt='o',
            color='red', markersize=3.5, capsize=2, elinewidth=0.5, 
            capthick=0.5, ecolor='black', label='Obs data')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('Tully-Fisher Relation')
    ax.set_xlim([1, 3])
    ax.set_ylim([7, 12])
    ax.legend()

    plt.savefig(plotname, dpi=300)


def explore_hist(data, nfw_data, bar_data, mean, std, obs_mean, obs_err, galaxy_mass, galaxy_ll, galaxy_tll, output_filename):

    # Create a range of x values for the Gaussian curve
    x = np.linspace(mean - 5*std, mean + 5*std, 200)
    # Compute the Gaussian curve values
    y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std)**2)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # Get the histogram for mock velocity data (without scaling)
    n, bins, patches = ax.hist(data, bins=100, color='red', alpha=0.5, density=False, label='Mock Velocity')

    # Get the histograms for NFW and Bar velocities (without scaling)
    nfw_n, nfw_bins = np.histogram(nfw_data, bins=100, density=False)
    bar_n, bar_bins = np.histogram(bar_data, bins=100, density=False)

    # Calculate scaling factors to match the peak of the mock velocity data
    nfw_scale_factor = max(n) / max(nfw_n)
    bar_scale_factor = max(n) / max(bar_n)

    # Plot the scaled NFW velocity histogram
    ax.hist(nfw_data, bins=100, color='blue', alpha=0.4, weights=np.ones_like(nfw_data) * nfw_scale_factor, label='NFW velocity')

    # Plot the scaled Bar velocity histogram
    ax.hist(bar_data, bins=100, color='green', alpha=0.4, weights=np.ones_like(bar_data) * bar_scale_factor, label='Bar velocity')

    # Plot the normalized Gaussian curve
    ax.plot(x, y, color='red', alpha=0.7, label='Gaussian Fit to Mock Data')

    # Add mean and standard deviation lines for the data
    ax.axvline(mean, color='red', linestyle='--', label='Mean')
    ax.axvline(mean + std, color='red', linestyle=':', label='1 sigma')
    ax.axvline(mean - std, color='red', linestyle=':')

    # Add vertical lines for the observed mean and observed error
    ax.axvline(obs_mean, color='orange', linestyle='--', label='Observed Mean')
    ax.axvline(obs_mean + obs_err, color='orange', linestyle=':', label='Observed +1 sigma')
    ax.axvline(obs_mean - obs_err, color='orange', linestyle=':')

    #ax.plot(x, y, color='blue', alpha=0.7, label='Gaussian of Obs Data')

    # Set labels and title
    ax.set_xlabel(xlabel)
    #ax.set_xlim(np.min(data), np.max(data))  # Adjust x-axis limit to fit data
    ax.set_ylabel('Normalized Frequency')
    ax.set_title(f'Galaxy of mass {galaxy_mass:.4g}, Log Likelihood: {galaxy_ll:.3g}, New Log Likelihood: {galaxy_tll:.3g}')
    ax.legend()

    # Save the plot
    plt.savefig(output_filename)
    plt.close()  # Close the figure to avoid display issues in some environments


def scatter_plot(x, y, xlabel, ylabel, title, output_filename):

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    ax.scatter(x, y, s=0.01, color='red', alpha=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.set_ylim([0, 500])

    plt.savefig(output_filename)
    plt.close()  # Close the figure to avoid display issues in some