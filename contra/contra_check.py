import numpy as np
from contra_unvect import do_contra
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.signal import find_peaks

# Set the font to Computer Modern (LaTeX default) and enable LaTeX rendering
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern']
rcParams['text.usetex'] = True

# Define radii and other parameters
ri = np.logspace(-9, 1, 30)

# Combinations of concentration, baryonic fraction, and scale radius
concs = np.array([6.250551925273973])
fbs = np.array([0.7891196485223975]) 
rbs = np.array([0.0013257113655901094]) 
nu = -3.0 #0.28

#how to find the upper bound? find the local maxima of this plot here. First, take the maxima corresponding to the 
#lowest ri value. 
#Check if directly computed rf value from this ri value less or greater than the target rf value. If it is larger, 
#set this ri value as the upper bound. If it is smaller, set the next local maxima as the upper bound.

# Prepare the plot
plt.figure(figsize=(8, 6), dpi=300)

# Loop through each combination
for i in range(len(concs)):
    for j in range(len(fbs)):
        for k in range(len(rbs)):
            c = concs[i]
            fb = fbs[j]
            rb = rbs[k]
            
            # Compute the initial and final dark matter mass fractions
            rf, mhi = do_contra(ri, c, fb, rb, A=1.6, w=0.8)

            rf = (rf / ri)**(nu) * ri

            # find local maxima of the plot
            mask = rf > 10**(-4.1755102040816325)
            if np.any(mask):
                upper_bound = ri[np.argmax(mask)]

            # Find local maxima in rf_probe_array
            peaks, _ = find_peaks(rf)  

            print(10**(-4.1755102040816325) < rf[peaks[0]])     
            
            # Plot the result
            plt.plot(ri, rf, label=f'(c = {c:.4f}, fb = {fb:.4f}, rb = {rb:.4f})', linewidth=2)
            plt.scatter(ri[peaks], rf[peaks], color='r', s=50, label='local maxima')
            plt.hlines(10**(-4.1755102040816325), 10**(-8.5), 10, colors='k', linestyles='dashed', linewidth=1)
            plt.vlines(10**(-7), 1e-7, 10, colors='k', linestyles='dashed', linewidth=1)
    
# Set the x-axis to logarithmic scale
plt.xscale('log')
plt.yscale('log')

# Add labels, grid, and legend
plt.xlabel('ri')
plt.ylabel('rf')
plt.title('rf/ri vs ri, A = 1,6, w = 0.8')
#plt.legend(fontsize=4)

# Show the plot
plt.savefig("/Users/fedorboreiko/Documents/Oxford/btfr_z/plots/contra_check.png", dpi=300)


    