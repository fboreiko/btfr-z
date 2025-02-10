import numpy as np
from contra_unvect import do_contra
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set the font to Computer Modern (LaTeX default) and enable LaTeX rendering
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern']
rcParams['text.usetex'] = True

# Define radii and other parameters
ri = np.logspace(-7, 0, 20)

# Combinations of concentration, baryonic fraction, and scale radius
concs = np.array([1])
fbs = np.array([0.0005]) 
rbs = np.array([0.0012]) 

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

            rf = (rf / ri)**(-1.8) * ri
            
            # Plot the result
            plt.plot(ri, rf, label=f'(c = {c:.4f}, fb = {fb:.4f}, rb = {rb:.4f})', linewidth=2)
    
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


    