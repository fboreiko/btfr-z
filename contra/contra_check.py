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
ri = np.logspace(-3, 2, 30)

halos = np.load("/Users/fedorboreiko/Documents/Oxford/Personal_codes/Codebase/halos_z_0p00.npy")

random_index = 150000
halo = halos[random_index]

Mvir = halo['Mvir'] / 0.7
Rvir = halo['Rvir'] / 0.7
rs = halo['rs'] / 0.7

Reff = 5
Mbar = 1e10

rb = Reff / 1.67835
c = Rvir / rs
fb = Mbar / (Mvir + Mbar)
rb_uless = rb / Rvir
ri_uless = ri / Rvir
nu = -1.0

print(rb_uless)

exit()


#how to find the upper bound? find the local maxima of this plot here. First, take the maxima corresponding to the 
#lowest ri value. 
#Check if directly computed rf value from this ri value less or greater than the target rf value. If it is larger, 
#set this ri value as the upper bound. If it is smaller, set the next local maxima as the upper bound.

# Prepare the plot
plt.figure(figsize=(8, 6), dpi=300)

# Compute the initial and final dark matter mass fractions
rf, mhi = do_contra(ri_uless, c, fb, rb_uless, A=1.6, w=0.8)

print('Pre contraction radii:', ri_uless)
print('Post contraction radii:', rf)

exit()


"""Post contraction radii: [1.37411178e-05 2.04329033e-05 3.03807688e-05 4.51662026e-05
 6.71358878e-05 9.97689342e-05 1.48217321e-04 2.20098269e-04
 3.26650193e-04 4.84408151e-04 7.17612165e-04 1.06163393e-03
 1.56779562e-03 2.31006563e-03 3.39427829e-03 4.97082737e-03
 7.25245674e-03 1.05403244e-02 1.52649687e-02 2.20561827e-02
 3.18714841e-02 4.62472251e-02 6.78133528e-02 1.01368378e-01
 1.55938996e-01 2.47285926e-01 3.97607255e-01 6.35255659e-01
 1.00302981e+00 1.56780967e+00]"""




rf = (rf / ri_uless)**(nu) * ri_uless

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


    