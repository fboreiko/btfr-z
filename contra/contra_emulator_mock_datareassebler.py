import numpy as np
import glob

# Step 1: Use glob to get all file names that match the pattern
c_file_list = sorted(glob.glob("contra_emulator_da/c_rank_*_realization_*.npy"))
cs = [np.load(file) for file in c_file_list]
c = np.stack(cs, axis=0)
np.save("contra_emulator_data/c.npy", c)
del c_file_list, cs, c

print('c saved')

fb_file_list = sorted(glob.glob("contra_emulator/fb_rank_*_realization_*.npy"))
fbs = [np.load(file) for file in fb_file_list]
fb = np.stack(fbs, axis=0)
np.save("contra_emulator_data/fb.npy", fb)
del fb_file_list, fbs, fb

print('fb saved')

rb_file_list = sorted(glob.glob("contra_emulator/rb_rank_*_realization_*.npy"))
rbs = [np.load(file) for file in rb_file_list]
rb = np.stack(rbs, axis=0)
np.save("contra_emulator_data/rb.npy", rb)
del rb_file_list, rbs, rb

print('rb saved')

ri_file_list = sorted(glob.glob("contra_emulator/ri_rank_*_realization_*.npy"))
ris = [np.load(file) for file in ri_file_list]
ri = np.stack(ris, axis=0)
np.save("contra_emulator_data/ri.npy", ri)

print("ri saved")
import numpy as np
import glob

# Step 1: Use glob to get all file names that match the pattern
c_file_list = sorted(glob.glob("contra_emulator/c_rank_*_realization_*.npy"))
cs = [np.load(file) for file in c_file_list]
c = np.stack(cs, axis=0)
np.save("contra_emulator_data/c.npy", c)
del c_file_list, cs, c

print('c saved')

fb_file_list = sorted(glob.glob("contra_emulator/fb_rank_*_realization_*.npy"))
fbs = [np.load(file) for file in fb_file_list]
fb = np.stack(fbs, axis=0)
np.save("contra_emulator_data/fb.npy", fb)
del fb_file_list, fbs, fb

print('fb saved')

rb_file_list = sorted(glob.glob("contra_emulator/rb_rank_*_realization_*.npy"))
rbs = [np.load(file) for file in rb_file_list]
rb = np.stack(rbs, axis=0)
np.save("contra_emulator_data/rb.npy", rb)
del rb_file_list, rbs, rb

print('rb saved')

Mri_file_list = sorted(glob.glob("contra_emulator/Mri_rank_*_realization_*.npy"))
Mris = [np.load(file) for file in Mri_file_list]
Mri = np.stack(Mris, axis=0)
np.save("contra_emulator_data/Mri.npy", Mri)
del Mri_file_list, Mris, Mri

print('Mri saved')

rf_file_list = sorted(glob.glob("contra_emulator/rf_rank_*_realization_*.npy"))
rfs = [np.load(file) for file in rf_file_list]
rf = np.stack(rfs, axis=0)
np.save("contra_emulator_data/rf.npy", rf)
del rf_file_list, rfs, rf

print("rf saved")

ri_file_list = sorted(glob.glob("contra_emulator/ri_rank_*_realization_*.npy"))
ris = [np.load(file) for file in ri_file_list]
ri = np.stack(ris, axis=0)
np.save("contra_emulator_data/ri.npy", ri)

print("ri saved")