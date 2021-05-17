import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py

def load(filename):
    return np.array(nib.load(filename).dataobj).transpose(2,1,0)

n_runs = 10

dirname = './results/mess_P1/'
dirname_dess = './results/dess_P1/'
S = 126 # Slice to visualize
image_dir = './images/figure4/'

os.makedirs(image_dir, exist_ok=True)

# Load data for W/F noise analysis
ref_dess = load(dirname_dess + 'w_run0.nii.gz')
ref_mess = load(dirname + 'w_run0.nii.gz')
ref_mess_f = load(dirname + 'f_run0.nii.gz')

reps_dess = np.zeros((n_runs,) + ref_dess.shape, dtype=ref_dess.dtype)
reps_mess = np.zeros((n_runs,) + ref_mess.shape, dtype=ref_mess.dtype)
reps_mess_f = np.zeros((n_runs,) + ref_mess_f.shape, dtype=ref_mess_f.dtype)

for run in range(n_runs):
    print('Loading', run+1)
    reps_dess[run] = load(dirname_dess + f'w_run{run+1}.nii.gz')
    reps_mess[run] = load(dirname + f'w_run{run+1}.nii.gz')
    reps_mess_f[run] = load(dirname + f'f_run{run+1}.nii.gz')

# Calculate noise levels
sd_dess = reps_dess.std(axis=0)
sd_mess = reps_mess.std(axis=0)
sd_mess_f = reps_mess_f.std(axis=0)

mean_sd_dess = sd_dess.mean()

# Write images
plt.imsave(image_dir + 'dess_w_noise.png', sd_dess[S]/mean_sd_dess,vmin=0,vmax=2,cmap=plt.get_cmap('seismic'))
plt.imsave(image_dir + 'mess_w_noise.png', sd_mess[S]/mean_sd_dess,vmin=0,vmax=2,cmap=plt.get_cmap('seismic'))
plt.imsave(image_dir + 'mess_f_noise.png', sd_mess_f[S]/mean_sd_dess,vmin=0,vmax=2,cmap=plt.get_cmap('seismic'))

snr_dess = abs(reps_dess.mean(axis=0))/sd_dess
snr_mess = abs(reps_mess.mean(axis=0))/sd_mess
snr_mess_f = abs(reps_mess_f.mean(axis=0))/sd_mess_f

plt.imsave(image_dir + 'dess_w_snr.png', snr_dess[S], vmin=0,vmax=80,cmap=plt.get_cmap('seismic'))
plt.imsave(image_dir + 'mess_w_snr.png', snr_mess[S], vmin=0,vmax=80,cmap=plt.get_cmap('seismic'))
plt.imsave(image_dir + 'mess_f_snr.png', snr_mess_f[S], vmin=0,vmax=80,cmap=plt.get_cmap('seismic'))

plt.imsave(image_dir + 'cmap_seismic.png', np.linspace(1,0,256)[:,None], cmap=plt.get_cmap('seismic'))


#%%
# Load data for T2 analysis
ref_dess_t2 = 1 / (load(dirname_dess + 'r2_run0.nii.gz') + 1e-12)
ref_mess_t2 = 1 / (load(dirname + 'r2_run0.nii.gz') + 1e-12)

reps_dess_t2 = np.zeros((n_runs,) + ref_dess_t2.shape, dtype=ref_dess_t2.dtype)
reps_mess_t2 = np.zeros((n_runs,) + ref_mess_t2.shape, dtype=ref_dess_t2.dtype)

for run in range(n_runs):
    print('Loading', run+1)
    reps_dess_t2[run] = 1 / (load(dirname_dess + f'r2_run{run+1}.nii.gz') + 1e-12)
    reps_mess_t2[run] = 1 / (load(dirname + f'r2_run{run+1}.nii.gz') + 1e-12)

with h5py.File('./data/P1_seg.h5', 'r') as f:
    mask_dess = np.array(f['dess_cartilage'])
    mask_mess = np.array(f['mess_cartilage'])

# Calculate T2 distributions
steps = 20
dess_mean = np.zeros(steps)
dess_q5 = np.zeros(steps)
dess_q95 = np.zeros(steps)
mess_mean = np.zeros(steps)
mess_q5 = np.zeros(steps)
mess_q95 = np.zeros(steps)
bounds = np.linspace(0,0.1,steps+1)
b = (bounds[:-1] + bounds[1:])/2

dess_mask = abs(ref_dess)>0.2
mess_mask = (abs(ref_mess)>abs(ref_mess_f)) & ((abs(ref_mess)+abs(ref_mess_f))>0.2)
mess_mask_f = (abs(ref_mess_f)>abs(ref_mess)) & ((abs(ref_mess)+abs(ref_mess_f))>0.2)

reps_dess_masked = np.clip(reps_dess_t2[:,dess_mask],0,0.25)
reps_mess_masked = np.clip(reps_mess_t2[:,mess_mask],0,0.25)

# For every bin in the reference images, find the distribution (5%, mean, 95%) of the T2 values in the pseudo-replicates
for i in range(len(bounds)-1):
    x = bounds[i]
    x2 = bounds[i+1]

    r = ref_dess_t2[dess_mask]
    m = (r>x) & (r<=x2)
    if m.sum() > 0:
        dess_q5[i] = np.percentile(reps_dess_masked[:,m],5)
        dess_q95[i] = np.percentile(reps_dess_masked[:,m],95)
        dess_mean[i] = reps_dess_masked[:,m].mean()
    else:
        dess_q5[i] = np.nan
        dess_q95[i] = np.nan
        dess_mean[i] = np.nan

    r = ref_mess_t2[mess_mask]
    m = (r>x) & (r<=x2)
    if m.sum() > 0:
        mess_q5[i] = np.percentile(reps_mess_masked[:,m],5)
        mess_q95[i] = np.percentile(reps_mess_masked[:,m],95)
        mess_mean[i] = reps_mess_masked[:,m].mean()
    else:
        mess_q5[i] = np.nan
        mess_q95[i] = np.nan
        mess_mean[i] = np.nan

# Plot and save T2 figure    
plt.figure()
plt.plot(b*1000,b*1000,'k--')
plt.plot(b*1000,dess_q5*1000, 'r--')
plt.plot(b*1000,dess_mean*1000, 'r')
plt.plot(b*1000,dess_q95*1000, 'r--')

plt.plot(b*1000,mess_q5*1000, 'b--')
plt.plot(b*1000,mess_mean*1000, 'b')
plt.plot(b*1000,mess_q95*1000, 'b--')

plt.title('$T_2$ (water-dominant voxels)')
plt.legend(['Reference', 'DESS 5%', 'DESS mean', 'DESS 95%', 'MESS 5%', 'MESS mean', 'MESS 95%'])
plt.xlabel('Reference $T_2$ (ms)')
plt.ylabel('Pseudo-replicate $T_2$ (ms)')

plt.savefig(image_dir + 't2.pdf',dpi=600)

