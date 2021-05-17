import torch
import torch.fft
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import nibabel as nib

from mess_fitting import fit_mess

#%% pytorch init
# Select GPU if available, otherwise fall back to CPU computation (much slower!)
if not torch.cuda.is_available():
    device = torch.device('cpu')
    print('Device: CPU')
else:
    device = torch.device('cuda:0')
    print('Device: {}'.format(torch.cuda.get_device_name(device.index)))

#%%
pad_readout = 5 # Number of pixels to pad the readout dimension with (could be determined based on bandwidth)
n_runs = 10 # Number of pseudo-replicates

options = {'lambda_b': 0.1, # Smoothness of b-parameter
           'na_threshold': 3, # Noise amplification threshold to enable kx regularization
           'lambda_k': 0.05, # Regularization of kx frequencies
           'lr_omega':0.01, # Learning rate for optimization
           'lr_omega_b':0.001, # Learning rate for b-parameters
           'iterations1': 200, # Number of iterations for first phase of optimization (real-valued, establish b-parameters)
           'iterations2': 200} # Number of iterations for second phase of optimization (complex-valued, b-parameters fixed)

# TODO: Voxel size

#%%
for P in [1,2,3,4,5]:
    print(P)
    filename = f'./data/P{P}.h5'
    
    dirName = f'./results/mess_P{P}/'
    os.makedirs(dirName, exist_ok=True)

    #%%
    for run in range(n_runs+1):
        # Load image
        with h5py.File(filename, 'r') as f:
            img_orig = np.array(f['mess'])
            mask = np.array(f['normalization_mask'])
            
            te = f['mess'].attrs['echoTimes']
            tr = f['mess'].attrs['repetitionTime']
            bw = f['mess'].attrs['readoutBandwidth']
            B0 = f['mess'].attrs['fieldStrength']
            fa = f['mess'].attrs['flipAngle']
    
        te = te.astype(np.float32)
    
        te_t2 = te.copy()
        te_t2[2:] += tr
    
        te[2:] -= tr
    
        # Normalize muscle tissue to 0.5 intensity
        scale = abs(img_orig[0,mask]).mean()*2 
        img_orig /= scale
        
        if run != 0:
            # Add noise, SNR 40 at muscle (intensity 0.5)
            img = img_orig + (np.random.randn(*img_orig.shape) + 1j * np.random.randn(*img_orig.shape)) * (1/80)
        else:
            img = img_orig.copy()
        
        # Fit parameters to MESS image
        r = fit_mess(img, te, te_t2, tr, fa, bw, B0, options, polarity=[1,-1,-1,1], readout_dir=1, device=device)
    
        #%% Write results to nifti
        # TODO: Use voxel size
        def save(img, filename):
            M = [[0,0,-0.65,0],
                 [-0.5,0,0,0],
                 [0,-0.5,0,0],
                 [0,0,0,1]]
            img = nib.Nifti1Image(np.transpose(img, list(range(len(img.shape)-1,-1,-1))), np.array(M))
            nib.save(img, filename)
    
        plt.imsave(dirName + f'w_run{run}.png', abs(r['w'][126]),cmap=plt.get_cmap('gray'),vmin=0,vmax=0.75)
        plt.imsave(dirName + f'f_run{run}.png', abs(r['f'][126]),cmap=plt.get_cmap('gray'),vmin=0,vmax=1)
        
        save(r['w'] * np.exp(1j * r['b_plus']), dirName + f'w_run{run}.nii.gz')
        save(r['f'] * np.exp(1j * r['b_plus']), dirName + f'f_run{run}.nii.gz')
        save(r['r2'], dirName + f'r2_run{run}.nii.gz')

