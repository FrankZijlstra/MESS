import matplotlib.pyplot as plt

import numpy as np
import os
import nibabel as nib
import h5py


# TODO: Include voxel size
def save(img, filename):
    M = [[0,0,-0.65,0],
         [-0.5,0,0,0],
         [0,-0.5,0,0],
         [0,0,0,1]]
    img = nib.Nifti1Image(np.transpose(img, list(range(len(img.shape)-1,-1,-1))), np.array(M))
    nib.save(img, filename)

n_runs = 10

for P in [1,2,3,4,5]:
    print(P)
    filename = f'./data/P{P}.h5'
    
    dirName = f'./results/dess_P{P}/'
    os.makedirs(dirName, exist_ok=True)

    #%%
    with h5py.File(filename, 'r') as f:
        img_orig = np.array(f['dess'])
        mask = np.array(f['normalization_mask'])
        
        te_ref = f['mess'].attrs['echoTimes']
        te = f['dess'].attrs['echoTimes']
        tr = f['dess'].attrs['repetitionTime']
        bw = f['dess'].attrs['readoutBandwidth']
        fa = f['dess'].attrs['flipAngle']
        
        # 16.4% reduction in bandwidth by adding 1.56ms (difference between DESS spoiler and MESS spoiler duration)
        # split over the two DESS readouts
        bw = 1/(1/bw + 1.56e-3/2) 
        
        bw_mess = f['mess'].attrs['readoutBandwidth']  
    
    dte = (te[0] - (tr+te[1]))
    
    # Calculate T2
    t1 = 1.2 # Cartilage
    t2_scale_w = np.sin(fa/180 * np.pi/2)**2 * (1 + np.exp(-tr/t1))/(1-np.cos(fa/180 * np.pi) * np.exp(-tr/t1))
    r2 = -np.log(abs(img_orig[0]/(img_orig[1]/t2_scale_w + 1e-12))) / dte
    
    # Difference in intensity of the muscle region at TE_mess[0] and TE_dess[0] due to T2
    te_diff_factor = np.exp(r2[mask].mean() * (te[0] - te_ref[0]))
    
    # Normalize muscle tissue to 0.5 intensity at TE_mess[0]
    scale = abs(img_orig[0,mask] * te_diff_factor).mean()*2
    img_orig /= scale
    
    #%% Loop over pseudo-replications
    for run in range(n_runs+1):
        print(run)
        img = img_orig.copy()
        if run > 0:
            # Add noise, SNR 40 * np.sqrt(bw_mess/bw_dess) at muscle (intensity 0.5)
            img += (np.random.randn(*img.shape) + 1j*np.random.randn(*img.shape))*(1/(80 * np.sqrt(bw_mess/bw)))
            
        # Scale image to intensity of TE_mess[0]
        img *= te_diff_factor
        
        # Calculate T2
        r2 = -np.log(abs(img[0]/(img[1]/t2_scale_w + 1e-12))) / dte
        
        # Save images to nifti
        save(img[0], dirName + f'w_run{run}.nii.gz')
        save(img[1], dirName + f'w2_run{run}.nii.gz')
        save(r2, dirName + f'r2_run{run}.nii.gz')
        
        plt.imsave(dirName + f'w_run{run}.png', abs(img[0,126]),cmap=plt.get_cmap('gray'),vmin=0,vmax=0.75)

