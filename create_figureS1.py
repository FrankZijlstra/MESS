import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os

def load(filename):
    return np.array(nib.load(filename).dataobj).transpose(2,1,0)

image_dir = './images/figureS1/'
os.makedirs(image_dir, exist_ok=True)
    
slices = [126,126,120,120,120]
slices_dess = [126,126,120,120,118]

for P in [1,2,3,4,5]:
    dirname = f'./mess_final3_P{P}/'
    dirname_dess = f'./dess_final_P{P}/'

    w = load(dirname + 'w_run0.nii.gz')
    f = load(dirname + 'f_run0.nii.gz')
    
    dess_splus = load(dirname_dess + 'w_run0.nii.gz')

    plt.imsave(image_dir + f'P{P}_mess_w.png', abs(w[slices[P-1]]), vmin=0, vmax=0.9, cmap=plt.get_cmap('gray'))
    plt.imsave(image_dir + f'P{P}_mess_f.png', abs(f[slices[P-1]]), vmin=0, vmax=1.2, cmap=plt.get_cmap('gray'))

    plt.imsave(image_dir + f'P{P}_dess_w.png', abs(dess_splus[slices_dess[P-1]]), vmin=0, vmax=0.9, cmap=plt.get_cmap('gray'))
