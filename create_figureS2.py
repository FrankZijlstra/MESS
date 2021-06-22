import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os

def load(filename):
    return np.array(nib.load(filename).dataobj).transpose(2,1,0)

image_dir = './images/figureS2/'
os.makedirs(image_dir, exist_ok=True)
    
slices = [126,126,120,120,120]
slices_dess = [126,126,120,120,118]

for P in [1]:
    dirname = f'./results/mess_P{P}/'
    dirname2 = f'./results/mess_less_regularization2_P{P}/'

    w = load(dirname + 'w.nii.gz')
    f = load(dirname + 'f.nii.gz')
    w2 = load(dirname2 + 'w.nii.gz')
    f2 = load(dirname2 + 'f.nii.gz')
    
    plt.imsave(image_dir + f'P{P}_mess_w.png', abs(w[slices[P-1]]), vmin=0, vmax=0.9, cmap=plt.get_cmap('gray'))
    plt.imsave(image_dir + f'P{P}_mess_f.png', abs(f[slices[P-1]]), vmin=0, vmax=1.2, cmap=plt.get_cmap('gray'))
    
    plt.imsave(image_dir + f'P{P}_mess2_w.png', abs(w2[slices[P-1]]), vmin=0, vmax=0.9, cmap=plt.get_cmap('gray'))
    plt.imsave(image_dir + f'P{P}_mess2_f.png', abs(f2[slices[P-1]]), vmin=0, vmax=1.2, cmap=plt.get_cmap('gray'))
