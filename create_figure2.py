import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py

def load(filename):
    return np.array(nib.load(filename).dataobj).transpose(2,1,0)


dirname = './results/mess_P1/'
S = 126 # Slice to visualize
image_dir = './images/figure2/'

os.makedirs(image_dir, exist_ok=True)

# Load data
with h5py.File('./data/P1.h5', 'r') as f:
    img = np.array(f['mess'])
    mask = np.array(f['normalization_mask'])
    img /= abs(img[0,mask]).mean()*2

w = load(dirname + 'w.nii.gz')
f = load(dirname + 'f.nii.gz')
r2 = load(dirname + 'r2.nii.gz')

b = load(dirname + 'b.nii.gz')
b_plus = load(dirname + 'b_plus.nii.gz')
b_minus = load(dirname + 'b_minus.nii.gz')

# Write images
plt.imsave(image_dir + 'echo1.png', abs(img[0,S]), vmin=0, vmax=0.9, cmap=plt.get_cmap('gray'))
plt.imsave(image_dir + 'echo2.png', abs(img[1,S]), vmin=0, vmax=0.9, cmap=plt.get_cmap('gray'))
plt.imsave(image_dir + 'echo3.png', abs(img[2,S]), vmin=0, vmax=0.9, cmap=plt.get_cmap('gray'))
plt.imsave(image_dir + 'echo4.png', abs(img[3,S]), vmin=0, vmax=0.9, cmap=plt.get_cmap('gray'))

plt.imsave(image_dir + 'w_magn.png', abs(w[S]), vmin=0, vmax=0.9, cmap=plt.get_cmap('gray'))
plt.imsave(image_dir + 'f_magn.png', abs(f[S]), vmin=0, vmax=1.2, cmap=plt.get_cmap('gray'))
plt.imsave(image_dir + 'w_phase.png', np.angle(w[S]), vmin=-np.pi, vmax=np.pi, cmap=plt.get_cmap('gray'))
plt.imsave(image_dir + 'f_phase.png', np.angle(f[S]), vmin=-np.pi, vmax=np.pi, cmap=plt.get_cmap('gray'))

plt.imsave(image_dir + 'r2.png', r2[S], vmin=0, vmax=100, cmap=plt.get_cmap('jet'))

plt.imsave(image_dir + 'b.png', np.angle(np.exp(1j*b[S])), vmin=-np.pi, vmax=np.pi, cmap=plt.get_cmap('gray'))
plt.imsave(image_dir + 'b_plus.png', np.angle(np.exp(1j*b_plus[S] + 1j*np.pi)), vmin=-np.pi, vmax=np.pi, cmap=plt.get_cmap('gray'))
plt.imsave(image_dir + 'b_minus.png', np.angle(np.exp(-1j*b_minus[S] + 1j*np.pi)), vmin=-np.pi, vmax=np.pi, cmap=plt.get_cmap('gray'))

plt.imsave(image_dir + 'cmap_gray.png', np.linspace(1,0,256)[:,None], cmap=plt.get_cmap('gray'))
plt.imsave(image_dir + 'cmap_jet.png', np.linspace(1,0,256)[:,None], cmap=plt.get_cmap('jet'))
