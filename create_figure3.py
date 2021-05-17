import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import skimage

def load(filename):
    return np.array(nib.load(filename).dataobj).transpose(2,1,0)

dirname = './results/mess_P1/'
dirname_dess = './results/dess_P1/'
S = 126 # Slice to visualize
image_dir = './images/figure3/'

os.makedirs(image_dir, exist_ok=True)

# Load data
with h5py.File('./data/P1.h5', 'r') as f:
    te_dess = f['dess'].attrs['echoTimes']
    tr_dess = f['dess'].attrs['repetitionTime']
    te_mess = f['mess'].attrs['echoTimes']
    tr_mess = f['mess'].attrs['repetitionTime']
    fa = f['dess'].attrs['flipAngle']

with h5py.File('./data/P1_seg.h5', 'r') as f:
    dess_femoral_cartilage = np.array(f['dess_femoral_cartilage'])
    mess_femoral_cartilage = np.array(f['mess_femoral_cartilage'])
    dess_tibial_cartilage = np.array(f['dess_tibial_cartilage'])
    mess_tibial_cartilage = np.array(f['mess_tibial_cartilage'])
    

w = load(dirname + 'w.nii.gz')
f = load(dirname + 'f.nii.gz')
r2 = load(dirname + 'r2.nii.gz')

dess_splus = load(dirname_dess + 'w_run0.nii.gz')
dess_sminus = load(dirname_dess + 'w2_run0.nii.gz')
dess_r2 = load(dirname_dess + 'r2_run0.nii.gz')

t1 = 1.2 # Cartilage
t2_scale_w = np.sin(fa/180 * np.pi/2)**2 * (1 + np.exp(-tr_dess/t1))/(1-np.cos(fa/180 * np.pi) * np.exp(-tr_dess/t1))

# Write images
plt.imsave(image_dir + 'dess_splus.png', abs(dess_splus[S]), vmin=0, vmax=0.9, cmap=plt.get_cmap('gray'))
plt.imsave(image_dir + 'dess_sminus.png', abs(dess_sminus[S]) / t2_scale_w, vmin=0, vmax=0.9, cmap=plt.get_cmap('gray'))

plt.imsave(image_dir + 'mess_splus.png', abs(w[S]) * np.exp(-r2[S] * (te_dess[0] - te_mess[0])), vmin=0, vmax=0.9, cmap=plt.get_cmap('gray'))
plt.imsave(image_dir + 'mess_sminus.png', abs(w[S]) * np.exp(-r2[S] * (tr_dess + te_dess[1] - te_mess[0])), vmin=0, vmax=0.9, cmap=plt.get_cmap('gray'))

plt.imsave(image_dir + 't2_mess.png', 1 / r2[S] * (abs(w[S])>abs(f[S])) * ((abs(w[S])+abs(f[S]))>0.2), vmin=0, vmax=0.05, cmap=plt.get_cmap('jet'))
plt.imsave(image_dir + 't2_dess.png', 1 / dess_r2[S] * (abs(dess_splus[S])>0.2), vmin=0, vmax=0.05, cmap=plt.get_cmap('jet'))

plt.imsave(image_dir + 't2_mess_f.png', 1 / r2[S] * (abs(f[S])>abs(w[S])) * ((abs(w[S])+abs(f[S]))>0.2), vmin=0, vmax=0.15, cmap=plt.get_cmap('jet'))

plt.imsave(image_dir + 'cmap_gray.png', np.linspace(1,0,256)[:,None], cmap=plt.get_cmap('gray'))
plt.imsave(image_dir + 'cmap_jet.png', np.linspace(1,0,256)[:,None], cmap=plt.get_cmap('jet'))

plt.imsave(image_dir + 'mess_ip.png', abs(w[S]) + abs(f[S]), vmin=0, vmax=1.2, cmap=plt.get_cmap('gray'))
plt.imsave(image_dir + 'mess_op.png', abs(abs(w[S]) - abs(f[S])), vmin=0, vmax=1.2, cmap=plt.get_cmap('gray'))

# Write images with cartilage overlay
dpi = 100
fig = plt.figure(figsize=(dess_splus[S].shape[0]/dpi,dess_splus[S].shape[1]/dpi), dpi=dpi*2, frameon=False)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(abs(dess_splus[S]), vmin=0, vmax=0.9, cmap=plt.get_cmap('gray'))
c = skimage.measure.find_contours(dess_femoral_cartilage[S])
ax.plot(c[0][:,1],c[0][:,0], color=(1,0,0), linewidth=0.5)
c = skimage.measure.find_contours(dess_tibial_cartilage[S])
ax.plot(c[0][:,1],c[0][:,0], color=(0,1,0), linewidth=0.5)
fig.savefig(image_dir + 'dess_splus_cartilage.png')

fig = plt.figure(figsize=(dess_splus[S].shape[0]/dpi,dess_splus[S].shape[1]/dpi), dpi=dpi*2, frameon=False)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(abs(w[S]) * np.exp(-r2[S] * (te_dess[0] - te_mess[0])), vmin=0, vmax=0.9, cmap=plt.get_cmap('gray'))
c = skimage.measure.find_contours(mess_femoral_cartilage[S])
ax.plot(c[0][:,1],c[0][:,0], color=(1,0,0), linewidth=0.5)
c = skimage.measure.find_contours(mess_tibial_cartilage[S])
ax.plot(c[0][:,1],c[0][:,0], color=(0,1,0), linewidth=0.5)
fig.savefig(image_dir + 'mess_splus_cartilage.png')
