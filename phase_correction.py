import numpy as np
from utils import rescale_fft

def phasecorrection_mess(img, scale=4):
    # Reduced image resolution (does not matter much, because we're only fitting a quadratic polynomial)
    img2 = rescale_fft(img, tuple([img.shape[0]] + [x//scale for x in img.shape[1:]]))

    # Mask for voxel with reliable phase information
    tmp = abs(img2).mean(axis=0)
    mask = tmp > tmp.max()*0.1
    
    # Image locations at reduced scale
    # TODO: Most indices unused
    indZ = np.indices([img2[0].shape[0],1,1])[0] / img2[0].shape[0] - 0.5
    indY = np.indices([1,img2[0].shape[1],1])[1] / img2[0].shape[1] - 0.5
    indX = np.indices([1,1,img2[0].shape[2]])[2] / img2[0].shape[2] - 0.5
        
    zi, yi, xi = np.meshgrid(indZ, indY, indX, indexing='ij')
    
    # Desired phase corrections
    # - Minimize phase difference between echo 1 and 4 and echo 2 and 3
    # - Minimize difference between phase differences of echo 1 and 2 and echo 3 and 4
    corr1 = img2[0] * img2[3]
    corr2 = img2[1] * img2[2]
    corr3 = img2[0] * img2[1].conj() * img2[2].conj() * img2[3]
    
    # Create coefficient matrix (A)
    a = np.stack((np.ones(xi[mask].shape), yi[mask]), axis=1)
    
    a = np.concatenate((np.concatenate((1*a, 0*a, 0*a, 1*a),axis=1),
                        np.concatenate((0*a, 1*a, 1*a, 0*a),axis=1),
                        np.concatenate((1*a,-1*a,-1*a, 1*a),axis=1)), axis=0)
    
    # Output vector (y)
    y = np.angle(np.concatenate((corr1[mask],
                                 corr2[mask],
                                 corr3[mask]),axis=0))
    
    # Solve Ax = y
    x, res, rank, s = np.linalg.lstsq(a,y, 1e-6)
    
    # Image locations at full scale
    indZ = np.indices([img[0].shape[0],1,1])[0] / img[0].shape[0] - 0.5
    indY = np.indices([1,img[0].shape[1],1])[1] / img[0].shape[1] - 0.5
    indX = np.indices([1,1,img[0].shape[2]])[2] / img[0].shape[2] - 0.5
    
    a = np.array([1, indY], dtype=np.object)

    # Phase correction (y = Ax)
    y = np.stack(a@(x.reshape(4,2).T),axis=0)
    img *= np.exp(-1j*y)
    
    return img