import numpy as np

from projected_power import binarySelectionMaxCut
from utils import rescale_fft, convolve_fft


def getSixPeakFatModel ():
    freqs = np.array([-3.80, -3.40, -2.60, -1.94, -0.39, 0.60])*1e-6
    amps = np.array([0.087, 0.693, 0.128, 0.004, 0.039, 0.048])
    amps /= amps.sum()
    
    return {'frequencies':freqs, 'amplitudes':amps}

def getSinglePeakFatModel ():
    freqs = np.array([-3.40])*1e-6
    amps = np.array([1])

    return {'frequencies':freqs, 'amplitudes':amps}

def getFatSignal (fieldStrength, echoTimes, fatModel):
    gm = 42.57747892e6 # Hz/T
    return np.sum(fatModel['amplitudes'][echoTimes.ndim * (None,) + (slice(None),)] * np.exp(1j * 2 * np.pi * fatModel['frequencies'][echoTimes.ndim * (None,) + (slice(None),)]*gm*fieldStrength * echoTimes[...,None]), axis=-1)


# Get bA and bB phasor candidates as per Berglund et al
def phasor_candidates (img, fat):
    ai0s = abs(img[0])**2
    ai1s = abs(img[1])**2
    
    c1 = ai0s * (1 - np.real(fat[1])) - ai1s * (1 - np.real(fat[0]))
    c2 = ai0s * (abs(fat[1])**2 - np.real(fat[1])) - ai1s * (abs(fat[0])**2 - np.real(fat[0]))
    c3 = ai0s * ai1s * abs(fat[0] - fat[1])**2 - (np.imag(fat[0]) * ai1s - np.imag(fat[1]) * ai0s)**2
    del ai0s, ai1s
    
    mask = (c3>0) & ((c1+c2)!=0)
    
    QA = (c1[mask]+np.sqrt(c3[mask])) / (c1[mask]+c2[mask])
    QB = (c1[mask]-np.sqrt(c3[mask])) / (c1[mask]+c2[mask])
    del c1, c2, c3

    bA = np.ones(img.shape[1:], dtype=np.complex64)
    bB = -np.ones(img.shape[1:], dtype=np.complex64)
    
    bA[mask] = img[1][mask] * (1+QA*(fat[0]-1)) / (img[0][mask] * (1+QA*(fat[1]-1)))
    bB[mask] = img[1][mask] * (1+QB*(fat[0]-1)) / (img[0][mask] * (1+QB*(fat[1]-1)))

    del QA,QB
    
    bA[mask] /= abs(bA[mask])
    bB[mask] /= abs(bB[mask])

    return bA, bB
   
# Real-valued water-fat separation
def separate_real (img, fat):
    A = np.array([[1, np.real(fat[0])], [1, np.real(fat[1])], [0, np.imag(fat[0])], [0, np.imag(fat[1])]], dtype=np.float32)
    Ainv = np.linalg.pinv(A)
    
    S0 = np.concatenate((np.real(img), np.imag(img)), axis=0)
    
    res = S0.transpose(1,2,3,0)@Ainv.T
    
    w = res[...,0]
    f = res[...,1]
    
    return w, f

# Complex-valued water-fat separation
def separate_complex (img, fat):
    A = np.array([[1, fat[0]], [1, fat[1]]], dtype=np.complex64)
    Ainv = np.linalg.pinv(A)
        
    res = img.transpose(1,2,3,0)@Ainv.T
    
    w = res[...,0]
    f = res[...,1]
    
    return w, f

# Real-valued water-fat separation with chemical-shift correction
# TODO: Readout axis is hardcoded
def separate_real_cs (img, fieldStrength, bw, te, fatModel, polarity=[1,-1]):
    polarity = np.array(polarity)
    
    tek = (np.fft.fftfreq(img.shape[2]) * 1/bw)[None,:] * polarity[:,None] + te[:,None]
    F = np.fft.fft(img, axis=2)
    
    F_W = np.zeros(F.shape[1:], dtype=np.complex64)
    F_F = np.zeros(F.shape[1:], dtype=np.complex64)
    
    for i in range(img.shape[2]):
        if i == 0:
            fat = getFatSignal(fieldStrength, tek[:,i], fatModel)
            A = np.array([[1,fat[0]],[1,fat[1]]], dtype=np.complex64)
            Ainv = np.linalg.pinv(A)
            
            res = Ainv @ F[:,:,i,:].transpose(1,2,0)[...,np.newaxis]
            F_W[:,i,:] = res[:,:,0,0]
            F_F[:,i,:] = res[:,:,1,0]
            
        else:
            fat2 = getFatSignal(fieldStrength, tek[:,i], fatModel)
            fat3 = getFatSignal(fieldStrength, tek[:,img.shape[2]-i], fatModel)
    
            A = np.array([[1,fat2[0]],[1,fat2[1]],[1,fat3[0].conj()],[1,fat3[1].conj()]], dtype=np.complex64)
            Ainv = np.linalg.pinv(A)
            
            res = Ainv @ np.concatenate((F[:,:,i,:].transpose(1,2,0)[...,np.newaxis],
                                          F[:,:,img.shape[2]-i,:].transpose(1,2,0)[...,np.newaxis].conj()), axis=2)
            F_W[:,i,:] = res[:,:,0,0]
            F_F[:,i,:] = res[:,:,1,0]
    
    w = np.fft.ifft(F_W, axis=1)
    f = np.fft.ifft(F_F, axis=1)
    
    return np.real(w),np.real(f)

# Complex-valued water-fat separation with chemical-shift correction
# TODO: Readout axis is hardcoded
def separate_complex_cs (img, fieldStrength, bw, te, fatModel, polarity=[1,-1]):
    polarity = np.array(polarity)
    
    tek = (np.fft.fftfreq(img.shape[2]) * 1/bw)[None,:] * polarity[:,None] + te[:,None]
    F = np.fft.fft(img, axis=2)
    
    F_W = np.zeros(F.shape[1:], dtype=np.complex64)
    F_F = np.zeros(F.shape[1:], dtype=np.complex64)
    
    for i in range(img.shape[2]):
        fat = getFatSignal(fieldStrength, tek[:,i], fatModel)
        A = np.array([[1,fat[0]],[1,fat[1]]], dtype=np.complex64)
        Ainv = np.linalg.pinv(A)
        
        res = Ainv @ F[:,:,i,:].transpose(1,2,0)[...,np.newaxis]
        F_W[:,i,:] = res[:,:,0,0]
        F_F[:,i,:] = res[:,:,1,0]
    
    w = np.fft.ifft(F_W, axis=1)
    f = np.fft.ifft(F_F, axis=1)
    
    return w,f

# Calculate b0 phasor from image and b phasor
def calculate_b0(img, fat, b):
    b0 = (img[0] * (1-fat[1]) - img[1] * (1-fat[0]) / b) / (fat[0] - fat[1])
    b0[abs(b0) == 0] = 1
    b0 /= abs(b0)
    return b0

# Two point dixon method
def dixon_2point (img, echoTimes, fieldStrength, readoutBandwidth=None, fieldPhasor=None, fatModel='6peak', realValued=True, chemicalShiftCorrection=False, projectedPower=True, projectPhasor=True, smoothPhasor=True, verbose=True):
    img = img.astype(np.complex64)
    
    if fatModel == '6peak':
        fatModel = getSixPeakFatModel()
    elif fatModel == '1peak':
        fatModel = getSinglePeakFatModel()
    else:
        raise RuntimeError('Invalid option for fatModel')
        
    fat = getFatSignal(fieldStrength, echoTimes, fatModel).astype(np.complex64)
    
    if fieldPhasor is not None:
        b = fieldPhasor.copy()
    else:
        b = np.ones(img.shape[1:], dtype=np.complex64)
    
    if projectedPower:
        # Get b phasor by projected power method as per Zhang et al, 2017
        if verbose:
            print('Projected power...')
    
        scale = 4
        # Downscale image for speed
        target_size = (img.shape[0],) + tuple(x//scale for x in img.shape[1:])
        zoomed_img = rescale_fft(img, target_size)
        
        # TODO: Make neighbourhood parameter, preferably in mm
        bA, bB = phasor_candidates(zoomed_img, fat)
        b_zoomed = binarySelectionMaxCut(bA, bB, im=abs(zoomed_img[0]) + abs(zoomed_img[1]))

        # Upscale b phasor
        b = rescale_fft(b_zoomed, img.shape[1:])
        b /= abs(b)

    if projectPhasor:
        # Project upscaled b phasor onto high resolution phasor candidates
        if verbose:
            print('Projecting onto phasors...')
            
        bA, bB = phasor_candidates(img, fat)
        b2 = bB.copy()
        smallest = abs(np.angle(bA / b)) < abs(np.angle(bB / b))
        b2[smallest] = bA[smallest]
        b = b2
        
        del b2, smallest
        
    if smoothPhasor:
        # Smooth b phasor
        if verbose:
            print('Post-smoothing...')
            
        # Smooth phasor
        kernel = np.ones((11,11,11), dtype=np.float32) # TODO: Make parameter, preferably in mm  
        
        mag = abs(img[0]) + abs(img[1])
        b = convolve_fft(b * mag, kernel)
        b /= abs(b)
        del mag


    # Get the phasor at the first echo (b0)
    b0 = calculate_b0(img, fat, b)

    # Remove b and b0 phasors from img
    img[0] /= b0
    img[1] /= b0*b
    
    # Perform water-fat separation
    if realValued:
        if chemicalShiftCorrection:
            if not readoutBandwidth:
                raise RuntimeError('Readout bandwidth needs to be specified for chemical shift correction')
            w,f = separate_real_cs(img, fieldStrength, readoutBandwidth, echoTimes, fatModel)
        else:
            w,f = separate_real(img, fat)
    else:
        if chemicalShiftCorrection:
            if not readoutBandwidth:
                raise RuntimeError('Readout bandwidth needs to be specified for chemical shift correction')
            w,f = separate_complex_cs(img, fieldStrength, readoutBandwidth, echoTimes, fatModel)
        else:
            w,f = separate_complex(img, fat)
    
    return {'w':w, 'f':f, 'b0':b0, 'b':b}
