import torch
import numpy as np
import time
import math

from phase_correction import phasecorrection_mess
from dixon import getSixPeakFatModel, getFatSignal, dixon_2point, calculate_b0

# L2 loss
def L2(x):
    return (x**2).mean()

# Weighted L2 loss
def L2_weighted(x,c):
    return (abs(x)**2 * c).mean()

# L2 loss on 2D phase differences
def L2_phase(x):
    return (torch.angle(torch.exp(1j * x[:,:-1] - 1j * x[:,1:]))**2).mean() + \
           (torch.angle(torch.exp(1j * x[:,:,:-1] - 1j * x[:,:,1:]))**2).mean()

# Noise amplification formula as defined by Lu et al
def noise_amplification_lu(A):
    return np.trace(np.linalg.inv(np.matrix(A).H @ A)).real / A.shape[1] * A.shape[0]

# Optimize params over a given loss function using pytorch
def optimize(params, loss_func, its):
    for p in params:
        for x in p['params']:
            x.requires_grad = True
    
    opt = torch.optim.RMSprop(params, momentum=0.95)

    for i in range(its):
        loss_value = loss_func()
        opt.zero_grad()
        loss_value.backward()
        opt.step()
        # print(i,loss_value.item())
   
    for p in params:
        for x in p['params']:
            x.requires_grad = False

# MESS fitting procedure:
#   - Phase correction
#   - 2-point Dixon initialization
#   - Real-valued optimization to refine b-parameters
#   - Complex-valued optimization with fixed b-parameters
def fit_mess(img, te, te_t2, tr, fa, bw, B0, options, polarity=[1,-1,-1,1], readout_dir=2, device=None):
    pad_readout = 5    
    t0 = time.time()

    #%% Linear phase correction
    img = phasecorrection_mess(img)
    print('Time phase correction:', time.time() - t0)

    #%% Pad readout dimension to prevent signal wrapping around because of chemical shift correction
    img = np.pad(img,((0,0),(0,0),(pad_readout,pad_readout),(0,0)))

    #%% Initialize water-fat separation
    N = img.shape[readout_dir+1]

    # Readout polarity
    polarity = np.array(polarity)

    fatModel = getSixPeakFatModel()
    te_k = (np.fft.fftfreq(N).astype(np.float32) * 1/bw)[None,:] * polarity[:,None] + te[:,None]

    fat_kspace = getFatSignal(B0, te_k, fatModel)
    fat = torch.from_numpy(fat_kspace).to(device, dtype=torch.complex64)
    
    t1 = 1.2 # Cartilage
    t2_scale_w = np.sin(fa/180 * np.pi/2)**2 * (1 + np.exp(-tr/t1))/(1-np.cos(fa/180 * np.pi) * np.exp(-tr/t1))
    t1 = 0.365 # Marrow
    t2_scale_f = np.sin(fa/180 * np.pi/2)**2 * (1 + np.exp(-tr/t1))/(1-np.cos(fa/180 * np.pi) * np.exp(-tr/t1))


    #%% Estimate delta-B0 using standard real-valued non-CS-corrected 2-point Dixon (Berglund et al) on the first two echoes
    t0 = time.time()
    
    out = dixon_2point(img[0:2], te[0:2], fieldStrength=B0)
    fat_dixon = getFatSignal(B0, te, fatModel)

    b_init = out['b']
    b_plus_init = out['b0']
    
    # Calculate b_minus based on 3rd and 4th echo (same as b0 in dixon_2point)
    b_minus_init = calculate_b0(img[[3,2]], fat_dixon[[3,2]], b_init.conj())
    
    print('Time Dixon:', time.time() - t0)

    #%% Calculate noise-amplification factor for kx-regularization parameter (Lu et al)
    na = np.zeros(fat_kspace.shape[1])
    for i in range(fat_kspace.shape[1]):
        na[i] = noise_amplification_lu(np.concatenate((np.ones((2,1)), fat_kspace[0:2,[i]]), axis=1))

    lambda_k = (torch.from_numpy(na).to(device)[None,None,:]>options['na_threshold']) * options['lambda_k']

    #%% Iterative fitting
    # TODO: Move functions somewhere nicer, implement transpose
    # Fits MESS parameters to a subset of slices in the image: img[sb:se]
    def fit(sb,se):
        
        # Initialize parameters (transpose if necessary)
        if readout_dir == 1:
            sh_x,sh_y = img.shape[2:]
        else:
            sh_y,sh_x = img.shape[2:]

        r2 = torch.ones((se-sb,sh_y,sh_x),device=device)*20

        w_real = torch.zeros((se-sb,sh_y,sh_x),device=device)
        w_imag = torch.zeros((se-sb,sh_y,sh_x),device=device)
        f_real = torch.zeros((se-sb,sh_y,sh_x),device=device)
        f_imag = torch.zeros((se-sb,sh_y,sh_x),device=device)

        # Initialize from 2-point Dixon field calculation
        if readout_dir == 1:
            # Transpose
            b = torch.from_numpy(np.angle(b_init[sb:se].transpose(0,2,1))).to(device,dtype=torch.float32)
            b_plus = torch.from_numpy(np.angle(b_plus_init[sb:se].transpose(0,2,1))).to(device,dtype=torch.float32)
            b_minus = torch.from_numpy(np.angle(b_minus_init[sb:se].transpose(0,2,1))).to(device,dtype=torch.float32)
    
            y = torch.from_numpy(img[:,sb:se].transpose(0,1,3,2)).to(device,dtype=torch.complex64)
        else:
            b = torch.from_numpy(np.angle(b_init[sb:se])).to(device,dtype=torch.float32)
            b_plus = torch.from_numpy(np.angle(b_plus_init[sb:se])).to(device,dtype=torch.float32)
            b_minus = torch.from_numpy(np.angle(b_minus_init[sb:se])).to(device,dtype=torch.float32)
    
            y = torch.from_numpy(img[:,sb:se]).to(device,dtype=torch.complex64)
        
        # MESS signal model
        def model():
            w = w_real + 1j*w_imag
            f = f_real + 1j*f_imag
            
            phase = torch.stack([b_plus, b_plus+b, b_minus-b, b_minus], dim=0)    
            phase = torch.exp(1j*phase)

            sig = []
            for i in range(2):
                sig_w = w * torch.exp(-r2*(te_t2[i] - te_t2[0])) * phase[i]
                sig_f = f * torch.exp(-r2*(te_t2[i] - te_t2[0])) * phase[i]
                s = sig_w + torch.fft.ifft(torch.fft.fft(sig_f, dim=2) * fat[[i],None], dim=2)

                sig.append(s[None])
            for i in range(2,4):
                sig_w = w * torch.exp(-r2*(te_t2[i] - te_t2[0])) * t2_scale_w * phase[i]
                sig_f = f * torch.exp(-r2*(te_t2[i] - te_t2[0])) * t2_scale_f * phase[i]
                
                s = sig_w + torch.fft.ifft(torch.fft.fft(sig_f, dim=2) * fat[[i],None], dim=2)
                sig.append(s[None])
            sig = torch.cat(sig,dim=0)
            return sig
        
        # Optimization objective function
        def loss():
            w = w_real + 1j*w_imag
            f = f_real + 1j*f_imag
            
            # Consistency with acquired data
            loss_value = (abs(model() - y)**2).mean()
            
            # B0-phasor smoothness
            loss_value += options['lambda_b'] * (L2_phase(b))

            # kx regularization
            phase = torch.exp(1j*(b_plus))
            loss_kx = L2_weighted(torch.fft.fft(w * phase, dim=2, norm='ortho'), lambda_k) + \
                     L2_weighted(torch.fft.fft(f * phase, dim=2, norm='ortho'), lambda_k)
            loss_value += loss_kx
            
            return loss_value * y.numel()
        
        # Real-valued W/F fitting, for refining B0-parameters to all MESS echoes
        omega = [w_real,f_real,r2]
        omega_b = [b,b_plus,b_minus]

        optimize([{'params': omega, 'lr':options['lr_omega']},
                  {'params': omega_b, 'lr':options['lr_omega_b']}], loss, options['iterations1'])
        
        # Complex-valued W/F fitting, with fixed B0-parameters
        omega = [w_real,w_imag,f_real,f_imag,r2]
        optimize([{'params': omega, 'lr':options['lr_omega']}], loss, options['iterations2'])

        # Gather results and transfer to CPU/numpy
        result =  {'w':w_real[:,:,pad_readout:-pad_readout].cpu().numpy() + 1j*w_imag[:,:,pad_readout:-pad_readout].cpu().numpy(),
                   'f':f_real[:,:,pad_readout:-pad_readout].cpu().numpy() + 1j*f_imag[:,:,pad_readout:-pad_readout].cpu().numpy(),
                   'r2':r2[:,:,pad_readout:-pad_readout].cpu().numpy(),
                   'b':b[:,:,pad_readout:-pad_readout].cpu().numpy(),
                   'b_plus':b_plus[:,:,pad_readout:-pad_readout].cpu().numpy(),
                   'b_minus':b_minus[:,:,pad_readout:-pad_readout].cpu().numpy()}
        
        if readout_dir == 1:
            result = {k:x.transpose(0,2,1) for k,x in result.items()}

        return result

    #%% Divide slices of image in chunks and fit each chunk
    r = []
    
    if device != torch.device('cpu'):
        gpu_ram = torch.cuda.get_device_properties(0).total_memory / 1024**3 # GB
        ram_intercept = 1.14 # GPU RAM base (GB)
        ram_slope = 4.91e-7 # GPU RAM per voxel (GB)
        slices_per_chunk = (gpu_ram - ram_intercept) // (ram_slope * img.shape[2] * img.shape[3])
        chunks = math.ceil(img.shape[1] / slices_per_chunk)
    else:
        chunks = 10 #img.shape[1]
        
    x = math.ceil(img.shape[1]/chunks)
    t0 = time.time()
    for i in range(chunks):
        print(i+1,'/',chunks,i*x,min((i+1)*x, img.shape[1]))
        r.append(fit(i*x,min((i+1)*x, img.shape[1])))
    
    r = {k:np.concatenate([x[k] for x in r], axis=0) for k in r[0]}

    print('Time fitting:', time.time() - t0)
    
    return r
