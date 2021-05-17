import numpy as np

# FFT-based rescaling (cropping/zero-padding)
def rescale_fft (image, target_shape):
    f_image = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image))) / np.sqrt(image.size)
    f_image = f_image[tuple(slice(max(x[1]//2 - x[0]//2,0),min(x[1]//2 - x[0]//2 + x[0], x[1])) for x in zip(target_shape, f_image.shape))]
    f_tmp = np.zeros(target_shape, dtype=np.complex64)
    f_tmp[tuple(slice(x[1]//2 - x[0]//2,x[1]//2 - x[0]//2 + x[0]) for x in zip(f_image.shape, target_shape))] = f_image
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(f_tmp))) * np.sqrt(f_tmp.size) / np.sqrt(image.size / np.prod(target_shape))

# FFT-based convolution ('same' padding)
def convolve_fft (im, kernel):
    padz = (kernel.shape[0]//2, kernel.shape[0] - kernel.shape[0]//2 - 1)
    pady = (kernel.shape[1]//2, kernel.shape[1] - kernel.shape[1]//2 - 1)
    padx = (kernel.shape[2]//2, kernel.shape[2] - kernel.shape[2]//2 - 1)
    im_pad = np.pad(im, (padz, pady, padx), mode='constant')

    d_z = im_pad.shape[0] - kernel.shape[0]
    d_y = im_pad.shape[1] - kernel.shape[1]
    d_x = im_pad.shape[2] - kernel.shape[2]
    
    kernel_pad = np.pad(kernel, ((d_z - d_z//2, d_z//2), (d_y - d_y//2, d_y//2), (d_x - d_x//2, d_x//2)), mode='constant')
    f_kernel = np.fft.fftn(kernel_pad)
    
    im_pad = np.fft.ifftshift(np.fft.ifftn(np.fft.fftn(im_pad) * f_kernel))
    im = im_pad[padz[0]:(-padz[1] or None),pady[0]:(-pady[1] or None),padx[0]:(-padx[1] or None)]
    return im
