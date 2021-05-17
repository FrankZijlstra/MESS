# Projected power method as per Zhang et al, 2017
# Adapted from http://mrsrl.stanford.edu/~tao/software.html

import numpy as np
import matplotlib.pyplot as plt

def binarySelectionMaxCut(p1,p2,p3=None,im=None):
    eps = 1e-10
    
    p1 /= (abs(p1)+eps)
    p2 /= (abs(p2)+eps)
    
    if p3 is None:
        p3 = np.ones(p1.shape,dtype=np.float32)

    if im is None:
        im = np.ones(p1.shape,dtype=np.float32)

    p1 = p1.astype(np.complex64)
    p2 = p2.astype(np.complex64)
    p3 = p3.astype(np.complex64)
    im = im.astype(np.float32)
    
    # initialization
    pd1 = abs(p3-p1)
    pd2 = abs(p3-p2)
    
    X = pd1<pd2
    
    # power order for |p1 - p2|^pr
    pr = 2
    
    nb = [(1,0,0),(0,1,0),(0,0,1),
          (1,1,0),(-1,1,0),(1,0,1),(-1,0,1),(0,1,1),(0,-1,1),
          (1,1,1),(1,1,-1),(1,-1,1),(-1,1,1)]
    
    nb = []
    r = 3
    r2 = int(np.floor(r))
    for x in range(-r2,r2+1):
        for y in range(-r2,r2+1):
            for z in range(-r2,r2+1):
                if x == 0 and y == 0 and z == 0:
                    continue
                if np.sqrt(x**2 + y**2 + z**2) <= r and (-x,-y,-z) not in nb:
                    nb.append((x,y,z))

    nb_slice = []
    D = []
    
    for n in nb:
        nb_slice.append([tuple(slice(max(-x,0), (min(-x,0) or None)) for x in n),
                         tuple(slice(max(x,0), (min(x,0) or None)) for x in n)])
        dist = np.sqrt(n[0]**2 + n[1]**2 + n[2]**2)
        
        W = np.minimum(abs(im[nb_slice[-1][0]]),abs(im[nb_slice[-1][1]]))
        D.append([-abs(p1[nb_slice[-1][0]] - p1[nb_slice[-1][1]])**pr * W / dist,
                  -abs(p1[nb_slice[-1][0]] - p2[nb_slice[-1][1]])**pr * W / dist,
                  -abs(p2[nb_slice[-1][0]] - p1[nb_slice[-1][1]])**pr * W / dist,
                  -abs(p2[nb_slice[-1][0]] - p2[nb_slice[-1][1]])**pr * W / dist])     
    
    vis = False
    
    if vis:
        plt.ion()
        plt.figure()
    
    XX = np.empty(p1.shape,dtype=np.float32)
    YY = np.empty(p1.shape,dtype=np.float32)
        
    for it in range(200):
        # print(it)
        XX.fill(0)
        YY.fill(0)

    
        for i,n in enumerate(nb_slice):
            XX[n[0]] += X[n[1]] * (D[i][0] - D[i][1]) + D[i][1]
            XX[n[1]] += X[n[0]] * (D[i][0] - D[i][2]) + D[i][2]
            
            YY[n[0]] += X[n[1]] * (D[i][2] - D[i][3]) + D[i][3]
            YY[n[1]] += X[n[0]] * (D[i][1] - D[i][3]) + D[i][3]

        if it>100:
            X = XX > YY
        else:
            X = YY / (XX+YY+eps)

        p3 = X * p1 + (1-X) * p2
        
        if vis:
            plt.imshow(np.angle(np.concatenate((p1[p1.shape[0]//2],p2[p1.shape[0]//2],p3[p1.shape[0]//2]),axis=1)))
            plt.draw_all()
            plt.pause(1e-4)
    
    return p3

