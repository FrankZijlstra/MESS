import numpy as np
from skimage import filters
from skimage import measure
import scipy.spatial
import h5py


# Calculate thickness of a mesh
def mesh_thickness(verts, normals):
    kdtree = scipy.spatial.KDTree(verts)
    
    mindist = np.zeros(verts.shape[0])
    
    # For every vertex, find the closest point more or less along the vertex normal
    for i in range(verts.shape[0]):
        # print(i)
        nn = kdtree.query_ball_point(verts[i], 5)
        nn_verts = verts[[x for x in nn if x != i]]
        
        # Find angle between normal and vector pointing towards neighbour
        direction = verts[i] - nn_verts
        direction_norm = np.linalg.norm(direction,axis=1)
        angle = np.arccos(np.clip((normals[[i]] * direction).sum(axis=1) /  direction_norm, -1, 1))
        mask = angle<np.pi/6
        
        # If no points are found (within 5mm), set to nan
        if mask.sum() == 0:
            mindist[i] = np.nan
        else:
            mindist[i] = direction_norm[mask].min()
    
    # Return thickness statistics
    return mindist[~np.isnan(mindist)].mean(), mindist[~np.isnan(mindist)].std()


#%%
with open('./images/table1_thickness.csv', 'w') as fp:
    fp.write('P,sequence,f_mean,t_mean\n')
    
    f_dess_mean = 0
    t_dess_mean = 0
    f_mess_mean = 0
    t_mess_mean = 0
    
    for P in [1,2,3,4,5]:
        print(P)
        with h5py.File(f'./data/P{P}_seg.h5', 'r') as f:
            segF_MESS = np.array(f['mess_femoral_cartilage'])
            segT_MESS = np.array(f['mess_tibial_cartilage'])
            segF_DESS = np.array(f['dess_femoral_cartilage'])
            segT_DESS = np.array(f['dess_tibial_cartilage'])

        segFs = filters.gaussian(segF_MESS, sigma=1)
        verts, faces, normals, values = measure.marching_cubes(segFs, 0.5,spacing=(0.65,0.5,0.5))
        tF_MESS, sF_MESS = mesh_thickness(verts, normals)
        
        segFs = filters.gaussian(segF_DESS, sigma=1)
        verts, faces, normals, values = measure.marching_cubes(segFs, 0.5,spacing=(0.65,0.5,0.5))
        tF_DESS, sF_DESS = mesh_thickness(verts, normals)
                
        segTs = filters.gaussian(segT_MESS, sigma=1)
        verts, faces, normals, values = measure.marching_cubes(segTs, 0.5,spacing=(0.65,0.5,0.5))
        tT_MESS, sT_MESS = mesh_thickness(verts, normals)
        
        segTs = filters.gaussian(segT_DESS, sigma=1)
        verts, faces, normals, values = measure.marching_cubes(segTs, 0.5,spacing=(0.65,0.5,0.5))
        tT_DESS, sT_DESS = mesh_thickness(verts, normals)
        
        f_dess_mean += tF_DESS/5
        t_dess_mean += tT_DESS/5
        f_mess_mean += tF_MESS/5
        t_mess_mean += tT_MESS/5
        
        fp.write(f'{P},DESS,{tF_DESS:.2f},{tT_DESS:.2f}\n')
        fp.write(f'{P},MESS,{tF_MESS:.2f},{tT_MESS:.2f}\n')
        
    fp.write(f'overall,DESS,{f_dess_mean:.2f},{t_dess_mean:.2f}\n')
    fp.write(f'overall,MESS,{f_mess_mean:.2f},{t_mess_mean:.2f}\n')
