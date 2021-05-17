import numpy as np
import nibabel as nib
import h5py

with open('./images/table1_t2.csv', 'w') as fp:
    fp.write('P,sequence,f_mean,f_std,t_mean,t_std\n')
    
    allvalues = [np.zeros((0)),np.zeros((0)),np.zeros((0)),np.zeros((0))]
    
    for P in [1,2,3,4,5]:
        # Load segmentations
        with h5py.File(f'./data/P{P}_seg.h5', 'r') as f:
            segF_MESS = np.array(f['mess_femoral_cartilage'])
            segT_MESS = np.array(f['mess_tibial_cartilage'])
            segF_DESS = np.array(f['dess_femoral_cartilage'])
            segT_DESS = np.array(f['dess_tibial_cartilage'])

        # Load R2 maps
        dirMESS = f'./results/mess_P{P}/'
        dirDESS = f'./results/dess_P{P}/'
               
        mess_r2 = nib.load(dirMESS + 'r2_run0.nii.gz').get_fdata().transpose(2,1,0)
        dess_r2 = nib.load(dirDESS + 'r2_run0.nii.gz').get_fdata().transpose(2,1,0)
        
        mess_t2 =  1000/(mess_r2 + 1e-12)
        dess_t2 =  1000/(dess_r2 + 1e-12)
        
        # Get R2 for femoral/tibial cartilage
        mess_f = mess_t2[segF_MESS]
        dess_f = dess_t2[segF_DESS]
        mess_t = mess_t2[segT_MESS]
        dess_t = dess_t2[segT_DESS]
        
        # Keep all values for overall mean/std
        allvalues[0] = np.concatenate((allvalues[0], dess_f))
        allvalues[1] = np.concatenate((allvalues[1], mess_f))
        allvalues[2] = np.concatenate((allvalues[2], dess_t))
        allvalues[3] = np.concatenate((allvalues[3], mess_t))
        
        # Write statistics to table
        fp.write(f'{P},DESS,{dess_f.mean():.1f},{dess_f.std():.1f},{dess_t.mean():.1f},{dess_t.std():.1f}\n')
        fp.write(f'{P},MESS,{mess_f.mean():.1f},{mess_f.std():.1f},{mess_t.mean():.1f},{mess_t.std():.1f}\n')

    # Write overall statistics to table
    fp.write(f'overall_mean,DESS,{allvalues[0].mean():.1f},{allvalues[0].std():.1f},{allvalues[2].mean():.1f},{allvalues[2].std():.1f}\n')
    fp.write(f'overall_mean,MESS,{allvalues[1].mean():.1f},{allvalues[1].std():.1f},{allvalues[3].mean():.1f},{allvalues[3].std():.1f}\n')
