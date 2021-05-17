import nibabel as nib
import numpy as np

def load(filename):
    return np.array(nib.load(filename).dataobj).transpose(2,1,0)

n_runs = 10

with open('./images/tableS2_noise.csv', 'w') as f_noise:
    with open('./images/tableS2_SNR.csv', 'w') as f_snr:
    
        f_noise.write('P,noise_overall,noise_w,noise_f\n')
        f_snr.write('P,snr_dess,snr_mess_w,snr_mess_f,snr_dess_w,snr_mess_w_w,snr_mess_f_f\n')

        f_noise.write('All,DESS,100%,,\n')

        noise_dess_avg = 0
        noise_mess_w_avg = 0
        noise_mess_f_avg = 0
        
        noise_mess_w_w_avg = 0
        noise_mess_w_f_avg = 0
        noise_mess_f_w_avg = 0
        noise_mess_f_f_avg = 0

        snr_dess_avg = 0
        snr_mess_w_avg = 0
        snr_mess_f_avg = 0

        snr_dess_w_avg = 0
        snr_mess_w_w_avg = 0
        snr_mess_f_f_avg = 0

        for P in [1,2,3,4,5]:
            dirname = f'./results/mess_P{P}/'
            dirname_dess = f'./results/dess_P{P}/'
            
            ref_dess = load(dirname_dess + 'w_run0.nii.gz')
            ref_mess = load(dirname + 'w_run0.nii.gz')
            ref_mess_f = load(dirname + 'f_run0.nii.gz')
            
            reps_dess = np.zeros((n_runs,) + ref_dess.shape, dtype=ref_dess.dtype)
            reps_mess = np.zeros((n_runs,) + ref_mess.shape, dtype=ref_mess.dtype)
            reps_mess_f = np.zeros((n_runs,) + ref_mess_f.shape, dtype=ref_mess_f.dtype)
            
            # Load data
            for run in range(n_runs):
                print('Loading', P, run+1)
                reps_dess[run] = load(dirname_dess + f'w_run{run+1}.nii.gz')
                reps_mess[run] = load(dirname + f'w_run{run+1}.nii.gz')
                reps_mess_f[run] = load(dirname + f'f_run{run+1}.nii.gz')
            
            # Calculate statistics
            sd_dess = reps_dess.std(axis=0)
            sd_mess = reps_mess.std(axis=0)
            sd_mess_f = reps_mess_f.std(axis=0)
            
            snr_dess = abs(reps_dess.mean(axis=0))/sd_dess
            snr_mess = abs(reps_mess.mean(axis=0))/sd_mess
            snr_mess_f = abs(reps_mess_f.mean(axis=0))/sd_mess_f
            
            mean_sd_dess = sd_dess.mean()
    
            # Calculate water-dominant and fat-dominant masks
            dess_mask = abs(ref_dess)>0.2
            mess_mask_w = (abs(ref_mess)>abs(ref_mess_f)) & ((abs(ref_mess)+abs(ref_mess_f))>0.2)
            mess_mask_f = (abs(ref_mess_f)>abs(ref_mess)) & ((abs(ref_mess)+abs(ref_mess_f))>0.2)
            
            # Calculate means
            noise_dess = sd_dess.mean() / mean_sd_dess
            noise_mess_w = sd_mess.mean() / mean_sd_dess
            noise_mess_f = sd_mess_f.mean() / mean_sd_dess
            
            noise_mess_w_w = sd_mess[mess_mask_w].mean() / mean_sd_dess
            noise_mess_w_f = sd_mess[mess_mask_f].mean() / mean_sd_dess
            noise_mess_f_w = sd_mess_f[mess_mask_w].mean() / mean_sd_dess
            noise_mess_f_f = sd_mess_f[mess_mask_f].mean() / mean_sd_dess
            
            sr_dess = snr_dess.mean()
            sr_mess_w = snr_mess.mean()
            sr_mess_f = snr_mess_f.mean()
            
            sr_dess_w = snr_dess[dess_mask].mean()
            sr_mess_w_w = snr_mess[mess_mask_w].mean()
            sr_mess_f_f = snr_mess_f[mess_mask_f].mean()
    
            # Calculate averages
            noise_dess_avg += noise_dess/5
            noise_mess_w_avg += noise_mess_w/5
            noise_mess_f_avg += noise_mess_f/5
            
            noise_mess_w_w_avg += noise_mess_w_w/5
            noise_mess_w_f_avg += noise_mess_w_f/5
            noise_mess_f_w_avg += noise_mess_f_w/5
            noise_mess_f_f_avg += noise_mess_f_f/5

            snr_dess_avg += sr_dess/5
            snr_mess_w_avg += sr_mess_w/5
            snr_mess_f_avg += sr_mess_f/5

            snr_dess_w_avg += sr_dess_w/5
            snr_mess_w_w_avg += sr_mess_w_w/5
            snr_mess_f_f_avg += sr_mess_f_f/5
            
            # Write to CSV
            f_noise.write(f'{P},MESS W,{noise_mess_w*100:.0f}%,{noise_mess_w_w*100:.0f}%,{noise_mess_w_f*100:.0f}%\n')
            f_noise.write(f'{P},MESS F,{noise_mess_f*100:.0f}%,{noise_mess_f_w*100:.0f}%,{noise_mess_f_f*100:.0f}%\n')

            f_snr.write(f'{P},{sr_dess:.1f},{sr_mess_w:.1f},{sr_mess_f:.1f},{sr_dess_w:.1f},{sr_mess_w_w:.1f},{sr_mess_f_f:.1f}\n')
            
        f_noise.write(f'mean,MESS W,{noise_mess_w_avg*100:.0f}%,{noise_mess_w_w_avg*100:.0f}%,{noise_mess_w_f_avg*100:.0f}%\n')
        f_noise.write(f'mean,MESS F,{noise_mess_f_avg*100:.0f}%,{noise_mess_f_w_avg*100:.0f}%,{noise_mess_f_f_avg*100:.0f}%\n')

        f_snr.write(f'mean,{snr_dess_avg:.1f},{snr_mess_w_avg:.1f},{snr_mess_f_avg:.1f},{snr_dess_w_avg:.1f},{snr_mess_w_w_avg:.1f},{snr_mess_f_f_avg:.1f}\n')
