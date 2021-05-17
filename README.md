### Parameter fitting for the MESS sequence
In this repository we provide the code necessary to fit parameters to images acquired with our Multiple-Echo Steady-State (MESS) MRI acquisition, to accompany our manuscript "Multiple-echo steady-state (MESS): Extending DESS for joint T2 mapping and chemical-shift corrected water-fat separation" (currently under review).

### Installation
To run our code, you need to have python 3 installed, along with the following packages:
- numpy
- scipy
- pytorch (version 1.8 or higher)
- skimage
- h5py
- matplotlib
- nibabel

The easiest way to accomplish this is to install Anaconda (https://www.anaconda.com/products/individual#Downloads) and run the following in the Anaconda command prompt:
`conda install pytorch -c pytorch`
`conda install h5py`
`pip install nibabel`

### Running the experiments
To run our experiments and recreate the image for the figures in the manuscript, first you need to download the DESS and MESS images from our OSF repository: https://osf.io/896rw/
Please ensure the *.h5 files are placed in the `data` folder.

To fit the parameters to the images, run the following:
`python run_mess.py`
`python run_dess_pseudo_replications.py`
`python run_mess_pseudo_replications.py`
Note that the MESS pseudo-replications take over 4 hours on a decent GPU and is needed only to create Figures 4 and S1, and Table S2. If you only want Figure S1, `n_runs` could be changed to 0 in the script to make it finish a lot faster.

After fitting, the data for our figures and tables can be reproduced by running (replacing XXX for the desired Figure or Table):
`python create_XXX.py`
Results will be produced in the images folder.

Note that the code is designed to be run on a NVIDIA GPU, but will fall back to CPU if this is not available. However, fitting on CPU is much slower.
