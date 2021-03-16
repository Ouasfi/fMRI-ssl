### script to prepare the narratives data
### This will take all the preprocessed MNI152NLin2009cAsym files, parcellate them on the STG_middle ROIs from MIST and remove noise confounds

import os

from nilearn.input_data import NiftiMasker

from load_confounds import Params24,Params36
### See here for details on fmriprep confounds https://fmriprep.org/en/stable/outputs.html#confounds
### and here to see which ones are loaded with https://github.com/SIMEXP/load_confounds#predefined-denoising-strategies

import os,sys
import numpy as np

import pandas as pd 
from matplotlib import pyplot as plt 
from nilearn.plotting import plot_matrix
from datalad import api


### Check if the STG ROI Mask is already calculated or not 

if os.path.isfile('parcellation/STG_middle.nii.gz'):
    print("Mask was already computed")
else:
    print("Computing parcellation mask for bilateral STG middle")
    from parcellation.MIST_auditorymasks import allrois

### first install the fmriprep derivatives using datalad 
### `datalad install -r datalad -r install ///labs/hasson/narratives/derivatives/fmriprep`

### Path to fmriprep derivatives folder 
basepath = '/media/nfarrugi/datapal/narratives/fmriprep/'

savepath = '/media/nfarrugi/datapal/narratives/parcellated'

def clean_parcel(filepath_fmri,roimask,save=False,savepath='./results',visualize=False):
    ## From the filepath_fmri, deduce the tsvfile 
    
    filedir,filebase = os.path.split(filepath_fmri)    
    filebasesplit = filebase.split('_')
    tsvfile_fmri = os.path.join(filedir, filebasesplit[0] + '_' + filebasesplit[1] + '_' + 'desc-confounds_regressors.tsv')
    print(tsvfile_fmri)

    ## fetch the files if they are not here already 
    api.get(tsvfile_fmri,dataset=api.Dataset(basepath))
    api.get(filepath_fmri,dataset=api.Dataset(basepath))

    
    resultfile = os.path.join(savepath,filebasesplit[0] + '_' + filebasesplit[1] + '_'  + filebasesplit[2] + filebasesplit[3] + '.npz')
    print(resultfile)

    mymasker = NiftiMasker(mask_img=roimask,standardize=False,detrend=False,smoothing_fwhm=None)

    mymasker.fit()
    if os.path.isfile(resultfile):
        print("File {} already exists".format(resultfile))
        X = np.load(resultfile)['X']
    else:

        

        # Load the confounds using the strategy , see here for all strategies https://github.com/SIMEXP/load_confounds#predefined-denoising-strategies
        confounds = Params36().load(tsvfile_fmri)
        ## Apply the masker 

        X = mymasker.fit_transform(filepath_fmri,confounds=confounds)

        ## Normalize after the masking 
        from sklearn.preprocessing import normalize
        X = normalize(X,axis=0) ### axis = 0 normalizes the features independently, so each voxel taken independently will fill N(0,1)
        
        if save:
            os.makedirs(savepath,exist_ok=True)
            
            np.savez_compressed(resultfile,X=X)

    if visualize:
        f,ax = plt.subplots(nrows=2,ncols=1,squeeze=True)
        ax[0].plot(X.mean(axis=1))
        from nilearn.plotting import plot_img,plot_stat_map
        from nilearn.image import mean_img        
        parcellated_mean = (mymasker.inverse_transform(X.mean(axis=0)))
        plot_stat_map(parcellated_mean,figure=f,axes=ax[1])
        plt.show()
        
    return X

for s in os.walk(basepath):
    try:
        curdir = s[0]
        if (curdir[-4:]=='func'):
            print(curdir)
            print('list of files : ')
            
            for curfile in s[2]:
                curid = (curfile.find('MNI152NLin2009cAsym_res-native_desc-preproc_bold.nii.gz'))
                
                if curid >0:
                    print('Parcellating file ' + os.path.join(curdir,curfile))
                    X = clean_parcel(os.path.join(curdir,curfile),roimask='parcellation/STG_middle.nii.gz',save=True,savepath=savepath,visualize=False)        
                    print("Shape is {}".format(X.shape))        
    except Exception as e:
        print("Error with file {}".format(curfile))
        print(e)
