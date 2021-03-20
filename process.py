import numpy as np
import pandas  as pd 
import argparse
import multiprocessing
from multiprocessing import Pool, TimeoutError
import time
import os
import re
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import json
import glob
def subjects2stmulis(DIR: str):
    subjects_stims = {}
    stimulis = []
    parcellations = glob.glob(DIR + r'/sub-*_task-*_space-MNI152NLin2009cAsymres-native.npz')
    regex = r'sub-(.*)_task-(.*)_space-MNI152NLin2009cAsymres-native.npz'
    for  fmri in parcellations:
        subject, stim = re.findall(regex, fmri)[0]
        stimulis.append(stim)
        subjects_stims[subject].append(stim) if len(subjects_stims.get(subject, []))>0  else subjects_stims.update({subject: [stim]}) 
    return subjects_stims, stimulis

def process_audio(path):
        if os.path.exists('np_stims/' + path.split('/')[-1].split('.')[0]+'.npy'): 
            print(f'File {path} Already saved')
        else : 
            print(path)
            resample = T.Resample(new_freq = 11100 )
            waveform, sample_rate = torchaudio.load(path)
            resample.orig_freq = sample_rate
            wv_resampled = resample(waveform).numpy()
            print('save', path)
            np.save(DATA_DIR + path.split('/')[-1].split('.')[0], wv_resampled)

def get_metadata(stimuli):
    df = pd.DataFrame()
    df['paths'] = stimuli
    df['sample_rate']= df['paths'].apply(lambda r : torchaudio.info(r)[0].rate);
    df['length']= df['paths'].apply(lambda r : torchaudio.info(r)[0].length);
    df['N_channels']= df['paths'].apply(lambda r : torchaudio.info(r)[0].channels);
    df.index = df['paths'].apply(lambda r : r.split('/')[-1])
    return df.drop(columns = ['paths']

if __name__ == "__main__":

    DIR = 'parcellation'
    stimuli = glob.glob('stimuli/*.wav'); 
    subjects_stims, _ = subjects2stmulis(DIR )
    with open('mapping.json', 'w') as f :
        json.dump(subjects_stims, f)
    df = get_metadata(stimuli)
    parser = argparse.ArgumentParser('Processing fmri data')
    parser.add_argument('-d','--save_dir',help = "Dir ", type=str, default= './')
    DATA_DIR = parser.parse_args().save_dir
    with Pool() as pool:
        pool.map_async(process_audio, df.paths).get(60*8)
    #vx = np.load('data/fmri/raw/' + file, allow_pickle =True)['X']
    #np.save('data/fmri/processed/sub_1.npy', vx)