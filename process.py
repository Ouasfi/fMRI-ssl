#%%
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

        

#%%
def subjects2stmulis(DIR: str):
    subjects_stims = {}
    stimulis = []
    parcellations = glob.glob(DIR + r'/sub-*_task-*_space-MNI152NLin2009cAsymres-native.npz')
    regex = r'sub-(.*)_task-(.*)_space-MNI152NLin2009cAsymres-native.npz'
    excludes = json.load(open("data/exclude_scans.json", 'r'))
    for  fmri in parcellations:
        subject, stim = re.findall(regex, fmri)[0]
        stimulis.append(stim)
        if f'sub-{subject}' not in excludes.get(stim.split('_')[0], []):
            subjects_stims[subject].append(stim) if len(subjects_stims.get(subject, []))>0  else subjects_stims.update({subject: [stim]}) 
    return subjects_stims, stimulis
#json.dump(subjects2stmulis('data/parcellated' )[0], open('metadata/mapping.json', 'w'))
#%%
def process_audio(path):
        if os.path.exists(AUDIODIR+ path.split('/')[-1].split('.')[0]+'.npy'): 
            print(f'File {path} Already saved')
        else : 
            print(path)
            resample = T.Resample(new_freq = 11100 )
            waveform, sample_rate = torchaudio.load(path)
            resample.orig_freq = sample_rate
            wv_resampled = resample(waveform).numpy()
            np.save(AUDIODIR + path.split('/')[-1].split('.')[0], wv_resampled)
            print('Saved:', path)

def get_metadata(stimuli):
    df = pd.DataFrame()
    df['paths'] = stimuli
    df['sample_rate']= df['paths'].apply(lambda r : torchaudio.info(r).sample_rate);
    df['length']= df['paths'].apply(lambda r : torchaudio.info(r).num_frames);
    df['N_channels']= df['paths'].apply(lambda r : torchaudio.info(r).num_channels);
    df.index = df['paths'].apply(lambda r : r.split('/')[-1])
    return df#.drop(columns = ['paths'])

def parse_events(subject, stim_run):
    """
    Parse events for Schema dataset
    """
    tr, sr  = 1.5, 11100
    #load events tsv
    events = pd.read_csv(f'data/narratives/{subject}/func/{subject}_task-{stim_run}_events.tsv', sep = '\t')
    events = events.dropna(axis = 0, subset = ['stim_file'])
    events.index = events.stim_file
    stim, run = stim_run.split('_') # eg. schema_run-4
    #load fmri data
    fmri = np.load(f'{FMRI_DIR}/{subject}_task-{stim_run}_space-MNI152NLin2009cAsymres-native.npz', \
        mmap_mode = 'c')['X']
    # for each stim crop the correspanding fmri window
    for stimuli in events.index:
        stim = stimuli.split('_')[0]+ '_'+ run
        onset = int(events.loc[stimuli].onset/tr)
        offset = int((events.loc[stimuli].duration + events.loc[stimuli].onset)/tr)
        fmri_ = fmri[onset:offset]
        duration = np.floor(np.load(f'{AUDIODIR}/{stimuli.split(".")[0]}.npy', mmap_mode = 'c').shape[1]/(sr*tr))
        assert abs(fmri_.shape[0] - duration)<tr, f'fmri {fmri_.shape[0]} duration {duration}'
        np.savez(f'{FMRI_DIR}/{subject}_task-{stim}_space-MNI152NLin2009cAsymres-native',fmri_)
    return 
if __name__ == "__main__":

    parser = argparse.ArgumentParser('Processing fmri data')
    parser.add_argument('-d','--save_dir',help = "Dir ", type=str, default= 'data/stimuli/')
    AUDIODIR = parser.parse_args().save_dir
    FMRI_DIR = 'data/parcellated'

    stimuli = glob.glob('data/stimuli/*.wav')
    df = pd.read_csv('metadata/participants.csv', sep = '\t')
    df.index = df['participant_id']
    for i in range(1,5): 
        list(map(lambda x : parse_events(x, stim_run = f'schema_run-{i}') \
            , df[df['task'].apply(lambda r : 'schema' in r.split(','))].participant_id ))
    
    json.dump(subjects2stmulis(FMRI_DIR )[0], open('metadata/mapping.json', 'w'))
    df = get_metadata(stimuli)
    
    with Pool() as pool:
        pool.map_async(process_audio, df.paths).get(60*8)

    
    #vx = np.load('data/fmri/raw/' + file, allow_pickle =True)['X']
    #np.save('data/fmri/processed/sub_1.npy', vx)