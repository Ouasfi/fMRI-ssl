import numpy as np
import os
from pylab import *
import torch 

from fastai.data.core import DataLoaders
import re
import random
from fastai.vision.all import *
from glob import glob
import sys
AUDIODIR = 'data/stimuli'
FMRIDIR = 'data/parcellated'


class RPSampler(torch.utils.data.sampler.Sampler):
    r"""Pour chaque subject sample n windows de sorte à avoir un dataset équilibré en 
    terme de subject. `weights` permet de controler la proportion des negative samples.

    Arguments:
    ---------
        dataset (Dataset): dataset to sample from
        size (int): The total number of sequences to sample
    returns:
        a tuple (t,target,subject) such as :
        - The anchor window starts at index t in the fmri recording.
        - The target is the class associeted to the pretexe task (eithre -1 or 1)
    """

    def __init__(self,dataset, batch_size,size,  weights):
    
        
        self.batch_size = batch_size
        self.size = size
        self.dataset = dataset
        self.serie_len = 0
        self.n_subjects = len(self.dataset.subjects); 
        self.weights = torch.DoubleTensor(weights)
        self.f = self.dataset.f # fin de la serie temporelle
        self.d = self.dataset.d
        self.tr = self.dataset.tr_
        
    def __iter__(self):
        num_batches = self.size// self.batch_size
        n_subject_samples = self.batch_size //self.n_subjects
        while num_batches > 0:
            if num_batches % 100 == 0 :
                print("batches restants :",num_batches)
            #iterate on each subject in the dataset
            for subject in self.dataset.subjects:
                sampled = 0
                #sample `n_subject_samples` per subject
                while sampled < n_subject_samples:
                    # each sample is a target and an anchor window. Positive or/and negative windows are sampled in the Dataset class. 
                    target  = 2*torch.multinomial(
                self.weights, 1, replacement=True) -1
                    t = np.random.rand() if not isinstance(self.dataset, RP_Dataset) else choice(arange(self.d, self.f-int(self.dataset.w/self.tr), 1))
                    sampled += 1
                    yield (t,target,subject)
            
            num_batches -=1

    def __len__(self):
        return self.size 

#@title Relative positioning class
T = 947
class Abstract_Dataset(torch.utils.data.Dataset):
    '''
    Classe dataset  pour les differents sampling
    '''
    def __init__(self, subjects, wind_len , n_features, T = 947):
        self.subjects = subjects
        self.time_series = []
        self.w = wind_len
        self.n = n_features
        self.T = T
    def get_windows(self,index):
        '''
        a method to load  a sequence 
        '''
        raise NotImplementedError
    def get_pos(self, t_anchor):
        '''
        a method to get positive samples
        '''
        raise NotImplementedError
    def load_ts(self, index):
        '''
        a method to get positive samples
        '''
        raise NotImplementedError
    def get_neg(self, t_anchor):
        '''
       a method to get negative samples
        '''
        raise NotImplementedError
    def get_targets(self, index):
        '''
        a method to get labels
        '''
        raise NotImplementedError
    def __getitem__(self, index):
        windows = self.get_windows(index)
        target = self.get_targets(index)
        return windows, target
    def __len__(self): return self.T

class RP_Dataset(Abstract_Dataset):
    r"""Pour chaque subject sample n windows de sorte à avoir un dataset équilibré en 
    terme de subject. `weights` permet de controler la proportion des negative samples.

    Arguments:
    ---------
        subjects (List): list of subjects  to use during training
        sampling_params (tuple(int,int)): positive and negative sampling windows
        wind_len (int): Windows lenght
        debut(int): Starting indice of the considered time series.
    """
    def __init__(self, subjects, sampling_params, wind_len , debut = 0, fin = 946, dry_run = False, tr=1.4 ):
        
        super().__init__(subjects, wind_len = wind_len, n_features = 3)
        self.tr = tr
        self.audio, self.sr = self.load_audio()
        self.pos , self.neg = sampling_params[0]*self.sr, sampling_params[1]*self.sr
        self.d, self.f = debut, fin #in tr
        self.d_audio , self.f_audio = int(self.d*self.tr*self.sr), int(self.f*self.tr*self.sr)
        self.dry_run = dry_run
    @property
    def tr_(self):
        return self.tr
    def get_windows(self,index):
        '''
        a method to get sampled windows
        '''
        #fmri index
        (t, target,subject) = index
        # sample a positive or negative audio index
        t_ = self.get_pos(t) if target>0 else self.get_neg(t)
        if self.dry_run:
            return (t, t_)
        #load fmri data
        fmri =self.load_fmri(subject)
        # slice 
        fmri_w = fmri[t:t+int(self.w/self.tr)] # fmri index*TR -->seconds
        # sample a positive or negative audio window
        audio_w = self.audio[t_:t_+self.w*self.sr] # could be negative or positive
        return (fmri_w, audio_w)
    
    def load_audio(self):
        return np.load( os.path.join(AUDIODIR, 'sherlock_audio.npy'),mmap_mode = "c" ), 11100
    def load_fmri(self, subject):
        path_= os.path.join(FMRIDIR, f"sub-{subject}_task-sherlock_space-MNI152NLin2009cAsymres-native.npz")
        return np.load(path_,mmap_mode = "c")['X']
    def get_targets(self, index):
        return (index[1]>0.5)*1
    def get_pos(self, t_anchor):
        w = self.w*self.sr #frmi  to audio window lenght
        t = int(t_anchor*self.tr*self.sr) #frmi indice to audio
        start = max(self.d_audio,t-self.pos ) 
        end = min(self.f_audio - w-1,t+self.pos) # to get a sequence of lenght self.w
        t_ = choice(arange(start,end, 1)) 
        return t_
    def get_neg(self, t_anchor):
        w = self.w*self.sr
        t = int(t_anchor*self.tr*self.sr)
        left_idx = arange(self.d_audio, max(self.d_audio, t - self.neg), 1)
        right_idx =arange(min(self.f_audio-w-1, t + self.neg-1),self.f_audio-w-1 ,1)
        t_ = choice(hstack([left_idx, right_idx])) # 
        return t_

def ssl_collate(batch):
    anchors = torch.stack([torch.from_numpy(item[0][0]) for item in batch])
    try:
        sampled = torch.stack([torch.from_numpy(item[0][1]) for item in batch])
    except:
        print("error")
    targets = torch.stack([item[1] for item in batch])
    
    return (anchors, sampled), targets

class RP_Dataset_Multi(Abstract_Dataset):
    r"""Pour chaque subject sample n windows de sorte à avoir un dataset équilibré en 
    terme de subject. `weights` permet de controler la proportion des negative samples.

    Arguments:
    ---------
        subjects (List): list of subjects  to use during training
        sampling_params (tuple(int,int)): positive and negative sampling windows
        wind_len (int): Windows lenght
        debut(int): Starting indice of the considered time series.
    """
    def __init__(self, subjects, sampling_params, wind_len , debut = 0,\
                 fin = None, dry_run = False,sr = 22050, tr=1.5, mode = 'train' ):
        
        super().__init__(subjects, wind_len = wind_len, n_features = 3)
        self.tr, self.sr = tr , sr
        self.tr_ = self.tr
        self.pos , self.neg = sampling_params[0]*self.sr, sampling_params[1]*self.sr
        self.sub2stims = json.load(open('metadata/mapping.json', 'r'))
        self.meta =  pd.read_csv('metadata/participants.csv', sep = '\t'); self.meta.index = self.meta['participant_id']
        self.events = json.load(open('metadata/events.json', 'r'))
        self.excluded = ['piemanpni', 'schema']
        #self.sub2stims = {key: value.__iter__() for key, value in self.sub2stims.items()}
        self.d, self.f = debut, fin #in tr
        self.dry_run = dry_run
        self.mode = mode
    def get_windows(self,index):
        '''
        a method to get sampled windows
        '''
        #fmri index
        (t, target,subject) = index
        # select a stim
        stim = self.select_stimuli(subject)
        fmri_onset = self.events[stim.split('_')[0]].get(stim.split('_')[0], {'onset': 3})['onset']
        # load audio
        audio = self.load_audio( subject, stim)
        self.d_audio , self.f_audio = 0, audio.shape[1]
        #define tr and audio intervals 
        self.f = int(self.f_audio/(self.tr*self.sr))+fmri_onset
        #load fmri data
        fmri = self.load_fmri(subject, stim)[fmri_onset:self.f]
        #define tr and audio intervals 
        #rescale t`
        #fmri_index*TR -->seconds
        w_tr = int(np.floor(self.w/self.tr))
        end, start =  self.f-2*w_tr, self.d
        t = int(t*(end-start+1) + start)
        # slice fmri window 
        # use self.tr_ to have a fixed window length(instead of using pad_sequence in collate_fn)
        fmri_w = fmri[t:t+w_tr]
        assert  fmri_w.shape[0]== 10 , f'{subject}, {stim},{ fmri_w.shape[0]}, {w_tr}, {end} , {fmri.shape}'
        # sample a positive or negative audio index
        try : 
            t_ = self.get_pos(t) if target>0 else self.get_neg(t)
        except:
            print(stim, subject, t, self.f_audio, self.f, fmri_onset)
            sys.exit()
        if self.dry_run:
            return (t, t_)
        # sample a positive or negative audio window
        audio_w = audio[0,t_:t_+self.w*self.sr]
        return (fmri_w, audio_w)
    def select_stimuli(self,subject):
        # find  and filter all available  stimuli
        #print(subject)
        stims = [stim for stim in self.sub2stims[subject] if stim.split('_')[0] not in self.excluded]
        #print(stim.split('_')[0])
        index = randint(low = 0, high = len(stims[:-1])) if self.mode == 'train' and len(stims)>1 else -1
        # select a stim randomly during training 
        return stims[index]
    def load_audio(self, subject, stim):
        stim = stim + self.get_condition(subject, stim) if stim == "milkyway" else stim.split('_')[0] #stim_run-1
        stim_path = os.path.join(AUDIODIR, f'{stim}_audio.npy')
        return np.load( stim_path,mmap_mode = "c" )
    def load_fmri(self ,subject ,stim ):
        path_= os.path.join(FMRIDIR,  f'sub-{subject}_task-{stim}_space-MNI152NLin2009cAsymres-native.npz')
        return np.load(path_,mmap_mode = "c")['X']
    def get_targets(self, index):
        return (index[1]>0.5)*1
    def get_pos(self, t_anchor):
        # convert `self.w` from seconds to the number of audio frames
        w = self.w*self.sr 
        # convert `t` from tr to the number of audio frames
        t = int(t_anchor*self.tr*self.sr)
        # self.d_audio and self.f_audio are updated in get_window to adapt to audio length
        start = max(self.d_audio ,t-self.pos ) 
        end   = min(self.f_audio-w-1 , t+self.pos) # to get a sequence of lenght self.w
        t_    = choice(arange(start , end, 1))
        return t_
    def get_neg(self, t_anchor):
        w = self.w*self.sr
        t = int(t_anchor*self.tr*self.sr)
        left_idx = arange(self.d_audio, max(self.d_audio, t - self.neg), 1)
        right_idx =arange(min(self.f_audio-w-1, t + self.neg-1),self.f_audio-w-1 ,1)
        t_ = choice(hstack([left_idx, right_idx])) # 
        return t_
    def get_condition(self, subject, stim):
        metadata = self.meta.loc[f'sub-{subject}']
        return metadata.condition.split(',')[metadata.task.split(',').index(stim)]