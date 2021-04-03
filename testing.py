import RP_dataset as rpd
import model as md
import torch
from torch.nn.functional import soft_margin_loss
import json
import pandas as pd
from datalad import api
import sys, os
import glob

from fastai.data.transforms import get_image_files
from fastai.data.core import DataLoaders
from torch.nn.functional import soft_margin_loss
import PIL
import re
import random
from fastai.vision.all import *
if __name__ == "__main__":
    
    mapping = json.load(open('metadata/mapping.json', 'r'))
    meta =  pd.read_csv('metadata/participants.csv', sep = '\t'); meta.index = meta['participant_id']
    subjects = [sub for sub in mapping if sub !="139" ]#if 'schema' not in meta.loc[f'sub-{sub}'].task.split(',') ]
    ## Dataset params
    sampling_params = (1,30)
    wind_len = 15
    sr = 11100
    batch_size = 100
    train_size = 80000 #total number of training windows
    val_size = 800 
    weights = [0.6]*2
    n_subjects = 50 # number of subjects per batch
    trainset =  rpd.RP_Dataset_Multi( subjects,
                                sampling_params= sampling_params, \
                                wind_len = wind_len,
                                sr = sr,  
                                mode = 'train')#en Sec

    sampler = rpd.RPSampler(trainset,
                        batch_size = batch_size,
                        size = train_size,
                        weights = weights,
                        n_subjects = n_subjects)
    train_loader = torch.utils.data.DataLoader(trainset,
                                            batch_size=batch_size, 
                                            num_workers=15,
                                            sampler = sampler,
                                            collate_fn=rpd.ssl_collate)

    val_generator = rpd.RP_Dataset_Multi( 
                                        subjects = subjects, 
                                        sampling_params= sampling_params,
                                        wind_len = wind_len ,
                                        mode = "val",
                                        sr = sr
                                        )
    val_sampler = rpd.RPSampler(val_generator, batch_size = batch_size,size = val_size
    ,  weights = weights, n_subjects = n_subjects)
    val_dataset = torch.utils.data.Subset(val_generator, indices= list(val_sampler))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, 
                                         num_workers=15, collate_fn=rpd.ssl_collate)
    # # %%
    # val_list = list(val_loader)
    # for i, data in enumerate(val_loader):
    #     assert abs(data[0][0] - val_list[i][0][0]).sum() == 0
    # print('\nLoaders sucessfully initialised \n')
    # %%
    print('- Audio encoder ...\n')
    m = md.AudioEmbed()
    print(m)
    (x_fmri, x_audio), y = next(iter(val_loader))
    B = x_fmri.shape[0]
    assert m(x_audio.float()).shape == torch.Size([B, 32,10]) , 'Size mismatch !'
    print('Keys sucessfully matched!')
    # %%
    print('- FMRI encoder ...\n')
    m = md.FMRIEmbed(voxels = 556)
    print(m)
    (x_fmri, x_audio), y = next(iter(val_loader))
    assert m(x_fmri.float()).shape == torch.Size([B, 32,10])
    print('Keys sucessfully matched!')
    #%%
    print('- Siamese model ...\n')
    m = md.SiameseModel_2(hidden_dim = 32*10, voxels = 556)
    print(m)
    (x_fmri, x_audio), y = next(iter(val_loader))
    #assert m((x_fmri.float(), x_audio)).shape == torch.Size([30, 2]), 'Size mismatch !'
    #assert soft_margin_loss(m((x_fmri.float(), x_audio)), y).requires_grad == True, 'Broken computational graph !'
    print('Keys sucessfully matched!')

    model = m.to(float)
    model.float()
    # loss_fn 
    loss_fn = CrossEntropyLossFlat() #soft_margin_loss
    # fastai learner
    dls = DataLoaders(train_loader, val_loader) # a validation set need to be defined
    learn = Learner(dls, model, loss_func=loss_fn, opt_func=Adam, metrics=[accuracy, F1Score(), Precision(), Recall()], 
                    cbs = [md.ToDevice()])
    learn.unfreeze()
    learn.fit(500, 3e-04)
    learn.export()