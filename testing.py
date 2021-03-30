import RP_dataset as rpd
import model as md
import torch
from torch.nn.functional import soft_margin_loss
import json
import pandas as pd
from datalad import api
import sys, os
import glob
if __name__ == "__main__":
    mapping = json.load(open('metadata/mapping.json', 'r'))
    meta =  pd.read_csv('metadata/participants.csv', sep = '\t'); meta.index = meta['participant_id']
    subjects = [sub for sub in mapping ]#if 'schema' not in meta.loc[f'sub-{sub}'].task.split(',') ]
    val_generator = rpd.RP_Dataset_Multi( subjects = subjects, 
                                        sampling_params= (1,30),
                                         wind_len = 15 ,
                                         mode = "val",
                                         sr = 11100)
    val_sampler = rpd.RPSampler(val_generator, batch_size = len(subjects),size = 6000
    ,  weights = [0.5]*2)
    val_dataset = torch.utils.data.Subset(val_generator, indices= list(val_sampler))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=30, 
                                         num_workers=0, collate_fn=rpd.ssl_collate)
    # %%
    val_list = list(val_loader)
    for i, data in enumerate(val_loader):
        assert abs(data[0][0] - val_list[i][0][0]).sum() == 0
    print('\nLoaders sucessfully initialised \n')
