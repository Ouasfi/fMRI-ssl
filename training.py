#%%
from importlib import reload
import RP_dataset as rpd; #reload(rpd)
import model as md
import train

import torch
import json
import pandas as pd
#from datalad import api
import sys, os
import glob

from fastai.data.transforms import get_image_files
from fastai.data.core import DataLoaders
from torch.nn.functional import soft_margin_loss
import PIL
import re
import random
from fastai.vision.all import *
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score

import configargparse
#%%
def ssl_loaders(train_subjects, test_subjects, args):
    sr = 11100
    trainset =  rpd.RP_Dataset_Multi( train_subjects,
                            sampling_params= (args.pos, args.neg), \
                            wind_len = args.wind_len,
                            sr = sr,  
                            mode = 'train',
                            scenario = args.scenario )

    sampler = rpd.RPSampler(trainset,
                        batch_size = args.batch_size,
                        size = args.train_size,
                        weights = [args.weights,1-args.weights] ,
                        n_subjects = args.n_subjects)
    train_loader = torch.utils.data.DataLoader(trainset,
                                            batch_size=args.batch_size, 
                                            num_workers=2,
                                            sampler = sampler,
                                            collate_fn=rpd.ssl_collate)

    val_generator = rpd.RP_Dataset_Multi( 
                                        subjects = test_subjects, 
                                        sampling_params= (args.pos, args.neg),
                                        wind_len = args.wind_len ,
                                        mode = "val",
                                        sr = sr,
                                        scenario = args.scenario
                                        )
    val_sampler = rpd.RPSampler(val_generator, batch_size = args.batch_size,size = args.val_size
    ,  weights = [args.weights,1-args.weights], n_subjects = args.n_subjects)
    val_dataset = torch.utils.data.Subset(val_generator, indices= list(val_sampler))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, 
                                         num_workers=2, collate_fn=rpd.ssl_collate)
    return train_loader, val_loader
def get_learner(train_loader, val_loader, args):
    model = md.SiameseModel_corr(hidden_dim = 32*10, n_classes = 2,voxels = 556).float()
    # loss_fn 
    loss_fn = md.CrosscCorrLoss(hidden_dim = 32*10,
                                scale =args.scale ,
                                lambd = args.reg).cuda()
    # fastai learner
    dls = DataLoaders(train_loader, val_loader)
    def siamese_splitter(model):
        return [params(model), params(loss_fn)]
    learn = Learner(dls,
                    model,
                    loss_func=loss_fn,
                    opt_func=Adam,
                    metrics=[], 
                    splitter = siamese_splitter,
                    cbs = [md.ToDevice()])
    learn.unfreeze()
    return learn

#%%
if __name__ == "__main__":

    # %%
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')


    # General training options
    p.add_argument('--batch_size', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
    p.add_argument('--scale', type=float, default=0.01, help='scale')
    p.add_argument('--reg', type=float, default=1e-2, help='Regularization term')
    p.add_argument('--epochs', type=int, default=50,
                help='Number of epochs to train for.')
    #Dataset options
    p.add_argument('--pos', type=int, default=1,
                help='Number of epochs to train for.')
    p.add_argument('--neg', type=int, default=60,
                help='Number of epochs to train for.')
    p.add_argument('--wind_len', type=int, default=15,
                help='Len of sampling windows.')
    #Dataloader options
    p.add_argument('--train_size', type=int, default=100000, help='total number of training windows')
    p.add_argument('--val_size', type=int, default=1000, help='val_size')
    p.add_argument('--weights', type=float, default=0.5, help='sampling weights')
    p.add_argument('--n_subjects', type=float, default=10, help='number of subjects per batch')
    p.add_argument('--mode', type=str, default='encoding', help='ssl or encoding')
    p.add_argument('--scenario', type=str, default='subjects', help='subjects or stims')
    p.add_argument('--m', type=str, default='cor_model_50_2', help='Model pathname')
    
    args = p.parse_args()
    mapping = json.load(open('metadata/mapping.json', 'r'))
    subjects = [sub for sub in mapping if sub !="139" ]
    
    split = len(subjects)//2+1
    train_subjects, test_subjects = subjects[:split], subjects[split:]
    ## Dataloaders
    train_loader, val_loader = ssl_loaders(train_subjects, test_subjects, args)

    learn = get_learner(train_loader, val_loader, args)
    # %%
   
    if args.mode == 'ssl':
        learn.fit(args.epochs, args.lr)
        learn.save(args.m+'_')
        learn.fit_one_cycle(20, 1e-04)
        learn.save(args.m+'_20')

    if args.mode == 'encoding':
        learn.load(args.m)

   # %%
    for i in range(10):
        try:
            test_subject = np.random.choice(test_subjects)#
            test_stim = np.random.choice( mapping[test_subject])
            

            print('Subject:', test_subject, 'Stimuli:' , test_stim)
            print()
            (z_train, y_train), (z_test, y_test) = train.get_regression_data(learn, train_loader.dataset, test_subject, 
                                                    test_stim, batch_size = 300 )

            
            clf, tr_scores, val_scores = train.ridge_encoding (z_train, y_train,z_test,  y_test )
            print("Train scores:")
            print('------------')
            print(" Mean: ", tr_scores.mean(), "Max;", tr_scores.max())
            print(f"R2 scores in {(tr_scores>0.).mean():.2f}% of regions higher than {0.}")
            print(f"R2 scores in {(tr_scores>0.1).mean():.2f}% of regions higher than {0.1}")
            print("Test scores:")
            print('-----------')
            print(" Mean: ", val_scores.mean(), "Max;", val_scores.max())
            print(f"R2 scores in {(val_scores>0.).mean():.2f}% of regions higher than {0.}")
            print(f"R2 scores in {(val_scores>0.1).mean():.2f}% of regions higher than {0.1}")
            print('--------------------------------------------------')
            print  ()      
        except:
            print(':/')

    # %%


# %%
#learn.save('cor_model_50_2')
    #learn.fit(50, 1e-04)
    #learn.save('cor_model_100_2')
    #learn.fit_one_cycle(50, 1e-03)
    #learn.save('cor_model_50')
    #learn.fit_one_cycle(50, 1e-04)
    #learn.save('cor_model_100')
    #learn.load('cor_model_50_2')