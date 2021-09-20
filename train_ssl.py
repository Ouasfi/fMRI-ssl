#%%
from training import get_args, get_learner, ssl_loaders
#%%
import yaml
import json
import sys, os
#%%
if __name__ == "__main__":

    # %%
    args = get_args()
    mapping = json.load(open('metadata/mapping.json', 'r'))
    subjects = [sub for sub in mapping if sub !="139" ]
    if args.subject is not None:
        train_subjects = test_subjects = [args.subject]
    else:
        split = len(subjects)//2+1
        train_subjects, test_subjects = subjects[:split], subjects[split:]
    
    ## Dataloaders
    train_loader, val_loader = ssl_loaders(train_subjects, test_subjects, args)
    learn = get_learner(train_loader, val_loader, args)
    # %%
 
    learn.fit(args.epochs, args.lr)
    learn.save(args.m+'_')


   