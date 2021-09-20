from training import get_args, get_learner, ssl_loaders
import encoding
import torch
import json
import numpy as np
# %%
    
def validate_on_subject(subject,train_stims, test_stim, learn, dataset, ):
    '''a function to train ridge regressor on `train_stims` and compute predictions and r2 scores
     on test stim using the audio feature extractor from `learn` 


    Args:
        subject (str): subject id
        train_stims (list): list of the stimuli used to train ridge regression
        test_stim (str): stimuli to use for validation
        learn (fastai learner): learner object with trained audio and fmri extrators
        dataset (torch.utils.data.Dataset): Dataset object 

    Returns:
        tuple: mean, max train and val scores and val scores
    '''    
    data = [encoding.get_data( dataset, subject, 
                                                stim, learn) for stim in train_stims]
    z_train, y_train  = torch.vstack([x for x,_ in data ]), torch.vstack([y for _,y in data ])
    z_test, y_test = encoding.get_data( dataset, subject, 
                                                test_stim, learn)
    clf, tr_scores, val_scores = encoding.ridge_encoding (z_train, y_train,z_test,  y_test )
    #results = {'train' : {'mean':tr_scores.mean(), 'max': val_scores.max() },'val' : {'mean':val_scores.mean(), 'max': tr_scores.max()} 
    return (tr_scores.mean(),tr_scores.max()), (val_scores.mean(), val_scores.max()), val_scores

#%%
if __name__ == "__main__":
    args = get_args()
    mapping = json.load(open('metadata/mapping.json', 'r')); mapping.pop('139', None)
    subjects = list(mapping.keys())
    train_loader, val_loader = ssl_loaders(subjects, subjects, args)
    learn = get_learner(train_loader, val_loader, args)
    learn.load(args.m)
    subject = subjects[0]#'249'#
    stims = [stim for stim in mapping[subject] if stim.split('_')[0] not in ['schema', 'piemanpni' ]]
    
    
    test_stim = stims.pop(-1)

    tr_results, val_results, val_scores = validate_on_subject(subject,stims[:2], test_stim, learn, train_loader.dataset, )
    print(tr_results)
    print(val_results)
# %%
