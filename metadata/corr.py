#%%
import pandas as pd
from multiprocessing import Pool, TimeoutError
from functools import partial
import scipy.stats
import matplotlib.pyplot as plt 
import numpy as np
def parse_scores(path = 'participants.csv'):
    p = pd.read_csv(path, sep = '\t')
    pp = p.dropna(subset = ['comprehension'])
    d_ = pd.DataFrame()
    for row in pp.transpose():
        c = p.iloc[row]['comprehension'].split(',')
        subject = p.iloc[row]['participant_id'].split('-')[-1]
        for i, score in enumerate(c):
            d = {}
            if score!='n/a':
                d['subject'] = subject
                d['task'] = p.iloc[row]['task'].split(',')[i]
                d['score'] = float(score)
                d_ = d_.append(d, ignore_index = True)
    return d_
def get_scores(scores, comprehension,reduction = 'mean', voxel = None ):
    all_scores = pd.DataFrame()
    for substim in scores:
        subject = substim[:3]
        stim = substim[3:].split('_')[0]
        dd = comprehension[comprehension['subject']== subject]
        ddd = dd[dd['task']==stim]
        ddd['r2'] = scores[substim].drop(columns = "Unnamed: 0").mean() if reduction=='mean' else scores[substim].drop(columns = "Unnamed: 0")[voxel]
        if len(ddd):
            all_scores =all_scores.append(ddd)
    return all_scores

def pearson_corr(scores, comprehension, voxel, reduction = None , filter_stims= False):
    
    all_scores = get_scores(scores, comprehension, reduction = reduction, voxel = voxel)
    all_scores['c'] = all_scores['task'].apply(lambda r : r in STIMS) if filter_stims else True
    x = all_scores['r2'][all_scores['c']]
    y = all_scores['score'][all_scores['c']]
    #y = (y -y.mean())/(y.var())
    R , _ = scipy.stats.pearsonr(x, y)
    #if R> 0.2:
        #print(voxel,R )
    return R 

def per_voxel_r(scores, comprehension, filter_stims= False):
    corr_ = partial(pearson_corr, scores, 
                                    comprehension,
                                    filter_stims = filter_stims)
    with Pool() as pool:
        rmap = pool.map_async(corr_,
                            scores.index).get(25)
        pool.close()
    return rmap
def get_map(scores):
    from nilearn.input_data import NiftiMasker    
    mymasker = NiftiMasker(mask_img='../parcellation/STG_middle.nii.gz')
    mymasker.fit()
    r2map = scores.copy().reshape(1,556)
    #r2map[r2map<0] = 0
    R2_img = mymasker.inverse_transform(r2map)
    return R2_img
def plot_scores(rmap):
    rmap_ = np.array(rmap)
    r2img = get_map(rmap_)
    from nilearn.plotting import plot_stat_map,plot_glass_brain
    f,(ax1, ax2) = plt.subplots(1,2)
    plot_stat_map(r2img,figure=f,axes=ax1)
    plot_glass_brain(r2img,figure=f,axes=ax2)

def plot_corr(scores, 
            comprehension, 
            voxel = None , 
            reduction = None,
            filter_stims= True):
    all_scores = get_scores(scores, comprehension, reduction = reduction, voxel = voxel)
    all_scores['c'] = all_scores['task'].apply(lambda r : r in STIMS) if filter_stims else True
    x = all_scores['r2'][all_scores['c']]
    y = all_scores['score'][all_scores['c']]
    #y = (y -y.mean())/(y.var())
    R , _ = scipy.stats.pearsonr(x, y)
    plt.scatter(y, np.sqrt(x)  )
    plt.title('Comprehension and SSL brain scores')
    plt.xlabel('Comprehension')
    plt.ylabel('Pearson correlation')
    i = np.sqrt(x).max()- 0.015
    plt.text(0.1,i,f'R  = {R:.3f}', fontsize  = 'large', 
            fontstyle = 'italic',
            fontweight = 'heavy')
#%% 
comprehension = parse_scores()
# %%
STIMS = ['piemanpni', 'slumlord','reach','slumlordreach', 'merlin', 'sherlock', 'bronx', 'forgot', 'black']
# %%
scores = pd.read_csv('../scores/scores_per_voxel_cor_model_50_2.csv')

plot_corr  (scores, 
            comprehension, 
            reduction = 'mean' ,
            filter_stims= True)
plt.title('Comprehension and average SSL brain scores', fontweight = 'heavy')

#%%
rmap1 = per_voxel_r(scores, comprehension, filter_stims = True)
# %%
voxel = np.array(rmap1).argmax()
#cond = (comprehension['score']<1)
plot_corr  (scores, 
            comprehension, 
            voxel = voxel, 
            filter_stims= True)# %%
plt.title(f'Comprehension and  SSL brain \nscores for voxel {voxel} ', fontweight = 'heavy')
# %%
plot_scores(rmap1)
# %%
