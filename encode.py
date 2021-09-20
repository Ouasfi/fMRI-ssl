#%%
import torch
from fastai.vision.all import *

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score

#%%
@torch.no_grad()
def get_data(trainset, subject, stim, learn):
     
    unfold_audio = nn.Unfold(padding = (0,trainset.w*trainset.sr//2-int(trainset.tr*trainset.sr)//2),
                    kernel_size =(1,trainset.w*trainset.sr), 
                    stride = (1,int(trainset.tr*trainset.sr)) )
    
    #load audio
    X_train = trainset.load_audio(subject,stim )[0] # (N,) where N = N_sec *sr
    x_train = unfold_audio( torch.from_numpy(X_train).reshape(1,1,1, -1)) # (1, k, B ) where k = w*sr,  B = N//(tr*sr) ie.a feature map per tr
    x_train = x_train.transpose(0,-1).squeeze(-1)#(B, k)
    #load fmri
    f = int(X_train.shape[0]/(trainset.sr*trainset.tr))
    fmri = trainset.load_fmri(subject ,stim )
    onset = trainset.get_onset(stim)
    y_train = torch.from_numpy(fmri[onset:f+onset] )
    print(x_train.shape, y_train.shape)
    N = min(x_train.shape[0], y_train.shape[0])
    train_dataset = torch.utils.data.TensorDataset(x_train[:N], y_train[:N])
    train_sampler = torch.utils.data.SequentialSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               sampler = train_sampler,
                                               batch_size= x_train.shape[0])
    x_train , y_train = train_loader.__iter__().__next__()  #(B, k)
    z_train= nn.Flatten()(learn.model.audio(x_train)).cpu() #(B, F) F is the dim of the extracted features
    return  (z_train, y_train)


def siamese_splitter(model):
    return [params(model.encoder), params(model.head)]
    
def loss_func(out, targ):
    return CrossEntropyLossFlat()(out, targ.long())
def ridge_encoding (z_train, y_train,z_test,  y_test ):
    clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10,100, 250, 2500, 25000],
                        ).fit(z_train.numpy(), y_train.cpu().numpy())
    z , y = z_train,  y_train
    predictions = clf.predict(z.numpy())
    
    tr_scores = r2_score(y.cpu().numpy(), predictions, multioutput='raw_values')
    tr_scores[tr_scores<0] = 0
    z , y = z_test,  y_test
    predictions = clf.predict(z.numpy())
    val_scores = r2_score(y.cpu().numpy(), predictions, multioutput='raw_values')
    val_scores[val_scores<0] = 0
    return clf, tr_scores, val_scores

def get_map(scores):
    from nilearn.input_data import NiftiMasker    
    mymasker = NiftiMasker(mask_img='parcellation/STG_middle.nii.gz')
    mymasker.fit()
    r2map = scores.copy().reshape(1,556)
    #r2map[r2map<0] = 0
    R2_img = mymasker.inverse_transform(r2map)
    return R2_img
def plot_r2map(R2_img, name = 'R2.png'):
    from nilearn.plotting import plot_stat_map,plot_glass_brain

    f,ax = plt.subplots()
    plot_stat_map(R2_img,output_file =name,  figure=f,axes=ax)
    #plot_glass_brain(R2_img,figure=f,axes=ax)
    f.savefig('R2.png')
    R2_img.to_filename('R2map.nii.gz')


def get_regdata(trainset, test_subject, test_stim, unfold = False, batch_size = None):
    #load audio
    audio = trainset.load_audio(test_subject,test_stim )[0]
    #load fmri
    f = int(audio.shape[0]/(trainset.sr*trainset.tr))
    fmri = trainset.load_fmri(test_subject ,test_stim )
    onset = trainset.get_onset(test_stim)
    fmri = fmri[onset:f+onset]
    # Data split

    middle_tr = int(fmri.shape[0]/2)
    middle_frames = int(middle_tr*trainset.sr*trainset.tr)
    X_train = (audio[:middle_frames])
    X_test = (audio[middle_frames:])
    y_train = fmri[:middle_tr] 
    y_test = fmri[middle_tr:]
    unfold_audio = nn.Unfold(padding = (0,trainset.w*trainset.sr//2-int(trainset.tr*trainset.sr)//2),
                    kernel_size =(1,trainset.w*trainset.sr), 
                    stride = (1,int(trainset.tr*trainset.sr)) )
    
    unfold_fmri = nn.Unfold(padding = (0,trainset.w//int(trainset.tr*2)-int(1)//2),
                    kernel_size =(1,int(trainset.w/trainset.tr)), 
                    stride = (1,1 )) if unfold else None 
    return get_loaders((X_train, y_train),
                        (X_test, y_test),
                        unfold_audio = unfold_audio,
                        unfold_fmri = unfold_fmri, 
                        batch_size = batch_size)

def get_loaders(trainset, testset, unfold_audio,  batch_size, unfold_fmri = None):

    X_train, y_train = trainset
    X_test, y_test = testset
    x_train = unfold_audio( torch.from_numpy(X_train).reshape(1,1,1, -1))
    x_train = x_train.transpose(0,-1).squeeze(-1)
    x_test = unfold_audio( torch.from_numpy(X_test).reshape(1,1,1, -1))
    x_test = x_test.transpose(0,-1).squeeze(-1)
    y_train, y_test = torch.from_numpy(y_train), torch.from_numpy(y_test)
    if unfold_fmri is not None:
        y_train = unfold_fmri(y_train.transpose(1,0).unsqueeze(1).unsqueeze(1)).transpose(-1, 0)
        y_test = unfold_fmri(y_test.transpose(1,0).unsqueeze(1).unsqueeze(1)).transpose(-1, 0)
    #print(y_train.shape, x_train.shape)
    #train
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_sampler = torch.utils.data.SequentialSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               sampler = train_sampler,
                                               batch_size= x_train.shape[0])
    assert len(train_loader)==1, 'batch size' 
    #test
    try:
        test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
    except:
        N = min(x_test.shape[0], y_test.shape[0])
        test_dataset = torch.utils.data.TensorDataset(x_test[:N], y_test[:N])
    test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               sampler = test_sampler,
                                               batch_size= x_test.shape[0])
    assert len(test_loader)==1, 'batch size'                                      
    return train_loader, test_loader




def get_regression_data(learn, val_generator, test_subject, test_stim, batch_size = None ):
    tr_loader, test_loader = get_regdata(val_generator, test_subject, test_stim, batch_size = batch_size)
    x_train , y_train = tr_loader.__iter__().__next__()
    x_test , y_test = test_loader.__iter__().__next__()
    
    with torch.no_grad():
        z_train= nn.Flatten()(learn.model.audio(x_train)).cpu()
        z_test= nn.Flatten()(learn.model.audio(x_test)).cpu()
    return (z_train, y_train), (z_test, y_test)