
from fastai.vision.all import *
import torch
import numpy as np



#@title SSL models
class AudioEmbed(nn.Module):
    """
    VGG 4 layers
    """
    def __init__(self):
        super(AudioEmbed, self).__init__()
        power = 4

        self.conv1 = nn.Conv1d(1, 2**power, 64, 2)
        self.bn1 = nn.BatchNorm1d(2**power)
        self.pool1 = nn.MaxPool1d(8)

        self.conv2 = nn.Conv1d(2**power, 2**power, 4, stride = 2)
        self.bn2 = nn.BatchNorm1d(2**power)
        self.pool2 = nn.MaxPool1d(8)

        self.conv3 = nn.Conv1d(2**power, 2**(power+1), 4, stride = 4)
        self.bn3 = nn.BatchNorm1d(2**(power+1))
        self.pool3 = nn.MaxPool1d(4)

        self.conv4 = nn.Conv1d(2**(power+1),2**(power+1),4, stride = 4)
        self.bn4 = nn.BatchNorm1d(2**(power+1))
        self.pool4 = nn.AdaptiveMaxPool1d(10)
    def forward(self, x):
        x = self.conv1(x.unsqueeze(1))
        x = self.bn1(x).relu()
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x).relu()
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x).relu()
        x = self.pool3(x)
        
        x = self.conv4(x); 
        x = self.bn4(x).relu(); 
        x = self.pool4(x); 
        return x
class FMRIEmbed(nn.Module):
    """
    VGG 4 layers
    """
    def __init__(self, voxels = 3):
        super(FMRIEmbed, self).__init__()
        power = 4

        self.conv1 = nn.Conv1d(voxels, 2**(power+1), 1)
        self.bn1 = nn.BatchNorm1d(2**(power+1))
        
    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.conv1(x); 
        x = self.bn1(x).relu(); 
        return x

class SiameseModel(nn.Module):
    """
    SiameseModel
    """
    def __init__(self, hidden_dim, n_classes = 2, voxels = 3):
        super(SiameseModel, self).__init__()
        self.audio = AudioEmbed().float()
        self.fmri = FMRIEmbed(voxels).float()
        self.metric = nn.Linear(hidden_dim, out_features = n_classes)
        self.flatten= nn.Flatten()
    def forward(self, x_fmri, x_audio):
        #x_fmri, x_audio = fmri_audio
        x_audio = self.flatten(self.audio(x_audio))
        x_fmri = self.flatten( self.fmri(x_fmri.float()) )
        x = self.metric(torch.abs(x_audio - x_fmri))
        return x
class SiameseModel_cat(nn.Module):
    """
    SiameseModel
    """
    def __init__(self, hidden_dim, n_classes = 2, voxels = 3):
        super(SiameseModel_cat, self).__init__()
        self.audio = AudioEmbed().float()
        self.fmri = FMRIEmbed(voxels).float()
        self.metric = nn.Linear(hidden_dim*2, out_features = n_classes)
        self.flatten= nn.Flatten()
    def forward(self, x_fmri, x_audio):
        #x_fmri, x_audio = fmri_audio
        x_audio = self.flatten(self.audio(x_audio))
        x_fmri = self.flatten( self.fmri(x_fmri.float()) )
        x = self.metric(torch.cat([x_audio , x_fmri], dim = 1))
        return x
class ToDevice(Callback):
    "Move data to CUDA device and convert it to float"
    def __init__(self, device=None): self.device = torch.device('cuda')#ifnone(device, DEVICE)
    def before_batch(self): 
        self.learn.xb = (self.learn.xb[0][0].float(), self.learn.xb[0][1].float())
        self.learn.xb,self.learn.yb = to_device(self.xb),to_device(self.yb)
    def before_fit(self): self.model.to(self.device)

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class SiameseModel_corr(nn.Module):
    """
    SiameseModel
    """

    def __init__(self, hidden_dim, n_classes = 2, voxels = 3):
        super(SiameseModel_corr, self).__init__()
        self.audio = AudioEmbed().float()
        self.fmri = FMRIEmbed(voxels).float()
        #self.metric = md.Cross_corr(hidden_dim = 32*10)
        self.flatten= nn.Flatten()
    def forward(self, x_fmri, x_audio):
        #x_fmri, x_audio = fmri_audio
        x_audio = self.flatten(self.audio(x_audio))
        x_fmri = self.flatten( self.fmri(x_fmri.float()) )
        #x = self.metric(x_audio , x_fmri)
        return (x_fmri, x_audio)

class CrosscCorrLoss(nn.Module):
    """
    Cross_corr
    """
    def __init__(self, hidden_dim, scale = 1, lambd = 0.001):
        super(CrosscCorrLoss, self).__init__()
        self.bn = nn.BatchNorm1d(hidden_dim, affine=False)
        self.bn2 = nn.BatchNorm1d(hidden_dim, affine=False)
        self.scale_loss = scale
        self.lambd = lambd
    def forward(self, out, y ):
        y = 2*y.float()-1
        x_fmri, x_audio = out[0], out[1]
        N, D = x_fmri.shape
        c = (self.bn(x_fmri).mul_(y)).T@(self.bn2(x_audio))
        c.div_(N)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(self.scale_loss)
        off_diag = off_diagonal(c).pow_(2).sum().mul(self.scale_loss)
        return on_diag + self.lambd * off_diag