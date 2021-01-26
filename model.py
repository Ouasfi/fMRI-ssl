
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
        x_audio = self.flatten(self.audio(x_audio))
        x_fmri = self.flatten( self.fmri(x_fmri) )
        x = self.metric(torch.abs(x_audio - x_fmri))
        return x