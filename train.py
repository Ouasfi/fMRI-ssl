from fastai.vision.all import *
import torch
import numpy as np

def siamese_splitter(model):
    return [params(model.encoder), params(model.head)]
    
def loss_func(out, targ):
    return CrossEntropyLossFlat()(out, targ.long())