import RP_dataset as rpd; #reload(rpd)
import torch
import model as md
from fastai.vision.all import *
import configargparse

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
                                            num_workers=3,
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
                                         num_workers=3, collate_fn=rpd.ssl_collate)
    return train_loader, val_loader

def get_learner(train_loader, val_loader, args):
    model = md.SiameseModel_corr(hidden_dim = 32*10, n_classes = 2,voxels = 556,
            temporal_filter = args.temporal_filter).float()
    # loss_fn
    if args.ub: 
        loss_fn = md.CrosscCorrLoss(hidden_dim = 32*10,
                                    scale =args.scale ,
                                    lambd = args.reg).cuda()
    else:
        loss_fn = md.CrossCorrLoss(hidden_dim = 32*10,
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
def get_args():
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
    p.add_argument('--subject', type=str, default=None, help='Subject ID')
    p.add_argument('--s', type=str, default='N', help='Scoring mode')
    p.add_argument('-ub', dest='ub', action='store_true', help = 'Upper Bound loss')
    p.add_argument('-save', dest='save', action='store_true', help= 'save encoding scores')
    p.add_argument('-temporal_filter', dest='temporal_filter', action='store_true',
                        help = 'Filter the temporal dimension of fmri windows')
    return p.parse_args()