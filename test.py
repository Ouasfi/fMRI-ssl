
# %%
import RP_dataset as rpd
import model as md
import torch
from torch.nn.functional import soft_margin_loss

if __name__ == "__main__":
    # %%
    val_generator = rpd.RP_Dataset( subjects = [1], sampling_params= (1,90), wind_len = 15 , debut = 662)
    val_sampler = rpd.RPSampler(val_generator, batch_size = 30,size = 60,  weights = [0.5]*2)
    val_dataset = torch.utils.data.Subset(val_generator, indices= list(val_sampler))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=30, 
                                         num_workers=0, collate_fn=rpd.ssl_collate)
    # %%
    val_list = list(val_loader)
    for i, data in enumerate(val_loader):
        assert abs(data[0][0] - val_list[i][0][0]).sum() == 0
    print('\nLoaders sucessfully initialised \n')
    # %%
    val_generator = rpd.RP_Dataset_Multi( subjects = ['048', '096', '101'], sampling_params= (1,90), wind_len = 15 ,mode = "val")
    val_sampler = rpd.RPSampler(val_generator, batch_size = 30,size = 60,  weights = [0.5]*2)
    val_dataset = torch.utils.data.Subset(val_generator, indices= list(val_sampler))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=30, 
                                         num_workers=0, collate_fn=rpd.ssl_collate)
    # %%
    val_list = list(val_loader)
    for i, data in enumerate(val_loader):
        assert abs(data[0][0] - val_list[i][0][0]).sum() == 0
    print('\nLoaders sucessfully initialised \n')

    # %%
    print('- Audio encoder ...\n')
    m = md.AudioEmbed()
    print(m)
    (x_fmri, x_audio), y = next(iter(val_loader))
    assert m(x_audio.float()).shape == torch.Size([30, 32,10]) , 'Size mismatch !'
    print('Keys sucessfully matched!')
    # %%
    print('- FMRI encoder ...\n')
    m = md.FMRIEmbed(voxels = 97)
    print(m)
    (x_fmri, x_audio), y = next(iter(val_loader))
    assert m(x_fmri.float()).shape == torch.Size([30, 32,10])
    print('Keys sucessfully matched!')
    #%%
    print('- Siamese model ...\n')
    m = md.SiameseModel(hidden_dim = 32*10, voxels = 97)
    print(m)
    (x_fmri, x_audio), y = next(iter(val_loader))
    assert m(x_fmri.float(), x_audio).shape == torch.Size([30, 2]), 'Size mismatch !'
    assert soft_margin_loss(m(x_fmri.float(), x_audio), y).requires_grad == True, 'Broken computational graph !'
    print('Keys sucessfully matched!')