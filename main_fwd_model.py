import torch
from torch import nn
import json 
import sys
import dataclasses
import numpy as np

# Quick script to train a forward model!

from torch.autograd import Variable

import pickle

load_data = False

class CNN_Decoder(nn.Module):
    def __init__(self, embedding_size, input_size=(1, 28, 28)):
        super(CNN_Decoder, self).__init__()
        self.input_height = input_size[1]
        self.input_width = input_size[2]
    
        self.input_dim = embedding_size
        self.channel_mult = 16
        self.output_channels = input_size[0]
        self.fc_output_dim = 512

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.fc_output_dim),
            nn.BatchNorm1d(self.fc_output_dim),
            nn.ReLU(True)
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.fc_output_dim, self.channel_mult*4,
                                4, 1, 0, bias=False),
            nn.BatchNorm2d(self.channel_mult*4),
            nn.ReLU(True),
            # state size: self.channel_mult*4 x 4 x 4
            nn.ConvTranspose2d(self.channel_mult*4, self.channel_mult*2,
                                4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*2),
            nn.ReLU(True),
            # state size: self.channel_mult*2 x 8 x 8
            nn.ConvTranspose2d(self.channel_mult*2, self.channel_mult*1,
                                4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*1),
            nn.ReLU(True),
            # state size: self.channel_mult*1 x 16 x 16
            nn.ConvTranspose2d(self.channel_mult*1, self.channel_mult//2,
                                4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult//2),
            nn.ReLU(True),
            # state size: self.channel_mult//2 x 32 x 32
            nn.ConvTranspose2d(self.channel_mult//2, self.output_channels, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size: self.output_channels x 64 x 64

        )

    def forward(self, x):
        x = self.fc(x)
        # print(x.shape)
        x = x.view(-1, self.fc_output_dim, 1, 1)
        # print(x.shape)
        x = self.deconv(x)
        # print(x.shape)
        return x
        # return x.view(-1, self.input_width*self.input_height)


class CNN_Encoder(nn.Module):
    def __init__(self, output_size, input_size=(1, 28, 28)):
        super(CNN_Encoder, self).__init__()

        self.input_size = input_size
        self.channel_mult = 16

        #convolutions
        nc = input_size[0]
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=nc,
                     out_channels=self.channel_mult*1,
                     kernel_size=4,
                     stride=1,
                     padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult*1, self.channel_mult*2, 4, 2, 1),
            nn.BatchNorm2d(self.channel_mult*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult*2, self.channel_mult*4, 4, 2, 1),
            nn.BatchNorm2d(self.channel_mult*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult*4, self.channel_mult*8, 4, 2, 1),
            nn.BatchNorm2d(self.channel_mult*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult*8, self.channel_mult*16, 3, 2, 1),
            nn.BatchNorm2d(self.channel_mult*16),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.flat_fts = self.get_flat_fts(self.conv)
        print(self.flat_fts)

        self.linear = nn.Sequential(
            nn.Linear(self.flat_fts, self.flat_fts),
            nn.BatchNorm1d(self.flat_fts),
            nn.LeakyReLU(0.2),
            nn.Linear(self.flat_fts, output_size),
            nn.Tanh()
                    #latent_code = torch.tanh(latent_code)
        )

    def get_flat_fts(self, fts):
        f = fts(Variable(torch.ones(1, *self.input_size)))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):
        # x = self.conv(x)
        x = self.conv(x.view(-1, *self.input_size))
        x = x.view(-1, self.flat_fts)
        return self.linear(x)


class Fwd_model(nn.Module):
    def __init__(self,nx,nu):
        super(Fwd_model, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(nu + nx, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, nx)
        )

    def forward(self, x,u ):
        return self.fc(torch.cat([x, u], dim=-1))
    

# data is a list of tuples (u, img, img_next)

import torch
from torchvision import transforms
from PIL import Image

folder_name = '/home/quim/code/denoising-diffusion-pytorch/image_based/plots_trajs/2024-07-11/trajs_2024-07-11--13-55-06_G8OO3W/'
folder_name = '/home/quim/code/denoising-diffusion-pytorch/image_based/plots_trajs/2024-07-11/trajs_2024-07-11--13-52-35_6NVJR7/'


import tqdm
if load_data:
    file_name = 'all_data.json'


    with open(folder_name + file_name, 'r') as f:
        d = json.load(f)
    print('data has been loaded!')


    datapoints = []

    trajectories = []

    #dict_keys(['states', 'controls', 'images', 'id'])

    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize the image to 64x64
        transforms.ToTensor()      ,   # Convert the image to a tensor
        # clamp between .1 and .9
        transforms.Lambda(lambda x: x.clamp(0.1, 0.9))  # Clamp values between 0.1 and 0.9
        ])

    for dd in tqdm.tqdm(d):
        nseq = len(dd['states'])
        for i in range(nseq-1):
            datapoint = {}
            datapoint['x'] = torch.tensor(dd['states'][i])
            datapoint['xnext'] = torch.tensor(dd['states'][i+1])
            datapoint['u'] = torch.tensor(dd['controls'][i])

            image_path = dd['images'][i]
            # 'path/to/your/image.png'
            image = Image.open(image_path).convert('RGB')  # Convert to RGB to ensure 3 channels
            # Apply the transformations
            tensor_image = transform(image)
            datapoint['img'] = tensor_image

            image_path = dd['images'][i+1]
            image = Image.open(image_path).convert('RGB')  # Convert to RGB to ensure 3 channels
            tensor_image = transform(image)
            datapoint['img_next'] = tensor_image
            datapoints.append(datapoint)

        trajectory = {}
        trajectory = {'x': torch.tensor(dd['states']), 'u': torch.tensor(dd['controls'])}

        imgs = []
        for img_path in dd['images']:
            image = Image.open(img_path).convert('RGB')
            tensor_image = transform(image)
            imgs.append(tensor_image)
        imgs = torch.stack(imgs)
        trajectory['imgs'] = imgs
        trajectories.append(trajectory)
    
    # lets save it to a file with pickle

    import pickle
    with open(folder_name + 'datapoints.pkl', 'wb') as f:
        pickle.dump(datapoints, f)

    with open(folder_name + 'trajectories.pkl', 'wb') as f:
        pickle.dump(trajectories, f)

    print('exiting')
    sys.exit(0)
    
else: 
    # load with pickle
    print('loading data with pickle')
    with open(folder_name + 'datapoints.pkl', 'rb') as f:
        datapoints = pickle.load(f)
    with open(folder_name + 'trajectories.pkl', 'rb') as f:
        trajectories = pickle.load(f)
    print('data has been loaded with pickle!')
    
from torch.utils.data import Dataset, DataLoader
class CustomDataset(Dataset):
    def __init__(self, datapoints, device):
        self.datapoints = datapoints
        self.device = device

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        datapoint = self.datapoints[idx]
        # return {
        #     'x': datapoint['x'].to(self.device),
        #     'xnext': datapoint['xnext'].to(self.device),
        #     'u': datapoint['u'].to(self.device),
        #     'img': datapoint['img'].to(self.device),
        #     'img_next': datapoint['img_next'].to(self.device)
        # }
        return {
            'x': datapoint['x'],
            'xnext': datapoint['xnext'],
            'u': datapoint['u'],
            'img': datapoint['img'],
            'img_next': datapoint['img_next']
        }


device = 'cuda' if torch.cuda.is_available() else 'cpu'

device = torch.device(device)
print('device: ', device)

batch_size = 64

dataset = CustomDataset(datapoints,device)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)




image_size = 64
nu = 2
nx = 8

nz = 8

img_size = (3,64,64)
fwd_model = Fwd_model(nx,nu)
encoder = CNN_Encoder(nz,img_size)
decoder = CNN_Decoder(nz, img_size)

fwd_model = fwd_model.to(device)
encoder = encoder.to(device)
decoder = decoder.to(device)

# todo: load the data correctly!!

opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(fwd_model.parameters()), lr=1e-4)


loss_fn = nn.MSELoss()
# tod: loss on z and loss on x
num_epochs = 5*1000
z_weight = .0001
recon_weight = .1
recon_weight_next = 1.

data_out = {'img_next_loss': [], 'z_loss': [] , 'img_loss': []}

import random
import string

def generate_id():
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


exp_id = generate_id()

import matplotlib.pyplot as plt


folder_out = 'results/' + exp_id + '/'
import pathlib
pathlib.Path(folder_out).mkdir(parents=True, exist_ok=True)


from torchvision import transforms as T, utils

from pympler import asizeof


# Function to print memory usage of variables
def print_memory_usage(*variables):
    for var in variables:
        print(f"Memory usage of variable {var}: {asizeof.asizeof(var)} bytes")



def print_all_memory_usage():
    for var_name, var_value in globals().items():
        if not var_name.startswith("__") and not callable(var_value):
            print(f"Memory usage of variable {var_name}: {asizeof.asizeof(var_value)} bytes")

# Call the function
#print_all_memory_usage()


# Function to print memory usage of all defined variables
def print_all_memory_usage(local_vars):
    for var_name, var_value in local_vars.items():
        if not var_name.startswith("__") and not callable(var_value):
            try:
                size = asizeof.asizeof(var_value)
                print(f"Memory usage of variable {var_name}: {size} bytes")
            except Exception as e:
                print(f"Could not determine size of {var_name} due to: {e}")

local_vars = locals()

# Print memory usage
print_all_memory_usage(local_vars)






for epoch in tqdm.tqdm(range(num_epochs)):
    epoch_loss = {'img_next_loss': 0, 'z_loss': 0 , 'img_loss': 0}
    batch_counter = 0 

    for batch in dataloader:
        encoder.train()
        decoder.train()
        fwd_model.train()
        opt.zero_grad()
        #x = batch['x']
        #xnext = batch['xnext']
        u = batch['u'].to(device)
        img = batch['img'].to(device)
        img_next = batch['img_next'].to(device)
        z = encoder(img)
        xnext = fwd_model(z,u)
        img_next_recon = decoder(xnext)

        img_next_loss = recon_weight_next * loss_fn(img_next_recon, img_next)
        z_loss = z_weight * (z * z).mean()


        img_recon = decoder(z)
        img_loss = recon_weight * loss_fn(img_recon, img)

        epoch_loss['img_next_loss'] += img_next_loss.item()
        epoch_loss['z_loss'] += z_loss.item()
        epoch_loss['img_loss'] += img_loss.item()

        loss = img_next_loss + z_loss + img_loss
       
        batch_counter += 1
        loss.backward()
        opt.step()

    data_out['img_next_loss'].append(epoch_loss['img_next_loss']/batch_counter)
    data_out['z_loss'].append(epoch_loss['z_loss']/batch_counter)
    data_out['img_loss'].append(epoch_loss['img_loss']/batch_counter)


    if epoch % 10 == 0:

        encoder.eval()
        decoder.eval()
        fwd_model.eval()

        print(f'Epoch {epoch}, img_next_loss: {epoch_loss["img_next_loss"]/batch_counter}, img_loss: {epoch_loss["img_loss"]/batch_counter}, z_loss: {epoch_loss["z_loss"]/batch_counter}')

        # plot the losses:
        plt.plot(data_out['img_next_loss'], label='img_next_loss')
        fout = folder_out + f'img_next_loss_{epoch:05d}.png'
        print('saving to: ', fout)
        plt.savefig(fout)
        plt.close()
        #plt.show()

        fout = folder_out + f'z_loss_{epoch:05d}.png'
        plt.plot(data_out['z_loss'], label='z_loss')
        print('saving to: ', fout)
        plt.savefig(fout)
        plt.close()
        #plt.show()

        fout = folder_out + f'img_loss_{epoch:05d}.png'
        plt.plot(data_out['img_loss'], label='img_loss')
        print('saving to: ', fout)
        plt.savefig(fout)
        plt.close()


        # lets evaluate the encoder/decoder
        # get some images from the dataseimgst
        with torch.no_grad():
            num_imgs = 10
            ids = [ 100 * i for i in range(num_imgs)]
            imgs = [dataset[i]['img'] for i in ids]
            us = torch.stack([dataset[i]['u'] for i in ids]).to(device)
            imgs = torch.stack(imgs).to(device)
            z = encoder(imgs)
            imgs_recon = decoder(z)
            # plot the images
            all_imgs = torch.cat([imgs, imgs_recon], dim=0)
            print('loss fun is ', loss_fn(imgs_recon, imgs))
            fout = folder_out + f'imgs_recon_{epoch:05}.png'
            print('saving to: ', fout)
            utils.save_image(all_imgs, fout, nrow = 10) 

            # lets evaluate the fwd model
            z = encoder(imgs)
            xnext = fwd_model(z,us)
            img_next_recon = decoder(xnext)
            imgs_next = torch.stack([dataset[i]['img_next'] for i in ids]).to(device)

            all_imgs = torch.cat([imgs_next, img_next_recon], dim=0)
            fout = folder_out + f'imgs_next_recon_{epoch:05}.png'
            print('loss fun is ', loss_fn(img_next_recon, imgs_next))
            print('saving to: ', fout)
            utils.save_image(all_imgs, fout, nrow = 10) 


            # lets plot 5 trajectories!!

            #trajectory = {'x': torch.tensor(dd['states']), 'u': torch.tensor(dd['controls'])}

            all_trajs = []
            num_trajs_eval = 5
            for i in range(num_trajs_eval):
                nseq = len(trajectories[i]['x'])
                print('nseq: ', nseq)
                z0 = encoder(trajectories[i]['imgs'][0].unsqueeze(0).to(device))
                
                imgs_recon = []
                img_recon_0 = decoder(z0)
                imgs_recon.append(img_recon_0)

                for j in range(nseq-1):
                    z0 = fwd_model(z0, trajectories[i]['u'][j].unsqueeze(0).to(device))
                    img_next_recon = decoder(z0)
                    imgs_recon.append(img_next_recon)

                both_imgs = torch.cat(
                  [ trajectories[i]['imgs'].to(device).clamp(0.1,.8), # small trick so that real images appear darker
                    torch.cat(imgs_recon,dim=0) ], dim=0 )
                all_trajs.append(both_imgs)
            all_trajs = torch.cat(all_trajs, dim=0)
            fout = folder_out + f'rollout_{epoch:05}.png'
            print('saving to: ', fout)
            utils.save_image(all_trajs, fout, nrow = nseq) 

            #local_vars = locals()

            # Print memory usage
            #print_all_memory_usage(local_vars)
            #sys.exit()



       
      

            