import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer, MLPNet, GaussianDiffusionMLP, TrainerMLP
from denoising_diffusion_pytorch import GaussianDiffusionMLPEncoder, TrainerMLPEncoder
    # Unet, GaussianDiffusion, Trainer, MLPNet, GaussianDiffusionMLP, TrainerMLP
# I -> encoder ->  z
# Diffusion in z space
# I need to learn a decoder that goes from z to I

# Two losses:
# Encoder/Decoder Reconstruction loss
# Diffusion loss (diffusion network and encoder)

from torch import nn
from torch.autograd import Variable
import numpy as np


import random
import string
import pathlib

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

torch.set_num_threads(4)



import sys
sys.path.append('../Conv-VAE-PyTorch')

from model.model import VanillaVAE

pretrained = True

vision_model = VanillaVAE(in_channels=3, latent_dims=8)
if pretrained:
    path = '/home/quim/code/Conv-VAE-PyTorch/output/ZIN9X2/state_dict_ZIN9X2_e09900.pth'
    vision_model.load_state_dict(torch.load(path))





def generate_id():
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


class CNN_Decoder(nn.Module):
    def __init__(self, embedding_size, input_size=(1, 28, 28)):
        super(CNN_Decoder, self).__init__()
        self.input_height = input_size[1]
        self.input_width = input_size[2]
        print("height, widht")
        print(self.input_height, self.input_width)
        self.input_dim = embedding_size
        self.channel_mult = 16
        self.output_channels = input_size[0]
        self.fc_output_dim = 512

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.fc_output_dim),
            #nn.BatchNorm1d(self.fc_output_dim),
            nn.ReLU(True),
            nn.Linear(self.fc_output_dim, self.fc_output_dim),
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
           # nn.BatchNorm1d(self.flat_fts),
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


transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize images to a fixed size, e.g., 128x128
    #clamp between .1 and .9
    transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Lambda(lambda img: img.clamp(0.1, 0.9)),

])



# Load the images using ImageFolder
dataset = datasets.ImageFolder(f'/home/quim/code/denoising-diffusion-pytorch/tmp/all_img_color_fake_class/', transform=transform)

# Convert the dataset to a DataLoader to iterate through the data
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

# Prepare lists to hold the data and labels
all_images = []
all_labels = []

# Iterate through the DataLoader and collect the data and labels
for images, labels in dataloader:
    all_images.append(images)
    all_labels.append(labels)

# Concatenate all batches to form a single tensor dataset
all_images = torch.cat(all_images)
all_labels = torch.cat(all_labels)

# Create a TensorDataset
tensor_dataset = TensorDataset(all_images, all_labels)

nx = 8 # hidden dim
model = MLPNet(
    dim =  nx
)

img_size = (3,64,64)

encoder = CNN_Encoder(nx,img_size)
decoder = CNN_Decoder(nx,img_size)


class VAEencoder(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        self.sample_noise = False
    def __call__(self,img):
        mu, log_var, z = self.vae.encode(img)
        if self.sample_noise:
            return z
        else:
            return mu
    
class VAEdecoder(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
    def __call__(self,z):
        return self.vae.decode(z)
    
encoder = VAEencoder(vision_model)
decoder = VAEdecoder(vision_model)

clip_denoised = False


diffusion = GaussianDiffusionMLPEncoder(


    model,
    vector_size = nx,
    #objective = "pred_noise",
    objective = "pred_noise",
    #objective = "pred_x0",
    image_size = 64,
    beta_schedule = 'sigmoid',
    timesteps = 100,    # number of steps
    auto_normalize= False,
    encoder = encoder,
    decoder = decoder,
    loss_in_image_space=True,
    fix_encoder_decoder=False,
    clip_denoised=clip_denoised,
    z_space_weight = .1,
)

X = diffusion.sample(batch_size=128, return_all_timesteps=False)
# import pdb; pdb.set_trace()

id = generate_id()

writer = SummaryWriter(f'tb/runs/{id}')

results_folder = f"./results/{id}/"
pathlib.Path(results_folder).mkdir(exist_ok=True, parents=True)


trainer = TrainerMLPEncoder(
    diffusion,
   # 'tmp/all_img',
    '/home/quim/code/denoising-diffusion-pytorch/tmp/all_img_color_fake_class/class1/',
    train_batch_size = 32,
    #ema_update_every=100000,
    #train_lr = 1e-3,
    train_lr = 1e-4,
    train_num_steps = 1000*1000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    save_and_sample_every = 2*1000,
    ema_decay = 0.995,                # exponential moving average decay
    amp = False,                       # turn on mixed precision
    calculate_fid = False,              # whether to calculate fid during training
   # dataset = tensor_dataset,
    results_folder = results_folder,
    augment_horizontal_flip=False,
    weight_diffusion=1.,
    z_weight=.0001,
    recon_weight=1e-3,
    tb_writer = writer,
  
)


trainer.train()

out_info = trainer.out_info

fields = [ 'loss', 'recon_loss' , 'diff_loss'] 

for field in fields:
    plt.plot(out_info[field], label=field)
    plt.legend()
    plt.savefig(f"{results_folder}/{field}.png")
    plt.close()

writer.close()


# lets try to get something working!!

