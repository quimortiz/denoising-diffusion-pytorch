import torch
#from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d_encoder import GaussianDiffusion1D, Unet1D, Trainer1D, Dataset1D

from vision_model.model import VanillaVAE

from torch import nn

import random
import string
import pickle

def generate_exp_id():
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=6))






import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp_id', type=str, default=generate_exp_id(),help='X')
parser.add_argument('--pretrained', action='store_true',help='X')
parser.add_argument('--lr', type=float, default=1e-4,help='X')
parser.add_argument('--recon_weight', type=float, default=.0,help='X')
parser.add_argument('--z_weight', type=float, default=1e-5, help='X')
parser.add_argument('--z_diff', type=float, default=1e-5, help='X')
parser.add_argument('--cond', action='store_true', help='X')
parser.add_argument('--size', type=int, default=32, help='X')
parser.add_argument('--mod_lr', action='store_true', help='X')
parser.add_argument('--cond_combined', action='store_true',help='X')


args = parser.parse_args()
print(args)

#args.size = 64

#args.cond = True
















batch = 1024
w = 64

h = 64
c = 3
n = 16

rand_data = torch.rand(batch, n , c, h , w)

folder_name = '/home/quim/code/denoising-diffusion-pytorch/image_based/plots_trajs/2024-07-11/trajs_2024-07-11--13-52-35_6NVJR7/'

    # load with pickle
print('loading data with pickle')
# with open(folder_name + 'datapoints.pkl', 'rb') as f:
#     datapoints = pickle.load(f)
with open(folder_name + 'trajectories.pkl', 'rb') as f:
    trajectories = pickle.load(f)
    print('data has been loaded with pickle!')

my_data = torch.stack( [ traj['imgs'] for traj in trajectories] )
del trajectories
import torch.nn.functional as F

# Slice the tensor to take one every two elements
my_data_reduced = my_data[:, ::2, :, :, :]


if args.size == 32:

    my_data_resized = torch.stack( [
    F.interpolate(traj, size=(32, 32), mode='bilinear', align_corners=False) for traj in my_data_reduced
    ])
elif args.size == 64:
    my_data_resized = my_data_reduced
else:
    raise ValueError('only 32 and 64 are supported')



# i want to reduce the sequence length from 16 to 8


# del datapoints






vision_model = VanillaVAE(in_channels=3, latent_dims=8 , size = args.size)
if args.pretrained:
    path = '/home/quim/code/Conv-VAE-PyTorch/output/ZIN9X2/state_dict_ZIN9X2_e09900.pth'
    vision_model.load_state_dict(torch.load(path))



model = Unet1D(
    dim = 32,
    dim_mults = (1, 2, 4),
    channels = 8, 
    y_cond = args.cond
)

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


diffusion = GaussianDiffusion1D(
    model,
    seq_length = 8,
    timesteps = 100,
    objective = 'pred_v' ,
    auto_normalize = False,
    encoder =  encoder   ,
    decoder = decoder,
    z_diff=args.z_diff
    )


loss = diffusion(my_data_resized[:32])
loss.backward()

# Or using trainer

dataset = Dataset1D(my_data_resized)  # this is just an example, but you can formulate your own Dataset and pass it into the `Trainer1D` below

trainer = Trainer1D(
    diffusion,
    dataset = dataset,
    train_batch_size = 64,
    train_lr = args.lr,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 1,    # gradient accumulation steps
    save_and_sample_every = 1000,
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    results_folder= f"./results/{args.exp_id}/",
    recon_weight=args.recon_weight,
    z_weight=args.z_weight,
    y_cond = args.cond,
    mod_lr = args.mod_lr,
    cond_combined = args.cond_combined
)
trainer.train()

# after a lot of training

sampled_seq = diffusion.sample(batch_size = 4)
sampled_seq.shape # (4, 32, 128)
