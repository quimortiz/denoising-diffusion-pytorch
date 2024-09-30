import torch

# from denoising_diffusion_pytorch import (
#     Unet1D,
#     GaussianDiffusion1D,
#     Trainer1D,
#     Dataset1D,
#     Dataset1DCond,
# )
#
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer, MLPNet, GaussianDiffusionMLP, TrainerMLP


import random
import string
import torch.nn.functional as F
from einops import rearrange, reduce

from vision_model.model import VanillaVAE
def generate_exp_id():
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=6))

# lets modify this to learn the latent codes of the VAE.


from torchvision import utils

import sys # noqa
sys.path.append('resnet-18-autoencoder/src') # noqa
from classes.resnet_autoencoder import AE
import pathlib


from torch.utils.data import TensorDataset


torch.set_num_threads(1)



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
parser.add_argument('--y_cond_as_x', action='store_true',help='X')
parser.add_argument('--weight_decay', type=float, default=0.0, help='X')
parser.add_argument('--train_num_steps', type=int, default=100000, help='X')
parser.add_argument('--train_u', action='store_true', help='X')
parser.add_argument('--fix_encoder', action='store_true', help='X')
parser.add_argument("--resnet", action="store_true", help="X")

args = parser.parse_args()


nz = 8
n_elements = 1
seq_length = int ( 16 // n_elements)
vision_model = VanillaVAE(in_channels=3, latent_dims=nz , size = args.size)
# model = Unet1D(dim=64, dim_mults=(1, 2, 4), nx=8, nu=0, ny=8)

model = MLPNet(
    dim =  nz
)

diffusion = GaussianDiffusionMLP(
    model,
    vector_size = nz,
    objective = "pred_v",
    # image_size = 64,
    beta_schedule = 'cosine',
    timesteps = 100,    # number of steps
    # auto_normalize= False
)


if not args.train_u:

    data_in = "./new_data_all_2024-09-05.pt"


    data = torch.load(data_in)['imgs']
    nu = 0
    data = data[:, ::n_elements, ...] # take one every n_elements

else: 
    nu = 2
    data = torch.load("./new_data_wu_img_THURSDAY.pt")
    data_us =  torch.load("./new_data_wu_us_THURSDAY.pt")
    data = data[:, ::n_elements, ...] # take one every two elments
    data_us = data_us[:,::n_elements, ...]
    my_data_us_reduced = data_us

data = data.clamp(.1, .9)
my_data_resized = data


if args.size == 32:
    target_size = (32, 32)
    batch_size, seq_length, channels, height, width = my_data_resized.shape
    dataset_reshaped = my_data_resized.view(-1, channels, height, width)
    dataset_resized = F.interpolate(dataset_reshaped, size=target_size, mode='bilinear', align_corners=False)
    dataset_resized = dataset_resized.view(batch_size, seq_length, channels, *target_size)
    my_data_resized = dataset_resized




if args.resnet:
    vision_model = AE('light') # try the default one!
else:
    vision_model = VanillaVAE(in_channels=3, latent_dims=nz, size=args.size)


if args.pretrained:
    if args.size == 32:
        path = "results/i2n6ce/model-95000.pt"
        vision_model.load_state_dict(torch.load(path)['model'])
    if args.size == 64:
        # path = "results/74idb0/model-95000.pt"
        path = "results/6ghoqo/model-95000.pt"

        if args.resnet:
            # path =  "results/5knvwh/model-95000.pt"
            path = "results/la90ra/model-490000.pt"
        # vision_model = torch.load(path)['model_full']
        print('loading model from ', path, '...')
        vision_model.load_state_dict(torch.load(path)['model'])




trajs_latent = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vision_model = vision_model.to(device)

print("encoding data...")
with torch.no_grad():
    for traj in my_data_resized[:100]:
        traj_latent = vision_model.encode(traj.to(device))[0]
        trajs_latent.append(traj_latent.cpu())

trajs_latent = torch.stack(trajs_latent)

# rearrange to (B, channels, seq)

states = rearrange(trajs_latent, 'b seq c -> (b seq) c').to(torch.device("cpu"))
rand_idx = torch.randperm(states.size(0))

dataset = TensorDataset(states)


results_folder = pathlib.Path(f"results/{args.exp_id}")
results_folder.mkdir(parents=True, exist_ok=True)
diffusion.to(torch.device("cpu"))

def callback(model, milestone):
    """

    """
    # sample with the model
    
    yout = model.sample(batch_size=32)
    # convert to images
    vision_model_device = next(vision_model.parameters()).device
    imgs = vision_model.decode(yout.to(vision_model_device))

    fout = str(results_folder / f'sample-imgs-{milestone:05d}.png')
    nrow = 8
    print(f'saving to {fout}')
    utils.save_image(imgs, fout , nrow = nrow)                                         



trainer = TrainerMLP(
    diffusion,
    folder=None,
    train_batch_size = 32,
    train_lr = 5*1e-4,
    train_num_steps = int(1e6),         # total training steps
    save_and_sample_every = 5000,
    ema_decay = 0.995,                # exponential moving average decay
    amp = False,                       # turn on mixed precision
    calculate_fid = False,              # whether to calculate fid during training
    dataset = dataset,
    # TensorDataset(data),
    results_folder = str(results_folder),
    callback=callback,
    # autonormalize = False
    # image_model=vision_model,
)


trainer.train()

