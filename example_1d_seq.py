import torch
from denoising_diffusion_pytorch import (
    Unet1D,
    GaussianDiffusion1D,
    Trainer1D,
    Dataset1D,
)

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
n_elements = 2
seq_length = int ( 16 // n_elements)
vision_model = VanillaVAE(in_channels=3, latent_dims=nz , size = args.size)
model = Unet1D(dim=64, dim_mults=(1, 2, 4), channels=8)

diffusion = GaussianDiffusion1D(
    model, seq_length=seq_length, timesteps=100, objective="pred_v"
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

with torch.no_grad():
    for traj in my_data_resized:
        traj_latent = vision_model.encode(traj.to(device))[0]
        trajs_latent.append(traj_latent.cpu())

trajs_latent = torch.stack(trajs_latent)
print("trajs latent are ready!!")

# rearrange to (B, channels, seq)

trajs_latent = rearrange(trajs_latent, 'b seq c -> b c seq')


dataset = Dataset1D(
    trajs_latent
)  # this is just an example, but you can formulate your own Dataset and pass it into the `Trainer1D` below


sampled_seq = diffusion.sample(batch_size=4)


import pathlib
results_folder = pathlib.Path(f"results/{args.exp_id}")
results_folder.mkdir(parents=True, exist_ok=True)


def callback(all_samples, milestone):
    """

    """
    # get images from latent codes
    seq_length = all_samples.shape[2]
    imgs = vision_model.decode(
        rearrange(all_samples, 'b c seq -> (b seq) c')
        )

    fout = str(results_folder / f'sample-imgs-{milestone}.png')
    print(f'saving to {fout}')
    utils.save_image(imgs, fout , nrow = seq_length)                                         


trainer = Trainer1D(
    diffusion,
    dataset=dataset,
    train_batch_size=32,
    train_lr=1e-4,
    train_num_steps=int(1e6),  # total training steps
    gradient_accumulate_every=1,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    amp=False,  # turn on mixed precision
    callback = callback,
    results_folder = str(results_folder)
)
trainer.train()
#
# # after a lot of training
#
sampled_seq = diffusion.sample(batch_size=4)
# sampled_seq.shape  # (4, 32, 128)
