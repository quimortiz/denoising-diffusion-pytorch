import sys
import torch

# from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d_encoder import (
    GaussianDiffusion1D,
    Unet1D,
    Trainer1D,
    Dataset1D,
    Dataset1D_img_and_u,
)

from vision_model.model import VanillaVAE

from torch import nn, ones_like

import random
import string
import pickle
import sys
from example_load_data import load_data, load_data_v2
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from einops import rearrange, reduce

import pathlib


from torchvision import utils

import sys # noqa
sys.path.append('resnet-18-autoencoder/src') # noqa
from classes.resnet_autoencoder import AE



def generate_exp_id():
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=6))


import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--exp_id", type=str, default=generate_exp_id(), help="X")
parser.add_argument("--pretrained", action="store_true", help="X")
parser.add_argument("--lr", type=float, default=1e-4, help="X")
parser.add_argument("--recon_weight", type=float, default=0.0, help="X")
parser.add_argument("--z_weight", type=float, default=1e-5, help="X")
parser.add_argument("--z_diff", type=float, default=1e-5, help="X")
parser.add_argument("--cond", action="store_true", help="X")
parser.add_argument("--size", type=int, default=32, help="X")
parser.add_argument("--mod_lr", action="store_true", help="X")
parser.add_argument("--cond_combined", action="store_true", help="X")
parser.add_argument("--y_cond_as_x", action="store_true", help="X")
parser.add_argument("--weight_decay", type=float, default=0.0, help="X")
parser.add_argument("--train_num_steps", type=int, default=100000, help="X")
parser.add_argument("--train_u", action="store_true", help="X")

args = parser.parse_args()


# args.cond = True
# args.y_cond_as_x = True

print(args)

# args.size = 64

# args.cond = True


# json_file = "/home/quim/code/nlp_diffusion/image_based/plots_trajs/2024-09-05/all_r_v1__2024-09-16--10-57-16.json"
# all_data = load_data_v2(json_file)
# print(data_img.shape)
# print(data_us.shape)
# torch.save(all_data, "./new_data_all_2024-09-05.pt")
# torch.save(data_us, "./new_data_wu_us_2024-09-05.pt")
# sys.exit()


nz = 8
n_elements = 4

if not args.train_u:

    data_in = "./new_data_all_2024-09-05.pt"

    data = torch.load(data_in)["imgs"]
    nu = 0
    data = data[:, ::n_elements, ...]  # take one every n_elements

else:
    nu = 2
    data = torch.load("./new_data_wu_img_THURSDAY.pt")
    data_us = torch.load("./new_data_wu_us_THURSDAY.pt")
    data = data[:, ::n_elements, ...]  # take one every two elments
    data_us = data_us[:, ::n_elements, ...]
    my_data_us_reduced = data_us


data = data.clamp(0.1, 0.9)
my_data_resized = data

if args.size == 32:
    target_size = (32, 32)

    # Reshape each image in the dataset to 32x32 using interpolate
    # We need to reshape the tensor to combine Batch and Seq dimensions, resize, and then reshape back
    batch_size, seq_length, channels, height, width = my_data_resized.shape

    # Reshape to (Batch * Seq, Channels, H, W) to apply the resize
    dataset_reshaped = my_data_resized.view(-1, channels, height, width)

    # Apply interpolation (resize)
    dataset_resized = F.interpolate(
        dataset_reshaped, size=target_size, mode="bilinear", align_corners=False
    )

    # Reshape back to (Batch, Seq, Channels, 32, 32)
    dataset_resized = dataset_resized.view(
        batch_size, seq_length, channels, *target_size
    )
    my_data_resized = dataset_resized

    # Now dataset_resized has shape (Batch, Seq, Channels, 32, 32)
    print(dataset_resized.shape)  # This should print torch.Size([10, 5, 3, 32, 32])

vision_model = VanillaVAE(in_channels=3, latent_dims=nz, size=args.size)



vision_model = AE('light') # try the default one!



if args.train_u:
    dataset = Dataset1D_img_and_u(my_data_resized, my_data_us_reduced)
else:
    dataset = Dataset1D(my_data_resized)
# dataset = Dataset1D_img_and_u(my_data_resized, my_data_us_reduced)


print(f"len dataset {len(dataset)}")
print(len(dataset))


def cycle(dl):
    while True:
        for data in dl:
            yield data


train_batch_size = 64


plot_data = torch.stack([dataset[i]["imgs"] for i in range(16)])  # 16 trajectories

dl = DataLoader(
    dataset,
    batch_size=train_batch_size,
    shuffle=True,
    pin_memory=False,
    num_workers=4,
    drop_last=True,
)


dl = cycle(dl)

train_lr = args.lr


opt = Adam(vision_model.parameters(), lr=train_lr, weight_decay=args.weight_decay)

max_steps = args.train_num_steps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vision_model.to(device)
print(f"device {device}")

results_folder = pathlib.Path(f"results/{args.exp_id}")
results_folder.mkdir(parents=True, exist_ok=True)

for i in range(max_steps):
    batch = next(dl)
    # print(f"batch {batch['imgs'].shape}")
    vision_model.train()
    imgs = batch["imgs"].to(device)
    imgs = rearrange(imgs, "b n c h w -> (b n) c h w")
    z_raw, _, _ = vision_model.encode(imgs)
    # normalize the z
    z_norms = torch.max(z_raw.norm(dim=-1, keepdim=True), torch.tensor(1e-5))
    # z = z_raw / z_norms
    z = z_raw
    # z_loss = torch.mean((torch.ones_like(z_norms) - z_norms) ** 2)
    z_loss = torch.mean(z ** 2)
    fake_imgs = vision_model.decode(z)
    img_loss = F.mse_loss(fake_imgs, imgs, reduction="mean")
    # z_loss = torch.mean(z ** 2)
    z_traj = rearrange(z, "(b n) c -> b n c", b=train_batch_size)
    z_traj_loss = torch.mean(
        torch.sum((z_traj[:, 1:, :] - z_traj[:, :-1, :]) ** 2, dim=-1)
    )
    total_loss = img_loss + args.z_weight * z_loss + args.z_diff * z_traj_loss
    # total_loss = img_loss
    opt.zero_grad()
    total_loss.backward()
    opt.step()

    if i % 1000 == 0:
        print(f"step {i} loss {total_loss.item()}")
        print("Average z norm", z_raw.norm(dim=-1).mean().item())
        print(
            "Raw loss:",
            f"img_loss {img_loss.item()} z_loss {z_loss.item()} z_traj_loss {z_traj_loss.item()}",
        )
        print(
            "Weighted Loss:",
            f"img_loss {img_loss.item()} z_loss {args.z_weight * z_loss.item()} z_traj_loss {args.z_diff * z_traj_loss.item()}",
        )
    if i % 5000 == 0:
        seq_length = plot_data.shape[1]
        b = plot_data.shape[0]
        _data = rearrange(plot_data, "b n c h w -> (b n) c h w")
        fout = str(results_folder / f"original-{i:04d}.png")
        print(f"saving to {fout}")
        utils.save_image(_data, fout, nrow=seq_length)

        # lets reconstruct the data
        z_plot, _, _ = vision_model.encode(
            rearrange(plot_data, "b n c h w -> (b n) c h w").to(device)
        )
        fake_imgs = vision_model.decode(z_plot)
        # fake_imgs_traj = rearrange(fake_imgs, '(b n) c h w -> b n c h w', b=b)

        fout = str(results_folder / f"recon-{i:04d}.png")
        print(f"saving to {fout}")
        utils.save_image(fake_imgs, fout, nrow=seq_length)


        # save the model 
        fout = str(results_folder / f"model-{i:04d}.pt")
        out = {
            'i': i,
            'model': vision_model.state_dict(),
            'model_full' : vision_model }
        torch.save(out, fout)

        # lets interploate the data betweeen the first and last image.
        # get all the first images:
        z_plot_traj = rearrange(z_plot, "(b n) c -> b n c", b=b)
        z_first = z_plot_traj[:, 0, :]
        z_last = z_plot_traj[:, -1, :]
        # how many interpolations?
        z_interp = torch.zeros_like(z_plot_traj)
        for j in range(seq_length):
            z_interp[:, j, :] = z_first + (z_last - z_first) * j / (seq_length - 1)

        z_interp = rearrange(z_interp, "b n c -> (b n) c")
        fake_imgs_interp = vision_model.decode(z_interp)
        
        fout = str(results_folder / f"interp-{i:04d}.png")
        print(f"saving to {fout}")
        utils.save_image(fake_imgs_interp, fout, nrow=seq_length)


# _z = self.model.encoder(_data)

# lets train only the encoder


# # Compute the difference between consecutive time steps
# diff = trajectories[:, 1:, :] - trajectories[:, :-1, :]
#
# # Compute the L2 squared norm along the last dimension (n_x)
# l2_squared_norm = torch.sum(diff ** 2, dim=-1)
#
# # l2_squared_norm now has shape (B, n_seq-1)


# second trainer with the


# Lets try to train only the diffusion part!!
# results/sjc0bq/sample-cond-110.png


# after a lot of training

# sampled_seq = diffusion.sample(batch_size = 4)
