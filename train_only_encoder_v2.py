import sys
import torch
from transformers.models.idefics import vision

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

import sys  # noqa


print(sys.path)
sys.path = ["resnet-18-autoencoder/src"] + sys.path
# sys.path.append("resnet-18-autoencoder/src")  # noqa
# append but before.
sys.path.append("VAE-ResNet18-PyTorch")

print(sys.path)
from classes.resnet_autoencoder import AE

import torch


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()  # Total bytes

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()  # Total bytes

    total_size = param_size + buffer_size  # Combine parameter and buffer sizes
    total_size_gb = total_size / (1024**3)  # Convert bytes to GB
    return total_size_gb


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
parser.add_argument("--resnet", action="store_true", help="X")
parser.add_argument("--resnet_vae_new", action="store_true", help="X")
parser.add_argument("--train_u", action="store_true", help="X")
parser.add_argument(
    "--z_interp_weight",
    type=float,
    default=1e-5,
    help="Weight for z interpolation loss",
)  # Existing
parser.add_argument(
    "--img_interp_weight",
    type=float,
    default=1e-2,
    help="Weight for image interpolation loss",
)  # Added this line
parser.add_argument("--noise_z", type=float, default=1e-2, help="X")
parser.add_argument("--noise_img", type=float, default=1e-2, help="X")
parser.add_argument("--z_dim", type=int, default=8, help="X")
parser.add_argument(
    "--max_first_dist_weight",
    type=float,
    default=1e-10,
    help="Weight for maximizing the distance between the first states in a batch",
)

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


nz = args.z_dim
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


if args.resnet:
    vision_model = AE("light", nz=nz)  # try the default one!

elif args.resnet_vae_new:
    from model import VAE as mVAE
    from model import VAEpretrained

    use_mlp = True
    vision_model = VAEpretrained(z_dim=nz, use_mlp=use_mlp)

else:
    vision_model = VanillaVAE(in_channels=3, latent_dims=nz, size=args.size)


model_size_gb = get_model_size(vision_model)
print(f"Model size: {model_size_gb:.4f} GB")


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


train_batch_size = 32


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


# opt = Adam(vision_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


# Learning Rate: Use a smaller learning rate for the pretrained layers and a larger one for the new layers.
#
# python


if args.resnet_vae_new:

    layer_4_fixed = False
    print(" layer 4  fixed", layer_4_fixed)

    for param in vision_model.encoder.resnet.parameters():
        param.requires_grad = False
    # Unfreeze the last few layers if needed
    if not layer_4_fixed:
        for param in vision_model.encoder.resnet.layer4.parameters():
            param.requires_grad = True

    params = []
    if not layer_4_fixed:
        params += vision_model.encoder.resnet.layer4.parameters()
    if use_mlp:
        params += vision_model.encoder.mlp.parameters()
    if not use_mlp:
        params += vision_model.encoder.fc_mu.parameters()
        params += vision_model.encoder.fc_logvar.parameters()
    params += vision_model.decoder.parameters()

else:
    params = vision_model.parameters()

opt = torch.optim.Adam(params, lr=args.lr)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vision_model.to(device)
print(f"device {device}")

results_folder = pathlib.Path(f"results/{args.exp_id}")
results_folder.mkdir(parents=True, exist_ok=True)

add_noise_img = True

i = 0

fout = str(results_folder / f"model-{i:04d}.pt")
out = {"i": i, "model": vision_model.state_dict(), "model_full": vision_model}
torch.save(out, fout)


vision_model.load_state_dict(torch.load(fout)["model"])


for i in range(args.train_num_steps):
    batch = next(dl)
    vision_model.train()
    imgs = batch["imgs"].to(device)
    imgs = rearrange(imgs, "b n c h w -> (b n) c h w")
    if add_noise_img:
        input_imgs = imgs + (2 * torch.rand_like(imgs) - 1) * args.noise_img
        input_imgs = input_imgs.clamp(0.1, 0.9)
    else:
        input_imgs = imgs

    z_raw, _, _ = vision_model.encode(input_imgs)
    z_raw = z_raw + (2 * torch.rand_like(z_raw) - 1) * args.noise_z
    z_norms = torch.max(z_raw.norm(dim=-1, keepdim=True), torch.tensor(1e-5).to(device))
    z = z_raw
    z_loss = torch.mean(z**2)
    fake_imgs = vision_model.decode(z)
    img_loss = F.mse_loss(fake_imgs, imgs, reduction="mean")
    z_traj = rearrange(z, "(b n) c -> b n c", b=train_batch_size)

    # Existing loss terms
    z_traj_loss = torch.mean(
        torch.sum((z_traj[:, 1:, :] - z_traj[:, :-1, :]) ** 2, dim=-1)
    )

    b = z_traj.shape[0]
    n = z_traj.shape[1]
    z_first = z_traj[:, 0, :]
    z_last = z_traj[:, -1, :]
    t = torch.arange(n, device=z_traj.device).float() / (n - 1)
    t = t.unsqueeze(0).unsqueeze(-1)
    z_interp = z_first.unsqueeze(1) + (z_last - z_first).unsqueeze(1) * t
    z_interp_loss = torch.mean((z_traj - z_interp) ** 2)

    img_interp_loss = torch.tensor(0.0)
    if args.img_interp_weight > 1e-12:
        img_interp = vision_model.decode(rearrange(z_interp, "b n c -> (b n) c"))
        img_interp_loss = F.mse_loss(imgs, img_interp, reduction="mean")

    # New loss term: Maximize distance between first states
    distances = torch.cdist(z_first, z_first, p=2)
    mask = ~torch.eye(z_first.size(0), device=device).bool()
    avg_distance = distances[mask].mean()
    max_first_dist_loss = -avg_distance  # Negative for maximization

    # Total loss
    total_loss = (
        img_loss
        + args.z_weight * z_loss
        + args.z_diff * z_traj_loss
        + args.z_interp_weight * z_interp_loss
        + args.img_interp_weight * img_interp_loss
        + args.max_first_dist_weight * max_first_dist_loss  # New term
    )

    opt.zero_grad()
    total_loss.backward()
    opt.step()

    if i % 2000 == 0:
        print(f"step {i} loss {total_loss.item()}")
        print("Average z norm", z_raw.norm(dim=-1).mean().item())
        print(
            "Raw loss:",
            f"img_loss {img_loss.item()} z_loss {z_loss.item()} z_traj_loss {z_traj_loss.item()} z_interp_loss {z_interp_loss.item()} img_interp_loss {img_interp_loss.item()} max_first_dist_loss {max_first_dist_loss.item()}",
        )
        print(
            "Weighted Loss:",
            f"img_loss {img_loss.item()} z_loss {args.z_weight * z_loss.item()} z_traj_loss {args.z_diff * z_traj_loss.item()} z_interp_loss {args.z_interp_weight * z_interp_loss.item()} img_interp_loss {args.img_interp_weight * img_interp_loss.item()} max_first_dist_loss {args.max_first_dist_weight * max_first_dist_loss.item()}",
        )
    if i % 10000 == 0:
        seq_length = plot_data.shape[1]
        b_plot = plot_data.shape[0]
        _data = rearrange(plot_data, "b n c h w -> (b n) c h w")
        fout = str(results_folder / f"original-{i:04d}.png")
        print(f"saving to {fout}")
        utils.save_image(_data, fout, nrow=seq_length)

        # lets reconstruct the data
        vision_model.eval()
        z_plot, _, _ = vision_model.encode(
            rearrange(plot_data, "b n c h w -> (b n) c h w").to(device)
        )
        fake_imgs_plot = vision_model.decode(z_plot)
        # fake_imgs_traj = rearrange(fake_imgs, '(b n) c h w -> b n c h w', b=b)

        fout = str(results_folder / f"recon-{i:04d}.png")
        print(f"saving to {fout}")
        utils.save_image(fake_imgs_plot, fout, nrow=seq_length)

        # save the model
        fout = str(results_folder / f"model-{i:04d}.pt")
        out = {"i": i, "model": vision_model.state_dict(), "model_full": vision_model}
        torch.save(out, fout)

        # lets interpolate the data between the first and last image.
        # get all the first images:
        z_plot_traj = rearrange(z_plot, "(b n) c -> b n c", b=b_plot)
        z_first_plot = z_plot_traj[:, 0, :]
        z_last_plot = z_plot_traj[:, -1, :]
        # how many interpolations?
        z_interp_plot = torch.zeros_like(z_plot_traj)
        for j in range(seq_length):
            z_interp_plot[:, j, :] = z_first_plot + (z_last_plot - z_first_plot) * j / (
                seq_length - 1
            )

        z_interp_plot = rearrange(z_interp_plot, "b n c -> (b n) c")
        fake_imgs_interp = vision_model.decode(z_interp_plot)

        fout = str(results_folder / f"interp-{i:04d}.png")
        print(f"saving to {fout}")
        utils.save_image(fake_imgs_interp, fout, nrow=seq_length)
