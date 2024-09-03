# Let use u's also!

import torch
#from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d_encoder import GaussianDiffusion1D, Unet1D, Trainer1D, Dataset1D, Dataset1D_img_and_u

from vision_model.model import VanillaVAE

from torch import nn

import random
import string
import pickle
import sys
from example_load_data import load_data, load_data_v2
import torch
import torch.nn.functional as F


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
parser.add_argument('--y_cond_as_x', action='store_true',help='X')
parser.add_argument('--weight_decay', type=float, default=0.0, help='X')
parser.add_argument('--train_num_steps', type=int, default=100000, help='X')
parser.add_argument('--train_u', action='store_true', help='X')

args = parser.parse_args()


# args.cond = True
# args.y_cond_as_x = True

print(args)

#args.size = 64

#args.cond = True








import sys



# json_file = "/home/quim/code/nlp_diffusion/image_based/plots_trajs/2024-08-30/all_r_v0.json"
# data = load_data_v2(json_file)
# torch.save(data, "./new_data.pt")
# sys.exit()


# data = new_data.p


data_in = "new_data.pt"
data = torch.load(data_in)


# i want to reduce the sequence length from 16 to 8


# del datapoints

load_pt = True
# my_data_resized, my_data_us_reduced =  load_data( load_pt = load_pt, size=args.size)
# nu = my_data_us_reduced.shape[-1]

nz = 12
nu = 0

data_in = "new_data.pt"
data = torch.load(data_in)
data = data.clamp(.1, .9)
data = data[:, ::2, ...] # take one every two elments
my_data_resized = data

if args.size == 32:
    target_size = (32, 32)

    # Reshape each image in the dataset to 32x32 using interpolate
    # We need to reshape the tensor to combine Batch and Seq dimensions, resize, and then reshape back
    batch_size, seq_length, channels, height, width = my_data_resized.shape

    # Reshape to (Batch * Seq, Channels, H, W) to apply the resize
    dataset_reshaped = my_data_resized.view(-1, channels, height, width)

    # Apply interpolation (resize)
    dataset_resized = F.interpolate(dataset_reshaped, size=target_size, mode='bilinear', align_corners=False)

    # Reshape back to (Batch, Seq, Channels, 32, 32)
    dataset_resized = dataset_resized.view(batch_size, seq_length, channels, *target_size)
    my_data_resized = dataset_resized

# Now dataset_resized has shape (Batch, Seq, Channels, 32, 32)
    print(dataset_resized.shape)  # This should print torch.Size([10, 5, 3, 32, 32])

vision_model = VanillaVAE(in_channels=3, latent_dims=nz , size = args.size)
if args.pretrained:
    path = '/home/quim/code/Conv-VAE-PyTorch/output/ZIN9X2/state_dict_ZIN9X2_e09900.pth'
    vision_model.load_state_dict(torch.load(path))



model = Unet1D(
    dim = 32,
    dim_mults = (1, 2, 4),
    nz = nz, 
    nu = nu, 
    y_cond = args.cond,
    y_cond_as_x=args.y_cond_as_x
)

# class VAEencoder(nn.Module):
#     def __init__(self, vae):
#         super().__init__()
#         self.vae = vae
#         self.sample_noise = False
#     def __call__(self,img):
#         mu, log_var, z = self.vae.encode(img)
#         if self.sample_noise:
#             return z
#         else:
#             return mu
#     
# class VAEdecoder(nn.Module):
#     def __init__(self, vae):
#         super().__init__()
#         self.vae = vae
#     def __call__(self,z):
#         return self.vae.decode(z)
#     
# encoder = VAEencoder(vision_model)
# decoder = VAEdecoder(vision_model)
#

diffusion = GaussianDiffusion1D(
    model,
    seq_length = 8,
    timesteps = 100,
    objective = 'pred_v' ,
    auto_normalize = False,
    vision_model = vision_model,
    )


diffusion = diffusion.cuda()
loss_dict = diffusion(my_data_resized[:32].cuda(), u = my_data_us_reduced[:32].cuda() if args.train_u else None, y = my_data_resized[:32, 0, ...].cuda())
samples = diffusion.sample( y = my_data_resized[:32, 0, ...].cuda())
# print('sample', samples.shape)
# sys.exit()

loss = loss_dict[ 'z_loss' ] + loss_dict[ 'img_loss' ]
loss.backward()

# Or using trainer


print('max is ', my_data_resized.max())
print('min is ', my_data_resized.min())

# dataset = Dataset1D(my_data_resized)  # this is just an example, but you can formulate your own Dataset and pass it into the `Trainer1D` below


if args.train_u:
    dataset = Dataset1D_img_and_u(my_data_resized, my_data_us_reduced)
else:
    dataset = Dataset1D(my_data_resized)
# dataset = Dataset1D_img_and_u(my_data_resized, my_data_us_reduced)


print(f'len dataset {len(dataset)}')
print(len(dataset))


# lets load a 
trainer = Trainer1D(
    diffusion,
    dataset = dataset,
    train_batch_size = 64,
    train_lr = args.lr,
    train_num_steps = args.train_num_steps,         # total training steps
    gradient_accumulate_every = 1,    # gradient accumulation steps
    save_and_sample_every = 1000,
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    results_folder= f"./results/{args.exp_id}/",
    recon_weight=args.recon_weight,
    z_weight=args.z_weight,
    y_cond = args.cond,
    mod_lr = args.mod_lr,
    cond_combined = args.cond_combined,
    weight_decay=args.weight_decay,
    z_diff_weight = args.z_diff
)




# python    example_1d_seq_encoder_wu.py --cond --y_cond_as_x --lr 1e-3 --mod_lr
# model = 'results/sjc0bq/model-86.pt'
# _model = torch.load(model)
# diffusion.load_state_dict(_model['model'])
# this is working! TODO: lets evaluate the model a couple of time to see what happens!!
# TODO: lets evaluate the distance between the z's

trainer.train()

trainer = Trainer1D(
    diffusion,
    dataset = dataset,
    train_batch_size = 64,
    train_lr = args.lr,
    train_num_steps = args.train_num_steps,
    gradient_accumulate_every = 1,    # gradient accumulation steps
    save_and_sample_every = 1000,
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    results_folder= f"./results/{args.exp_id}-fine/",
    recon_weight=args.recon_weight,
    z_weight=args.z_weight,
    y_cond = args.cond,
    mod_lr = args.mod_lr,
    cond_combined = args.cond_combined,
    weight_decay=args.weight_decay,
    z_diff_weight = args.z_diff,
    freeze_encoder = True
)

trainer.train()


#second trainer with the 


# Lets try to train only the diffusion part!!
# results/sjc0bq/sample-cond-110.png






# after a lot of training

# sampled_seq = diffusion.sample(batch_size = 4)
# sampled_seq.shape # (4, 32, 128)





