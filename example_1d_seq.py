import torch
from denoising_diffusion_pytorch import (
    Unet1D,
    GaussianDiffusion1D,
    Trainer1D,
    Dataset1D,
    Dataset1DCond,
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
import pathlib

torch.set_num_threads(2)



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
parser.add_argument("--noise_z", type=float, default=1e-3, help="X")
parser.add_argument("--noise_y", type=float, default=1e-2, help="X")
parser.add_argument("--learn_dx", action="store_true", help="X")

args = parser.parse_args()

nz = 8
n_elements = 1

data_in = "./new_data_all_2024-09-05.pt"

if not args.train_u:
    data = torch.load(data_in)['imgs']
    nu = 0
    data = data[:, ::n_elements, ...] # take one every n_elements

else: 
    nu = 2
    # data = torch.load("./new_data_wu_img_THURSDAY.pt")
    # data_us =  torch.load("./new_data_wu_us_THURSDAY.pt")
    
    data_in = torch.load(data_in)
    data = data_in['imgs']
    data_us = data_in['us']
    data = data[:, ::n_elements, ...] # take one every n_elements
    data_us = data_us[:, ::n_elements, ...] # take one every n_elements
    # print("Caution, setting us to zero for now")
    # data_us = data_us.clamp(0.,0.)

    # Compute minimum per channel (dim=0)
    min_per_channel = torch.amin(data_us, dim=(0,1))
    print("\nus Minimum per channel:", min_per_channel)

    # Compute maximum per channel (dim=0)
    max_per_channel = torch.amax(data_us, dim=(0,1))
    print("us Maximum per channel:", max_per_channel)

    # sys.exit()


    xs = data_in['xs']
    data_xs = xs[:, ::n_elements, ...] # take one every n_elements

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
            path = "results/la90ra/model-490000.pt"  # the original i was using
            # path = "results/la90ra/model-995000.pt"
            # path = "results/nysoh1/model-995000.pt"
            path =  "results/z91yo7/model-990000.pt"
            path = "results/y6owhr/model-460000.pt"
        print('loading model from ', path, '...')
        # model_full = torch.load(path)['model_full']
        # print(type(model_full))
        vision_model.load_state_dict(torch.load(path)['model'])
        # vision_model = torch.load(path)['model_full']



vision_model.eval() # i was missing this... 


trajs_latent = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vision_model = vision_model.to(device)

num_trajs=-1
print("encoding data...")
with torch.no_grad():
    for traj in my_data_resized[:num_trajs]:
        traj_latent = vision_model.encode(traj.to(device))[0]
        trajs_latent.append(traj_latent.cpu())

trajs_latent = torch.stack(trajs_latent)



#concatenate with the us

save = True

if save:
    torch.save(
         {
            'zs': trajs_latent,
            'us': data_us[:num_trajs],
            'xs': data_xs[:num_trajs],
            'vision_model': path
        },
    "trajs_latent_all_y6owhr_460000.pt"
    )


sys.exit()

trajs_latent = torch.cat([trajs_latent, data_us[:num_trajs]], dim=2)


# torch.save(trajs_latent, "trajs_latent_v0.pt")
#note trajs latent v0 uses  the model resnet path = "results/la90ra/model-490000.pt"


# rearrange to (B, channels, seq)

trajs_latent = rearrange(trajs_latent, 'b seq c -> b c seq')



Y = trajs_latent[:,:nz, 0].clone()
trajs_latent[:, :nz, :] -= Y.unsqueeze(2)
dataset = Dataset1DCond( trajs_latent   ,Y)

rand_idx = torch.randperm(len(dataset))
y_eval = Y[rand_idx[:16]]



seq_length = int ( 16 // n_elements)
model = Unet1D(dim=64, dim_mults=(1, 2, 4), nx=8, nu=nu, ny=8)

diffusion = GaussianDiffusion1D(
    model, seq_length=seq_length, timesteps=100, objective="pred_v", auto_normalize = False, 
)


sampled_seq = diffusion.sample(batch_size=4, y = trajs_latent[:4,:nz, 0])


results_folder = pathlib.Path(f"results/{args.exp_id}")
results_folder.mkdir(parents=True, exist_ok=True)


def callback(model, milestone):
    """

    """
    samples_per_y =4
    model.eval()
    device = next(model.parameters()).device
    device_vision_model = next(vision_model.parameters()).device
    with torch.no_grad():
        ys  = y_eval[:6].repeat_interleave(samples_per_y, dim=0).to(device)
        all_samples_cond = model.sample( batch_size=ys.shape[0], y = ys)

        # add the ys
        all_samples_cond[:,:nz,:] += ys.unsqueeze(2)

        seq_length = all_samples_cond.shape[2]
        imgs = vision_model.decode(
            rearrange(all_samples_cond, 'b c seq -> (b seq) c').to(device_vision_model)[:,:nz]
            )

        fout = str(results_folder / f'sample-imgs-cond-{milestone:05d}.png')

        print(f'saving to {fout}')
        utils.save_image(imgs, fout , nrow = seq_length)                                         

        all_samples = model.sample( batch_size=y_eval.shape[0], y = y_eval.to(device))
        all_samples[:,:nz,:] += y_eval.unsqueeze(2)
        seq_length = all_samples.shape[2]
        imgs = vision_model.decode(
            rearrange(all_samples, 'b c seq -> (b seq) c').to(device_vision_model)[:,:nz]
            )
        fout = str(results_folder / f'sample-imgs-{milestone:05d}.png')
        print(f'saving to {fout}')
        utils.save_image(imgs, fout , nrow = seq_length)

    # get images from latent codes


print('training...')
trainer = Trainer1D(
    diffusion,
    dataset=dataset,
    save_and_sample_every=1000,
    train_batch_size=32,
    train_lr=1e-4,
    train_num_steps=int(1e6),  # total training steps
    gradient_accumulate_every=1,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    amp=False,  # turn on mixed precision
    callback = callback,
    results_folder = str(results_folder),
    noise_z = args.noise_z,
    noise_y = args.noise_y
)
trainer.train()
