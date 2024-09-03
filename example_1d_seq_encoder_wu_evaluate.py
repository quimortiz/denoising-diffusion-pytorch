import torch
#from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d_encoder import GaussianDiffusion1D, Unet1D, Trainer1D, Dataset1D, Dataset1D_img_and_u

from vision_model.model import VanillaVAE
from torchvision import transforms

from torch import nn

import random
import string
import pickle
import sys
from example_load_data import load_data
from einops import rearrange, reduce

from torchvision import utils

from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image, ImageOps, ImageDraw

def fix_seed(seed):
    # Set seed for random number generators
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # If you are using CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    
    # Make the deterministic flag true if you want reproducibility for operations with randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Example usage
fix_seed(1)  # Replace with your desired seed



def generate_exp_id():
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=6))


def recenter_img(img):


    img = img.cpu().permute(1, 2, 0)
    plt.imshow(img)
    plt.show(block=True)


    # Convert the image to a numpy array
    image_np = np.array(img)

    image = Image.fromarray((255 * image_np).astype(np.uint8))
    # Define the color range for green (adjust these values as necessary)
    lower_green = np.array([0, 100, 0]) / 255.
    upper_green = np.array([100, 255, 100]) / 255. 

    # Reshape the color boundaries to be broadcastable with the image shape
    lower_green = lower_green.reshape(1, 1, 3)
    upper_green = upper_green.reshape(1, 1, 3)

    # Create a mask for the green color
    mask = np.all((image_np >= lower_green) & (image_np <= upper_green), axis=-1)




    # Find the bounding box of the green ball
    coords = np.argwhere(mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)


    # Calculate the center of the green area
    center_y = (y_min + y_max) // 2
    center_x = (x_min + x_max) // 2

    # Draw a red cross at the center of the green ball on the original image

    copy_of_image = image.copy()

    draw = ImageDraw.Draw(copy_of_image)
    cross_size = 5  # Size of the cross
    draw.line((center_x - cross_size, center_y, center_x + cross_size, center_y), fill="red", width=1)
    draw.line((center_x, center_y - cross_size, center_x, center_y + cross_size), fill="red", width=1)
    # Display the original image with the red cross marking the center
    # copy_of_image.show()

    # Display the image using matplotlib
    plt.imshow(copy_of_image)
    plt.axis('off')  # Turn off axis labels
    plt.show(block=True)  # This will block execution until the window is closed



    # Expand the original image by 16 pixels on all sides (fill with white)
    expanded_image = ImageOps.expand(image, border=16, fill=(int(.9*255), int(.9 * 255), int(.9*255)))

    # Calculate the new center after expansion
    new_center_x = center_x + 16
    new_center_y = center_y + 16

    # Define the box for a 32x32 crop around the new center
    half_crop_size = 16
    left = max(0, new_center_x - half_crop_size)
    top = max(0, new_center_y - half_crop_size)
    right = min(expanded_image.width, new_center_x + half_crop_size)
    bottom = min(expanded_image.height, new_center_y + half_crop_size)

    # Crop the 32x32 region centered around the robot
    cropped_image = expanded_image.crop((left, top, right, bottom))

    # Display the cropped image
    # cropped_image.show()
    plt.imshow(cropped_image)
    plt.axis('off')  # Turn off axis labels
    plt.show(block=True)  # This will block execution until the window is closed

    # Optionally, save the cropped image
    cropped_image.save("/tmp/cropped_image_32x32.png")


    # cropped image is good!

    # Convert the cropped image to a PyTorch tensor
    transform = transforms.ToTensor()  # Converts a PIL image to a tensor
    tensor_image = transform(cropped_image)

    return tensor_image



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

args = parser.parse_args()


# args.cond = True
# args.y_cond_as_x = True

print(args)

#args.size = 64

#args.cond = True

















batch = 1024
w = 64

h = 64
c = 3
n = 16

rand_data = torch.rand(batch, n , c, h , w)


# i want to reduce the sequence length from 16 to 8


# del datapoints

load_pt = True
my_data_resized, my_data_us_reduced = load_data( load_pt = load_pt, size=args.size)


nu = my_data_us_reduced.shape[-1]
nz = 8



vision_model = VanillaVAE(in_channels=3, latent_dims=8 , size = args.size)
if args.pretrained:
    path = '/home/quim/code/Conv-VAE-PyTorch/output/ZIN9X2/state_dict_ZIN9X2_e09900.pth'
    vision_model.load_state_dict(torch.load(path))



model = Unet1D(
    dim = 32,
    dim_mults = (1, 2, 4),
    nz = 8, 
    nu = 2, 
    # channels = nz + nu,
    y_cond = args.cond,
    y_cond_as_x=args.y_cond_as_x
)

diffusion = GaussianDiffusion1D(
    model,
    seq_length = 8,
    timesteps = 100,
    objective = 'pred_v' ,
    auto_normalize = False,
    vision_model = vision_model,
    )

# lets load the model!
model = 'results/sjc0bq/model-109.pt'
_model = torch.load(model)
diffusion.load_state_dict(_model['model'])


diffusion = diffusion.cuda()
loss_dict = diffusion(my_data_resized[:32].cuda(), u = my_data_us_reduced[:32].cuda(), y = my_data_resized[:32, 0, ...].cuda())



# my data_resized

# lets encode and decode a trajectory

id_traj = 0

traj = my_data_resized[id_traj].cuda()

zs = diffusion.encoder( traj)


for z in zs:
    print(z)

# lets compute the distance from the first to all the 

print('norms')
z0 = zs[0]
for i, z in enumerate(zs):
    print(torch.norm(z0 - z))


id_traj_2 = 1

traj2 = my_data_resized[id_traj_2].cuda()

zs2 = diffusion.encoder(traj2)

# for z in zs2:
#     print(zs2)

# lets compute the distance from the first to all the 

print('norms')
z02 = zs2[0]
for i, z in enumerate(zs2):
    print(torch.norm(z02 - z))

for i, z in enumerate(zs2):
    print(torch.norm(z0 - z))

sys.exit()



import time
with torch.no_grad():


    y =  my_data_resized[:2, 0, ...].cuda()
    tic = time.time()
    samples = diffusion.sample( y = y)
    toc = time.time()
    print('32 : toc - tic' , toc - tic)

    y =  my_data_resized[:32, 0, ...].cuda()
    tic = time.time()
    samples = diffusion.sample( y = y)
    toc = time.time()
    print('32 : toc - tic' , toc - tic)

    y =  my_data_resized[:128, 0, ...].cuda()
    tic = time.time()
    samples = diffusion.sample( y = y)
    toc = time.time()
    print('128 : toc - tic' , toc - tic)


    y =  my_data_resized[:1024, 0, ...].cuda()
    tic = time.time()
    samples = diffusion.sample( y = y)
    toc = time.time()
    print('1024 : toc - tic' , toc - tic)

loss = loss_dict[ 'z_loss' ] + loss_dict[ 'img_loss' ]
loss.backward()


# lets check the results :)


# Or using trainer


print('max is ', my_data_resized.max())
print('min is ', my_data_resized.min())

# dataset = Dataset1D(my_data_resized)  # this is just an example, but you can formulate your own Dataset and pass it into the `Trainer1D` below

dataset = Dataset1D_img_and_u(my_data_resized, my_data_us_reduced)

batch_size = 16
n = 16

from torch.utils.data import Dataset, DataLoader
dl = DataLoader(dataset, batch_size = batch_size, shuffle = False, pin_memory = False, num_workers = 4)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = diffusion

batch = next(iter(dl))

data =batch['imgs'].to(device) 
us = batch['us'].to(device) 

y_cond = True
all_samples = diffusion.sample(batch_size=n , y = None if not y_cond else  data[:,0,...])

if y_cond:
    # we take 4 first samples, and use that as conditioning.
    _y = data[:4,0,:,:,:]
    _y = _y.repeat_interleave(4, dim=0)
    all_samples_cond = diffusion.sample(batch_size=n , y = _y)
    # we repeat them so that we have [s1,s1,s1,s1 , s2,s2,s2,s2, ...]

#all_samples_list = list(map(lambda n: self.model.sample(batch_size=n), batches))


#all_samples = torch.cat(all_samples_list, dim = 0)
seq_length = all_samples.shape[1]
all_samples = rearrange(all_samples, 'b n c h w -> (b n) c h w')
fout = "/tmp/all_samples.png"
print(f'saving to {fout}')
utils.save_image(all_samples,fout , nrow = seq_length)

_data = rearrange(data, 'b n c h w -> (b n) c h w')
fout = "/tmp/data.png"
print(f'saving to {fout}')
utils.save_image(_data,fout , nrow = seq_length)

# lets reconstruct the data
_z = model.encoder(_data)
_data_fake = model.decoder(_z)
# print the average norm
print(f'average norm of z {_z.norm(dim=1).mean()}')
fout = f"/tmp/recon.png"
print(f'saving to {fout}')
utils.save_image(_data_fake,fout , nrow = seq_length)

all_samples_cond = rearrange(all_samples_cond, 'b n c h w -> (b n) c h w')
fout = "/tmp/all_samples_cond.png"
print(f'saving to {fout}')
utils.save_image(all_samples_cond,fout , nrow = seq_length)


# lets take 
y0_id = 3
_y0 = data[y0_id:y0_id+1,0,:,:,:] # i now this one is nice!
_y0 = _y0.repeat_interleave(4, dim=0) # repeat 4 times

all_samples_cond = diffusion.sample(batch_size=4 , y = _y0)

all_samples_cond = rearrange(all_samples_cond, 'b n c h w -> (b n) c h w')
fout = "/tmp/all_samples_cond2.png"
print(f'saving to {fout}')
utils.save_image(all_samples_cond,fout , nrow = seq_length)

traj_id = 2
img = all_samples_cond[7+8*traj_id,...]

fout = f"/tmp/all_samples_cond2_{traj_id}.png"
print(f'saving to {fout}')
utils.save_image(all_samples_cond[8*traj_id:8*(traj_id+1)],fout , nrow = seq_length)


# evaluate 4 times.


img = recenter_img(img)


_y0 = img.to(device).unsqueeze(0).repeat_interleave(8, dim=0) # repeat 4 times
all_samples_cond = diffusion.sample(batch_size=_y0.shape[0] , y = _y0)

all_samples_cond = rearrange(all_samples_cond, 'b n c h w -> (b n) c h w')
fout = "/tmp/evaluated_on_fake.png"
print(f'saving to {fout}')
utils.save_image(all_samples_cond,fout , nrow = seq_length)

traj_id = 2
img = all_samples_cond[7+8*traj_id,...]


# save this trajectoy
fout = f"/tmp/evaluated_on_fake_{traj_id}.png"
print(f'saving to {fout}')
utils.save_image(all_samples_cond[8*traj_id:8*(traj_id+1)],fout , nrow = seq_length)


img = recenter_img(img)
_y0 = img.to(device).unsqueeze(0).repeat_interleave(8, dim=0) # repeat 4 times
all_samples_cond = diffusion.sample(batch_size=_y0.shape[0] , y = _y0)

all_samples_cond = rearrange(all_samples_cond, 'b n c h w -> (b n) c h w')
fout = "/tmp/evaluated_on_fake2.png"
print(f'saving to {fout}')
utils.save_image(all_samples_cond,fout , nrow = seq_length)

traj_id = 2
fout = f"/tmp/evaluated_on_fake2_{traj_id}.png"
print(f'saving to {fout}')
utils.save_image(all_samples_cond[8*traj_id:8*(traj_id+1)],fout , nrow = seq_length)

# it is possible to concatenate!!

