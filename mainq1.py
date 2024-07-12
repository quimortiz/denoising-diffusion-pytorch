import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from torch.utils.data import TensorDataset
from torchvision import transforms as T, utils
from tqdm import tqdm
import pathlib

model = Unet(
    dim = 32,
    channels = 3,
    dim_mults = (1, 2,4),
    flash_attn = False
)

diffusion = GaussianDiffusion(
    model,
    image_size = 64,
    timesteps = 100    # number of steps
)

# imgs_tensor = torch.load("/home/quim/code/nlp_diffusion/imgs_tensor_64x64_color.pt")
# print("max", torch.max(imgs_tensor))
# print("min", torch.min(imgs_tensor))
# all_images = torch.cat( [i for  i in imgs_tensor[:16] ],dim=0)
# utils.save_image(all_images, "/tmp/imgs.png", 4 )

# Load the tensor
# imgs_tensor = torch.load("data/imgs_tensor_64x64_color.pt")

# import pathlib
# path_out = "tmp/all_img_color/"
# pathlib.Path(path_out).mkdir(exist_ok=True,parents=True)
# for i,img in enumerate(imgs_tensor):
#     utils.save_image(img,  f"{path_out}/{i:04d}.png")



# # Check the max and min values in the tensor
# print("max", torch.max(imgs_tensor))
# print("min", torch.min(imgs_tensor))

# # Verify the shape of imgs_tensor
# print("Shape of imgs_tensor:", imgs_tensor.shape)

# # Concatenate the first 16 images
# # all_images = torch.cat([img.unsqueeze(0) for img in imgs_tensor[:16]], dim=0)

# # Save the concatenated images as a single image grid
# utils.save_image(imgs_tensor[:16],  "/tmp/imgs.png", nrow=4)
# # utils.save_image(all_images * 2 - 1, "/tmp/imgs.png", nrow=4)



#print(type(imgs_tensor))


import random
import string
def generate_id():
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))

id = generate_id()

print("id", id)


results_folder = f"./results/{id}/"
pathlib.Path(results_folder).mkdir(exist_ok=True, parents=True)

trainer = Trainer(
    diffusion,
   # 'tmp/all_img',
    '/home/quim/code/denoising-diffusion-pytorch/tmp/all_img_color_fake_class/class1',
    train_batch_size = 32,
    train_lr = 1e-4,
    train_num_steps = 10000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    save_and_sample_every = 100,
    ema_decay = 0.995,                # exponential moving average decay
    amp = False,                       # turn on mixed precision
    calculate_fid = False,              # whether to calculate fid during training
   # dataset =  f'/home/quim/code/denoising-diffusion-pytorch/tmp/all_img_color_fake_class/class1',
    augment_horizontal_flip=False,
   # TensorDataset(imgs_tensor),
    results_folder = results_folder
)


trainer.train()
