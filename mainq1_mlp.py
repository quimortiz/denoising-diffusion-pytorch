import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer, MLPNet, GaussianDiffusionMLP, TrainerMLP
from torch.utils.data import TensorDataset
from torchvision import transforms as T, utils
from tqdm import tqdm
import pathlib
import random
import string
import matplotlib.pyplot as plt

import sys
sys.path.append( "/home/quim/code/resnet-18-autoencoder/src/")

from classes.resnet_autoencoder import AE



from scripts.utils import (
    train_epoch,
    test_epoch,
    plot_ae_outputs,
    plot_ae_interpolations,
    checkpoint,
    resume,
    get_rand_id
)



torch.set_num_threads(4)

import random
import string
from torchvision import datasets, transforms





# lets load the model!
# filename = "/home/quim/code/resnet-18-autoencoder/data/65nzbo/model_65nzbo.ckpt"
# checkpoint = torch.load(filename)
# model = AE("cnn").cpu()
# model, epoch, loss = resume(model, filename)
# model.eval()





# lets try to run diffusion using only a vector!! 

def plot_data(ax,X):
    XX = X[:,0]
    YY = X[:,1]
    ax.plot(XX,YY, 'o')

# TODO: Use an external encoder
# TODO: train jointly with the encoder


# nx = 4
# nx = 4
nx = 8

model = MLPNet(
    dim =  nx
)

vision_model = AE("light")


filename = "/home/quim/code/resnet-18-autoencoder/data/crjlgf/model_crjlgf.ckpt"


# filename = "/home/quim/code/resnet-18-autoencoder/data/65nzbo/model_65nzbo.ckpt"
checkpoint = torch.load(filename)
vision_model.load_state_dict(checkpoint['model_state_dict'])
vision_model.eval()

# Load the images



# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['val_loss']
# return model, epoch, loss

class ClampTransform:
    def __call__(self, tensor):
        return torch.clamp(tensor, 0., 1.)


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        return torch.clamp(noisy_tensor, 0.0, 1.0)

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'



transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
    ClampTransform(),
    # AddGaussianNoise(0., 0.05)
])

BATCH_SIZE=64

id = "color"

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(f'/home/quim/code/denoising-diffusion-pytorch/tmp/all_img_{id}_fake_class/',
                         transform=transform),
    batch_size=BATCH_SIZE,  # Ensure batch size is greater than 1
    shuffle=True,
    # **kwargs
)
    

train_dataset = train_loader.dataset

device = "cpu"
id = "00"
plot_ae_outputs(vision_model, "train_dataset", 0, train_dataset, device, n=10, id=id)
plot_ae_interpolations(vision_model, "train_dataset", 0, train_dataset, device, n=10, id=id)

# Let's encode the full dataset

with torch.no_grad():
    zs = torch.concat( [ vision_model.encoder_low(img[0].unsqueeze(0)) for img in train_dataset ] )

# remove grad

# zs = zs.requires_grad_(False)

# print(zs.shape)
# sys.exit()

# lets continue here :)





diffusion = GaussianDiffusionMLP(
    model,
    vector_size = nx,
    objective = "pred_noise",
    # image_size = 64,
    beta_schedule = 'cosine',
    timesteps = 100,    # number of steps
    auto_normalize= False
)

# imgs_tensor = torch.load("/home/quim/code/nlp_diffusion/imgs_tensor_64x64_color.pt")
# print("max", torch.max(imgs_tensor))
# print("min", torch.min(imgs_tensor))
# all_images = torch.cat( [i for  i in imgs_tensor[:16] ],dim=0)
# utils.save_image(all_images, "/tmp/imgs.png", 4 )

# Load the tensor
# imgs_tensor = torch.load("data/imgs_tensor_64x64_color.pt")
#
# import pathlib
# path_out = "tmp/all_img_color/"
# pathlib.Path(path_out).mkdir(exist_ok=True,parents=True)
# for i,img in enumerate(imgs_tensor):
#     utils.save_image(img,  f"{path_out}/{i:04d}.png")
#
#
#
# # Check the max and min values in the tensor
# print("max", torch.max(imgs_tensor))
# print("min", torch.min(imgs_tensor))
#
# # Verify the shape of imgs_tensor
# print("Shape of imgs_tensor:", imgs_tensor.shape)
#
# # Concatenate the first 16 images
# # all_images = torch.cat([img.unsqueeze(0) for img in imgs_tensor[:16]], dim=0)
#
# # Save the concatenated images as a single image grid
# utils.save_image(imgs_tensor[:16],  "/tmp/imgs.png", nrow=4)
# # utils.save_image(all_images * 2 - 1, "/tmp/imgs.png", nrow=4)
#
#
#
# print(type(imgs_tensor))
#
#
def generate_id():
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
#
#
#
results_folder = f"./results/{generate_id()}/"
pathlib.Path(results_folder).mkdir(exist_ok=True, parents=True)

# data = torch.randn(1024, 16)

# lets generate points in a circumference or radius .5

num_points = 1024
X = .5 * torch.cos(torch.linspace(0, 2*3.1415, num_points))
Y = .5 * torch.sin(torch.linspace(0, 2*3.1415, num_points))
X1 = .5 * torch.cos(torch.linspace(0, 2*3.1415, num_points))
Y1 = .5 * torch.sin(torch.linspace(0, 2*3.1415, num_points))

data = torch.stack([X,Y,
                    X1, Y1
                    ], dim=1)


fig, ax = plt.subplots(1,1)
plot_data(ax,data)
ax.set_aspect('equal', 'box')
ax.set_xlim(-1.5,1.5)
ax.set_ylim(-1.5,1.5)
plt.show()




# Continue here with the learned encoder!!
# Try to train everything end to end?

id = get_rand_id()
trainer = TrainerMLP(
    diffusion,
    'tmp/all_img',
    train_batch_size = 32,
    train_lr = 5*1e-3,
    train_num_steps = 100000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    save_and_sample_every = 100,
    ema_decay = 0.995,                # exponential moving average decay
    amp = False,                       # turn on mixed precision
    calculate_fid = False,              # whether to calculate fid during training
    dataset = 
    TensorDataset(zs),
    # TensorDataset(data),
    results_folder = results_folder,
    # autonormalize = False
    image_model=vision_model,
    id = id
)


trainer.train()




# X = diffusion.sample(batch_size = 128, return_all_timesteps = False)
#
#
# fig, ax = plt.subplots(1,1)
#
# plot_data(ax,X)
# ax.set_aspect('equal', 'box')
# ax.set_xlim(-1.5,1.5)
# ax.set_ylim(-1.5,1.5)
# plt.show()



# import pdb; pdb.set_trace()


        # # (h, w), channels = self.image_size, self.channels
        # sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        # return sample_fn((batch_size,self.vector_size), return_all_timesteps = return_all_timesteps)
        #


# lets generate data



# how to evaluate the model?
