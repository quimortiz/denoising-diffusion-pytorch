import pickle
import torch
import sys
import torchvision
import json
from PIL import Image
from torchvision import transforms
import pathlib
import matplotlib.pyplot as plt



def load_data( load_pt = False, size=32):

    if load_pt:
        folder_name = '/home/quim/code/denoising-diffusion-pytorch/image_based/plots_trajs/2024-07-11/trajs_2024-07-11--13-52-35_6NVJR7/'

            # load with pickle
        print('loading data with pickle')
        # with open(folder_name + 'datapoints.pkl', 'rb') as f:
        #     datapoints = pickle.load(f)
        with open(folder_name + 'trajectories.pkl', 'rb') as f:
            trajectories = pickle.load(f)
            print('data has been loaded with pickle!')

        # import pdb; pdb.set_trace()
        my_data = torch.stack( [ traj['imgs'] for traj in trajectories] )
        my_data_us = torch.stack( [ traj['u'] for traj in trajectories] )
        del trajectories
        import torch.nn.functional as F

        # Slice the tensor to take one every two elements
        my_data_reduced = my_data[:, ::2, ...]
        my_data_us_reduced = my_data_us[:,::2,...]

        print("max min in us", my_data_us_reduced.max(), my_data_us_reduced.min())

        my_data_reduced = my_data_reduced.clamp(.1, .9)

        if size == 32:

            my_data_resized = torch.stack( [
            F.interpolate(traj, size=(32, 32), mode='bilinear', align_corners=False) for traj in my_data_reduced
            ])
        elif size == 64:
            my_data_resized = my_data_reduced
        else:
            raise ValueError('only 32 and 64 are supported')

        torch.save(my_data_resized, f'/tmp/my_data_resized_{size}.pt')
        torch.save(my_data_us_reduced, f'/tmp/my_data_us_reduced_{size}.pt')

    else:
        if size == 32:
            my_data_resized = torch.load('/tmp/my_data_resized.pt')
            my_data_us_reduced = torch.load('/tmp/my_data_us_reduced.pt')
        elif size == 64:
            my_data_resized = torch.load('/tmp/my_data_resized_64.pt')
            my_data_us_reduced = torch.load('/tmp/my_data_us_reduced_64.pt')
    return my_data_resized, my_data_us_reduced


def load_data_v2(json_file):
    """
    """
    # load the images
    with open(json_file, 'r') as f:
        data = json.load(f)

    print(len(data))
    print(data[0].keys())
    trajs_out = []
    trajs_out_filenames = []
    trajs_us_out = []
    trajs_xs_out = []
    for i, traj in enumerate(data):
        imgs = traj['images']
        u = traj['controls']
        x = traj['states']
        u_pt = torch.tensor(u)
        x_pt = torch.tensor(x)
        # print(u_pt.shape)
        trajs_us_out.append(u_pt)
        trajs_xs_out.append(x_pt)
        # do something with the images
        # do something with the u

        traj_imgs = []
        for img in imgs:


            # Load the image
            # print("loading ", img)
            image = Image.open( "/home/quim/code/nlp_diffusion/" + img).convert("RGB")  # Convert to RGB, ignoring alpha channel

            # Define the transformation to convert the image to a tensor with values in the range [0, 1]
            transform = transforms.Compose([
                transforms.Resize((64, 64)),  # Resize the image to 64x64
                transforms.ToTensor(),  # Converts the image to a tensor and scales the pixel values to [0, 1]
            ])



            # Apply the transformation
            image_tensor = transform(image)
            # print(image_tensor.shape)


            # Convert the tensor back to a NumPy array for visualization
            # image_np = image_tensor.permute(1, 2, 0).numpy()  # Change the tensor from (C, H, W) to (H, W, C)

            # Plot the image using Matplotlib
            # plt.imshow(image_np)
            # plt.axis('off')  # Turn off axis labels
            # plt.show()


            traj_imgs.append(image_tensor)
        pt = torch.stack(traj_imgs)
        # save this to a file? 

        fout = f"./new_data/trajs_v0/trajs/traj_{id}.pt"
        pathlib.Path(fout).parent.mkdir(exist_ok=True,parents=True)
        trajs_out_filenames.append(fout)
        torch.save(pt,fout)

        trajs_out.append( pt )

    log_out = "./new_data/trajs_v0/" + "data.json"

    pathlib.Path(log_out).parent.mkdir(exist_ok=True,parents=True)
    with open(log_out, "w") as f:
        json.dump(trajs_out_filenames, f)

    trajs_out = torch.stack(trajs_out)
    trajs_us_out = torch.stack(trajs_us_out)
    trajs_xs_out = torch.stack(trajs_xs_out)
    return { "imgs": trajs_out, "us": trajs_us_out, "xs": trajs_xs_out }
        
