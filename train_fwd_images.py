import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from einops import rearrange
from torchvision import utils
import random
import string
import argparse
import pathlib

from vision_model.model import VanillaVAE

import einops

import torch.nn.functional as F
from einops import rearrange, reduce
from torch.utils.data import Dataset

import sys  # noqa

sys.path.append("resnet-18-autoencoder/src")  # noqa
from classes.resnet_autoencoder import AE


# Define MLPForwardModel as above

torch.set_num_threads(1)

import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from typing import List, Dict, Any, Union
import torch
import numpy as np


def get_n_samples_from_dataset(dataset, n_samples, device, randomized=False):
    """
    Extracts n_samples from the given dataset.

    Args:
        dataset (torch.utils.data.Dataset): The Dataset to extract samples from.
        n_samples (int): The number of samples to extract.
        device (torch.device): The device to move the samples to.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        dict: A dictionary containing the extracted samples for each key.
    """

    dataset_size = len(dataset)
    if n_samples > dataset_size:
        raise ValueError(
            f"Requested {n_samples} samples, but the dataset only contains {dataset_size} samples."
        )

    # Randomly sample unique indices
    if randomized:
        indices = random.sample(range(dataset_size), n_samples)
    else:
        indices = list(range(n_samples))

    # Initialize a dictionary to hold the samples
    samples = {}
    for idx in indices:
        sample = dataset[idx]  # Assuming dataset[idx] returns a dict
        for key, value in sample.items():
            if key not in samples:
                samples[key] = []
            samples[key].append(value.to(device))

    # Stack lists into tensors
    for key in samples:
        samples[key] = torch.stack(samples[key], dim=0)

    return samples


def mean_of_dicts(list_of_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes the mean of values for each key across a list of dictionaries.

    Args:
        list_of_dicts (List[Dict[str, Any]]): A list where each element is a dictionary with identical keys.

    Returns:
        Dict[str, Any]: A dictionary with the same keys, where each value is the mean of the corresponding values in the input dictionaries.

    Raises:
        ValueError: If input list is empty or dictionaries have differing keys.
        TypeError: If values corresponding to the same key are of incompatible types.
    """
    if not list_of_dicts:
        raise ValueError("The input list is empty.")

    # Ensure all dictionaries have the same keys
    first_keys = set(list_of_dicts[0].keys())
    for idx, d in enumerate(list_of_dicts):
        if set(d.keys()) != first_keys:
            raise ValueError(f"Dictionary at index {idx} has different keys.")

    # Initialize a defaultdict to accumulate values
    accumulators = defaultdict(list)

    # Populate the accumulators
    for d in list_of_dicts:
        for key, value in d.items():
            accumulators[key].append(value)

    # Compute the mean for each key
    mean_dict = {}
    for key, values in accumulators.items():
        # Determine the type of the first value to decide how to compute the mean
        first_val = values[0]

        if isinstance(first_val, torch.Tensor):
            # Stack tensors and compute mean
            try:
                stacked = torch.stack(values)
                mean_val = torch.mean(stacked, dim=0)
            except Exception as e:
                raise TypeError(
                    f"Error computing mean for key '{key}' with tensor values: {e}"
                )

        elif isinstance(first_val, np.ndarray):
            # Stack numpy arrays and compute mean
            try:
                stacked = np.stack(values)
                mean_val = np.mean(stacked, axis=0)
            except Exception as e:
                raise TypeError(
                    f"Error computing mean for key '{key}' with numpy array values: {e}"
                )

        elif isinstance(first_val, (int, float)):
            # Compute arithmetic mean for numeric values
            try:
                mean_val = sum(values) / len(values)
            except Exception as e:
                raise TypeError(
                    f"Error computing mean for key '{key}' with numeric values: {e}"
                )

        else:
            raise TypeError(
                f"Unsupported value type for key '{key}': {type(first_val)}"
            )

        mean_dict[key] = mean_val

    return mean_dict


class MLPForwardModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(MLPForwardModel, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, state, control):
        # Concatenate state and control inputs
        predict_delta = True
        x = torch.cat([state, control], dim=1)
        if predict_delta:
            next_state = self.model(x) + state
        else:
            next_state = self.model(x)
        return next_state


# Define TrajectoryDataset as above


class TrajectoryDataset(Dataset):
    def __init__(self, imgs, us, xs):
        """
        trajs_latent: Tensor of shape (batch_size, channels, total_sequence_length)
        us: Tensor of shape (batch_size, control_channels, total_sequence_length)
        sequence_length: Number of steps in each sample (e.g., 5 steps)
        """
        self.imgs = imgs
        self.us = us
        self.xs = xs
        assert imgs.shape[0] == us.shape[0]
        assert imgs.shape[0] == xs.shape[0]

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        return {"imgs": self.imgs[idx], "us": self.us[idx], "xs": self.xs[idx]}


def generate_exp_id():
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=6))


# TODO: continue from here!! -- check this out!
def generate_special_trajectories(
    model, test_trajs_latent, test_data_us, vision_model, N, M, results_folder, epoch
):
    """
    Generates M trajectories, each is a concatenation of N trajectories.

    Args:
        model: The forward model.
        test_trajs_latent: Tensor of shape (num_test_trajectories, channels, seq_length).
        test_data_us: Tensor of shape (num_test_trajectories, control_channels, seq_length).
        vision_model: VAE model for decoding latent states to images.
        N: Number of concatenations per trajectory.
        M: Number of trajectories to generate.
        results_folder: Folder to save the generated trajectory images.
        epoch: Current epoch number (for naming saved images).
    """
    model.eval()
    vision_model.eval()
    device = next(model.parameters()).device
    vision_device = next(vision_model.parameters()).device

    with torch.no_grad():
        num_trajectories, channels, seq_length = test_trajs_latent.shape
        control_channels = test_data_us.shape[1]

        # Precompute the start states of all test trajectories
        start_states = test_trajs_latent[:, :, 0].to(
            device
        )  # Shape: (num_trajectories, C)

        generated_trajectories = []

        for m in range(M):
            # Select a random starting trajectory index from the test set
            traj_idx = random.randint(0, num_trajectories - 1)
            start_state = test_trajs_latent[traj_idx, :, 0].to(device)  # Shape: (C,)
            # print(f"Trajectory {m+1}/{M}: Selected starting trajectory index: {traj_idx}")

            generated_states = [start_state]
            current_state = start_state

            for n in range(N):
                # Compute distances between current_state and all test start states
                distances = torch.norm(start_states - current_state.unsqueeze(0), dim=1)

                # Find the index of the closest trajectory
                closest_idx = torch.argmin(distances).item()
                print(
                    f"Trajectory {m+1}, Step {n+1}/{N}: Closest trajectory index: {closest_idx} with distance: {distances[closest_idx].item():.4f}"
                )

                # Retrieve the control inputs from the closest trajectory
                controls = test_data_us[
                    closest_idx
                ]  # Shape: (control_channels, seq_length)

                # Apply the control inputs step by step
                for u in controls.permute(
                    1, 0
                ):  # Transpose to get shape (seq_length, control_channels)
                    # Apply the control input to predict the next state
                    current_state = model(
                        current_state.unsqueeze(0), u.unsqueeze(0)
                    ).squeeze(
                        0
                    )  # Shape: (C,)
                    generated_states.append(current_state)

                print(
                    f"Trajectory {m+1}, Step {n+1}/{N}: Applied control inputs and predicted next states."
                )

            # Stack the generated states into a tensor and add to the list
            generated_states_tensor = torch.stack(
                generated_states
            )  # Shape: (total_length, C)
            generated_trajectories.append(generated_states_tensor)

        # Stack all trajectories into a tensor
        generated_trajectories_tensor = torch.stack(
            generated_trajectories
        )  # Shape: (M, total_length, C)

        # Reshape for decoding
        M, total_length, C = generated_trajectories_tensor.shape
        decoded_imgs = vision_model.decode(
            generated_trajectories_tensor.view(-1, C).to(vision_device)
        )

        # Save the images with each row representing a trajectory
        fout = str(results_folder / f"special-trajectories-epoch-{epoch+1:05d}.png")
        print(f"Saving special trajectories visualization to {fout}")
        utils.save_image(decoded_imgs.cpu(), fout, nrow=total_length)


def generate_special_trajectory(
    model, test_trajs_latent, test_data_us, vision_model, N, results_folder, epoch
):
    """
    Generates a special trajectory by selecting control inputs from the closest matching trajectories.

    Args:
        model: The forward model.
        test_trajs_latent: Tensor of shape (num_test_trajectories, channels, seq_length).
        test_data_us: Tensor of shape (num_test_trajectories, control_channels, seq_length).
        vision_model: VAE model for decoding latent states to images.
        N: Number of steps to generate.
        results_folder: Folder to save the generated trajectory images.
        epoch: Current epoch number (for naming saved images).
    """
    model.eval()  # Set model to evaluation mode
    device = next(model.parameters()).device
    vision_device = next(vision_model.parameters()).device
    with torch.no_grad():
        num_trajectories, channels, seq_length = test_trajs_latent.shape
        control_channels = test_data_us.shape[1]

        # Select a random trajectory index from the test set
        traj_idx = random.randint(0, num_trajectories - 1)
        start_state = test_trajs_latent[traj_idx, :, 0].to(device)  # Shape: (C,)
        # print(f"Selected starting trajectory index: {traj_idx}")

        generated_states = [start_state]
        current_state = start_state

        start_states = test_trajs_latent[:, :, 0].to(
            device
        )  # Shape: (num_trajectories, C)
        for step in range(N):
            # Compute distances between current_state and all test start states
            # test_trajs_latent[:, :, 0] has shape (num_trajectories, C)
            distances = torch.norm(
                start_states - current_state.unsqueeze(0), dim=1
            )  # Shape: (num_trajectories,)

            # Find the index of the closest trajectory
            closest_idx = torch.argmin(distances).item()
            # print(f"Step {step+1}: Closest trajectory index: {closest_idx} with distance: {distances[closest_idx].item():.4f}")
            #
            # Retrieve the control inputs from the closest trajectory
            controls = test_data_us[
                closest_idx
            ]  # Shape: (control_channels, seq_length)

            # Select the control input corresponding to the current step
            # To avoid index out of range, use modulo
            for u in torch.transpose(controls, 0, 1):
                # Apply the control input to predict the next state
                current_state = model(
                    current_state.unsqueeze(0), u.unsqueeze(0)
                ).squeeze(
                    0
                )  # Shape: (C,)
                generated_states.append(current_state)

            # print(f"Step {step+1}: Applied control input and predicted next states.")

        # Stack the generated states into a tensor of shape (1, C, N+1)
        generated_states_tensor = torch.stack(generated_states)

        # Decode the generated latent states to images
        decoded_imgs = vision_model.decode(
            generated_states_tensor.to(vision_device)
        )  # Shape: (1, C, N+1)

        # Rearrange to (B*(N+1), C)
        # decoded_imgs = rearrange(decoded_imgs, "b c seq -> (b seq) c")

        # Save the images
        fout = str(results_folder / f"special-trajectory-epoch-{epoch+1:05d}.png")
        # print(f"saving special trajectory visualization to {fout}")
        utils.save_image(decoded_imgs.cpu(), fout, nrow=16)  # Arrange images in a row


def evaluate_model(
    model,
    dataloader,
    criterion,
    sequence_length,
    noise_z,
    noise_u,
    one_step_weight,
    multi_step_weight,
    tag="",
    vision_model=None,
    save_images=False,
    results_folder=None,
    epoch=None,
):
    """
    Evaluates the model on the provided dataloader.

    Args:
        model: The forward model to evaluate.
        dataloader: DataLoader for the dataset (train or test).
        criterion: Loss function.
        device: Device to run the evaluation on.
        sequence_length: Number of steps in each sample.
        noise_z: Noise weight for z.
        noise_u: Noise weight for u.
        one_step_weight: Weight for one-step loss.
        multi_step_weight: Weight for multi-step loss.
        vision_model: VAE model for decoding (required if save_images is True).
        save_images: Whether to save predicted images.
        results_folder: Folder to save images.
        epoch: Current epoch number (for naming saved images).

    Returns:
        avg_total_loss: Combined weighted loss.
        avg_loss_one_step: Average one-step loss.
        avg_loss_multi_step: Average multi-step loss.
    """
    model.eval()  # Set model to evaluation mode
    total_loss_one_step = 0.0
    total_loss_multi_step = 0.0
    total_samples = 0

    # To handle image saving, we can use the first batch
    images_saved = False
    device = next(model.parameters()).device
    device_vision = (
        next(vision_model.parameters()).device if vision_model is not None else None
    )
    with torch.no_grad():
        for batch_idx, (states, controls) in enumerate(dataloader):
            states = states.to(device)
            controls = controls.to(device)

            batch_size, channels, seq_length = states.shape
            control_channels = controls.shape[1]

            # One-step prediction
            current_state = states[:, :, 0]  # Shape: (batch_size, channels)
            control_input = controls[:, :, 0]  # Shape: (batch_size, control_channels)
            next_state = states[:, :, 1]  # Shape: (batch_size, channels)

            predicted_next_state = model(
                current_state + noise_z * torch.randn_like(current_state),
                control_input + noise_u * torch.randn_like(control_input),
            )
            loss_one_step = criterion(predicted_next_state, next_state)

            # Multi-step predictions
            predicted_states = [current_state, predicted_next_state]
            for t in range(1, sequence_length - 1):
                control_input_t = controls[:, :, t]
                predicted_next = model(
                    predicted_states[-1]
                    + noise_z * torch.randn_like(predicted_states[-1]),
                    control_input_t + noise_u * torch.randn_like(control_input_t),
                )
                predicted_states.append(predicted_next)

            # Compute multi-step loss
            loss_multi_step = 0.0
            for t in range(1, sequence_length):
                loss_t = criterion(predicted_states[t], states[:, :, t])
                loss_multi_step += loss_t

            total_loss_one_step += loss_one_step.item() * batch_size
            total_loss_multi_step += loss_multi_step.item() * batch_size
            total_samples += batch_size

            # Save images for the first batch if required
            if save_images and not images_saved:
                if (
                    vision_model is not None
                    and results_folder is not None
                    and epoch is not None
                ):
                    predicted_states_tensor = torch.stack(
                        predicted_states, dim=2
                    )  # Shape: (B, C, Seq)
                    # Decode the predicted latent states to images
                    imgs = vision_model.decode(
                        rearrange(predicted_states_tensor, "b c seq -> (b seq) c").to(
                            device_vision
                        )
                    ).cpu()
                    # Save the images
                    fout = str(
                        results_folder / f"sample-imgs-{tag}-epoch-{epoch+1:05d}.png"
                    )
                    # print(f"Saving sample images to {fout}")
                    utils.save_image(imgs, fout, nrow=sequence_length)
                    images_saved = True

    avg_loss_one_step = total_loss_one_step / total_samples
    avg_loss_multi_step = total_loss_multi_step / total_samples
    avg_total_loss = (
        one_step_weight * avg_loss_one_step + multi_step_weight * avg_loss_multi_step
    )

    return avg_total_loss, avg_loss_one_step, avg_loss_multi_step


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--exp_id", type=str, default=generate_exp_id(), help="Experiment ID"
    )
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained VAE")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--train_num_steps", type=int, default=100000, help="Number of training steps"
    )
    parser.add_argument(
        "--fix_encoder", action="store_true", help="Fix the VAE encoder"
    )
    parser.add_argument("--resnet", action="store_true", help="Use ResNet encoder")
    parser.add_argument(
        "--noise_z", type=float, default=1e-2, help="Noise weight for z"
    )
    parser.add_argument(
        "--noise_u", type=float, default=1e-2, help="Noise weight for u"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--z_predict", type=float, default=1e-1, help="Weight for z prediction loss"
    )
    parser.add_argument(
        "--z_reg", type=float, default=1e-5, help="Weight for z regularization loss"
    )

    args = parser.parse_args()
    print(args)

    nu = 2
    nz = 12  # Latent dimension
    n_elements = 1

    # Initialize Vision Model
    if args.resnet:
        vision_model = AE("light", nz=nz)  # Ensure this is correctly implemented
    else:
        vision_model = VanillaVAE(in_channels=3, latent_dims=nz, size=64)

    # # Load Pretrained Weights if specified
    # if args.size == 32:
    #     path = "results/i2n6ce/model-95000.pt"
    #     vision_model.load_state_dict(torch.load(path)["model"])
    # elif args.size == 64:
    #     # path = "results/la90ra/model-995000.pt"  # Adjust the path as needed
    #     # path = "results/la90ra/model-490000.pt"  # the original i was using
    #     path = "results/z91yo7/model-990000.pt"  # the one i used for the 64x64
    #     path = "results/y6owhr/model-460000.pt"
    #     print("Loading model from ", path, "...")
    #     vision_model.load_state_dict(torch.load(path)["model"])
    # elif args.size == 224:
    #     path = "path_to_pretrained_model_for_224.pt"  # Provide the correct path
    #     print("Loading model from ", path, "...")
    #     vision_model.load_state_dict(torch.load(path)["model"])
    # else:
    #     raise ValueError(f"Unsupported size {args.size} for pretrained model.")

    vision_model.eval()  # Set to evaluation mode

    # Encode data using the vision model
    device_model = "cpu"
    device_vision = "cuda" if torch.cuda.is_available() else "cpu"

    device_model = device_vision = device = "cuda"

    vision_model = vision_model.to(device_vision)

    data_in = "./new_data_all_2024-09-05.pt"
    data_in = torch.load(data_in)
    imgs = data_in["imgs"]
    us = data_in["us"]
    xs = data_in["xs"]

    # data = torch.load("trajs_latent_all_v1.pt")
    # data = torch.load("trajs_latent_all_z91yo7.pt")
    # v1.pt")
    # generate the data using the vision model

    # trajs_latent = data["zs"]
    # trajs_x = data["xs"]
    # data_us = data["us"]

    print("before")
    print("max in data_us", torch.max(us))
    print("min in data_us", torch.min(us))
    print("sum squares ", torch.sum(us**2))

    # data_us = data_us[:,:,:] - data_xs[:,:,3:5]

    print("after")
    print("max in data_us", torch.max(us))
    print("min in data_us", torch.min(us))
    print("sum squares ", torch.sum(us**2))

    # data_us = einops.rearrange(data_us, "b seq c -> b c seq")

    # Rearrange to (B, channels, seq)
    # trajs_latent = rearrange(trajs_latent, "b seq c -> b c seq")

    # -------------------- Data Splitting Begins Here -------------------- #

    # Set random seed for reproducibility
    random_seed = 42
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    num_trajectories = imgs.shape[0]

    test_size = int(0.05 * num_trajectories)  # 5% for testing
    train_size = num_trajectories - test_size

    # Generate shuffled indices
    indices = list(range(num_trajectories))
    random.shuffle(indices)

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Create Dataset instances

    train_dataset = TrajectoryDataset(
        imgs=imgs[train_indices],
        xs=xs[train_indices],
        us=us[train_indices],
    )
    test_dataset = TrajectoryDataset(
        imgs=imgs[test_indices],
        xs=xs[test_indices],
        us=us[test_indices],
    )

    # Create DataLoader instances
    dataloader_train = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    dataloader_test = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    train_small_eval = get_n_samples_from_dataset(train_dataset, 32, device)
    test_small_eval = get_n_samples_from_dataset(test_dataset, 32, device)

    # -------------------- Data Splitting Ends Here -------------------- #

    # Initialize Forward Model
    input_dim = nz + nu  # Latent dimension + control input dimension
    hidden_dim = 128
    output_dim = nz  # Predicting next latent state

    forward_model = MLPForwardModel(input_dim, hidden_dim, output_dim, num_layers=4).to(
        device_model
    )

    # Define Loss Function
    criterion = nn.MSELoss()

    # Optionally, fix the encoder if specified
    params = []
    params = list(forward_model.parameters())
    if args.fix_encoder:
        for param in vision_model.parameters():
            param.requires_grad = False
    else:
        params += list(vision_model.parameters())

    # Define Optimizer
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    # Training Loop
    num_epochs = args.train_num_steps // len(dataloader_train) + 1
    forward_model.train()
    vision_model.train()

    results_folder = pathlib.Path(f"results/{args.exp_id}")
    (results_folder / "imgs").mkdir(parents=True, exist_ok=True)
    (results_folder / "checkpoints").mkdir(parents=True, exist_ok=True)

    # Compute multi-step loss
    img_recon = 1.0
    img_predict = 1.0
    z_predict = args.z_predict
    z_reg = args.z_reg

    def compute(data):
        xs = data["xs"].to(device)
        us = data["us"].to(device)
        imgs = data["imgs"].to(device)
        sequence_length = xs.shape[1]

        zs = rearrange(
            vision_model.encode(rearrange(imgs, "b seq ... -> (b seq) ..."))[0],
            "(b seq) ... -> b seq ...",
            seq=sequence_length,
        )

        imgs_recon = rearrange(
            vision_model.decode(rearrange(zs, "b seq ... -> (b seq) ...")),
            "(b seq) ... -> b seq ...",
            seq=sequence_length,
        )

        current_z = zs[:, 0, :]

        # Multi-step predictions
        predicted_zs = [current_z]
        for t in range(1, sequence_length):
            control_input_t = us[:, t, :]
            predicted_z = forward_model(
                predicted_zs[-1] + args.noise_z * torch.randn_like(predicted_zs[-1]),
                control_input_t + args.noise_u * torch.randn_like(control_input_t),
            )
            predicted_zs.append(predicted_z)

        predicted_zs = torch.stack(predicted_zs, dim=1)

        # print("predicted_zs", predicted_zs.shape)
        imgs_predicted = rearrange(
            vision_model.decode(rearrange(predicted_zs, "b seq ... -> (b seq) ...")),
            "(b seq) ... -> b seq ...",
            seq=sequence_length,
        )
        # print(imgs_predicted.shape)

        loss_multistep_z = torch.tensor(0.0).to(device)
        loss_multistep_img = torch.tensor(0.0).to(device)

        # for t in range(sequence_length):
        loss_z = criterion(predicted_zs, zs)
        loss_img = criterion(imgs_predicted, imgs)
        loss_multistep_z += loss_z
        loss_multistep_img += loss_img

        # Combine losses
        loss_img_recon = criterion(imgs_recon, imgs)
        total_loss = (
            z_predict * loss_multistep_z
            + img_predict * loss_multistep_img
            + img_recon * loss_img_recon
        )
        total_loss += z_reg * torch.mean(zs**2)

        return {
            "loss": {
                "total_loss": total_loss,
                "loss_multistep_z": loss_multistep_z,
                "loss_multistep_img": loss_multistep_img,
                "loss_img_recon": loss_img_recon,
                "z_reg": torch.mean(zs**2),
                "w_loss_multistep_z": z_predict * loss_multistep_z,
                "w_loss_multistep_img": img_predict * loss_multistep_img,
                "w_loss_img_recon": img_recon * loss_img_recon,
                "w_z_reg": z_reg * torch.mean(zs**2),
            },
            "imgs_recon": imgs_recon,
            "imgs_predicted": imgs_predicted,
            "predicted_zs": predicted_zs,
            "imgs": imgs,
        }

    def loss_info(out):
        str_out = ""
        for key, value in out.items():
            str_out += f"{key}: {value.item():.3e}  "
        print(str_out)

    def save_images(out, tag):
        imgs = out["imgs"]
        imgs_recon = out["imgs_recon"]
        imgs_predicted = out["imgs_predicted"]

        sequence_length = imgs.shape[1]

        fout = results_folder / f"imgs/orig-imgs-{tag}.png"
        utils.save_image(
            rearrange(imgs, "b seq ... -> (b seq) ..."), fout, nrow=sequence_length
        )

        fout = results_folder / f"imgs/recon-imgs-{tag}.png"
        utils.save_image(
            rearrange(imgs_recon, "b seq ... -> (b seq) ..."),
            fout,
            nrow=sequence_length,
        )

        fout = results_folder / f"imgs/predicted-imgs-{tag}.png"
        utils.save_image(
            rearrange(imgs_predicted, "b seq ... -> (b seq) ..."),
            fout,
            nrow=sequence_length,
        )

    print("experiment id: ", args.exp_id)
    for epoch in range(num_epochs):

        forward_model.train()  # Ensure model is in training mode
        if args.fix_encoder:
            vision_model.eval()
        else:
            vision_model.train()

        for batch_idx, data in enumerate(dataloader_train):

            out = compute(data)
            total_loss = out["loss"]["total_loss"]

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Logging
            if batch_idx % 1000 == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs} -- batch {batch_idx+1}/{len(dataloader_train)} Train set"
                )
                loss_info(out["loss"])

        # get a batch from the test set.
        vision_model.eval()
        forward_model.eval()
        with torch.no_grad():
            infos = []
            for batch_idx, data in enumerate(dataloader_test):
                out = compute(data)
                infos.append(out["loss"])

            info = mean_of_dicts(infos)
            print(f"Epoch {epoch+1}/{num_epochs} -- test set")
            loss_info(info)

            out = compute(train_small_eval)
            save_images(out, f"train-e{epoch+1:05d}")

            out = compute(test_small_eval)
            save_images(out, f"test-e{epoch+1:05d}")

        fout = results_folder / f"checkpoints/model-epoch-{epoch+1:05d}.pt"
        torch.save(
            {
                "fwd_state_dict": forward_model.state_dict(),
                "fwd_model": forward_model,
                "vision_state_dict": vision_model.state_dict(),
                "vison_model": vision_model,
            },
            str(fout),
        )

    print("Training completed!")


if __name__ == "__main__":
    main()
