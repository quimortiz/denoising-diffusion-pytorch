import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from einops import rearrange
from torchvision import utils
import random
import string
import pathlib
import sys
import datetime
import wandb

# Add custom module paths
sys.path.append("resnet-18-autoencoder/src")  # Adjust as needed

from vision_model.model import VanillaVAE
from classes.resnet_autoencoder import AE

# Import Hydra
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, to_absolute_path

# Import TensorBoard SummaryWriter
from torch.utils.tensorboard import SummaryWriter

from collections import defaultdict
from typing import List, Dict, Any
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
    def __init__(self, nz, nu,  hidden_dim=64, num_layers=3):
        super(MLPForwardModel, self).__init__()
        layers = []
        input_dim = nz + nu
        hidden_dim = nz
        output_dim = nz
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


class TrajectoryDatasetDisk(Dataset):
    def __init__(self, base_path, file_list):
        """
        trajs_latent: Tensor of shape (batch_size, channels, total_sequence_length)
        us: Tensor of shape (batch_size, control_channels, total_sequence_length)
        sequence_length: Number of steps in each sample (e.g., 5 steps)
        """
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = torch.load(self.file_list[idx])
        return data


def generate_exp_id():
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=6))


exp_id = generate_exp_id()
print("experiment id: ", exp_id)


OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    config_path="configs/autoencoder_fwd", config_name="config", version_base=None
)
def main(config: DictConfig):
    print("Configuration:\n", OmegaConf.to_yaml(config))

    # [wandb] Initialize wandb
    wandb.init(
        project=config.wandb.project_name,
        config=OmegaConf.to_container(config, resolve=True),
    )

    # Set random seed for reproducibility
    random_seed = config.training.seed
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    vision_model = instantiate(config.vision_model.model)
    vision_model = vision_model.to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    vision_model.eval()  # Set to evaluation mode

    # Encode data using the vision model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision_model = vision_model.to(device)

    data_folder = config.data_folder

    # get how many files are in data_folder
    num_trajs = len(list(pathlib.Path(data_folder).rglob("*.pt")))
    print(num_trajs, "num_trajs")

    num_trajectories = num_trajs

    test_size = int(
        config.training.test_size_ratio * num_trajectories
    )  # Define test_size_ratio in config
    train_size = num_trajectories - test_size

    # Generate shuffled indices
    indices = list(range(num_trajectories))
    random.shuffle(indices)

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Create Dataset instances
    train_dataset = TrajectoryDatasetDisk(
        data_folder, [f"{data_folder}/traj_{i:05d}.pt" for i in train_indices]
    )

    test_dataset = TrajectoryDatasetDisk(
        data_folder, [f"{data_folder}/traj_{i:05d}.pt" for i in test_indices]
    )

    # Create DataLoader instances
    dataloader_train = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=False,
        drop_last=True,
    )

    dataloader_test = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=False,
        drop_last=True,
    )

    train_small_eval = get_n_samples_from_dataset(train_dataset, 32, device)
    test_small_eval = get_n_samples_from_dataset(test_dataset, 32, device)

    # -------------------- Data Splitting Ends Here -------------------- #

    # Instantiate Forward Model using Hydra
    forward_model = instantiate(config.fwd_model.model).to(device)

    # Define Loss Function
    criterion = nn.MSELoss()

    # Optionally, fix the encoder if specified
    params = list(forward_model.parameters())
    if config.training.fix_encoder:
        for param in vision_model.parameters():
            param.requires_grad = False
    else:
        params += list(vision_model.parameters())

    # Define Optimizer
    optimizer = optim.Adam(
        params, lr=config.training.lr, weight_decay=config.training.weight_decay
    )

    # Training Loop
    num_epochs = config.training.train_num_steps // len(dataloader_train) + 1
    forward_model.train()
    vision_model.train()


    # get time stamp in format YYYY-MM-DD--HH-MM-SS
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

    results_folder = pathlib.Path(f"results/{exp_id}")
    (results_folder / "imgs").mkdir(parents=True, exist_ok=True)
    (results_folder / "checkpoints").mkdir(parents=True, exist_ok=True)

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=to_absolute_path(f"runs/{exp_id}__{time_stamp}"))

    # [wandb] Log hyperparameters
    wandb.config.update(
        OmegaConf.to_container(config.training, resolve=True), allow_val_change=True
    )
    wandb.config.update(
        OmegaConf.to_container(config.fwd_model, resolve=True), allow_val_change=True
    )
    wandb.config.update(
        OmegaConf.to_container(config.vision_model, resolve=True), allow_val_change=True
    )

    img_recon_weight = config.training.img_recon
    img_predict_weight = config.training.img_predict
    z_predict_weight = config.training.z_predict
    z_reg_weight = config.training.z_reg
    repulsion_weight = config.training.repulsion

    def compute(data):
        xs = data["xs"].to(device)
        us = data["us"].to(device)
        imgs = data["imgs"].to(device)
        sequence_length = xs.shape[1]

        _imgs = imgs + config.training.noise_img * torch.randn_like(imgs)

        zs = rearrange(
            vision_model.encode(rearrange(_imgs, "b seq ... -> (b seq) ..."))[0],
            "(b seq) ... -> b seq ...",
            seq=sequence_length,
        )

        _zs = zs + config.training.noise_z * torch.randn_like(zs)

        imgs_recon = rearrange(
            vision_model.decode(rearrange(_zs, "b seq ... -> (b seq) ...")),
            "(b seq) ... -> b seq ...",
            seq=sequence_length,
        )

        current_z = zs[:, 0, :]

        # Multi-step predictions
        predicted_zs = [current_z]
        for t in range(1, sequence_length):
            control_input_t = us[:, t, :]
            predicted_z = forward_model(
                predicted_zs[-1]
                + config.training.noise_z * torch.randn_like(predicted_zs[-1]),
                control_input_t
                + config.training.noise_u * torch.randn_like(control_input_t),
            )
            predicted_zs.append(predicted_z)

        predicted_zs = torch.stack(predicted_zs, dim=1)

        # Decode predicted zs to images
        imgs_predicted = rearrange(
            vision_model.decode(rearrange(predicted_zs, "b seq ... -> (b seq) ...")),
            "(b seq) ... -> b seq ...",
            seq=sequence_length,
        )

        # Compute losses
        loss_z = criterion(predicted_zs, zs)
        loss_img = criterion(imgs_predicted, imgs)

        # Combine losses
        loss_img_recon = criterion(imgs_recon, imgs)
        total_loss = (
            z_predict_weight * loss_z
            + img_predict_weight * loss_img
            + img_recon_weight * loss_img_recon
        )

        z_first = zs[:, 0, :]
        # New loss term: Maximize distance between first states
        distances = torch.cdist(z_first, z_first, p=2)
        mask = ~torch.eye(z_first.size(0), device=device).bool()
        avg_distance = distances[mask].mean()
        max_first_dist_loss = -avg_distance  # Negative for maximization




        total_loss += z_reg_weight * torch.mean(zs**2)
        total_loss += repulsion_weight * max_first_dist_loss

        return {
            "loss": {
                "total_loss": total_loss,
                "loss_multistep_z": loss_z,
                "loss_multistep_img": loss_img,
                "loss_img_recon": loss_img_recon,
                "z_reg": torch.mean(zs**2),
                "repulsion_loss": max_first_dist_loss,
                "w_loss_multistep_z": z_predict_weight * loss_z,
                "w_loss_multistep_img": img_predict_weight * loss_img,
                "w_loss_img_recon": img_recon_weight * loss_img_recon,
                "w_z_reg": z_reg_weight * torch.mean(zs**2),
                "w_repulsion_loss": repulsion_weight * max_first_dist_loss,
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

    def save_images(out, tag, epoch):
        imgs = out["imgs"]
        imgs_recon = out["imgs_recon"]
        imgs_predicted = out["imgs_predicted"]

        sequence_length = imgs.shape[1]

        fout = results_folder / f"imgs/orig-imgs-{tag}-e{epoch:05d}.png"
        utils.save_image(
            rearrange(imgs, "b seq ... -> (b seq) ..."), fout, nrow=sequence_length
        )

        fout = results_folder / f"imgs/recon-imgs-{tag}-e{epoch:05d}.png"
        utils.save_image(
            rearrange(imgs_recon, "b seq ... -> (b seq) ..."),
            fout,
            nrow=sequence_length,
        )

        fout = results_folder / f"imgs/predicted-imgs-{tag}-e{epoch:05d}.png"
        utils.save_image(
            rearrange(imgs_predicted, "b seq ... -> (b seq) ..."),
            fout,
            nrow=sequence_length,
        )

        # Log images to TensorBoard
        # Select a subset to log (e.g., first 4 images)
        num_images_to_log = min(4, imgs.shape[0])
        grid_orig = utils.make_grid(
            rearrange(imgs[:num_images_to_log], "b seq c h w -> (b seq) c h w"),
            nrow=sequence_length,
        )
        grid_recon = utils.make_grid(
            rearrange(imgs_recon[:num_images_to_log], "b seq c h w -> (b seq) c h w"),
            nrow=sequence_length,
        )
        grid_pred = utils.make_grid(
            rearrange(
                imgs_predicted[:num_images_to_log], "b seq c h w -> (b seq) c h w"
            ),
            nrow=sequence_length,
        )

        writer.add_image(f"Original Images/{tag}", grid_orig, global_step=epoch)
        writer.add_image(f"Reconstructed Images/{tag}", grid_recon, global_step=epoch)
        writer.add_image(f"Predicted Images/{tag}", grid_pred, global_step=epoch)

        # [wandb] Log images to wandb
        wandb.log(
            {
                f"Original Images/{tag}": wandb.Image(grid_orig.cpu()),
                f"Reconstructed Images/{tag}": wandb.Image(grid_recon.cpu()),
                f"Predicted Images/{tag}": wandb.Image(grid_pred.cpu()),
            },
            step=epoch,
        )

    for epoch in range(num_epochs):

        forward_model.train()  # Ensure model is in training mode
        if config.training.fix_encoder:
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
            if batch_idx == 0:
                print(
                    f"Epoch {epoch}/{num_epochs} -- batch {batch_idx+1}/{len(dataloader_train)} Train set"
                )
                loss_info(out["loss"])

                # Log training losses to TensorBoard
                for loss_name, loss_value in out["loss"].items():
                    writer.add_scalar(
                        f"Train/{loss_name}",
                        loss_value.item(),
                        epoch * len(dataloader_train) + batch_idx,
                    )

                # [wandb] Log training losses to wandb
                wandb.log(
                    {
                        f"Train/{loss_name}": loss_value.item()
                        for loss_name, loss_value in out["loss"].items()
                    },
                    step=epoch,
                )

        # Get a batch from the test set.
        vision_model.eval()
        forward_model.eval()
        with torch.no_grad():
            infos = []
            for batch_idx, data in enumerate(dataloader_test):
                out = compute(data)
                infos.append(out["loss"])

            info = mean_of_dicts(infos)
            print(f"Epoch {epoch}/{num_epochs} -- test set")
            loss_info(info)

            # Log test losses to TensorBoard
            for loss_name, loss_value in info.items():
                writer.add_scalar(f"Test/{loss_name}", loss_value.item(), epoch)

                # [wandb] Log test losses to wandb
                wandb.log({f"Test/{loss_name}": loss_value.item()}, step=epoch)

            # Save and log images with epoch number
            out_train = compute(train_small_eval)
            save_images(out_train, f"train", epoch)

            out_test = compute(test_small_eval)
            save_images(out_test, f"test", epoch)

        # Save model checkpoint
        fout = results_folder / f"checkpoints/model-epoch-{epoch:05d}.pt"
        torch.save(
            {
                "fwd_state_dict": forward_model.state_dict(),
                "fwd_model": forward_model,
                "vision_state_dict": vision_model.state_dict(),
                "vision_model": vision_model,
            },
            str(fout),
        )

        # [wandb] Optionally, save checkpoints as wandb artifacts
        # Uncomment the following lines if you want to track model checkpoints in wandb
        """
        artifact = wandb.Artifact('model_checkpoint', type='model')
        artifact.add_file(str(fout))
        wandb.log_artifact(artifact)
        """

    print("Training completed!")

    # After training, log hyperparameters and final metrics
    metric_dict = {
        "final_total_loss": info["total_loss"].item(),
        "final_loss_multistep_z": info["loss_multistep_z"].item(),
        "final_loss_multistep_img": info["loss_multistep_img"].item(),
        "final_loss_img_recon": info["loss_img_recon"].item(),
        "final_z_reg": info["z_reg"].item(),
    }

    # [wandb] Log final metrics
    wandb.log(metric_dict)

    writer.add_hparams(
        OmegaConf.to_container(config.training, resolve=True), metric_dict
    )

    # Close the TensorBoard writer
    writer.close()

    # [wandb] Finish the wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
