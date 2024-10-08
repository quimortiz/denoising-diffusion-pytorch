import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
    def __init__(self, trajs_latent, us, sequence_length=5):
        """
        trajs_latent: Tensor of shape (batch_size, channels, total_sequence_length)
        us: Tensor of shape (batch_size, control_channels, total_sequence_length)
        sequence_length: Number of steps in each sample (e.g., 5 steps)
        """
        self.trajs = trajs_latent
        self.us = us
        self.total_sequence_length = trajs_latent.shape[2]
        self.sequence_length = sequence_length

    def __len__(self):
        return self.trajs.shape[0] * (self.total_sequence_length - self.sequence_length)

    def __getitem__(self, idx):
        batch_idx = idx // (self.total_sequence_length - self.sequence_length)
        step_idx = idx % (self.total_sequence_length - self.sequence_length)

        states = self.trajs[
            batch_idx, :, step_idx : step_idx + self.sequence_length
        ]  # Shape: (channels, sequence_length)
        controls = self.us[
            batch_idx, :, step_idx : step_idx + self.sequence_length - 1
        ]  # Shape: (control_channels, sequence_length -1)

        return states, controls


def generate_exp_id():
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=6))


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
        "--recon_weight", type=float, default=0.0, help="Reconstruction loss weight"
    )
    parser.add_argument(
        "--z_weight",
        type=float,
        default=1e-5,
        help="Latent space regularization weight",
    )
    parser.add_argument(
        "--z_diff",
        type=float,
        default=1e-5,
        help="Latent difference regularization weight",
    )
    parser.add_argument("--cond", action="store_true", help="Use conditioning")
    parser.add_argument("--size", type=int, default=32, help="Image size")
    parser.add_argument(
        "--mod_lr", action="store_true", help="Modify learning rate schedule"
    )
    parser.add_argument(
        "--cond_combined", action="store_true", help="Use combined conditioning"
    )
    parser.add_argument(
        "--y_cond_as_x", action="store_true", help="Use y conditioning as x"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--train_num_steps", type=int, default=100000, help="Number of training steps"
    )
    parser.add_argument(
        "--train_u", action="store_true", help="Train with control inputs (u)"
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

    args = parser.parse_args()
    print(args)

    nu = 2
    nz = 8  # Latent dimension
    n_elements = 1

    # Initialize Vision Model
    if args.resnet:
        vision_model = AE("light")  # Ensure this is correctly implemented
    else:
        vision_model = VanillaVAE(in_channels=3, latent_dims=nz, size=args.size)

    # Load Pretrained Weights if specified
    if args.size == 32:
        path = "results/i2n6ce/model-95000.pt"
        vision_model.load_state_dict(torch.load(path)["model"])
    elif args.size == 64:
        # path = "results/la90ra/model-995000.pt"  # Adjust the path as needed
        path = "results/la90ra/model-490000.pt"  # the original i was using
        print("Loading model from ", path, "...")
        vision_model.load_state_dict(torch.load(path)["model"])
    elif args.size == 224:
        path = "path_to_pretrained_model_for_224.pt"  # Provide the correct path
        print("Loading model from ", path, "...")
        vision_model.load_state_dict(torch.load(path)["model"])
    else:
        raise ValueError(f"Unsupported size {args.size} for pretrained model.")

    vision_model.eval()  # Set to evaluation mode

    # Encode data using the vision model
    trajs_latent = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    vision_model = vision_model.to(device)

    data = torch.load("trajs_latent_all_v1.pt")

    trajs_latent = data["zs"]
    data_us = data["us"]
    data_us = einops.rearrange(data_us, "b seq c -> b c seq")

    # Rearrange to (B, channels, seq)
    trajs_latent = rearrange(trajs_latent, "b seq c -> b c seq")

    # Create Dataset and DataLoader

    sequence_length = 15
    dataset = TrajectoryDataset(trajs_latent, data_us, sequence_length=sequence_length)

    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )

    # Initialize Forward Model
    input_dim = nz + nu  # Latent dimension + control input dimension
    hidden_dim = 256
    output_dim = nz  # Predicting next latent state

    forward_model = MLPForwardModel(input_dim, hidden_dim, output_dim, num_layers=4).to(
        device
    )

    # Define Optimizer
    optimizer = optim.Adam(
        forward_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Define Loss Function
    criterion = nn.MSELoss()

    # Optionally, fix the encoder if specified
    if args.fix_encoder:
        for param in vision_model.parameters():
            param.requires_grad = False

    # Training Loop
    num_epochs = args.train_num_steps // len(dataloader) + 1
    forward_model.train()

    results_folder = pathlib.Path(f"results/{args.exp_id}")
    results_folder.mkdir(parents=True, exist_ok=True)

    one_step_loss_weight = 0.1
    multi_step_loss_weight = 1.0

    for epoch in range(num_epochs):
        for batch_idx, (states, controls) in enumerate(dataloader):
            states = states.to(device)  # Shape: (batch_size, channels, sequence_length)
            controls = controls.to(
                device
            )  # Shape: (batch_size, control_channels, sequence_length -1)

            batch_size, channels, sequence_length = states.shape
            # print(f"Batch size: {batch_size}, Channels: {channels}, Sequence length: {sequence_length}")
            control_channels = controls.shape[1]

            # Initialize loss
            total_loss = 0.0

            # One-step prediction
            current_state = states[:, :, 0]  # Shape: (batch_size, channels)
            control_input = controls[:, :, 0]  # Shape: (batch_size, control_channels)
            next_state = states[:, :, 1]  # Shape: (batch_size, channels)

            predicted_next_state = forward_model(
                current_state + args.noise_z * torch.randn_like(current_state),
                control_input + args.noise_u * torch.randn_like(control_input),
            )
            loss_one_step = criterion(predicted_next_state, next_state)

            # Multi-step predictions (up to 4 steps ahead)
            predicted_states = [current_state, predicted_next_state]
            for t in range(1, sequence_length - 1):
                control_input_t = controls[:, :, t]
                predicted_next_state = forward_model(
                    predicted_states[-1]
                    + args.noise_z * torch.randn_like(predicted_states[-1]),
                    control_input_t + args.noise_u * torch.randn_like(control_input_t),
                )
                predicted_states.append(predicted_next_state)

            # Compute multi-step loss
            loss_multi_step = 0.0
            for t in range(1, sequence_length):
                loss_t = criterion(predicted_states[t], states[:, :, t])
                loss_multi_step += loss_t

            # Combine losses
            total_loss = (
                one_step_loss_weight * loss_one_step
                + multi_step_loss_weight * loss_multi_step
            )

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Logging
            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(dataloader)}] "
                    f"Loss: {total_loss.item():.4f} (One-step: {loss_one_step.item():.4f}, "
                    f"Multi-step: {loss_multi_step.item():.4f})"
                )

        # visualize the trajectories
        predicted_states = torch.stack(predicted_states, dim=2)

        # display predicted next states
        imgs = vision_model.decode(
            einops.rearrange(predicted_states, "b c seq -> (b seq) c").to(device)
        )

        fout = str(results_folder / f"sample-imgs-{epoch:05d}.png")
        print(f"saving to {fout}")
        utils.save_image(imgs, fout, nrow=sequence_length)

        # save the model
        fout = results_folder / f"checkpoints/model-{epoch:05d}.pt"
        fout.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": forward_model.state_dict(),
                "model": forward_model,
            },
            str(fout),
        )

    print("Training completed!")


if __name__ == "__main__":
    main()
