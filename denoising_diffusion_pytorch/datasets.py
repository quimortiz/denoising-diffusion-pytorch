from torch.utils.data import DataLoader, Dataset
import torch

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


class TrajectoryDatasetLatent(Dataset):
    def __init__(self, zs, us, xs = None):
        """
        trajs_latent: Tensor of shape (batch_size, channels, total_sequence_length)
        us: Tensor of shape (batch_size, control_channels, total_sequence_length)
        sequence_length: Number of steps in each sample (e.g., 5 steps)
        """
        self.zs = zs
        self.us = us
        self.xs = xs
        assert zs.shape[0] == us.shape[0]
        if xs is not None:
            assert zs.shape[0] == xs.shape[0]

    def __len__(self):
        return self.zs.shape[0]

    def __getitem__(self, idx):
        return {"zs": self.zs[idx], "us": self.us[idx], "xs": self.xs[idx]}



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

