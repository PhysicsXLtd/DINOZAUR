"""Data scaling classes definitions."""

import logging
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(stream=sys.stdout))
logger.setLevel(logging.INFO)


class IdentityScaler:
    """A class which applies no normalization to the data."""

    def __init__(self, dataloader: DataLoader | None = None, key: str | None = None):
        """Initialize IdentityScaler.

        Args:
            dataloader: Placeholder for dataloader. Defaults to None.
            key: Key in data sample to apply normalization to. Defaults to None.
        """
        pass

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transform."""
        return x

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply reverse transform."""
        return x


class GaussianScaler:
    """GaussianScaler."""

    def __init__(self, dataloader: DataLoader, key: str):
        """Initialize GaussianScaler.

        Args:
            dataloader: Placeholder for dataloader.
            key: Key in data sample to apply normalization to.
        """
        # Initialize accumulators
        channel_sum = None
        channel_sum_sq = None
        total_points = 0

        logger.info("Fitting data scaler.")

        for batch in tqdm(dataloader):
            data = batch[key]

            # Get batch shape info
            channels = data.shape[-1]

            # Reshape to (batch * all_spatial_dims, channels)
            # This flattens all dimensions except the last (channels)
            data_flat = data.reshape(-1, channels)

            # Initialize accumulators on first batch
            if channel_sum is None:
                channel_sum = torch.zeros(channels, dtype=torch.float64, device=data.device)
                channel_sum_sq = torch.zeros(channels, dtype=torch.float64, device=data.device)

            # Accumulate statistics
            channel_sum += data_flat.sum(dim=0, dtype=torch.float64)
            channel_sum_sq += (data_flat**2).sum(dim=0, dtype=torch.float64)
            total_points += data_flat.shape[0]

        # Compute mean and std
        self.mean = (channel_sum / total_points).float()
        self.std = torch.sqrt(channel_sum_sq / total_points - self.mean**2).float()

        # Prevent division by zero in normalization
        self.std = torch.clamp(self.std, min=1e-8)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transform."""
        return (x - self.mean) / (self.std)

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply reverse transform."""
        return x * (self.std) + self.mean
