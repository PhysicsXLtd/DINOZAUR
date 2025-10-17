"""H5Dataset class definition."""

import logging
import os
import sys
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(stream=sys.stdout))
logger.setLevel(logging.INFO)


class H5Dataset(Dataset):
    """Dataset for .h5 files."""

    def __init__(
        self,
        dir_name: str,
        device: str,
        features: list[str],
        target: str,
        extra_inputs: dict[str, str] = {},
    ):
        """Initialize H5Dataset.

        Args:
            dir_name: Dataset directory.
            device: Device.
            features: List of dataset keys to be concatenated as model input.
            target: Target dataset key.
            extra_inputs: Dict to map dataset keys to extra model inputs. Defaults to {}.
        """
        self.features = features
        self.target = target
        self.extra_inputs = extra_inputs
        self.input_keys = list(features) + [target] + list(extra_inputs.values())
        data = []
        location = Path(dir_name)
        self.files = os.listdir(location)
        logger.info("Creating dataset.")
        for file in tqdm(self.files):
            file_path = f"{location}/{file}"
            if os.path.isfile(file_path):
                sample = {}
                with h5py.File(file_path, "r") as f:
                    for key in self.input_keys:
                        sample[key] = torch.Tensor(f[key][:])
                data.append(sample)
        self.data = data
        self.device = device

    def __len__(self):
        """Get length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Get item from the dataset."""
        sample = {}
        data_sample = self.data[idx]
        features = torch.cat([data_sample[key] for key in self.features], -1)

        sample["x"] = features
        sample["target"] = data_sample[self.target]
        for key, value in self.extra_inputs.items():
            sample[key] = data_sample[value]
        for key, value in sample.items():
            sample[key] = value.to(self.device)
        return sample
