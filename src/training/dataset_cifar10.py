"""
CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
"""

from torch.utils.data import Dataset
from torchvision import transforms
from natsort import natsorted
from pathlib import Path
import numpy as np
import torch
import pickle
from src.training.augmentations import AddGaussianNoise


def unpickle(file):
    """
    specifically for cifar-10
    """
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


class Cifar10Dataset(Dataset):
    """
    ------------------------------------------------------------------------------------------------
    Download the dataset from https://www.cs.toronto.edu/~kriz/cifar.html
    This class assumes that the cifar-10 dataset is stored in the following directory structure:
    ------------------------------------------------------------------------------------------------
    data / cifar-10
        test_batch
        batches.meta
        data_batch_1..., data_batch_5

    ------------------------------------------------------------------------------------------------
    Downsampling
    ------------------------------------------------------------------------------------------------
    Linnea's Paper: The CIFAR-10 dataset is downsampled 50 times to form a set of only 1000 training images.
    - downsample can only be applied when the mode is TRAIN
    - set a random seed (42) and take 1000 training samples
    - since there are only 10 classes, each class will have 100 images only
    - maybe torch subset function can do this?

    """

    def __init__(self, mode="TRAIN", gaussian_noise_std=0.1):
        self.mode = mode
        self.PATH_DATASETS = Path("data", "cifar-10")
        self.file_test = self.PATH_DATASETS / "test_batch"
        self.files_train = list(self.PATH_DATASETS.glob("data_batch_*"))
        self.files_train = natsorted(self.files_train)
        self.train_file_lengths = [self._get_num_samples_in_file(file) for file in self.files_train]
        self.norm_means, self.norm_stds = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.norm_means, std=self.norm_stds),
                AddGaussianNoise(mean=0.0, std=gaussian_noise_std),
            ]
        )

    def _get_num_samples_in_file(self, file):
        data_dict = unpickle(file)
        return len(data_dict[b"labels"])

    def _get_local_index_and_file(self, idx):
        """
        Since the cifar-10 dataset is stored in many small files, this function returns the local index of the sample in the file, as well as the file itself.
        """
        cumulative_samples = 0
        for file, file_len in zip(self.files_train, self.train_file_lengths):
            if idx < cumulative_samples + file_len:
                return idx - cumulative_samples, file
            cumulative_samples += file_len
        raise ValueError("Index out of range!")

    def _get_filename(self, idx):
        """Returns the filename of the sample at the given index."""
        local_idx, file = self._get_local_index_and_file(idx)
        data_dict = unpickle(file)
        return data_dict[b"filenames"][local_idx]

    # -------- dunders --------
    def __len__(self):
        if self.mode == "TRAIN":
            return sum(self.train_file_lengths)
        if self.mode == "TEST":
            return self._get_num_samples_in_file(self.file_test)

    def __getitem__(self, idx):
        assert self.mode in ["TRAIN", "TEST"], f"Invalid mode! Got {self.mode}"

        if self.mode == "TRAIN":
            local_idx, file = self._get_local_index_and_file(idx)
            data_dict = unpickle(file)
            images = np.array(data_dict[b"data"])
            images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            labels = np.array(data_dict[b"labels"])
            image_tensor, label_tensor = self.transforms(images[local_idx]), torch.tensor(labels[local_idx])

        elif self.mode == "TEST":
            if idx >= self._get_num_samples_in_file(self.file_test):
                raise IndexError("Index out of range for TEST mode!")
            data_dict = unpickle(self.file_test)
            images = np.array(data_dict[b"data"])
            images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            labels = np.array(data_dict[b"labels"])
            image_tensor, label_tensor = self.transforms(images[idx]), torch.tensor(labels[idx])

        return image_tensor, label_tensor


if __name__ == "__main__":
    pass
