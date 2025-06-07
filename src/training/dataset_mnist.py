from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torch
from pathlib import Path
from src.training.augmentations import AddGaussianNoise


class MNISTDataset(Dataset):
    def __init__(self, mode="TRAIN"):
        assert mode in ["TRAIN", "TEST"], f"Invalid mode: {mode}"
        self.mode = mode
        self.PATH_DATASETS = Path("data", "mnist")

        self.norm_means = [0.1307]
        self.norm_stds = [0.3081]

        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.norm_means, std=self.norm_stds),
            ]
        )

        self.dataset = datasets.MNIST(
            root=self.PATH_DATASETS,
            train=True if mode == "TRAIN" else False,
            download=True,
            transform=self.transforms,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image_tensor = image.repeat(3, 1, 1)  # [1, 28, 28] → [3, 28, 28]
        label_tensor = torch.tensor(label)
        return image_tensor, label_tensor


if __name__ == "__main__":
    dataset = MNISTDataset("TRAIN")
    print(len(dataset))
    img, label = dataset[0]
    print(img.shape, label)
