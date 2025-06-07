#!/Users/donyin/miniconda3/envs/imperial/bin/python

"""
ImageNet 100: kaggle datasets download -d ambityga/imagenet100
"""

from PIL import Image
import torch, os, json
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.datasets import ImageFolder

"""
the folder structure
5 folders: train.X1 - X4 + val.X
each folder contains x subfolders, where the subfolder names are the class keys that can be used on the self.labels
each subfolder contains the images
"""


class ImageNet100Dataset(Dataset):
    def __init__(self, folder: Path, mode: str = "TRAIN"):
        self.folder = folder
        self.mode = mode.upper()
        self.labels = json.load(open(folder / "Labels.json"))

        transform_list = [transforms.Resize((224, 224), antialias=True)]
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.transforms = transforms.Compose(transform_list)
        self.image_paths, self.image_labels_str = [], []
        self._load_image_paths_labels()

    def _load_image_paths_labels(self):
        folders = [f"train.X{i}" for i in range(1, 5)] if self.mode == "TRAIN" else ["val.X"]
        for folder_name in folders:
            path = self.folder / folder_name
            for class_key in os.listdir(path):
                class_folder = path / class_key
                if os.path.isdir(class_folder):
                    for image_name in os.listdir(class_folder):
                        self.image_paths.append(class_folder / image_name)
                        self.image_labels_str.append(self.labels[class_key])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = read_image(str(image_path)).float()
        image = image / 255.0

        # -------- [deal with irregular channel] --------
        if image.shape[0] == 1:  # if 1 channel, repeat it 3 times
            image = image.repeat(3, 1, 1)
        if image.shape[0] == 4:  # if 4 channels, remove the alpha channel
            image = image[:3, :, :]

        image = self.transforms(image)
        return image, list(self.labels.values()).index(self.image_labels_str[idx])


if __name__ == "__main__":
    dataset = ImageNet100Dataset(folder=Path("/Users/donyin/Desktop/imagenet100"), mode="TRAIN")
