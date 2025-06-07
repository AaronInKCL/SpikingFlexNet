from src.training.dataset_cifar10 import Cifar10Dataset
from src.training.dataset_imagenet import ImageNet100Dataset
from src.training.dataset_subset import create_balanced_subset, create_random_subset
from src.utils.server import is_on_server
from pathlib import Path
from src.training.dataset_mnist import MNISTDataset

def get_dataset_obj(dataset_name: str, mode: str):
    """
    example:
        dataset = get_dataset_obj("cifar10", "TRAIN")
        dataset = get_dataset_obj("cifar10", "TEST")
        dataset = get_dataset_obj("imagenet100", "TRAIN")
        dataset = get_dataset_obj("imagenet100", "TEST")

    cifar10 / imagenet100 / cifar10-down-50 / cifar10-random-small-100 / cifar10-random-small-500
    """
    match dataset_name:
        case "mnist":
            return MNISTDataset(mode)
        case "cifar10":
            return Cifar10Dataset(mode)
        case "cifar10-down-50":  # [NOTE] 10 classes, 100 images per class / totalling 1000 images / Linnea used
            cifar10_train = Cifar10Dataset(mode)
            return create_balanced_subset(cifar10_train, num_classes=10, num_samples_per_class=100)
        case "cifar10-random-small-100":
            cifar10_train = Cifar10Dataset(mode)
            return create_random_subset(cifar10_train, num_samples=100, seed=42)
        case "cifar10-random-small-500":
            cifar10_train = Cifar10Dataset(mode)
            return create_random_subset(cifar10_train, num_samples=500, seed=42)
        case "cifar10-random-small-gaussian-noise-0.5":
            cifar10_train = Cifar10Dataset(mode, gaussian_noise_std=0.5)
            return create_random_subset(cifar10_train, num_samples=100, seed=42)
        case "imagenet100":
            if is_on_server():
                return ImageNet100Dataset(folder=Path("/home/zephyr/flexnet/Flexible-Neurons-main/data/imagenet100"), mode=mode)
            else:
                return ImageNet100Dataset(folder=Path("data/imagenet100"), mode=mode)
        case _:
            raise ValueError(f"Invalid dataset name: {dataset_name}")
