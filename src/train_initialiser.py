"""
Main training script for image classification models.

This script supports:
- Command-line usage with a JSON config
- External module invocation via `train_model()`

Features:
- Dataset loading
- Training loop control
- Run folder creation and config saving
"""

import sys
sys.path.append("/home/zephyr/flexnet/Flexible-Neurons-main")

import os
import json
from rich import print

from src.utils.general import banner
from src.training.train import main_training_loop
from src.training.dataset_select import get_dataset_obj
from src.training.dataset_subset import create_random_subset
from src.analysis.run_loader import RunLoader


def get_num_batches_per_log(dataset, batch_size, logs_per_epoch):
    """
    Compute how many batches should pass between each log output.

    Args:
        dataset (Dataset): The training dataset.
        batch_size (int): Batch size used during training.
        logs_per_epoch (int): Desired number of log entries per epoch.

    Returns:
        int: Number of batches per logging event.
    """
    num_batches = len(dataset) // batch_size
    num_batches_per_log = num_batches // logs_per_epoch
    return max(1, num_batches_per_log)


def continue_training(run_loader: RunLoader, epochs: int = 50, logs_per_epoch: int = 36):
    """
    Continue or start training using a given RunLoader.

    Args:
        run_loader (RunLoader): Object containing model and config.
        epochs (int): Number of training epochs.
        logs_per_epoch (int): Number of logs per epoch.
    """
    banner("Configurations")
    print(run_loader.config)

    dataset_train = get_dataset_obj(run_loader.config["dataset"], "TRAIN")
    dataset_valid = get_dataset_obj(run_loader.config["dataset"], "TEST")

    log_every_n_batch = get_num_batches_per_log(dataset_train, run_loader.config["batch_size"], logs_per_epoch)

    main_training_loop(
        epochs=epochs,
        run_loader=run_loader,
        dataset_train=dataset_train,
        dataset_valid=dataset_valid,
        log_every_n_batch=log_every_n_batch,
    )


def make_run_folder_name(config: dict):
    """
    Generate a unique folder name based on model configuration.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        str: Name of the run folder.
    """
    model = config.get("network", "Model")
    spk_flag = "SNN" if config.get("spiking", False) else "ANN"
    stbp_flag = "STBP" if config.get("use_stbp", False) else "NoSTBP"
    flex_flag = "FLEX" if config.get("use_flex", False) else "NoFLEX"
    dataset = config.get("dataset", "unknown")
    return f"{dataset}_{model}_{spk_flag}_{stbp_flag}_{flex_flag}"


def train_model(config: dict):
    """
    Main training entry point when called as a module.

    Args:
        config (dict): Model training configuration.

    Returns:
        dict: Contains the name of the created run.
    """
    run_name = make_run_folder_name(config)
    run_folder = os.path.join("checkpoints", run_name)
    os.makedirs(run_folder, exist_ok=True)

    # Save the current config to the run folder
    with open(os.path.join(run_folder, "configurations.json"), "w") as f:
        json.dump(config, f, indent=4)

    # Load model and start training
    run_loader = RunLoader(run_folder)
    continue_training(run_loader=run_loader, epochs=config.get("epochs", 30))
    return {"run_name": run_name}


if __name__ == "__main__":
    # Command-line interface for standalone usage
    import argparse
    parser = argparse.ArgumentParser(description="Train with injected configuration")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    train_model(config)
