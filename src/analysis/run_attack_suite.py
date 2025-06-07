import sys
sys.path.append("/home/zephyr/flexnet/Flexible-Neurons-main")

import argparse
from pathlib import Path
from src.analysis.run_loader_image.attacks import Attack
from src.analysis.run_loader import RunLoader
from src.training.dataset_select import get_dataset_obj
from src.training.dataset_subset import create_random_subset
from torch.utils.data import DataLoader
import torch
import numpy as np

"""
Adversarial attack evaluation script.

This script loads a trained model from a checkpoint folder and applies a specified 
adversarial attack (e.g., PGD, FGSM). It evaluates top-n accuracy and saves results.
"""


def run_attack(checkpoint_dir: str, attack_type: str = "PGD", top_n: int = 1, sample_size: int = 128):
    """
    Runs an adversarial attack on a model checkpoint.

    Args:
        checkpoint_dir (str): Path to the model checkpoint folder.
        attack_type (str): Type of adversarial attack (e.g., PGD, FGSM).
        top_n (int): Compute top-n accuracy.
        sample_size (int): Number of test samples to use.
    """
    print(f"[✓] Loading model from: {checkpoint_dir}")
    run_loader = RunLoader(Path(checkpoint_dir), whether_load_checkpoint=False)
    model = run_loader.model

    model.eval()
    dummy_input = torch.randn(1, *run_loader.config.in_dimensions).to(run_loader.device)
    model(dummy_input)

    # Load and subsample the test dataset
    dataset = get_dataset_obj(run_loader.config.dataset, "TEST")
    dataset = create_random_subset(dataset, sample_size)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Create output directory for attack results
    attack_results_dir = Path(checkpoint_dir) / "results" / attack_type
    attack_results_dir.mkdir(parents=True, exist_ok=True)

    # Initialize and run the attack
    attack_runner = Attack(
        run_loader=run_loader,
        attack=attack_type,
        data_loader=dataloader,
        accuracy_top_n=top_n,
        save_to=attack_results_dir,
    )
    attack_runner.set_model(model)
    attack_runner.test_save_data()

    print(f"[✓] Attack results saved to: {attack_results_dir}")


if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Run adversarial attack evaluation")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--attack", type=str, default="PGD", help="Attack type: PGD, FGSM, Jitter, SPSA, etc.")
    parser.add_argument("--top_n", type=int, default=1, help="Top-n accuracy to compute")
    parser.add_argument("--sample_size", type=int, default=128, help="Number of samples to test")
    args = parser.parse_args()

    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run attack with provided configuration
    run_attack(
        checkpoint_dir=args.checkpoint_dir,
        attack_type=args.attack,
        top_n=args.top_n,
        sample_size=args.sample_size,
    )
