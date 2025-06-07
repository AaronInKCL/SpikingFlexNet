import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.append("/home/zephyr/flexnet/Flexible-Neurons-main")

from src.analysis.run_loader import RunLoader


def collect_accuracies_from_all_runs(base_folder):
    base_path = Path(base_folder)
    run_folders = [f for f in base_path.iterdir() if f.is_dir()]
    
    training_accuracies = {}
    test_accuracies = {}

    for run in run_folders:
        try:
            run_loader = RunLoader(run, whether_load_checkpoint=False)
            df = run_loader.logger.get_dataframe()

            # Training accuracy over epochs
            if "Train Accuracy" in df.columns:
                training_accuracies[run.name] = df["Train Accuracy"].tolist()

            # Test accuracy over epochs (not just last epoch)
            if "Valid Accuracy" in df.columns:
                test_accuracies[run.name] = df["Valid Accuracy"].tolist()
        except Exception as e:
            print(f"Skipping {run.name} due to error: {e}")

    return training_accuracies, test_accuracies


def plot_accuracies(accuracy_dict, title, ylabel, save_path):
    plt.figure(figsize=(12, 6))
    for run_name, acc_list in accuracy_dict.items():
        plt.plot(range(1, len(acc_list) + 1), acc_list, label=run_name)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    base_folder = "checkpoints"  
    training_accs, test_accs = collect_accuracies_from_all_runs(base_folder)

    plot_accuracies(training_accs, "Training Accuracy Over Epochs", "Train Accuracy", "all_training_accuracy.png")
    plot_accuracies(test_accs, "Test Accuracy Over Epochs", "Test Accuracy", "all_test_accuracy.png")
