import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def save_sparsity_plot(spike_record, save_path: Path, title="Sparsity Over Time"):
    """
    Saves a plot showing sparsity (fraction of active neurons) per timestep.
    """
    sns.set_theme(style="whitegrid")

    spike_matrix = torch.stack(spike_record).cpu().numpy()  # [T, N]
    T, N = spike_matrix.shape

    sparsity_over_time = spike_matrix.sum(axis=1) / N  # [T]

    plt.figure(figsize=(12, 5))
    plt.plot(np.arange(T), sparsity_over_time, color="crimson", linewidth=2)
    plt.xlabel("Timestep")
    plt.ylabel("Sparsity (Active Ratio)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
