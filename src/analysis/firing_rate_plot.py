import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def save_firing_rate_plot(spike_record, save_path: Path, title="Firing Rate Map"):
    """
    spike_record: list of [N] tensors (T steps)
    save_path: where to save the plot
    """
    sns.set_theme(style="whitegrid")

    spike_matrix = torch.stack(spike_record).cpu().numpy()  # [T, N]
    T, N = spike_matrix.shape

    firing_rates = spike_matrix.sum(axis=0) / T  # [N]

    plt.figure(figsize=(14, 5))
    plt.bar(np.arange(N), firing_rates, color="skyblue", edgecolor="gray", alpha=0.8)
    plt.xlabel("Neuron Index")
    plt.ylabel("Firing Rate (Spikes per Timestep)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
