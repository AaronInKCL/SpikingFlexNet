# src/analysis/spike_raster_plot.py

import matplotlib.pyplot as plt
import torch
from pathlib import Path
import seaborn as sns
import numpy as np
import torch.nn as nn

def plot_spike_raster(spike_tensor, title="Spike Raster", save_path=None):
    T, C, H, W = spike_tensor.shape
    spike_tensor_flat = spike_tensor.view(T, -1)  # [T, N]

    plt.figure(figsize=(10, 6))
    for neuron_idx in range(spike_tensor_flat.shape[1]):
        spike_times = torch.nonzero(spike_tensor_flat[:, neuron_idx], as_tuple=False).squeeze(1)
        plt.scatter(spike_times.numpy(), [neuron_idx] * len(spike_times), s=2, color='black')

    plt.xlabel("Timestep")
    plt.ylabel("Neuron Index")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    plt.close()



def save_spike_raster_plots(model, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    layer_id = 0

    for name, module in model.named_modules():
        if hasattr(module, "last_spike") and module.last_spike is not None:
            spikes = module.last_spike.view(-1)  # flatten
            plt.figure(figsize=(12, 2))
            plt.plot(spikes.numpy(), "|", markersize=2)
            plt.title(f"Spike Raster - {name}")
            plt.tight_layout()
            plt.savefig(save_dir / f"raster_{layer_id:02d}_{name.replace('.', '_')}.png")
            plt.close()
            layer_id += 1


def save_spike_raster_plot(spike_record, save_path: Path, title="Spike Raster"):
    """
    spike_record: list of 1D spike vectors (each is shape [N]) → total length T
    """
    sns.set_theme(style="whitegrid")

    # --- Handle inconsistent spike vector lengths ---
    min_len = min(s.numel() for s in spike_record)
    trimmed_record = [s[:min_len] for s in spike_record]

    # --- Construct spike matrix [T, N] ---
    spike_matrix = torch.stack(trimmed_record).cpu().numpy()
    times, neuron_ids = np.where(spike_matrix > 0.5)

    # --- Plot ---
    plt.figure(figsize=(12, 6))
    plt.scatter(neuron_ids, times, s=4, c="royalblue", alpha=0.6, edgecolors="none")
    plt.xlabel("Neuron Index")
    plt.ylabel("Time Step")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_all_layer_spike_rasters(model: nn.Module, save_dir: Path):
    """
    For every layer that has `spike_history`, generate a spike raster plot,
    saving each with a filename based on its index in the loop.
    """
    from .spike_raster_plot import save_spike_raster_plot 
    save_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    for name, module in model.named_modules():
        if hasattr(module, "spike_history") and isinstance(module.spike_history, list) and len(module.spike_history) > 0:
            plot_path = save_dir / f"layer_{idx}_raster.png"
            save_spike_raster_plot(module.spike_history, plot_path, title=f"Spike Raster - {name}")
            idx += 1


