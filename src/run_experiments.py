"""
Batch experiment launcher for multiple training configurations.

This script:
- Generates combinations of key model parameters
- Launches training jobs in parallel (multi-process)
- Logs success or failure for each configuration
"""



from multiprocessing import Pool
from itertools import product
import os
import yaml
import json
from train_initialiser import train_model  # Your previously defined training wrapper



def generate_configs():
    """
    Generate all valid combinations of model configurations.

    Returns:
        list: A list of configuration dictionaries.
    """
    spiking_opts = [True, False]
    use_stbp_opts = [True, False]
    mask_opts = ["soft", "hard"]
    ratios = [(1, 1), (2, 1), (3, 1)]

    all_configs = []
    for s, stbp, m, r in product(spiking_opts, use_stbp_opts, mask_opts, ratios):
        # Disallow STBP when spiking is False
        if not s and stbp:
            continue

        config = {
            "spiking": s,
            "use_STBP": stbp,
            "mask_type": m,
            "conv_pool_ratio": r,
            "dataset": "cifar10",
            "epochs": 30,
            "batch_size": 64,
            "learning_rate": 0.001
        }
        all_configs.append(config)
    return all_configs


def train_with_config(config):
    """
    Training function for a single configuration.

    Args:
        config (dict): A configuration dictionary to run training with.
    """
    try:
        result = train_model(config)
        print(f"[✓] Finished: {result['run_name']}")
    except Exception as e:
        print(f"[✗] Failed: {config} → {e}")


if __name__ == "__main__":
    all_configs = generate_configs()
    os.makedirs("checkpoints", exist_ok=True)

    # Run training jobs in parallel using 4 worker processes
    with Pool(processes=4) as pool:
        pool.map(train_with_config, all_configs)
