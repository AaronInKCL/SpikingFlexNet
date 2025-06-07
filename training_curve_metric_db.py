#!/Users/donyin/miniconda3/envs/imperial/bin/python

# import donware
# from donware import inspect_package
# inspect_package(donware)

import numpy as np
import seaborn as sns
import os, pandas, json
from pathlib import Path
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/zephyr/flexnet/Flexible-Neurons-main")


from src.plot.params import FIGURE_SIZE, FONTSIZE_TITLE, FONTSIZE_LABEL, FONTSIZE_TICK, PAD_TITLE, DPI_FIGURE

FIGURE_SIZE = (FIGURE_SIZE[0] * 0.6, FIGURE_SIZE[1] * 0.8)

# the data files are located at: data/training_curve/metric-db-imagenet-100-epoch-121-batch-16
# this dir contains some csv files, eac his named such as 000011.csv
# these are named as the following:
# e.g.,
# {
#     "000011": "VGG6",
#     "000001": "Flex: Threshold + Scaled Sigmoid",
#     "000002": "Flex: Threshold + Hard Sigmoid",
#     "000003": "Flex: Threshold + STE",
#     "000004": "Flex: Threshold + SR",
#     "000006": "Flex: SAB + Scaled Sigmoid",
#     "000007": "Flex: SAB + Hard Sigmoid",
#     "000008": "Flex: SAB + STE",
#     "000009": "Flex: SAB + SR",
#     "000010": "Flex: CMP",
# }

# each file contains the keys of Train Accuracy Balanced
# Train Loss
# Valid Accuracy Balanced
# Valid Loss
# Epoch (each row has a epoch, it can repeat many rows for the same epoch)
# id (step, 1, 2, 3...)


def smooth_curve(data, weight=0.9, start_from=3, baseline=0.1):
    """
    Smoothing function for data. The chosen smoothing function is an exponential moving average (EMA), which is generally preferred over a simple moving average (SMA) in certain contexts due to its ability to weight recent data points more heavily. This can make the EMA more responsive to recent changes and trends in the data, which is often beneficial when plotting training curves to observe recent performance trends. The EMA can also handle larger datasets more efficiently as it doesn’t require recalculating the average of a moving window.
    """
    smoothed = []
    for i, point in enumerate(data):
        if i < start_from:
            smoothed.append(baseline * weight + point * (1 - weight))
        else:
            if i == start_from:
                last = point
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
    return smoothed


def plot_training_curve(
    file_paths,
    smooth=True,
    rolling_std_window=10,
    save_as=None,
    label_every_n_epochs=20,
    linewidth=4,
    title="Training and Validation Curve",
    total_epochs=120,
    keep_first_n_epoch=120,
    exclude_the_first_n_epoch=0,
    plot_loss=False,  # New parameter to control plotting loss or accuracy
    plot_which=["train", "valid"],
):
    """
    40 steps per epoch, 180 epochs
    """

    legend_size_adjustment_factor = 0.75

    # compute the data to keep
    train_data = pandas.read_csv(next(iter(file_paths.values())))
    # num_steps_per_epoch = len(train_data) // total_epochs
    num_steps_per_epoch = 39
    keep_first_n_row = num_steps_per_epoch * keep_first_n_epoch

    plt.figure(figsize=FIGURE_SIZE)
    color_map = {}

    for model, path in file_paths.items():
        data = pandas.read_csv(path)

        # Exclude the first n epochs
        data = data.iloc[exclude_the_first_n_epoch * num_steps_per_epoch :]
        data.reset_index(drop=True, inplace=True)

        # Function to handle outliers
        data.fillna(0, inplace=True)

        def handle_outliers(data, lower_percentile=3, upper_percentile=97):
            float_columns = data.select_dtypes(include=[np.float64, np.float32]).columns
            for column in float_columns:
                lower_bound = np.percentile(data[column], lower_percentile)
                upper_bound = np.percentile(data[column], upper_percentile)
                data.loc[data[column] < lower_bound, column] = lower_bound
                data.loc[data[column] > upper_bound, column] = upper_bound
            return data

        # Handle outliers for data
        data = handle_outliers(data)

        # keep the first keep_first_n rows
        data = data.iloc[:keep_first_n_row]

        if plot_loss:
            train_values_raw = data["Train Loss"]
            valid_values_raw = data["Valid Loss"]
            train_values_raw.fillna(0, inplace=True)
            valid_values_raw.fillna(0, inplace=True)

        else:
            train_values_raw = data["Train Accuracy Balanced"]
            valid_values_raw = data["Valid Accuracy Balanced"]
            train_values_raw.fillna(0, inplace=True)
            valid_values_raw.fillna(0, inplace=True)

        train_values = smooth_curve(train_values_raw, weight=0.9) if smooth else train_values_raw
        valid_values = smooth_curve(valid_values_raw, weight=0.998) if smooth else valid_values_raw

        train_std = train_values_raw.rolling(window=rolling_std_window).std().fillna(0)
        valid_std = valid_values_raw.rolling(window=rolling_std_window).std().fillna(0)

        color = sns.color_palette("tab10")[list(file_paths.keys()).index(model)]
        color_map[model] = color

        if "train" in plot_which:
            sns.lineplot(x=data["Epoch"], y=train_values, linestyle="-", linewidth=1.5, color=color)
            plt.fill_between(data["Epoch"], train_values - train_std, train_values + train_std, color=color, alpha=0.2)

        if "valid" in plot_which:
            plt.plot(data["Epoch"], valid_values, linestyle="--", linewidth=1.5, color=color)
            plt.fill_between(data["Epoch"], valid_values - valid_std, valid_values + valid_std, color=color, alpha=0.2)

    # Custom legend for models
    model_lines = [plt.Line2D([0], [0], color=color_map[model], linestyle="-", linewidth=linewidth) for model in file_paths.keys()]
    model_legend = plt.legend(
        model_lines,
        file_paths.keys(),
        loc="lower right",
        bbox_to_anchor=(2, 0),
        fontsize=FONTSIZE_LABEL * legend_size_adjustment_factor,
    )

    _h_ = model_legend.get_window_extent().height / 6000

    # Custom legend for train vs validation
    legend_lines = []
    legend_labels = []
    if "train" in plot_which:
        legend_lines.append(plt.Line2D([0], [0], color="black", linestyle="-", linewidth=linewidth))
        legend_labels.append("Train")
    if "valid" in plot_which:
        legend_lines.append(plt.Line2D([0], [0], color="black", linestyle="--", linewidth=linewidth))
        legend_labels.append("Validation")

    train_valid_legend = plt.legend(
        legend_lines,
        legend_labels,
        loc="lower right",
        bbox_to_anchor=(2, _h_),
        fontsize=FONTSIZE_LABEL * legend_size_adjustment_factor,
    )

    # Add both legends to the plot
    plt.gca().add_artist(train_valid_legend)
    plt.gca().add_artist(model_legend)

    # Manually amend x-axis labels to show epochs
    max_epoch = data["Epoch"].max()
    epochs = range(0, max_epoch + 1)
    epoch_steps = [e for e in epochs if e % label_every_n_epochs == 0 or e == epochs[-1]]
    plt.xticks(ticks=epoch_steps, labels=epoch_steps, fontsize=FONTSIZE_TICK)

    plt.title(title, fontsize=FONTSIZE_TITLE, pad=PAD_TITLE)
    plt.xlabel("Epochs", fontsize=FONTSIZE_LABEL, labelpad=PAD_TITLE)
    plt.ylabel("Loss" if plot_loss else "Accuracy", fontsize=FONTSIZE_LABEL, labelpad=PAD_TITLE)

    plt.tick_params(axis="y", labelsize=FONTSIZE_TICK)
    plt.tight_layout()
    plt.savefig(save_as, dpi=DPI_FIGURE, bbox_extra_artists=(model_legend, train_valid_legend), bbox_inches="tight")


if __name__ == "__main__":
    # --------------------------------
    save_dir = Path("figures") / "training_curve"
    save_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path("data_csv/training_curve")

    for file in os.listdir(data_dir):
        print(file)

    # Load the registry
    with open(data_dir / "registry.json", "r") as f:
        registry = json.load(f)

    # -------------------------------- accuracy of imagenet-100 --------------------------------

    file_paths_imagenet_accuracy = {registry[key]: str(data_dir / f"{key}.csv") for key in registry}

    plot_training_curve(
        file_paths=file_paths_imagenet_accuracy,
        smooth=True,
        rolling_std_window=12,
        save_as=save_dir / "imagenet-100-accuracy.png",
        title="Accuracy Curves\n VGG16 vs Flex\n(Different Mechanisms)",
        plot_loss=False,
    )

    plot_training_curve(
        file_paths=file_paths_imagenet_accuracy,
        smooth=True,
        rolling_std_window=12,
        save_as=save_dir / "imagenet-100-accuracy-train.png",
        title="Training Accuracy Curves\n VGG16 vs Flex\n(Different Mechanisms)",
        plot_loss=False,
        plot_which=["train"],
    )

    plot_training_curve(
        file_paths=file_paths_imagenet_accuracy,
        smooth=True,
        rolling_std_window=12,
        save_as=save_dir / "imagenet-100-accuracy-valid.png",
        title="Validation Accuracy Curves\n VGG16 vs Flex\n(Different Mechanisms)",
        plot_loss=False,
        plot_which=["valid"],
    )

    # -------------------------------- loss of imagenet-100 --------------------------------

    plot_training_curve(
        file_paths=file_paths_imagenet_accuracy,
        smooth=True,
        rolling_std_window=12,
        save_as=save_dir / "imagenet-100-loss.png",
        title="Loss Curves\n VGG16 vs Flex\n(Different Mechanisms)",
        plot_loss=True,
    )

    plot_training_curve(
        file_paths=file_paths_imagenet_accuracy,
        smooth=True,
        rolling_std_window=12,
        save_as=save_dir / "imagenet-100-loss-train.png",
        title="Training Loss Curves\n VGG16 vs Flex\n(Different Mechanisms)",
        plot_loss=True,
        plot_which=["train"],
    )

    plot_training_curve(
        file_paths=file_paths_imagenet_accuracy,
        smooth=True,
        rolling_std_window=12,
        save_as=save_dir / "imagenet-100-loss-valid.png",
        title="Validation Loss Curves\n VGG16 vs Flex\n(Different Mechanisms)",
        plot_loss=True,
        plot_which=["valid"],
    )
