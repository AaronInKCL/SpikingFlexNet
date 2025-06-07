#!/Users/donyin/miniconda3/envs/imperial/bin/python

# import donware
# from donware import inspect_package
# inspect_package(donware)

import os, pandas
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


from src.plot.params import FIGURE_SIZE, FONTSIZE_TITLE, FONTSIZE_LABEL, FONTSIZE_TICK, PAD_TITLE, DPI_FIGURE

FIGURE_SIZE = (FIGURE_SIZE[0] * 0.6, FIGURE_SIZE[1] * 0.8)


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
    total_epochs=210,
    keep_first_n_epoch=120,
    exclude_the_first_n_epoch=0,
):
    """
    40 steps per epoch, 180 epochs
    """
    legend_size_adjustment_factor = 0.75

    # compute the data to keep
    train_data = pandas.read_csv(next(iter(file_paths.values()))["train"])
    num_steps_per_epoch = len(train_data) // total_epochs
    keep_first_n_row = num_steps_per_epoch * keep_first_n_epoch

    plt.figure(figsize=FIGURE_SIZE)
    color_map = {}

    for model, paths in file_paths.items():
        train_data = pandas.read_csv(paths["train"])
        valid_data = pandas.read_csv(paths["valid"])

        # Exclude the first n epochs
        train_data = train_data.iloc[exclude_the_first_n_epoch * num_steps_per_epoch :]
        valid_data = valid_data.iloc[exclude_the_first_n_epoch * num_steps_per_epoch :]
        train_data.reset_index(drop=True, inplace=True)
        valid_data.reset_index(drop=True, inplace=True)

        # Function to handle outliers
        def handle_outliers(data, lower_percentile=3, upper_percentile=97):
            float_columns = data.select_dtypes(include=[np.float64, np.float32]).columns
            for column in float_columns:
                lower_bound = np.percentile(data[column], lower_percentile)
                upper_bound = np.percentile(data[column], upper_percentile)
                data.loc[data[column] < lower_bound, column] = lower_bound
                data.loc[data[column] > upper_bound, column] = upper_bound
            return data

        # Handle outliers for train and validation data
        train_data = handle_outliers(train_data)
        valid_data = handle_outliers(valid_data)

        # keep the first keep_first_n rows
        train_data = train_data.iloc[:keep_first_n_row]
        valid_data = valid_data.iloc[:keep_first_n_row]

        train_values_raw = train_data.iloc[:, 1]
        valid_values_raw = valid_data.iloc[:, 1]

        train_values = smooth_curve(train_values_raw, weight=0.9) if smooth else train_values_raw
        valid_values = smooth_curve(valid_values_raw, weight=0.998) if smooth else valid_values_raw

        train_std = train_values_raw.rolling(window=rolling_std_window).std().fillna(0)
        valid_std = valid_values_raw.rolling(window=rolling_std_window).std().fillna(0)

        color = sns.color_palette("tab10")[list(file_paths.keys()).index(model)]
        color_map[model] = color

        sns.lineplot(x=train_data.iloc[:, 0], y=train_values, linestyle="-", linewidth=1.5, color=color)
        plt.fill_between(train_data.iloc[:, 0], train_values - train_std, train_values + train_std, color=color, alpha=0.2)

        sns.lineplot(x=valid_data.iloc[:, 0], y=valid_values, linestyle="--", linewidth=1.5, color=color)
        plt.fill_between(valid_data.iloc[:, 0], valid_values - valid_std, valid_values + valid_std, color=color, alpha=0.2)

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
    train_line = plt.Line2D([0], [0], color="black", linestyle="-", linewidth=linewidth)
    valid_line = plt.Line2D([0], [0], color="black", linestyle="--", linewidth=linewidth)
    train_valid_legend = plt.legend(
        [train_line, valid_line],
        ["Train", "Validation"],
        loc="lower right",
        bbox_to_anchor=(2, _h_),
        fontsize=FONTSIZE_LABEL * legend_size_adjustment_factor,
    )

    # Add both legends to the plot
    plt.gca().add_artist(train_valid_legend)
    plt.gca().add_artist(model_legend)

    # Manually amend x-axis labels to show epochs
    max_step = train_data.iloc[:, 0].max()
    epochs = range(0, (max_step // num_steps_per_epoch) + 1)
    epoch_steps = [e * num_steps_per_epoch for e in epochs if e % label_every_n_epochs == 0 or e == epochs[-1]]
    epoch_labels = [e for e in epochs if e % label_every_n_epochs == 0 or e == epochs[-1]]
    epoch_labels[-1] += 1
    plt.xticks(ticks=epoch_steps, labels=epoch_labels, fontsize=FONTSIZE_TICK)

    plt.title(title, fontsize=FONTSIZE_TITLE, pad=PAD_TITLE)
    plt.xlabel("Epochs", fontsize=FONTSIZE_LABEL, labelpad=PAD_TITLE)
    plt.ylabel("Accuracy", fontsize=FONTSIZE_LABEL, labelpad=PAD_TITLE)

    plt.tick_params(axis="y", labelsize=FONTSIZE_TICK)
    plt.tight_layout()
    plt.savefig(save_as, dpi=DPI_FIGURE, bbox_extra_artists=(model_legend, train_valid_legend), bbox_inches="tight")


if __name__ == "__main__":
    # --------------------------------
    save_dir = Path("figures") / "training_curve"
    save_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path("data/training_curve/cifar-10-epoch-210-batch-16")

    for file in os.listdir(data_dir):
        print(file)

    # -------------------------------- loss of cifar-10 --------------------------------
    file_paths_cifar_loss = {
        "VGG16": {
            "train": "data/training_curve/cifar-10-epoch-210-batch-16/original-vgg16-train-loss.csv",
            "valid": "data/training_curve/cifar-10-epoch-210-batch-16/original-vgg16-valid-loss.csv",
        },
        "VGG16 (Flex): Threshold + Scaled Sigmoid": {
            "train": "data/training_curve/cifar-10-epoch-210-batch-16/threshold-scaled-sigmoid-train-loss.csv",
            "valid": "data/training_curve/cifar-10-epoch-210-batch-16/threshold-scaled-sigmoid-valid-loss.csv",
        },
        "VGG16 (Flex): Threshold + Hard Sigmoid": {
            "train": "data/training_curve/cifar-10-epoch-210-batch-16/threshold-hard-sigmoid-train-loss.csv",
            "valid": "data/training_curve/cifar-10-epoch-210-batch-16/threshold-hard-sigmoid-valid-loss.csv",
        },
        "VGG16 (Flex): Threshold + STE": {
            "train": "data/training_curve/cifar-10-epoch-210-batch-16/threshold-ste-train-loss.csv",
            "valid": "data/training_curve/cifar-10-epoch-210-batch-16/threshold-ste-valid-loss.csv",
        },
        "VGG16 (Flex): Threshold + SR": {
            "train": "data/training_curve/cifar-10-epoch-210-batch-16/threshold-sr-train-loss.csv",
            "valid": "data/training_curve/cifar-10-epoch-210-batch-16/threshold-sr-valid-loss.csv",
        },
        "VGG16 (Flex): SAB + Scaled Sigmoid": {
            "train": "data/training_curve/cifar-10-epoch-210-batch-16/sab-scaled-sigmoid-train-loss.csv",
            "valid": "data/training_curve/cifar-10-epoch-210-batch-16/sab-scaled-sigmoid-valid-loss.csv",
        },
        "VGG16 (Flex): SAB + Hard Sigmoid": {
            "train": "data/training_curve/cifar-10-epoch-210-batch-16/sab-hard-sigmoid-train-loss.csv",
            "valid": "data/training_curve/cifar-10-epoch-210-batch-16/sab-hard-sigmoid-valid-loss.csv",
        },
        "VGG16 (Flex): SAB + STE": {
            "train": "data/training_curve/cifar-10-epoch-210-batch-16/sab-ste-train-loss.csv",
            "valid": "data/training_curve/cifar-10-epoch-210-batch-16/sab-ste-valid-loss.csv",
        },
        "VGG16 (Flex): SAB + SR": {
            "train": "data/training_curve/cifar-10-epoch-210-batch-16/sab-sr-train-loss.csv",
            "valid": "data/training_curve/cifar-10-epoch-210-batch-16/sab-sr-valid-loss.csv",
        },
        "VGG16 (Flex): Channelwise MaxPool": {
            "train": "data/training_curve/cifar-10-epoch-210-batch-16/cmp-train-loss.csv",
            "valid": "data/training_curve/cifar-10-epoch-210-batch-16/cmp-valid-loss.csv",
        },
    }

    plot_training_curve(
        file_paths=file_paths_cifar_loss,
        smooth=True,
        rolling_std_window=12,
        save_as=save_dir / "cifar-loss.png",
        title="Loss Curves \n VGG16 vs Flex\n(Different Mechanisms)",
    )

    # -------------------------------- accuracy of cifar-10 --------------------------------

    file_paths_cifar_accuracy = {
        "VGG16 (Original)": {
            "train": "data/training_curve/cifar-10-epoch-210-batch-16/original-vgg16-train-accuracy-balanced.csv",
            "valid": "data/training_curve/cifar-10-epoch-210-batch-16/original-vgg16-valid-accuracy-balanced.csv",
        },
        "VGG16 (Flex): Threshold + Scaled Sigmoid": {
            "train": "data/training_curve/cifar-10-epoch-210-batch-16/threshold-scaled-sigmoid-train-accuracy-balanced.csv",
            "valid": "data/training_curve/cifar-10-epoch-210-batch-16/threshold-scaled-sigmoid-valid-accuracy-balanced.csv",
        },
        "VGG16 (Flex): Threshold + Hard Sigmoid": {
            "train": "data/training_curve/cifar-10-epoch-210-batch-16/threshold-hard-sigmoid-train-accuracy-balanced.csv",
            "valid": "data/training_curve/cifar-10-epoch-210-batch-16/threshold-hard-sigmoid-valid-accuracy-balanced.csv",
        },
        "VGG16 (Flex): Threshold + STE": {
            "train": "data/training_curve/cifar-10-epoch-210-batch-16/threshold-ste-train-accuracy-balanced.csv",
            "valid": "data/training_curve/cifar-10-epoch-210-batch-16/threshold-ste-valid-accuracy-balanced.csv",
        },
        "VGG16 (Flex): Threshold + SR": {
            "train": "data/training_curve/cifar-10-epoch-210-batch-16/threshold-sr-train-accuracy-balanced.csv",
            "valid": "data/training_curve/cifar-10-epoch-210-batch-16/threshold-sr-valid-accuracy-balanced.csv",
        },
        "VGG16 (Flex): SAB + Scaled Sigmoid": {
            "train": "data/training_curve/cifar-10-epoch-210-batch-16/sab-scaled-sigmoid-train-accuracy-balanced.csv",
            "valid": "data/training_curve/cifar-10-epoch-210-batch-16/sab-scaled-sigmoid-valid-accuracy-balanced.csv",
        },
        "VGG16 (Flex): SAB + Hard Sigmoid": {
            "train": "data/training_curve/cifar-10-epoch-210-batch-16/sab-hard-sigmoid-train-accuracy-balanced.csv",
            "valid": "data/training_curve/cifar-10-epoch-210-batch-16/sab-hard-sigmoid-valid-accuracy-balanced.csv",
        },
        "VGG16 (Flex): SAB + STE": {
            "train": "data/training_curve/cifar-10-epoch-210-batch-16/sab-ste-train-accuracy-balanced.csv",
            "valid": "data/training_curve/cifar-10-epoch-210-batch-16/sab-ste-valid-accuracy-balanced.csv",
        },
        "VGG16 (Flex): SAB + SR": {
            "train": "data/training_curve/cifar-10-epoch-210-batch-16/sab-sr-train-accuracy-balanced.csv",
            "valid": "data/training_curve/cifar-10-epoch-210-batch-16/sab-sr-valid-accuracy-balanced.csv",
        },
        "VGG16 (Flex): Channelwise MaxPool": {
            "train": "data/training_curve/cifar-10-epoch-210-batch-16/cmp-train-accuracy-balanced.csv",
            "valid": "data/training_curve/cifar-10-epoch-210-batch-16/cmp-valid-accuracy-balanced.csv",
        },
    }

    plot_training_curve(
        file_paths=file_paths_cifar_accuracy,
        smooth=True,
        rolling_std_window=12,
        save_as=save_dir / "cifar-accuracy.png",
        title="Accuracy Curves\n VGG16 vs Flex\n(Different Mechanisms)",
    )
