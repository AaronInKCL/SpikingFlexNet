import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import sys
sys.path.append("/home/zephyr/flexnet/Flexible-Neurons-main")
from src.plot.params import FIGURE_SIZE, FONTSIZE_TITLE, FONTSIZE_LABEL, FONTSIZE_TICK, PAD_TITLE, DPI_FIGURE

FIGURE_SIZE = (FIGURE_SIZE[0] * 0.6, FIGURE_SIZE[1] * 0.8)

def smooth_curve(data, weight=0.9, start_from=3, baseline=0.1):
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

def plot_training_curve_combined(
    file_paths,
    smooth=True,
    rolling_std_window=10,
    save_as=None,
    label_every_n_epochs=20,
    linewidth=4,
    total_epochs=120,
    keep_first_n_epoch=120,
    exclude_the_first_n_epoch=0,
    best_point_start_epoch=20,
):
    legend_size_adjustment_factor = 0.75
    num_steps_per_epoch = 39
    keep_first_n_row = num_steps_per_epoch * keep_first_n_epoch

    fig, axs = plt.subplots(1, 2, figsize=(FIGURE_SIZE[0]*2, FIGURE_SIZE[1]), sharex=False)

    color_map = {}
    best_acc_points = []   # 记录 accuracy 子图的最优点 (epoch, value, color)
    best_loss_points = []  # 记录 loss 子图的最优点 (epoch, value, color)

    for model, path in file_paths.items():
        data = pd.read_csv(path)

        data = data.iloc[exclude_the_first_n_epoch * num_steps_per_epoch:]
        data.reset_index(drop=True, inplace=True)
        data.fillna(0, inplace=True)

        def handle_outliers(data, lower_percentile=3, upper_percentile=97):
            float_columns = data.select_dtypes(include=[np.float64, np.float32]).columns
            for column in float_columns:
                lower_bound = np.percentile(data[column], lower_percentile)
                upper_bound = np.percentile(data[column], upper_percentile)
                data.loc[data[column] < lower_bound, column] = lower_bound
                data.loc[data[column] > upper_bound, column] = upper_bound
            return data

        data = handle_outliers(data)
        data = data.iloc[:keep_first_n_row]

        train_acc_raw = data["Train Accuracy Balanced"]
        valid_acc_raw = data["Valid Accuracy Balanced"]
        train_loss_raw = data["Train Loss"]
        valid_loss_raw = data["Valid Loss"]

        train_acc = pd.Series(smooth_curve(train_acc_raw, weight=0.9) if smooth else train_acc_raw)
        valid_acc = pd.Series(smooth_curve(valid_acc_raw, weight=0.998) if smooth else valid_acc_raw)
        train_loss = pd.Series(smooth_curve(train_loss_raw, weight=0.9) if smooth else train_loss_raw)
        valid_loss = pd.Series(smooth_curve(valid_loss_raw, weight=0.998) if smooth else valid_loss_raw)

        train_acc_std = train_acc_raw.rolling(window=rolling_std_window).std().fillna(0)
        valid_acc_std = valid_acc_raw.rolling(window=rolling_std_window).std().fillna(0)
        train_loss_std = train_loss_raw.rolling(window=rolling_std_window).std().fillna(0)
        valid_loss_std = valid_loss_raw.rolling(window=rolling_std_window).std().fillna(0)

        color = sns.color_palette("tab10")[list(file_paths.keys()).index(model)]
        color_map[model] = color

        epochs = data["Epoch"]

        # 绘制主曲线
        sns.lineplot(ax=axs[0], x=epochs, y=train_acc, linestyle="-", linewidth=1.5, color=color)
        axs[0].fill_between(epochs, train_acc - train_acc_std, train_acc + train_acc_std, color=color, alpha=0.2)

        sns.lineplot(ax=axs[1], x=epochs, y=train_loss, linestyle="-", linewidth=1.5, color=color)
        axs[1].fill_between(epochs, train_loss - train_loss_std, train_loss + train_loss_std, color=color, alpha=0.2)

        axs[0].plot(epochs, valid_acc, linestyle="--", linewidth=1.5, color=color)
        axs[0].fill_between(epochs, valid_acc - valid_acc_std, valid_acc + valid_acc_std, color=color, alpha=0.2)

        axs[1].plot(epochs, valid_loss, linestyle="--", linewidth=1.5, color=color)
        axs[1].fill_between(epochs, valid_loss - valid_loss_std, valid_loss + valid_loss_std, color=color, alpha=0.2)

        # -----------------------
        # Best point: 只从第 20 epoch 开始考虑
        mask = epochs >= best_point_start_epoch
        epochs_valid = epochs[mask].reset_index(drop=True)
        valid_acc_sub = valid_acc[mask]
        valid_loss_sub = valid_loss[mask]

        # 保存最佳 Validation Accuracy
        best_acc_idx = np.argmax(valid_acc_sub)
        best_acc_epoch = epochs_valid[best_acc_idx]
        best_acc_value = valid_acc_sub.iloc[best_acc_idx]
        best_acc_points.append((best_acc_epoch, best_acc_value, color))

        # 保存最佳 Validation Loss
        best_loss_idx = np.argmin(valid_loss_sub)
        best_loss_epoch = epochs_valid[best_loss_idx]
        best_loss_value = valid_loss_sub.iloc[best_loss_idx]
        best_loss_points.append((best_loss_epoch, best_loss_value, color))

    # ---------- Accuracy 图上画最佳点 ----------
    offset_unit_acc = (axs[0].get_ylim()[1] - axs[0].get_ylim()[0]) * 0.04
    for i, (epoch, value, color) in enumerate(best_acc_points):
        axs[0].axvline(x=epoch, linestyle=":", linewidth=1, color=color, alpha=0.7)
        axs[0].scatter(epoch, value, color=color, s=40, marker='o')
        offset = (i + 1) * offset_unit_acc
        axs[0].text(epoch, value + offset, f"{value:.3f}", fontsize=9, ha='center', va='bottom')

    # ---------- Loss 图上画最佳点 ----------
    offset_unit_loss = (axs[1].get_ylim()[1] - axs[1].get_ylim()[0]) * 0.07
    for i, (epoch, value, color) in enumerate(best_loss_points):
        axs[1].axvline(x=epoch, linestyle=":", linewidth=1, color=color, alpha=0.7)
        axs[1].scatter(epoch, value, color=color, s=40, marker='o')
        offset = (i + 1) * offset_unit_loss
        axs[1].text(epoch, value + offset, f"{value:.3f}", fontsize=9, ha='center', va='bottom')

    # ---------- 画标题 ----------
    axs[0].set_title("Accuracy Curves\nVGG16 vs Flex\n(Different Mechanisms)", fontsize=FONTSIZE_TITLE, pad=PAD_TITLE)
    axs[0].set_xlabel("Epochs", fontsize=FONTSIZE_LABEL, labelpad=PAD_TITLE)
    axs[0].set_ylabel("Accuracy", fontsize=FONTSIZE_LABEL, labelpad=PAD_TITLE)

    axs[1].set_title("Loss Curves\nVGG16 vs Flex\n(Different Mechanisms)", fontsize=FONTSIZE_TITLE, pad=PAD_TITLE)
    axs[1].set_xlabel("Epochs", fontsize=FONTSIZE_LABEL, labelpad=PAD_TITLE)
    axs[1].set_ylabel("Loss", fontsize=FONTSIZE_LABEL, labelpad=PAD_TITLE)

    # ---------- 画 Train / Validation 图例 ----------
    for ax in axs:
        legend_lines = []
        legend_labels = []
        legend_lines.append(plt.Line2D([0], [0], color="black", linestyle="-", linewidth=4))
        legend_labels.append("Train")
        legend_lines.append(plt.Line2D([0], [0], color="black", linestyle="--", linewidth=4))
        legend_labels.append("Validation")
        ax.legend(
            legend_lines,
            legend_labels,
            loc="center right",
            fontsize=FONTSIZE_LABEL * legend_size_adjustment_factor,
        )

    # ---------- 画模型图例 ----------
    model_lines = [plt.Line2D([0], [0], color=color_map[model], linestyle="-", linewidth=4) for model in file_paths.keys()]
    model_legend = fig.legend(
        model_lines,
        file_paths.keys(),
        loc="center right",
        bbox_to_anchor=(1.25, 0.5),
        fontsize=FONTSIZE_LABEL * legend_size_adjustment_factor,
    )

    plt.tight_layout()
    plt.savefig(save_as, dpi=DPI_FIGURE, bbox_inches="tight")
    
    # 保存 PDF
    save_as_pdf = str(save_as).replace('.png', '.pdf')
    plt.savefig(save_as_pdf, dpi=DPI_FIGURE, bbox_inches="tight", format='pdf')

    plt.show()

# ------------- Main --------------
if __name__ == "__main__":
    save_dir = Path("figures") / "training_curve"
    save_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path("data_csv/training_curve")

    with open(data_dir / "registry.json", "r") as f:
        registry = json.load(f)

    file_paths = {registry[key]: str(data_dir / f"{key}.csv") for key in registry}

    plot_training_curve_combined(
        file_paths=file_paths,
        smooth=True,
        rolling_std_window=12,
        save_as=save_dir / "cifar10-vgg16-combined-accuracy-loss.png",
    )
