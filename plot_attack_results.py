import os
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ---------- 配置区 ----------
checkpoints_dir = './checkpoints'  # 你的 checkpoints 文件夹路径
attack_types = ['FGSM', 'PGD', 'SPGD']  # 你有的攻击类型
model_prefix = 'cifar10_VGG16_'  # 模型文件夹前缀
output_path = './attack_plots/combined_attack_plot_final.png'  # 输出图片路径

# 你自己定义的简洁版模型名字，顺序要跟你的模型文件夹一一对应！
model_display_names = [
    'ANN_VGG16', 
    'ANN_FLEX', 
    'SNN_Hybrid_FLEX', 
    'SNN_Hybrid_VGG16', 
    'SNN_STBP_VGG16', 
    'SNN_STBP_FLEX'
]
# ----------------------------

os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Seaborn 美化
sns.set(style="whitegrid", context="talk", palette="tab10")
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['font.family'] = 'sans-serif'

def load_attack_data(model_dir, attack_type):
    """加载单个模型、单种攻击的accuracy和auc数据"""
    attack_dir = os.path.join(model_dir, 'results', attack_type)
    accuracy_file = os.path.join(attack_dir, f'{attack_type}_model_accuracies_top_1.json')
    auc_file = os.path.join(attack_dir, f'{attack_type}_model_auc_score_top_1.json')

    if not os.path.exists(accuracy_file) or not os.path.exists(auc_file):
        print(f"Warning: Missing files for {attack_type} in {model_dir}")
        return None, None

    with open(accuracy_file, 'r') as f:
        accuracy_data = json.load(f)
    with open(auc_file, 'r') as f:
        auc_data = json.load(f)

    epsilons = accuracy_data['epsilons']
    accuracies = accuracy_data['accuracies']
    auc_key = f"{attack_type} Attack Area Under Curve Top 1"
    auc_score = auc_data[auc_key]

    return epsilons, accuracies, auc_score

def plot_combined_attack(models_data, model_names, display_names):
    """绘制合并图"""
    n_attacks = len(attack_types)
    fig, axes = plt.subplots(2, n_attacks, figsize=(7 * n_attacks, 10), gridspec_kw={'height_ratios': [3, 1]})
    plt.suptitle('Adversarial Attack Performance', fontsize=24, fontweight='bold')

    # 固定颜色和 marker
    colors = sns.color_palette("tab10", n_colors=len(model_names))
    markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X']

    for idx, attack_type in enumerate(attack_types):
        # 上面：折线图 (Accuracy vs Epsilon)
        ax1 = axes[0, idx]
        for model_idx, model_name in enumerate(model_names):
            data = models_data[model_name][attack_type]
            epsilons, accuracies, auc_score = data
            ax1.plot(
                epsilons, accuracies,
                label=display_names[model_idx],
                color=colors[model_idx],
                marker=markers[model_idx % len(markers)],
                linewidth=2,
                markersize=6
            )
        ax1.set_title(f'{attack_type} Attack', fontsize=18)
        ax1.set_xlabel('Epsilon', fontsize=16)
        if idx == 0:
            ax1.set_ylabel('Top-1 Accuracy', fontsize=16)
        ax1.set_ylim(0, 0.1)  # 设置成 0-0.1
        ax1.grid(True, linestyle='--', alpha=0.7)

        # 下面：柱状图 (AUC vs Model)
        ax2 = axes[1, idx]
        auc_values = [models_data[m][attack_type][2] for m in model_names]
        bars = ax2.bar(
            np.arange(len(display_names)), auc_values, 
            color=colors,
            alpha=0.8
        )
        ax2.set_xlabel('Models', fontsize=16)
        if idx == 0:
            ax2.set_ylabel('AUC', fontsize=16)
        ax2.set_ylim(0, max(auc_values) * 1.2)

        # ⚠️ 去掉柱状图 x 轴模型名字
        ax2.set_xticks([])  # 不显示x轴标签

        ax2.grid(True, linestyle='--', alpha=0.7)

        # 在每个柱子上标数值
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.4f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords='offset points',
                         ha='center', va='bottom', fontsize=10)

    # --- 统一 Legend ---
    lines = []
    labels = []
    for model_idx, short_name in enumerate(display_names):
        line = plt.Line2D(
            [], [], color=colors[model_idx],
            marker=markers[model_idx % len(markers)],
            linestyle='-', linewidth=2, markersize=8,
            label=short_name
        )
        lines.append(line)
        labels.append(short_name)

    fig.legend(
        handles=lines,
        labels=labels,
        loc='lower center',
        ncol=len(model_names),
        fontsize=20,  # Legend 字体大一些
        frameon=False,
        bbox_to_anchor=(0.5, -0.05)
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot: {output_path}")
    plt.close()

def main():
    all_models = [d for d in os.listdir(checkpoints_dir) if d.startswith(model_prefix)]
    if not all_models:
        print("No models found with prefix", model_prefix)
        return

    print(f"Found models: {all_models}")

    models_data = {}

    for model in all_models:
        model_dir = os.path.join(checkpoints_dir, model)
        models_data[model] = {}
        for attack_type in attack_types:
            epsilons, accuracies, auc_score = load_attack_data(model_dir, attack_type)
            if epsilons is not None and accuracies is not None:
                models_data[model][attack_type] = (epsilons, accuracies, auc_score)

    if models_data:
        plot_combined_attack(models_data, list(models_data.keys()), model_display_names)

if __name__ == "__main__":
    main()
