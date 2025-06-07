import sys
sys.path.append("/home/zephyr/flexnet/Flexible-Neurons-main")

import matplotlib.pyplot as plt
from pathlib import Path
from src.analysis.run_loader import RunLoader


def plot_train_valid_accuracies(run_path):
    run_loader = RunLoader(Path(run_path), whether_load_checkpoint=False)
    logger = run_loader.logger

    # 从 SQLite 数据库中提取表为 DataFrame
    df = logger.get_dataframe()  # ←←← 这是关键

    # 输出列名供调试
    print("列名:", df.columns.tolist())

    # 提取各项数据
    epochs = df["Epoch"]
    train_acc = df["Train Accuracy"]
    valid_acc = df["Valid Accuracy"]
    train_acc_bal = df["Train Accuracy Balanced"]
    valid_acc_bal = df["Valid Accuracy Balanced"]

    # 可视化
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, valid_acc, label="Valid Accuracy")
    plt.plot(epochs, train_acc_bal, label="Train Balanced Accuracy", linestyle="--")
    plt.plot(epochs, valid_acc_bal, label="Valid Balanced Accuracy", linestyle="--")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train and Valid Accuracy Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("train_valid_accuracy.png")
    plt.show()



if __name__ == "__main__":
    run_path = "checkpoints/mnist_VGG6_SNN_NoSTBP_NoFLEX"  # 修改为你的运行路径
    plot_train_valid_accuracies(run_path)
