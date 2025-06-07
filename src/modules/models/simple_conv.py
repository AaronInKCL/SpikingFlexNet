import torch
import torch.nn as nn
from src.utils.hyperparams.settings import AttrDict
from src.utils.general import apply_kaiming_initialization
from src.modules.layers.spike import SpikingActivation  # ✅ 你这行路径可能是 spike 而非 layers.spike


class SimpleConvNet(nn.Module):
    def __init__(self, config: AttrDict):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = None  # ✅ 延迟定义
        self.fc2 = nn.Linear(128, 10)

        if config.get("spiking", False):
            self.activation = SpikingActivation(surrogate=config.get("use_STBP", False))
        else:
            self.activation = nn.ReLU()

        # Kaiming 初始化稍后调用（因为 fc1 还没定义）
        self.config = config

    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = self.pool(self.activation(self.conv3(x)))
        x = self.pool(self.activation(self.conv4(x)))
        x = torch.flatten(x, 1)

        # ✅ 第一次 forward 时定义 fc1
        if self.fc1 is None:
            in_features = x.shape[1]
            self.fc1 = nn.Linear(in_features, 128).to(x.device)
            apply_kaiming_initialization(self.fc1)

        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x
