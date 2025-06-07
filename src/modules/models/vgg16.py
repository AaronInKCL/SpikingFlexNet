import torch, math
import torch.nn as nn
import torch.nn.functional as F
from rich import print
from torchsummary import summary
from torch.nn import BatchNorm2d

from donware import banner
from src.modules.layers.flex import Flex2D
from src.modules.models.utils import DimensionTracer
from src.modules.layers.spike import SpikingActivation


class VGG16(nn.Module):
    def __init__(self, config, verbose=True):
        super().__init__()
        self.config = config
        self.default_dense_size = 4096
        self.dropout = nn.Dropout(p=self.config.get("dropout", 0.2))
        self.fc1 = None
        self.fc2 = nn.Linear(self.default_dense_size, self.default_dense_size)
        self.fc3 = nn.Linear(self.default_dense_size, config.get("num_classes"))

        self.activation = (
            SpikingActivation(surrogate=config.get("use_STBP", False))
            if config.get("spiking", False)
            else nn.ReLU()
        )

        dimension_tracer = DimensionTracer(config.get("in_dimensions"))

        class Conv2d(Flex2D if config.get("use_flex") else nn.Conv2d):
            def __init__(self, *args, **kwargs):
                if isinstance(self, Flex2D):
                    kwargs["config"] = config
                super().__init__(*args, **kwargs)
                self.in_dimensions = dimension_tracer.calculate_dimension()
                dimension_tracer(**kwargs)
                self.out_dimensions = dimension_tracer.calculate_dimension()
                if isinstance(self, Flex2D):
                    self.init_dimension_dependent_modules()

        self.features = nn.Sequential(
            # Block 1
            Conv2d(3, 64, 3, padding=1), BatchNorm2d(64), self.activation,
            Conv2d(64, 64, 3, padding=1), BatchNorm2d(64), self.activation,
            Conv2d(64, 64, 2, stride=2),  # Pool

            # Block 2
            Conv2d(64, 128, 3, padding=1), BatchNorm2d(128), self.activation,
            Conv2d(128, 128, 3, padding=1), BatchNorm2d(128), self.activation,
            Conv2d(128, 128, 2, stride=2),  # Pool

            # Block 3
            Conv2d(128, 256, 3, padding=1), BatchNorm2d(256), self.activation,
            Conv2d(256, 256, 3, padding=1), BatchNorm2d(256), self.activation,
            Conv2d(256, 256, 3, padding=1), BatchNorm2d(256), self.activation,
            Conv2d(256, 256, 2, stride=2),  # Pool

            # Block 4
            Conv2d(256, 512, 3, padding=1), BatchNorm2d(512), self.activation,
            Conv2d(512, 512, 3, padding=1), BatchNorm2d(512), self.activation,
            Conv2d(512, 512, 3, padding=1), BatchNorm2d(512), self.activation,
            Conv2d(512, 512, 2, stride=2),  # Pool

            # Block 5
            Conv2d(512, 512, 3, padding=1), BatchNorm2d(512), self.activation,
            Conv2d(512, 512, 3, padding=1), BatchNorm2d(512), self.activation,
            Conv2d(512, 512, 3, padding=1), BatchNorm2d(512), self.activation,
            Conv2d(512, 512, 2, stride=2),  # Pool
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)

        if self.fc1 is None:
            in_features = x.shape[1]
            self.fc1 = nn.Linear(in_features, self.default_dense_size).to(x.device)

        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def check_binariness(self):
        assert self.config.get("use_flex")
        return [m.homogeneity for m in self.modules() if isinstance(m, Flex2D)]

    def _get_cp_id_matrices(self):
        assert self.config.get("use_flex")
        return [(m.cp_identity_matrix >= 0.5).float() for m in self.modules() if isinstance(m, Flex2D)]

    def check_conv_ratio(self):
        assert self.config.get("use_flex")
        return [m.conv_ratio for m in self.modules() if isinstance(m, Flex2D)]


if __name__ == "__main__":
    config = {
        "use_flex": False,
        "in_dimensions": (3, 32, 32),
        "num_classes": 10,
        "spiking": True,
        "use_STBP": True
    }
    model = VGG16(config)
    summary(model, (3, 32, 32))
