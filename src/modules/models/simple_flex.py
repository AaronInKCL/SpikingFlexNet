import math, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d

from src.modules.layers.flex import Flex2D
from src.utils.hyperparams.settings import AttrDict
from src.modules.models.utils import DimensionTracer
from src.utils.general import apply_kaiming_initialization
from src.modules.layers.spike import SpikingActivation


class SimpleFlexNet(nn.Module):
    def __init__(self, config: AttrDict, num_layers: int = 4, use_batch_norm: bool = False):
        super().__init__()
        self.config = config
        self.__name__ = config.get("__name__", "SimpleFlexNet")
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm

        # ✅ 动态选择激活函数
        self.activation = (
            SpikingActivation(surrogate=config.get("use_STBP", False))
            if config.get("spiking", False)
            else nn.ReLU()
        )

        self.dropout = nn.Dropout(p=config.get("dropout", 0.2))
        self.fc1 = None
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, config.get("num_classes"))

        dimension_tracer = DimensionTracer(config.get("in_dimensions"))

        class Conv2d(Flex2D if config.get("use_flex", True) else nn.Conv2d):
            def __init__(self, *args, **kwargs):
                if config.get("use_flex", True):
                    kwargs["config"] = config
                super().__init__(*args, **kwargs)
                self.in_dimensions = dimension_tracer.calculate_dimension()
                dimension_tracer(**kwargs)
                self.out_dimensions = dimension_tracer.calculate_dimension()
                if isinstance(self, Flex2D):
                    self.init_dimension_dependent_modules()

        class MaxPool2d(nn.MaxPool2d):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.in_dimensions = dimension_tracer.calculate_dimension()
                dimension_tracer(**kwargs)
                self.out_dimensions = dimension_tracer.calculate_dimension()

            def forward(self, x):
                self.output, self.indices = F.max_pool2d(
                    x, self.kernel_size, self.stride, self.padding,
                    self.dilation, self.ceil_mode, return_indices=True
                )
                return self.output

        # -------- build features --------
        layers, in_channels, out_channels = [], 3, 16
        for i in range(self.num_layers):
            layers.append(Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1))
            if self.use_batch_norm:
                layers.append(BatchNorm2d(out_channels))
            layers.append(self.activation)

            if (i + 1) % 2 == 0 and out_channels < 512:
                layers.append(MaxPool2d(kernel_size=2, stride=2))

            in_channels = out_channels
            if out_channels < 512:
                out_channels *= 2

        self.features = nn.Sequential(*layers)
        apply_kaiming_initialization(self)

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)

        if self.fc1 is None:
            in_features = x.shape[1]
            self.fc1 = nn.Linear(in_features, 128).to(x.device)

        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def check_binariness(self):
        assert self.config.get("use_flex"), "This model does not use Flex2D layers for checking binariness."
        return [m.homogeneity for m in self.modules() if isinstance(m, Flex2D)]

    def _get_cp_id_matrices(self):
        assert self.config.get("use_flex"), "This model does not use Flex2D layers for checking conv ratio."
        return [(m.cp_identity_matrix >= 0.5).float() for m in self.modules() if isinstance(m, Flex2D)]

    def check_conv_ratio(self):
        assert self.config.get("use_flex"), "This model does not use Flex2D layers for checking conv ratio."
        return [m.conv_ratio for m in self.modules() if isinstance(m, Flex2D)]
