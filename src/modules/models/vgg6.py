#!/Users/donyin/miniconda3/envs/imperial/bin/python

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


class VGG6(nn.Module):
    def __init__(self, config, verbose=True):
        super().__init__()
        self.config = config
        self.default_dense_size = 256
        self.dropout = nn.Dropout(p=self.config.get("dropout", 0.2))
        self.fc1 = None  
        self.fc2 = nn.Linear(self.default_dense_size, self.default_dense_size)
        self.fc3 = nn.Linear(self.default_dense_size, config.get("num_classes"))

        # -------- 获取激活函数 --------
        self.activation = (
            SpikingActivation(surrogate=config.get("use_STBP", False))
            if config.get("spiking", False)
            else nn.ReLU()
        )

        # -------- dimension tracker --------
        dimension_tracer = DimensionTracer(config.get("in_dimensions"))

        # -------- dynamic Conv2d --------
        class Conv2d(Flex2D if self.config.get("use_flex") else nn.Conv2d):
            def __init__(conv_self, *args, **kwargs):  # 用 conv_self 避免和外层 self 混淆
                if self.config.get("use_flex"):
                    kwargs["config"] = config
                super(Conv2d, conv_self).__init__(*args, **kwargs)

                conv_self.in_dimensions = dimension_tracer.calculate_dimension()
                dimension_tracer(**kwargs)
                conv_self.out_dimensions = dimension_tracer.calculate_dimension()

                # 强制调用初始化
                if self.config.get("use_flex"):
                    conv_self.init_dimension_dependent_modules()


        # -------- MaxPool wrapper --------
        class MaxPool2d(nn.MaxPool2d):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.in_dimensions = dimension_tracer.calculate_dimension()
                dimension_tracer(**kwargs)
                self.out_dimensions = dimension_tracer.calculate_dimension()

            def forward(self, x):
                self.output, self.indices = F.max_pool2d(
                    x, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, return_indices=True
                )
                return self.output

        # -------- features part --------
        self.features = nn.Sequential(
            Conv2d(3, 64, 3, padding=1),
            BatchNorm2d(64),
            self.activation,

            Conv2d(64, 64, 3, padding=1),
            BatchNorm2d(64),
            self.activation,

            Conv2d(64, 64, 2, stride=2),  # downsample

            Conv2d(64, 128, 3, padding=1),
            BatchNorm2d(128),
            self.activation,

            Conv2d(128, 128, 3, padding=1),
            BatchNorm2d(128),
            self.activation,

            Conv2d(128, 128, 2, stride=2),  # downsample
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
        assert self.config.get("use_flex"), "Flex2D required"
        return [m.homogeneity for m in self.modules() if isinstance(m, Flex2D)]

    def _get_cp_id_matrices(self):
        assert self.config.get("use_flex"), "Flex2D required"
        return [(m.cp_identity_matrix >= 0.5).float() for m in self.modules() if isinstance(m, Flex2D)]

    def check_conv_ratio(self):
        assert self.config.get("use_flex"), "Flex2D required"
        return [m.conv_ratio for m in self.modules() if isinstance(m, Flex2D)]
    



if __name__ == "__main__":
    config = {
        "use_flex": False,
        "in_dimensions": (3, 32, 32),
        "num_classes": 10,
        "spiking": True,
        "use_STBP": True,
    }
    model = VGG6(config)
    summary(model, (3, 32, 32))
