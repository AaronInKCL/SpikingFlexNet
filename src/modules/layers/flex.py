#!/Users/donyin/miniconda3/envs/imperial/bin/python

import torch
import torch.nn as nn
from pathlib import Path
from src.modules import masks
from src.modules import logits
from torch.nn import functional as F
from src.utils.device import select_device
from src.modules.layers._utils import channel_interpolate
from src.modules.joint.channelwise_maxpool import channel_wise_maxpool
from src.training.mask_monitor import measure_homogeneity, count_conv_ratio
from src.modules.masks.sigmoid import sigmoid_plain, HardsigmoidFunc
from src.modules.layers.spike import SpikingActivation  # ✅ 新增：导入脉冲激活函数


class MaxPool2d(nn.MaxPool2d):
    """this is a wrapper that returns indices for computing hessian"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        self.output, self.indices = F.max_pool2d(
            x, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, return_indices=True
        )
        return self.output


class Flex2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, config=None):
        super().__init__()
        assert config, "Missing config file for Flex2D"
        self.config = config
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.device = select_device()

        self.flex_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.flex_pool = MaxPool2d(kernel_size, stride, padding)
        self.bn_logits = nn.BatchNorm2d(self.out_channels)

        self.homogeneity = 0
        self.conv_ratio = 0
        self.cp_identity_matrix = None

        if "SpatialAttention" in self.config.get("logits_mechanism", ""):
            self.spatial_attention_block = getattr(logits, self.config.logits_mechanism)(
                num_blocks=self.config.num_spatial_attention_block,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
            )

        # ✅ 新增：根据 spiking 和 STBP 初始化激活函数
        if self.config.get("spiking", False):
            self.activation = SpikingActivation(surrogate=self.config.get("use_STBP", False))
        else:
            self.activation = nn.ReLU()

    def init_dimension_dependent_modules(self):
        assert hasattr(self, "out_dimensions"), "out_dimensions must be specified before initializing threshold"
        self.threshold = nn.Parameter(torch.randn(*self.out_dimensions)).to(self.device)
        nn.init.kaiming_uniform_(self.threshold)

    def forward(self, x):
        t_flex_pool = self.flex_pool(x)
        t_flex_conv = self.flex_conv(x)
        t_flex_pool = channel_interpolate(t_flex_pool, self.out_channels)

        match self.config.get("joint_mechanism", False):
            case "CHANNELWISE_MAXPOOL":
                output, self.conv_ratio, self.cp_identity_matrix = channel_wise_maxpool(t_flex_pool, t_flex_conv)
            case False:
                logits = self.get_logits(x, t_flex_pool)
                mask = self.masking(logits)

                with torch.no_grad():
                    self.cp_identity_matrix = mask

                output = (t_flex_pool * (1 - mask)) + (t_flex_conv * mask)
            case _:
                raise NotImplementedError

        return output

    def get_logits(self, x, t_flex_pool):
        match self.config.logits_mechanism:
            case "THRESHOLD":
                logits = t_flex_pool - self.threshold
            case "SpatialAttentionBlock":
                logits = self.spatial_attention_block(x)
            case _:
                raise ValueError(f"Unknown logits mechanism: {self.config.logits_mechanism}")

        if self.config.logits_use_batchnorm:
            logits = self.bn_logits(logits)

        logits = self.activation(logits)  # ✅ 新增：应用脉冲或普通激活

        return logits

    def masking(self, logits):
        match self.config.masking_mechanism:
            case "SIGMOID_MUL":
                mask = sigmoid_plain(logits * self.config.get("sigmoid_mul_factor"))
            case "SIGMOID_HARD":
                mask = HardsigmoidFunc.apply(logits)
            case _:
                pass

        if "StochasticRound" in self.config.masking_mechanism:
            mask = getattr(masks, self.config.masking_mechanism).apply(logits)

        if "STE" in self.config.masking_mechanism:
            mask = getattr(masks, self.config.masking_mechanism).apply(logits)

        self._monitor_mask(mask)
        return mask

    def _monitor_mask(self, mask):
        with torch.no_grad():
            self.homogeneity = measure_homogeneity(mask)
            self.conv_ratio = count_conv_ratio(mask)
            if self.config.get("joint_mechanism", False) == "CHANNELWISE_MAXPOOL":
                self.homogeneity = 1


if __name__ == "__main__":
    pass
