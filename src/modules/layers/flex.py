
#!/Users/donyin/miniconda3/envs/imperial/bin/python
"""
This is the core module of the entire programme, Flex2D. Named so we can have Flex3D, Flex4D, etc. in the future.

IMPORTANT:
Two essential methods of the flex2d class:
    1. masking:
        This method applies a specified masking mechanism on the input logits to generate a binary mask. The mechanism applied depends on the masking_mechanism attribute of the Flex2D instance.
        Args: logits (torch.Tensor): The input logits tensor to be masked.
        Returns: torch.Tensor: The masked tensor, where the mask has been applied to the logits.

    2. get_logits:
        This method computes the logits used to create the binary mask during the forward pass. The method for computing logits is determined by the logits_mechanism attribute of the Flex2D instance.
        Args:
            t_flex_pool (torch.Tensor): The tensor obtained after applying max pooling to the input tensor.
            t_flex_conv (torch.Tensor): The tensor obtained after applying convolution to the input tensor.
        Returns:
            torch.Tensor: The logits tensor which will be passed to a masking function to create a binary mask.
"""

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
from src.modules.masks.sigmoid import sigmoid_smoothed, HardsigmoidFunc, sigmoid_plain


class MaxPool2d(nn.MaxPool2d):
    """this is a wrapper that returns indices for computing hessian"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # -------- figure out all the in and out dimensions --------

    def forward(self, x):
        self.output, self.indices = F.max_pool2d(
            x, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, return_indices=True
        )

        return self.output


class Flex2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, config=None):
        super().__init__()
        """
        # in dimensions: in this case [C, H, W]
        # --------
        # logits_mechanism: THRESHOLD or SPATIAL_ATTENTION_(1-3)
        # masking_mechanism: "SIGMOID", "STE", "SIGMOID_SMOOTHED", "STE_SIGMOID"
        # num_spatial_attention_block: int
        # logits_use_batchnorm: bool

        # about parameter vs variable:
        variable is almost deprecated and works the same as just plain tensor. And a Parameters is a specific Tensor that is marked as being a parameter from an nn.Module and so will be returned when calling .parameters() on this Module.
        """
        # -------- set configs --------
        assert config, "Missing config file for Flex2D"
        self.config = config
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.device = select_device()

        # -------- Initialize layers --------
        self.flex_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.flex_pool = MaxPool2d(kernel_size, stride, padding)
        self.bn_logits = nn.BatchNorm2d(self.out_channels)

        # -------- Initialize monitored variables --------
        self.homogeneity = 0  # for monitoring the binariness of the mask later on
        self.conv_ratio = 0  # for later updating
        self.cp_identity_matrix = None  # store the matrix indicating the channel pool identity

        # -------- initialize logits related modules --------
        if "SpatialAttention" in self.config.get("logits_mechanism", ""):
            self.spatial_attention_block = getattr(logits, self.config.logits_mechanism)(
                num_blocks=self.config.num_spatial_attention_block,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
            )

    def init_dimension_dependent_modules(self):
        """This is initialized before the actual running, like init"""
        # -------- Initialize threshold --------

        assert hasattr(self, "out_dimensions"), "out_dimensions must be specified before initializing threshold"
        self.threshold = nn.Parameter(torch.randn(*self.out_dimensions)).to(self.device)
        nn.init.kaiming_uniform_(self.threshold)

    def forward(self, x):
        t_flex_pool = self.flex_pool(x)  # 只调用一次
        t_flex_conv = self.flex_conv(x)
        t_flex_pool = channel_interpolate(t_flex_pool, self.out_channels)

        # 初始化 threshold
        if not hasattr(self, "threshold") or self.threshold.shape != t_flex_pool.shape:
            with torch.no_grad():
                self.threshold = nn.Parameter(torch.randn(1, *t_flex_pool.shape[1:]).to(self.device))
                nn.init.kaiming_uniform_(self.threshold)

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
        """
        x: the input tensor
        t_flex_pool: the input tensor after max pooling
        t_flex_conv: the input tensor after convolution
        """
        # -------- calculate logits --------
        match self.config.logits_mechanism:
            case "THRESHOLD":
                # print("t_flex_pool", t_flex_pool.shape, "threshold", self.threshold.shape)

                # logits = t_flex_pool - self.threshold
                logits = t_flex_pool - self.threshold 
            case "SpatialAttentionBlock":
                logits = self.spatial_attention_block(x)
            case _:
                raise ValueError(f"Unknown logits mechanism: {self.config.logits_mechanism}")

        # -------- apply batchnorm --------
        if self.config.logits_use_batchnorm:
            logits = self.bn_logits(logits)  # this solves the SMOOTHED-SIGMOID-not-learning-at-all problem

        return logits

    def masking(self, logits):
        match self.config.masking_mechanism:
            case "SIGMOID_MUL":
                mask = sigmoid_plain(logits * self.config.get("sigmoid_mul_factor"))  # already improved version
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
        with torch.no_grad():  # This block won't be part of the computation graph
            self.homogeneity = measure_homogeneity(mask)
            self.conv_ratio = count_conv_ratio(mask)
            self.homogeneity = 1 if self.config.get("joint_mechanism", False) == "CHANNELWISE_MAXPOOL" else self.homogeneity


if __name__ == "__main__":
    """The testing of the module is at tests/"""
    pass