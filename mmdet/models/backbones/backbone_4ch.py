
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import math

from mmdet.registry import MODELS

class BasicBlock(nn.Module):
    """Basic building block for CSP backbone."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class CSPBlock(nn.Module):
    """CSP Block for feature extraction."""

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 1):
        super().__init__()
        mid_channels = out_channels // 2

        self.main_conv = BasicBlock(in_channels, mid_channels, 1)
        self.short_conv = BasicBlock(in_channels, mid_channels, 1)

        self.blocks = nn.Sequential(*[
            BasicBlock(mid_channels, mid_channels) for _ in range(num_blocks)
        ])

        self.final_conv = BasicBlock(out_channels, out_channels, 1)

    def forward(self, x):
        main = self.main_conv(x)
        main = self.blocks(main)

        short = self.short_conv(x)

        out = torch.cat([main, short], dim=1)
        return self.final_conv(out)

@MODELS.register_module()
class CSPNeXt4Ch(nn.Module):
    """
    4-channel CSPNeXt backbone for RTMDet.
    Designed to process RGB + Heatmap input.
    """

    def __init__(self, 
                 arch: str = 'P5',
                 widen_factor: float = 0.375,
                 deepen_factor: float = 0.167,
                 out_indices: Tuple[int] = (2, 3, 4),
                 norm_cfg: Dict = None,
                 act_cfg: Dict = None,
                 init_cfg: Dict = None,
                 **kwargs):  # Accept any other kwargs to be compatible
        super().__init__()

        self.arch = arch
        self.widen_factor = widen_factor
        self.deepen_factor = deepen_factor
        self.out_indices = out_indices

        # Calculate channel numbers
        base_channels = int(256 * widen_factor)  # ~96 for tiny
        channels = [base_channels * (2 ** i) for i in range(5)]  # [96, 192, 384, 768, 1536]

        # Calculate block numbers
        base_blocks = [3, 6, 6, 3]
        blocks = [max(1, int(b * deepen_factor)) for b in base_blocks]  # [1, 1, 1, 1] for tiny

        # Stem: 4-channel input -> base_channels
        self.stem = BasicBlock(4, channels[0] // 2, 3)  # 4 -> 48
        self.stem2 = BasicBlock(channels[0] // 2, channels[0], 3)  # 48 -> 96

        # Stages
        self.stages = nn.ModuleList()
        in_ch = channels[0]

        for i, (out_ch, num_blocks) in enumerate(zip(channels[1:], blocks)):
            # Downsample with stride=2
            downsample = BasicBlock(in_ch, out_ch, 3, stride=2)
            # CSP Block
            csp_block = CSPBlock(out_ch, out_ch, num_blocks)

            stage = nn.Sequential(downsample, csp_block)
            self.stages.append(stage)
            in_ch = out_ch

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 4, H, W)

        Returns:
            Tuple of feature tensors at different scales
        """
        if x.shape[1] != 4:
            raise ValueError(f"Expected 4-channel input, got {x.shape[1]} channels")

        # Stem
        x = self.stem(x)
        x = self.stem2(x)

        # Collect outputs
        outputs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                outputs.append(x)

        return tuple(outputs)

    def __repr__(self):
        return f"CSPNeXt4Ch(arch={self.arch}, widen_factor={self.widen_factor}, " \
               f"deepen_factor={self.deepen_factor})"

def build_4ch_backbone(**kwargs):
    """Factory function to build 4-channel backbone."""
    return CSPNeXt4Ch(**kwargs)
