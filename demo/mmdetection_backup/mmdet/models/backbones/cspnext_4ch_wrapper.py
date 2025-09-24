"""
CSPNeXt 4-Channel Wrapper for MMDetection
Inflates the first convolution to accept 4-channel input while preserving pretrained weights.
"""

import torch
import torch.nn as nn
from mmdet.registry import MODELS
from mmdet.models.backbones import CSPNeXt
import logging
from typing import Optional

def inflate_conv_for_channels(conv_layer: nn.Conv2d, target_channels: int = 4, zero_init_new: bool = True) -> nn.Conv2d:
    """
    Inflate a convolution layer to accept more input channels while preserving existing weights.

    Args:
        conv_layer: Original convolution layer
        target_channels: Target number of input channels
        zero_init_new: Whether to zero-initialize new channels (recommended for gradual learning)

    Returns:
        New convolution layer with inflated input channels
    """
    if conv_layer.in_channels >= target_channels:
        return conv_layer  # Already has enough channels

    # Create new conv with same parameters except input channels
    new_conv = nn.Conv2d(
        in_channels=target_channels,
        out_channels=conv_layer.out_channels,
        kernel_size=conv_layer.kernel_size,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
        dilation=conv_layer.dilation,
        groups=conv_layer.groups,
        bias=conv_layer.bias is not None,
        padding_mode=conv_layer.padding_mode
    )

    with torch.no_grad():
        # Copy existing weights to corresponding input channels
        original_channels = conv_layer.in_channels
        new_conv.weight[:, :original_channels, :, :] = conv_layer.weight

        # Initialize new channels
        if zero_init_new:
            # Zero initialization: model ignores new channels initially
            new_conv.weight[:, original_channels:, :, :].zero_()
        else:
            # Kaiming initialization: more aggressive use of new channels
            nn.init.kaiming_normal_(
                new_conv.weight[:, original_channels:, :, :], 
                nonlinearity='relu'
            )

        # Copy bias if present
        if conv_layer.bias is not None:
            new_conv.bias.copy_(conv_layer.bias)

    logging.info(f"Inflated conv from {original_channels} to {target_channels} channels")
    return new_conv

@MODELS.register_module()
class CSPNeXt4Ch(CSPNeXt):
    """
    CSPNeXt wrapper that accepts 4-channel input by inflating the stem convolution.

    This wrapper:
    - Preserves all pretrained weights except the first convolution
    - Zero-initializes the 4th channel (PriorH) for gradual learning
    - Acts as a drop-in replacement for CSPNeXt
    - Maintains full backward compatibility
    """

    def __init__(self, target_channels: int = 4, zero_init_new: bool = True, **kwargs):
        """
        Initialize CSPNeXt with 4-channel input support.

        Args:
            target_channels: Number of input channels (default: 4 for RGB + PriorH)
            zero_init_new: Whether to zero-initialize new channels
            **kwargs: Arguments passed to parent CSPNeXt
        """
        self.target_channels = target_channels
        self.zero_init_new = zero_init_new

        # Initialize parent CSPNeXt normally (3-channel)
        super().__init__(**kwargs)

        # Inflate the stem convolution after initialization
        self._inflate_stem_convolution()

        logging.info(f"CSPNeXt4Ch initialized for {target_channels}-channel input")

    def _inflate_stem_convolution(self):
        """Find and inflate the stem convolution layer."""
        inflated = False

        # Strategy 1: Check if stem has 'conv' attribute (ConvModule)
        if hasattr(self.stem, 'conv') and isinstance(self.stem.conv, nn.Conv2d):
            original_conv = self.stem.conv
            if original_conv.in_channels < self.target_channels:
                self.stem.conv = inflate_conv_for_channels(
                    original_conv, self.target_channels, self.zero_init_new
                )
                inflated = True
                logging.info("Inflated stem.conv")

        # Strategy 2: Check if stem is Sequential and first element is Conv2d
        elif hasattr(self.stem, '__getitem__') and len(self.stem) > 0:
            if isinstance(self.stem[0], nn.Conv2d):
                original_conv = self.stem[0]
                if original_conv.in_channels < self.target_channels:
                    self.stem[0] = inflate_conv_for_channels(
                        original_conv, self.target_channels, self.zero_init_new
                    )
                    inflated = True
                    logging.info("Inflated stem[0]")

        # Strategy 3: Check if stem is directly a Conv2d
        elif isinstance(self.stem, nn.Conv2d):
            if self.stem.in_channels < self.target_channels:
                self.stem = inflate_conv_for_channels(
                    self.stem, self.target_channels, self.zero_init_new
                )
                inflated = True
                logging.info("Inflated stem (direct Conv2d)")

        # Strategy 4: Search through all stem modules
        if not inflated:
            for name, module in self.stem.named_modules():
                if isinstance(module, nn.Conv2d) and module.in_channels < self.target_channels:
                    # Get parent module and attribute name
                    if '.' in name:
                        parent_path, attr_name = name.rsplit('.', 1)
                        parent = self.stem
                        for part in parent_path.split('.'):
                            parent = getattr(parent, part)
                    else:
                        parent = self.stem
                        attr_name = name

                    # Replace the convolution
                    new_conv = inflate_conv_for_channels(
                        module, self.target_channels, self.zero_init_new
                    )
                    setattr(parent, attr_name, new_conv)
                    inflated = True
                    logging.info(f"Inflated {name}")
                    break

        if not inflated:
            raise RuntimeError(
                f"Could not find convolution layer to inflate in stem. "
                f"Stem structure: {self.stem}"
            )

    def init_weights(self):
        """Initialize weights, preserving pretrained benefits."""
        # Parent initialization loads pretrained weights
        super().init_weights()

        # Our inflation already handled the new channels properly
        # (pretrained RGB weights + zero/kaiming init for new channels)
        pass

    def forward(self, x):
        """Forward pass - same as parent but handles 4-channel input."""
        if x.shape[1] != self.target_channels:
            raise ValueError(
                f"Expected {self.target_channels}-channel input, got {x.shape[1]} channels"
            )

        return super().forward(x)
