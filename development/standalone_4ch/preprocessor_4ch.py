
"""4-channel data preprocessor for RGB + Heatmap input.

This module provides a data preprocessor specifically designed for 4-channel
input consisting of RGB channels plus a heatmap channel. It handles proper
normalization for each channel type and supports device-aware operations.

The preprocessor applies channel-specific normalization:
    - RGB channels: Standard ImageNet statistics (adjusted for BGR)
    - Heatmap channel: Custom statistics for probability/confidence maps

Key features:
    - Channel-specific normalization (RGB vs Heatmap)
    - Device-aware tensor operations with buffers
    - Support for BGR to RGB conversion
    - Compatible with MMDetection data pipeline
    - Memory efficient batch processing

Example:
    Basic usage:
        preprocessor = DetDataPreprocessor4Ch(
            mean=[103.53, 116.28, 123.675, 0.0],  # BGR + Heatmap
            std=[57.375, 57.12, 57.375, 1.0],     # BGR + Heatmap
            bgr_to_rgb=False
        )
        
        # Process 4-channel batch
        normalized = preprocessor(batch_4ch)  # Shape: (B, 4, H, W)

Author: 4-Channel RTMDet Development Team  
Date: September 2025
"""

import torch
import torch.nn as nn
from typing import List, Optional, Union
import numpy as np

class DetDataPreprocessor4Ch(nn.Module):
    """4-channel data preprocessor for RGB + Heatmap input.
    
    This preprocessor handles RGBH input by processing each channel 
    appropriately. RGB channels use standard ImageNet normalization while
    the heatmap channel uses custom normalization for probability maps.
    
    Args:
        mean: List of mean values for each channel [R, G, B, Heatmap].
            Defaults to BGR ImageNet means plus zero for heatmap.
        std: List of standard deviation values for each channel.
            Defaults to BGR ImageNet stds plus 1.0 for heatmap.
        bgr_to_rgb: Whether to convert BGR to RGB order for first 3 channels.
            
    Attributes:
        mean (torch.Tensor): Registered buffer for mean values.
        std (torch.Tensor): Registered buffer for std values.
        bgr_to_rgb (bool): Channel reordering flag.
    """

    def __init__(self, 
                 mean: List[float] = [103.53, 116.28, 123.675, 0.0],
                 std: List[float] = [57.375, 57.12, 57.375, 1.0],
                 bgr_to_rgb: bool = False):
        super().__init__()

        # Register as buffers for proper device handling
        self.register_buffer('mean', torch.tensor(mean).view(-1, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(-1, 1, 1))
        self.bgr_to_rgb = bgr_to_rgb

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Process 4-channel input data with proper normalization.
        
        Applies channel-specific normalization to RGB + Heatmap input.
        Optionally converts BGR to RGB for the first three channels while
        preserving the heatmap channel unchanged.
        
        Args:
            data (torch.Tensor): Input tensor of shape (B, 4, H, W) where
                channels are [R/B, G, B/R, Heatmap] depending on bgr_to_rgb.
                
        Returns:
            torch.Tensor: Normalized 4-channel tensor with same shape as input.
            
        Raises:
            ValueError: If input doesn't have exactly 4 channels.
            
        Example:
            >>> preprocessor = DetDataPreprocessor4Ch()
            >>> batch = torch.randn(8, 4, 640, 640)  # Batch of 4-channel images
            >>> normalized = preprocessor(batch)
            >>> print(normalized.shape)  # torch.Size([8, 4, 640, 640])
        """
        # Ensure we have 4 channels
        if data.shape[1] != 4:
            raise ValueError(f"Expected 4 channels, got {data.shape[1]}")

        # Convert BGR to RGB if needed (only for first 3 channels)
        if self.bgr_to_rgb:
            data = data[:, [2, 1, 0, 3], :, :]  # BGR -> RGB, keep H

        # Normalize
        data = (data - self.mean) / self.std

        return data

    def __repr__(self) -> str:
        """Return string representation of the preprocessor.
        
        Returns:
            str: Human-readable representation showing configuration parameters.
        """
        return (f"DetDataPreprocessor4Ch("
                f"mean={self.mean.squeeze().tolist()}, "
                f"std={self.std.squeeze().tolist()}, "
                f"bgr_to_rgb={self.bgr_to_rgb})")
