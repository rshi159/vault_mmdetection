
import torch
import torch.nn as nn
from typing import List, Optional, Union
import numpy as np

class DetDataPreprocessor4Ch(nn.Module):
    """
    4-channel data preprocessor for RGB + Heatmap input.
    Handles RGBH input by processing each channel appropriately.
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
        """
        Process 4-channel input data.

        Args:
            data (torch.Tensor): Input tensor of shape (B, 4, H, W)

        Returns:
            torch.Tensor: Normalized 4-channel tensor
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

    def __repr__(self):
        return f"DetDataPreprocessor4Ch(mean={self.mean.squeeze().tolist()}, " \
               f"std={self.std.squeeze().tolist()}, bgr_to_rgb={self.bgr_to_rgb})"
