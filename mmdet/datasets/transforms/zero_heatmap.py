"""
Zero Heatmap Transform for RGB-only training.

This transform forces the 4th channel (heatmap) to all zeros during Stage 1 training,
allowing the model to learn strong RGB visual features without heatmap dependency.
"""

import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class ZeroHeatmapTransform(BaseTransform):
    """Transform that forces the heatmap channel (4th channel) to zeros.
    
    This is used in Stage 1 training to ensure the model learns RGB features
    without relying on heatmap information.
    
    Args:
        enforce_4ch (bool): Whether to enforce that input has exactly 4 channels.
            Defaults to True for safety.
    """
    
    def __init__(self, enforce_4ch: bool = True):
        self.enforce_4ch = enforce_4ch
        
    def transform(self, results: dict) -> dict:
        """Zero out the heatmap channel (4th channel) for RGB-only training.
        
        Args:
            results (dict): Result dict from the data pipeline.
            
        Returns:
            dict: Modified result dict with zeroed heatmap channel.
        """
        img = results['img']
        
        # Validate input has 4 channels
        if img.shape[2] != 4:
            raise ValueError(f"Expected 4-channel image, got {img.shape[2]} channels")
        
        # Zero out the heatmap channel (4th channel, index 3)
        img[:, :, 3] = 0.0
        
        return results
        
    def __repr__(self):
        return f"{self.__class__.__name__}(enforce_4ch={self.enforce_4ch})"