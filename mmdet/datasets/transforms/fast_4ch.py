"""
Fast 4-Channel Padding for RGB-Only Training.

This module provides optimized transforms for RGB-only training that bypass
expensive heatmap generation entirely.
"""

import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class FastPad4Channel(BaseTransform):
    """
    Ultra-fast 4-channel padding for RGB-only training.
    
    This transform directly adds a zero-filled 4th channel without any
    heatmap generation, making it ideal for RGB foundation training.
    
    This is much faster than:
    1. RobustHeatmapGeneration -> Pad4Channel -> ZeroHeatmapTransform
    2. Even Pad4Channel -> ZeroHeatmapTransform
    
    Since we're doing RGB-only training, we can skip all heatmap processing.
    
    Args:
        pad_value (float): Value to fill the 4th channel. Defaults to 0.0.
    """
    
    def __init__(self, pad_value: float = 0.0):
        self.pad_value = pad_value
        
    def transform(self, results: dict) -> dict:
        """Add zero-filled 4th channel directly to RGB image.
        
        Args:
            results (dict): Result dict from the data pipeline.
            
        Returns:
            dict: Modified result dict with 4-channel image.
        """
        img = results['img']
        
        # Validate input is 3-channel RGB
        if img.shape[2] != 3:
            raise ValueError(f"Expected 3-channel RGB image, got {img.shape[2]} channels")
        
        height, width = img.shape[:2]
        
        # Create 4th channel filled with pad_value
        fourth_channel = np.full((height, width, 1), self.pad_value, dtype=img.dtype)
        
        # Concatenate RGB + zero channel
        img_4ch = np.concatenate([img, fourth_channel], axis=2)
        
        # Update results
        results['img'] = img_4ch
        
        return results
        
    def __repr__(self):
        return f"{self.__class__.__name__}(pad_value={self.pad_value})"


@TRANSFORMS.register_module() 
class RGBOnly4Channel(BaseTransform):
    """
    Single transform that handles RGB-only 4-channel setup.
    
    This combines FastPad4Channel functionality and ensures the 4th channel
    stays zeroed throughout the pipeline. This is the most efficient approach
    for RGB-only training.
    
    Replaces the pipeline:
    - Pad4Channel -> ZeroHeatmapTransform
    
    With a single optimized transform.
    """
    
    def __init__(self):
        pass
        
    def transform(self, results: dict) -> dict:
        """Convert RGB to RGB+Zero in one optimized step.
        
        Args:
            results (dict): Result dict from the data pipeline.
            
        Returns:
            dict: Modified result dict with 4-channel image (RGB+0).
        """
        img = results['img']
        
        # Handle both 3-channel and 4-channel inputs
        if img.shape[2] == 3:
            # Add zero 4th channel to RGB
            height, width = img.shape[:2]
            fourth_channel = np.zeros((height, width, 1), dtype=img.dtype)
            img = np.concatenate([img, fourth_channel], axis=2)
        elif img.shape[2] == 4:
            # Force 4th channel to zero (in case it was already 4-channel)
            img[:, :, 3] = 0.0
        else:
            raise ValueError(f"Expected 3 or 4 channels, got {img.shape[2]} channels")
        
        # Update results
        results['img'] = img
        
        return results
        
    def __repr__(self):
        return f"{self.__class__.__name__}()"