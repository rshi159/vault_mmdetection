"""
Custom transform to generate heatmap channel for 4-channel input.
Integrates with existing heatmap_generator.py for conveyor detection.
"""

import numpy as np
import cv2
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS
from pathlib import Path
import sys

# Add development directory to find heatmap_generator
dev_dir = Path(__file__).parent.parent.parent / 'development'
sys.path.insert(0, str(dev_dir))

try:
    from heatmap_generator import ConveyorHeatmapGenerator
except ImportError:
    print("Warning: ConveyorHeatmapGenerator not found. Using dummy heatmap generation.")
    ConveyorHeatmapGenerator = None


@TRANSFORMS.register_module()
class GenerateHeatmapChannel(BaseTransform):
    """
    Transform to generate heatmap channel and create 4-channel input.
    
    This transform:
    1. Takes RGB image (3 channels)
    2. Generates heatmap using ConveyorHeatmapGenerator
    3. Combines into 4-channel image (RGB + Heatmap)
    
    Args:
        method (str): Heatmap generation method. Default: 'prior_based'
        heatmap_strength (float): Strength of heatmap signal. Default: 1.0
        normalize_heatmap (bool): Whether to normalize heatmap to [0,1]. Default: True
    """
    
    def __init__(self, 
                 method: str = 'prior_based',
                 heatmap_strength: float = 1.0,
                 normalize_heatmap: bool = True):
        self.method = method
        self.heatmap_strength = heatmap_strength
        self.normalize_heatmap = normalize_heatmap
        
        # Initialize heatmap generator
        if ConveyorHeatmapGenerator is not None:
            self.heatmap_generator = ConveyorHeatmapGenerator()
        else:
            self.heatmap_generator = None
            print("Using dummy heatmap generation")
    
    def transform(self, results: dict) -> dict:
        """
        Generate heatmap and create 4-channel image.
        
        Args:
            results (dict): Result dict containing 'img' key with RGB image
            
        Returns:
            dict: Updated results with 4-channel image
        """
        img = results['img']  # RGB image (H, W, 3)
        h, w = img.shape[:2]
        
        # Generate heatmap
        if self.heatmap_generator is not None:
            # Use real heatmap generator
            heatmap = self._generate_real_heatmap(img)
        else:
            # Use dummy heatmap for testing
            heatmap = self._generate_dummy_heatmap(h, w)
        
        # Ensure heatmap is right shape and type
        if len(heatmap.shape) == 2:
            heatmap = heatmap[..., np.newaxis]  # Add channel dimension
        
        # Normalize heatmap if requested
        if self.normalize_heatmap:
            heatmap = heatmap.astype(np.float32)
            if heatmap.max() > heatmap.min():
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        # Apply strength scaling
        heatmap = heatmap * self.heatmap_strength
        
        # Combine RGB + Heatmap
        img_4ch = np.concatenate([img, heatmap], axis=2)  # (H, W, 4)
        
        # Update results
        results['img'] = img_4ch
        results['img_shape'] = img_4ch.shape
        
        return results
    
    def _generate_real_heatmap(self, img: np.ndarray) -> np.ndarray:
        """Generate real heatmap using ConveyorHeatmapGenerator."""
        try:
            # Convert BGR to RGB if needed (MMDet uses BGR by default)
            if img.shape[2] == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img
            
            # Generate heatmap
            heatmap = self.heatmap_generator.generate_conveyor_heatmap(
                img_rgb, method=self.method
            )
            
            return heatmap.astype(np.float32)
            
        except Exception as e:
            print(f"Error generating real heatmap: {e}")
            return self._generate_dummy_heatmap(img.shape[0], img.shape[1])
    
    def _generate_dummy_heatmap(self, height: int, width: int) -> np.ndarray:
        """Generate dummy heatmap for testing when real generator unavailable."""
        # Create simple center-focused heatmap
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        
        # Distance from center
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        # Create heatmap with higher values near center
        heatmap = 1.0 - (dist / max_dist)
        heatmap = np.clip(heatmap, 0, 1)
        
        return heatmap.astype(np.float32)
    
    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'method={self.method}, '
                f'heatmap_strength={self.heatmap_strength}, '
                f'normalize_heatmap={self.normalize_heatmap})')


@TRANSFORMS.register_module() 
class Pad4Channel(BaseTransform):
    """
    Custom padding for 4-channel images.
    Extends regular padding to handle the 4th channel properly.
    """
    
    def __init__(self, size, pad_val=dict(img=(114, 114, 114, 0))):
        self.size = size
        self.pad_val = pad_val
    
    def transform(self, results):
        """Pad 4-channel image."""
        img = results['img']
        
        if len(img.shape) == 3 and img.shape[2] == 4:
            # Handle 4-channel image
            h, w = img.shape[:2]
            target_h, target_w = self.size
            
            # Calculate padding
            pad_h = max(0, target_h - h)
            pad_w = max(0, target_w - w)
            
            if pad_h > 0 or pad_w > 0:
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                
                # Get pad values
                if isinstance(self.pad_val['img'], (list, tuple)) and len(self.pad_val['img']) == 4:
                    pad_values = self.pad_val['img']
                else:
                    pad_values = (114, 114, 114, 0)  # Default RGBH padding
                
                # Pad each channel
                padded_img = np.zeros((target_h, target_w, 4), dtype=img.dtype)
                
                # Fill with padding values
                for c in range(4):
                    padded_img[:, :, c] = pad_values[c]
                
                # Place original image
                padded_img[pad_top:pad_top+h, pad_left:pad_left+w] = img
                
                results['img'] = padded_img
                results['img_shape'] = padded_img.shape
        
        return results