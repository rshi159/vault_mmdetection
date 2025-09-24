"""
Robust heatmap generation for training that avoids overfitting to perfect priors.
Generates realistic, noisy heatmaps with deliberate uncertainty and errors.
"""

import numpy as np
import cv2
import random
from typing import Dict, List, Tuple, Optional
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RobustHeatmapGeneration(BaseTransform):
    """
    Generate robust, noisy heatmaps for training that avoid overfitting.
    
    Key features:
    - Adds noise to center points and keypoints
    - Introduces deliberate errors in some heatmaps
    - Variable heatmap quality during training
    - Uses realistic prior knowledge, not ground truth
    
    Args:
        noise_ratio (float): Probability of adding noise to heatmaps (0.0-1.0)
        error_ratio (float): Probability of deliberate errors (0.0-0.3)
        center_noise_std (float): Standard deviation for center point noise (pixels)
        keypoint_noise_std (float): Standard deviation for keypoint noise (pixels)
        quality_variance (bool): Whether to vary heatmap quality randomly
        min_sigma (float): Minimum Gaussian sigma for heatmap generation
        max_sigma (float): Maximum Gaussian sigma for heatmap generation
        no_heatmap_ratio (float): Fraction of images with NO heatmap (pure RGB training)
    """
    
    def __init__(self,
                 noise_ratio: float = 0.7,
                 error_ratio: float = 0.15,
                 center_noise_std: float = 8.0,
                 keypoint_noise_std: float = 12.0,
                 quality_variance: bool = True,
                 min_sigma: float = 10.0,
                 max_sigma: float = 40.0,
                 no_heatmap_ratio: float = 0.25,
                 partial_heatmap_ratio: float = 0.15,  # NEW: Some parcels missing heatmaps
                 global_noise_ratio: float = 0.0,
                 global_noise_std: float = 0.05,
                 multiplicative_noise_ratio: float = 0.0,
                 multiplicative_noise_range: tuple = (0.8, 1.2),
                 background_noise_std: float = 0.0):
        """
        Args:
            noise_ratio: Probability of adding positional noise to centers/keypoints
            error_ratio: Probability of adding deliberate errors
            center_noise_std: Standard deviation for center position noise
            keypoint_noise_std: Standard deviation for keypoint position noise  
            quality_variance: Whether to vary heatmap quality
            min_sigma: Minimum sigma for Gaussian heatmaps
            max_sigma: Maximum sigma for Gaussian heatmaps
            no_heatmap_ratio: Probability of generating zero heatmap (pure RGB)
            partial_heatmap_ratio: Probability of missing heatmaps for some parcels
            global_noise_ratio: Probability of adding global noise to entire heatmap
            global_noise_std: Standard deviation for global noise
            multiplicative_noise_ratio: Probability of adding multiplicative noise
            multiplicative_noise_range: Range for multiplicative noise factor (min, max)
            background_noise_std: Standard deviation for background noise (0 = no noise)
        """
        
        self.noise_ratio = noise_ratio
        self.error_ratio = error_ratio
        self.center_noise_std = center_noise_std
        self.keypoint_noise_std = keypoint_noise_std
        self.quality_variance = quality_variance
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.no_heatmap_ratio = no_heatmap_ratio
        self.partial_heatmap_ratio = partial_heatmap_ratio  # NEW
        self.global_noise_ratio = global_noise_ratio
        self.global_noise_std = global_noise_std
        self.multiplicative_noise_ratio = multiplicative_noise_ratio
        self.multiplicative_noise_range = multiplicative_noise_range
        self.background_noise_std = background_noise_std
    
    def transform(self, results: dict) -> dict:
        """Generate robust heatmap and create 4-channel input."""
        img = results['img']
        h, w = img.shape[:2]
        
        # CRITICAL: Some images get NO heatmap for pure RGB training
        if random.random() < self.no_heatmap_ratio:
            # Zero heatmap - forces model to rely on RGB only
            heatmap = np.zeros((h, w, 1), dtype=np.float32)
        else:
            # Get ground truth for heatmap generation (but add noise!)
            gt_bboxes = results.get('gt_bboxes', [])
            
            # NEW: Support partial heatmaps - some parcels missing
            if random.random() < self.partial_heatmap_ratio and len(gt_bboxes) > 1:
                # Randomly skip some bboxes to simulate missing heatmaps
                num_to_keep = max(1, random.randint(1, len(gt_bboxes) - 1))
                indices = random.sample(range(len(gt_bboxes)), num_to_keep)
                partial_bboxes = [gt_bboxes[i] for i in indices]
                heatmap = self._generate_noisy_heatmap(h, w, partial_bboxes)
            else:
                # Generate heatmap with deliberate imperfections
                heatmap = self._generate_noisy_heatmap(h, w, gt_bboxes)
            
            # Ensure correct shape
            if len(heatmap.shape) == 2:
                heatmap = heatmap[..., np.newaxis]
        
        # Combine RGB + Heatmap (ensure consistent float32 dtype)
        img = img.astype(np.float32)  # Ensure RGB is float32
        heatmap = heatmap.astype(np.float32)  # Ensure heatmap is float32
        img_4ch = np.concatenate([img, heatmap], axis=2)
        
        # Update results
        results['img'] = img_4ch
        results['img_shape'] = img_4ch.shape
        
        return results
    
    def _generate_noisy_heatmap(self, height, width, gt_bboxes):
        """Generate a noisy heatmap with intentional imperfections"""
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        if len(gt_bboxes) == 0:
            return self._generate_background_prior(height, width)
        
        for bbox in gt_bboxes:
            # Get the tensor data from HorizontalBoxes object and ensure it's on CPU
            bbox_tensor = bbox.tensor.squeeze(0) if bbox.tensor.dim() > 1 else bbox.tensor
            # Move to CPU if it's on GPU
            bbox_tensor = bbox_tensor.cpu() if bbox_tensor.is_cuda else bbox_tensor
            
            # These are already in xyxy format (x1, y1, x2, y2)
            x1, y1, x2, y2 = bbox_tensor
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            # Convert to numpy for compatibility with meshgrid
            cx = cx.item() if hasattr(cx, 'item') else float(cx)
            cy = cy.item() if hasattr(cy, 'item') else float(cy)
            
            # Add noise to center with some probability
            if random.random() < self.noise_ratio:
                cx += np.random.normal(0, self.center_noise_std)
                cy += np.random.normal(0, self.center_noise_std)
            
            # Clamp to image bounds
            cx = np.clip(cx, 0, width - 1)
            cy = np.clip(cy, 0, height - 1)
            
            # Variable sigma for different heatmap qualities
            if self.quality_variance:
                sigma = np.random.uniform(self.min_sigma, self.max_sigma)
            else:
                sigma = (self.min_sigma + self.max_sigma) / 2
            
            # Add deliberate errors occasionally
            if random.random() < self.error_ratio:
                # Introduce systematic errors
                error_type = random.choice(['offset', 'multi_peak', 'weak_signal'])
                heatmap_part = self._generate_error_heatmap(
                    height, width, cx, cy, sigma, error_type
                )
            else:
                # Generate normal noisy heatmap
                heatmap_part = self._generate_gaussian_heatmap(
                    height, width, cx, cy, sigma
                )
            
            # Accumulate heatmaps
            heatmap = np.maximum(heatmap, heatmap_part)
        
        # Add global noise to entire heatmap (configurable)
        if self.global_noise_ratio > 0 and random.random() < self.global_noise_ratio:
            noise = np.random.normal(0, self.global_noise_std, (height, width))
            heatmap = np.clip(heatmap + noise, 0, 1)
        
        return heatmap
    
    def _generate_gaussian_heatmap(self, h: int, w: int, cx: float, cy: float, sigma: float) -> np.ndarray:
        """Generate Gaussian heatmap with added noise."""
        y, x = np.ogrid[:h, :w]
        
        # Distance from center
        dist_sq = (x - cx)**2 + (y - cy)**2
        
        # Gaussian with configurable noise
        heatmap = np.exp(-dist_sq / (2 * sigma**2))
        
        # Add multiplicative noise to make it realistic (configurable)
        if self.multiplicative_noise_ratio > 0 and random.random() < self.multiplicative_noise_ratio:
            noise_min, noise_max = self.multiplicative_noise_range
            noise_factor = np.random.uniform(noise_min, noise_max)
            heatmap *= noise_factor
        
        return np.clip(heatmap, 0, 1).astype(np.float32)
    
    def _generate_error_heatmap(self, h: int, w: int, cx: float, cy: float, 
                               sigma: float, error_type: str) -> np.ndarray:
        """Generate deliberately imperfect heatmaps."""
        
        if error_type == 'offset':
            # Offset the center significantly
            offset_x = np.random.normal(0, 20)
            offset_y = np.random.normal(0, 20)
            new_cx = np.clip(cx + offset_x, 0, w - 1)
            new_cy = np.clip(cy + offset_y, 0, h - 1)
            return self._generate_gaussian_heatmap(h, w, new_cx, new_cy, sigma)
            
        elif error_type == 'multi_peak':
            # Multiple peaks (false positives)
            heatmap = self._generate_gaussian_heatmap(h, w, cx, cy, sigma)
            
            # Add 1-2 additional peaks
            num_extra = random.randint(1, 2)
            for _ in range(num_extra):
                extra_cx = np.random.uniform(0, w - 1)
                extra_cy = np.random.uniform(0, h - 1)
                extra_sigma = sigma * random.uniform(0.5, 1.5)
                extra_peak = self._generate_gaussian_heatmap(h, w, extra_cx, extra_cy, extra_sigma)
                extra_peak *= random.uniform(0.3, 0.7)  # Weaker peaks
                heatmap = np.maximum(heatmap, extra_peak)
            
            return heatmap
            
        elif error_type == 'weak_signal':
            # Very weak or diffuse signal
            weak_sigma = sigma * random.uniform(2.0, 4.0)  # Much larger
            heatmap = self._generate_gaussian_heatmap(h, w, cx, cy, weak_sigma)
            heatmap *= random.uniform(0.2, 0.5)  # Much weaker
            return heatmap
        
        else:
            return self._generate_gaussian_heatmap(h, w, cx, cy, sigma)
    
    def _generate_background_prior(self, height: int, width: int) -> np.ndarray:
        """Generate background prior when no objects present."""
        # Subtle center bias (conveyor belt center)
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        
        # Very weak center preference
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        prior = 0.05 * (1.0 - dist / max_dist)  # Very weak signal
        
        # Add configurable noise
        if self.background_noise_std > 0:
            noise = np.random.normal(0, self.background_noise_std, (height, width))
            prior = np.clip(prior + noise, 0, 0.1)  # Keep background low
        
        return prior.astype(np.float32)
    
    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'noise_ratio={self.noise_ratio}, '
                f'error_ratio={self.error_ratio}, '
                f'no_heatmap_ratio={self.no_heatmap_ratio}, '
                f'center_noise_std={self.center_noise_std})')


@TRANSFORMS.register_module()
class HeatmapQualityScheduler(BaseTransform):
    """
    Gradually improve heatmap quality during training to avoid overfitting.
    Early training: Very noisy heatmaps
    Late training: Higher quality heatmaps
    """
    
    def __init__(self, training_progress: float = 0.0):
        self.training_progress = training_progress  # 0.0 to 1.0
    
    def set_training_progress(self, progress: float):
        """Update training progress (called by training hook)."""
        self.training_progress = np.clip(progress, 0.0, 1.0)
    
    def transform(self, results: dict) -> dict:
        """Modify heatmap quality based on training progress."""
        # Get current heatmap
        img = results['img']
        if img.shape[2] != 4:
            return results
            
        heatmap = img[:, :, 3]  # 4th channel
        
        # Early training: add more noise
        # Late training: reduce noise
        noise_level = 0.1 * (1.0 - self.training_progress)  # Decreases over time
        
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, heatmap.shape)
            heatmap = np.clip(heatmap + noise, 0, 1)
            
            # Update image
            img[:, :, 3] = heatmap
            results['img'] = img
        
        return results