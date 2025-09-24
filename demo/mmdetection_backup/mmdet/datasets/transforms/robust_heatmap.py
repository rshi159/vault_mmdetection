"""
Robust heatmap generation for training that avoids overfitting to perfect priors.
Generates realistic, noisy heatmaps with deliberate uncertainty and errors.
Uses YOLO keypoint data for enhanced heatmap generation.
"""

import numpy as np
import cv2
import random
import os
from typing import Dict, List, Tuple, Optional
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS

# Import the YOLO keypoint lookup system
try:
    from mmdet.datasets.yolo_keypoint_lookup import create_lookup_singleton
    KEYPOINT_LOOKUP_AVAILABLE = True
except ImportError:
    print("⚠️ YOLO keypoint lookup not available - using bbox-only heatmaps")
    KEYPOINT_LOOKUP_AVAILABLE = False


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
                 yolo_dataset_path: Optional[str] = None):
        
        self.noise_ratio = noise_ratio
        self.error_ratio = error_ratio
        self.center_noise_std = center_noise_std
        self.keypoint_noise_std = keypoint_noise_std
        self.quality_variance = quality_variance
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.no_heatmap_ratio = no_heatmap_ratio
        
        # Initialize YOLO keypoint lookup
        self.keypoint_lookup = None
        if KEYPOINT_LOOKUP_AVAILABLE and yolo_dataset_path:
            try:
                self.keypoint_lookup = create_lookup_singleton(yolo_dataset_path)
                print("✅ YOLO keypoint lookup initialized for enhanced heatmap generation")
            except Exception as e:
                print(f"⚠️ Failed to initialize keypoint lookup: {e}")
        
        # Auto-detect YOLO dataset path if not provided
        if self.keypoint_lookup is None and KEYPOINT_LOOKUP_AVAILABLE:
            # Try to find YOLO dataset path relative to current working directory
            potential_paths = [
                "development/augmented_data_production",
                "../development/augmented_data_production",
                "../../development/augmented_data_production"
            ]
            
            for path in potential_paths:
                if os.path.exists(path):
                    try:
                        self.keypoint_lookup = create_lookup_singleton(path)
                        print(f"✅ Auto-detected YOLO dataset at: {path}")
                        break
                    except Exception as e:
                        continue
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.no_heatmap_ratio = no_heatmap_ratio
    
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
            img_filename = os.path.basename(results.get('img_path', ''))
            
            # Try to get keypoint data for enhanced heatmap generation
            keypoint_data = None
            if self.keypoint_lookup and img_filename:
                keypoint_data = self.keypoint_lookup.get_keypoints(img_filename)
            
            # Generate heatmap with deliberate imperfections
            heatmap = self._generate_noisy_heatmap(h, w, gt_bboxes, keypoint_data)
            
            # Ensure correct shape
            if len(heatmap.shape) == 2:
                heatmap = heatmap[..., np.newaxis]
        
        # Combine RGB + Heatmap
        img_4ch = np.concatenate([img, heatmap], axis=2)
        
        # Update results
        results['img'] = img_4ch
        results['img_shape'] = img_4ch.shape
        
        return results
    
    def _generate_noisy_heatmap(self, height: int, width: int, gt_bboxes, keypoint_data: Optional[List[Dict]] = None) -> np.ndarray:
        """Generate heatmap with realistic noise and errors, using keypoints when available."""
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        # No objects = pure background prior
        if len(gt_bboxes) == 0:
            return self._generate_background_prior(height, width)
        
        # Use keypoint data if available (more realistic)
        if keypoint_data:
            return self._generate_keypoint_heatmap(height, width, keypoint_data)
        else:
            # Fallback to bbox-based heatmap
            return self._generate_bbox_heatmap(height, width, gt_bboxes)
    
    def _generate_keypoint_heatmap(self, height: int, width: int, keypoint_data: List[Dict]) -> np.ndarray:
        """Generate heatmap using YOLO keypoint data for enhanced realism."""
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        for annotation in keypoint_data:
            keypoints = annotation.get('keypoints_normalized', [])
            bbox_norm = annotation.get('bbox_normalized', [])
            
            if not keypoints or len(bbox_norm) < 4:
                continue
            
            # Convert normalized coordinates to pixel coordinates
            # YOLO bbox format: [x_center, y_center, width, height] (normalized)
            cx_norm, cy_norm, w_norm, h_norm = bbox_norm
            
            # Add noise to keypoints with some probability
            processed_keypoints = []
            for kp in keypoints:
                if len(kp) < 3:
                    continue
                    
                kx_norm, ky_norm, visibility = kp
                
                # Skip non-visible keypoints sometimes (realistic)
                if visibility == 0 or (visibility == 1 and random.random() < 0.3):
                    continue
                
                # Convert to pixel coordinates
                kx = kx_norm * width
                ky = ky_norm * height
                
                # Add noise with some probability
                if random.random() < self.noise_ratio:
                    kx += np.random.normal(0, self.keypoint_noise_std)
                    ky += np.random.normal(0, self.keypoint_noise_std)
                
                # Clamp to image bounds
                kx = np.clip(kx, 0, width - 1)
                ky = np.clip(ky, 0, height - 1)
                
                processed_keypoints.append((kx, ky, visibility))
            
            # Generate heatmap from keypoints
            if processed_keypoints:
                self._add_keypoint_gaussian_to_heatmap(heatmap, processed_keypoints, height, width)
            else:
                # Fallback to bbox center if no valid keypoints
                cx = cx_norm * width
                cy = cy_norm * height
                if random.random() < self.noise_ratio:
                    cx += np.random.normal(0, self.center_noise_std)
                    cy += np.random.normal(0, self.center_noise_std)
                cx = np.clip(cx, 0, width - 1)
                cy = np.clip(cy, 0, height - 1)
                self._add_gaussian_to_heatmap(heatmap, cx, cy, height, width)
        
        return heatmap
    
    def _generate_bbox_heatmap(self, height: int, width: int, gt_bboxes) -> np.ndarray:
        """Generate heatmap using bbox centers (fallback method)."""
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        for bbox in gt_bboxes:
            # Convert bbox to numpy array if it's a BaseBoxes object
            if hasattr(bbox, 'tensor'):
                bbox_array = bbox.tensor.cpu().numpy().flatten()
            elif hasattr(gt_bboxes, 'tensor'):
                # gt_bboxes is a BaseBoxes object itself
                bbox_array = gt_bboxes.tensor[0].cpu().numpy().flatten() if len(gt_bboxes.tensor) > 0 else []
                if len(bbox_array) < 4:
                    continue
            else:
                bbox_array = np.array(bbox).flatten()
            
            # Ensure we have at least 4 coordinates
            if len(bbox_array) < 4:
                continue
                
            # Get bbox center (but add noise) - format: [x1, y1, x2, y2]
            cx = (bbox_array[0] + bbox_array[2]) / 2
            cy = (bbox_array[1] + bbox_array[3]) / 2
            
            # Add noise to center with some probability
            if random.random() < self.noise_ratio:
                cx += np.random.normal(0, self.center_noise_std)
                cy += np.random.normal(0, self.center_noise_std)
            
            # Clamp to image bounds
            cx = np.clip(cx, 0, width - 1)
            cy = np.clip(cy, 0, height - 1)
            
            # Add Gaussian to heatmap
            self._add_gaussian_to_heatmap(heatmap, cx, cy, height, width)
        
        return heatmap
    
    def _generate_gaussian_heatmap(self, h: int, w: int, cx: float, cy: float, sigma: float) -> np.ndarray:
        """Generate Gaussian heatmap with added noise."""
        y, x = np.ogrid[:h, :w]
        
        # Distance from center
        dist_sq = (x - cx)**2 + (y - cy)**2
        
        # Gaussian with noise
        heatmap = np.exp(-dist_sq / (2 * sigma**2))
        
        # Add multiplicative noise to make it realistic
        if random.random() < 0.5:
            noise_factor = np.random.uniform(0.8, 1.2)
            heatmap *= noise_factor
        
        return np.clip(heatmap, 0, 1).astype(np.float32)
    
    def _add_keypoint_gaussian_to_heatmap(self, heatmap: np.ndarray, keypoints: List[Tuple], height: int, width: int):
        """Add Gaussian blobs for each keypoint to the heatmap."""
        for kx, ky, visibility in keypoints:
            # Adjust sigma based on visibility and add some randomness
            base_sigma = (self.min_sigma + self.max_sigma) / 2
            if visibility == 1:  # Occluded
                sigma = base_sigma * random.uniform(1.5, 2.5)  # Larger, more uncertain
            else:  # Visible
                sigma = base_sigma * random.uniform(0.8, 1.2)
            
            self._add_gaussian_to_heatmap(heatmap, kx, ky, height, width, sigma)
    
    def _add_gaussian_to_heatmap(self, heatmap: np.ndarray, cx: float, cy: float, 
                                height: int, width: int, sigma: Optional[float] = None):
        """Add a single Gaussian blob to the heatmap."""
        if sigma is None:
            if self.quality_variance:
                sigma = np.random.uniform(self.min_sigma, self.max_sigma)
            else:
                sigma = (self.min_sigma + self.max_sigma) / 2
        
        # Generate Gaussian
        y, x = np.ogrid[:height, :width]
        dist_sq = (x - cx)**2 + (y - cy)**2
        gaussian = np.exp(-dist_sq / (2 * sigma**2))
        
        # Add multiplicative noise to make it realistic
        if random.random() < 0.5:
            noise_factor = np.random.uniform(0.8, 1.2)
            gaussian *= noise_factor
        
        # Add to heatmap (take maximum to avoid overlapping issues)
        heatmap[:] = np.maximum(heatmap, gaussian)
    
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
        
        # Add noise
        noise = np.random.normal(0, 0.02, (height, width))
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