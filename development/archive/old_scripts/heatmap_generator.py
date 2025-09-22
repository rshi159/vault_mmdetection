"""
Prior Heatmap Generator Module

A standalone module for generating various types of prior heatmaps for object detection.
Supports multiple heatmap generation modes including YOLO keypoints with adaptive uncertainty.

YOLO Keypoint Visibility Format:
    0 = not labeled (keypoint not annotated)
    1 = occluded (keypoint exists but is hidden) 
    2 = visible (keypoint is clearly visible)

Usage:
    from heatmap_generator import HeatmapGenerator
    
    generator = HeatmapGenerator()
    heatmap = generator.generate_heatmap(
        data=keypoints,
        bbox=bbox_coords,
        img_shape=(height, width),
        heatmap_type='yolo_keypoints',
        uncertainty_mode='adaptive',
        visibilities=[2, 2, 1, 0]  # visible, visible, occluded, not labeled
    )
"""

import os
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum


class HeatmapType(Enum):
    """Supported heatmap generation types."""
    GAUSSIAN_BLOB = "gaussian_blob"
    RECTANGULAR = "rectangular"
    SHAPE_AWARE = "shape_aware"
    YOLO_KEYPOINTS = "yolo_keypoints"
    HYBRID_KEYPOINT_GAUSSIAN = "hybrid_keypoint_gaussian"
    PCA_SHAPE_GAUSSIAN = "pca_shape_gaussian"


class UncertaintyMode(Enum):
    """Uncertainty modeling modes for keypoint-based heatmaps."""
    UNIFORM = "uniform"
    BBOX_BASED = "bbox_based"
    ADAPTIVE = "adaptive"


class HeatmapGenerator:
    """
    Generate prior heatmaps for object detection with multiple generation modes.
    
    Supports:
    - Gaussian blob heatmaps (traditional)
    - Rectangular heatmaps (box-based)
    - Shape-aware heatmaps (enhanced rectangular)
    - YOLO keypoint heatmaps with adaptive uncertainty (recommended)
    - Hybrid keypoint + Gaussian heatmaps (best of both worlds)
    - PCA shape Gaussian heatmaps (orientation + shape aware)
    """
    
    def __init__(self, default_uncertainty: float = 15.0):
        """
        Initialize the heatmap generator.
        
        Args:
            default_uncertainty: Default uncertainty value in pixels
        """
        self.default_uncertainty = default_uncertainty
    
    def generate_heatmap(self,
                        data: Union[np.ndarray, List],
                        bbox: List[float],
                        img_shape: Tuple[int, int],
                        heatmap_type: str = "yolo_keypoints",
                        uncertainty_mode: str = "adaptive",
                        base_uncertainty: Optional[float] = None,
                        **kwargs) -> np.ndarray:
        """
        Generate a prior heatmap based on the specified type and data.
        
        Args:
            data: Input data for heatmap generation
                - For keypoints: (N, 2) array of [x, y] coordinates
                - For other types: not used (can be None)
            bbox: Bounding box as [x1, y1, x2, y2] in pixel coordinates
            img_shape: Image dimensions as (height, width)
            heatmap_type: Type of heatmap to generate
            uncertainty_mode: Uncertainty modeling mode (for keypoint heatmaps)
            base_uncertainty: Base uncertainty value (defaults to self.default_uncertainty)
            **kwargs: Additional parameters (e.g., visibilities for keypoints)
            
        Returns:
            numpy.ndarray: Generated heatmap of shape (H, W) with values in [0, 1]
        """
        if base_uncertainty is None:
            base_uncertainty = self.default_uncertainty
            
        heatmap_type = heatmap_type.lower()
        
        if heatmap_type == HeatmapType.GAUSSIAN_BLOB.value:
            return self._generate_gaussian_blob(bbox, img_shape, base_uncertainty)
        elif heatmap_type == HeatmapType.RECTANGULAR.value:
            return self._generate_rectangular(bbox, img_shape)
        elif heatmap_type == HeatmapType.SHAPE_AWARE.value:
            return self._generate_shape_aware(bbox, img_shape, base_uncertainty)
        elif heatmap_type == HeatmapType.YOLO_KEYPOINTS.value:
            visibilities = kwargs.get('visibilities', None)
            return self._generate_yolo_keypoints(
                data, visibilities, bbox, img_shape, uncertainty_mode, base_uncertainty
            )
        elif heatmap_type == HeatmapType.HYBRID_KEYPOINT_GAUSSIAN.value:
            visibilities = kwargs.get('visibilities', None)
            gaussian_weight = kwargs.get('gaussian_weight', 0.3)  # Default 30% Gaussian contribution
            return self._generate_hybrid_keypoint_gaussian(
                data, visibilities, bbox, img_shape, uncertainty_mode, base_uncertainty, gaussian_weight
            )
        elif heatmap_type == HeatmapType.PCA_SHAPE_GAUSSIAN.value:
            visibilities = kwargs.get('visibilities', None)
            alpha = kwargs.get('alpha', 0.4)  # Fraction converting axis length -> std
            keypoint_weight = kwargs.get('keypoint_weight', 0.3)  # Optional keypoint peaks on top
            return self._generate_pca_shape_gaussian(
                data, visibilities, bbox, img_shape, base_uncertainty, alpha, keypoint_weight
            )
        else:
            raise ValueError(f"Unknown heatmap type: {heatmap_type}")
    
    def _generate_gaussian_blob(self, bbox: List[float], img_shape: Tuple[int, int], 
                               sigma: float) -> np.ndarray:
        """Generate traditional Gaussian blob heatmap at bbox center."""
        H, W = img_shape
        heatmap = np.zeros((H, W), dtype=np.float32)
        
        # Calculate center
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        
        # Generate Gaussian
        y_coords, x_coords = np.ogrid[:H, :W]
        heatmap = np.exp(-((x_coords - cx)**2 + (y_coords - cy)**2) / (2 * sigma**2))
        
        return heatmap
    
    def _generate_rectangular(self, bbox: List[float], img_shape: Tuple[int, int]) -> np.ndarray:
        """Generate simple rectangular heatmap."""
        H, W = img_shape
        heatmap = np.zeros((H, W), dtype=np.float32)
        
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        
        if x2 > x1 and y2 > y1:
            heatmap[y1:y2, x1:x2] = 1.0
            
        return heatmap
    
    def _generate_shape_aware(self, bbox: List[float], img_shape: Tuple[int, int], 
                             sigma: float) -> np.ndarray:
        """Generate enhanced rectangular heatmap with Gaussian falloff."""
        H, W = img_shape
        heatmap = np.zeros((H, W), dtype=np.float32)
        
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        
        if x2 > x1 and y2 > y1:
            # Core region with high confidence
            heatmap[y1:y2, x1:x2] = 1.0
            
            # Gradient falloff around edges
            edge_width = min(int(sigma/2), (x2-x1)//4, (y2-y1)//4)
            if edge_width > 0:
                for i in range(edge_width):
                    weight = 0.8 * (edge_width - i) / edge_width
                    # Expand region with decreasing weight
                    expand_x1 = max(0, x1 - i - 1)
                    expand_y1 = max(0, y1 - i - 1)
                    expand_x2 = min(W, x2 + i + 1)
                    expand_y2 = min(H, y2 + i + 1)
                    
                    heatmap[expand_y1:expand_y2, expand_x1:expand_x2] = np.maximum(
                        heatmap[expand_y1:expand_y2, expand_x1:expand_x2], weight)
        
        return heatmap
    
    def _generate_yolo_keypoints(self, 
                                keypoints: np.ndarray,
                                visibilities: Optional[np.ndarray],
                                bbox: List[float],
                                img_shape: Tuple[int, int],
                                uncertainty_mode: str,
                                base_uncertainty: float) -> np.ndarray:
        """
        Generate prior heatmap from YOLO keypoint annotations with adaptive uncertainty.
        
        This is the recommended method as it provides the most realistic prior information
        by using actual object keypoints with uncertainty modeling.
        
        Args:
            keypoints: (N, 2) array of keypoint coordinates [x, y]
            visibilities: (N,) array of visibility flags (0=not visible, 1=visible, 2=occluded)
            bbox: [x1, y1, x2, y2] bounding box
            img_shape: (H, W) image dimensions
            uncertainty_mode: 'uniform', 'bbox_based', or 'adaptive'
            base_uncertainty: Base uncertainty value in pixels
            
        Returns:
            numpy.ndarray: Prior heatmap (H, W) with values in [0, 1]
        """
        H, W = img_shape
        heatmap = np.zeros((H, W), dtype=np.float32)
        
        if len(keypoints) == 0:
            return heatmap
        
        # Default to all visible if no visibility info provided
        if visibilities is None:
            visibilities = np.ones(len(keypoints), dtype=int)
        
        # Convert bbox to width/height for uncertainty estimation
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]
        bbox_size = (bbox_w + bbox_h) / 2  # Average dimension
        
        for i, (kp, vis) in enumerate(zip(keypoints, visibilities)):
            if vis == 0:  # Not visible, skip
                continue
                
            kp_x, kp_y = kp
            
            # Skip keypoints outside image bounds
            if not (0 <= kp_x < W and 0 <= kp_y < H):
                continue
            
            # Determine uncertainty based on mode
            if uncertainty_mode == UncertaintyMode.UNIFORM.value:
                sigma = base_uncertainty
            elif uncertainty_mode == UncertaintyMode.BBOX_BASED.value:
                # Scale uncertainty with bbox size
                sigma = base_uncertainty * (bbox_size / 100.0)  # Normalize to ~100px bbox
            elif uncertainty_mode == UncertaintyMode.ADAPTIVE.value:
                # Higher uncertainty for occluded points, lower for visible
                # YOLO format: 0=not labeled, 1=occluded, 2=visible
                if vis == 1:  # Occluded
                    sigma = base_uncertainty * 1.5
                elif vis == 2:  # Visible
                    sigma = base_uncertainty * 0.8
                else:  # Not labeled (vis == 0)
                    sigma = base_uncertainty
            else:
                sigma = base_uncertainty
                
            # Clamp sigma to reasonable range
            sigma = np.clip(sigma, 5.0, 50.0)
            
            # Generate Gaussian for this keypoint
            y_coords, x_coords = np.ogrid[:H, :W]
            gaussian = np.exp(-((x_coords - kp_x)**2 + (y_coords - kp_y)**2) / (2 * sigma**2))
            
            # Weight by visibility (visible points get full weight, occluded get reduced)
            # YOLO format: 0=not labeled, 1=occluded, 2=visible
            if vis == 2:  # Visible
                weight = 1.0
            elif vis == 1:  # Occluded
                weight = 0.7
            else:  # Not labeled (vis == 0)
                weight = 0.3
            heatmap = np.maximum(heatmap, weight * gaussian)
        
        # Normalize to [0, 1]
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap
    
    def _generate_hybrid_keypoint_gaussian(self, 
                                         keypoints: np.ndarray,
                                         visibilities: Optional[np.ndarray],
                                         bbox: List[float],
                                         img_shape: Tuple[int, int],
                                         uncertainty_mode: str,
                                         base_uncertainty: float,
                                         gaussian_weight: float = 0.3) -> np.ndarray:
        """
        Generate hybrid heatmap combining YOLO keypoints with shape-aware 2D Gaussian.
        
        This method combines the structured information from keypoints with a central
        focus bias from a 2D Gaussian ellipse fitted to the bounding box. This provides:
        - Detailed object structure from keypoints (adaptive uncertainty)
        - Shape-aware central focus bias for object detection (2D elliptical Gaussian)
        - Balanced representation for training stability
        
        Args:
            keypoints: (N, 2) array of keypoint coordinates [x, y]
            visibilities: (N,) array of visibility flags (0=not visible, 1=visible, 2=occluded)
            bbox: [x1, y1, x2, y2] bounding box
            img_shape: (H, W) image dimensions
            uncertainty_mode: 'uniform', 'bbox_based', or 'adaptive'
            base_uncertainty: Base uncertainty value in pixels
            gaussian_weight: Weight for 2D Gaussian component (0.0-1.0). Default 0.3 = 30%
            
        Returns:
            numpy.ndarray: Hybrid heatmap (H, W) with values in [0, 1]
        """
        H, W = img_shape
        
        # Generate keypoint component (70% weight by default)
        keypoint_heatmap = self._generate_yolo_keypoints(
            keypoints, visibilities, bbox, img_shape, uncertainty_mode, base_uncertainty
        )
        
        # Generate soft center Gaussian component (30% weight by default)
        # Use shape-aware 2D Gaussian for better object representation
        center_sigma = base_uncertainty * 1.5  # Larger for softer center bias
        gaussian_heatmap = self._generate_shape_aware(bbox, img_shape, center_sigma)
        
        # Combine the heatmaps with weighted sum
        keypoint_weight = 1.0 - gaussian_weight
        hybrid_heatmap = (keypoint_weight * keypoint_heatmap + 
                         gaussian_weight * gaussian_heatmap)
        
        # Normalize to [0, 1] to maintain proper probability distribution
        if hybrid_heatmap.max() > 0:
            hybrid_heatmap = hybrid_heatmap / hybrid_heatmap.max()
        
        return hybrid_heatmap
    
    def _generate_pca_shape_gaussian(self, 
                                   keypoints: np.ndarray,
                                   visibilities: Optional[np.ndarray],
                                   bbox: List[float],
                                   img_shape: Tuple[int, int],
                                   base_uncertainty: float,
                                   alpha: float = 0.4,
                                   keypoint_weight: float = 0.0) -> np.ndarray:
        """
        Generate PCA-based shape Gaussian heatmap from keypoints.
        
        Uses keypoint distribution to estimate object orientation and extent,
        creating a rotated Gaussian that captures shape information naturally.
        This approach encodes center, shape, and orientation in a unified way.
        
        Args:
            keypoints: (N, 2) array of keypoint coordinates [x, y]
            visibilities: (N,) array of visibility flags (0=not visible, 1=visible, 2=occluded)
            bbox: [x1, y1, x2, y2] bounding box (used for center and fallback)
            img_shape: (H, W) image dimensions
            base_uncertainty: Base uncertainty value in pixels
            alpha: Fraction converting axis length -> std (0.35-0.5 typical)
            keypoint_weight: Optional weight for keypoint peaks on top of shape Gaussian
            
        Returns:
            numpy.ndarray: Shape-aware Gaussian heatmap (H, W) with values in [0, 1]
        """
        H, W = img_shape
        
        # Calculate center from bbox
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        center_xy = np.array([cx, cy], dtype=np.float32)
        
        # Handle empty keypoints - fallback to axis-aligned bbox Gaussian
        if len(keypoints) == 0:
            bbox_w = bbox[2] - bbox[0]
            bbox_h = bbox[3] - bbox[1]
            sigma_x = max(alpha * bbox_w / 2.0, base_uncertainty)
            sigma_y = max(alpha * bbox_h / 2.0, base_uncertainty)
            return self._generate_axis_aligned_gaussian(
                center_xy, (sigma_x, sigma_y), img_shape
            )
        
        # Prepare keypoint weights based on visibility
        if visibilities is None:
            weights = np.ones(len(keypoints), dtype=np.float32)
        else:
            # Weight visible points higher than occluded
            # YOLO format: 0=not labeled, 1=occluded, 2=visible
            weights = np.ones(len(keypoints), dtype=np.float32)
            weights[visibilities == 2] = 1.0  # Visible: full weight
            weights[visibilities == 1] = 0.7  # Occluded: reduced weight
            weights[visibilities == 0] = 0.1  # Not labeled: minimal weight
        
        # Center keypoints around bbox center
        K = keypoints.astype(np.float32)
        centered_keypoints = K - center_xy[None, :]  # (N, 2)
        
        # Weighted PCA: compute covariance matrix
        Wsum = np.sum(weights) + 1e-8
        weighted_centered = centered_keypoints * weights[:, None]  # (N, 2)
        cov_matrix = (weighted_centered.T @ centered_keypoints) / Wsum  # (2, 2)
        
        # SVD to get principal axes and eigenvalues
        U, S, _ = np.linalg.svd(cov_matrix)  # U: eigenvectors, S: eigenvalues
        
        # Project keypoints onto principal axes to measure extents
        projections = centered_keypoints @ U  # (N, 2)
        
        # Measure extents along principal axes
        L1 = projections[:, 0].max() - projections[:, 0].min()  # Extent along 1st axis
        L2 = projections[:, 1].max() - projections[:, 1].min()  # Extent along 2nd axis
        
        # Convert extents to Gaussian standard deviations
        sigma1 = max(alpha * (L1 / 2.0), base_uncertainty)
        sigma2 = max(alpha * (L2 / 2.0), base_uncertainty)
        
        # Build covariance matrix in image coordinates: Σ = R * diag(σ²) * R^T
        R = U  # Rotation matrix (principal directions)
        sigma_diag = np.diag([sigma1**2, sigma2**2])
        covariance = R @ sigma_diag @ R.T
        
        # Generate coordinate grids
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        coords = np.stack([xx - cx, yy - cy], axis=-1)  # (H, W, 2)
        
        # Compute Mahalanobis distance using covariance inverse
        cov_inv = np.linalg.inv(covariance + 1e-8 * np.eye(2))  # Regularize for stability
        # Vectorized computation: (H, W, 2) @ (2, 2) -> (H, W, 2), then element-wise multiply and sum
        mahalanobis = np.sum((coords @ cov_inv) * coords, axis=-1)  # (H, W)
        
        # Generate shape Gaussian
        shape_gaussian = np.exp(-0.5 * mahalanobis).astype(np.float32)
        
        # Optionally add keypoint peaks on top
        if keypoint_weight > 0.0 and len(keypoints) > 0:
            keypoint_heatmap = self._generate_yolo_keypoints(
                keypoints, visibilities, bbox, img_shape, 'adaptive', base_uncertainty
            )
            # Combine with weighted sum
            combined_heatmap = ((1.0 - keypoint_weight) * shape_gaussian + 
                              keypoint_weight * keypoint_heatmap)
        else:
            combined_heatmap = shape_gaussian
        
        # Normalize to [0, 1]
        if combined_heatmap.max() > 0:
            combined_heatmap = combined_heatmap / combined_heatmap.max()
        
        return np.clip(combined_heatmap, 0.0, 1.0)
    
    def _generate_axis_aligned_gaussian(self, center_xy: np.ndarray, sigmas_xy: Tuple[float, float], 
                                      img_shape: Tuple[int, int]) -> np.ndarray:
        """Helper: Generate axis-aligned Gaussian (fallback for degenerate keypoints)."""
        cx, cy = center_xy
        sigma_x, sigma_y = sigmas_xy
        H, W = img_shape
        
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        dx = (xx - cx) / (sigma_x + 1e-8)
        dy = (yy - cy) / (sigma_y + 1e-8)
        gaussian = np.exp(-0.5 * (dx*dx + dy*dy))
        
        return np.clip(gaussian, 0.0, 1.0)
    
    def parse_yolo_label(self, label_path: str, img_width: int, img_height: int) -> Dict:
        """
        Parse YOLO label file with keypoint annotations.
        
        Expected format: class cx cy w h x1 y1 v1 x2 y2 v2 ... x8 y8 v8
        Where coordinates are normalized [0,1] and visibility: 0=not visible, 1=visible, 2=occluded
        
        Args:
            label_path: Path to YOLO .txt label file
            img_width: Image width in pixels
            img_height: Image height in pixels
            
        Returns:
            Dict with keys: 'keypoints', 'visibilities', 'bbox', 'class_id'
        """
        if not os.path.exists(label_path):
            return {
                'keypoints': np.array([]).reshape(0, 2),
                'visibilities': np.array([]),
                'bbox': [0, 0, 0, 0],
                'class_id': 0
            }
        
        with open(label_path, 'r') as f:
            line = f.readline().strip()
        
        if not line:
            return {
                'keypoints': np.array([]).reshape(0, 2),
                'visibilities': np.array([]),
                'bbox': [0, 0, 0, 0],
                'class_id': 0
            }
        
        parts = line.split()
        if len(parts) < 5:
            return {
                'keypoints': np.array([]).reshape(0, 2),
                'visibilities': np.array([]),
                'bbox': [0, 0, 0, 0],
                'class_id': 0
            }
        
        # Parse class and bbox (normalized)
        class_id = int(parts[0])
        cx_norm = float(parts[1])
        cy_norm = float(parts[2])
        w_norm = float(parts[3])
        h_norm = float(parts[4])
        
        # Convert to pixel coordinates
        cx = cx_norm * img_width
        cy = cy_norm * img_height
        w = w_norm * img_width
        h = h_norm * img_height
        
        # Convert to x1, y1, x2, y2
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        bbox = [x1, y1, x2, y2]
        
        # Parse keypoints if present
        keypoints = []
        visibilities = []
        
        if len(parts) > 5:
            keypoint_data = parts[5:]
            # Each keypoint has 3 values: x, y, visibility
            for i in range(0, len(keypoint_data), 3):
                if i + 2 < len(keypoint_data):
                    kp_x_norm = float(keypoint_data[i])
                    kp_y_norm = float(keypoint_data[i + 1])
                    kp_vis = int(keypoint_data[i + 2])
                    
                    # Convert to pixel coordinates
                    kp_x = kp_x_norm * img_width
                    kp_y = kp_y_norm * img_height
                    
                    keypoints.append([kp_x, kp_y])
                    visibilities.append(kp_vis)
        
        return {
            'keypoints': np.array(keypoints) if keypoints else np.array([]).reshape(0, 2),
            'visibilities': np.array(visibilities) if visibilities else np.array([]),
            'bbox': bbox,
            'class_id': class_id
        }
    
    def generate_from_yolo_file(self,
                               label_path: str,
                               img_shape: Tuple[int, int],
                               heatmap_type: str = "yolo_keypoints",
                               uncertainty_mode: str = "adaptive",
                               base_uncertainty: Optional[float] = None) -> np.ndarray:
        """
        Generate heatmap directly from YOLO label file.
        
        Args:
            label_path: Path to YOLO .txt label file
            img_shape: Image dimensions as (height, width)
            heatmap_type: Type of heatmap to generate
            uncertainty_mode: Uncertainty modeling mode
            base_uncertainty: Base uncertainty value
            
        Returns:
            numpy.ndarray: Generated heatmap of shape (H, W)
        """
        H, W = img_shape
        label_data = self.parse_yolo_label(label_path, W, H)
        
        return self.generate_heatmap(
            data=label_data['keypoints'],
            bbox=label_data['bbox'],
            img_shape=img_shape,
            heatmap_type=heatmap_type,
            uncertainty_mode=uncertainty_mode,
            base_uncertainty=base_uncertainty,
            visibilities=label_data['visibilities']
        )


def main():
    """Example usage of the HeatmapGenerator."""
    # Example: Generate heatmap from YOLO keypoints
    generator = HeatmapGenerator()
    
    # Sample data (replace with your actual data)
    sample_keypoints = np.array([
        [100, 120],  # keypoint 1
        [150, 130],  # keypoint 2
        [200, 140],  # keypoint 3
        [250, 150],  # keypoint 4
    ])
    sample_visibilities = np.array([1, 1, 2, 1])  # visible, visible, occluded, visible
    sample_bbox = [80, 100, 270, 180]  # x1, y1, x2, y2
    img_shape = (480, 640)  # height, width
    
    # Generate different types of heatmaps
    heatmap_types = [
        ("gaussian_blob", "Gaussian Blob"),
        ("rectangular", "Rectangular"),
        ("shape_aware", "Shape-Aware"),
        ("yolo_keypoints", "YOLO Keypoints"),
        ("hybrid_keypoint_gaussian", "Hybrid Keypoint+Gaussian"),
        ("pca_shape_gaussian", "PCA Shape Gaussian (NEW!)")
    ]
    
    print("🎯 HeatmapGenerator Example Usage")
    print("=" * 50)
    
    for heatmap_type, description in heatmap_types:
        if heatmap_type in ["yolo_keypoints", "hybrid_keypoint_gaussian", "pca_shape_gaussian"]:
            # Special parameters for keypoint-based types
            extra_params = {}
            if heatmap_type == "hybrid_keypoint_gaussian":
                extra_params['gaussian_weight'] = 0.3  # 30% Gaussian, 70% keypoints
            elif heatmap_type == "pca_shape_gaussian":
                extra_params['alpha'] = 0.4  # Shape extent factor
                extra_params['keypoint_weight'] = 0.1  # Optional keypoint peaks
                
            heatmap = generator.generate_heatmap(
                data=sample_keypoints,
                bbox=sample_bbox,
                img_shape=img_shape,
                heatmap_type=heatmap_type,
                uncertainty_mode="adaptive",
                visibilities=sample_visibilities,
                **extra_params
            )
        else:
            heatmap = generator.generate_heatmap(
                data=None,
                bbox=sample_bbox,
                img_shape=img_shape,
                heatmap_type=heatmap_type
            )
        
        coverage = np.sum(heatmap > 0.1)
        peak = heatmap.max()
        
        print(f"{description}:")
        print(f"  - Coverage: {coverage} pixels")
        print(f"  - Peak value: {peak:.3f}")
        print(f"  - Mean value: {heatmap.mean():.6f}")
        print()
    
    print("✅ Example completed! Try 'pca_shape_gaussian' for shape-aware orientation!")
    print("\n🎯 Recommendations:")
    print("  • Pure keypoints: Use 'yolo_keypoints' with 'adaptive' uncertainty")
    print("  • Hybrid approach: Use 'hybrid_keypoint_gaussian' with gaussian_weight=0.2-0.4")
    print("  • Shape-aware: Use 'pca_shape_gaussian' with alpha=0.35-0.5")
    print("  • Traditional: Use 'gaussian_blob' for baseline comparison")


if __name__ == "__main__":
    main()