#!/usr/bin/env python3

import torch
import numpy as np
import sys
import os
import cv2
import matplotlib.pyplot as plt

# Add the repository root to Python path
repo_root = os.path.abspath('.')
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from mmdet.datasets.transforms.robust_heatmap import RobustHeatmapGeneration

def analyze_heatmap_generation():
    """Analyze what our heatmap generation is actually producing."""
    
    print("🔍 Analyzing Heatmap Generation...")
    
    # Create transform
    heatmap_gen = RobustHeatmapGeneration(
        noise_ratio=0.1,
        min_sigma=15.0,
        max_sigma=20.0,
        center_noise_std=4.0,
        keypoint_noise_std=6.0,
        error_ratio=0.01,
        quality_variance=False
    )
    
    # Simulate some detection data
    from mmdet.structures.bbox import HorizontalBoxes
    results = {
        'img': np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8),
        'gt_bboxes': HorizontalBoxes(np.array([[50, 50, 150, 150], [200, 200, 300, 300]], dtype=np.float32)),
        'gt_labels': np.array([0, 0], dtype=np.int64)
    }
    
    print(f"📊 Input image shape: {results['img'].shape}")
    print(f"📊 Input bboxes: {results['gt_bboxes']}")
    
    # Apply transform
    results = heatmap_gen(results)
    
    # Check the results
    img = results['img']
    print(f"📊 Output image shape: {img.shape}")
    print(f"📊 Output image dtype: {img.dtype}")
    
    if img.shape[2] == 4:  # Check if we have 4 channels
        rgb = img[:, :, :3]
        heatmap = img[:, :, 3]
        
        print(f"📊 RGB channels - min: {rgb.min():.3f}, max: {rgb.max():.3f}, mean: {rgb.mean():.3f}")
        print(f"📊 Heatmap channel - min: {heatmap.min():.3f}, max: {heatmap.max():.3f}, mean: {heatmap.mean():.3f}")
        print(f"📊 Heatmap std: {heatmap.std():.3f}")
        
        # Check if heatmap has meaningful structure
        non_zero_pixels = (heatmap > 0.1).sum()
        total_pixels = heatmap.size
        print(f"📊 Non-zero pixels in heatmap: {non_zero_pixels}/{total_pixels} ({100*non_zero_pixels/total_pixels:.1f}%)")
        
        # Check values around bbox centers
        bbox1_center = ((50+150)//2, (50+150)//2)  # (100, 100)
        bbox2_center = ((200+300)//2, (200+300)//2)  # (250, 250)
        
        print(f"📊 Heatmap value at bbox1 center {bbox1_center}: {heatmap[bbox1_center[1], bbox1_center[0]]:.3f}")
        print(f"📊 Heatmap value at bbox2 center {bbox2_center}: {heatmap[bbox2_center[1], bbox2_center[0]]:.3f}")
        
        return heatmap, rgb
    else:
        print("❌ Transform didn't add 4th channel!")
        return None, None

def test_normalization_effect():
    """Test how current normalization affects our data."""
    
    print("\n🔍 Testing Normalization Effects...")
    
    # Simulate the preprocessing
    from mmdet.models.data_preprocessors.data_preprocessor_4ch import DetDataPreprocessor4Ch
    
    preprocessor = DetDataPreprocessor4Ch(
        mean=[103.53, 116.28, 123.675, 0.0],
        std=[57.375, 57.12, 57.375, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32
    )
    
    # Create sample 4-channel data
    rgb_data = np.random.randint(0, 255, (1, 320, 320, 3))  # Typical image values
    heatmap_data = np.random.exponential(0.5, (1, 320, 320, 1))  # Exponential-like heatmap
    
    four_channel_data = np.concatenate([rgb_data, heatmap_data], axis=-1)
    
    print(f"📊 Before normalization:")
    print(f"   RGB - min: {rgb_data.min():.1f}, max: {rgb_data.max():.1f}, mean: {rgb_data.mean():.1f}")
    print(f"   Heatmap - min: {heatmap_data.min():.3f}, max: {heatmap_data.max():.3f}, mean: {heatmap_data.mean():.3f}")
    
    # Convert to tensor format expected by preprocessor
    tensor_data = torch.from_numpy(four_channel_data).permute(0, 3, 1, 2).float()
    
    # Apply normalization manually to see effect
    normalized = tensor_data.clone()
    mean = torch.tensor([103.53, 116.28, 123.675, 0.0]).view(1, 4, 1, 1)
    std = torch.tensor([57.375, 57.12, 57.375, 1.0]).view(1, 4, 1, 1)
    
    normalized = (normalized - mean) / std
    
    print(f"📊 After normalization:")
    print(f"   RGB channels - min: {normalized[0, :3].min():.3f}, max: {normalized[0, :3].max():.3f}, mean: {normalized[0, :3].mean():.3f}")
    print(f"   Heatmap channel - min: {normalized[0, 3].min():.3f}, max: {normalized[0, 3].max():.3f}, mean: {normalized[0, 3].mean():.3f}")

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Heatmap Analysis")
    print("=" * 60)
    
    heatmap, rgb = analyze_heatmap_generation()
    test_normalization_effect()
    
    print("\n" + "=" * 60)
    print("🎯 Analysis Summary:")
    print("=" * 60)
    print("Check if:")
    print("1. Heatmaps have reasonable value ranges")
    print("2. Heatmaps have structure around bbox centers") 
    print("3. Normalization parameters are appropriate")
    print("4. The 4th channel provides useful signal vs noise")