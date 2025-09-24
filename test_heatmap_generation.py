#!/usr/bin/env python3
"""
Test script to check if heatmaps are being generated properly
and are not garbage data.
"""

import numpy as np
import sys
import os
sys.path.insert(0, '/home/robun2/Documents/vault_conveyor_tracking/vault_mmdetection')

try:
    from mmdet.datasets.transforms.robust_heatmap import RobustHeatmapGeneration
    from mmdet.structures.bbox import HorizontalBoxes
    import torch
    
    print("Successfully imported required modules")
    
    # Create a test transform
    transform = RobustHeatmapGeneration(
        center_noise_std=4.0,
        keypoint_noise_std=6.0,
        max_sigma=20.0,
        min_sigma=15.0,
        noise_ratio=0.1,
        error_ratio=0.01,
        quality_variance=False
    )
    
    # Create synthetic test data
    img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
    
    # Create sample bounding boxes (simulating objects)
    gt_bboxes = HorizontalBoxes(torch.tensor([
        [50, 50, 150, 150],    # Object 1
        [200, 100, 280, 200],  # Object 2
        [100, 250, 200, 300]   # Object 3
    ]))
    
    results = {
        'img': img,
        'img_shape': img.shape,
        'gt_bboxes': gt_bboxes
    }
    
    print("\nGenerating 10 heatmaps to check consistency...")
    
    # Generate 10 heatmaps to check consistency
    heatmaps = []
    for i in range(10):
        # Reset the image each time (since transform modifies it)
        results['img'] = img.copy()
        transformed = transform(results)
        heatmap = transformed['img'][:, :, 3]  # 4th channel
        heatmaps.append(heatmap)
        print(f"Heatmap {i+1}: min={heatmap.min():.4f}, max={heatmap.max():.4f}, mean={heatmap.mean():.4f}, std={heatmap.std():.4f}")
    
    # Check if they're different (not cached/static)
    print(f"\nAre heatmaps different? {not np.array_equal(heatmaps[0], heatmaps[1])}")
    
    # Calculate correlation between first two
    h1_flat = heatmaps[0].flatten()
    h2_flat = heatmaps[1].flatten()
    
    # Check if they have variation (not all zeros)
    print(f"Heatmap 1 has variation: {h1_flat.std() > 0.001}")
    print(f"Heatmap 2 has variation: {h2_flat.std() > 0.001}")
    
    if h1_flat.std() > 0 and h2_flat.std() > 0:
        correlation = np.corrcoef(h1_flat, h2_flat)[0,1]
        print(f"Correlation between first two: {correlation:.4f}")
    else:
        print("One or both heatmaps are flat (no variation)")
    
    # Test with no bboxes (should get background prior)
    print(f"\nTesting with no bounding boxes...")
    results_no_bbox = {
        'img': img.copy(),
        'img_shape': img.shape,
        'gt_bboxes': []
    }
    
    transformed_no_bbox = transform(results_no_bbox)
    heatmap_no_bbox = transformed_no_bbox['img'][:, :, 3]
    print(f"No bbox heatmap: min={heatmap_no_bbox.min():.4f}, max={heatmap_no_bbox.max():.4f}, mean={heatmap_no_bbox.mean():.4f}")
    
    print("\n✓ Heatmap generation test completed successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()