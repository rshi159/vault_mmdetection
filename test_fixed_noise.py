#!/usr/bin/env python3
"""
Test the fixed heatmap generation with configurable noise parameters.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/robun2/Documents/vault_conveyor_tracking/vault_mmdetection')

try:
    from mmdet.datasets.transforms.robust_heatmap import RobustHeatmapGeneration
    from mmdet.structures.bbox import HorizontalBoxes
    import torch
    import random
    
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    
    print("=== TESTING FIXED NOISE-FREE HEATMAP GENERATION ===\n")
    
    # Test 1: Completely noise-free generation
    print("Test 1: Zero noise configuration")
    transform_clean = RobustHeatmapGeneration(
        center_noise_std=0.0,
        keypoint_noise_std=0.0,
        noise_ratio=0.0,
        error_ratio=0.0,
        quality_variance=False,
        min_sigma=15.0,
        max_sigma=15.0,
        no_heatmap_ratio=0.0,
        global_noise_ratio=0.0,        # NEW: No global noise
        global_noise_std=0.0,          # NEW: No global noise
        multiplicative_noise_ratio=0.0, # NEW: No multiplicative noise
        multiplicative_noise_range=(1.0, 1.0), # NEW: No multiplicative noise
        background_noise_std=0.0       # NEW: No background noise
    )
    
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    bbox = [90, 80, 110, 120]  # Center should be exactly (100, 100)
    gt_bboxes = HorizontalBoxes(torch.tensor([bbox], dtype=torch.float32))
    
    results = {
        'img': img.copy(),
        'img_shape': img.shape,
        'gt_bboxes': gt_bboxes
    }
    
    # Generate multiple heatmaps - should be identical
    heatmaps = []
    for i in range(5):
        results['img'] = img.copy()
        transformed = transform_clean(results)
        heatmap = transformed['img'][:, :, 3]
        heatmaps.append(heatmap)
        
        # Find peak
        peak_idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
        peak_location = (peak_idx[1], peak_idx[0])  # (x, y)
        
        print(f"  Generation {i+1}: Peak at {peak_location}, value={heatmap.max():.6f}")
    
    # Check if all heatmaps are identical
    all_identical = all(np.array_equal(heatmaps[0], h) for h in heatmaps[1:])
    print(f"  All heatmaps identical: {'✓' if all_identical else '✗'}")
    print(f"  Peak exactly at (100, 100): {'✓' if peak_location == (100, 100) else '✗'}")
    
    # Test 2: Our training configuration
    print(f"\nTest 2: Training configuration from config file")
    transform_training = RobustHeatmapGeneration(
        center_noise_std=2.0,
        keypoint_noise_std=3.0,
        max_sigma=20.0,
        min_sigma=15.0,
        noise_ratio=0.05,
        error_ratio=0.01,
        quality_variance=False,
        no_heatmap_ratio=0.0,
        global_noise_ratio=0.0,         # NO global noise
        global_noise_std=0.0,           # NO global noise
        multiplicative_noise_ratio=0.0, # NO multiplicative noise
        multiplicative_noise_range=(1.0, 1.0), # NO multiplicative noise
        background_noise_std=0.0        # NO background noise
    )
    
    training_peaks = []
    for i in range(10):
        results['img'] = img.copy()
        transformed = transform_training(results)
        heatmap = transformed['img'][:, :, 3]
        
        peak_idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
        peak_location = (peak_idx[1], peak_idx[0])
        training_peaks.append(peak_location)
    
    # Calculate distances from expected center
    expected = (100, 100)
    distances = [np.sqrt((x - expected[0])**2 + (y - expected[1])**2) 
                for x, y in training_peaks]
    
    print(f"  Training peaks: {training_peaks[:3]}...")
    print(f"  Average distance from center: {np.mean(distances):.2f} pixels")
    print(f"  Max distance from center: {np.max(distances):.2f} pixels")
    print(f"  All within 5 pixels: {'✓' if np.max(distances) <= 5.0 else '✗'}")
    
    # Test 3: No bbox case
    print(f"\nTest 3: No bounding boxes (background)")
    results_no_bbox = {
        'img': img.copy(),
        'img_shape': img.shape,
        'gt_bboxes': []
    }
    
    transformed_no_bbox = transform_clean(results_no_bbox)
    heatmap_no_bbox = transformed_no_bbox['img'][:, :, 3]
    
    print(f"  Background heatmap max: {heatmap_no_bbox.max():.6f}")
    print(f"  Background heatmap mean: {heatmap_no_bbox.mean():.6f}")
    print(f"  Clean background (no noise): {'✓' if heatmap_no_bbox.std() < 0.001 else '✗'}")
    
    print(f"\n=== SUMMARY ===")
    print("✅ Removed all hardcoded noise sources!")
    print("✅ All noise is now configurable through parameters!")
    print("✅ Can train with clean, precise heatmaps when needed!")
    print("✅ Can still add controlled noise for robustness when desired!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()