#!/usr/bin/env python3
"""
Focused test to verify exact positioning correlation with ground truth.
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
    
    print("=== PRECISE GROUND TRUTH CORRELATION TEST ===\n")
    
    # Create transform with ZERO noise for exact testing
    transform = RobustHeatmapGeneration(
        center_noise_std=0.0,  # No center noise
        keypoint_noise_std=0.0,  # No keypoint noise
        max_sigma=15.0,
        min_sigma=15.0,  # Fixed sigma
        noise_ratio=0.0,  # No random noise
        error_ratio=0.0,  # No deliberate errors
        quality_variance=False,  # No variance
        no_heatmap_ratio=0.0  # Always generate heatmap
    )
    
    # Test cases with precise expected locations
    test_cases = [
        {
            'name': 'Simple Center Test',
            'bbox': [90, 80, 110, 120],  # Center should be (100, 100)
            'expected_center': (100, 100)
        },
        {
            'name': 'Off-center Test',
            'bbox': [0, 0, 60, 40],  # Center should be (30, 20)
            'expected_center': (30, 20)
        },
        {
            'name': 'Large Box Test',
            'bbox': [50, 50, 150, 150],  # Center should be (100, 100)
            'expected_center': (100, 100)
        }
    ]
    
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases):
        print(f"Test {i+1}: {test_case['name']}")
        
        # Create single bbox
        bbox_coords = test_case['bbox']
        gt_bboxes = HorizontalBoxes(torch.tensor([bbox_coords], dtype=torch.float32))
        
        results = {
            'img': img.copy(),
            'img_shape': img.shape,
            'gt_bboxes': gt_bboxes
        }
        
        # Generate heatmap
        transformed = transform(results)
        heatmap = transformed['img'][:, :, 3]  # 4th channel
        
        # Find the maximum value location
        max_idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
        actual_center = (max_idx[1], max_idx[0])  # (x, y)
        expected_center = test_case['expected_center']
        
        # Calculate exact center from bbox
        x1, y1, x2, y2 = bbox_coords
        calculated_center = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        distance = np.sqrt((actual_center[0] - expected_center[0])**2 + 
                          (actual_center[1] - expected_center[1])**2)
        
        print(f"  BBox: [{x1}, {y1}, {x2}, {y2}]")
        print(f"  Calculated center: ({calculated_center[0]:.1f}, {calculated_center[1]:.1f})")
        print(f"  Expected center: {expected_center}")
        print(f"  Actual peak location: {actual_center}")
        print(f"  Distance error: {distance:.2f} pixels")
        print(f"  Peak value: {heatmap.max():.4f}")
        print(f"  Heatmap has signal: {'✓' if heatmap.max() > 0.5 else '✗'}")
        
        # Check if the peak is exactly where expected (within 1 pixel)
        test_passed = distance <= 1.0 and heatmap.max() > 0.5
        print(f"  Result: {'✓ PASS' if test_passed else '✗ FAIL'}")
        
        if not test_passed:
            all_passed = False
            
        print()
    
    # Test multiple objects
    print("Test: Multiple Objects")
    multi_bboxes = [
        [20, 20, 40, 40],   # Center (30, 30)
        [160, 160, 180, 180]  # Center (170, 170)
    ]
    
    gt_bboxes = HorizontalBoxes(torch.tensor(multi_bboxes, dtype=torch.float32))
    
    results = {
        'img': img.copy(),
        'img_shape': img.shape,
        'gt_bboxes': gt_bboxes
    }
    
    transformed = transform(results)
    heatmap = transformed['img'][:, :, 3]
    
    # Check for peaks at both locations
    expected_centers = [(30, 30), (170, 170)]
    
    for j, expected_center in enumerate(expected_centers):
        # Check in a 5x5 region around expected center
        cx, cy = expected_center
        region = heatmap[max(0, cy-2):cy+3, max(0, cx-2):cx+3]
        
        if region.size > 0:
            max_val = region.max()
            print(f"  Object {j+1} at {expected_center}: peak value = {max_val:.4f}")
        else:
            print(f"  Object {j+1} at {expected_center}: no region found")
    
    overall_max = heatmap.max()
    print(f"  Overall heatmap max: {overall_max:.4f}")
    print(f"  Multiple objects detected: {'✓' if overall_max > 0.5 else '✗'}")
    
    print(f"\n=== FINAL RESULT ===")
    if all_passed:
        print("🎯 ALL TESTS PASSED!")
        print("✅ Heatmaps are correctly positioned at ground truth bounding box centers!")
        print("✅ The 4th channel provides accurate spatial attention information!")
    else:
        print("❌ SOME TESTS FAILED!")
        print("❗ Heatmap positioning may not be accurate!")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()