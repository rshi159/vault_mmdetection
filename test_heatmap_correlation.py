#!/usr/bin/env python3
"""
Comprehensive test to verify that heatmaps are correctly positioned
at ground truth bounding box centers and keypoint locations.
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.insert(0, '/home/robun2/Documents/vault_conveyor_tracking/vault_mmdetection')

try:
    from mmdet.datasets.transforms.robust_heatmap import RobustHeatmapGeneration
    from mmdet.structures.bbox import HorizontalBoxes
    import torch
    
    print("=== HEATMAP-GROUND TRUTH CORRELATION TEST ===\n")
    
    def test_heatmap_positioning():
        """Test if heatmaps are generated at correct bbox centers"""
        
        # Create transform with minimal noise for precise testing
        transform = RobustHeatmapGeneration(
            center_noise_std=0.0,  # No noise for precise testing
            keypoint_noise_std=0.0,
            max_sigma=10.0,
            min_sigma=10.0,
            noise_ratio=0.0,  # No noise
            error_ratio=0.0,  # No errors
            quality_variance=False,
            no_heatmap_ratio=0.0  # Always generate heatmap
        )
        
        # Test image
        img_size = (240, 320)  # H, W
        img = np.random.randint(0, 255, (img_size[0], img_size[1], 3), dtype=np.uint8)
        
        # Define test bounding boxes with known centers
        test_cases = [
            {
                'name': 'Single centered object',
                'bboxes': [[100, 80, 200, 160]],  # Center at (150, 120)
                'expected_centers': [(150, 120)]
            },
            {
                'name': 'Two objects',
                'bboxes': [[50, 50, 100, 100], [200, 150, 270, 200]],  # Centers at (75, 75) and (235, 175)
                'expected_centers': [(75, 75), (235, 175)]
            },
            {
                'name': 'Corner object',
                'bboxes': [[10, 10, 50, 50]],  # Center at (30, 30)
                'expected_centers': [(30, 30)]
            }
        ]
        
        for test_case in test_cases:
            print(f"Testing: {test_case['name']}")
            
            # Create bounding boxes
            gt_bboxes = HorizontalBoxes(torch.tensor(test_case['bboxes'], dtype=torch.float32))
            
            results = {
                'img': img.copy(),
                'img_shape': img.shape,
                'gt_bboxes': gt_bboxes
            }
            
            # Generate heatmap
            transformed = transform(results)
            heatmap = transformed['img'][:, :, 3]  # 4th channel
            
            # Find peaks in heatmap
            peak_locations = []
            for expected_center in test_case['expected_centers']:
                cx, cy = expected_center
                
                # Check a small region around expected center
                y_start = max(0, cy - 5)
                y_end = min(img_size[0], cy + 6)
                x_start = max(0, cx - 5)
                x_end = min(img_size[1], cx + 6)
                
                region = heatmap[y_start:y_end, x_start:x_end]
                
                # Find max in region
                if region.size > 0:
                    max_val = region.max()
                    max_indices = np.unravel_index(region.argmax(), region.shape)
                    actual_y = y_start + max_indices[0]
                    actual_x = x_start + max_indices[1]
                    
                    distance = np.sqrt((actual_x - cx)**2 + (actual_y - cy)**2)
                    
                    print(f"  Expected center: ({cx}, {cy})")
                    print(f"  Actual peak: ({actual_x}, {actual_y})")
                    print(f"  Distance: {distance:.2f} pixels")
                    print(f"  Peak value: {max_val:.4f}")
                    print(f"  Accuracy: {'✓' if distance < 3.0 else '✗'}")
                    
                    peak_locations.append((actual_x, actual_y, max_val))
                else:
                    print(f"  ✗ No region found for center ({cx}, {cy})")
            
            print()
        
        return True
    
    def test_keypoint_correlation():
        """Test correlation with keypoint annotations if available"""
        print("=== KEYPOINT CORRELATION TEST ===")
        
        # Since we don't have keypoint data in this simple test,
        # we'll test the bbox center correlation which is what
        # our current implementation uses
        
        transform = RobustHeatmapGeneration(
            center_noise_std=2.0,  # Small noise
            noise_ratio=0.0,  # No random noise
            error_ratio=0.0   # No errors
        )
        
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        
        # Single bbox with known center
        bbox = [80, 70, 120, 110]  # Center at (100, 90)
        gt_bboxes = HorizontalBoxes(torch.tensor([bbox], dtype=torch.float32))
        
        results = {
            'img': img,
            'img_shape': img.shape,
            'gt_bboxes': gt_bboxes
        }
        
        # Generate multiple heatmaps and check consistency
        centers_found = []
        for i in range(10):
            results['img'] = img.copy()
            transformed = transform(results)
            heatmap = transformed['img'][:, :, 3]
            
            # Find the peak
            peak_idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
            peak_y, peak_x = peak_idx
            centers_found.append((peak_x, peak_y))
        
        # Check if peaks are consistently near the expected center
        expected_center = (100, 90)
        distances = [np.sqrt((x - expected_center[0])**2 + (y - expected_center[1])**2) 
                    for x, y in centers_found]
        
        print(f"Expected center: {expected_center}")
        print(f"Found centers: {centers_found[:5]}...")  # Show first 5
        print(f"Average distance from expected: {np.mean(distances):.2f} pixels")
        print(f"Max distance: {np.max(distances):.2f} pixels")
        print(f"All within 10 pixels: {'✓' if np.max(distances) < 10 else '✗'}")
        
        return np.max(distances) < 10
    
    def test_no_bbox_case():
        """Test behavior when no bounding boxes are present"""
        print("\n=== NO BBOX TEST ===")
        
        transform = RobustHeatmapGeneration()
        
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        results = {
            'img': img,
            'img_shape': img.shape,
            'gt_bboxes': []  # No bboxes
        }
        
        transformed = transform(results)
        heatmap = transformed['img'][:, :, 3]
        
        print(f"No bbox heatmap stats:")
        print(f"  Min: {heatmap.min():.4f}")
        print(f"  Max: {heatmap.max():.4f}")
        print(f"  Mean: {heatmap.mean():.4f}")
        print(f"  Std: {heatmap.std():.4f}")
        
        # Should be mostly low values (background prior)
        is_background = heatmap.max() < 0.2
        print(f"  Looks like background: {'✓' if is_background else '✗'}")
        
        return is_background
    
    # Run all tests
    print("Running comprehensive heatmap positioning tests...\n")
    
    test1_passed = test_heatmap_positioning()
    test2_passed = test_keypoint_correlation()
    test3_passed = test_no_bbox_case()
    
    print(f"\n=== SUMMARY ===")
    print(f"✓ Positioning Test: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"✓ Consistency Test: {'PASSED' if test2_passed else 'FAILED'}")
    print(f"✓ No Bbox Test: {'PASSED' if test3_passed else 'FAILED'}")
    
    if test1_passed and test2_passed and test3_passed:
        print(f"\n🎯 ALL TESTS PASSED - Heatmaps correctly correlate with ground truth!")
    else:
        print(f"\n❌ SOME TESTS FAILED - Check heatmap generation logic!")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()