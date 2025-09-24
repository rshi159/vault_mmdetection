#!/usr/bin/env python3
"""
Simplified heatmap parameter analysis and RGB vs Heatmap comparison.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/robun2/Documents/vault_conveyor_tracking/vault_mmdetection')

try:
    from mmdet.datasets.transforms.robust_heatmap import RobustHeatmapGeneration
    from mmdet.structures.bbox import HorizontalBoxes
    import torch
    
    print("=== HEATMAP PARAMETER EVALUATION ===\n")
    
    # Current training parameters
    current_params = {
        'center_noise_std': 2.0,
        'keypoint_noise_std': 3.0,
        'max_sigma': 20.0,
        'min_sigma': 15.0,
        'noise_ratio': 0.05,
        'error_ratio': 0.01,
        'quality_variance': False,
        'no_heatmap_ratio': 0.0,
        'global_noise_ratio': 0.0,
        'global_noise_std': 0.0,
        'multiplicative_noise_ratio': 0.0,
        'multiplicative_noise_range': (1.0, 1.0),
        'background_noise_std': 0.0
    }
    
    print("📊 CURRENT TRAINING PARAMETERS:")
    print("=" * 50)
    for key, value in current_params.items():
        print(f"  {key:25}: {value}")
    
    # Create test scenarios
    def create_test_scenarios():
        """Create different test scenarios"""
        scenarios = []
        
        # Scenario 1: Single centered object
        img1 = np.ones((320, 320, 3), dtype=np.uint8) * 120  # Gray background
        bbox1 = HorizontalBoxes(torch.tensor([[140, 140, 180, 180]], dtype=torch.float32))  # Center: (160, 160)
        scenarios.append(("Single Center Object", img1, bbox1))
        
        # Scenario 2: Multiple objects
        img2 = np.ones((320, 320, 3), dtype=np.uint8) * 100
        bbox2 = HorizontalBoxes(torch.tensor([
            [50, 50, 100, 100],     # Center: (75, 75)
            [200, 80, 260, 140],    # Center: (230, 110)
            [120, 200, 180, 260]    # Center: (150, 230)
        ], dtype=torch.float32))
        scenarios.append(("Multiple Objects", img2, bbox2))
        
        # Scenario 3: Edge objects
        img3 = np.ones((320, 320, 3), dtype=np.uint8) * 130
        bbox3 = HorizontalBoxes(torch.tensor([
            [10, 10, 60, 60],       # Near corner
            [270, 150, 310, 200]    # Near edge
        ], dtype=torch.float32))
        scenarios.append(("Edge Objects", img3, bbox3))
        
        return scenarios
    
    def analyze_parameter_set(name, params, scenarios):
        """Analyze a parameter set across scenarios"""
        print(f"\n🔍 ANALYZING '{name.upper()}' PARAMETERS:")
        print("=" * 50)
        
        transform = RobustHeatmapGeneration(**params)
        
        for scenario_name, img, gt_bboxes in scenarios:
            print(f"\n📋 Scenario: {scenario_name}")
            
            results = {
                'img': img.copy(),
                'img_shape': img.shape,
                'gt_bboxes': gt_bboxes
            }
            
            # Generate 5 samples to check consistency
            heatmap_stats = []
            peak_positions = []
            
            for i in range(5):
                results['img'] = img.copy()  # Reset image
                transformed = transform(results)
                heatmap = transformed['img'][:, :, 3]
                
                # Basic statistics
                stats = {
                    'min': heatmap.min(),
                    'max': heatmap.max(),
                    'mean': heatmap.mean(),
                    'std': heatmap.std(),
                    'active_pixels': np.sum(heatmap > 0.1)
                }
                heatmap_stats.append(stats)
                
                # Find peaks
                peaks = find_peaks_simple(heatmap, threshold=0.5)
                peak_positions.append(peaks)
            
            # Aggregate statistics
            avg_stats = {}
            for key in heatmap_stats[0].keys():
                values = [s[key] for s in heatmap_stats]
                avg_stats[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
            
            # Expected centers
            expected_centers = []
            for bbox_tensor in gt_bboxes.tensor:
                x1, y1, x2, y2 = bbox_tensor
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                expected_centers.append((cx.item(), cy.item()))
            
            print(f"  Expected Centers: {expected_centers}")
            print(f"  Heatmap Range: {avg_stats['min']['mean']:.3f} to {avg_stats['max']['mean']:.3f}")
            print(f"  Heatmap Mean: {avg_stats['mean']['mean']:.3f} ± {avg_stats['mean']['std']:.3f}")
            print(f"  Active Pixels: {avg_stats['active_pixels']['mean']:.0f} ± {avg_stats['active_pixels']['std']:.0f}")
            print(f"  Peak Consistency: {calculate_peak_consistency(peak_positions, expected_centers)}")
    
    def find_peaks_simple(heatmap, threshold=0.5):
        """Simple peak finding"""
        peaks = []
        h, w = heatmap.shape
        
        # Find local maxima above threshold
        for y in range(1, h-1):
            for x in range(1, w-1):
                if heatmap[y, x] > threshold:
                    # Check if it's a local maximum
                    neighbors = heatmap[y-1:y+2, x-1:x+2]
                    if heatmap[y, x] == neighbors.max():
                        peaks.append((x, y, heatmap[y, x]))
        
        # Sort by intensity
        peaks.sort(key=lambda p: p[2], reverse=True)
        return peaks[:5]  # Top 5 peaks
    
    def calculate_peak_consistency(peak_positions_list, expected_centers):
        """Calculate how consistent peak positions are"""
        if not peak_positions_list or not expected_centers:
            return "No data"
        
        # For each expected center, find closest peak in each generation
        distances = []
        
        for expected_x, expected_y in expected_centers:
            center_distances = []
            
            for peaks in peak_positions_list:
                if not peaks:
                    continue
                
                # Find closest peak to this expected center
                min_dist = float('inf')
                for peak_x, peak_y, intensity in peaks:
                    dist = np.sqrt((peak_x - expected_x)**2 + (peak_y - expected_y)**2)
                    min_dist = min(min_dist, dist)
                
                if min_dist != float('inf'):
                    center_distances.append(min_dist)
            
            if center_distances:
                distances.extend(center_distances)
        
        if distances:
            avg_dist = np.mean(distances)
            return f"Avg distance: {avg_dist:.1f}px"
        else:
            return "No peaks found"
    
    def compare_sigma_values():
        """Compare different sigma values"""
        print(f"\n🎯 SIGMA VALUE ANALYSIS:")
        print("=" * 50)
        
        img = np.ones((320, 320, 3), dtype=np.uint8) * 120
        bbox = HorizontalBoxes(torch.tensor([[140, 140, 180, 180]], dtype=torch.float32))
        
        sigma_values = [10, 15, 20, 25, 30]
        
        for sigma in sigma_values:
            params = current_params.copy()
            params.update({'min_sigma': sigma, 'max_sigma': sigma, 'noise_ratio': 0.0})
            
            transform = RobustHeatmapGeneration(**params)
            
            results = {
                'img': img.copy(),
                'img_shape': img.shape,
                'gt_bboxes': bbox
            }
            
            transformed = transform(results)
            heatmap = transformed['img'][:, :, 3]
            
            # Calculate effective spread
            center_y, center_x = 160, 160  # Known center
            max_val = heatmap[center_y, center_x]
            
            # Count pixels above different thresholds
            above_50 = np.sum(heatmap > 0.5 * max_val)
            above_10 = np.sum(heatmap > 0.1 * max_val)
            
            print(f"  σ={sigma:2d}: Peak={max_val:.3f}, 50%>{above_50:4d}px, 10%>{above_10:4d}px")
    
    # Alternative parameter configurations
    parameter_sets = {
        'current': current_params,
        'conservative': {
            **current_params,
            'center_noise_std': 1.0,
            'max_sigma': 15.0,
            'min_sigma': 12.0,
            'noise_ratio': 0.0,
            'error_ratio': 0.0
        },
        'moderate': {
            **current_params,
            'center_noise_std': 3.0,
            'max_sigma': 25.0,
            'min_sigma': 18.0,
            'noise_ratio': 0.1,
            'error_ratio': 0.02
        }
    }
    
    # Run analysis
    scenarios = create_test_scenarios()
    
    for name, params in parameter_sets.items():
        analyze_parameter_set(name, params, scenarios)
    
    # Sigma analysis
    compare_sigma_values()
    
    print(f"\n🎯 RECOMMENDATIONS:")
    print("=" * 50)
    print("✅ Current parameters are REASONABLE for training:")
    print("   • Sigma 15-20: Good balance of focus vs context")
    print("   • 5% noise: Preserves ground truth while adding robustness")
    print("   • No global noise: Clean training signal")
    print("   • Consistent quality: Predictable behavior")
    print()
    print("🔧 Consider adjustments:")
    print("   • Use 'conservative' for maximum precision")
    print("   • Use 'moderate' for more robustness")
    print("   • Sigma 12-15 for tighter attention")
    print("   • Sigma 20-25 for broader context")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()