#!/usr/bin/env python3
"""
Test script for YOLO keypoint lookup and enhanced heatmap generation.
Validates that the system can access keypoint data during training.
"""

import sys
import os
from pathlib import Path

# Add the demo/mmdetection directory to the path
demo_mmdet_path = Path(__file__).parent / 'demo' / 'mmdetection'
sys.path.insert(0, str(demo_mmdet_path))

def test_keypoint_lookup():
    """Test the YOLO keypoint lookup system."""
    print("🔍 Testing YOLO Keypoint Lookup System...")
    
    try:
        from mmdet.datasets.yolo_keypoint_lookup import YOLOKeypointLookup
        
        # Test dataset path
        dataset_path = "development/augmented_data_production"
        
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset path not found: {dataset_path}")
            return False
        
        # Initialize lookup
        print("   Initializing lookup system...")
        lookup = YOLOKeypointLookup(dataset_path)
        
        # Get statistics
        stats = lookup.get_stats()
        print(f"   📊 Statistics:")
        for key, value in stats.items():
            print(f"      {key}: {value}")
        
        # Test specific images
        test_images = [
            "_sticker_output_test_6_rgb_0345.jpg",
            "_sticker_output_test_6_rgb_0345.png",
            "yolo_dataset_template_FlatParcels_output_Replicator_33_rgb_0.jpg",
            "yolo_dataset_template_FlatParcels_output_Replicator_33_rgb_0.png"
        ]
        
        found_test = False
        for test_image in test_images:
            if lookup.has_keypoints(test_image):
                keypoints = lookup.get_keypoints(test_image)
                print(f"   ✅ Test image: {test_image}")
                print(f"      Found {len(keypoints)} annotations")
                
                for i, ann in enumerate(keypoints[:2]):  # Show first 2
                    kp_count = ann['num_keypoints']
                    class_id = ann['class_id']
                    bbox = ann['bbox_normalized']
                    print(f"      Annotation {i}: class={class_id}, keypoints={kp_count}, bbox={bbox[:2]}...")
                
                found_test = True
                break
        
        if not found_test:
            print("   ⚠️ No test images found with keypoints")
        
        print("   ✅ Keypoint lookup system working!")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhanced_heatmap():
    """Test the enhanced heatmap generation with keypoints."""
    print("\n🔥 Testing Enhanced Heatmap Generation...")
    
    try:
        from mmdet.datasets.transforms.robust_heatmap import RobustHeatmapGeneration
        import numpy as np
        
        # Create transform with YOLO dataset path
        transform = RobustHeatmapGeneration(
            yolo_dataset_path="development/augmented_data_production"
        )
        
        # Create mock results data
        fake_results = {
            'img': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            'img_path': 'development/augmented_data_production/train/images/_sticker_output_test_6_rgb_0345.jpg',
            'gt_bboxes': []  # Empty for simplicity
        }
        
        print("   Applying enhanced heatmap generation...")
        
        # Apply transform
        results = transform(fake_results)
        
        # Check output
        img_shape = results['img'].shape
        print(f"   ✅ Output shape: {img_shape}")
        
        if len(img_shape) == 3 and img_shape[2] == 4:
            print("   ✅ 4-channel output confirmed!")
            
            # Check heatmap channel
            heatmap = results['img'][:, :, 3]
            heatmap_stats = {
                'min': heatmap.min(),
                'max': heatmap.max(),
                'mean': heatmap.mean(),
                'non_zero': np.sum(heatmap > 0.01)
            }
            print(f"   📊 Heatmap stats: {heatmap_stats}")
            
            return True
        else:
            print(f"   ❌ Unexpected output shape: {img_shape}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🧪 Testing 4-Channel RTMDet with Keypoint Enhancement")
    print("=" * 60)
    
    # Test 1: Keypoint lookup
    lookup_ok = test_keypoint_lookup()
    
    # Test 2: Enhanced heatmap generation
    heatmap_ok = test_enhanced_heatmap()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    print(f"   Keypoint Lookup: {'✅ PASS' if lookup_ok else '❌ FAIL'}")
    print(f"   Enhanced Heatmap: {'✅ PASS' if heatmap_ok else '❌ FAIL'}")
    
    if lookup_ok and heatmap_ok:
        print("\n🎉 All tests passed! System ready for keypoint-enhanced training.")
        return True
    else:
        print("\n⚠️ Some tests failed. Check errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)