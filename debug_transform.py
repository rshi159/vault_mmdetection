#!/usr/bin/env python3
"""
Debug script to test the Pad4Channel transform specifically to see if it sets metadata.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, '/home/robun2/Documents/vault_conveyor_tracking/vault_mmdetection')

from mmdet.datasets.transforms.heatmap_transforms import Pad4Channel
import numpy as np

def test_pad4channel():
    """Test if Pad4Channel is setting the metadata keys."""
    
    # Create a sample data dict like MMDetection would have after other transforms
    sample_data = {
        'img': np.random.randint(0, 255, (640, 640, 4), dtype=np.uint8),  # 4-channel image
        'img_shape': (640, 640),
        'ori_shape': (480, 480),
        'scale_factor': 1.33333,
        'img_id': 1,
        'img_path': '/path/to/image.jpg'
    }
    
    print("Testing Pad4Channel transform...")
    print(f"Input keys: {list(sample_data.keys())}")
    print(f"Input image shape: {sample_data['img'].shape}")
    
    # Create the transform
    pad_transform = Pad4Channel(size=(640, 640), pad_val=dict(img=(114, 114, 114, 0)))
    
    try:
        # Apply the transform
        result = pad_transform(sample_data)
        
        print(f"\nOutput keys: {list(result.keys())}")
        print(f"Output image shape: {result['img'].shape}")
        
        # Check for metadata keys we're looking for
        metadata_keys = ['pad_shape', 'pad_fixed_size', 'pad_size_divisor']
        for key in metadata_keys:
            if key in result:
                print(f"✅ {key}: {result[key]}")
            else:
                print(f"❌ {key}: MISSING")
        
        return result
        
    except Exception as e:
        print(f"ERROR in Pad4Channel: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_standard_pad():
    """Test what standard Pad transform does for comparison."""
    try:
        from mmcv.transforms import Pad
        
        sample_data = {
            'img': np.random.randint(0, 255, (480, 480, 3), dtype=np.uint8),
            'img_shape': (480, 480),
            'ori_shape': (480, 480),
        }
        
        print("\n\nTesting standard Pad transform for comparison...")
        pad_transform = Pad(size=(640, 640))
        result = pad_transform(sample_data)
        
        print(f"Standard Pad output keys: {list(result.keys())}")
        metadata_keys = ['pad_shape', 'pad_fixed_size', 'pad_size_divisor']
        for key in metadata_keys:
            if key in result:
                print(f"✅ Standard Pad {key}: {result[key]}")
            else:
                print(f"❌ Standard Pad {key}: MISSING")
                
    except Exception as e:
        print(f"Couldn't test standard Pad: {e}")

if __name__ == "__main__":
    test_pad4channel()
    test_standard_pad()