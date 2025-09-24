#!/usr/bin/env python3
"""
Debug script to test PackDetInputs specifically and see if metainfo flows correctly.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, '/home/robun2/Documents/vault_conveyor_tracking/vault_mmdetection')

from mmdet.datasets.transforms.formatting import PackDetInputs
from mmdet.datasets.transforms.heatmap_transforms import Pad4Channel
import numpy as np

def test_pack_det_inputs():
    """Test if PackDetInputs correctly packs the metadata."""
    
    # Create sample data that would be output by Pad4Channel
    sample_data = {
        'img': np.random.randint(0, 255, (640, 640, 4), dtype=np.uint8),
        'img_shape': (640, 640),
        'ori_shape': (480, 480),
        'scale_factor': 1.33333,
        'img_id': 1,
        'img_path': '/path/to/image.jpg',
        'pad_shape': (640, 640, 4),  # This is what Pad4Channel sets
        'pad_fixed_size': (640, 640),
        'pad_size_divisor': None,
        'gt_bboxes': np.array([[100, 100, 200, 200]], dtype=np.float32),
        'gt_bboxes_labels': np.array([0], dtype=np.int64),
        'instances': []  # Add empty instances for compatibility
    }
    
    print("Testing PackDetInputs transform...")
    print(f"Input keys: {list(sample_data.keys())}")
    
    # Test the transform with all metadata keys
    pack_transform = PackDetInputs(
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'pad_shape', 'pad_fixed_size', 'pad_size_divisor')
    )
    
    try:
        result = pack_transform(sample_data)
        
        print(f"\nOutput keys: {list(result.keys())}")
        
        # Check if we have the expected structure
        if 'data_samples' in result:
            data_sample = result['data_samples']
            print(f"data_sample type: {type(data_sample)}")
            
            # Check metainfo
            if hasattr(data_sample, 'metainfo'):
                metainfo = data_sample.metainfo
                print(f"metainfo keys: {list(metainfo.keys())}")
                
                # Check for our critical metadata
                critical_keys = ['pad_shape', 'pad_fixed_size', 'pad_size_divisor']
                for key in critical_keys:
                    if key in metainfo:
                        print(f"✅ metainfo['{key}']: {metainfo[key]}")
                    else:
                        print(f"❌ metainfo['{key}']: MISSING")
            else:
                print("❌ data_sample has no metainfo attribute")
        else:
            print("❌ No 'data_samples' in result")
            
        return result
        
    except Exception as e:
        print(f"ERROR in PackDetInputs: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_unpack_gt_instances():
    """Test the unpack function that the model uses."""
    
    # First create a data sample using PackDetInputs
    sample_data = {
        'img': np.random.randint(0, 255, (640, 640, 4), dtype=np.uint8),
        'img_shape': (640, 640),
        'ori_shape': (480, 480),
        'scale_factor': 1.33333,
        'img_id': 1,
        'img_path': '/path/to/image.jpg',
        'pad_shape': (640, 640, 4),
        'pad_fixed_size': (640, 640),
        'pad_size_divisor': None,
        'gt_bboxes': np.array([[100, 100, 200, 200]], dtype=np.float32),
        'gt_bboxes_labels': np.array([0], dtype=np.int64),
        'instances': []
    }
    
    pack_transform = PackDetInputs(
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'pad_shape', 'pad_fixed_size', 'pad_size_divisor')
    )
    
    packed_result = pack_transform(sample_data)
    
    if 'data_samples' in packed_result:
        # Test unpack_gt_instances like the model does
        try:
            from mmdet.models.utils.misc import unpack_gt_instances
            
            batch_data_samples = [packed_result['data_samples']]
            
            print("\n\nTesting unpack_gt_instances (used by model)...")
            outputs = unpack_gt_instances(batch_data_samples)
            batch_gt_instances, batch_gt_instances_ignore, batch_img_metas = outputs
            
            print(f"batch_img_metas type: {type(batch_img_metas)}")
            print(f"Number of img_metas: {len(batch_img_metas)}")
            
            if len(batch_img_metas) > 0:
                img_meta = batch_img_metas[0]
                print(f"img_meta type: {type(img_meta)}")
                print(f"img_meta keys: {list(img_meta.keys())}")
                
                # Check for pad_shape
                if 'pad_shape' in img_meta:
                    print(f"✅ img_meta['pad_shape']: {img_meta['pad_shape']}")
                else:
                    print("❌ img_meta['pad_shape']: MISSING")
                    print("Available keys:", list(img_meta.keys()))
                    
        except Exception as e:
            print(f"ERROR in unpack_gt_instances: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_pack_det_inputs()
    test_unpack_gt_instances()