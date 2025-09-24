#!/usr/bin/env python3

import sys
import os.path as osp

# Add repository root to Python path
repo_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
    print(f"✅ Added repository root to Python path: {repo_root}")

from mmdet.datasets import CocoDataset

# Test basic COCO dataset loading
data_root = '/home/robun2/Documents/vault_conveyor_tracking/vault_mmdetection/development/augmented_data_production/'

print("Testing COCO dataset loading...")

try:
    dataset = CocoDataset(
        data_root=data_root,
        ann_file='train/annotations.json',
        data_prefix=dict(img='train/images/'),
        test_mode=False,
        pipeline=[],  # Empty pipeline for testing
        metainfo=dict(classes=('conveyor_object',))
    )
    
    print(f"✅ Dataset loaded successfully!")
    print(f"Number of samples: {len(dataset)}")
    
    if len(dataset) > 0:
        print("✅ Dataset has samples")
        # Try to get the first sample
        sample = dataset.get_data_info(0)
        print(f"First sample: {sample}")
    else:
        print("❌ Dataset is empty!")
        
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    import traceback
    traceback.print_exc()