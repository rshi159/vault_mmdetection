#!/usr/bin/env python3

import sys
import os.path as osp

# Add repository root to Python path
repo_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from mmdet.datasets import CocoDataset

# Test samples with annotations
data_root = '/home/robun2/Documents/vault_conveyor_tracking/vault_mmdetection/development/augmented_data_production/'

dataset = CocoDataset(
    data_root=data_root,
    ann_file='train/annotations.json',
    data_prefix=dict(img='train/images/'),
    test_mode=False,
    pipeline=[],
    metainfo=dict(classes=('conveyor_object',))
)

print(f"Total samples: {len(dataset)}")

# Check first 10 samples for instances
samples_with_instances = 0
samples_without_instances = 0

for i in range(min(100, len(dataset))):
    sample = dataset.get_data_info(i)
    if len(sample['instances']) > 0:
        samples_with_instances += 1
        if samples_with_instances == 1:
            print(f"✅ Sample {i} has {len(sample['instances'])} instances")
            print(f"   First instance: {sample['instances'][0]}")
    else:
        samples_without_instances += 1

print(f"\nIn first 100 samples:")
print(f"Samples with instances: {samples_with_instances}")
print(f"Samples without instances: {samples_without_instances}")

# Check if filter_empty_gt helps
if samples_without_instances > 0:
    print(f"\n⚠️ Found {samples_without_instances} samples without annotations!")
    print("This might cause issues when filter_empty_gt=True is used")