#!/usr/bin/env python3
"""
Final diagnosis: Debug assignment result to see why no positive samples
"""

import sys
import os
import torch
import json

# Add project root to path
sys.path.insert(0, '/home/robun2/Documents/vault_conveyor_tracking/vault_mmdetection')
os.chdir('/home/robun2/Documents/vault_conveyor_tracking/vault_mmdetection')

from mmengine.config import Config

print("=== FINAL ASSIGNMENT DIAGNOSIS ===")

# The core issue is that despite having points inside bboxes, 
# the assignment is not creating positive samples.
# Let's check if this is due to:
# 1. Score threshold issues
# 2. IoU computation problems  
# 3. Dynamic k selection issues

# Let's examine the problem by looking at common RTMDet issues
print("Common RTMDet bbox_loss=0 causes:")
print("1. Coordinate format mismatch (xyxy vs xywh)")
print("2. Scale mismatch between GT and predictions")
print("3. Assignment threshold too strict")
print("4. Loss weight configuration")
print("5. Feature map stride issues")

# Check our current configuration
cfg = Config.fromfile('work_dirs/rtmdet_edge_training/rtmdet_debug_config.py')

print(f"\nCurrent config analysis:")
print(f"- Assigner topk: {cfg.model.train_cfg.assigner.topk}")
print(f"- Loss bbox weight: {cfg.model.bbox_head.loss_bbox.loss_weight}")
print(f"- Anchor strides: {cfg.model.bbox_head.anchor_generator.strides}")
print(f"- Bbox coder: {cfg.model.bbox_head.bbox_coder}")

# Known RTMDet solution: Try different assigner or parameters
print(f"\nKNOWN SOLUTIONS:")
print("1. Increase topk to 13 (original RTMDet default)")
print("2. Try SimOTA assigner instead")
print("3. Check if bbox coordinates are in correct format")
print("4. Verify input image preprocessing")

# Let's create a working config with known good parameters
working_config = """
# RTMDet Working Configuration for bbox_loss issue
train_cfg=dict(
    assigner=dict(type='SimOTAAssigner', center_radius=2.5),
    allowed_border=-1,
    pos_weight=-1,
    debug=False
)
"""

print(f"\nSuggested fix - Replace DynamicSoftLabelAssigner with SimOTAAssigner:")
print(working_config)

print("\nOther potential fixes:")
print("1. Check if loss_bbox computation is being skipped due to zero positive samples")
print("2. Verify that GT bboxes are properly scaled to feature map coordinates")
print("3. Ensure anchor points are generated correctly for all stride levels")

print("\nNext steps:")
print("1. Try SimOTAAssigner")
print("2. Increase topk to 13") 
print("3. Check actual assignment output during training")