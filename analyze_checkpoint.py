#!/usr/bin/env python3
"""Investigate what the checkpoint actually contains"""

import torch
import sys
sys.path.insert(0, '.')

# Load the 4-channel checkpoint and examine its structure
checkpoint_path = './work_dirs/rtmdet_optimized_training/best_coco_bbox_mAP_epoch_195_4ch.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("🔍 Checkpoint Analysis:")
print(f"Keys in checkpoint: {list(checkpoint.keys())}")

if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
    print(f"\nModel parameters in checkpoint:")
    
    # Look at backbone structure
    backbone_keys = [k for k in state_dict.keys() if k.startswith('backbone.stem.0.conv.weight')]
    if backbone_keys:
        first_conv_key = backbone_keys[0]
        first_conv_shape = state_dict[first_conv_key].shape
        print(f"First conv shape: {first_conv_shape}")
        
    # Look for stage output channel patterns
    stage_keys = [k for k in state_dict.keys() if k.startswith('backbone.stage') and 'weight' in k and 'conv' in k]
    print(f"\nStage keys sample:")
    for key in sorted(stage_keys)[:10]:  # First 10 keys
        print(f"  {key}: {state_dict[key].shape}")
        
    # Look at neck input dimensions
    neck_keys = [k for k in state_dict.keys() if k.startswith('neck.') and 'weight' in k]
    print(f"\nNeck keys sample:")
    for key in sorted(neck_keys)[:5]:  # First 5 keys  
        print(f"  {key}: {state_dict[key].shape}")
        
    # Look at bbox head
    bbox_keys = [k for k in state_dict.keys() if k.startswith('bbox_head.') and 'weight' in k]
    print(f"\nBBox head keys sample:")
    for key in sorted(bbox_keys)[:5]:  # First 5 keys
        print(f"  {key}: {state_dict[key].shape}")

    print(f"\n🎯 Analysis Summary:")
    print(f"   Total parameters: {len(state_dict)}")
    
    # Check if this is truly a 4-channel checkpoint or 3-channel
    if 'backbone.stem.0.conv.weight' in state_dict:
        shape = state_dict['backbone.stem.0.conv.weight'].shape
        input_channels = shape[1]
        print(f"   Input channels: {input_channels} ({'4-channel' if input_channels == 4 else '3-channel'})")
        print(f"   First conv: {shape}")

else:
    print("No 'state_dict' key found in checkpoint!")