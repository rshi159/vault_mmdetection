#!/usr/bin/env python3
"""Find the EXACT configuration that matches the checkpoint"""

import torch
import sys
sys.path.insert(0, '.')

from mmdet.registry import MODELS
from mmengine.config import Config
import mmdet.models.backbones.backbone_4ch

# Load the checkpoint
checkpoint = torch.load('./work_dirs/rtmdet_optimized_training/best_coco_bbox_mAP_epoch_195_4ch.pth', map_location='cpu')
checkpoint_state = checkpoint['state_dict']

def test_full_model_match(deepen_factor, widen_factor, neck_in_channels, neck_out_channels, bbox_channels):
    """Test if a full model configuration matches the checkpoint"""
    
    # Build the exact config
    cfg_dict = {
        'type': 'RTMDet',
        'data_preprocessor': {
            'type': 'DetDataPreprocessor4Ch',
            'mean': [123.675, 116.28, 103.53, 0.0],
            'std': [58.395, 57.12, 57.375, 1.0],
            'bgr_to_rgb': True,
            'pad_size_divisor': 32
        },
        'backbone': {
            'type': 'CSPNeXt4Ch',
            'arch': 'P5',
            'expand_ratio': 0.5,
            'deepen_factor': deepen_factor,
            'widen_factor': widen_factor,
            'channel_attention': True,
            'norm_cfg': {'type': 'BN'},
            'act_cfg': {'type': 'SiLU'},
            'out_indices': (1, 2, 3),
        },
        'neck': {
            'type': 'CSPNeXtPAFPN',
            'in_channels': neck_in_channels,
            'out_channels': neck_out_channels,
            'num_csp_blocks': 1,
            'expand_ratio': 0.5,
            'norm_cfg': {'type': 'BN'},
            'act_cfg': {'type': 'SiLU'}
        },
        'bbox_head': {
            'type': 'RTMDetHead',
            'num_classes': 1,
            'in_channels': bbox_channels,
            'feat_channels':bbox_channels,
            'stacked_convs': 2,
            'anchor_generator': {'type': 'MlvlPointGenerator', 'offset': 0, 'strides': [8, 16, 32]},
            'bbox_coder': {'type': 'DistancePointBBoxCoder'},
            'loss_cls': {'type': 'QualityFocalLoss', 'use_sigmoid': True, 'beta': 2.0, 'loss_weight': 1.0},
            'loss_bbox': {'type': 'CIoULoss', 'loss_weight': 2.0},
            'norm_cfg': {'type': 'BN'},
            'act_cfg': {'type': 'SiLU'}
        }
    }
    
    try:
        model = MODELS.build(cfg_dict)
        model_state = model.state_dict()
        
        # Count matching parameters
        matches = 0
        mismatches = 0
        total_ckpt_params = 0
        
        for key, ckpt_param in checkpoint_state.items():
            total_ckpt_params += 1
            if key in model_state:
                model_param = model_state[key]
                if ckpt_param.shape == model_param.shape:
                    matches += 1
                else:
                    mismatches += 1
                    # Print first few mismatches for debugging
                    if mismatches <= 3:
                        print(f"    MISMATCH {key}: ckpt {ckpt_param.shape} vs model {model_param.shape}")
            else:
                mismatches += 1
                if mismatches <= 3:
                    print(f"    MISSING {key} in model")
        
        match_rate = matches / total_ckpt_params * 100
        print(f"  deepen={deepen_factor}, widen={widen_factor}, neck_in={neck_in_channels}, neck_out={neck_out_channels}, bbox={bbox_channels}")
        print(f"    Matches: {matches}/{total_ckpt_params} ({match_rate:.1f}%), Mismatches: {mismatches}")
        
        return match_rate > 95  # Consider >95% match as success
        
    except Exception as e:
        print(f"  ERROR with deepen={deepen_factor}, widen={widen_factor}: {e}")
        return False

print("🔍 Finding exact configuration match...")
print("Testing configurations:")

# Test different combinations based on our analysis
configs_to_test = [
    # deepen, widen, neck_in, neck_out, bbox
    (0.167, 0.375, [48, 96, 192], 48, 48),   # Small model matching backbone output
    (0.167, 0.375, [96, 192, 384], 96, 96),  # Original failing config
    (0.33, 0.5, [48, 96, 192], 48, 48),      # RTMDet-S with small channels
    (0.33, 0.5, [96, 192, 384], 96, 96),     # RTMDet-S with large channels  
]

best_match = None
best_rate = 0

for config in configs_to_test:
    print(f"\nTesting: {config}")
    is_match = test_full_model_match(*config)
    if is_match:
        print(f"  ✅ PERFECT MATCH FOUND!")
        best_match = config
        break

if best_match:
    print(f"\n🎯 EXACT CONFIGURATION: {best_match}")
else:
    print(f"\n❌ No perfect match found. Need to investigate further.")