#!/usr/bin/env python3
"""Find the right parameters to match checkpoint dimensions"""

import torch
import sys
sys.path.insert(0, '.')

from mmdet.registry import MODELS
import mmdet.models.backbones.backbone_4ch

def test_backbone_config(deepen_factor, widen_factor):
    """Test backbone configuration and return output channels"""
    backbone_cfg = dict(
        type='CSPNeXt4Ch',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        channel_attention=True,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=None,
        out_indices=(1, 2, 3),
    )
    
    backbone = MODELS.build(backbone_cfg)
    x = torch.randn(1, 4, 640, 640)
    outputs = backbone(x)
    channels = [out.shape[1] for out in outputs]
    
    return channels

# Target channels from successful checkpoint
target_channels = [96, 192, 384]

print("Testing different widen_factor values:")
print(f"Target: {target_channels}")
print()

# Test various widen_factor values with deepen_factor=0.167
test_configs = [
    (0.167, 0.375),  # Current failing config
    (0.167, 0.5),    # RTMDet-S config  
    (0.167, 0.75),   # 2x our current
    (0.167, 1.0),    # Even larger
    (0.33, 0.375),   # RTMDet-S deepen with our widen
    (0.33, 0.5),     # RTMDet-S config
]

for deepen, widen in test_configs:
    try:
        channels = test_backbone_config(deepen, widen)
        match = "✅" if channels == target_channels else "❌"
        print(f"deepen={deepen:>5}, widen={widen:>4} → {channels} {match}")
    except Exception as e:
        print(f"deepen={deepen:>5}, widen={widen:>4} → ERROR: {e}")

print(f"\n🎯 Need to find config that outputs: {target_channels}")