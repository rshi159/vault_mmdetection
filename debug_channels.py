#!/usr/bin/env python3
"""Debug channel mismatch issue"""

import torch
import sys
sys.path.insert(0, '.')

from mmdet.registry import MODELS
import mmdet.models.backbones.backbone_4ch

# Test the exact backbone configuration from the config
backbone_cfg = dict(
    type='CSPNeXt4Ch',
    arch='P5',
    expand_ratio=0.5,
    deepen_factor=0.167,      # RTMDet-Tiny dimensions
    widen_factor=0.375,       # 0.375 from successful config
    channel_attention=True,
    norm_cfg=dict(type='BN'),
    act_cfg=dict(type='SiLU'),
    init_cfg=None,
    out_indices=(1, 2, 3),
)

print("Building backbone...")
backbone = MODELS.build(backbone_cfg)

print("\nTesting with 4-channel input...")
x = torch.randn(32, 4, 640, 640)  # Batch size 32 like in the error
outputs = backbone(x)

print("\nBackbone outputs:")
for i, out in enumerate(outputs):
    print(f"  Output {i}: {out.shape}")

print("\nExpected neck input channels: [96, 192, 384]")
print("Actual backbone output channels:")
for i, out in enumerate(outputs):
    print(f"  Output {i}: channels = {out.shape[1]}")

# Check if there's a mismatch
expected_channels = [96, 192, 384]
actual_channels = [out.shape[1] for out in outputs]

print(f"\nChannel match analysis:")
for i, (expected, actual) in enumerate(zip(expected_channels, actual_channels)):
    match = "✅" if expected == actual else "❌"
    print(f"  Level {i}: Expected {expected}, Got {actual} {match}")
    
if expected_channels != actual_channels:
    print("\n🚨 MISMATCH DETECTED!")
    print(f"Expected: {expected_channels}")
    print(f"Actual:   {actual_channels}")
    
    # Let's check what configuration would work
    print("\n🔧 DIAGNOSIS:")
    print("The original CSPNeXt architecture might have different channel scaling")
    print("Need to check the original RTMDet configuration or adjust our config")
else:
    print("\n✅ Channels match perfectly!")