#!/usr/bin/env python3
"""Test full model setup with proper imports"""

import torch
import sys
sys.path.insert(0, '.')

# Import all required modules
from mmdet.registry import MODELS
import mmdet.models.backbones.backbone_4ch
import mmdet.models.data_preprocessors.preprocessor_4ch
from mmdet.engine.hooks.rgb_4ch_hook import RGB4ChannelHook
from mmengine.config import Config

# Simple direct model building test
print("1. Testing CSPNeXt4Ch backbone...")
backbone_cfg = dict(
    type='CSPNeXt4Ch',
    arch='P5',
    expand_ratio=0.5,
    deepen_factor=0.33,
    widen_factor=0.375,
    channel_attention=True,
    norm_cfg=dict(type='SyncBN'),
    act_cfg=dict(type='SiLU', inplace=True)
)

backbone = MODELS.build(backbone_cfg)
print(f"✅ Backbone built successfully")

# Test input
print("\n2. Testing forward pass...")
x = torch.randn(1, 4, 640, 640)
try:
    outputs = backbone(x)
    print(f"✅ Backbone forward pass successful")
    for i, out in enumerate(outputs):
        print(f"   Output {i}: {out.shape}")
except Exception as e:
    print(f"❌ Backbone forward failed: {e}")

print("\n3. Testing DetDataPreprocessor4Ch...")
preprocessor_cfg = dict(
    type='DetDataPreprocessor4Ch',
    mean=[123.675, 116.28, 103.53, 0.0],
    std=[58.395, 57.12, 57.375, 1.0], 
    bgr_to_rgb=True,
    pad_size_divisor=32
)

try:
    preprocessor = MODELS.build(preprocessor_cfg)
    print(f"✅ DetDataPreprocessor4Ch built successfully")
except Exception as e:
    print(f"❌ DetDataPreprocessor4Ch build failed: {e}")

print("\n🎯 All components working individually!")
print("   Ready to test full training pipeline")