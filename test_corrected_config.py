#!/usr/bin/env python3
"""Test the full model configuration"""

import torch
import sys
sys.path.insert(0, '.')

from mmdet.registry import MODELS
from mmengine.config import Config
import mmdet.models.backbones.backbone_4ch
import mmdet.models.data_preprocessors.preprocessor_4ch

# Load the corrected config
cfg = Config.fromfile('configs/rtmdet/rtmdet_4ch_zeroheatmap_fresh.py')

print("Testing corrected configuration...")
print(f"Backbone config: deepen_factor={cfg.model.backbone.deepen_factor}, widen_factor={cfg.model.backbone.widen_factor}")
print(f"Neck input channels: {cfg.model.neck.in_channels}")
print(f"Neck output channels: {cfg.model.neck.out_channels}")
print(f"BBox head input channels: {cfg.model.bbox_head.in_channels}")

# Test backbone
backbone = MODELS.build(cfg.model.backbone)
x = torch.randn(2, 4, 640, 640)
backbone_outputs = backbone(x)

print(f"\nBackbone outputs:")
for i, out in enumerate(backbone_outputs):
    print(f"  Level {i}: {out.shape}")

# Test neck
neck = MODELS.build(cfg.model.neck)
neck_outputs = neck(backbone_outputs)

print(f"\nNeck outputs:")
for i, out in enumerate(neck_outputs):
    print(f"  Level {i}: {out.shape}")

print(f"\n✅ Configuration test successful!")
print(f"   Backbone → Neck: {[out.shape[1] for out in backbone_outputs]} → {cfg.model.neck.in_channels}")
print(f"   Neck → BBox: {neck_outputs[0].shape[1]} → {cfg.model.bbox_head.in_channels}")

# Quick dimensionality check
expected_neck_in = cfg.model.neck.in_channels
actual_backbone_out = [out.shape[1] for out in backbone_outputs]
expected_bbox_in = cfg.model.bbox_head.in_channels
actual_neck_out = neck_outputs[0].shape[1]

if expected_neck_in == actual_backbone_out and expected_bbox_in == actual_neck_out:
    print("🎯 All dimensions match perfectly!")
else:
    print("❌ Dimension mismatch detected")