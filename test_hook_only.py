#!/usr/bin/env python3
"""Test the RGB4ChannelHook in isolation"""

import torch
from mmdet.registry import MODELS, RUNNERS, HOOKS
from mmengine.config import Config
import mmdet.models.backbones.backbone_4ch
import mmdet.models.data_preprocessors.preprocessor_4ch
from mmdet.engine.hooks.rgb_4ch_hook import RGB4ChannelHook

# Test the backbone + hook
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

print("1. Building backbone...")
backbone = MODELS.build(backbone_cfg)

# Check first conv weight
first_conv = backbone.stem[0].conv
print(f"First conv weight shape: {first_conv.weight.shape}")
print(f"4th channel norm before hook: {first_conv.weight[:, 3, :, :].norm():.6f}")

print("\n2. Creating hook...")
hook = RGB4ChannelHook(
    zero_4th_channel=True,
    freeze_4th_channel=True,
    monitor_weights=True,
    log_interval=500,
    first_conv_name='backbone.stem.0.conv'
)

print("\n3. Simulating model loading hook...")
# Create a fake runner with model
class FakeRunner:
    def __init__(self, model):
        self.model = model

runner = FakeRunner(backbone)

# Call the hook's after_load_checkpoint method
# First set the correct conv name for standalone backbone
hook.first_conv_name = 'stem.0.conv'
hook.after_load_checkpoint(runner, checkpoint=None)

print(f"4th channel norm after hook: {first_conv.weight[:, 3, :, :].norm():.6f}")

# Check if it's actually zeroed
if first_conv.weight[:, 3, :, :].norm() < 1e-6:
    print("✅ SUCCESS: 4th channel is properly zeroed!")
else:
    print("❌ FAILED: 4th channel is not zeroed")
    
print(f"RGB channels norm: {first_conv.weight[:, :3, :, :].norm():.6f}")