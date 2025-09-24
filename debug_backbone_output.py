import sys
sys.path.insert(0, '/home/robun2/Documents/vault_conveyor_tracking/vault_mmdetection')

import torch
from mmdet.models.backbones.backbone_4ch import CSPNeXt4Ch

# Test the actual backbone configuration
backbone = CSPNeXt4Ch(
    arch='P5',
    widen_factor=0.25,
    deepen_factor=0.167,
    out_indices=(2, 3, 4),
    norm_cfg=dict(type='SyncBN'),
    act_cfg=dict(type='SiLU', inplace=True),
)

print(f"widen_factor: {backbone.widen_factor}")
print(f"out_indices: {backbone.out_indices}")

# Calculate the channels like the backbone does
base_channels = int(256 * backbone.widen_factor)  # = 64
channels = [base_channels * (2 ** i) for i in range(5)]  # [64, 128, 256, 512, 1024]
print(f"base_channels: {base_channels}")
print(f"all_channels: {channels}")
print(f"output_indices channels: {[channels[i] for i in backbone.out_indices]}")

# Test with dummy input
x = torch.randn(1, 4, 320, 320)
outputs = backbone(x)
print(f"\nActual outputs:")
print(f"Number of outputs: {len(outputs)}")
for i, out in enumerate(outputs):
    print(f"Output {i}: {out.shape}")
