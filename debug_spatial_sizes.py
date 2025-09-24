import sys
sys.path.insert(0, '/home/robun2/Documents/vault_conveyor_tracking/vault_mmdetection')

import torch
from mmdet.models.backbones.backbone_4ch import CSPNeXt4Ch

# Test the backbone with actual training input size
backbone = CSPNeXt4Ch(
    arch='P5',
    widen_factor=0.25,
    deepen_factor=0.167,
    out_indices=(1, 2, 3),
    norm_cfg=dict(type='SyncBN'),
    act_cfg=dict(type='SiLU', inplace=True),
)

# Test with the training input size (320x320 after resize and pad)
x = torch.randn(1, 4, 320, 320)
outputs = backbone(x)

print(f"Input shape: {x.shape}")
print(f"Number of outputs: {len(outputs)}")
for i, out in enumerate(outputs):
    stage_idx = [1, 2, 3][i]
    downscale = 2 ** (stage_idx + 1)  # Stage 1 = 2^2=4x, Stage 2 = 2^3=8x, Stage 3 = 2^4=16x
    expected_size = 320 // downscale
    print(f"Output {i} (stage {stage_idx}): {out.shape}, expected spatial: {expected_size}x{expected_size}")
