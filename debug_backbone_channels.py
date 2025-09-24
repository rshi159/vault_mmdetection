import torch
from mmdet.models.backbones.backbone_4ch import CSPNeXt4Ch

# Test with widen_factor=0.25
backbone = CSPNeXt4Ch(
    arch='P5',
    widen_factor=0.25,
    out_indices=(2, 3, 4)
)

# Get channel counts
print(f"widen_factor: {backbone.widen_factor}")
print(f"base_channels: {backbone.base_channels}")

# Calculate expected channels for each stage
stage_channels = []
for i, stage in enumerate(backbone.stages):
    if hasattr(stage, '__len__') and len(stage) > 1 and hasattr(stage[1], 'out_channels'):
        channels = stage[1].out_channels
        print(f"Stage {i} out_channels: {channels}")
        stage_channels.append(channels)

print(f"\nBackbone output channels: {stage_channels}")
print(f"Expected output indices (2,3,4): {[stage_channels[i] for i in [2,3,4]]}")

# Test with dummy input
x = torch.randn(1, 4, 320, 320)
outputs = backbone(x)
print(f"\nActual output shapes:")
for i, out in enumerate(outputs):
    print(f"Output {i}: {out.shape}")
