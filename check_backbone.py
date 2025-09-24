#!/usr/bin/env python3
import torch

def check_checkpoint():
    # Load checkpoint
    checkpoint = torch.load('./work_dirs/rtmdet_optimized_training/best_coco_bbox_mAP_epoch_195_4ch.pth', map_location='cpu')
    
    print("🔍 Checking all 4-channel conv layers in checkpoint:")
    for key in sorted(checkpoint['state_dict'].keys()):
        if 'conv.weight' in key and 'backbone' in key:
            weight = checkpoint['state_dict'][key]
            if len(weight.shape) == 4 and weight.shape[1] == 4:
                fourth_norm = weight[:, 3, :, :].norm().item()
                first3_norm = weight[:, :3, :, :].norm().item()
                print(f"   • {key}")
                print(f"     Shape: {weight.shape}")
                print(f"     4th ch norm: {fourth_norm:.6f}")
                print(f"     RGB ch norm: {first3_norm:.6f}")
                print(f"     Zeroed?: {'✅' if fourth_norm < 1e-6 else '❌'}")

    print(f"\n🎯 Testing CSPNeXt4Ch backbone directly:")
    
    # Import and build backbone only
    import mmdet.models.backbones.backbone_4ch
    from mmdet.registry import MODELS
    
    backbone_cfg = {
        'type': 'CSPNeXt4Ch',
        'arch': 'P5',
        'expand_ratio': 0.5,
        'deepen_factor': 0.167,
        'widen_factor': 0.375,
        'channel_attention': True,
        'norm_cfg': {'type': 'BN'},
        'act_cfg': {'type': 'SiLU'},
        'out_indices': (1, 2, 3),
    }
    
    backbone = MODELS.build(backbone_cfg)
    
    # Load relevant weights
    backbone_weights = {k.replace('backbone.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('backbone.')}
    backbone.load_state_dict(backbone_weights, strict=False)
    
    # Check first conv - find it dynamically
    print(f"   • Backbone structure: {type(backbone.stem)}")
    print(f"   • Stem modules:")
    for i, module in enumerate(backbone.stem.named_children()):
        print(f"     [{i}] {module[0]}: {type(module[1])}")
    
    # Find first conv layer
    first_conv = None
    for name, module in backbone.named_modules():
        if isinstance(module, torch.nn.Conv2d) and module.in_channels == 4:
            first_conv = module
            print(f"   • Found first 4ch conv: {name}")
            break
    
    if first_conv:
        print(f"   • First conv weight shape: {first_conv.weight.shape}")
        print(f"   • 4th channel norm: {first_conv.weight[:, 3, :, :].norm().item():.6f}")
    else:
        print("   ❌ Could not find 4-channel conv layer")
    
    # Test forward
    dummy_input = torch.randn(1, 4, 640, 640)
    dummy_input[:, 3] = 0  # Zero out 4th channel
    
    features = backbone(dummy_input)
    print(f"   • Forward pass successful")
    print(f"   • Output shapes: {[f.shape for f in features]}")

if __name__ == "__main__":
    check_checkpoint()