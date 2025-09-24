#!/usr/bin/env python3
import torch
from mmdet.registry import MODELS
from mmengine.config import Config

# Import custom modules
import mmdet.models.data_preprocessors.preprocessor_4ch
import mmdet.models.backbones.backbone_4ch
import mmdet.datasets.transforms.fast_4ch
import mmdet.engine.hooks.rgb_4ch_hook

def check_weights():
    # Load config
    cfg = Config.fromfile('configs/rtmdet/rtmdet_4ch_zeroheatmap_fresh.py')
    
    # Build model
    model = MODELS.build(cfg.model)
    
    # Load checkpoint manually
    checkpoint = torch.load(cfg.load_from, map_location='cpu')
    
    print("🔍 Checking checkpoint keys:")
    for key in sorted(checkpoint['state_dict'].keys()):
        if 'conv.weight' in key and 'backbone' in key:
            weight = checkpoint['state_dict'][key]
            if len(weight.shape) == 4 and weight.shape[1] == 4:
                print(f"   • {key}: {weight.shape}, 4th ch norm: {weight[:, 3, :, :].norm().item():.6f}")
    
    # Load checkpoint into model
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    print("\n🔍 Checking model weights after loading:")
    for name, param in model.named_parameters():
        if 'conv.weight' in name and 'backbone' in name:
            if len(param.shape) == 4 and param.shape[1] == 4:
                print(f"   • {name}: {param.shape}, 4th ch norm: {param[:, 3, :, :].norm().item():.6f}")

if __name__ == "__main__":
    check_weights()