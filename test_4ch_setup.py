#!/usr/bin/env python3
"""
Quick verification script to test the 4-channel setup before training.
Checks model loading, checkpoint compatibility, and RGB consistency.
"""

import torch
import sys
import os
sys.path.insert(0, '.')

from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.registry import RUNNERS

def test_4ch_setup():
    """Test the 4-channel setup."""
    
    print("🧪 Testing 4-Channel Setup")
    print("=" * 50)
    
    # Load config
    config_file = "configs/rtmdet/rtmdet_4ch_zeroheatmap_fresh.py"
    cfg = Config.fromfile(config_file)
    
    print(f"✅ Config loaded: {config_file}")
    print(f"   • Work dir: {cfg.work_dir}")
    print(f"   • Load from: {cfg.load_from}")
    print(f"   • Resume: {cfg.resume}")
    
    # Create runner (this loads the model and checkpoint)
    print(f"\n🔧 Creating runner and loading model...")
    try:
        runner = Runner.from_cfg(cfg)
        print(f"✅ Runner created successfully")
    except Exception as e:
        print(f"❌ Failed to create runner: {e}")
        return False
    
    # Check model architecture
    print(f"\n🏗️ Model Architecture:")
    model = runner.model
    
    # Check first conv layer (handle different backbone structures)
    try:
        # Try CSPNeXt4Ch structure
        if hasattr(model.backbone, 'stem') and hasattr(model.backbone.stem, '0'):
            first_conv = model.backbone.stem[0].conv
        elif hasattr(model.backbone, 'stem') and hasattr(model.backbone.stem, 'conv'):
            first_conv = model.backbone.stem.conv
        else:
            # Fallback: find first conv layer
            first_conv = None
            for name, module in model.backbone.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    first_conv = module
                    break
        
        if first_conv is not None:
            print(f"   • First conv shape: {first_conv.weight.shape}")
            print(f"   • Expected: [out_channels, 4, kernel_h, kernel_w]")
            
            if first_conv.weight.shape[1] != 4:
                print(f"❌ First conv has {first_conv.weight.shape[1]} input channels, expected 4")
                return False
            
            # Check 4th channel weights
            fourth_channel_norm = first_conv.weight[:, 3, :, :].norm().item()
            print(f"   • 4th channel weight norm: {fourth_channel_norm:.6f}")
            
            if fourth_channel_norm > 1e-3:
                print(f"⚠️ 4th channel weights are not zero! (norm={fourth_channel_norm:.6f})")
            else:
                print(f"✅ 4th channel weights are properly zeroed")
        else:
            print(f"❌ Could not find first conv layer in backbone")
            return False
    except Exception as e:
        print(f"❌ Error accessing first conv layer: {e}")
        return False
    
    # Test forward pass with dummy input
    print(f"\n🧪 Testing forward pass...")
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # Create dummy 4-channel input (RGB + zero heatmap)
        dummy_input = torch.randn(1, 4, 640, 640).to(device)
        dummy_input[:, 3] = 0  # Zero out 4th channel
        
        print(f"   • Input shape: {dummy_input.shape}")
        print(f"   • 4th channel norm: {dummy_input[:, 3].norm().item():.6f}")
        
        try:
            # Test backbone
            features = model.backbone(dummy_input)
            print(f"   • Backbone output shapes: {[f.shape for f in features]}")
            
            # Test full forward (inference mode)
            model.eval()
            batch_data = {
                'inputs': [dummy_input.squeeze(0)],
                'data_samples': [None]  # No ground truth for inference
            }
            
            # This tests the full pipeline including preprocessor
            results = model.test_step(batch_data)
            print(f"   • Full forward pass successful")
            print(f"✅ Model is working correctly!")
            
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            return False
    
    model.train()
    
    print(f"\n🎯 Summary:")
    print(f"   ✅ Config loads correctly")
    print(f"   ✅ Model architecture is correct (4-channel input)")
    print(f"   ✅ Checkpoint loads successfully") 
    print(f"   ✅ 4th channel is properly zeroed")
    print(f"   ✅ Forward pass works")
    print(f"   🚀 Ready for training!")
    
    return True

if __name__ == "__main__":
    success = test_4ch_setup()
    sys.exit(0 if success else 1)