#!/usr/bin/env python3

"""
Fixed training script with proper 4th channel handling
"""

import os
import warnings
import torch
import sys

# Suppress known warnings
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
warnings.filterwarnings('ignore', category=UserWarning, module='torch.functional')

from mmdet.apis import init_detector
from mmengine.runner import Runner
from mmengine.config import Config

def freeze_4th_channel(model):
    """
    Properly freeze the 4th channel of the first conv layer
    """
    try:
        # Access the first conv layer in the backbone
        conv = model.backbone.stem[0].conv
        
        if conv.weight.shape[1] >= 4:
            print("✅ Freezing 4th channel weights...")
            
            # Zero out the 4th channel weights
            with torch.no_grad():
                conv.weight[:, 3].zero_()
            
            # Register hook to keep 4th channel gradients at zero
            def _zero_grad_ch4(grad):
                if grad is not None and grad.shape[1] >= 4:
                    grad[:, 3] = 0
                return grad
            
            conv.weight.register_hook(_zero_grad_ch4)
            
            print(f"   4th channel frozen for layer: {conv}")
            print(f"   Weight shape: {conv.weight.shape}")
            
            # Verify it's zeroed
            ch4_norm = conv.weight[:, 3].norm().item()
            print(f"   4th channel norm: {ch4_norm:.6f}")
            
            return True
    except Exception as e:
        print(f"❌ Could not freeze 4th channel: {e}")
        return False

def main():
    """Main training function with proper setup"""
    
    print("🚀 Starting RTMDet 4-Channel Training (Fixed Pipeline)...")
    
    # Load config
    config_file = 'configs/rtmdet/rtmdet_4ch_zeroheatmap_fresh.py'
    cfg = Config.fromfile(config_file)
    
    print(f"📋 Config loaded: {config_file}")
    print(f"🎯 Loading checkpoint: {cfg.load_from}")
    print(f"📁 Work directory: {cfg.work_dir}")
    
    # Initialize runner
    runner = Runner.from_cfg(cfg)
    
    # Freeze 4th channel after model initialization
    print("\n🔧 Setting up 4th channel freezing...")
    freeze_success = freeze_4th_channel(runner.model)
    
    if freeze_success:
        print("✅ 4th channel properly frozen")
    else:
        print("⚠️  Warning: Could not freeze 4th channel")
    
    print("\n🏃 Starting training...")
    
    # Start training
    runner.train()
    
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)