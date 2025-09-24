#!/usr/bin/env python
"""
Simple training script for 4-channel RTMDet without hooks
Uses RGBOnly4Channel transform in pipeline to handle 4th channel zeroing
"""

import os
import sys
import warnings
import torch

# Suppress known warnings
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
warnings.filterwarnings('ignore', category=UserWarning, module='torch.functional')

from mmdet.apis import init_detector
from mmengine.runner import Runner
from mmengine.config import Config

def main():
    """
    Main training function - simplified without gradient hooks
    """
    print("🚀 Starting 4-channel RTMDet training (simplified)")
    print("   • 4th channel zeroing handled by RGBOnly4Channel transform")
    print("   • No gradient hooks needed")
    print()
    
    # Load config
    config_path = './configs/rtmdet/rtmdet_4ch_zeroheatmap_fresh.py'
    cfg = Config.fromfile(config_path)
    
    # Initialize the runner and start training
    try:
        print("📋 Building runner from config...")
        runner = Runner.from_cfg(cfg)
        
        print("▶️  Starting training...")
        print("=" * 60)
        runner.train()
        
    except Exception as error:
        print(f"\n💥 Training failed: {error}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n✅ Training completed successfully!")
    return 0

if __name__ == '__main__':
    sys.exit(main())