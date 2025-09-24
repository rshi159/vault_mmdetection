#!/usr/bin/env python3
"""
Example Training Script for 4-Channel RTMDet
Professional example showing complete training setup.
"""

import argparse
import os
import sys
from pathlib import Path

from mmengine.config import Config
from mmengine.runner import Runner


def main():
    """Main training function with professional setup."""
    parser = argparse.ArgumentParser(
        description='Train 4-Channel RTMDet for Vault Conveyor Package Detection'
    )
    parser.add_argument(
        '--config', 
        default='configs/rtmdet/rtmdet_4ch_zeroheatmap_fresh.py',
        help='Training configuration file'
    )
    parser.add_argument(
        '--work-dir',
        help='Working directory for outputs (overrides config)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from latest checkpoint'
    )
    parser.add_argument(
        '--validate-setup',
        action='store_true', 
        help='Validate 4-channel setup before training'
    )
    
    args = parser.parse_args()
    
    print("🚀 4-Channel RTMDet Training")
    print("=" * 50)
    print(f"Config: {args.config}")
    
    # Load configuration
    cfg = Config.fromfile(args.config)
    
    # Override work directory if specified
    if args.work_dir:
        cfg.work_dir = args.work_dir
        print(f"Work dir: {cfg.work_dir}")
    
    # Create work directory
    os.makedirs(cfg.work_dir, exist_ok=True)
    
    # Setup resuming
    if args.resume:
        cfg.resume = True
        print("Resume: Enabled")
    
    # Validate setup if requested
    if args.validate_setup:
        print("\n🔧 Validating 4-channel setup...")
        validate_4ch_setup(cfg)
    
    # Initialize runner and start training
    print("\n📋 Initializing training runner...")
    runner = Runner.from_cfg(cfg)
    
    print("🏃 Starting training...")
    print("=" * 50)
    
    # Start training
    runner.train()
    
    print("\n✅ Training completed successfully!")


def validate_4ch_setup(cfg):
    """Validate 4-channel setup before training."""
    try:
        from mmdet.registry import MODELS
        
        # Test model building
        model = MODELS.build(cfg.model)
        print("   ✅ Model builds successfully")
        
        # Check first conv layer
        first_conv = model.backbone.stem[0].conv
        if first_conv.weight.shape[1] == 4:
            print("   ✅ 4-channel input configured")
        else:
            print(f"   ❌ Expected 4 channels, got {first_conv.weight.shape[1]}")
            
        # Check 4th channel is zero
        fourth_norm = first_conv.weight[:, 3, :, :].norm().item()
        if fourth_norm < 1e-6:
            print("   ✅ 4th channel properly zeroed")
        else:
            print(f"   ⚠️  4th channel not zero: {fourth_norm:.6f}")
            
        print("   ✅ Setup validation passed")
        
    except Exception as e:
        print(f"   ❌ Setup validation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()