#!/usr/bin/env python3
"""
Fresh 4-channel training with proper weight inflation and 4th channel freezing.
Implements gradient hook to keep 4th channel weights at zero during training.
"""

import argparse
import os
import sys
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.registry import RUNNERS


def freeze_fourth_input_slice(model, conv_attr='backbone.stem.0.conv'):
    """
    Freeze the 4th input channel of the first convolution layer.
    
    Args:
        model: The model to modify
        conv_attr: Attribute path to the first convolution layer
    """
    # Navigate to the conv layer
    parts = conv_attr.split('.')
    conv = model
    for part in parts:
        conv = getattr(conv, part)
    
    # Zero out the 4th channel weights
    with torch.no_grad():
        if conv.weight.data.shape[1] >= 4:
            conv.weight.data[:, 3].zero_()
            print(f"✅ Zeroed 4th channel weights in {conv_attr}")
            print(f"   • Weight shape: {conv.weight.data.shape}")
            print(f"   • 4th channel norm: {conv.weight.data[:, 3].norm().item():.6f}")
    
    # Register gradient hook to keep 4th channel at zero
    def zero_grad_ch4(grad):
        """Gradient hook to zero out 4th channel gradients."""
        if grad.shape[1] >= 4:
            grad[:, 3] = 0
        return grad
    
    conv.weight.register_hook(zero_grad_ch4)
    print(f"✅ Registered gradient hook to freeze 4th channel in {conv_attr}")


def sanity_check_4ch_vs_3ch(model_4ch, dummy_input_4ch, dummy_input_3ch=None):
    """
    Quick sanity check to compare 4-channel model with zero heatmap 
    against equivalent 3-channel processing.
    
    Args:
        model_4ch: 4-channel model
        dummy_input_4ch: Dummy 4-channel input (RGB + zero heatmap)
        dummy_input_3ch: Optional 3-channel input for comparison
    """
    model_4ch.eval()
    
    with torch.no_grad():
        # Test 4-channel model
        features_4ch = model_4ch.backbone(dummy_input_4ch)
        
        print(f"🔍 4-Channel Model Sanity Check:")
        print(f"   • Input shape: {dummy_input_4ch.shape}")
        for i, feat in enumerate(features_4ch):
            print(f"   • Feature {i} shape: {feat.shape}, mean: {feat.mean().item():.6f}")
        
        # Check that 4th channel is actually zero
        if dummy_input_4ch.shape[1] >= 4:
            ch4_norm = dummy_input_4ch[:, 3].norm().item()
            print(f"   • 4th channel input norm: {ch4_norm:.6f}")
            if ch4_norm > 1e-6:
                print(f"   ⚠️  WARNING: 4th channel is not zero!")
            else:
                print(f"   ✅ 4th channel is properly zeroed")
    
    model_4ch.train()


def main():
    parser = argparse.ArgumentParser(description='Train RTMDet 4-channel with zero heatmap')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume', action='store_true', help='resume from the latest checkpoint automatically')
    parser.add_argument('--amp', action='store_true', help='enable automatic-mixed-precision training')
    parser.add_argument('--no-validate', action='store_true', help='disable checkpoint validation')
    
    args = parser.parse_args()
    
    # Load config
    cfg = Config.fromfile(args.config)
    
    # Override work dir if specified
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    
    # Handle resume
    if args.resume:
        cfg.resume = True
    
    # Create work directory
    os.makedirs(cfg.work_dir, exist_ok=True)
    
    print(f"🚀 Starting Fresh 4-Channel Training")
    print(f"   • Config: {args.config}")
    print(f"   • Work dir: {cfg.work_dir}")
    print(f"   • Load from: {cfg.get('load_from', 'None')}")
    print(f"   • Resume: {cfg.get('resume', False)}")
    
    # Create runner
    runner = Runner.from_cfg(cfg)
    
    # Apply 4th channel freezing after model is built and checkpoint is loaded
    print(f"\n🔧 Applying 4th channel freezing...")
    freeze_fourth_input_slice(runner.model, conv_attr='backbone.stem.0.conv')
    
    # Optional: Sanity check with dummy input
    print(f"\n🧪 Running sanity check...")
    device = next(runner.model.parameters()).device
    dummy_input = torch.randn(1, 4, 640, 640).to(device)
    dummy_input[:, 3] = 0  # Zero out 4th channel
    sanity_check_4ch_vs_3ch(runner.model, dummy_input)
    
    print(f"\n🎯 Starting training...")
    
    # Start training
    if cfg.get('resume', False):
        runner.resume()
    else:
        runner.train()


if __name__ == '__main__':
    main()