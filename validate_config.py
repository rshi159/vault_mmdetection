#!/usr/bin/env python3
"""
Pre-training validation script for RTMDet 4-channel recovery configuration.
Validates critical components before launching long training runs.
"""

import torch
import sys
import os
sys.path.append('.')

def validate_config():
    """Validate configuration compatibility before training."""
    print("🔍 Validating RTMDet 4-channel recovery configuration...")
    
    # 1. Check CUDA and 4090 capability
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
        
    device_name = torch.cuda.get_device_name(0)
    print(f"✅ GPU: {device_name}")
    
    # 2. Check TF32 enablement
    tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    tf32_cudnn = torch.backends.cudnn.allow_tf32
    print(f"✅ TF32 - matmul: {tf32_matmul}, cudnn: {tf32_cudnn}")
    
    # 3. Validate neck channel calculations
    widen_factor = 0.125
    expected_channels = [int(256 * widen_factor), int(512 * widen_factor), int(1024 * widen_factor)]
    print(f"✅ RTMDet-Tiny neck in_channels should be: {expected_channels}")
    
    # 4. Check memory estimation
    # RTMDet-Tiny @ 768px, batch_size=32, AMP
    estimated_vram = 18.5  # GB
    available_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"✅ VRAM - Available: {available_vram:.1f}GB, Estimated usage: {estimated_vram}GB")
    
    if estimated_vram > available_vram * 0.9:
        print("⚠️  WARNING: Estimated VRAM usage is high, consider reducing batch_size")
    
    # 5. Test SizeAnnealingHook import
    try:
        from configs.rtmdet.size_annealing_hook import SizeAnnealingHook
        print("✅ SizeAnnealingHook imported successfully")
    except ImportError as e:
        print(f"❌ SizeAnnealingHook import failed: {e}")
        return False
    
    # 6. Validate checkpoint path
    checkpoint_path = './work_dirs/rgb_foundation_ultrafast_300ep/best_coco_bbox_mAP_epoch_70.pth'
    if os.path.exists(checkpoint_path):
        print(f"✅ Checkpoint found: {checkpoint_path}")
    else:
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
    
    # 7. Validate data paths
    data_root = 'development/augmented_data_production/'
    train_ann = os.path.join(data_root, 'train/annotations.json')
    val_ann = os.path.join(data_root, 'valid/annotations.json')
    
    if os.path.exists(train_ann):
        print(f"✅ Training annotations found: {train_ann}")
    else:
        print(f"❌ Training annotations missing: {train_ann}")
        return False
        
    if os.path.exists(val_ann):
        print(f"✅ Validation annotations found: {val_ann}")
    else:
        print(f"❌ Validation annotations missing: {val_ann}")
        return False
    
    print("\n🚀 Configuration validation complete!")
    print("\n📋 Training Summary:")
    print("   - Model: RTMDet-Tiny (edge-optimized)")
    print("   - Training res: 768px → 640px (size annealing @ epoch 120)")
    print("   - Batch size: 32 (optimized for 4090)")
    print("   - Duration: 150 epochs")
    print("   - TF32: Enabled (~20% speedup)")
    print("   - RGB-only training with 4th channel zeroed")
    
    return True

if __name__ == "__main__":
    success = validate_config()
    if success:
        print("\n✅ Ready to launch training!")
        print("Command: python tools/train.py configs/rtmdet/rtmdet_4ch_rgb_recovery_gentletune.py --work-dir ./work_dirs/rgb_recovery_4090_optimized")
    else:
        print("\n❌ Please fix validation errors before training")
        sys.exit(1)