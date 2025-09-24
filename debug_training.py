#!/usr/bin/env python3
"""
Minimal training test with debug output to see where metadata is lost.
"""

import os
import sys

# Add project root to Python path
sys.path.insert(0, '/home/robun2/Documents/vault_conveyor_tracking/vault_mmdetection')

from mmdet.apis import init_detector
from mmengine.config import Config
import torch

def debug_training():
    """Debug the actual training pipeline to see where metadata is lost."""
    
    config_file = 'configs/rtmdet/rtmdet_tiny_4ch_dynamic.py'
    
    print("Loading config...")
    cfg = Config.fromfile(config_file)
    
    # Create a minimal batch to test
    print("Building model...")
    try:
        model = init_detector(cfg, checkpoint=None, device='cpu')
        print("✅ Model built successfully")
        
        # Try to get a single data sample from the dataset
        print("Building dataset...")
        from mmdet.datasets import build_dataset
        
        train_dataset = build_dataset(cfg.train_dataloader.dataset)
        print(f"✅ Dataset built with {len(train_dataset)} samples")
        
        # Get a single sample
        print("Getting sample from dataset...")
        sample = train_dataset[0]
        print(f"Sample keys: {list(sample.keys())}")
        
        if 'data_samples' in sample:
            data_sample = sample['data_samples']
            if hasattr(data_sample, 'metainfo'):
                metainfo = data_sample.metainfo
                print(f"Sample metainfo keys: {list(metainfo.keys())}")
                if 'pad_shape' in metainfo:
                    print(f"✅ Sample has pad_shape: {metainfo['pad_shape']}")
                else:
                    print("❌ Sample missing pad_shape")
                    print("Available metainfo keys:", list(metainfo.keys()))
            else:
                print("❌ Sample data_samples has no metainfo")
        else:
            print("❌ Sample has no data_samples")
            
        # Test the model forward
        print("\nTesting model forward with real data...")
        model.train()
        
        # Create a batch from the sample
        batch_inputs = sample['inputs'].unsqueeze(0)  # Add batch dimension
        batch_data_samples = [sample['data_samples']]
        
        print(f"Batch inputs shape: {batch_inputs.shape}")
        print(f"Batch data samples length: {len(batch_data_samples)}")
        
        # Test model loss
        try:
            losses = model.loss(batch_inputs, batch_data_samples)
            print("✅ Model forward completed successfully")
            print(f"Losses: {list(losses.keys())}")
        except Exception as e:
            print(f"❌ Model forward failed: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_training()