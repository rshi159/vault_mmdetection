#!/usr/bin/env python3
"""
Test 4-channel RTMDet memory usage with minimal setup
"""
import torch
import sys
import os
from pathlib import Path

# Setup path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

# Import custom components
from mmdet.models.backbones.backbone_4ch import CSPNeXt4Ch
from mmdet.models.data_preprocessors.preprocessor_4ch import DetDataPreprocessor4Ch

def test_memory_usage():
    """Test GPU memory usage of 4-channel components"""
    
    if not torch.cuda.is_available():
        print("CUDA not available, testing on CPU")
        device = 'cpu'
    else:
        device = 'cuda'
        torch.cuda.empty_cache()
        print(f"Starting GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # Create components
    print("Creating 4-channel preprocessor...")
    preprocessor = DetDataPreprocessor4Ch(
        mean=[103.53, 116.28, 123.675, 0.0],
        std=[57.375, 57.12, 57.375, 1.0]
    ).to(device)
    
    if device == 'cuda':
        print(f"After preprocessor: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    print("Creating 4-channel backbone...")
    backbone = CSPNeXt4Ch(
        arch='P5',
        widen_factor=0.375,    # tiny
        deepen_factor=0.167,   # tiny  
        out_indices=(2, 3, 4)
    ).to(device)
    
    if device == 'cuda':
        print(f"After backbone: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # Test with different batch sizes and resolutions
    test_configs = [
        (1, 320, 320),  # Small
        (1, 640, 640),  # Medium  
        (1, 1280, 1280), # Large
        (4, 320, 320),  # Small batch
    ]
    
    backbone.eval()
    with torch.no_grad():
        for batch_size, h, w in test_configs:
            try:
                print(f"\nTesting batch_size={batch_size}, resolution={h}x{w}")
                
                # Create test data
                test_data = torch.randn(batch_size, 4, h, w, device=device)
                print(f"  Input size: {test_data.shape}, memory: {test_data.numel() * 4 / 1024**3:.3f} GB")
                
                if device == 'cuda':
                    print(f"  Before forward: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                
                # Forward pass
                features = backbone(test_data)
                
                if device == 'cuda':
                    print(f"  After forward: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                
                # Print feature sizes
                total_feature_memory = 0
                for i, feat in enumerate(features):
                    feat_memory = feat.numel() * 4 / 1024**3  # float32 = 4 bytes
                    total_feature_memory += feat_memory
                    print(f"    P{i+3}: {feat.shape}, memory: {feat_memory:.3f} GB")
                
                print(f"  Total feature memory: {total_feature_memory:.3f} GB")
                
                # Clean up
                del test_data, features
                if device == 'cuda':
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                print(f"  ERROR: {e}")
                if device == 'cuda':
                    torch.cuda.empty_cache()

if __name__ == "__main__":
    test_memory_usage()