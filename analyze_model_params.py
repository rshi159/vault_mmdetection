#!/usr/bin/env python3
"""
Script to calculate and compare parameter counts for different RTMDet configurations.
"""

import torch
import torch.nn as nn
from mmengine.config import Config
from mmdet.registry import MODELS
import sys
import os

# Add the project root to Python path
sys.path.insert(0, '/home/robun2/Documents/vault_conveyor_tracking/vault_mmdetection')

# Import custom modules to register them
try:
    # This will register our custom components
    from mmdet.models.data_preprocessors.preprocessor_4ch import DetDataPreprocessor4Ch
    from mmdet.models.backbones.backbone_4ch import CSPNeXt4Ch
    print("✅ Custom 4-channel components loaded successfully")
except ImportError as e:
    print(f"❌ Failed to load custom components: {e}")
    print("Running analysis with standard components only...")

def count_parameters(model):
    """Count total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def get_model_size_mb(model):
    """Calculate model size in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024

def analyze_model_components(model):
    """Analyze parameter distribution across model components."""
    components = {}
    
    if hasattr(model, 'backbone'):
        backbone_params = sum(p.numel() for p in model.backbone.parameters())
        components['backbone'] = backbone_params
    
    if hasattr(model, 'neck'):
        neck_params = sum(p.numel() for p in model.neck.parameters())
        components['neck'] = neck_params
    
    if hasattr(model, 'bbox_head'):
        head_params = sum(p.numel() for p in model.bbox_head.parameters())
        components['bbox_head'] = head_params
    
    return components

def analyze_config(config_path, config_name):
    """Analyze a specific configuration."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {config_name}")
    print(f"Config: {config_path}")
    print(f"{'='*60}")
    
    try:
        # Load config
        cfg = Config.fromfile(config_path)
        
        # Build model
        model = MODELS.build(cfg.model)
        model.eval()
        
        # Count parameters
        total_params, trainable_params = count_parameters(model)
        model_size_mb = get_model_size_mb(model)
        
        # Analyze components
        components = analyze_model_components(model)
        
        # Print results
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Model Size: {model_size_mb:.2f} MB")
        print(f"Parameters in millions: {total_params/1e6:.2f}M")
        
        print(f"\nComponent Breakdown:")
        total_component_params = sum(components.values())
        for component, params in components.items():
            percentage = (params / total_component_params) * 100
            print(f"  {component}: {params:,} ({percentage:.1f}%)")
        
        # Memory usage estimation (rough)
        print(f"\nMemory Estimates (FP32):")
        print(f"  Model weights: {model_size_mb:.2f} MB")
        print(f"  Inference (batch=1): ~{model_size_mb * 3:.2f} MB")
        print(f"  Training (batch=16): ~{model_size_mb * 8:.2f} MB")
        print(f"  Training (batch=32): ~{model_size_mb * 15:.2f} MB")
        
        return {
            'name': config_name,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'size_mb': model_size_mb,
            'components': components
        }
        
    except Exception as e:
        print(f"Error analyzing {config_name}: {e}")
        return None

def main():
    # Configurations to analyze
    configs = [
        ('configs/rtmdet/rtmdet_tiny_4ch_simple_test.py', 'RTMDet-Tiny 4Ch (Simple Test)'),
        ('configs/rtmdet/rtmdet_tiny_4ch_production.py', 'RTMDet-Tiny 4Ch (Production)'),
        ('configs/rtmdet/rtmdet_edge_4ch.py', 'RTMDet-Edge 4Ch (Ultra-lightweight)'),
    ]
    
    results = []
    
    print("RTMDet 4-Channel Model Parameter Analysis")
    print("="*60)
    
    for config_path, config_name in configs:
        if os.path.exists(config_path):
            result = analyze_config(config_path, config_name)
            if result:
                results.append(result)
        else:
            print(f"Config not found: {config_path}")
    
    # Comparison table
    if len(results) > 1:
        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(f"{'Configuration':<30} {'Parameters':<12} {'Size (MB)':<10} {'Ratio':<8}")
        print(f"{'-'*80}")
        
        baseline = results[0]['total_params'] if results else 1
        for result in results:
            ratio = result['total_params'] / baseline
            print(f"{result['name']:<30} {result['total_params']/1e6:>8.2f}M {result['size_mb']:>8.2f} {ratio:>6.2f}x")
    
    print(f"\n{'='*80}")
    print("EDGE DEPLOYMENT RECOMMENDATIONS")
    print(f"{'='*80}")
    print("For edge deployment with single-class detection:")
    print("• Target: <1M parameters for mobile devices")
    print("• Target: <5MB model size")
    print("• Consider quantization (INT8) for 4x size reduction")
    print("• Consider pruning for additional 2-3x parameter reduction")
    print("• ONNX/TensorRT conversion for inference optimization")

if __name__ == "__main__":
    main()