#!/usr/bin/env python3

"""
Compare the effective channel calculations between original and our 4ch backbone
"""

import torch
from mmdet.registry import MODELS

def calculate_channels():
    """Calculate channel sizes for both architectures"""
    
    # Original RTMDet-Tiny parameters
    arch_setting = [[64, 128, 3, True, False], [128, 256, 6, True, False],
                   [256, 512, 6, True, False], [512, 1024, 3, False, True]]
    deepen_factor = 0.167
    widen_factor = 0.375
    out_indices = (2, 3, 4)
    
    print("=== Original RTMDet-Tiny Channel Calculation ===")
    print(f"deepen_factor: {deepen_factor}")
    print(f"widen_factor: {widen_factor}")
    print(f"out_indices: {out_indices}")
    
    output_channels = []
    for i, (in_channels, out_channels, num_blocks, add_identity, use_spp) in enumerate(arch_setting):
        effective_out = int(out_channels * widen_factor)
        effective_blocks = max(round(num_blocks * deepen_factor), 1)
        print(f"Stage {i+1}: {in_channels} -> {out_channels} * {widen_factor} = {effective_out} (blocks: {effective_blocks})")
        
        if (i + 1) in out_indices:  # Convert to 1-based indexing
            output_channels.append(effective_out)
    
    print(f"\nExpected output channels: {output_channels}")
    
    print("\n=== Testing CSPNeXt4Ch Implementation ===")
    
    # Test our implementation
    model_cfg = dict(
        type='CSPNeXt4Ch',
        arch='P5',
        deepen_factor=0.167,
        widen_factor=0.375,
        out_indices=(2, 3, 4),
    )
    
    try:
        model = MODELS.build(model_cfg)
        
        # Test with dummy input
        dummy_input = torch.randn(1, 4, 640, 640)
        with torch.no_grad():
            outputs = model(dummy_input)
        
        actual_channels = [output.shape[1] for output in outputs]
        print(f"Actual output channels: {actual_channels}")
        
        if actual_channels == output_channels:
            print("✅ Channel dimensions match!")
        else:
            print("❌ Channel dimensions don't match!")
            print(f"Expected: {output_channels}")
            print(f"Got: {actual_channels}")
            
        # Show detailed layer info
        print(f"\nModel stages:")
        for i, layer_name in enumerate(model.layers):
            layer = getattr(model, layer_name)
            print(f"  {layer_name}: {layer}")
            
    except Exception as e:
        print(f"Error building model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    calculate_channels()