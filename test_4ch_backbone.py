#!/usr/bin/env python3

import torch
import sys
import os

# Add the repository root to Python path
repo_root = os.path.abspath('.')
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Initialize MODELS registry
from mmdet.models.backbones import CSPNeXt4Ch

def test_4ch_backbone():
    """Test the 4-channel backbone with actual 4-channel input."""
    
    print("🔍 Testing 4-channel backbone...")
    
    # Create backbone
    backbone = CSPNeXt4Ch(
        arch='P5',
        deepen_factor=0.167,
        widen_factor=0.25,
        out_indices=(1, 2, 3)
    )
    
    # Create 4-channel input: [batch, 4, height, width]
    batch_size = 1
    height, width = 320, 320
    
    # Simulate RGB + Heatmap input
    rgb_input = torch.randn(batch_size, 3, height, width)
    heatmap_input = torch.randn(batch_size, 1, height, width) 
    four_channel_input = torch.cat([rgb_input, heatmap_input], dim=1)
    
    print(f"📊 Input shape: {four_channel_input.shape}")
    print(f"📊 Input channels: {four_channel_input.shape[1]}")
    
    # Test forward pass
    try:
        with torch.no_grad():
            outputs = backbone(four_channel_input)
            
        print(f"✅ Forward pass successful!")
        print(f"📊 Number of output feature maps: {len(outputs)}")
        
        for i, output in enumerate(outputs):
            print(f"📊 Output {i} shape: {output.shape}")
            
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False

def test_3ch_vs_4ch():
    """Compare what happens with 3-channel vs 4-channel input."""
    
    print("\n🔍 Testing channel compatibility...")
    
    # Test with 3-channel input (should fail)
    try:
        backbone = CSPNeXt4Ch()
        three_channel_input = torch.randn(1, 3, 320, 320)
        
        with torch.no_grad():
            outputs = backbone(three_channel_input)
        print("❌ 3-channel input worked (this is unexpected!)")
        
    except Exception as e:
        print(f"✅ 3-channel input correctly failed: {type(e).__name__}")
        
    # Test with 4-channel input (should work)
    try:
        backbone = CSPNeXt4Ch()
        four_channel_input = torch.randn(1, 4, 320, 320)
        
        with torch.no_grad():
            outputs = backbone(four_channel_input)
        print(f"✅ 4-channel input works correctly")
        
    except Exception as e:
        print(f"❌ 4-channel input failed: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 4-Channel Backbone Testing")
    print("=" * 60)
    
    test_4ch_backbone()
    test_3ch_vs_4ch()
    
    print("\n" + "=" * 60)
    print("🎯 Conclusion:")
    print("=" * 60)
    print("If tests pass, the 4-channel backbone is working correctly.")
    print("The rising loss might be due to:")
    print("1. No pretrained weights (training from scratch)")
    print("2. Learning rate too high")
    print("3. Data normalization issues")
    print("4. Heatmap generation creating noise")