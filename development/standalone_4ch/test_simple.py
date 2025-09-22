#!/usr/bin/env python3
"""
Simple test script for 4-channel RTMDet implementation.
Run this to verify everything works correctly.
"""

import torch
import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from preprocessor_4ch import DetDataPreprocessor4Ch
from backbone_4ch import CSPNeXt4Ch

def main():
    print("Testing 4-channel RTMDet implementation...")
    
    # Create components
    preprocessor = DetDataPreprocessor4Ch()
    backbone = CSPNeXt4Ch()
    
    # Create test data: RGB + Heatmap
    batch_size = 1
    rgb = torch.randint(0, 256, (batch_size, 3, 640, 640), dtype=torch.float32)
    heatmap = torch.rand(batch_size, 1, 640, 640)
    input_4ch = torch.cat([rgb, heatmap], dim=1)
    
    print(f"Input shape: {input_4ch.shape}")
    
    # Process
    with torch.no_grad():
        processed = preprocessor(input_4ch)
        features = backbone(processed)
        
        print(f"Output features: {len(features)} levels")
        for i, feat in enumerate(features):
            print(f"  P{i+3}: {feat.shape}")
    
    print("✓ Success! 4-channel processing works.")
    return True

if __name__ == "__main__":
    main()