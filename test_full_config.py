#!/usr/bin/env python3
"""Test full config loading step by step"""

import torch
import sys
sys.path.insert(0, '.')

from mmdet.registry import MODELS
from mmengine.config import Config
import mmdet  # Import full mmdet to ensure all registrations
import mmdet.models.backbones.backbone_4ch
import mmdet.models.data_preprocessors.preprocessor_4ch

# Load config
print("1. Loading config...")
cfg = Config.fromfile('configs/rtmdet/rtmdet_4ch_zeroheatmap_fresh.py')
print("✅ Config loaded successfully")

# Try building just the data preprocessor
print("\n2. Building data preprocessor...")
try:
    preprocessor = MODELS.build(cfg.model.data_preprocessor)
    print("✅ Data preprocessor built successfully")
except Exception as e:
    print(f"❌ Data preprocessor failed: {e}")
    import traceback
    traceback.print_exc()

# Try building just the backbone
print("\n3. Building backbone...")
try:
    backbone = MODELS.build(cfg.model.backbone)
    print("✅ Backbone built successfully")
except Exception as e:
    print(f"❌ Backbone failed: {e}")
    import traceback
    traceback.print_exc()

# Try building the full model
print("\n4. Building full model...")
try:
    model = MODELS.build(cfg.model)
    print("✅ Full model built successfully")
    
    # Test with dummy input
    print("\n5. Testing forward pass...")
    dummy_data = dict(
        inputs=torch.randn(1, 4, 640, 640),
        data_samples=[]
    )
    
    with torch.no_grad():
        result = model.forward(**dummy_data, mode='predict')
        print("✅ Forward pass successful")
        
except Exception as e:
    print(f"❌ Model build/forward failed: {e}")
    import traceback
    traceback.print_exc()