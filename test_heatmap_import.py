#!/usr/bin/env python3

import sys
import os
from pathlib import Path

# Add the repository root to Python path (same as training script)
repo_root = os.path.abspath('.')
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

print("🔍 Testing HeatmapGenerator Import in Training Context...")

# Simulate the exact import logic from heatmap_transforms.py
print(f"📁 Current working directory: {os.getcwd()}")
print(f"📁 __file__ would be: {__file__}")

# Calculate the dev_dir path exactly like in heatmap_transforms.py
# Note: We're simulating from mmdet/datasets/transforms/heatmap_transforms.py
transforms_dir = Path(repo_root) / 'mmdet' / 'datasets' / 'transforms'
dev_dir = transforms_dir.parent.parent.parent / 'development' / 'heatmap_generation'

print(f"📁 Calculated dev_dir: {dev_dir}")
print(f"📁 dev_dir exists: {dev_dir.exists()}")
print(f"📁 dev_dir absolute: {dev_dir.absolute()}")

if dev_dir.exists():
    print(f"📁 Files in dev_dir: {list(dev_dir.iterdir())}")

# Add to path and try import
sys.path.insert(0, str(dev_dir))
print(f"📁 Added to sys.path: {str(dev_dir)}")

try:
    from heatmap_generator import HeatmapGenerator
    print("✅ HeatmapGenerator imported successfully!")
    print(f"✅ HeatmapGenerator: {HeatmapGenerator}")
    
    # Test creating an instance
    generator = HeatmapGenerator()
    print(f"✅ HeatmapGenerator instance created: {generator}")
    
except ImportError as e:
    print(f"❌ ImportError: {e}")
    print(f"❌ sys.path: {sys.path}")
except Exception as e:
    print(f"❌ Other exception: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()