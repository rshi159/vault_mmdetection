# TF32 Optimization for Ada (4090) Architecture
# Provides ~20% speedup for matrix operations with minimal accuracy impact

import torch
import os

def enable_tf32_optimization():
    """Enable TF32 for significant speedup on Ada architecture (RTX 4090)"""
    
    # Enable TF32 for CUDA matrix operations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Set environment variable for NVIDIA driver
    os.environ['NVIDIA_TF32_OVERRIDE'] = '1'
    
    print("🚀 TF32 optimization enabled for RTX 4090:")
    print(f"   - CUDA matmul TF32: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"   - cuDNN TF32: {torch.backends.cudnn.allow_tf32}")
    print(f"   - NVIDIA TF32 override: {os.environ.get('NVIDIA_TF32_OVERRIDE', 'Not set')}")
    print("   - Expected speedup: ~20% for matrix operations")

# Enable TF32 when this module is imported
enable_tf32_optimization()