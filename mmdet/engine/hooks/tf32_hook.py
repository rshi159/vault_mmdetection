# TF32 Optimization Hook for RTX 4090
# Enables TF32 tensor cores for faster mixed precision training

from mmengine.hooks import Hook
from mmdet.registry import HOOKS

@HOOKS.register_module()
class TF32Hook(Hook):
    """Enable TF32 for RTX 4090 optimization at runtime.
    
    This hook enables TF32 tensor cores which can provide ~20% speedup
    on RTX 30/40 series GPUs for mixed precision training.
    """
    
    def __init__(self):
        super().__init__()
        
    def before_run(self, runner):
        """Enable TF32 before training starts."""
        try:
            import torch
            import os
            
            # Enable TF32 for matmul operations
            torch.backends.cuda.matmul.allow_tf32 = True
            # Enable TF32 for cuDNN operations  
            torch.backends.cudnn.allow_tf32 = True
            # Set environment variable for NVIDIA driver
            os.environ['NVIDIA_TF32_OVERRIDE'] = '1'
            
            runner.logger.info('✅ TF32 enabled for RTX 4090 optimization')
            
        except Exception as e:
            runner.logger.warning(f'⚠️  Failed to enable TF32: {e}')