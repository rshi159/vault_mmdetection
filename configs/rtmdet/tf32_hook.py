# TF32 Hook for Runtime Optimization
from mmengine.hooks import Hook
from mmdet.registry import HOOKS

@HOOKS.register_module()
class TF32Hook(Hook):
    """Enable TF32 for RTX 4090 optimization at runtime."""
    
    def __init__(self):
        super().__init__()
    
    def before_run(self, runner):
        """Enable TF32 before training starts."""
        import torch
        import os
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        os.environ['NVIDIA_TF32_OVERRIDE'] = '1'
        runner.logger.info('✅ TF32 enabled for RTX 4090 optimization')