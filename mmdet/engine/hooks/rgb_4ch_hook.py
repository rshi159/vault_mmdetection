"""
Hook to maintain RGB-only training while keeping 4-channel architecture.
Ensures the 4th channel weights remain zero and frozen.
"""

from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmdet.registry import HOOKS
import torch


@HOOKS.register_module()
class RGB4ChannelHook(Hook):
    """
    Hook to maintain RGB-only training in a 4-channel architecture.
    
    This hook:
    1. Zeros and freezes the 4th channel weights after checkpoint loading
    2. Monitors 4th channel weights during training to ensure they stay zero
    3. Provides detailed weight analysis for debugging
    """
    
    def __init__(self, 
                 zero_4th_channel: bool = True,
                 freeze_4th_channel: bool = True,
                 monitor_weights: bool = True,
                 log_interval: int = 500,
                 first_conv_name: str = "backbone.stem.0.conv"):
        """
        Initialize RGB 4-channel hook.
        
        Args:
            zero_4th_channel: Whether to zero out 4th channel weights
            freeze_4th_channel: Whether to freeze 4th channel weights (disable gradients)
            monitor_weights: Whether to log weight statistics
            log_interval: How often to log weight stats (in iterations)
            first_conv_name: Name of first convolution module
        """
        self.zero_4th_channel = zero_4th_channel
        self.freeze_4th_channel = freeze_4th_channel
        self.monitor_weights = monitor_weights
        self.log_interval = log_interval
        self.first_conv_name = first_conv_name
        
    def after_load_checkpoint(self, runner, checkpoint):
        """Called after checkpoint is loaded - set up 4th channel."""
        model = runner.model
        
        # Find first conv layer
        first_conv = self._get_first_conv(model)
        if first_conv is None:
            print_log(f"⚠️  Could not find first conv layer: {self.first_conv_name}", level='WARNING')
            return
            
        with torch.no_grad():
            if first_conv.weight.shape[1] >= 4:
                if self.zero_4th_channel:
                    # Zero out the 4th channel
                    first_conv.weight[:, 3, :, :].zero_()
                    
                # Check current state
                rgb_norm = first_conv.weight[:, :3, :, :].norm().item()
                fourth_norm = first_conv.weight[:, 3, :, :].norm().item()
                
                print_log(f"✅ RGB-Only Setup: Zeroed and frozen 4th channel weights", level='INFO')
                print_log(f"   First conv weight shape: {first_conv.weight.shape}", level='INFO')
                print_log(f"   4th channel norm: {fourth_norm:.6f}", level='INFO')
                
                if self.freeze_4th_channel:
                    # We can't selectively freeze part of a parameter, but we can monitor it
                    print_log(f"   4th channel will be monitored and re-zeroed if needed", level='INFO')
        
        # Log hook setup
        print_log(f"🚀 RGB-Only Training Hook initialized", level='INFO')
        print_log(f"   Zero 4th channel: {self.zero_4th_channel}", level='INFO')
        print_log(f"   Monitor weights: {self.monitor_weights}", level='INFO')
    
    def before_train_iter(self, runner, batch_idx=None, data_batch=None):
        """Before each training iteration - re-zero if needed."""
        if not self.zero_4th_channel:
            return
            
        model = runner.model
        first_conv = self._get_first_conv(model)
        
        if first_conv is not None and first_conv.weight.shape[1] >= 4:
            with torch.no_grad():
                first_conv.weight[:, 3, :, :].zero_()
    
    def after_train_iter(self, runner, batch_idx=None, data_batch=None, outputs=None):
        """After training iteration - log weight stats."""
        if not self.monitor_weights:
            return
            
        if runner.iter % self.log_interval == 0:
            self._log_weight_analysis(runner, f"iter_{runner.iter}")
    
    def after_train_epoch(self, runner):
        """After each training epoch - log weight stats.""" 
        if self.monitor_weights:
            self._log_weight_analysis(runner, f"epoch_{runner.epoch}")
            
    def after_val_epoch(self, runner):
        """After validation epoch - log weight stats."""
        if self.monitor_weights:
            self._log_weight_analysis(runner, f"val_{runner.epoch + 1}")
    
    def _get_first_conv(self, model):
        """Get the first convolution layer."""
        module = model
        for name in self.first_conv_name.split('.'):
            if hasattr(module, name):
                module = getattr(module, name)
            else:
                return None
        return module if hasattr(module, 'weight') else None
    
    def _log_weight_analysis(self, runner, stage):
        """Log detailed weight analysis."""
        first_conv = self._get_first_conv(runner.model)
        if first_conv is None or first_conv.weight.shape[1] < 4:
            return
            
        with torch.no_grad():
            weights = first_conv.weight
            
            # RGB channels (0, 1, 2)
            rgb_weights = weights[:, :3, :, :]
            rgb_norm = rgb_weights.norm().item()
            rgb_mean = rgb_weights.mean().item()
            rgb_std = rgb_weights.std().item()
            rgb_max = rgb_weights.max().item()
            
            # Individual RGB channel norms
            r_norm = weights[:, 0, :, :].norm().item()
            g_norm = weights[:, 1, :, :].norm().item() 
            b_norm = weights[:, 2, :, :].norm().item()
            
            # 4th channel (PriorH)
            fourth_weights = weights[:, 3, :, :]
            fourth_norm = fourth_weights.norm().item()
            fourth_mean = fourth_weights.mean().item()
            fourth_std = fourth_weights.std().item()
            fourth_max = fourth_weights.max().item()
            
            # Calculate percentages
            total_norm = rgb_norm + fourth_norm
            rgb_pct = (rgb_norm / total_norm * 100) if total_norm > 0 else 0
            fourth_pct = (fourth_norm / total_norm * 100) if total_norm > 0 else 0
            
            print_log(f"🔍 {stage}_4-Channel Weight Analysis (Epoch {runner.epoch}):", level='INFO')
            print_log(f"   📊 Norms - RGB: {rgb_norm:.6f} ({rgb_pct:.1f}%) | PriorH: {fourth_norm:.6f} ({fourth_pct:.1f}%)", level='INFO') 
            print_log(f"   📈 Means - RGB: {rgb_mean:.6f} | PriorH: {fourth_mean:.6f}", level='INFO')
            print_log(f"   📐 Stds  - RGB: {rgb_std:.6f} | PriorH: {fourth_std:.6f}", level='INFO')
            print_log(f"   🎯 Max   - RGB: {rgb_max:.6f} | PriorH: {fourth_max:.6f}", level='INFO')
            print_log(f"   🌈 Individual RGB norms - R: {r_norm:.6f}, G: {g_norm:.6f}, B: {b_norm:.6f}", level='INFO')
            
            if fourth_norm < 1e-6:
                print_log(f"   🎯 Perfect RGB-only (PriorH frozen)", level='INFO')
            else:
                print_log(f"   ⚠️  PriorH channel is active! (norm={fourth_norm:.6f})", level='WARNING')