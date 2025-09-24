"""
RGB-Only Training Hook for true RGB foundation training.

This hook ensures that the 4th channel (PriorH) weights are properly handled
for RGB-only training by freezing them at zero values.
"""

import torch
from mmengine.hooks import Hook
from mmdet.registry import HOOKS


@HOOKS.register_module()
class RGBOnlyTrainingHook(Hook):
    """Hook to enforce true RGB-only training by managing 4th channel weights.
    
    This hook:
    1. Zeros out 4th channel weights in the first convolutional layer
    2. Freezes these weights during training to prevent updates
    3. Monitors weight evolution to ensure RGB-only learning
    
    Args:
        zero_4th_channel (bool): Whether to zero and freeze 4th channel weights.
            Defaults to True.
        monitor_weights (bool): Whether to log weight analysis.
            Defaults to True.
        log_interval (int, optional): Interval for iteration-based logging.
            If None, only logs after epochs.
    """
    
    def __init__(self, 
                 zero_4th_channel: bool = True,
                 monitor_weights: bool = True,
                 log_interval: int = None):
        self.zero_4th_channel = zero_4th_channel
        self.monitor_weights = monitor_weights
        self.log_interval = log_interval
        self.first_layer_processed = False
        
    def before_train(self, runner):
        """Initialize RGB-only training setup."""
        if self.zero_4th_channel:
            self._setup_rgb_only_weights(runner)
        
        if self.monitor_weights:
            runner.logger.info("🚀 RGB-Only Training Hook initialized")
            runner.logger.info(f"   Zero 4th channel: {self.zero_4th_channel}")
            runner.logger.info(f"   Monitor weights: {self.monitor_weights}")
    
    def _setup_rgb_only_weights(self, runner):
        """Zero and freeze 4th channel weights in first conv layer."""
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
            
        # Find the first conv layer (stem)
        first_conv = None
        if hasattr(model.backbone, 'stem') and hasattr(model.backbone.stem, 'conv'):
            first_conv = model.backbone.stem.conv
        elif hasattr(model.backbone, 'stem') and hasattr(model.backbone.stem, 'conv1'):
            first_conv = model.backbone.stem.conv1
        
        if first_conv is not None and hasattr(first_conv, 'weight'):
            weight = first_conv.weight
            if weight.size(1) >= 4:  # Has 4+ input channels
                # Zero the 4th channel weights (index 3)
                with torch.no_grad():
                    weight[:, 3, :, :] = 0.0
                
                # Freeze the 4th channel weights by creating a custom mask
                def freeze_4th_channel_grad(grad):
                    if grad is not None:
                        grad[:, 3, :, :] = 0.0
                    return grad
                
                weight.register_hook(freeze_4th_channel_grad)
                self.first_layer_processed = True
                
                runner.logger.info("✅ RGB-Only Setup: Zeroed and frozen 4th channel weights")
                runner.logger.info(f"   First conv weight shape: {weight.shape}")
                runner.logger.info(f"   4th channel norm: {weight[:, 3, :, :].norm().item():.6f}")
        else:
            runner.logger.warning("⚠️ Could not find first conv layer for RGB-only setup")
    
    def _get_conv_weights(self, runner):
        """Get first convolutional layer weights for analysis."""
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
            
        if hasattr(model.backbone, 'stem') and hasattr(model.backbone.stem, 'conv'):
            conv_layer = model.backbone.stem.conv
        else:
            return None
            
        if hasattr(conv_layer, 'weight'):
            return conv_layer.weight.data
        return None
    
    def _log_weight_analysis(self, runner, weights, prefix=""):
        """Log detailed weight analysis for RGB vs PriorH channels."""
        if weights.size(1) < 4:
            return
            
        # Split weights: RGB (channels 0,1,2) vs PriorH (channel 3)
        rgb_weights = weights[:, :3, :, :].reshape(-1)
        priorh_weights = weights[:, 3, :, :].reshape(-1)
        
        # Calculate norms
        rgb_norm = rgb_weights.norm().item()
        priorh_norm = priorh_weights.norm().item()
        total_norm = (rgb_norm ** 2 + priorh_norm ** 2) ** 0.5
        
        # Calculate percentages
        rgb_pct = (rgb_norm / total_norm * 100) if total_norm > 0 else 0
        priorh_pct = (priorh_norm / total_norm * 100) if total_norm > 0 else 0
        
        # Calculate statistics
        rgb_mean = rgb_weights.mean().item()
        rgb_std = rgb_weights.std().item()
        rgb_max = rgb_weights.abs().max().item()
        
        priorh_mean = priorh_weights.mean().item()
        priorh_std = priorh_weights.std().item()
        priorh_max = priorh_weights.abs().max().item()
        
        # Individual RGB channel norms
        r_norm = weights[:, 0, :, :].norm().item()
        g_norm = weights[:, 1, :, :].norm().item()
        b_norm = weights[:, 2, :, :].norm().item()
        
        # Log analysis
        runner.logger.info(f"🔍 {prefix}4-Channel Weight Analysis (Epoch {runner.epoch}):")
        runner.logger.info(f"   📊 Norms - RGB: {rgb_norm:.6f} ({rgb_pct:.1f}%) | PriorH: {priorh_norm:.6f} ({priorh_pct:.1f}%)")
        runner.logger.info(f"   📈 Means - RGB: {rgb_mean:.6f} | PriorH: {priorh_mean:.6f}")
        runner.logger.info(f"   📐 Stds  - RGB: {rgb_std:.6f} | PriorH: {priorh_std:.6f}")
        runner.logger.info(f"   🎯 Max   - RGB: {rgb_max:.6f} | PriorH: {priorh_max:.6f}")
        runner.logger.info(f"   🌈 Individual RGB norms - R: {r_norm:.6f}, G: {g_norm:.6f}, B: {b_norm:.6f}")
        
        # Determine utilization status
        if priorh_pct < 1.0:
            status = "🎯 Perfect RGB-only (PriorH frozen)"
        elif priorh_pct < 5.0:
            status = "✅ Excellent RGB-focus"
        elif priorh_pct < 15.0:
            status = "👍 Good RGB-focus"
        elif priorh_pct < 30.0:
            status = "⚠️ Moderate PriorH usage"
        else:
            status = "❌ High PriorH utilization"
            
        runner.logger.info(f"   {status}")
    
    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """Monitor weights during training iterations."""
        if not self.monitor_weights:
            return
            
        if self.log_interval is not None and (runner.iter + 1) % self.log_interval == 0:
            weights = self._get_conv_weights(runner)
            if weights is not None:
                self._log_weight_analysis(runner, weights, prefix="iter_")
    
    def after_train_epoch(self, runner):
        """Log weight analysis after each epoch."""
        if not self.monitor_weights:
            return
            
        weights = self._get_conv_weights(runner)
        if weights is not None:
            self._log_weight_analysis(runner, weights, prefix="epoch_")
    
    def after_val_epoch(self, runner, metrics=None):
        """Log weight analysis after validation."""
        if not self.monitor_weights:
            return
            
        weights = self._get_conv_weights(runner)
        if weights is not None:
            self._log_weight_analysis(runner, weights, prefix="val_")