"""Hook for monitoring PriorH channel weight norms during training."""

import torch
from mmengine.hooks import Hook
from mmdet.registry import HOOKS
from typing import Optional


@HOOKS.register_module()
class PriorHMonitorHook(Hook):
    """Hook to monitor RGB vs PriorH channel weight norms during training.
    
    This hook tracks the weight norms of the first 3 channels (RGB) vs the 4th channel
    (PriorH) in the backbone stem to verify the model is learning to use the heatmap channel.
    
    Args:
        log_interval (int): Interval to log weight norms. Default: 50.
        log_after_epoch (bool): Whether to log after each epoch. Default: True.
        backbone_path (str): Path to the backbone stem conv layer. 
            Default: 'backbone.stem.conv'.
    """
    
    def __init__(self, 
                 log_interval: int = None,  # Changed: Only log per epoch by default
                 log_after_epoch: bool = True,
                 backbone_path: str = 'backbone.stem.conv'):
        self.log_interval = log_interval  # None means no iteration logging
        self.log_after_epoch = log_after_epoch
        self.backbone_path = backbone_path
        
    def _get_conv_weights(self, runner) -> Optional[torch.Tensor]:
        """Get the stem conv weights from the model."""
        try:
            # Navigate to the stem conv layer
            model = runner.model
            if hasattr(model, 'module'):  # Handle DDP/DP wrapped models
                model = model.module
                
            # Split path by dots and navigate
            parts = self.backbone_path.split('.')
            conv_layer = model
            for part in parts:
                conv_layer = getattr(conv_layer, part)
                
            # Get the actual conv layer weights
            if hasattr(conv_layer, 'weight'):
                return conv_layer.weight
            elif hasattr(conv_layer, 'conv') and hasattr(conv_layer.conv, 'weight'):
                return conv_layer.conv.weight
            else:
                runner.logger.warning(f"Could not find conv weights at {self.backbone_path}")
                return None
                
        except AttributeError as e:
            runner.logger.warning(f"Error accessing backbone path {self.backbone_path}: {e}")
            return None
    
    def _log_weight_analysis(self, runner, weights: torch.Tensor, prefix: str = ""):
        """Log detailed weight analysis for RGB vs PriorH channels."""
        if weights.size(1) != 4:
            runner.logger.info(f"⚠️  {prefix}Expected 4-channel input weights, got {weights.size(1)} channels")
            return
            
        # Split RGB and PriorH weights
        rgb_weights = weights[:, :3, :, :]  # First 3 channels (RGB)
        priorh_weights = weights[:, 3:4, :, :]  # 4th channel (PriorH)
        
        # Calculate comprehensive statistics
        rgb_norm = torch.norm(rgb_weights).item()
        priorh_norm = torch.norm(priorh_weights).item()
        
        rgb_mean = rgb_weights.mean().item()
        priorh_mean = priorh_weights.mean().item()
        
        rgb_std = rgb_weights.std().item()
        priorh_std = priorh_weights.std().item()
        
        rgb_abs_max = rgb_weights.abs().max().item()
        priorh_abs_max = priorh_weights.abs().max().item()
        
        # Channel-wise norms
        rgb_channel_norms = [torch.norm(weights[:, i, :, :]).item() for i in range(3)]
        
        # Calculate ratios safely
        total_norm = rgb_norm + priorh_norm
        rgb_ratio = (rgb_norm / total_norm * 100) if total_norm > 0 else 0
        priorh_ratio = (priorh_norm / total_norm * 100) if total_norm > 0 else 0
        
        # Log comprehensive analysis
        runner.logger.info(f"🔍 {prefix}4-Channel Weight Analysis (Epoch {runner.epoch + 1}):")
        runner.logger.info(f"   📊 Norms - RGB: {rgb_norm:.6f} ({rgb_ratio:.1f}%) | PriorH: {priorh_norm:.6f} ({priorh_ratio:.1f}%)")
        runner.logger.info(f"   📈 Means - RGB: {rgb_mean:.6f} | PriorH: {priorh_mean:.6f}")
        runner.logger.info(f"   📐 Stds  - RGB: {rgb_std:.6f} | PriorH: {priorh_std:.6f}")
        runner.logger.info(f"   🎯 Max   - RGB: {rgb_abs_max:.6f} | PriorH: {priorh_abs_max:.6f}")
        runner.logger.info(f"   🌈 Individual RGB norms - R: {rgb_channel_norms[0]:.6f}, G: {rgb_channel_norms[1]:.6f}, B: {rgb_channel_norms[2]:.6f}")
        
        # Analysis interpretation
        if priorh_norm < 0.001:
            runner.logger.info(f"   ⚠️  PriorH weights extremely small - channel likely unused!")
        elif priorh_ratio < 5:
            runner.logger.info(f"   ⚠️  PriorH weights very small ({priorh_ratio:.1f}%) - minimal utilization")
        elif priorh_ratio > 30:
            runner.logger.info(f"   ✅ Strong PriorH utilization ({priorh_ratio:.1f}%)")
        else:
            runner.logger.info(f"   📌 Moderate PriorH utilization ({priorh_ratio:.1f}%)")
            
        # Store metrics for potential plotting/analysis
        if not hasattr(runner, 'weight_history'):
            runner.weight_history = []
        runner.weight_history.append({
            'epoch': runner.epoch + 1,
            'rgb_norm': rgb_norm,
            'priorh_norm': priorh_norm,
            'rgb_ratio': rgb_ratio,
            'priorh_ratio': priorh_ratio
        })
    
    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs=None):
        """Log weight norms every N iterations (only if log_interval is set)."""
        if self.log_interval is not None and (runner.iter + 1) % self.log_interval == 0:
            weights = self._get_conv_weights(runner)
            if weights is not None:
                self._log_weight_analysis(runner, weights, prefix="iter_")
    
    def after_train_epoch(self, runner):
        """Log weight norms after each epoch."""
        if self.log_after_epoch:
            weights = self._get_conv_weights(runner)
            if weights is not None:
                self._log_weight_analysis(runner, weights, prefix="epoch_")
    
    def after_val_epoch(self, runner, metrics=None):
        """Log weight norms after validation."""
        weights = self._get_conv_weights(runner)
        if weights is not None:
            self._log_weight_analysis(runner, weights, prefix="val_")