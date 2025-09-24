"""
Weight inflation utilities for converting 3-channel pretrained models to 4-channel.
Ensures the 4th channel is properly initialized as zeros and optionally frozen.
"""

import torch
from typing import Dict, Any, Optional
from mmengine.logging import print_log


def inflate_3ch_to_4ch_weights(
    state_dict: Dict[str, Any], 
    first_conv_key: str = "backbone.stem.conv.weight",
    freeze_4th_channel: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Inflate 3-channel pretrained weights to 4-channel by adding zero weights for the 4th channel.
    
    Args:
        state_dict: Pretrained model state dict with 3-channel first conv
        first_conv_key: Key name for the first convolution layer weights
        freeze_4th_channel: Whether to mark 4th channel for freezing
        verbose: Whether to print inflation details
        
    Returns:
        Modified state dict with 4-channel first conv (4th channel = zeros)
    """
    if first_conv_key not in state_dict:
        if verbose:
            print_log(f"⚠️  First conv key '{first_conv_key}' not found in state_dict", level='WARNING')
            print_log(f"Available keys: {list(state_dict.keys())[:10]}...", level='WARNING')
        return state_dict
    
    # Get the first conv weights
    conv_weight = state_dict[first_conv_key]  # Shape: [out_channels, 3, kernel_h, kernel_w]
    
    if conv_weight.shape[1] != 3:
        if verbose:
            print_log(f"⚠️  Expected 3 input channels, got {conv_weight.shape[1]}", level='WARNING')
        return state_dict
    
    # Create 4-channel version by adding zero channel
    out_channels, _, kernel_h, kernel_w = conv_weight.shape
    zero_channel = torch.zeros(out_channels, 1, kernel_h, kernel_w, dtype=conv_weight.dtype, device=conv_weight.device)
    
    # Concatenate: [RGB channels, Zero channel]
    conv_weight_4ch = torch.cat([conv_weight, zero_channel], dim=1)
    
    # Update state dict
    state_dict[first_conv_key] = conv_weight_4ch
    
    if verbose:
        print_log(f"✅ Inflated {first_conv_key}: {conv_weight.shape} → {conv_weight_4ch.shape}", level='INFO')
        print_log(f"   • 4th channel initialized as zeros", level='INFO')
        if freeze_4th_channel:
            print_log(f"   • 4th channel marked for freezing", level='INFO')
    
    return state_dict


def freeze_4th_channel_weights(model: torch.nn.Module, first_conv_name: str = "backbone.stem.conv"):
    """
    Freeze the 4th channel weights of the first convolution to keep them at zero.
    
    Args:
        model: The 4-channel model
        first_conv_name: Name of the first convolution module
    """
    # Navigate to the first conv layer
    module = model
    for name in first_conv_name.split('.'):
        if hasattr(module, name):
            module = getattr(module, name)
        else:
            print_log(f"⚠️  Could not find module {first_conv_name}", level='WARNING')
            return
    
    # Freeze the 4th channel (index 3)
    if hasattr(module, 'weight') and module.weight.shape[1] >= 4:
        with torch.no_grad():
            # Zero out the 4th channel
            module.weight[:, 3, :, :].zero_()
            # Disable gradients for 4th channel
            module.weight.requires_grad_(True)  # Enable overall gradients
            
        print_log(f"✅ Frozen 4th channel weights in {first_conv_name}", level='INFO')
        print_log(f"   • Shape: {module.weight.shape}", level='INFO')
        print_log(f"   • 4th channel norm: {module.weight[:, 3, :, :].norm().item():.6f}", level='INFO')
    else:
        print_log(f"⚠️  Could not freeze 4th channel in {first_conv_name}", level='WARNING')


def get_first_conv_name(model: torch.nn.Module) -> Optional[str]:
    """
    Automatically find the first convolution layer name in common backbones.
    
    Args:
        model: The model to inspect
        
    Returns:
        Name of the first convolution layer or None if not found
    """
    # Common patterns for first conv in different backbones
    candidates = [
        "backbone.stem.conv",
        "backbone.stem.conv1", 
        "backbone.conv1",
        "backbone.stem.0",
        "backbone.conv_stem"
    ]
    
    for candidate in candidates:
        module = model
        found = True
        for name in candidate.split('.'):
            if hasattr(module, name):
                module = getattr(module, name)
            else:
                found = False
                break
        
        if found and hasattr(module, 'weight'):
            return candidate
    
    return None


class RGB4ChannelInitHook:
    """
    Hook to ensure proper 4-channel initialization when loading 3-channel checkpoints.
    Use this as a custom hook in your training config.
    """
    
    def __init__(self, freeze_4th_channel: bool = True):
        self.freeze_4th_channel = freeze_4th_channel
    
    def after_load_checkpoint(self, runner, checkpoint):
        """Called after checkpoint is loaded."""
        model = runner.model
        
        # Find and freeze 4th channel
        first_conv_name = get_first_conv_name(model)
        if first_conv_name and self.freeze_4th_channel:
            freeze_4th_channel_weights(model, first_conv_name)


def apply_rgb_4ch_checkpoint_surgery(checkpoint_path: str, save_path: str, first_conv_key: str = "backbone.stem.conv.weight"):
    """
    Convert a 3-channel checkpoint to 4-channel by inflating the first conv layer.
    
    Args:
        checkpoint_path: Path to 3-channel checkpoint
        save_path: Path to save 4-channel checkpoint
        first_conv_key: Key for first conv weights in state_dict
    """
    print_log(f"🔧 Converting 3-channel checkpoint to 4-channel", level='INFO')
    print_log(f"   • Input: {checkpoint_path}", level='INFO')
    print_log(f"   • Output: {save_path}", level='INFO')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Inflate weights
    if 'state_dict' in checkpoint:
        checkpoint['state_dict'] = inflate_3ch_to_4ch_weights(checkpoint['state_dict'], first_conv_key)
    else:
        checkpoint = inflate_3ch_to_4ch_weights(checkpoint, first_conv_key)
    
    # Save inflated checkpoint
    torch.save(checkpoint, save_path)
    print_log(f"✅ Saved 4-channel checkpoint to {save_path}", level='INFO')