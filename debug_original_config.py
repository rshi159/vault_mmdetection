#!/usr/bin/env python3

"""
Debug script to understand the original checkpoint architecture
"""

import torch
from mmdet.apis import init_detector

def analyze_original_rtmdet():
    """Analyze the original RTMDet-tiny config that matches our checkpoint"""
    
    # Load the original 3-channel config that was used to create our checkpoint
    config_file = 'configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py'
    checkpoint_file = './work_dirs/rtmdet_optimized_training/best_coco_bbox_mAP_epoch_195_4ch.pth'
    
    print("=== Analyzing Original RTMDet-Tiny Architecture ===")
    
    try:
        # Initialize the original model
        model = init_detector(config_file, checkpoint=None, device='cpu')
        
        print("\n1. Original Model Backbone Output Channels:")
        # Check backbone output
        import torch
        dummy_input = torch.randn(1, 3, 640, 640)
        
        # Get backbone features
        features = model.backbone(dummy_input)
        for i, feat in enumerate(features):
            print(f"   Stage {i}: {feat.shape}")
        
        print("\n2. Original Model Neck Input/Output:")
        neck_outputs = model.neck(features)
        for i, output in enumerate(neck_outputs):
            print(f"   Output {i}: {output.shape}")
            
        print("\n3. Original Model Configuration:")
        print(f"   Backbone: {model.backbone.__class__.__name__}")
        print(f"   Backbone channels: {getattr(model.backbone, 'out_channels', 'N/A')}")
        
        # Check if we can extract the specific parameters
        if hasattr(model.backbone, 'widen_factor'):
            print(f"   Widen factor: {model.backbone.widen_factor}")
        if hasattr(model.backbone, 'deepen_factor'):
            print(f"   Deepen factor: {model.backbone.deepen_factor}")
            
        print("\n4. Neck Configuration:")
        print(f"   Neck: {model.neck.__class__.__name__}")
        if hasattr(model.neck, 'in_channels'):
            print(f"   Neck in_channels: {model.neck.in_channels}")
        if hasattr(model.neck, 'out_channels'):
            print(f"   Neck out_channels: {model.neck.out_channels}")
            
    except Exception as e:
        print(f"Error loading original model: {e}")
        
    print("\n=== Checkpoint Analysis ===")
    try:
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # Find backbone output channels by looking at neck input
        backbone_keys = [k for k in state_dict.keys() if k.startswith('backbone.') and 'conv.weight' in k and '.stage' in k]
        
        print("\n5. Backbone Stages from Checkpoint:")
        stage_channels = {}
        for key in backbone_keys:
            if 'main_conv.conv.weight' in key and 'stage' in key:
                stage_num = key.split('.stage')[1].split('.')[0]
                weight = state_dict[key]
                out_channels = weight.shape[0]
                stage_channels[f'stage{stage_num}'] = out_channels
                print(f"   {key}: out_channels = {out_channels}")
                
        print("\n6. Neck Input Channels from Checkpoint:")
        neck_keys = [k for k in state_dict.keys() if k.startswith('neck.reduce_layers')]
        for key in neck_keys:
            if 'conv.weight' in key:
                weight = state_dict[key]
                in_channels = weight.shape[1]
                out_channels = weight.shape[0]
                print(f"   {key}: {in_channels} -> {out_channels}")
                
    except Exception as e:
        print(f"Error analyzing checkpoint: {e}")

if __name__ == '__main__':
    analyze_original_rtmdet()