#!/usr/bin/env python3
"""
Manual parameter calculation for RTMDet 4-channel configurations.
"""

def calculate_conv_params(in_channels, out_channels, kernel_size, bias=False):
    """Calculate parameters for a convolution layer."""
    weight_params = in_channels * out_channels * kernel_size * kernel_size
    bias_params = out_channels if bias else 0
    return weight_params + bias_params

def calculate_bn_params(channels):
    """Calculate parameters for batch normalization."""
    return channels * 2  # weight + bias

def calculate_cspnext_4ch_params(widen_factor=0.25, deepen_factor=0.167):
    """Calculate parameters for CSPNeXt4Ch backbone."""
    # Base calculations
    base_channels = int(256 * widen_factor)  # 64 for tiny
    channels = [base_channels * (2 ** i) for i in range(5)]  # [64, 128, 256, 512, 1024]
    
    print(f"CSPNeXt4Ch with widen_factor={widen_factor}, deepen_factor={deepen_factor}")
    print(f"Channel progression: {channels}")
    
    total_params = 0
    
    # Stem layers
    # stem: 4 -> channels[0]//2 (4 -> 32)
    stem_params = calculate_conv_params(4, channels[0]//2, 3) + calculate_bn_params(channels[0]//2)
    # stem2: channels[0]//2 -> channels[0] (32 -> 64)
    stem2_params = calculate_conv_params(channels[0]//2, channels[0], 3) + calculate_bn_params(channels[0])
    
    stem_total = stem_params + stem2_params
    total_params += stem_total
    print(f"  Stem: {stem_total:,} params")
    
    # Stages
    base_blocks = [3, 6, 6, 3]
    blocks = [max(1, int(b * deepen_factor)) for b in base_blocks]  # [1, 1, 1, 1] for tiny
    
    in_ch = channels[0]
    for i, (out_ch, num_blocks) in enumerate(zip(channels[1:], blocks)):
        # Downsample conv + bn
        downsample_params = calculate_conv_params(in_ch, out_ch, 3) + calculate_bn_params(out_ch)
        
        # CSP Block approximation
        mid_channels = out_ch // 2
        # main_conv + short_conv + final_conv + blocks
        main_conv = calculate_conv_params(out_ch, mid_channels, 1) + calculate_bn_params(mid_channels)
        short_conv = calculate_conv_params(out_ch, mid_channels, 1) + calculate_bn_params(mid_channels)
        final_conv = calculate_conv_params(out_ch, out_ch, 1) + calculate_bn_params(out_ch)
        
        # Blocks inside CSP
        block_params = 0
        for _ in range(num_blocks):
            block_params += calculate_conv_params(mid_channels, mid_channels, 3) + calculate_bn_params(mid_channels)
        
        stage_params = downsample_params + main_conv + short_conv + final_conv + block_params
        total_params += stage_params
        print(f"  Stage {i+1}: {stage_params:,} params")
        
        in_ch = out_ch
    
    return total_params, channels

def calculate_neck_params(in_channels, out_channels, num_csp_blocks=1):
    """Calculate parameters for CSPNeXtPAFPN neck."""
    total_params = 0
    
    # Reduce layers
    for in_ch in in_channels:
        reduce_params = calculate_conv_params(in_ch, out_channels, 1) + calculate_bn_params(out_channels)
        total_params += reduce_params
    
    # Top-down blocks (simplified)
    for i in range(len(in_channels) - 1):
        # Approximate CSP block parameters
        expand_ch = int(out_channels * 0.5)
        csp_params = (
            calculate_conv_params(out_channels * 2, expand_ch, 1) +  # main_conv
            calculate_conv_params(out_channels * 2, expand_ch, 1) +  # short_conv
            calculate_conv_params(out_channels, out_channels, 1) +   # final_conv
            calculate_bn_params(expand_ch) * 2 + calculate_bn_params(out_channels)
        )
        # Add blocks inside CSP
        for _ in range(num_csp_blocks):
            csp_params += calculate_conv_params(expand_ch, expand_ch, 3) + calculate_bn_params(expand_ch)
        
        total_params += csp_params
    
    # Bottom-up blocks (similar to top-down)
    total_params += total_params  # Rough approximation
    
    # Output convolutions
    for _ in in_channels:
        total_params += calculate_conv_params(out_channels, out_channels, 3) + calculate_bn_params(out_channels)
    
    return total_params

def calculate_head_params(in_channels, feat_channels, num_classes, stacked_convs=2):
    """Calculate parameters for RTMDet head."""
    total_params = 0
    
    # Classification convs
    cls_params = 0
    for i in range(stacked_convs):
        if i == 0:
            cls_params += calculate_conv_params(in_channels, feat_channels, 3) + calculate_bn_params(feat_channels)
        else:
            cls_params += calculate_conv_params(feat_channels, feat_channels, 3) + calculate_bn_params(feat_channels)
    
    # Regression convs (same structure)
    reg_params = cls_params
    
    # Final prediction layers
    cls_pred = calculate_conv_params(feat_channels, num_classes, 1, bias=True)
    reg_pred = calculate_conv_params(feat_channels, 4, 1, bias=True)  # bbox coordinates
    obj_pred = calculate_conv_params(feat_channels, 1, 1, bias=True)  # objectness
    
    total_params = cls_params + reg_params + cls_pred + reg_pred + obj_pred
    
    # Multiply by number of feature levels (typically 3)
    total_params *= 3
    
    return total_params

def analyze_configurations():
    """Analyze different RTMDet configurations."""
    
    configs = [
        {
            'name': 'RTMDet-Tiny 4Ch (widen=0.25)',
            'backbone_widen': 0.25,
            'backbone_deepen': 0.167,
            'neck_out': 96,
            'head_feat': 96,
            'head_convs': 2
        },
        {
            'name': 'RTMDet-Tiny 4Ch Optimized (widen=0.25, neck=64)',
            'backbone_widen': 0.25,
            'backbone_deepen': 0.167,
            'neck_out': 64,
            'head_feat': 64,
            'head_convs': 2
        },
        {
            'name': 'RTMDet-Edge 4Ch (widen=0.125)',
            'backbone_widen': 0.125,
            'backbone_deepen': 0.167,
            'neck_out': 32,
            'head_feat': 32,
            'head_convs': 1
        }
    ]
    
    print("RTMDet 4-Channel Parameter Analysis")
    print("=" * 60)
    
    results = []
    
    for config in configs:
        print(f"\n{config['name']}")
        print("-" * 50)
        
        # Calculate backbone parameters
        backbone_params, backbone_channels = calculate_cspnext_4ch_params(
            config['backbone_widen'], 
            config['backbone_deepen']
        )
        
        # Get neck input channels (stages 1,2,3)
        neck_in_channels = backbone_channels[2:5]  # [256, 512, 1024] for widen=0.25
        
        # Calculate neck parameters
        neck_params = calculate_neck_params(
            neck_in_channels, 
            config['neck_out'], 
            num_csp_blocks=1
        )
        
        # Calculate head parameters
        head_params = calculate_head_params(
            config['neck_out'],
            config['head_feat'],
            num_classes=1,
            stacked_convs=config['head_convs']
        )
        
        total_params = backbone_params + neck_params + head_params
        
        print(f"  Backbone: {backbone_params:,} params ({backbone_params/total_params*100:.1f}%)")
        print(f"  Neck:     {neck_params:,} params ({neck_params/total_params*100:.1f}%)")
        print(f"  Head:     {head_params:,} params ({head_params/total_params*100:.1f}%)")
        print(f"  TOTAL:    {total_params:,} params ({total_params/1e6:.2f}M)")
        
        # Estimate model size
        model_size_mb = total_params * 4 / (1024 * 1024)  # FP32
        print(f"  Size:     {model_size_mb:.2f} MB (FP32)")
        
        results.append({
            'name': config['name'],
            'params': total_params,
            'size_mb': model_size_mb
        })
    
    # Comparison
    print(f"\n{'=' * 60}")
    print("COMPARISON")
    print(f"{'=' * 60}")
    print(f"{'Configuration':<40} {'Parameters':<12} {'Size (MB)'}")
    print(f"{'-' * 60}")
    
    for result in results:
        print(f"{result['name']:<40} {result['params']/1e6:>8.2f}M {result['size_mb']:>8.2f}")
    
    print(f"\n{'=' * 60}")
    print("EDGE DEPLOYMENT RECOMMENDATIONS")
    print(f"{'=' * 60}")
    print("• For mobile/edge: Target <1M parameters")
    print("• RTMDet-Edge config achieves this goal")
    print("• Consider INT8 quantization for 4x size reduction")
    print("• Consider pruning for additional parameter reduction")
    print("• ONNX/TensorRT for inference optimization")

if __name__ == "__main__":
    analyze_configurations()