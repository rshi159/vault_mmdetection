# 4-Channel RTMDet Development Guide

## Architecture Overview

This document provides detailed technical information for developers working on the 4-channel RTMDet implementation.

## Core Components

### 1. CSPNeXt4Ch Backbone (`mmdet/models/backbones/backbone_4ch.py`)

**Purpose**: Modified CSPNeXt backbone that processes 4-channel inputs (RGB + PriorH).

**Key Features**:
- Accepts 4-channel input tensors `[B, 4, H, W]`
- First convolution layer: `Conv2d(4, out_channels, ...)`
- Maintains identical architecture to original CSPNeXt after first layer
- Compatible with existing 3-channel checkpoints via weight inflation

**Implementation Details**:
```python
# First convolution modified for 4 channels
self.stem = nn.Sequential(
    ConvModule(
        4,  # 4-channel input instead of 3
        int(arch_setting[0][0] * widen_factor // 2),
        3, padding=1, stride=2,
        norm_cfg=norm_cfg, act_cfg=act_cfg
    ),
    # ... rest of stem layers
)
```

**Weight Initialization**:
- RGB channels: Initialize from pretrained 3-channel weights
- 4th channel: Initialize to zeros and keep frozen during RGB-only training

### 2. DetDataPreprocessor4Ch (`mmdet/models/data_preprocessors/preprocessor_4ch.py`)

**Purpose**: Data preprocessor that supports arbitrary channel counts.

**Key Features**:
- Removes MMDetection's hardcoded 1/3 channel restriction
- Per-channel normalization: `mean=[123.675, 116.28, 103.53, 0.0]`
- Proper device handling for GPU training
- Maintains compatibility with MMDetection pipeline

**Critical Implementation**:
```python
# Override channel validation
self.num_channels = len(mean)  # Support any number of channels

# Proper normalization setup
self.mean = torch.tensor(mean).view(-1, 1, 1) 
self.std = torch.tensor(std).view(-1, 1, 1)
```

### 3. RGBOnly4Channel Transform (`mmdet/datasets/transforms/fast_4ch.py`)

**Purpose**: Pipeline transform that adds zeroed 4th channel to RGB images.

**Implementation**:
```python
@TRANSFORMS.register_module()
class RGBOnly4Channel(BaseTransform):
    def transform(self, results: dict) -> dict:
        img = results['img']  # Shape: [H, W, 3]
        
        # Add zero channel for PriorH placeholder
        zero_channel = np.zeros((img.shape[0], img.shape[1], 1), dtype=img.dtype)
        img_4ch = np.concatenate([img, zero_channel], axis=2)  # [H, W, 4]
        
        results['img'] = img_4ch
        return results
```

### 4. RGB4ChannelHook (`mmdet/engine/hooks/rgb_4ch_hook.py`)

**Purpose**: Training hook that monitors and enforces 4th channel weight freezing.

**Key Functions**:
- `after_load_checkpoint()`: Zero and freeze 4th channel weights after checkpoint loading
- `before_train_iter()`: Re-zero 4th channel weights before each iteration
- `after_train_iter()`: Log weight statistics for monitoring

**Implementation Pattern**:
```python
def before_train_iter(self, runner, batch_idx=None, data_batch=None):
    if not self.zero_4th_channel:
        return
        
    first_conv = self._get_first_conv(runner.model)
    if first_conv is not None and first_conv.weight.shape[1] >= 4:
        with torch.no_grad():
            first_conv.weight[:, 3, :, :].zero_()
```

## Training Pipeline

### Data Flow
1. **Image Loading**: Standard COCO dataset loading `[H, W, 3]`
2. **Transform Pipeline**: RGBOnly4Channel adds zero channel `[H, W, 4]`
3. **Preprocessing**: DetDataPreprocessor4Ch normalizes per-channel
4. **Model Input**: CSPNeXt4Ch processes `[B, 4, H, W]` tensors
5. **Weight Monitoring**: RGB4ChannelHook ensures 4th channel stays frozen

### Key Design Decisions

#### Why Zero 4th Channel?
- **Architecture Preservation**: Maintains 4-channel capability for future heatmap integration
- **Training Isolation**: RGB training unaffected by uninitialized heatmap weights
- **Checkpoint Compatibility**: Easy transition from 3-channel to 4-channel models

#### Why Freeze 4th Channel Weights?
- **Gradient Isolation**: Prevents backpropagation through unused channel
- **Training Stability**: Eliminates random weight updates in unused channel
- **Reproducibility**: Consistent zero weights across training runs

## Configuration System

### Main Production Config: `rtmdet_4ch_zeroheatmap_fresh.py`

**Key Sections**:

```python
# Custom imports for 4-channel components
custom_imports = dict(
    imports=[
        'mmdet.models.data_preprocessors.preprocessor_4ch',
        'mmdet.models.backbones.backbone_4ch', 
        'mmdet.datasets.transforms.fast_4ch',
        'mmdet.engine.hooks.rgb_4ch_hook',
    ],
    allow_failed_imports=False
)

# 4-channel model configuration
model = dict(
    type='RTMDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor4Ch',
        mean=[123.675, 116.28, 103.53, 0.0],  # RGB + zero heatmap
        std=[58.395, 57.12, 57.375, 1.0],     # RGB + unity heatmap
    ),
    backbone=dict(
        type='CSPNeXt4Ch',  # 4-channel backbone
        # ... standard RTMDet backbone config
    ),
    # ... standard RTMDet head configs
)

# Pipeline with 4-channel transform
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RGBOnly4Channel'),  # Critical: adds zero 4th channel
    # ... rest of pipeline
]
```

### GPU Optimization Settings

**Batch Size Scaling**:
- Target: Maximum GPU utilization on 24GB VRAM
- Current: `batch_size=128` (~15-20GB usage)
- Learning Rate: `lr=4e-4` (scaled proportionally)

**Memory Optimization**:
```python
# AMP settings for memory efficiency
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale=512.0,  # Fixed to prevent NaN gradients
    clip_grad=dict(max_norm=10, norm_type=2),
)

# DataLoader optimization
train_dataloader = dict(
    batch_size=128,
    num_workers=16,      # Fast data loading
    persistent_workers=True,
    pin_memory=True,
)
```

## Checkpoint Management

### 3-Channel to 4-Channel Conversion

**Script**: `convert_3ch_to_4ch.py`

**Process**:
1. Load 3-channel checkpoint
2. Find first convolution layer weights `[out, 3, k, k]`
3. Add zero channel: `[out, 4, k, k]` where `weight[:, 3, :, :] = 0`
4. Save as 4-channel checkpoint

**Usage**:
```bash
python convert_3ch_to_4ch.py \
    --in pretrained_3ch_model.pth \
    --out inflated_4ch_model.pth
```

### Weight Analysis

**Monitoring 4th Channel**:
```python
# Check if 4th channel stays frozen
fourth_norm = first_conv.weight[:, 3, :, :].norm().item()
if fourth_norm < 1e-6:
    print("✅ 4th channel properly frozen")
else:
    print(f"⚠️ 4th channel active: {fourth_norm:.6f}")
```

## Performance Considerations

### Memory Usage
- **4-channel tensors**: ~33% larger than 3-channel
- **Gradient memory**: Additional memory for 4th channel gradients (though frozen)
- **Batch size**: Reduced by ~25% compared to 3-channel training

### Training Speed
- **Forward pass**: Minimal overhead (~5-10% slower)
- **Backward pass**: 4th channel gradients computed but zeroed
- **Data loading**: No significant impact

### Convergence
- **Learning rate**: Scales linearly with batch size
- **Warmup**: Standard 1000 iteration warmup
- **Scheduler**: Cosine annealing over 200 epochs

## Testing and Validation

### Component Testing
1. **Backbone**: Verify 4-channel input processing
2. **Preprocessor**: Check normalization and device placement
3. **Transform**: Validate zero channel addition
4. **Hook**: Monitor weight freezing effectiveness

### Integration Testing
1. **Full pipeline**: End-to-end training validation
2. **Checkpoint loading**: Verify 3ch→4ch conversion
3. **GPU utilization**: Monitor memory usage and speed
4. **Convergence**: Compare with 3-channel baseline

## Future Development

### Heatmap Integration Roadmap

**Phase 1: Current (RGB-only)**
- 4th channel frozen at zero
- Train on RGB data only
- Maintain architecture for future expansion

**Phase 2: Heatmap Data Preparation**
- Develop PriorH heatmap generation pipeline
- Create heatmap augmentation strategies
- Build heatmap-aware evaluation metrics

**Phase 3: Joint Training**
- Unfreeze 4th channel weights
- Implement heatmap-aware loss functions
- Balance RGB and heatmap contributions

**Phase 4: Production Deployment**
- Real-time heatmap generation
- Inference optimization
- Performance monitoring

### Potential Enhancements

1. **Dynamic Channel Training**
   - Randomly enable/disable heatmap channel during training
   - Improve robustness to missing heatmap data

2. **Channel Attention**
   - Learn channel importance weights
   - Adaptive fusion of RGB and heatmap features

3. **Multi-Scale Heatmaps**
   - Process heatmaps at multiple resolutions
   - Hierarchical heatmap feature extraction

## Debugging Guide

### Common Issues

**Issue**: CUDA out of memory
**Solution**: Reduce batch size or use gradient accumulation

**Issue**: 4th channel weights not staying zero
**Solution**: Verify RGB4ChannelHook is registered and first_conv_name is correct

**Issue**: Model architecture mismatch
**Solution**: Check backbone out_indices and neck in_channels alignment

**Issue**: Slow training speed
**Solution**: Increase num_workers, use SSD storage, enable persistent_workers

### Debugging Tools

```python
# Check 4-channel weight status
def check_4ch_weights(model):
    conv = model.backbone.stem[0].conv
    rgb_norm = conv.weight[:, :3, :, :].norm().item()
    fourth_norm = conv.weight[:, 3, :, :].norm().item()
    print(f"RGB norm: {rgb_norm:.6f}, 4th norm: {fourth_norm:.6f}")

# Monitor GPU memory
import torch
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
print(f"GPU cached: {torch.cuda.memory_reserved() / 1e9:.2f}GB")
```

## Contributing Guidelines

### Code Style
- Follow MMDetection conventions
- Use type hints for new functions
- Add docstrings for public methods
- Include unit tests for new components

### Testing Requirements
- Test on different GPU memory sizes
- Validate checkpoint compatibility
- Benchmark against 3-channel baseline
- Verify memory usage patterns

### Documentation
- Update this guide for architectural changes
- Document configuration parameter changes
- Include performance impact analysis
- Provide migration guides for breaking changes