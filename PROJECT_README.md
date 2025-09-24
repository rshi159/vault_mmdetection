# 4-Channel RTMDet for Vault Conveyor Package Detection

## Overview

This project implements a 4-channel RTMDet architecture for enhanced package detection in vault conveyor systems. The system extends the standard RTMDet model to accept RGB + PriorH (heatmap) input while maintaining full compatibility with existing 3-channel pretrained models.

## Key Features

- **4-Channel Architecture**: Processes RGB + PriorH heatmap inputs for enhanced detection
- **RGB-Only Training Mode**: Trains with RGB channels while maintaining 4-channel architecture 
- **Zero-Heatmap Pipeline**: Uses zeroed 4th channel as placeholder for future heatmap integration
- **Checkpoint Compatibility**: Seamless conversion from 3-channel to 4-channel models
- **GPU Optimized**: Configured for efficient training on 24GB GPUs with large batch sizes

## Architecture

### Core Components

1. **CSPNeXt4Ch Backbone**: Modified CSPNeXt that accepts 4-channel inputs
2. **DetDataPreprocessor4Ch**: Custom preprocessor supporting arbitrary channel counts
3. **RGBOnly4Channel Transform**: Pipeline transform that adds zeroed 4th channel
4. **RGB4ChannelHook**: Training hook that monitors and freezes 4th channel weights

### Model Configuration

- **Base Model**: RTMDet-Tiny with 4-channel input capability
- **Input Channels**: 4 (RGB + PriorH placeholder)
- **Training Mode**: RGB-only with zeroed heatmap channel
- **Batch Size**: 128 (optimized for 24GB GPU)
- **Learning Rate**: 4e-4 (scaled for large batch size)

## Quick Start

### Prerequisites

```bash
# Install MMDetection 3.1.1 environment
conda create -n mmdet311 python=3.8
conda activate mmdet311
pip install torch torchvision
pip install mmengine mmcv mmdet
```

### Training

```bash
# Activate environment
source ~/.venvs/mmdet311/bin/activate

# Start 4-channel training
python train_simple.py
```

### Inference

```bash
# Run inference on trained model
python demo/image_demo.py \
    demo/demo.jpg \
    configs/rtmdet/rtmdet_4ch_zeroheatmap_fresh.py \
    work_dirs/rtmdet_4ch_zeroheatmap_fresh/best_coco_bbox_mAP.pth
```

### Checkpoint Conversion

Convert existing 3-channel checkpoints to 4-channel:

```bash
python convert_3ch_to_4ch.py \
    --in path/to/3ch_checkpoint.pth \
    --out path/to/4ch_checkpoint.pth
```

## Configuration Files

### Production Configurations

- **`rtmdet_4ch_zeroheatmap_fresh.py`**: Main production config for 4-channel RGB-only training
  - Batch size: 128
  - Learning rate: 4e-4  
  - GPU optimized for 24GB VRAM
  - Uses DynamicSoftLabelAssigner for RTMDet

### Key Configuration Sections

```python
# 4-Channel Model Definition
model = dict(
    type='RTMDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor4Ch',
        mean=[123.675, 116.28, 103.53, 0.0],  # RGB + zero heatmap
        std=[58.395, 57.12, 57.375, 1.0],     # RGB + unity heatmap
    ),
    backbone=dict(
        type='CSPNeXt4Ch',  # 4-channel backbone
        # ... configuration
    ),
    # ... rest of model config
)

# RGB-Only Training Pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RGBOnly4Channel'),  # Adds zero 4th channel
    # ... rest of pipeline
]
```

## Development Workflow

### Training Pipeline

1. **Data Loading**: Standard COCO-format dataset loading
2. **Transform Pipeline**: RGBOnly4Channel transform adds zeroed 4th channel
3. **Model Processing**: 4-channel backbone processes [R,G,B,0] tensors
4. **Weight Monitoring**: RGB4ChannelHook ensures 4th channel weights stay frozen
5. **Loss Computation**: Standard RTMDet loss functions

### Key Design Decisions

- **Architecture Preservation**: Maintains 4-channel capability for future heatmap integration
- **Training Efficiency**: RGB-only mode allows training without heatmap data
- **Checkpoint Compatibility**: Inflation mechanism preserves pretrained RGB features
- **Gradient Isolation**: 4th channel weights frozen to prevent interference

## File Structure

```
├── configs/rtmdet/
│   ├── rtmdet_4ch_zeroheatmap_fresh.py    # Main production config
│   └── archive_old/                        # Archived experimental configs
├── mmdet/
│   ├── models/
│   │   ├── backbones/backbone_4ch.py       # 4-channel CSPNeXt backbone
│   │   └── data_preprocessors/preprocessor_4ch.py  # 4-channel preprocessor
│   ├── datasets/transforms/fast_4ch.py     # RGBOnly4Channel transform
│   └── engine/hooks/rgb_4ch_hook.py        # 4th channel monitoring hook
├── convert_3ch_to_4ch.py                   # Checkpoint conversion utility
├── train_simple.py                         # Simple training script
└── PROJECT_README.md                       # This documentation
```

## Performance

### Training Metrics

- **GPU Utilization**: ~15-20GB / 24GB VRAM (excellent utilization)
- **Batch Processing**: 128 samples per batch for fast convergence
- **Training Speed**: Optimized data loading with 16 workers
- **Memory Efficiency**: AMP training with fixed loss scaling

### Model Performance

- **mAP**: Comparable to standard RTMDet on RGB-only data
- **4th Channel**: Properly frozen at zero weights throughout training
- **Convergence**: Stable training with large batch size and scaled learning rate

## Future Enhancements

### Planned Features

1. **Heatmap Integration**: Enable training with actual PriorH heatmap data
2. **Multi-Scale Training**: Dynamic image size training for robustness
3. **Advanced Augmentations**: Specialized augmentations for conveyor systems
4. **Real-time Inference**: Optimizations for production deployment

### Integration Roadmap

1. **Phase 1**: RGB-only training (current implementation)
2. **Phase 2**: Heatmap data collection and preprocessing
3. **Phase 3**: Joint RGB+Heatmap training
4. **Phase 4**: Production deployment and monitoring

## Troubleshooting

### Common Issues

**Q: CUDA out of memory errors**
A: Reduce batch size from 128 to 64 or 32 in the config file

**Q: 4th channel weights not staying zero**
A: Ensure RGB4ChannelHook is properly configured in custom_hooks

**Q: Checkpoint loading errors**
A: Use convert_3ch_to_4ch.py to properly inflate 3-channel checkpoints

**Q: Slow training speed**
A: Increase num_workers in dataloaders, ensure SSD storage for dataset

### Debugging Tools

- **Weight Analysis**: Check 4th channel weight norms during training
- **Pipeline Verification**: Validate RGBOnly4Channel transform output
- **Memory Profiling**: Monitor GPU memory usage with different batch sizes

## Contributing

### Code Review Checklist

- [ ] All debug/test files removed from production branch
- [ ] Configuration files properly documented
- [ ] Training scripts follow MMDetection conventions
- [ ] 4th channel freezing properly implemented
- [ ] GPU memory usage optimized
- [ ] Documentation up to date

### Development Guidelines

1. Use semantic commit messages
2. Test configurations before pushing
3. Document any architectural changes
4. Profile GPU memory usage for new features
5. Maintain compatibility with MMDetection updates

## License

This project extends MMDetection which is released under the Apache 2.0 license.