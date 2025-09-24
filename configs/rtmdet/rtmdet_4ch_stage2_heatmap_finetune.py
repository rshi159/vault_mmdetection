"""
RTMDet 4-Channel Stage 2: Heatmap Fine-tuning Configuration

This config fine-tunes the RGB-trained model from Stage 1 by introducing
heatmap spatial priors at a reduced learning rate. This prevents the heatmap
from overwhelming the RGB features while adding spatial knowledge.

Training Strategy:
- Load Stage 1 checkpoint (RGB features established)
- Introduce augmented heatmaps with aggressive balancing
- Use reduced learning rate (1/5th of Stage 1)
- Shorter training (50-75 epochs for fine-tuning)
- Monitor RGB:PriorH ratio to ensure balance

Key Changes from Stage 1:
1. RobustHeatmapGeneration replaces ZeroHeatmapTransform
2. Reduced learning rate for fine-tuning
3. Shorter epoch count
4. Load from Stage 1 checkpoint
"""

_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# Model configuration for 4-channel input
model = dict(
    type='RTMDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor4Ch',
        mean=[103.53, 116.28, 123.675, 0.0],  # RGB + zero mean for heatmap
        std=[57.375, 57.12, 58.395, 1.0],     # RGB + unit std for heatmap
        bgr_to_rgb=True,
        batch_augments=None
    ),
    backbone=dict(
        type='CSPNeXt4Ch',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=1.0,
        widen_factor=1.0,
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True),
        in_channels=4,  # RGB + heatmap
        # No init_cfg - will load from Stage 1 checkpoint
    ),
    neck=dict(
        type='CSPNeXtPAFPN',
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)
    ),
    bbox_head=dict(
        type='RTMDetHead',
        num_classes=1,  # Single class: parcel
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True),
        share_conv=True,
        pred_kernel_size=1,
        featmap_strides=[8, 16, 32]
    )
)

# Dataset configuration
dataset_type = 'CocoDataset'
data_root = '/home/robun2/Documents/vault_conveyor_tracking/all_images_datasets/augmented_parcels_4ch/'

# Backend arguments
backend_args = None

# Training pipeline - KEY: RobustHeatmapGeneration with aggressive balancing
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    
    # Load and pad to 4 channels
    dict(type='Pad4Channel'),
    
    # CRITICAL: Aggressive heatmap balancing to prevent over-dependence
    dict(
        type='RobustHeatmapGeneration',
        no_heatmap_ratio=0.40,        # 40% complete heatmap suppression
        partial_heatmap_ratio=0.30,   # 30% partial heatmap (some missing)
        noise_heatmap_ratio=0.10,     # 10% noisy heatmap
        heatmap_noise_std=0.15,       # Moderate noise
        missing_bbox_ratio=0.3,       # In partial mode, 30% bboxes missing
        center_bias=1.5,              # Slight center bias
        size_factor=2.0,              # Reasonable size factor
        min_size=20,                  # Minimum heatmap size
        max_size=100                  # Maximum heatmap size
    ),
    
    # Standard augmentations
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    
    # Final formatting
    dict(type='PackDetInputs')
]

# Validation pipeline (with heatmap)
val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Pad4Channel'),
    # Use normal heatmap generation for validation
    dict(
        type='RobustHeatmapGeneration',
        no_heatmap_ratio=0.0,         # Always use heatmap in validation
        partial_heatmap_ratio=0.0,
        noise_heatmap_ratio=0.0
    ),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# Test pipeline (with heatmap)
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Pad4Channel'),
    dict(
        type='RobustHeatmapGeneration',
        no_heatmap_ratio=0.0,         # Always use heatmap in testing
        partial_heatmap_ratio=0.0,
        noise_heatmap_ratio=0.0
    ),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# Dataset definitions
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/train.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args,
        metainfo=dict(classes=('parcel',))
    )
)

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/val.json',
        data_prefix=dict(img='val/'),
        test_mode=True,
        pipeline=val_pipeline,
        backend_args=backend_args,
        metainfo=dict(classes=('parcel',))
    )
)

test_dataloader = val_dataloader

# Evaluation
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args
)
test_evaluator = val_evaluator

# Training schedule for Stage 2: Fine-tuning with heatmap
max_epochs = 60  # Shorter fine-tuning phase
stage_2_epochs = max_epochs

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=3,  # More frequent validation during fine-tuning
    dynamic_intervals=[(max_epochs-10, 1)]
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Reduced learning rate for fine-tuning (1/5th of Stage 1)
base_lr = 0.0008  # Reduced from 0.004

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,  # Gentler warmup for fine-tuning
        by_epoch=False,
        begin=0,
        end=500  # Shorter warmup
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs,
        eta_min=1e-7,  # Lower minimum LR
        begin=0,
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True
    )
]

# Optimizer configuration with reduced learning rate
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.01),  # Reduced weight decay too
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        bypass_duplicate=True
    )
)

# Enhanced monitoring for Stage 2
custom_hooks = [
    dict(
        type='PriorHMonitorHook',
        backbone_path='backbone.stem.conv',
        log_after_epoch=True,
        log_interval=None,  # Only log after epochs
    ),
    # More aggressive early stopping for fine-tuning
    dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP',
        patience=10,  # Shorter patience
        min_delta=0.0005  # Smaller improvement threshold
    )
]

# Logging and checkpointing
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggingHook', interval=25),  # More frequent logging
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=3,  # More frequent checkpointing
        max_keep_ckpts=5,
        save_best='coco/bbox_mAP',
        rule='greater'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

# Environment
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)

# Visualization config
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

# Logging
log_processor = dict(type='LogProcessor', window_size=25, by_epoch=True)
log_level = 'INFO'

# Loading configuration - will load Stage 1 checkpoint
load_from = None  # Will be specified via command line (Stage 1 checkpoint)
resume = False

# Work directory will be specified via command line
work_dir = './work_dirs/stage2_heatmap_finetune'

# Random seed
randomness = dict(seed=42)

# Auto-scale learning rate
auto_scale_lr = dict(enable=False)

# Compile model for faster training (if supported)
compile_config = dict(backend='inductor', mode='default')

print("🔧 Stage 2 Config: Heatmap fine-tuning with RGB features preserved")
print(f"📊 Epochs: {max_epochs}")
print(f"🎓 Learning Rate: {base_lr} (1/5th of Stage 1)")
print(f"🎯 Strategy: Add heatmap knowledge while preserving RGB features")
print(f"⚖️ Heatmap Balance: 40% none + 30% partial + 10% noise = 80% suppression")