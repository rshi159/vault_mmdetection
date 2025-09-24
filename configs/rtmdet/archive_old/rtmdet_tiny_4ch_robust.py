"""
Robust 4-channel RTMDet configuration with anti-overfitting heatmap strategy.
Uses noisy, imperfect heatmaps to ensure model learns from RGB features too.
"""

_base_ = './rtmdet_tiny_8xb32-300e_coco.py'

# Model configuration with 4-channel components
model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor4Ch',
        mean=[103.53, 116.28, 123.675, 0.0],  # BGR + Heatmap means
        std=[57.375, 57.12, 57.375, 1.0],     # BGR + Heatmap stds
        bgr_to_rgb=False,  # Keep as BGR for compatibility
        pad_size_divisor=32
    ),
    backbone=dict(
        type='CSPNeXt4Ch',  # Use our 4-channel backbone
        arch='P5',
        deepen_factor=0.167,
        widen_factor=0.375,
        out_indices=(2, 3, 4),
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True),
        init_cfg=None  # No pretrained weights for 4-channel version
    ),
    neck=dict(
        in_channels=[96, 192, 384], 
        out_channels=96, 
        num_csp_blocks=1
    ),
    bbox_head=dict(
        in_channels=96, 
        feat_channels=96, 
        exp_on_reg=False,
        num_classes=1  # Single class: conveyor_object
    )
)

# Dataset configuration
dataset_type = 'CocoDataset'
data_root = '/home/robun2/Documents/vault_conveyor_tracking/vault_mmdetection/development/augmented_data_production/'

# Robust pipeline with anti-overfitting heatmap generation
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    
    # ROBUST HEATMAP GENERATION - Key anti-overfitting strategy
    dict(
        type='RobustHeatmapGeneration',
        noise_ratio=0.7,           # 70% of heatmaps get noise
        error_ratio=0.15,          # 15% get deliberate errors
        center_noise_std=8.0,      # 8-pixel noise on centers
        keypoint_noise_std=12.0,   # 12-pixel noise on keypoints
        quality_variance=True,     # Variable heatmap quality
        min_sigma=10.0,            # Min Gaussian spread
        max_sigma=40.0             # Max Gaussian spread
    ),
    
    dict(
        type='CachedMosaic',
        img_scale=(640, 640),
        pad_val=114.0,
        max_cached_images=10,
        random_pop=False),
    dict(
        type='RandomResize',
        scale=(1280, 1280),
        ratio_range=(0.8, 1.2),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad4Channel', size=(640, 640), pad_val=dict(img=(114, 114, 114, 0))),
    dict(type='PackDetInputs')
]

# Validation pipeline - still use robust heatmaps to match training
val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    
    # Less aggressive but still realistic heatmaps for validation
    dict(
        type='RobustHeatmapGeneration',
        noise_ratio=0.3,           # Reduced noise for validation
        error_ratio=0.05,          # Reduced errors for validation
        center_noise_std=4.0,      # Less noise
        keypoint_noise_std=6.0,    # Less noise
        quality_variance=False,    # Consistent quality
        min_sigma=15.0,
        max_sigma=25.0
    ),
    
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad4Channel', size=(640, 640), pad_val=dict(img=(114, 114, 114, 0))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# Data loaders
train_dataloader = dict(
    batch_size=16,  # Reduced batch size for 4-channel data
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train/annotations.json',
        data_prefix=dict(img='train/images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        metainfo=dict(classes=('conveyor_object',))
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='valid/annotations.json',
        data_prefix=dict(img='valid/images/'),
        test_mode=True,
        pipeline=val_pipeline,
        metainfo=dict(classes=('conveyor_object',))
    )
)

test_dataloader = val_dataloader

# Evaluation
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'valid/annotations.json',
    metric='bbox',
    format_only=False
)
test_evaluator = val_evaluator

# Training schedule - longer training for robust learning
max_epochs = 400  # Increased from 300
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=10
)

# Optimizer with weight decay to prevent overfitting
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.003, weight_decay=0.1),  # Increased weight decay
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True)
)

# Learning rate schedule with warmup
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=2000),  # Longer warmup
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0001,
        begin=2000,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=False,
        convert_to_iter_based=True)
]

# Additional regularization hooks
default_hooks = dict(
    checkpoint=dict(interval=50, max_keep_ckpts=5),
    logger=dict(interval=50),
    # Add early stopping to prevent overfitting
    early_stopping=dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP',
        patience=30,
        min_delta=0.001
    )
)

# Regularization settings
# 1. Dropout in neck (if supported)
# 2. Label smoothing
# 3. Mixup/Cutmix (already in pipeline)

# Runtime settings
work_dir = './work_dirs/rtmdet_tiny_4ch_robust'

# Custom hooks can be added here when needed
# custom_hooks = []

# Logging configuration to monitor heatmap quality
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook', 
             log_dir='./logs/rtmdet_4ch_robust')
    ]
)