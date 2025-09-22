"""
Advanced 4-channel RTMDet configuration with dynamic per-iteration heatmap generation.
Regenerates heatmaps every iteration for maximum robustness and prevents overfitting.
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

# ADVANCED DYNAMIC PIPELINE - Regenerates heatmaps every iteration
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    
    # DYNAMIC HEATMAP GENERATION - Different every iteration!
    dict(
        type='RobustHeatmapGeneration',
        # Core robustness parameters
        noise_ratio=0.8,           # 80% of heatmaps get noise  
        error_ratio=0.2,           # 20% get deliberate errors
        no_heatmap_ratio=0.25,     # 25% get NO heatmap (pure RGB training)
        
        # Noise characteristics  
        center_noise_std=10.0,     # 10-pixel noise on centers
        keypoint_noise_std=15.0,   # 15-pixel noise on keypoints
        quality_variance=True,     # Always vary quality
        
        # Heatmap quality range
        min_sigma=8.0,             # Very tight heatmaps
        max_sigma=50.0             # Very diffuse heatmaps
    ),
    
    # Standard augmentation pipeline
    dict(
        type='CachedMosaic',
        img_scale=(640, 640),
        pad_val=114.0,
        max_cached_images=8,       # Reduced for memory with 4-channel
        random_pop=False),
    dict(
        type='RandomResize',
        scale=(1280, 1280),
        ratio_range=(0.75, 1.25),  # Moderate scaling
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    
    # 4-channel aware padding
    dict(type='Pad4Channel', size=(640, 640), pad_val=dict(img=(114, 114, 114, 0))),
    
    # Final packaging
    dict(type='PackDetInputs')
]

# Validation pipeline - moderate robustness to match real conditions
val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    
    # Validation heatmaps - realistic but less chaotic than training
    dict(
        type='RobustHeatmapGeneration',
        noise_ratio=0.4,           # 40% noise in validation
        error_ratio=0.1,           # 10% errors in validation  
        no_heatmap_ratio=0.15,     # 15% no heatmap in validation
        center_noise_std=5.0,      # Less noise for validation
        keypoint_noise_std=8.0,    # Less noise for validation
        quality_variance=True,     # Still vary quality
        min_sigma=12.0,
        max_sigma=30.0
    ),
    
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad4Channel', size=(640, 640), pad_val=dict(img=(114, 114, 114, 0))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# Test pipeline - production-like conditions (minimal heatmap assistance)
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    
    # Test with minimal heatmap assistance to simulate production
    dict(
        type='RobustHeatmapGeneration',
        noise_ratio=0.6,           # High noise to test robustness
        error_ratio=0.15,          # Some errors to test robustness
        no_heatmap_ratio=0.3,      # 30% pure RGB to test RGB capability
        center_noise_std=8.0,
        keypoint_noise_std=12.0,
        quality_variance=True,
        min_sigma=15.0,
        max_sigma=35.0
    ),
    
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad4Channel', size=(640, 640), pad_val=dict(img=(114, 114, 114, 0))),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# Data loaders with appropriate settings for dynamic generation
train_dataloader = dict(
    batch_size=12,              # Slightly reduced for 4-channel + dynamic generation
    num_workers=6,              # Increased workers for dynamic processing
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        metainfo=dict(classes=('conveyor_object',))
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=val_pipeline,
        metainfo=dict(classes=('conveyor_object',))
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val.json',  # Use val set for testing
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=dict(classes=('conveyor_object',))
    )
)

# Evaluation
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric='bbox',
    format_only=False
)
test_evaluator = val_evaluator

# Extended training for robust learning with dynamic heatmaps
max_epochs = 450  # Longer training for dynamic robustness
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=15  # More frequent validation
)

# Strong regularization for robustness
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.002, weight_decay=0.15),  # Higher weight decay
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True)
)

# Conservative learning rate schedule
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.0005, by_epoch=False, begin=0, end=3000),  # Gentle warmup
    dict(
        type='CosineAnnealingLR',
        eta_min=0.00005,           # Lower minimum LR
        begin=3000,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=False,
        convert_to_iter_based=True)
]

# Enhanced monitoring and regularization
default_hooks = dict(
    checkpoint=dict(interval=30, max_keep_ckpts=8),  # More frequent checkpoints
    logger=dict(interval=25),                        # More frequent logging
    # Enhanced early stopping for dynamic training
    early_stopping=dict(
        monitor='coco/bbox_mAP',
        patience=40,              # Longer patience for dynamic training
        min_delta=0.0005          # Smaller improvement threshold
    )
)

# Runtime settings
work_dir = './work_dirs/rtmdet_tiny_4ch_dynamic'

# Enhanced logging for dynamic heatmap monitoring
log_config = dict(
    interval=25,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook', 
             log_dir='./logs/rtmdet_4ch_dynamic'),
        # Custom hook to log heatmap statistics
        dict(type='WandbLoggerHook', 
             init_kwargs=dict(project='rtmdet_4ch_dynamic'))
    ]
)

# Comments for monitoring
# During training, watch for:
# 1. Model performance with no_heatmap samples (pure RGB capability)
# 2. Consistent improvement despite heatmap noise
# 3. No overfitting to heatmap patterns
# 4. Good generalization to test set with different heatmap conditions