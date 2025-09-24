# RTMDet-Tiny 4-Channel Production Training Configuration
# Optimized for RTX 4090 + AMD Ryzen 9 7950X (16-core)
# Based on reference: work_dirs/rtmdet_optimized_training/rtmdet_optimized_config.py

# Custom hooks for early stopping and monitoring
custom_hooks = [
    dict(
        check_finite=True,
        min_delta=0.001,
        monitor='coco/bbox_mAP',
        patience=20,                              # Early stopping patience
        type='EarlyStoppingHook'),
]

# Dataset configuration
data_root = 'development/augmented_data_production/'
dataset_type = 'CocoDataset'

# Hooks configuration - output every epoch
default_hooks = dict(
    checkpoint=dict(
        interval=1,                               # Save every epoch
        max_keep_ckpts=5,
        rule='greater',
        save_best='coco/bbox_mAP',                # Save best model by mAP
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),  # Log every 50 iterations for production
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))

default_scope = 'mmdet'

# Environment configuration - enable cudnn benchmark
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))

launcher = 'none'
load_from = None                                  # No pretrained weights for 4-channel
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)

# Class metadata
metainfo = dict(
    classes=('package',), 
    palette=[(220, 20, 60,),])

# Model configuration - Edge-optimized 4-channel RTMDet for single-class detection
model = dict(
    backbone=dict(
        act_cfg=dict(type='SiLU'),
        arch='P5',
        channel_attention=False,                  # Disabled for edge deployment
        deepen_factor=0.167,                      # Keep minimal depth
        expand_ratio=0.5,
        norm_cfg=dict(type='BN'),
        type='CSPNeXt4Ch',                        # 4-channel backbone
        widen_factor=0.125,                       # Much smaller for edge (was 0.25)
        out_indices=(1, 2, 3)),                   # 3 detection scales
    bbox_head=dict(
        act_cfg=dict(type='SiLU'),
        anchor_generator=dict(
            offset=0, 
            strides=[8, 16, 32], 
            type='MlvlPointGenerator'),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        feat_channels=32,                         # Edge-optimized (was 64)
        in_channels=32,                           # Edge-optimized (was 64)  
        loss_bbox=dict(loss_weight=2.0, type='GIoULoss'),
        loss_cls=dict(
            beta=2.0,
            loss_weight=1.0,
            type='QualityFocalLoss',
            use_sigmoid=True),
        norm_cfg=dict(type='BN'),
        num_classes=1,                            # Single class: package
        stacked_convs=1,                          # Reduced for edge deployment (was 2)
        type='RTMDetHead'),
    data_preprocessor=dict(
        batch_augments=None,
        bgr_to_rgb=False,
        mean=[103.53, 116.28, 123.675, 0.0],      # RGB + Heatmap means
        std=[57.375, 57.12, 58.395, 1.0],         # RGB + Heatmap stds (match ref std)
        type='DetDataPreprocessor4Ch'),           # 4-channel preprocessor
    neck=dict(
        act_cfg=dict(type='SiLU'),
        expand_ratio=0.5,
        in_channels=[64, 128, 256],               # Correct for widen_factor=0.125
        norm_cfg=dict(type='BN'),
        num_csp_blocks=1,
        out_channels=32,                          # Edge-optimized (was 64)
        type='CSPNeXtPAFPN'),
    test_cfg=dict(
        max_per_img=100,                          # Reduced for edge (was 300)
        min_bbox_size=0,
        nms=dict(iou_threshold=0.55, type='nms'),
        nms_pre=1000,                             # Reduced for edge (was 30000)
        score_thr=0.001),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(topk=13, type='DynamicSoftLabelAssigner'),
        debug=False,
        pos_weight=-1),
    type='RTMDet')

# Optimizer configuration - use AmpOptimWrapper like reference
optim_wrapper = dict(
    loss_scale='dynamic',                         # Mixed precision training
    optimizer=dict(lr=0.002, type='AdamW', weight_decay=0.05),  # Higher LR for smaller model and larger batch
    paramwise_cfg=dict(
        bias_decay_mult=0, 
        bypass_duplicate=True, 
        norm_decay_mult=0),
    type='AmpOptimWrapper')

# Learning rate scheduler - match reference pattern
param_scheduler = [
    dict(
        begin=0, 
        by_epoch=False, 
        end=1000, 
        start_factor=1e-05,
        type='LinearLR'),                         # Warmup phase
    dict(
        begin=1000,
        by_epoch=False,
        end=200000,                               # Long cosine schedule  
        eta_min=0.0002,
        type='CosineAnnealingLR'),
]

resume = False

# Training pipeline with 4-channel heatmap generation
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    
    # Data augmentation like reference but without CachedMosaic (4-channel not compatible)
    dict(keep_ratio=True, scale=(640, 640), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    
    # Generate 4th channel heatmap  
    dict(
        type='RobustHeatmapGeneration',
        center_noise_std=2.0,                     # Optimized from our analysis
        keypoint_noise_std=3.0,
        max_sigma=22.0,                           # Slightly larger for 640x640
        min_sigma=16.0,
        noise_ratio=0.08,                         # Small robustness
        error_ratio=0.02,
        quality_variance=True,
        no_heatmap_ratio=0.05,                    # 5% pure RGB
        global_noise_ratio=0.0,
        global_noise_std=0.0,
        multiplicative_noise_ratio=0.02,
        multiplicative_noise_range=(0.95, 1.05),
        background_noise_std=0.01
    ),
    
    dict(
        pad_val=dict(img=(114, 114, 114, 0)),     # 4-channel padding
        size=(640, 640),
        type='Pad4Channel'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'pad_shape')
    ),
]

# Validation pipeline - clean heatmaps
val_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(640, 640), type='Resize'),
    
    # Clean heatmap generation for validation
    dict(
        type='RobustHeatmapGeneration',
        center_noise_std=0.0,                     # No noise for validation
        keypoint_noise_std=0.0,
        max_sigma=20.0,
        min_sigma=18.0,
        noise_ratio=0.0,
        error_ratio=0.0,
        quality_variance=False,
        no_heatmap_ratio=0.0,
        global_noise_ratio=0.0,
        global_noise_std=0.0,
        multiplicative_noise_ratio=0.0,
        multiplicative_noise_range=(1.0, 1.0),
        background_noise_std=0.0
    ),
    
    dict(
        pad_val=dict(img=(114, 114, 114, 0)),     # 4-channel padding  
        size=(640, 640),
        type='Pad4Channel'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path', 
            'ori_shape',
            'img_shape',
            'scale_factor',
            'pad_shape',
        ),
        type='PackDetInputs'),
]

# Training configuration
train_cfg = dict(
    max_epochs=200, 
    type='EpochBasedTrainLoop', 
    val_interval=10)                              # Validate every 10 epochs for production

# Dataloader configuration - optimized for your hardware
train_dataloader = dict(
    batch_size=24,                                # Increased due to smaller model
    dataset=dict(
        ann_file='train/annotations.json',
        data_prefix=dict(img='train/images/'),
        data_root=data_root,
        filter_cfg=dict(filter_empty_gt=False, min_size=0),
        metainfo=metainfo,
        pipeline=train_pipeline,
        type=dataset_type),
    num_workers=12,                               # Increased workers for Ryzen 9 7950X
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))

val_dataloader = dict(
    batch_size=32,                                # Increased due to smaller model (3.22M params)
    dataset=dict(
        ann_file='valid/annotations.json',        # Use validation set
        data_prefix=dict(img='valid/images/'),
        data_root=data_root,
        filter_cfg=dict(filter_empty_gt=False, min_size=0),
        metainfo=metainfo,
        pipeline=val_pipeline,
        type=dataset_type),
    num_workers=12,                               # Increased workers
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))

test_dataloader = val_dataloader                 # Use same as validation

# Evaluation configuration
val_evaluator = dict(
    ann_file=data_root + 'valid/annotations.json',
    format_only=False,
    metric='bbox',
    type='CocoMetric')

test_evaluator = val_evaluator

# Validation and test configuration
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Working directory
work_dir = 'work_dirs/rtmdet_tiny_4ch_production'