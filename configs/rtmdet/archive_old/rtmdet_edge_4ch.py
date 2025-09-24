# RTMDet Edge-Optimized 4-Channel Configuration
# Designed for single-class detection on edge devices
# Minimal model size with maximum efficiency

# Custom hooks for early stopping and monitoring
custom_hooks = [
    dict(
        check_finite=True,
        min_delta=0.001,
        monitor='coco/bbox_mAP',
        patience=15,                              # Reduced patience for faster training
        type='EarlyStoppingHook'),
]

# Dataset configuration
data_root = 'development/augmented_data_production/'
dataset_type = 'CocoDataset'

# Hooks configuration - reduced logging for edge focus
default_hooks = dict(
    checkpoint=dict(
        interval=1,                               
        max_keep_ckpts=3,                         # Keep fewer checkpoints
        rule='greater',
        save_best='coco/bbox_mAP',                
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),  
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))

default_scope = 'mmdet'

# Environment configuration
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))

launcher = 'none'
load_from = None                                  
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)

# Class metadata
metainfo = dict(
    classes=('package',), 
    palette=[(220, 20, 60,),])

# ULTRA-LIGHTWEIGHT MODEL CONFIGURATION FOR EDGE DEPLOYMENT
model = dict(
    type='RTMDet',
    
    # Minimal 4-channel backbone - heavily reduced for edge
    backbone=dict(
        type='CSPNeXt4Ch',
        arch='P5',
        deepen_factor=0.167,                      # Keep minimal depth
        widen_factor=0.125,                       # MUCH smaller than 0.25 - save significant memory
        channel_attention=False,                  # Disable attention for speed
        expand_ratio=0.5,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU'),
        out_indices=(1, 2, 3),                    # Keep 3 scales for detection
    ),
    
    # Minimal neck - heavily reduced channels
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[64, 128, 256],               # Correct channels for widen_factor=0.125
        out_channels=32,                          # MUCH smaller than 64/96 - significant memory savings
        num_csp_blocks=1,                         # Minimal CSP blocks
        expand_ratio=0.5,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU'),
    ),
    
    # Minimal detection head - optimized for single class
    bbox_head=dict(
        type='RTMDetHead',
        num_classes=1,                            # Single class only
        in_channels=32,                           # Match neck output
        feat_channels=32,                         # Minimal feature channels
        stacked_convs=1,                          # Reduced from 2 - less computation
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU'),
        anchor_generator=dict(
            type='MlvlPointGenerator',
            offset=0,
            strides=[8, 16, 32]                   # Standard strides
        ),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0
        ),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
    ),
    
    # 4-channel data preprocessor
    data_preprocessor=dict(
        type='DetDataPreprocessor4Ch',
        mean=[103.53, 116.28, 123.675, 0.0],      
        std=[57.375, 57.12, 58.395, 1.0],         
        bgr_to_rgb=False,
        batch_augments=None,
    ),
    
    # Training configuration
    train_cfg=dict(
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False
    ),
    
    # Test configuration - optimized for speed
    test_cfg=dict(
        nms_pre=1000,                             # Much smaller than 30000
        min_bbox_size=0,
        score_thr=0.01,                           # Higher threshold for edge
        nms=dict(type='nms', iou_threshold=0.55),
        max_per_img=100                           # Much smaller than 300
    )
)

# Optimizer configuration - optimized for smaller model
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW',
        lr=0.002,                                 # Higher LR for smaller model
        weight_decay=0.05
    ),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True
    )
)

# Learning rate schedule - faster convergence for edge model
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-5,
        by_epoch=False,
        begin=0,
        end=500                                   # Shorter warmup
    ),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0002,
        begin=500,
        end=100000,                               # Shorter schedule
        by_epoch=False,
    )
]

# Training configuration
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=150,                               # Reduced epochs for edge model
    val_interval=5                                # More frequent validation
)

# Validation and test configuration
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Training pipeline with minimal augmentation for edge deployment
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', prob=0.3),            # Reduced flip probability
    
    # Minimal heatmap generation for training
    dict(
        type='RobustHeatmapGeneration',
        min_sigma=18.0,
        max_sigma=20.0,
        noise_ratio=0.05,                         # Minimal noise for edge
        keypoint_noise_std=2.0,
        center_noise_std=1.0,
        background_noise_std=0.005,
        quality_variance=False,                   # Disable for consistency
        multiplicative_noise_ratio=0.01,
        multiplicative_noise_range=(0.98, 1.02),
        global_noise_ratio=0.0,
        global_noise_std=0.0,
        error_ratio=0.01,
        no_heatmap_ratio=0.02
    ),
    
    dict(
        type='Pad4Channel',
        size=(640, 640),
        pad_val=dict(img=(114, 114, 114, 0))
    ),
    
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'pad_shape')
    )
]

# Validation pipeline - no augmentation
val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    
    # Clean heatmap generation for validation
    dict(
        type='RobustHeatmapGeneration',
        min_sigma=19.0,
        max_sigma=19.0,                           # Fixed sigma for consistency
        noise_ratio=0.0,
        keypoint_noise_std=0.0,
        center_noise_std=0.0,
        background_noise_std=0.0,
        quality_variance=False,
        multiplicative_noise_ratio=0.0,
        multiplicative_noise_range=(1.0, 1.0),
        global_noise_ratio=0.0,
        global_noise_std=0.0,
        error_ratio=0.0,
        no_heatmap_ratio=0.0
    ),
    
    dict(
        type='Pad4Channel',
        size=(640, 640),
        pad_val=dict(img=(114, 114, 114, 0))
    ),
    
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'pad_shape')
    )
]

# Dataloader configuration - optimized for edge development
train_dataloader = dict(
    batch_size=32,                                # Can handle larger batches with smaller model
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train/annotations.json',
        data_prefix=dict(img='train/images/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=0),
        metainfo=metainfo,
        pipeline=train_pipeline
    ),
    num_workers=8,                                # Reasonable for edge development
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True)
)

val_dataloader = dict(
    batch_size=24,                                # Larger batches for validation
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='valid/annotations.json',
        data_prefix=dict(img='valid/images/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=0),
        metainfo=metainfo,
        pipeline=val_pipeline
    ),
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False)
)

test_dataloader = val_dataloader

# Evaluation configuration
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'valid/annotations.json',
    metric='bbox',
    format_only=False
)

test_evaluator = val_evaluator