"""
Simple test configuration for 4-channel RTMDet without complex transforms.
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
        widen_factor=0.25,  # Reduced to save GPU memory
        out_indices=(1, 2, 3),
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True),
        init_cfg=None  # No pretrained weights for 4-channel version
    ),
    neck=dict(
        in_channels=[256, 512, 1024],  # Correct channels from backbone stages (1,2,3)
        out_channels=64, 
        num_csp_blocks=1
    ),
    bbox_head=dict(
        type='RTMDetSepBNHead',
        num_classes=1,  # Only one class: package
        in_channels=64,   # Updated for smaller neck
        feat_channels=64, # Updated for smaller neck
        stacked_convs=2,
        share_conv=True,
        pred_kernel_size=1,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True),
        anchor_generator=dict(
            type='MlvlPointGenerator',
            offset=0,
            strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        with_objectness=False,
        exp_on_reg=False,
    )
)

# Dataset configuration
dataset_type = 'CocoDataset'
data_root = '/home/robun2/Documents/vault_conveyor_tracking/vault_mmdetection/development/augmented_data_production/'

# Simple pipeline without complex transforms
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    
    # Resize FIRST to reduce memory usage - smaller for testing
    dict(type='Resize', scale=(320, 320), keep_ratio=True),
    
    # Generate heatmap at reduced resolution - CLEAN for precise training
    dict(
        type='RobustHeatmapGeneration',
        center_noise_std=2.0,           # Minimal center noise
        keypoint_noise_std=3.0,         # Minimal keypoint noise
        max_sigma=20.0,
        min_sigma=15.0,
        noise_ratio=0.05,               # Very low noise ratio
        error_ratio=0.01,               # Very low error ratio
        quality_variance=False,         # Consistent quality
        no_heatmap_ratio=0.0,          # Always generate heatmap
        global_noise_ratio=0.0,         # NO global noise
        global_noise_std=0.0,           # NO global noise
        multiplicative_noise_ratio=0.0, # NO multiplicative noise
        multiplicative_noise_range=(1.0, 1.0), # NO multiplicative noise
        background_noise_std=0.0        # NO background noise
    ),
    
    # Pad to final size
    dict(
        type='Pad4Channel',
        size=(320, 320),
        pad_val=dict(img=(114, 114, 114, 0))
    ),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'pad_shape')
    )
]

val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    
    # Resize FIRST to reduce memory usage - smaller for testing
    dict(type='Resize', scale=(320, 320), keep_ratio=True),
    
    # Generate heatmap at reduced resolution
    dict(
        type='RobustHeatmapGeneration',
        center_noise_std=2.0,
        keypoint_noise_std=3.0,
        max_sigma=20.0,
        min_sigma=15.0,
        noise_ratio=0.05,  # Lower noise for validation
        error_ratio=0.01,
        quality_variance=False
    ),
    
    # Pad to final size
    dict(
        type='Pad4Channel',
        size=(320, 320),
        pad_val=dict(img=(114, 114, 114, 0))
    ),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'pad_shape')
    )
]

# Data loaders
train_dataloader = dict(
    batch_size=1,
    num_workers=0,  # Disable multiprocessing to avoid device issues
    persistent_workers=False,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train/annotations.json',
        data_prefix=dict(img='train/images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        metainfo=dict(classes=('package',))
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=0,  # Disable multiprocessing to avoid device issues
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='valid/annotations.json',
        data_prefix=dict(img='valid/images/'),
        pipeline=val_pipeline,
        test_mode=True,
        metainfo=dict(classes=('package',))
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

# Training schedule - short for testing
max_epochs = 2  # Very short for testing
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

# Work directory
work_dir = './work_dirs/rtmdet_tiny_4ch_simple_test'