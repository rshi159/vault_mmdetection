"""
RTMDet-tiny configuration for 4-channel conveyor belt detection.
RGB + Heatmap input processing with native 4-channel support.
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

# Custom pipeline for 4-channel data loading
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    # Generate heatmap channel to create 4-channel input
    dict(type='GenerateHeatmapChannel', method='prior_based', heatmap_strength=1.0),
    dict(
        type='CachedMosaic',
        img_scale=(640, 640),
        pad_val=114.0,
        max_cached_images=10,  # Reduced for memory
        random_pop=False),
    dict(
        type='RandomResize',
        scale=(1280, 1280),
        ratio_range=(0.8, 1.2),  # Less aggressive for conveyor
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad4Channel', size=(640, 640), pad_val=dict(img=(114, 114, 114, 0))),  # 4-channel padding
    dict(type='PackDetInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    # Generate heatmap channel for validation
    dict(type='GenerateHeatmapChannel', method='prior_based', heatmap_strength=1.0),
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
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='images/'),
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
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=val_pipeline,
        metainfo=dict(classes=('conveyor_object',))
    )
)

test_dataloader = val_dataloader

# Evaluation
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric='bbox',
    format_only=False
)
test_evaluator = val_evaluator

# Training schedule
max_epochs = 300
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=10
)

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.004, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True)
)

# Learning rate schedule
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0002,
        begin=1000,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=False,
        convert_to_iter_based=True)
]

# Hooks
default_hooks = dict(
    checkpoint=dict(interval=50, max_keep_ckpts=3),
    logger=dict(interval=50)
)

# Runtime settings
work_dir = './work_dirs/rtmdet_tiny_4ch_conveyor'