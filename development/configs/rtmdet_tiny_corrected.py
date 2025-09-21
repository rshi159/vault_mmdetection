# RTMDet-tiny Package Detection Configuration with Corrected COCO Annotations
# Fixed paths and updated for corrected COCO file_name format

_base_ = [
    '../configs/_base_/models/rtmdet_tiny.py',
    '../configs/_base_/datasets/coco_detection.py', 
    '../configs/_base_/schedules/schedule_1x.py',
    '../configs/_base_/default_runtime.py'
]

# Dataset settings
dataset_type = 'CocoDataset'
data_root = 'development/augmented_data_production/'

# Classes and metadata
metainfo = {
    'classes': ('package',),
    'palette': [(220, 20, 60)]  # Crimson red for packages
}

# Training data configuration
train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/annotations.json',
        data_prefix=dict(img='train/images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=(640, 640), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackDetInputs')
        ]
    )
)

# Validation data configuration  
val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='valid/annotations.json',
        data_prefix=dict(img='valid/images/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(640, 640), keep_ratio=True),
            dict(type='PackDetInputs')
        ]
    )
)

# Test dataloader (same as validation)
test_dataloader = val_dataloader

# Model configuration for single class
model = dict(
    bbox_head=dict(
        num_classes=1,  # Only 'package' class
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
    )
)

# Training schedule
max_epochs = 50
train_cfg = dict(max_epochs=max_epochs, val_interval=5)

# Optimizer settings
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.004, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True
    )
)

# Learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000
    ),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0002,
        begin=1,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True
    )
]

# Runtime settings
default_hooks = dict(
    checkpoint=dict(interval=5, max_keep_ckpts=3, save_best='auto'),
    logger=dict(interval=50),
)

# Evaluation settings
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'valid/annotations.json',
    metric='bbox',
    format_only=False
)

test_evaluator = val_evaluator

# Reduce log verbosity and frequency
log_level = 'INFO'
log_processor = dict(window_size=50)

# Load pretrained RTMDet-tiny weights
load_from = None  # Will be set via command line