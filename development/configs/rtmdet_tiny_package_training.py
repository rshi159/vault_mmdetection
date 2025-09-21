
# RTMDet-tiny Package Detection Training Configuration
_base_ = [
    '/home/robun2/Documents/vault_conveyor_tracking/vault_mmdetection/configs/_base_/models/rtmdet_tiny.py',
    '/home/robun2/Documents/vault_conveyor_tracking/vault_mmdetection/configs/_base_/datasets/coco_detection.py',
    '/home/robun2/Documents/vault_conveyor_tracking/vault_mmdetection/configs/_base_/schedules/schedule_1x.py',
    '/home/robun2/Documents/vault_conveyor_tracking/vault_mmdetection/configs/_base_/default_runtime.py'
]

# Dataset settings
dataset_type = 'CocoDataset'
data_root = '/home/robun2/Documents/vault_conveyor_tracking/vault_mmdetection/development/augmented_data/'
metainfo = {
    'classes': ('package',),
    'palette': [(255, 0, 0)]  # Red color for package class
}

# Model settings - modify for single class
model = dict(
    bbox_head=dict(
        num_classes=1,  # Only package class
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
    ),
    # Use pre-trained COCO weights as starting point
    init_cfg=dict(
        type='Pretrained',
        checkpoint='https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
    )
)

# Data pipeline with augmentation
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomShift', shift_ratio=0.1, prob=0.5),
    dict(
        type='ColorTransform',
        level=5,
        prob=0.5
    ),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs')
]

# Dataset configuration
train_dataloader = dict(
    batch_size=8,  # Adjust based on GPU memory
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='train/annotations.json',  # Will be created
        data_prefix=dict(img='train/images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline
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
        metainfo=metainfo,
        data_root=data_root,
        ann_file='valid/annotations.json',  # Will be created
        data_prefix=dict(img='valid/images/'),
        test_mode=True,
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader

# Evaluation settings
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'valid/annotations.json',
    metric='bbox',
    format_only=False
)
test_evaluator = val_evaluator

# Training schedule - fine-tuning setup
max_epochs = 50  # Fewer epochs for fine-tuning
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer settings - lower learning rate for fine-tuning
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05),  # Lower LR
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True
    )
)

# Learning rate schedule
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=1000
    ),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[30, 40],
        gamma=0.1
    )
]

# Logging and checkpointing
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', 
        interval=5,  # Save every 5 epochs
        max_keep_ckpts=3,  # Keep only 3 checkpoints
        save_best='auto'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

# Work directory
work_dir = '/home/robun2/Documents/vault_conveyor_tracking/vault_mmdetection/development/work_dirs/rtmdet_tiny_package_training'

# Load pre-trained weights but reset classification head for new classes
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
resume = False

# Visualization settings
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)
