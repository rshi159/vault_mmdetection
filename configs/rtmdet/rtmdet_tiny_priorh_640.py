
# RTMDet Tiny + Prior Heatmap Configuration  
# Based on rtmdet_tiny_8xb32-300e_coco.py with PriorH integration

# We'll define the full config here instead of importing base to avoid syntax issues

# Custom imports for PriorH components
custom_imports = dict(
    imports=[
        'mmdet.datasets.transforms.transforms',  # For GeneratePriorH
    ],
    allow_failed_imports=False
)

# Dataset metainfo
metainfo = dict(
    classes=('package',),
    palette=[(255, 0, 0)]  # Red for packages
)

# Data root and paths
data_root = 'development/augmented_data_production/'
data_prefix = dict(
    train=dict(img_path='train/images/', ann_file='train/annotations.json'),
    val=dict(img_path='valid/images/', ann_file='valid/annotations.json')
)

# Training pipeline with PriorH generation
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # Geometric transforms MUST come before GeneratePriorH
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    # Generate prior heatmap AFTER geometric transforms
    dict(
        type='GeneratePriorH',
        out_key='prior_h',
        kw=0.15,  # Width scaling factor
        kh=0.15,  # Height scaling factor  
        sigma_min=2.0,  # Minimum std in pixels
        prior_drop_p=0.3,  # 30% chance to zero heatmap
        jitter_px_frac=0.01,  # ±1% image size jitter
        sigma_scale_jitter=(0.9, 1.1),  # Random std scaling
        intensity_scale=(0.9, 1.1),  # Random intensity scaling
        apply_blur=False
    ),
    # Modified PackDetInputs to handle prior_h
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 
                   'scale_factor', 'flip', 'flip_direction')
    ),
]

# Validation pipeline (clean priors, no dropout/noise)
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    # Clean prior generation for validation
    dict(
        type='GeneratePriorH',
        out_key='prior_h',
        kw=0.15,
        kh=0.15,
        sigma_min=2.0,
        prior_drop_p=0.0,  # No dropout in validation
        jitter_px_frac=0.0,  # No jitter in validation
        sigma_scale_jitter=(1.0, 1.0),  # No noise in validation
        intensity_scale=(1.0, 1.0),  # No noise in validation
        apply_blur=False
    ),
    dict(type='PackDetInputs'),
]

# Test pipeline (same as validation)
test_pipeline = val_pipeline

# Dataset configuration
train_dataloader = dict(
    batch_size=8,  # Adjust based on GPU memory with 4-channel input
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=data_prefix['train'],
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=data_prefix['val'],
        test_mode=True,
        pipeline=val_pipeline,
    )
)

test_dataloader = val_dataloader

# Model configuration with 4-channel backbone
model = dict(
    type='RTMDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53, 0.0],  # RGB + PriorH (heatmap mean ~0)
        std=[58.395, 57.12, 57.375, 1.0],     # RGB + PriorH (heatmap std ~1)
        bgr_to_rgb=True,
        batch_augments=None
    ),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.167,
        widen_factor=0.375,
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True),
        in_channels=4  # IMPORTANT: Native 4-channel input (RGB + PriorH)
    ),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[96, 192, 384],
        out_channels=96,
        num_csp_blocks=1,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)
    ),
    bbox_head=dict(
        type='RTMDetHead',
        num_classes=1,  # Single class: package
        in_channels=96,
        stacked_convs=2,
        feat_channels=96,
        anchor_generator=dict(
            type='MlvlPointGenerator',
            offset=0,
            strides=[8, 16, 32]
        ),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0
        ),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        with_objectness=False,
        exp_on_reg=True,
        share_conv=True,
        pred_kernel_size=1,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)
    ),
    train_cfg=dict(
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False
    ),
    test_cfg=dict(
        nms_pre=30000,
        min_bbox_size=0,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300
    )
)

# Training configuration
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.004, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True
    )
)

# Learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000
    ),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0002,
        begin=1000,
        end=300,
        T_max=299000,
        by_epoch=False,
        convert_to_iter_based=True
    )
]

# Training settings
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=300,
    val_interval=10
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Evaluation metrics
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + data_prefix['val']['ann_file'],
    metric='bbox',
    format_only=False
)

test_evaluator = val_evaluator

# Runtime settings
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10, save_best='coco/bbox_mAP'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
