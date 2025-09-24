# RTMDet 4-Channel Clean Training Configuration
# Fresh start from properly inflated 3ch→4ch checkpoint
# Zero heatmap channel maintained for RGB-only training

# ============================================================================
# BASE CONFIGURATIONS
# =========================================        first_conv_name='backbone.stem.conv',       first_conv_name='backbone.stem.conv',==================================
default_scope = 'mmdet'

# Custom imports for 4-channel components
custom_imports = dict(
    imports=[
        'mmdet.models.data_preprocessors.preprocessor_4ch',
        'mmdet.models.backbones.backbone_4ch',
        'mmdet.datasets.transforms.fast_4ch',
        'mmdet.engine.hooks.rgb_4ch_hook',
    ],
    allow_failed_imports=False
)

# ============================================================================
# MODEL - RTMDet with 4-Channel Input (Zero Heatmap)
# ============================================================================

model = dict(
    type='RTMDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor4Ch',
        bgr_to_rgb=True,                         # 🔧 CONSISTENT: RGB order everywhere
        mean=[123.675, 116.28, 103.53, 0.0],    # 🔧 CONSISTENT: ImageNet RGB + zero heatmap
        std=[58.395, 57.12, 57.375, 1.0],       # 🔧 CONSISTENT: ImageNet RGB + unity heatmap
        pad_size_divisor=32,
        pad_value=0,
        batch_augments=None
    ),
    backbone=dict(
        type='CSPNeXt4Ch',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.167,                     # RTMDet-Tiny depth  
        widen_factor=0.375,                      # 🔧 REVERTED: Back to checkpoint's original widen_factor
        channel_attention=True,                  # 🔧 MATCH REFERENCE: Enabled in successful config
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=None,
        out_indices=(2, 3, 4),
    ),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[96, 192, 384],              # ✅ CORRECT: Match CSPNeXt4Ch actual output channels
        out_channels=96,                         # 🔧 KEEP: This works from original successful config
        num_csp_blocks=1,
        expand_ratio=0.5,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU')
    ),
    bbox_head=dict(
        type='RTMDetHead',
        num_classes=1,
        in_channels=96,                          # 🔧 CORRECTED: Match neck out_channels  
        feat_channels=96,                        # 🔧 CORRECTED: Match checkpoint bbox head
        stacked_convs=2,                         # 🔧 MATCH REFERENCE: More capacity
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
        loss_bbox=dict(
            type='GIoULoss',                     # Match working config
            loss_weight=2.0
        ),
        # 🔧 RTMDet-specific parameters (only supported ones)
        with_objectness=False,                   # RTMDet doesn't use objectness branch
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU')
    ),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(topk=13, type='DynamicSoftLabelAssigner'),
        debug=False,
        pos_weight=-1),
    test_cfg=dict(
        nms_pre=4000,                            # 🔧 UPGRADE: More candidates for small objects
        min_bbox_size=0,
        score_thr=0.03,                          # 🔧 UPGRADE: Lower threshold for small objects
        nms=dict(type='nms', iou_threshold=0.55), # 🔧 UPGRADE: Tighter NMS
        max_per_img=300
    )
)

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

data_root = 'development/augmented_data_production/'
dataset_type = 'CocoDataset'
metainfo = {
    'classes': ('package',),
    'palette': [(220, 20, 60)]
}

# 🔧 CLEAN PIPELINE: RGB-only with zero heatmap
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RGBOnly4Channel'),                # 🔧 CANONICAL: Zero heatmap before normalization
    dict(
        type='Resize',
        scale=(640, 640),
        keep_ratio=True,
        clip_object_border=False
    ),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='FilterAnnotations',
        min_gt_bbox_wh=(1, 1),
        keep_empty=False,
        by_box=True,
        by_mask=False
    ),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'pad_shape')
    )
]

val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RGBOnly4Channel'),                # 🔧 CANONICAL: Zero heatmap before normalization
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'pad_shape')
    )
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='RGBOnly4Channel'),                # 🔧 CANONICAL: Zero heatmap before normalization
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'pad_shape')
    )
]

# Dataset configurations
train_dataloader = dict(
    batch_size=128,                           # 🔧 GPU OPTIMIZATION: Increase for better utilization (24GB VRAM)
    num_workers=16,                          # 🔧 GPU OPTIMIZATION: More workers for faster data loading
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/annotations.json',
        data_prefix=dict(img='train/images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=1),  # ✅ FIX: Allow tiny objects
        pipeline=train_pipeline,
        backend_args=None
    )
)

val_dataloader = dict(
    batch_size=128,                           # 🔧 GPU OPTIMIZATION: Match train batch size
    num_workers=16,                          # 🔧 GPU OPTIMIZATION: More workers
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='valid/annotations.json',
        data_prefix=dict(img='valid/images/'),
        test_mode=False,
        pipeline=val_pipeline,
        backend_args=None
    )
)

test_dataloader = dict(
    batch_size=128,                           # 🔧 GPU OPTIMIZATION: Match train batch size
    num_workers=16,                          # 🔧 GPU OPTIMIZATION: More workers
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='valid/annotations.json',
        data_prefix=dict(img='valid/images/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=None
    )
)

# Evaluation
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'valid/annotations.json',
    metric='bbox',
    format_only=False,
    backend_args=None
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'valid/annotations.json',
    metric='bbox',
    format_only=False,
    backend_args=None
)

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# 🔧 GPU OPTIMIZATION: Adjusted for 4x larger batch size (64 vs 16)
optim_wrapper = dict(
    type='AmpOptimWrapper',                  
    optimizer=dict(
        type='AdamW', 
        lr=4e-4,                             # 🔧 GPU OPTIMIZATION: Scale LR for larger batch (4x increase)
        weight_decay=0.05,                   
        betas=(0.9, 0.999)
    ),
    paramwise_cfg=dict(
        norm_decay_mult=0, 
        bias_decay_mult=0, 
        bypass_duplicate=True
    ),
    clip_grad=dict(max_norm=10, norm_type=2), # 🔧 FIX: More aggressive gradient clipping
    loss_scale=512.0                         # 🔧 FIX: Fixed loss scale to prevent NaN gradients
)

# ✅ FIX: Proper warmup schedule for RTMDet 
param_scheduler = [
    dict(
        type='LinearLR', 
        start_factor=1e-3,                   # ✅ Standard warmup factor
        by_epoch=False,
        begin=0, 
        end=1000                             # ✅ 1000 iteration warmup
    ),
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        T_max=200,                           # 🔧 UPGRADE: 200 epochs
        eta_min=1e-6
    )
]

# Training loop
train_cfg = dict(
    type='EpochBasedTrainLoop', 
    max_epochs=200,                          # 🔧 UPGRADE: More epochs for convergence
    val_interval=10
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ============================================================================
# HOOKS CONFIGURATION
# ============================================================================

# 🔧 CLEAN: Basic hooks only
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', 
        interval=10,
        max_keep_ckpts=5,
        save_best='coco/bbox_mAP',
        rule='greater'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook')
    # 🔧 REMOVED: DetVisualizationHook (causes CPU slowdown)
)

# ✅ FIX: Only EMA hook (pipeline already zeros 4th channel)
custom_hooks = [
    dict(
        type='EMAHook',                      # ✅ EMA for better convergence
        momentum=2e-4,
        priority=49
    )
]

# ============================================================================
# ENVIRONMENT & PATHS
# ============================================================================

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', 
    vis_backends=vis_backends, 
    name='visualizer'
)

log_processor = dict(
    type='LogProcessor',
    window_size=50,
    by_epoch=True
)

log_level = 'INFO'

# ============================================================================
# CHECKPOINT CONFIGURATION - FRESH START
# ============================================================================

work_dir = './work_dirs/rtmdet_4ch_zeroheatmap_fresh'  # 🔧 FRESH: New directory
load_from = './work_dirs/rtmdet_optimized_training/best_coco_bbox_mAP_epoch_195_4ch.pth'  # 🔧 INFLATED: 4ch checkpoint
resume = False                                         # 🔧 FRESH: No auto-resume

# ✅ FIX WARNING: Allow partial checkpoint loading (missing data_preprocessor keys are OK)
find_unused_parameters = False

# Reproducibility
randomness = dict(
    seed=42,
    deterministic=False
)

auto_scale_lr = dict(enable=False, base_batch_size=128)  # 🔧 GPU OPTIMIZATION: Match new batch size