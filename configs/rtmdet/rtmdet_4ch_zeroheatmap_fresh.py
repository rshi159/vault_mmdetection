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
        pad_value=114,                           # 🔧 CONSISTENCY: Match Mosaic pad_val for uniform borders
        batch_augments=None
    ),
    backbone=dict(
        type='CSPNeXt4Ch',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.33,                      # 🔧 HOT-START: RTMDet-S for strong pretrained features
        widen_factor=0.5,                        # 🔧 HOT-START: RTMDet-S width matches pretrained checkpoint
        channel_attention=True,                  # 🔧 MATCH REFERENCE: Enabled in successful config
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=None,
        out_indices=(1, 2, 3, 4),                # 🔧 P2 ADDITION: P2,P3,P4,P5 -> strides 4,8,16,32
    ),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[64, 128, 256, 512],         # 🔧 HOT-START: RTMDet-S channels (widen_factor=0.5)
        out_channels=128,                        # 🔧 HOT-START: More capacity for P2 features
        num_csp_blocks=1,
        expand_ratio=0.5,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU')
    ),
    bbox_head=dict(
        type='RTMDetHead',
        num_classes=1,
        in_channels=128,                         # 🔧 HOT-START: Match neck out_channels
        feat_channels=128,                       # 🔧 HOT-START: More capacity for P2 tiny objects
        stacked_convs=2,                         # ✅ REVERT: Original layers to match checkpoint
        anchor_generator=dict(
            type='MlvlPointGenerator', 
            offset=0, 
            strides=[4, 8, 16, 32]  # 🔧 P2 LEVEL: stride=4 (P2) for tiny parcels
        ),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0                      # 🔧 CONSERVATIVE: Prevent early grad spikes
        ),
        loss_bbox=dict(
            type='CIoULoss',                     # 🔧 TINY BOXES: CIoU better for small object regression
            loss_weight=2.0                      # 🔧 CONSERVATIVE: Prevent early grad spikes
        ),
        # 🔧 RTMDet-specific parameters
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU')
    ),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(
            type='DynamicSoftLabelAssigner',
            topk=13,                             # 🔧 OPTIMAL: Standard topk for RTMDet-S
            iou_calculator=dict(type='BboxOverlaps2D')
        ),
        debug=False,
        pos_weight=-1),
    test_cfg=dict(
        nms_pre=8000,                            # 🔧 SMALL OBJ: Even more candidates for better recall
        min_bbox_size=0,                         # ✅ SMALL OBJ: Allow smallest boxes
        score_thr=0.02,                          # 🔧 SMALL OBJ: Lower threshold for small parcels
        nms=dict(type='nms', iou_threshold=0.6), # 🔧 SMALL OBJ: Relaxed NMS for packed parcels
        max_per_img=300                          # 🔧 SMALL OBJ: More detections for dense scenes
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
    # 🔧 MOSAIC: Loading/annotations handled in MultiImageMixDataset.dataset.pipeline
    dict(type='Mosaic', img_scale=(640, 640), pad_val=114.0, prob=0.8),
    dict(
        type='RandomAffine', 
        scaling_ratio_range=(0.8, 1.2),     # 🔧 AABB SAFE: Gentle scaling only
        max_rotate_degree=0,                # 🔧 AABB CRITICAL: NO rotation - breaks axis-aligned labels  
        max_translate_ratio=0.1,            # 🔧 AABB SAFE: Small translation only
        border=(-320, -320)
    ),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RGBOnly4Channel'),                # 🔧 POST-MOSAIC: Convert to 4ch after Mosaic (which only supports 3ch)
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),  # 🔧 AABB SAFE: Horizontal flip OK for conveyor
    dict(
        type='FilterAnnotations',
        min_gt_bbox_wh=(4, 4),                   # 🔧 STABILITY: Filter sub-pixel boxes that spike CIoU grads
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
    batch_size=40,                            # 🔧 OPTIMAL: Increased for better GPU utilization (19.2/24 GB usage)
    num_workers=16,                          # 🔧 CPU OPTIMAL: Better utilization of Ryzen 9 7950X (16C/32T)
    prefetch_factor=4,                       # 🔧 PERFORMANCE: Prefetch batches to reduce data_time
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='MultiImageMixDataset',         # 🔧 MOSAIC: Wrapper needed for Mosaic augmentation
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            metainfo=metainfo,
            ann_file='train/annotations.json',
            data_prefix=dict(img='train/images/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=1),  # ✅ FIX: Allow tiny objects
            pipeline=[
                dict(type='LoadImageFromFile', backend_args=None),
                dict(type='LoadAnnotations', with_bbox=True),
                # Keep as 3-channel for Mosaic compatibility
            ],
            backend_args=None
        ),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=40,                            # 🔧 MATCH: Same as training batch size
    num_workers=10,                          # 🔧 IMPROVED: Better CPU utilization for validation
    prefetch_factor=2,                       # 🔧 PERFORMANCE: Light prefetch for validation
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
    batch_size=40,                            # 🔧 MATCH: Same as training batch size
    num_workers=10,                          # 🔧 IMPROVED: Better CPU utilization for test
    prefetch_factor=2,                       # 🔧 PERFORMANCE: Light prefetch for test
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

# 🔧 GPU OPTIMIZATION: Adjusted for 2.5x larger batch size (40 vs 16)
optim_wrapper = dict(
    type='AmpOptimWrapper',                  
    optimizer=dict(
        type='AdamW', 
        lr=1.25e-4,                          # 🔧 SCALED: 2e-4 * (40/64) for batch_size=40
        weight_decay=0.05,                   
        betas=(0.9, 0.999)
    ),
    paramwise_cfg=dict(
        norm_decay_mult=0, 
        bias_decay_mult=0, 
        bypass_duplicate=True
    ),
    clip_grad=dict(max_norm=10, norm_type=2), # 🔧 FIX: More aggressive gradient clipping
    loss_scale='dynamic'                     # 🔧 P2+MOSAIC: Dynamic scaling prevents overflow with P2 and Mosaic
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
    val_interval=5                           # 🔧 MONITORING: More frequent validation for better tracking
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ============================================================================
# HOOKS CONFIGURATION
# ============================================================================

# 🔧 CLEAN: Basic hooks only
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=150),  # 🔧 REDUCED NOISE: Less frequent logging
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', 
        interval=5,
        max_keep_ckpts=10,
        save_best='coco/bbox_mAP',
        rule='greater'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook')
    # 🔧 REMOVED: DetVisualizationHook (causes CPU slowdown)
)

# 🔧 TRAINING FROM SCRATCH: Optimized hooks for better performance
custom_hooks = [
    dict(
        type='EMAHook',                      # ✅ EMA for better convergence
        momentum=4e-4,                       # 🔧 FASTER EMA: Better for from-scratch training
        priority=49
    ),
    dict(
        type='NumClassCheckHook',            # 🔧 SAFETY: Verify class consistency
        priority=48
    ),
    dict(
        type='YOLOXModeSwitchHook',          # 🔧 STABILITY: Disable Mosaic in last 10 epochs
        num_last_epochs=10,
        priority=48
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

work_dir = './work_dirs/rtmdet_4ch_s_fromscratch'        # 🔧 FROM SCRATCH: No pretrained checkpoint available yet
load_from = None                                         # 🔧 FROM SCRATCH: Train from scratch until we create inflated checkpoint
resume = False                                         # 🔧 FRESH: No auto-resume

# ✅ FIX WARNING: Allow partial checkpoint loading (missing data_preprocessor keys are OK)
find_unused_parameters = False

# Reproducibility
randomness = dict(
    seed=42,
    deterministic=False
)

auto_scale_lr = dict(enable=False, base_batch_size=40)   # 🔧 FROM SCRATCH: Match batch size