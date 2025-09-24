"""Production configuration for 300-epoch training with balanced heatmap usage."""

_base_ = './rtmdet_tiny_4ch_production_fixed.py'

# Production training configuration - 300 epochs
max_epochs = 300
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=300,
    val_interval=5)  # Validate every 5 epochs

# Enhanced data pipeline with aggressive heatmap balancing
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', prob=0.3),
    dict(
        type='RobustHeatmapGeneration',
        min_sigma=18.0,
        max_sigma=20.0,
        center_noise_std=1.0,
        keypoint_noise_std=2.0,
        background_noise_std=0.005,
        noise_ratio=0.05,
        no_heatmap_ratio=0.30,  # 30% completely no heatmap (vs 2% before)
        partial_heatmap_ratio=0.20,  # NEW: 20% missing heatmaps for some parcels
        error_ratio=0.01,
        multiplicative_noise_ratio=0.01,
        multiplicative_noise_range=(0.98, 1.02),
        global_noise_ratio=0.0,
        global_noise_std=0.0,
        quality_variance=False
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

# Update train dataloader with new pipeline
train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root='development/augmented_data_production/',
        ann_file='train/annotations.json',
        data_prefix=dict(img='train/images/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=0),
        pipeline=train_pipeline,
        metainfo=dict(
            classes=('package',),
            palette=[(220, 20, 60)]
        )
    )
)

# Production monitoring and hooks  
custom_hooks = [
    # Early stopping with patience starting after epoch 100
    dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP',
        patience=20,  # 20 epoch patience
        min_delta=0.001,
        check_finite=True,
        start_epoch=100),  # Only start checking after epoch 100
    
    # PriorH Channel Weight Monitoring - EPOCH LEVEL ONLY
    dict(
        type='PriorHMonitorHook',
        backbone_path='backbone.stem',
        log_interval=None,  # NO iteration logging
        log_after_epoch=True),  # Only epoch-level logging
    
    # Prediction Visualization every 10 epochs for production
    dict(
        type='PredictionVisualizationHook',
        vis_interval=10,  # Every 10 epochs for production
        output_dir='work_dirs/vis_outputs',
        max_images=10,
        score_thr=0.1,
        show_gt=True),
]

# Enhanced checkpointing for long training
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=9999),  # Minimal iteration logging
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,  # Save every 5 epochs
        save_best='coco/bbox_mAP',
        rule='greater',
        max_keep_ckpts=10),  # Keep more checkpoints for long training
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

# Enhanced logging processor
log_processor = dict(
    type='LogProcessor', 
    window_size=50, 
    by_epoch=True)

# Pretrained weights (set via command line)
load_from = None

# Enhanced learning rate schedule for 300 epochs
param_scheduler = [
    # Warmup for first 1000 iterations
    dict(
        type='LinearLR',
        start_factor=1e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    # Cosine annealing for remaining training
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0001,  # Lower minimum LR for longer training
        by_epoch=False,
        begin=1000,
        end=300000)  # Approximate total iterations for 300 epochs
]

# Validation configuration
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')