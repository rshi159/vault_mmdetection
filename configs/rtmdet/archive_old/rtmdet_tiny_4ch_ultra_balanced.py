"""Ultra-aggressive heatmap balancing test configuration."""

_base_ = './rtmdet_tiny_4ch_production_fixed.py'

# Quick test settings
max_epochs = 2
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=2,
    val_interval=1)

# Ultra-aggressive heatmap balancing
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
        no_heatmap_ratio=0.40,  # 40% completely no heatmap!
        partial_heatmap_ratio=0.25,  # 25% missing heatmaps for some parcels
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

# Update train dataloader
train_dataloader = dict(
    batch_size=8,  # Smaller for testing
    num_workers=4,
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

val_dataloader = dict(batch_size=8, num_workers=4)

# Monitoring hooks
custom_hooks = [
    # PriorH Channel Weight Monitoring - epoch level only
    dict(
        type='PriorHMonitorHook',
        backbone_path='backbone.stem',
        log_interval=25,  # Keep some iteration logging for testing
        log_after_epoch=True),
    
    # Prediction Visualization
    dict(
        type='PredictionVisualizationHook',
        vis_interval=1,  # Every epoch for testing
        output_dir='work_dirs/vis_outputs',
        max_images=5,
        score_thr=0.1,
        show_gt=True),
]

# Summary of heatmap strategy:
# - 40% images: ZERO heatmap (pure RGB)
# - 25% images: PARTIAL heatmaps (some parcels missing)
# - 35% images: FULL heatmaps (all parcels have heatmaps)
# This should drastically reduce PriorH dependency!