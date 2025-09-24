# Use the working model configuration as base
_base_ = './rtmdet_tiny_4ch_production_fixed.py'

# Override to 300 epochs for RGB foundation training
max_epochs = 300
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=5,
    dynamic_intervals=[(max_epochs-10, 1)]
)

# Override the training pipeline to add ZeroHeatmapTransform
# Follow the working pipeline structure exactly
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', prob=0.3),
    
    # First generate heatmap normally, then zero it out
    dict(
        type='RobustHeatmapGeneration',
        no_heatmap_ratio=1.0,  # Force 100% no heatmap since we're zeroing anyway
        noise_ratio=0.0,
        error_ratio=0.0,
        multiplicative_noise_ratio=0.0,
        global_noise_ratio=0.0,
        background_noise_std=0.0,
        keypoint_noise_std=0.0,
        center_noise_std=0.0,
        multiplicative_noise_range=(1.0, 1.0),
        min_sigma=18.0,
        max_sigma=20.0,
        global_noise_std=0.0,
        quality_variance=False
    ),
    dict(
        type='Pad4Channel',
        size=(640, 640),
        pad_val=dict(img=(114, 114, 114, 0))
    ),
    # CRITICAL: Apply ZeroHeatmapTransform AFTER Pad4Channel creates 4th channel
    dict(type='ZeroHeatmapTransform'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'pad_shape')
    )
]

# Override train dataloader to use our pipeline
train_dataloader = dict(
    dataset=dict(pipeline=train_pipeline)
)

# Override default hooks for epoch-based logging
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,  # Save checkpoint every 5 epochs
        max_keep_ckpts=3,
        save_best='coco/bbox_mAP',
        rule='greater'
    ),
    logger=dict(
        type='LoggerHook', 
        interval=1  # Back to epoch-only logging
    )
)

# Override log processor for epoch-based processing
log_processor = dict(
    by_epoch=True, 
    type='LogProcessor', 
    window_size=1,  # Single epoch window
    num_digits=4
)

# Custom hooks for monitoring RGB foundation training
custom_hooks = [
    dict(
        type='PriorHMonitorHook',
        backbone_path='backbone.stem',
        log_after_epoch=True,
        log_interval=10,  # Temporarily log every 10 iterations for quick verification
    ),
    dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP',
        patience=25,
        min_delta=0.001
    )
]

# Work directory
work_dir = 'work_dirs/rgb_foundation_300ep'

print("🔧 RGB Foundation Training: 300 epochs with zero heatmap")
print(f"📊 Strategy: Use working config + ZeroHeatmapTransform")