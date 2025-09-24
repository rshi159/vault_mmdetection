"""Production configuration with PriorH monitoring and visualization hooks."""

_base_ = './rtmdet_tiny_4ch_production_fixed.py'

# Override max_epochs if needed for testing
# max_epochs = 1  # Uncomment for quick testing

# Add monitoring hooks to the custom_hooks
custom_hooks = [
    # Early stopping
    dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP',
        patience=15,
        min_delta=0.001,
        check_finite=True),
    
    # PriorH Channel Weight Monitoring
    dict(
        type='PriorHMonitorHook',
        log_interval=50,           # Log every 50 iterations
        log_after_epoch=True,      # Log after each epoch
        backbone_path='backbone.stem'  # Path to stem layer
    ),
    
    # Prediction Visualization
    dict(
        type='PredictionVisualizationHook',
        vis_interval=5,            # Visualize every 5 epochs
        score_thr=0.1,            # Low threshold to see early predictions
        output_dir='work_dirs/vis_outputs',
        max_images=10,            # Visualize 10 validation images
        show_gt=True              # Show ground truth boxes
    )
]

# Enhanced logging for monitoring
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=25),  # More frequent logging
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='coco/bbox_mAP',
        rule='greater',
        max_keep_ckpts=5),  # Keep more checkpoints for analysis
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

# More detailed logging processor
log_processor = dict(
    type='LogProcessor', 
    window_size=50, 
    by_epoch=True)

# Pretrained weights (can be overridden via command line)
# Example: --cfg-options load_from=work_dirs/rtmdet_production_corrected/epoch_13.pth
load_from = None  # Set to None by default, override with command line

# Validation configuration - run validation more frequently
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=150,
    val_interval=2)  # Validate every 2 epochs for better monitoring