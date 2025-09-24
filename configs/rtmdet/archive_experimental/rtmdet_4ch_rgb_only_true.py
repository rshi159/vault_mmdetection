# True RGB-Only Foundation Training - 300 Epochs
# This config implements proper RGB-only training by:
# 1. Zeroing heatmap input data (ZeroHeatmapTransform)
# 2. Freezing 4th channel weights in first conv layer (RGBOnlyTrainingHook)
# 3. Monitoring weight evolution to ensure RGB-focus

_base_ = './rtmdet_tiny_4ch_production_fixed.py'

# ============================================================================
# TRAINING SETTINGS - 300 Epoch Foundation Training
# ============================================================================

# Extended training for foundation model
max_epochs = 300
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=10)

# Learning rate schedule - extended for 300 epochs
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.002 * 0.05,  # 5% of base LR
        begin=0,
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
]

# ============================================================================
# DATA PIPELINE - RGB-Only with Heatmap Zeroing
# ============================================================================

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Pad4Channel'),  # Add 4th channel
    dict(type='ZeroHeatmapTransform'),  # Force 4th channel to zeros
    dict(type='CachedMosaic',
         img_scale=(640, 640),
         pad_val=114.0,
         max_cached_images=40,
         random_pop=False),
    dict(type='RandomResize',
         scale=(1280, 1280),
         ratio_range=(0.1, 2.0),
         keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114, 0))),  # Pad 4th channel with 0
    dict(type='CachedMixUp', img_scale=(640, 640), ratio_range=(1.0, 1.0), max_cached_images=20, random_pop=False, pad_val=(114, 114, 114, 0), prob=0.5),
    dict(type='PackDetInputs')
]

# ============================================================================
# LOGGING & MONITORING CONFIGURATION
# ============================================================================

# Epoch-based logging only
log_level = 'INFO'
log_processor = dict(
    by_epoch=True, 
    type='LogProcessor', 
    window_size=1,  # Single epoch window
    num_digits=4
)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(
        type='LoggerHook', 
        interval=1  # Epoch-only logging
    ),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

# ============================================================================
# RGB-ONLY TRAINING HOOK - True RGB Foundation Training
# ============================================================================

custom_hooks = [
    dict(
        type='RGBOnlyTrainingHook',
        zero_4th_channel=True,  # Zero and freeze 4th channel weights
        monitor_weights=True,   # Monitor weight evolution
        log_interval=None       # Only log after epochs
    )
]

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

work_dir = './work_dirs/rgb_foundation_true_300ep'

# Visualization settings for monitoring
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# Load from existing checkpoint if resuming
# load_from = './work_dirs/rgb_foundation_true_300ep/epoch_X.pth'  # Uncomment if resuming