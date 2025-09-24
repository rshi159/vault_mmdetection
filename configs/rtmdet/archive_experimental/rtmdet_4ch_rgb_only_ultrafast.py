# Ultra-Fast RGB-Only Foundation Training - 300 Epochs (ENHANCED)
# This config implements the fastest possible RGB-only training by:
# 1. Using RGBOnly4Channel for direct RGB->RGBZ conversion (bypasses all heatmap generation)
# 2. Freezing 4th channel weights in first conv layer (RGBOnlyTrainingHook)
# 3. EMA for smoother training and better validation AP
# 4. bfloat16 AMP for faster training
# 5. Improved augmentation stability and gradient clipping

_base_ = './rtmdet_tiny_4ch_production_fixed.py'

# ============================================================================
# TRAINING SETTINGS - 300 Epoch Foundation Training
# ============================================================================

# Extended training for foundation model with improved validation cadence
max_epochs = 300
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=5)  # 🔧 IMPROVED: More frequent validation

# Learning rate schedule - extended for 300 epochs (IMPROVED)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),  # warmup in iters
    dict(
        type='CosineAnnealingLR',
        eta_min=0.002 * 0.05,  # 5% of base LR
        begin=0,
        end=max_epochs,
        by_epoch=True          # 🔧 FIXED: Removed convert_to_iter_based redundancy
    ),
]

# ============================================================================
# OPTIMIZER WITH IMPROVED AMP - High Performance Training
# ============================================================================

# Enhanced optimizer with bfloat16 AMP and gradient clipping
optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='bfloat16',           # 🔧 IMPROVED: bfloat16 for better stability (fallback to float16 if unsupported)
    optimizer=dict(
        type='AdamW', 
        lr=0.002, 
        weight_decay=0.05
    ),
    clip_grad=dict(max_norm=5.0),  # 🔧 NEW: Gradient clipping for stability
)

# ============================================================================
# ULTRA-FAST RGB-ONLY DATA PIPELINE - IMPROVED & STABLE
# ============================================================================

# Main training pipeline with proper 4-channel handling and AABB-safe augmentations
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='YOLOXHSVRandomAug'),  # RGB-only photometric augmentation BEFORE 4-channel conversion
    dict(type='RGBOnly4Channel'),    # Add zero 4th channel AFTER color augmentation
    
    dict(type='CachedMosaic',
         img_scale=(640, 640),
         pad_val=(114, 114, 114, 0),  # 🔧 FIXED: Proper 4-channel padding
         max_cached_images=40,
         random_pop=False),
    
    dict(type='RandomResize',
         scale=(1280, 1280),
         ratio_range=(0.7, 1.3),      # 🔧 FIXED: Gentler scale range (was 0.1-2.0)
         keep_ratio=True),
    
    # 🔧 REMOVED: RandomCrop after Mosaic (can chop objects and add label churn)
    # Keep only AABB-safe geometric transforms
    dict(type='RandomFlip', prob=0.5),
    
    # 🔧 NEW: Filter bad boxes after geometric operations
    dict(type='FilterAnnotations', 
         min_gt_bbox_wh=(2, 2),       # Remove tiny boxes (< 2x2 pixels)
         keep_empty=False),           # Remove images with no valid boxes
    
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114, 0))),  # Correct 4-channel padding
    
    dict(type='CachedMixUp',
         img_scale=(640, 640),
         ratio_range=(0.8, 1.0),      # 🔧 FIXED: Gentler blending (was 1.0-1.0)
         max_cached_images=20,
         random_pop=False,
         pad_val=(114, 114, 114, 0),  # 🔧 FIXED: Proper 4-channel padding
         prob=0.25),                  # 🔧 FIXED: Reduced probability (was 0.5)
    
    dict(type='PackDetInputs')
]

# ============================================================================
# LOGGING & MONITORING CONFIGURATION - OVERNIGHT TRAINING
# ============================================================================

# Epoch-based logging for overnight monitoring
log_level = 'INFO'
log_processor = dict(
    by_epoch=True, 
    type='LogProcessor', 
    window_size=1,  # Single epoch window for clean logging
    num_digits=4
)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(
        type='LoggerHook', 
        interval=200        # 🔧 IMPROVED: More frequent logging (was 1000) for mid-epoch signals
    ),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', 
        interval=5,         # 🔧 IMPROVED: Align with validation (every 5 epochs)
        max_keep_ckpts=8,   # 🔧 IMPROVED: Keep more checkpoints for longer training
        save_best='auto'    # Save best model automatically
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

# ============================================================================
# ENHANCED TRAINING HOOKS - EMA + RGB-Only Training
# ============================================================================

custom_hooks = [
    dict(
        type='EMAHook', 
        momentum=0.0002,            # 🔧 NEW: Exponential Moving Average for smoother training
        priority=49,                # High priority to run before other hooks
        update_buffers=True         # Update batch norm buffers in EMA model
    ),
    dict(
        type='RGBOnlyTrainingHook',
        zero_4th_channel=True,      # Zero and freeze 4th channel weights
        monitor_weights=False,      # Disable weight monitoring for max speed
        log_interval=None           # No weight logging
    )
]

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

work_dir = './work_dirs/rgb_foundation_ultrafast_300ep'

# Visualization settings for monitoring
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# ============================================================================
# DATALOADER OPTIMIZATION - Ultra-Fast Throughput
# ============================================================================

# Optimized dataloaders for maximum throughput
train_dataloader = dict(
    persistent_workers=True,    # 🔧 NEW: Keep workers alive between epochs
    pin_memory=True,           # 🔧 NEW: Faster GPU transfers
    num_workers=8              # 🔧 NEW: Parallel data loading (adjust based on your CPU)
)

val_dataloader = dict(
    persistent_workers=True,    # 🔧 NEW: Keep workers alive for validation
    pin_memory=True            # 🔧 NEW: Faster GPU transfers
)

# ============================================================================
# REPRODUCIBILITY & RANDOMNESS
# ============================================================================

# Seed for reproducibility while maintaining speed
randomness = dict(
    seed=42,                   # 🔧 NEW: Fixed seed for reproducibility
    deterministic=False        # Keep False for cudnn autotune speed
)

# Load from existing checkpoint - resuming from epoch 20
resume = True                  # 🔧 IMPROVED: Use resume instead of load_from for continuing training
# load_from = './work_dirs/rgb_foundation_ultrafast_300ep/epoch_20.pth'  # Commented out - using resume instead

# Print configuration summary
print("🚀 Ultra-Fast RGB Foundation Training - 300 Epochs (IMPROVED)")
print("📈 Pipeline Optimizations:")
print("   ✅ RGBOnly4Channel: Direct RGB->RGBZ (bypasses all heatmap generation)")
print("   ✅ RGBOnlyTrainingHook: Freezes 4th channel weights")
print("   ✅ No RobustHeatmapGeneration: Eliminates computational bottleneck")
print("   🔧 FIXED: Proper 4-channel padding in Mosaic/MixUp")
print("   🔧 FIXED: Gentler augmentation ranges for stability")
print("   FIXED: Reduced MixUp probability (0.5→0.25)")
print("   🔧 FIXED: Improved scale jitter (0.1-2.0→0.7-1.3)")
print("   Expected speedup: 10-50x faster data loading")
print(f"📊 Target: {max_epochs} epochs of pure RGB feature learning")
print("")
print("💡 CURRICULUM LEARNING SUGGESTION:")
print("   • Epochs 0-20: Current stable pipeline (resuming from epoch 20)")
print("   • Epochs 255-300: Consider disabling Mosaic/MixUp for final polish")
print("   • Monitor training stability and adjust augmentation strength as needed")
print("   • EMA weights will be automatically used for validation")

print("")
print("🚀 ENHANCED FEATURES:")
print("   🚀 NEW: EMA (momentum=0.0002) for smoother training")
print("   🚀 NEW: bfloat16 AMP for faster training")  
print("   🚀 NEW: Gradient clipping (max_norm=5.0)")
print("   🚀 NEW: More frequent validation & checkpoints (every 5 epochs)")
print("   🚀 NEW: Optimized dataloaders (persistent_workers + pin_memory)")
print("   � NEW: Better logging cadence (interval=200 for mid-epoch signals)")
print("   🔧 REMOVED: RandomCrop after Mosaic (preserves AABB integrity)")
print("   🔧 IMPROVED: Using resume=True for checkpoint continuation")
print("   Expected speedup: 10-50x faster data loading + AMP acceleration")
print("")
print("⚠️  LATE-PHASE POLISH RECOMMENDATION:")
print("   • At epoch ~255: Create polish config with Mosaic/MixUp disabled")
print("   • This makes training distribution closer to eval for final convergence")
print("   • Keep only: YOLOXHSVRandomAug → RGBOnly4Channel → RandomResize → RandomFlip → Pad")