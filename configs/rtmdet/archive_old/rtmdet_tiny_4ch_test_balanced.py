"""Quick test config to verify balanced heatmap usage."""

_base_ = './rtmdet_tiny_4ch_production_300ep.py'

# Quick test settings
max_epochs = 2
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=2,
    val_interval=1)  # Validate every epoch for testing

# Smaller batch for quick testing
train_dataloader = dict(
    batch_size=8,
    num_workers=4)

val_dataloader = dict(
    batch_size=8,
    num_workers=4)

# Test custom hooks with epoch-only logging
custom_hooks = [
    # No early stopping for quick test
    
    # PriorH Channel Weight Monitoring - epoch level only
    dict(
        type='PriorHMonitorHook',
        backbone_path='backbone.stem',
        log_interval=None,  # No iteration logging
        log_after_epoch=True),  # Only epoch-level logging
    
    # Prediction Visualization every epoch for testing
    dict(
        type='PredictionVisualizationHook',
        vis_interval=1,  # Every epoch for testing
        output_dir='work_dirs/vis_outputs',
        max_images=5,  # Fewer images for testing
        score_thr=0.1,
        show_gt=True),
]