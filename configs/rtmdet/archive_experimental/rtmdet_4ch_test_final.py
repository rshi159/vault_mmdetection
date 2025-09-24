"""Quick test for final production configuration."""

_base_ = './rtmdet_4ch_production_final.py'

# Quick test settings - 2 epochs
max_epochs = 2
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=2,
    val_interval=1)  # Validate every epoch

# Smaller batches for testing
train_dataloader = dict(
    batch_size=8,
    num_workers=4)

val_dataloader = dict(
    batch_size=8,
    num_workers=4)

# Test hooks with some iteration logging for verification
custom_hooks = [
    # PriorH Channel Weight Monitoring
    dict(
        type='PriorHMonitorHook',
        backbone_path='backbone.stem',
        log_interval=50,  # Some iteration logging for testing
        log_after_epoch=True),
    
    # Prediction Visualization every epoch for testing
    dict(
        type='PredictionVisualizationHook',
        vis_interval=1,
        output_dir='work_dirs/vis_outputs',
        max_images=5,
        score_thr=0.1,
        show_gt=True),
]