#!/usr/bin/env python3
"""
Check our actual training configuration to understand the heatmap parameters.
"""

import sys
sys.path.insert(0, '/home/robun2/Documents/vault_conveyor_tracking/vault_mmdetection')

# Import the config
try:
    from mmengine.config import Config
    cfg = Config.fromfile('configs/rtmdet/rtmdet_tiny_4ch_simple_test.py')
    
    print("=== HEATMAP GENERATION CONFIGURATION ===")
    
    # Find the RobustHeatmapGeneration in the pipeline
    train_pipeline = cfg.train_pipeline
    
    for i, transform in enumerate(train_pipeline):
        if transform['type'] == 'RobustHeatmapGeneration':
            print(f"Found RobustHeatmapGeneration at position {i}:")
            for key, value in transform.items():
                if key != 'type':
                    print(f"  {key}: {value}")
            break
    else:
        print("RobustHeatmapGeneration not found in pipeline!")
        
    print(f"\n=== TRAINING PARAMETERS ===")
    print(f"Training epochs: {cfg.train_cfg.max_epochs}")
    print(f"Dataset: {cfg.dataset_type}")
    print(f"Data root: {cfg.data_root}")
    
    # Check batch size and data loading
    print(f"Batch size: {cfg.train_dataloader.batch_size}")
    print(f"Num workers: {cfg.train_dataloader.num_workers}")
    print(f"Persistent workers: {cfg.train_dataloader.persistent_workers}")
    
    print(f"\n=== HEATMAP QUALITY ANALYSIS ===")
    print("Based on the configuration:")
    
    heatmap_config = None
    for transform in train_pipeline:
        if transform['type'] == 'RobustHeatmapGeneration':
            heatmap_config = transform
            break
    
    if heatmap_config:
        noise_ratio = heatmap_config.get('noise_ratio', 0.2)
        error_ratio = heatmap_config.get('error_ratio', 0.05)
        quality_variance = heatmap_config.get('quality_variance', True)
        no_heatmap_ratio = heatmap_config.get('no_heatmap_ratio', 0.0)
        
        print(f"• {noise_ratio*100:.1f}% of heatmaps get positional noise")
        print(f"• {error_ratio*100:.1f}% of heatmaps get deliberate errors")
        print(f"• Quality variance enabled: {quality_variance}")
        print(f"• {no_heatmap_ratio*100:.1f}% of samples get zero heatmaps (pure RGB)")
        
        # Calculate expected signal quality
        good_heatmaps = (1 - error_ratio) * 100
        print(f"• Expected ~{good_heatmaps:.1f}% good quality heatmaps")
        
except Exception as e:
    print(f"Error loading config: {e}")
    import traceback
    traceback.print_exc()