"""Demo script to start monitored training with PriorH analysis."""

import subprocess
import sys
from pathlib import Path


def run_monitored_training():
    """Start training with PriorH monitoring enabled."""
    print("🚀 Starting RTMDet training with PriorH monitoring...")
    
    # Training command
    cmd = [
        "python", "tools/train.py",
        "configs/rtmdet/rtmdet_tiny_4ch_monitored.py",
        "--work-dir", "work_dirs/rtmdet_monitored_training"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("\n" + "="*60)
    print("MONITORING FEATURES ENABLED:")
    print("="*60)
    print("📊 PriorH Weight Norms: Every 50 iterations + each epoch")
    print("🖼️  Prediction Visualizations: Every 5 epochs") 
    print("📈 Enhanced Logging: Every 25 iterations")
    print("💾 Checkpointing: Every epoch with best model saving")
    print("="*60)
    
    print("\nWhat to watch for:")
    print("✅ PriorH norm should start near 0 and gradually increase")
    print("✅ RGB/PriorH ratio should stabilize around 10-20% by epoch 50+")
    print("✅ Prediction visualizations should show improving box quality")
    print("✅ mAP should start improving after PriorH norm reaches ~0.1")
    
    print(f"\nOutputs will be saved to:")
    print(f"  - Logs: work_dirs/rtmdet_monitored_training/")
    print(f"  - Visualizations: work_dirs/vis_outputs/")
    print(f"  - Checkpoints: work_dirs/rtmdet_monitored_training/")
    
    print("\nPress Ctrl+C to stop training at any time...")
    print("="*60)
    
    try:
        # Run the training
        result = subprocess.run(cmd, check=True)
        print("✅ Training completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")


def show_analysis_commands():
    """Show commands for analyzing results."""
    print("\n" + "="*60)
    print("POST-TRAINING ANALYSIS COMMANDS:")
    print("="*60)
    
    print("\n1. Analyze PriorH weight evolution:")
    print("   python analyze_priorh_logs.py work_dirs/rtmdet_monitored_training/[timestamp].log")
    
    print("\n2. View prediction visualizations:")
    print("   ls work_dirs/vis_outputs/")
    print("   # Open images to see learning progress")
    
    print("\n3. TensorBoard (if available):")
    print("   tensorboard --logdir work_dirs/rtmdet_monitored_training/")
    
    print("\n4. Check final model performance:")
    print("   python tools/test.py \\")
    print("     configs/rtmdet/rtmdet_tiny_4ch_monitored.py \\")
    print("     work_dirs/rtmdet_monitored_training/best_coco_bbox_mAP_*.pth \\")
    print("     --show-dir results_visualization")


def main():
    """Main demo function."""
    print("=== RTMDet 4-Channel PriorH Monitoring Demo ===\n")
    
    # Check if config exists
    config_path = Path("configs/rtmdet/rtmdet_tiny_4ch_monitored.py")
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        print("Please ensure you're in the MMDetection root directory")
        return
    
    print("Configuration: configs/rtmdet/rtmdet_tiny_4ch_monitored.py")
    print("Features: PriorH monitoring + prediction visualization + enhanced logging")
    
    # Ask user if they want to start training
    response = input("\nStart monitored training? (y/n): ").lower().strip()
    
    if response == 'y' or response == 'yes':
        run_monitored_training()
        show_analysis_commands()
    else:
        print("Training not started. You can run it manually with:")
        print("  python tools/train.py configs/rtmdet/rtmdet_tiny_4ch_monitored.py --work-dir work_dirs/rtmdet_monitored")
        show_analysis_commands()


if __name__ == '__main__':
    main()