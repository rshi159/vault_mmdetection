#!/usr/bin/env python3
"""4-Channel RTMDet training script for conveyor belt object detection.

This script provides a complete training pipeline for 4-channel RTMDet models
that process RGB + Heatmap input. It supports multiple training configurations
with varying levels of robustness and anti-overfitting strategies.

Key features:
    - Native 4-channel processing (RGB + Heatmap)
    - Dynamic per-iteration heatmap generation
    - Anti-overfitting strategies (noise, errors, no-heatmap samples)
    - Production-ready configurations
    - Comprehensive validation and error checking

Usage:
    # Test setup (recommended first)
    python train_4channel.py --dry-run
    
    # Start training with dynamic config (recommended)
    python train_4channel.py --config configs/rtmdet/rtmdet_tiny_4ch_dynamic.py
    
    # Alternative robust training
    python train_4channel.py --config configs/rtmdet/rtmdet_tiny_4ch_robust.py

Author: 4-Channel RTMDet Development Team
Date: September 2025
Version: 2.0
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional


def _print_configuration_info(mmdet_root: Path, config_path: Path, 
                             work_dir: Path, config_name: str) -> None:
    """Print comprehensive configuration information.
    
    Args:
        mmdet_root: Path to MMDetection root directory.
        config_path: Path to training configuration file.
        work_dir: Path to training output directory.
        config_name: Name of the configuration file.
    """
    print(f"MMDetection root: {mmdet_root}")
    print(f"Config file: {config_path}")
    print(f"Work directory: {work_dir}")
    
    # Print config-specific information
    if "dynamic" in config_name:
        print("🎯 Using DYNAMIC config - heatmaps regenerated every iteration!")
        print("   • 25% images get NO heatmap (pure RGB training)")
        print("   • 80% get noise, 20% get deliberate errors")
        print("   • Maximum robustness and anti-overfitting")
    elif "robust" in config_name:
        print("🛡️ Using ROBUST config - noisy heatmaps for anti-overfitting")
    else:
        print("⚠️ Using BASIC config - may overfit to heatmaps")


def _validate_setup(config_path: Path) -> bool:
    """Validate training setup including config and data availability.
    
    Args:
        config_path: Path to the training configuration file.
        
    Returns:
        bool: True if setup is valid, False otherwise.
    """
    # Verify config exists
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    # Verify data directory exists
    data_root = Path("/home/robun2/Documents/vault_conveyor_tracking/"
                    "vault_mmdetection/development/augmented_data_production/")
    if not data_root.exists():
        print(f"❌ Data directory not found: {data_root}")
        print("Please ensure your augmented dataset is in the correct location.")
        return False
    
    print(f"✅ Data directory found: {data_root}")
    
    # Check for required training data files
    required_files = {
        "Training annotations": data_root / "train/annotations.json",
        "Validation annotations": data_root / "valid/annotations.json",
        "Training images": data_root / "train/images",
        "Validation images": data_root / "valid/images"
    }
    
    for name, path in required_files.items():
        if not path.exists():
            print(f"❌ {name} not found: {path}")
            return False
        print(f"✅ {name}: {path}")
    
    return True


def _execute_training(mmdet_root: Path, config_path: Path, 
                     work_dir: Path, config_name: str) -> bool:
    """Execute the training command with proper error handling.
    
    Args:
        mmdet_root: Path to MMDetection root directory.
        config_path: Path to training configuration file.
        work_dir: Path to training output directory.
        config_name: Name of the configuration file.
        
    Returns:
        bool: True if training completed successfully, False otherwise.
    """
    # Create work directory
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Build training command
    train_script = mmdet_root / "tools/train.py"
    cmd = [
        sys.executable,
        str(train_script),
        str(config_path),
        "--work-dir", str(work_dir)
    ]
    
    # Print execution information
    print("\\n🎯 Training Command:")
    print(" ".join(cmd))
    print("\\n" + "=" * 60)
    
    if "dynamic" in config_name:
        print("🔄 Dynamic heatmap generation - Each iteration creates new heatmaps!")
        print("📊 Monitor: RGB-only performance, noise robustness, no overfitting")
    
    print("Starting training... (Press Ctrl+C to stop)")
    print("=" * 60)
    
    # Change to MMDetection root directory for execution
    original_cwd = os.getcwd()
    os.chdir(mmdet_root)
    
    try:
        # Execute training
        result = subprocess.run(cmd, check=True)
        print("\\n✅ Training completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\\n❌ Training failed with error code: {e.returncode}")
        return False
        
    except KeyboardInterrupt:
        print("\\n⏹️ Training interrupted by user")
        return False
        
    finally:
        # Restore original working directory
        os.chdir(original_cwd)

def main() -> bool:
    """Main training function with comprehensive setup validation.
    
    Handles configuration selection, path validation, data verification,
    and training execution with proper error handling.
    
    Returns:
        bool: True if training completed successfully, False otherwise.
    """
    print("🚀 Starting 4-channel RTMDet training for conveyor detection")
    print("=" * 60)
    
    # Handle config argument with intelligent defaults
    config_name = "rtmdet_tiny_4ch_dynamic.py"  # Default to dynamic config
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv):
            if arg == "--config" and i + 1 < len(sys.argv):
                config_name = Path(sys.argv[i + 1]).name
                break
    
    # Set up paths with validation
    mmdet_root = Path(__file__).parent.parent
    config_path = mmdet_root / f"configs/rtmdet/{config_name}"
    work_dir = mmdet_root / f"work_dirs/{Path(config_name).stem}"
    
    _print_configuration_info(mmdet_root, config_path, work_dir, config_name)
    
    # Comprehensive validation checks
    if not _validate_setup(config_path):
        return False
    
    # Build and execute training command
    return _execute_training(mmdet_root, config_path, work_dir, config_name)

def dry_run() -> bool:
    """Validate 4-channel training setup without starting actual training.
    
    Performs comprehensive validation of:
        - 4-channel component imports
        - Component instantiation
        - Configuration loading
        - Pipeline compatibility
    
    Returns:
        bool: True if all validation checks pass, False otherwise.
    """
    print("🔍 Dry run - checking 4-channel training setup")
    print("=" * 60)
    
    try:
        # Add MMDetection to Python path
        mmdet_root = Path(__file__).parent.parent
        sys.path.insert(0, str(mmdet_root))
        
        # Test 4-channel component imports
        print("Testing component imports...")
        from mmdet.models.data_preprocessors import DetDataPreprocessor4Ch
        from mmdet.models.backbones import CSPNeXt4Ch
        from mmdet.datasets.transforms import (RobustHeatmapGeneration, 
                                               Pad4Channel)
        
        print("✅ All 4-channel components imported successfully")
        
        # Test component instantiation
        print("Testing component instantiation...")
        preprocessor = DetDataPreprocessor4Ch()
        backbone = CSPNeXt4Ch()
        transform = RobustHeatmapGeneration()
        
        print("✅ All components can be instantiated")
        
        # Test configuration loading
        print("Testing configuration loading...")
        config_path = (mmdet_root / 
                      "configs/rtmdet/rtmdet_tiny_4ch_dynamic.py")
        
        if config_path.exists():
            # Simple config validation without full MMEngine dependency
            with open(config_path, 'r') as f:
                config_content = f.read()
                
            # Basic validation - check for key components
            required_components = [
                'DetDataPreprocessor4Ch',
                'CSPNeXt4Ch',
                'RobustHeatmapGeneration'
            ]
            
            missing_components = [
                comp for comp in required_components 
                if comp not in config_content
            ]
            
            if missing_components:
                print(f"❌ Config missing components: {missing_components}")
                return False
                
            print("✅ Training config validation passed")
            print("   - Model type: RTMDet")
            print("   - Backbone: CSPNeXt4Ch")
            print("   - Preprocessor: DetDataPreprocessor4Ch") 
            print("   - Transforms: RobustHeatmapGeneration")
        else:
            print(f"❌ Config file not found: {config_path}")
            return False
            
        print("\\n🎯 Setup verification complete - ready for training!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you've run the setup and copied 4-channel components")
        return False
        
    except Exception as e:
        print(f"❌ Setup error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--dry-run":
        success = dry_run()
    else:
        success = main()
    
    sys.exit(0 if success else 1)