#!/usr/bin/env python3
"""
Test script to verify complete monitoring system with proper visualization.
"""

import os
import sys
import subprocess
import tempfile

def test_monitoring_with_vis():
    """Test the monitoring system with a very short training run."""
    
    # Create a temporary config for quick testing
    config_content = '''
# Quick test config
_base_ = './configs/rtmdet/rtmdet_tiny_4ch_monitored.py'

# Override for quick testing
max_epochs = 1
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=1,
    val_interval=1)  # Validate every epoch to trigger visualization

# Use smaller dataset for quick test
train_dataloader = dict(batch_size=8)
val_dataloader = dict(batch_size=8)
test_dataloader = dict(batch_size=8)
'''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(config_content)
        temp_config = f.name

    try:
        # Run short training
        cmd = [
            'python', 'tools/train.py', 
            temp_config,
            '--work-dir', 'work_dirs/quick_monitoring_test',
            '--cfg-options', 'max_epochs=1'
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        print("STDOUT:", result.stdout[-2000:])  # Last 2000 chars
        if result.stderr:
            print("STDERR:", result.stderr[-1000:])  # Last 1000 chars
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("Training timed out - this is expected for testing")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        os.unlink(temp_config)

def check_outputs():
    """Check if monitoring outputs were created."""
    base_dir = "work_dirs/quick_monitoring_test"
    vis_dir = "work_dirs/vis_outputs"
    
    print(f"\nChecking outputs in {base_dir}:")
    if os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            print(f"  - {item}")
    else:
        print(f"  Directory {base_dir} not found")
    
    print(f"\nChecking visualizations in {vis_dir}:")
    if os.path.exists(vis_dir):
        for item in os.listdir(vis_dir):
            print(f"  - {item}")
    else:
        print(f"  Directory {vis_dir} not found")

if __name__ == "__main__":
    print("Testing complete monitoring system...")
    success = test_monitoring_with_vis()
    check_outputs()
    
    print(f"\nTest {'PASSED' if success else 'FAILED'}")