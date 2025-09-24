#!/usr/bin/env python
"""
Test script to use the standard tools/train.py with our 4-channel config
This uses the standard MMDetection training infrastructure
"""

import os
import sys
import subprocess

def main():
    """
    Run training using the standard MMDetection tools
    """
    print("🚀 Starting 4-channel RTMDet training with standard tools/train.py")
    print("   • Using official MMDetection training infrastructure")
    print("   • Should handle device placement correctly")
    print()
    
    # Change to the mmdetection directory
    os.chdir('/home/robun2/Documents/vault_conveyor_tracking/vault_mmdetection')
    
    # Set up command
    cmd = [
        'python', 'tools/train.py',
        'configs/rtmdet/rtmdet_4ch_zeroheatmap_fresh.py',
        '--work-dir', './work_dirs/rtmdet_4ch_zeroheatmap_fresh'
    ]
    
    try:
        print("📋 Running command:", ' '.join(cmd))
        print("=" * 60)
        
        # Run the training command
        result = subprocess.run(cmd, check=True)
        
        print("\n✅ Training completed successfully!")
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"\n💥 Training failed with exit code: {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"\n💥 Training failed: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())