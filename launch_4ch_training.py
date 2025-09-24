#!/usr/bin/env python3
"""
Launch fresh 4-channel training with the clean config.
Uses standard MMDetection tools/train.py but applies 4th channel freezing hook.
"""

import os
import sys
import subprocess

def main():
    """Launch training with the clean 4-channel config."""
    
    config_file = "configs/rtmdet/rtmdet_4ch_zeroheatmap_fresh.py"
    
    print("🚀 Launching Fresh 4-Channel Training")
    print(f"   • Config: {config_file}")
    print(f"   • Zero heatmap mode: RGB channels with 4th channel frozen at zero")
    print(f"   • Load from: Properly inflated 4-channel checkpoint")
    print(f"   • Fresh work directory: No auto-resume")
    print()
    
    # Build command
    cmd = [
        "python", "tools/train.py", 
        config_file,
        "--work-dir", "work_dirs/rtmdet_4ch_zeroheatmap_fresh",
        "--amp"  # Enable automatic mixed precision
    ]
    
    print("🎯 Command:")
    print("   " + " ".join(cmd))
    print()
    
    # Execute training
    try:
        # Change to the correct directory
        os.chdir("/home/robun2/Documents/vault_conveyor_tracking/vault_mmdetection")
        
        # Activate environment and run
        env = os.environ.copy()
        env["PATH"] = "/home/robun2/.venvs/mmdet311/bin:" + env["PATH"]
        
        subprocess.run(cmd, env=env, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        sys.exit(0)

if __name__ == "__main__":
    main()