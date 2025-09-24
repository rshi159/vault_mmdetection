#!/usr/bin/env python3

"""
Training script with warnings suppressed
"""

import os
import warnings

# Suppress known warnings
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
warnings.filterwarnings('ignore', category=UserWarning, module='torch.functional')

# Run training
import subprocess
import sys

def main():
    cmd = [
        sys.executable, 'tools/train.py',
        'configs/rtmdet/rtmdet_4ch_zeroheatmap_fresh.py'
    ]
    
    print("🚀 Starting 4-channel training with warnings suppressed...")
    result = subprocess.run(cmd, cwd='/home/robun2/Documents/vault_conveyor_tracking/vault_mmdetection')
    return result.returncode

if __name__ == '__main__':
    sys.exit(main())