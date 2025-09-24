#!/bin/bash
# RGB Recovery Training with 4090 Optimizations
# This script enables TF32 and launches training with proper environment

cd /home/robun2/Documents/vault_conveyor_tracking/vault_mmdetection
source ~/.venvs/mmdet311/bin/activate

# Enable TF32 for RTX 4090 optimization
export NVIDIA_TF32_OVERRIDE=1

# Enable TF32 in Python before training
python -c "
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print('✅ TF32 enabled for RTX 4090 optimization')

# Now run training
import subprocess
import sys
result = subprocess.run([
    sys.executable, 'tools/train.py', 
    'configs/rtmdet/rtmdet_4ch_rgb_recovery_gentletune.py',
    '--work-dir', './work_dirs/rgb_recovery_4090_optimized'
], cwd='.')
sys.exit(result.returncode)
"