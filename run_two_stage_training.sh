#!/bin/bash
"""
Two-Stage Training Script for RTMDet 4-Channel Model

This script implements the two-stage training strategy:
1. Stage 1: RGB-only training (heatmap = zeros) to establish visual features
2. Stage 2: Heatmap fine-tuning at reduced learning rate

Usage:
    ./run_two_stage_training.sh [stage1_epochs] [stage2_epochs]
    
Example:
    ./run_two_stage_training.sh 120 60
"""

# Configuration
STAGE1_CONFIG="configs/rtmdet/rtmdet_4ch_stage1_rgb_only.py"
STAGE2_CONFIG="configs/rtmdet/rtmdet_4ch_stage2_heatmap_finetune.py"
PRETRAINED_WEIGHTS="https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_l_8xb32-300e_coco/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth"

# Work directories
STAGE1_WORKDIR="work_dirs/stage1_rgb_foundation"
STAGE2_WORKDIR="work_dirs/stage2_heatmap_integration"

# Parse arguments
STAGE1_EPOCHS=${1:-120}
STAGE2_EPOCHS=${2:-60}

echo "🔧 Two-Stage Training Strategy"
echo "=================================="
echo "Stage 1: RGB Foundation ($STAGE1_EPOCHS epochs)"
echo "Stage 2: Heatmap Integration ($STAGE2_EPOCHS epochs)"
echo ""

# Stage 1: RGB Foundation Training
echo "🚀 Starting Stage 1: RGB Foundation Training..."
echo "   • Config: $STAGE1_CONFIG"
echo "   • Epochs: $STAGE1_EPOCHS"
echo "   • Strategy: Heatmap channel forced to zeros"
echo "   • Goal: Establish strong RGB visual features"
echo ""

python tools/train.py $STAGE1_CONFIG \
    --work-dir $STAGE1_WORKDIR \
    --cfg-options \
    max_epochs=$STAGE1_EPOCHS \
    load_from=None

STAGE1_EXIT_CODE=$?

if [ $STAGE1_EXIT_CODE -ne 0 ]; then
    echo "❌ Stage 1 training failed with exit code $STAGE1_EXIT_CODE"
    exit $STAGE1_EXIT_CODE
fi

echo "✅ Stage 1 completed successfully!"

# Find the best Stage 1 checkpoint
STAGE1_BEST_CKPT="$STAGE1_WORKDIR/best_coco_bbox_mAP_epoch_*.pth"
if ls $STAGE1_BEST_CKPT 1> /dev/null 2>&1; then
    STAGE1_CHECKPOINT=$(ls $STAGE1_BEST_CKPT | head -1)
    echo "📁 Using Stage 1 checkpoint: $STAGE1_CHECKPOINT"
else
    # Fallback to latest epoch
    STAGE1_CHECKPOINT="$STAGE1_WORKDIR/epoch_$STAGE1_EPOCHS.pth"
    echo "📁 Using Stage 1 final checkpoint: $STAGE1_CHECKPOINT"
fi

# Stage 2: Heatmap Fine-tuning
echo ""
echo "🎯 Starting Stage 2: Heatmap Integration Fine-tuning..."
echo "   • Config: $STAGE2_CONFIG"
echo "   • Epochs: $STAGE2_EPOCHS"
echo "   • Strategy: 80% heatmap suppression + reduced LR"
echo "   • Goal: Add spatial knowledge without overwhelming RGB"
echo "   • Loading from: $STAGE1_CHECKPOINT"
echo ""

python tools/train.py $STAGE2_CONFIG \
    --work-dir $STAGE2_WORKDIR \
    --cfg-options \
    max_epochs=$STAGE2_EPOCHS \
    load_from="$STAGE1_CHECKPOINT"

STAGE2_EXIT_CODE=$?

if [ $STAGE2_EXIT_CODE -ne 0 ]; then
    echo "❌ Stage 2 training failed with exit code $STAGE2_EXIT_CODE"
    exit $STAGE2_EXIT_CODE
fi

echo ""
echo "🎉 Two-Stage Training Completed Successfully!"
echo "=============================================="
echo "Stage 1 (RGB Foundation): $STAGE1_WORKDIR"
echo "Stage 2 (Heatmap Integration): $STAGE2_WORKDIR"
echo ""
echo "📊 Next steps:"
echo "1. Check weight monitoring logs for RGB:PriorH ratio evolution"
echo "2. Validate that RGB features are preserved in Stage 2"
echo "3. Test inference with both checkpoints"
echo ""
echo "💡 Expected outcome:"
echo "   • Stage 1: Strong RGB features, balanced channel weights"
echo "   • Stage 2: Enhanced spatial knowledge, controlled PriorH:RGB ratio (<5x)"