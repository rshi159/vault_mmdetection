#!/bin/bash
"""
RGB Foundation Model Training Script

Trains a 300-epoch RGB-only foundation model with heatmap channel forced to zeros.
This creates a strong baseline that can be loaded for any subsequent heatmap experiments.

Usage:
    ./train_rgb_foundation.sh
    
The resulting model will be saved to work_dirs/rgb_foundation_300ep/ and can be loaded
for any future experiments with:
    --cfg-options load_from=work_dirs/rgb_foundation_300ep/best_coco_bbox_mAP_epoch_*.pth
"""

CONFIG="configs/rtmdet/rtmdet_4ch_stage1_rgb_only.py"
WORKDIR="work_dirs/rgb_foundation_300ep"

echo "🔧 RGB Foundation Model Training"
echo "================================="
echo "📋 Config: $CONFIG"
echo "📁 Output: $WORKDIR"
echo "📊 Epochs: 300"
echo "🎯 Strategy: RGB-only with zero heatmap"
echo ""
echo "💡 This creates a foundation model that can be loaded for any subsequent experiments"
echo "🚀 Starting training..."
echo ""

python tools/train.py $CONFIG \
    --work-dir $WORKDIR \
    --cfg-options \
    load_from=None

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "🎉 RGB Foundation Training Completed!"
    echo "====================================="
    echo "📁 Model saved to: $WORKDIR"
    echo ""
    echo "📋 To use this foundation model in future experiments:"
    echo "   python tools/train.py your_config.py \\"
    echo "     --cfg-options load_from=\"$WORKDIR/best_coco_bbox_mAP_epoch_*.pth\""
    echo ""
    echo "💡 Expected characteristics:"
    echo "   • Strong RGB visual features"
    echo "   • Balanced channel weights (no heatmap bias)"
    echo "   • Ready for heatmap integration experiments"
    echo ""
    
    # Show available checkpoints
    echo "📄 Available checkpoints:"
    ls -la $WORKDIR/*.pth 2>/dev/null | head -5
    
else
    echo "❌ Training failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi