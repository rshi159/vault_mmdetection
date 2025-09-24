#!/usr/bin/env python3
"""
Test if augmentation artifacts are causing detection issues
Run inference with completely clean pipeline (no padding artifacts)
"""

import os
import cv2
import numpy as np
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
import matplotlib.pyplot as plt
import matplotlib.patches as patches

register_all_modules()

def test_clean_inference():
    """Test inference with completely clean images (no padding artifacts)"""
    
    # Model configuration
    config_file = 'configs/rtmdet/rtmdet_4ch_rgb_only_ultrafast.py'
    checkpoint_file = './work_dirs/rgb_foundation_ultrafast_300ep/best_coco_bbox_mAP_epoch_70.pth'
    
    print("🔍 Loading model...")
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    
    # Test with a simple clean image
    test_images = [
        'development/augmented_data_production/train/images/NoColorLights_GlossyParcels_output_Replicator_33_rgb_862.png',
        'development/augmented_data_production/train/images/KFL_overhead_images_10.4.5.64_frame_178.jpg',
        'development/augmented_data_production/train/images/FlatParcels_output_Replicator_33_rgb_11_aug_0.png'
    ]
    
    for img_path in test_images:
        if not os.path.exists(img_path):
            continue
            
        print(f"\n📸 Testing: {os.path.basename(img_path)}")
        
        # Load original image
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        print(f"   Original size: {w}x{h}")
        
        # Test different input sizes to see padding effects
        test_sizes = [
            (640, 640),   # Standard training size
            (w, h),       # Original size (no resize)
            (800, 800),   # Larger size
        ]
        
        results = {}
        for size in test_sizes:
            # Manually resize without any padding artifacts
            if size == (w, h):
                # Original size - no resize
                test_img = img.copy()
                size_label = f"{w}x{h}_original"
            else:
                # Clean resize
                test_img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
                size_label = f"{size[0]}x{size[1]}_resized"
            
            # Convert to RGB and add zero channel
            test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
            test_img_4ch = np.zeros((test_img_rgb.shape[0], test_img_rgb.shape[1], 4), dtype=np.uint8)
            test_img_4ch[:, :, :3] = test_img_rgb
            
            # Run inference
            result = inference_detector(model, test_img_4ch)
            
            # Count detections
            scores = result.pred_instances.scores.cpu().numpy()
            valid_dets = np.sum(scores > 0.3)
            high_conf_dets = np.sum(scores > 0.5)
            
            results[size_label] = {
                'total_dets': valid_dets,
                'high_conf': high_conf_dets,
                'max_score': np.max(scores) if len(scores) > 0 else 0.0
            }
            
            print(f"   {size_label}: {valid_dets} detections (>0.3), {high_conf} high-conf (>0.5), max_score: {results[size_label]['max_score']:.3f}")
        
        # Check for size-dependent detection changes (padding artifacts)
        original_dets = results.get(f"{w}x{h}_original", {}).get('total_dets', 0)
        resized_dets = results.get("640x640_resized", {}).get('total_dets', 0)
        
        if abs(original_dets - resized_dets) > 2:
            print(f"   ⚠️  PADDING ARTIFACT DETECTED: {original_dets} → {resized_dets} detections")
        else:
            print(f"   ✅ Consistent detections across sizes")

if __name__ == '__main__':
    test_clean_inference()