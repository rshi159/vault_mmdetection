#!/usr/bin/env python3
"""
Quick visualization script to generate inference images
"""

import os
import random
import json
import cv2
import numpy as np
import torch
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.structures import DetDataSample
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Register all modules
register_all_modules()

def visualize_predictions_and_gt(img_path, model, gt_annotations, output_path, score_thr=0.3):
    """Run inference and save visualization with ground truth"""
    
    # Load and prepare image
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Could not load image: {img_path}")
        return False
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Run inference
    result = inference_detector(model, img_path)
    
    # Create side-by-side visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Left: Ground Truth
    ax1.imshow(img_rgb)
    ax1.set_title(f"Ground Truth: {os.path.basename(img_path)}", fontsize=14)
    ax1.axis('off')
    
    # Draw ground truth boxes
    gt_count = 0
    if gt_annotations:
        for ann in gt_annotations:
            x, y, w, h = ann['bbox']
            area = w * h
            
            # Color code by size (like COCO evaluation)
            if area < 32*32:
                color = 'yellow'  # Small
                size_label = 'S'
            elif area < 96*96:
                color = 'orange'  # Medium  
                size_label = 'M'
            else:
                color = 'green'   # Large
                size_label = 'L'
            
            rect = patches.Rectangle((x, y), w, h, 
                                   linewidth=2, edgecolor=color, facecolor='none')
            ax1.add_patch(rect)
            
            # Add size label
            ax1.text(x, y-5, size_label, 
                   color=color, fontsize=12, weight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
            gt_count += 1
    
    # Right: Predictions
    ax2.imshow(img_rgb)
    ax2.set_title(f"Predictions: {os.path.basename(img_path)}", fontsize=14)
    ax2.axis('off')
    
    # Draw predictions
    pred_instances = result.pred_instances
    scores = pred_instances.scores.cpu().numpy()
    bboxes = pred_instances.bboxes.cpu().numpy()
    
    pred_count = 0
    valid_indices = scores > score_thr
    if np.any(valid_indices):
        valid_scores = scores[valid_indices]
        valid_bboxes = bboxes[valid_indices]
        
        for bbox, score in zip(valid_bboxes, valid_scores):
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            area = w * h
            
            # Color code by size
            if area < 32*32:
                color = 'yellow'  # Small
            elif area < 96*96:
                color = 'orange'  # Medium  
            else:
                color = 'red'     # Large (red for predictions)
            
            rect = patches.Rectangle((x1, y1), w, h, 
                                   linewidth=2, edgecolor=color, facecolor='none')
            ax2.add_patch(rect)
            
            # Add score text
            ax2.text(x1, y1-5, f'{score:.2f}', 
                   color=color, fontsize=10, weight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            pred_count += 1
    
    # Add stats text
    ax1.text(10, 30, f'GT Objects: {gt_count}', 
           color='white', fontsize=12, weight='bold',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.7))
    
    ax2.text(10, 30, f'Predictions: {pred_count}', 
           color='white', fontsize=12, weight='bold',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.7))
    
    # Add legend
    legend_elements = [
        patches.Patch(color='yellow', label='Small (<32²px)'),
        patches.Patch(color='orange', label='Medium (32²-96²px)'), 
        patches.Patch(color='green', label='Large (>96²px) - GT'),
        patches.Patch(color='red', label='Large (>96²px) - Pred')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path} (GT: {gt_count}, Pred: {pred_count})")
    return True

def main():
    # Configuration
    config_file = 'configs/rtmdet/rtmdet_4ch_rgb_only_ultrafast.py'
    checkpoint_file = './work_dirs/rgb_foundation_ultrafast_300ep/best_coco_bbox_mAP_epoch_70.pth'
    data_root = 'development/augmented_data_production/'
    ann_file = 'train/annotations.json'  # Use training set for much more variety
    data_prefix = 'train/images/'
    output_dir = './visualization_results'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("🔍 Initializing model...")
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    
    print("📂 Loading random validation images...")
    with open(os.path.join(data_root, ann_file), 'r') as f:
        coco_data = json.load(f)
    
    # Create image_id -> annotations mapping
    img_id_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)
    
    # Get diverse sample of images - filter by different prefixes for variety
    all_images = coco_data['images']
    print(f"📊 Total images available: {len(all_images)}")
    
    # Group by image type prefixes to ensure diversity
    image_groups = {}
    for img in all_images:
        filename = img['file_name']
        # Extract prefix pattern (before first underscore or number)
        if '_' in filename:
            prefix = filename.split('_')[0]
        else:
            prefix = filename[:3]  # fallback
        
        if prefix not in image_groups:
            image_groups[prefix] = []
        image_groups[prefix].append(img)
    
    print(f"🎨 Found {len(image_groups)} different image types:")
    for prefix, imgs in image_groups.items():
        print(f"   • {prefix}: {len(imgs)} images")
    
    # Sample diverse images - try to get different types
    random.seed(42)
    diverse_images = []
    
    # First, try to get one from each major group
    sorted_groups = sorted(image_groups.items(), key=lambda x: len(x[1]), reverse=True)
    for prefix, imgs in sorted_groups[:12]:  # Top 12 groups
        if len(diverse_images) < 12:
            diverse_images.append(random.choice(imgs))
    
    # If we need more, add random samples
    if len(diverse_images) < 12:
        remaining = [img for img in all_images if img not in diverse_images]
        diverse_images.extend(random.sample(remaining, min(12 - len(diverse_images), len(remaining))))
    
    print(f"🎯 Selected {len(diverse_images)} diverse images for visualization")
    
    print(f"🎨 Generating visualizations with Ground Truth...")
    success_count = 0
    
    for i, img_info in enumerate(diverse_images):
        img_path = os.path.join(data_root, data_prefix, img_info['file_name'])
        output_path = os.path.join(output_dir, f"diverse_gt_pred_{i+1:02d}_{img_info['file_name']}")
        
        # Get ground truth annotations for this image
        img_id = img_info['id']
        gt_annotations = img_id_to_anns.get(img_id, [])
        
        if visualize_predictions_and_gt(img_path, model, gt_annotations, output_path):
            success_count += 1
    
    print(f"\n✅ Generated {success_count}/{len(diverse_images)} visualizations!")
    print(f"📁 Saved in: {output_dir}")
    print(f"🔍 Check the images to see Ground Truth vs Predictions across diverse image types")
    print(f"📊 Legend: Yellow=Small, Orange=Medium, Green=Large(GT), Red=Large(Pred)")

if __name__ == '__main__':
    main()