#!/usr/bin/env python3
"""
Inference visualization script for the best RGB checkpoint
Visualizes predictions on random validation images to diagnose model performance
"""

import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
import mmcv
import json

# Register all modules
register_all_modules()

def load_random_val_images(data_root, ann_file, num_images=12):
    """Load random validation images with their annotations"""
    with open(os.path.join(data_root, ann_file), 'r') as f:
        coco_data = json.load(f)
    
    # Get random images
    random.seed(42)  # For reproducibility
    random_images = random.sample(coco_data['images'], num_images)
    
    # Create image_id to annotations mapping
    img_id_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)
    
    return random_images, img_id_to_anns

def convert_to_4channel(img):
    """Convert RGB image to 4-channel RGBZ format"""
    if len(img.shape) == 3 and img.shape[2] == 3:
        # Add zero channel for heatmap
        zero_channel = np.zeros((img.shape[0], img.shape[1], 1), dtype=img.dtype)
        img_4ch = np.concatenate([img, zero_channel], axis=2)
        return img_4ch
    return img

def visualize_results(images_info, img_id_to_anns, model, data_root, data_prefix, output_dir):
    """Visualize inference results"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for idx, img_info in enumerate(images_info):
        if idx >= 12:
            break
            
        # Load image
        img_path = os.path.join(data_root, data_prefix, img_info['file_name'])
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to 4-channel for model
        img_4ch = convert_to_4channel(img_rgb)
        
        # Run inference
        result = inference_detector(model, img_4ch)
        
        # Get ground truth annotations
        img_id = img_info['id']
        gt_anns = img_id_to_anns.get(img_id, [])
        
        # Visualize
        ax = axes[idx]
        ax.imshow(img_rgb)
        ax.set_title(f"Image {img_id}: {img_info['file_name']}", fontsize=10)
        ax.axis('off')
        
        # Draw ground truth (green)
        for ann in gt_anns:
            bbox = ann['bbox']  # [x, y, w, h]
            rect_gt = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                                  linewidth=2, edgecolor='green', facecolor='none', 
                                  label='GT' if ann == gt_anns[0] else "")
            ax.add_patch(rect_gt)
        
        # Draw predictions (red)
        pred_instances = result.pred_instances
        if len(pred_instances.bboxes) > 0:
            scores = pred_instances.scores.cpu().numpy()
            bboxes = pred_instances.bboxes.cpu().numpy()
            
            # Filter by score threshold
            score_thr = 0.3
            valid_indices = scores > score_thr
            
            if np.any(valid_indices):
                valid_scores = scores[valid_indices]
                valid_bboxes = bboxes[valid_indices]
                
                for i, (bbox, score) in enumerate(zip(valid_bboxes, valid_scores)):
                    x1, y1, x2, y2 = bbox
                    rect_pred = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                            linewidth=2, edgecolor='red', facecolor='none',
                                            label=f'Pred (score>{score_thr})' if i == 0 else "")
                    ax.add_patch(rect_pred)
                    
                    # Add score text
                    ax.text(x1, y1-5, f'{score:.2f}', color='red', fontsize=8, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        
        # Add legend only for first subplot
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)
        
        # Print stats
        num_gt = len(gt_anns)
        num_pred = len(pred_instances.bboxes)
        num_valid_pred = np.sum(pred_instances.scores.cpu().numpy() > 0.3)
        
        print(f"Image {img_id}: GT={num_gt}, Pred={num_pred}, Valid_Pred(>0.3)={num_valid_pred}")
        if num_gt > 0:
            gt_areas = [ann['area'] for ann in gt_anns]
            avg_gt_area = np.mean(gt_areas)
            print(f"  Average GT area: {avg_gt_area:.1f}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'inference_visualization.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved to: {os.path.join(output_dir, 'inference_visualization.png')}")

def main():
    # Configuration
    config_file = 'configs/rtmdet/rtmdet_4ch_rgb_only_ultrafast.py'
    checkpoint_file = './work_dirs/rgb_foundation_ultrafast_300ep/best_coco_bbox_mAP_epoch_70.pth'
    data_root = 'development/augmented_data_production/'
    ann_file = 'valid/annotations.json'
    data_prefix = 'valid/images/'
    output_dir = './visualization_results'
    
    print("🔍 Initializing model...")
    print(f"Config: {config_file}")
    print(f"Checkpoint: {checkpoint_file}")
    
    # Initialize model
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    
    print("📂 Loading random validation images...")
    images_info, img_id_to_anns = load_random_val_images(data_root, ann_file, num_images=12)
    
    print("🎨 Running inference and visualization...")
    visualize_results(images_info, img_id_to_anns, model, data_root, data_prefix, output_dir)
    
    print("✅ Visualization complete!")

if __name__ == '__main__':
    main()