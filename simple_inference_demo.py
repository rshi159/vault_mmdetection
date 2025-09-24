#!/usr/bin/env python3
"""
Simple inference script using MMDet's image demo functionality
Runs inference on random validation images to visualize model performance
"""

import os
import random
import json
import shutil
from mmdet.utils import register_all_modules

# Register all modules
register_all_modules()

def get_random_val_images(data_root, ann_file, num_images=12):
    """Get paths to random validation images"""
    with open(os.path.join(data_root, ann_file), 'r') as f:
        coco_data = json.load(f)
    
    # Get random images
    random.seed(42)  # For reproducibility
    random_images = random.sample(coco_data['images'], num_images)
    
    return [img['file_name'] for img in random_images]

def main():
    # Configuration
    config_file = 'configs/rtmdet/rtmdet_4ch_rgb_only_ultrafast.py'
    checkpoint_file = './work_dirs/rgb_foundation_ultrafast_300ep/best_coco_bbox_mAP_epoch_70.pth'
    data_root = 'development/augmented_data_production/'
    ann_file = 'valid/annotations.json'
    data_prefix = 'valid/images/'
    output_dir = './visualization_results'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("📂 Getting random validation images...")
    random_images = get_random_val_images(data_root, ann_file, num_images=12)
    
    print(f"🎨 Running inference on {len(random_images)} images...")
    print(f"Config: {config_file}")
    print(f"Checkpoint: {checkpoint_file}")
    print(f"Output: {output_dir}")
    
    # Run inference on each image using MMDet's image demo
    for i, img_name in enumerate(random_images):
        img_path = os.path.join(data_root, data_prefix, img_name)
        output_path = os.path.join(output_dir, f"result_{i:02d}_{img_name}")
        
        if os.path.exists(img_path):
            print(f"Processing {i+1}/{len(random_images)}: {img_name}")
            
            # Use MMDet's image demo script
            cmd = f"""
            python3 demo/image_demo.py \\
                "{img_path}" \\
                "{config_file}" \\
                "{checkpoint_file}" \\
                --out-file "{output_path}" \\
                --device cuda:0 \\
                --score-thr 0.3
            """
            
            # Execute the command
            exit_code = os.system(cmd)
            if exit_code != 0:
                print(f"❌ Failed to process {img_name}")
            else:
                print(f"✅ Saved: {output_path}")
        else:
            print(f"❌ Image not found: {img_path}")
    
    print(f"✅ Visualization complete! Results saved in: {output_dir}")
    print(f"📊 You can check the inference results to analyze:")
    print("   • Detection quality on different object sizes")
    print("   • False positives and missed detections") 
    print("   • Model confidence scores")

if __name__ == '__main__':
    main()