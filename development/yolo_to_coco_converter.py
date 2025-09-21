#!/usr/bin/env python3
"""
YOLO to COCO Format Converter for MMDetection

This script converts YOLO format annotations (.txt files) to COCO format (.json files)
required by MMDetection framework for object detection training.

Usage:
    python yolo_to_coco_converter.py --dataset_path development/augmented_data_production
    
Author: RTMDet Edge Training Pipeline
Date: September 2025
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from PIL import Image
import sys

def convert_yolo_to_coco(
    dataset_path: str,
    split: str = 'train',
    class_names: List[str] = ['package']
) -> bool:
    """Convert YOLO format annotations to COCO format.
    
    Args:
        dataset_path: Path to the dataset directory.
        split: Dataset split ('train' or 'valid').
        class_names: List of class names.
        
    Returns:
        bool: True if conversion successful.
    """
    dataset_root = Path(dataset_path)
    images_dir = dataset_root / split / 'images'
    labels_dir = dataset_root / split / 'labels'
    output_file = dataset_root / split / 'annotations.json'
    
    if not images_dir.exists() or not labels_dir.exists():
        print(f"❌ Required directories not found for {split} split")
        print(f"   Expected: {images_dir} and {labels_dir}")
        return False
    
    print(f"📂 Processing {split} split:")
    print(f"   Images: {images_dir}")
    print(f"   Labels: {labels_dir}")
    print(f"   Output: {output_file}")
    
    # Initialize COCO format structure
    coco_data = {
        "info": {
            "description": f"Package Detection Dataset - {split.title()} Split",
            "url": "",
            "version": "1.0",
            "year": 2025,
            "contributor": "RTMDet Edge Training Pipeline",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [
            {
                "id": 1,
                "name": "Custom License",
                "url": ""
            }
        ],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Add categories (COCO format requires category IDs to start from 1)
    for idx, class_name in enumerate(class_names):
        coco_data["categories"].append({
            "id": idx + 1,  # COCO category IDs start from 1
            "name": class_name,
            "supercategory": "object"
        })
    
    # Process images and annotations
    image_id = 0
    annotation_id = 0
    processed_images = 0
    total_annotations = 0
    
    print(f"🔄 Converting {split} annotations...")
    
    # Get all image files
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    total_images = len(image_files)
    
    if total_images == 0:
        print(f"❌ No image files found in {images_dir}")
        return False
    
    for idx, img_file in enumerate(image_files):
        # Progress indicator
        if idx % 1000 == 0 or idx == total_images - 1:
            print(f"   Processing image {idx + 1}/{total_images}...")
        
        # Corresponding label file
        label_file = labels_dir / f"{img_file.stem}.txt"
        
        # Get image dimensions
        try:
            with Image.open(img_file) as img:
                width, height = img.size
        except Exception as e:
            print(f"⚠️ Error reading image {img_file}: {e}")
            continue
        
        # Add image info
        image_info = {
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": img_file.name,  # Just the filename, no path prefix
            "license": 1,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": ""
        }
        coco_data["images"].append(image_info)
        processed_images += 1
        
        # Process annotations if label file exists
        if label_file.exists():
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        bbox_width = float(parts[3])
                        bbox_height = float(parts[4])
                    except ValueError as e:
                        print(f"⚠️ Error parsing annotation in {label_file}: {line}")
                        continue
                    
                    # Convert from YOLO format (normalized) to COCO format (absolute)
                    x_min = (x_center - bbox_width / 2) * width
                    y_min = (y_center - bbox_height / 2) * height
                    bbox_width_abs = bbox_width * width
                    bbox_height_abs = bbox_height * height
                    
                    # Ensure bbox is within image bounds
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    bbox_width_abs = min(bbox_width_abs, width - x_min)
                    bbox_height_abs = min(bbox_height_abs, height - y_min)
                    
                    # Skip invalid bboxes
                    if bbox_width_abs <= 0 or bbox_height_abs <= 0:
                        continue
                    
                    # COCO format: [x_min, y_min, width, height]
                    bbox = [x_min, y_min, bbox_width_abs, bbox_height_abs]
                    area = bbox_width_abs * bbox_height_abs
                    
                    annotation_info = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id + 1,  # Convert from 0-based YOLO to 1-based COCO
                        "segmentation": [],
                        "area": area,
                        "bbox": bbox,
                        "iscrowd": 0
                    }
                    coco_data["annotations"].append(annotation_info)
                    annotation_id += 1
                    total_annotations += 1
                    
            except Exception as e:
                print(f"⚠️ Error processing label {label_file}: {e}")
        
        image_id += 1
    
    # Save COCO format annotations
    try:
        # Create output directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"✅ Conversion complete for {split} split:")
        print(f"   📊 Images processed: {processed_images}")
        print(f"   📊 Annotations created: {total_annotations}")
        print(f"   📁 Output file: {output_file}")
        print(f"   📦 Categories: {[cat['name'] for cat in coco_data['categories']]}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving annotations: {e}")
        return False

def validate_dataset_structure(dataset_path: str) -> Dict[str, Any]:
    """Validate dataset structure before conversion.
    
    Args:
        dataset_path: Path to the dataset directory.
        
    Returns:
        Dict containing validation results.
    """
    dataset_root = Path(dataset_path)
    
    validation_results = {
        'dataset_exists': dataset_root.exists(),
        'train_valid': False,
        'valid_valid': False,
        'train_images': 0,
        'train_labels': 0,
        'valid_images': 0,
        'valid_labels': 0,
    }
    
    if not dataset_root.exists():
        return validation_results
    
    # Check train split
    train_images_dir = dataset_root / 'train' / 'images'
    train_labels_dir = dataset_root / 'train' / 'labels'
    
    if train_images_dir.exists() and train_labels_dir.exists():
        validation_results['train_images'] = len(list(train_images_dir.glob('*.jpg')) + list(train_images_dir.glob('*.png')))
        validation_results['train_labels'] = len(list(train_labels_dir.glob('*.txt')))
        validation_results['train_valid'] = validation_results['train_images'] > 0 and validation_results['train_labels'] > 0
    
    # Check valid split
    valid_images_dir = dataset_root / 'valid' / 'images'
    valid_labels_dir = dataset_root / 'valid' / 'labels'
    
    if valid_images_dir.exists() and valid_labels_dir.exists():
        validation_results['valid_images'] = len(list(valid_images_dir.glob('*.jpg')) + list(valid_images_dir.glob('*.png')))
        validation_results['valid_labels'] = len(list(valid_labels_dir.glob('*.txt')))
        validation_results['valid_valid'] = validation_results['valid_images'] > 0 and validation_results['valid_labels'] > 0
    
    return validation_results

def main():
    """Main function to run the YOLO to COCO converter."""
    parser = argparse.ArgumentParser(
        description="Convert YOLO format annotations to COCO format for MMDetection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python yolo_to_coco_converter.py --dataset_path development/augmented_data_production
    python yolo_to_coco_converter.py --dataset_path /path/to/dataset --class_names package box item
        """
    )
    
    parser.add_argument(
        '--dataset_path',
        type=str,
        required=True,
        help='Path to the dataset directory containing train/valid splits'
    )
    
    parser.add_argument(
        '--class_names',
        nargs='+',
        default=['package'],
        help='List of class names (default: package)'
    )
    
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'valid'],
        help='Dataset splits to convert (default: train valid)'
    )
    
    args = parser.parse_args()
    
    print("🔄 YOLO to COCO Format Converter")
    print("=" * 40)
    print(f"📂 Dataset path: {args.dataset_path}")
    print(f"📋 Class names: {args.class_names}")
    print(f"📊 Splits: {args.splits}")
    print()
    
    # Validate dataset structure
    print("🔍 Validating dataset structure...")
    validation = validate_dataset_structure(args.dataset_path)
    
    if not validation['dataset_exists']:
        print(f"❌ Dataset directory does not exist: {args.dataset_path}")
        sys.exit(1)
    
    print(f"📊 Dataset validation results:")
    for split in args.splits:
        split_key = f"{split}_valid"
        images_key = f"{split}_images"
        labels_key = f"{split}_labels"
        
        if split_key in validation:
            status = "✅" if validation[split_key] else "❌"
            print(f"   {status} {split}: {validation.get(images_key, 0)} images, {validation.get(labels_key, 0)} labels")
        else:
            print(f"   ❌ {split}: Not found")
    
    print()
    
    # Convert each split
    success_count = 0
    for split in args.splits:
        split_key = f"{split}_valid"
        if validation.get(split_key, False):
            print(f"🔄 Converting {split} split...")
            if convert_yolo_to_coco(args.dataset_path, split, args.class_names):
                success_count += 1
            print()
        else:
            print(f"⏭️ Skipping {split} split (not valid)")
    
    # Summary
    print("📋 Conversion Summary:")
    print(f"   ✅ Successfully converted: {success_count}/{len(args.splits)} splits")
    
    if success_count == len(args.splits):
        print("\n🎉 All conversions completed successfully!")
        print("📝 Your dataset is now ready for MMDetection training!")
        return 0
    else:
        print(f"\n⚠️ {len(args.splits) - success_count} conversions failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())