"""
YOLO keypoint lookup system for accessing original keypoint data during training.
Provides interface between COCO annotations and original YOLO keypoint data.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle


class YOLOKeypointLookup:
    """
    Lookup system to access original YOLO keypoint data during training.
    
    Maps image filenames to their corresponding YOLO keypoint annotations,
    allowing the training pipeline to access rich keypoint information
    even when using COCO format for the main dataset.
    """
    
    def __init__(self, yolo_dataset_path: str, cache_file: Optional[str] = None):
        """
        Initialize the lookup system.
        
        Args:
            yolo_dataset_path: Path to the YOLO dataset directory
            cache_file: Optional path to cache the lookup table
        """
        self.yolo_dataset_path = Path(yolo_dataset_path)
        self.cache_file = cache_file
        self.lookup_table = {}
        
        # Build or load the lookup table
        if cache_file and os.path.exists(cache_file):
            self._load_cache()
        else:
            self._build_lookup_table()
            if cache_file:
                self._save_cache()
    
    def _build_lookup_table(self):
        """Build the lookup table from YOLO dataset."""
        print("🔄 Building YOLO keypoint lookup table...")
        
        for split in ['train', 'valid']:
            labels_dir = self.yolo_dataset_path / split / 'labels'
            images_dir = self.yolo_dataset_path / split / 'images'
            
            if not labels_dir.exists():
                continue
                
            split_count = 0
            for label_file in labels_dir.glob('*.txt'):
                # Get corresponding image filename
                image_name = label_file.stem
                
                # Find the actual image file (could be .jpg or .png)
                image_file = None
                for ext in ['.jpg', '.png']:
                    candidate = images_dir / f"{image_name}{ext}"
                    if candidate.exists():
                        image_file = candidate
                        break
                
                if not image_file:
                    continue
                
                # Parse YOLO annotations
                annotations = self._parse_yolo_file(label_file)
                if annotations:
                    # Use just the image filename as key (not full path)
                    self.lookup_table[image_file.name] = {
                        'split': split,
                        'annotations': annotations,
                        'label_file': str(label_file),
                        'image_file': str(image_file)
                    }
                    split_count += 1
            
            print(f"   {split}: {split_count} images indexed")
        
        print(f"✅ Lookup table built: {len(self.lookup_table)} total images")
    
    def _parse_yolo_file(self, label_file: Path) -> List[Dict]:
        """Parse a YOLO label file and extract keypoint data."""
        annotations = []
        
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 5:  # Need at least class + bbox
                    continue
                
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Parse keypoints (remaining parts should be keypoint data)
                    keypoints = []
                    keypoint_data = parts[5:]
                    
                    # Group keypoints by sets of 3 (x, y, visibility)
                    for i in range(0, len(keypoint_data), 3):
                        if i + 2 < len(keypoint_data):
                            try:
                                kx = float(keypoint_data[i])
                                ky = float(keypoint_data[i + 1])
                                kv = int(keypoint_data[i + 2])
                                keypoints.append([kx, ky, kv])
                            except (ValueError, IndexError):
                                break
                    
                    annotation = {
                        'class_id': class_id,
                        'bbox_normalized': [x_center, y_center, width, height],
                        'keypoints_normalized': keypoints,
                        'num_keypoints': len(keypoints)
                    }
                    
                    annotations.append(annotation)
                
                except (ValueError, IndexError) as e:
                    print(f"⚠️ Error parsing line in {label_file}: {line}")
                    continue
        
        except Exception as e:
            print(f"⚠️ Error reading {label_file}: {e}")
            return []
        
        return annotations
    
    def get_keypoints(self, image_filename: str) -> Optional[List[Dict]]:
        """
        Get keypoint data for a given image filename.
        
        Args:
            image_filename: Name of the image file (e.g., 'image001.jpg')
            
        Returns:
            List of annotation dictionaries with keypoint data, or None if not found
        """
        return self.lookup_table.get(image_filename, {}).get('annotations', None)
    
    def has_keypoints(self, image_filename: str) -> bool:
        """Check if an image has keypoint data available."""
        return image_filename in self.lookup_table
    
    def get_stats(self) -> Dict:
        """Get statistics about the lookup table."""
        total_images = len(self.lookup_table)
        total_annotations = 0
        total_keypoints = 0
        
        splits = {'train': 0, 'valid': 0}
        
        for image_data in self.lookup_table.values():
            splits[image_data['split']] += 1
            annotations = image_data['annotations']
            total_annotations += len(annotations)
            
            for ann in annotations:
                total_keypoints += ann['num_keypoints']
        
        return {
            'total_images': total_images,
            'total_annotations': total_annotations,
            'total_keypoints': total_keypoints,
            'splits': splits,
            'avg_keypoints_per_annotation': total_keypoints / max(total_annotations, 1)
        }
    
    def _save_cache(self):
        """Save lookup table to cache file."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.lookup_table, f)
            print(f"✅ Lookup table cached to {self.cache_file}")
        except Exception as e:
            print(f"⚠️ Failed to save cache: {e}")
    
    def _load_cache(self):
        """Load lookup table from cache file."""
        try:
            with open(self.cache_file, 'rb') as f:
                self.lookup_table = pickle.load(f)
            print(f"✅ Lookup table loaded from cache: {len(self.lookup_table)} images")
        except Exception as e:
            print(f"⚠️ Failed to load cache: {e}")
            self._build_lookup_table()


def create_lookup_singleton(yolo_dataset_path: str) -> YOLOKeypointLookup:
    """Create a singleton instance of the lookup system."""
    if not hasattr(create_lookup_singleton, '_instance'):
        cache_file = os.path.join(yolo_dataset_path, 'yolo_keypoint_lookup.pkl')
        create_lookup_singleton._instance = YOLOKeypointLookup(
            yolo_dataset_path=yolo_dataset_path,
            cache_file=cache_file
        )
    return create_lookup_singleton._instance


if __name__ == "__main__":
    # Test the lookup system
    dataset_path = "/home/robun2/Documents/vault_conveyor_tracking/vault_mmdetection/development/augmented_data_production"
    
    lookup = YOLOKeypointLookup(dataset_path)
    
    print("\n📊 Lookup Statistics:")
    stats = lookup.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test a specific image
    test_images = [
        "_sticker_output_test_6_rgb_0345.jpg",
        "_sticker_output_test_6_rgb_0345.png"
    ]
    
    for test_image in test_images:
        if lookup.has_keypoints(test_image):
            keypoints = lookup.get_keypoints(test_image)
            print(f"\n🔍 Test image: {test_image}")
            print(f"   Found {len(keypoints)} annotations")
            for i, ann in enumerate(keypoints[:2]):  # Show first 2 annotations
                print(f"   Annotation {i}: {ann['num_keypoints']} keypoints")
            break