#!/usr/bin/env python3
"""
Example Inference Script for 4-Channel RTMDet
Professional example showing inference on trained models.
"""

import argparse
import cv2
import numpy as np
from pathlib import Path

def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description='Run inference with 4-Channel RTMDet'
    )
    parser.add_argument('image', help='Input image path')
    parser.add_argument('config', help='Model config file')
    parser.add_argument('checkpoint', help='Model checkpoint file')
    parser.add_argument(
        '--output', 
        default='output.jpg',
        help='Output image path'
    )
    parser.add_argument(
        '--score-threshold',
        type=float,
        default=0.3,
        help='Detection score threshold'
    )
    
    args = parser.parse_args()
    
    print("🎯 4-Channel RTMDet Inference")
    print("=" * 50)
    print(f"Image: {args.image}")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output}")
    
    # Import MMDetection components
    try:
        from mmdet.apis import init_detector, inference_detector
        from mmdet.utils import register_all_modules
        
        # Register custom modules
        register_all_modules()
        
    except ImportError as e:
        print(f"❌ MMDetection import failed: {e}")
        print("Please ensure MMDetection is properly installed.")
        return
    
    # Initialize model
    print("\n📋 Loading model...")
    try:
        model = init_detector(args.config, args.checkpoint, device='cuda:0')
        print("   ✅ Model loaded successfully")
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        return
    
    # Load and validate image
    print(f"\n🖼️ Loading image...")
    try:
        image = cv2.imread(args.image)
        if image is None:
            raise ValueError(f"Could not load image: {args.image}")
        print(f"   ✅ Image loaded: {image.shape}")
    except Exception as e:
        print(f"   ❌ Image loading failed: {e}")
        return
    
    # Run inference
    print(f"\n🧠 Running inference...")
    try:
        results = inference_detector(model, image)
        print("   ✅ Inference completed")
        
        # Process results
        detections = process_results(results, args.score_threshold)
        print(f"   📊 Found {len(detections)} detections above threshold {args.score_threshold}")
        
    except Exception as e:
        print(f"   ❌ Inference failed: {e}")
        return
    
    # Visualize results
    print(f"\n🎨 Visualizing results...")
    try:
        output_image = visualize_detections(image, detections)
        cv2.imwrite(args.output, output_image)
        print(f"   ✅ Results saved to: {args.output}")
    except Exception as e:
        print(f"   ❌ Visualization failed: {e}")
        return
    
    print(f"\n✅ Inference completed successfully!")


def process_results(results, score_threshold):
    """Process detection results and filter by score."""
    detections = []
    
    # Extract predictions (format depends on MMDetection version)
    if hasattr(results, 'pred_instances'):
        # MMDetection 3.x format
        pred_instances = results.pred_instances
        boxes = pred_instances.bboxes.cpu().numpy()
        scores = pred_instances.scores.cpu().numpy()
        labels = pred_instances.labels.cpu().numpy()
    else:
        # Legacy format
        boxes = results[0]  # Assuming single class
        scores = boxes[:, 4]
        boxes = boxes[:, :4]
        labels = np.zeros(len(boxes), dtype=int)
    
    # Filter by score threshold
    valid_indices = scores >= score_threshold
    
    for i in range(len(boxes)):
        if valid_indices[i]:
            detections.append({
                'bbox': boxes[i],
                'score': scores[i], 
                'label': labels[i]
            })
    
    return detections


def visualize_detections(image, detections):
    """Visualize detection results on image."""
    output_image = image.copy()
    
    for detection in detections:
        bbox = detection['bbox']
        score = detection['score']
        
        # Draw bounding box
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw confidence score
        label_text = f"Package: {score:.2f}"
        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(output_image, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(output_image, label_text, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return output_image


if __name__ == '__main__':
    main()