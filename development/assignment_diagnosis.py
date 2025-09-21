#!/usr/bin/env python3
"""
Deep dive into assignment process to identify why bbox_loss = 0.0000
This script tests the assignment process with actual model forward pass
"""

import sys
import os
import torch
import json
import numpy as np

# Add project root to path
project_root = '/home/robun2/Documents/vault_conveyor_tracking/vault_mmdetection'
sys.path.insert(0, project_root)
os.chdir(project_root)

from mmdet.apis import init_detector
from mmdet.datasets import build_dataset
from mmdet.registry import MODELS
from mmengine.config import Config
from mmdet.models.task_modules.assigners import DynamicSoftLabelAssigner

print("=== ASSIGNMENT PROCESS DEEP DIVE ===")

# Load the debug config
debug_config_path = 'work_dirs/rtmdet_edge_training/rtmdet_debug_config.py'
cfg = Config.fromfile(debug_config_path)

# Build model
print(f"Building model with config...")
model = MODELS.build(cfg.model)
model.eval()

# Create a simple dataset for testing
print(f"Creating test dataset...")
dataset_cfg = cfg.train_dataloader.dataset
test_dataset = build_dataset(dataset_cfg)

print(f"Dataset size: {len(test_dataset)}")

# Get a few samples
print(f"Loading test samples...")
sample_indices = [0, 1, 2]  # Test first 3 samples
for i, idx in enumerate(sample_indices):
    print(f"\n{'='*50}")
    print(f"Sample {i+1} (index {idx})")
    print(f"{'='*50}")
    
    # Get raw sample
    sample = test_dataset[idx]
    
    print(f"Image shape: {sample['inputs'].shape}")
    print(f"GT instances:")
    print(f"  - Bboxes shape: {sample['data_samples'].gt_instances.bboxes.shape}")
    print(f"  - Labels shape: {sample['data_samples'].gt_instances.labels.shape}")
    print(f"  - Bboxes: {sample['data_samples'].gt_instances.bboxes}")
    print(f"  - Labels: {sample['data_samples'].gt_instances.labels}")
    
    # Test model forward (without training step)
    with torch.no_grad():
        # Prepare input for model
        inputs = sample['inputs'].unsqueeze(0)  # Add batch dimension
        data_samples = [sample['data_samples']]
        
        print(f"Model input shape: {inputs.shape}")
        
        # Get model predictions at different scales
        try:
            # Run backbone and neck
            x = model.backbone(inputs)
            neck_outputs = model.neck(x)
            
            print(f"Backbone outputs: {[feat.shape for feat in x]}")
            print(f"Neck outputs: {[feat.shape for feat in neck_outputs]}")
            
            # Get model head predictions
            cls_scores, bbox_preds = model.bbox_head(neck_outputs)
            
            print(f"Classification predictions: {[score.shape for score in cls_scores]}")
            print(f"Bbox predictions: {[pred.shape for pred in bbox_preds]}")
            
            # Test assignment manually
            assigner = DynamicSoftLabelAssigner(topk=3)
            
            # Get GT info
            gt_instances = data_samples[0].gt_instances
            gt_bboxes = gt_instances.bboxes
            gt_labels = gt_instances.labels
            
            print(f"\nGT for assignment:")
            print(f"  - GT bboxes: {gt_bboxes}")
            print(f"  - GT labels: {gt_labels}")
            
            # Generate anchor points for assignment
            # Get feature map sizes
            featmap_sizes = [feat.shape[-2:] for feat in neck_outputs]
            mlvl_points = model.bbox_head.anchor_generator.grid_priors(
                featmap_sizes, device=inputs.device, with_stride=True)
            
            print(f"\nFeature map sizes: {featmap_sizes}")
            print(f"Multi-level points: {[points.shape for points in mlvl_points]}")
            
            # Flatten predictions for assignment
            flatten_cls_scores = [cls_score.flatten(0, 1) for cls_score in cls_scores]
            flatten_bbox_preds = [bbox_pred.flatten(0, 1) for bbox_pred in bbox_preds]
            flatten_points = torch.cat(mlvl_points, dim=0)
            
            print(f"\nFlattened cls scores: {[score.shape for score in flatten_cls_scores]}")
            print(f"Flattened bbox preds: {[pred.shape for pred in flatten_bbox_preds]}")
            print(f"Flattened points shape: {flatten_points.shape}")
            
            # Concatenate across batch (single sample)
            cls_scores_concat = torch.cat([score[0:1] for score in flatten_cls_scores], dim=1)
            bbox_preds_concat = torch.cat([pred[0:1] for pred in flatten_bbox_preds], dim=1)
            
            print(f"\nConcatenated cls scores shape: {cls_scores_concat.shape}")
            print(f"Concatenated bbox preds shape: {bbox_preds_concat.shape}")
            
            print(f"\nSpatial Analysis:")
            print(f"Points range: X[{flatten_points[:, 0].min():.1f}, {flatten_points[:, 0].max():.1f}] Y[{flatten_points[:, 1].min():.1f}, {flatten_points[:, 1].max():.1f}]")
            print(f"GT bbox range: X[{gt_bboxes[:, 0].min():.1f}, {gt_bboxes[:, 0].max():.1f}] Y[{gt_bboxes[:, 1].min():.1f}, {gt_bboxes[:, 1].max():.1f}]")
            
            # Check if any points are inside GT bboxes
            points_xy = flatten_points[:, :2]  # Remove stride info
            total_inside = 0
            for j, gt_bbox in enumerate(gt_bboxes):
                x1, y1, x2, y2 = gt_bbox
                inside = (points_xy[:, 0] >= x1) & (points_xy[:, 0] <= x2) & \
                        (points_xy[:, 1] >= y1) & (points_xy[:, 1] <= y2)
                inside_count = inside.sum().item()
                total_inside += inside_count
                print(f"Points inside GT bbox {j+1} {gt_bbox}: {inside_count}")
            
            if total_inside == 0:
                print("\n⚠️  WARNING: No anchor points are inside any GT bboxes!")
                print("This explains why assignment fails - scale mismatch issue")
                
                # Check overlaps
                print("\nChecking bbox overlaps with anchor grid...")
                for level, (points, featmap_size) in enumerate(zip(mlvl_points, featmap_sizes)):
                    level_points = points[:, :2]
                    stride = points[0, 2].item()
                    
                    print(f"\nLevel {level} (stride {stride}):")
                    print(f"  Feature map size: {featmap_size}")
                    print(f"  Points range: X[{level_points[:, 0].min():.1f}, {level_points[:, 0].max():.1f}] Y[{level_points[:, 1].min():.1f}, {level_points[:, 1].max():.1f}]")
                    
                    for j, gt_bbox in enumerate(gt_bboxes):
                        x1, y1, x2, y2 = gt_bbox
                        inside = (level_points[:, 0] >= x1) & (level_points[:, 0] <= x2) & \
                                (level_points[:, 1] >= y1) & (level_points[:, 1] <= y2)
                        inside_count = inside.sum().item()
                        
                        # Also check proximity
                        distances = torch.sqrt((level_points[:, 0] - (x1 + x2)/2)**2 + 
                                             (level_points[:, 1] - (y1 + y2)/2)**2)
                        min_distance = distances.min().item()
                        
                        print(f"    GT bbox {j+1}: {inside_count} points inside, min distance: {min_distance:.1f}")
            else:
                print(f"\n✅ Total points inside GT bboxes: {total_inside}")
                
                # Try actual assignment
                print("\nTesting assignment process...")
                try:
                    # Prepare data for assignment
                    assign_result = assigner.assign(
                        cls_scores_concat[0],  # Remove batch dim
                        bbox_preds_concat[0],  # Remove batch dim
                        flatten_points,
                        gt_bboxes,
                        gt_labels
                    )
                    
                    print(f"Assignment result:")
                    print(f"  - Num GT: {assign_result.num_gts}")
                    print(f"  - GT indices shape: {assign_result.gt_inds.shape}")
                    print(f"  - Max label: {assign_result.max_overlaps.max()}")
                    print(f"  - Positive samples: {(assign_result.gt_inds > 0).sum()}")
                    
                except Exception as e:
                    print(f"Assignment failed: {e}")
                    import traceback
                    traceback.print_exc()
            
        except Exception as e:
            print(f"Error during forward pass: {e}")
            import traceback
            traceback.print_exc()
    
    if i >= 2:  # Limit to 3 samples to avoid too much output
        break

print(f"\n{'='*50}")
print("DIAGNOSIS COMPLETE")
print(f"{'='*50}")