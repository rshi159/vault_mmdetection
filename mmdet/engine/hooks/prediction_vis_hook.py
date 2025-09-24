"""Hook for periodic visualization of predictions on fixed validation images."""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional

from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmdet.registry import HOOKS
from mmdet.apis import inference_detector
from mmdet.structures import DetDataSample
from mmcv import imread
import mmcv


@HOOKS.register_module()
class PredictionVisualizationHook(Hook):
    """Hook to visualize predictions on fixed validation images during training.
    
    This hook provides qualitative assessment of model learning by saving
    prediction visualizations on a fixed set of validation images.
    
    Args:
        image_paths (List[str]): List of paths to validation images to visualize.
        vis_interval (int): Interval (in epochs) to save visualizations. Default: 5.
        score_thr (float): Score threshold for displaying predictions. Default: 0.1.
        output_dir (str): Directory to save visualization images. Default: 'vis_outputs'.
        max_images (int): Maximum number of images to visualize. Default: 10.
        show_gt (bool): Whether to show ground truth annotations. Default: True.
    """
    
    def __init__(self,
                 image_paths: Optional[List[str]] = None,
                 vis_interval: int = 5,
                 score_thr: float = 0.1,
                 output_dir: str = 'vis_outputs',
                 max_images: int = 10,
                 show_gt: bool = True):
        self.image_paths = image_paths or []
        self.vis_interval = vis_interval
        self.score_thr = score_thr
        self.output_dir = output_dir
        self.max_images = max_images
        self.show_gt = show_gt
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _auto_select_images(self, runner) -> List[str]:
        """Auto-select validation images if none provided."""
        try:
            # Get validation dataset
            val_dataloader = runner.val_dataloader
            if val_dataloader is None:
                return []
                
            dataset = val_dataloader.dataset
            
            # Select evenly spaced images from validation set
            total_images = len(dataset)
            if total_images == 0:
                return []
                
            step = max(1, total_images // self.max_images)
            indices = list(range(0, min(total_images, self.max_images * step), step))
            
            selected_paths = []
            for idx in indices:
                try:
                    data_info = dataset.get_data_info(idx)
                    img_path = data_info.get('img_path', '')
                    if img_path and os.path.exists(img_path):
                        selected_paths.append(img_path)
                except:
                    continue
                    
            return selected_paths[:self.max_images]
            
        except Exception as e:
            runner.logger.warning(f"Could not auto-select validation images: {e}")
            return []
    
    def _get_ground_truth(self, runner, img_path: str) -> Optional[DetDataSample]:
        """Get ground truth annotations for an image."""
        if not self.show_gt:
            return None
            
        try:
            val_dataset = runner.val_dataloader.dataset
            
            # Find the image in the dataset
            for i in range(len(val_dataset)):
                data_info = val_dataset.get_data_info(i)
                if data_info.get('img_path', '') == img_path:
                    # Load the full data sample with annotations
                    data_sample = val_dataset[i]
                    return data_sample.get('data_samples', None)
                    
        except Exception as e:
            runner.logger.warning(f"Could not load ground truth for {img_path}: {e}")
            
        return None
    
    def _draw_bboxes(self, img: np.ndarray, bboxes: np.ndarray, labels: np.ndarray, 
                     scores: np.ndarray, color: tuple = (0, 255, 0), thickness: int = 2):
        """Draw bounding boxes on image."""
        img_with_boxes = img.copy()
        
        for bbox, label, score in zip(bboxes, labels, scores):
            if score < self.score_thr:
                continue
                
            x1, y1, x2, y2 = bbox.astype(int)
            
            # Draw rectangle
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label and score
            label_text = f'package: {score:.2f}'
            (text_width, text_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            cv2.rectangle(img_with_boxes, (x1, y1 - text_height - 5), 
                         (x1 + text_width, y1), color, -1)
            cv2.putText(img_with_boxes, label_text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                       
        return img_with_boxes
    
    def _visualize_image(self, runner, img_path: str, epoch: int):
        """Visualize predictions for a single image."""
        try:
            # Load image
            if not os.path.exists(img_path):
                runner.logger.warning(f"Image not found: {img_path}")
                return
                
            img = imread(img_path)
            if img is None:
                runner.logger.warning(f"Could not load image: {img_path}")
                return
            
            # Run inference
            model = runner.model
            if hasattr(model, 'module'):
                model = model.module
                
            with torch.no_grad():
                result = inference_detector(model, img)
            
            # Extract predictions
            pred_instances = result.pred_instances
            pred_bboxes = pred_instances.bboxes.cpu().numpy()
            pred_scores = pred_instances.scores.cpu().numpy()
            pred_labels = pred_instances.labels.cpu().numpy()
            
            # Draw predictions (green boxes)
            img_with_pred = self._draw_bboxes(
                img, pred_bboxes, pred_labels, pred_scores,
                color=(0, 255, 0), thickness=2
            )
            
            # Draw ground truth if available (red boxes)
            gt_sample = self._get_ground_truth(runner, img_path)
            if gt_sample is not None and hasattr(gt_sample, 'gt_instances'):
                gt_bboxes = gt_sample.gt_instances.bboxes.cpu().numpy()
                gt_labels = gt_sample.gt_instances.labels.cpu().numpy()
                gt_scores = np.ones(len(gt_bboxes))  # GT has score 1.0
                
                img_with_pred = self._draw_bboxes(
                    img_with_pred, gt_bboxes, gt_labels, gt_scores,
                    color=(0, 0, 255), thickness=2
                )
            
            # Save visualization
            img_name = Path(img_path).stem
            output_path = os.path.join(
                self.output_dir, f'epoch_{epoch:03d}_{img_name}_pred.jpg'
            )
            cv2.imwrite(output_path, img_with_pred)
            
            # Count predictions for logging
            high_conf_preds = np.sum(pred_scores > 0.5)
            all_preds = len(pred_scores)
            
            runner.logger.info(
                f"Epoch {epoch} - {img_name}: {all_preds} predictions "
                f"({high_conf_preds} > 0.5 conf) -> {output_path}"
            )
            
        except Exception as e:
            runner.logger.error(f"Error visualizing {img_path}: {e}")
    
    def after_val_epoch(self, runner, metrics=None):
        """Create visualizations after validation epochs."""
        current_epoch = runner.epoch
        
        # Only visualize on specified intervals
        if current_epoch % self.vis_interval != 0:
            return
            
        # Get images to visualize
        if not self.image_paths:
            self.image_paths = self._auto_select_images(runner)
            
        if not self.image_paths:
            runner.logger.warning("No validation images found for visualization")
            return
            
        runner.logger.info(
            f"Creating prediction visualizations for epoch {current_epoch}..."
        )
        
        # Visualize each image
        for img_path in self.image_paths:
            self._visualize_image(runner, img_path, current_epoch)
            
        runner.logger.info(
            f"Saved {len(self.image_paths)} visualizations to {self.output_dir}"
        )