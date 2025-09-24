"""
Custom 4-Channel Data Preprocessor for MMDetection
Removes channel restrictions and supports arbitrary input channels.
"""

from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmdet.registry import MODELS
from mmengine.model.base_model.data_preprocessor import BaseDataPreprocessor
import torch
import warnings
import numpy as np

@MODELS.register_module()
class DetDataPreprocessor4Ch(DetDataPreprocessor):
    """
    DetDataPreprocessor that supports arbitrary channel counts (not just 1/3).

    Key features:
    - Removes channel count restrictions
    - Supports per-channel normalization for any number of channels
    - Designed for 4-channel input (RGB + PriorH/Heatmap)
    - Drop-in replacement for standard DetDataPreprocessor
    """

    def __init__(self, mean=None, std=None, bgr_to_rgb=False, pad_size_divisor=1, pad_value=0, batch_augments=None, **kwargs):
        """
        Initialize 4-channel preprocessor.

        Args:
            mean: Per-channel mean values (list of length = input channels)
            std: Per-channel std values (list of length = input channels)
            bgr_to_rgb: Whether to convert BGR to RGB
            pad_size_divisor: Pad size divisor
            pad_value: Value used for padding
            batch_augments: Batch augmentations
            **kwargs: Other arguments passed to parent
        """
        # Default 4-channel normalization (BGR + PriorH)
        if mean is None:
            mean = [103.53, 116.28, 123.675, 0.0]  # BGR ImageNet + no-op for PriorH
        if std is None:
            std = [57.375, 57.12, 57.375, 1.0]     # BGR ImageNet + no-op for PriorH

        # Validate channel consistency
        if len(mean) != len(std):
            raise ValueError(f"Mean ({len(mean)}) and std ({len(std)}) must have same length")

        self.num_channels = len(mean)

        # We can't call super().__init__() because DetDataPreprocessor has hardcoded
        # assertions for 1 or 3 channels. Instead, we'll manually set up what we need.
        
        # Initialize base model components from BaseDataPreprocessor
        BaseDataPreprocessor.__init__(self, **kwargs)
        
        # Set up 4-channel specific attributes
        self.bgr_to_rgb = bgr_to_rgb
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value
        self.batch_augments = batch_augments
        
        # Set up normalization for 4 channels
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
        
        # Register as buffers so they move with the model
        self.register_buffer('_mean', self.mean)
        self.register_buffer('_std', self.std)
        
        # Set up attributes that DetDataPreprocessor normally sets
        self._channel_conversion = bgr_to_rgb and self.num_channels == 3
        self._non_blocking = True

        if self.num_channels == 4:
            print(f"✅ DetDataPreprocessor4Ch initialized for 4-channel input")
            print(f"   • Channels: RGB + PriorH")
            print(f"   • Mean: {mean}")
            print(f"   • Std: {std}")
        else:
            print(f"✅ DetDataPreprocessor4Ch initialized for {self.num_channels}-channel input")

    def forward(self, data, training=False):
        """
        Forward pass that properly handles 4-channel input with DetDataPreprocessor functionality.

        Args:
            data: Input data with arbitrary channels
            training: Whether in training mode

        Returns:
            Preprocessed data ready for backbone
        """
        # Get pad shapes before calling parent forward (like DetDataPreprocessor does)
        batch_pad_shape = self._get_pad_shape(data)
        
        # Call the BaseDataPreprocessor forward for basic processing (handles device movement, etc.)
        data = BaseDataPreprocessor.forward(self, data=data, training=training)
        inputs, data_samples = data['inputs'], data['data_samples']

        # Process the inputs - normalize and batch them properly
        if isinstance(inputs, (list, tuple)):
            processed_inputs = []
            max_h, max_w = 0, 0
            
            # First pass: normalize and find max dimensions
            for input_tensor in inputs:
                # Ensure it's a tensor on the correct device
                if not isinstance(input_tensor, torch.Tensor):
                    input_tensor = torch.as_tensor(input_tensor, dtype=torch.float32)
                
                # Move to the same device as our registered buffers (model device)
                input_tensor = input_tensor.to(device=self._mean.device, dtype=torch.float32)
                
                # Normalize the 4-channel input
                input_tensor = self.normalize(input_tensor)
                
                # Track max dimensions
                _, h, w = input_tensor.shape
                max_h = max(max_h, h)
                max_w = max(max_w, w)
                
                processed_inputs.append(input_tensor)
            
            # Second pass: pad all tensors to divisible dimensions (not just max dims)
            padded_inputs = []
            
            # Calculate the target dimensions that are divisible by pad_size_divisor
            target_h = int(np.ceil(max_h / self.pad_size_divisor)) * self.pad_size_divisor
            target_w = int(np.ceil(max_w / self.pad_size_divisor)) * self.pad_size_divisor
            
            for input_tensor in processed_inputs:
                c, h, w = input_tensor.shape
                
                # Calculate padding needed to reach target dimensions
                pad_h = target_h - h
                pad_w = target_w - w
                
                if pad_h > 0 or pad_w > 0:
                    # Pad with zeros (bottom, right padding)
                    padding = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
                    input_tensor = torch.nn.functional.pad(input_tensor, padding, value=self.pad_value)
                
                padded_inputs.append(input_tensor)
            
            # Stack into batch tensor: (N, C, H, W)
            inputs = torch.stack(padded_inputs, dim=0)
        else:
            # Already a batched tensor, just normalize and ensure proper device/dtype
            inputs = inputs.to(device=self._mean.device, dtype=torch.float32)
            inputs = self.normalize(inputs)

        # Update the data dict
        data['inputs'] = inputs

        if data_samples is not None:
            # Set batch input shape and pad shape (like DetDataPreprocessor does)
            batch_input_shape = tuple(inputs.size()[-2:])
            for data_sample, pad_shape in zip(data_samples, batch_pad_shape):
                data_sample.set_metainfo({
                    'batch_input_shape': batch_input_shape,
                    'pad_shape': pad_shape
                })

        return data

    def _get_pad_shape(self, data: dict):
        """Get the pad_shape of each image based on data and pad_size_divisor."""
        _batch_inputs = data['inputs']
        if isinstance(_batch_inputs, list):
            # Find the maximum dimensions across all images in the batch
            max_h, max_w = 0, 0
            for ori_input in _batch_inputs:
                if isinstance(ori_input, torch.Tensor):
                    max_h = max(max_h, ori_input.shape[-2])
                    max_w = max(max_w, ori_input.shape[-1])
                else:
                    max_h = max(max_h, ori_input.shape[0])
                    max_w = max(max_w, ori_input.shape[1])
            
            # Calculate target dimensions for the whole batch (divisible by pad_size_divisor)
            target_h = int(np.ceil(max_h / self.pad_size_divisor)) * self.pad_size_divisor
            target_w = int(np.ceil(max_w / self.pad_size_divisor)) * self.pad_size_divisor
            
            # All images in the batch will be padded to the same target dimensions
            batch_pad_shape = [(target_h, target_w)] * len(_batch_inputs)
        else:
            # Handle tensor batch case
            pad_h = int(np.ceil(_batch_inputs.shape[-2] / self.pad_size_divisor)) * self.pad_size_divisor
            pad_w = int(np.ceil(_batch_inputs.shape[-1] / self.pad_size_divisor)) * self.pad_size_divisor
            batch_pad_shape = [(pad_h, pad_w)] * _batch_inputs.shape[0]
        
        return batch_pad_shape

    def normalize(self, img):
        """Normalize image with 4-channel support."""
        # img should be (C, H, W) format
        if img.shape[0] != self.num_channels:
            warnings.warn(f"Expected {self.num_channels} channels, got {img.shape[0]}")
            
        # Ensure tensors are on the same device - use float32 consistently for normalization
        # AMP will handle the final dtype conversion when it enters the backbone
        mean = self._mean.to(device=img.device, dtype=torch.float32)
        std = self._std.to(device=img.device, dtype=torch.float32)
        
        # Ensure img is float32 for normalization
        if img.dtype != torch.float32:
            img = img.to(dtype=torch.float32)
        
        # Apply normalization
        img = (img - mean) / std
        return img
        
    def cast_data(self, data):
        """Cast data to appropriate types."""
        # Basic implementation - just ensure inputs are tensors
        if hasattr(data, 'inputs'):
            data.inputs = [torch.as_tensor(inp) if not isinstance(inp, torch.Tensor) else inp 
                          for inp in data.inputs]
        return data
        
    def pad_data(self, data):
        """Handle padding for size divisibility."""
        # Basic implementation - for 4-channel we mainly need this to work
        # The actual padding logic can be simplified since we removed complex augmentations
        return data

    def preprocess_img(self, _img):
        """Preprocess single image - handles arbitrary channels."""
        # Parent method works fine for arbitrary channels
        return super().preprocess_img(_img)
