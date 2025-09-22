
"""
Usage example for 4-channel RTMDet implementation.
"""

import torch
from preprocessor_4ch import DetDataPreprocessor4Ch
from backbone_4ch import CSPNeXt4Ch

def create_4channel_model():
    """Create a complete 4-channel model."""

    # 1. Data preprocessor
    preprocessor = DetDataPreprocessor4Ch(
        mean=[103.53, 116.28, 123.675, 0.0],  # BGR + Heatmap
        std=[57.375, 57.12, 57.375, 1.0],     # BGR + Heatmap
        bgr_to_rgb=False  # Keep as BGR for compatibility
    )

    # 2. Backbone
    backbone = CSPNeXt4Ch(
        arch='P5',
        widen_factor=0.375,    # For RTMDet-tiny
        deepen_factor=0.167,   # For RTMDet-tiny
        out_indices=(2, 3, 4)  # Output P3, P4, P5
    )

    return preprocessor, backbone

def process_4channel_batch(preprocessor, backbone, images, heatmaps):
    """
    Process a batch of RGB images with heatmaps.

    Args:
        preprocessor: DetDataPreprocessor4Ch instance
        backbone: CSPNeXt4Ch instance
        images: Tensor of shape (B, 3, H, W) - RGB images
        heatmaps: Tensor of shape (B, 1, H, W) - Heatmap channel

    Returns:
        Tuple of feature tensors
    """
    # Combine RGB + Heatmap
    input_4ch = torch.cat([images, heatmaps], dim=1)  # (B, 4, H, W)

    # Preprocess
    processed = preprocessor(input_4ch)

    # Extract features
    features = backbone(processed)

    return features

# Example usage
if __name__ == "__main__":
    print("Creating 4-channel model...")

    preprocessor, backbone = create_4channel_model()

    print(f"Preprocessor: {preprocessor}")
    print(f"Backbone: {backbone}")

    # Example data
    batch_size = 4
    height, width = 640, 640

    # RGB images (normalized to 0-255)
    images = torch.randint(0, 256, (batch_size, 3, height, width), dtype=torch.float32)

    # Heatmaps (probability maps 0-1)
    heatmaps = torch.rand(batch_size, 1, height, width)

    print(f"\nProcessing batch of {batch_size} images...")
    print(f"Image shape: {images.shape}")
    print(f"Heatmap shape: {heatmaps.shape}")

    # Process
    with torch.no_grad():
        features = process_4channel_batch(preprocessor, backbone, images, heatmaps)

        print(f"\nOutput features:")
        for i, feat in enumerate(features):
            print(f"  P{i+3}: {feat.shape}")

    print("\n✓ 4-channel processing successful!")
