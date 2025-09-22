
import torch
import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from preprocessor_4ch import DetDataPreprocessor4Ch
from backbone_4ch import CSPNeXt4Ch

def test_4channel_standalone():
    """Test the standalone 4-channel implementation."""
    print("Testing standalone 4-channel implementation...")

    # Test 1: Create components
    print("\n1. Creating components...")

    preprocessor = DetDataPreprocessor4Ch(
        mean=[103.53, 116.28, 123.675, 0.0],
        std=[57.375, 57.12, 57.375, 1.0]
    )
    print(f"   Preprocessor: {preprocessor}")

    backbone = CSPNeXt4Ch(
        arch='P5',
        widen_factor=0.375,
        deepen_factor=0.167,
        out_indices=(2, 3, 4)
    )
    print(f"   Backbone: {backbone}")

    # Test 2: Test with sample data
    print("\n2. Testing with sample data...")

    # Create 4-channel test data (RGB + Heatmap)
    batch_size = 2
    height, width = 640, 640

    # RGB channels (0-255 range)
    rgb_data = torch.randint(0, 256, (batch_size, 3, height, width), dtype=torch.float32)

    # Heatmap channel (0-1 range, representing probability)
    heatmap_data = torch.rand(batch_size, 1, height, width)

    # Combine to 4-channel input
    input_data = torch.cat([rgb_data, heatmap_data], dim=1)
    print(f"   Input shape: {input_data.shape}")
    print(f"   Input range: RGB [{rgb_data.min():.1f}, {rgb_data.max():.1f}], "
          f"Heatmap [{heatmap_data.min():.3f}, {heatmap_data.max():.3f}]")

    # Test 3: Preprocessing
    print("\n3. Testing preprocessing...")

    preprocessor.eval()
    with torch.no_grad():
        processed = preprocessor(input_data)
        print(f"   Processed shape: {processed.shape}")
        print(f"   Processed range: [{processed.min():.3f}, {processed.max():.3f}]")

    # Test 4: Backbone forward pass
    print("\n4. Testing backbone...")

    backbone.eval()
    with torch.no_grad():
        features = backbone(processed)
        print(f"   Number of feature levels: {len(features)}")
        for i, feat in enumerate(features):
            print(f"   Feature {i}: {feat.shape}")

    # Test 5: Memory and parameters
    print("\n5. Model information...")

    total_params = sum(p.numel() for p in backbone.parameters())
    trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)

    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # Test memory usage
    with torch.no_grad():
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        _ = backbone(processed)

        memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        memory_used = (memory_after - memory_before) / 1024 / 1024  # MB

        if torch.cuda.is_available():
            print(f"   GPU memory used: {memory_used:.1f} MB")
        else:
            print("   Running on CPU")

    print("\n" + "="*50)
    print("✓ All tests passed! 4-channel implementation works.")
    print("="*50)

    return True

if __name__ == "__main__":
    test_4channel_standalone()
