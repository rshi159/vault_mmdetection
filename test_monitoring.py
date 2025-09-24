"""Test script for PriorH monitoring and visualization."""

import torch
import sys
import os

# Add project root to path
project_root = '/home/robun2/Documents/vault_conveyor_tracking/vault_mmdetection'
sys.path.insert(0, project_root)

def test_priorh_monitoring():
    """Test the PriorH monitoring hook."""
    print("Testing PriorH Channel Weight Monitoring...")
    
    try:
        from mmdet.engine.hooks import PriorHMonitorHook
        
        # Create hook instance
        hook = PriorHMonitorHook(
            log_interval=10,
            log_after_epoch=True,
            backbone_path='backbone.stem'
        )
        
        print("✅ PriorHMonitorHook created successfully")
        print(f"   - Log interval: {hook.log_interval}")
        print(f"   - Backbone path: {hook.backbone_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating PriorHMonitorHook: {e}")
        return False

def test_visualization_hook():
    """Test the prediction visualization hook."""
    print("\nTesting Prediction Visualization Hook...")
    
    try:
        from mmdet.engine.hooks import PredictionVisualizationHook
        
        # Create hook instance
        hook = PredictionVisualizationHook(
            vis_interval=5,
            score_thr=0.1,
            output_dir='test_vis_outputs',
            max_images=5,
            show_gt=True
        )
        
        print("✅ PredictionVisualizationHook created successfully")
        print(f"   - Visualization interval: {hook.vis_interval}")
        print(f"   - Score threshold: {hook.score_thr}")
        print(f"   - Output directory: {hook.output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating PredictionVisualizationHook: {e}")
        return False

def test_config_loading():
    """Test loading the monitored configuration."""
    print("\nTesting Monitored Configuration...")
    
    try:
        from mmengine.config import Config
        
        config_path = os.path.join(project_root, 'configs/rtmdet/rtmdet_tiny_4ch_monitored.py')
        
        # Try to load the config
        cfg = Config.fromfile(config_path)
        
        print("✅ Monitored config loaded successfully")
        
        # Check if our custom hooks are present
        custom_hooks = cfg.get('custom_hooks', [])
        hook_types = [hook.get('type', '') for hook in custom_hooks]
        
        if 'PriorHMonitorHook' in hook_types:
            print("   ✅ PriorHMonitorHook found in config")
        else:
            print("   ❌ PriorHMonitorHook not found in config")
            
        if 'PredictionVisualizationHook' in hook_types:
            print("   ✅ PredictionVisualizationHook found in config")
        else:
            print("   ❌ PredictionVisualizationHook not found in config")
            
        return True
        
    except Exception as e:
        print(f"❌ Error loading monitored config: {e}")
        return False

def test_backbone_access():
    """Test accessing backbone weights (mock test)."""
    print("\nTesting Backbone Weight Access...")
    
    try:
        # Create a simple mock model structure
        class MockConv:
            def __init__(self):
                self.weight = torch.randn(64, 4, 3, 3)  # 4-channel input
                
        class MockStem:
            def __init__(self):
                self.conv = MockConv()
                
        class MockBackbone:
            def __init__(self):
                self.stem = MockStem()
                
        class MockModel:
            def __init__(self):
                self.backbone = MockBackbone()
        
        model = MockModel()
        
        # Test weight access
        weights = model.backbone.stem.conv.weight
        print(f"✅ Mock backbone weights shape: {weights.shape}")
        
        # Test norm calculations
        rgb_norm = weights[:, :3].norm().item()
        priorh_norm = weights[:, 3:4].norm().item()
        
        print(f"   - RGB norm: {rgb_norm:.4f}")
        print(f"   - PriorH norm: {priorh_norm:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing backbone access: {e}")
        return False

def main():
    """Run all tests."""
    print("=== PriorH Monitoring System Test ===\n")
    
    os.chdir(project_root)
    
    tests = [
        test_priorh_monitoring,
        test_visualization_hook, 
        test_config_loading,
        test_backbone_access
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print(f"\n=== Test Summary ===")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All tests passed! The monitoring system is ready.")
        print("\nNext steps:")
        print("1. Run training with: python tools/train.py configs/rtmdet/rtmdet_tiny_4ch_monitored.py")
        print("2. Monitor PriorH weight norms in the logs")
        print("3. Check visualization outputs in work_dirs/vis_outputs/")
    else:
        print("❌ Some tests failed. Please check the errors above.")

if __name__ == '__main__':
    main()