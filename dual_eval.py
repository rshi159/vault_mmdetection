#!/usr/bin/env python3
"""
Dual Resolution Evaluation Script
Evaluates trained model at both 640px (deploy) and 768px (training) resolutions
"""

import argparse
import subprocess
import sys

def run_evaluation(config_path, checkpoint_path, work_dir):
    """Run evaluation at both resolutions."""
    
    print("🔍 Dual Resolution Evaluation")
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Work dir: {work_dir}")
    print("-" * 60)
    
    # 1. Evaluate at 640px (deployment resolution)
    print("🎯 Evaluating at 640px (deployment resolution)...")
    cmd_640 = [
        "python", "tools/test.py",
        config_path,
        checkpoint_path,
        "--work-dir", f"{work_dir}/eval_640px",
        "--show-dir", f"{work_dir}/eval_640px/vis"
    ]
    
    try:
        result = subprocess.run(cmd_640, check=True, capture_output=True, text=True)
        print("✅ 640px evaluation completed")
        print("Deploy metrics (640px):")
        # Extract key metrics from output
        lines = result.stdout.split('\n')
        for line in lines:
            if 'bbox_mAP' in line or 'mAP' in line:
                print(f"   {line.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"❌ 640px evaluation failed: {e}")
        print(f"Error output: {e.stderr}")
        return False
    
    print("-" * 40)
    
    # 2. Evaluate at 768px (training resolution) 
    print("📈 Evaluating at 768px (training resolution)...")
    
    # Create a temporary config that uses 768px test pipeline
    temp_config = f"{work_dir}/temp_768_config.py"
    with open(config_path, 'r') as f:
        config_content = f.read()
    
    # Replace test_dataloader to use 768px pipeline
    config_content = config_content.replace(
        "pipeline=test_pipeline_640,",
        "pipeline=test_pipeline_768,"
    )
    
    with open(temp_config, 'w') as f:
        f.write(config_content)
    
    cmd_768 = [
        "python", "tools/test.py", 
        temp_config,
        checkpoint_path,
        "--work-dir", f"{work_dir}/eval_768px",
        "--show-dir", f"{work_dir}/eval_768px/vis"
    ]
    
    try:
        result = subprocess.run(cmd_768, check=True, capture_output=True, text=True)
        print("✅ 768px evaluation completed")
        print("Training metrics (768px):")
        # Extract key metrics from output
        lines = result.stdout.split('\n')
        for line in lines:
            if 'bbox_mAP' in line or 'mAP' in line:
                print(f"   {line.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"❌ 768px evaluation failed: {e}")
        print(f"Error output: {e.stderr}")
        return False
    
    print("\n" + "="*60)
    print("🎯 Dual Resolution Evaluation Complete!")
    print("📊 Results saved to:")
    print(f"   • 640px (deploy): {work_dir}/eval_640px/")
    print(f"   • 768px (train):  {work_dir}/eval_768px/")
    print("📈 Use higher training resolution (768px) metrics for model selection")
    print("🚀 Use deployment resolution (640px) metrics for production planning")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual resolution evaluation")
    parser.add_argument("config", help="Path to config file")
    parser.add_argument("checkpoint", help="Path to checkpoint file") 
    parser.add_argument("--work-dir", default="./work_dirs/dual_eval", help="Work directory")
    
    args = parser.parse_args()
    
    success = run_evaluation(args.config, args.checkpoint, args.work_dir)
    sys.exit(0 if success else 1)