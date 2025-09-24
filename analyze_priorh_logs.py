"""Utility script to analyze PriorH weight evolution from logs."""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def parse_priorh_logs(log_file_path):
    """Parse PriorH weight norms from training logs.
    
    Args:
        log_file_path (str): Path to the training log file.
        
    Returns:
        dict: Parsed data with epochs, iterations, and norm values.
    """
    data = {
        'epochs': [],
        'iterations': [], 
        'rgb_norms': [],
        'priorh_norms': [],
        'rgb_ratios': [],
        'priorh_ratios': []
    }
    
    # Regex patterns for different log types
    patterns = {
        'epoch': r'epoch_\[Conv1 Weight Norms\] RGB: ([\d.]+) \(([\d.]+)\), PriorH: ([\d.]+) \(([\d.]+)\)',
        'iter': r'iter_\[Conv1 Weight Norms\] RGB: ([\d.]+) \(([\d.]+)\), PriorH: ([\d.]+) \(([\d.]+)\)',
        'epoch_num': r'Epoch\(train\)\s+\[(\d+)\]'
    }
    
    current_epoch = 0
    
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                # Track current epoch
                epoch_match = re.search(patterns['epoch_num'], line)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))
                
                # Parse epoch-level norms
                epoch_match = re.search(patterns['epoch'], line)
                if epoch_match:
                    rgb_norm, rgb_ratio, priorh_norm, priorh_ratio = map(float, epoch_match.groups())
                    data['epochs'].append(current_epoch)
                    data['rgb_norms'].append(rgb_norm)
                    data['priorh_norms'].append(priorh_norm)
                    data['rgb_ratios'].append(rgb_ratio)
                    data['priorh_ratios'].append(priorh_ratio)
                    
    except FileNotFoundError:
        print(f"Log file not found: {log_file_path}")
        return None
    except Exception as e:
        print(f"Error parsing log file: {e}")
        return None
        
    return data if data['epochs'] else None


def plot_priorh_evolution(data, output_dir='plots'):
    """Create plots showing PriorH weight evolution.
    
    Args:
        data (dict): Parsed log data.
        output_dir (str): Directory to save plots.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('PriorH Channel Weight Evolution During Training', fontsize=16)
    
    epochs = np.array(data['epochs'])
    rgb_norms = np.array(data['rgb_norms'])
    priorh_norms = np.array(data['priorh_norms'])
    rgb_ratios = np.array(data['rgb_ratios'])
    priorh_ratios = np.array(data['priorh_ratios'])
    
    # Plot 1: Absolute weight norms
    axes[0, 0].plot(epochs, rgb_norms, 'b-', label='RGB Channels', linewidth=2)
    axes[0, 0].plot(epochs, priorh_norms, 'r-', label='PriorH Channel', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Weight Norm')
    axes[0, 0].set_title('Absolute Weight Norms')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Weight ratios (relative importance)
    axes[0, 1].plot(epochs, rgb_ratios, 'b-', label='RGB Ratio', linewidth=2)
    axes[0, 1].plot(epochs, priorh_ratios, 'r-', label='PriorH Ratio', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Weight Ratio')
    axes[0, 1].set_title('Relative Weight Importance')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: PriorH vs RGB ratio
    if len(priorh_norms) > 0 and len(rgb_norms) > 0:
        ratio_priorh_to_rgb = priorh_norms / (rgb_norms + 1e-8)
        axes[1, 0].plot(epochs, ratio_priorh_to_rgb, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('PriorH/RGB Norm Ratio')
        axes[1, 0].set_title('PriorH Learning Progress')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add horizontal line at reasonable ratio
        axes[1, 0].axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, 
                          label='Target (10% of RGB)')
        axes[1, 0].legend()
    
    # Plot 4: PriorH norm growth rate
    if len(priorh_norms) > 1:
        growth_rate = np.diff(priorh_norms)
        axes[1, 1].plot(epochs[1:], growth_rate, 'purple', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('PriorH Norm Change')
        axes[1, 1].set_title('PriorH Learning Rate')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / 'priorh_evolution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 PriorH evolution plot saved to: {output_path}")
    
    # Show summary statistics
    print("\n=== PriorH Learning Summary ===")
    if len(priorh_norms) > 0:
        print(f"Initial PriorH norm: {priorh_norms[0]:.4f}")
        print(f"Final PriorH norm: {priorh_norms[-1]:.4f}")
        print(f"Total growth: {priorh_norms[-1] - priorh_norms[0]:.4f}")
        print(f"Final PriorH/RGB ratio: {priorh_norms[-1] / (rgb_norms[-1] + 1e-8):.4f}")
        
        if priorh_norms[-1] / (rgb_norms[-1] + 1e-8) > 0.05:
            print("✅ Model is learning to use PriorH channel!")
        else:
            print("⚠️  PriorH learning may be slow - consider reducing PriorDrop or increasing learning rate")


def main():
    parser = argparse.ArgumentParser(description='Analyze PriorH weight evolution from training logs')
    parser.add_argument('log_file', help='Path to training log file')
    parser.add_argument('--output-dir', default='plots', help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Parse logs
    data = parse_priorh_logs(args.log_file)
    if data is None:
        print("❌ Could not parse log data")
        return
        
    if not data['epochs']:
        print("❌ No PriorH monitoring data found in logs")
        return
        
    # Create plots
    plot_priorh_evolution(data, args.output_dir)


if __name__ == '__main__':
    main()