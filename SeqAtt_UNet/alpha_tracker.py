"""
================================================================================
ALPHA TRACKER - INTEGRATION FOR SeqAtt-UNet
================================================================================
Module used to track, log and visualize the learned α values of the CBAM
modules during training.

Designed to be integrated directly into train_with_alpha_tracker.py

FEATURES:
1. Automatically detect the CBAM modules inside the model
2. Log the α values of every epoch into a JSON file
3. Log to TensorBoard
4. Build the visualization used in the paper
5. Aggregate the results of several runs

USAGE:
    See the detailed guide in the file HUONG_DAN_TICH_HOP_ALPHA_TRACKER.md

Author: SeqAtt-UNet Project
Last updated: 02/01/2026
================================================================================
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch


class AlphaTracker:
    """
    Class used to track and log the learned α values of the CBAM modules.
    
    How it works:
    - At every epoch, call log_epoch() to record the α values
    - At the end of training, call save_logs() and plot_evolution()
    - Data is stored as JSON so that it is easy to aggregate several runs
    
    Attributes:
        model: PyTorch model containing the CBAM modules
        log_dir: Directory where the logs are saved
        logs: Dictionary holding the logged data
    """
    
    def __init__(self, model, log_dir='./alpha_logs', experiment_name=None):
        """
        Initialize the AlphaTracker.
        
        Args:
            model: PyTorch model (VisionTransformer with CBAM)
            log_dir: Directory where the logs are saved
            experiment_name: Experiment name (used to tell the runs apart)
        """
        self.model = model
        self.log_dir = log_dir
        self.experiment_name = experiment_name or datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Initialize the logs structure
        self.logs = {
            'experiment_name': self.experiment_name,
            'created_at': datetime.now().isoformat(),
            'epochs': [],
            'cbam1_alpha': [],
            'cbam2_alpha': [],
            'cbam3_alpha': [],
            'mean_alpha': [],
            'timestamps': [],
            'losses': [],  # Optional: track the loss at the same time
        }
        
        # Create the log directory if it does not exist yet
        os.makedirs(log_dir, exist_ok=True)
        
        # Find the CBAM modules inside the model
        self.cbam_modules = self._find_cbam_alpha_params(model)
        
        if self.cbam_modules:
            print(f"[AlphaTracker] Initialized. Found {len(self.cbam_modules)} CBAM modules:")
            for name in self.cbam_modules.keys():
                print(f"  - {name}")
        else:
            print("[AlphaTracker] WARNING: No CBAM modules found!")
    
    def _find_cbam_alpha_params(self, model):
        """
        Find every CBAM module and its alpha parameters.
        
        [UPDATED] Supports both versions:
        - Old version: self.alpha is directly an nn.Parameter
        - New version: self.alpha_raw is the nn.Parameter and self.alpha is a property (softplus)
        
        Returns a dictionary: {short_name: module} (so that module.alpha can be read)
        """
        cbam_modules = {}
        
        # Look for CBAM modules exposing an alpha property or an alpha_raw parameter
        for name, module in model.named_modules():
            if 'cbam' in name.lower():
                # Check whether an alpha property is available
                if hasattr(module, 'alpha'):
                    # Take the short name (cbam1, cbam2, cbam3)
                    parts = name.split('.')
                    short_name = None
                    for part in parts:
                        if 'cbam' in part.lower():
                            short_name = part
                            break
                    if short_name is None:
                        short_name = parts[-1] if parts else name
                    
                    cbam_modules[short_name] = module
        
        return cbam_modules
    
    def get_current_alphas(self):
        """
        Read the current α value of every CBAM module.
        
        [UPDATED] Read through the alpha property (softplus applied, always >= 0)
        
        Returns:
            Dictionary {cbam_name: alpha_value}
        """
        alphas = {}
        
        for short_name, module in self.cbam_modules.items():
            try:
                # alpha is a property that returns softplus(alpha_raw) >= 0
                alpha_val = module.alpha.item()
                alphas[short_name] = alpha_val
            except Exception as e:
                # Fallback: try reading the raw parameter directly
                if hasattr(module, 'alpha_raw'):
                    import torch.nn.functional as F
                    alphas[short_name] = F.softplus(module.alpha_raw).item()
        
        return alphas
    
    def log_epoch(self, epoch, loss=None, writer=None):
        """
        Log the α values of the current epoch.
        
        Args:
            epoch: Current epoch number
            loss: Training loss (optional, tracked at the same time)
            writer: TensorBoard SummaryWriter (optional)
        
        Returns:
            Dictionary holding the current α values
        """
        alphas = self.get_current_alphas()
        
        # Write into the logs
        self.logs['epochs'].append(epoch)
        self.logs['timestamps'].append(datetime.now().isoformat())
        
        if loss is not None:
            self.logs['losses'].append(loss)
        
        # Log each CBAM module
        for cbam_name in ['cbam1', 'cbam2', 'cbam3']:
            if cbam_name in alphas:
                self.logs[f'{cbam_name}_alpha'].append(alphas[cbam_name])
            else:
                self.logs[f'{cbam_name}_alpha'].append(None)
        
        # Log mean alpha
        valid_alphas = [v for v in alphas.values() if v is not None]
        mean_alpha = sum(valid_alphas) / len(valid_alphas) if valid_alphas else 0
        self.logs['mean_alpha'].append(mean_alpha)
        
        # Log to TensorBoard when a writer is available
        if writer is not None:
            for cbam_name, alpha_val in alphas.items():
                writer.add_scalar(f'cbam_alpha/{cbam_name}', alpha_val, epoch)
            writer.add_scalar('cbam_alpha/mean', mean_alpha, epoch)
        
        return alphas
    
    def save_logs(self, filename=None):
        """
        Save the logs to a JSON file.
        
        Args:
            filename: File name (default: alpha_logs_{experiment_name}.json)
        """
        if filename is None:
            filename = f'alpha_logs_{self.experiment_name}.json'
        
        filepath = os.path.join(self.log_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, indent=2, ensure_ascii=False)
        
        print(f"[AlphaTracker] Logs saved to: {filepath}")
        return filepath
    
    def load_logs(self, filepath):
        """
        Load logs from a JSON file, with error handling.
        
        Args:
            filepath: Path to the JSON log file
            
        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file is not valid JSON
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Alpha log file not found: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.logs = json.load(f)
            print(f"[AlphaTracker] Loaded logs from: {filepath}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {filepath}: {e}")
    
    def plot_evolution(self, save_path=None, show=True, title_suffix=''):
        """
        Plot the evolution of the α values.
        
        Args:
            save_path: Path where the figure is saved (optional)
            show: Whether to display the plot
            title_suffix: Extra text appended to the title
        
        Returns:
            matplotlib Figure object
        """
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 6))
        
        epochs = self.logs['epochs']
        
        colors = {
            'cbam1': '#1f77b4',  # Blue
            'cbam2': '#ff7f0e',  # Orange
            'cbam3': '#2ca02c'   # Green
        }
        
        labels = {
            'cbam1': 'CBAM-1 (56×56×256)',
            'cbam2': 'CBAM-2 (28×28×512)',
            'cbam3': 'CBAM-3 (14×14×1024)'
        }
        
        # Plot each CBAM module
        for cbam_name in ['cbam1', 'cbam2', 'cbam3']:
            values = self.logs.get(f'{cbam_name}_alpha', [])
            if values and values[0] is not None:
                ax.plot(epochs, values, 
                       color=colors[cbam_name], 
                       linewidth=2.5,
                       label=labels[cbam_name],
                       marker='o',
                       markersize=3,
                       markevery=max(1, len(epochs)//20))  # One marker every 5% of the epochs
        
        # Reference line at α = 0.01 (the initialization value)
        ax.axhline(y=0.01, color='gray', linestyle='--', 
                  linewidth=1.5, alpha=0.7, label='Initial α = 0.01')
        
        # Styling
        ax.set_xlabel('Training Epoch', fontsize=14, fontweight='bold')
        ax.set_ylabel('Learned α Value', fontsize=14, fontweight='bold')
        ax.set_title(f'Evolution of CBAM α Values During Training{title_suffix}', 
                    fontsize=16, fontweight='bold')
        ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
        ax.set_xlim(0, max(epochs) if epochs else 150)
        ax.set_ylim(0, max(0.6, max(self.logs['mean_alpha']) * 1.2) if self.logs['mean_alpha'] else 0.6)
        ax.grid(True, alpha=0.3)
        
        # Annotate the final α value
        if epochs:
            for cbam_name in ['cbam1', 'cbam2', 'cbam3']:
                values = self.logs.get(f'{cbam_name}_alpha', [])
                if values and values[-1] is not None:
                    ax.annotate(f'{values[-1]:.3f}',
                               xy=(epochs[-1], values[-1]),
                               xytext=(5, 0),
                               textcoords='offset points',
                               fontsize=10,
                               color=colors[cbam_name],
                               fontweight='bold')
        
        plt.tight_layout()
        
        # Save the figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"[AlphaTracker] Plot saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def get_summary(self):
        """
        Get the summary statistics of the α values.
        
        Returns:
            Dictionary with the statistics of each CBAM module
        """
        summary = {
            'experiment_name': self.experiment_name,
            'total_epochs': len(self.logs['epochs']),
        }
        
        for cbam_name in ['cbam1', 'cbam2', 'cbam3']:
            values = [v for v in self.logs.get(f'{cbam_name}_alpha', []) if v is not None]
            
            if values:
                summary[cbam_name] = {
                    'initial': values[0],
                    'final': values[-1],
                    'max': max(values),
                    'min': min(values),
                    'mean_last_10': np.mean(values[-10:]) if len(values) >= 10 else np.mean(values),
                    'change': values[-1] - values[0],
                    'change_percent': ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else float('inf')
                }
        
        return summary
    
    def print_summary(self):
        """Print the summary to the console."""
        summary = self.get_summary()
        
        print("\n" + "=" * 70)
        print("CBAM ALPHA VALUES SUMMARY")
        print("=" * 70)
        print(f"Experiment: {summary['experiment_name']}")
        print(f"Total epochs: {summary['total_epochs']}")
        print("-" * 70)
        
        for cbam_name in ['cbam1', 'cbam2', 'cbam3']:
            if cbam_name in summary:
                stats = summary[cbam_name]
                print(f"\n{cbam_name.upper()}:")
                print(f"  Initial value:     {stats['initial']:.6f}")
                print(f"  Final value:       {stats['final']:.6f}")
                print(f"  Change:            {stats['change']:+.6f} ({stats['change_percent']:+.1f}%)")
                print(f"  Max value:         {stats['max']:.6f}")
                print(f"  Mean (last 10):    {stats['mean_last_10']:.6f}")
        
        print("\n" + "=" * 70)


def aggregate_multiple_runs(log_files, output_dir='./aggregated_logs'):
    """
    Aggregate the α logs of several runs to compute mean ± std.
    
    Args:
        log_files: List of paths to the JSON log files
        output_dir: Output directory
    
    Returns:
        Dictionary with the aggregated data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_runs = {
        'cbam1': [],
        'cbam2': [],
        'cbam3': []
    }
    
    # Load every run
    for filepath in log_files:
        with open(filepath, 'r') as f:
            logs = json.load(f)
        
        for cbam_name in ['cbam1', 'cbam2', 'cbam3']:
            values = logs.get(f'{cbam_name}_alpha', [])
            if values and values[0] is not None:
                all_runs[cbam_name].append(values)
    
    # Compute the mean and the std
    aggregated = {
        'num_runs': len(log_files),
        'epochs': logs['epochs'],  # Assumes that every run has the same number of epochs
    }
    
    for cbam_name in ['cbam1', 'cbam2', 'cbam3']:
        if all_runs[cbam_name]:
            # Make sure that every run has the same length
            min_len = min(len(run) for run in all_runs[cbam_name])
            runs_array = np.array([run[:min_len] for run in all_runs[cbam_name]])
            
            aggregated[f'{cbam_name}_mean'] = runs_array.mean(axis=0).tolist()
            aggregated[f'{cbam_name}_std'] = runs_array.std(axis=0).tolist()
            aggregated[f'{cbam_name}_final_mean'] = float(runs_array[:, -1].mean())
            aggregated[f'{cbam_name}_final_std'] = float(runs_array[:, -1].std())
    
    # Save aggregated data
    output_path = os.path.join(output_dir, 'aggregated_alpha_logs.json')
    with open(output_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"[AlphaTracker] Aggregated {len(log_files)} runs to: {output_path}")
    
    return aggregated


def plot_aggregated_evolution(aggregated_logs, save_path=None, show=True):
    """
    Plot the aggregated α evolution with error bands.
    
    Args:
        aggregated_logs: Dictionary returned by aggregate_multiple_runs()
        save_path: Path where the figure is saved
        show: Whether to display the plot
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    epochs = aggregated_logs['epochs']
    colors = {'cbam1': '#1f77b4', 'cbam2': '#ff7f0e', 'cbam3': '#2ca02c'}
    
    for cbam_name in ['cbam1', 'cbam2', 'cbam3']:
        mean_key = f'{cbam_name}_mean'
        std_key = f'{cbam_name}_std'
        
        if mean_key in aggregated_logs:
            mean = np.array(aggregated_logs[mean_key])
            std = np.array(aggregated_logs[std_key])
            
            final_mean = aggregated_logs.get(f'{cbam_name}_final_mean', mean[-1])
            final_std = aggregated_logs.get(f'{cbam_name}_final_std', std[-1])
            
            label = f'CBAM-{cbam_name[-1]} (final: {final_mean:.3f}±{final_std:.3f})'
            
            ax.plot(epochs[:len(mean)], mean, color=colors[cbam_name], 
                   linewidth=2.5, label=label)
            ax.fill_between(epochs[:len(mean)], mean - std, mean + std, 
                           color=colors[cbam_name], alpha=0.2)
    
    ax.axhline(y=0.01, color='gray', linestyle='--', linewidth=1.5, 
              alpha=0.7, label='Initial α = 0.01')
    
    ax.set_xlabel('Training Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Learned α Value', fontsize=14, fontweight='bold')
    ax.set_title(f"CBAM α Evolution ({aggregated_logs['num_runs']} runs, mean ± std)", 
                fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.set_xlim(0, max(epochs))
    ax.set_ylim(0, 0.6)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"[AlphaTracker] Aggregated plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


# =============================================================================
# DEMO & TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ALPHA TRACKER - Demo Mode")
    print("=" * 70)
    
    # Create the demo logs
    print("\nCreating demo alpha logs...")
    
    np.random.seed(42)
    num_epochs = 150
    
    def simulate_alpha(init, final, epochs, noise=0.008):
        """Simulate smooth alpha increase."""
        x = np.linspace(-6, 6, epochs)
        base = init + (final - init) / (1 + np.exp(-x))
        return (base + noise * np.random.randn(epochs)).tolist()
    
    # Create a fake tracker
    class FakeModel:
        def named_parameters(self):
            return []
    
    tracker = AlphaTracker(FakeModel(), log_dir='./demo_logs', experiment_name='demo_run')
    
    # Fill demo data
    tracker.logs['epochs'] = list(range(num_epochs))
    tracker.logs['cbam1_alpha'] = simulate_alpha(0.01, 0.23, num_epochs)
    tracker.logs['cbam2_alpha'] = simulate_alpha(0.01, 0.41, num_epochs)
    tracker.logs['cbam3_alpha'] = simulate_alpha(0.01, 0.18, num_epochs)
    tracker.logs['mean_alpha'] = [
        (a + b + c) / 3 
        for a, b, c in zip(
            tracker.logs['cbam1_alpha'],
            tracker.logs['cbam2_alpha'],
            tracker.logs['cbam3_alpha']
        )
    ]
    
    # Save and plot
    tracker.save_logs()
    tracker.plot_evolution(save_path='./demo_logs/demo_alpha_evolution.png', show=False)
    tracker.print_summary()
    
    print("\n[OK] Demo completed! Check ./demo_logs/ for outputs.")
