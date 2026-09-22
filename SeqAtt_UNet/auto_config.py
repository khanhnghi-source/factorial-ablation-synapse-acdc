"""
AUTO-CONFIG MODULE FOR SeqAtt-UNet - PHASE 3
============================================
Self-configuring hyperparameter selection in the spirit of nnU-Net.

Contents:
1. Dataset Analysis: compute dataset statistics
2. Patch Size Optimization: automatically select the optimal patch size
3. Batch Size & Learning Rate Scaling: linear scaling rule
4. Network Topology: adjust the number of layers and channels
5. Memory Estimation: estimate the required GPU memory

References:
- nnU-Net: "Automated Design of Deep Learning Methods for Biomedical Image Segmentation"
- Linear Scaling Rule: Goyal et al., "Accurate, Large Minibatch SGD"

Author: SeqAtt-UNet Project - Phase 3 Improvements
"""

import os
import sys
import math
import json
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DatasetStatistics:
    """
    Dataset statistics - the basis for automatic configuration.

    Attributes:
        name: Dataset name (Synapse, ACDC, etc.)
        num_samples: Number of training samples
        num_classes: Number of segmentation classes
        image_shape: Original image size (H, W), or (D, H, W) for 3D
        voxel_spacing: Spacing between voxels (for medical images)
        intensity_range: (min, max) of the intensity values
        class_frequencies: Occurrence frequency of each class
        median_shape: Median image size across the dataset
        is_3d: Whether the dataset is 3D
    """
    name: str
    num_samples: int
    num_classes: int
    image_shape: Tuple[int, ...]
    voxel_spacing: Optional[Tuple[float, ...]] = None
    intensity_range: Tuple[float, float] = (0.0, 1.0)
    class_frequencies: Optional[Dict[int, float]] = None
    median_shape: Optional[Tuple[int, ...]] = None
    is_3d: bool = False


@dataclass
class AutoConfig:
    """
    Configuration automatically derived from the dataset statistics.

    Attributes:
        patch_size: Optimal patch size
        batch_size: Recommended batch size
        base_lr: Base learning rate
        num_epochs: Recommended number of epochs
        encoder_channels: Number of channels for the encoder stages
        decoder_channels: Number of channels for the decoder stages
        num_pool_stages: Number of pooling stages
        deep_supervision_weights: Weights for deep supervision
        augmentation_config: Augmentation configuration
        estimated_memory_gb: Estimated GPU memory requirement
    """
    # Core hyperparameters
    patch_size: Tuple[int, int] = (224, 224)
    batch_size: int = 12
    base_lr: float = 0.01
    num_epochs: int = 150
    
    # Network architecture
    encoder_channels: Tuple[int, ...] = (64, 128, 256, 512)
    decoder_channels: Tuple[int, ...] = (256, 128, 64, 16)
    num_pool_stages: int = 4
    
    # Deep supervision
    use_deep_supervision: bool = True
    deep_supervision_weights: Tuple[float, ...] = (1.0, 0.4, 0.2, 0.1)
    
    # Augmentation
    augmentation_strength: str = 'strong'
    
    # Memory estimate
    estimated_memory_gb: float = 8.0
    
    # Metadata
    config_source: str = 'auto'
    dataset_name: str = ''
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary."""
        return asdict(self)

    def save(self, path: str):
        """Save the config to a JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'AutoConfig':
        """Load the config from a JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        # Convert lists back to tuples
        for key in ['patch_size', 'encoder_channels', 'decoder_channels', 'deep_supervision_weights']:
            if key in data and isinstance(data[key], list):
                data[key] = tuple(data[key])
        return cls(**data)


# =============================================================================
# DATASET ANALYZER
# =============================================================================

class DatasetAnalyzer:
    """
    Compute dataset statistics that drive the auto-config.

    Supported:
    - Synapse dataset (multi-organ CT)
    - ACDC dataset (cardiac MRI)
    - Generic NPZ/H5 datasets
    """

    def __init__(self, root_path: str, dataset_type: str = 'Synapse'):
        """
        Args:
            root_path: Path to the data directory
            dataset_type: Dataset type ('Synapse', 'ACDC', 'generic')
        """
        self.root_path = Path(root_path)
        self.dataset_type = dataset_type
        
    def analyze(self) -> DatasetStatistics:
        """
        Analyze the whole dataset.

        Returns:
            DatasetStatistics object holding the computed statistics
        """
        if self.dataset_type == 'Synapse':
            return self._analyze_synapse()
        elif self.dataset_type == 'ACDC':
            return self._analyze_acdc()
        else:
            return self._analyze_generic()
    
    def _analyze_synapse(self) -> DatasetStatistics:
        """Analyze the Synapse dataset."""
        # Synapse dataset defaults
        num_classes = 9
        image_shape = (224, 224)  # After resizing

        # Count the samples
        num_samples = 0
        if self.root_path.exists():
            npz_files = list(self.root_path.glob('*.npz'))
            num_samples = len(npz_files)

        # Analyze the sample data when it is available
        class_frequencies = self._compute_class_frequencies_synapse() if num_samples > 0 else None
        
        return DatasetStatistics(
            name='Synapse',
            num_samples=num_samples,
            num_classes=num_classes,
            image_shape=image_shape,
            voxel_spacing=(1.0, 1.0),  # Normalized
            intensity_range=(-1.0, 1.0),  # After normalization
            class_frequencies=class_frequencies,
            median_shape=image_shape,
            is_3d=False
        )
    
    def _analyze_acdc(self) -> DatasetStatistics:
        """Analyze the ACDC dataset."""
        num_classes = 4
        image_shape = (224, 224)

        # Count the samples
        num_samples = 0
        slices_dir = self.root_path / 'ACDC_training_slices'
        if slices_dir.exists():
            h5_files = list(slices_dir.glob('*.h5'))
            num_samples = len(h5_files)
        
        return DatasetStatistics(
            name='ACDC',
            num_samples=num_samples,
            num_classes=num_classes,
            image_shape=image_shape,
            voxel_spacing=(1.0, 1.0),
            intensity_range=(-1.0, 1.0),
            class_frequencies=None,
            median_shape=image_shape,
            is_3d=False
        )
    
    def _analyze_generic(self) -> DatasetStatistics:
        """Analyze a generic dataset."""
        return DatasetStatistics(
            name='generic',
            num_samples=0,
            num_classes=2,
            image_shape=(224, 224),
            is_3d=False
        )
    
    def _compute_class_frequencies_synapse(self, max_samples: int = 100) -> Dict[int, float]:
        """
        Compute the occurrence frequency of each class in the Synapse dataset.

        Args:
            max_samples: Maximum number of samples to analyze (to save time)

        Returns:
            Dictionary mapping class index -> frequency
        """
        npz_files = list(self.root_path.glob('*.npz'))[:max_samples]
        
        if not npz_files:
            return {}
        
        class_counts = {}
        total_pixels = 0
        
        for npz_file in npz_files:
            try:
                data = np.load(npz_file)
                label = data['label']
                
                for c in range(9):  # 9 classes in Synapse
                    count = np.sum(label == c)
                    class_counts[c] = class_counts.get(c, 0) + count
                
                total_pixels += label.size
            except Exception as e:
                logger.warning(f"Error loading {npz_file}: {e}")
                continue
        
        # Convert to frequencies
        if total_pixels > 0:
            return {c: count / total_pixels for c, count in class_counts.items()}
        return {}


# =============================================================================
# AUTO-CONFIG GENERATOR
# =============================================================================

class AutoConfigGenerator:
    """
    Automatically generate an optimal configuration from the dataset statistics.

    The algorithm follows the nnU-Net design principles:
    1. The patch size is chosen from the median image size and the GPU memory
    2. The batch size is maximized within the GPU memory budget
    3. The learning rate follows the linear scaling rule
    4. The network topology is adjusted to the resolution
    """

    # Configuration constants
    DEFAULT_GPU_MEMORY_GB = 11.0  # GTX 1080 Ti
    BASE_BATCH_SIZE = 12  # Batch size reference
    BASE_LR = 0.01  # Learning rate reference
    MIN_PATCH_SIZE = 64
    MAX_PATCH_SIZE = 512
    
    # Dataset-specific defaults
    DATASET_DEFAULTS = {
        'Synapse': {
            'patch_size': (224, 224),
            'batch_size': 12,
            'num_epochs': 150,
            'num_classes': 9,
        },
        'ACDC': {
            'patch_size': (224, 224),
            'batch_size': 12,
            'num_epochs': 200,
            'num_classes': 4,
        },
    }
    
    def __init__(self, 
                 gpu_memory_gb: float = DEFAULT_GPU_MEMORY_GB,
                 use_amp: bool = True):
        """
        Args:
            gpu_memory_gb: Available GPU memory (GB)
            use_amp: Whether Automatic Mixed Precision is used
        """
        self.gpu_memory_gb = gpu_memory_gb
        self.use_amp = use_amp
        # AMP reduces memory usage by roughly 30-40%
        self.effective_memory = gpu_memory_gb * (1.4 if use_amp else 1.0)
    
    def generate(self, stats: DatasetStatistics) -> AutoConfig:
        """
        Generate an optimal configuration from the dataset statistics.

        Args:
            stats: DatasetStatistics object

        Returns:
            AutoConfig holding the optimized hyperparameters
        """
        logger.info(f"Generating auto-config for {stats.name} dataset...")
        
        # 1. Patch size optimization
        patch_size = self._optimize_patch_size(stats)
        
        # 2. Batch size optimization
        batch_size = self._optimize_batch_size(stats, patch_size)
        
        # 3. Learning rate with linear scaling
        base_lr = self._compute_learning_rate(batch_size)

        # 4. Number of epochs based on the dataset size
        num_epochs = self._compute_num_epochs(stats)
        
        # 5. Network topology
        encoder_channels, decoder_channels = self._optimize_network_topology(patch_size)
        
        # 6. Deep supervision weights
        ds_weights = self._compute_deep_supervision_weights()
        
        # 7. Augmentation strength
        aug_strength = self._determine_augmentation_strength(stats)
        
        # 8. Memory estimation
        memory_estimate = self._estimate_memory(batch_size, patch_size)
        
        config = AutoConfig(
            patch_size=patch_size,
            batch_size=batch_size,
            base_lr=base_lr,
            num_epochs=num_epochs,
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            num_pool_stages=4,
            use_deep_supervision=True,
            deep_supervision_weights=ds_weights,
            augmentation_strength=aug_strength,
            estimated_memory_gb=memory_estimate,
            config_source='auto',
            dataset_name=stats.name
        )
        
        # Log config
        self._log_config(config, stats)
        
        return config
    
    def _optimize_patch_size(self, stats: DatasetStatistics) -> Tuple[int, int]:
        """
        Optimize the patch size from the image size and the GPU memory.

        Guidelines:
        - The patch size should be a multiple of 16 (for the ViT patches)
        - It should not be too small (< 96) so that enough context is captured
        - It should not be too large (> 384) to avoid memory overflow
        """
        if stats.name in self.DATASET_DEFAULTS:
            return self.DATASET_DEFAULTS[stats.name]['patch_size']

        # Based on the median shape
        if stats.median_shape:
            h, w = stats.median_shape[:2]
        else:
            h, w = stats.image_shape[:2]

        # Find a suitable patch size
        target_size = min(h, w)

        # Round to nearest multiple of 16
        candidate_sizes = [96, 128, 160, 192, 224, 256, 288, 320, 352, 384]

        # Pick the size closest to the target without exceeding it
        patch_size = 224  # default
        for size in candidate_sizes:
            if size <= target_size:
                patch_size = size
            else:
                break

        # Memory constraint: shrink the patch when GPU memory is limited
        if self.effective_memory < 8:
            patch_size = min(patch_size, 192)
        elif self.effective_memory < 6:
            patch_size = min(patch_size, 160)
        
        return (patch_size, patch_size)
    
    def _optimize_batch_size(self, 
                             stats: DatasetStatistics, 
                             patch_size: Tuple[int, int]) -> int:
        """
        Optimize the batch size from the available GPU memory.

        Empirical memory estimation formula:
        memory_per_sample ~= patch_size^2 * num_channels * 4 bytes * multiplier

        with multiplier ~ 50-100 for TransUNet (activations, gradients, etc.)
        """
        if stats.name in self.DATASET_DEFAULTS:
            base_bs = self.DATASET_DEFAULTS[stats.name]['batch_size']
        else:
            base_bs = self.BASE_BATCH_SIZE
        
        # Estimate the memory needed for batch_size = 1
        h, w = patch_size
        # Rough estimate: ~0.5GB per sample at 224x224 for TransUNet
        memory_per_sample = (h * w / (224 * 224)) * 0.5

        # Largest batch size that fits in memory
        # Reserve ~2GB for the PyTorch overhead
        available_memory = self.effective_memory - 2.0
        max_batch_size = max(1, int(available_memory / memory_per_sample))

        # Never exceed the default batch size (keeps training stable)
        batch_size = min(base_bs, max_batch_size)

        # Round down to an even number (so gradient accumulation works well)
        if batch_size > 2:
            batch_size = (batch_size // 2) * 2
        
        return max(1, batch_size)
    
    def _compute_learning_rate(self, batch_size: int) -> float:
        """
        Compute the learning rate with the Linear Scaling Rule.

        Rule: when the batch size grows by a factor of k, scale the LR by k.

        Reference: Goyal et al., "Accurate, Large Minibatch SGD"
        """
        # The base LR was tuned for batch_size = 12
        scale_factor = batch_size / self.BASE_BATCH_SIZE

        # Apply linear scaling within bounds
        lr = self.BASE_LR * scale_factor

        # Clamp to avoid an LR that is too high or too low
        lr = max(0.001, min(lr, 0.1))
        
        return lr
    
    def _compute_num_epochs(self, stats: DatasetStatistics) -> int:
        """
        Determine the number of epochs from the dataset size.

        Small datasets need more epochs for the model to converge.
        """
        if stats.name in self.DATASET_DEFAULTS:
            return self.DATASET_DEFAULTS[stats.name]['num_epochs']

        # Heuristic: fewer than 1000 samples -> 150-200 epochs are needed
        if stats.num_samples < 1000:
            return 150
        elif stats.num_samples < 5000:
            return 100
        else:
            return 80
    
    def _optimize_network_topology(self, 
                                   patch_size: Tuple[int, int]
                                   ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """
        Adjust the network topology to the patch size.

        TransUNet encoder output = patch_size / 16
        The decoder needs enough capacity to upsample back to full resolution
        """
        # Standard TransUNet decoder channels
        # Designed for a 224x224 input
        encoder_channels = (64, 128, 256, 512)
        decoder_channels = (256, 128, 64, 16)

        # For a smaller patch size the channel widths can be reduced
        h, w = patch_size
        if h < 160:
            encoder_channels = (32, 64, 128, 256)
            decoder_channels = (128, 64, 32, 16)
        
        return encoder_channels, decoder_channels
    
    def _compute_deep_supervision_weights(self) -> Tuple[float, ...]:
        """
        Compute the Deep Supervision weights.

        The weights decay for auxiliary outputs at lower resolutions.
        """
        # Standard weights: [1.0, 0.4, 0.2, 0.1]
        # Final output: weight = 1.0
        # Intermediate outputs: weights decreasing
        return (1.0, 0.4, 0.2, 0.1)
    
    def _determine_augmentation_strength(self, stats: DatasetStatistics) -> str:
        """
        Determine the augmentation strength from the dataset size.

        Small datasets need stronger augmentation to avoid overfitting.
        """
        if stats.num_samples < 500:
            return 'strong'
        elif stats.num_samples < 2000:
            return 'medium'
        else:
            return 'light'
    
    def _estimate_memory(self, 
                        batch_size: int, 
                        patch_size: Tuple[int, int]) -> float:
        """
        Estimate the required GPU memory (GB).
        """
        h, w = patch_size
        # Empirical formula for TransUNet
        memory = batch_size * (h * w / (224 * 224)) * 0.5
        # Add overhead
        memory = memory * 1.2 + 2.0
        return round(memory, 1)
    
    def _log_config(self, config: AutoConfig, stats: DatasetStatistics):
        """Log the generated configuration."""
        logger.info("=" * 60)
        logger.info("AUTO-CONFIG GENERATED")
        logger.info("=" * 60)
        logger.info(f"Dataset: {stats.name} ({stats.num_samples} samples)")
        logger.info(f"Patch size: {config.patch_size}")
        logger.info(f"Batch size: {config.batch_size}")
        logger.info(f"Learning rate: {config.base_lr}")
        logger.info(f"Epochs: {config.num_epochs}")
        logger.info(f"Deep supervision: {config.use_deep_supervision}")
        logger.info(f"DS weights: {config.deep_supervision_weights}")
        logger.info(f"Augmentation: {config.augmentation_strength}")
        logger.info(f"Estimated memory: {config.estimated_memory_gb} GB")
        logger.info("=" * 60)


# =============================================================================
# LEARNING RATE SCHEDULERS
# =============================================================================

class WarmupCosineScheduler:
    """
    Learning rate scheduler with Warmup + Cosine Annealing.

    Widely used by state-of-the-art methods (ViT, DeiT, etc.)
    """
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 warmup_epochs: int,
                 total_epochs: int,
                 base_lr: float,
                 min_lr: float = 1e-6):
        """
        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of warmup epochs
            total_epochs: Total number of epochs
            base_lr: Base learning rate
            min_lr: Minimum learning rate
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
    
    def step(self, epoch: Optional[int] = None):
        """
        Update the learning rate for the next epoch.

        Args:
            epoch: Current epoch (optional, incremented automatically when omitted)
        """
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        lr = self._compute_lr(self.current_epoch)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def _compute_lr(self, epoch: int) -> float:
        """Compute the LR for a specific epoch."""
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        
        return lr
    
    def get_lr(self) -> float:
        """Get the current LR."""
        return self._compute_lr(self.current_epoch)


class WarmupPolynomialScheduler:
    """
    Learning rate scheduler with Warmup + Polynomial Decay.

    Used in nnU-Net and in many medical imaging methods.
    """
    
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 warmup_epochs: int,
                 total_epochs: int,
                 base_lr: float,
                 power: float = 0.9,
                 min_lr: float = 1e-6):
        """
        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of warmup epochs
            total_epochs: Total number of epochs
            base_lr: Base learning rate
            power: Exponent for the polynomial decay
            min_lr: Minimum learning rate
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.power = power
        self.min_lr = min_lr
        self.current_epoch = 0
    
    def step(self, epoch: Optional[int] = None):
        """Update the learning rate."""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        lr = self._compute_lr(self.current_epoch)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def _compute_lr(self, epoch: int) -> float:
        """Compute the LR for a specific epoch."""
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Polynomial decay
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * (1 - progress) ** self.power
            lr = max(lr, self.min_lr)
        
        return lr
    
    def get_lr(self) -> float:
        """Get the current LR."""
        return self._compute_lr(self.current_epoch)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_auto_config(dataset_type: str, 
                    root_path: str,
                    gpu_memory_gb: float = 11.0,
                    use_amp: bool = True) -> AutoConfig:
    """
    Helper function that builds the auto-config for a dataset.

    Args:
        dataset_type: 'Synapse' or 'ACDC'
        root_path: Path to the data
        gpu_memory_gb: GPU memory (GB)
        use_amp: Whether AMP is used

    Returns:
        AutoConfig object
    """
    analyzer = DatasetAnalyzer(root_path, dataset_type)
    stats = analyzer.analyze()
    
    generator = AutoConfigGenerator(gpu_memory_gb, use_amp)
    config = generator.generate(stats)
    
    return config


def apply_linear_scaling(base_lr: float, 
                        base_batch_size: int,
                        new_batch_size: int) -> float:
    """
    Apply the Linear Scaling Rule to the learning rate.

    Args:
        base_lr: Original learning rate
        base_batch_size: Original batch size
        new_batch_size: New batch size

    Returns:
        The scaled learning rate
    """
    scale = new_batch_size / base_batch_size
    return base_lr * scale


def get_scheduler(scheduler_type: str,
                  optimizer: torch.optim.Optimizer,
                  **kwargs) -> Any:
    """
    Factory function that builds an LR scheduler.

    Args:
        scheduler_type: 'warmup_cosine' or 'warmup_polynomial'
        optimizer: PyTorch optimizer
        **kwargs: Arguments forwarded to the scheduler

    Returns:
        Scheduler instance
    """
    if scheduler_type == 'warmup_cosine':
        return WarmupCosineScheduler(optimizer, **kwargs)
    elif scheduler_type == 'warmup_polynomial':
        return WarmupPolynomialScheduler(optimizer, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


# =============================================================================
# TEST CODE
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("AUTO-CONFIG MODULE TEST")
    print("=" * 60)
    
    # Test with a dummy dataset
    stats = DatasetStatistics(
        name='Synapse',
        num_samples=2212,
        num_classes=9,
        image_shape=(224, 224),
        is_3d=False
    )
    
    # Generate config
    generator = AutoConfigGenerator(gpu_memory_gb=11.0, use_amp=True)
    config = generator.generate(stats)
    
    print("\nGenerated Config:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    
    # Test schedulers
    print("\n" + "=" * 60)
    print("SCHEDULER TEST")
    print("=" * 60)
    
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Test warmup cosine
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs=10, total_epochs=100, base_lr=0.01
    )
    
    print("\nWarmup Cosine Schedule (first 20 epochs):")
    for epoch in range(20):
        lr = scheduler.step(epoch)
        print(f"  Epoch {epoch:3d}: LR = {lr:.6f}")
    
    print("\n[OK] Auto-config module test completed!")
