"""
Dataset module for SeqAtt_UNet
==============================

Supported datasets:
- Synapse: Multi-organ abdominal CT segmentation (9 classes)
- ACDC: Cardiac MRI segmentation (4 classes)
"""

from .dataset_synapse import Synapse_dataset, RandomGenerator
from .dataset_acdc import ACDC_dataset, RandomGenerator as ACDCRandomGenerator

__all__ = [
    'Synapse_dataset', 
    'ACDC_dataset',
    'RandomGenerator',
    'ACDCRandomGenerator'
]
