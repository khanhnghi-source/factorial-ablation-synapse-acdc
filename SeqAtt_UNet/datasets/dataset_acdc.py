# =============================================================================
# NOTICE OF MODIFICATION (Apache License 2.0, Section 4(b))
#
# This file is derived from TransUNet:
#     https://github.com/Beckschen/TransUNet -- datasets/dataset_synapse.py
#     Chen et al., "TransUNet: Transformers Make Strong Encoders for Medical
#     Image Segmentation," arXiv:2102.04306, 2021. Licensed under Apache-2.0.
#
# This file HAS BEEN MODIFIED by the authors of the present study. See NOTICE
# and THIRD_PARTY.md in the repository root, and the full licence text in
# LICENSE.
# =============================================================================
"""
ACDC Dataset Loader cho SeqAtt_UNet
====================================
ACDC (Automated Cardiac Diagnosis Challenge) dataset cho cardiac MRI segmentation.

Dataset structure:
- 100 patients (patient001 - patient100)
- Test set: patient001 - patient020  
- Validation set: patient021 - patient030
- Training set: patient031 - patient100
- 4 classes: background (0), RV (1), MYO (2), LV (3)

Data format:
- Training: slices (ACDC_training_slices/*.h5)
- Validation/Test: volumes (ACDC_training_volumes/*.h5)

Author: SeqAtt_UNet Project
"""

import os
import re
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    """Random rotation (0, 90, 180, 270 degrees) and random flip."""
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    """Random rotation within -20 to 20 degrees."""
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    """
    Random data augmentation generator for ACDC dataset.
    Applies random rotation, flip, and resize to target output size.
    """
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        # Random augmentation
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        
        # Resize if needed
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            # Use order=3 (cubic) for image, order=0 (nearest) for label
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        # Convert to tensor
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        
        sample = {'image': image, 'label': label.long()}
        return sample


class ACDC_dataset(Dataset):
    """
    ACDC Dataset class for cardiac MRI segmentation.
    
    Args:
        base_dir: Root directory containing ACDC data
        split: 'train', 'val', or 'test'
        transform: Optional transform to apply
        list_dir: Not used (kept for compatibility), splits are determined automatically
    
    Note:
        - Training uses 2D slices from ACDC_training_slices/
        - Validation/Testing uses 3D volumes from ACDC_training_volumes/
    """
    
    def __init__(self, base_dir=None, split='train', list_dir=None, transform=None):
        self._base_dir = base_dir
        self.split = split
        self.transform = transform
        self.sample_list = []
        
        # Get patient IDs for each split
        train_ids, val_ids, test_ids = self._get_patient_ids()
        
        if 'train' in self.split:
            # Training: use individual slices
            slices_dir = os.path.join(self._base_dir, "ACDC_training_slices")
            if os.path.exists(slices_dir):
                all_slices = os.listdir(slices_dir)
                for patient_id in train_ids:
                    # Match slices for this patient (e.g., patient031_frame01_slice_0.h5)
                    patient_slices = [s for s in all_slices if re.match(f'{patient_id}.*', s)]
                    self.sample_list.extend(patient_slices)
        
        elif 'val' in self.split:
            # Validation: use volumes
            volumes_dir = os.path.join(self._base_dir, "ACDC_training_volumes")
            if os.path.exists(volumes_dir):
                all_volumes = os.listdir(volumes_dir)
                for patient_id in val_ids:
                    patient_volumes = [v for v in all_volumes if re.match(f'{patient_id}.*', v)]
                    self.sample_list.extend(patient_volumes)
        
        elif 'test' in self.split:
            # Testing: use volumes
            volumes_dir = os.path.join(self._base_dir, "ACDC_training_volumes")
            if os.path.exists(volumes_dir):
                all_volumes = os.listdir(volumes_dir)
                for patient_id in test_ids:
                    patient_volumes = [v for v in all_volumes if re.match(f'{patient_id}.*', v)]
                    self.sample_list.extend(patient_volumes)
        
        print(f"ACDC Dataset [{self.split}]: loaded {len(self.sample_list)} samples")

    def _get_patient_ids(self):
        """
        Define patient ID splits for ACDC dataset.
        
        Returns:
            Tuple of (training_ids, validation_ids, testing_ids)
        """
        # All 100 patients
        all_cases = [f"patient{i:03d}" for i in range(1, 101)]
        
        # Standard ACDC split
        testing_set = [f"patient{i:03d}" for i in range(1, 21)]      # patient001-020
        validation_set = [f"patient{i:03d}" for i in range(21, 31)]  # patient021-030
        training_set = [p for p in all_cases if p not in testing_set + validation_set]  # patient031-100
        
        return training_set, validation_set, testing_set

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        
        if 'train' in self.split:
            # Load individual slice for training
            filepath = os.path.join(self._base_dir, "ACDC_training_slices", case)
            with h5py.File(filepath, 'r') as h5f:
                image = h5f['image'][:]
                label = h5f['label'][:]
            
            sample = {'image': image, 'label': label}
            
            # Apply transforms (augmentation + resize)
            if self.transform:
                sample = self.transform(sample)
        else:
            # Load full volume for validation/testing
            filepath = os.path.join(self._base_dir, "ACDC_training_volumes", case)
            with h5py.File(filepath, 'r') as h5f:
                image = h5f['image'][:]
                label = h5f['label'][:]
            
            sample = {'image': image, 'label': label}
        
        sample['idx'] = idx
        sample['case_name'] = case.replace('.h5', '')
        
        return sample


# Alias for compatibility with different naming conventions
ACDCDataset = ACDC_dataset
BaseDataSets = ACDC_dataset
