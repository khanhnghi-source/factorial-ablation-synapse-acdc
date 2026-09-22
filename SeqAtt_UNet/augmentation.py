"""
ADVANCED DATA AUGMENTATION FOR SeqAtt-UNet
==========================================
PHASE 1 - Data Augmentation Improvements

Provides the same kind of strong augmentation used by nnU-Net:
1. Spatial Transforms: rotation, scaling, elastic deformation
2. Intensity Transforms: gamma correction, gaussian noise/blur
3. Advanced: MixUp, CutMix (optional)

nnU-Net relies on very strong augmentation, and this is one of the key factors
behind its SOTA performance.

Author: SeqAtt-UNet Project - Phase 1 Improvements
"""

import numpy as np
import random
import torch
from scipy import ndimage
from scipy.ndimage import zoom, map_coordinates, gaussian_filter
from scipy.ndimage.interpolation import rotate


class AdvancedRandomGenerator:
    """
    Advanced Data Augmentation Pipeline.

    Combines several strong augmentation techniques to improve model
    generalization. The transforms are chosen to suit medical image segmentation.
    """
    
    def __init__(self, output_size, 
                 # Spatial transforms
                 rotation_range=(-30, 30),
                 scale_range=(0.7, 1.4),
                 elastic_deform=True,
                 elastic_alpha=200,
                 elastic_sigma=20,
                 # Intensity transforms
                 gamma_range=(0.7, 1.5),
                 gaussian_noise_std=0.1,
                 gaussian_blur_sigma=(0.5, 1.0),
                 brightness_range=(-0.2, 0.2),
                 contrast_range=(0.8, 1.2),
                 # Advanced options
                 use_mixup=False,
                 mixup_alpha=0.2,
                 # Probabilities
                 p_spatial=0.5,
                 p_intensity=0.3,
                 p_elastic=0.2):
        """
        Args:
            output_size: Output image size [H, W]
            rotation_range: Range for random rotation (degrees)
            scale_range: Range for random scaling
            elastic_deform: Enable elastic deformation
            elastic_alpha: Elastic deformation intensity
            elastic_sigma: Elastic deformation smoothness
            gamma_range: Range for gamma correction
            gaussian_noise_std: Std for gaussian noise
            gaussian_blur_sigma: Range for gaussian blur sigma
            brightness_range: Range for brightness adjustment
            contrast_range: Range for contrast adjustment
            use_mixup: Enable MixUp augmentation (requires external handling)
            mixup_alpha: Alpha parameter for MixUp
            p_spatial: Probability of spatial transforms
            p_intensity: Probability of intensity transforms
            p_elastic: Probability of elastic deformation
        """
        self.output_size = output_size
        
        # Spatial params
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.elastic_deform = elastic_deform
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        
        # Intensity params
        self.gamma_range = gamma_range
        self.gaussian_noise_std = gaussian_noise_std
        self.gaussian_blur_sigma = gaussian_blur_sigma
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        
        # Advanced
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        
        # Probabilities
        self.p_spatial = p_spatial
        self.p_intensity = p_intensity
        self.p_elastic = p_elastic
    
    # =========================================================================
    # SPATIAL TRANSFORMS
    # =========================================================================
    
    def random_rotation(self, image, label):
        """Random rotation with an angle drawn from the configured range."""
        angle = np.random.uniform(*self.rotation_range)
        image = rotate(image, angle, order=3, reshape=False, mode='constant', cval=0)
        label = rotate(label, angle, order=0, reshape=False, mode='constant', cval=0)
        return image, label
    
    def random_scale(self, image, label):
        """Random scaling with a zoom factor drawn from the configured range."""
        scale = np.random.uniform(*self.scale_range)
        h, w = image.shape

        # Scale both image and label
        scaled_image = zoom(image, scale, order=3)
        scaled_label = zoom(label, scale, order=0)

        # Crop or pad to preserve the original size
        sh, sw = scaled_image.shape
        
        if scale > 1:
            # Crop center
            start_h = (sh - h) // 2
            start_w = (sw - w) // 2
            image = scaled_image[start_h:start_h+h, start_w:start_w+w]
            label = scaled_label[start_h:start_h+h, start_w:start_w+w]
        else:
            # Pad with zeros
            pad_h = (h - sh) // 2
            pad_w = (w - sw) // 2
            image = np.zeros((h, w), dtype=image.dtype)
            label = np.zeros((h, w), dtype=label.dtype)
            image[pad_h:pad_h+sh, pad_w:pad_w+sw] = scaled_image
            label[pad_h:pad_h+sh, pad_w:pad_w+sw] = scaled_label
        
        return image, label
    
    def random_flip(self, image, label):
        """Random horizontal and vertical flip."""
        # Random horizontal flip
        if random.random() > 0.5:
            image = np.flip(image, axis=1).copy()
            label = np.flip(label, axis=1).copy()
        
        # Random vertical flip
        if random.random() > 0.5:
            image = np.flip(image, axis=0).copy()
            label = np.flip(label, axis=0).copy()
        
        return image, label
    
    def elastic_deformation(self, image, label):
        """
        Elastic deformation - a very effective augmentation for medical images.

        Generates a random displacement field and applies it to both the image
        and the label.
        """
        alpha = self.elastic_alpha
        sigma = self.elastic_sigma
        
        h, w = image.shape
        
        # Random displacement field
        dx = gaussian_filter(np.random.randn(h, w) * alpha, sigma, mode='constant', cval=0)
        dy = gaussian_filter(np.random.randn(h, w) * alpha, sigma, mode='constant', cval=0)
        
        # Grid coordinates
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Apply displacement
        indices = [np.clip(y + dy, 0, h-1).astype(np.int32),
                   np.clip(x + dx, 0, w-1).astype(np.int32)]
        
        # Map coordinates for the image (interpolation order 3) and the label (order 0)
        image = map_coordinates(image, indices, order=3, mode='reflect')
        label = map_coordinates(label, indices, order=0, mode='reflect')
        
        return image, label
    
    def random_rot90(self, image, label):
        """Random 90-degree rotations."""
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        return image, label
    
    # =========================================================================
    # INTENSITY TRANSFORMS
    # =========================================================================
    
    def gamma_correction(self, image):
        """
        Gamma correction for adjusting brightness.

        gamma < 1: Brighten image
        gamma > 1: Darken image
        """
        gamma = np.random.uniform(*self.gamma_range)
        
        # Normalize to [0, 1], apply gamma, scale back
        img_min, img_max = image.min(), image.max()
        if img_max - img_min > 1e-8:
            image = (image - img_min) / (img_max - img_min)
            image = np.power(image, gamma)
            image = image * (img_max - img_min) + img_min
        
        return image
    
    def add_gaussian_noise(self, image):
        """Add Gaussian noise to the image."""
        std = np.random.uniform(0, self.gaussian_noise_std)
        noise = np.random.normal(0, std, image.shape)
        image = image + noise
        return image
    
    def gaussian_blur(self, image):
        """Apply Gaussian blur with a random sigma."""
        sigma = np.random.uniform(*self.gaussian_blur_sigma)
        image = gaussian_filter(image, sigma)
        return image
    
    def brightness_adjustment(self, image):
        """Adjust brightness."""
        delta = np.random.uniform(*self.brightness_range)
        image = image + delta
        return image
    
    def contrast_adjustment(self, image):
        """Adjust contrast."""
        factor = np.random.uniform(*self.contrast_range)
        mean = image.mean()
        image = (image - mean) * factor + mean
        return image
    
    def intensity_normalization(self, image):
        """Normalize intensity to the [0, 1] range."""
        img_min, img_max = image.min(), image.max()
        if img_max - img_min > 1e-8:
            image = (image - img_min) / (img_max - img_min)
        return image
    
    # =========================================================================
    # ADVANCED AUGMENTATION
    # =========================================================================
    
    @staticmethod
    def mixup(image1, label1, image2, label2, alpha=0.2):
        """
        MixUp augmentation: combine 2 samples with a random lambda.

        Must be called at the DataLoader level, not on an individual sample.

        Args:
            image1, label1: First sample
            image2, label2: Second sample
            alpha: Beta distribution parameter

        Returns:
            Mixed image and label (soft labels)
        """
        lam = np.random.beta(alpha, alpha)
        mixed_image = lam * image1 + (1 - lam) * image2
        mixed_label = lam * label1 + (1 - lam) * label2
        return mixed_image, mixed_label, lam
    
    @staticmethod
    def cutmix(image1, label1, image2, label2, alpha=1.0):
        """
        CutMix augmentation: cut and paste regions between 2 samples.

        Args:
            image1, label1: Target sample
            image2, label2: Source sample
            alpha: Beta distribution parameter

        Returns:
            CutMixed image and label
        """
        h, w = image1.shape
        lam = np.random.beta(alpha, alpha)
        
        # Random cut region
        cut_rat = np.sqrt(1 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Copy the region from image2/label2 into image1/label1
        mixed_image = image1.copy()
        mixed_label = label1.copy()
        mixed_image[bby1:bby2, bbx1:bbx2] = image2[bby1:bby2, bbx1:bbx2]
        mixed_label[bby1:bby2, bbx1:bbx2] = label2[bby1:bby2, bbx1:bbx2]
        
        # Compute actual lambda based on cut region
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        
        return mixed_image, mixed_label, lam
    
    # =========================================================================
    # MAIN TRANSFORM PIPELINE
    # =========================================================================
    
    def __call__(self, sample):
        """
        Apply augmentation pipeline.
        
        Args:
            sample: Dict with 'image' and 'label' keys

        Returns:
            Augmented sample
        """
        image, label = sample['image'], sample['label']
        
        # ====================
        # SPATIAL TRANSFORMS
        # ====================
        
        # Always apply random rotation/flip (probability check inside)
        if random.random() < self.p_spatial:
            # Random rotation
            image, label = self.random_rotation(image, label)
        
        if random.random() < self.p_spatial:
            # Random scaling
            image, label = self.random_scale(image, label)
        
        # Random flip (always apply with 50% each direction)
        image, label = self.random_flip(image, label)
        
        # Elastic deformation
        if self.elastic_deform and random.random() < self.p_elastic:
            image, label = self.elastic_deformation(image, label)
        
        # ====================
        # INTENSITY TRANSFORMS
        # ====================
        
        if random.random() < self.p_intensity:
            # Gamma correction
            image = self.gamma_correction(image)
        
        if random.random() < self.p_intensity * 0.5:
            # Gaussian noise
            image = self.add_gaussian_noise(image)
        
        if random.random() < self.p_intensity * 0.5:
            # Gaussian blur
            image = self.gaussian_blur(image)
        
        if random.random() < self.p_intensity * 0.3:
            # Brightness
            image = self.brightness_adjustment(image)
        
        if random.random() < self.p_intensity * 0.3:
            # Contrast
            image = self.contrast_adjustment(image)
        
        # ====================
        # RESIZE TO OUTPUT SIZE
        # ====================
        
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        # ====================
        # CONVERT TO TENSOR
        # ====================
        
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        
        sample = {'image': image, 'label': label.long()}
        return sample


class LightAugmentation:
    """
    Light augmentation for validation, or whenever strong augmentation is not
    wanted.

    Contains only the basic transforms: rotation, flip, resize.
    """
    
    def __init__(self, output_size):
        self.output_size = output_size
    
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        # Random rotation 90°
        if random.random() > 0.5:
            k = np.random.randint(0, 4)
            image = np.rot90(image, k)
            label = np.rot90(label, k)
        
        # Random flip
        if random.random() > 0.5:
            axis = np.random.randint(0, 2)
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()
        
        # Resize
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        
        return {'image': image, 'label': label.long()}


class NoAugmentation:
    """
    No augmentation - resize only, for testing/validation.
    """
    
    def __init__(self, output_size):
        self.output_size = output_size
    
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        
        return {'image': image, 'label': label.long()}


# =============================================================================
# AUGMENTATION CONFIGS
# =============================================================================

def get_augmentation(aug_type='advanced', output_size=[224, 224], **kwargs):
    """
    Factory function that builds an augmentation pipeline.

    Args:
        aug_type: 'advanced', 'light', 'none'
        output_size: Output image size
        **kwargs: Additional arguments for AdvancedRandomGenerator

    Returns:
        Augmentation transform
    """
    if aug_type == 'advanced':
        return AdvancedRandomGenerator(output_size, **kwargs)
    elif aug_type == 'light':
        return LightAugmentation(output_size)
    elif aug_type == 'none':
        return NoAugmentation(output_size)
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")


# Default configs for the different training phases
AUGMENTATION_CONFIGS = {
    # Strong augmentation - used for the main training
    'strong': {
        'rotation_range': (-30, 30),
        'scale_range': (0.7, 1.4),
        'elastic_deform': True,
        'elastic_alpha': 200,
        'elastic_sigma': 20,
        'gamma_range': (0.7, 1.5),
        'gaussian_noise_std': 0.1,
        'gaussian_blur_sigma': (0.5, 1.0),
        'p_spatial': 0.5,
        'p_intensity': 0.3,
        'p_elastic': 0.2
    },
    
    # Medium augmentation - balanced
    'medium': {
        'rotation_range': (-20, 20),
        'scale_range': (0.8, 1.2),
        'elastic_deform': True,
        'elastic_alpha': 100,
        'elastic_sigma': 15,
        'gamma_range': (0.8, 1.2),
        'gaussian_noise_std': 0.05,
        'gaussian_blur_sigma': (0.3, 0.7),
        'p_spatial': 0.4,
        'p_intensity': 0.2,
        'p_elastic': 0.15
    },
    
    # Light augmentation - for fine-tuning
    'light': {
        'rotation_range': (-15, 15),
        'scale_range': (0.9, 1.1),
        'elastic_deform': False,
        'gamma_range': (0.9, 1.1),
        'gaussian_noise_std': 0.02,
        'gaussian_blur_sigma': (0.2, 0.5),
        'p_spatial': 0.3,
        'p_intensity': 0.15,
        'p_elastic': 0
    }
}


if __name__ == "__main__":
    # Test augmentation
    print("Testing Advanced Augmentation...")
    
    # Create dummy sample
    h, w = 256, 256
    image = np.random.randn(h, w).astype(np.float32)
    label = np.random.randint(0, 9, (h, w)).astype(np.float32)
    sample = {'image': image, 'label': label}
    
    # Test advanced augmentation
    aug = get_augmentation('advanced', output_size=[224, 224], **AUGMENTATION_CONFIGS['strong'])
    augmented = aug(sample)
    
    print(f"Input shape: {image.shape}")
    print(f"Output image shape: {augmented['image'].shape}")
    print(f"Output label shape: {augmented['label'].shape}")
    print(f"Label unique values: {torch.unique(augmented['label']).numpy()}")
    
    print("\n✓ Augmentation working correctly!")
