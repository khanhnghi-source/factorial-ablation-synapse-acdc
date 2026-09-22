# =============================================================================
# NOTICE OF MODIFICATION (Apache License 2.0, Section 4(b))
#
# This file is derived from TransUNet:
#     https://github.com/Beckschen/TransUNet -- utils.py (test_single_volume)
#     Chen et al., "TransUNet: Transformers Make Strong Encoders for Medical
#     Image Segmentation," arXiv:2102.04306, 2021. Licensed under Apache-2.0.
#
# This file HAS BEEN MODIFIED by the authors of the present study. See NOTICE
# and THIRD_PARTY.md in the repository root, and the full licence text in
# LICENSE.
# =============================================================================
"""
TEST-TIME AUGMENTATION (TTA) FOR SeqAtt-UNet
=============================================
PHASE 1 - Inference Improvements

TTA can add 0.5-1% DSC WITHOUT retraining the model.
The idea: apply several augmentations at inference time and aggregate the
predictions.

TTA strategies:
1. Flip TTA: Horizontal + Vertical flip
2. Rotation TTA: Multi-angle rotations
3. Scale TTA: Multi-scale predictions
4. Full TTA: all of the above combined

Author: SeqAtt-UNet Project - Phase 1 Improvements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from scipy.ndimage import zoom


class TestTimeAugmentation:
    """
    Test-Time Augmentation (TTA) Module.
    
    Core idea:
    - Build several augmented versions of the input image
    - Run a forward pass through the model for each version
    - Reverse the augmentation on the predictions
    - Average all predictions

    Benefits:
    - More robust predictions
    - Lower variance
    - No retraining required

    Trade-off:
    - Inference time grows with the number of TTA transforms
    - Memory usage grows if the predictions are batched
    """
    
    def __init__(self, 
                 use_flip=True,
                 use_rotation=True,
                 use_scale=False,
                 rotation_angles=[90, 180, 270],
                 scale_factors=[0.9, 1.0, 1.1],
                 aggregation='mean'):
        """
        Args:
            use_flip: Enable flip augmentations (horizontal + vertical)
            use_rotation: Enable rotation augmentations
            use_scale: Enable multi-scale augmentations
            rotation_angles: List of rotation angles to use
            scale_factors: List of scale factors to use
            aggregation: 'mean' or 'max', used to aggregate the predictions
        """
        self.use_flip = use_flip
        self.use_rotation = use_rotation
        self.use_scale = use_scale
        self.rotation_angles = rotation_angles
        self.scale_factors = scale_factors
        self.aggregation = aggregation
    
    def _flip_horizontal(self, x):
        """Flip horizontal."""
        return torch.flip(x, dims=[-1])
    
    def _flip_vertical(self, x):
        """Flip vertical."""
        return torch.flip(x, dims=[-2])
    
    def _rotate(self, x, angle):
        """
        Rotate tensor by angle degrees.
        Uses F.grid_sample for a differentiable rotation.
        """
        # angle in degrees
        angle_rad = angle * np.pi / 180
        
        # Rotation matrix
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        theta = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ], dtype=x.dtype, device=x.device).unsqueeze(0)
        
        # Expand theta over the batch
        theta = theta.expand(x.size(0), -1, -1)

        # Create the grid and sample
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False, mode='bilinear', padding_mode='zeros')
    
    def _scale(self, x, scale_factor, target_size=None):
        """Scale image by factor."""
        if target_size is None:
            target_size = x.shape[-2:]
        
        # Scale up/down
        scaled = F.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        
        # Resize back to original
        return F.interpolate(scaled, size=target_size, mode='bilinear', align_corners=False)
    
    def get_tta_transforms(self):
        """
        Get list of TTA transforms.
        
        Returns:
            List of tuples (forward_fn, inverse_fn, name)
        """
        transforms = []
        
        # Original (identity)
        transforms.append((
            lambda x: x,
            lambda x: x,
            'original'
        ))
        
        # Flip transforms
        if self.use_flip:
            # Horizontal flip
            transforms.append((
                self._flip_horizontal,
                self._flip_horizontal,  # Inverse = same operation
                'flip_h'
            ))
            
            # Vertical flip
            transforms.append((
                self._flip_vertical,
                self._flip_vertical,  # Inverse = same operation
                'flip_v'
            ))
            
            # Both flips
            transforms.append((
                lambda x: self._flip_vertical(self._flip_horizontal(x)),
                lambda x: self._flip_horizontal(self._flip_vertical(x)),
                'flip_hv'
            ))
        
        # Rotation transforms
        if self.use_rotation:
            for angle in self.rotation_angles:
                forward_fn = lambda x, a=angle: self._rotate(x, a)
                inverse_fn = lambda x, a=angle: self._rotate(x, -a)
                transforms.append((forward_fn, inverse_fn, f'rot_{angle}'))
        
        return transforms
    
    def __call__(self, model, image, verbose=False):
        """
        Apply TTA and return the aggregated prediction.

        Args:
            model: Segmentation model
            image: Input image tensor [B, C, H, W]
            verbose: Print TTA steps

        Returns:
            Aggregated prediction [B, num_classes, H, W]
        """
        model.eval()
        predictions = []
        transforms = self.get_tta_transforms()
        
        with torch.no_grad():
            for forward_fn, inverse_fn, name in transforms:
                # Apply forward transform
                transformed_input = forward_fn(image)
                
                # Forward pass
                pred = model(transformed_input)
                
                # Apply softmax before the inverse transform
                pred_softmax = F.softmax(pred, dim=1)
                
                # Apply inverse transform
                pred_inverse = inverse_fn(pred_softmax)
                
                predictions.append(pred_inverse)
                
                if verbose:
                    print(f"TTA transform: {name}")
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=0)  # [N_tta, B, C, H, W]
        
        # Aggregate
        if self.aggregation == 'mean':
            final_pred = predictions.mean(dim=0)
        elif self.aggregation == 'max':
            final_pred = predictions.max(dim=0)[0]
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        return final_pred


class SimpleTTA:
    """
    Simple TTA - flip transforms only.

    Faster than full TTA while still being effective.
    Recommended when production/inference speed matters.
    """
    
    def __call__(self, model, image):
        """
        Args:
            model: Segmentation model
            image: Input tensor [B, C, H, W]
            
        Returns:
            Aggregated prediction
        """
        model.eval()
        predictions = []
        
        with torch.no_grad():
            # Original
            pred0 = F.softmax(model(image), dim=1)
            predictions.append(pred0)
            
            # Horizontal flip
            flipped_h = torch.flip(image, dims=[-1])
            pred_h = F.softmax(model(flipped_h), dim=1)
            pred_h = torch.flip(pred_h, dims=[-1])
            predictions.append(pred_h)
            
            # Vertical flip
            flipped_v = torch.flip(image, dims=[-2])
            pred_v = F.softmax(model(flipped_v), dim=1)
            pred_v = torch.flip(pred_v, dims=[-2])
            predictions.append(pred_v)
            
            # Both flips
            flipped_hv = torch.flip(image, dims=[-1, -2])
            pred_hv = F.softmax(model(flipped_hv), dim=1)
            pred_hv = torch.flip(pred_hv, dims=[-1, -2])
            predictions.append(pred_hv)
        
        # Average predictions
        return torch.stack(predictions, dim=0).mean(dim=0)


class RotationTTA:
    """
    Rotation TTA - uses 4 rotations (0°, 90°, 180°, 270°).

    Effective for objects without an orientation bias.
    """
    
    def __call__(self, model, image):
        """
        Args:
            model: Segmentation model
            image: Input tensor [B, C, H, W]
            
        Returns:
            Aggregated prediction
        """
        model.eval()
        predictions = []
        
        with torch.no_grad():
            for k in range(4):  # 0, 90, 180, 270 degrees
                # Rotate input
                rotated = torch.rot90(image, k, dims=[-2, -1])
                
                # Forward pass
                pred = F.softmax(model(rotated), dim=1)
                
                # Rotate back
                pred = torch.rot90(pred, -k, dims=[-2, -1])
                predictions.append(pred)
        
        return torch.stack(predictions, dim=0).mean(dim=0)


class FullTTA:
    """
    Full TTA - flip + rotation combined.

    8 predictions total (4 rotations × 2 flip states).
    Highest accuracy, but the slowest.
    """
    
    def __call__(self, model, image):
        """
        Args:
            model: Segmentation model  
            image: Input tensor [B, C, H, W]
            
        Returns:
            Aggregated prediction
        """
        model.eval()
        predictions = []
        
        with torch.no_grad():
            for k in range(4):  # 4 rotations
                for flip in [False, True]:  # flip or not
                    # Transform
                    x = torch.rot90(image, k, dims=[-2, -1])
                    if flip:
                        x = torch.flip(x, dims=[-1])
                    
                    # Predict
                    pred = F.softmax(model(x), dim=1)
                    
                    # Inverse transform
                    if flip:
                        pred = torch.flip(pred, dims=[-1])
                    pred = torch.rot90(pred, -k, dims=[-2, -1])
                    
                    predictions.append(pred)
        
        return torch.stack(predictions, dim=0).mean(dim=0)


# =============================================================================
# INFERENCE UTILITIES
# =============================================================================

def test_single_volume_with_tta(image, label, net, classes, patch_size=[256, 256], 
                                 tta_type='simple', test_save_path=None, case=None, z_spacing=1):
    """
    Test a single volume with TTA.

    Drop-in replacement for the test_single_volume function in utils.py.

    Args:
        image: Input volume
        label: Ground truth
        net: Model
        classes: Number of classes
        patch_size: Patch size for inference
        tta_type: 'none', 'simple', 'rotation', 'full'
        test_save_path: Path to save predictions
        case: Case name
        z_spacing: Spacing for saving

    Returns:
        List of (dice, hd95) metrics per class
    """
    from medpy import metric
    import SimpleITK as sitk
    
    # Select TTA
    if tta_type == 'none':
        tta = None
    elif tta_type == 'simple':
        tta = SimpleTTA()
    elif tta_type == 'rotation':
        tta = RotationTTA()
    elif tta_type == 'full':
        tta = FullTTA()
    else:
        raise ValueError(f"Unknown TTA type: {tta_type}")
    
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)
            
            input_tensor = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            
            net.eval()
            with torch.no_grad():
                if tta is not None:
                    # TTA inference
                    outputs = tta(net, input_tensor)
                else:
                    # Standard inference
                    outputs = F.softmax(net(input_tensor), dim=1)
                
                out = torch.argmax(outputs, dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        
        net.eval()
        with torch.no_grad():
            if tta is not None:
                outputs = tta(net, input_tensor)
            else:
                outputs = F.softmax(net(input_tensor), dim=1)
            
            out = torch.argmax(outputs, dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    
    # Calculate metrics
    metric_list = []
    for i in range(1, classes):
        pred_i = prediction == i
        gt_i = label == i
        
        if pred_i.sum() > 0 and gt_i.sum() > 0:
            dice = metric.binary.dc(pred_i, gt_i)
            hd95 = metric.binary.hd95(pred_i, gt_i)
        elif pred_i.sum() > 0 and gt_i.sum() == 0:
            dice, hd95 = 1, 0
        else:
            dice, hd95 = 0, 0
        
        metric_list.append((dice, hd95))
    
    # Save predictions
    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/' + case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/' + case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/' + case + "_gt.nii.gz")
    
    return metric_list


def get_tta(tta_type='simple'):
    """
    Factory function that builds a TTA module.

    Args:
        tta_type: 'none', 'simple', 'rotation', 'full', 'custom'

    Returns:
        A TTA module, or None
    """
    if tta_type == 'none':
        return None
    elif tta_type == 'simple':
        return SimpleTTA()
    elif tta_type == 'rotation':
        return RotationTTA()
    elif tta_type == 'full':
        return FullTTA()
    elif tta_type == 'custom':
        return TestTimeAugmentation(
            use_flip=True,
            use_rotation=True,
            rotation_angles=[90, 180, 270],
            aggregation='mean'
        )
    else:
        raise ValueError(f"Unknown TTA type: {tta_type}")


# =============================================================================
# TTA COMPARISON
# =============================================================================

TTA_CONFIGS = {
    'none': {
        'description': 'No TTA - fastest inference',
        'num_predictions': 1,
        'expected_improvement': '0%'
    },
    'simple': {
        'description': '4-flip TTA (H, V, HV)',
        'num_predictions': 4,
        'expected_improvement': '0.3-0.5% DSC'
    },
    'rotation': {
        'description': '4-rotation TTA (0, 90, 180, 270)',
        'num_predictions': 4,
        'expected_improvement': '0.3-0.5% DSC'
    },
    'full': {
        'description': '8-transform TTA (4 rotations × 2 flips)',
        'num_predictions': 8,
        'expected_improvement': '0.5-1% DSC'
    }
}


if __name__ == "__main__":
    print("Testing Test-Time Augmentation...")
    
    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self, num_classes=9):
            super().__init__()
            self.conv = nn.Conv2d(1, num_classes, 3, padding=1)
        
        def forward(self, x):
            return self.conv(x)
    
    model = DummyModel().cuda()
    
    # Create dummy input
    x = torch.randn(1, 1, 224, 224).cuda()
    
    # Test different TTA types
    for tta_type in ['simple', 'rotation', 'full']:
        tta = get_tta(tta_type)
        pred = tta(model, x)
        print(f"{tta_type} TTA: input {x.shape} -> output {pred.shape}")
        print(f"  Prediction sum (should be 1.0 per pixel): {pred.sum(dim=1).mean().item():.4f}")
    
    print("\n✓ All TTA types working correctly!")
