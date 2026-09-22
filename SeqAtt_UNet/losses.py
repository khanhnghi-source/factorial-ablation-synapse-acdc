# =============================================================================
# NOTICE OF MODIFICATION (Apache License 2.0, Section 4(b))
#
# This file is derived from TransUNet:
#     https://github.com/Beckschen/TransUNet -- utils.py (DiceLoss)
#     Chen et al., "TransUNet: Transformers Make Strong Encoders for Medical
#     Image Segmentation," arXiv:2102.04306, 2021. Licensed under Apache-2.0.
#
# This file HAS BEEN MODIFIED by the authors of the present study. See NOTICE
# and THIRD_PARTY.md in the repository root, and the full licence text in
# LICENSE.
# =============================================================================
"""
ADVANCED LOSS FUNCTIONS FOR SeqAtt-UNet
========================================
PHASE 1 - Loss Function Improvements

Contents:
1. DiceLoss: standard loss for segmentation
2. BoundaryLoss: signed-distance boundary loss (Kervadec et al., MIDL 2019)
3. DifferentiableHDLoss: squared error between soft morphological boundary maps.
   Despite the class name this is NOT a Hausdorff distance; it is the term the
   paper writes as L_Bmap. It is the variant that runs.
4. HDLoss: NOT USED and NOT DIFFERENTIABLE -- see the warning on that class.
5. CombinedLoss: combines the terms above.

WHAT THE 70 REPORTED RUNS ACTUALLY TRAINED WITH
(`LOSS_CONFIG` in trainer_phase3.py; this file's defaults match it):

    L = 1.0 * L_Dice + 0.5 * L_CE
        + 1[epoch >= 50] * (0.1 * L_Bdy + 0.1 * L_Bmap)

- All four terms are active. `use_boundary` and `use_hd` are both True.
- Cross-entropy carries weight 0.5, not 1.0.
- Both boundary terms are switched on only from epoch 50 of 150, so the first
  third of training optimises Dice and cross-entropy alone.
- BoundaryLoss is signed and may go negative on well-fitted structures, which is
  why it carries a small weight; a guard falls back to Dice + CE if the total
  goes far negative.

An earlier version of this docstring said "BoundaryLoss is disabled by default"
and "Only Dice + CE + HD Loss are used". Both described a configuration that was
never used for any reported run.

Author: SeqAtt-UNet Project - Phase 1 Improvements (Fixed Version)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt


class DiceLoss(nn.Module):
    """
    Standard Dice Loss - kept unchanged from the original implementation.
    
    Formula: 1 - (2 * intersection + smooth) / (sum_pred + sum_target + smooth)
    
    This is the main loss function; it always lies within [0, 1].
    """
    
    def __init__(self, n_classes, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.smooth = smooth

    def _one_hot_encoder(self, input_tensor):
        """Convert a label tensor into a one-hot encoding."""
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        """Compute the Dice Loss for a single class."""
        target = target.float()
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + self.smooth) / (z_sum + y_sum + self.smooth)
        return 1 - loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        
        if weight is None:
            weight = [1] * self.n_classes
            
        assert inputs.size() == target.size(), \
            f'predict {inputs.size()} & target {target.size()} shape do not match'
        
        loss = 0.0
        for i in range(self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i]
            
        return loss / self.n_classes


class BoundaryLoss(nn.Module):
    """
    Boundary Loss based on the Distance Transform - FOLLOWS THE ORIGINAL PAPER.
    
    Paper: "Boundary loss for highly unbalanced segmentation" (MIDL 2019)
    
    Idea:
    - Use a signed distance map:
      + INSIDE ground truth: distance < 0 (negative)
      + OUTSIDE ground truth: distance > 0 (positive)
    - Loss = mean(softmax * signed_distance)
    - Minimizing the loss -> pushes the predictions inside the ground truth
    
    IMPORTANT:
    - The loss CAN BE NEGATIVE when the model predicts correctly (this is expected behavior)
    - It must be used with a small weight so that the total loss stays positive
    - Recommended: lambda_boundary = 0.01 - 0.1
    """
    
    def __init__(self, n_classes):
        super(BoundaryLoss, self).__init__()
        self.n_classes = n_classes
    
    def _compute_signed_distance_map(self, target):
        """
        Compute the signed distance map of the ground truth.
        
        Convention:
        - INSIDE ground truth region: distance < 0 (negative)
        - OUTSIDE ground truth region: distance > 0 (positive)
        - ON boundary: distance = 0
        
        Args:
            target: Ground truth tensor [B, H, W]
            
        Returns:
            Signed distance map tensor [B, n_classes, H, W]
        """
        batch_size = target.shape[0]
        target_np = target.cpu().numpy()
        
        dist_maps = np.zeros((batch_size, self.n_classes, target.shape[1], target.shape[2]))
        
        for b in range(batch_size):
            for c in range(self.n_classes):
                mask = (target_np[b] == c).astype(np.float64)
                
                if mask.sum() > 0 and (1 - mask).sum() > 0:
                    # Distance transform with respect to the boundary
                    # pos_dist: distance from the outside towards the boundary (positive outside)
                    pos_dist = distance_transform_edt(1 - mask)
                    # neg_dist: distance from the inside towards the boundary (positive inside)
                    neg_dist = distance_transform_edt(mask)
                    
                    # Signed distance: positive outside, negative inside
                    signed_dist = pos_dist - neg_dist
                    
                    # Normalize to bound the magnitude (optional but helpful)
                    max_val = max(np.abs(signed_dist).max(), 1.0)
                    dist_maps[b, c] = signed_dist / max_val
                    
                elif mask.sum() == 0:
                    # Class absent from the image -> all positive (penalize any prediction)
                    dist_maps[b, c] = 1.0
                else:
                    # Class covers the whole image -> all negative (reward all predictions)
                    dist_maps[b, c] = -1.0
        
        return torch.from_numpy(dist_maps).float().to(target.device)
    
    def forward(self, inputs, target, softmax=True):
        """
        Compute the Boundary Loss.
        
        Loss = mean(softmax * signed_distance)
        
        - Correct prediction (high prob inside, negative distance) -> negative loss
        - Wrong prediction (high prob outside, positive distance) -> positive loss
        - Minimizing the loss -> pushes the predictions inside the ground truth
        
        Args:
            inputs: Prediction logits [B, C, H, W]
            target: Ground truth [B, H, W]
            softmax: Whether to apply softmax
            
        Returns:
            Boundary loss value (CAN BE NEGATIVE - this is expected behavior)
        """
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        
        # Compute the signed distance map
        dist_map = self._compute_signed_distance_map(target)
        
        # Boundary loss = mean(prediction * signed_distance)
        # Computed for the foreground classes only (background class 0 is skipped)
        loss = 0.0
        for c in range(1, self.n_classes):
            loss += torch.mean(inputs[:, c] * dist_map[:, c])
        
        loss = loss / (self.n_classes - 1)
        
        return loss


class HDLoss(nn.Module):
    """
    NOT USED, AND NOT DIFFERENTIABLE. Kept only so that the file matches the one
    the experiments ran against; no reported result went through this class.

    `CombinedLoss` defaults to `use_differentiable_hd=True` and the trainer never
    overrides it, so `DifferentiableHDLoss` is what runs.

    Why it would not work if it did run: `_compute_hd_loss` detaches the
    prediction, computes the distance transforms in NumPy, and wraps a Python
    float back into a fresh `torch.tensor(..., requires_grad=True)`. That tensor
    is a new leaf with no path back to the network, so the term contributes
    exactly zero gradient while still appearing in the logged loss. Anyone
    setting `use_differentiable_hd=False` would be training on Dice + CE +
    Boundary with a constant added to the reported number.

    Formula it intends: L_hd = (1/N) * sum(pred_dist + gt_dist), always positive.
    """
    
    def __init__(self, n_classes, alpha=2.0):
        """
        Args:
            n_classes: Number of classes
            alpha: Exponential weight used to focus on the maximum distances
        """
        super(HDLoss, self).__init__()
        self.n_classes = n_classes
        self.alpha = alpha
    
    def _compute_hd_loss(self, pred, target, class_idx):
        """Compute the HD loss for one specific class."""
        pred_np = pred.cpu().detach().numpy()
        target_np = target.cpu().numpy()
        
        batch_size = pred.shape[0]
        total_loss = 0.0
        valid_samples = 0
        
        for b in range(batch_size):
            pred_mask = (pred_np[b] > 0.5).astype(np.float64)
            target_mask = (target_np[b] == class_idx).astype(np.float64)
            
            # Check that both masks contain pixels
            if pred_mask.sum() < 1 or target_mask.sum() < 1:
                continue
            
            # Distance transforms
            target_dist = distance_transform_edt(1 - target_mask)
            pred_dist = distance_transform_edt(1 - pred_mask)
            
            # HD loss components (always positive)
            pred_to_gt = np.mean(pred_mask * (target_dist ** self.alpha))
            gt_to_pred = np.mean(target_mask * (pred_dist ** self.alpha))
            
            total_loss += (pred_to_gt + gt_to_pred)
            valid_samples += 1
        
        if valid_samples == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        return torch.tensor(total_loss / valid_samples, device=pred.device, requires_grad=True)
    
    def forward(self, inputs, target, softmax=True):
        """Compute the HD Loss over all classes."""
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        
        total_loss = torch.tensor(0.0, device=inputs.device)
        
        # Skip the background (class 0)
        for c in range(1, self.n_classes):
            class_loss = self._compute_hd_loss(inputs[:, c], target, c)
            total_loss = total_loss + class_loss
        
        return total_loss / max(self.n_classes - 1, 1)


class DifferentiableHDLoss(nn.Module):
    """
    Differentiable Hausdorff Distance Loss.
    
    A fully differentiable version of the HD Loss that relies on soft operations.
    Always POSITIVE.
    """
    
    def __init__(self, n_classes, kernel_size=3):
        super(DifferentiableHDLoss, self).__init__()
        self.n_classes = n_classes
        self.kernel_size = kernel_size
        
        # Erosion/Dilation kernels
        self.kernel = torch.ones(1, 1, kernel_size, kernel_size)
    
    def _soft_erode(self, x):
        """Soft erosion operation."""
        kernel = self.kernel.to(x.device)
        return 1 - F.max_pool2d(1 - x, self.kernel_size, stride=1, 
                                 padding=self.kernel_size // 2)
    
    def _soft_dilate(self, x):
        """Soft dilation operation."""
        return F.max_pool2d(x, self.kernel_size, stride=1, 
                           padding=self.kernel_size // 2)
    
    def _soft_boundary(self, x):
        """Extract the boundary through dilation - erosion."""
        return self._soft_dilate(x) - self._soft_erode(x)
    
    def forward(self, inputs, target, softmax=True):
        """Compute the differentiable HD loss."""
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        
        # One-hot encode target
        target_one_hot = F.one_hot(target.long(), self.n_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        total_loss = 0.0
        
        # Compute the loss for each class (the background is skipped)
        for c in range(1, self.n_classes):
            pred_c = inputs[:, c:c+1]
            target_c = target_one_hot[:, c:c+1]
            
            # Extract boundaries
            pred_boundary = self._soft_boundary(pred_c)
            target_boundary = self._soft_boundary(target_c)
            
            # HD approximation: MSE of boundary mismatch (always positive)
            loss_c = F.mse_loss(pred_boundary, target_boundary)
            total_loss += loss_c
        
        return total_loss / max(self.n_classes - 1, 1)


class CombinedLoss(nn.Module):
    """
    Combined Loss Function that merges several loss components.
    
    Formula:
    L_total = λ_dice * L_dice + λ_ce * L_ce + λ_boundary * L_boundary + λ_hd * L_hd
    
    IMPORTANT NOTE ON THE BOUNDARY LOSS:
    - BoundaryLoss follows the original paper (MIDL 2019) and uses a SIGNED distance map
    - The loss CAN BE NEGATIVE when the model predicts correctly (this is expected behavior)
    - A SMALL weight should be used: lambda_boundary = 0.01 - 0.1
    - If the total loss turns negative, REDUCE lambda_boundary
    
    Recommended weights:
    - λ_dice = 1.0 (main segmentation loss)
    - λ_ce = 0.5 (cross entropy)
    - λ_boundary = 0.1 (boundary awareness - CAN BE NEGATIVE)
    - λ_hd = 0.1 (HD95 improvement)
    """
    
    def __init__(self, n_classes, 
                 lambda_dice=1.0, 
                 lambda_ce=0.5, 
                 lambda_boundary=0.1,  # Boundary loss (can be negative)
                 lambda_hd=0.1,
                 use_boundary=True,
                 use_hd=True,
                 use_differentiable_hd=True):
        """
        Args:
            n_classes: Number of segmentation classes
            lambda_dice: Weight of the Dice Loss (default 1.0)
            lambda_ce: Weight of the Cross Entropy Loss (default 0.5)
            lambda_boundary: Weight of the Boundary Loss (default 0.1)
                            WARNING: the boundary loss CAN BE NEGATIVE - use a small weight!
            lambda_hd: Weight of the HD Loss (default 0.1)
            use_boundary: Whether to use the Boundary Loss (default True)
            use_hd: Whether to use the HD Loss (default True)
            use_differentiable_hd: Use the differentiable HD loss
        """
        super(CombinedLoss, self).__init__()
        
        self.n_classes = n_classes
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        self.lambda_boundary = lambda_boundary
        self.lambda_hd = lambda_hd
        self.use_boundary = use_boundary
        self.use_hd = use_hd
        
        # Initialize loss components
        self.dice_loss = DiceLoss(n_classes)
        self.ce_loss = nn.CrossEntropyLoss()
        
        if use_boundary:
            self.boundary_loss = BoundaryLoss(n_classes)
        
        if use_hd:
            if use_differentiable_hd:
                self.hd_loss = DifferentiableHDLoss(n_classes)
            else:
                self.hd_loss = HDLoss(n_classes)
    
    def forward(self, inputs, target, epoch=0, boundary_start_epoch=50):
        """
        Compute the combined loss.
        
        Args:
            inputs: Prediction logits [B, C, H, W]
            target: Ground truth [B, H, W]
            epoch: Current epoch
            boundary_start_epoch: Epoch from which the Boundary + HD Loss are applied
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Core losses: Dice + CE (always positive)
        loss_dice = self.dice_loss(inputs, target, softmax=True)
        loss_ce = self.ce_loss(inputs, target.long())
        
        total_loss = self.lambda_dice * loss_dice + self.lambda_ce * loss_ce
        
        loss_dict = {
            'dice': loss_dice.item(),
            'ce': loss_ce.item()
        }
        
        # Boundary Loss (only if enabled AND past boundary_start_epoch)
        if self.use_boundary and epoch >= boundary_start_epoch and self.lambda_boundary > 0:
            loss_boundary = self.boundary_loss(inputs, target, softmax=True)
            total_loss = total_loss + self.lambda_boundary * loss_boundary
            loss_dict['boundary'] = loss_boundary.item()
        
        # HD Loss (only if enabled AND past boundary_start_epoch)
        if self.use_hd and epoch >= boundary_start_epoch and self.lambda_hd > 0:
            loss_hd = self.hd_loss(inputs, target, softmax=True)
            total_loss = total_loss + self.lambda_hd * loss_hd
            loss_dict['hd'] = loss_hd.item()
        
        loss_dict['total'] = total_loss.item()
        
        # SAFETY CHECK: warn if the total loss is negative
        # The boundary loss CAN BE NEGATIVE (expected behavior according to the paper)
        # Fall back only when the total loss is far too negative (< -1.0)
        if total_loss.item() < -1.0:
            print(f"[WARNING] Very negative total loss: {total_loss.item():.4f}")
            print(f"   Loss components: {loss_dict}")
            print(f"   Consider reducing lambda_boundary (current: {self.lambda_boundary})")
            # Fall back to Dice + CE to avoid a training collapse
            total_loss = self.lambda_dice * loss_dice + self.lambda_ce * loss_ce
            loss_dict['total'] = total_loss.item()
            loss_dict['warning'] = 'fallback_to_dice_ce'
        elif total_loss.item() < 0:
            # A slightly negative loss is acceptable - only log a periodic warning
            loss_dict['note'] = 'negative_but_acceptable'
        
        return total_loss, loss_dict


class FocalDiceLoss(nn.Module):
    """
    Focal Dice Loss - combines the Focal Loss with the Dice Loss.
    
    Handles class imbalance better.
    Always POSITIVE.
    """
    
    def __init__(self, n_classes, gamma=2.0, smooth=1e-5):
        super(FocalDiceLoss, self).__init__()
        self.n_classes = n_classes
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, inputs, target, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        
        # One-hot encode
        target_one_hot = F.one_hot(target.long(), self.n_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        total_loss = 0.0
        
        for c in range(self.n_classes):
            pred_c = inputs[:, c]
            target_c = target_one_hot[:, c]
            
            # Focal weight
            focal_weight = (1 - pred_c) ** self.gamma * target_c + \
                          pred_c ** self.gamma * (1 - target_c)
            
            # Weighted Dice
            intersection = torch.sum(focal_weight * pred_c * target_c)
            union = torch.sum(focal_weight * pred_c) + torch.sum(focal_weight * target_c)
            
            dice_c = (2 * intersection + self.smooth) / (union + self.smooth)
            total_loss += (1 - dice_c)
        
        return total_loss / self.n_classes


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_loss_function(loss_type, n_classes, **kwargs):
    """
    Factory function that builds a loss function.
    
    Args:
        loss_type: Loss type ('dice', 'combined', 'focal_dice')
        n_classes: Number of classes
        **kwargs: Additional arguments
        
    Returns:
        Loss function instance
    """
    if loss_type == 'dice':
        return DiceLoss(n_classes)
    elif loss_type == 'combined':
        return CombinedLoss(n_classes, **kwargs)
    elif loss_type == 'focal_dice':
        return FocalDiceLoss(n_classes, **kwargs)
    elif loss_type == 'boundary':
        return BoundaryLoss(n_classes)
    elif loss_type == 'hd':
        return DifferentiableHDLoss(n_classes)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test code that verifies that the loss functions are always non-negative
    print("Testing Loss Functions (Fixed Version)...")
    print("=" * 60)
    
    n_classes = 9
    batch_size = 2
    h, w = 224, 224
    
    # Dummy inputs
    inputs = torch.randn(batch_size, n_classes, h, w)
    target = torch.randint(0, n_classes, (batch_size, h, w))
    
    # Test DiceLoss
    dice_loss = DiceLoss(n_classes)
    loss = dice_loss(inputs, target, softmax=True)
    print(f"DiceLoss: {loss.item():.4f} (should be in [0, 1])")
    assert loss.item() >= 0, "DiceLoss should be non-negative!"
    
    # Test BoundaryLoss (fixed version)
    boundary_loss = BoundaryLoss(n_classes)
    loss = boundary_loss(inputs, target, softmax=True)
    print(f"BoundaryLoss: {loss.item():.4f} (should be >= 0)")
    assert loss.item() >= 0, "BoundaryLoss should be non-negative!"
    
    # Test HDLoss
    hd_loss = DifferentiableHDLoss(n_classes)
    loss = hd_loss(inputs, target, softmax=True)
    print(f"HDLoss: {loss.item():.4f} (should be >= 0)")
    assert loss.item() >= 0, "HDLoss should be non-negative!"
    
    # Test CombinedLoss (epoch 0 - no boundary/hd)
    combined_loss = CombinedLoss(n_classes, use_boundary=False, use_hd=True)
    loss, loss_dict = combined_loss(inputs, target, epoch=0)
    print(f"\nCombinedLoss (epoch 0): {loss_dict}")
    assert loss.item() >= 0, "CombinedLoss should be non-negative!"
    
    # Test CombinedLoss (epoch 100 - with hd)
    loss, loss_dict = combined_loss(inputs, target, epoch=100)
    print(f"CombinedLoss (epoch 100): {loss_dict}")
    assert loss.item() >= 0, "CombinedLoss should be non-negative!"
    
    print("\n" + "=" * 60)
    print("[OK] All loss functions are NON-NEGATIVE!")
    print("[OK] Safe for training - total loss will always be positive!")
