# =============================================================================
# NOTICE OF MODIFICATION (Apache License 2.0, Section 4(b))
#
# This file is derived from TransUNet:
#     https://github.com/Beckschen/TransUNet -- networks/vit_seg_modeling.py (DecoderBlock, DecoderCup) and utils.py (DiceLoss)
#     Chen et al., "TransUNet: Transformers Make Strong Encoders for Medical
#     Image Segmentation," arXiv:2102.04306, 2021. Licensed under Apache-2.0.
#
# This file HAS BEEN MODIFIED by the authors of the present study. See NOTICE
# and THIRD_PARTY.md in the repository root, and the full licence text in
# LICENSE.
# =============================================================================
"""
DEEP SUPERVISION MODULE FOR SeqAtt-UNet - PHASE 3
================================================
Adds auxiliary losses at the decoder stages to improve gradient flow.

Rationale:
- Deep supervision lets gradients reach the deeper layers directly
- Mitigates the vanishing gradient problem
- Improves convergence speed and final performance

Formulation:
    L_total = sum_i (w_i * L_i)

    where:
    - L_i: Loss at output i (i=0 is the final output)
    - w_i: Weight for output i (decaying for outputs at lower resolutions)

Typical weights: [1.0, 0.4, 0.2, 0.1]

Reference:
- Lee et al., "Deeply-Supervised Nets" (AISTATS 2015)
- nnU-Net implementation

Author: SeqAtt-UNet Project - Phase 3 Improvements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Any


# =============================================================================
# DEEP SUPERVISION LOSS
# =============================================================================

class DeepSupervisionLoss(nn.Module):
    """
    Deep Supervision Loss for multi-scale outputs.

    Combines the losses from several scales with decaying weights.
    Supports several base loss types (Dice, CE, Combined).
    """
    
    def __init__(self,
                 base_loss: nn.Module,
                 weights: Tuple[float, ...] = (1.0, 0.4, 0.2, 0.1),
                 use_softmax: bool = True):
        """
        Args:
            base_loss: Base loss function (applied at every scale)
            weights: Tuple of weights, one per output scale
                     [final, aux1, aux2, aux3] - the final output has the largest weight
            use_softmax: Whether to apply softmax before computing the loss
        """
        super(DeepSupervisionLoss, self).__init__()
        
        self.base_loss = base_loss
        self.weights = weights
        self.use_softmax = use_softmax
        
        # Normalize the weights so that they sum to 1 (optional)
        weight_sum = sum(weights)
        self.normalized_weights = tuple(w / weight_sum for w in weights)
    
    def forward(self, 
                outputs: List[torch.Tensor], 
                target: torch.Tensor,
                **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the Deep Supervision Loss.

        Args:
            outputs: List of predictions at the different scales
                    [final_pred, aux_pred_1, aux_pred_2, aux_pred_3]
                    - final_pred: Full resolution (B, C, H, W)
                    - aux_pred_i: Lower resolution outputs
            target: Ground truth (B, H, W)
            **kwargs: Additional arguments passed to base_loss

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        if not isinstance(outputs, (list, tuple)):
            # With a single output there is nothing to supervise deeply
            if hasattr(self.base_loss, 'forward'):
                result = self.base_loss(outputs, target, **kwargs)
                if isinstance(result, tuple):
                    return result
                return result, {'total': result.item()}
        
        total_loss = 0.0
        loss_dict = {}
        
        # Compute the loss at every output scale
        num_outputs = len(outputs)
        
        for i, output in enumerate(outputs):
            if i >= len(self.weights):
                break
            
            weight = self.weights[i]
            
            if weight == 0:
                continue
            
            # Resize the output if needed (auxiliary outputs may have a different resolution)
            if output.shape[-2:] != target.shape[-2:]:
                output = F.interpolate(
                    output, 
                    size=target.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            # Compute the loss with the base loss function
            if hasattr(self.base_loss, 'forward'):
                loss_result = self.base_loss(output, target, **kwargs)
                
                # Handle different return types
                if isinstance(loss_result, tuple):
                    loss = loss_result[0]
                    if len(loss_result) > 1 and isinstance(loss_result[1], dict):
                        for key, val in loss_result[1].items():
                            if key != 'total':
                                loss_dict[f'scale{i}_{key}'] = val
                else:
                    loss = loss_result
            else:
                loss = self.base_loss(output, target)
            
            # Weighted sum
            total_loss = total_loss + weight * loss
            loss_dict[f'scale{i}'] = loss.item()
        
        loss_dict['total'] = total_loss.item()
        loss_dict['weights_used'] = self.weights[:num_outputs]
        
        return total_loss, loss_dict


class DeepSupervisionDiceCELoss(nn.Module):
    """
    Deep Supervision with a Dice + Cross Entropy Loss.

    This is the most common combination in medical image segmentation.
    """
    
    def __init__(self,
                 n_classes: int,
                 weights: Tuple[float, ...] = (1.0, 0.4, 0.2, 0.1),
                 dice_weight: float = 1.0,
                 ce_weight: float = 0.5,
                 smooth: float = 1e-5):
        """
        Args:
            n_classes: Number of segmentation classes
            weights: Weights for deep supervision
            dice_weight: Weight of the Dice loss
            ce_weight: Weight of the CE loss
            smooth: Smoothing factor for the Dice term
        """
        super(DeepSupervisionDiceCELoss, self).__init__()
        
        self.n_classes = n_classes
        self.ds_weights = weights
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth
        
        self.ce_loss = nn.CrossEntropyLoss()
    
    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the Dice loss for a single output."""
        pred_soft = torch.softmax(pred, dim=1)
        
        # One-hot encode target
        target_one_hot = F.one_hot(target.long(), self.n_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        # Dice per class
        total_dice = 0.0
        for c in range(self.n_classes):
            pred_c = pred_soft[:, c]
            target_c = target_one_hot[:, c]
            
            intersection = torch.sum(pred_c * target_c)
            union = torch.sum(pred_c) + torch.sum(target_c)
            
            dice_c = (2 * intersection + self.smooth) / (union + self.smooth)
            total_dice += dice_c
        
        return 1 - total_dice / self.n_classes
    
    def forward(self,
                outputs: List[torch.Tensor],
                target: torch.Tensor,
                **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the Deep Supervision Dice + CE Loss.

        Args:
            outputs: List of predictions
            target: Ground truth

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        
        total_loss = 0.0
        loss_dict = {}
        
        for i, output in enumerate(outputs):
            if i >= len(self.ds_weights):
                break
            
            weight = self.ds_weights[i]
            
            if weight == 0:
                continue
            
            # Resize if needed
            if output.shape[-2:] != target.shape[-2:]:
                output = F.interpolate(
                    output,
                    size=target.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            # Compute losses
            dice_loss = self._dice_loss(output, target)
            ce_loss = self.ce_loss(output, target.long())
            
            combined = self.dice_weight * dice_loss + self.ce_weight * ce_loss
            
            total_loss = total_loss + weight * combined
            
            loss_dict[f'scale{i}_dice'] = dice_loss.item()
            loss_dict[f'scale{i}_ce'] = ce_loss.item()
            loss_dict[f'scale{i}_combined'] = combined.item()
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


# =============================================================================
# DEEP SUPERVISION DECODER
# =============================================================================

class DeepSupervisionHead(nn.Module):
    """
    Segmentation head with support for Deep Supervision outputs.

    Creates multiple output heads at the different resolutions.
    """
    
    def __init__(self,
                 in_channels_list: List[int],
                 out_channels: int,
                 kernel_size: int = 3):
        """
        Args:
            in_channels_list: List of input channels, one per scale
                             [final_channels, aux1_channels, aux2_channels, ...]
            out_channels: Number of output channels (= num_classes)
            kernel_size: Kernel size of the convolution
        """
        super(DeepSupervisionHead, self).__init__()
        
        self.heads = nn.ModuleList()
        
        for in_ch in in_channels_list:
            head = nn.Sequential(
                nn.Conv2d(in_ch, out_channels, kernel_size=kernel_size,
                         padding=kernel_size // 2),
            )
            self.heads.append(head)
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Generate the predictions for every scale.

        Args:
            features: List of feature maps coming from the decoder

        Returns:
            List of predictions at the different scales
        """
        outputs = []
        
        for feature, head in zip(features, self.heads):
            out = head(feature)
            outputs.append(out)
        
        return outputs


class DeepSupervisionDecoder(nn.Module):
    """
    Decoder that emits Deep Supervision outputs.

    Replaces the TransUNet DecoderCup so that auxiliary outputs are supported.
    """
    
    def __init__(self,
                 config,
                 num_ds_outputs: int = 4):
        """
        Args:
            config: Model config
            num_ds_outputs: Number of deep supervision outputs
        """
        super(DeepSupervisionDecoder, self).__init__()
        
        self.config = config
        self.num_ds_outputs = num_ds_outputs
        
        head_channels = 512
        
        # Initial conv
        self.conv_more = nn.Sequential(
            nn.Conv2d(config.hidden_size, head_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True)
        )
        
        # Decoder blocks
        decoder_channels = config.decoder_channels  # (256, 128, 64, 16)
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        
        # Skip channels
        if config.n_skip != 0:
            skip_channels = list(config.skip_channels)
            for i in range(4 - config.n_skip):
                skip_channels[3 - i] = 0
        else:
            skip_channels = [0, 0, 0, 0]
        
        self.blocks = nn.ModuleList()
        for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels):
            block = DecoderBlock(in_ch, out_ch, sk_ch)
            self.blocks.append(block)
        
        # Deep supervision heads
        # Channels at each decoder stage (after each block)
        ds_channels = list(decoder_channels[:num_ds_outputs])
        self.ds_heads = DeepSupervisionHead(
            in_channels_list=ds_channels,
            out_channels=config.n_classes,
            kernel_size=3
        )
    
    def forward(self, 
                hidden_states: torch.Tensor,
                features: Optional[List[torch.Tensor]] = None,
                return_deep_supervision: bool = True
                ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Decoder forward pass with optional deep supervision outputs.

        Args:
            hidden_states: Encoder output (B, N, D)
            features: Skip connection features from the encoder
            return_deep_supervision: Whether to return the auxiliary outputs

        Returns:
            Tuple of:
            - final_output: Full resolution prediction (B, C, H, W)
            - ds_outputs: List of auxiliary outputs (if return_deep_supervision=True)
        """
        B, n_patch, hidden = hidden_states.size()
        h = w = int(np.sqrt(n_patch))
        
        # Reshape to 2D
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        
        # Collect intermediate features for deep supervision
        intermediate_features = []
        
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            
            x = decoder_block(x, skip=skip)
            
            if return_deep_supervision and i < self.num_ds_outputs:
                intermediate_features.append(x)
        
        # Generate outputs
        if return_deep_supervision and intermediate_features:
            ds_outputs = self.ds_heads(intermediate_features)
            return ds_outputs[-1], ds_outputs  # final output and all outputs
        
        return x, None


class DecoderBlock(nn.Module):
    """Decoder block with a skip connection."""
    
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int = 0):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
    
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.up(x)
        
        if skip is not None:
            # Handle size mismatch
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        
        x = self.conv1(x)
        x = self.conv2(x)
        
        return x


# =============================================================================
# INTEGRATION WRAPPER
# =============================================================================

class DeepSupervisionWrapper(nn.Module):
    """
    Wrapper that adds Deep Supervision to an existing model.

    Usage:
    ```python
    model = VisionTransformer(config, ...)
    model = DeepSupervisionWrapper(model, num_classes=9)
    
    # Training
    outputs = model(images)  # Returns list of outputs
    loss = ds_loss(outputs, labels)
    ```
    """
    
    def __init__(self,
                 model: nn.Module,
                 num_classes: int,
                 num_ds_outputs: int = 4,
                 ds_channels: Tuple[int, ...] = (256, 128, 64, 16)):
        """
        Args:
            model: Base segmentation model
            num_classes: Number of classes
            num_ds_outputs: Number of deep supervision outputs
            ds_channels: Number of channels at each decoder stage
        """
        super(DeepSupervisionWrapper, self).__init__()
        
        self.model = model
        self.num_classes = num_classes
        self.num_ds_outputs = num_ds_outputs
        
        # Auxiliary segmentation heads
        self.aux_heads = nn.ModuleList()
        
        for i, ch in enumerate(ds_channels[:num_ds_outputs - 1]):
            # Skip final output (handled by main segmentation head)
            aux_head = nn.Conv2d(ch, num_classes, kernel_size=1)
            self.aux_heads.append(aux_head)
        
        self._deep_supervision_enabled = True
    
    def enable_deep_supervision(self):
        """Enable deep supervision."""
        self._deep_supervision_enabled = True
    
    def disable_deep_supervision(self):
        """Disable deep supervision (for inference)."""
        self._deep_supervision_enabled = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        When deep supervision is enabled: returns the list [final, aux1, aux2, aux3]
        When deep supervision is disabled: returns the single final output
        """
        # The model must expose a method returning the intermediate features.
        # If the base model does not support it, only a single output is returned.

        if self._deep_supervision_enabled:
            if hasattr(self.model, 'forward_with_features'):
                final_output, features = self.model.forward_with_features(x)
                
                outputs = [final_output]
                
                # Generate auxiliary outputs
                for feat, aux_head in zip(features, self.aux_heads):
                    aux_out = aux_head(feat)
                    # Upsample to full resolution
                    aux_out = F.interpolate(
                        aux_out,
                        size=final_output.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                    outputs.append(aux_out)
                
                return outputs
        
        # Default: single output
        return self.model(x)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_deep_supervision_loss(base_loss_type: str,
                              n_classes: int,
                              ds_weights: Tuple[float, ...] = (1.0, 0.4, 0.2, 0.1),
                              **kwargs) -> nn.Module:
    """
    Factory function that builds a Deep Supervision Loss.

    Args:
        base_loss_type: Base loss type ('dice', 'ce', 'combined')
        n_classes: Number of classes
        ds_weights: Weights for deep supervision
        **kwargs: Additional args forwarded to the base loss

    Returns:
        Deep Supervision Loss module
    """
    if base_loss_type == 'dice_ce':
        return DeepSupervisionDiceCELoss(
            n_classes=n_classes,
            weights=ds_weights,
            **kwargs
        )
    else:
        # Use with external base loss
        from losses import get_loss_function
        base_loss = get_loss_function(base_loss_type, n_classes, **kwargs)
        return DeepSupervisionLoss(base_loss, weights=ds_weights)


def compute_ds_weights(num_outputs: int,
                       strategy: str = 'exponential',
                       base: float = 0.5) -> Tuple[float, ...]:
    """
    Compute the Deep Supervision weights.

    Args:
        num_outputs: Number of outputs
        strategy: 'exponential', 'linear', or 'uniform'
        base: Base value for the exponential strategy

    Returns:
        Tuple of weights
    """
    if strategy == 'exponential':
        # Exponentially decaying weights: [1.0, 0.5, 0.25, 0.125, ...]
        weights = [base ** i for i in range(num_outputs)]
        weights[0] = 1.0  # The final output always has weight = 1

    elif strategy == 'linear':
        # Linearly decaying weights
        weights = [1.0 - (i * (1.0 - 0.1) / num_outputs) for i in range(num_outputs)]

    elif strategy == 'uniform':
        # Uniform weights
        weights = [1.0 / num_outputs] * num_outputs
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return tuple(weights)


# =============================================================================
# TEST CODE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DEEP SUPERVISION MODULE TEST")
    print("=" * 60)
    
    n_classes = 9
    batch_size = 2
    h, w = 224, 224
    
    # Test DeepSupervisionDiceCELoss
    print("\n1. Testing DeepSupervisionDiceCELoss...")
    
    ds_loss = DeepSupervisionDiceCELoss(
        n_classes=n_classes,
        weights=(1.0, 0.4, 0.2, 0.1)
    )
    
    # Simulate multi-scale outputs
    outputs = [
        torch.randn(batch_size, n_classes, h, w),      # Final (full res)
        torch.randn(batch_size, n_classes, h//2, w//2),  # Aux 1
        torch.randn(batch_size, n_classes, h//4, w//4),  # Aux 2
        torch.randn(batch_size, n_classes, h//8, w//8),  # Aux 3
    ]
    
    target = torch.randint(0, n_classes, (batch_size, h, w))
    
    loss, loss_dict = ds_loss(outputs, target)
    
    print(f"  Total loss: {loss.item():.4f}")
    print(f"  Loss components: {loss_dict}")
    
    # Test with single output
    print("\n2. Testing with single output...")
    single_output = torch.randn(batch_size, n_classes, h, w)
    loss, loss_dict = ds_loss(single_output, target)
    print(f"  Total loss: {loss.item():.4f}")
    
    # Test DeepSupervisionHead
    print("\n3. Testing DeepSupervisionHead...")
    
    head = DeepSupervisionHead(
        in_channels_list=[256, 128, 64, 16],
        out_channels=n_classes
    )
    
    features = [
        torch.randn(batch_size, 256, h//2, w//2),
        torch.randn(batch_size, 128, h, w),
        torch.randn(batch_size, 64, h, w),
        torch.randn(batch_size, 16, h, w),
    ]
    
    outputs = head(features)
    print(f"  Number of outputs: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"    Output {i}: {out.shape}")
    
    # Test compute_ds_weights
    print("\n4. Testing DS weight computation...")
    
    for strategy in ['exponential', 'linear', 'uniform']:
        weights = compute_ds_weights(4, strategy=strategy)
        print(f"  {strategy}: {weights}")
    
    print("\n" + "=" * 60)
    print("[OK] Deep Supervision module test completed!")
    print("=" * 60)
