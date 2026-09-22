# =============================================================================
# NOTICE OF MODIFICATION (Apache License 2.0, Section 4(b))
#
# This file is derived from TransUNet:
#     https://github.com/Beckschen/TransUNet -- networks/vit_seg_modeling.py
#     Chen et al., "TransUNet: Transformers Make Strong Encoders for Medical
#     Image Segmentation," arXiv:2102.04306, 2021. Licensed under Apache-2.0.
#
# This file HAS BEEN MODIFIED by the authors of the present study. See NOTICE
# and THIRD_PARTY.md in the repository root, and the full licence text in
# LICENSE.
# =============================================================================
"""
TransUNet with CBAM + BiLSTM + Deep Supervision - PHASE 3
=========================================================
Extends vit_seg_modeling_xlstm.py with deep supervision support.

PHASE 3 IMPROVEMENTS:
1. DecoderCupDS: decoder that produces deep supervision outputs
2. VisionTransformerDS: main model with multi-scale output support
3. Wrapper for integrating with the existing model

Deep supervision lets gradients reach the decoder stages directly,
which improves gradient flow and boosts performance.

Author: SeqAtt-UNet Project - Phase 3 Improvements
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from typing import List, Tuple, Optional, Dict, Any

# Import from the original model
from networks import vit_seg_configs as configs
from networks.resnet_cbam import ResNetV2WithCBAM, ResNetV2

# Import components from vit_seg_modeling_xlstm
from networks.vit_seg_modeling_xlstm import (
    Attention,
    Mlp,
    BiLSTMMlp,
    HybridMlp,
    create_mlp,
    Block,
    Encoder,
    Embeddings,
    Transformer,
    Conv2dReLU,
    np2th,
    CONFIGS
)


logger = logging.getLogger(__name__)


# =============================================================================
# DEEP SUPERVISION COMPONENTS
# =============================================================================

class DecoderBlockDS(nn.Module):
    """
    Decoder block that can expose its intermediate features.

    Same as the original DecoderBlock, but it can return the features before
    upsampling so that they can be used for deep supervision.
    """
    
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super().__init__()
        
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels, out_channels,
            kernel_size=3, padding=1, use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels, out_channels,
            kernel_size=3, padding=1, use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        
        if skip is not None:
            # Handle size mismatch
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        
        x = self.conv1(x)
        x = self.conv2(x)
        
        return x


class SegmentationHeadDS(nn.Module):
    """
    Segmentation head supporting multiple outputs for deep supervision.

    Builds prediction heads at several different scales.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        
        if upsampling > 1:
            self.upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling)
        else:
            self.upsampling = nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.upsampling(x)
        return x


class DecoderCupDS(nn.Module):
    """
    Decoder with deep supervision - extends DecoderCup.

    Main changes:
    - Builds a segmentation head at every decoder stage
    - Returns a list of outputs for deep supervision

    Attributes:
        num_ds_outputs: Number of deep supervision outputs (default=4)
        ds_enabled: Flag that turns deep supervision on/off
    """
    
    def __init__(self, config, num_ds_outputs=4):
        """
        Args:
            config: Model configuration
            num_ds_outputs: Number of DS outputs (at most the number of decoder blocks)
        """
        super().__init__()
        
        self.config = config
        self.num_ds_outputs = num_ds_outputs
        self._ds_enabled = True
        
        head_channels = 512
        
        # Initial conv
        self.conv_more = Conv2dReLU(
            config.hidden_size, head_channels,
            kernel_size=3, padding=1, use_batchnorm=True,
        )
        
        # Decoder channels
        decoder_channels = config.decoder_channels  # (256, 128, 64, 16)
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        
        # Skip channels
        if self.config.n_skip != 0:
            skip_channels = list(self.config.skip_channels)
            for i in range(4 - self.config.n_skip):
                skip_channels[3 - i] = 0
        else:
            skip_channels = [0, 0, 0, 0]
        
        # Decoder blocks
        self.blocks = nn.ModuleList([
            DecoderBlockDS(in_ch, out_ch, sk_ch)
            for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ])
        
        # Deep Supervision heads - one head per decoder output
        self.ds_heads = nn.ModuleList()
        
        for i, ch in enumerate(decoder_channels[:num_ds_outputs]):
            # Compute the upsampling factor: each block upsamples by 2x
            # Block 0: 14x14 -> 28x28
            # Block 1: 28x28 -> 56x56
            # Block 2: 56x56 -> 112x112
            # Block 3: 112x112 -> 224x224
            # Upsampling factor required to reach full resolution
            upsample_factor = 2 ** (len(decoder_channels) - 1 - i)
            
            head = SegmentationHeadDS(
                in_channels=ch,
                out_channels=config.n_classes,
                kernel_size=3,
                upsampling=upsample_factor
            )
            self.ds_heads.append(head)
    
    def enable_deep_supervision(self):
        """Enable deep supervision (used during training)."""
        self._ds_enabled = True
    
    def disable_deep_supervision(self):
        """Disable deep supervision (used during inference)."""
        self._ds_enabled = False
    
    def forward(self, hidden_states, features=None):
        """
        Forward pass of the decoder.

        Args:
            hidden_states: Output of the Transformer encoder (B, N, D)
            features: Skip connection features from the hybrid backbone

        Returns:
            If ds_enabled=True: List[Tensor] of predictions at several scales
            If ds_enabled=False: a single Tensor prediction at full resolution
        """
        B, n_patch, hidden = hidden_states.size()
        h = w = int(np.sqrt(n_patch))
        
        # Reshape back to 2D
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        
        # Collect intermediate outputs for deep supervision
        ds_outputs = []
        
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            
            x = decoder_block(x, skip=skip)
            
            # Generate DS output at this scale
            if self._ds_enabled and i < self.num_ds_outputs:
                ds_out = self.ds_heads[i](x)
                ds_outputs.append(ds_out)
        
        if self._ds_enabled:
            # Reverse so that the final output comes first in the list
            # [aux_1, aux_2, aux_3, final] -> [final, aux_3, aux_2, aux_1]
            ds_outputs = ds_outputs[::-1]
            return ds_outputs
        else:
            # Return the final output only
            final_output = self.ds_heads[-1](x) if hasattr(self, 'ds_heads') else x
            return final_output


# =============================================================================
# MAIN MODEL WITH DEEP SUPERVISION
# =============================================================================

class VisionTransformerDS(nn.Module):
    """
    TransUNet with CBAM + BiLSTM + Deep Supervision.

    Extends the original VisionTransformer with:
    - DecoderCupDS: decoder that supports deep supervision
    - Multiple output heads
    - A flag that turns deep supervision on/off

    Usage:
    ```python
    model = VisionTransformerDS(config, use_deep_supervision=True)

    # Training (with deep supervision)
    outputs = model(images)  # List [final, aux1, aux2, aux3]

    # Inference (final output only)
    model.disable_deep_supervision()
    output = model(images)  # Single tensor
    ```
    """
    
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False,
                 use_cbam=True, mlp_type='standard', lstm_hidden_size=None,
                 use_deep_supervision=True, num_ds_outputs=4, **kwargs):
        """
        Args:
            config: Model configuration
            img_size: Input image size
            num_classes: Number of output classes
            zero_head: Zero initialize classifier head
            vis: Return attention weights
            use_cbam: Use CBAM attention in ResNet
            mlp_type: Type of MLP ('standard', 'bilstm', 'hybrid')
            lstm_hidden_size: Hidden size for LSTM
            use_deep_supervision: Enable deep supervision
            num_ds_outputs: Number of deep supervision outputs
        """
        super(VisionTransformerDS, self).__init__()
        
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.use_cbam = use_cbam
        self.mlp_type = mlp_type
        self.use_deep_supervision = use_deep_supervision
        self.num_ds_outputs = num_ds_outputs
        
        # Transformer (encoder)
        self.transformer = Transformer(
            config, img_size, vis,
            use_cbam=use_cbam,
            mlp_type=mlp_type,
            lstm_hidden_size=lstm_hidden_size,
            **kwargs
        )
        
        # Decoder with deep supervision
        if use_deep_supervision:
            self.decoder = DecoderCupDS(config, num_ds_outputs=num_ds_outputs)
        else:
            # Fallback to standard decoder
            from networks.vit_seg_modeling_xlstm import DecoderCup
            self.decoder = DecoderCup(config)
            self.segmentation_head = SegmentationHeadDS(
                in_channels=config['decoder_channels'][-1],
                out_channels=config['n_classes'],
                kernel_size=3,
            )
        
        self.config = config
        self._ds_enabled = use_deep_supervision
    
    def enable_deep_supervision(self):
        """Enable deep supervision."""
        self._ds_enabled = True
        if hasattr(self.decoder, 'enable_deep_supervision'):
            self.decoder.enable_deep_supervision()
    
    def disable_deep_supervision(self):
        """Disable deep supervision (used during inference)."""
        self._ds_enabled = False
        if hasattr(self.decoder, 'disable_deep_supervision'):
            self.decoder.disable_deep_supervision()
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input images (B, C, H, W)
            
        Returns:
            If deep_supervision is enabled: list of predictions at several scales
            If it is disabled: a single prediction tensor
        """
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Encoder
        x, attn_weights, features = self.transformer(x)
        
        # Decoder
        if self.use_deep_supervision:
            outputs = self.decoder(x, features)
            return outputs
        else:
            x = self.decoder(x, features)
            if hasattr(self, 'segmentation_head'):
                logits = self.segmentation_head(x)
            else:
                logits = x
            return logits
    
    def forward_with_features(self, x):
        """
        Forward pass that also returns the intermediate features.

        Useful for attaching auxiliary heads externally.
        
        Returns:
            Tuple of (final_output, intermediate_features)
        """
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        x, attn_weights, features = self.transformer(x)
        
        # Get decoder intermediate features
        B, n_patch, hidden = x.size()
        h = w = int(np.sqrt(n_patch))
        
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.decoder.conv_more(x)
        
        intermediate_features = []
        
        for i, decoder_block in enumerate(self.decoder.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            
            x = decoder_block(x, skip=skip)
            intermediate_features.append(x)
        
        # Final output
        if hasattr(self.decoder, 'ds_heads'):
            final_output = self.decoder.ds_heads[-1](x)
        elif hasattr(self, 'segmentation_head'):
            final_output = self.segmentation_head(x)
        else:
            final_output = x
        
        return final_output, intermediate_features
    
    def load_from(self, weights):
        """Load pretrained weights from a numpy file."""
        with torch.no_grad():
            res_weight = weights
            
            self.transformer.embeddings.patch_embeddings.weight.copy_(
                np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(
                np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(
                np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(
                np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1] - 1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                if self.use_cbam:
                    self.transformer.embeddings.hybrid_model.load_from(res_weight)
                else:
                    self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(
                        np2th(res_weight["conv_root/kernel"], conv=True))
                    gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                    gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                    self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                    self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                    for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                        for uname, unit in block.named_children():
                            unit.load_from(res_weight, n_block=bname, n_unit=uname)
        
        print(f"✓ Loaded pretrained weights")
        print(f"✓ CBAM: {self.use_cbam}, MLP type: {self.mlp_type}")
        print(f"✓ Deep Supervision: {self.use_deep_supervision}, Outputs: {self.num_ds_outputs}")


# =============================================================================
# WRAPPER FOR EXISTING MODEL
# =============================================================================

class DeepSupervisionAdapter(nn.Module):
    """
    Adapter that adds deep supervision to an existing model.

    It allows the original VisionTransformer to be used with deep supervision
    without modifying the model code.

    Usage:
    ```python
    from networks.vit_seg_modeling_xlstm import VisionTransformer
    
    base_model = VisionTransformer(config, ...)
    model = DeepSupervisionAdapter(base_model, num_classes=9)
    ```
    """
    
    def __init__(self, 
                 base_model: nn.Module,
                 num_classes: int,
                 num_ds_outputs: int = 4,
                 decoder_channels: Tuple[int, ...] = (256, 128, 64, 16)):
        """
        Args:
            base_model: Base model (VisionTransformer)
            num_classes: Number of classes
            num_ds_outputs: Number of DS outputs
            decoder_channels: Channel count at each decoder stage
        """
        super().__init__()
        
        self.base_model = base_model
        self.num_classes = num_classes
        self.num_ds_outputs = num_ds_outputs
        self._ds_enabled = True
        
        # Auxiliary heads for the intermediate outputs
        self.aux_heads = nn.ModuleList()
        
        for i, ch in enumerate(decoder_channels[:num_ds_outputs - 1]):
            # Each aux head is applied to one intermediate feature map
            upsample_factor = 2 ** (len(decoder_channels) - 1 - i)
            
            aux_head = nn.Sequential(
                nn.Conv2d(ch, num_classes, kernel_size=1),
                nn.UpsamplingBilinear2d(scale_factor=upsample_factor) if upsample_factor > 1 else nn.Identity()
            )
            self.aux_heads.append(aux_head)
    
    def enable_deep_supervision(self):
        self._ds_enabled = True
    
    def disable_deep_supervision(self):
        self._ds_enabled = False
    
    def forward(self, x):
        """
        Forward pass.
        
        When DS is enabled: returns the list [final, aux1, aux2, ...]
        When DS is disabled: returns a single final output
        """
        if self._ds_enabled and hasattr(self.base_model, 'forward_with_features'):
            final_output, features = self.base_model.forward_with_features(x)
            
            outputs = [final_output]
            
            for feat, aux_head in zip(features[:-1], self.aux_heads):
                aux_out = aux_head(feat)
                outputs.append(aux_out)
            
            return outputs
        else:
            return self.base_model(x)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_model_with_ds(config_name: str = 'R50-ViT-B_16',
                         img_size: int = 224,
                         num_classes: int = 9,
                         use_cbam: bool = True,
                         mlp_type: str = 'bilstm',
                         use_deep_supervision: bool = True,
                         num_ds_outputs: int = 4,
                         **kwargs) -> nn.Module:
    """
    Factory function that builds a model with deep supervision.
    
    Args:
        config_name: Config name ('R50-ViT-B_16', 'ViT-B_16', etc.)
        img_size: Input image size
        num_classes: Number of classes
        use_cbam: Whether to use CBAM
        mlp_type: MLP variant
        use_deep_supervision: Whether to enable deep supervision
        num_ds_outputs: Number of DS outputs
        
    Returns:
        Model instance
    """
    config = CONFIGS[config_name]
    config.n_classes = num_classes
    
    if config_name.find('R50') != -1:
        config.patches.grid = (
            int(img_size / 16),
            int(img_size / 16)
        )
    
    model = VisionTransformerDS(
        config,
        img_size=img_size,
        num_classes=num_classes,
        use_cbam=use_cbam,
        mlp_type=mlp_type,
        use_deep_supervision=use_deep_supervision,
        num_ds_outputs=num_ds_outputs,
        **kwargs
    )
    
    return model


# =============================================================================
# TEST CODE
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 3 MODEL TEST - Deep Supervision")
    print("=" * 70)
    
    # Test configuration
    config_name = 'R50-ViT-B_16'
    img_size = 224
    num_classes = 9
    batch_size = 2
    
    print(f"\nConfig: {config_name}")
    print(f"Image size: {img_size}x{img_size}")
    print(f"Classes: {num_classes}")
    print(f"Batch size: {batch_size}")
    
    # Create model
    print("\n1. Creating VisionTransformerDS...")
    
    model = create_model_with_ds(
        config_name=config_name,
        img_size=img_size,
        num_classes=num_classes,
        use_cbam=True,
        mlp_type='bilstm',
        use_deep_supervision=True,
        num_ds_outputs=4
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Test the forward pass with deep supervision
    print("\n2. Testing forward pass with Deep Supervision...")
    
    x = torch.randn(batch_size, 3, img_size, img_size)
    
    model.eval()
    with torch.no_grad():
        outputs = model(x)
    
    if isinstance(outputs, list):
        print(f"   Number of outputs: {len(outputs)}")
        for i, out in enumerate(outputs):
            print(f"   Output {i}: {out.shape}")
    else:
        print(f"   Single output: {outputs.shape}")
    
    # Test with deep supervision disabled
    print("\n3. Testing forward pass without Deep Supervision...")
    
    model.disable_deep_supervision()
    
    with torch.no_grad():
        output = model(x)
    
    if isinstance(output, list):
        print(f"   Output (list): {[o.shape for o in output]}")
    else:
        print(f"   Output: {output.shape}")
    
    # Test forward_with_features
    print("\n4. Testing forward_with_features...")
    
    model.enable_deep_supervision()
    
    with torch.no_grad():
        final, features = model.forward_with_features(x)
    
    print(f"   Final output: {final.shape}")
    print(f"   Number of features: {len(features)}")
    for i, feat in enumerate(features):
        print(f"   Feature {i}: {feat.shape}")
    
    print("\n" + "=" * 70)
    print("✓ Phase 3 model test completed!")
    print("=" * 70)
