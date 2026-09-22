# =============================================================================
# NOTICE OF MODIFICATION (Apache License 2.0, Section 4(b))
#
# This file is derived from TransUNet:
#     https://github.com/Beckschen/TransUNet -- networks/vit_seg_modeling.py (DecoderCup)
#     Chen et al., "TransUNet: Transformers Make Strong Encoders for Medical
#     Image Segmentation," arXiv:2102.04306, 2021. Licensed under Apache-2.0.
#
# This file HAS BEEN MODIFIED by the authors of the present study. See NOTICE
# and THIRD_PARTY.md in the repository root, and the full licence text in
# LICENSE.
# =============================================================================
"""
SKIP-CBAM MODULE FOR SeqAtt-UNet
================================
Phase 4 improvement: apply CBAM at the SKIP CONNECTIONS inside the decoder, not
only at the encoder stages. Goal: filter noise out of the skip features and bias
them spatially BEFORE they are concatenated with the upsampled decoder features.

SCIENTIFIC MOTIVATION:
- The encoder CBAM (already present) refines features inside the encoder pipeline.
- Skip features take a shortcut (the skip connection) to the decoder and carry
  both useful information (boundary cues) and noise (irrelevant texture).
- Skip-CBAM gates the skip features before they are merged with decoder features,
  letting the decoder select information per spatial location and per channel.

ARCHITECTURE:
- Each decoder block has its own SkipCBAM gating its own skip feature.
- Each SkipCBAM has its own alpha (learnable, softplus constrained to >= 0).
- Alpha starts at ~0.01 -> identity at init -> pretrained behavior is preserved.

INTEGRATION:
1. The original `vit_seg_modeling_phase3.py` is left untouched.
2. Subclass `DecoderCupDS` -> `DecoderCupSkipCBAM`.
3. Wrap `VisionTransformerDS` -> `VisionTransformerSkipCBAM` (factory function).
4. Reuse the same pretrained-loading pipeline and alpha tracker.

MEASURED ABLATION OUTCOME (5 seeds per dataset; supersedes the expectations that
stood here while the module was being written):
- Synapse: 79.76 +/- 0.30 % mean DSC against 81.00 +/- 0.61 % for the
  deep-supervision Baseline -- 1.24 pp WORSE, not the gain that was anticipated.
- ACDC: lowest mean HD95 of any configuration (1.41 +/- 0.34 voxels), but the
  advantage holds in only two of five seeds and the mean is carried by one of
  them. Reported in the paper as an observation, not as a gain.
- ACDC mean DSC 90.32 %, marginally below +CBAM+BiLSTM+DS at 90.39 %.

COST (measured, not estimated):
- Params: +0.08 M (374.71 M with the extension against 374.63 M without).
  The three modules act on the skip features themselves, BEFORE concatenation,
  so each is only as wide as its own skip: 512, 256 and 64 channels, which comes
  to 83,241 parameters. An earlier note here said "+0.3 M"; that figure belongs
  to a design in which the module spans the fused tensor after concatenation,
  which is not what this file implements.
- Throughput: no measurable cost (within the +/-2 % run-to-run variation).
- VRAM: +1 MB (1570 against 1569 MB).

REFERENCES:
- Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.
- Oktay et al., "Attention U-Net", arXiv 2018 (precedent for skip gating).
- Schlemper et al., "Attention Gated Networks", MedIA 2019.

Author: SeqAtt-UNet Project - Phase 4 (Skip-CBAM)
Date: 2026-06
"""

from __future__ import absolute_import, division, print_function

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse from resnet_cbam (keeps the CBAM block consistent)
from networks.resnet_cbam import ChannelAttention, SpatialAttention

# Reuse decoder blocks & model classes
from networks.vit_seg_modeling_phase3 import (
    DecoderBlockDS,
    SegmentationHeadDS,
    VisionTransformerDS,
)
from networks.vit_seg_modeling_xlstm import Conv2dReLU


__all__ = [
    'SkipCBAM',
    'DecoderBlockSkipCBAM',
    'DecoderCupSkipCBAM',
    'VisionTransformerSkipCBAM',
    'create_skip_cbam_model',
]


# =============================================================================
# SKIP-CBAM MODULE (CBAM applied to skip features only)
# =============================================================================

class SkipCBAM(nn.Module):
    """
    CBAM-based gating for a skip connection.

    Formula (same as ResidualCBAM):
        F'_skip = F_skip + alpha * (CBAM(F_skip) - F_skip)

    where alpha = softplus(alpha_raw) >= 0, initialized at ~0.01 (near identity).

    DIFFERENCES from the encoder ResidualCBAM:
    - It sits on the skip path, not in the main encoder forward pass.
    - It owns its alpha, independent of the encoder alpha, so the two can
      learn different balances.
    - Smaller reduction (8 instead of 16) since skip features have fewer channels
      (256/512).
    """

    ALPHA_INIT = 0.01

    def __init__(self, channels: int, reduction: int = 8, spatial_kernel: int = 7):
        super().__init__()
        self.channels = channels
        self.channel_attention = ChannelAttention(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel)

        # Learnable alpha with a softplus constraint
        if self.ALPHA_INIT > 0:
            init_raw = math.log(math.exp(self.ALPHA_INIT) - 1 + 1e-8)
        else:
            init_raw = -5.0
        self.alpha_raw = nn.Parameter(torch.tensor([init_raw]))

    @property
    def alpha(self) -> torch.Tensor:
        """Alpha after the >= 0 constraint (for logging/tracking)."""
        return F.softplus(self.alpha_raw)

    def forward(self, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            skip: Skip feature [B, C, H, W]

        Returns:
            Gated skip feature [B, C, H, W]
        """
        ca = self.channel_attention(skip)
        attended = skip * ca

        sa = self.spatial_attention(attended)
        attended = attended * sa

        alpha = F.softplus(self.alpha_raw)
        return skip + alpha * (attended - skip)


# =============================================================================
# DECODER BLOCK WITH SKIP-CBAM
# =============================================================================

class DecoderBlockSkipCBAM(nn.Module):
    """
    Decoder block identical to DecoderBlockDS, except that Skip-CBAM is applied
    to the skip features BEFORE the concatenation.

    Flow:
        x_up = Upsample(x)
        skip_refined = SkipCBAM(skip)   # <- newly added
        x = concat([x_up, skip_refined])
        x = Conv -> ReLU -> Conv -> ReLU
    """

    def __init__(self, in_channels: int, out_channels: int,
                 skip_channels: int = 0, use_batchnorm: bool = True,
                 skip_cbam_reduction: int = 8):
        super().__init__()

        self.use_skip_cbam = skip_channels > 0

        self.conv1 = Conv2dReLU(
            in_channels + skip_channels, out_channels,
            kernel_size=3, padding=1, use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels, out_channels,
            kernel_size=3, padding=1, use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # SkipCBAM on the skip path (only when a skip exists)
        if self.use_skip_cbam:
            self.skip_cbam = SkipCBAM(
                channels=skip_channels,
                reduction=skip_cbam_reduction,
                spatial_kernel=7,
            )
        else:
            self.skip_cbam = None

    def forward(self, x: torch.Tensor,
                skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.up(x)

        if skip is not None:
            # Handle size mismatch
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:],
                                  mode='bilinear', align_corners=False)
            # [NEW] Apply Skip-CBAM BEFORE the concatenation
            if self.skip_cbam is not None:
                skip = self.skip_cbam(skip)
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        return x


# =============================================================================
# DECODER CUP WITH SKIP-CBAM (Subclass DecoderCupDS to inherit DS logic)
# =============================================================================

class DecoderCupSkipCBAM(nn.Module):
    """
    Decoder cup with Skip-CBAM + deep supervision.

    Same structure as DecoderCupDS, but DecoderBlockDS is replaced by
    DecoderBlockSkipCBAM in the first 3 blocks (those with skips); the last has none.
    """

    def __init__(self, config, num_ds_outputs: int = 4,
                 skip_cbam_reduction: int = 8):
        super().__init__()

        self.config = config
        self.num_ds_outputs = num_ds_outputs
        self._ds_enabled = True

        head_channels = 512

        self.conv_more = Conv2dReLU(
            config.hidden_size, head_channels,
            kernel_size=3, padding=1, use_batchnorm=True,
        )

        decoder_channels = config.decoder_channels  # (256, 128, 64, 16)
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if config.n_skip != 0:
            skip_channels = list(config.skip_channels)
            for i in range(4 - config.n_skip):
                skip_channels[3 - i] = 0
        else:
            skip_channels = [0, 0, 0, 0]

        # [NEW] Decoder blocks that use SkipCBAM
        self.blocks = nn.ModuleList([
            DecoderBlockSkipCBAM(
                in_ch, out_ch, sk_ch,
                skip_cbam_reduction=skip_cbam_reduction,
            )
            for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ])

        # Deep Supervision heads (same as DecoderCupDS)
        self.ds_heads = nn.ModuleList()
        for i, ch in enumerate(decoder_channels[:num_ds_outputs]):
            upsample_factor = 2 ** (len(decoder_channels) - 1 - i)
            head = SegmentationHeadDS(
                in_channels=ch,
                out_channels=config.n_classes,
                kernel_size=3,
                upsampling=upsample_factor,
            )
            self.ds_heads.append(head)

    def enable_deep_supervision(self):
        self._ds_enabled = True

    def disable_deep_supervision(self):
        self._ds_enabled = False

    def forward(self, hidden_states: torch.Tensor,
                features: Optional[List[torch.Tensor]] = None):
        B, n_patch, hidden = hidden_states.size()
        h = w = int(np.sqrt(n_patch))

        x = hidden_states.permute(0, 2, 1).contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)

        ds_outputs = []

        for i, decoder_block in enumerate(self.blocks):
            if features is not None and i < self.config.n_skip:
                skip = features[i]
            else:
                skip = None

            x = decoder_block(x, skip=skip)

            if self._ds_enabled and i < self.num_ds_outputs:
                ds_out = self.ds_heads[i](x)
                ds_outputs.append(ds_out)

        if self._ds_enabled:
            # Same as DecoderCupDS: reverse -> [final, aux_*]
            ds_outputs = ds_outputs[::-1]
            return ds_outputs
        else:
            final_output = self.ds_heads[-1](x) if hasattr(self, 'ds_heads') else x
            return final_output

    def get_skip_cbam_alphas(self) -> dict:
        """Return a dict {block_idx: alpha_value} for tracking."""
        alphas = {}
        for i, block in enumerate(self.blocks):
            if block.skip_cbam is not None:
                alphas[f'skip_cbam_block{i}'] = block.skip_cbam.alpha.item()
        return alphas


# =============================================================================
# MAIN MODEL: VisionTransformerSkipCBAM
# =============================================================================

class VisionTransformerSkipCBAM(VisionTransformerDS):
    """
    SeqAtt-UNet with the additional Skip-CBAM.

    Inherits from VisionTransformerDS (keeping encoder CBAM + BiLSTM-MLP + DS)
    and ONLY swaps the decoder for DecoderCupSkipCBAM.

    Usage:
    ```python
    model = VisionTransformerSkipCBAM(
        config, img_size=224, num_classes=9,
        use_cbam=True,                  # Encoder CBAM unchanged
        mlp_type='bilstm',
        use_deep_supervision=True,
        skip_cbam_reduction=8,          # New parameter
    )
    model.load_from(np.load(pretrained_path))  # Pretrained weights still work
    ```
    """

    def __init__(self, config, img_size: int = 224, num_classes: int = 21843,
                 zero_head: bool = False, vis: bool = False,
                 use_cbam: bool = True, mlp_type: str = 'standard',
                 lstm_hidden_size: Optional[int] = None,
                 use_deep_supervision: bool = True,
                 num_ds_outputs: int = 4,
                 skip_cbam_reduction: int = 8,
                 **kwargs):

        # Call the parent __init__ (builds the encoder, transformer, old decoder)
        super().__init__(
            config=config,
            img_size=img_size,
            num_classes=num_classes,
            zero_head=zero_head,
            vis=vis,
            use_cbam=use_cbam,
            mlp_type=mlp_type,
            lstm_hidden_size=lstm_hidden_size,
            use_deep_supervision=use_deep_supervision,
            num_ds_outputs=num_ds_outputs,
            **kwargs,
        )

        # [NEW] Swap the decoder for the SkipCBAM variant
        if use_deep_supervision:
            self.decoder = DecoderCupSkipCBAM(
                config,
                num_ds_outputs=num_ds_outputs,
                skip_cbam_reduction=skip_cbam_reduction,
            )
        else:
            raise NotImplementedError(
                "VisionTransformerSkipCBAM requires use_deep_supervision=True. "
                "For a variant without DS, subclass VisionTransformerSkipCBAM "
                "and override `decoder`."
            )

        self.skip_cbam_reduction = skip_cbam_reduction

    def get_all_alphas(self) -> dict:
        """
        Return all alpha values (encoder CBAM + skip CBAM) for the alpha tracker.
        """
        alphas = {}

        # Encoder CBAM alphas (from ResNetV2WithCBAM)
        for name, module in self.named_modules():
            if hasattr(module, 'alpha_raw') and isinstance(module.alpha_raw, nn.Parameter):
                short_name = name.split('.')[-1] if '.' in name else name
                alphas[f'encoder_{short_name}'] = F.softplus(module.alpha_raw).item()

        # Skip-CBAM alphas (from DecoderCupSkipCBAM)
        if hasattr(self.decoder, 'get_skip_cbam_alphas'):
            alphas.update(self.decoder.get_skip_cbam_alphas())

        return alphas


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_skip_cbam_model(config_name: str = 'R50-ViT-B_16',
                           img_size: int = 224,
                           num_classes: int = 9,
                           use_cbam: bool = True,
                           mlp_type: str = 'bilstm',
                           use_deep_supervision: bool = True,
                           num_ds_outputs: int = 4,
                           skip_cbam_reduction: int = 8,
                           **kwargs) -> VisionTransformerSkipCBAM:
    """
    Factory that builds a VisionTransformerSkipCBAM.

    Args:
        config_name: ViT config name ('R50-ViT-B_16', ...)
        img_size: Input image size
        num_classes: Number of segmentation classes
        use_cbam: Enable the encoder CBAM
        mlp_type: MLP variant ('standard', 'bilstm', 'hybrid')
        use_deep_supervision: Enable deep supervision
        num_ds_outputs: Number of DS outputs
        skip_cbam_reduction: Reduction factor for SkipCBAM (default=8)

    Returns:
        Model instance
    """
    from networks.vit_seg_modeling_xlstm import CONFIGS

    config = CONFIGS[config_name]
    config.n_classes = num_classes

    if 'R50' in config_name:
        config.patches.grid = (img_size // 16, img_size // 16)

    model = VisionTransformerSkipCBAM(
        config=config,
        img_size=img_size,
        num_classes=num_classes,
        use_cbam=use_cbam,
        mlp_type=mlp_type,
        use_deep_supervision=use_deep_supervision,
        num_ds_outputs=num_ds_outputs,
        skip_cbam_reduction=skip_cbam_reduction,
        **kwargs,
    )

    return model


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == '__main__':
    print('=' * 70)
    print('SKIP-CBAM MODULE TEST')
    print('=' * 70)

    # Test the SkipCBAM module on its own
    print('\n[1] SkipCBAM forward shape test')
    skip_cbam = SkipCBAM(channels=256, reduction=8)
    x = torch.randn(2, 256, 28, 28)
    y = skip_cbam(x)
    print(f'    Input:  {x.shape}')
    print(f'    Output: {y.shape}')
    print(f'    Alpha:  {skip_cbam.alpha.item():.6f}')
    assert y.shape == x.shape

    # Test the full model with a 3-channel input
    print('\n[2] VisionTransformerSkipCBAM forward')
    model = create_skip_cbam_model(
        config_name='R50-ViT-B_16',
        img_size=224,
        num_classes=9,
        use_cbam=True,
        mlp_type='bilstm',
        use_deep_supervision=True,
        num_ds_outputs=4,
        skip_cbam_reduction=8,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f'    Total params: {n_params:,}')

    x = torch.randn(2, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        outputs = model(x)

    if isinstance(outputs, list):
        for i, o in enumerate(outputs):
            print(f'    Output {i}: {o.shape}')
    else:
        print(f'    Output: {outputs.shape}')

    # Test alpha tracking
    print('\n[3] Alpha tracker')
    alphas = model.get_all_alphas()
    for name, val in alphas.items():
        print(f'    {name:30s}: {val:.6f}')

    print('\n' + '=' * 70)
    print('✓ Skip-CBAM module test passed!')
    print('=' * 70)
