# =============================================================================
# NOTICE OF MODIFICATION (Apache License 2.0, Section 4(b))
#
# This file is derived from TransUNet:
#     https://github.com/Beckschen/TransUNet -- networks/vit_seg_modeling_resnet_skip.py
#     Chen et al., "TransUNet: Transformers Make Strong Encoders for Medical
#     Image Segmentation," arXiv:2102.04306, 2021. Licensed under Apache-2.0.
#
# This file HAS BEEN MODIFIED by the authors of the present study. See NOTICE
# and THIRD_PARTY.md in the repository root, and the full licence text in
# LICENSE.
# =============================================================================
"""
ResNetV2 Backbone with Residual CBAM for TransUNet
==================================================
Improvements:
1. Residual CBAM with learnable scaling (alpha=0 init)
2. No DCN - standard convolutions are kept unchanged
3. Fully compatible with the pretrained weights

Author: TransUNet_CBAM_LSTM Project
"""

import math
from os.path import join as pjoin
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def np2th(weights, conv=False):
    """Convert a weight tensor from HWIO to OIHW layout for PyTorch."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


# =============================================================================
# CBAM Module with Residual Connection
# =============================================================================

class ChannelAttention(nn.Module):
    """
    Channel Attention Module of CBAM.
    Learns "what" - which feature channels are important.
    """
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        reduced_channels = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module of CBAM.
    Learns "where" - which spatial locations are important.
    """
    
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))


class ResidualCBAM(nn.Module):
    """
    CBAM with a residual connection and learnable scaling.

    Formula: output = x + alpha * (CBAM(x) - x)

    [IMPROVEMENT V2] Use softplus to guarantee alpha >= 0

    Problem observed previously:
    - alpha could drift to negative values during training
    - A negative alpha means the network is inverting the attention effect
    - This indicates that CBAM was hurting performance

    Solution:
    - Store alpha_raw (an unconstrained parameter)
    - Apply softplus(alpha_raw) so that alpha >= 0
    - softplus(x) = log(1 + exp(x)), a smooth approximation of ReLU
    - Initialize alpha_raw such that softplus(alpha_raw) is about 0.01

    Expected range of alpha after training:
    - alpha in [0, 0.1]: CBAM has a weak effect
    - alpha in [0.1, 0.3]: CBAM is moderately active
    - alpha in [0.3, 0.5]: CBAM is strongly active
    - alpha > 0.5: CBAM has a very large influence
    """
    
    # [CONFIG] Change this value to run different experiments
    ALPHA_INIT_VALUE = 0.01  # Initial alpha value (after softplus)
    
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)
        
        # [IMPROVEMENT V2] Learnable scaling factor with the constraint alpha >= 0
        # Compute alpha_raw such that softplus(alpha_raw) is about ALPHA_INIT_VALUE
        # softplus(x) = log(1 + exp(x))
        # Inverse: x = log(exp(y) - 1) for y > 0
        if self.ALPHA_INIT_VALUE > 0:
            # Compute the initial value of alpha_raw
            init_raw = math.log(math.exp(self.ALPHA_INIT_VALUE) - 1 + 1e-8)
        else:
            init_raw = -5.0  # softplus(-5) is about 0.007
        
        self.alpha_raw = nn.Parameter(torch.tensor([init_raw]))
    
    @property
    def alpha(self):
        """Return alpha after the >= 0 constraint has been applied."""
        return F.softplus(self.alpha_raw)
        
    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        attended = x * ca
        
        # Spatial attention  
        sa = self.spatial_attention(attended)
        attended = attended * sa
        
        # Residual connection with learnable scaling (alpha >= 0)
        alpha_constrained = F.softplus(self.alpha_raw)
        return x + alpha_constrained * (attended - x)


# =============================================================================
# Standard Convolution with Weight Standardization
# =============================================================================

class StdConv2d(nn.Conv2d):
    """2D convolution with weight standardization."""
    
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, 
                       self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride, 
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride, 
                     padding=0, bias=bias)


# =============================================================================
# PreActBottleneck Block
# =============================================================================

class PreActBottleneck(nn.Module):
    """Pre-activation Bottleneck Block."""
    
    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)
        
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or cin != cout:
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

    def load_from(self, weights, n_block, n_unit):
        """Load pretrained weights."""
        conv1_weight = np2th(weights[f'{n_block}/{n_unit}/conv1/kernel'], conv=True)
        conv2_weight = np2th(weights[f'{n_block}/{n_unit}/conv2/kernel'], conv=True)
        conv3_weight = np2th(weights[f'{n_block}/{n_unit}/conv3/kernel'], conv=True)

        gn1_weight = np2th(weights[f'{n_block}/{n_unit}/gn1/scale'])
        gn1_bias = np2th(weights[f'{n_block}/{n_unit}/gn1/bias'])
        gn2_weight = np2th(weights[f'{n_block}/{n_unit}/gn2/scale'])
        gn2_bias = np2th(weights[f'{n_block}/{n_unit}/gn2/bias'])
        gn3_weight = np2th(weights[f'{n_block}/{n_unit}/gn3/scale'])
        gn3_bias = np2th(weights[f'{n_block}/{n_unit}/gn3/bias'])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))
        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))
        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[f'{n_block}/{n_unit}/conv_proj/kernel'], conv=True)
            proj_gn_weight = np2th(weights[f'{n_block}/{n_unit}/gn_proj/scale'])
            proj_gn_bias = np2th(weights[f'{n_block}/{n_unit}/gn_proj/bias'])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))


# =============================================================================
# ResNetV2 with CBAM
# =============================================================================

class ResNetV2WithCBAM(nn.Module):
    """
    ResNetV2 with a Residual CBAM after each block.
    """
    
    def __init__(self, block_units, width_factor, use_cbam=True, 
                 cbam_reduction=16, cbam_spatial_kernel=7):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width
        self.use_cbam = use_cbam

        # Root block
        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
        ]))

        # Block 1
        self.body_block1 = nn.Sequential(OrderedDict(
            [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
            [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) 
             for i in range(2, block_units[0] + 1)]
        ))
        if use_cbam:
            self.cbam1 = ResidualCBAM(width * 4, cbam_reduction, cbam_spatial_kernel)

        # Block 2
        self.body_block2 = nn.Sequential(OrderedDict(
            [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
            [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) 
             for i in range(2, block_units[1] + 1)]
        ))
        if use_cbam:
            self.cbam2 = ResidualCBAM(width * 8, cbam_reduction, cbam_spatial_kernel)

        # Block 3
        self.body_block3 = nn.Sequential(OrderedDict(
            [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
            [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) 
             for i in range(2, block_units[2] + 1)]
        ))
        if use_cbam:
            self.cbam3 = ResidualCBAM(width * 16, cbam_reduction, cbam_spatial_kernel)

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        
        x = self.root(x)
        features.append(x)
        
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        
        # Block 1 + CBAM
        x = self.body_block1(x)
        if self.use_cbam:
            x = self.cbam1(x)
        
        right_size = int(in_size / 4)
        if x.size()[2] != right_size:
            pad = right_size - x.size()[2]
            assert pad < 3 and pad > 0
            feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
            feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
        else:
            feat = x
        features.append(feat)
        
        # Block 2 + CBAM
        x = self.body_block2(x)
        if self.use_cbam:
            x = self.cbam2(x)
            
        right_size = int(in_size / 8)
        if x.size()[2] != right_size:
            pad = right_size - x.size()[2]
            assert pad < 3 and pad > 0
            feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
            feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
        else:
            feat = x
        features.append(feat)
        
        # Block 3 + CBAM
        x = self.body_block3(x)
        if self.use_cbam:
            x = self.cbam3(x)

        return x, features[::-1]
    
    def load_from(self, weights):
        """Load pretrained weights."""
        with torch.no_grad():
            self.root.conv.weight.copy_(np2th(weights['conv_root/kernel'], conv=True))
            self.root.gn.weight.copy_(np2th(weights['gn_root/scale']).view(-1))
            self.root.gn.bias.copy_(np2th(weights['gn_root/bias']).view(-1))
            
            for bname, block in [('block1', self.body_block1), 
                                  ('block2', self.body_block2), 
                                  ('block3', self.body_block3)]:
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=bname, n_unit=uname)
        
        print("✓ Loaded pretrained weights for ResNetV2")
        if self.use_cbam:
            print(f"✓ CBAM modules initialized with α={ResidualCBAM.ALPHA_INIT_VALUE} "
                  f"(near-identity mapping)")


# =============================================================================
# Original ResNetV2 (Backward Compatibility)
# =============================================================================

class ResNetV2(nn.Module):
    """Original ResNetV2."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) 
                 for i in range(2, block_units[0] + 1)]
            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) 
                 for i in range(2, block_units[1] + 1)]
            ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) 
                 for i in range(2, block_units[2] + 1)]
            ))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i+1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        
        x = self.body[-1](x)
        return x, features[::-1]
