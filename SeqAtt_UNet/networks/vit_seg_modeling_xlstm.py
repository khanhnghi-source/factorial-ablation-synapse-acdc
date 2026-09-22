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
TransUNet with CBAM + xLSTM in the feed-forward sub-block
=====================================
Extended version with xLSTM support (sLSTM and mLSTM).

Main additions:
1. ResNetV2 + Residual CBAM
2. xLSTM inside the Transformer feed-forward sub-block (replacing or supplementing the BiLSTM)
3. Several variants: bilstm, slstm, mlstm, hybrid_slstm, hybrid_mlstm

Usage:
    model = VisionTransformer(config, mlp_type='mlstm')  # use mLSTM
    model = VisionTransformer(config, mlp_type='slstm')  # use sLSTM
    model = VisionTransformer(config, mlp_type='hybrid_mlstm')  # hybrid with mLSTM

Author: SeqAtt_UNet Project (Extended Version)
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
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

from . import vit_seg_configs as configs
from .resnet_cbam import ResNetV2WithCBAM, ResNetV2

# Import xLSTM modules
try:
    from .xlstm_modules import xLSTMMlp, HybridxLSTMMlp, create_xlstm_mlp
    XLSTM_AVAILABLE = True
except ImportError:
    XLSTM_AVAILABLE = False
    print("Warning: xLSTM modules not found. Only BiLSTM will be available.")


logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


# =============================================================================
# Attention module (unchanged from TransUNet)
# =============================================================================

class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        
        return attention_output, weights


# =============================================================================
# MLP Modules (Standard + BiLSTM + xLSTM variants)
# =============================================================================

class Mlp(nn.Module):
    """Standard MLP Module."""
    
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class BiLSTMMlp(nn.Module):
    """
    Feed-forward sub-block with a BiLSTM, to capture sequential dependencies between patches.
    
    Architecture:
    Input (B, N, D) → FC1 → GELU → BiLSTM → LayerNorm → FC2 → Output
    
    [CONSIDERED, NOT ADOPTED] Raising lstm_hidden_size from mlp_dim//4 to mlp_dim//2
    
    Rationale that was considered:
    - As shipped: lstm_hidden = 768 (mlp_dim=3072 / 4)
    - BiLSTM output = 768 * 2 = 1536, a 50% narrowing relative to mlp_dim
    
    - Alternative: lstm_hidden = 1536 (mlp_dim=3072 / 2)  
    - BiLSTM output = 1536 * 2 = 3072, matching mlp_dim
    - Less information loss, more BiLSTM capacity
    
    Trade-off:
    - GPU memory rises by roughly 20-30% for the BiLSTM layers
    - Fall back to mlp_dim // 4 if memory is short
    
    Options:
    - LSTM_HIDDEN_DIVISOR = 2: more capacity
    - LSTM_HIDDEN_DIVISOR = 4: less memory. THIS IS THE VALUE USED FOR EVERY REPORTED RUN.
    """
    
    # [CONFIG] This constant sets the BiLSTM capacity.
    # IMPORTANT: keep it at 4; anything smaller explodes the parameter count.
    # - DIVISOR = 4: lstm_hidden = 768, ~23.6M params/block  <-- used for all reported runs
    # - DIVISOR = 2: lstm_hidden = 1536, ~56.6M params/block  (not used)
    # - DIVISOR = 3: lstm_hidden = 1024, ~35M params/block   (not used)
    LSTM_HIDDEN_DIVISOR = 4  # keep at 4; see the note above
    
    def __init__(self, config, lstm_hidden_size=None, num_lstm_layers=1):
        super(BiLSTMMlp, self).__init__()
        
        hidden_size = config.hidden_size
        mlp_dim = config.transformer["mlp_dim"]
        dropout_rate = config.transformer["dropout_rate"]
        
        # NOTE: the divisor below is the one actually used (mlp_dim//4 -> 768)
        # with DIVISOR=4: lstm_hidden = 768, BiLSTM output = 1536, projected back to 768
        self.lstm_hidden = lstm_hidden_size or (mlp_dim // self.LSTM_HIDDEN_DIVISOR)
        
        self.fc1 = Linear(hidden_size, mlp_dim)
        self.act_fn = ACT2FN["gelu"]
        self.dropout1 = Dropout(dropout_rate)
        
        self.bilstm = nn.LSTM(
            input_size=mlp_dim,
            hidden_size=self.lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_lstm_layers > 1 else 0
        )
        
        self.lstm_norm = LayerNorm(self.lstm_hidden * 2, eps=1e-6)
        self.fc2 = Linear(self.lstm_hidden * 2, hidden_size)
        self.dropout2 = Dropout(dropout_rate)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)
        
        for name, param in self.bilstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)

    def forward(self, x):
        h = self.fc1(x)
        h = self.act_fn(h)
        h = self.dropout1(h)
        
        lstm_out, _ = self.bilstm(h)
        lstm_out = self.lstm_norm(lstm_out)
        
        out = self.fc2(lstm_out)
        out = self.dropout2(out)
        
        return out


class HybridMlp(nn.Module):
    """
    Hybrid feed-forward: standard projection and BiLSTM sharing one expansion layer,
    mixed by a single sigmoid gate.
    output = gate * BiLSTM_path + (1 - gate) * Standard_path
    """
    
    def __init__(self, config, lstm_hidden_size=None):
        super(HybridMlp, self).__init__()
        
        hidden_size = config.hidden_size
        mlp_dim = config.transformer["mlp_dim"]
        dropout_rate = config.transformer["dropout_rate"]
        
        # Standard MLP path
        self.fc1 = Linear(hidden_size, mlp_dim)
        self.fc2 = Linear(mlp_dim, hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(dropout_rate)
        
        # BiLSTM path
        self.lstm_hidden = lstm_hidden_size or (mlp_dim // 4)
        self.bilstm = nn.LSTM(
            input_size=mlp_dim,
            hidden_size=self.lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.lstm_proj = Linear(self.lstm_hidden * 2, hidden_size)
        self.lstm_norm = LayerNorm(self.lstm_hidden * 2, eps=1e-6)
        
        # Gating
        self.gate = nn.Parameter(torch.tensor(0.5))
        
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.lstm_proj.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)
        nn.init.normal_(self.lstm_proj.bias, std=1e-6)
        
        for name, param in self.bilstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)

    def forward(self, x):
        h = self.fc1(x)
        h = self.act_fn(h)
        h = self.dropout(h)
        std_out = self.fc2(h)
        std_out = self.dropout(std_out)
        
        lstm_out, _ = self.bilstm(h)
        lstm_out = self.lstm_norm(lstm_out)
        lstm_out = self.lstm_proj(lstm_out)
        lstm_out = self.dropout(lstm_out)
        
        g = torch.sigmoid(self.gate)
        out = g * lstm_out + (1 - g) * std_out
        
        return out
    
    def load_pretrained_fc(self, fc1_weight, fc1_bias, fc2_weight, fc2_bias):
        self.fc1.weight.copy_(fc1_weight)
        self.fc1.bias.copy_(fc1_bias)
        self.fc2.weight.copy_(fc2_weight)
        self.fc2.bias.copy_(fc2_bias)


def create_mlp(config, mlp_type='standard', lstm_hidden_size=None, **kwargs):
    """
    Factory returning the requested feed-forward module.
    
    Args:
        config: Model configuration
        mlp_type: which feed-forward variant:
            - 'standard': the canonical TransUNet MLP
            - 'bilstm': BiLSTM MLP
            - 'hybrid': Hybrid BiLSTM + Standard
            - 'slstm': sLSTM MLP (xLSTM)
            - 'mlstm': mLSTM MLP (xLSTM)
            - 'xlstm': Alias cho 'mlstm'
            - 'hybrid_slstm': Hybrid sLSTM
            - 'hybrid_mlstm': Hybrid mLSTM
            - 'hybrid_xlstm': Alias cho 'hybrid_mlstm'
        lstm_hidden_size: Hidden size cho LSTM/xLSTM
        **kwargs: Additional arguments cho xLSTM
    
    Returns:
        MLP module
    """
    if mlp_type == 'standard':
        return Mlp(config)
    elif mlp_type == 'bilstm':
        return BiLSTMMlp(config, lstm_hidden_size)
    elif mlp_type == 'hybrid':
        return HybridMlp(config, lstm_hidden_size)
    elif mlp_type in ['slstm', 'mlstm', 'xlstm', 'hybrid_slstm', 'hybrid_mlstm', 'hybrid_xlstm']:
        if not XLSTM_AVAILABLE:
            raise ImportError("xLSTM modules not available. Please ensure xlstm_modules.py is in the networks folder.")
        return create_xlstm_mlp(config, mlp_type, lstm_hidden_size=lstm_hidden_size, **kwargs)
    else:
        raise ValueError(f"Unknown mlp_type: {mlp_type}")


# =============================================================================
# Transformer Block
# =============================================================================

class Block(nn.Module):
    """
    Transformer block supporting several feed-forward variants.
    """
    
    def __init__(self, config, vis, mlp_type='standard', lstm_hidden_size=None, **kwargs):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.mlp_type = mlp_type
        
        # build the requested feed-forward module
        self.ffn = create_mlp(config, mlp_type, lstm_hidden_size, **kwargs)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[f'{ROOT}/{ATTENTION_Q}/kernel']).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[f'{ROOT}/{ATTENTION_K}/kernel']).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[f'{ROOT}/{ATTENTION_V}/kernel']).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[f'{ROOT}/{ATTENTION_OUT}/kernel']).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[f'{ROOT}/{ATTENTION_Q}/bias']).view(-1)
            key_bias = np2th(weights[f'{ROOT}/{ATTENTION_K}/bias']).view(-1)
            value_bias = np2th(weights[f'{ROOT}/{ATTENTION_V}/bias']).view(-1)
            out_bias = np2th(weights[f'{ROOT}/{ATTENTION_OUT}/bias']).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[f'{ROOT}/{FC_0}/kernel']).t()
            mlp_weight_1 = np2th(weights[f'{ROOT}/{FC_1}/kernel']).t()
            mlp_bias_0 = np2th(weights[f'{ROOT}/{FC_0}/bias']).t()
            mlp_bias_1 = np2th(weights[f'{ROOT}/{FC_1}/bias']).t()

            # load pretrained weights according to the feed-forward variant
            if self.mlp_type == 'standard':
                self.ffn.fc1.weight.copy_(mlp_weight_0)
                self.ffn.fc1.bias.copy_(mlp_bias_0)
                self.ffn.fc2.weight.copy_(mlp_weight_1)
                self.ffn.fc2.bias.copy_(mlp_bias_1)
            elif self.mlp_type in ['hybrid', 'hybrid_slstm', 'hybrid_mlstm', 'hybrid_xlstm']:
                if hasattr(self.ffn, 'load_pretrained_fc'):
                    self.ffn.load_pretrained_fc(mlp_weight_0, mlp_bias_0, mlp_weight_1, mlp_bias_1)
                else:
                    self.ffn.fc1.weight.copy_(mlp_weight_0)
                    self.ffn.fc1.bias.copy_(mlp_bias_0)
            elif self.mlp_type in ['bilstm', 'slstm', 'mlstm', 'xlstm']:
                # for LSTM-based variants only fc1 can be loaded
                self.ffn.fc1.weight.copy_(mlp_weight_0)
                self.ffn.fc1.bias.copy_(mlp_bias_0)

            self.attention_norm.weight.copy_(np2th(weights[f'{ROOT}/{ATTENTION_NORM}/scale']))
            self.attention_norm.bias.copy_(np2th(weights[f'{ROOT}/{ATTENTION_NORM}/bias']))
            self.ffn_norm.weight.copy_(np2th(weights[f'{ROOT}/{MLP_NORM}/scale']))
            self.ffn_norm.bias.copy_(np2th(weights[f'{ROOT}/{MLP_NORM}/bias']))


# =============================================================================
# Encoder
# =============================================================================

class Encoder(nn.Module):
    def __init__(self, config, vis, mlp_type='standard', lstm_hidden_size=None, **kwargs):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis, mlp_type=mlp_type, lstm_hidden_size=lstm_hidden_size, **kwargs)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


# =============================================================================
# Embeddings
# =============================================================================

class Embeddings(nn.Module):
    def __init__(self, config, img_size, in_channels=3, use_cbam=True):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        self.use_cbam = use_cbam
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            if use_cbam:
                self.hybrid_model = ResNetV2WithCBAM(
                    block_units=config.resnet.num_layers,
                    width_factor=config.resnet.width_factor,
                    use_cbam=True
                )
            else:
                self.hybrid_model = ResNetV2(
                    block_units=config.resnet.num_layers,
                    width_factor=config.resnet.width_factor
                )
            in_channels = self.hybrid_model.width * 16

        self.patch_embeddings = Conv2d(
            in_channels=in_channels,
            out_channels=config.hidden_size,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
            
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


# =============================================================================
# Transformer
# =============================================================================

class Transformer(nn.Module):
    def __init__(self, config, img_size, vis, use_cbam=True, 
                 mlp_type='standard', lstm_hidden_size=None, **kwargs):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size, use_cbam=use_cbam)
        self.encoder = Encoder(config, vis, mlp_type=mlp_type, 
                              lstm_hidden_size=lstm_hidden_size, **kwargs)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights, features


# =============================================================================
# Decoder
# =============================================================================

class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, 
                 stride=1, use_batchnorm=True):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                        stride=stride, padding=padding, bias=not use_batchnorm)
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
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
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                          padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        
        self.conv_more = Conv2dReLU(
            config.hidden_size, head_channels,
            kernel_size=3, padding=1, use_batchnorm=True,
        )
        
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4 - self.config.n_skip):
                skip_channels[3-i] = 0
        else:
            skip_channels = [0, 0, 0, 0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) 
            for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
            
        return x


# =============================================================================
# Main Model
# =============================================================================

class VisionTransformer(nn.Module):
    """
    TransUNet with CBAM + BiLSTM/xLSTM.
    
    Supported feed-forward variants:
    - standard: the canonical TransUNet MLP
    - bilstm: BiLSTM trong MLP
    - hybrid: Hybrid BiLSTM + Standard
    - slstm: sLSTM (xLSTM variant)
    - mlstm: mLSTM (xLSTM variant)
    - xlstm: Alias cho mlstm
    - hybrid_slstm, hybrid_mlstm, hybrid_xlstm: Hybrid variants
    """
    
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False,
                 use_cbam=True, mlp_type='standard', lstm_hidden_size=None, **kwargs):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.use_cbam = use_cbam
        self.mlp_type = mlp_type
        
        self.transformer = Transformer(
            config, img_size, vis,
            use_cbam=use_cbam,
            mlp_type=mlp_type,
            lstm_hidden_size=lstm_hidden_size,
            **kwargs
        )
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, attn_weights, features = self.transformer(x)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits

    def load_from(self, weights):
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


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}
