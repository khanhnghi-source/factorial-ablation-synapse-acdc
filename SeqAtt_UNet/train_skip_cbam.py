# =============================================================================
# NOTICE OF MODIFICATION (Apache License 2.0, Section 4(b))
#
# This file is derived from TransUNet:
#     https://github.com/Beckschen/TransUNet -- train.py
#     Chen et al., "TransUNet: Transformers Make Strong Encoders for Medical
#     Image Segmentation," arXiv:2102.04306, 2021. Licensed under Apache-2.0.
#
# This file HAS BEEN MODIFIED by the authors of the present study. See NOTICE
# and THIRD_PARTY.md in the repository root, and the full licence text in
# LICENSE.
# =============================================================================
"""
TRAIN ENTRY POINT - SKIP-CBAM VARIANT (Phase 4 with α-LR boost)
================================================================
Wrapper script that trains SeqAtt-UNet with Skip-CBAM.

[NEW Phase 4 update]:
  --alpha_lr_multiplier N  Raise the learning rate of the alpha params
                            (CBAM + Skip-CBAM) to N times base_lr.
                            Default = 1.0 (no boost).
                            Recommended = 10.0 when alpha gets stuck around 0.10.

train_phase3.py / trainer_phase3.py are left untouched -- the script only
monkey-patches `get_layer_wise_lr_params` to add a separate param group for alpha.

USAGE:
    # Default (no α boost)
    python train_skip_cbam.py --dataset ACDC --use_cbam --mlp_type bilstm \\
        --deep_supervision --seed 1234 --skip_cbam_reduction 8

    # With a 10x alpha LR boost (option 3)
    python train_skip_cbam.py --dataset ACDC --use_cbam --mlp_type bilstm \\
        --deep_supervision --seed 1234 --skip_cbam_reduction 8 \\
        --alpha_lr_multiplier 10.0

Output dir naming:
    TU_<dataset>_CBAM_<mlp>_Phase3_DS_SkipCBAM/                  (mult=1.0)
    TU_<dataset>_CBAM_<mlp>_Phase3_DS_SkipCBAM_alpha10x/         (mult>1.0)

Author: SeqAtt-UNet Phase 4
"""

import argparse
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict

# Reuse all CLI parsing & config updates from train_phase3
from train_phase3 import parse_args, update_phase3_config

# Import trainer module (we'll monkey-patch a function inside it)
import trainer_phase3
from trainer_phase3 import (
    trainer_synapse_phase3, trainer_acdc_phase3,
    PHASE3_CONFIG, LOSS_CONFIG,
)

from networks.skip_cbam import create_skip_cbam_model
from networks.vit_seg_modeling_xlstm import CONFIGS


# =============================================================================
# MONKEY-PATCH: alpha-aware layer-wise LR with separate group for α params
# =============================================================================

_ORIGINAL_GET_LR = trainer_phase3.get_layer_wise_lr_params

# Default multiplier (overridden by CLI in main())
_ALPHA_LR_MULTIPLIER = 1.0


def alpha_aware_layer_wise_lr_params(model: nn.Module,
                                      base_lr: float,
                                      decay: float = 0.9) -> List[Dict]:
    """
    Build parameter groups with:
      - alpha params (`alpha_raw`): separate group with LR = base_lr * _ALPHA_LR_MULTIPLIER
      - Other params: layer-wise decay (same as the original)

    With _ALPHA_LR_MULTIPLIER = 1.0 the behaviour is IDENTICAL to the original
    (the only cost is one extra param group when alpha params exist).
    """
    # 1. Separate α params from rest
    alpha_params = []
    other_named = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'alpha_raw' in name:
            alpha_params.append((name, param))
        else:
            other_named.append((name, param))

    # 2. Apply layer-wise decay on non-α (replicate original logic from trainer_phase3.py)
    if decay == 1.0:
        non_alpha_groups = [{'params': [p for _, p in other_named], 'lr': base_lr}]
    else:
        depth_map = {}
        for name, param in other_named:
            if 'embeddings' in name or 'hybrid_model' in name:
                depth = 0
            elif 'encoder' in name:
                if 'layer.' in name:
                    try:
                        layer_num = int(name.split('layer.')[1].split('.')[0])
                        depth = 1 + layer_num
                    except Exception:
                        depth = 5
                else:
                    depth = 5
            elif 'decoder' in name:
                depth = 10
            elif 'segmentation_head' in name or 'ds_heads' in name:
                depth = 15
            else:
                depth = 8

            depth_map.setdefault(depth, []).append(param)

        max_depth = max(depth_map.keys()) if depth_map else 0
        non_alpha_groups = []
        for depth, params in sorted(depth_map.items()):
            lr_scale = decay ** (max_depth - depth)
            non_alpha_groups.append({
                'params': params,
                'lr': base_lr * lr_scale,
                'depth': depth,
            })

    # 3. α group (if any) at boosted LR
    groups = list(non_alpha_groups)
    if alpha_params:
        alpha_lr = base_lr * _ALPHA_LR_MULTIPLIER
        groups.append({
            'params':       [p for _, p in alpha_params],
            'lr':           alpha_lr,
            'name':         'alpha_params',
            'is_alpha':     True,
        })
        print(f"[ALPHA-LR] {len(alpha_params)} α params with LR = "
              f"{base_lr} × {_ALPHA_LR_MULTIPLIER} = {alpha_lr}")
        for n, _ in alpha_params:
            print(f"           - {n}")

    return groups


# =============================================================================
# MAIN
# =============================================================================

def main():
    global _ALPHA_LR_MULTIPLIER

    # Extra parser for Phase 4 specific flags (parse BEFORE main parser)
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument('--skip_cbam_reduction', type=int, default=8,
                              help='Reduction factor for SkipCBAM channel attention')
    extra_parser.add_argument('--alpha_lr_multiplier', type=float, default=1.0,
                              help='Multiplier for α params LR (default=1.0 disabled, '
                                   'recommended=10.0 if α stuck around 0.10)')
    extra_args, remaining = extra_parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    args = parse_args()
    args.skip_cbam_reduction = extra_args.skip_cbam_reduction
    args.alpha_lr_multiplier = extra_args.alpha_lr_multiplier

    # Apply α-LR multiplier via monkey-patch
    _ALPHA_LR_MULTIPLIER = float(args.alpha_lr_multiplier)
    trainer_phase3.get_layer_wise_lr_params = alpha_aware_layer_wise_lr_params

    if _ALPHA_LR_MULTIPLIER != 1.0:
        print(f"\n[CONFIG] α-LR multiplier ACTIVE: {_ALPHA_LR_MULTIPLIER}×")
    else:
        print(f"\n[CONFIG] α-LR multiplier disabled (= 1.0)")

    # Set seeds
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Dataset config
    dataset_config = {
        'Synapse': {
            'root_path': '../data/Synapse/train_npz',
            'list_dir': './lists/lists_Synapse',
            'volume_path': '../data/Synapse/test_vol_h5',
            'num_classes': 9,
        },
        'ACDC': {
            'root_path': '../data/ACDC',
            'list_dir': None,
            'volume_path': '../data/ACDC',
            'num_classes': 4,
        },
    }
    cfg = dataset_config[args.dataset]
    args.num_classes = args.num_classes or cfg['num_classes']
    args.root_path = args.root_path or cfg['root_path']
    args.list_dir = args.list_dir or cfg['list_dir']
    args.volume_path = args.volume_path or cfg['volume_path']

    if not os.path.exists(args.root_path):
        print(f'❌ Data path not found: {args.root_path}')
        return

    update_phase3_config(args)

    # Experiment name, with a suffix when the alpha-LR boost is used
    exp_name = f'TU_{args.dataset}{args.img_size}'
    if args.use_cbam:
        exp_name += '_CBAM'
    if args.mlp_type != 'standard':
        exp_name += f'_{args.mlp_type.upper()}'
    exp_name += '_Phase3'
    if args.deep_supervision:
        exp_name += '_DS'
    exp_name += '_SkipCBAM'
    if args.alpha_lr_multiplier != 1.0:
        # Encode multiplier in folder name to distinguish
        mult_str = f"{args.alpha_lr_multiplier:g}".replace('.', 'p')
        exp_name += f'_alpha{mult_str}x'

    snapshot_path = os.path.join(
        '../model', exp_name,
        f'TU_pretrain_{args.vit_name}_skip{args.n_skip}_'
        f'epo{args.max_epochs}_bs{args.batch_size}_aug{args.augmentation}_'
        f'lr{args.base_lr}_{args.img_size}_s{args.seed}'
    )
    os.makedirs(snapshot_path, exist_ok=True)
    args.snapshot_path = snapshot_path
    args.exp_name = exp_name

    print('=' * 80)
    print('🚀 SeqAtt-UNet PHASE 4 - Skip-CBAM Variant')
    print('=' * 80)
    print(f'Dataset:                {args.dataset}  Classes: {args.num_classes}')
    print(f'Model:                  R50-ViT-B/16 + Encoder-CBAM={args.use_cbam} + MLP={args.mlp_type}')
    print(f'                        + Skip-CBAM (reduction={args.skip_cbam_reduction})')
    print(f'                        + DS={args.deep_supervision} weights={args.ds_weights_tuple}')
    print(f'Alpha-LR multiplier:    {args.alpha_lr_multiplier}×')
    print(f'Epochs:                 {args.max_epochs}  Batch: {args.batch_size}  Base LR: {args.base_lr}')
    print(f'Output:                 {snapshot_path}')
    print('=' * 80)

    # Build model
    model = create_skip_cbam_model(
        config_name=args.vit_name,
        img_size=args.img_size,
        num_classes=args.num_classes,
        use_cbam=args.use_cbam,
        mlp_type=args.mlp_type,
        use_deep_supervision=args.deep_supervision,
        num_ds_outputs=len(args.ds_weights_tuple),
        skip_cbam_reduction=args.skip_cbam_reduction,
    )

    if hasattr(model, 'config'):
        model.config.n_skip = args.n_skip

    # Load pretrained
    if args.resume:
        print(f'📥 Resuming from: {args.resume}')
        model.load_state_dict(torch.load(args.resume, weights_only=True))
    elif os.path.exists(args.pretrained_path):
        print(f'📥 Loading pretrained: {args.pretrained_path}')
        model.load_from(np.load(args.pretrained_path))
    else:
        print('⚠️ No pretrained weights found.')

    model = model.cuda()
    n_params = sum(p.numel() for p in model.parameters())
    n_alpha = sum(1 for n, _ in model.named_parameters() if 'alpha_raw' in n)
    print(f'📊 Params total: {n_params:,}  ({n_alpha} α parameters)')

    # Train (uses monkey-patched alpha-aware LR)
    print('\n🏃 Starting Phase 4 training (Skip-CBAM)...\n')
    if args.dataset == 'Synapse':
        trainer_synapse_phase3(args, model, snapshot_path)
    else:
        trainer_acdc_phase3(args, model, snapshot_path)

    print('\n✅ Phase 4 training (Skip-CBAM) completed!')
    print(f'📁 Saved to: {snapshot_path}')


if __name__ == '__main__':
    main()
