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
PHASE 3 TRAIN SCRIPT FOR SeqAtt-UNet
====================================
Entry point for running training with all the Phase 3 improvements.

PHASE 3: ADVANCED TRAINING TECHNIQUES (expected +1-2% DSC)

3.1. Self-configuring, in the spirit of nnU-Net:
    --auto_config           : Auto-tune the hyperparameters from the dataset
    --auto_adjust_lr        : Linear scaling of batch size & LR
    
3.2. Pre-training Strategy:
    --scheduler             : LR scheduler (warmup_cosine/warmup_polynomial)
    --warmup_epochs         : Number of warmup epochs
    --layer_lr_decay        : Layer-wise LR decay factor
    --pretrained_path       : Path to pretrained weights
    
3.3. Deep Supervision:
    --deep_supervision      : Enable deep supervision
    --ds_weights            : Weights for the DS outputs (default: 1.0,0.4,0.2,0.1)

INHERITED FROM PHASE 1:
    - Combined Loss Function
    - Advanced Data Augmentation
    - CBAM Alpha Tracking
    - Test-Time Augmentation

Usage (Terminal):
    # Full Phase 3 with every feature enabled
    python train_phase3.py --dataset Synapse --use_cbam --mlp_type bilstm --deep_supervision --scheduler warmup_cosine --warmup_epochs 10 --seed 1234

    # Phase 3 with auto-config
    python train_phase3.py --dataset Synapse --auto_config --deep_supervision

    # Deep Supervision only (no auto-config)
    python train_phase3.py --dataset Synapse --deep_supervision --no_auto_config

Usage (Google Colab):
    !python train_phase3.py --dataset Synapse --use_cbam --mlp_type bilstm --deep_supervision --auto_config

Author: SeqAtt-UNet Project - Phase 3 Improvements
"""

import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

# Import Phase 3 trainer
from trainer_phase3 import (
    trainer_synapse_phase3, 
    trainer_acdc_phase3,
    PHASE3_CONFIG,
    LOSS_CONFIG
)

# Import models
from networks.vit_seg_modeling_xlstm import VisionTransformer, CONFIGS
from networks.vit_seg_modeling_phase3 import VisionTransformerDS, create_model_with_ds


def parse_args():
    parser = argparse.ArgumentParser(description='SeqAtt-UNet Phase 3 Training')
    
    # ========================
    # DATASET ARGUMENTS
    # ========================
    parser.add_argument('--dataset', type=str, default='Synapse',
                        choices=['Synapse', 'ACDC'],
                        help='Dataset name: Synapse or ACDC')
    parser.add_argument('--root_path', type=str, default=None,
                        help='Path to training data (auto-set if not provided)')
    parser.add_argument('--list_dir', type=str, default=None,
                        help='Path to list files (for Synapse)')
    parser.add_argument('--volume_path', type=str, default=None,
                        help='Path to test volumes (for inference)')
    
    # ========================
    # MODEL ARGUMENTS
    # ========================
    parser.add_argument('--num_classes', type=int, default=None,
                        help='Number of classes (auto-set from dataset)')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16',
                        help='ViT model name')
    parser.add_argument('--vit_patches_size', type=int, default=16,
                        help='ViT patch size')
    parser.add_argument('--n_skip', type=int, default=3,
                        help='Number of skip connections')
    
    # ========================
    # TRAINING ARGUMENTS
    # ========================
    parser.add_argument('--max_epochs', type=int, default=150,
                        help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=12,
                        help='Batch size per GPU')
    parser.add_argument('--base_lr', type=float, default=0.01,
                        help='Base learning rate')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed')
    parser.add_argument('--n_gpu', type=int, default=1,
                        help='Number of GPUs')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='Use deterministic mode')
    
    # ========================
    # PRETRAINED WEIGHTS
    # ========================
    parser.add_argument('--pretrained_path', type=str,
                        default='../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz',
                        help='Path to pretrained ViT weights')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')
    
    # ========================
    # PHASE 1 OPTIONS (inherited)
    # ========================
    parser.add_argument('--use_cbam', action='store_true', default=True,
                        help='Use CBAM attention')
    parser.add_argument('--no_cbam', action='store_false', dest='use_cbam',
                        help='Disable CBAM attention')
    parser.add_argument('--mlp_type', type=str, default='bilstm',
                        choices=['standard', 'bilstm', 'hybrid'],
                        help='MLP type in Transformer blocks')
    parser.add_argument('--augmentation', type=str, default='strong',
                        choices=['strong', 'medium', 'light'],
                        help='Augmentation strength')
    
    # ========================
    # PHASE 3 OPTIONS - 3.1 Self-Configuring
    # ========================
    parser.add_argument('--auto_config', action='store_true', default=True,
                        help='Enable auto-configuration from dataset analysis')
    parser.add_argument('--no_auto_config', action='store_false', dest='auto_config',
                        help='Disable auto-configuration')
    parser.add_argument('--auto_adjust_lr', action='store_true', default=True,
                        help='Auto-adjust LR based on batch size (linear scaling)')
    parser.add_argument('--no_auto_adjust_lr', action='store_false', dest='auto_adjust_lr',
                        help='Disable auto LR adjustment')
    
    # ========================
    # PHASE 3 OPTIONS - 3.2 Pre-training Strategy
    # ========================
    parser.add_argument('--scheduler', type=str, default='warmup_cosine',
                        choices=['warmup_cosine', 'warmup_polynomial'],
                        help='Learning rate scheduler type')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Number of warmup epochs')
    parser.add_argument('--layer_lr_decay', type=float, default=0.9,
                        help='Layer-wise LR decay factor (1.0 = disabled)')
    
    # ========================
    # PHASE 3 OPTIONS - 3.3 Deep Supervision
    # ========================
    parser.add_argument('--deep_supervision', action='store_true', default=True,
                        help='Enable deep supervision')
    parser.add_argument('--no_deep_supervision', action='store_false', dest='deep_supervision',
                        help='Disable deep supervision')
    parser.add_argument('--ds_weights', type=str, default='1.0,0.4,0.2,0.1',
                        help='Deep supervision weights (comma-separated)')
    parser.add_argument('--ds_start_epoch', type=int, default=0,
                        help='Epoch to start deep supervision')
    
    # ========================
    # LOSS CONFIGURATION
    # ========================
    parser.add_argument('--lambda_dice', type=float, default=1.0,
                        help='Weight for Dice loss')
    parser.add_argument('--lambda_ce', type=float, default=0.5,
                        help='Weight for CE loss')
    parser.add_argument('--lambda_boundary', type=float, default=0.1,
                        help='Weight for Boundary loss')
    parser.add_argument('--lambda_hd', type=float, default=0.1,
                        help='Weight for HD loss')
    parser.add_argument('--use_boundary', action='store_true', default=True,
                        help='Use boundary loss')
    parser.add_argument('--no_boundary', action='store_false', dest='use_boundary',
                        help='Disable boundary loss')
    parser.add_argument('--use_hd', action='store_true', default=True,
                        help='Use HD loss')
    parser.add_argument('--no_hd', action='store_false', dest='use_hd',
                        help='Disable HD loss')
    parser.add_argument('--boundary_start_epoch', type=int, default=50,
                        help='Epoch to start boundary/HD loss')
    
    args = parser.parse_args()
    
    # Parse DS weights
    args.ds_weights_tuple = tuple(float(x) for x in args.ds_weights.split(','))
    
    return args


def update_phase3_config(args):
    """
    Update PHASE3_CONFIG from the command line arguments.
    
    Allows the configuration to be adjusted flexibly through CLI arguments.
    """
    global PHASE3_CONFIG, LOSS_CONFIG
    
    # 3.1 Self-Configuring
    PHASE3_CONFIG['use_auto_config'] = args.auto_config
    PHASE3_CONFIG['auto_adjust_lr'] = args.auto_adjust_lr
    
    # 3.2 Pre-training Strategy
    PHASE3_CONFIG['scheduler_type'] = args.scheduler
    PHASE3_CONFIG['warmup_epochs'] = args.warmup_epochs
    PHASE3_CONFIG['layer_wise_lr_decay'] = args.layer_lr_decay
    
    # 3.3 Deep Supervision
    PHASE3_CONFIG['use_deep_supervision'] = args.deep_supervision
    PHASE3_CONFIG['ds_weights'] = args.ds_weights_tuple
    PHASE3_CONFIG['ds_start_epoch'] = args.ds_start_epoch
    
    # Loss config
    LOSS_CONFIG['lambda_dice'] = args.lambda_dice
    LOSS_CONFIG['lambda_ce'] = args.lambda_ce
    LOSS_CONFIG['lambda_boundary'] = args.lambda_boundary
    LOSS_CONFIG['lambda_hd'] = args.lambda_hd
    LOSS_CONFIG['use_boundary'] = args.use_boundary
    LOSS_CONFIG['use_hd'] = args.use_hd
    LOSS_CONFIG['boundary_start_epoch'] = args.boundary_start_epoch


def main():
    args = parse_args()
    
    # ========================
    # SET RANDOM SEED
    # ========================
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    # ========================
    # DATASET CONFIG
    # ========================
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
    
    config = dataset_config[args.dataset]
    args.num_classes = args.num_classes or config['num_classes']
    args.root_path = args.root_path or config['root_path']
    args.list_dir = args.list_dir or config['list_dir']
    args.volume_path = args.volume_path or config['volume_path']
    
    # ========================
    # VERIFY DATA EXISTS
    # ========================
    if not os.path.exists(args.root_path):
        print(f"\n❌ ERROR: Data path does not exist: {args.root_path}")
        print(f"\nExpected directory structure for {args.dataset}:")
        if args.dataset == 'Synapse':
            print("""
    data/
    └── Synapse/
        ├── train_npz/          # ← root_path should point here
        │   ├── case0005_slice000.npz
        │   └── ...
        └── test_vol_h5/
            └── ...
            """)
        else:
            print("""
    data/
    └── ACDC/                   # ← root_path should point here
        ├── ACDC_training_slices/
        │   └── ...
        └── ACDC_training_volumes/
            └── ...
            """)
        return
    
    # ========================
    # UPDATE PHASE 3 CONFIG
    # ========================
    update_phase3_config(args)
    
    # ========================
    # CREATE OUTPUT DIRECTORY
    # ========================
    exp_name = f'TU_{args.dataset}{args.img_size}'
    if args.use_cbam:
        exp_name += '_CBAM'
    if args.mlp_type != 'standard':
        exp_name += f'_{args.mlp_type.upper()}'
    exp_name += '_Phase3'
    
    # Add Phase 3 specific info
    if args.deep_supervision:
        exp_name += '_DS'
    if args.scheduler != 'warmup_cosine':
        exp_name += f'_{args.scheduler}'
    
    snapshot_path = os.path.join(
        '../model',
        exp_name,
        f"TU_pretrain_{args.vit_name}_skip{args.n_skip}_" +
        f"epo{args.max_epochs}_bs{args.batch_size}_aug{args.augmentation}_" +
        f"lr{args.base_lr}_{args.img_size}_s{args.seed}"
    )
    os.makedirs(snapshot_path, exist_ok=True)
    
    args.snapshot_path = snapshot_path
    args.exp_name = exp_name
    
    # ========================
    # PRINT CONFIGURATION
    # ========================
    print("=" * 80)
    print("🚀 SeqAtt-UNet PHASE 3 TRAINING - Advanced Training Techniques")
    print("=" * 80)
    print()
    print("📊 DATASET & MODEL:")
    print(f"   Dataset:        {args.dataset}")
    print(f"   Model:          {args.vit_name} + CBAM={args.use_cbam} + MLP={args.mlp_type}")
    print(f"   Input size:     {args.img_size}x{args.img_size}")
    print(f"   Classes:        {args.num_classes}")
    print()
    print("⚙️ PHASE 3 IMPROVEMENTS:")
    print()
    print("   [3.1] Self-Configuring:")
    print(f"         Auto-config:        {args.auto_config}")
    print(f"         Auto-adjust LR:     {args.auto_adjust_lr}")
    print()
    print("   [3.2] Pre-training Strategy:")
    print(f"         LR Scheduler:       {args.scheduler}")
    print(f"         Warmup epochs:      {args.warmup_epochs}")
    print(f"         Layer-wise decay:   {args.layer_lr_decay}")
    print()
    print("   [3.3] Deep Supervision:")
    print(f"         Enabled:            {args.deep_supervision}")
    print(f"         Weights:            {args.ds_weights_tuple}")
    print(f"         Start epoch:        {args.ds_start_epoch}")
    print()
    print("📈 TRAINING CONFIG:")
    print(f"   Epochs:         {args.max_epochs}")
    print(f"   Batch size:     {args.batch_size}")
    print(f"   Base LR:        {args.base_lr}")
    print(f"   Augmentation:   {args.augmentation}")
    print()
    print("🎯 LOSS FUNCTION:")
    print(f"   λ_dice:         {args.lambda_dice}")
    print(f"   λ_ce:           {args.lambda_ce}")
    print(f"   λ_boundary:     {args.lambda_boundary} (start epoch {args.boundary_start_epoch})")
    print(f"   λ_hd:           {args.lambda_hd}")
    print(f"   Boundary loss:  {args.use_boundary}")
    print(f"   HD loss:        {args.use_hd}")
    print()
    print("-" * 80)
    print(f"📁 Output: {snapshot_path}")
    print("=" * 80)
    print()
    
    # ========================
    # CREATE MODEL
    # ========================
    config_vit = CONFIGS[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size),
            int(args.img_size / args.vit_patches_size)
        )
    
    # Select the model matching the deep supervision option
    if args.deep_supervision:
        print("🔧 Using VisionTransformerDS model with Deep Supervision support")
        model = VisionTransformerDS(
            config_vit,
            img_size=args.img_size,
            num_classes=config_vit.n_classes,
            use_cbam=args.use_cbam,
            mlp_type=args.mlp_type,
            use_deep_supervision=True,
            num_ds_outputs=len(args.ds_weights_tuple)  # One DS output per weight
        )
    else:
        print("🔧 Using standard VisionTransformer model")
        model = VisionTransformer(
            config_vit,
            img_size=args.img_size,
            num_classes=config_vit.n_classes,
            use_cbam=args.use_cbam,
            mlp_type=args.mlp_type
        )
    
    # ========================
    # LOAD PRETRAINED WEIGHTS
    # ========================
    if args.resume:
        print(f"📥 Resuming from checkpoint: {args.resume}")
        model.load_state_dict(torch.load(args.resume, weights_only=True))
    elif os.path.exists(args.pretrained_path):
        print(f"📥 Loading pretrained weights from: {args.pretrained_path}")
        model.load_from(np.load(args.pretrained_path))
    else:
        print("⚠️ No pretrained weights found. Training from scratch.")
    
    # ========================
    # MOVE TO GPU
    # ========================
    model = model.cuda()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model Parameters:")
    print(f"   Total:     {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    
    # ========================
    # START TRAINING
    # ========================
    print("\n🏃 Starting Phase 3 training...\n")
    
    if args.dataset == 'Synapse':
        trainer_synapse_phase3(args, model, snapshot_path)
    elif args.dataset == 'ACDC':
        trainer_acdc_phase3(args, model, snapshot_path)
    
    print("\n" + "=" * 80)
    print("✅ Phase 3 Training completed!")
    print(f"📁 Models saved to: {snapshot_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
