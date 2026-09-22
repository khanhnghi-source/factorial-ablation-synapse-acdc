# =============================================================================
# NOTICE OF MODIFICATION (Apache License 2.0, Section 4(b))
#
# This file is derived from TransUNet:
#     https://github.com/Beckschen/TransUNet -- test.py and utils.py
#     Chen et al., "TransUNet: Transformers Make Strong Encoders for Medical
#     Image Segmentation," arXiv:2102.04306, 2021. Licensed under Apache-2.0.
#
# This file HAS BEEN MODIFIED by the authors of the present study. See NOTICE
# and THIRD_PARTY.md in the repository root, and the full licence text in
# LICENSE.
# =============================================================================
"""
TEST SCRIPT FOR PHASE 4 SKIP-CBAM MODELS
==========================================
Standalone test script that loads VisionTransformerSkipCBAM checkpoints.
Reuses inference utilities from test_phase3.py but builds the correct model class.

100% compatible with the output format of test_phase3.py (per-patient + per-class +
FINAL RESULTS with HD95 in voxel + mm), and therefore with aggregate_results.py.

USAGE:
    python test_skip_cbam.py --dataset ACDC --seed 1234 --tta simple \\
        --model_path /path/to/best_model.pth

    # Or auto-discover the checkpoint from the Drive folder structure:
    python test_skip_cbam.py --dataset ACDC --use_cbam --mlp_type bilstm \\
        --deep_supervision --seed 1234 --tta simple

Author: SeqAtt-UNet Phase 4
"""

import argparse
import logging
import os
import sys
import glob
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

# Reuse inference utilities from test_phase3.py
from test_phase3 import (
    calculate_metric_percase,
    test_single_volume,
    inference_synapse,
    inference_acdc,
    DEFAULT_VOXELSPACING_MM,
    log_cbam_alphas,
)


def build_model_path_skip_cbam(args):
    """Find model path on Drive for Skip-CBAM variant."""
    exp_name = f'TU_{args.dataset}{args.img_size}'
    if args.use_cbam:
        exp_name += '_CBAM'
    if args.mlp_type != 'standard':
        exp_name += f'_{args.mlp_type.upper()}'
    exp_name += '_Phase3'
    if args.deep_supervision:
        exp_name += '_DS'
    exp_name += '_SkipCBAM'   # ★ Phase 4 suffix

    model_dir = os.path.join('../model', exp_name)
    if not os.path.exists(model_dir):
        return None, exp_name, None

    pattern = f"*_bs{args.batch_size}_*_s{args.seed}"
    subdirs = glob.glob(os.path.join(model_dir, pattern))
    if not subdirs:
        subdirs = glob.glob(os.path.join(model_dir, f"*_s{args.seed}"))
    if not subdirs:
        subdirs = [d for d in glob.glob(os.path.join(model_dir, '*')) if os.path.isdir(d)]

    if not subdirs:
        return None, exp_name, model_dir

    snap = max(subdirs, key=os.path.getmtime)
    best = os.path.join(snap, 'best_model.pth')
    if os.path.exists(best):
        return best, exp_name, snap

    epoch_models = glob.glob(os.path.join(snap, 'epoch_*.pth'))
    if epoch_models:
        latest = max(epoch_models, key=lambda x: int(x.split('_')[-1].replace('.pth', '')))
        return latest, exp_name, snap
    return None, exp_name, snap


def main():
    parser = argparse.ArgumentParser(description="Test SeqAtt-UNet Phase 4 Skip-CBAM")

    parser.add_argument('--dataset', type=str, default='ACDC', choices=['Synapse', 'ACDC'])
    parser.add_argument('--volume_path', type=str, default=None)
    parser.add_argument('--list_dir', type=str, default=None)
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16')
    parser.add_argument('--vit_patches_size', type=int, default=16)
    parser.add_argument('--n_skip', type=int, default=3)

    parser.add_argument('--use_cbam', action='store_true')
    parser.add_argument('--mlp_type', type=str, default='bilstm',
                        choices=['standard', 'bilstm', 'hybrid'])
    parser.add_argument('--deep_supervision', action='store_true', default=True)
    parser.add_argument('--skip_cbam_reduction', type=int, default=8)

    parser.add_argument('--max_epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--base_lr', type=float, default=0.01)
    parser.add_argument('--augmentation', type=str, default='strong')
    parser.add_argument('--seed', type=int, default=1234)

    parser.add_argument('--tta', type=str, default='simple',
                        choices=['none', 'simple', 'full'])
    parser.add_argument('--is_savenii', action='store_true')
    parser.add_argument('--test_save_dir', type=str, default='../predictions')
    parser.add_argument('--voxelspacing', type=str, default=None)
    parser.add_argument('--deterministic', type=int, default=1)
    parser.add_argument('--model_path', type=str, default=None)

    args = parser.parse_args()

    # Determinism
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # Auto dataset defaults
    if args.dataset == 'Synapse':
        args.num_classes = args.num_classes or 9
        args.volume_path = args.volume_path or '../data/Synapse/test_vol_h5'
        args.list_dir = args.list_dir or './lists/lists_Synapse'
    else:
        args.num_classes = args.num_classes or 4
        args.volume_path = args.volume_path or '../data/ACDC'

    # ---- Find model path ----
    exp_name = None
    snapshot_path = None
    if args.model_path is None:
        args.model_path, exp_name, snapshot_path = build_model_path_skip_cbam(args)
        if args.model_path is None:
            print("\n❌ Model file not found.")
            print(f"Expected dir: {snapshot_path}")
            sys.exit(1)
    else:
        parts = args.model_path.split(os.sep)
        for p in parts:
            if p.startswith('TU_'):
                exp_name = p
                break
        exp_name = exp_name or f'TU_{args.dataset}_Phase4_SkipCBAM'

    if not os.path.exists(args.model_path):
        print(f"\n❌ Model file does not exist: {args.model_path}")
        sys.exit(1)

    # ---- Logging ----
    log_folder = f'./test_log/test_log_{exp_name}'
    os.makedirs(log_folder, exist_ok=True)
    snapshot_name = os.path.basename(os.path.dirname(args.model_path))
    log_filename = f'{snapshot_name}_tta_{args.tta}.txt'
    logging.basicConfig(
        filename=os.path.join(log_folder, log_filename),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S',
        force=True,
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info("=" * 60)
    logging.info("🧪 SeqAtt-UNet PHASE 4 (Skip-CBAM) TESTING")
    logging.info("=" * 60)
    logging.info(f"Dataset:                {args.dataset}")
    logging.info(f"Model (Skip-CBAM):      {args.model_path}")
    logging.info(f"CBAM:                   {args.use_cbam}")
    logging.info(f"MLP Type:               {args.mlp_type}")
    logging.info(f"Skip-CBAM reduction:    {args.skip_cbam_reduction}")
    logging.info(f"TTA:                    {args.tta}")
    logging.info("=" * 60)

    # ---- BUILD VisionTransformerSkipCBAM (correct class for Phase 4) ----
    from networks.vit_seg_modeling_xlstm import CONFIGS
    from networks.skip_cbam import VisionTransformerSkipCBAM

    config_vit = CONFIGS[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if 'R50' in args.vit_name:
        config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size),
            int(args.img_size / args.vit_patches_size)
        )

    model = VisionTransformerSkipCBAM(
        config_vit,
        img_size=args.img_size,
        num_classes=config_vit.n_classes,
        use_cbam=args.use_cbam,
        mlp_type=args.mlp_type,
        use_deep_supervision=True,
        num_ds_outputs=4,
        skip_cbam_reduction=args.skip_cbam_reduction,
    ).cuda()
    model.disable_deep_supervision()
    logging.info("✓ Built VisionTransformerSkipCBAM with DS DISABLED for inference")

    # Load weights
    state_dict = torch.load(args.model_path, weights_only=False)
    model.load_state_dict(state_dict)
    logging.info(f"✓ Loaded checkpoint")

    if args.use_cbam:
        log_cbam_alphas(model, logging)

    # Skip-CBAM α tracking
    if hasattr(model, 'get_all_alphas'):
        alphas = model.get_all_alphas()
        logging.info(f"\n[Skip-CBAM α values after training]")
        for name, val in alphas.items():
            logging.info(f"   {name}: {val:.4f}")

    # Save path
    test_save_path = None
    if args.is_savenii:
        test_save_path = os.path.join(args.test_save_dir,
                                       f'{exp_name}_{snapshot_name}_tta_{args.tta}')
        os.makedirs(test_save_path, exist_ok=True)

    # Voxelspacing
    if args.voxelspacing is None:
        vs = DEFAULT_VOXELSPACING_MM.get(args.dataset)
    elif args.voxelspacing.lower() == 'voxel':
        vs = None
    else:
        vs = tuple(float(x) for x in args.voxelspacing.split(','))
    logging.info(f"📐 HD95 voxelspacing: {vs}")

    # ---- RUN INFERENCE ----
    if args.dataset == 'Synapse':
        mean_dice, mean_hd95_voxel, mean_hd95_mm = inference_synapse(
            args, model, test_save_path=test_save_path,
            tta_type=args.tta, voxelspacing=vs,
        )
    else:
        mean_dice, mean_hd95_voxel, mean_hd95_mm = inference_acdc(
            args, model, test_save_path=test_save_path,
            tta_type=args.tta, voxelspacing=vs,
        )

    logging.info("\n" + "=" * 60)
    logging.info("📊 FINAL RESULTS (Phase 4 Skip-CBAM)")
    logging.info("=" * 60)
    logging.info(f"  Mean Dice:         {mean_dice:.4f} ({mean_dice*100:.2f}%)")
    logging.info(f"  Mean HD95 (voxel): {mean_hd95_voxel:.4f}")
    if mean_hd95_mm is not None:
        logging.info(f"  Mean HD95 (mm):    {mean_hd95_mm:.4f}  [voxelspacing={vs}]")
    logging.info(f"  TTA:               {args.tta}")
    logging.info("=" * 60)

    return mean_dice, mean_hd95_voxel, mean_hd95_mm


if __name__ == "__main__":
    main()
