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
PHASE 3 TEST SCRIPT FOR SeqAtt-UNet
==================================
Test the Phase 3 model with Deep Supervision support.
Deep Supervision is disabled automatically at inference time so the model
returns a single output.

Usage:
    # Option 1: resolve the model path automatically from the parameters (RECOMMENDED)
    python test_phase3.py --dataset Synapse --use_cbam --mlp_type bilstm --deep_supervision --seed 1234

    # Option 2: point to the model path directly
    python test_phase3.py --dataset Synapse --model_path ../model/.../best_model.pth --deep_supervision

    # With Test-Time Augmentation
    python test_phase3.py --dataset Synapse --use_cbam --mlp_type bilstm --deep_supervision --seed 1234 --tta simple

TTA Types:
    - none:   no TTA (fastest)
    - simple: four forward passes -- identity, horizontal flip, vertical flip,
              and both flips -- inverse-transformed and averaged. NO rotations:
              `RotationTTA` is a separate class that `simple` does not use.
              This is what every reported run used.
    - full:   8-transform TTA (not used for any reported result)

Author: SeqAtt-UNet Project - Phase 3 Testing
"""

import argparse
import logging
import os
import sys

# Windows CP1252 console cannot encode emoji / non-ASCII text -- force UTF-8
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import random
from scipy.ndimage import zoom
from tqdm import tqdm
from medpy import metric
import torch.nn.functional as F
import glob

# Import TTA
from inference_tta import SimpleTTA, FullTTA, get_tta


def build_model_path_phase3(args):
    """
    Build the model path automatically from the given parameters.
    Follows the naming format used by train_phase3.py
    """
    # Build experiment name (same convention as train_phase3.py)
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

    # [PATCH 2026-06-Phase4] Suffix for Skip-CBAM variant
    if getattr(args, 'use_skip_cbam', False):
        exp_name += '_SkipCBAM'
    
    # Locate the model directory
    model_dir = os.path.join('../model', exp_name)
    
    if not os.path.exists(model_dir):
        return None, exp_name, None
    
    # Find the sub-directory matching the requested batch_size and seed
    # Pattern: TU_pretrain_R50-ViT-B_16_skip3_epo{epochs}_bs{batch_size}_aug{aug}_lr{lr}_{img_size}_s{seed}
    pattern_exact = f"*_bs{args.batch_size}_*_s{args.seed}"
    subdirs = glob.glob(os.path.join(model_dir, pattern_exact))
    
    if not subdirs:
        # Fall back to matching the seed only when no exact match is found
        pattern_seed = f"*_s{args.seed}"
        subdirs = glob.glob(os.path.join(model_dir, pattern_seed))
        if subdirs:
            print(f"⚠️ No model found for bs{args.batch_size}, using the model with seed {args.seed}")
    
    if not subdirs:
        # Fall back to any available sub-directory
        subdirs = [d for d in glob.glob(os.path.join(model_dir, '*')) if os.path.isdir(d)]
        if subdirs:
            print(f"⚠️ No model found for seed {args.seed}, using the most recent one")
    
    if not subdirs:
        return None, exp_name, model_dir
    
    # Take the most recently modified directory
    snapshot_path = max(subdirs, key=os.path.getmtime)
    
    # Look for best_model.pth, otherwise the last epoch checkpoint
    best_model_path = os.path.join(snapshot_path, 'best_model.pth')
    
    if os.path.exists(best_model_path):
        return best_model_path, exp_name, snapshot_path
    
    # Look for the latest epoch checkpoint
    epoch_models = glob.glob(os.path.join(snapshot_path, 'epoch_*.pth'))
    if epoch_models:
        latest_epoch = max(epoch_models, key=lambda x: int(x.split('_')[-1].replace('.pth', '')))
        print(f"⚠️ best_model.pth not found, using {os.path.basename(latest_epoch)}")
        return latest_epoch, exp_name, snapshot_path
    
    return None, exp_name, snapshot_path


def calculate_metric_percase(pred, gt, voxelspacing=None):
    """
    Compute Dice and HD95 for a single case.

    [PATCH 2026-06] Added the voxelspacing parameter so HD95 can be measured in mm
    instead of voxels.

    Args:
        pred: Binary prediction mask
        gt: Binary ground-truth mask
        voxelspacing: tuple of floats, or None.
            - None -> HD95 in voxels (compatible with the original TransUNet code)
            - (sx, sy[, sz]) -> HD95 in mm

    Returns:
        Tuple (dice, hd95).
        IMPORTANT: hd95 is expressed in the unit implied by the voxelspacing passed in.
        To report both voxel and mm, call this function twice with different voxelspacing.
    """
    pred[pred > 0] = 1
    gt[gt > 0] = 1

    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt, voxelspacing=voxelspacing)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


# Default native voxel spacing (in-plane, mm) for the two benchmark datasets.
# Source:
#   - ACDC: Bernard et al., IEEE TMI 2018 (median 1.52 mm in-plane)
#   - Synapse: Landman et al., MICCAI 2015 (median 0.76 mm in-plane)
# If a specific dataset uses a different spacing, override it with --voxelspacing.
DEFAULT_VOXELSPACING_MM = {
    'Synapse': (0.76, 0.76),
    'ACDC':    (1.52, 1.52),
}


def get_voxelspacing_from_h5(filepath, fallback=None):
    """
    Read the voxelspacing from the HDF5 attributes (when available).

    ACDC/Synapse h5 files sometimes store the spacing in attrs under the keys:
      'spacing', 'pixdim', 'voxelspacing', 'pixel_size'

    Args:
        filepath: Path to the h5 file
        fallback: Default spacing used when nothing is found in attrs

    Returns:
        Spacing tuple, or the fallback value
    """
    import h5py
    keys_to_check = ['spacing', 'pixdim', 'voxelspacing', 'pixel_size',
                     'PixelSpacing', 'voxel_size']
    try:
        with h5py.File(filepath, 'r') as f:
            # Check file-level attrs
            for k in keys_to_check:
                if k in f.attrs:
                    sp = f.attrs[k]
                    if hasattr(sp, '__len__') and len(sp) >= 2:
                        return tuple(float(s) for s in sp[:2])
            # Check image-level attrs
            if 'image' in f:
                for k in keys_to_check:
                    if k in f['image'].attrs:
                        sp = f['image'].attrs[k]
                        if hasattr(sp, '__len__') and len(sp) >= 2:
                            return tuple(float(s) for s in sp[:2])
    except Exception:
        pass
    return fallback


def test_single_volume(image, label, model, classes, patch_size=[224, 224],
                       tta_type='none', test_save_path=None, case=None, z_spacing=1,
                       voxelspacing=None, report_both_units=True):
    """
    Run inference and evaluation on a single volume.

    [PATCH 2026-06] Added voxelspacing so HD95 can also be reported in mm.

    Args:
        ...
        voxelspacing: Native in-plane spacing (sx, sy) in mm.
            None -> return the voxel-based HD95 only.
            (sx, sy) -> also return HD95 in mm.
        report_both_units: If True and voxelspacing is not None, return an
            (n_classes-1, 3) array: [dice, hd95_voxel, hd95_mm].
            If False, return only (n_classes-1, 2): [dice, hd95_voxel or mm].

    Returns:
        metric_list: per-class metrics; the shape depends on report_both_units.
    """
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    
    # Get TTA object
    tta = get_tta(tta_type)
    
    prediction = np.zeros_like(label)
    
    for ind in range(image.shape[0]):
        slice_data = image[ind, :, :]
        x, y = slice_data.shape[0], slice_data.shape[1]
        
        if x != patch_size[0] or y != patch_size[1]:
            slice_data = zoom(slice_data, (patch_size[0] / x, patch_size[1] / y), order=3)
        
        input_tensor = torch.from_numpy(slice_data).unsqueeze(0).unsqueeze(0).float().cuda()
        input_tensor = input_tensor.repeat(1, 3, 1, 1)  # Grayscale -> 3 channels
        
        model.eval()
        with torch.no_grad():
            if tta is not None:
                outputs = tta(model, input_tensor)
            else:
                outputs = model(input_tensor)
            
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
        
        if x != patch_size[0] or y != patch_size[1]:
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        else:
            pred = out
        
        prediction[ind] = pred
    
    # Save the prediction when test_save_path is provided
    if test_save_path is not None and case is not None:
        import nibabel as nib
        img_itk = nib.Nifti1Image(prediction.astype(np.float32), np.eye(4))
        nib.save(img_itk, os.path.join(test_save_path, case + "_pred.nii.gz"))
    
    # Compute metrics for every class (class 1 to classes-1)
    # [PATCH] Report both HD95_voxel and HD95_mm for a fair comparison with the literature
    metric_list = []
    for i in range(1, classes):
        pred_mask = (prediction == i).astype(np.uint8)
        gt_mask = (label == i).astype(np.uint8)
        if report_both_units and voxelspacing is not None:
            dice, hd95_voxel = calculate_metric_percase(pred_mask.copy(), gt_mask.copy(),
                                                       voxelspacing=None)
            # Expand in-plane spacing to 3D if volume is 3D (slices, H, W)
            vs_mm = voxelspacing
            if pred_mask.ndim == 3 and len(vs_mm) == 2:
                vs_mm = (float(z_spacing),) + tuple(vs_mm)
            _, hd95_mm = calculate_metric_percase(pred_mask.copy(), gt_mask.copy(),
                                                 voxelspacing=vs_mm)
            metric_list.append((dice, hd95_voxel, hd95_mm))
        else:
            vs = voxelspacing
            if vs is not None and pred_mask.ndim == 3 and len(vs) == 2:
                vs = (float(z_spacing),) + tuple(vs)
            metric_list.append(calculate_metric_percase(pred_mask, gt_mask,
                                                      voxelspacing=vs))

    return metric_list


def inference_synapse(args, model, test_save_path=None, tta_type='none',
                      voxelspacing=None):
    """
    Run inference on the Synapse test set.

    [PATCH 2026-06] Added voxelspacing so HD95 is reported in mm alongside voxels.
    """
    from datasets.dataset_synapse import Synapse_dataset
    from torch.utils.data import DataLoader

    db_test = Synapse_dataset(
        base_dir=args.volume_path,
        list_dir=args.list_dir,
        split="test_vol"
    )

    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)

    # Synapse class names (8 organs, class 1-8)
    class_names = ['Aorta', 'Gallbladder', 'Kidney(L)', 'Kidney(R)',
                   'Liver', 'Pancreas', 'Spleen', 'Stomach']

    # [PATCH] Resolve voxelspacing by priority: args > h5 attrs > default
    if voxelspacing is None:
        voxelspacing = DEFAULT_VOXELSPACING_MM['Synapse']
    report_both = voxelspacing is not None

    logging.info(f"\n{'='*60}")
    logging.info(f"Testing on {len(db_test)} volumes")
    logging.info(f"TTA: {tta_type}")
    logging.info(f"Voxelspacing (mm): {voxelspacing} | report_both_units={report_both}")
    logging.info(f"{'='*60}")

    model.eval()
    n_metric_cols = 3 if report_both else 2
    metric_list = np.zeros((args.num_classes - 1, n_metric_cols))

    for i_batch, sampled_batch in tqdm(enumerate(testloader), total=len(testloader), desc="Testing"):
        image = sampled_batch['image']
        label = sampled_batch['label']
        case_name = sampled_batch['case_name'][0]

        # Try to read the per-case voxelspacing from h5 (if present), else use the default
        case_fp = os.path.join(args.volume_path, f"{case_name}.npy.h5")
        case_vs = get_voxelspacing_from_h5(case_fp, fallback=voxelspacing)

        metric_i = test_single_volume(
            image, label, model,
            classes=args.num_classes,
            patch_size=[args.img_size, args.img_size],
            tta_type=tta_type,
            test_save_path=test_save_path,
            case=case_name,
            voxelspacing=case_vs,
            report_both_units=report_both
        )

        metric_list += np.array(metric_i)

        case_dice = np.mean([m[0] for m in metric_i])
        if report_both:
            case_hd_v = np.mean([m[1] for m in metric_i])
            case_hd_mm = np.mean([m[2] for m in metric_i])
            logging.info(f'{case_name}: DSC={case_dice:.4f}, '
                         f'HD95_vox={case_hd_v:.2f}, HD95_mm={case_hd_mm:.2f}')
        else:
            case_hd = np.mean([m[1] for m in metric_i])
            logging.info(f'{case_name}: DSC={case_dice:.4f}, HD95={case_hd:.2f}')

    metric_list = metric_list / len(db_test)

    logging.info(f"\n{'='*60}")
    logging.info("PER-CLASS RESULTS:")
    logging.info(f"{'='*60}")

    for i in range(args.num_classes - 1):
        class_name = class_names[i] if i < len(class_names) else f'Class{i+1}'
        if report_both:
            logging.info(f'{class_name:15s}: DSC={metric_list[i][0]:.4f}, '
                         f'HD95_vox={metric_list[i][1]:.2f}, HD95_mm={metric_list[i][2]:.2f}')
        else:
            logging.info(f'{class_name:15s}: DSC={metric_list[i][0]:.4f}, '
                         f'HD95={metric_list[i][1]:.2f}')

    mean_dice = np.mean(metric_list, axis=0)[0]
    mean_hd95_voxel = np.mean(metric_list, axis=0)[1]
    mean_hd95_mm = np.mean(metric_list, axis=0)[2] if report_both else None

    return mean_dice, mean_hd95_voxel, mean_hd95_mm


def inference_acdc(args, model, test_save_path=None, tta_type='none',
                   voxelspacing=None):
    """
    Run inference on the ACDC test set.

    [PATCH 2026-06] Added voxelspacing -> HD95 is reported in mm alongside voxels.
    """
    from datasets.dataset_acdc import ACDC_dataset
    from torch.utils.data import DataLoader

    db_test = ACDC_dataset(
        base_dir=args.volume_path,
        split='test'
    )

    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)

    class_names = ['RV', 'Myo', 'LV']

    if voxelspacing is None:
        voxelspacing = DEFAULT_VOXELSPACING_MM['ACDC']
    report_both = voxelspacing is not None

    logging.info(f"\n{'='*60}")
    logging.info(f"Testing on {len(db_test)} samples")
    logging.info(f"TTA: {tta_type}")
    logging.info(f"Voxelspacing (mm): {voxelspacing} | report_both_units={report_both}")
    logging.info(f"{'='*60}")

    model.eval()
    n_metric_cols = 3 if report_both else 2
    metric_list = np.zeros((args.num_classes - 1, n_metric_cols))

    for i_batch, sampled_batch in tqdm(enumerate(testloader), total=len(testloader), desc="Testing"):
        image = sampled_batch['image']
        label = sampled_batch['label']
        case_name = sampled_batch.get('case_name', [f'case_{i_batch}'])[0]

        # Read the voxelspacing from the h5 attrs when available
        case_fp = os.path.join(args.volume_path, 'ACDC_training_volumes',
                               f"{case_name}.h5")
        case_vs = get_voxelspacing_from_h5(case_fp, fallback=voxelspacing)

        metric_i = test_single_volume(
            image, label, model,
            classes=args.num_classes,
            patch_size=[args.img_size, args.img_size],
            tta_type=tta_type,
            test_save_path=test_save_path,
            case=case_name,
            voxelspacing=case_vs,
            report_both_units=report_both
        )

        metric_list += np.array(metric_i)

        case_dice = np.mean([m[0] for m in metric_i])
        if report_both:
            case_hd_v = np.mean([m[1] for m in metric_i])
            case_hd_mm = np.mean([m[2] for m in metric_i])
            logging.info(f'{case_name}: DSC={case_dice:.4f}, '
                         f'HD95_vox={case_hd_v:.2f}, HD95_mm={case_hd_mm:.2f}')
        else:
            case_hd = np.mean([m[1] for m in metric_i])
            logging.info(f'{case_name}: DSC={case_dice:.4f}, HD95={case_hd:.2f}')

    metric_list = metric_list / len(db_test)

    logging.info(f"\n{'='*60}")
    logging.info("PER-CLASS RESULTS:")
    logging.info(f"{'='*60}")

    for i in range(args.num_classes - 1):
        class_name = class_names[i] if i < len(class_names) else f'Class{i+1}'
        if report_both:
            logging.info(f'{class_name:15s}: DSC={metric_list[i][0]:.4f}, '
                         f'HD95_vox={metric_list[i][1]:.2f}, HD95_mm={metric_list[i][2]:.2f}')
        else:
            logging.info(f'{class_name:15s}: DSC={metric_list[i][0]:.4f}, '
                         f'HD95={metric_list[i][1]:.2f}')

    mean_dice = np.mean(metric_list, axis=0)[0]
    mean_hd95_voxel = np.mean(metric_list, axis=0)[1]
    mean_hd95_mm = np.mean(metric_list, axis=0)[2] if report_both else None

    return mean_dice, mean_hd95_voxel, mean_hd95_mm


def log_cbam_alphas(model, logger):
    """Log CBAM alpha values."""
    alphas = []
    for name, param in model.named_parameters():
        if 'cbam' in name.lower() and 'alpha' in name.lower():
            alphas.append((name, param.item()))
    
    if alphas:
        logger.info("\n📊 CBAM Alpha Values:")
        for name, val in alphas:
            logger.info(f"   {name}: {val:.4f}")


def main():
    parser = argparse.ArgumentParser(description='SeqAtt-UNet Phase 3 Testing')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='Synapse', choices=['Synapse', 'ACDC'])
    parser.add_argument('--volume_path', type=str, default=None)
    parser.add_argument('--list_dir', type=str, default=None)
    parser.add_argument('--num_classes', type=int, default=None)
    
    # Model
    parser.add_argument('--model_path', type=str, default=None, help='Direct path to model weights')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16')
    parser.add_argument('--vit_patches_size', type=int, default=16)
    parser.add_argument('--n_skip', type=int, default=3)
    
    # Phase 3 specific - must match the training configuration
    parser.add_argument('--use_cbam', action='store_true', help='Use CBAM attention')
    parser.add_argument('--mlp_type', type=str, default='standard',
                        choices=['standard', 'bilstm', 'convlstm', 'xlstm', 'hybrid'])
    parser.add_argument('--deep_supervision', action='store_true', help='Model trained with Deep Supervision')
    parser.add_argument('--scheduler', type=str, default='warmup_cosine',
                        choices=['warmup_cosine', 'warmup_polynomial'])

    # [PATCH 2026-06-Phase4] Skip-CBAM variant (Phase 4)
    parser.add_argument('--use_skip_cbam', action='store_true',
                        help='Use VisionTransformerSkipCBAM (Phase 4) instead of VisionTransformerDS')
    parser.add_argument('--skip_cbam_reduction', type=int, default=8,
                        help='Reduction factor for Skip-CBAM channel attention')
    
    # Training params (for auto model path)
    parser.add_argument('--max_epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--base_lr', type=float, default=0.01)
    parser.add_argument('--augmentation', type=str, default='advanced')
    parser.add_argument('--seed', type=int, default=1234)
    
    # Test options
    parser.add_argument('--tta', type=str, default='none',
                        choices=['none', 'simple', 'full'],
                        help='Test-Time Augmentation type')
    parser.add_argument('--is_savenii', action='store_true', help='Save predictions as NIfTI')
    parser.add_argument('--test_save_dir', type=str, default='../predictions')

    # [PATCH 2026-06] HD95 unit handling
    parser.add_argument('--voxelspacing', type=str, default=None,
                        help='In-plane voxel spacing (mm), comma-separated, '
                             'e.g. "1.52,1.52" for ACDC, "0.76,0.76" for Synapse. '
                             'If None, the default from DEFAULT_VOXELSPACING_MM is used. '
                             'Set "voxel" to report in voxels only (no mm).')
    
    # Other
    parser.add_argument('--deterministic', type=int, default=1)
    
    args = parser.parse_args()
    
    # Deterministic
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    
    # Auto-set dataset parameters
    if args.dataset == 'Synapse':
        args.num_classes = args.num_classes or 9
        args.volume_path = args.volume_path or '../data/Synapse/test_vol_h5'
        args.list_dir = args.list_dir or './lists/lists_Synapse'
    elif args.dataset == 'ACDC':
        args.num_classes = args.num_classes or 4
        args.volume_path = args.volume_path or '../data/ACDC'
    
    # ========================
    # FIND MODEL PATH
    # ========================
    exp_name = None
    snapshot_path = None
    
    if args.model_path is None:
        args.model_path, exp_name, snapshot_path = build_model_path_phase3(args)
        
        if args.model_path is None:
            print("\n❌ ERROR: Model file not found!")
            print("\nOptions:")
            print("1. Check that training has completed successfully")
            print("2. Specify model path directly:")
            print(f"   python test_phase3.py --dataset {args.dataset} --model_path ../model/.../best_model.pth --deep_supervision")
            
            if snapshot_path:
                print(f"\nExpected model directory: {snapshot_path}")
                if os.path.exists(snapshot_path):
                    files = os.listdir(snapshot_path)
                    print(f"Files found: {files}")
            sys.exit(1)
    else:
        # Extract exp_name from path
        parts = args.model_path.split(os.sep)
        for p in parts:
            if p.startswith('TU_'):
                exp_name = p
                break
        if exp_name is None:
            exp_name = f'TU_{args.dataset}{args.img_size}_Phase3'
    
    if not os.path.exists(args.model_path):
        print(f"\n❌ ERROR: Model file does not exist: {args.model_path}")
        sys.exit(1)
    
    # ========================
    # SETUP LOGGING
    # ========================
    log_folder = f'./test_log/test_log_{exp_name}'
    os.makedirs(log_folder, exist_ok=True)
    
    snapshot_name = os.path.basename(os.path.dirname(args.model_path))
    log_filename = f'{snapshot_name}_tta_{args.tta}.txt'
    
    logging.basicConfig(
        filename=os.path.join(log_folder, log_filename),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    logging.info("=" * 60)
    logging.info("🧪 SeqAtt-UNet PHASE 3 TESTING")
    logging.info("=" * 60)
    logging.info(f"Dataset:          {args.dataset}")
    logging.info(f"Model:            {args.model_path}")
    logging.info(f"Deep Supervision: {args.deep_supervision}")
    logging.info(f"CBAM:             {args.use_cbam}")
    logging.info(f"MLP Type:         {args.mlp_type}")
    logging.info(f"TTA:              {args.tta}")
    logging.info(f"Classes:          {args.num_classes}")
    logging.info("=" * 60)
    
    # ========================
    # LOAD MODEL
    # ========================
    from networks.vit_seg_modeling_xlstm import CONFIGS
    
    config_vit = CONFIGS[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size),
            int(args.img_size / args.vit_patches_size)
        )
    
    # Select the matching model class
    if args.deep_supervision:
        # [PATCH 2026-06-Phase4] Detect Phase 4 (Skip-CBAM) vs Phase 3
        if args.use_skip_cbam:
            from networks.skip_cbam import VisionTransformerSkipCBAM
            logging.info("→ Building VisionTransformerSkipCBAM (Phase 4)")
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
        else:
            from networks.vit_seg_modeling_phase3 import VisionTransformerDS
            logging.info("→ Building VisionTransformerDS (Phase 3)")
            model = VisionTransformerDS(
                config_vit,
                img_size=args.img_size,
                num_classes=config_vit.n_classes,
                use_cbam=args.use_cbam,
                mlp_type=args.mlp_type,
                use_deep_supervision=True
            ).cuda()

        # IMPORTANT: disable deep supervision at inference time
        model.disable_deep_supervision()
        logging.info("✓ Deep Supervision DISABLED for inference (single output)")
    else:
        from networks.vit_seg_modeling_xlstm import VisionTransformer

        model = VisionTransformer(
            config_vit,
            img_size=args.img_size,
            num_classes=config_vit.n_classes,
            use_cbam=args.use_cbam,
            mlp_type=args.mlp_type
        ).cuda()
    
    # Load weights
    state_dict = torch.load(args.model_path)
    model.load_state_dict(state_dict)
    logging.info(f"✓ Loaded model from {args.model_path}")
    
    # Log CBAM alphas
    if args.use_cbam:
        log_cbam_alphas(model, logging)
    
    # ========================
    # SETUP SAVE PATH
    # ========================
    test_save_path = None
    if args.is_savenii:
        test_save_path = os.path.join(
            args.test_save_dir,
            f'{exp_name}_{snapshot_name}_tta_{args.tta}'
        )
        os.makedirs(test_save_path, exist_ok=True)
        logging.info(f"📁 Predictions will be saved to: {test_save_path}")
    
    # ========================
    # [PATCH 2026-06] Parse voxelspacing arg
    # ========================
    if args.voxelspacing is None:
        vs = DEFAULT_VOXELSPACING_MM.get(args.dataset)
    elif args.voxelspacing.lower() == 'voxel':
        vs = None  # report in voxels only
    else:
        vs = tuple(float(x) for x in args.voxelspacing.split(','))

    logging.info(f"📐 HD95 voxelspacing setting: {vs}")
    logging.info(f"    → Default for {args.dataset}: "
                 f"{DEFAULT_VOXELSPACING_MM.get(args.dataset)}")

    # ========================
    # RUN INFERENCE
    # ========================
    if args.dataset == 'Synapse':
        mean_dice, mean_hd95_voxel, mean_hd95_mm = inference_synapse(
            args, model,
            test_save_path=test_save_path,
            tta_type=args.tta,
            voxelspacing=vs,
        )
    else:
        mean_dice, mean_hd95_voxel, mean_hd95_mm = inference_acdc(
            args, model,
            test_save_path=test_save_path,
            tta_type=args.tta,
            voxelspacing=vs,
        )

    # ========================
    # FINAL RESULTS
    # ========================
    logging.info("\n" + "=" * 60)
    logging.info("📊 FINAL RESULTS")
    logging.info("=" * 60)
    logging.info(f"  Mean Dice:        {mean_dice:.4f} ({mean_dice*100:.2f}%)")
    logging.info(f"  Mean HD95 (voxel): {mean_hd95_voxel:.4f}")
    if mean_hd95_mm is not None:
        logging.info(f"  Mean HD95 (mm):    {mean_hd95_mm:.4f}  "
                     f"[voxelspacing={vs}]")
    logging.info(f"  TTA:              {args.tta}")
    logging.info("=" * 60)

    if test_save_path:
        logging.info(f"  Predictions saved to: {test_save_path}")

    logging.info(f"\n📝 Log saved to: {os.path.join(log_folder, log_filename)}")

    return mean_dice, mean_hd95_voxel, mean_hd95_mm


if __name__ == "__main__":
    main()
