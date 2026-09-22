"""
VERIFY HD95 UNIT - Voxel vs Millimeter
=======================================
Checks a critical bug: does medpy.metric.binary.hd95 return voxels or millimeters?

SUSPECTED ROOT CAUSE:
  test_phase3.py:115 calls `metric.binary.hd95(pred, gt)` WITHOUT passing voxelspacing
  → per the medpy docs: "If not specified, a grid spacing of unity is implied"
  → the value is therefore expressed in VOXEL UNITS, not in mm as the paper claims.

USAGE:
  python verify_hd95_unit.py --check_synthetic
  python verify_hd95_unit.py --check_acdc  --acdc_path /path/to/data/ACDC
  python verify_hd95_unit.py --check_synapse --synapse_path /path/to/data/Synapse/test_vol_h5

OUTPUT:
  - Prints HD95 in voxels vs HD95 in mm for the same prediction
  - Reports the native voxel spacing of each dataset
  - Proposes a correction factor

Author: SeqAtt-UNet Phase 3 - HD95 Verification
"""
import argparse
import os
import glob
import numpy as np
from medpy import metric


def demo_synthetic():
    """Demonstrate the unit issue on a synthetic shape offset by 5 pixels."""
    print("=" * 70)
    print("[A] SYNTHETIC TEST -- two squares exactly 5 voxels apart")
    print("=" * 70)

    gt = np.zeros((50, 50), dtype=np.uint8)
    pred = np.zeros((50, 50), dtype=np.uint8)
    gt[20:30, 20:30] = 1
    pred[20:30, 25:35] = 1  # shifted by 5 voxels along the x axis

    cases = [
        ("voxelspacing=None (the default used in test_phase3.py)", None),
        ("voxelspacing=(1.0, 1.0)", (1.0, 1.0)),
        ("voxelspacing=(1.52, 1.52) - ACDC native ~1.5mm", (1.52, 1.52)),
        ("voxelspacing=(0.76, 0.76) - Synapse CT ~0.76mm", (0.76, 0.76)),
    ]
    for desc, vs in cases:
        hd = metric.binary.hd95(pred, gt, voxelspacing=vs)
        unit = "VOXEL" if vs is None else "mm"
        print(f"  {desc:55s} → HD95 = {hd:6.3f} [{unit}]")
    print()
    print("  CONCLUSION: with voxelspacing=None the distance is measured in VOXELS.")
    print("  To obtain mm, the dataset's true voxelspacing MUST be passed.\n")


def check_acdc(acdc_path):
    """Read ACDC volumes to recover their native voxel spacing."""
    print("=" * 70)
    print("[B] CHECK ACDC NATIVE VOXEL SPACING")
    print("=" * 70)

    import h5py
    vol_dir = os.path.join(acdc_path, "ACDC_training_volumes")
    if not os.path.isdir(vol_dir):
        vol_dir = acdc_path  # fallback

    h5_files = sorted(glob.glob(os.path.join(vol_dir, "*.h5")))[:5]
    if not h5_files:
        print(f"  ❌ No *.h5 file found in {vol_dir}")
        return

    for fp in h5_files:
        try:
            with h5py.File(fp, "r") as f:
                keys = list(f.keys())
                shape = f["image"].shape if "image" in f else None
                # ACDC h5 files usually do not store spacing, so inspect attrs
                attrs = dict(f.attrs) if hasattr(f, "attrs") else {}
                # Check attrs from image dataset
                if "image" in f:
                    img_attrs = dict(f["image"].attrs)
                    attrs.update({f"image.{k}": v for k, v in img_attrs.items()})
                print(f"  {os.path.basename(fp):30s} shape={shape}  attrs={attrs}")
        except Exception as e:
            print(f"  ❌ {fp}: {e}")
    print()
    print("  NOTE: if the ACDC h5 files do not store spacing in attrs, it has to be")
    print("  read from the original NIfTI metadata. ACDC native in-plane spacing:")
    print("    - Range: 1.37 - 1.68 mm (median ~1.5 mm)")
    print("    - Slice thickness: 5 - 10 mm")
    print()
    print("  Source: Bernard et al., IEEE TMI 2018 (ACDC challenge paper)")
    print()


def check_synapse(synapse_path):
    """Read Synapse volumes to recover their native voxel spacing."""
    print("=" * 70)
    print("[C] CHECK SYNAPSE NATIVE VOXEL SPACING")
    print("=" * 70)

    import h5py
    h5_files = sorted(glob.glob(os.path.join(synapse_path, "*.h5")))[:5]
    if not h5_files:
        print(f"  ❌ No *.h5 file found in {synapse_path}")
        return

    for fp in h5_files:
        try:
            with h5py.File(fp, "r") as f:
                shape = f["image"].shape if "image" in f else None
                attrs = dict(f.attrs)
                if "image" in f:
                    img_attrs = dict(f["image"].attrs)
                    attrs.update({f"image.{k}": v for k, v in img_attrs.items()})
                print(f"  {os.path.basename(fp):30s} shape={shape}  attrs={attrs}")
        except Exception as e:
            print(f"  ❌ {fp}: {e}")
    print()
    print("  Synapse Multi-Atlas (Beyond Cranial Vault) native spacing:")
    print("    - In-plane: 0.54 - 0.98 mm (median ~0.76 mm)")
    print("    - Slice thickness: 2.5 - 5.0 mm")
    print()


def correction_table():
    """Table of correction factors for converting HD95 from voxels to mm."""
    print("=" * 70)
    print("[D] CORRECTION TABLE: HD95_voxel → HD95_mm")
    print("=" * 70)
    print()
    print("  Assumes pred and label are at native resolution (resized back before hd95)")
    print()
    print("  ACDC (spacing ~1.5 mm):")
    print("    - Paper reports 1.43 mm → true value ~ 1.43 × 1.5 = 2.15 mm")
    print("    - Paper reports 7.23 mm (TransUNet)")
    print("      → With the same (voxel) pipeline, TransUNet = 7.23 × 1.5 = 10.85 mm")
    print("      → Improvement 7.23v → 1.43v = 80%; equally 10.85mm → 2.15mm = 80%")
    print()
    print("  Synapse (spacing ~0.76 mm):")
    print("    - Paper reports 20.50 mm (DS Only mean) → this may in fact be voxels")
    print("      → If voxels: 20.50 × 0.76 ≈ 15.58 mm")
    print()
    print("  CASE 1 (best case for the paper): SeqAtt and the TransUNet baseline are")
    print("  BOTH measured in voxels → the ratio holds → '80% reduction' stays valid.")
    print()
    print("  CASE 2 (worst case): the TransUNet 7.23 figure is reported in mm (quoted")
    print("  from the TransUNet paper, not re-run) while SeqAtt 1.43 is in voxels.")
    print("  → The comparison is NOT fair; the paper MUST re-run TransUNet through the")
    print("  same pipeline to obtain a correct baseline.")
    print()


def fix_test_code_suggestion():
    """Print the suggested patch for test_phase3.py."""
    print("=" * 70)
    print("[E] SUGGESTED FIX for test_phase3.py")
    print("=" * 70)
    print()
    print("  CURRENT CODE (test_phase3.py:113-120):")
    print("    def calculate_metric_percase(pred, gt):")
    print("        pred[pred > 0] = 1; gt[gt > 0] = 1")
    print("        if pred.sum() > 0 and gt.sum() > 0:")
    print("            dice = metric.binary.dc(pred, gt)")
    print("            hd95 = metric.binary.hd95(pred, gt)   # ❌ voxel")
    print()
    print("  PROPOSED FIX:")
    print("    def calculate_metric_percase(pred, gt, voxelspacing=None):")
    print("        pred[pred > 0] = 1; gt[gt > 0] = 1")
    print("        if pred.sum() > 0 and gt.sum() > 0:")
    print("            dice = metric.binary.dc(pred, gt)")
    print("            hd95 = metric.binary.hd95(pred, gt,")
    print("                                     voxelspacing=voxelspacing)  # ✓ mm")
    print()
    print("  And inside test_single_volume, pass the dataset voxelspacing:")
    print("    ACDC: voxelspacing=(z_spacing, 1.52, 1.52)  # or read it from h5 attrs")
    print("    Synapse: voxelspacing=(z_spacing, 0.76, 0.76)")
    print()
    print("  OR, to stay comparable with the 2D-on-ACDC literature that uses voxels:")
    print("    STATE EXPLICITLY in the paper: 'HD95 reported in pixel units, consistent")
    print("    with TransUNet original codebase'.")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check_synthetic", action="store_true", default=True)
    ap.add_argument("--check_acdc", action="store_true")
    ap.add_argument("--check_synapse", action="store_true")
    ap.add_argument("--acdc_path", type=str,
                    default=os.environ.get('SEQATT_ACDC_ROOT', 'data/ACDC'))
    ap.add_argument("--synapse_path", type=str,
                    default=os.environ.get('SEQATT_SYNAPSE_ROOT', 'data/Synapse/test_vol_h5'))
    args = ap.parse_args()

    if args.check_synthetic:
        demo_synthetic()
    if args.check_acdc:
        check_acdc(args.acdc_path)
    if args.check_synapse:
        check_synapse(args.synapse_path)
    correction_table()
    fix_test_code_suggestion()


if __name__ == "__main__":
    main()
