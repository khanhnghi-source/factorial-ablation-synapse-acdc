"""
Diagnose the true source of voxel spacing -- prerequisite for C1 (R1.5).

BACKGROUND
----------
R1.5: "HD95 should be computed using each volume's actual post-processing voxel
geometry, or the authors must document an explicit physical resampling operation
that makes the fixed spacings correct."

Established from the source code (2026-09-07):
  - `test_phase3.py:328,416` DOES call `get_voxelspacing_from_h5()` per case, but
    falls back to `DEFAULT_VOXELSPACING_MM` when the .h5 file carries no attrs.
    The test logs print `(0.76, 0.76)` -> the fallback is likely being used for
    EVERY case.
  - `z_spacing` only ever takes its default of 1; no call site passes a different
    value. So HD95 in mm is actually computed with (1.0, 0.76, 0.76), whereas
    Section IV-D of the manuscript states (3.0, 0.76, 0.76). This is a
    code-vs-paper contradiction.
  - `test_single_volume` resizes the prediction BACK to the original size
    (lines 245-246) before computing metrics, so the part of R1.5 claiming that
    the resize to 224 distorts spacing does NOT apply -- the metrics are computed
    in the original geometry.

This script answers two questions:
  1. Do the .h5 files store spacing, and under which key?
  2. If not, is the original data (.nii.gz) available to recover the true spacing?

USAGE
-----
    python diagnose_spacing.py --data_root "/path/to/data"
    python diagnose_spacing.py --data_root ... --raw_root "/path/to/RawData"
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Optional

SPACING_KEYS = ['spacing', 'pixdim', 'voxelspacing', 'pixel_size',
                'PixelSpacing', 'voxel_size', 'affine', 'origin', 'direction']


def inspect_h5(path: str, label: str, n: int = 3) -> None:
    """Print all attrs and shapes of the first few .h5 files."""
    import h5py
    fs = sorted(glob.glob(path))
    print(f"\n{'─' * 76}")
    print(f"{label}   ({len(fs)} files)")
    print(f"  pattern: {path}")
    print('─' * 76)
    if not fs:
        print("  🔴 No files found -- check the path.")
        return

    for f in fs[:n]:
        print(f"\n  ▸ {os.path.basename(f)}")
        try:
            with h5py.File(f, 'r') as h:
                print(f"      dataset  : {list(h.keys())}")
                for k in h.keys():
                    try:
                        print(f"        {k:<8} shape={h[k].shape} dtype={h[k].dtype}")
                    except Exception:
                        pass
                fa = dict(h.attrs)
                print(f"      attrs file-level : {fa if fa else '(NONE)'}")
                for k in h.keys():
                    ia = dict(h[k].attrs)
                    if ia:
                        print(f"      attrs '{k}' : {ia}")
                # Any spacing key present?
                found = [k for k in SPACING_KEYS
                        if k in fa or any(k in h[d].attrs for d in h.keys())]
                print(f"      -> spacing keys: {found if found else '🔴 NONE'}")
        except Exception as e:
            print(f"      🔴 read error: {e}")


def inspect_nifti(pattern: str, label: str, n: int = 5) -> None:
    """Read the true spacing from the original .nii/.nii.gz files.

    recursive=True is required for '**' in the pattern to work.
    """
    fs = sorted(glob.glob(pattern, recursive=True))
    print(f"\n{'─' * 76}")
    print(f"{label}   ({len(fs)} files)")
    print(f"  pattern: {pattern}")
    print('─' * 76)
    if not fs:
        print("  ⚠️  Not found -- the original data is needed to recover the true spacing.")
        return
    try:
        import nibabel as nib
    except ImportError:
        print("  🔴 nibabel is not installed: pip install nibabel")
        return

    print(f"\n  {'file':<28}{'shape':>18}{'spacing (z, y, x)':>28}")
    print("  " + "-" * 72)
    spacings = []
    for f in fs[:n]:
        try:
            img = nib.load(f)
            zoom = img.header.get_zooms()
            sp = tuple(round(float(z), 4) for z in zoom[:3])
            spacings.append(sp)
            print(f"  {os.path.basename(f)[:27]:<28}{str(img.shape):>18}{str(sp):>28}")
        except Exception as e:
            print(f"  {os.path.basename(f)[:27]:<28}  error: {e}")
    if len(spacings) > 1:
        differs = len(set(spacings)) > 1
        print(f"\n  -> spacing {'DIFFERS across cases ⚠️' if differs else 'identical'}")
        if differs:
            print("    This is exactly the point R1.5 raises: a single fixed value cannot be used.")


def main() -> None:
    p = argparse.ArgumentParser(description="Diagnose the source of voxel spacing (C1 / R1.5)")
    p.add_argument('--data_root', type=str, required=True,
                   help='data/ directory containing Synapse/ and ACDC/')
    p.add_argument('--raw_root', type=str, default=None,
                   help='Directory with the ORIGINAL .nii.gz data (if available), e.g. Synapse RawData')
    args = p.parse_args()

    print("=" * 76)
    print("VOXEL SPACING DIAGNOSIS -- prerequisite for C1 (R1.5)")
    print("=" * 76)
    print(f"data_root: {args.data_root}")
    print(f"raw_root : {args.raw_root or '(not provided)'}")

    # ── 1. The .h5 files used for testing ─────────────────────────────────
    inspect_h5(os.path.join(args.data_root, 'Synapse', 'test_vol_h5', '*.h5'),
           '[1] Synapse test volumes (.h5) -- current source of HD95')
    inspect_h5(os.path.join(args.data_root, 'ACDC', 'ACDC_training_volumes', '*.h5'),
           '[2] ACDC volumes (.h5)')

    # ── 2. Original .nii.gz data ──────────────────────────────────────────
    if args.raw_root:
        inspect_nifti(os.path.join(args.raw_root, '**', '*.nii.gz'),
                  '[3] ORIGINAL .nii.gz data -- true spacing')
    else:
        for cand in ['RawData', 'Synapse/RawData', 'Synapse/averaged-training-images',
                     'ACDC/database', 'ACDC/training']:
            pt = os.path.join(args.data_root, cand, '**', '*.nii*')
            if glob.glob(pt, recursive=True):
                inspect_nifti(pt, f'[3] Found .nii under {cand}')
                break
        else:
            print(f"\n{'─' * 76}")
            print("[3] Original .nii.gz data")
            print('─' * 76)
            print("  ⚠️  Not found automatically. Pass --raw_root if it is available.")

    # ── 3. Conclusions and next steps ─────────────────────────────────────
    print("\n" + "=" * 76)
    print("HOW TO READ THE RESULTS")
    print("=" * 76)
    print("""
  A. If the .h5 files DO carry a spacing key (sections 1/2 print the key name):
     -> test_phase3.py already reads it per case. Only z_spacing needs fixing,
        then re-run the evaluation over the 50 checkpoints. NO retraining needed.

  B. If the .h5 files have NO spacing but the original .nii.gz exists (section 3):
     -> Build a case -> spacing lookup table from the .nii.gz headers and pass it
        to test_phase3.py. Again, no retraining needed.

  C. If neither is available:
     -> The original metadata must be re-downloaded from the dataset source
        (Synapse: BTCV challenge; ACDC: Bernard 2018). Headers are enough; the
        images are not needed.

  In EVERY case, the following two items are mandatory and independent:
    - Fix Section IV-D: the manuscript states (3.0, 0.76, 0.76) and
      (5.0, 1.52, 1.52), but the code uses z_spacing = 1.0. It must match what
      was actually run.
    - Answer the second half of R1.5: the reviewer argues that the resize to 224
      distorts spacing, but test_single_volume (lines 245-246) resizes the
      prediction BACK to the original size before computing metrics, so the
      metrics live in the original geometry. This point can be settled by
      quoting the code; no recomputation is required.
""")


if __name__ == '__main__':
    main()
