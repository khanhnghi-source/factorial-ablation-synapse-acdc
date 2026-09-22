"""
AGGREGATE 40 TEST LOGS → KetQua_v2.xlsx
========================================
Parse the log files under test_log/ and aggregate them into an Excel workbook
with 5 sheets:

  1. Summary          : mean±std (DSC, HD95 voxel, HD95 mm) per config × dataset
  2. Synapse_per_org  : per-organ DSC + HD95, mean±std across 5 seeds
  3. ACDC_per_str     : per-structure DSC + HD95, mean±std across 5 seeds
  4. All_runs         : raw data for every run (40 rows)
  5. Per_patient      : per-patient DSC + HD95 (input to the Wilcoxon test)

USAGE:
    python aggregate_results.py
    python aggregate_results.py --test_log_dir ./test_log --output KetQua_v2.xlsx

Author: SeqAtt-UNet Phase 4
"""

import argparse
import os
import re
import glob
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# REGEX patterns
# =============================================================================

# Per-patient line:
# [HH:MM:SS.fff] patient_name: DSC=0.9436, HD95_vox=1.00, HD95_mm=1.17
RE_PATIENT = re.compile(
    r'\]\s+(?P<case>\S+):\s+DSC=(?P<dsc>[\d.]+),\s+HD95_vox=(?P<hv>[\d.]+),\s+HD95_mm=(?P<hm>[\d.]+)'
)

# Per-class line:
# [HH:MM:SS.fff] RV             : DSC=0.9013, HD95_vox=1.32, HD95_mm=1.69
RE_PERCLASS = re.compile(
    r'\]\s+(?P<cls>[A-Za-z()\.]+)\s*:\s+DSC=(?P<dsc>[\d.]+),\s+HD95_vox=(?P<hv>[\d.]+),\s+HD95_mm=(?P<hm>[\d.]+)'
)

# Mean Dice line:
#   Mean Dice:        0.9017 (90.17%)
RE_MEAN_DICE = re.compile(r'Mean Dice:\s+(?P<dice>[\d.]+)\s+\(')

# Mean HD95 voxel:
#   Mean HD95 (voxel): 1.2286
RE_MEAN_HV = re.compile(r'Mean HD95 \(voxel\):\s+(?P<v>[\d.]+)')

# Mean HD95 mm:
#   Mean HD95 (mm):    1.6113  [voxelspacing=(1.52, 1.52)]
RE_MEAN_HM = re.compile(r'Mean HD95 \(mm\):\s+(?P<v>[\d.]+)')

# Folder/file naming patterns
# Phase 3: test_log_TU_<DATASET><SIZE>_[CBAM_]?[BILSTM|HYBRID]?_Phase3[_DS]/
# Phase 4: test_log_TU_<DATASET><SIZE>_[CBAM_]?[BILSTM|HYBRID]?_Phase3_DS_SkipCBAM/
# Baseline (no CBAM/BiLSTM, with DS):  test_log_TU_ACDC224_Phase3_DS
#
# [FIX 2026-09-06] `_DS` became OPTIONAL.
#   The factorial ablation for R1.3 introduces two new exp_names:
#       TU_<DS>224_Phase3            <- baseline WITHOUT DS     (group a, 10 runs)
#       TU_<DS>224_BILSTM_Phase3_DS  <- BiLSTM+DS without CBAM  (group b, 10 runs)
#   The old regex required '_Phase3_DS', so group (a) was SKIPPED ENTIRELY -- 10
#   runs, ~55 GPU-hours, would never reach the tables and nothing would warn.
RE_EXP_DIR = re.compile(
    r'test_log_TU_(?P<dataset>Synapse|ACDC)(?P<size>\d+)'
    r'(?:_(?P<config>.+?))?_Phase3'
    r'(?P<ds>_DS)?'
    r'(?P<phase4>_SkipCBAM)?$'
)

# Inner file pattern: ..._bs<BS>_..._s<SEED>_tta_<TTA>.txt
RE_SEED_FILE = re.compile(
    r'_bs(?P<bs>\d+)_.*_s(?P<seed>\d+)_tta_(?P<tta>\w+)\.txt$'
)

# Default batch size to keep (matches paper). Set to None to keep all.
KEEP_BATCH_SIZE = 12


# Class names per dataset (from test_phase3.py)
CLASS_NAMES = {
    'Synapse': ['Aorta', 'Gallbladder', 'Kidney(L)', 'Kidney(R)',
                'Liver', 'Pancreas', 'Spleen', 'Stomach'],
    'ACDC':    ['RV', 'Myo', 'LV'],
}


# =============================================================================
# PARSING
# =============================================================================

def parse_config_name(raw: str, phase4: bool = False, ds: bool = True) -> str:
    """
    Standardise config name from folder.

    Args:
        raw: config string from folder (e.g., 'CBAM_BILSTM' or '')
        phase4: True if folder has _SkipCBAM suffix
        ds:     True if folder has _DS suffix (deep supervision enabled)

    Examples:
        '',            ds=True   -> 'Baseline (DS only)'
        '',            ds=False  -> 'Baseline (no DS)'          <- R1.3 group (a)
        'BILSTM',      ds=True   -> '+BiLSTM+DS (no CBAM)'      <- R1.3 group (b)
        'CBAM_BILSTM', ds=True   -> '+CBAM+BiLSTM+DS'
        'CBAM_BILSTM', phase4=True -> '+CBAM+BiLSTM+DS+SkipCBAM'
    """
    raw = raw.strip('_')
    if not raw:
        # Baseline: distinguish WITH from WITHOUT deep supervision. The 'no DS'
        # branch is the mandatory control that R1.3 calls the "minimum required
        # control".
        base = 'Baseline (DS only)' if ds else 'Baseline (no DS)'
    elif raw == 'CBAM':
        base = '+CBAM+DS' if ds else '+CBAM (no DS)'
    elif raw == 'CBAM_BILSTM':
        base = '+CBAM+BiLSTM+DS' if ds else '+CBAM+BiLSTM (no DS)'
    elif raw == 'CBAM_HYBRID':
        base = '+CBAM+Hybrid+DS' if ds else '+CBAM+Hybrid (no DS)'
    elif raw == 'BILSTM':
        # BiLSTM WITHOUT CBAM -- the configuration R1.3 asks for in order to
        # isolate the contribution of the sequential block on its own.
        base = '+BiLSTM+DS (no CBAM)' if ds else '+BiLSTM (no CBAM, no DS)'
    elif raw == 'HYBRID':
        base = '+Hybrid+DS (no CBAM)' if ds else '+Hybrid (no CBAM, no DS)'
    else:
        base = raw if ds else f'{raw} (no DS)'

    if phase4:
        base += '+SkipCBAM'
    return base


def parse_log_file(filepath: str) -> Optional[Dict]:
    """
    Parse single log file.

    Returns dict with:
        per_patient: list of {case, dsc, hd95_vox, hd95_mm}
        per_class:   list of {cls, dsc, hd95_vox, hd95_mm}
        mean_dice:   float
        mean_hd95_vox: float
        mean_hd95_mm: float
    """
    if not os.path.exists(filepath):
        return None

    result = {
        'per_patient': [],
        'per_class':   [],
        'mean_dice':   None,
        'mean_hd95_vox': None,
        'mean_hd95_mm': None,
    }

    in_perclass_section = False

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.rstrip()

            # Track section boundaries
            if 'PER-CLASS RESULTS' in line:
                in_perclass_section = True
                continue
            if 'FINAL RESULTS' in line:
                in_perclass_section = False
                continue

            # Mean stats
            m = RE_MEAN_DICE.search(line)
            if m:
                result['mean_dice'] = float(m.group('dice'))
                continue

            m = RE_MEAN_HV.search(line)
            if m:
                result['mean_hd95_vox'] = float(m.group('v'))
                continue

            m = RE_MEAN_HM.search(line)
            if m:
                result['mean_hd95_mm'] = float(m.group('v'))
                continue

            # Per-class (after "PER-CLASS RESULTS" header)
            if in_perclass_section:
                m = RE_PERCLASS.search(line)
                if m:
                    cls_name = m.group('cls').strip()
                    # Skip header lines that match accidentally
                    if cls_name.lower() not in ('per', 'class', 'final'):
                        result['per_class'].append({
                            'cls':       cls_name,
                            'dsc':       float(m.group('dsc')),
                            'hd95_vox':  float(m.group('hv')),
                            'hd95_mm':   float(m.group('hm')),
                        })
                continue

            # Per-patient
            m = RE_PATIENT.search(line)
            if m:
                case_name = m.group('case')
                # Filter out non-patient lines that match by accident
                # (case_name should start with "patient" or "case")
                if (case_name.startswith('patient') or case_name.startswith('case') or
                        case_name.startswith('amos') or case_name.startswith('pt')):
                    result['per_patient'].append({
                        'case':      case_name,
                        'dsc':       float(m.group('dsc')),
                        'hd95_vox':  float(m.group('hv')),
                        'hd95_mm':   float(m.group('hm')),
                    })

    # ------------------------------------------------------------------
    # [FIX 2026-09-05] Dedup the per-patient records.
    #
    # Some logs contain TWO passes of per-patient results (the test was re-run and
    # the log was appended to instead of overwritten). The loop above reads
    # sequentially, so it appends both passes and every case shows up twice with
    # identical values.
    #
    # Observed consequence: ACDC `+CBAM+BiLSTM+DS` seed 1234 produced 80 rows
    # instead of 40. Because the tests run per case POOLED ACROSS SEEDS, seed 1234
    # was given double weight:
    #     wrong  : (2×0.9027 + 0.9041+0.9052+0.9040+0.9034)/6 = 0.903676
    #     correct: (  0.9027 + 0.9041+0.9052+0.9040+0.9034)/5 = 0.903871
    # -> primary endpoint E3 (ACDC DSC) was reported as p=0.0446 instead of
    #    p=0.0409, and mean_A drifted away from the 90.39% stated in the Abstract.
    #
    # Keep the FIRST occurrence in reading order. Since the duplicates carry
    # identical values, this choice changes no reported number.
    # ------------------------------------------------------------------
    if result['per_patient']:
        seen_cases: set = set()
        deduped: List[Dict] = []
        for rec in result['per_patient']:
            if rec['case'] in seen_cases:
                continue
            seen_cases.add(rec['case'])
            deduped.append(rec)
        if len(deduped) != len(result['per_patient']):
            print(f"[DEDUP] {os.path.basename(filepath)}: "
                  f"{len(result['per_patient'])} -> {len(deduped)} per-patient rows")
        result['per_patient'] = deduped

    return result


def _short_path(path: str, n_parts: int = 3) -> str:
    """Shorten a log path for the `filepath` column.

    Uses `os.path.relpath` when possible, but on Windows that function raises
    `ValueError` when the file and the current directory sit on TWO DIFFERENT
    DRIVES -- a routine situation in this project: `test_log` lives on drive `H:`
    (Google Drive) while the repo lives on drive `E:`. In that case the last
    `n_parts` components are kept, which yields the same compact and stable form
    as before, for example:
        test_log\\test_log_TU_Synapse224_CBAM_Phase3_DS\\TU_pretrain_..._s3456_tta_simple.txt

    Parameters
    ----------
    path : str
        Absolute path to the log file.
    n_parts : int, optional
        Number of trailing components to keep when relpath cannot be computed.

    Returns
    -------
    str
        The shortened path.
    """
    try:
        return os.path.relpath(path)
    except ValueError:
        parts = os.path.normpath(path).split(os.sep)
        return os.path.join(*parts[-n_parts:]) if len(parts) >= n_parts else path


def discover_runs(test_log_dir: str) -> List[Dict]:
    """
    Scan test_log directory and discover all (dataset, config, seed) combinations.

    Returns list of dicts: {dataset, config, seed, filepath}
    """
    runs = []

    for exp_dir in sorted(glob.glob(os.path.join(test_log_dir, 'test_log_TU_*'))):
        dir_name = os.path.basename(exp_dir)
        m = RE_EXP_DIR.match(dir_name)
        if not m:
            print(f"[WARN] Folder does not match the pattern: {dir_name}")
            continue

        dataset = m.group('dataset')
        cfg_raw = m.group('config') or ''
        phase4 = bool(m.group('phase4'))
        ds = bool(m.group('ds'))
        config = parse_config_name(cfg_raw, phase4=phase4, ds=ds)

        # Find log files inside
        for fp in sorted(glob.glob(os.path.join(exp_dir, '*.txt'))):
            fname = os.path.basename(fp)
            ms = RE_SEED_FILE.search(fname)
            if not ms:
                continue
            bs = int(ms.group('bs'))
            seed = int(ms.group('seed'))
            tta = ms.group('tta')

            # Filter by batch size if configured
            if KEEP_BATCH_SIZE is not None and bs != KEEP_BATCH_SIZE:
                print(f"[SKIP] bs={bs} (keep only bs={KEEP_BATCH_SIZE}): {fname}")
                continue

            runs.append({
                'dataset':  dataset,
                'config':   config,
                'seed':     seed,
                'batch':    bs,
                'tta':      tta,
                'filepath': fp,
            })

    return runs


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate(runs: List[Dict]) -> Dict[str, pd.DataFrame]:
    """
    Build 5 DataFrames for Excel sheets.
    """
    all_runs_rows = []
    per_patient_rows = []
    per_class_rows = []

    for r in runs:
        parsed = parse_log_file(r['filepath'])
        if parsed is None:
            print(f"[WARN] Cannot parse: {r['filepath']}")
            continue

        # Sheet 4: all_runs (one row per run)
        all_runs_rows.append({
            'dataset':     r['dataset'],
            'config':      r['config'],
            'seed':        r['seed'],
            'tta':         r['tta'],
            'mean_dsc':    parsed['mean_dice'],
            'mean_hd95_vox': parsed['mean_hd95_vox'],
            'mean_hd95_mm':  parsed['mean_hd95_mm'],
            'n_patients':  len(parsed['per_patient']),
            'filepath':    _short_path(r['filepath']),
        })

        # Sheet 5: per_patient
        for p in parsed['per_patient']:
            per_patient_rows.append({
                'dataset':   r['dataset'],
                'config':    r['config'],
                'seed':      r['seed'],
                'case':      p['case'],
                'dsc':       p['dsc'],
                'hd95_vox':  p['hd95_vox'],
                'hd95_mm':   p['hd95_mm'],
            })

        # Sheet 2/3: per_class
        for c in parsed['per_class']:
            per_class_rows.append({
                'dataset':   r['dataset'],
                'config':    r['config'],
                'seed':      r['seed'],
                'class':     c['cls'],
                'dsc':       c['dsc'],
                'hd95_vox':  c['hd95_vox'],
                'hd95_mm':   c['hd95_mm'],
            })

    df_all = pd.DataFrame(all_runs_rows)
    df_pat = pd.DataFrame(per_patient_rows)
    df_cls = pd.DataFrame(per_class_rows)

    # ---- Sheet 1: Summary -----------------------------------------------------
    summary = (
        df_all
        .groupby(['dataset', 'config'])
        .agg(
            n_seeds=('seed', 'count'),
            mean_dsc=('mean_dsc', 'mean'),
            std_dsc=('mean_dsc', 'std'),
            mean_hd95_vox=('mean_hd95_vox', 'mean'),
            std_hd95_vox=('mean_hd95_vox', 'std'),
            mean_hd95_mm=('mean_hd95_mm', 'mean'),
            std_hd95_mm=('mean_hd95_mm', 'std'),
        )
        .reset_index()
    )
    # Pretty format
    summary['DSC (mean ± std)'] = summary.apply(
        lambda r: f"{r['mean_dsc']*100:.2f} ± {r['std_dsc']*100:.2f}", axis=1)
    summary['HD95 voxel (mean ± std)'] = summary.apply(
        lambda r: f"{r['mean_hd95_vox']:.2f} ± {r['std_hd95_vox']:.2f}", axis=1)
    summary['HD95 mm (mean ± std)'] = summary.apply(
        lambda r: f"{r['mean_hd95_mm']:.2f} ± {r['std_hd95_mm']:.2f}", axis=1)

    summary_display = summary[[
        'dataset', 'config', 'n_seeds',
        'DSC (mean ± std)', 'HD95 voxel (mean ± std)', 'HD95 mm (mean ± std)',
        'mean_dsc', 'std_dsc',
        'mean_hd95_vox', 'std_hd95_vox',
        'mean_hd95_mm', 'std_hd95_mm',
    ]]

    # ---- Sheet 2: Synapse per-organ ------------------------------------------
    syn_cls = df_cls[df_cls['dataset'] == 'Synapse']
    syn_per_org = build_per_class_pivot(syn_cls, CLASS_NAMES['Synapse'])

    # ---- Sheet 3: ACDC per-structure -----------------------------------------
    acdc_cls = df_cls[df_cls['dataset'] == 'ACDC']
    acdc_per_str = build_per_class_pivot(acdc_cls, CLASS_NAMES['ACDC'])

    return {
        'Summary':         summary_display,
        'Synapse_per_org': syn_per_org,
        'ACDC_per_str':    acdc_per_str,
        'All_runs':        df_all.sort_values(['dataset', 'config', 'seed']),
        'Per_patient':     df_pat.sort_values(['dataset', 'config', 'seed', 'case']),
    }


def build_per_class_pivot(df_cls: pd.DataFrame, expected_classes: List[str]) -> pd.DataFrame:
    """
    Build a per-class summary table: rows = config, columns = class × metric.

    Outputs mean±std per (config, class) for DSC and HD95.
    """
    if df_cls.empty:
        return pd.DataFrame()

    # Aggregate per (config, class)
    agg = (
        df_cls
        .groupby(['config', 'class'])
        .agg(
            n_seeds=('seed', 'count'),
            mean_dsc=('dsc', 'mean'),
            std_dsc=('dsc', 'std'),
            mean_hv=('hd95_vox', 'mean'),
            std_hv=('hd95_vox', 'std'),
            mean_hm=('hd95_mm', 'mean'),
            std_hm=('hd95_mm', 'std'),
        )
        .reset_index()
    )

    # Pretty format
    agg['DSC'] = agg.apply(lambda r: f"{r['mean_dsc']*100:.2f} ± {r['std_dsc']*100:.2f}", axis=1)
    agg['HD95_vox'] = agg.apply(lambda r: f"{r['mean_hv']:.2f} ± {r['std_hv']:.2f}", axis=1)
    agg['HD95_mm'] = agg.apply(lambda r: f"{r['mean_hm']:.2f} ± {r['std_hm']:.2f}", axis=1)

    # Pivot: rows = config, cols = class × metric
    pivot_dsc = agg.pivot(index='config', columns='class', values='DSC')
    pivot_hv = agg.pivot(index='config', columns='class', values='HD95_vox')
    pivot_hm = agg.pivot(index='config', columns='class', values='HD95_mm')

    # Reorder columns to expected order
    cls_in_data = [c for c in expected_classes if c in pivot_dsc.columns]
    extra = [c for c in pivot_dsc.columns if c not in expected_classes]
    cols_ordered = cls_in_data + extra

    pivot_dsc = pivot_dsc.reindex(columns=cols_ordered)
    pivot_hv  = pivot_hv.reindex(columns=cols_ordered)
    pivot_hm  = pivot_hm.reindex(columns=cols_ordered)

    # Flatten columns (avoid MultiIndex, which openpyxl doesn't handle well with index=False)
    pivot_dsc.columns = [f"DSC | {c}"        for c in pivot_dsc.columns]
    pivot_hv.columns  = [f"HD95_vox | {c}"   for c in pivot_hv.columns]
    pivot_hm.columns  = [f"HD95_mm | {c}"    for c in pivot_hm.columns]

    combined = pd.concat([pivot_dsc, pivot_hv, pivot_hm], axis=1)
    return combined.reset_index()


# =============================================================================
# EXCEL EXPORT
# =============================================================================

def export_excel(sheets: Dict[str, pd.DataFrame], output_path: str):
    """
    Write sheets to Excel, PRESERVING sheets written by other scripts.

    ``pd.ExcelWriter`` in the default write mode replaces the entire workbook.
    On 2026-09-14 that silently destroyed the ``Computational`` sheet — the
    seven-configuration efficiency measurement behind Table V, which takes ten
    minutes of GPU timing to reproduce — simply because this script was re-run
    to refresh the result sheets. Nothing warned; the sheet was just gone.

    This script owns only the sheets it builds. Any other sheet found in an
    existing workbook (``Computational`` from ``compute_flops.py``,
    ``Primary_endpoints`` from ``statistical_tests_primary.py``) is read back
    and rewritten untouched.
    """
    preserved: Dict[str, pd.DataFrame] = {}
    if os.path.exists(output_path):
        try:
            existing = pd.ExcelFile(output_path)
            for sn in existing.sheet_names:
                if sn not in sheets:
                    preserved[sn] = pd.read_excel(output_path, sheet_name=sn)
            existing.close()
        except Exception as exc:                                # pragma: no cover
            print(f"[WARN] Could not read the existing workbook ({exc}); "
                  f"sheets written by other scripts MAY be lost.")

    if preserved:
        print(f"[PRESERVED] {len(preserved)} sheet(s) owned by other scripts: "
              f"{', '.join(preserved)}")

    sheets = {**sheets, **preserved}

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)

            # Auto-fit column widths
            ws = writer.sheets[name]
            for col_idx, col_name in enumerate(df.columns):
                col_letter = ws.cell(row=1, column=col_idx + 1).column_letter
                max_len = max(
                    [len(str(col_name))] +
                    [len(str(v)) for v in df.iloc[:, col_idx].astype(str)]
                )
                ws.column_dimensions[col_letter].width = min(max_len + 2, 50)

    print(f"\n✓ Saved Excel: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--test_log_dir', type=str, default='./test_log',
                   help='Directory containing test_log_TU_* subfolders')
    p.add_argument('--output', type=str, default='../KetQua_v2.xlsx',
                   help='Output Excel path')
    args = p.parse_args()

    print(f"Scanning {args.test_log_dir} ...")
    runs = discover_runs(args.test_log_dir)
    print(f"Found {len(runs)} log files.")

    if len(runs) == 0:
        print("No logs found. Check --test_log_dir path.")
        return

    # Quick overview
    counts = defaultdict(int)
    for r in runs:
        counts[(r['dataset'], r['config'])] += 1
    print("\nRun coverage:")
    for (ds, cfg), n in sorted(counts.items()):
        marker = "✓" if n == 5 else "⚠"
        print(f"  {marker} {ds:8s} | {cfg:30s} | {n} seeds")

    print("\nParsing and aggregating ...")
    sheets = aggregate(runs)

    print("\n=== SUMMARY (Sheet 1) ===")
    summary = sheets['Summary']
    summary_display = summary[['dataset', 'config', 'n_seeds',
                                'DSC (mean ± std)', 'HD95 voxel (mean ± std)',
                                'HD95 mm (mean ± std)']]
    print(summary_display.to_string(index=False))

    export_excel(sheets, args.output)

    print("\nNext steps:")
    print("  1. Open KetQua_v2.xlsx and check the numbers")
    print("  2. Run statistical_tests.py (Wilcoxon + Bonferroni-Holm)")
    print("  3. Cross-check against the paper's claim of '1.43 mm HD95' on ACDC")


if __name__ == '__main__':
    main()
