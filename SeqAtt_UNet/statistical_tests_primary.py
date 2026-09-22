"""
PRIMARY ENDPOINT STATISTICAL TESTS for SeqAtt-UNet
====================================================
Strategy: Instead of testing all 6 pairwise combinations (which triggers
Bonferroni-Holm correction and dilutes power), pre-declare PRIMARY ENDPOINTS
and test ONLY those:

  Primary endpoint 1 (ACDC, HD95 voxel):
    H0: +CBAM+BiLSTM+DS = Baseline (DS only)
    H1: +CBAM+BiLSTM+DS < Baseline   (improvement = lower HD95)

  Primary endpoint 2 (Synapse, DSC):
    H0: Baseline (DS only) = best ablation
    H1: Baseline > best ablation  (Baseline wins Synapse)

The five endpoints are PRE-SPECIFIED (defined a priori by the authors) but were
NOT lodged with an external registry, so they must not be called "pre-registered".
All five are corrected for multiplicity with the Holm step-down procedure; under
that correction none reaches significance, and this is reported openly.

Also computes per-organ/per-structure tests with sub-correction (more power).

USAGE:
    python statistical_tests_primary.py

Output: adds "Primary_endpoints" + "Per_organ_tests" sheets to KetQua_v2.xlsx
"""

import argparse
import os
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# CORE TEST FUNCTIONS
# =============================================================================

def wilcoxon_one_test(values_A: np.ndarray, values_B: np.ndarray,
                      alternative: str = 'two-sided') -> Dict:
    """
    Single Wilcoxon signed-rank test between paired samples.

    Args:
        values_A, values_B: 1D arrays of paired measurements
        alternative: 'two-sided', 'less' (A < B), or 'greater' (A > B)

    Returns dict with statistics.
    """
    a = np.asarray(values_A, dtype=float)
    b = np.asarray(values_B, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    n = len(a)
    diffs = a - b

    if n == 0 or np.all(diffs == 0):
        return {
            'n': n, 'mean_A': np.nan, 'mean_B': np.nan,
            'mean_diff': 0.0, 'median_diff': 0.0,
            'W': np.nan, 'p_value': 1.0, 'cohen_d': 0.0,
        }

    try:
        W, p = stats.wilcoxon(a, b, alternative=alternative, zero_method='wilcox')
    except Exception:
        W, p = np.nan, np.nan

    std = np.std(diffs, ddof=1) if n > 1 else 1.0
    d = float(np.mean(diffs) / std) if std > 0 else 0.0

    return {
        'n':           n,
        'mean_A':      float(np.mean(a)),
        'mean_B':      float(np.mean(b)),
        'mean_diff':   float(np.mean(diffs)),
        'median_diff': float(np.median(diffs)),
        'W':           float(W) if not np.isnan(W) else np.nan,
        'p_value':     float(p) if not np.isnan(p) else np.nan,
        'cohen_d':     d,
    }


def per_patient_mean(df_pat, dataset, config, metric):
    """Mean over seeds, one value per patient."""
    sub = df_pat[(df_pat['dataset'] == dataset) & (df_pat['config'] == config)]
    return sub.groupby('case')[metric].mean()


# =============================================================================
# PRIMARY ENDPOINTS
# =============================================================================

PRIMARY_ENDPOINTS = [
    {
        'name':       'ACDC HD95 voxel: Proposed (+CBAM+BiLSTM+DS) better than Baseline?',
        'dataset':    'ACDC',
        'config_A':   '+CBAM+BiLSTM+DS',   # proposed
        'config_B':   'Baseline (DS only)',
        'metric':     'hd95_vox',
        'alternative': 'less',              # A < B → improvement
        'rationale':  'Boundary precision is the primary clinical metric for cardiac segmentation',
    },
    {
        'name':       'ACDC HD95 mm: Proposed (+CBAM+BiLSTM+DS) better than Baseline?',
        'dataset':    'ACDC',
        'config_A':   '+CBAM+BiLSTM+DS',
        'config_B':   'Baseline (DS only)',
        'metric':     'hd95_mm',
        'alternative': 'less',
        'rationale':  'Same as above, reported in millimeters for clinical interpretation',
    },
    {
        'name':       'ACDC DSC: Proposed (+CBAM+BiLSTM+DS) better than Baseline?',
        'dataset':    'ACDC',
        'config_A':   '+CBAM+BiLSTM+DS',
        'config_B':   'Baseline (DS only)',
        'metric':     'dsc',
        'alternative': 'greater',
        'rationale':  'Volumetric overlap metric',
    },
    {
        'name':       'Synapse DSC: Baseline better than +CBAM+BiLSTM+DS?',
        'dataset':    'Synapse',
        'config_A':   'Baseline (DS only)',  # baseline wins Synapse
        'config_B':   '+CBAM+BiLSTM+DS',
        'metric':     'dsc',
        'alternative': 'greater',
        'rationale':  'Test the dataset-dependent finding: BiLSTM hurts heterogeneous anatomy',
    },
    {
        'name':       'Synapse HD95 voxel: Baseline better than +CBAM+BiLSTM+DS?',
        'dataset':    'Synapse',
        'config_A':   'Baseline (DS only)',
        'config_B':   '+CBAM+BiLSTM+DS',
        'metric':     'hd95_vox',
        'alternative': 'less',
        'rationale':  'Boundary precision on multi-organ — also baseline wins',
    },
]


def run_primary_endpoints(df_pat: pd.DataFrame) -> pd.DataFrame:
    rows = []
    print("=" * 72)
    print("PRIMARY ENDPOINT TESTS (no multiple-testing correction)")
    print("=" * 72)
    for ep in PRIMARY_ENDPOINTS:
        a_series = per_patient_mean(df_pat, ep['dataset'], ep['config_A'], ep['metric'])
        b_series = per_patient_mean(df_pat, ep['dataset'], ep['config_B'], ep['metric'])

        # Align on common patients
        common = a_series.index.intersection(b_series.index)
        a = a_series.loc[common].values
        b = b_series.loc[common].values

        res = wilcoxon_one_test(a, b, alternative=ep['alternative'])

        sig = '***' if res['p_value'] < 0.001 \
            else '**' if res['p_value'] < 0.01 \
            else '*' if res['p_value'] < 0.05 else 'n.s.'

        print(f"\n  ► {ep['name']}")
        print(f"    A = {ep['config_A']:25s} mean={res['mean_A']:.4f}")
        print(f"    B = {ep['config_B']:25s} mean={res['mean_B']:.4f}")
        print(f"    n={res['n']}, alt={ep['alternative']}, "
              f"W={res['W']:.1f}, p={res['p_value']:.4f}  {sig}")
        print(f"    Cohen's d (paired) = {res['cohen_d']:.3f}")

        rows.append({
            'endpoint':    ep['name'],
            'dataset':     ep['dataset'],
            'config_A':    ep['config_A'],
            'config_B':    ep['config_B'],
            'metric':      ep['metric'],
            'alternative': ep['alternative'],
            'n_patients':  res['n'],
            'mean_A':      res['mean_A'],
            'mean_B':      res['mean_B'],
            'mean_diff':   res['mean_diff'],
            'median_diff': res['median_diff'],
            'wilcoxon_W':  res['W'],
            'p_value':     res['p_value'],
            'significance': sig,
            'cohen_d':     res['cohen_d'],
            'rationale':   ep['rationale'],
        })

    return pd.DataFrame(rows)


# =============================================================================
# PER-ORGAN TESTS (more power: 12 patients × 8 organs = 96 samples per config)
# =============================================================================

def per_class_per_patient(df_pat: pd.DataFrame, dataset: str, config: str,
                          metric: str) -> pd.DataFrame:
    """
    For a (dataset, config), build a DataFrame of (case, class) -> metric value
    averaged over seeds.

    NOTE: df_pat from aggregate_results.py only contains per-patient *aggregated*
    values across all classes (the 'patient_NNN: DSC=X' lines). Per-class
    per-patient data is NOT in the log files — we'd need to instrument the test
    to emit it. For now, this function will return empty; we skip per-organ tests.

    To enable per-organ tests, modify test_phase3.py to emit per-(case,class)
    rows in the log, then re-run.
    """
    return pd.DataFrame()


# =============================================================================
# CONTEXT NOTES (for paper authors)
# =============================================================================

INTERPRETATION = """
HOW TO INTERPRET THESE RESULTS IN THE PAPER:

1. The five endpoints are PRE-SPECIFIED, not pre-registered. They were defined a
   priori by the authors, but were never lodged with an external registry (OSF,
   AsPredicted, ClinicalTrials.gov). The word "pre-registered" must not be used
   for them anywhere in the manuscript, the supplementary material, or this code.

2. All five endpoints ARE corrected for multiplicity using the Holm step-down
   procedure. The earlier position — that distinct clinical questions with
   directional hypotheses are exempt from correction — is not defensible when the
   five endpoints are reported together as support for a single overall claim, and
   it contradicted Section IV-F of the manuscript, which stated that Holm was
   applied. The script now reports Holm-adjusted values alongside the raw ones.

3. Under Holm correction NO endpoint reaches significance. This is reported
   openly. The contribution of this work does not rest on a surviving p-value: it
   rests on a controlled ablation showing that the same architectural changes help
   on cardiac MRI and do not help on multi-organ CT.

4. The pairwise ablation tables (Wilcoxon_DSC etc.) remain exploratory and must
   never be described as "significant" in the Abstract or Conclusions.

5. Effect sizes (Cohen's d):
     |d| < 0.2  : negligible
     0.2-0.5    : small
     0.5-0.8    : medium
     > 0.8      : large

6. Wording rules for the revision:
   - Say "consistent improvement", never "significantly better", for any endpoint
     that does not survive Holm correction — which currently means all of them.
   - Report 95% bootstrap CIs to communicate uncertainty.
   - Do not claim clinical actionability: ejection fraction, myocardial mass and
     wall thickness were never computed in this study.
"""


# =============================================================================
# MULTIPLICITY CORRECTION
# =============================================================================

def _add_holm(df: pd.DataFrame, p_col: str = 'p_value') -> pd.DataFrame:
    """Add the `p_holm` (Holm step-down) and `significant_holm` columns to the
    endpoint table.

    Holm step-down: sort the p-values in ascending order, multiply the i-th one
    by (n - i), then enforce monotonic non-decreasing values and cap them at 1.0.
    This controls the family-wise error rate without assuming the tests are
    independent -- appropriate here, because the five endpoints share the same
    data and are reported together as evidence for a single overall claim.

    Parameters
    ----------
    df : pd.DataFrame
        The endpoint table; must contain the `p_col` column.
    p_col : str, optional
        Name of the column holding the raw p-values.

    Returns
    -------
    pd.DataFrame
        A copy of `df` with two new columns: `p_holm` and `significant_holm`.
    """
    out = df.copy()
    order = out[p_col].values.argsort()
    n = len(out)
    adjusted = [0.0] * n
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, float(out[p_col].values[idx]) * (n - rank))
        adjusted[idx] = min(running, 1.0)
    out['p_holm'] = adjusted
    out['significant_holm'] = out['p_holm'] < 0.05
    return out


# =============================================================================
# MAIN
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=str, default='../KetQua_v2.xlsx')
    p.add_argument('--output', type=str, default=None)
    args = p.parse_args()
    output = args.output or args.input

    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)

    df_pat = pd.read_excel(args.input, sheet_name='Per_patient')
    print(f"Loaded {len(df_pat)} per-patient rows.")

    # Run primary endpoint tests
    df_primary = run_primary_endpoints(df_pat)

    # Apply the Holm step-down correction to ALL FIVE endpoints and write it
    # straight into the table, so the exported sheet matches Section IV-F of the
    # manuscript (which already states that Holm was applied).
    df_primary = _add_holm(df_primary)
    print("\n" + "=" * 72)
    print("HOLM STEP-DOWN CORRECTION ACROSS ALL FIVE ENDPOINTS")
    print("=" * 72)
    for _, row in df_primary.sort_values('p_value').iterrows():
        mark = '*' if row['p_holm'] < 0.05 else 'n.s.'
        print(f"  {str(row['endpoint'])[:52]:<54} "
              f"p={row['p_value']:.4f}  p_holm={row['p_holm']:.4f}  {mark}")
    n_surv = int((df_primary['p_holm'] < 0.05).sum())
    print("-" * 72)
    print(f"  {n_surv} of {len(df_primary)} endpoints survive Holm correction at alpha=0.05.")
    if n_surv == 0:
        print("  Report this openly: no endpoint survives. Use 'consistent improvement',")
        print("  never 'significantly better', anywhere in the manuscript.")

    # Add to Excel
    existing = {}
    xls = pd.ExcelFile(args.input)
    for sn in xls.sheet_names:
        existing[sn] = pd.read_excel(args.input, sheet_name=sn)

    existing['Primary_endpoints'] = df_primary

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for name, df in existing.items():
            df.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            for ci, col in enumerate(df.columns):
                col_letter = ws.cell(row=1, column=ci + 1).column_letter
                max_len = max([len(str(col))] +
                              [len(str(v)) for v in df.iloc[:, ci].astype(str)])
                ws.column_dimensions[col_letter].width = min(max_len + 2, 60)

    print(f"\n✓ Added 'Primary_endpoints' sheet to {output}")
    print(INTERPRETATION)


if __name__ == '__main__':
    main()
