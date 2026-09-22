"""
STATISTICAL TESTS for SeqAtt-UNet ablation comparisons
=======================================================
Reads per-patient data from KetQua_v2.xlsx (sheet "Per_patient")
and computes:

  1. Pairwise Wilcoxon signed-rank tests (per-patient DSC, HD95 voxel, HD95 mm)
     - For each dataset
     - For each pair of (config_A, config_B)
     - Across 5 seeds → mean per-patient value per config

  2. Bonferroni-Holm correction across the 6 pairs per dataset

  3. Bootstrap 95% confidence intervals for each config's mean Dice
     (bias-corrected accelerated bootstrap, 10000 resamples)

  4. Effect size (Cohen's d, paired)

Output: appends 3 new sheets to KetQua_v2.xlsx
  - Wilcoxon_DSC
  - Wilcoxon_HD95vox
  - Bootstrap_CI

Recommended for the ablation-table footnotes.

USAGE:
    python statistical_tests.py
    python statistical_tests.py --input ../KetQua_v2.xlsx --bootstrap_n 10000

Author: SeqAtt-UNet — statistical analysis for the ablation study
"""

import argparse
import os
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# UTILITIES
# =============================================================================

def per_patient_means(df_pat: pd.DataFrame, dataset: str, metric: str) -> pd.DataFrame:
    """
    For a given dataset and metric column, compute mean over 5 seeds for each
    (config, case). This gives one value per patient per config.

    Returns DataFrame: rows=case, cols=config, values=metric.
    """
    sub = df_pat[df_pat['dataset'] == dataset]
    grouped = (
        sub
        .groupby(['config', 'case'])[metric]
        .mean()
        .reset_index()
    )
    pivot = grouped.pivot(index='case', columns='config', values=metric)
    return pivot


def wilcoxon_pairwise(pivot: pd.DataFrame, alternative: str = 'two-sided',
                       direction: str = 'greater') -> pd.DataFrame:
    """
    Pairwise Wilcoxon signed-rank tests between all config columns.

    Args:
        pivot: DataFrame rows=case, cols=config, values=metric
        alternative: 'two-sided' for symmetric test
        direction: 'greater' (better config has HIGHER value, e.g. DSC)
                   or 'less' (better config has LOWER value, e.g. HD95)

    Returns long-form DataFrame with columns:
        config_A, config_B, n_patients, median_diff,
        statistic_W, p_value_raw, p_value_holm, cohen_d, conclusion
    """
    configs = sorted(pivot.columns.tolist())
    rows = []

    for ca, cb in combinations(configs, 2):
        # Paired data (drop NaN rows where either is missing)
        paired = pivot[[ca, cb]].dropna()
        a = paired[ca].values
        b = paired[cb].values
        n = len(paired)
        diffs = a - b

        # Wilcoxon (handle case with all-zero differences)
        try:
            if np.all(diffs == 0):
                W, p = np.nan, 1.0
            else:
                W, p = stats.wilcoxon(a, b, alternative=alternative, zero_method='wilcox')
        except Exception as e:
            W, p = np.nan, np.nan

        # Cohen's d (paired)
        d = np.mean(diffs) / np.std(diffs, ddof=1) if np.std(diffs, ddof=1) > 0 else 0.0

        # Median difference (more interpretable than mean for nonparametric)
        med_diff = float(np.median(diffs))

        rows.append({
            'config_A':       ca,
            'config_B':       cb,
            'n_patients':     n,
            'median_diff_A_minus_B': med_diff,
            'mean_A':          float(np.mean(a)),
            'mean_B':          float(np.mean(b)),
            'statistic_W':     float(W) if not np.isnan(W) else np.nan,
            'p_value_raw':     float(p) if not np.isnan(p) else np.nan,
            'cohen_d_paired':  d,
        })

    df = pd.DataFrame(rows)

    # Bonferroni-Holm correction
    p_raw = df['p_value_raw'].fillna(1.0).values
    order = np.argsort(p_raw)
    n_tests = len(p_raw)
    p_holm = np.zeros_like(p_raw)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (n_tests - rank) * p_raw[idx])
        running_max = max(running_max, adj)
        p_holm[idx] = running_max
    df['p_value_holm'] = p_holm

    # Conclusion at α = 0.05
    def conclude(row, direction=direction):
        if pd.isna(row['p_value_raw']):
            return 'N/A'
        if row['p_value_holm'] >= 0.05:
            return 'n.s. (not significant)'
        # Direction: higher = better (Dice) vs lower = better (HD95)
        better_a = (row['median_diff_A_minus_B'] > 0) if direction == 'greater' \
                   else (row['median_diff_A_minus_B'] < 0)
        winner = row['config_A'] if better_a else row['config_B']
        if row['p_value_holm'] < 0.001:
            sig = '***'
        elif row['p_value_holm'] < 0.01:
            sig = '**'
        else:
            sig = '*'
        return f"{winner} better {sig}"

    df['conclusion'] = df.apply(conclude, axis=1)
    df = df.sort_values('p_value_holm')
    return df


def bootstrap_ci(values: np.ndarray, n_resamples: int = 10000,
                 ci_level: float = 0.95, seed: int = 42) -> Tuple[float, float, float]:
    """
    Percentile bootstrap confidence interval for the mean.

    NOTE (2026-09-14): this was previously documented as "BCa-style", which it is
    not — there is no bias correction and no acceleration term, only the 2.5th and
    97.5th percentiles of the resampled means. The code is unchanged; only the
    description is corrected, because this file is released with the paper and a
    reader comparing the docstring against the implementation would rightly flag
    the mismatch.

    Returns (mean, ci_low, ci_high).
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    if n < 2:
        return float(np.mean(values)) if n == 1 else 0.0, np.nan, np.nan
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = np.mean(values[idx])
    alpha = 1.0 - ci_level
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(np.mean(values)), lo, hi


def bootstrap_per_config(df_pat: pd.DataFrame, dataset: str, metric: str,
                          n_resamples: int = 10000) -> pd.DataFrame:
    """
    Compute bootstrap 95% CI for mean metric, per config.

    The "value" for each patient is the mean across seeds.
    Then we bootstrap over patients to estimate the population mean.
    """
    rows = []
    pivot = per_patient_means(df_pat, dataset, metric)
    for config in pivot.columns:
        vals = pivot[config].dropna().values
        mean, lo, hi = bootstrap_ci(vals, n_resamples=n_resamples)
        rows.append({
            'dataset':    dataset,
            'config':     config,
            'metric':     metric,
            'n_patients': len(vals),
            'mean':       mean,
            'ci_95_low':  lo,
            'ci_95_high': hi,
            'ci_95':      f"[{lo:.4f}, {hi:.4f}]",
        })
    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=str, default='../KetQua_v2.xlsx',
                   help='Excel from aggregate_results.py')
    p.add_argument('--output', type=str, default=None,
                   help='Output Excel (default: overwrite input with new sheets)')
    p.add_argument('--bootstrap_n', type=int, default=10000)
    args = p.parse_args()

    output = args.output or args.input

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")

    print(f"Reading {args.input} ...")
    df_pat = pd.read_excel(args.input, sheet_name='Per_patient')
    print(f"Per_patient rows: {len(df_pat)}")
    print(f"Datasets: {sorted(df_pat['dataset'].unique())}")
    print(f"Configs:  {sorted(df_pat['config'].unique())}")

    # ---- Wilcoxon tests on DSC and HD95 -------------------------------------
    wilcoxon_results = []
    for dataset in sorted(df_pat['dataset'].unique()):
        print(f"\n=== Wilcoxon tests: {dataset} ===")
        for metric, direction, label in [
            ('dsc',      'greater', 'DSC (higher=better)'),
            ('hd95_vox', 'less',    'HD95_voxel (lower=better)'),
            ('hd95_mm',  'less',    'HD95_mm (lower=better)'),
        ]:
            pivot = per_patient_means(df_pat, dataset, metric)
            print(f"  {label}: {pivot.shape[0]} patients, {pivot.shape[1]} configs")
            res = wilcoxon_pairwise(pivot, direction=direction)
            res.insert(0, 'dataset', dataset)
            res.insert(1, 'metric', metric)
            wilcoxon_results.append(res)
            # Show top significant pairs
            top = res[res['p_value_holm'] < 0.05]
            if len(top) > 0:
                print(f"    Significant pairs (Holm-adjusted p<0.05):")
                for _, r in top.iterrows():
                    print(f"      {r['config_A']} vs {r['config_B']}: "
                          f"p_holm={r['p_value_holm']:.4f}, {r['conclusion']}")
            else:
                print(f"    No significant pairs after Holm correction.")

    df_wilcoxon = pd.concat(wilcoxon_results, ignore_index=True)

    # ---- Bootstrap 95% CI ----------------------------------------------------
    print(f"\n=== Bootstrap 95% CI (n_resamples={args.bootstrap_n}) ===")
    bootstrap_results = []
    for dataset in sorted(df_pat['dataset'].unique()):
        for metric in ['dsc', 'hd95_vox', 'hd95_mm']:
            bootstrap_results.append(
                bootstrap_per_config(df_pat, dataset, metric, args.bootstrap_n)
            )
    df_bootstrap = pd.concat(bootstrap_results, ignore_index=True)

    # Show summary
    print("\nBootstrap CI summary (Dice only):")
    summary = df_bootstrap[df_bootstrap['metric'] == 'dsc'][
        ['dataset', 'config', 'n_patients', 'mean', 'ci_95']
    ]
    print(summary.to_string(index=False))

    # ---- Write to Excel (preserve existing sheets) --------------------------
    print(f"\nWriting results to {output} ...")
    existing = {}
    if os.path.exists(args.input):
        xls = pd.ExcelFile(args.input)
        for sn in xls.sheet_names:
            existing[sn] = pd.read_excel(args.input, sheet_name=sn)

    # Replace/add new sheets
    existing['Wilcoxon_all'] = df_wilcoxon
    existing['Bootstrap_CI'] = df_bootstrap

    # Pretty Wilcoxon sheets per metric
    for metric, sheet in [('dsc', 'Wilcoxon_DSC'),
                          ('hd95_vox', 'Wilcoxon_HD95vox'),
                          ('hd95_mm',  'Wilcoxon_HD95mm')]:
        existing[sheet] = df_wilcoxon[df_wilcoxon['metric'] == metric].copy()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for name, df in existing.items():
            df.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            for ci, col in enumerate(df.columns):
                col_letter = ws.cell(row=1, column=ci + 1).column_letter
                max_len = max([len(str(col))] + [len(str(v)) for v in df.iloc[:, ci].astype(str)])
                ws.column_dimensions[col_letter].width = min(max_len + 2, 50)

    print(f"\n✓ Done. New sheets added to {output}:")
    print(f"    Wilcoxon_DSC, Wilcoxon_HD95vox, Wilcoxon_HD95mm, Wilcoxon_all, Bootstrap_CI")
    print("\nFor paper Table III footnote, use:")
    print("  * Wilcoxon signed-rank, p < 0.05 after Holm-Bonferroni correction")
    print("  * 95% CI from percentile bootstrap (10,000 resamples)")


if __name__ == '__main__':
    main()
