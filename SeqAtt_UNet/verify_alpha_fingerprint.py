"""
verify_alpha_fingerprint.py -- alpha fingerprint check: checkpoint vs. test log
==================================================================================

WHY THIS SCRIPT EXISTS
----------------------
On 11/09/2026 we found that the ``best_model.pth`` of ``Synapse +CBAM+DS seed
3456`` had been OVERWRITTEN by a later training run, while the matching test log
was never regenerated. The numbers reported in ``KetQua_v2.xlsx`` therefore
belong to a model that NO LONGER EXISTS::

    January log  : cbam1 = -2.2610 , cbam2 = -2.1888 , cbam3 = -2.0835
    11/09 re-run : cbam1 = -2.2605 , cbam2 = -2.1886 , cbam3 = -2.0813
    Mean Dice    : 0.7973  ->  0.8062   (a gap of 0.89 points)

Comparing file modification times is not trustworthy enough: the results live on
a cloud-synchronised drive, and the sync client may rewrite mtime. This script
relies on evidence that cannot be forged instead: ``test_phase3.py`` PRINTS the
``cbam*.alpha_raw`` values it read from the checkpoint at the very top of the
test log. Reading those same tensors back from the ``best_model.pth`` that is
currently on disk and comparing the two gives:

* a match    -> the log really does belong to this checkpoint, the numbers stand;
* a mismatch -> the checkpoint was replaced after the log was written, so the
  test MUST be re-run.

SCOPE
-----
Only configurations WITH CBAM can be checked this way (``+CBAM+DS``,
``+CBAM+BiLSTM+DS``, ``+CBAM+Hybrid+DS``, ``+SkipCBAM``). The configurations
without CBAM -- ``Baseline``, ``Baseline no DS``, ``+BiLSTM+DS`` -- print no
parameter to the log at all, so they can only be verified by re-running the
test. The script still lists them under the ``NO ALPHA`` label, so that it stays
clear which part of the results table is still unverified.

USAGE
-----
::

    python verify_alpha_fingerprint.py --model_root /path/to/model \\
                                       --log_roots results/test_log
    python verify_alpha_fingerprint.py --csv fingerprint.csv

The script is READ-ONLY -- it changes nothing and deletes nothing.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

# An alpha line as it appears in a test log:
#   [08:49:38.083]    transformer.embeddings.hybrid_model.cbam1.alpha_raw: -2.2610
RE_LOG_ALPHA = re.compile(
    r'\]\s+(?P<key>\S*alpha_raw):\s*(?P<val>-?[\d.]+)'
)

def _default_log_roots() -> List[str]:
    """
    Default directories in which to look for test logs.

    No machine-specific absolute path is hard-coded here: this script ships with
    the paper, so a private path would be useless to anyone else and would also
    expose a personal drive layout. Resolution order:

    1. the ``SEQATT_LOG_ROOTS`` environment variable (several paths, separated by
       ``os.pathsep``)
    2. ``./results/test_log`` and ``./test_log`` relative to the current directory
    3. ``../results/test_log`` and ``./test_log`` relative to this script's location
    """
    env = os.environ.get('SEQATT_LOG_ROOTS')
    if env:
        return [p for p in env.split(os.pathsep) if p]
    here = os.path.dirname(os.path.abspath(__file__))
    return [
        os.path.join(os.getcwd(), 'results', 'test_log'),
        os.path.join(os.getcwd(), 'test_log'),
        os.path.join(here, '..', 'results', 'test_log'),
        os.path.join(here, 'test_log'),
    ]


def build_log_index(roots: List[str]) -> Dict[str, str]:
    """
    Index every test log found underneath the given roots.

    The key is ``"<parent directory name>|<file name>"`` rather than the file
    name alone, because the file name encodes neither the dataset nor the
    configuration: ``TU_pretrain_..._s1234_tta_simple.txt`` appears identically
    under both the ACDC and the Synapse directories.

    Parameters
    ----------
    roots : list of str
        Root directories to scan recursively. Roots that do not exist are skipped.

    Returns
    -------
    dict
        Mapping from key -> absolute path.
    """
    index: Dict[str, str] = {}
    for root in roots:
        if not os.path.isdir(root):
            print(f'  [skipped] {root}  (does not exist)')
            continue
        n = 0
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                if not fn.endswith('_tta_simple.txt'):
                    continue
                key = os.path.basename(dirpath) + '|' + fn
                index.setdefault(key, os.path.join(dirpath, fn))
                n += 1
        print(f'  [scanned] {root}  ->  {n} logs')
    return index


def read_log_alphas(log_path: str) -> Dict[str, float]:
    """
    Read the ``alpha_raw`` values that the test log printed.

    If a log holds several runs appended one after another (which happens when a
    test is re-run and writes into the same file), the LAST occurrence is kept:
    it belongs to the most recent run, that is, the one matching the latest
    results in the file.
    """
    alphas: Dict[str, float] = {}
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            m = RE_LOG_ALPHA.search(line)
            if m:
                alphas[m.group('key')] = float(m.group('val'))
    return alphas


def read_ckpt_alphas(ckpt_path: str) -> Dict[str, float]:
    """Read the ``*alpha_raw`` tensors from a checkpoint, on CPU."""
    import torch

    obj = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    state = obj
    if isinstance(obj, dict):
        for k in ('state_dict', 'model', 'model_state_dict'):
            if k in obj and isinstance(obj[k], dict):
                state = obj[k]
                break

    out: Dict[str, float] = {}
    if isinstance(state, dict):
        for k, v in state.items():
            if k.endswith('alpha_raw'):
                try:
                    out[k] = float(v.reshape(-1)[0].item())
                except Exception:
                    pass
    return out


def compare(ckpt: Dict[str, float], log: Dict[str, float],
            tol: float) -> Tuple[str, str]:
    """
    Compare the two sets of alpha values.

    Returns
    -------
    (status, detail)
        status is one of {'MATCH', 'MISMATCH', 'NO ALPHA', 'LOG NO ALPHA'}.
    """
    if not ckpt:
        return 'NO ALPHA', 'checkpoint has no alpha_raw parameter (non-CBAM configuration)'
    if not log:
        return 'LOG NO ALPHA', 'log prints no alpha -- probably an older script version'

    worst_key, worst_diff = '', 0.0
    n_cmp = 0
    for k, cv in sorted(ckpt.items()):
        if k not in log:
            continue
        n_cmp += 1
        d = abs(cv - log[k])
        if d > worst_diff:
            worst_diff, worst_key = d, k

    if n_cmp == 0:
        return 'LOG NO ALPHA', 'no alpha key is shared between the log and the checkpoint'
    if worst_diff > tol:
        short = worst_key.split('.')[-2] if '.' in worst_key else worst_key
        return 'MISMATCH', (f'{short}: ckpt={ckpt[worst_key]:.4f} '
                            f'log={log[worst_key]:.4f} diff={worst_diff:.5f}')
    return 'MATCH', f'{n_cmp} alpha values match within tolerance {tol:g}'


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model_root', default='../model',
                   help='Directory holding the TU_* directories (default ../model)')
    p.add_argument('--log_roots', nargs='*', default=None,
                   help='Root directories that contain the test logs')
    p.add_argument('--tol', type=float, default=1e-4,
                   help='Tolerance of the comparison. Logs print 4 decimal places, '
                        'so 1e-4 is just tight enough; the true discrepancy of '
                        'seed 3456 was 5e-4.')
    p.add_argument('--skip_bs24', action='store_true', default=True,
                   help='Skip the quarantined batch_size=24 runs')
    p.add_argument('--csv', default='', help='Write the results to a CSV file')
    args = p.parse_args()
    if not args.log_roots:
        args.log_roots = _default_log_roots()

    try:
        import torch  # noqa: F401
    except ImportError:
        print('ERROR: torch is not installed.')
        return 1

    if not os.path.isdir(args.model_root):
        print(f'ERROR: model root not found: {args.model_root}')
        return 1

    print('\nIndexing test logs:')
    log_index = build_log_index(args.log_roots)
    print(f'  Total: {len(log_index)} logs\n')
    if not log_index:
        print('ERROR: no log could be indexed. Pass the correct --log_roots.')
        return 1

    rows: List[Dict[str, str]] = []

    for cfg_dir in sorted(os.listdir(args.model_root)):
        if not cfg_dir.startswith('TU_'):
            continue
        cfg_path = os.path.join(args.model_root, cfg_dir)
        if not os.path.isdir(cfg_path):
            continue

        for run_name in sorted(os.listdir(cfg_path)):
            run_path = os.path.join(cfg_path, run_name)
            ckpt = os.path.join(run_path, 'best_model.pth')
            if not os.path.isfile(ckpt):
                continue
            if args.skip_bs24 and '_bs24_' in run_name:
                continue

            key = f'test_log_{cfg_dir}|{run_name}_tta_simple.txt'
            log_path: Optional[str] = log_index.get(key)

            if log_path is None:
                status, detail = 'NO LOG', 'no matching test log found'
            else:
                try:
                    status, detail = compare(read_ckpt_alphas(ckpt),
                                             read_log_alphas(log_path), args.tol)
                except Exception as exc:                       # pragma: no cover
                    status, detail = 'ERROR', f'{type(exc).__name__}: {exc}'

            seed_m = re.search(r'_s(\d+)$', run_name)
            rows.append({
                'status': status,
                'config': cfg_dir[3:] if cfg_dir.startswith('TU_') else cfg_dir,
                'seed':   seed_m.group(1) if seed_m else '?',
                'detail': detail,
                'run':    run_name,
            })
            print(f'  {status:<13} {rows[-1]["config"]:<38} s{rows[-1]["seed"]:<5} {detail}')

    if not rows:
        print('\nNo checkpoint found.')
        return 0

    print('\n' + '=' * 78)
    print('SUMMARY')
    print('=' * 78)
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r['status']] = counts.get(r['status'], 0) + 1
    for k in sorted(counts):
        print(f'  {k:<13} {counts[k]:>3}')
    print(f'  {"TOTAL":<13} {len(rows):>3}')

    print('\nHOW TO READ THE STATUS')
    print('  MATCH        checkpoint alpha matches the log -> the numbers are usable')
    print('  MISMATCH     checkpoint WAS REPLACED after the log -> test MUST be re-run')
    print('  NO ALPHA     configuration without CBAM, cannot be checked this way')
    print('               -> can only be verified by re-running the test')
    print('  LOG NO ALPHA log written by an older script version, prints no alpha')
    print('  NO LOG       trained but not tested yet\n')

    if args.csv:
        import csv
        with open(args.csv, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=['status', 'config', 'seed', 'detail', 'run'])
            w.writeheader()
            w.writerows(rows)
        print(f'Written: {args.csv}')

    return 1 if counts.get('MISMATCH') else 0


if __name__ == '__main__':
    sys.exit(main())
