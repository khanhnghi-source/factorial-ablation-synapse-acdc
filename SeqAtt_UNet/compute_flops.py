"""
COMPUTE FLOPS, PARAMS, THROUGHPUT for the computational-efficiency table
================================================================
Measures for all 7 configs:

  - Baseline, no DS         (mlp=standard, no CBAM, no DS)   <- R1.3 control
  - Baseline (DS only, mlp=standard, no CBAM)
  - +CBAM+DS                (mlp=standard, use_cbam=True)
  - +BiLSTM+DS (no CBAM)    (mlp=bilstm,   use_cbam=False)   <- R1.3 control
  - +CBAM+BiLSTM+DS         (mlp=bilstm,   use_cbam=True)
  - +CBAM+Hybrid+DS         (mlp=hybrid,   use_cbam=True)
  - +CBAM+BiLSTM+DS+SkipCBAM (Phase 4)

WARNING: throughput (img/s) and peak VRAM are HARDWARE-DEPENDENT. Table V in the
manuscript reports these on a local NVIDIA GTX 1080 Ti. Re-measure every row on
the SAME GPU in ONE run, otherwise the img/s column mixes devices and is not
comparable. Params and GFLOPs are hardware-independent.

Metrics:
  - Total params (M)
  - Trainable params (M)
  - GFLOPs (forward at 224×224)
  - Throughput (img/s, GPU forward batch=1, batch=8)
  - Peak VRAM (MB)

Output: KetQua_v2.xlsx sheet "Computational"

USAGE:
    python compute_flops.py                  # measure all configs
    python compute_flops.py --device cpu     # CPU only (no throughput)
    python compute_flops.py --img_size 224 --num_classes 9
"""

import argparse
import os
import sys
import time
import warnings
from typing import Dict, List

warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import pandas as pd


def measure_config(config_name: str, build_fn, img_size: int = 224,
                   num_classes: int = 9, device: str = 'cuda',
                   n_warmup: int = 50, n_iters: int = 20) -> Dict:
    """
    Measure one model config.

    Args:
        config_name: human-readable label
        build_fn: callable() → nn.Module
        img_size, num_classes: input dims
        device: 'cuda' or 'cpu'
        n_warmup, n_iters: throughput measurement params

    Returns dict of metrics.
    """
    import torch
    import torch.nn as nn

    print(f"\n{'='*60}")
    print(f"Measuring: {config_name}")
    print(f"{'='*60}")

    model = build_fn()
    model.eval()

    # ---- Params ----
    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Params total:     {total_p/1e6:.2f} M")
    print(f"  Params trainable: {train_p/1e6:.2f} M")

    # ---- GFLOPs ----
    flops_g = None
    try:
        from fvcore.nn import FlopCountAnalysis, parameter_count
        x = torch.randn(1, 3, img_size, img_size)
        if device == 'cuda' and torch.cuda.is_available():
            model_for_flop = model.cpu()  # fvcore works better on CPU
        else:
            model_for_flop = model

        # Disable DS for cleaner forward (single output)
        if hasattr(model_for_flop, 'disable_deep_supervision'):
            model_for_flop.disable_deep_supervision()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            flops = FlopCountAnalysis(model_for_flop, x)
            flops.unsupported_ops_warnings(False)
            flops.uncalled_modules_warnings(False)
            flops_total = flops.total()
            flops_g = flops_total / 1e9
            print(f"  GFLOPs (224×224, bs=1): {flops_g:.2f}")
    except ImportError:
        print()
        print("  " + "!" * 70)
        print("  fvcore IS NOT INSTALLED -> the GFLOPs column will be all N/A and will")
        print("  OVERWRITE the previous values in the Excel sheet.")
        print("  Install it and re-run:   pip install fvcore")
        print("  " + "!" * 70)
        print()
    except Exception as e:
        print(f"  [WARN] FLOPs computation failed: {e}")

    # ---- Throughput + Peak VRAM ----
    throughput_bs1 = None
    throughput_bs8 = None
    peak_vram_mb = None

    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
        if hasattr(model, 'disable_deep_supervision'):
            model.disable_deep_supervision()

        for batch_size in [1, 8]:
            try:
                x = torch.randn(batch_size, 3, img_size, img_size).cuda()
                # Warmup
                with torch.no_grad():
                    for _ in range(n_warmup):
                        _ = model(x)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

                # Timing
                t0 = time.perf_counter()
                with torch.no_grad():
                    for _ in range(n_iters):
                        _ = model(x)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0

                ips = (n_iters * batch_size) / dt
                if batch_size == 1:
                    throughput_bs1 = ips
                    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
                else:
                    throughput_bs8 = ips
                print(f"  Throughput bs={batch_size}: {ips:.1f} img/s")
            except RuntimeError as e:
                print(f"  [WARN] Throughput bs={batch_size} failed: {e}")
                break

        if peak_vram_mb is not None:
            print(f"  Peak VRAM (bs=1): {peak_vram_mb:.0f} MB")

        # Cleanup
        del model
        torch.cuda.empty_cache()

    return {
        'config':            config_name,
        'params_M':          total_p / 1e6,
        'trainable_params_M': train_p / 1e6,
        'gflops':            flops_g,
        'throughput_bs1_ips': throughput_bs1,
        'throughput_bs8_ips': throughput_bs8,
        'peak_vram_MB':      peak_vram_mb,
    }


# =============================================================================
# Build functions for each config
# =============================================================================

def build_baseline_ds(img_size, num_classes):
    """+DS only, no CBAM, mlp=standard"""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from networks.vit_seg_modeling_phase3 import create_model_with_ds
    return create_model_with_ds(
        config_name='R50-ViT-B_16',
        img_size=img_size,
        num_classes=num_classes,
        use_cbam=False,
        mlp_type='standard',
        use_deep_supervision=True,
        num_ds_outputs=4,
    )


def build_cbam_ds(img_size, num_classes):
    from networks.vit_seg_modeling_phase3 import create_model_with_ds
    return create_model_with_ds(
        config_name='R50-ViT-B_16',
        img_size=img_size,
        num_classes=num_classes,
        use_cbam=True,
        mlp_type='standard',
        use_deep_supervision=True,
        num_ds_outputs=4,
    )


def build_cbam_bilstm_ds(img_size, num_classes):
    from networks.vit_seg_modeling_phase3 import create_model_with_ds
    return create_model_with_ds(
        config_name='R50-ViT-B_16',
        img_size=img_size,
        num_classes=num_classes,
        use_cbam=True,
        mlp_type='bilstm',
        use_deep_supervision=True,
        num_ds_outputs=4,
    )


def build_cbam_hybrid_ds(img_size, num_classes):
    from networks.vit_seg_modeling_phase3 import create_model_with_ds
    return create_model_with_ds(
        config_name='R50-ViT-B_16',
        img_size=img_size,
        num_classes=num_classes,
        use_cbam=True,
        mlp_type='hybrid',
        use_deep_supervision=True,
        num_ds_outputs=4,
    )


def build_cbam_bilstm_ds_skipcbam(img_size, num_classes):
    from networks.skip_cbam import create_skip_cbam_model
    return create_skip_cbam_model(
        config_name='R50-ViT-B_16',
        img_size=img_size,
        num_classes=num_classes,
        use_cbam=True,
        mlp_type='bilstm',
        use_deep_supervision=True,
        num_ds_outputs=4,
        skip_cbam_reduction=8,
    )


# --------------------------------------------------------------------------
# R1.3 factorial controls — added 2026-09-11.
#
# Table V originally had no row for either factorial control, which leaves the
# title's "3.6x the parameter cost" claim unverifiable for the configuration the
# paper actually recommends (+BiLSTM+DS, no CBAM). Both rows are needed:
#
#   1. Baseline, no DS  -> isolates the parameter cost of the DS heads themselves
#                          (R1.3: "the same implementation trained with and
#                           without DS on the same splits and seeds")
#   2. +BiLSTM+DS       -> the recommended practical configuration and the best
#                          ACDC cell in Table VI (90.45); a reader will ask what
#                          it costs (R2.3 cost-benefit)
#
# Expected: (2) ~= 374.45 M, since CBAM contributes ~0.18 M (105.49 - 105.31),
# so the 3.56x ratio against the DS-only baseline (105.31 M) should hold. MEASURE
# rather than assume — the estimate is not a substitute for the measured value.
# --------------------------------------------------------------------------

def build_baseline_nods(img_size, num_classes):
    """Factorial control: no CBAM, no DS, mlp=standard."""
    from networks.vit_seg_modeling_phase3 import create_model_with_ds
    return create_model_with_ds(
        config_name='R50-ViT-B_16',
        img_size=img_size,
        num_classes=num_classes,
        use_cbam=False,
        mlp_type='standard',
        use_deep_supervision=False,
        num_ds_outputs=4,
    )


def build_bilstm_ds(img_size, num_classes):
    """Factorial control / recommended config: BiLSTM + DS, no CBAM."""
    from networks.vit_seg_modeling_phase3 import create_model_with_ds
    return create_model_with_ds(
        config_name='R50-ViT-B_16',
        img_size=img_size,
        num_classes=num_classes,
        use_cbam=False,
        mlp_type='bilstm',
        use_deep_supervision=True,
        num_ds_outputs=4,
    )


CONFIGS = [
    ('Baseline, no DS',             build_baseline_nods),             # R1.3 control
    ('Baseline (DS only)',          build_baseline_ds),
    ('+CBAM+DS',                    build_cbam_ds),
    ('+BiLSTM+DS (no CBAM)',        build_bilstm_ds),                 # R1.3 control
    ('+CBAM+BiLSTM+DS',             build_cbam_bilstm_ds),
    ('+CBAM+Hybrid+DS',             build_cbam_hybrid_ds),
    ('+CBAM+BiLSTM+DS+SkipCBAM',    build_cbam_bilstm_ds_skipcbam),  # Phase 4
]


# =============================================================================
# MAIN
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--num_classes', type=int, default=9)
    p.add_argument('--device', type=str, default='cuda',
                   choices=['cuda', 'cpu'])
    p.add_argument('--n_iters', type=int, default=20,
                   help='Iters for throughput timing')
    p.add_argument('--n_warmup', type=int, default=50,
                   help='Warmup iters before timing starts (default 50; 5 is far too '
                        'few and penalises the first config measured, because cuDNN '
                        'has not settled yet)')
    p.add_argument('--output', type=str, default='../KetQua_v2.xlsx',
                   help='Excel to append "Computational" sheet')
    p.add_argument('--configs', type=str, default='all',
                   help='Comma-separated list, or "all". Names: '
                        '"baseline_nods", "baseline", "cbam", "bilstm_nocbam", '
                        '"bilstm", "hybrid", "skip_cbam"')
    p.add_argument('--allow_partial', action='store_true',
                   help='Permit writing a subset of configs to the Excel sheet, '
                        'discarding rows not measured in this run. Off by default.')
    args = p.parse_args()

    # Import torch lazily to give clear error if missing
    try:
        import torch
    except ImportError:
        print("ERROR: torch not installed. Run: pip install torch")
        return

    print(f"Device: {args.device}")
    print(f"Torch: {torch.__version__}")
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            args.device = 'cpu'
        else:
            print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Filter configs if requested
    selected = CONFIGS
    if args.configs != 'all':
        wanted = set(args.configs.split(','))
        name_map = {
            'baseline_nods':  'Baseline, no DS',           # R1.3 control
            'baseline':       'Baseline (DS only)',
            'cbam':           '+CBAM+DS',
            'bilstm_nocbam':  '+BiLSTM+DS (no CBAM)',      # R1.3 control
            'bilstm':         '+CBAM+BiLSTM+DS',
            'hybrid':         '+CBAM+Hybrid+DS',
            'skip_cbam':      '+CBAM+BiLSTM+DS+SkipCBAM',
        }
        wanted_full = {name_map.get(w, w) for w in wanted}
        selected = [(n, fn) for (n, fn) in CONFIGS if n in wanted_full]

    print(f"\nMeasuring {len(selected)} configs at {args.img_size}×{args.img_size}, "
          f"num_classes={args.num_classes}, n_warmup={args.n_warmup}, n_iters={args.n_iters}")

    # ------------------------------------------------------------------
    # Warm the GPU up BEFORE measuring the first config.
    #
    # Observed on 11/09 and 14/09: the FIRST config measured is always slower
    # than the second, even though the two Baseline rows run EXACTLY THE SAME
    # forward graph (deep supervision is disabled before timing). The cause is
    # cuDNN autotuning plus GPU clocks that have not settled after only a few
    # warmup iterations. Magnitude:
    #     torch 2.4.1, 5 warmup  -> 0.7% gap
    #     torch 2.6.0, 5 warmup  -> 4.9% gap   <- unacceptable
    # Run a small conv stack for a few seconds to bring the GPU into a steady
    # state before the measurement loop starts, and raise the default n_warmup
    # to 50.
    # ------------------------------------------------------------------
    if args.device == 'cuda' and torch.cuda.is_available():
        print("Warming up the GPU before measuring ...", end=' ', flush=True)
        import torch.nn as nn
        warm = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
                             nn.Conv2d(64, 64, 3, padding=1)).cuda().eval()
        x = torch.randn(8, 3, args.img_size, args.img_size).cuda()
        with torch.no_grad():
            for _i in range(200):
                # Do NOT bind the result to a variable. This was originally
                # written as "_ = warm(x)", and `_` then held on to the output
                # tensor (8,64,224,224) = 98.0 MB AFTER the loop finished;
                # `del warm, x` never touched it. Consequence: every row of the
                # Peak VRAM column was inflated by exactly 98 MB (458 -> 556,
                # 1569 -> 1667, 1678 -> 1776). The constant offset was the giveaway.
                warm(x)
        torch.cuda.synchronize()
        del warm, x
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        resid = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"done ({resid:.1f} MB still held).")
        if resid > 5:
            print("  WARNING: the warmup did not release everything -> the Peak VRAM column will be inflated.")

    results = []
    for name, build_fn in selected:
        try:
            row = measure_config(name, lambda fn=build_fn: fn(args.img_size, args.num_classes),
                                  img_size=args.img_size, num_classes=args.num_classes,
                                  device=args.device, n_iters=args.n_iters,
                                  n_warmup=args.n_warmup)
            results.append(row)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[ERROR] Failed to measure {name}: {e}")

    if not results:
        print("No results.")
        return

    df = pd.DataFrame(results)

    # Format nicely for paper
    df['Params (M)'] = df['params_M'].map(lambda x: f"{x:.2f}" if x else "N/A")
    df['GFLOPs'] = df['gflops'].map(lambda x: f"{x:.2f}" if x else "N/A")
    df['Throughput bs=1 (img/s)'] = df['throughput_bs1_ips'].map(lambda x: f"{x:.1f}" if x else "N/A")
    df['Throughput bs=8 (img/s)'] = df['throughput_bs8_ips'].map(lambda x: f"{x:.1f}" if x else "N/A")
    df['Peak VRAM (MB)'] = df['peak_vram_MB'].map(lambda x: f"{x:.0f}" if x else "N/A")

    display = df[['config', 'Params (M)', 'GFLOPs',
                  'Throughput bs=1 (img/s)', 'Throughput bs=8 (img/s)',
                  'Peak VRAM (MB)']]

    print("\n" + "=" * 80)
    print("FINAL TABLE V (Computational Analysis)")
    print("=" * 80)
    print(display.to_string(index=False))

    # ------------------------------------------------------------------
    # MEASUREMENT QUALITY CHECK
    #
    # The two Baseline rows run exactly the same forward graph (DS is disabled
    # before timing), so the gap between them is PURE NOISE -- it measures the
    # noise floor directly, with no assumptions.
    #
    # The 2.5% threshold comes from FOUR observations of this very Baseline pair
    # on the GTX 1080 Ti, not from a single measurement:
    #     11/09 torch 2.4.1, warmup 5   -> 0.7%
    #     14/09 torch 2.6.0, warmup 5   -> 4.9%
    #     14/09 warmup 100 (VRAM leak)  -> 1.4%
    #     14/09 warmup 100 (fixed)      -> 1.9%
    # The sign of the gap flips between runs -> it really is noise. The original
    # 1.5% threshold came from ONE lucky observation and was refuted by the three
    # that followed. The conclusion to use in the paper: throughput on this
    # machine reproduces to within about +-2%, so any gap below ~2% in the table
    # is not interpretable.
    # ------------------------------------------------------------------
    bl = {r['config']: r['throughput_bs1_ips'] for r in results
          if r['config'] in ('Baseline, no DS', 'Baseline (DS only)')
          and r['throughput_bs1_ips']}
    if len(bl) == 2:
        a, b = bl['Baseline, no DS'], bl['Baseline (DS only)']
        noise = abs(a - b) / max(a, b) * 100
        print(f"\nNoise floor (the two Baseline rows, same forward graph): "
              f"{a:.1f} vs {b:.1f} img/s = {noise:.1f}%")
        if noise > 2.5:
            print("  " + "!" * 70)
            print(f"  WARNING: {noise:.1f}% exceeds the usual noise level (~2%).")
            print("  Close background applications, let the GPU cool down, and re-run.")
            print("  " + "!" * 70)
        else:
            print(f"  OK -- within the usual noise level. Any gap below ~{noise:.0f}% in the")
            print("  throughput column is NOT interpretable; state this in the paper.")

    try:
        import torch as _t
        print(f"\nEnvironment: torch {_t.__version__}"
              + (f" · {_t.cuda.get_device_name(0)}" if _t.cuda.is_available() else ""))
        print("  Record this version in the paper: throughput depends on the runtime, not only on the hardware.")
    except Exception:
        pass

    # Save to Excel (append "Computational" sheet)
    # NOTE: the sheet is REPLACED, not merged. Running a subset of configs would
    # silently drop the rows measured previously, so refuse to do that unless the
    # caller explicitly opts in.
    if (os.path.exists(args.output) and len(selected) < len(CONFIGS)
            and not args.allow_partial):
        print("\n" + "!" * 78)
        print("REFUSING TO WRITE: this would REPLACE the 'Computational' sheet with")
        print(f"only {len(selected)} of {len(CONFIGS)} configs, discarding the rest.")
        print("Re-run without --configs (measures all 7 on one GPU, which is also")
        print("what makes the img/s column comparable), or pass --allow_partial if")
        print("you really intend to overwrite the sheet with a subset.")
        print("!" * 78)
        print("\nMeasured values above are still valid — they simply were not saved.")
        return

    if os.path.exists(args.output):
        existing = {}
        xls = pd.ExcelFile(args.output)
        for sn in xls.sheet_names:
            existing[sn] = pd.read_excel(args.output, sheet_name=sn)
        existing['Computational'] = df

        with pd.ExcelWriter(args.output, engine='openpyxl') as writer:
            for name, ddf in existing.items():
                ddf.to_excel(writer, sheet_name=name, index=False)
                ws = writer.sheets[name]
                for ci, col in enumerate(ddf.columns):
                    col_letter = ws.cell(row=1, column=ci + 1).column_letter
                    max_len = max([len(str(col))] +
                                  [len(str(v)) for v in ddf.iloc[:, ci].astype(str)])
                    ws.column_dimensions[col_letter].width = min(max_len + 2, 50)
        print(f"\n[OK] Appended 'Computational' sheet to {args.output}")
    else:
        df.to_excel(args.output, sheet_name='Computational', index=False)
        print(f"\n[OK] Created {args.output}")

    # LaTeX snippet for paper
    print("\n--- LaTeX snippet for IEEE paper (Table V) ---")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Computational analysis at $224\times 224$ input resolution.}")
    print(r"\label{tab:computational}")
    print(r"\begin{tabular}{lrrrr}")
    print(r"\toprule")
    print(r"Method & Params (M) & GFLOPs & Throughput (img/s) & VRAM (MB) \\")
    print(r"\midrule")
    for _, r in df.iterrows():
        params = f"{r['params_M']:.1f}" if r['params_M'] else '-'
        gflops = f"{r['gflops']:.1f}" if r['gflops'] else '-'
        tput   = f"{r['throughput_bs1_ips']:.1f}" if r['throughput_bs1_ips'] else '-'
        vram   = f"{r['peak_vram_MB']:.0f}" if r['peak_vram_MB'] else '-'
        cfg = r['config'].replace('+', r'+').replace('SkipCBAM', r'\textbf{SkipCBAM}')
        print(f"{cfg} & {params} & {gflops} & {tput} & {vram} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == '__main__':
    main()
