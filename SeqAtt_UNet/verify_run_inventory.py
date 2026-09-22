"""
Inventory of the runs already present under model/, before the 20 factorial runs
(R1.3) are launched.

Purpose:
    1. List the exp_name directories that ACTUALLY exist, in order to check whether
       the 4 ablations of the paper carry the names recorded in the project notes.
    2. Pre-compute the exp_name of the 2 new factorial configurations and WARN when
       one collides with an existing directory.
       (Critical Don't #5: log.txt and alpha_logs_*.json are results that CANNOT BE
       REPRODUCED.)

Why this script is needed:
    In train_phase3.py, argparse declares `--use_cbam` and `--deep_supervision` with
    `default=True`. Omitting a flag does NOT disable it -- `--no_cbam` /
    `--no_deep_supervision` must be used instead. With the wrong flags, exp_name
    collides with an existing directory and `os.makedirs(..., exist_ok=True)`
    overwrites the earlier results.

Usage:
    python verify_run_inventory.py
    python verify_run_inventory.py --model_root "/path/to/model"
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

# The five official seeds -- DO NOT change (R1.3 requires "same splits and seeds").
OFFICIAL_SEEDS: Tuple[int, ...] = (1234, 2345, 3456, 4567, 5678)

# Hyperparameters used by the 50 original runs, read from run_all_experiments.bat
# together with the argparse defaults.
TRAIN_DEFAULTS: Dict[str, object] = {
    "vit_name": "R50-ViT-B_16",
    "n_skip": 3,
    "max_epochs": 150,
    # ⚠️ 12, NOT 24. Inventory of model/ on 2026-09-05: every configuration has
    #    5 bs12 runs (the 5 official seeds) + 1 bs24 run (seed 1234 only, a trial).
    #    Table I of the paper states "Batch size 24" -- WRONG, needs fixing.
    "batch_size": 12,
    "augmentation": "strong",
    "base_lr": 0.01,
    "img_size": 224,
}


def build_exp_name(
    dataset: str,
    img_size: int,
    use_cbam: bool,
    mlp_type: str,
    deep_supervision: bool,
    scheduler: str = "warmup_cosine",
) -> str:
    """Reproduce exactly the exp_name logic of train_phase3.py (lines 304-315).

    Parameters
    ----------
    dataset : str
        'Synapse' or 'ACDC'.
    img_size : int
        Input image size.
    use_cbam : bool
        Whether CBAM is enabled.
    mlp_type : str
        'standard' | 'bilstm' | 'hybrid'.
    deep_supervision : bool
        Whether deep supervision is enabled.
    scheduler : str, optional
        Scheduler name; only appended to exp_name when it differs from 'warmup_cosine'.

    Returns
    -------
    str
        The exact exp_name that train_phase3.py would produce.
    """
    exp_name = f"TU_{dataset}{img_size}"
    if use_cbam:
        exp_name += "_CBAM"
    if mlp_type != "standard":
        exp_name += f"_{mlp_type.upper()}"
    exp_name += "_Phase3"
    if deep_supervision:
        exp_name += "_DS"
    if scheduler != "warmup_cosine":
        exp_name += f"_{scheduler}"
    return exp_name


def build_run_dir(seed: int, **overrides: object) -> str:
    """Reproduce the per-seed sub-directory level (train_phase3.py lines 320-322)."""
    cfg = {**TRAIN_DEFAULTS, **overrides}
    return (
        f"TU_pretrain_{cfg['vit_name']}_skip{cfg['n_skip']}_"
        f"epo{cfg['max_epochs']}_bs{cfg['batch_size']}_aug{cfg['augmentation']}_"
        f"lr{cfg['base_lr']}_{cfg['img_size']}_s{seed}"
    )


# ---------------------------------------------------------------------------
# Configuration definitions
# ---------------------------------------------------------------------------

# Configurations ALREADY RUN (per the results workbook). Used for the
# inventory only; they are not re-run.
EXISTING_CONFIGS: List[Dict[str, object]] = [
    {"label": "Baseline (DS only)",        "use_cbam": False, "mlp_type": "standard", "ds": True},
    {"label": "+CBAM+DS",                  "use_cbam": True,  "mlp_type": "standard", "ds": True},
    {"label": "+CBAM+BiLSTM+DS",           "use_cbam": True,  "mlp_type": "bilstm",   "ds": True},
    {"label": "+CBAM+Hybrid+DS",           "use_cbam": True,  "mlp_type": "hybrid",   "ds": True},
]

# The 2 NEW configurations for the R1.3 factorial.
NEW_CONFIGS: List[Dict[str, object]] = [
    {
        "label": "(a) Baseline WITHOUT DS  [R1.3 minimum required control]",
        "use_cbam": False, "mlp_type": "standard", "ds": False,
        "flags": "--no_cbam --no_deep_supervision --mlp_type standard",
    },
    {
        "label": "(b) BiLSTM+DS WITHOUT CBAM  [isolates the sequential contribution]",
        "use_cbam": False, "mlp_type": "bilstm", "ds": True,
        "flags": "--no_cbam --mlp_type bilstm",
    },
]


def parse_run_dir(name: str) -> Optional[Dict[str, str]]:
    """Extract the hyperparameters from a seed directory name.

    Expected form:
        TU_pretrain_<vit>_skip<N>_epo<E>_bs<B>_aug<A>_lr<LR>_<size>_s<SEED>

    Returns
    -------
    dict or None
        None when the name does not match the pattern (unexpected directory).
    """
    import re
    m = re.match(
        r"TU_pretrain_(?P<vit>.+?)_skip(?P<skip>\d+)_epo(?P<epo>\d+)_"
        r"bs(?P<bs>\d+)_aug(?P<aug>\w+?)_lr(?P<lr>[\d.]+)_(?P<size>\d+)_s(?P<seed>\d+)$",
        name,
    )
    return m.groupdict() if m else None


def detail_scan(model_root: str) -> None:
    """Detailed inventory of each seed directory: hyperparameters, unexpected seeds,
    incomplete runs.

    This serves two purposes:
      - R1.3: show that the factorial uses EXACTLY the 5 seeds and identical
              hyperparameters.
      - B2  : the Supplementary must list precisely which runs entered the paper.
    """
    print()
    print("=" * 78)
    print("DETAILED INVENTORY -- one entry per seed directory")
    print("=" * 78)

    exps = sorted(
        d for d in os.listdir(model_root)
        if os.path.isdir(os.path.join(model_root, d)) and d.startswith("TU_")
    )

    anomalies: List[str] = []

    for exp in exps:
        exp_dir = os.path.join(model_root, exp)
        subs = sorted(
            d for d in os.listdir(exp_dir)
            if os.path.isdir(os.path.join(exp_dir, d))
        )
        print(f"\n▸ {exp}  ({len(subs)} sub-directories)")

        seeds_found: List[int] = []
        hp_seen: Dict[str, int] = {}

        for sub in subs:
            sub_dir = os.path.join(exp_dir, sub)
            parsed = parse_run_dir(sub)

            # Completeness status of the run
            has_log = os.path.isfile(os.path.join(sub_dir, "log.txt"))
            pths = [f for f in os.listdir(sub_dir) if f.endswith(".pth")]
            alphas = [f for f in os.listdir(sub_dir) if f.startswith("alpha_logs")]
            events = [f for f in os.listdir(sub_dir) if f.startswith("events.out")]

            if parsed is None:
                print(f"    ⚠️  UNEXPECTED DIRECTORY (does not match the pattern): {sub}")
                anomalies.append(f"{exp}/{sub} -- name does not match the pattern")
                continue

            seed = int(parsed["seed"])
            seeds_found.append(seed)
            hp_key = (
                f"skip{parsed['skip']}_epo{parsed['epo']}_bs{parsed['bs']}_"
                f"aug{parsed['aug']}_lr{parsed['lr']}_{parsed['size']}"
            )
            hp_seen[hp_key] = hp_seen.get(hp_key, 0) + 1

            status = []
            status.append("log" if has_log else "NO log")
            status.append(f"{len(pths)} pth" if pths else "NO pth")
            if alphas:
                status.append(f"{len(alphas)} alpha")
            if events:
                status.append(f"{len(events)} tb")

            official = "  " if seed in OFFICIAL_SEEDS else " ⚠️UNEXPECTED SEED"
            complete = "" if (has_log and pths) else "   ← INCOMPLETE RUN"
            print(f"    seed {seed}{official}  [{', '.join(status)}]{complete}")

            if seed not in OFFICIAL_SEEDS:
                anomalies.append(f"{exp}/s{seed} -- seed outside the 5 official seeds")
            if not (has_log and pths):
                anomalies.append(f"{exp}/s{seed} -- incomplete run (log.txt or .pth missing)")

        # Compare against the 5 official seeds
        missing = [s for s in OFFICIAL_SEEDS if s not in seeds_found]
        extra = sorted(set(seeds_found) - set(OFFICIAL_SEEDS))
        dupes = sorted({s for s in seeds_found if seeds_found.count(s) > 1})

        if missing:
            print(f"    🔴 MISSING seeds: {missing}")
            anomalies.append(f"{exp} -- missing seeds {missing}")
        if extra:
            print(f"    ⚠️  EXTRA seeds: {extra}")
        if dupes:
            print(f"    ⚠️  DUPLICATE seeds (different hyperparameters?): {dupes}")

        # Hyperparameters must be identical within one exp
        if len(hp_seen) > 1:
            print("    🔴 INCONSISTENT HYPERPARAMETERS within the same configuration:")
            for k, v in sorted(hp_seen.items()):
                print(f"         {k}   × {v} runs")
            anomalies.append(f"{exp} -- {len(hp_seen)} different hyperparameter sets")

    print()
    print("=" * 78)
    if anomalies:
        print(f"SUMMARY: {len(anomalies)} point(s) to review")
        for a in anomalies:
            print(f"  • {a}")
        print()
        print("  ⚠️  Before answering R1.3, it must be settled EXACTLY which 5 runs of")
        print("      each configuration were used in KetQua_v2.xlsx. Extra or incomplete")
        print("      runs must not leak into the results table, and the Supplementary")
        print("      (B2) must list the correct ones.")
    else:
        print("SUMMARY: ✓ All configurations: 5 official seeds, consistent hyperparameters.")
    print("=" * 78)


def scan(model_root: Optional[str]) -> None:
    """Print the inventory report and warn about directory collisions."""
    root_ok = model_root is not None and os.path.isdir(model_root)

    print("=" * 78)
    print("RUN INVENTORY -- before launching the 20 factorial runs (R1.3)")
    print("=" * 78)
    print(f"model_root: {model_root}")
    if not root_ok:
        print("  ⚠️  This directory cannot be accessed.")
        print("      Re-run the script on a machine where the results drive is mounted.")
        print("      The predicted directory names below remain correct and usable.")
    print()

    # --- Part 1: directories that actually exist --------------------------
    actual: List[str] = []
    if root_ok:
        actual = sorted(
            d for d in os.listdir(model_root)
            if os.path.isdir(os.path.join(model_root, d)) and d.startswith("TU_")
        )
        print(f"[1] TU_* directories actually present ({len(actual)}):")
        for d in actual:
            n_seeds = 0
            sub = os.path.join(model_root, d)
            try:
                n_seeds = sum(
                    1 for s in os.listdir(sub)
                    if os.path.isdir(os.path.join(sub, s))
                )
            except OSError:
                pass
            print(f"    - {d}   ({n_seeds} seed directories)")
        print()

    # --- Part 2: check the 4 configurations already run -------------------
    print("[2] Matching the 4 ablations of the paper against the expected names:")
    for dataset in ("Synapse", "ACDC"):
        for cfg in EXISTING_CONFIGS:
            name = build_exp_name(
                dataset, TRAIN_DEFAULTS["img_size"],  # type: ignore[arg-type]
                bool(cfg["use_cbam"]), str(cfg["mlp_type"]), bool(cfg["ds"]),
            )
            if root_ok:
                mark = "✓ found" if name in actual else "✗ NOT FOUND"
            else:
                mark = "?"
            print(f"    {mark:>12}  {name:<42} = {cfg['label']}")
    print()
    if root_ok:
        print("    ⚠️  If 'TU_*_Phase3_DS' (Baseline without CBAM) is NOT FOUND, then the")
        print("        Baseline of the paper may have been run WITH CBAM -- the whole")
        print("        ablation table must be re-checked before answering R1.3.")
        print()

    # --- Part 3: the 2 new configurations + guard -------------------------
    print("[3] The two new factorial configurations -- collision check:")
    collision = False
    for cfg in NEW_CONFIGS:
        print(f"\n    {cfg['label']}")
        print(f"      CLI flags: {cfg['flags']}")
        for dataset in ("Synapse", "ACDC"):
            name = build_exp_name(
                dataset, TRAIN_DEFAULTS["img_size"],  # type: ignore[arg-type]
                bool(cfg["use_cbam"]), str(cfg["mlp_type"]), bool(cfg["ds"]),
            )
            if root_ok and name in actual:
                collision = True
                print(f"      🔴 COLLISION: {name}  ← STOP, this would overwrite old data!")
            else:
                print(f"      ✓ new:   {name}")
    print()

    # --- Part 4: argparse flag warning ------------------------------------
    print("[4] ⚠️  ARGPARSE TRAP -- read this before running:")
    print("      train_phase3.py declares:")
    print("        --use_cbam          action='store_true', default=True")
    print("        --deep_supervision  action='store_true', default=True")
    print("        --mlp_type          default='bilstm'")
    print("      → OMITTING A FLAG DOES NOT DISABLE IT. Use --no_cbam / --no_deep_supervision.")
    print("      → A command like 'python train_phase3.py --dataset Synapse --mlp_type standard'")
    print("        yields TU_Synapse224_CBAM_Phase3_DS and OVERWRITES the +CBAM run.")
    print()

    print("=" * 78)
    if collision:
        print("CONCLUSION: 🔴 DIRECTORY COLLISION -- do not run the factorial until resolved.")
    elif root_ok:
        print("CONCLUSION: ✓ No collision. Safe to run run_factorial_r13.")
    else:
        print("CONCLUSION: ? Drive not checked -- re-run where the results drive is mounted.")
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inventory existing runs and detect directory collisions before the factorial."
    )
    parser.add_argument(
        "--model_root", type=str,
        default=os.path.join("..", "model"),
        help="Directory holding the TU_* runs (default ../model, the path train_phase3.py uses)",
    )
    parser.add_argument(
        "--detail", action="store_true",
        help="Detailed per-seed inventory: hyperparameters, unexpected seeds, incomplete runs",
    )
    args = parser.parse_args()
    scan(args.model_root)
    if args.detail:
        if os.path.isdir(args.model_root):
            detail_scan(args.model_root)
        else:
            print("\n⚠️  --detail requires access to model_root. Skipped.")


if __name__ == "__main__":
    main()
