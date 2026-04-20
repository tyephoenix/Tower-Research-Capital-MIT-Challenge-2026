"""
Problem 1 standalone — run P1a (index detection) + P1b (coefficients)
without matrix imputation.

  --method tye   (default): row-residual test + NNLS coefficient recovery
  --method deven          : 7-stage stability verification of hardcoded decomps

Writes:
  ../answers/problem1a_answer-<method>.csv
  ../answers/problem1b_answer-<method>.csv

The full pipeline (P1 + P2 + EM refinement) lives in ../problem-2/main.py.
"""

import argparse
import importlib
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def run_tye(args):
    sys.path.insert(0, str(HERE))
    import numpy as np
    import pandas as pd
    from tye.candidates import (process_candidates, hardcoded_candidates,
                                DEFAULT_MIN_ROWS, DEFAULT_N_SPLITS,
                                DEFAULT_RMSE_THRESHOLD)
    from tye.coefficients import recover_coefficients, distribute_through

    np.random.seed(42)
    df = pd.read_csv(REPO / "data" / "limestone_data_challenge_2026.data.csv")
    cols = [c for c in df.columns if c.startswith("col_")]
    data = df[cols].copy()
    arr = data.values.astype(np.float64)
    vmask = ~np.isnan(arr)
    D = len(cols)

    if args.candidates:
        result = hardcoded_candidates(cols, args.candidates)
    else:
        result = process_candidates(
            arr, vmask, cols,
            min_rows=args.min_rows, n_splits=DEFAULT_N_SPLITS,
            rmse_threshold=args.threshold,
        )

    decompositions, filled, fmask = recover_coefficients(
        data, arr, vmask, cols,
        result["index_cols"], result["index_idxs"],
        result["farmer_cols"], result["farmer_idxs"],
    )

    accepted_cols = list(decompositions.keys())
    accepted_idxs = [cols.index(c) for c in accepted_cols]
    farmer_idx_set = set(range(D)) - set(accepted_idxs)
    farmer_idxs = sorted(farmer_idx_set)
    farmer_cols = [cols[i] for i in farmer_idxs]

    expanded = distribute_through(decompositions, cols)
    return cols, accepted_cols, farmer_cols, decompositions, expanded


def run_deven(args):
    scripts_dir = HERE / "deven" / "scripts"
    deven_dir = HERE / "deven"
    sys.path.insert(0, str(scripts_dir))
    sys.path.insert(0, str(deven_dir))
    sys.path.insert(0, str(HERE))

    import numpy as np
    import pandas as pd

    csv_path = REPO / "data" / "limestone_data_challenge_2026.data.csv"

    # Run all 7 stability/verification stages to disk in a scratch dir,
    # then discard. The actual decomposition constants live in common.py.
    with tempfile.TemporaryDirectory() as tmp:
        outdir = Path(tmp)
        stages = [
            "01_validate_known_decomps",
            "02_stability_validation",
            "03_farmer_only_stepwise",
            "04_fill_and_rerun",
            "05_exact_fill_all",
            "06_refine_col30_col46",
        ]
        for stage in stages:
            print(f"\n{'=' * 70}\n  Deven stage: {stage}\n{'=' * 70}")
            old_argv = sys.argv
            sys.argv = [stage, "--csv", str(csv_path), "--outdir", str(outdir)]
            mod = importlib.import_module(stage)
            mod.main()
            sys.argv = old_argv

        # Stage 07 expects the filled matrix output from stage 05.
        # Its diagnostic classification is informational; skip cleanly if it
        # errors on sparse-row edge cases.
        print(f"\n{'=' * 70}\n  Deven stage: 07_classify_remaining\n{'=' * 70}")
        old_argv = sys.argv
        sys.argv = ["07_classify_remaining",
                    "--csv", str(outdir / "filled_matrix.csv"),
                    "--outdir", str(outdir)]
        try:
            mod = importlib.import_module("07_classify_remaining")
            mod.main()
        except Exception as e:
            print(f"  (stage 07 diagnostic skipped: {type(e).__name__}: {e})")
        sys.argv = old_argv

    # Emit CSVs from Deven's verified constants
    from deven.common import CONFIRMED_INDICES, KNOWN_DECOMPS_EXACT

    df = pd.read_csv(csv_path)
    cols = [c for c in df.columns if c.startswith("col_")]
    accepted_cols = sorted(CONFIRMED_INDICES)
    farmer_cols = [c for c in cols if c not in CONFIRMED_INDICES]

    # One canonical decomp per index — prefer the latest (most refined) entry,
    # which is the last occurrence in KNOWN_DECOMPS_EXACT.
    chosen = {}
    for idx_col, weights in KNOWN_DECOMPS_EXACT:
        chosen[idx_col] = weights  # overwrite — last one wins

    # Package to match the shape downstream code expects
    decompositions = {}
    for c, w in chosen.items():
        farmer_names = list(w.keys())
        decompositions[c] = {
            "farmer_idxs": [cols.index(n) for n in farmer_names],
            "coefs": np.array([w[n] for n in farmer_names], dtype=float),
        }
    # distribute_through (simple inline version since Tye's coefficients.py
    # may not be on path)
    expanded = {}
    index_set = set(decompositions.keys())
    memo = {}

    def dist(col):
        if col in memo:
            return memo[col]
        if col not in decompositions:
            memo[col] = {col: 1.0}
            return memo[col]
        out = {}
        dec = decompositions[col]
        for fi, wt in zip(dec["farmer_idxs"], dec["coefs"]):
            child = cols[fi]
            if child in index_set:
                for k, v in dist(child).items():
                    out[k] = out.get(k, 0.0) + float(wt) * v
            else:
                out[child] = out.get(child, 0.0) + float(wt)
        memo[col] = out
        return out

    for c in decompositions:
        d = dist(c)
        farmer_names = list(d.keys())
        expanded[c] = {
            "farmer_idxs": [cols.index(n) for n in farmer_names],
            "coefs": np.array([d[n] for n in farmer_names], dtype=float),
        }

    return cols, accepted_cols, farmer_cols, decompositions, expanded


def write_answers(method, cols, accepted_cols, farmer_cols,
                  decompositions, expanded):
    import pandas as pd

    answers_dir = REPO / "answers"
    answers_dir.mkdir(exist_ok=True)
    suffix = f"-{method}"

    p1a = answers_dir / f"problem1a_answer{suffix}.csv"
    p1b = answers_dir / f"problem1b_answer{suffix}.csv"

    pd.DataFrame({
        "column": farmer_cols + accepted_cols,
        "is_index": [False] * len(farmer_cols) + [True] * len(accepted_cols),
    }).to_csv(p1a, index=False)

    rows = []
    for idx_col in accepted_cols:
        source = expanded.get(idx_col) or decompositions.get(idx_col)
        if source is None:
            continue
        for fi, w in zip(source["farmer_idxs"], source["coefs"]):
            if w > 1e-4:
                rows.append({
                    "index_col": idx_col,
                    "constituent_col": cols[fi],
                    "coef": round(float(w), 6),
                })
    pd.DataFrame(rows).to_csv(p1b, index=False)
    print(f"\n  Saved {p1a}")
    print(f"  Saved {p1b}")


def main():
    parser = argparse.ArgumentParser(
        description="Problem 1 standalone (no imputation)")
    parser.add_argument("--method", choices=["tye", "deven"], default="tye")
    parser.add_argument("--min-rows", type=int, default=25)
    parser.add_argument("--threshold", type=float, default=3.5)
    parser.add_argument("--candidates", type=int, nargs="+", default=None,
                        help="(tye only) hardcode candidates by col number")
    args = parser.parse_args()

    print(f"Method: {args.method}")
    if args.method == "tye":
        cols, accepted, farmers, decomps, expanded = run_tye(args)
    else:
        cols, accepted, farmers, decomps, expanded = run_deven(args)

    print(f"\n  Indices ({len(accepted)}): {sorted(accepted)}")
    print(f"  Farmers: {len(farmers)}")
    write_answers(args.method, cols, accepted, farmers, decomps, expanded)


if __name__ == "__main__":
    main()
