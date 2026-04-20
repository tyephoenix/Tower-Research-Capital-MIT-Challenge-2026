"""
Problem 5 pipeline — compute σ, build cache, run backtest.

Without --intermediates: computes σ in memory, runs backtest directly.
With --intermediates:    also saves intermediates/sigma.json to disk.

Prerequisites: run problem-2/main.py first (needs answers/ and
               problem-2/intermediates/coefficients.json).
"""

import os, sys, json, time, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from trade import (trading_problem_5, _load_decompositions, _topo_order,
                   _build_classes, DEFAULT_SIGMA)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data",
                         "limestone_data_challenge_2026.data.csv")
COMPLETED_PATH = os.path.join(os.path.dirname(__file__), "..", "answers",
                              "problem2_answer-tye.csv")

KNN_K = 20
PROJECTION_RANK = 12
KNN_WEIGHT = 0.5
N_SAMPLES = 500


def compute_sigma(arr, completed, vmask, cols, verbose=True):
    """LOO cross-validation: per-column prediction error σ."""
    col_means = completed.mean(axis=0)
    centered = completed - col_means
    _, s, Vt = np.linalg.svd(centered, full_matrices=False)
    Vt_r = Vt[:PROJECTION_RANK]

    rng = np.random.default_rng(42)
    sigmas = {}

    for j, col in enumerate(cols):
        obs_rows = np.where(vmask[:, j])[0]
        if len(obs_rows) > N_SAMPLES:
            sample_rows = rng.choice(obs_rows, N_SAMPLES, replace=False)
        else:
            sample_rows = obs_rows

        errors = []
        for row_idx in sample_rows:
            true_val = arr[row_idx, j]
            row_vals = arr[row_idx].copy()
            row_vals[j] = np.nan

            obs = ~np.isnan(row_vals)
            obs_idxs = np.where(obs)[0]
            if len(obs_idxs) < 5:
                continue

            dists = np.sum((completed[:, obs_idxs] - row_vals[obs_idxs]) ** 2,
                           axis=1)
            nn_idxs = np.argsort(dists)[:KNN_K]
            w = 1.0 / (np.sqrt(dists[nn_idxs]) + 1e-8)
            w /= w.sum()
            knn_pred = completed[nn_idxs, j] @ w

            V_obs = Vt_r[:, obs_idxs].T
            centered_obs = row_vals[obs_idxs] - col_means[obs_idxs]
            alpha, *_ = np.linalg.lstsq(V_obs, centered_obs, rcond=None)
            lr_pred = col_means[j] + Vt_r[:, j] @ alpha

            pred = KNN_WEIGHT * knn_pred + (1 - KNN_WEIGHT) * lr_pred
            errors.append(pred - true_val)

        sigma = np.std(errors) if errors else 10.0
        sigmas[col] = round(float(sigma), 4)
        if verbose:
            print(f"  {col}: σ = {sigma:.4f}  (n={len(errors)})")

    return sigmas


def backtest(raw_df, cols, completed, cache, n_rows=500):
    """Run backtest on training rows, return summary dict."""
    rng = np.random.default_rng(42)
    test_rows = sorted(rng.choice(3650, n_rows, replace=False))

    total_score = 0.0
    total_fills = 0
    total_orders = 0
    total_qty_filled = 0
    col_selected = {}
    t0 = time.time()

    for i, t in enumerate(test_rows):
        row = raw_df.iloc[t]
        row_vals = np.array([row[c] for c in cols], dtype=float)
        nan_idxs = np.where(np.isnan(row_vals))[0]

        result = trading_problem_5(row, _cache=cache)

        true_prices = completed[t]
        true_median = float(np.median(true_prices))

        for _, order in result.iterrows():
            col_name = order["order_col"]
            bid = order["px"]
            qty = int(order["qty"])
            j = cols.index(col_name)

            total_orders += 1
            col_selected[col_name] = col_selected.get(col_name, 0) + 1

            if bid >= true_prices[j]:
                total_score += qty * (true_median - bid)
                total_fills += 1
                total_qty_filled += qty

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1:3d}/{n_rows}] "
                  f"score={total_score:>10.0f}  "
                  f"fills={total_fills}/{total_orders}  "
                  f"time={elapsed:.1f}s")

    elapsed = time.time() - t0
    fill_rate = total_fills / max(total_orders, 1) * 100

    return {
        "total_score": total_score,
        "avg_profit": total_score / n_rows,
        "fill_rate": fill_rate,
        "total_fills": total_fills,
        "total_orders": total_orders,
        "total_qty_filled": total_qty_filled,
        "elapsed": elapsed,
        "col_selected": col_selected,
    }


def main():
    parser = argparse.ArgumentParser(description="Problem 5 pipeline")
    parser.add_argument("--intermediates", action="store_true",
                        help="Save intermediates/sigma.json to disk")
    parser.add_argument("--backtest-rows", type=int, default=500,
                        help="Number of rows to backtest (default 500)")
    args = parser.parse_args()

    t_start = time.time()

    if not os.path.exists(COMPLETED_PATH):
        print(f"ERROR: {COMPLETED_PATH} not found. "
              f"Run problem-2/main.py first.")
        sys.exit(1)

    print(f"{'='*70}")
    print(f"  Problem 5 Pipeline")
    print(f"{'='*70}\n")

    # ── Load data ─────────────────────────────────────────────────────────
    raw_df = pd.read_csv(DATA_PATH)
    comp_df = pd.read_csv(COMPLETED_PATH)
    cols = [c for c in raw_df.columns if c != "time"]
    arr = raw_df[cols].values.astype(float)
    completed = comp_df[cols].values.astype(float)
    vmask = ~np.isnan(arr)

    print(f"  Data: {arr.shape[0]}x{arr.shape[1]}, "
          f"NaN rate: {(~vmask).mean():.3f}")

    # ── Step 1: compute per-column σ ──────────────────────────────────────
    print(f"\n  Computing per-column σ (LOO CV, {N_SAMPLES} samples/col)...")
    sigmas = compute_sigma(arr, completed, vmask, cols)
    avg_sigma = np.mean(list(sigmas.values()))
    print(f"\n  Avg σ: {avg_sigma:.4f}")

    if args.intermediates:
        os.makedirs("intermediates", exist_ok=True)
        with open("intermediates/sigma.json", "w") as f:
            json.dump(sigmas, f, indent=2)
        print(f"  Saved intermediates/sigma.json")

    # ── Step 2: build cache ───────────────────────────────────────────────
    print(f"\n  Building classification matrix...")
    decompositions = _load_decompositions(cols)
    decomp_order = _topo_order(decompositions)
    classes_full, _ = _build_classes(arr, vmask, decompositions, cols)

    n_given = (classes_full == 0).sum()
    n_det = (classes_full == 1).sum()
    n_imp = (classes_full == 2).sum()
    print(f"  Classification: {n_given:,} given, {n_det:,} deterministic, "
          f"{n_imp:,} imputed")

    cache = {
        "cols": cols,
        "completed": completed,
        "decompositions": decompositions,
        "decomp_order": decomp_order,
        "sigma": sigmas,
        "classes_full": classes_full,
    }

    # ── Step 3: backtest ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Backtest: {args.backtest_rows} training rows")
    print(f"{'='*70}\n")

    results = backtest(raw_df, cols, completed, cache,
                       n_rows=args.backtest_rows)

    print(f"\n{'='*70}")
    print(f"  Results")
    print(f"{'='*70}")
    print(f"  Total score (profit):  {results['total_score']:>12.0f}")
    print(f"  Avg profit / day:      {results['avg_profit']:>12.1f}")
    print(f"  Fill rate:             {results['total_fills']}/"
          f"{results['total_orders']} "
          f"({results['fill_rate']:.1f}%)")
    print(f"  Total qty filled:      {results['total_qty_filled']}")
    print(f"  Backtest time:         {results['elapsed']:.1f}s")

    print(f"\n  Top 10 selected columns:")
    for col, cnt in sorted(results["col_selected"].items(),
                           key=lambda x: -x[1])[:10]:
        s = sigmas.get(col, DEFAULT_SIGMA)
        print(f"    {col}: {cnt} times (σ={s:.2f})")

    print(f"\n  Total pipeline time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
