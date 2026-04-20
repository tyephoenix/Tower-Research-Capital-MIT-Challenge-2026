"""
Phase 1 — Row-Residual Index Detection.

For each column, measure how well it can be predicted as a convex combination
of its top-K correlated columns on co-observed rows (train/test split).
True indices have near-zero test RMSE; farmers have positive RMSE.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tye.coefficients import fit_convex


DEFAULT_MIN_ROWS = 30
DEFAULT_N_SPLITS = 15
DEFAULT_RMSE_THRESHOLD = 3.5


def pairwise_abs_corr(arr, vmask, min_obs=50):
    """Pairwise absolute Pearson correlation on co-observed rows."""
    D = arr.shape[1]
    corr = np.full((D, D), np.nan)
    for i in range(D):
        for j in range(i + 1, D):
            both = vmask[:, i] & vmask[:, j]
            if both.sum() < min_obs:
                continue
            r = np.corrcoef(arr[both, i], arr[both, j])[0, 1]
            corr[i, j] = abs(r)
            corr[j, i] = abs(r)
    return corr


def process_candidates(arr, vmask, cols, min_rows=DEFAULT_MIN_ROWS,
                       n_splits=DEFAULT_N_SPLITS,
                       rmse_threshold=DEFAULT_RMSE_THRESHOLD, verbose=True):
    """
    Detect index columns via row-residual sparse convex test.

    For each column, greedily add the next most-correlated predictor as long
    as co-observed rows stay >= min_rows. This gives each column the maximum
    K the data supports — no arbitrary cap.

    Returns dict with index_cols, farmer_cols, residual_data, etc.
    """
    D = len(cols)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  PHASE 1: Row-Residual Index Detection (adaptive K, min_rows={min_rows})")
        print(f"{'='*70}\n")
        print(f"  Computing pairwise correlations...")

    corr = pairwise_abs_corr(arr, vmask)

    if verbose:
        print(f"  Running convex residual test ({n_splits} splits per column)...")

    residual_data = []
    for ci in range(D):
        other_corrs = [(j, corr[ci, j]) for j in range(D)
                       if j != ci and not np.isnan(corr[ci, j])]
        other_corrs.sort(key=lambda x: -x[1])

        # Greedily add predictors while co-observed rows >= min_rows
        selected = []
        co_obs = vmask[:, ci].copy()
        for j, _ in other_corrs:
            candidate_obs = co_obs & vmask[:, j]
            if candidate_obs.sum() < min_rows:
                break
            selected.append(j)
            co_obs = candidate_obs

        rows = np.where(co_obs)[0]
        K_used = len(selected)

        if K_used < 2:
            residual_data.append({
                "col": cols[ci], "col_idx": ci,
                "rmse": float("inf"), "std": 0.0,
                "co_obs": int(len(rows)), "K": K_used,
            })
            continue

        y_all = arr[rows, ci]
        X_all = arr[np.ix_(rows, selected)]

        rmses = []
        for _ in range(n_splits):
            idx = np.random.permutation(len(rows))
            half = len(idx) // 2
            tr, te = idx[:half], idx[half:]
            if len(tr) < 10 or len(te) < 10:
                continue
            coefs, _ = fit_convex(y_all[tr], X_all[tr])
            pred = X_all[te] @ coefs
            rmse = np.sqrt(np.mean((y_all[te] - pred) ** 2))
            rmses.append(rmse)

        mean_rmse = float(np.mean(rmses)) if rmses else float("inf")
        std_rmse = float(np.std(rmses)) if rmses else 0.0
        residual_data.append({
            "col": cols[ci], "col_idx": ci,
            "rmse": mean_rmse, "std": std_rmse,
            "co_obs": int(len(rows)), "K": K_used,
        })

    residual_data.sort(key=lambda x: x["rmse"])

    index_cols = []
    index_idxs = []
    for rd in residual_data:
        if rd["rmse"] < rmse_threshold:
            index_cols.append(rd["col"])
            index_idxs.append(rd["col_idx"])

    farmer_idxs = sorted(set(range(D)) - set(index_idxs))
    farmer_cols = [cols[i] for i in farmer_idxs]

    if verbose:
        print(f"\n  {'Rank':>4}  {'Column':<8}  {'RMSE':>10}  {'Std':>8}  {'CoObs':>6}  {'K':>3}")
        print(f"  {'-'*51}")
        for rank, rd in enumerate(residual_data, 1):
            tag = " *" if rd["rmse"] < rmse_threshold else ""
            print(f"  {rank:4d}  {rd['col']:<8}  {rd['rmse']:10.4f}"
                  f"  {rd['std']:8.4f}  {rd['co_obs']:6d}  {rd.get('K',0):3d}{tag}")
        print(f"\n  ┌─────────────────────────────────────────┐")
        print(f"  │  Detected {len(index_cols)} indices (threshold={rmse_threshold})")
        print(f"  │  {index_cols}")
        print(f"  └─────────────────────────────────────────┘")

    return {
        "index_cols": index_cols,
        "index_idxs": index_idxs,
        "farmer_cols": farmer_cols,
        "farmer_idxs": farmer_idxs,
        "residual_data": residual_data,
        "rmse_threshold": rmse_threshold,
    }


def hardcoded_candidates(cols, candidate_nums, verbose=True):
    """Skip detection — use hardcoded column numbers as indices."""
    D = len(cols)
    col_num_map = {int(c.split("_")[1]): i for i, c in enumerate(cols)}

    index_idxs = []
    index_cols = []
    for n in candidate_nums:
        if n not in col_num_map:
            raise ValueError(f"col_{n:02d} not found in columns")
        idx = col_num_map[n]
        index_idxs.append(idx)
        index_cols.append(cols[idx])

    farmer_idxs = sorted(set(range(D)) - set(index_idxs))
    farmer_cols = [cols[i] for i in farmer_idxs]

    if verbose:
        print(f"\n  Hardcoded candidates: {index_cols}")

    return {
        "index_cols": index_cols,
        "index_idxs": index_idxs,
        "farmer_cols": farmer_cols,
        "farmer_idxs": farmer_idxs,
        "residual_data": [],
        "rmse_threshold": 0,
    }


# ── Data availability diagnostic ─────────────────────────────────────────

def print_data_availability(arr, vmask, cols, residual_data, min_rows=20):
    """
    For each candidate, show how co-observation decays as predictors are added.
    Flags columns where the true decomposition might be unverifiable on
    original data (the "col_42 problem").
    """
    D = len(cols)
    corr = pairwise_abs_corr(arr, vmask)

    print(f"\n{'='*70}")
    print(f"  DATA AVAILABILITY DIAGNOSTIC")
    print(f"  (co-observed rows as top-correlated predictors are added)")
    print(f"{'='*70}\n")

    flagged = []
    for rd in residual_data:
        ci = rd["col_idx"]
        col_name = rd["col"]

        other_corrs = [(j, corr[ci, j]) for j in range(D)
                       if j != ci and not np.isnan(corr[ci, j])]
        other_corrs.sort(key=lambda x: -x[1])

        # Track co-obs as we add predictors
        co_obs = vmask[:, ci].copy()
        base_obs = co_obs.sum()
        decay = [(0, base_obs, "-")]
        for k, (j, r) in enumerate(other_corrs, 1):
            co_obs_new = co_obs & vmask[:, j]
            n = co_obs_new.sum()
            decay.append((k, n, cols[j]))
            if n < min_rows:
                break
            co_obs = co_obs_new

        max_k = decay[-1][0] if decay[-1][1] >= min_rows else decay[-2][0]
        drop_rate = 1.0 - (decay[min(5, len(decay)-1)][1] / base_obs) if base_obs > 0 else 0

        # Check for "blind spots": pairs of top-10 correlated columns that
        # have 0 co-obs with the target
        top10 = [j for j, _ in other_corrs[:10]]
        blind_pairs = []
        for i in range(len(top10)):
            for j in range(i+1, len(top10)):
                tri = vmask[:, ci] & vmask[:, top10[i]] & vmask[:, top10[j]]
                if tri.sum() == 0:
                    blind_pairs.append((cols[top10[i]], cols[top10[j]]))

        is_flagged = len(blind_pairs) > 0 or max_k < 5

        print(f"  {col_name}  (obs={base_obs}, max_K={max_k}, "
              f"RMSE={rd['rmse']:.3f})")
        print(f"    K:    ", end="")
        for k, n, _ in decay[:11]:
            print(f" {k:>4}", end="")
        print()
        print(f"    rows: ", end="")
        for k, n, _ in decay[:11]:
            marker = "!" if n < min_rows else " "
            print(f" {n:>3}{marker}", end="")
        print()

        if blind_pairs:
            print(f"    ⚠ BLIND SPOTS: {len(blind_pairs)} top-10 pairs "
                  f"with 0 triple co-obs:")
            for a, b in blind_pairs[:5]:
                print(f"      {col_name} + {a} + {b}: 0 rows")
            if len(blind_pairs) > 5:
                print(f"      ... and {len(blind_pairs)-5} more")
            flagged.append(col_name)
        elif max_k < 5:
            print(f"    ⚠ LOW MAX K: only {max_k} predictors fit "
                  f"with >={min_rows} co-obs")
            flagged.append(col_name)
        print()

    if flagged:
        print(f"  ┌─────────────────────────────────────────────────────┐")
        print(f"  │  ⚠ {len(flagged)} columns with data availability issues:")
        print(f"  │    {flagged}")
        print(f"  │  These may have unverifiable decompositions on")
        print(f"  │  original data (need filled data to prove).")
        print(f"  └─────────────────────────────────────────────────────┘")
    else:
        print(f"  ✓ No data availability issues detected.")


# ── Plotting ──────────────────────────────────────────────────────────────

def plot_residual_ranking(residual_data, rmse_threshold,
                          save_path="analysis/residuals.png"):
    """Horizontal bar chart: all columns ranked by test RMSE."""
    finite = [d for d in residual_data if np.isfinite(d["rmse"])]
    inf_cols = [d for d in residual_data if not np.isfinite(d["rmse"])]
    if not finite:
        return

    n = len(finite)
    fig, ax = plt.subplots(figsize=(11, max(8, n * 0.28)))

    rmses = [d["rmse"] for d in finite]
    labels = [d["col"] for d in finite]
    y_pos = np.arange(n)

    c_index = "#26A69A"
    c_farmer = "#B0BEC5"
    colors = [c_index if r < rmse_threshold else c_farmer for r in rmses]

    bars = ax.barh(y_pos, rmses, color=colors, edgecolor="white",
                   linewidth=0.5, height=0.75)
    ax.axvline(x=rmse_threshold, color="#E53935", linestyle="--",
               linewidth=2, alpha=0.8, label=f"threshold = {rmse_threshold:.1f}")

    k_vals = [d.get("K", 0) for d in finite]
    for i, (rmse, label, k) in enumerate(zip(rmses, labels, k_vals)):
        ax.text(rmse + 0.15, i, f"{rmse:.2f}  (K={k})",
                va="center", fontsize=7, color="#555")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8, fontfamily="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("Test RMSE (convex fit, train/test split)", fontsize=11)
    ax.set_title("Phase 1: Row-Residual Index Detection\n"
                 "Columns ranked by predictability — lower = more likely index",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
    ax.grid(axis="x", alpha=0.2)
    ax.set_xlim(0, max(rmses) * 1.12)

    n_idx = sum(1 for r in rmses if r < rmse_threshold)
    ax.axhspan(-0.5, n_idx - 0.5, alpha=0.06, color=c_index, zorder=0)

    if inf_cols:
        ax.text(0.98, 0.02,
                f"+{len(inf_cols)} cols with insufficient co-obs (not shown)",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color="#999")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


def plot_residual_gaps(residual_data, rmse_threshold,
                       save_path="analysis/gaps.png"):
    """Two-panel: sorted RMSE values + gap ratios between consecutive."""
    finite = [d for d in residual_data if np.isfinite(d["rmse"])]
    if len(finite) < 3:
        return

    rmses = [d["rmse"] for d in finite]
    labels = [d["col"] for d in finite]
    n = len(rmses)
    x = np.arange(n)

    gaps = [rmses[i + 1] / rmses[i] if rmses[i] > 0 else 1.0
            for i in range(n - 1)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9),
                                    gridspec_kw={"height_ratios": [3, 2]})

    c_index = "#26A69A"
    c_farmer = "#B0BEC5"
    colors = [c_index if r < rmse_threshold else c_farmer for r in rmses]

    ax1.bar(x, rmses, color=colors, edgecolor="white", linewidth=0.5, width=0.8)
    ax1.axhline(y=rmse_threshold, color="#E53935", linestyle="--",
                linewidth=2, alpha=0.8, label=f"threshold = {rmse_threshold:.1f}")
    ax1.set_ylabel("Test RMSE", fontsize=11)
    ax1.set_title("Phase 1: Sorted RMSE with Gap Analysis", fontsize=13,
                  fontweight="bold")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(axis="y", alpha=0.2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=90, fontsize=7, fontfamily="monospace")

    n_idx = sum(1 for r in rmses if r < rmse_threshold)
    if n_idx > 0 and n_idx < n:
        ax1.axvline(x=n_idx - 0.5, color="#E53935", linewidth=1.5, alpha=0.5)

    gap_x = np.arange(len(gaps))
    gap_colors = ["#FF7043" if g > 1.15 else "#90A4AE" for g in gaps]
    ax2.bar(gap_x, gaps, color=gap_colors, edgecolor="white",
            linewidth=0.5, width=0.8)
    ax2.axhline(y=1.0, color="#999", linewidth=0.5)
    ax2.axhline(y=1.15, color="#FF7043", linestyle=":", linewidth=1.5,
                alpha=0.6, label="1.15x")
    ax2.set_ylabel("Gap ratio (next / current)", fontsize=11)
    ax2.set_xlabel("Rank", fontsize=11)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(axis="y", alpha=0.2)

    gap_labels = [f"{labels[i]}" for i in range(len(gaps))]
    ax2.set_xticks(gap_x)
    ax2.set_xticklabels(gap_labels, rotation=90, fontsize=7, fontfamily="monospace")

    for i, g in enumerate(gaps):
        if g > 1.15:
            ax2.text(i, g + 0.01, f"{g:.2f}", ha="center", fontsize=7,
                     color="#D84315", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


# ── Serialization ─────────────────────────────────────────────────────────

def save_candidates(result, path="intermediates/candidates.json"):
    """Serialize candidates result to JSON."""
    import json, os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = {
        "index_cols": result["index_cols"],
        "index_idxs": result["index_idxs"],
        "farmer_cols": result["farmer_cols"],
        "farmer_idxs": result["farmer_idxs"],
        "residual_data": result.get("residual_data", []),
        "rmse_threshold": result.get("rmse_threshold", 0),
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved {path}")


def load_candidates(path="intermediates/candidates.json"):
    """Load candidates result from JSON."""
    import json
    with open(path) as f:
        return json.load(f)


# ── Standalone entry point ────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser(description="Phase 1: Index detection")
    parser.add_argument("--min-rows", type=int, default=DEFAULT_MIN_ROWS,
                        help=f"Min co-observed rows per column (default {DEFAULT_MIN_ROWS})")
    parser.add_argument("--splits", type=int, default=DEFAULT_N_SPLITS,
                        help=f"Train/test splits (default {DEFAULT_N_SPLITS})")
    parser.add_argument("--threshold", type=float, default=DEFAULT_RMSE_THRESHOLD,
                        help=f"RMSE threshold for index detection (default {DEFAULT_RMSE_THRESHOLD})")
    parser.add_argument("--candidates", type=int, nargs="+", default=None,
                        help="Hardcode candidates by col number (e.g. --candidates 11 50)")
    args = parser.parse_args()

    os.makedirs("analysis", exist_ok=True)

    np.random.seed(42)

    df = pd.read_csv("../data/limestone_data_challenge_2026.data.csv")
    cols = [c for c in df.columns if c.startswith("col_")]
    data = df[cols].copy()
    arr = data.values.astype(np.float64)
    vmask = ~np.isnan(arr)
    print(f"Data: {data.shape[0]}x{data.shape[1]}, "
          f"NaN rate: {data.isna().mean().mean():.3f}")

    if args.candidates:
        result = hardcoded_candidates(cols, args.candidates)
    else:
        result = process_candidates(
            arr, vmask, cols,
            min_rows=args.min_rows, n_splits=args.splits,
            rmse_threshold=args.threshold,
        )
        if result["residual_data"]:
            plot_residual_ranking(result["residual_data"],
                                 result["rmse_threshold"],
                                 save_path="analysis/residuals.png")
            plot_residual_gaps(result["residual_data"],
                               result["rmse_threshold"],
                               save_path="analysis/gaps.png")
            print_data_availability(arr, vmask, cols, result["residual_data"])

    save_candidates(result)
