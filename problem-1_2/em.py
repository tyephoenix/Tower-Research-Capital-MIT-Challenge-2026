"""
EM pipeline — loads coefficients from intermediates, runs EM loop
(SVD ↔ re-regression) to refine uncertain indices, outputs to answers/.

Run coefficients.py first to generate intermediates/coefficients.json.
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time, argparse, json

from coefficients import (load_coefficients, fill_from_known,
                          reregress_on_completed, save_coefficients,
                          distribute_through)
from matrix import iterative_svd_complete


EM_COEF_TOL = 1e-4


def verify_rmse(idx_col, cols, decompositions, data):
    """RMSE on co-observed original rows."""
    dec = decompositions[idx_col]
    farmer_names = [cols[fi] for fi in dec["farmer_idxs"]]
    valid = data[[idx_col] + farmer_names].dropna()
    if len(valid) < 3:
        return float("inf")
    pred = valid[farmer_names].values @ dec["coefs"]
    return float(np.sqrt(np.mean((pred - valid[idx_col].values) ** 2)))


def main():
    parser = argparse.ArgumentParser(description="EM refinement pipeline")
    parser.add_argument("--em-iters", type=int, default=10,
                        help="Max EM iterations (default 10)")
    args = parser.parse_args()

    coef_path = "intermediates/coefficients.json"
    if not os.path.exists(coef_path):
        print(f"ERROR: {coef_path} not found. Run coefficients.py first.")
        sys.exit(1)

    os.makedirs("analysis", exist_ok=True)
    t_start = time.time()

    # ── Load data ─────────────────────────────────────────────────────────
    df = pd.read_csv("../data/limestone_data_challenge_2026.data.csv")
    cols = [c for c in df.columns if c.startswith("col_")]
    data = df[cols].copy()
    T, D = data.shape
    print(f"Data: {T}x{D}, NaN rate: {data.isna().mean().mean():.3f}")

    arr = data.values.astype(np.float64)
    vmask = ~np.isnan(arr)

    # ── Load coefficients from intermediate ───────────────────────────────
    coef_data = load_coefficients(coef_path)
    decompositions = coef_data["decompositions"]
    index_cols = coef_data["index_cols"]
    farmer_idxs = coef_data["farmer_idxs"]
    print(f"Loaded coefficients for: {index_cols}")

    # Rebuild filled matrix from the loaded decompositions
    filled, fmask = fill_from_known(arr, vmask, decompositions, cols,
                                     verbose=True)

    # Classify proven vs uncertain
    proven = []
    uncertain = []
    for c in index_cols:
        v = verify_rmse(c, cols, decompositions, data)
        tag = "proven" if v < 0.01 else "uncertain"
        if v < 0.01:
            proven.append(c)
        else:
            uncertain.append(c)
        print(f"  {c}: verify RMSE={v:.6f} [{tag}]")

    print(f"\n  Proven (locked): {proven}")
    print(f"  Uncertain (EM needed): {uncertain}")

    # ── SVD completion ────────────────────────────────────────────────────
    best_rank = len(farmer_idxs)
    obs_pct = fmask.sum() / fmask.size

    print(f"\n{'='*70}")
    print(f"  SVD + EM Refinement (rank={best_rank}, "
          f"warm start {obs_pct:.1%} observed)")
    print(f"{'='*70}\n")

    completed, obs_rmse = iterative_svd_complete(
        filled, fmask, best_rank, max_iter=500, rel_tol=1e-6)
    print(f"  Initial SVD obs RMSE: {obs_rmse:.6f}")

    # ── EM loop ───────────────────────────────────────────────────────────
    em_history = {c: [verify_rmse(c, cols, decompositions, data)]
                  for c in index_cols}
    em_svd_rmse = []
    em_deltas = []

    if uncertain:
        best_verify = {c: verify_rmse(c, cols, decompositions, data)
                       for c in uncertain}
        best_decomp = {c: dict(decompositions[c]) for c in uncertain}

        print(f"\n  EM LOOP: Re-regressing {uncertain}")
        for c in uncertain:
            print(f"    {c}: initial verify={best_verify[c]:.6f}")

        for em_it in range(1, args.em_iters + 1):
            print(f"\n  -- EM iteration {em_it} --")

            prev_coefs = {c: np.array(decompositions[c]["coefs"], dtype=float)
                          for c in uncertain}

            for c in uncertain:
                idx_i = cols.index(c)
                re_fi, re_coefs, re_rmse = reregress_on_completed(
                    idx_i, farmer_idxs, completed, cols, top_k=30)
                if re_coefs is not None:
                    candidate = {
                        "method": "em",
                        "farmer_idxs": re_fi,
                        "coefs": re_coefs,
                    }
                    old_dec = decompositions[c]
                    decompositions[c] = candidate
                    v = verify_rmse(c, cols, decompositions, data)

                    if v < best_verify[c]:
                        best_verify[c] = v
                        best_decomp[c] = dict(candidate)
                        print(f"    {c}: RMSE={re_rmse:.6f}, k={len(re_fi)}, "
                              f"verify={v:.6f} (improved)")
                        for fi, w in sorted(zip(re_fi, re_coefs),
                                            key=lambda x: -x[1]):
                            if w > 0.01:
                                print(f"        {cols[fi]}: {w:.6f}")
                    else:
                        decompositions[c] = old_dec
                        print(f"    {c}: verify={v:.6f} > "
                              f"best {best_verify[c]:.6f}, reverted")

            for c in index_cols:
                em_history[c].append(
                    verify_rmse(c, cols, decompositions, data))

            working = completed.copy()
            for c in index_cols:
                idx_i = cols.index(c)
                dec = decompositions[c]
                working[:, idx_i] = (completed[:, dec["farmer_idxs"]]
                                     @ dec["coefs"])
            working[vmask] = arr[vmask]

            completed, obs_rmse = iterative_svd_complete(
                arr, vmask, best_rank, max_iter=300, rel_tol=1e-6,
                warm_start=working)
            em_svd_rmse.append(obs_rmse)
            print(f"    SVD obs RMSE: {obs_rmse:.6f}")

            max_delta = 0.0
            for c in uncertain:
                new_c = np.array(decompositions[c]["coefs"], dtype=float)
                old_c = prev_coefs[c]
                if len(new_c) == len(old_c):
                    delta = np.max(np.abs(new_c - old_c))
                else:
                    delta = 1.0
                max_delta = max(max_delta, delta)
            em_deltas.append(max_delta)

            print(f"    Max coef delta: {max_delta:.8f}")
            if max_delta < EM_COEF_TOL:
                print(f"\n  EM converged after {em_it} iterations.")
                break
        else:
            print(f"\n  EM reached max iterations ({args.em_iters}).")

        for c in uncertain:
            decompositions[c] = best_decomp[c]
        print(f"\n  EM done — restored best coefficients per column:")
        for c in uncertain:
            print(f"    {c}: best verify={best_verify[c]:.6f}")

        # ── Convergence plot ──────────────────────────────────────────────
        n_em = len(em_deltas)
        if n_em > 0:
            fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            iters = list(range(n_em))

            for c in uncertain:
                axes[0].plot(iters, em_history[c][1:], "o-", label=c,
                             linewidth=2, markersize=6)
            axes[0].set_ylabel("Verify RMSE")
            axes[0].set_title("EM Convergence")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(iters, em_svd_rmse, "s-", color="#2196F3",
                         linewidth=2, markersize=6)
            axes[1].set_ylabel("SVD Obs RMSE")
            axes[1].grid(True, alpha=0.3)

            axes[2].plot(iters, em_deltas, "^-", color="#FF5722",
                         linewidth=2, markersize=6)
            axes[2].axhline(y=EM_COEF_TOL, color="red", linestyle="--",
                             alpha=0.7, label=f"tol={EM_COEF_TOL}")
            axes[2].set_ylabel("Max Coef Delta")
            axes[2].set_xlabel("EM Iteration")
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig("analysis/em_convergence.png", dpi=150)
            plt.close()
            print(f"  Saved analysis/em_convergence.png")
    else:
        print(f"\n  All indices proven — skipping EM loop.")

    # ── Save EM coefficients ──────────────────────────────────────────────
    accepted_cols = list(decompositions.keys())
    accepted_idxs = [cols.index(c) for c in accepted_cols]
    farmer_idx_set = set(range(D)) - set(accepted_idxs)
    farmer_idxs_final = sorted(farmer_idx_set)
    farmer_cols_final = [cols[i] for i in farmer_idxs_final]

    save_coefficients(
        decompositions,
        accepted_cols, accepted_idxs,
        farmer_cols_final, farmer_idxs_final,
        cols,
        path="intermediates/em_coefficients.json",
    )

    print(f"\nTotal time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
