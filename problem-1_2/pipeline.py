"""
Full pipeline — runs everything from scratch.
candidates → coefficients → SVD → EM → answers/

With --intermediates: also saves intermediates/ and analysis/ plots.
Without: just answers, no plots, no extra IO.
"""

import os
import numpy as np
import pandas as pd
import time, argparse

from candidates import (process_candidates, hardcoded_candidates,
                        DEFAULT_MIN_ROWS, DEFAULT_N_SPLITS, DEFAULT_RMSE_THRESHOLD)
from coefficients import (recover_coefficients, refit_weights,
                          distribute_through)
from matrix import iterative_svd_complete
from trend import fit_all_columns, build_warm_start


ANSWERS_DIR = "../answers"
EM_COEF_TOL = 1e-4


def verify_rmse(idx_col, cols, decompositions, data):
    dec = decompositions[idx_col]
    farmer_names = [cols[fi] for fi in dec["farmer_idxs"]]
    valid = data[[idx_col] + farmer_names].dropna()
    if len(valid) < 3:
        return float("inf")
    pred = valid[farmer_names].values @ dec["coefs"]
    return float(np.sqrt(np.mean((pred - valid[idx_col].values) ** 2)))


def main():
    parser = argparse.ArgumentParser(description="Limestone — full pipeline")
    parser.add_argument("--min-rows", type=int, default=DEFAULT_MIN_ROWS)
    parser.add_argument("--threshold", type=float, default=DEFAULT_RMSE_THRESHOLD)
    parser.add_argument("--em-iters", type=int, default=10)
    parser.add_argument("--intermediates", action="store_true",
                        help="Save intermediates + analysis plots")
    parser.add_argument("--candidates", type=int, nargs="+", default=None,
                        help="Hardcode candidates by col number")
    args = parser.parse_args()

    os.makedirs(ANSWERS_DIR, exist_ok=True)
    if args.intermediates:
        os.makedirs("intermediates", exist_ok=True)
        os.makedirs("analysis", exist_ok=True)

    t_start = time.time()
    np.random.seed(42)

    df = pd.read_csv("../data/limestone_data_challenge_2026.data.csv")
    cols = [c for c in df.columns if c.startswith("col_")]
    data = df[cols].copy()
    T, D = data.shape
    print(f"Data: {T}x{D}, NaN rate: {data.isna().mean().mean():.3f}")

    arr = data.values.astype(np.float64)
    vmask = ~np.isnan(arr)

    # ── Phase 1: candidates ───────────────────────────────────────────────
    if args.candidates:
        result = hardcoded_candidates(cols, args.candidates)
    else:
        result = process_candidates(
            arr, vmask, cols,
            min_rows=args.min_rows, n_splits=DEFAULT_N_SPLITS,
            rmse_threshold=args.threshold,
        )

    if args.intermediates:
        from candidates import (save_candidates, plot_residual_ranking,
                                plot_residual_gaps, print_data_availability)
        save_candidates(result)
        if result["residual_data"]:
            plot_residual_ranking(result["residual_data"],
                                 result["rmse_threshold"])
            plot_residual_gaps(result["residual_data"],
                               result["rmse_threshold"])
            print_data_availability(arr, vmask, cols, result["residual_data"])

    # ── Phase 2: coefficient recovery ─────────────────────────────────────
    decompositions, filled, fmask = recover_coefficients(
        data, arr, vmask, cols,
        result["index_cols"], result["index_idxs"],
        result["farmer_cols"], result["farmer_idxs"],
    )

    # Update index/farmer lists based on what was actually accepted
    accepted_cols = list(decompositions.keys())
    accepted_idxs = [cols.index(c) for c in accepted_cols]
    reclassified = [c for c in result["index_cols"] if c not in decompositions]

    farmer_idx_set = set(range(D)) - set(accepted_idxs)
    farmer_idxs = sorted(farmer_idx_set)
    farmer_cols = [cols[i] for i in farmer_idxs]
    n_farmers = len(farmer_idxs)

    print(f"\n  Final classification: {len(accepted_cols)} indices, {n_farmers} farmers")
    if reclassified:
        print(f"  Reclassified as farmers: {reclassified}")

    # Classify as proven vs uncertain
    proven = []
    uncertain = []
    for c in accepted_cols:
        if decompositions[c].get("proven", False):
            proven.append(c)
        else:
            uncertain.append(c)
    print(f"  Proven: {proven}")
    print(f"  Uncertain (EM needed): {uncertain}")

    if args.intermediates:
        from coefficients import save_coefficients
        save_coefficients(decompositions, accepted_cols, accepted_idxs,
                          farmer_cols, farmer_idxs, cols)

    # ── Phase 2.5: fit trend model per column for warm start ────────────
    t_all = df["time"].values
    print("\n  Fitting triangle-wave trend model per column...")
    trend_fits = fit_all_columns(arr, vmask, t_all, cols, verbose=True)

    trend_warm = build_warm_start(arr, vmask, t_all, cols, trend_fits)

    # Merge: keep coefficient-filled values where available, else use trend
    warm = filled.copy()
    still_nan = np.isnan(warm)
    warm[still_nan] = trend_warm[still_nan]
    warm_mask = fmask | still_nan  # everything is now filled

    # ── Phase 3: SVD completion ───────────────────────────────────────────
    best_rank = n_farmers
    obs_filled = fmask.sum() / fmask.size
    print(f"\n  SVD completion at rank={best_rank} ({n_farmers} farmers), "
          f"warm start from coeff fill ({obs_filled:.1%} observed) "
          f"+ trend model...")

    completed, obs_rmse = iterative_svd_complete(
        arr, vmask, best_rank, warm_start=warm)
    print(f"  SVD obs RMSE={obs_rmse:.6f}")

    if args.intermediates:
        from matrix import save_matrix_result
        save_matrix_result(decompositions, best_rank, obs_rmse, [], cols)

    # ── Phase 4: EM loop ──────────────────────────────────────────────────
    em_history = {c: [verify_rmse(c, cols, decompositions, data)]
                  for c in accepted_cols} if args.intermediates else None
    em_svd_rmse = [] if args.intermediates else None
    em_deltas = [] if args.intermediates else None

    if uncertain:
        fixed_farmers = {c: list(decompositions[c]["farmer_idxs"])
                         for c in uncertain}
        best_verify = {c: verify_rmse(c, cols, decompositions, data)
                       for c in uncertain}
        best_decomp = {c: dict(decompositions[c]) for c in uncertain}

        print(f"\n  EM loop ({args.em_iters} max iters)...")
        for c in uncertain:
            print(f"    {c}: farmers = {[cols[fi] for fi in fixed_farmers[c]]}, "
                  f"initial verify={best_verify[c]:.6f}")

        for em_it in range(1, args.em_iters + 1):
            prev_coefs = {c: np.array(decompositions[c]["coefs"], dtype=float)
                          for c in uncertain}

            for c in uncertain:
                idx_i = cols.index(c)
                re_fi, re_coefs, re_rmse = refit_weights(
                    idx_i, fixed_farmers[c], completed)
                candidate = {
                    "method": "em",
                    "farmer_idxs": re_fi,
                    "coefs": re_coefs,
                    "proven": False,
                }
                old_dec = decompositions[c]
                decompositions[c] = candidate
                v = verify_rmse(c, cols, decompositions, data)

                if v < best_verify[c]:
                    best_verify[c] = v
                    best_decomp[c] = dict(candidate)
                    print(f"    EM {em_it} | {c}: verify={v:.6f} (improved), "
                          f"rmse_completed={re_rmse:.6f}")
                else:
                    decompositions[c] = old_dec
                    print(f"    EM {em_it} | {c}: verify={v:.6f} > "
                          f"best {best_verify[c]:.6f}, reverted")

            if args.intermediates:
                for c in accepted_cols:
                    em_history[c].append(
                        verify_rmse(c, cols, decompositions, data))

            # Reconstruct index columns, then re-SVD
            working = completed.copy()
            for c in accepted_cols:
                idx_i = cols.index(c)
                dec = decompositions[c]
                working[:, idx_i] = (completed[:, dec["farmer_idxs"]]
                                     @ dec["coefs"])
            working[vmask] = arr[vmask]

            completed, obs_rmse = iterative_svd_complete(
                arr, vmask, best_rank, max_iter=300, rel_tol=1e-6,
                warm_start=working)

            max_delta = 0.0
            for c in uncertain:
                new_c = np.array(decompositions[c]["coefs"], dtype=float)
                old_c = prev_coefs[c]
                if len(new_c) == len(old_c):
                    max_delta = max(max_delta, np.max(np.abs(new_c - old_c)))
                else:
                    max_delta = max(max_delta, 1.0)

            if args.intermediates:
                em_svd_rmse.append(obs_rmse)
                em_deltas.append(max_delta)

            print(f"    EM {em_it} | SVD RMSE={obs_rmse:.6f}, "
                  f"coef delta={max_delta:.8f}")
            if max_delta < EM_COEF_TOL:
                print(f"  EM converged at iteration {em_it}.")
                break

        for c in uncertain:
            decompositions[c] = best_decomp[c]
        print(f"\n  EM done — restored best coefficients per column:")
        for c in uncertain:
            print(f"    {c}: best verify={best_verify[c]:.6f}")

    if args.intermediates and em_deltas:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_em = len(em_deltas)
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

    if args.intermediates:
        from coefficients import save_coefficients as _sc
        _sc(decompositions, accepted_cols, accepted_idxs,
            farmer_cols, farmer_idxs, cols,
            path="intermediates/em_coefficients.json")

    # ── Final reconstruction ──────────────────────────────────────────────
    for c in accepted_cols:
        idx_i = cols.index(c)
        dec = decompositions[c]
        completed[:, idx_i] = (completed[:, dec["farmer_idxs"]]
                               @ dec["coefs"])
    completed[vmask] = arr[vmask]

    # ── Distribute through for problem 1b ─────────────────────────────────
    expanded = distribute_through(decompositions, cols)

    # ── Save answers ──────────────────────────────────────────────────────
    p1a = os.path.join(ANSWERS_DIR, "problem1a_answer.csv")
    p1b = os.path.join(ANSWERS_DIR, "problem1b_answer.csv")
    p2 = os.path.join(ANSWERS_DIR, "problem2_answer.csv")

    # Problem 1a: which columns are indices
    pd.DataFrame({
        "column": farmer_cols + accepted_cols,
        "is_index": [False] * n_farmers + [True] * len(accepted_cols),
    }).to_csv(p1a, index=False)

    # Problem 1b: coefficient matrix (distributed to pure farmers)
    rows_1b = []
    for idx_col in accepted_cols:
        if idx_col in expanded:
            exp = expanded[idx_col]
            for fi, w in zip(exp["farmer_idxs"], exp["coefs"]):
                if w > 1e-4:
                    rows_1b.append({
                        "index_col": idx_col,
                        "constituent_col": cols[fi],
                        "coef": round(float(w), 6),
                    })
        elif idx_col in decompositions:
            dec = decompositions[idx_col]
            for fi, w in zip(dec["farmer_idxs"], dec["coefs"]):
                if w > 1e-4:
                    rows_1b.append({
                        "index_col": idx_col,
                        "constituent_col": cols[fi],
                        "coef": round(float(w), 6),
                    })
    pd.DataFrame(rows_1b).to_csv(p1b, index=False)

    # Problem 2: completed matrix
    out = df.copy()
    for j, c in enumerate(cols):
        out[c] = completed[:, j]
    out.to_csv(p2, index=False)

    print(f"\n  Saved {p1a}, {p1b}, {p2}")
    remaining_nan = np.isnan(completed).sum()
    print(f"  Completed matrix: {remaining_nan} NaN remaining")
    print(f"  Total time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
