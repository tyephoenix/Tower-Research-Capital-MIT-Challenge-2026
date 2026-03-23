"""
Phase 3 — Matrix Completion & Rebuild.
SVD-based iterative completion, then optional re-regression of uncertain
index coefficients on the completed data.
"""

import numpy as np
import pandas as pd


def iterative_svd_complete(data_np, mask_np, rank,
                           max_iter=500, rel_tol=1e-4, warm_start=None):
    """
    Iterative SVD using torch (GPU if available).

    Parameters
    ----------
    warm_start : np.ndarray or None
        If provided, use this as the initial fill for missing values
        instead of column means. Must be same shape as data_np with no NaN.

    Returns (completed, obs_rmse).
    """
    import torch

    if torch.cuda.is_available():
        dev = torch.device("cuda")
        dtype = torch.float64
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        dtype = torch.float32
    else:
        dev = torch.device("cpu")
        dtype = torch.float64

    if warm_start is not None:
        X_np = warm_start.copy()
    else:
        col_means = np.nanmean(data_np, axis=0)
        X_np = data_np.copy()
        X_np[~mask_np] = np.take(col_means, np.where(~mask_np)[1])

    X = torch.tensor(X_np, dtype=dtype, device=dev)
    mask = torch.tensor(mask_np, dtype=torch.bool, device=dev)
    data_obs = torch.tensor(np.nan_to_num(data_np, nan=0.0),
                            dtype=dtype, device=dev)

    prev_rmse = float("inf")
    for it in range(max_iter):
        U, S, Vt = torch.linalg.svd(X, full_matrices=False)
        recon = U[:, :rank] @ torch.diag(S[:rank]) @ Vt[:rank, :]
        X = torch.where(mask, data_obs, recon)

        diff = (data_obs - recon) * mask
        rmse = torch.sqrt((diff ** 2).sum() / mask.sum()).item()
        if abs(prev_rmse - rmse) / (rmse + 1e-15) < rel_tol:
            break
        prev_rmse = rmse

    return X.cpu().numpy(), rmse


def rank_sweep(arr, vmask, ranks=None, max_iter=300, rel_tol=1e-5,
               verbose=True):
    """
    Sweep over ranks, return list of (rank, obs_rmse).
    """
    if ranks is None:
        ranks = list(range(2, 21)) + list(range(25, 46, 5))

    if verbose:
        print(f"  Rank sweep (full {arr.shape[1]}-col matrix)...")
        print(f"  {'Rank':>5} {'ObsRMSE':>10}")

    results = []
    for rank in ranks:
        _, obs_rmse = iterative_svd_complete(
            arr, vmask, rank, max_iter=max_iter, rel_tol=rel_tol)
        results.append((rank, obs_rmse))
        if verbose:
            print(f"  {rank:5d} {obs_rmse:10.6f}")

    return results


def pick_best_rank(rank_results, rmse_threshold=0.1):
    """Pick first rank below threshold, else lowest RMSE."""
    good = [(r, e) for r, e in rank_results if e < rmse_threshold]
    if good:
        return good[0][0], good[0][1]
    best = min(rank_results, key=lambda x: x[1])
    return best[0], best[1]


def svd_best_rank(arr, vmask, ranks=None, verbose=True):
    """Rank sweep → pick best → complete at that rank. Returns (rank, completed, obs_rmse)."""
    rank_results = rank_sweep(arr, vmask, ranks=ranks, verbose=verbose)
    best_rank, _ = pick_best_rank(rank_results)
    if verbose:
        print(f"  Selected rank={best_rank}")
    completed, obs_rmse = iterative_svd_complete(
        arr, vmask, best_rank, max_iter=500, rel_tol=1e-6)
    return best_rank, completed, obs_rmse


def complete_matrix(arr, vmask, cols, decompositions,
                    filled=None, fmask=None, rank=None, verbose=True):
    """
    SVD completion + reconstruct index columns from decompositions.
    If filled/fmask provided, uses them as warm start.
    Returns (completed, obs_rmse).
    """
    if rank is None:
        n_indices = len(decompositions)
        rank = len(cols) - n_indices

    if verbose:
        print(f"\n  Matrix completion: rank={rank}, "
              f"{len(decompositions)} index columns")

    if filled is None or fmask is None:
        from coefficients import fill_from_known
        filled, fmask = fill_from_known(arr, vmask, decompositions, cols,
                                         verbose=verbose)

    completed, obs_rmse = iterative_svd_complete(filled, fmask, rank)
    if verbose:
        print(f"  SVD obs RMSE: {obs_rmse:.6f}")

    for idx_col, dec in decompositions.items():
        idx_i = cols.index(idx_col)
        completed[:, idx_i] = (completed[:, dec["farmer_idxs"]]
                               @ dec["coefs"])

    completed[vmask] = arr[vmask]
    return completed, obs_rmse


def save_matrix_result(decompositions, best_rank, final_rmse, rank_results,
                       cols, path="intermediates/matrix.json"):
    """Serialize matrix completion metadata to JSON."""
    import json, os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = {
        "best_rank": int(best_rank),
        "final_rmse": float(final_rmse),
        "rank_results": [[int(r), float(e)] for r, e in rank_results],
        "decompositions": {},
    }
    for idx_col, dec in decompositions.items():
        out["decompositions"][idx_col] = {
            "method": dec["method"],
            "farmer_idxs": [int(x) for x in dec["farmer_idxs"]],
            "coefs": [float(x) for x in dec["coefs"]],
        }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved {path}")


# ── Standalone entry point ────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from coefficients import load_coefficients

    coef_path = "intermediates/coefficients.json"
    try:
        coef_data = load_coefficients(coef_path)
    except FileNotFoundError:
        print(f"ERROR: {coef_path} not found. Run coefficients.py first.")
        sys.exit(1)

    df = pd.read_csv("../data/limestone_data_challenge_2026.data.csv")
    cols = [c for c in df.columns if c.startswith("col_")]
    data = df[cols].copy()
    arr = data.values.astype(np.float64)
    vmask = ~np.isnan(arr)

    print(f"Data: {data.shape[0]}×{data.shape[1]}, "
          f"NaN rate: {data.isna().mean().mean():.3f}")
    print(f"Loaded coefficients for: {coef_data['index_cols']}")

    decompositions = coef_data["decompositions"]

    completed, obs_rmse = complete_matrix(arr, vmask, cols, decompositions)

    out = df.copy()
    for j, c in enumerate(cols):
        out[c] = completed[:, j]
    out.to_csv("../answers/problem2_answer.csv", index=False)
    print(f"  Saved ../answers/problem2_answer.csv")

    save_matrix_result(decompositions,
                       len(cols) - len(decompositions), obs_rmse,
                       [], cols)
