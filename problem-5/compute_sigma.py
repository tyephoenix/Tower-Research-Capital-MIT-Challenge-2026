"""
Compute per-column σ via leave-one-out cross-validation.

For each column j, for ~500 rows where column j is observed:
  - Hide column j
  - Predict it using KNN + low-rank projection from remaining observed columns
  - Record the error

σ_j = std(errors)

Saves result to intermediates/sigma.json.
"""

import os, json
import numpy as np
import pandas as pd

DATA_PATH = "../data/limestone_data_challenge_2026.data.csv"
COMPLETED_PATH = "../answers/problem2_answer.csv"

KNN_K = 20
PROJECTION_RANK = 12
KNN_WEIGHT = 0.5
N_SAMPLES = 500


def main():
    raw = pd.read_csv(DATA_PATH)
    comp = pd.read_csv(COMPLETED_PATH)
    cols = [c for c in raw.columns if c != "time"]
    arr = raw[cols].values.astype(float)
    completed = comp[cols].values.astype(float)
    vmask = ~np.isnan(arr)

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

            dists = np.sum((completed[:, obs_idxs] - row_vals[obs_idxs]) ** 2, axis=1)
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
        print(f"  {col}: σ = {sigma:.4f}  (n={len(errors)})")

    os.makedirs("intermediates", exist_ok=True)
    with open("intermediates/sigma.json", "w") as f:
        json.dump(sigmas, f, indent=2)
    print(f"\nSaved intermediates/sigma.json")

    avg = np.mean(list(sigmas.values()))
    print(f"Avg σ: {avg:.4f}")


if __name__ == "__main__":
    main()
