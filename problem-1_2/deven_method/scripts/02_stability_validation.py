from __future__ import annotations

import numpy as np

from common import CANDIDATE_STABILITY_SETS, fit_convex, load_matrix, parse_args, save_json


def stability_test(target_col, predictor_cols, M, mask, ci, n_trials=200, frac=0.7):
    ti = ci[target_col]
    pis = [ci[c] for c in predictor_cols]
    obs = mask[:, ti].copy()
    for pi in pis:
        obs &= mask[:, pi]
    all_rows = np.where(obs)[0]
    n_available = len(all_rows)
    if n_available < 15:
        return None, None, None
    sample_size = max(int(n_available * frac), len(pis) + 3)
    all_weights = []
    all_rmses = []
    for _ in range(n_trials):
        rows = np.random.choice(all_rows, size=min(sample_size, n_available), replace=False)
        y = M[rows, ti]
        X = M[np.ix_(rows, pis)]
        w, rmse, _ = fit_convex(y, X, equality=False, maxiter=500, ftol=1e-12)
        if w is not None:
            all_weights.append(w)
            all_rmses.append(rmse)
    return np.array(all_weights), np.array(all_rmses), all_rows


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    _, _, M, mask, ci = load_matrix(args.csv)
    np.random.seed(42)
    output = {}
    print(f"Matrix shape: {M.shape}, Observed: {mask.mean():.3f}")
    for target, pred_sets in CANDIDATE_STABILITY_SETS.items():
        output[target] = []
        print('\n' + '=' * 70)
        print(target)
        print('=' * 70)
        for pred_set in pred_sets:
            print(f"\n── {target} ~ {pred_set} ──")
            weights, rmses, rows = stability_test(target, pred_set, M, mask, ci)
            record = {'predictors': pred_set, 'n_rows': 0}
            if weights is None:
                print('  NOT ENOUGH ROWS')
                output[target].append(record)
                continue
            record['n_rows'] = int(len(rows))
            record['mean_weights'] = {pred_set[j]: float(weights[:, j].mean()) for j in range(weights.shape[1])}
            record['std_weights'] = {pred_set[j]: float(weights[:, j].std()) for j in range(weights.shape[1])}
            record['mean_rmse'] = float(rmses.mean())
            record['std_rmse'] = float(rmses.std())
            print(f"  Total co-observed rows: {len(rows)}")
            print(f"\n  Weight statistics across {len(rmses)} subsamples:")
            print(f"  {'Predictor':>10} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'CoV':>10}")
            print(f"  " + '-' * 65)
            for j, col in enumerate(pred_set):
                w = weights[:, j]
                mean = w.mean()
                std = w.std()
                cov = std / (mean + 1e-10)
                print(f"  {col:>10} {mean:>10.6f} {std:>10.6f} {w.min():>10.6f} {w.max():>10.6f} {cov:>10.4f}")
            wsum = weights.sum(axis=1)
            print(f"  {'sum(w)':>10} {wsum.mean():>10.6f} {wsum.std():>10.6f} {wsum.min():>10.6f} {wsum.max():>10.6f}")
            print(f"\n  RMSE: mean={rmses.mean():.6f}, std={rmses.std():.6f}, min={rmses.min():.6f}, max={rmses.max():.6f}")
            ti = ci[target]
            pis = [ci[c] for c in pred_set]
            y_full = M[rows, ti]
            X_full = M[np.ix_(rows, pis)]
            w_full, rmse_full, _ = fit_convex(y_full, X_full, equality=False, maxiter=500, ftol=1e-12)
            record['full_fit'] = {
                'rmse': float(rmse_full),
                'weights': {pred_set[j]: float(w_full[j]) for j in range(len(pred_set)) if w_full[j] > 1e-6},
                'sum': float(w_full.sum()),
            }
            print(f"\n  Full-data fit (n={len(rows)}): RMSE={rmse_full:.6f}")
            for j, col in enumerate(pred_set):
                if w_full[j] > 1e-6:
                    print(f"    {col}: {w_full[j]:.6f}")
            print(f"    sum(w) = {w_full.sum():.6f}")
            output[target].append(record)
    save_json(output, args.outdir / '02_stability_validation.json')


if __name__ == '__main__':
    main()
