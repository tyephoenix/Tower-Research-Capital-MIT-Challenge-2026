from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from common import KNOWN_DECOMPS_EXACT, fill_all, load_matrix, parse_args, save_json


def fit_convex_exact(y, X):
    def obj(w):
        r = y - X @ w
        return float(np.dot(r, r))

    def jac(w):
        return -2.0 * X.T @ (y - X @ w)

    p = X.shape[1]
    bounds = [(0, None)] * p
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    w0 = np.ones(p) / p
    result = minimize(obj, w0, jac=jac, method='SLSQP', bounds=bounds, constraints=constraints,
                      options={'maxiter': 2000, 'ftol': 1e-15})
    return result.x, result


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    df, col_names, M, mask, ci = load_matrix(args.csv)
    filled = np.zeros_like(mask, dtype=bool)
    print('Filling from exact decompositions...')
    fill_all(M, mask, filled, ci, KNOWN_DECOMPS_EXACT[:6], verbose=True)
    print(f'Observation rate after exact fills: {mask.mean():.4f}')
    targets = [
        ('col_30', ['col_26', 'col_19', 'col_34', 'col_40', 'col_09', 'col_45', 'col_24']),
        ('col_46', ['col_15', 'col_34', 'col_32', 'col_09', 'col_23', 'col_05', 'col_37', 'col_20', 'col_04']),
    ]
    out = {}
    for target, farmers in targets:
        ti = ci[target]
        fis = [ci[c] for c in farmers]
        obs = mask[:, ti].copy()
        for fi in fis:
            obs &= mask[:, fi]
        rows = np.where(obs)[0]
        print('\n' + '=' * 70)
        print(f'RE-SOLVING {target}')
        print(f'  Predictors: {farmers}')
        print(f'  Co-observed rows (with filled data): {len(rows)}')
        y = M[rows, ti]
        X = M[np.ix_(rows, fis)]
        w, result = fit_convex_exact(y, X)
        pred = X @ w
        residuals = y - pred
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        max_err = float(np.max(np.abs(residuals)))
        print('\n  REFINED WEIGHTS (full precision):')
        for k, c in enumerate(farmers):
            print(f'    {c}: {w[k]:.15f}')
        print(f'    Σw = {w.sum():.15f}')
        print(f'\n  RMSE: {rmse:.15f}')
        print(f'  Max |error|: {max_err:.15f}')
        print(f'  Solver success: {result.success}')
        print(f'  Solver message: {result.message}')
        M_orig = df[col_names].values.astype(np.float64)
        mask_orig = ~np.isnan(M_orig)
        obs_orig = mask_orig[:, ti].copy()
        for fi in fis:
            obs_orig &= mask_orig[:, fi]
        rows_orig = np.where(obs_orig)[0]
        validation = None
        if len(rows_orig) > 0:
            y_orig = M_orig[rows_orig, ti]
            X_orig = M_orig[np.ix_(rows_orig, fis)]
            pred_orig = X_orig @ w
            res_orig = y_orig - pred_orig
            validation = {
                'complete_rows': int(len(rows_orig)),
                'rmse': float(np.sqrt(np.mean(res_orig ** 2))),
                'max_abs_error': float(np.max(np.abs(res_orig))),
            }
            print('\n  VALIDATION ON ORIGINAL (unfilled) DATA:')
            print(f"    Complete rows: {validation['complete_rows']}")
            print(f"    RMSE: {validation['rmse']:.15f}")
            print(f"    Max |error|: {validation['max_abs_error']:.15f}")
        out[target] = {
            'predictors': farmers,
            'weights': {farmers[k]: float(w[k]) for k in range(len(farmers))},
            'train_rows': int(len(rows)),
            'train_rmse': rmse,
            'train_max_abs_error': max_err,
            'validation_original': validation,
        }
    save_json(out, args.outdir / '06_refine_col30_col46.json')


if __name__ == '__main__':
    main()
