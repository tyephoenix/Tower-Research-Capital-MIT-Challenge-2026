from __future__ import annotations

import numpy as np

from common import fit_convex, load_matrix, parse_args, save_json

HIGH_CONF_FARMERS = {
    'col_24', 'col_34', 'col_31', 'col_52', 'col_32', 'col_12', 'col_49',
    'col_07', 'col_20', 'col_22', 'col_26', 'col_28', 'col_42',
}
MEDIUM_CONF = {
    'col_05', 'col_23', 'col_45', 'col_15', 'col_09', 'col_08', 'col_01', 'col_04', 'col_06',
    'col_10', 'col_14', 'col_19', 'col_25', 'col_27', 'col_29', 'col_33', 'col_35',
    'col_36', 'col_37', 'col_38', 'col_39', 'col_41', 'col_02', 'col_03', 'col_00',
    'col_40', 'col_47', 'col_21',
}
CONFIRMED_INDICES = {'col_11', 'col_50'}
FARMER_CANDIDATES = sorted(HIGH_CONF_FARMERS | MEDIUM_CONF)


def forward_stepwise_farmers(target_col, farmer_pool, M, mask, ci, max_predictors=8, min_rows=25):
    ti = ci[target_col]
    pool = [ci[c] for c in farmer_pool if c != target_col]
    pool_names = [c for c in farmer_pool if c != target_col]
    target_rows = np.where(mask[:, ti])[0]
    y_var = np.var(M[target_rows, ti])
    selected, selected_names, history = [], [], []
    for step in range(max_predictors):
        best_rmse = np.inf
        best_j = None
        best_jname = None
        for j, jname in zip(pool, pool_names):
            if j in selected:
                continue
            trial = selected + [j]
            obs = mask[:, ti].copy()
            for col in trial:
                obs &= mask[:, col]
            rows = np.where(obs)[0]
            if len(rows) < min_rows:
                continue
            y = M[rows, ti]
            X = M[np.ix_(rows, trial)]
            w, rmse, _ = fit_convex(y, X, equality=False, maxiter=500, ftol=1e-12)
            if w is not None and rmse < best_rmse:
                best_rmse = rmse
                best_j = j
                best_jname = jname
        if best_j is None:
            break
        selected.append(best_j)
        selected_names.append(best_jname)
        obs = mask[:, ti].copy()
        for col in selected:
            obs &= mask[:, col]
        rows = np.where(obs)[0]
        y = M[rows, ti]
        X = M[np.ix_(rows, selected)]
        w_final, rmse_final, _ = fit_convex(y, X, equality=False, maxiter=500, ftol=1e-12)
        r2 = 1 - (rmse_final ** 2) / y_var if y_var > 0 else 0
        weights = {selected_names[k]: float(w_final[k]) for k in range(len(selected)) if w_final[k] > 1e-6}
        history.append({
            'step': step + 1,
            'added': best_jname,
            'rmse': float(rmse_final),
            'r2': float(r2),
            'n_rows': int(len(rows)),
            'weights': weights,
            'w_sum': float(np.sum(w_final)),
        })
        wstr = ', '.join(f'{selected_names[k]}={w_final[k]:.4f}' for k in range(len(selected)) if w_final[k] > 1e-4)
        print(f"    step {step+1}: +{best_jname:>8}, RMSE={rmse_final:.6f}, R²={r2:.8f}, n={len(rows):>4}, Σw={np.sum(w_final):.6f}  [{wstr}]")
        if rmse_final < 0.01:
            print('    >>> EXACT FIT FOUND!')
            break
        if step > 0 and history[-2]['rmse'] - rmse_final < 0.005:
            break
    return history


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    _, col_names, M, mask, ci = load_matrix(args.csv)
    print(f"Matrix shape: {M.shape}, Observed: {mask.mean():.3f}")
    print(f"\nConfirmed indices: {sorted(CONFIRMED_INDICES)}")
    print(f"Farmer candidates ({len(FARMER_CANDIDATES)}): {FARMER_CANDIDATES}")
    remaining = sorted(set(col_names) - CONFIRMED_INDICES - set(FARMER_CANDIDATES))
    print(f"Remaining to classify: {remaining}")
    print('\n' + '=' * 70)
    print('FORWARD STEPWISE WITH FARMER-ONLY PREDICTORS')
    print('=' * 70)
    all_results = {}
    for col in col_names:
        print(f"\n── {col} ──")
        hist = forward_stepwise_farmers(col, FARMER_CANDIDATES, M, mask, ci, max_predictors=8, min_rows=25)
        all_results[col] = hist
    summary = []
    for col in col_names:
        hist = all_results[col]
        if hist:
            final = hist[-1]
            summary.append({
                'col': col,
                'rmse': final['rmse'],
                'r2': final['r2'],
                'n_pred': final['step'],
                'n_rows': final['n_rows'],
                'w_sum': final['w_sum'],
                'weights': final['weights'],
            })
        else:
            summary.append({'col': col, 'rmse': float('inf'), 'r2': 0, 'n_pred': 0, 'n_rows': 0, 'w_sum': 0, 'weights': {}})
    summary.sort(key=lambda r: r['rmse'])
    print('\n' + '=' * 70)
    print('SUMMARY: Final fit for each column (farmer-only predictors)')
    print('=' * 70)
    print(f"\n{'Rk':>3} {'Column':>8} {'RMSE':>10} {'R²':>12} {'#p':>3} {'#rows':>5} {'Σw':>8}  Components")
    print('-' * 100)
    for rank_i, r in enumerate(summary):
        marker = ''
        if r['col'] in CONFIRMED_INDICES:
            marker = ' ← INDEX'
        elif r['col'] in HIGH_CONF_FARMERS:
            marker = ' ← farmer'
        preds = ', '.join(f"{c}={w:.3f}" for c, w in sorted(r['weights'].items(), key=lambda x: -x[1]) if w > 0.005)
        print(f"{rank_i+1:>3} {r['col']:>8} {r['rmse']:>10.4f} {r['r2']:>12.8f} {r['n_pred']:>3} {r['n_rows']:>5} {r['w_sum']:>8.4f}  {preds}{marker}")
    save_json({'all_results': all_results, 'summary': summary}, args.outdir / '03_farmer_only_stepwise.json')


if __name__ == '__main__':
    main()
