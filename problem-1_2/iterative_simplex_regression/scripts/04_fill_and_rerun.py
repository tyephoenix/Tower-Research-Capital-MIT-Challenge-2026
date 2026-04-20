from __future__ import annotations

import numpy as np

from common import KNOWN_DECOMPS_EXACT, fill_all, load_matrix, parse_args, save_json, fit_convex

# Start from 4 confirmed decompositions before exact confirmation of 30 and 46
ROUND_DECOMPS = KNOWN_DECOMPS_EXACT[:6]
APPROX = [
    ("col_46", {"col_15": 0.295, "col_34": 0.237, "col_09": 0.122, "col_32": 0.112,
                "col_23": 0.084, "col_05": 0.080, "col_37": 0.031, "col_20": 0.026,
                "col_04": 0.012}),
    ("col_30", {"col_26": 0.219, "col_19": 0.214, "col_34": 0.148, "col_40": 0.146,
                "col_09": 0.126, "col_45": 0.079, "col_24": 0.067}),
]


def run_stepwise_all(M, mask, ci, col_names, predictor_pool, max_pred=10, min_rows=25):
    summaries = []
    for col in col_names:
        ti = ci[col]
        pool = [(ci[c], c) for c in predictor_pool if c != col]
        target_rows = np.where(mask[:, ti])[0]
        y_var = np.var(M[target_rows, ti]) if len(target_rows) else 0.0
        selected_is, selected_names = [], []
        history = []
        for step in range(max_pred):
            best_rmse, best_j, best_jname = np.inf, None, None
            for j, jname in pool:
                if j in selected_is:
                    continue
                trial = selected_is + [j]
                obs = mask[:, ti].copy()
                for c in trial:
                    obs &= mask[:, c]
                rows = np.where(obs)[0]
                if len(rows) < min_rows:
                    continue
                y = M[rows, ti]
                X = M[np.ix_(rows, trial)]
                w, rmse, _ = fit_convex(y, X, equality=False, maxiter=500, ftol=1e-12)
                if w is not None and rmse < best_rmse:
                    best_rmse, best_j, best_jname = rmse, j, jname
            if best_j is None:
                break
            selected_is.append(best_j)
            selected_names.append(best_jname)
            obs = mask[:, ti].copy()
            for c in selected_is:
                obs &= mask[:, c]
            rows = np.where(obs)[0]
            y = M[rows, ti]
            X = M[np.ix_(rows, selected_is)]
            w_final, rmse_final, _ = fit_convex(y, X, equality=False, maxiter=500, ftol=1e-12)
            r2 = 1 - (rmse_final ** 2) / y_var if y_var > 0 else 0
            history.append({'step': step + 1, 'rmse': float(rmse_final), 'r2': float(r2), 'n_rows': int(len(rows)),
                            'weights': {selected_names[k]: float(w_final[k]) for k in range(len(selected_is)) if w_final[k] > 1e-6},
                            'w_sum': float(np.sum(w_final))})
            if rmse_final < 0.01:
                break
            if step > 0 and history[-2]['rmse'] - rmse_final < 0.005:
                break
        if history:
            final = history[-1]
            summaries.append({'col': col, **final})
    summaries.sort(key=lambda r: r['rmse'])
    return summaries


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    _, col_names, M, mask, ci = load_matrix(args.csv)
    filled = np.zeros_like(mask, dtype=bool)
    print(f'Matrix shape: {M.shape}, Initial observed: {mask.mean():.4f} ({mask.sum()} entries)')
    print('\nExact decompositions:', len(ROUND_DECOMPS))
    print('Approximate decompositions:', len(APPROX))
    all_decomps = ROUND_DECOMPS + APPROX
    for idx_col, w in all_decomps:
        tag = ' [APPROX]' if (idx_col, w) in APPROX else ''
        wstr = ' + '.join(f'{v:.4f}·{k}' for k, v in sorted(w.items(), key=lambda x: -x[1]))
        print(f'  {idx_col} = {wstr}{tag}')
    print('\n' + '=' * 70)
    print('PHASE 1: AGGRESSIVE FILL (exact + approximate)')
    print('=' * 70)
    n_filled = fill_all(M, mask, filled, ci, all_decomps, verbose=True)
    print(f'\nTotal fills: {n_filled}')
    print(f'Observation rate: {mask.mean():.4f} ({mask.sum()} entries)')
    # Re-solve col_46 and col_30 using approximate supports
    targets = [
        ('col_46', ['col_15', 'col_34', 'col_09', 'col_32', 'col_23', 'col_05', 'col_37', 'col_20', 'col_04']),
        ('col_30', ['col_26', 'col_19', 'col_34', 'col_40', 'col_09', 'col_45', 'col_24']),
    ]
    refined = {}
    print('\n' + '=' * 70)
    print('PHASE 2: RE-SOLVE col_46 and col_30 WITH DENSER DATA')
    print('=' * 70)
    for target, preds in targets:
        ti = ci[target]
        pis = [ci[c] for c in preds]
        obs = mask[:, ti].copy()
        for pi in pis:
            obs &= mask[:, pi]
        rows = np.where(obs)[0]
        print(f'\n── Re-solving {target} ──')
        print(f'  Co-observed rows with original predictors: {len(rows)}')
        if len(rows) >= 20:
            y = M[rows, ti]
            X = M[np.ix_(rows, pis)]
            w, rmse, _ = fit_convex(y, X, equality=False, maxiter=2000, ftol=1e-15)
            refined[target] = {'predictors': preds, 'weights': {preds[k]: float(w[k]) for k in range(len(preds))}, 'rmse': float(rmse), 'rows': int(len(rows))}
            print(f'  Full fit: RMSE={rmse:.8f}, Σw={w.sum():.6f}')
            for k, c in enumerate(preds):
                print(f'    {c}: {w[k]:.6f}')
    predictor_pool = sorted(c for c in col_names if c not in {'col_11', 'col_42', 'col_48', 'col_50', 'col_46', 'col_30'})
    print('\n' + '=' * 70)
    print('PHASE 3: FULL STEPWISE FOR ALL NON-CONFIRMED COLUMNS')
    print('=' * 70)
    candidates = [c for c in col_names if c not in {'col_11', 'col_42', 'col_48', 'col_50', 'col_46', 'col_30'}]
    summaries = run_stepwise_all(M, mask, ci, candidates, predictor_pool, max_pred=10, min_rows=25)
    print(f"\n{'Rk':>3} {'Col':>8} {'RMSE':>10} {'R²':>12} {'#p':>3} {'#rows':>5} {'Σw':>8}  Components")
    print('-' * 100)
    for ri, r in enumerate(summaries[:25]):
        preds = ', '.join(f"{c}={w:.3f}" for c, w in sorted(r['weights'].items(), key=lambda x: -x[1]) if w > 0.005)
        print(f"{ri+1:>3} {r['col']:>8} {r['rmse']:>10.4f} {r['r2']:>12.8f} {r['step']:>3} {r['n_rows']:>5} {r['w_sum']:>8.4f}  {preds}")
    save_json({'fills': int(n_filled), 'refined': refined, 'summaries': summaries}, args.outdir / '04_fill_and_rerun.json')


if __name__ == '__main__':
    main()
