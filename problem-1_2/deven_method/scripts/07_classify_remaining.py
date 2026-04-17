from __future__ import annotations

from common import ALL_KNOWN_FARMERS, CONFIRMED_FARMERS, CONFIRMED_INDICES, LIKELY_FARMERS, UNCLASSIFIED, fit_convex, load_matrix, parse_args, save_json
import numpy as np


def fit_convex_eq(y, X):
    return fit_convex(y, X, equality=True, maxiter=2000, ftol=1e-15)


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    _, col_names, M, mask, ci = load_matrix(args.csv)
    print(f"Matrix shape: {M.shape}, Observed: {mask.mean():.4f}")
    print(f"\nConfirmed indices ({len(CONFIRMED_INDICES)}): {sorted(CONFIRMED_INDICES)}")
    print(f"Confirmed farmers ({len(CONFIRMED_FARMERS)}): {sorted(CONFIRMED_FARMERS)}")
    print(f"Likely farmers ({len(LIKELY_FARMERS)}): {sorted(LIKELY_FARMERS)}")
    print(f"Unclassified ({len(UNCLASSIFIED)}): {UNCLASSIFIED}")
    print('\n' + '=' * 70)
    print('TEST 1: Regress on ALL known farmers (complete cases)')
    print('=' * 70)
    farmer_is = [ci[c] for c in ALL_KNOWN_FARMERS]
    test1 = {}
    for col in UNCLASSIFIED:
        ti = ci[col]
        obs = mask[:, ti].copy()
        for fi in farmer_is:
            obs &= mask[:, fi]
        rows = np.where(obs)[0]
        if len(rows) < 5:
            print(f'  {col}: only {len(rows)} complete rows — skipping')
            test1[col] = {'rows': int(len(rows))}
            continue
        y = M[rows, ti]
        X = M[np.ix_(rows, farmer_is)]
        w, rmse, _ = fit_convex_eq(y, X)
        active = [(ALL_KNOWN_FARMERS[k], float(w[k])) for k in range(len(ALL_KNOWN_FARMERS)) if w[k] > 0.005]
        active.sort(key=lambda x: -x[1])
        print(f"  {col}: n={len(rows):>4}, RMSE={rmse:.6f}, Σw={w.sum():.4f}, [{', '.join(f'{c}={v:.3f}' for c, v in active[:8])}]")
        test1[col] = {'rows': int(len(rows)), 'rmse': float(rmse), 'weights': dict(active)}
    print('\n' + '=' * 70)
    print('TEST 2: Forward stepwise on known farmers only')
    print('=' * 70)
    results = []
    for col in UNCLASSIFIED:
        ti = ci[col]
        target_rows = np.where(mask[:, ti])[0]
        y_var = np.var(M[target_rows, ti])
        selected_is, selected_names = [], []
        best_hist = None
        for step in range(12):
            best_rmse, best_j, best_jname = np.inf, None, None
            for fname in ALL_KNOWN_FARMERS:
                j = ci[fname]
                if j in selected_is:
                    continue
                trial = selected_is + [j]
                obs = mask[:, ti].copy()
                for c in trial:
                    obs &= mask[:, c]
                rows = np.where(obs)[0]
                if len(rows) < 22:
                    continue
                y = M[rows, ti]
                X = M[np.ix_(rows, trial)]
                w, rmse, _ = fit_convex_eq(y, X)
                if rmse < best_rmse:
                    best_rmse, best_j, best_jname = rmse, j, fname
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
            w, rmse, _ = fit_convex_eq(y, X)
            r2 = 1 - (rmse ** 2) / y_var if y_var > 0 else 0
            best_hist = {'step': step + 1, 'rmse': float(rmse), 'r2': float(r2), 'n_rows': int(len(rows)), 'w_sum': float(w.sum()),
                         'weights': {selected_names[k]: float(w[k]) for k in range(len(selected_is)) if w[k] > 1e-6}}
            if rmse < 0.01:
                break
        if best_hist:
            results.append({'col': col, **best_hist})
    results.sort(key=lambda r: r['rmse'])
    print(f"\n{'Rk':>3} {'Col':>8} {'RMSE':>10} {'R²':>12} {'#p':>3} {'#rows':>5}  Components")
    print('-' * 90)
    for ri, r in enumerate(results):
        preds = ', '.join(f"{c}={w:.3f}" for c, w in sorted(r['weights'].items(), key=lambda x: -x[1]) if w > 0.005)
        marker = ' *** INDEX?' if r['rmse'] < 0.5 else ''
        print(f"{ri+1:>3} {r['col']:>8} {r['rmse']:>10.4f} {r['r2']:>12.8f} {r['n_rows']:>3} {r['n_rows']:>5}  {preds}{marker}")
    print('\n' + '=' * 70)
    print('SANITY CHECK: Confirmed farmers regressed on other farmers')
    print('=' * 70)
    sanity = {}
    for col in sorted(CONFIRMED_FARMERS):
        ti = ci[col]
        other_farmers = [ci[c] for c in ALL_KNOWN_FARMERS if c != col]
        obs = mask[:, ti].copy()
        for fi in other_farmers:
            obs &= mask[:, fi]
        rows = np.where(obs)[0]
        if len(rows) < 5:
            print(f'  {col}: {len(rows)} complete rows — skipping')
            sanity[col] = {'rows': int(len(rows))}
            continue
        y = M[rows, ti]
        X = M[np.ix_(rows, other_farmers)]
        w, rmse, _ = fit_convex_eq(y, X)
        print(f'  {col}: n={len(rows):>4}, RMSE={rmse:.4f}')
        sanity[col] = {'rows': int(len(rows)), 'rmse': float(rmse)}
    print('\n' + '=' * 70)
    print('FINAL CLASSIFICATION')
    print('=' * 70)
    classes = {}
    for r in results:
        if r['rmse'] < 0.5:
            label = 'LIKELY INDEX'
        elif r['rmse'] < 2.0:
            label = 'UNCERTAIN'
        else:
            label = 'LIKELY FARMER'
        classes[r['col']] = label
        print(f"  {r['col']}: {label} (RMSE={r['rmse']:.4f})")
    save_json({'test1': test1, 'results': results, 'sanity': sanity, 'classes': classes}, args.outdir / '07_classify_remaining.json')


if __name__ == '__main__':
    main()
