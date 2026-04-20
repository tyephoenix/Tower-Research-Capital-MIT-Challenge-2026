from __future__ import annotations

import numpy as np
import pandas as pd

from common import CONFIRMED_INDICES, KNOWN_DECOMPS_EXACT, fill_all, forward_stepwise, load_matrix, parse_args, save_json


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    df, col_names, M, mask, ci = load_matrix(args.csv)
    filled = np.zeros_like(mask, dtype=bool)
    print(f"Matrix shape: {M.shape}, Initial observed: {mask.mean():.4f} ({mask.sum()} entries)")
    print(f"Confirmed indices: {sorted(CONFIRMED_INDICES)}")
    print(f"Total decompositions: {len(KNOWN_DECOMPS_EXACT)}")
    print('\n' + '=' * 70)
    print('PHASE 1: AGGRESSIVE FILL (all 8 exact decompositions)')
    print('=' * 70)
    n_filled = fill_all(M, mask, filled, ci, KNOWN_DECOMPS_EXACT, verbose=True)
    print(f"\nTotal fills: {n_filled}")
    print(f"Observation rate: {mask.mean():.4f} ({mask.sum()} entries)")
    print('\nPer-column fills:')
    for j, c in enumerate(col_names):
        n = int(filled[:, j].sum())
        if n > 0:
            pct = mask[:, j].sum() / M.shape[0] * 100
            print(f"  {c}: +{n} ({pct:.1f}% observed now)")

    print('\n' + '=' * 70)
    print('PHASE 2: FORWARD STEPWISE')
    print('=' * 70)
    predictor_pool = sorted(c for c in col_names if c not in CONFIRMED_INDICES)
    candidates = [c for c in col_names if c not in CONFIRMED_INDICES]
    all_summaries = []
    new_exact = []
    for col in candidates:
        hist = forward_stepwise(col, M, mask, ci, predictor_pool, equality=True, max_steps=12, min_rows=25, verbose=True)
        if hist:
            final = hist[-1]
            all_summaries.append({'col': col, 'rmse': final.rmse, 'r2': final.r2, 'n_rows': final.n_rows,
                                  'weights': final.weights, 'w_sum': final.w_sum, 'n_pred': final.step})
            if final.rmse < 0.01:
                new_exact.append({'col': col, 'weights': final.weights})
    all_summaries.sort(key=lambda r: r['rmse'])
    print('\n' + '=' * 70)
    print('ALL COLUMNS RANKED BY RMSE')
    print('=' * 70)
    print(f"\n{'Rk':>3} {'Col':>8} {'RMSE':>10} {'R²':>12} {'#p':>3} {'#rows':>5} {'Σw':>8}  Components")
    print('-' * 100)
    for ri, r in enumerate(all_summaries):
        preds = ', '.join(f"{c}={w:.3f}" for c, w in sorted(r['weights'].items(), key=lambda x: -x[1]) if w > 0.005)
        print(f"{ri+1:>3} {r['col']:>8} {r['rmse']:>10.4f} {r['r2']:>12.8f} {r['n_pred']:>3} {r['n_rows']:>5} {r['w_sum']:>8.4f}  {preds}")
    print('\n' + '=' * 70)
    print('EXACT / NEAR-EXACT (RMSE < 0.5)')
    print('=' * 70)
    for r in all_summaries:
        if r['rmse'] < 0.5:
            print(f"\n{r['col']}: RMSE={r['rmse']:.6f}, R²={r['r2']:.10f}, n={r['n_rows']}, Σw={r['w_sum']:.6f}")
            for c, w in sorted(r['weights'].items(), key=lambda x: -x[1]):
                if w > 0.001:
                    print(f"  {c}: {w:.6f}")
    print('\n' + '=' * 70)
    print('FINAL REPORT')
    print('=' * 70)
    print(f"Observation rate: {0.5117:.4f} -> {mask.mean():.4f}")
    print(f"Entries filled: {filled.sum()}")
    print(f"Confirmed indices: {sorted(CONFIRMED_INDICES)}")
    if new_exact:
        print(f"Newly confirmed: {[d['col'] for d in new_exact]}")
    save_json({'fills': int(n_filled), 'summaries': all_summaries, 'new_exact': new_exact}, args.outdir / '05_exact_fill_all.json')
    filled_df = pd.DataFrame(M, columns=col_names)
    if 'time' in df.columns:
        filled_df.insert(0, 'time', df['time'])
    filled_df.to_csv(args.outdir / 'filled_matrix.csv', index=False)


if __name__ == '__main__':
    main()
