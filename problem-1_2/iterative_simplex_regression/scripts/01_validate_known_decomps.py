from __future__ import annotations

import numpy as np

from common import KNOWN_DECOMPS_EXACT, load_matrix, parse_args, save_json


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    _, col_names, M, mask, ci = load_matrix(args.csv)

    results = []
    print(f"Matrix shape: {M.shape}, Observed: {mask.mean():.4f}")

    for idx_col, weights in KNOWN_DECOMPS_EXACT[:6]:  # original exact set before adding 30/46
        ti = ci[idx_col]
        farmers = list(weights.keys())
        fis = [ci[c] for c in farmers]
        ws = [weights[c] for c in farmers]
        obs = mask[:, ti].copy()
        for fi in fis:
            obs &= mask[:, fi]
        rows = np.where(obs)[0]
        wsum = float(sum(ws))
        row_res = {
            'index': idx_col,
            'weights': weights,
            'weight_sum': wsum,
            'complete_rows': int(len(rows)),
        }
        print('=' * 70)
        wstr = ' + '.join(f'{w:.4f}·{c}' for c, w in sorted(weights.items(), key=lambda x: -x[1]))
        print(f'{idx_col} = {wstr}')
        print(f'  Σw = {wsum:.6f}')
        print(f'  Complete rows: {len(rows)}')
        if len(rows) == 0:
            pairs = {}
            print('  NO COMPLETE ROWS — cannot validate directly')
            for c, _w in sorted(weights.items(), key=lambda x: -x[1]):
                both = mask[:, ti] & mask[:, ci[c]]
                pairs[c] = int(both.sum())
                print(f'    {idx_col} & {c}: {both.sum()} co-observed rows')
            row_res['pair_counts'] = pairs
        else:
            y = M[rows, ti]
            pred = sum(ws[k] * M[rows, fis[k]] for k in range(len(farmers)))
            residuals = y - pred
            rmse = float(np.sqrt(np.mean(residuals ** 2)))
            max_err = float(np.max(np.abs(residuals)))
            denom = np.sum((y - y.mean()) ** 2)
            r2 = float(1 - np.sum(residuals ** 2) / denom) if denom > 0 else float('nan')
            row_res.update({
                'rmse': rmse,
                'max_abs_error': max_err,
                'r2': r2,
                'first5_residuals': residuals[:5].tolist(),
            })
            print(f'  RMSE: {rmse:.10f}')
            print(f'  Max |error|: {max_err:.10f}')
            print(f'  R²: {r2:.10f}')
            print(f'  First 5 residuals: {np.round(residuals[:5], 8)}')
        results.append(row_res)
    save_json(results, args.outdir / '01_validate_known_decomps.json')


if __name__ == '__main__':
    main()
