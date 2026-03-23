"""
Problem 4 — Arbitrage: buy from NaN src, sell to index dest.

Strategy:
  1. Predict all prices (lookup for t<=3649, KNN+LR otherwise).
  2. Find the cheapest NaN column (src) and most expensive index column (dest).
  3. If dest_price > src_price, trade 100kg for profit.
"""

import os, json
import numpy as np
import pandas as pd

DATA_PATH = "../data/limestone_data_challenge_2026.data.csv"
COMPLETED_PATH = "../answers/problem2_answer.csv"
ANSWER_1A = "../answers/problem1a_answer.csv"
ANSWER_1B = "../answers/problem1b_answer.csv"
COEFF_JSON = "../problem-1_2/intermediates/coefficients.json"

MAX_HISTORICAL_T = 3649
KNN_K = 20
PROJECTION_RANK = 12
KNN_WEIGHT = 0.5

# ── Lazy-loaded state ────────────────────────────────────────────────────────
_completed = None
_cols = None
_decompositions = None
_index_cols = None


def _load_decompositions(cols):
    """Load decompositions from intermediates or answer file."""
    if os.path.exists(COEFF_JSON):
        with open(COEFF_JSON) as f:
            raw = json.load(f)
        raw_dec = raw.get("decompositions", raw)
        decompositions = {}
        for idx_col, dec in raw_dec.items():
            fi_idxs = [int(i) for i in dec["farmer_idxs"]]
            decompositions[idx_col] = {
                "farmer_names": [cols[i] for i in fi_idxs],
                "coefs": [float(c) for c in dec["coefs"]],
            }
        return decompositions

    df = pd.read_csv(ANSWER_1B)
    decompositions = {}
    for idx_col, grp in df.groupby("index_col"):
        decompositions[idx_col] = {
            "farmer_names": grp["constituent_col"].tolist(),
            "coefs": grp["coef"].tolist(),
        }
    return decompositions


def _algebraic_fill(row_vals, cols, decompositions):
    """
    Try to deterministically fill NaN cells from decomposition constraints.
    Returns (known_mask, filled_row).
    """
    filled = row_vals.copy()
    known = ~np.isnan(filled)

    changed = True
    while changed:
        changed = False
        for idx_col, dec in decompositions.items():
            idx_i = cols.index(idx_col)
            fi_idxs = [cols.index(c) for c in dec["farmer_names"]]
            coefs = np.array(dec["coefs"], dtype=float)

            if known[idx_i]:
                farmer_known = np.array([known[fi] for fi in fi_idxs])
                if (~farmer_known).sum() == 1:
                    j_miss = int(np.where(~farmer_known)[0][0])
                    w = coefs[j_miss]
                    if w >= 1e-8:
                        known_sum = sum(coefs[j] * filled[fi_idxs[j]]
                                        for j in range(len(fi_idxs)) if j != j_miss)
                        filled[fi_idxs[j_miss]] = (filled[idx_i] - known_sum) / w
                        known[fi_idxs[j_miss]] = True
                        changed = True
            else:
                if all(known[fi] for fi in fi_idxs):
                    filled[idx_i] = sum(coefs[j] * filled[fi_idxs[j]]
                                        for j in range(len(fi_idxs)))
                    known[idx_i] = True
                    changed = True

    return known, filled


def _ensure_loaded():
    global _completed, _cols, _decompositions, _index_cols

    if _completed is not None:
        return

    if not os.path.exists(COMPLETED_PATH):
        raise FileNotFoundError(f"Run problem 2 first — need {COMPLETED_PATH}")

    comp_df = pd.read_csv(COMPLETED_PATH)
    _cols = [c for c in comp_df.columns if c != "time"]
    _completed = comp_df[_cols].values.astype(float)
    _decompositions = _load_decompositions(_cols)

    a1 = pd.read_csv(ANSWER_1A)
    _index_cols = set(a1.loc[a1["is_index"], "column"].tolist())


def _predict_row(row_vals, t):
    """Return predicted prices for all columns."""
    _ensure_loaded()

    if 0 <= t <= MAX_HISTORICAL_T:
        predicted = _completed[t].copy()
        obs = ~np.isnan(row_vals)
        predicted[obs] = row_vals[obs]
        return predicted

    known, det_filled = _algebraic_fill(row_vals, _cols, _decompositions)
    missing = ~known

    if not missing.any():
        return det_filled

    known_idxs = np.where(known)[0]
    dists = np.sum((_completed[:, known_idxs] - det_filled[known_idxs]) ** 2,
                   axis=1)
    nn_idxs = np.argsort(dists)[:KNN_K]
    weights = 1.0 / (np.sqrt(dists[nn_idxs]) + 1e-8)
    weights /= weights.sum()
    knn_pred = _completed[nn_idxs].T @ weights

    col_means = _completed.mean(axis=0)
    centered = _completed - col_means
    _, s, Vt = np.linalg.svd(centered, full_matrices=False)
    Vt_r = Vt[:PROJECTION_RANK]

    centered_known = det_filled[known_idxs] - col_means[known_idxs]
    V_obs = Vt_r[:, known_idxs].T
    alpha, *_ = np.linalg.lstsq(V_obs, centered_known, rcond=None)
    lr_pred = col_means + Vt_r.T @ alpha

    predicted = KNN_WEIGHT * knn_pred + (1 - KNN_WEIGHT) * lr_pred
    predicted[known] = det_filled[known]
    return predicted


# ── Public trading function ──────────────────────────────────────────────────

def trading_problem_4(row):
    """
    Problem 4 entry point.

    Returns
    -------
    pd.DataFrame with columns ``src_col``, ``dest_col``, ``qty``.
    """
    _ensure_loaded()

    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    t = int(row.get("time", -1))
    row_vals = np.array([row.get(c, np.nan) for c in _cols], dtype=float)

    predicted = _predict_row(row_vals, t)
    nan_mask = np.isnan(row_vals)

    src_idxs = np.where(nan_mask)[0]
    if len(src_idxs) == 0:
        return pd.DataFrame({"src_col": [], "dest_col": [], "qty": []})

    dest_idxs = np.array([i for i, c in enumerate(_cols) if c in _index_cols])
    if len(dest_idxs) == 0:
        return pd.DataFrame({"src_col": [], "dest_col": [], "qty": []})

    best_src = src_idxs[np.argmin(predicted[src_idxs])]
    best_dest = dest_idxs[np.argmax(predicted[dest_idxs])]

    src_price = predicted[best_src]
    dest_price = predicted[best_dest]

    if dest_price <= src_price:
        return pd.DataFrame({"src_col": [], "dest_col": [], "qty": []})

    return pd.DataFrame({
        "src_col": [_cols[best_src]],
        "dest_col": [_cols[best_dest]],
        "qty": [100],
    })


# ── CLI test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time as _time

    raw = pd.read_csv(DATA_PATH)
    comp = pd.read_csv(COMPLETED_PATH)
    cols = [c for c in raw.columns if c != "time"]

    _ensure_loaded()
    print(f"Index columns: {sorted(_index_cols)}\n")

    test_ts = [0, 1, 50, 100, 500, 1000, 1825, 2500, 3000, 3649]
    print(f"{'='*130}")
    print("TEST 1: Historical rows (t <= 3649)")
    print(f"{'='*130}")

    total_profit, total_oracle = 0.0, 0.0
    for t in test_ts:
        row = raw.iloc[t]
        row_vals = np.array([row[c] for c in cols], dtype=float)
        predicted = _predict_row(row_vals, t)

        trades = trading_problem_4(row)
        nan_mask = np.isnan(row_vals)
        nan_idxs = np.where(nan_mask)[0]
        idx_idxs = np.array([i for i, c in enumerate(cols) if c in _index_cols])

        if len(nan_idxs) > 0 and len(idx_idxs) > 0:
            best_src_price = min(comp.loc[t, cols[i]] for i in nan_idxs)
            best_dest_price = max(comp.loc[t, cols[i]] for i in idx_idxs)
            oracle_profit = max(0, (best_dest_price - best_src_price) * 100)
        else:
            oracle_profit = 0

        if len(trades) == 0:
            profit = 0
            print(f"  t={t:<5d} NO TRADE  oracle={oracle_profit:10.0f}")
        else:
            src = trades["src_col"].iloc[0]
            dest = trades["dest_col"].iloc[0]
            qty = trades["qty"].iloc[0]
            profit = qty * (comp.loc[t, dest] - comp.loc[t, src])
            print(f"  t={t:<5d} {qty:3d}kg {src} → {dest}  "
                  f"profit={profit:10.0f}  oracle={oracle_profit:10.0f}  "
                  f"gap={profit - oracle_profit:+9.0f}")

        total_profit += profit
        total_oracle += oracle_profit

    print(f"\n  Total profit: {total_profit:>12,.0f}  Oracle: {total_oracle:>12,.0f}")

    print(f"\n{'='*130}")
    print("TEST 2: Full historical sweep (3650 rows)")
    print(f"{'='*130}")

    full_profit, full_oracle = 0.0, 0.0
    n_trades, n_no_trade, violations = 0, 0, 0
    t2_start = _time.perf_counter()

    for t in range(3650):
        row = raw.iloc[t]
        row_vals = np.array([row[c] for c in cols], dtype=float)
        nan_mask = np.isnan(row_vals)
        nan_idxs = np.where(nan_mask)[0]
        idx_idxs = np.array([i for i, c in enumerate(cols) if c in _index_cols])

        trades = trading_problem_4(row)

        if len(nan_idxs) > 0 and len(idx_idxs) > 0:
            best_src_p = min(comp.loc[t, cols[i]] for i in nan_idxs)
            best_dest_p = max(comp.loc[t, cols[i]] for i in idx_idxs)
            oracle = max(0, (best_dest_p - best_src_p) * 100)
        else:
            oracle = 0

        if len(trades) == 0:
            n_no_trade += 1
        else:
            n_trades += 1
            for _, trade in trades.iterrows():
                if trade["src_col"] not in {cols[i] for i in nan_idxs}:
                    violations += 1
                if trade["dest_col"] not in _index_cols:
                    violations += 1
            src = trades["src_col"].iloc[0]
            dest = trades["dest_col"].iloc[0]
            qty = trades["qty"].iloc[0]
            full_profit += qty * (comp.loc[t, dest] - comp.loc[t, src])

        full_oracle += oracle

    t2_elapsed = _time.perf_counter() - t2_start
    print(f"  Total profit:      {full_profit:>14,.0f}")
    print(f"  Oracle profit:     {full_oracle:>14,.0f}")
    print(f"  Capture rate:      {100*full_profit/full_oracle:.1f}%" if full_oracle > 0 else "")
    print(f"  Rows traded:       {n_trades}")
    print(f"  Violations:        {violations}  {'PASS' if violations == 0 else 'FAIL'}")
    print(f"  Time:              {t2_elapsed:.2f}s ({1000*t2_elapsed/3650:.1f} ms/row)")

    print(f"\n{'='*130}")
    print("TEST 3: Synthetic future rows (t > 3649)")
    print(f"{'='*130}")

    idx_idxs = np.array([i for i, c in enumerate(cols) if c in _index_cols])
    syn_profit, syn_oracle = 0.0, 0.0
    t3_times = []
    base_rows = [0, 100, 500, 1000, 2000, 3000, 3649]
    for bt in base_rows:
        true_row = comp.iloc[bt]
        true_vals = np.array([true_row[c] for c in cols], dtype=float)

        rng = np.random.default_rng(42)
        mask = rng.random(len(cols)) < 0.5
        synthetic = true_vals.copy()
        synthetic[mask] = np.nan

        row_series = pd.Series({"time": MAX_HISTORICAL_T + 1})
        for i, c in enumerate(cols):
            row_series[c] = synthetic[i]

        t0 = _time.perf_counter()
        trades = trading_problem_4(row_series)
        elapsed = _time.perf_counter() - t0
        t3_times.append(elapsed)

        nan_idxs_syn = np.where(mask)[0]
        best_src_p = min(true_vals[i] for i in nan_idxs_syn)
        best_dest_p = max(true_vals[i] for i in idx_idxs)
        oracle = max(0, (best_dest_p - best_src_p) * 100)

        if len(trades) == 0:
            print(f"  t=SYN(base={bt:<4d}) NO TRADE  oracle={oracle:10.0f}")
        else:
            src = trades["src_col"].iloc[0]
            dest = trades["dest_col"].iloc[0]
            qty = trades["qty"].iloc[0]
            profit = qty * (true_vals[cols.index(dest)] - true_vals[cols.index(src)])
            syn_profit += profit
            print(f"  t=SYN(base={bt:<4d}) {qty:3d}kg {src} → {dest}  "
                  f"profit={profit:10.0f}  oracle={oracle:10.0f}  "
                  f"gap={profit - oracle:+9.0f}  time={elapsed:.2f}s")

        syn_oracle += oracle

    print(f"\n  Total profit: {syn_profit:>12,.0f}  Oracle: {syn_oracle:>12,.0f}")
    print(f"  Avg time/row: {sum(t3_times)/len(t3_times):.2f}s")
