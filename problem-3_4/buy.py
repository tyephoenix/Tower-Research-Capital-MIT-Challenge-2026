"""
Problem 3 — Buy 100kg of flour from NaN-priced columns only.

Strategy:
  1. Predict NaN prices (lookup for t<=3649, KNN+LR otherwise).
  2. Buy all 100kg from the cheapest predicted NaN column.
"""

import os, json
import numpy as np
import pandas as pd

DATA_PATH = "../data/limestone_data_challenge_2026.data.csv"
COMPLETED_PATH = "../answers/problem2_answer-tye.csv"
ANSWER_1B = "../answers/problem1b_answer-tye.csv"
COEFF_JSON = "../problem-2/intermediates/coefficients.json"

MAX_HISTORICAL_T = 3649
KNN_K = 20
PROJECTION_RANK = 12
KNN_WEIGHT = 0.5

# ── Lazy-loaded state ────────────────────────────────────────────────────────
_completed = None
_cols = None
_decompositions = None  # {idx_col: {"farmer_names": [...], "coefs": [...]}}


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
    Returns (known_mask, filled_row) where known[j] = True if observed or algebraic.
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
    global _completed, _cols, _decompositions

    if _completed is not None:
        return

    if not os.path.exists(COMPLETED_PATH):
        raise FileNotFoundError(f"Run problem 2 first — need {COMPLETED_PATH}")

    comp_df = pd.read_csv(COMPLETED_PATH)
    _cols = [c for c in comp_df.columns if c != "time"]
    _completed = comp_df[_cols].values.astype(float)
    _decompositions = _load_decompositions(_cols)


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

def trading_problem_3(row):
    """
    Problem 3 entry point.

    Parameters
    ----------
    row : pd.Series or single-row DataFrame
        One day's bulletin with time + col_00..col_52 (NaNs present).

    Returns
    -------
    pd.DataFrame with columns ``col`` and ``qty`` (ints, sum = 100).
    """
    _ensure_loaded()

    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    t = int(row.get("time", -1))
    row_vals = np.array([row.get(c, np.nan) for c in _cols], dtype=float)

    nan_mask = np.isnan(row_vals)
    nan_idxs = np.where(nan_mask)[0]

    if len(nan_idxs) == 0:
        return pd.DataFrame({"col": [_cols[0]], "qty": [100]})

    predicted = _predict_row(row_vals, t)
    prices = predicted[nan_idxs]
    best = nan_idxs[np.argmin(prices)]
    return pd.DataFrame({"col": [_cols[best]], "qty": [100]})


# ── CLI test ─────────────────────────────────────────────────────────────────

def _test_row(t, raw_df, comp_df, cols):
    row = raw_df.iloc[t]
    row_vals = np.array([row[c] for c in cols], dtype=float)

    trades = trading_problem_3(row)
    col = trades["col"].iloc[0]
    qty = trades["qty"].iloc[0]
    true_price = comp_df.loc[t, col]
    cost = qty * true_price

    predicted = _predict_row(row_vals, t)
    pred_price = predicted[cols.index(col)]

    nan_cols = [c for c in cols if np.isnan(row[c])]
    nan_true = [comp_df.loc[t, c] for c in nan_cols]
    best_possible = min(nan_true) * 100
    worst_possible = max(nan_true) * 100

    print(f"  t={t:<5d} buy {qty:3d}kg {col}  "
          f"pred={pred_price:8.2f}  true={true_price:8.2f}  "
          f"err={pred_price - true_price:+7.2f}  "
          f"cost={cost:10.0f}  oracle={best_possible:10.0f}  "
          f"gap={cost - best_possible:+8.0f}  "
          f"NaNs={len(nan_cols)}")

    return cost, best_possible, worst_possible


def _test_synthetic_row(cols, comp_df, base_t=0):
    true_row = comp_df.iloc[base_t]
    row_vals = np.array([true_row[c] for c in cols], dtype=float)

    rng = np.random.default_rng(42)
    mask = rng.random(len(cols)) < 0.5
    synthetic = row_vals.copy()
    synthetic[mask] = np.nan

    row_series = pd.Series({"time": MAX_HISTORICAL_T + 1})
    for i, c in enumerate(cols):
        row_series[c] = synthetic[i]

    trades = trading_problem_3(row_series)
    col = trades["col"].iloc[0]
    qty = trades["qty"].iloc[0]
    ci = cols.index(col)
    true_price = row_vals[ci]
    predicted = _predict_row(synthetic, MAX_HISTORICAL_T + 1)
    pred_price = predicted[ci]

    nan_idxs = np.where(mask)[0]
    best_possible = min(row_vals[i] for i in nan_idxs) * 100
    cost = qty * true_price

    print(f"  t=SYN   buy {qty:3d}kg {col}  "
          f"pred={pred_price:8.2f}  true={true_price:8.2f}  "
          f"err={pred_price - true_price:+7.2f}  "
          f"cost={cost:10.0f}  oracle={best_possible:10.0f}  "
          f"gap={cost - best_possible:+8.0f}  "
          f"NaNs={mask.sum()}")

    return cost, best_possible


if __name__ == "__main__":
    import time as _time

    raw = pd.read_csv(DATA_PATH)
    comp = pd.read_csv(COMPLETED_PATH)
    cols = [c for c in raw.columns if c != "time"]

    test_ts = [0, 1, 50, 100, 500, 1000, 1825, 2500, 3000, 3649]
    print(f"{'='*120}")
    print("TEST 1: Historical rows (t <= 3649)")
    print(f"{'='*120}")

    total_cost, total_oracle = 0.0, 0.0
    t1_times = []
    for t in test_ts:
        t0 = _time.perf_counter()
        c, o, _ = _test_row(t, raw, comp, cols)
        elapsed = _time.perf_counter() - t0
        t1_times.append(elapsed)
        total_cost += c
        total_oracle += o

    print(f"\n  Summary ({len(test_ts)} rows):")
    print(f"    Total cost:   {total_cost:>12,.0f}")
    print(f"    Oracle cost:  {total_oracle:>12,.0f}")
    print(f"    Total gap:    {total_cost - total_oracle:>+12,.0f}")
    print(f"    Avg time/row: {1000*sum(t1_times)/len(t1_times):>10.1f} ms")

    print(f"\n{'='*120}")
    print("TEST 2: Full historical sweep (3650 rows)")
    print(f"{'='*120}")

    full_cost, full_oracle, full_worst = 0.0, 0.0, 0.0
    hits = 0
    nan_violations = 0
    t2_start = _time.perf_counter()
    for t in range(3650):
        row = raw.iloc[t]
        row_vals = np.array([row[c] for c in cols], dtype=float)
        trades = trading_problem_3(row)

        nan_cols_set = {c for c in cols if np.isnan(row[c])}
        for _, trade in trades.iterrows():
            if trade["col"] not in nan_cols_set:
                nan_violations += 1
        assert trades["qty"].sum() == 100

        col_name = trades["col"].iloc[0]
        true_price = comp.loc[t, col_name]
        cost = trades["qty"].iloc[0] * true_price

        nan_true = [comp.loc[t, c] for c in nan_cols_set]
        oracle = min(nan_true) * 100
        worst = max(nan_true) * 100

        if cost <= oracle + 0.01:
            hits += 1
        full_cost += cost
        full_oracle += oracle
        full_worst += worst

    t2_elapsed = _time.perf_counter() - t2_start
    pct_saved = 100 * (full_worst - full_cost) / (full_worst - full_oracle) if full_worst != full_oracle else 0
    print(f"  Total cost:        {full_cost:>14,.0f}")
    print(f"  Oracle cost:       {full_oracle:>14,.0f}")
    print(f"  Gap vs oracle:     {full_cost - full_oracle:>+14,.0f}")
    print(f"  Oracle-hit rate:   {hits}/3650 ({100*hits/3650:.1f}%)")
    print(f"  Savings vs worst:  {pct_saved:.1f}%")
    print(f"  NaN violations:    {nan_violations}  {'PASS' if nan_violations == 0 else 'FAIL'}")
    print(f"  Time:              {t2_elapsed:.2f}s ({1000*t2_elapsed/3650:.1f} ms/row)")

    print(f"\n{'='*120}")
    print("TEST 3: Synthetic future rows (t > 3649)")
    print(f"{'='*120}")

    syn_cost, syn_oracle = 0.0, 0.0
    t3_times = []
    base_rows = [0, 100, 500, 1000, 2000, 3000, 3649]
    for bt in base_rows:
        t0 = _time.perf_counter()
        c, o = _test_synthetic_row(cols, comp, base_t=bt)
        elapsed = _time.perf_counter() - t0
        t3_times.append(elapsed)
        syn_cost += c
        syn_oracle += o

    print(f"\n  Summary ({len(base_rows)} synthetic rows):")
    print(f"    Total cost:   {syn_cost:>12,.0f}")
    print(f"    Oracle cost:  {syn_oracle:>12,.0f}")
    print(f"    Total gap:    {syn_cost - syn_oracle:>+12,.0f}")
    print(f"    Avg time/row: {sum(t3_times)/len(t3_times):>10.2f} s")
