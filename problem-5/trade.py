"""
Problem 5 — Limit-order buying with optimal bid pricing.

Single self-contained file. All data loaded from intermediates:
  - Decompositions from problem-1_2/intermediates/coefficients.json
  - Sigma from problem-5/intermediates/sigma.json
  - Classification built on the fly from raw data + decompositions

Score = Σ(qty_i × (median_px - px_i) × I{px_i >= true_px_i})
"""

import os, json
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm

_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(_DIR, "..", "data",
                         "limestone_data_challenge_2026.data.csv")
COMPLETED_PATH = os.path.join(_DIR, "..", "answers",
                              "problem2_answer.csv")
COEFF_JSON = os.path.join(_DIR, "..", "problem-1_2",
                          "intermediates", "coefficients.json")
ANSWER_1B = os.path.join(_DIR, "..", "answers",
                         "problem1b_answer.csv")
SIGMA_JSON = os.path.join(_DIR, "intermediates", "sigma.json")

MAX_HISTORICAL_T = 3649
KNN_K = 20
PROJECTION_RANK = 12
KNN_WEIGHT = 0.5

SIGMA_ALGEBRAIC = 0.02
SIGMA_P2_CAP = 3.0
SIGMA_OOT_INFLATE = 1.5
DEFAULT_SIGMA = 5.0

# ── Lazy-loaded internal cache ────────────────────────────────────────────────

_state = None


def _load_decompositions(cols):
    """Load raw decompositions (may include index-to-index deps)."""
    if os.path.exists(COEFF_JSON):
        with open(COEFF_JSON) as f:
            raw = json.load(f)
        raw_dec = raw.get("decompositions", raw)
        decompositions = {}
        for idx_col, dec in raw_dec.items():
            fi_idxs = [int(i) for i in dec["farmer_idxs"]]
            decompositions[idx_col] = {
                "farmer_idxs": fi_idxs,
                "farmer_names": [cols[i] for i in fi_idxs],
                "coefs": [float(c) for c in dec["coefs"]],
            }
        return decompositions

    df = pd.read_csv(ANSWER_1B)
    decompositions = {}
    for idx_col, grp in df.groupby("index_col"):
        fi_names = grp["constituent_col"].tolist()
        fi_idxs = [cols.index(c) for c in fi_names]
        decompositions[idx_col] = {
            "farmer_idxs": fi_idxs,
            "farmer_names": fi_names,
            "coefs": grp["coef"].tolist(),
        }
    return decompositions


def _topo_order(decompositions):
    """Topological sort: independent indices first, then dependents."""
    index_set = set(decompositions.keys())
    order = []
    remaining = set(index_set)
    while remaining:
        ready = [c for c in sorted(remaining)
                 if not any(dep in remaining and dep != c
                            for dep in decompositions[c]["farmer_names"]
                            if dep in index_set)]
        if not ready:
            order.extend(sorted(remaining))
            break
        order.extend(ready)
        remaining -= set(ready)
    return order


def _load_sigma():
    """Load per-column sigma from intermediates."""
    if os.path.exists(SIGMA_JSON):
        with open(SIGMA_JSON) as f:
            return json.load(f)
    return {}


def _build_classes(original_arr, vmask, decompositions, cols):
    """Build classification: 0=given, 1=deterministic, 2=imputed."""
    n_rows, n_cols = original_arr.shape
    classes = np.full((n_rows, n_cols), 2, dtype=np.int8)
    classes[vmask] = 0

    filled = original_arr.copy()
    fmask = vmask.copy()

    for _ in range(20):
        new_fills = 0
        for idx_col, dec in decompositions.items():
            idx_i = cols.index(idx_col)
            fi_list = dec["farmer_idxs"]
            coefs = np.array(dec["coefs"], dtype=float)

            for row in np.where(fmask[:, idx_i])[0]:
                farmer_obs = fmask[row, fi_list]
                if (~farmer_obs).sum() != 1:
                    continue
                j_miss = int(np.where(~farmer_obs)[0][0])
                fi_miss = fi_list[j_miss]
                w = coefs[j_miss]
                if w < 1e-8:
                    continue
                known_sum = sum(
                    coefs[j] * filled[row, fi_list[j]]
                    for j in range(len(fi_list)) if j != j_miss
                )
                filled[row, fi_miss] = (filled[row, idx_i] - known_sum) / w
                fmask[row, fi_miss] = True
                classes[row, fi_miss] = 1
                new_fills += 1

            for row in np.where(~fmask[:, idx_i])[0]:
                if fmask[row, fi_list].all():
                    filled[row, idx_i] = sum(
                        coefs[j] * filled[row, fi_list[j]]
                        for j in range(len(fi_list))
                    )
                    fmask[row, idx_i] = True
                    classes[row, idx_i] = 1
                    new_fills += 1

        if new_fills == 0:
            break

    return classes, filled


def _load_all():
    """Load everything needed, cache in module-level _state."""
    global _state
    if _state is not None:
        return _state

    comp_df = pd.read_csv(COMPLETED_PATH)
    cols = [c for c in comp_df.columns if c != "time"]
    completed = comp_df[cols].values.astype(float)

    decompositions = _load_decompositions(cols)
    decomp_order = _topo_order(decompositions)
    sigma = _load_sigma()

    raw_df = pd.read_csv(DATA_PATH)
    raw_arr = raw_df[cols].values.astype(float)
    raw_vmask = ~np.isnan(raw_arr)
    classes_full, _ = _build_classes(raw_arr, raw_vmask, decompositions, cols)

    _state = {
        "cols": cols,
        "completed": completed,
        "decompositions": decompositions,
        "decomp_order": decomp_order,
        "sigma": sigma,
        "classes_full": classes_full,
    }
    return _state


# ── Core helpers ──────────────────────────────────────────────────────────────

def _try_algebraic(row_prices, cols, decompositions, decomp_order):
    """Compute index prices from decompositions in topological order."""
    known = {}
    for c in cols:
        v = row_prices.get(c, np.nan)
        if not np.isnan(v):
            known[c] = v

    results = {}
    for idx_col in decomp_order:
        if idx_col in known:
            continue
        dec = decompositions[idx_col]
        if all(c in known for c in dec["farmer_names"]):
            price = sum(w * known[c]
                        for c, w in zip(dec["farmer_names"], dec["coefs"]))
            known[idx_col] = price
            results[idx_col] = price

    return results


def _optimal_bid(p_hat, sigma, median_est):
    """Find bid maximizing E[profit/kg] = (median - bid) * Phi((bid - p_hat) / sigma)."""
    if sigma < 0.1:
        return round(p_hat + 0.01, 2)

    if p_hat >= median_est:
        return round(p_hat + 0.01, 2)

    def neg_expected_profit(bid):
        fill_prob = norm.cdf((bid - p_hat) / sigma)
        return -(median_est - bid) * fill_prob

    lo = max(p_hat - 2 * sigma, 0.01)
    hi = median_est - 0.01

    if lo >= hi:
        return round(p_hat + 0.01, 2)

    result = minimize_scalar(neg_expected_profit, bounds=(lo, hi),
                             method="bounded")
    return round(result.x, 2)


# ── Public trading function ───────────────────────────────────────────────────

def trading_problem_5(row, _cache=None):
    """
    Problem 5 entry point.

    Parameters
    ----------
    row : pd.Series
        One day's bulletin with 'time' and col_00..col_52 (some NaN).
    _cache : dict or None
        Optional preloaded data for speed. Keys: 'completed', 'cols',
        'classes_full', 'decompositions', 'decomp_order', 'sigma'.
        If None, loads from disk (cached after first call).

    Returns
    -------
    pd.DataFrame with columns: order_col, px, qty
    """
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    if _cache is not None:
        cols = _cache["cols"]
        completed = _cache["completed"]
        decompositions = _cache.get("decompositions", {})
        decomp_order = _cache.get("decomp_order", [])
        sigma_dict = _cache.get("sigma", {})
        classes_full = _cache.get("classes_full", None)
    else:
        s = _load_all()
        cols = s["cols"]
        completed = s["completed"]
        decompositions = s["decompositions"]
        decomp_order = s["decomp_order"]
        sigma_dict = s["sigma"]
        classes_full = s["classes_full"]

    t = int(row.get("time", -1))
    row_vals = np.array([row.get(c, np.nan) for c in cols], dtype=float)
    nan_mask = np.isnan(row_vals)
    nan_idxs = np.where(nan_mask)[0]

    if len(nan_idxs) == 0:
        return pd.DataFrame({"order_col": [cols[0]], "px": [0.01], "qty": [0]})

    predicted = np.full(len(cols), np.nan)
    col_sigma = np.full(len(cols), np.nan)

    obs_mask = ~nan_mask
    predicted[obs_mask] = row_vals[obs_mask]
    col_sigma[obs_mask] = 0.0

    # Path A: historical lookup
    if 0 <= t <= MAX_HISTORICAL_T:
        for j in nan_idxs:
            predicted[j] = completed[t, j]
            if classes_full is not None and classes_full[t, j] == 1:
                col_sigma[j] = SIGMA_ALGEBRAIC
            else:
                col_sigma[j] = min(
                    sigma_dict.get(cols[j], DEFAULT_SIGMA), SIGMA_P2_CAP)

    # Path B: out-of-time prediction
    else:
        row_prices = {cols[j]: row_vals[j] for j in range(len(cols))
                      if not np.isnan(row_vals[j])}
        alg_fills = _try_algebraic(row_prices, cols, decompositions,
                                   decomp_order)

        for col_name, price in alg_fills.items():
            j = cols.index(col_name)
            if nan_mask[j]:
                predicted[j] = price
                col_sigma[j] = SIGMA_ALGEBRAIC

        still_missing = np.where(np.isnan(predicted))[0]
        if len(still_missing) > 0:
            known_idxs = np.where(~np.isnan(predicted))[0]
            known_vals = predicted[known_idxs]

            dists = np.sum((completed[:, known_idxs] - known_vals) ** 2,
                           axis=1)
            nn_idxs = np.argsort(dists)[:KNN_K]
            w = 1.0 / (np.sqrt(dists[nn_idxs]) + 1e-8)
            w /= w.sum()
            knn_pred = completed[nn_idxs].T @ w

            col_means = completed.mean(axis=0)
            centered = completed - col_means
            _, s, Vt = np.linalg.svd(centered, full_matrices=False)
            Vt_r = Vt[:PROJECTION_RANK]
            V_obs = Vt_r[:, known_idxs].T
            centered_known = known_vals - col_means[known_idxs]
            alpha, *_ = np.linalg.lstsq(V_obs, centered_known, rcond=None)
            lr_pred = col_means + Vt_r.T @ alpha

            blended = KNN_WEIGHT * knn_pred + (1 - KNN_WEIGHT) * lr_pred

            for j in still_missing:
                predicted[j] = blended[j]
                col_sigma[j] = sigma_dict.get(cols[j], DEFAULT_SIGMA) \
                               * SIGMA_OOT_INFLATE

    median_est = float(np.median(predicted))

    bids = []
    for j in nan_idxs:
        p_hat = predicted[j]
        sigma = col_sigma[j]
        bid = _optimal_bid(p_hat, sigma, median_est)

        if sigma < 0.1:
            fill_prob = 1.0
        else:
            fill_prob = norm.cdf((bid - p_hat) / sigma)

        exp_profit = (median_est - bid) * fill_prob
        bids.append((j, cols[j], bid, exp_profit, fill_prob))

    profitable = [(j, col, bid, ep, fp)
                  for j, col, bid, ep, fp in bids if ep > 0]

    if not profitable:
        best = max(bids, key=lambda x: x[3])
        return pd.DataFrame({
            "order_col": [best[1]], "px": [best[2]], "qty": [0]
        })

    profitable.sort(key=lambda x: -x[3])
    is_historical = (0 <= t <= MAX_HISTORICAL_T)

    if is_historical:
        top = profitable[0]
        return pd.DataFrame({
            "order_col": [top[1]], "px": [top[2]], "qty": [100]
        })
    else:
        top_n = profitable[:min(3, len(profitable))]
        total_ep = sum(x[3] for x in top_n)

        orders = []
        remaining = 100
        for i, (j, col, bid, ep, fp) in enumerate(top_n):
            if i == len(top_n) - 1:
                qty = remaining
            else:
                qty = max(1, int(round(100 * ep / total_ep)))
                qty = min(qty, remaining)
            remaining -= qty
            if qty > 0:
                orders.append({"order_col": col, "px": bid, "qty": qty})

        if not orders:
            top = profitable[0]
            orders = [{"order_col": top[1], "px": top[2], "qty": 100}]

        return pd.DataFrame(orders)


# ── Backtest ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time as _time

    state = _load_all()
    sigma_dict = state["sigma"]

    raw_df = pd.read_csv(DATA_PATH)
    cols = state["cols"]
    completed = state["completed"]

    cache = state

    rng = np.random.default_rng(42)
    test_rows = sorted(rng.choice(3650, 500, replace=False))

    print(f"{'='*100}")
    print(f"BACKTEST: Problem 5 on {len(test_rows)} training rows")
    print(f"{'='*100}")

    total_score = 0.0
    total_fills = 0
    total_orders = 0
    total_qty_filled = 0
    col_selected = {}
    t0 = _time.time()

    for i, t in enumerate(test_rows):
        row = raw_df.iloc[t]
        row_vals = np.array([row[c] for c in cols], dtype=float)
        nan_idxs = np.where(np.isnan(row_vals))[0]

        result = trading_problem_5(row, _cache=cache)

        true_prices = completed[t]
        true_median = float(np.median(true_prices))

        day_score = 0.0
        day_fills = 0
        for _, order in result.iterrows():
            col_name = order["order_col"]
            bid = order["px"]
            qty = int(order["qty"])
            j = cols.index(col_name)

            assert col_name in [cols[k] for k in nan_idxs], \
                f"t={t}: ordered non-NaN column {col_name}"

            total_orders += 1
            col_selected[col_name] = col_selected.get(col_name, 0) + 1

            true_price = true_prices[j]
            if bid >= true_price:
                profit = qty * (true_median - bid)
                day_score += profit
                day_fills += 1
                total_qty_filled += qty

        total_fills += day_fills
        total_score += day_score

        if (i + 1) % 100 == 0:
            elapsed = _time.time() - t0
            print(f"  [{i+1:3d}/{len(test_rows)}] "
                  f"score={total_score:>10.0f}  "
                  f"fills={total_fills}/{total_orders}  "
                  f"time={elapsed:.1f}s")

    elapsed = _time.time() - t0

    fill_rate = total_fills / max(total_orders, 1) * 100
    avg_profit = total_score / len(test_rows)

    print(f"\n{'='*100}")
    print(f"RESULTS")
    print(f"{'='*100}")
    print(f"  Total score (profit):    {total_score:>12.0f}")
    print(f"  Avg profit / day:        {avg_profit:>12.1f}")
    print(f"  Fill rate:               {total_fills}/{total_orders} "
          f"({fill_rate:.1f}%)")
    print(f"  Total qty filled:        {total_qty_filled}")
    print(f"  Avg time / row:          {1000*elapsed/len(test_rows):.0f} ms")
    print(f"  Total time:              {elapsed:.1f}s")

    print(f"\n  Top 10 selected columns:")
    for col, cnt in sorted(col_selected.items(), key=lambda x: -x[1])[:10]:
        s = sigma_dict.get(col, DEFAULT_SIGMA)
        print(f"    {col}: {cnt} times (σ={s:.2f})")
