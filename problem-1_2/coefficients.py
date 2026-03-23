"""
Phase 2 — Coefficient Recovery via NNLS.

Methods:
  A) NNLS on co-observed rows (adaptive pool, handles any k)
  B) NNLS on filled data (all rows where target observed, handles k=10+)
  C) Greedy forward selection with NNLS (fast)

Indices CAN appear as components of other indices. After proving,
distribute through to get the farmer-only decomposition.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls


def fit_convex(y, X):
    """Fit y ≈ X @ c with c >= 0, sum(c) = 1. Returns (coefs, rmse)."""
    n = X.shape[1]
    c0 = np.ones(n) / n
    res = minimize(
        lambda c: np.mean((y - X @ c) ** 2),
        c0, method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints={"type": "eq", "fun": lambda c: c.sum() - 1.0},
        options={"maxiter": 5000, "ftol": 1e-15},
    )
    return res.x, np.sqrt(max(res.fun, 0))


def fit_nnls(y, X):
    """NNLS: y ≈ X @ c, c >= 0. Returns (coefs, rmse, weight_sum)."""
    c, _ = nnls(X, y)
    rmse = float(np.sqrt(np.mean((y - X @ c) ** 2)))
    return c, rmse, float(c.sum())


def _verify(idx_i, active_fi, data, mask_arr, min_rows=10):
    """Refit with convex constraint on co-observed data (original or filled)."""
    mask = mask_arr[:, idx_i].copy()
    for fi in active_fi:
        mask &= mask_arr[:, fi]
    rows = np.where(mask)[0]
    if len(rows) < min_rows:
        return None, float("inf"), len(rows)
    y = data[rows, idx_i]
    X = data[np.ix_(rows, active_fi)]
    coefs, rmse = fit_convex(y, X)
    return coefs, rmse, len(rows)


def _best_verify(idx_i, active_fi, arr, vmask, filled, fmask, min_rows=10):
    """Verify on both original and filled data, return the better result."""
    c1, r1, n1 = _verify(idx_i, active_fi, arr, vmask, min_rows)
    c2, r2, n2 = _verify(idx_i, active_fi, filled, fmask, min_rows)
    if c1 is not None and (c2 is None or r1 <= r2):
        return c1, r1, n1, "orig"
    if c2 is not None:
        return c2, r2, n2, "filled"
    return None, float("inf"), 0, None


# ── Method A: NNLS on co-observed rows ────────────────────────────────────

def nnls_coobs(idx_i, pool, arr, vmask, max_pool=30, min_rows=20):
    """
    Greedily build predictor set from most correlated columns while
    maintaining >= min_rows co-observed. Then fit NNLS in one shot.
    """
    corr_vals = []
    for fi in pool:
        both = vmask[:, idx_i] & vmask[:, fi]
        if both.sum() < 10:
            continue
        r = np.corrcoef(arr[both, idx_i], arr[both, fi])[0, 1]
        if not np.isnan(r):
            corr_vals.append((fi, abs(r)))
    corr_vals.sort(key=lambda x: -x[1])

    selected = []
    co_obs = vmask[:, idx_i].copy()
    for fi, _ in corr_vals:
        candidate_obs = co_obs & vmask[:, fi]
        if candidate_obs.sum() < min_rows:
            continue
        selected.append(fi)
        co_obs = candidate_obs
        if len(selected) >= max_pool:
            break

    if len(selected) < 2:
        return None, None, np.inf, 0.0, 0, 0

    rows = np.where(co_obs)[0]
    y = arr[rows, idx_i]
    X = arr[np.ix_(rows, selected)]
    coefs, rmse, w_sum = fit_nnls(y, X)

    active_fi = [fi for fi, w in zip(selected, coefs) if w > 1e-6]
    active_w = np.array([w for w in coefs if w > 1e-6])
    return active_fi, active_w, rmse, w_sum, len(rows), len(active_fi)


# ── Method B: NNLS on filled data ────────────────────────────────────────

def nnls_filled(idx_i, pool, filled, fmask, top_n=30):
    """
    NNLS using filled data. Remaining missing predictors get column means.
    Uses ALL rows where target is observed.
    """
    col_means = np.nanmean(filled, axis=0)
    rows = np.where(fmask[:, idx_i])[0]
    if len(rows) < 20:
        return None, None, np.inf, 0.0, 0, 0

    corr_vals = []
    for fi in pool:
        both = fmask[rows, fi]
        if both.sum() < 10:
            continue
        vals_y = filled[rows[both], idx_i]
        vals_x = filled[rows[both], fi]
        r = np.corrcoef(vals_y, vals_x)[0, 1]
        if not np.isnan(r):
            corr_vals.append((fi, abs(r)))
    corr_vals.sort(key=lambda x: -x[1])
    screened = [fi for fi, _ in corr_vals[:top_n]]

    if len(screened) < 2:
        return None, None, np.inf, 0.0, 0, 0

    y = filled[rows, idx_i]
    X = np.zeros((len(rows), len(screened)))
    for j, fi in enumerate(screened):
        obs = fmask[rows, fi]
        X[obs, j] = filled[rows[obs], fi]
        X[~obs, j] = col_means[fi]

    coefs, rmse, w_sum = fit_nnls(y, X)

    active_fi = [fi for fi, w in zip(screened, coefs) if w > 1e-6]
    active_w = np.array([w for w in coefs if w > 1e-6])
    return active_fi, active_w, rmse, w_sum, len(rows), len(active_fi)


# ── Method C: Greedy forward selection ────────────────────────────────────

def greedy_nnls(idx_i, pool, arr, vmask, max_k=20, min_rows=20):
    """Greedy forward selection using NNLS (fast)."""
    corr_vals = []
    for fi in pool:
        both = vmask[:, idx_i] & vmask[:, fi]
        if both.sum() < 10:
            continue
        r = np.corrcoef(arr[both, idx_i], arr[both, fi])[0, 1]
        if not np.isnan(r):
            corr_vals.append((fi, abs(r)))
    corr_vals.sort(key=lambda x: -x[1])
    pool_sorted = [fi for fi, _ in corr_vals]

    selected = []
    best_rmse = np.inf
    best_coefs = None

    for step in range(max_k):
        best_col = None
        best_step_rmse = np.inf
        best_step_coefs = None

        base_mask = vmask[:, idx_i].copy()
        for si in selected:
            base_mask &= vmask[:, si]

        for trial in pool_sorted:
            if trial in selected:
                continue
            mask = base_mask & vmask[:, trial]
            if mask.sum() < min_rows:
                continue
            cols_idx = selected + [trial]
            y = arr[mask, idx_i]
            X = arr[np.ix_(mask, cols_idx)]
            c, rmse, _ = fit_nnls(y, X)
            if rmse < best_step_rmse:
                best_step_rmse = rmse
                best_col = trial
                best_step_coefs = c

        if best_col is None:
            break
        selected.append(best_col)
        best_rmse = best_step_rmse
        best_coefs = best_step_coefs
        if best_rmse < 0.001:
            break

    if best_coefs is None:
        return None, None, np.inf
    active_fi = [fi for fi, w in zip(selected, best_coefs) if w > 1e-6]
    active_w = np.array([w for w in best_coefs if w > 1e-6])
    return active_fi, active_w, best_rmse


# ── Constraint propagation ────────────────────────────────────────────────

def fill_from_known(arr, vmask, decompositions, cols, max_passes=10,
                    verbose=True):
    """
    Use proven decompositions to fill missing values deterministically.
    """
    filled = arr.copy()
    fmask = vmask.copy()
    total_filled = 0

    for pass_num in range(max_passes):
        new_fills = 0
        for idx_col, dec in decompositions.items():
            idx_i = cols.index(idx_col)
            fi_list = dec["farmer_idxs"]
            coefs = np.array(dec["coefs"], dtype=float)

            idx_obs = fmask[:, idx_i]
            for row in np.where(idx_obs)[0]:
                farmer_obs = fmask[row, fi_list]
                n_missing = (~farmer_obs).sum()
                if n_missing != 1:
                    continue
                missing_j = int(np.where(~farmer_obs)[0][0])
                missing_fi = fi_list[missing_j]
                w_missing = coefs[missing_j]
                if w_missing < 1e-8:
                    continue
                known_sum = sum(coefs[j] * filled[row, fi_list[j]]
                                for j in range(len(fi_list)) if j != missing_j)
                filled[row, missing_fi] = (filled[row, idx_i] - known_sum) / w_missing
                fmask[row, missing_fi] = True
                new_fills += 1

            for row in np.where(~idx_obs)[0]:
                if fmask[row, fi_list].all():
                    filled[row, idx_i] = sum(coefs[j] * filled[row, fi_list[j]]
                                             for j in range(len(fi_list)))
                    fmask[row, idx_i] = True
                    new_fills += 1

        total_filled += new_fills
        if new_fills == 0:
            break

    if verbose and total_filled > 0:
        obs_before = vmask.sum()
        obs_after = fmask.sum()
        print(f"  [Fill] {total_filled} values filled in {pass_num+1} passes "
              f"({obs_before} → {obs_after} observed)")
    return filled, fmask


# ── Distribute through: expand index components to farmers ────────────────

def distribute_through(decompositions, cols):
    """
    For each proven index, if its decomposition includes another proven index,
    substitute that index's decomposition to get all-farmer weights.
    Repeat until no index components remain.
    """
    proven = {c for c, d in decompositions.items() if d.get("proven")}
    expanded = {}

    for idx_col, dec in decompositions.items():
        if not dec.get("proven"):
            continue
        weights = {}
        for fi, w in zip(dec["farmer_idxs"], dec["coefs"]):
            col_name = cols[fi]
            weights[col_name] = weights.get(col_name, 0.0) + float(w)

        changed = True
        while changed:
            changed = False
            new_weights = {}
            for comp_col, w in weights.items():
                if comp_col in proven and comp_col != idx_col and comp_col in decompositions:
                    sub = decompositions[comp_col]
                    for sfi, sw in zip(sub["farmer_idxs"], sub["coefs"]):
                        sub_name = cols[sfi]
                        new_weights[sub_name] = new_weights.get(sub_name, 0.0) + w * float(sw)
                    changed = True
                else:
                    new_weights[comp_col] = new_weights.get(comp_col, 0.0) + w
            weights = new_weights

        farmer_names = sorted(weights.keys(), key=lambda c: -weights[c])
        farmer_idxs = [cols.index(c) for c in farmer_names if weights[c] > 1e-6]
        farmer_coefs = np.array([weights[c] for c in farmer_names if weights[c] > 1e-6])

        expanded[idx_col] = {
            "farmer_idxs": farmer_idxs,
            "coefs": farmer_coefs,
            "farmer_names": [cols[fi] for fi in farmer_idxs],
        }

    return expanded


# ── Re-regression helpers (for EM loop in pipeline) ───────────────────────

def refit_weights(idx_i, fixed_farmer_idxs, completed):
    """Re-fit convex weights on completed data with a fixed set of farmers."""
    y = completed[:, idx_i]
    X = completed[:, fixed_farmer_idxs]
    coefs, rmse = fit_convex(y, X)
    return list(fixed_farmer_idxs), coefs, rmse


def reregress_on_completed(idx_i, farmer_idxs, completed, cols, top_k=25):
    """Re-regress on SVD-completed data using greedy + NNLS."""
    y = completed[:, idx_i]
    corr_vals = []
    for fi in farmer_idxs:
        r = np.corrcoef(y, completed[:, fi])[0, 1]
        corr_vals.append((fi, abs(r) if not np.isnan(r) else 0.0))
    pool = [fi for fi, _ in sorted(corr_vals, key=lambda x: -x[1])[:top_k]]

    X = completed[:, pool]
    coefs, rmse, w_sum = fit_nnls(y, X)
    active_fi = [fi for fi, w in zip(pool, coefs) if w > 1e-6]
    active_w = np.array([w for w in coefs if w > 1e-6])

    if not active_fi:
        return pool[:3], np.ones(3) / 3, rmse

    y_v = completed[:, idx_i]
    X_v = completed[:, active_fi]
    c_final, rmse_final = fit_convex(y_v, X_v)
    return active_fi, c_final, rmse_final


# ── Orchestrator ─────────────────────────────────────────────────────────

PROVEN_THRESHOLD = 0.01

def _try_one_candidate(idx_col, idx_i, all_others, arr, vmask, filled, fmask,
                       cols, proven_idxs=None, verbose=True):
    """
    Try all methods for a single candidate, return best.
    proven_idxs: set of column indices already proven as indices.
    Methods A-C use full pool; Method D uses farmer-only pool (excludes proven).
    """
    best_method = None
    best_fi = None
    best_c = None
    best_score = np.inf
    if proven_idxs is None:
        proven_idxs = set()

    def _run_method(label, active_fi, active_w=None, raw_rmse=None,
                    raw_sum=None, raw_rows=None, raw_k=None):
        """Verify a method result on both original and filled, update best."""
        nonlocal best_method, best_fi, best_c, best_score
        if active_fi is None:
            return
        vc, vr, vn, vsrc = _best_verify(
            idx_i, active_fi, arr, vmask, filled, fmask)
        if verbose:
            extra = ""
            if raw_rmse is not None:
                extra += f"RMSE={raw_rmse:.6f}"
            if raw_sum is not None:
                extra += f", sum={raw_sum:.4f}"
            if raw_rows is not None:
                extra += f", rows={raw_rows}"
            if raw_k is not None:
                extra = f"pool→{raw_k} active, " + extra
            if extra:
                print(f"  [{label}] {extra}")
            if vc is not None:
                print(f"    → verify({vsrc}): RMSE={vr:.8f}, rows={vn}")
        if vc is not None and vr < best_score:
            best_method, best_fi, best_c, best_score = (
                label, active_fi, vc, vr)

    # Method A: NNLS on co-observed rows (original data)
    a_fi, a_w, a_rmse, a_sum, a_rows, a_k = nnls_coobs(
        idx_i, all_others, arr, vmask)
    _run_method("nnls_coobs", a_fi, a_w, a_rmse, a_sum, a_rows, a_k)

    # Method B: NNLS on filled data
    b_fi, b_w, b_rmse, b_sum, b_rows, b_k = nnls_filled(
        idx_i, all_others, filled, fmask)
    _run_method("nnls_filled", b_fi, b_w, b_rmse, b_sum, b_rows, b_k)

    # Method C: Greedy forward selection (original data, full pool)
    c_fi, c_w, c_rmse = greedy_nnls(idx_i, all_others, arr, vmask)
    if c_fi is not None:
        _run_method("greedy", c_fi, c_w, c_rmse, raw_k=len(c_fi))

    # Method D: Greedy on farmer-only pool (skip proven indices)
    if proven_idxs:
        farmer_pool = [j for j in all_others if j not in proven_idxs]

        d_fi, d_w, d_rmse = greedy_nnls(idx_i, farmer_pool, arr, vmask)
        if d_fi is not None:
            _run_method("greedy_farmers", d_fi, d_w, d_rmse,
                        raw_k=len(d_fi))

        # Greedy on farmer-only pool with FILLED data
        e_fi, e_w, e_rmse = greedy_nnls(idx_i, farmer_pool, filled, fmask)
        if e_fi is not None:
            _run_method("greedy_farmers_filled", e_fi, e_w, e_rmse,
                        raw_k=len(e_fi))

        # NNLS-coobs on farmer-only pool
        f_fi, f_w, f_rmse, f_sum, f_rows, f_k = nnls_coobs(
            idx_i, farmer_pool, arr, vmask)
        _run_method("nnls_coobs_farmers", f_fi, f_w, f_rmse,
                    f_sum, f_rows, f_k)

    if verbose:
        print(f"  >>> Best: {best_method} (RMSE={best_score:.8f})")
        if best_fi is not None:
            for fi, w in sorted(zip(best_fi, best_c), key=lambda x: -x[1]):
                if w > 0.005:
                    print(f"      {cols[fi]}: {w:.6f}")

    return best_method, best_fi, best_c, best_score


def recover_coefficients(data, arr, vmask, cols,
                         index_cols, index_idxs, farmer_cols, farmer_idxs,
                         max_passes=20, accept_threshold=0.6, verbose=True):
    """
    Iterative rank-and-peel coefficient recovery.

    Pool = ALL other columns (indices allowed — distribute through later).
    Fill from ALL accepted decompositions each pass.
    """
    D = len(cols)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  PHASE 2: Iterative Coefficient Recovery (NNLS)")
        print(f"  Rank-and-peel | proven<{PROVEN_THRESHOLD} | tentative<{accept_threshold}")
        print(f"{'='*70}")

    decompositions = {}
    accepted_set = set()
    remaining = list(zip(index_cols, index_idxs))

    for pass_num in range(1, max_passes + 1):
        if not remaining:
            break

        # Fill from ALL accepted decompositions
        if accepted_set:
            all_decs = {c: decompositions[c] for c in accepted_set}
            filled, fmask = fill_from_known(
                arr, vmask, all_decs, cols, verbose=verbose)
        else:
            filled, fmask = arr.copy(), vmask.copy()

        if verbose:
            obs_rate = fmask.sum() / fmask.size
            print(f"\n  ══ Pass {pass_num} ({len(remaining)} candidates, "
                  f"{obs_rate:.1%} observed) ══")

        # Build set of proven column indices for farmer-only pool
        proven_idxs = set()
        for c in accepted_set:
            ci = cols.index(c)
            proven_idxs.add(ci)

        # Score all remaining (pool = ALL other columns, never exclude)
        scores = []
        for idx_col, idx_i in remaining:
            if verbose:
                print(f"\n  --- {idx_col} ---")

            all_others = [j for j in range(D) if j != idx_i]
            method, fi, coefs, score = _try_one_candidate(
                idx_col, idx_i, all_others, arr, vmask,
                filled, fmask, cols,
                proven_idxs=proven_idxs, verbose=verbose)
            scores.append((idx_col, idx_i, method, fi, coefs, score))

        scores.sort(key=lambda x: x[5])

        if verbose:
            print(f"\n  Ranking:")
            for idx_col, _, _, _, _, score in scores:
                tag = ("PROVEN" if score < PROVEN_THRESHOLD else
                       "tentative" if score < accept_threshold else "reject")
                print(f"    {idx_col}: RMSE={score:.6f} [{tag}]")

        # Accept: all proven, or single best tentative
        proven = [(c, i, m, f, co, s) for c, i, m, f, co, s in scores
                  if s < PROVEN_THRESHOLD]
        new_accepted = []

        if proven:
            for idx_col, idx_i, method, fi, coefs, score in proven:
                decompositions[idx_col] = {
                    "method": method,
                    "farmer_idxs": fi if fi is not None else [],
                    "coefs": coefs if coefs is not None else np.array([]),
                    "proven": True,
                }
                accepted_set.add(idx_col)
                new_accepted.append(idx_col)
                if verbose:
                    print(f"  ✓ PROVEN: {idx_col} (RMSE={score:.8f})")
        else:
            best = scores[0]
            idx_col, idx_i, method, fi, coefs, score = best
            if score < accept_threshold and fi is not None:
                decompositions[idx_col] = {
                    "method": method,
                    "farmer_idxs": fi,
                    "coefs": coefs,
                    "proven": False,
                }
                accepted_set.add(idx_col)
                new_accepted.append(idx_col)
                if verbose:
                    print(f"  ~ TENTATIVE: {idx_col} (RMSE={score:.6f})")
            else:
                if verbose:
                    print(f"  No candidate below threshold — stopping.")
                break

        remaining = [(c, i) for c, i, _, _, _, _ in scores
                     if c not in accepted_set]

        if verbose:
            print(f"  Pass {pass_num}: accepted {new_accepted}, "
                  f"{len(remaining)} remaining")

    # Distribute through: expand index components to pure farmers
    expanded = distribute_through(decompositions, cols)
    if verbose and expanded:
        print(f"\n  ── Distributed to farmers ──")
        for idx_col, exp in expanded.items():
            names = [f"{cols[fi]}:{w:.4f}"
                     for fi, w in zip(exp["farmer_idxs"], exp["coefs"])]
            print(f"    {idx_col} = {' + '.join(names)}")

    # Final fill with all accepted decompositions
    if accepted_set:
        all_decs = {c: decompositions[c] for c in accepted_set}
        filled, fmask = fill_from_known(arr, vmask, all_decs, cols,
                                         verbose=verbose)
    else:
        filled, fmask = arr.copy(), vmask.copy()

    confirmed = [c for c in index_cols if c in accepted_set]
    failed = [c for c in index_cols if c not in accepted_set]
    if verbose:
        print(f"\n  ┌─────────────────────────────────────────┐")
        print(f"  │  Accepted indices: {confirmed}")
        if failed:
            print(f"  │  Reclassified as farmers: {failed}")
        obs_rate = fmask.sum() / fmask.size
        print(f"  │  Filled matrix: {obs_rate:.1%} observed")
        print(f"  └─────────────────────────────────────────┘")

    return decompositions, filled, fmask


# ── Serialization ─────────────────────────────────────────────────────────

def save_coefficients(decompositions, index_cols, index_idxs,
                      farmer_cols, farmer_idxs, cols,
                      path="intermediates/coefficients.json"):
    """Serialize decompositions to JSON."""
    import json, os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = {
        "index_cols": index_cols,
        "index_idxs": index_idxs,
        "farmer_cols": farmer_cols,
        "farmer_idxs": farmer_idxs,
        "cols": list(cols),
        "decompositions": {},
    }
    for idx_col, dec in decompositions.items():
        out["decompositions"][idx_col] = {
            "method": dec["method"],
            "farmer_idxs": [int(x) for x in dec["farmer_idxs"]],
            "coefs": [float(x) for x in dec["coefs"]],
        }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved {path}")


def load_coefficients(path="intermediates/coefficients.json"):
    """Load decompositions from JSON."""
    import json
    with open(path) as f:
        raw = json.load(f)
    for dec in raw["decompositions"].values():
        dec["coefs"] = np.array(dec["coefs"])
    return raw


# ── Standalone entry point ────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from candidates import load_candidates

    cand_path = "intermediates/candidates.json"
    try:
        cand = load_candidates(cand_path)
    except FileNotFoundError:
        print(f"ERROR: {cand_path} not found. Run candidates.py first.")
        sys.exit(1)

    df = pd.read_csv("../data/limestone_data_challenge_2026.data.csv")
    cols = [c for c in df.columns if c.startswith("col_")]
    data = df[cols].copy()
    arr = data.values.astype(np.float64)
    vmask = ~np.isnan(arr)

    print(f"Data: {data.shape[0]}x{data.shape[1]}, "
          f"NaN rate: {data.isna().mean().mean():.3f}")
    print(f"Loaded candidates: {cand['index_cols']}")

    decompositions, filled, fmask = recover_coefficients(
        data, arr, vmask, cols,
        cand["index_cols"], cand["index_idxs"],
        cand["farmer_cols"], cand["farmer_idxs"],
    )

    accepted_cols = list(decompositions.keys())
    accepted_idxs = [cols.index(c) for c in accepted_cols]
    farmer_idx_set = set(range(len(cols))) - set(accepted_idxs)
    farmer_idxs_final = sorted(farmer_idx_set)
    farmer_cols_final = [cols[i] for i in farmer_idxs_final]

    save_coefficients(
        decompositions,
        accepted_cols, accepted_idxs,
        farmer_cols_final, farmer_idxs_final,
        cols,
    )
