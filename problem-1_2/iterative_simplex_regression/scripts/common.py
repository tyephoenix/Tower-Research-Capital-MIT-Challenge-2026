from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def default_paths() -> tuple[Path, Path]:
    method_dir = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / 'data' / 'limestone_data_challenge_2026.data.csv', method_dir / 'results'


def parse_args(default_csv: Path | None = None, default_out: Path | None = None):
    if default_csv is None or default_out is None:
        default_csv2, default_out2 = default_paths()
        default_csv = default_csv or default_csv2
        default_out = default_out or default_out2
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=Path, default=default_csv)
    ap.add_argument('--outdir', type=Path, default=default_out)
    return ap.parse_args()


def load_matrix(csv_path: Path):
    df = pd.read_csv(csv_path)
    col_names = [c for c in df.columns if c != 'time']
    M = df[col_names].values.astype(np.float64)
    mask = ~np.isnan(M)
    ci = {name: i for i, name in enumerate(col_names)}
    return df, col_names, M, mask, ci


def fit_convex(y: np.ndarray, X: np.ndarray, equality: bool = False,
               maxiter: int = 2000, ftol: float = 1e-15):
    n, p = X.shape
    if n < p + 2:
        return None, np.inf, None

    def obj(w):
        r = y - X @ w
        return np.dot(r, r) / n

    def jac(w):
        return -2.0 * (X.T @ (y - X @ w)) / n

    bounds = [(0, None)] * p
    if equality:
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        w0 = np.ones(p) / p
    else:
        constraints = [{"type": "ineq", "fun": lambda w: 1.0 - np.sum(w)}]
        w0 = np.ones(p) / (2 * p)

    result = minimize(
        obj,
        w0,
        jac=jac,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': maxiter, 'ftol': ftol},
    )
    return result.x, float(np.sqrt(np.mean((y - X @ result.x) ** 2))), result


@dataclass
class StepwiseResult:
    col: str
    step: int
    rmse: float
    r2: float
    n_rows: int
    w_sum: float
    weights: Dict[str, float]


KNOWN_DECOMPS_EXACT: list[tuple[str, dict[str, float]]] = [
    ("col_42", {"col_26": 0.5955364658726565, "col_28": 0.40446096451682156}),
    ("col_50", {"col_42": 0.586385, "col_32": 0.224753, "col_26": 0.188861}),
    ("col_50", {"col_26": 0.586385 * 0.5955364658726565 + 0.188861,
                  "col_28": 0.586385 * 0.40446096451682156,
                  "col_32": 0.224753}),
    ("col_11", {"col_28": 0.342417, "col_42": 0.307571, "col_20": 0.212762,
                 "col_07": 0.074628, "col_22": 0.062622}),
    ("col_11", {"col_28": 0.342417 + 0.307571 * 0.40446096451682156,
                 "col_26": 0.307571 * 0.5955364658726565,
                 "col_20": 0.212762, "col_07": 0.074628, "col_22": 0.062622}),
    ("col_48", {"col_05": 0.5453628709082098, "col_45": 0.13482802176108735,
                 "col_23": 0.12688200806246352, "col_04": 0.09683806773729109,
                 "col_26": 0.09608857884372685}),
    ("col_46", {"col_15": 0.2950, "col_34": 0.2370, "col_32": 0.1120,
                 "col_09": 0.1220, "col_23": 0.0840, "col_05": 0.0800,
                 "col_37": 0.0310, "col_20": 0.0260, "col_04": 0.0120}),
    ("col_30", {"col_26": 0.2190, "col_19": 0.2140, "col_34": 0.1480,
                 "col_40": 0.1460, "col_09": 0.1260, "col_45": 0.0790,
                 "col_24": 0.0670}),
]

CONFIRMED_INDICES = {"col_11", "col_42", "col_48", "col_50", "col_30", "col_46"}
CONFIRMED_FARMERS = {
    "col_26", "col_28", "col_32", "col_20", "col_07", "col_22",
    "col_05", "col_45", "col_23", "col_04", "col_19", "col_34",
    "col_40", "col_09", "col_24", "col_15", "col_37",
}
LIKELY_FARMERS = {"col_31", "col_52", "col_49", "col_12"}
ALL_KNOWN_FARMERS = sorted(CONFIRMED_FARMERS | LIKELY_FARMERS)
UNCLASSIFIED = sorted({f'col_{i:02d}' for i in range(53)} - CONFIRMED_INDICES - CONFIRMED_FARMERS - LIKELY_FARMERS)


CANDIDATE_STABILITY_SETS = {
    "col_11": [["col_28", "col_42", "col_20", "col_07", "col_22"]],
    "col_50": [["col_42", "col_32", "col_26"]],
    "col_48": [
        ["col_05", "col_23", "col_45", "col_04", "col_42"],
        ["col_05", "col_23", "col_45", "col_04", "col_42", "col_40"],
        ["col_05", "col_23", "col_45", "col_04"],
        ["col_05", "col_23"],
    ],
    "col_46": [
        ["col_15", "col_34", "col_32", "col_30", "col_09", "col_20"],
        ["col_15", "col_34", "col_32", "col_30", "col_09", "col_20", "col_48"],
        ["col_15", "col_34", "col_32", "col_30"],
        ["col_15", "col_34"],
    ],
    "col_30": [
        ["col_46", "col_26", "col_19", "col_27", "col_24", "col_34"],
        ["col_46", "col_26", "col_19", "col_27"],
        ["col_46", "col_26"],
    ],
    "col_42": [
        ["col_11", "col_26"],
        ["col_11", "col_26", "col_50"],
        ["col_28", "col_20", "col_07", "col_22", "col_26"],
    ],
    "col_32": [
        ["col_50"],
        ["col_50", "col_42", "col_26"],
    ],
}


def fill_from_decomposition(M: np.ndarray, mask: np.ndarray, filled: np.ndarray,
                            ci: dict[str, int], index_col: str, weights: dict[str, float]) -> int:
    idx_i = ci[index_col]
    farmer_cols = list(weights.keys())
    farmer_is = [ci[c] for c in farmer_cols]
    farmer_ws = [weights[c] for c in farmer_cols]
    n_farmers = len(farmer_cols)
    new_fills = 0
    n_rows = M.shape[0]

    for t in range(n_rows):
        idx_obs = mask[t, idx_i]
        farmer_obs = [mask[t, fi] for fi in farmer_is]
        n_farmer_obs = sum(farmer_obs)

        if (not idx_obs) and n_farmer_obs == n_farmers:
            M[t, idx_i] = sum(farmer_ws[k] * M[t, farmer_is[k]] for k in range(n_farmers))
            mask[t, idx_i] = True
            filled[t, idx_i] = True
            new_fills += 1
        elif idx_obs and n_farmer_obs == n_farmers - 1:
            missing_k = next((k for k in range(n_farmers) if not farmer_obs[k]), None)
            if missing_k is not None and farmer_ws[missing_k] > 1e-12:
                other_sum = sum(farmer_ws[k] * M[t, farmer_is[k]] for k in range(n_farmers) if k != missing_k)
                M[t, farmer_is[missing_k]] = (M[t, idx_i] - other_sum) / farmer_ws[missing_k]
                mask[t, farmer_is[missing_k]] = True
                filled[t, farmer_is[missing_k]] = True
                new_fills += 1

    return new_fills


def fill_all(M: np.ndarray, mask: np.ndarray, filled: np.ndarray,
             ci: dict[str, int], decomps: Sequence[tuple[str, dict[str, float]]],
             verbose: bool = True) -> int:
    total = 0
    iteration = 0
    while True:
        iteration += 1
        round_fills = 0
        for index_col, weights in decomps:
            round_fills += fill_from_decomposition(M, mask, filled, ci, index_col, weights)
        total += round_fills
        if round_fills == 0:
            break
        if verbose:
            print(f"  Fill iteration {iteration}: {round_fills} new fills (total obs: {mask.sum()}, {mask.mean()*100:.2f}%)")
    return total


def forward_stepwise(col: str, M: np.ndarray, mask: np.ndarray, ci: dict[str, int],
                     predictor_names: Sequence[str], equality: bool = True,
                     max_steps: int = 12, min_rows: int = 25, verbose: bool = False):
    ti = ci[col]
    target_rows = np.where(mask[:, ti])[0]
    y_var = float(np.var(M[target_rows, ti])) if len(target_rows) else 0.0
    selected_is: list[int] = []
    selected_names: list[str] = []
    history: list[StepwiseResult] = []

    for step in range(max_steps):
        best_rmse = np.inf
        best_j = None
        best_jname = None
        for fname in predictor_names:
            j = ci[fname]
            if j in selected_is or j == ti:
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
            w, rmse, _ = fit_convex(y, X, equality=equality)
            if w is not None and rmse < best_rmse:
                best_rmse = rmse
                best_j = j
                best_jname = fname
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
        w_final, rmse_final, _ = fit_convex(y, X, equality=equality)
        r2 = 1 - (rmse_final ** 2) / y_var if y_var > 0 else 0.0
        weights = {selected_names[k]: float(w_final[k]) for k in range(len(selected_is)) if w_final[k] > 1e-6}
        res = StepwiseResult(col=col, step=step + 1, rmse=float(rmse_final), r2=float(r2),
                             n_rows=len(rows), w_sum=float(np.sum(w_final)), weights=weights)
        history.append(res)
        if verbose:
            wstr = ', '.join(f'{selected_names[k]}={w_final[k]:.4f}' for k in range(len(selected_is)) if w_final[k] > 1e-4)
            print(f"  {col} step {step+1}: +{best_jname:>8}, RMSE={rmse_final:.6f}, R²={r2:.8f}, n={len(rows):>4}, Σw={np.sum(w_final):.6f}  [{wstr}]")
        if rmse_final < 0.01:
            break
        if step > 0 and history[-2].rmse - rmse_final < 0.005:
            break
    return history


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)
