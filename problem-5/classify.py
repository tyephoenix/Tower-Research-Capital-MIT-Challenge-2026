"""
Classify every cell in the completed data matrix into confidence tiers:
  0 = given        (non-NaN in the original bulletin)
  1 = deterministic (NaN originally, but exactly recoverable from index decompositions)
  2 = imputed       (NaN originally, filled by SVD / EM)

Priority: given > deterministic > imputed.

Usage:
  python classify.py                   # prints summary, saves classification
  from classify import build_classes   # use programmatically
"""

import os, sys, json
import numpy as np
import pandas as pd

DATA_PATH = "../data/limestone_data_challenge_2026.data.csv"
COMPLETED_PATH = "../answers/problem2_answer.csv"
COEFF_INTERMEDIATE = "../problem-1_2/intermediates/coefficients.json"
ANSWER_1B = "../answers/problem1b_answer.csv"


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_decompositions_from_1b(path, cols):
    """Parse problem1b_answer.csv into the dict format used internally."""
    df = pd.read_csv(path)
    decompositions = {}
    for idx_col, grp in df.groupby("index_col"):
        fi_names = grp["constituent_col"].tolist()
        coefs = grp["coef"].tolist()
        fi_idxs = [cols.index(c) for c in fi_names]
        decompositions[idx_col] = {
            "farmer_idxs": fi_idxs,
            "farmer_names": fi_names,
            "coefs": coefs,
        }
    return decompositions


def _load_decompositions(cols):
    """Load decompositions from intermediates (preferred) or answer file."""
    if os.path.exists(COEFF_INTERMEDIATE):
        with open(COEFF_INTERMEDIATE) as f:
            raw = json.load(f)
        raw_dec = raw.get("decompositions", raw)
        decompositions = {}
        for idx_col, dec in raw_dec.items():
            fi_idxs = [int(i) for i in dec["farmer_idxs"]]
            fi_names = [cols[i] for i in fi_idxs]
            decompositions[idx_col] = {
                "farmer_idxs": fi_idxs,
                "farmer_names": fi_names,
                "coefs": [float(c) for c in dec["coefs"]],
            }
        return decompositions
    if os.path.exists(ANSWER_1B):
        return _load_decompositions_from_1b(ANSWER_1B, cols)
    raise FileNotFoundError(
        "Need either intermediates/coefficients.json or answers/problem1b_answer.csv"
    )


# ── core ─────────────────────────────────────────────────────────────────────

def build_classes(original_arr, vmask, decompositions, cols):
    """
    Build a classification matrix (same shape as data).

    Returns
    -------
    classes : ndarray of int, shape (n_rows, n_cols)
        0 = given, 1 = deterministic, 2 = imputed
    filled  : ndarray, the deterministically-filled data
    """
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

            # index observed → solve for 1 missing farmer
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

            # all farmers observed → compute index
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


def classify_row(row_vals, decompositions, cols):
    """
    Classify a single row (1-d array, may contain NaN).

    Returns
    -------
    classes : ndarray of int8, length n_cols
        0 = given, 1 = deterministic, 2 = imputed
    filled_row : ndarray, with deterministic fills applied
    """
    n = len(cols)
    classes = np.full(n, 2, dtype=np.int8)
    filled = row_vals.copy()
    obs = ~np.isnan(row_vals)
    classes[obs] = 0

    changed = True
    while changed:
        changed = False
        for idx_col, dec in decompositions.items():
            idx_i = cols.index(idx_col)
            fi_list = dec["farmer_idxs"]
            coefs = np.array(dec["coefs"], dtype=float)

            if classes[idx_i] <= 1:
                # index already known → solve for 1 missing farmer
                farmer_known = np.array([classes[fi] <= 1 for fi in fi_list])
                if (~farmer_known).sum() == 1:
                    j_miss = int(np.where(~farmer_known)[0][0])
                    fi_miss = fi_list[j_miss]
                    w = coefs[j_miss]
                    if w >= 1e-8:
                        known_sum = sum(
                            coefs[j] * filled[fi_list[j]]
                            for j in range(len(fi_list)) if j != j_miss
                        )
                        filled[fi_miss] = (filled[idx_i] - known_sum) / w
                        classes[fi_miss] = 1
                        changed = True
            else:
                # index unknown → compute if all farmers known
                if all(classes[fi] <= 1 for fi in fi_list):
                    filled[idx_i] = sum(
                        coefs[j] * filled[fi_list[j]]
                        for j in range(len(fi_list))
                    )
                    classes[idx_i] = 1
                    changed = True

    return classes, filled


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    raw = pd.read_csv(DATA_PATH)
    cols = [c for c in raw.columns if c != "time"]
    arr = raw[cols].values.astype(float)
    vmask = ~np.isnan(arr)

    decompositions = _load_decompositions(cols)

    classes, filled = build_classes(arr, vmask, decompositions, cols)

    n_total = classes.size
    n_given = (classes == 0).sum()
    n_det = (classes == 1).sum()
    n_imp = (classes == 2).sum()

    print(f"Total cells:   {n_total:>10,}")
    print(f"  Given:       {n_given:>10,}  ({100*n_given/n_total:.1f}%)")
    print(f"  Deterministic: {n_det:>8,}  ({100*n_det/n_total:.1f}%)")
    print(f"  Imputed:     {n_imp:>10,}  ({100*n_imp/n_total:.1f}%)")

    out = pd.DataFrame(classes, columns=cols)
    out.insert(0, "time", raw["time"])
    os.makedirs("intermediates", exist_ok=True)
    out.to_csv("intermediates/classification.csv", index=False)
    print(f"\nSaved → intermediates/classification.csv")
