"""
Per-column periodic trend model.

Finds dominant period via Lomb-Scargle periodogram (O(n log n)),
then fits harmonics via OLS. No grid search needed.
"""

import numpy as np
from scipy.signal import lombscargle

N_HARM = 5


def _find_periods(t, y, n_periods=2):
    """Find top dominant periods using Lomb-Scargle periodogram."""
    # Detrend
    slope, intercept = np.polyfit(t, y, 1)
    y_detrend = y - (slope * t + intercept)

    # Candidate angular frequencies
    min_period, max_period = 30.0, 300.0
    freqs = np.linspace(2 * np.pi / max_period, 2 * np.pi / min_period, 2000)

    power = lombscargle(t, y_detrend, freqs, normalize=True)

    periods_found = []
    for _ in range(n_periods):
        if len(power) == 0:
            break
        idx = np.argmax(power)
        P = 2 * np.pi / freqs[idx]
        periods_found.append(P)
        # Zero out around this peak and its harmonics
        for mult in [0.5, 1.0, 2.0]:
            mask = np.abs(freqs - mult * freqs[idx]) < freqs[idx] * 0.15
            power[mask] = 0.0

    return periods_found


def _build_basis(t, P, n_harm=N_HARM):
    """Design matrix for one period: sin/cos pairs + trend + intercept."""
    n = len(t)
    X = np.empty((n, 2 * n_harm + 2))
    for k in range(1, n_harm + 1):
        arg = 2 * np.pi * k * t / P
        X[:, 2*(k-1)] = np.sin(arg)
        X[:, 2*(k-1)+1] = np.cos(arg)
    X[:, -2] = t
    X[:, -1] = 1.0
    return X


def fit_column(t_obs, y_obs):
    """
    Fit harmonic trend model using Lomb-Scargle + OLS.
    Returns dict: P1, P2, coefs, n_harm, rmse
    """
    n = len(t_obs)
    if n < 10:
        mu = np.mean(y_obs) if n > 0 else 160.0
        p = 2 * N_HARM + 2
        c = np.zeros(p)
        c[-1] = mu
        return {"P1": 79.0, "P2": None, "coefs": c,
                "n_harm": N_HARM, "rmse": float("inf")}

    t = t_obs.astype(np.float64)
    y = y_obs.astype(np.float64)

    periods = _find_periods(t, y, n_periods=2)
    P1 = periods[0] if periods else 79.0

    # Single-period fit
    X1 = _build_basis(t, P1)
    coefs1, _, _, _ = np.linalg.lstsq(X1, y, rcond=None)
    sse1 = np.sum((y - X1 @ coefs1) ** 2)

    # Try double-period if we found a second
    P2 = None
    coefs_final = coefs1
    sse_final = sse1
    if len(periods) >= 2:
        P2_cand = periods[1]
        X_joint = np.column_stack([
            _build_basis(t, P1)[:, :-2],
            _build_basis(t, P2_cand),
        ])
        coefs_joint, _, _, _ = np.linalg.lstsq(X_joint, y, rcond=None)
        sse_joint = np.sum((y - X_joint @ coefs_joint) ** 2)

        p1_params = 2 * N_HARM + 2
        p2_params = 4 * N_HARM + 2
        bic1 = n * np.log(sse1 / n + 1e-15) + p1_params * np.log(n)
        bic2 = n * np.log(sse_joint / n + 1e-15) + p2_params * np.log(n)

        if bic2 < bic1 - 1.0:
            P2 = P2_cand
            coefs_final = coefs_joint
            sse_final = sse_joint

    rmse = np.sqrt(sse_final / n)
    return {"P1": float(P1), "P2": float(P2) if P2 else None,
            "coefs": coefs_final, "n_harm": N_HARM, "rmse": rmse}


def trend_model(t, params):
    """Evaluate fitted model at times t."""
    P1 = params["P1"]
    P2 = params["P2"]
    coefs = params["coefs"]
    nh = params["n_harm"]

    if P2 is None:
        X = _build_basis(t, P1, nh)
    else:
        X = np.column_stack([
            _build_basis(t, P1, nh)[:, :-2],
            _build_basis(t, P2, nh),
        ])
    return X @ coefs


def fit_all_columns(arr, vmask, t_all, cols, verbose=True):
    """Fit trend model for every column."""
    fits = {}
    for j, col in enumerate(cols):
        obs = vmask[:, j]
        t_obs = t_all[obs].astype(float)
        y_obs = arr[obs, j]
        fits[col] = fit_column(t_obs, y_obs)
        if verbose:
            f = fits[col]
            p2_str = f"P2={f['P2']:.1f}" if f['P2'] else "single"
            print(f"  {col}: P1={f['P1']:.1f} {p2_str:>12s}  "
                  f"rmse={f['rmse']:.2f}")
    return fits


def build_warm_start(arr, vmask, t_all, cols, fits):
    """Fill NaN cells with fitted trend model. Observed values preserved."""
    T, D = arr.shape
    warm = arr.copy()
    t_float = t_all.astype(float)
    for j in range(D):
        missing = ~vmask[:, j]
        if missing.any():
            warm[missing, j] = trend_model(t_float[missing], fits[cols[j]])
    return warm


if __name__ == "__main__":
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import time as _time

    df = pd.read_csv("../data/limestone_data_challenge_2026.data.csv")
    cols = [c for c in df.columns if c.startswith("col_")]
    arr = df[cols].values.astype(float)
    vmask = ~np.isnan(arr)
    t_all = df["time"].values

    print("Fitting trend models (Lomb-Scargle + OLS)...")
    t0 = _time.time()
    fits = fit_all_columns(arr, vmask, t_all, cols, verbose=True)
    elapsed = _time.time() - t0
    print(f"\nTotal fit time: {elapsed:.1f}s")

    n_single = sum(1 for f in fits.values() if f["P2"] is None)
    n_double = sum(1 for f in fits.values() if f["P2"] is not None)
    print(f"Single-period: {n_single}, Double-period: {n_double}")
    avg_rmse = np.mean([f["rmse"] for f in fits.values()])
    print(f"Avg RMSE: {avg_rmse:.2f}")

    test_cols = ["col_00", "col_05", "col_09", "col_12", "col_22",
                 "col_28", "col_35", "col_40", "col_47", "col_52"]

    fig, axes = plt.subplots(5, 2, figsize=(16, 22), sharex=True)
    axes = axes.ravel()
    t_dense = np.arange(0, t_all.max() + 1, dtype=float)

    for i, col in enumerate(test_cols):
        ax = axes[i]
        j = cols.index(col)
        obs = vmask[:, j]

        ax.scatter(t_all[obs], arr[obs, j], s=0.5, alpha=0.4,
                   color="blue", label="observed")

        f = fits[col]
        model_y = trend_model(t_dense, f)
        p2_str = f"+P2={f['P2']:.0f}" if f['P2'] else ""
        ax.plot(t_dense, model_y, color="red", linewidth=1.5,
                label=f"P1={f['P1']:.0f}{p2_str} (rmse={f['rmse']:.1f})")

        ax.set_title(f"{col}  P={f['P1']:.0f}{p2_str}  rmse={f['rmse']:.1f}")
        ax.set_ylabel("price")
        ax.legend(fontsize=7, markerscale=6)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("time (t)")
    axes[-2].set_xlabel("time (t)")
    fig.suptitle("Harmonic trend fits (Lomb-Scargle + OLS)", fontsize=14)
    fig.tight_layout()
    fig.savefig("./analysis/trend_fits.png", dpi=150, bbox_inches="tight")
    print("Saved ./analysis/trend_fits.png")
