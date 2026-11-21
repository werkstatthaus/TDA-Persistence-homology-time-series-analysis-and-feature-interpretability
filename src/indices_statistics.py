import itertools
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import binomtest, pearsonr, spearmanr


def _compute_corr_metric(
    x_arr, y_arr, method: str = "pearson", return_n: bool = False
):
    """
    Compute correlation coefficient, two-sided p-value, and optionally valid pair count.
    Supported methods: 'pearson', 'difference' (first differences), and 'spearman'.
    """
    x = np.asarray(x_arr, dtype=float)
    y = np.asarray(y_arr, dtype=float)

    if method == "difference":
        if len(x) < 3 or len(y) < 3:
            return (np.nan, np.nan, 0) if return_n else (np.nan, np.nan)
        x = np.diff(x)
        y = np.diff(y)

    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    n_valid = int(len(x_clean))

    if n_valid < 2 or np.std(x_clean) == 0.0 or np.std(y_clean) == 0.0:
        return (np.nan, np.nan, n_valid) if return_n else (np.nan, np.nan)

    if method in ("pearson", "difference"):
        r_val = float(np.corrcoef(x_clean, y_clean)[0, 1])
        try:
            _, p_val = pearsonr(x_clean, y_clean)
            p_val = float(p_val)
        except Exception:
            p_val = np.nan
        return (r_val, p_val, n_valid) if return_n else (r_val, p_val)

    if method == "spearman":
        try:
            r_val, p_val = spearmanr(x_clean, y_clean)
            return (
                (float(r_val), float(p_val), n_valid)
                if return_n
                else (float(r_val), float(p_val))
            )
        except Exception:
            rx = pd.Series(x_clean).rank().to_numpy()
            ry = pd.Series(y_clean).rank().to_numpy()
            if np.std(rx) == 0.0 or np.std(ry) == 0.0:
                return (np.nan, np.nan, n_valid) if return_n else (np.nan, np.nan)
            r_val = float(np.corrcoef(rx, ry)[0, 1])
            return (r_val, np.nan, n_valid) if return_n else (r_val, np.nan)

    return (np.nan, np.nan, 0) if return_n else (np.nan, np.nan)


# Backward-compatible alias if imported elsewhere in the app
_interval_compute_corr_metric = _compute_corr_metric


def _compute_acf_values(arr_1d, max_lag: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Compute sample autocorrelation for lags 1..max_lag using normalized autocovariance."""
    vals = np.asarray(arr_1d, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = len(vals)
    if n < 3:
        return np.array([], dtype=int), np.array([], dtype=float)

    max_l = max(1, min(int(max_lag), n - 1))
    centered = vals - np.mean(vals)
    denom = float(np.dot(centered, centered))
    if denom <= 0.0:
        return np.arange(1, max_l + 1, dtype=int), np.zeros(max_l, dtype=float)

    lags = np.arange(1, max_l + 1, dtype=int)
    acf_vals = np.array(
        [float(np.dot(centered[: n - lag], centered[lag:])) / denom for lag in lags],
        dtype=float,
    )
    return lags, acf_vals


# Backward-compatible alias if imported elsewhere in the app
_s9_compute_acf_values = _compute_acf_values


def _s9_compute_directional_metrics(
    x_arr,
    y_arr,
    return_p: bool = False,
    n_perms: int = 2000,
    seed: int = 42,
):
    """
    Compute two directional sign-alignment scores on paired interval values (x_i, y_i):
      1. uncentered_cosine: sum(x_i * y_i) / (||x||_2 * ||y||_2)
      2. sign_agreement: mean(sign(x_i) * sign(y_i)) over pairs where x_i != 0 and y_i != 0

    When return_p=True, also returns two-sided p-values (p_cos, p_sign):
      - p_cos: exact sign-flip test for n <= 12, else seeded Monte Carlo sign-flip test.
      - p_sign: exact two-sided binomial test against p = 0.5 over non-zero pairs.
    """
    x = np.asarray(x_arr, dtype=float)
    y = np.asarray(y_arr, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    xc = x[mask]
    yc = y[mask]

    if len(xc) == 0:
        return (np.nan, np.nan, 0, np.nan, np.nan) if return_p else (np.nan, np.nan, 0)

    norm_x = float(np.linalg.norm(xc))
    norm_y = float(np.linalg.norm(yc))
    uncentered_cos = (
        float(np.dot(xc, yc) / (norm_x * norm_y))
        if norm_x > 0.0 and norm_y > 0.0
        else np.nan
    )

    nz_mask = (xc != 0.0) & (yc != 0.0)
    n_nz = int(np.sum(nz_mask))
    if n_nz > 0:
        sign_prod = np.sign(xc[nz_mask]) * np.sign(yc[nz_mask])
        sign_agree = float(np.mean(sign_prod))
    else:
        sign_agree = np.nan
        sign_prod = np.array([], dtype=float)

    if not return_p:
        return uncentered_cos, sign_agree, n_nz

    # 1. Two-sided p-value for uncentered_cosine via sign-flip permutation
    if np.isfinite(uncentered_cos) and len(xc) >= 1:
        obs_abs = abs(uncentered_cos)
        denom_xy = norm_x * norm_y
        n_pts = len(xc)

        if n_pts <= 12:
            exceed = 0
            total = 0
            for flips in itertools.product((-1.0, 1.0), repeat=n_pts):
                cos_null = float(np.dot(xc, yc * np.asarray(flips)) / denom_xy)
                if abs(cos_null) >= obs_abs - 1e-12:
                    exceed += 1
                total += 1
            p_cos = float(exceed / total) if total > 0 else np.nan
        else:
            rng = np.random.default_rng(seed)
            flips_mc = rng.choice((-1.0, 1.0), size=(n_perms, n_pts))
            dots = (flips_mc * yc) @ xc
            cos_nulls = np.abs(dots / denom_xy)
            exceed = int(np.sum(cos_nulls >= obs_abs - 1e-12))
            p_cos = float((exceed + 1) / (n_perms + 1))
    else:
        p_cos = np.nan

    # 2. Two-sided exact binomial p-value for sign_agreement (null: agreement prob = 0.5)
    if n_nz > 0 and np.isfinite(sign_agree):
        n_agree = int(np.sum(sign_prod > 0.0))
        p_sign = float(binomtest(n_agree, n=n_nz, p=0.5, alternative="two-sided").pvalue)
    else:
        p_sign = np.nan

    return uncentered_cos, sign_agree, n_nz, p_cos, p_sign


def _compute_acf(
    series: np.ndarray,
    max_lag: int,
    criterion: str,
    include_lag_zero: bool,
    adaptive_lag: bool,
) -> tuple[np.ndarray, int, np.ndarray, float]:
    """Compute lags, optimal_tau, acf_curve, and acf_value at optimal_tau."""
    vals = np.asarray(series, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = len(vals)

    if n < 3:
        empty_lags = np.array([], dtype=int)
        empty_curve = np.array([], dtype=float)
        return empty_lags, 1, empty_curve, np.nan

    eff_max_lag = int(max_lag)
    if adaptive_lag and n < eff_max_lag * 2:
        eff_max_lag = max(5, n // 3)

    max_l = max(1, min(eff_max_lag, n - 1))
    lags = (
        np.arange(0, max_l + 1, dtype=int)
        if include_lag_zero
        else np.arange(1, max_l + 1, dtype=int)
    )

    centered = vals - np.mean(vals)
    denom = float(np.dot(centered, centered))

    if denom <= 0.0:
        acf_curve = (
            np.ones(len(lags), dtype=float)
            if include_lag_zero
            else np.zeros(len(lags), dtype=float)
        )
    else:
        acf_curve = np.array(
            [
                1.0 if k == 0 else float(np.dot(centered[:-k], centered[k:])) / denom
                for k in lags
            ],
            dtype=float,
        )

    if criterion == "Drop to 1/e (approx 0.368)":
        target = 1.0 / np.e
        idx = np.where((lags >= 1) & (acf_curve <= target))[0]
        optimal_tau = int(lags[idx[0]]) if len(idx) > 0 else max(1, int(lags[-1]))
    else:
        idx = np.where((lags >= 1) & (acf_curve <= 0.0))[0]
        optimal_tau = int(lags[idx[0]]) if len(idx) > 0 else 1

    match = np.where(lags == optimal_tau)[0]
    acf_value = float(acf_curve[match[0]]) if len(match) > 0 else float(acf_curve[-1])

    return lags, optimal_tau, acf_curve, acf_value


def render_s9_metric_methodology_help(expanded: bool = False) -> None:
    """Render a concise Streamlit explanation of correlation parameters and directional p-values."""
    with st.expander("How correlation metrics and p-values are computed", expanded=expanded):
        st.markdown(
            """
### 1. Correlation Parameters (`pearson`, `difference`, `spearman`)
All three methods first filter to **pairwise finite values** (`np.isfinite(x) & np.isfinite(y)`), require **at least 2 valid pairs**, and return `NaN` if either series has **zero standard deviation** (`std == 0.0`).

- **Pearson (`"pearson"`)**
  - **Input:** Raw aligned values (`x`, `y`).
  - **Computation & parameters:** Linear correlation via `np.corrcoef(x_clean, y_clean)[0, 1]` and two-sided test via `scipy.stats.pearsonr(x_clean, y_clean)` (`alternative="two-sided"`).
- **Difference (`"difference"`)**
  - **Input:** First-order lag-1 differences (`np.diff(x)`, `np.diff(y)` — i.e., `x[t+1] - x[t]`), requiring at least **3 raw points** (`len >= 3`) before differencing.
  - **Computation & parameters:** Applies the same Pearson correlation (`np.corrcoef` + `scipy.stats.pearsonr`, `alternative="two-sided"`) to the differenced series.
- **Spearman (`"spearman"`)**
  - **Input:** Raw aligned values (`x`, `y`).
  - **Computation & parameters:** Rank-order correlation via `scipy.stats.spearmanr(x_clean, y_clean)` using default `alternative="two-sided"` and average tie-breaking (`method="average"`), with a pandas rank fallback if needed.

### 2. Uncentered Cosine $p$-Value Method
- **`uncentered_cosine`** measures magnitude-weighted directional alignment without mean subtraction:
  $$\\cos_{\\text{uncentered}} = \\frac{\\sum_i x_i y_i}{\\sqrt{\\sum_i x_i^2}\\,\\sqrt{\\sum_i y_i^2}}$$
- **Null & $p$-value method:** Computed via a **two-sided sign-flip (Rademacher) test** (`|cos_null| >= |uncentered_cos|`), randomly flipping signs (`+1` / `-1`) to test against zero directional alignment:
  - **Exact sign-flip enumeration** over all $2^n$ sign combinations when `n <= 12`.
  - **Seeded Monte Carlo sign flips** (`n_perms=2000`, `seed=42`, `(exceed + 1) / (n_perms + 1)`) when `n > 12`.
- *(For reference, `Sign_Agreement_p_value` uses an exact two-sided binomial test against `p = 0.5` over non-zero sign pairs.)*
            """
        )