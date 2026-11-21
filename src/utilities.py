import numpy as np
import pandas as pd


def create_time_delay_embedding(
    series: np.ndarray,
    tau: int,
    dim: int,
    index=None,
    delay_direction: str = "Future (+ i * tau)",
    avg_mode: str = "Raw Value at Shift",
    avg_k: int = 1,
) -> tuple[pd.DataFrame | None, list[str] | None]:
    """Construct a time-delay embedding point cloud with configurable direction and averaging."""
    arr = np.asarray(series, dtype=float)
    n = len(arr)
    tau_int = max(1, int(tau))
    dim_int = max(1, int(dim))
    k_int = max(1, int(avg_k))
    max_shift = (dim_int - 1) * tau_int

    if n == 0:
        return None, None

    if delay_direction == "Future (+ i * tau)":
        valid_end = (
            n - max_shift - k_int + 1
            if avg_mode == "Average over Future k points"
            else n - max_shift
        )
        if valid_end <= 0:
            return None, None
        base_indices = np.arange(0, valid_end, dtype=int)
        sign = 1
    else:
        start_t = (
            max_shift + k_int - 1
            if avg_mode == "Average over Past k points"
            else max_shift
        )
        if start_t >= n or start_t < 0:
            return None, None
        base_indices = np.arange(start_t, n, dtype=int)
        sign = -1

    embedded = np.zeros((len(base_indices), dim_int), dtype=float)
    for i in range(dim_int):
        centers = base_indices + sign * (i * tau_int)
        if avg_mode == "Average over Past k points":
            embedded[:, i] = [
                float(np.mean(arr[max(0, c - k_int + 1) : c + 1])) for c in centers
            ]
        elif avg_mode == "Average over Future k points":
            embedded[:, i] = [
                float(np.mean(arr[c : min(n, c + k_int)])) for c in centers
            ]
        else:
            embedded[:, i] = arr[centers]

    new_index = (
        index[base_indices]
        if index is not None and len(index) >= n
        else range(len(base_indices))
    )
    col_names = [f"Delay_t{i * tau_int}" for i in range(dim_int)]
    return pd.DataFrame(embedded, index=new_index, columns=col_names), col_names


def compute_mi_optimal_tau(
    series: np.ndarray, max_lag: int = 50, bins: int = 20
) -> tuple[int, np.ndarray]:
    """Compute time-lagged mutual information and return the first local minimum lag."""
    arr = np.asarray(series, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 4:
        return 1, np.zeros(max(2, int(max_lag)), dtype=float)

    eff_max_lag = int(max_lag)
    if n < eff_max_lag * 2:
        eff_max_lag = max(5, n // 3)
    eff_max_lag = max(2, min(eff_max_lag, n - 1))

    n_bins = max(2, int(bins))
    mi_vals = np.zeros(eff_max_lag, dtype=float)

    for lag in range(1, eff_max_lag):
        s_t = arr[:-lag]
        s_lag = arr[lag:]
        hist_2d, _, _ = np.histogram2d(s_t, s_lag, bins=n_bins)
        total_counts = float(np.sum(hist_2d))
        if total_counts <= 0.0:
            continue

        pxy = hist_2d / total_counts
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)

        # Vectorized MI calculation over positive joint probabilities
        outer_p = np.outer(px, py)
        valid = (pxy > 0.0) & (outer_p > 0.0)
        mi = float(np.sum(pxy[valid] * np.log2(pxy[valid] / outer_p[valid])))
        mi_vals[lag] = max(0.0, mi)

    mi_vals[0] = mi_vals[1] if eff_max_lag > 1 else 0.0

    optimal_tau = 1
    for lag in range(2, eff_max_lag - 1):
        if mi_vals[lag] < mi_vals[lag - 1] and mi_vals[lag] < mi_vals[lag + 1]:
            optimal_tau = lag
            break

    if optimal_tau == 1 and eff_max_lag > 2 and np.any(mi_vals[1:] > 0.0):
        optimal_tau = int(np.argmin(mi_vals[1:]) + 1)

    return max(1, optimal_tau), mi_vals


def _s9_extract_interval_feature(
    arr_1d, start_indices, k_len: int, agg_mode: str = "diff"
) -> np.ndarray:
    """
    Compute an interval feature over inclusive slice [t, t + k_len] for each t in start_indices.
    Supported agg_mode values: 'diff', 'mean', 'sum'.
    """
    arr = np.asarray(arr_1d, dtype=float)
    starts = np.asarray(start_indices, dtype=int)
    k_int = int(k_len)

    if len(starts) == 0:
        return np.array([], dtype=float)

    if agg_mode == "diff":
        return arr[starts + k_int] - arr[starts]

    out = np.empty(len(starts), dtype=float)
    for i, t_start in enumerate(starts):
        seg = arr[t_start : t_start + k_int + 1]
        if len(seg) == 0 or not np.all(np.isfinite(seg)):
            out[i] = np.nan
        elif agg_mode == "mean":
            out[i] = float(np.mean(seg))
        elif agg_mode == "sum":
            out[i] = float(np.sum(seg))
        else:
            out[i] = np.nan
    return out


def _s9_align_to_timestamps(
    series_obj: pd.Series | None, target_timestamps, fallback_len: int | None = None
) -> np.ndarray:
    """Align a pandas Series to target timestamps with positional fallback."""
    if series_obj is None or len(target_timestamps) == 0:
        return np.full(len(target_timestamps), np.nan, dtype=float)

    aligned = []
    for idx_pos, ts in enumerate(target_timestamps):
        val = np.nan
        if ts in series_obj.index:
            found = series_obj.loc[ts]
            val = found.iloc[0] if isinstance(found, pd.Series) else found
        elif 0 <= idx_pos < len(series_obj):
            val = series_obj.iloc[idx_pos]

        try:
            aligned.append(float(val))
        except (TypeError, ValueError):
            aligned.append(np.nan)

    return np.asarray(aligned, dtype=float)


def build_traceback_dataframe(
    df: pd.DataFrame, indices, selected_dims: list[str]
) -> pd.DataFrame:
    """Map selected point-cloud indices back to original rows, timestamps, and coordinates."""
    idx_list = list(indices) if indices is not None else []
    if not idx_list:
        return pd.DataFrame(
            columns=["Absolute Array Index", "Dataset Row Number", "Timestamp"]
            + list(selected_dims)
        )

    rows = []
    has_get_loc = hasattr(df.index, "get_loc")
    for idx in idx_list:
        idx_int = int(idx)
        ts = df.index[idx_int]
        pt = df.iloc[idx_int][selected_dims].to_numpy(dtype=float)
        row_num = df.index.get_loc(ts) if has_get_loc else idx_int
        if isinstance(row_num, slice):
            row_num = row_num.start
        elif isinstance(row_num, np.ndarray):
            row_num = int(np.flatnonzero(row_num)[0])

        row_dict = {
            "Absolute Array Index": idx_int,
            "Dataset Row Number": int(row_num),
            "Timestamp": str(ts),
        }
        for d_idx, dim_name in enumerate(selected_dims):
            row_dict[dim_name] = round(float(pt[d_idx]), 5)
        rows.append(row_dict)

    return pd.DataFrame(rows)


def _align_series_to_s8_anchors(
    series_obj: pd.Series | None, s8_frame: pd.DataFrame
) -> pd.Series:
    """Align a pandas Series to s8_frame rows using Anchor_Timestamp or Window_Anchor_Row."""
    if series_obj is None or len(s8_frame) == 0:
        return pd.Series(np.nan, index=s8_frame.index, dtype=float)

    aligned_vals = []
    for _, row in s8_frame.iterrows():
        ts = row.get("Anchor_Timestamp", None)
        r_idx = row.get("Window_Anchor_Row", None)
        val = np.nan

        if ts is not None and ts in series_obj.index:
            found = series_obj.loc[ts]
            val = found.iloc[0] if isinstance(found, pd.Series) else found
        elif r_idx is not None:
            try:
                r_int = int(r_idx)
                if r_int in series_obj.index:
                    found = series_obj.loc[r_int]
                    val = found.iloc[0] if isinstance(found, pd.Series) else found
                elif 0 <= r_int < len(series_obj):
                    val = series_obj.iloc[r_int]
            except (TypeError, ValueError):
                val = np.nan

        try:
            aligned_vals.append(float(val))
        except (TypeError, ValueError):
            aligned_vals.append(np.nan)

    return pd.Series(aligned_vals, index=s8_frame.index, dtype=float)