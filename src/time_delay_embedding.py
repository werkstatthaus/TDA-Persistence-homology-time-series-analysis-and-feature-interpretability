import numpy as np
import pandas as pd


def compute_acf_optimal_tau(series: np.ndarray, max_lag: int = 50, criterion: str = "First Zero Crossing"):
    """Computes autocorrelation function (ACF) and finds optimal tau."""
    n = len(series)
    if n < max_lag * 2:
        max_lag = max(5, n // 3)
    series_centered = series - np.mean(series)
    c0 = np.dot(series_centered, series_centered)
    if c0 == 0:
        return 1, np.ones(max_lag)

    acf_vals = np.zeros(max_lag)
    for lag in range(max_lag):
        if lag == 0:
            acf_vals[lag] = 1.0
        else:
            acf_vals[lag] = np.dot(series_centered[:-lag], series_centered[lag:]) / c0

    optimal_tau = 1
    if criterion == "Drop to 1/e (approx 0.368)":
        target = 1.0 / np.e
        idx = np.where(acf_vals <= target)[0]
        optimal_tau = int(idx[0]) if len(idx) > 0 else max_lag - 1
    else:
        idx = np.where(acf_vals <= 0)[0]
        optimal_tau = int(idx[0]) if len(idx) > 0 else 1

    return max(1, optimal_tau), acf_vals


def compute_mi_optimal_tau(series: np.ndarray, max_lag: int = 50, bins: int = 20):
    """Computes mutual information between s(t) and s(t-tau) and finds the first local minimum."""
    n = len(series)
    if n < max_lag * 2:
        max_lag = max(5, n // 3)

    mi_vals = np.zeros(max_lag)
    for lag in range(1, max_lag):
        s_t = series[:-lag]
        s_lag = series[lag:]

        hist_2d, _, _ = np.histogram2d(s_t, s_lag, bins=bins, density=True)
        pxy = hist_2d / np.sum(hist_2d)
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)

        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += pxy[i, j] * np.log2(pxy[i, j] / (px[i] * py[j]))
        mi_vals[lag] = max(0.0, mi)

    mi_vals[0] = mi_vals[1] if max_lag > 1 else 0.0

    optimal_tau = 1
    for lag in range(2, max_lag - 1):
        if mi_vals[lag] < mi_vals[lag - 1] and mi_vals[lag] < mi_vals[lag + 1]:
            optimal_tau = lag
            break
    if optimal_tau == 1 and max_lag > 2:
        optimal_tau = int(np.argmin(mi_vals[1:]) + 1)

    return max(1, optimal_tau), mi_vals


def create_time_delay_embedding(
    series: np.ndarray,
    tau: int,
    dim: int,
    index=None,
    delay_direction="Future (+ i * tau)",
    avg_mode="Raw Value at Shift",
    avg_k=1,
):
    """Constructs a  time-delay embedding point cloud with direction and averaging."""
    n = len(series)
    max_shift = (dim - 1) * tau

    if delay_direction == "Future (+ i * tau)":
        if avg_mode == "Average over Future k points":
            valid_end = n - max_shift - avg_k + 1
        else:
            valid_end = n - max_shift

        if valid_end <= 0:
            return None, None

        base_indices = np.arange(0, valid_end)
        embedded = np.zeros((len(base_indices), dim))

        for i in range(dim):
            shift = i * tau
            for idx, t in enumerate(base_indices):
                center = t + shift
                if avg_mode == "Average over Past k points":
                    p_start = max(0, center - avg_k + 1)
                    embedded[idx, i] = np.mean(series[p_start : center + 1])
                elif avg_mode == "Average over Future k points":
                    p_end = min(n, center + avg_k)
                    embedded[idx, i] = np.mean(series[center:p_end])
                else:
                    embedded[idx, i] = series[center]

        new_index = index[base_indices] if index is not None and len(index) >= n else range(len(base_indices))
    else:
        start_t = max_shift + avg_k - 1 if avg_mode == "Average over Past k points" else max_shift

        if start_t >= n:
            return None, None

        base_indices = np.arange(start_t, n)
        embedded = np.zeros((len(base_indices), dim))

        for i in range(dim):
            shift = i * tau
            for idx, t in enumerate(base_indices):
                center = t - shift
                if avg_mode == "Average over Past k points":
                    p_start = max(0, center - avg_k + 1)
                    embedded[idx, i] = np.mean(series[p_start : center + 1])
                elif avg_mode == "Average over Future k points":
                    p_end = min(n, center + avg_k)
                    embedded[idx, i] = np.mean(series[center:p_end])
                else:
                    embedded[idx, i] = series[center]

        new_index = index[base_indices] if index is not None and len(index) >= n else range(len(base_indices))

    col_names = [f"Delay_t{i*tau}" for i in range(dim)]
    return pd.DataFrame(embedded, index=new_index, columns=col_names), col_names