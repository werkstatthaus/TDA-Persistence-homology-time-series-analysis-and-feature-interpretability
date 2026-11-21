import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data
def generate_synthetic_loops_data(n_steps: int = 120, seed: int = 42):
    """Generates 3-dimensional topological loops in 3D space."""
    np.random.seed(seed)
    pts_per_tube = n_steps // 3

    t1 = np.linspace(0, 2 * np.pi, pts_per_tube)
    x1 = 15 + 5 * np.cos(t1) + np.random.normal(0, 0.15, pts_per_tube)
    y1 = 15 + 5 * np.sin(t1) + np.random.normal(0, 0.15, pts_per_tube)
    z1 = np.random.normal(0, 0.15, pts_per_tube)

    t2 = np.linspace(0, 2 * np.pi, pts_per_tube)
    x2 = np.random.normal(0, 0.15, pts_per_tube)
    y2 = 45 + 5 * np.cos(t2) + np.random.normal(0, 0.15, pts_per_tube)
    z2 = 15 + 5 * np.sin(t2) + np.random.normal(0, 0.15, pts_per_tube)

    t3 = np.linspace(0, 2 * np.pi, pts_per_tube)
    x3 = 30 + 5 * np.cos(t3) + np.random.normal(0, 0.15, pts_per_tube)
    y3 = np.random.normal(0, 0.15, pts_per_tube)
    z3 = 30 + 5 * np.sin(t3) + np.random.normal(0, 0.15, pts_per_tube)

    X = np.concatenate([x1, x2, x3])
    Y = np.concatenate([y1, y2, y3])
    Z = np.concatenate([z1, z2, z3])

    if len(X) < n_steps:
        pad = n_steps - len(X)
        X = np.pad(X, (0, pad), "edge")
        Y = np.pad(Y, (0, pad), "edge")
        Z = np.pad(Z, (0, pad), "edge")
    elif len(X) > n_steps:
        X, Y, Z = X[:n_steps], Y[:n_steps], Z[:n_steps]

    dates = pd.date_range(start="2026-01-01", periods=n_steps, freq="D")
    return pd.DataFrame({"X": X, "Y": Y, "Z": Z}, index=dates)


@st.cache_data
def generate_synthetic_stock_data(n_steps: int = 150, n_stocks: int = 5, seed: int = 42):
    """Generates synthetic multi-stock time series data with cyclic market dynamics."""
    np.random.seed(seed)
    t = np.linspace(0, 4 * np.pi, n_steps)

    stock_data = {}
    for i in range(n_stocks):
        base_trend = 100 + 10 * np.sin(t + (i * 0.5))
        noise = np.cumsum(np.random.normal(0, 1.0, n_steps))
        stock_data[f"Stock_{i+1}"] = base_trend + noise

    dates = pd.date_range(start="2026-01-01", periods=n_steps, freq="D")
    return pd.DataFrame(stock_data, index=dates)


@st.cache_data
def generate_synthetic_sine_wave(n_steps: int = 300, seed: int = 42):
    """Generates a clean 1D sine wave for time delay embedding demonstration."""
    np.random.seed(seed)
    t = np.linspace(0, 8 * np.pi, n_steps)
    series = np.sin(t) + 0.1 * np.random.normal(0, 1, n_steps)
    dates = pd.date_range(start="2026-01-01", periods=n_steps, freq="D")
    return pd.DataFrame({"Signal_1D": series}, index=dates)