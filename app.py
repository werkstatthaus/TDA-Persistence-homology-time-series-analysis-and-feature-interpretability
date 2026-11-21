import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.signal import welch
from src.indices_statistics import _interval_compute_corr_metric, _compute_acf,_s9_compute_directional_metrics,_compute_corr_metric,render_s9_metric_methodology_help
from src.datagenerators import generate_synthetic_loops_data, generate_synthetic_stock_data,generate_synthetic_sine_wave
from src.tda_core import TDAPipeline, compute_fast_diagram_only
from src.utilities import build_traceback_dataframe, _align_series_to_s8_anchors, create_time_delay_embedding,_s9_extract_interval_feature, compute_mi_optimal_tau
from src.metrics import compute_betti_curve, generate_persistence_image, match_cycle_across_spaces, compute_landscape_lp_norm, compute_diagram_wasserstein, compute_landscape_layers,compute_persistence_entropy

st.set_page_config(page_title="TDA Persistence Images & Phase Space Inspector", layout="wide")

# =============================================================================
# DEFAULT CONFIGURATION INITIALIZATIONS 
# =============================================================================
rel_persistence_pct = 5
thresh_type = "Absolute (Fine Decimal)"
min_persistence_thresh = 0.0005
protect_critical_edges = True
pi_filter_mode = "Persistence Length (d - b)"
custom_filter_expr = "p"
standardize_data = True
homology_dim = 1
pc_stride = 1
stride_agg_mode = "Raw Value (Every n-th row)"
window_direction = "Future (Forward from Start Index)"
wasserstein_order = 1.0

# =============================================================================
# STREAMLIT USER INTERFACE & APP MODES
# =============================================================================

st.title(" TDA Persistence Images & Phase Space Inspector")
st.markdown(
    """
Explore $H_0, H_1, H_2$ persistence homology across multivariate spaces or time-delay embedded phase spaces.
"""
)

st.sidebar.header("0. Analysis Mode")
app_mode = st.sidebar.radio(
    "Select Workflow Mode", ["Multivariate Data Analysis", "Time-Delay Embedding (Takens')"]
)

st.sidebar.header("1. Data Source Setup")
raw_df = None
if app_mode == "Multivariate Data Analysis":
    data_option = st.sidebar.radio(
        "Data Source", ["Synthetic 3 Loops (3D)", "Synthetic Stock Dataset", "Upload CSV / Joblib"]
    )
    if data_option == "Synthetic 3 Loops (3D)":
        n_steps_gen = st.sidebar.number_input(
            "Total Time Steps / Rows", min_value=60, max_value=500, value=120, step=10
        )
        raw_df = generate_synthetic_loops_data(n_steps=n_steps_gen)
    elif data_option == "Synthetic Stock Dataset":
        n_steps_gen = st.sidebar.number_input(
            "Total Time Steps / Rows", min_value=60, max_value=500, value=150, step=10
        )
        n_stocks = st.sidebar.number_input(
            "Number of Stocks (Dimensions)", min_value=2, max_value=20, value=5, step=1
        )
        raw_df = generate_synthetic_stock_data(n_steps=n_steps_gen, n_stocks=n_stocks)
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload tabular CSV or joblib file", type=["csv", "pkl", "joblib"]
        )
        if uploaded_file is not None:
            try:
                uploaded_file.seek(0)
                if uploaded_file.name.endswith((".pkl", ".joblib")):
                    import joblib

                    raw_df = joblib.load(uploaded_file)
                else:
                    try:
                        raw_df = pd.read_csv(
                            uploaded_file,
                            index_col=0,
                            parse_dates=True,
                            encoding="utf-8",
                            quotechar='"',
                            escapechar="\\",
                            on_bad_lines="skip",
                        )
                    except Exception:
                        uploaded_file.seek(0)
                        raw_df = pd.read_csv(
                            uploaded_file,
                            index_col=0,
                            parse_dates=True,
                            encoding="latin1",
                            quotechar='"',
                            escapechar="\\",
                            on_bad_lines="skip",
                        )
                st.sidebar.success(
                    f"Successfully loaded dataset ({raw_df.shape[0]} rows, {raw_df.shape[1]} cols)"
                )
            except Exception as e:
                st.sidebar.error(f"Error reading file: {e}")
                raw_df = None
        else:
            st.sidebar.info("Please upload a CSV or joblib file to proceed.")
else:
    td_source = st.sidebar.radio(
        "1D Signal / Tabular Source",
        ["Synthetic Sine Wave + Noise", "Upload Tabular CSV (DataFrame [[]])"],
    )
    if td_source == "Synthetic Sine Wave + Noise":
        n_steps_1d = st.sidebar.number_input(
            "Signal Length", min_value=100, max_value=1000, value=300, step=50
        )
        raw_df = generate_synthetic_sine_wave(n_steps=n_steps_1d)
    else:
        uploaded_1d = st.sidebar.file_uploader("Upload Tabular CSV for Embedding", type=["csv"])
        if uploaded_1d is not None:
            try:
                raw_df = pd.read_csv(uploaded_1d, index_col=0, parse_dates=True)
            except Exception as e:
                st.sidebar.error(f"Error loading CSV: {e}")
                raw_df = None
        else:
            st.sidebar.info("Upload a tabular CSV file to begin.")

if raw_df is None:
    st.info(" Configure your data source or upload a valid data file in the sidebar to begin.")
    st.stop()
    


# =============================================================================
# INITIAL / RAW DATASET EXPLORATORY STATISTICS
# =============================================================================
st.markdown("---")
st.header("Initial Dataset Exploratory Analysis")
st.markdown("Statistics and overview of your raw loaded dataset prior to transformations.")

init_numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
st.info(f"**Initial Dataset Shape:** `{raw_df.shape[0]}` rows × `{raw_df.shape[1]}` columns")

init_eda_stats = pd.DataFrame(
    {
        "Mean": raw_df[init_numeric_cols].mean(),
        "Std Dev": raw_df[init_numeric_cols].std(),
        "Min": raw_df[init_numeric_cols].min(),
        "25%": raw_df[init_numeric_cols].quantile(0.25),
        "Median (50%)": raw_df[init_numeric_cols].median(),
        "75%": raw_df[init_numeric_cols].quantile(0.75),
        "Max": raw_df[init_numeric_cols].max(),
        "Missing Values": raw_df[init_numeric_cols].isnull().sum(),
        "Missing %": (raw_df[init_numeric_cols].isnull().mean() * 100).round(2),
    }
)
st.dataframe(init_eda_stats, use_container_width=True)


# =============================================================================
# TIME-DELAY EMBEDDING (IF APPLICABLE) WITH ACF & MI OPTIMIZER ABOVE DELAY CHOICE
# =============================================================================
selected_dims = []
if app_mode == "Time-Delay Embedding (Takens')":
    st.markdown("---")
    st.header(" Time-Delay Phase Space Embedding")

    st.sidebar.subheader(" Embedding Parameters")
    target_col = st.sidebar.selectbox("Select Target Column for 1D Series", options=init_numeric_cols)

    st.sidebar.markdown("---")
    st.sidebar.subheader(" Automated Optimizer (ACF & MI)")
    run_optimizer = st.sidebar.button("Run ACF & MI Optimization")

    series_vals = raw_df[target_col].dropna().values

    if "rec_acf_tau" not in st.session_state:
        st.session_state["rec_acf_tau"] = 5
    if "rec_mi_tau" not in st.session_state:
        st.session_state["rec_mi_tau"] = 5

    if run_optimizer:
        acf_crit = st.sidebar.selectbox(
            "ACF Target Criterion", ["First Zero Crossing", "Drop to 1/e "]
        )
        _,opt_acf_tau, acf_curve,_ = _compute_acf(
            series_vals, max_lag=60, criterion=acf_crit,include_lag_zero=False,
            adaptive_lag=False
        )
        opt_mi_tau, mi_curve = compute_mi_optimal_tau(series_vals, max_lag=60, bins=20)

        st.session_state["rec_acf_tau"] = opt_acf_tau
        st.session_state["rec_mi_tau"] = opt_mi_tau
        st.session_state["acf_curve"] = acf_curve
        st.session_state["mi_curve"] = mi_curve
        st.sidebar.success(
            f"Recommended ACF $\tau$: **{opt_acf_tau}** | MI $\tau$: **{opt_mi_tau}**"
        )

    if "acf_curve" in st.session_state and "mi_curve" in st.session_state:
        with st.expander(" View ACF & Mutual Information Diagnostic Curves", expanded=True):
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                fig_acf = go.Figure()
                fig_acf.add_trace(
                    go.Scatter(
                        y=st.session_state["acf_curve"],
                        mode="lines+markers",
                        line=dict(color="#1f77b4"),
                    )
                )
                fig_acf.add_vline(
                    x=st.session_state["rec_acf_tau"],
                    line_dash="solid",
                    line_color="red",
                    annotation_text=f"Rec τ = {st.session_state['rec_acf_tau']}",
                )
                fig_acf.update_layout(
                    title="Autocorrelation Function (ACF)",
                    xaxis_title="Lag (τ)",
                    yaxis_title="ACF Value",
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_acf, use_container_width=True)
                if st.button("👉 Apply Recommended ACF Tau"):
                    st.session_state["chosen_tau"] = st.session_state["rec_acf_tau"]
                    st.rerun()
            with col_d2:
                fig_mi = go.Figure()
                fig_mi.add_trace(
                    go.Scatter(
                        y=st.session_state["mi_curve"],
                        mode="lines+markers",
                        line=dict(color="#ff7f0e"),
                    )
                )
                fig_mi.add_vline(
                    x=st.session_state["rec_mi_tau"],
                    line_dash="solid",
                    line_color="green",
                    annotation_text=f"Rec τ = {st.session_state['rec_mi_tau']}",
                )
                fig_mi.update_layout(
                    title="Mutual Information (MI)",
                    xaxis_title="Lag (τ)",
                    yaxis_title="Mutual Information (bits)",
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_mi, use_container_width=True)
                if st.button("👉 Apply Recommended MI Tau"):
                    st.session_state["chosen_tau"] = st.session_state["rec_mi_tau"]
                    st.rerun()

    default_tau_val = st.session_state.get(
        "chosen_tau", st.session_state.get("rec_acf_tau", 5)
    )
    tau = st.sidebar.number_input(
        "Time Delay ($\tau$)", min_value=1, max_value=50, value=int(default_tau_val), step=1
    )
    dim_embed = st.sidebar.number_input(
        "Embedding Dimension ($d$)", min_value=2, max_value=10, value=3, step=1
    )
    delay_dir = st.sidebar.radio(
        "Delay Direction", ["Future (+ i * tau)", "Past (- i * tau)"], horizontal=True
    )
    embed_avg_mode = st.sidebar.selectbox(
        "Delay Coordinate Aggregation",
        [
            "Raw Value at Shift",
            "Average over Past k points",
            "Average over Future k points",
        ],
    )
    embed_k = st.sidebar.number_input(
        "Delay Averaging Window ($k$)", min_value=1, max_value=20, value=1, step=1
    )

    series_vals = raw_df[target_col].values
    embedded_df, selected_dims = create_time_delay_embedding(
        series_vals,
        tau=tau,
        dim=dim_embed,
        index=raw_df.index,
        delay_direction=delay_dir,
        avg_mode=embed_avg_mode,
        avg_k=int(embed_k),
    )
    if embedded_df is not None:
        raw_df = embedded_df
        st.success(
            f"Successfully constructed delay-embedded phase space ($\tau={tau}, d={dim_embed}$). New shape: `{raw_df.shape[0]}` rows × `{raw_df.shape[1]}` columns."
        )
    else:
        st.error(
            "Embedding parameters result in too few points. Reduce delay, dimension, or averaging window, or provide a longer series."
        )
        st.stop()


# =============================================================================
# FEATURE ENGINEERING: PER-DIMENSION & AVERAGE ROLLING VOLATILITY
# =============================================================================
st.markdown("---")
st.header("Feature Engineering: Per-Dimension & Average Rolling Volatility")
st.markdown(
    "Compute rolling volatility (rolling standard deviation) for every numeric dimension "
    "separately plus the average volatility across dimensions, aligned to the dataset index."
)

base_numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()

if "s1_vol_window" not in st.session_state:
    st.session_state["s1_vol_window"] = 20
if "s1_vol_ddof" not in st.session_state:
    st.session_state["s1_vol_ddof"] = 1
if "s1_vol_include_in_raw_df" not in st.session_state:
    st.session_state["s1_vol_include_in_raw_df"] = False
if "s1_vol_include_avg_in_raw_df" not in st.session_state:
    st.session_state["s1_vol_include_avg_in_raw_df"] = True

v_c1, v_c2, v_c3 = st.columns(3)
with v_c1:
    vol_window = st.number_input(
        "Rolling Volatility Window",
        min_value=2,
        max_value=max(2, len(raw_df)),
        value=int(st.session_state["s1_vol_window"]),
        step=1,
        key="s1_vol_window_input",
    )
    st.session_state["s1_vol_window"] = int(vol_window)
with v_c2:
    vol_ddof = st.selectbox(
        "Delta Degrees of Freedom (ddof)",
        options=[0, 1],
        index=0 if int(st.session_state["s1_vol_ddof"]) == 0 else 1,
        key="s1_vol_ddof_input",
    )
    st.session_state["s1_vol_ddof"] = int(vol_ddof)
with v_c3:
    vol_min_periods_full = st.checkbox(
        "Require full window (min_periods = window)",
        value=True,
        key="s1_vol_full_win",
    )

# Exclude previously added rolling volatility columns when computing base dimension volatilities
dim_cols_for_vol = [
    c for c in base_numeric_cols
    if not c.startswith("RollVol_") and c != "Avg_Rolling_Volatility"
]

min_p = int(vol_window) if vol_min_periods_full else 1
num_subset = raw_df[dim_cols_for_vol].apply(pd.to_numeric, errors="coerce")

# Per-dimension rolling volatility DataFrame aligned to raw_df.index
df_dim_vol = pd.DataFrame(index=raw_df.index)
for col in dim_cols_for_vol:
    df_dim_vol[f"RollVol_{col}_w{int(vol_window)}"] = (
        num_subset[col]
        .rolling(window=int(vol_window), min_periods=min_p)
        .std(ddof=int(vol_ddof))
    )

# Average volatility across all dimensions at each timestep (same index)
avg_vol_series = df_dim_vol.mean(axis=1, skipna=True)
df_avg_vol = pd.DataFrame(
    {
        "Row_Index": np.arange(len(raw_df), dtype=int),
        "Avg_Rolling_Volatility": avg_vol_series,
    },
    index=raw_df.index,
)

# Combined rolling volatility DataFrame (per-dimension + average)
df_all_vol = df_dim_vol.copy()
df_all_vol["Avg_Rolling_Volatility"] = avg_vol_series

# Save DataFrames in session_state
st.session_state["s1_dim_volatility_df"] = df_dim_vol
st.session_state["s1_avg_volatility_df"] = df_avg_vol
st.session_state["s1_all_volatility_df"] = df_all_vol

# Option to include per-dimension and/or average rolling volatility columns in raw_df
opt_c1, opt_c2 = st.columns(2)
with opt_c1:
    include_dim_cols = st.checkbox(
        "Include per-dimension volatility columns in dataset",
        value=bool(st.session_state["s1_vol_include_in_raw_df"]),
        key="s1_vol_inc_dim_cb",
    )
    st.session_state["s1_vol_include_in_raw_df"] = include_dim_cols
with opt_c2:
    include_avg_col = st.checkbox(
        "Include average volatility column in dataset",
        value=bool(st.session_state["s1_vol_include_avg_in_raw_df"]),
        key="s1_vol_inc_avg_cb",
    )
    st.session_state["s1_vol_include_avg_in_raw_df"] = include_avg_col

# Remove any stale generated volatility columns first, then add selected ones cleanly
stale_vol_cols = [c for c in raw_df.columns if c.startswith("RollVol_") or c == "Avg_Rolling_Volatility"]
if stale_vol_cols:
    raw_df = raw_df.drop(columns=stale_vol_cols)

if include_dim_cols:
    for col in df_dim_vol.columns:
        raw_df[col] = df_dim_vol[col].reindex(raw_df.index)

if include_avg_col:
    raw_df["Avg_Rolling_Volatility"] = df_avg_vol["Avg_Rolling_Volatility"].reindex(raw_df.index)

st.session_state["raw_df"] = raw_df

# Dataset shape info
st.markdown(
    f"**Dataset Shape (`raw_df`):** `{raw_df.shape[0]}` rows × `{raw_df.shape[1]}` columns "
    f"| **Per-Dimension Volatility DF:** `{df_dim_vol.shape[0]}` × `{df_dim_vol.shape[1]}` "
    f"| **Average Volatility DF:** `{df_avg_vol.shape[0]}` × `{df_avg_vol.shape[1]}`"
)

# Post-computation statistics table across every dimension volatility & average volatility
vol_stat_rows = []
for col in df_all_vol.columns:
    s_vals = pd.to_numeric(df_all_vol[col], errors="coerce").dropna()
    if len(s_vals) > 0:
        vol_stat_rows.append({
            "Series": col,
            "Valid_Timesteps": int(len(s_vals)),
            "NaN_Timesteps": int(len(df_all_vol) - len(s_vals)),
            "Mean": float(s_vals.mean()),
            "Std": float(s_vals.std(ddof=1)) if len(s_vals) > 1 else 0.0,
            "Median": float(s_vals.median()),
            "Min": float(s_vals.min()),
            "Max": float(s_vals.max()),
        })
    else:
        vol_stat_rows.append({
            "Series": col,
            "Valid_Timesteps": 0,
            "NaN_Timesteps": int(len(df_all_vol)),
            "Mean": np.nan,
            "Std": np.nan,
            "Median": np.nan,
            "Min": np.nan,
            "Max": np.nan,
        })

df_vol_stats = pd.DataFrame(vol_stat_rows)
st.session_state["s1_volatility_stats_df"] = df_vol_stats

st.subheader(" Rolling Volatility Statistics (Per Dimension & Average)")
st.dataframe(df_vol_stats, use_container_width=True)

current_numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()

# =============================================================================
# PROCESSED WORKING DATASET STATISTICS
# =============================================================================
st.markdown("---")
st.header(" Processed Working Dataset Statistics")
st.markdown("Statistics of the active working dataset (after embedding / feature engineering).")

st.info(f" **Processed Shape:** `{raw_df.shape[0]}` rows × `{raw_df.shape[1]}` columns")

proc_eda_stats = pd.DataFrame(
    {
        "Mean": raw_df[current_numeric_cols].mean(),
        "Std Dev": raw_df[current_numeric_cols].std(),
        "Min": raw_df[current_numeric_cols].min(),
        "25%": raw_df[current_numeric_cols].quantile(0.25),
        "Median (50%)": raw_df[current_numeric_cols].median(),
        "75%": raw_df[current_numeric_cols].quantile(0.75),
        "Max": raw_df[current_numeric_cols].max(),
        "Missing Values": raw_df[current_numeric_cols].isnull().sum(),
        "Missing %": (raw_df[current_numeric_cols].isnull().mean() * 100).round(2),
    }
)
st.dataframe(proc_eda_stats, use_container_width=True)

# Maintain dynamic selected dims state across column structure changes
if "prev_cols_signature" not in st.session_state or st.session_state["prev_cols_signature"] != str(
    current_numeric_cols
):
    st.session_state["prev_cols_signature"] = str(current_numeric_cols)
    st.session_state["active_dims_selection"] = (
        current_numeric_cols[:3] if len(current_numeric_cols) >= 3 else current_numeric_cols[:2]
    )

col_ed1, col_ed2 = st.columns([3, 1])
with col_ed1:
    columns_to_keep = st.multiselect(
        "Select Columns to Keep (Drop Unwanted Columns):",
        options=current_numeric_cols,
        default=current_numeric_cols,
    )
    if columns_to_keep:
        raw_df = raw_df[columns_to_keep]
        current_numeric_cols = columns_to_keep

    selected_dims = st.multiselect(
        "Select Active Dimensions (Columns) for TDA Space (Requires at least 2 for H1, 3 for H2):",
        options=current_numeric_cols,
        default=current_numeric_cols[:3]
        if len(current_numeric_cols) >= 3
        else current_numeric_cols[:2],
        key="active_dims_selection",
    )

with col_ed2:
    st.write("")
    st.write("")
    if st.button("Reset to Base Dataset"):
        st.session_state["added_corr_features"] = []
        st.rerun()

if len(selected_dims) < 1:
    st.error(" Please select or configure at least **1 dimension** for topological analysis.")
    st.stop()

max_total_rows = len(raw_df)

# =============================================================================
# SIDEBAR CONFIGURATION FORM FOR TDA PIPELINE
# =============================================================================
with st.sidebar.form("config_form"):
    st.subheader("Homology Dimension ($H_k$)")
    homology_dim = st.selectbox(
        "Select Homology Dimension",
        options=[0, 1, 2],
        index=1,
        format_func=lambda x: f"H_{x} ({'Connected Components' if x==0 else ('Loops / Cycles' if x==1 else 'Voids / Cavities')})",
    )

    st.subheader("Point Cloud Window & Direction")
    window_size = st.number_input(
        "Window Size (Number of Rows)",
        min_value=min(30, max_total_rows),
        max_value=max_total_rows,
        value=min(100, max_total_rows),
        step=1,
    )
    pc_stride = st.number_input("Point Cloud Row Stride ($n$)", min_value=1, max_value=100, value=1, step=1)
    stride_agg_mode = st.selectbox(
        "Stride Point Aggregation",
        [
            "Raw Value (Every n-th row)",
            "Average over Past n rows",
            "Average over Future n rows",
            "Sum of the Past n values",
        ],
    )

    window_direction = st.radio(
        "Window Direction",
        ["Future (Forward from Start Index)", "Past (Backward from Start Index)"],
        horizontal=True,
        help="Future takes rows from start index onwards. Past takes rows ending at start index.",
    )

    if window_direction == "Future (Forward from Start Index)":
        max_start_idx = max(0, max_total_rows - window_size)
        default_start_idx = 0
    else:
        max_start_idx = max(window_size - 1, max_total_rows - 1)
        default_start_idx = min(window_size - 1, max_start_idx)

    start_mode = st.radio(
        "Specify Window Start By:", ["Row Index (Integer)", "Timestamp Selection"], horizontal=True
    )

    if hasattr(raw_df.index, "__len__") and len(raw_df.index) > 0:
        sidebar_ts_options = raw_df.index.tolist()
        selected_timestamp = st.selectbox("Select Start Timestamp", options=sidebar_ts_options, index=0)
        start_idx_ts = int(raw_df.index.get_loc(selected_timestamp))
    else:
        start_idx_ts = default_start_idx

    min_row_val = (
        0
        if window_direction == "Future (Forward from Start Index)"
        else min(window_size - 1, max_total_rows - 1)
    )
    start_idx_num = st.number_input(
        "Window Start Index (Integer)",
        min_value=min_row_val,
        max_value=max(0, max_start_idx),
        value=default_start_idx,
        step=1,
    )

    if window_direction == "Future (Forward from Start Index)":
        start_idx = (
            min(start_idx_ts, max_start_idx)
            if start_mode == "Timestamp Selection"
            else min(start_idx_num, max_start_idx)
        )
    else:
        min_bwd = min(window_size - 1, max_total_rows - 1)
        start_idx = (
            max(min_bwd, min(start_idx_ts, max_start_idx))
            if start_mode == "Timestamp Selection"
            else max(min_bwd, min(start_idx_num, max_start_idx))
        )

    standardize_data = st.checkbox(
        "Standardize Data globally (Global Z-score: Mean=0, Std=1)",
        value=True,
        help="Centers data around zero mean and scales to unit variance across the dataset.",
    )

    st.subheader("2. Scale-Aware TDA Filtering")
    thresh_type = st.radio(
        "Threshold Mode", ["Absolute (Fine Decimal)", "Relative (% of Max Persistence)"], horizontal=True
    )

    if thresh_type == "Absolute (Fine Decimal)":
        min_persistence_thresh = st.number_input(
            "Min Persistence Cutoff",
            min_value=0.0,
            max_value=100.0,
            value=0.0005,
            format="%.5f",
            step=0.0001,
        )
    else:
        rel_persistence_pct = st.slider(
            "Min Persistence (% of Max Diagram Persistence)", 0, 50, 5, step=1
        )
        min_persistence_thresh = 0.0

    protect_critical_edges = st.checkbox("Protect Birth & Death Vertices from Pruning", value=True)

    st.subheader("3. Distance & Persistence Settings")
    
    wasserstein_order = st.number_input(
        "Wasserstein Distance Order ($p$)",
        min_value=1.0,
        max_value=5.0,
        value=1.0,
        step=1.0,
        help="Exponent p for p-Wasserstein distance metric.",
        key="wasserstein_order_input"
    )
    
    landscape_lp_order = st.number_input(
        "Persistence Landscape Norm Order ($p$)",
        min_value=1.0,
        max_value=5.0,
        value=2.0,
        step=1.0,
        help="Exponent p for the generalized Lp norm of persistence landscapes.",
        key="landscape_lp_order_input"
    )
    
    # Save to session state for global access across sections
    st.session_state["wasserstein_order"] = wasserstein_order
    st.session_state["landscape_lp_order"] = landscape_lp_order

    submit_button = st.form_submit_button(label=" Run Analysis / Compute TDA")

if max_total_rows < window_size:
    st.error(
        f" Dataset length ({max_total_rows}) is smaller than window size ({window_size}). Please reduce window size or provide more data."
    )
    st.stop()

if homology_dim == 1 and len(selected_dims) < 2:
    st.error(" $H_1$ homology (loops) requires at least **2 active dimensions**.")
    st.stop()
elif homology_dim == 2 and len(selected_dims) < 3:
    st.error(" $H_2$ homology (3D voids/cavities) requires at least **3 active dimensions**.")
    st.stop()

# =============================================================================
# INITIALIZES THE TDA PIPELINE & CACHE IN SESSION STATE
# =============================================================================
pipeline = TDAPipeline(
    df=raw_df,
    selected_dims=selected_dims,
    start_idx=start_idx,
    window_size=window_size,
    pc_stride=int(pc_stride),
    stride_agg_mode=stride_agg_mode,
    window_direction=window_direction,
    standardize=standardize_data,
    homology_dim=homology_dim,
)

current_config_key = f"{start_idx}_{window_size}_{pc_stride}_{stride_agg_mode}_{window_direction}_{selected_dims}_{standardize_data}_{thresh_type}_{min_persistence_thresh}_{rel_persistence_pct}_{homology_dim}_{wasserstein_order}"
if (
    "last_config_key" not in st.session_state
    or st.session_state["last_config_key"] != current_config_key
):
    st.session_state["last_config_key"] = current_config_key
    for key in ["ran_sec1", "ran_sec2", "ran_sec3", "ran_sec4"]:
        st.session_state[key] = False

raw_cycles = pipeline.compute_raw_cycles()
if window_direction == "Future (Forward from Start Index)":
    _win = raw_df[selected_dims].iloc[start_idx : start_idx + int(window_size)]
else:
    _start = max(0, start_idx - int(window_size) + 1)
    _win = raw_df[selected_dims].iloc[_start : start_idx + 1]


if raw_cycles:
    max_diag_p = max([c["persistence"] for c in raw_cycles])
    effective_thresh = (
        max_diag_p * (rel_persistence_pct / 100.0)
        if thresh_type == "Relative (% of Max Persistence)"
        else min_persistence_thresh
    )
    cycles = [c for c in raw_cycles if c["persistence"] >= effective_thresh]
else:
    cycles = []
    max_diag_p = 0.0
    effective_thresh = 0.0

st.sidebar.info(
    f" Max Diagram Persistence: `{max_diag_p:.5f}` | Applied Cutoff: `{effective_thresh:.5f}`"
)

def page_1_tda():
    # =============================================================================
    # SECTION 1: PERSISTENCE CYCLES TABLE & SUMMARY STATISTICS FOR SELECTED CYCLE
    # =============================================================================
    st.markdown("---")
    st.header(f"1. $H_{homology_dim}$ Persistence Features Table & Summary Statistics")
    
    if st.button(f"🚀 Run Section 1 (H_{homology_dim} Table)", key="button1") or submit_button:
        st.session_state["ran_sec1"] = True
    
    if st.session_state.get("ran_sec1", False):
        if not cycles:
            st.warning(f"No topological features ($H_{homology_dim}$) exceed the current persistence threshold.")
        else:
            c_pers = [c["persistence"] for c in cycles]
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Features Count", len(cycles))
            m2.metric("Total Persistence", f"{np.sum(c_pers):.4f}")
            m3.metric("Mean Persistence", f"{np.mean(c_pers):.4f}")
            m4.metric("Median Persistence", f"{np.median(c_pers):.4f}")
            m5.metric("Max Persistence", f"{np.max(c_pers):.4f}")
            m6.metric("Std Dev", f"{np.std(c_pers):.4f}")
    
            cycles_df = pd.DataFrame(cycles)
            st.dataframe(
                cycles_df[
                    [
                        "id",
                        "birth",
                        "death",
                        "persistence",
                        "peak_time",
                        "peak_height",
                        "n_points",
                    ]
                ],
                use_container_width=True,
            )
    
    # =============================================================================
    # SECTION 2: BETTI CURVES
    # =============================================================================
    st.markdown("---")
    st.header(f"2. Betti Curves ($B_{homology_dim}$)")
    
    if st.button(" Run Section 2 (Betti Curves)", key="button2_betti") or submit_button:
     st.session_state["ran_sec2_betti"] = True
    
    if st.session_state.get("ran_sec2_betti", False):
        if not cycles:
            st.warning("No topological features available to plot Betti curves.")
        else:
            t_grid, betti_vals = compute_betti_curve(cycles)
            if len(t_grid) > 0:
                fig_betti, ax_b = plt.subplots(figsize=(10, 3))
                ax_b.step(t_grid, betti_vals, where="mid", color="purple", linewidth=2)
                ax_b.fill_between(t_grid, betti_vals, step="mid", alpha=0.2, color="purple")
                ax_b.set_title(
                    r"Betti Curve ($\beta_%d(t)$): Active Topological Features vs. Filtration Distance ($t$)"
                    % homology_dim
                )
                ax_b.set_xlabel("Filtration Distance ($t$)")
                ax_b.set_ylabel(r"$\beta_%d(t)$" % homology_dim)
                ax_b.grid(True, linestyle="--", alpha=0.5)
                st.pyplot(fig_betti)
            else:
                st.info("Insufficient finite death intervals for Betti curve calculation.")
    
    # =========================================================================
    # SECTION 3: PERSISTENCE LANDSCAPE & LP NORM
    # =========================================================================
    st.markdown("---")
    st.header(f"3. Persistence Landscape & Norm ($H_{homology_dim}$)")

    p = st.session_state.get("landscape_lp_order", 2.0)
    st.session_state.setdefault("landscape_n_layers", 2)
    st.session_state.setdefault("landscape_num_pts", 200)

    # Restore widget keys if they were pruned while on another page
    st.session_state.setdefault("global_n_layers", st.session_state["landscape_n_layers"])
    st.session_state.setdefault("global_num_pts", st.session_state["landscape_num_pts"])

    with st.expander(" Landscape Parameter Configuration", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            n_layers_val = st.slider(
                "Landscape Layers ($k$)",
                min_value=1,
                max_value=5,
                step=1,
                key="global_n_layers",
                help="Number of landscape layers (lambda_1, lambda_2, ...) to evaluate.",
            )
            st.session_state["landscape_n_layers"] = n_layers_val
        with col2:
            num_pts_val = st.slider(
                "Landscape Bins / Resolution",
                min_value=50,
                max_value=500,
                step=50,
                key="global_num_pts",
                help="Number of evaluation points (bins) along the time axis.",
            )
            st.session_state["landscape_num_pts"] = num_pts_val

    form_submitted = st.session_state.get("tda_form_submitted", False)
    if st.button("Run Section 3 (Persistence Landscape & Norm)", key="button2") or form_submitted:
        st.session_state["ran_sec2"] = True

    if st.session_state.get("ran_sec2", False):
        if not cycles:
            st.warning("No topological features available to plot landscape.")
        else:
            # 1. Plotting the Landscape Tent Functions
            fig, ax = plt.subplots(figsize=(10, 3))
            for c in cycles:
                b, d = c["birth"], c["death"]
                ax.plot(
                    [b, c["peak_time"], d],
                    [0, c["peak_height"], 0],
                    label=f"C{c['id']}",
                    linewidth=2,
                )
                ax.scatter([c["peak_time"]], [c["peak_height"]], s=40)
                ax.annotate(
                    f"C{c['id']}",
                    (c["peak_time"], c["peak_height"]),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                )

            ax.set_title(f"Persistence Landscape Tent Functions ($H_{homology_dim}$)")
            ax.set_xlabel("Filtration Distance ($t$)")
            ax.set_ylabel(r"Height ($\lambda$)")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="upper right")
            st.pyplot(fig)

            # 2. Compute the Lp Norm using the persistent session state values
            active_layers = st.session_state["landscape_n_layers"]
            active_bins = st.session_state["landscape_num_pts"]

            landscape_norm = compute_landscape_lp_norm(
                cycles=cycles,
                num_pts=active_bins,
                n_layers=active_layers,
                p=p,
            )

            st.success(
                f"**Computed Persistence Landscape $L^p$ Norm ($k$={active_layers}, bins={active_bins}):** `{landscape_norm:.4f}`"
            )
            st.session_state["computed_landscape_norm"] = landscape_norm
    
    # =============================================================================
    # SECTION 4: PERSISTENCE IMAGES & MANUAL GRID-CELL INSPECTOR
    # =============================================================================
    st.markdown("---")
    st.header("4. Persistence Image & Manual Grid-Cell Inspector")
    st.markdown("Configure persistence image resolution, Gaussian smoothing, and filter weighting functions below.")
    
    col_pi1, col_pi2 = st.columns(2)
    st.session_state.setdefault("pi_res", None)
    st.session_state.setdefault("pi_sigma", None)
    st.session_state.setdefault("pi_res_input", st.session_state["pi_res"])
    st.session_state.setdefault("pi_sigma_input", st.session_state["pi_sigma"])
    with col_pi1:
        pi_res = st.number_input(
            "Persistence Image Resolution", min_value=10, max_value=50, value=25, step=1, key="pi_res_input"
        
        )
        

        
        st.session_state["pi_res"] = pi_res
            
    st.session_state.setdefault("pi_sigma", None)
    with col_pi2:
        pi_sigma = st.number_input(
            "Gaussian Spread ($\sigma$)",
            min_value=0.0001,
            max_value=100.0,
            value=0.15,
            step=0.0001,
            format="%.4f",
            key="pi_sigma_input"
        )
        
        
        st.session_state["pi_sigma"]=pi_sigma
    
    # --- Filter Function Settings (Moved from Sidebar / Section 3 into Section 4) ---
    st.markdown("####  Persistence Image Filter Function (Color Multiplier)")
    pi_filter_mode = st.selectbox(
        "Filter Function Mode",
        [
            "Persistence Length (d - b)",
            "None (Constant = 1.0)",
            "Normalized Persistence",
            "Custom Formula",
        ],
        key="pi_filter_mode_select"
    )
    custom_filter_expr = (
        st.text_input("Custom Expression (use 'b', 'p')", value="p**2", key="custom_filter_input")
        if pi_filter_mode == "Custom Formula"
        else "p"
    )
    
    if st.button(" Run Section 4 (Persistence Image & Inspector)", key="button4") or submit_button:
        st.session_state["ran_sec3"] = True
    
    if st.session_state.get("ran_sec3", False):
        if not cycles:
            st.warning("No topological features available for Persistence Image.")
        else:
            pi_matrix, b_bins, p_bins, cell_map = generate_persistence_image(
                cycles,
                res=pi_res,
                sigma=pi_sigma,
                filter_mode=pi_filter_mode,
                custom_expr=custom_filter_expr,
            )
    
            res_dim = pi_matrix.shape[0]
            hover_text_grid = []
            for r_idx in range(res_dim):
                row_texts = []
                for c_idx in range(res_dim):
                    flat_index = (r_idx * res_dim) + c_idx
                    val = pi_matrix[r_idx, c_idx]
                    b_val = b_bins[c_idx]
                    p_val = p_bins[res_dim - 1 - r_idx]
                    row_texts.append(
                        f"Row (r): {r_idx}<br>"
                        f"Col (c): {c_idx}<br>"
                        f"<b>Flattened Index: {flat_index}</b><br>"
                        f"Birth ($b$): {b_val:.4f}<br>"
                        f"Persistence ($d-b$): {p_val:.4f}<br>"
                        f"Pixel Value: {val:.5f}"
                    )
                hover_text_grid.append(row_texts)
    
            fig_pi = go.Figure(
                data=go.Heatmap(
                    z=pi_matrix,
                    x=np.round(b_bins, 5),
                    y=np.round(p_bins[::-1], 5),
                    colorscale="Viridis",
                    text=hover_text_grid,
                    hoverinfo="text",
                )
            )
            fig_pi.update_layout(
                title=f"Persistence Image (Filter: {pi_filter_mode}, $\sigma$={pi_sigma}) — Hover for Row, Col & Flattened Index",
                xaxis_title="Birth ($b$)",
                yaxis_title="Persistence ($d - b$)",
                height=500,
            )
            st.plotly_chart(fig_pi, use_container_width=True)
    
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                sel_row = st.number_input(
                    "Grid Pixel Row", min_value=0, max_value=pi_res - 1, value=pi_res // 2, step=1, key="pi_sel_row"
                )
            with col_g2:
                sel_col = st.number_input(
                    "Grid Pixel Column", min_value=0, max_value=pi_res - 1, value=pi_res // 2, step=1, key="pi_sel_col"
                )
    
            st.subheader(
                f" Pixel Cell Breakdown (Row: {sel_row}, Col: {sel_col} | Flat Index: {(sel_row * pi_res) + sel_col})"
            )
            contributing_details = cell_map.get((sel_row, sel_col), [])
            if contributing_details:
                st.dataframe(pd.DataFrame(contributing_details), use_container_width=True)
            else:
                st.info("No features contribute to this specific pixel grid cell.")
                
    # =============================================================================
    # SECTION 5: DYNAMIC POINT CLOUD VISUALIZATION & TRACEBACKS
    # =============================================================================
    st.markdown("---")
    st.header("5. Dynamic Point Cloud & Topological Highlighting")
    
    if st.button(" Run Section 5 (Point Cloud Highlighting & Traceback)", key="button5") or submit_button:
        st.session_state["ran_sec5"] = True
    
    if st.session_state.get("ran_sec5", False):
        if not cycles:
            st.warning("No features available for point cloud projection.")
        else:
            col_sel1, col_sel2 = st.columns(2)
            with col_sel1:
                selected_cycle_id = st.selectbox("Select Feature to Inspect", options=[c["id"] for c in cycles])
            with col_sel2:
                if homology_dim == 0:
                    highlight_mode = st.radio("Highlight Component:", ["Component Birth Point (1 pt)", "Component Merging Edge (Death Edge)", "Immediate Pre-Death Neighbors"], horizontal=True)
                elif homology_dim == 1:
                    highlight_mode = st.radio("Highlight Component:", ["Full 360° Circle Ring", "Rigorously Pruned Minimal Skeleton", "Minimal Cycle + Birth Neighbors", "Birth Edge", "Death Edge"], horizontal=True)
                else:
                    highlight_mode = st.radio("Highlight Component:", ["Full Cavity Shell (>= 6 pts)", "Rigorously Pruned Minimal Cavity (>= 6 pts)", "Birth Edge", "Death Edge"], horizontal=True)
    
            selected_cycle = next(c for c in cycles if c["id"] == selected_cycle_id)
            rigorous_skeleton = pipeline.extract_minimal_skeleton(selected_cycle, protect_critical=protect_critical_edges)
    
            birth_dist = selected_cycle["birth"]
            expanded_neighbors = set(rigorous_skeleton)
            for v in rigorous_skeleton:
                for j in range(pipeline.n_points):
                    if pipeline.distance_matrix[v, j] <= birth_dist:
                        expanded_neighbors.add(j)
            expanded_neighbors_list = sorted(list(expanded_neighbors))
    
            st.info(f" Projecting from active TDA pipeline space (`{', '.join(selected_dims)}`):")
            
            vis_dim_count = 2 if len(selected_dims) == 2 else st.radio("Visualization Mode", [2, 3], index=1, horizontal=True)
    
            plot_pc = pipeline.point_cloud if vis_dim_count == 3 else pipeline.point_cloud[:, :2]
    
            if vis_dim_count == 2:
                p1, p2 = st.columns(2)
                with p1:
                    plot_x_idx = st.selectbox("X-Axis Dimension", options=range(len(selected_dims)), format_func=lambda x: selected_dims[x], index=0, key="vis_x_2d")
                with p2:
                    plot_y_idx = st.selectbox("Y-Axis Dimension", options=range(len(selected_dims)), format_func=lambda x: selected_dims[x], index=min(1, len(selected_dims)-1), key="vis_y_2d")
                plot_coords = plot_pc[:, [plot_x_idx, plot_y_idx]]
                plot_labels = [selected_dims[plot_x_idx], selected_dims[plot_y_idx]]
            else:
                p1, p2, p3 = st.columns(3)
                with p1:
                    plot_x_idx = st.selectbox("X-Axis Dimension", options=range(len(selected_dims)), format_func=lambda x: selected_dims[x], index=0, key="vis_x_3d")
                with p2:
                    plot_y_idx = st.selectbox("Y-Axis Dimension", options=range(len(selected_dims)), format_func=lambda x: selected_dims[x], index=min(1, len(selected_dims)-1), key="vis_y_3d")
                with p3:
                    plot_z_idx = st.selectbox("Z-Axis Dimension", options=range(len(selected_dims)), format_func=lambda x: selected_dims[x], index=min(2, len(selected_dims)-1), key="vis_z_3d")
                plot_coords = plot_pc[:, [plot_x_idx, plot_y_idx, plot_z_idx]]
                plot_labels = [selected_dims[plot_x_idx], selected_dims[plot_y_idx], selected_dims[plot_z_idx]]
    
            marker_colors = ["rgba(70, 130, 180, 0.85)"] * pipeline.n_points
            marker_sizes = [6] * pipeline.n_points
            marker_symbols = ["circle"] * pipeline.n_points
    
            highlight_indices = []
            if homology_dim == 0:
                if highlight_mode == "Component Birth Point (1 pt)":
                    highlight_indices = selected_cycle["vertices"]
                    title_suffix = f"Component Birth Point ({len(highlight_indices)} point)"
                elif highlight_mode == "Component Merging Edge (Death Edge)":
                    highlight_indices = selected_cycle["death_edge"]
                    title_suffix = f"Component Merging Edge at t={selected_cycle['death']:.5f}"
                else:
                    highlight_indices = selected_cycle["pre_death_neighbors"]
                    title_suffix = f"Immediate Pre-Death Neighbors (dist <= {selected_cycle['death']:.4f}, {len(highlight_indices)} points)"
            elif homology_dim == 1:
                if highlight_mode == "Full 360° Circle Ring":
                    highlight_indices = selected_cycle["vertices"]
                    title_suffix = f"Full 360° Circle Ring ({len(highlight_indices)} points)"
                elif highlight_mode == "Rigorously Pruned Minimal Skeleton":
                    highlight_indices = rigorous_skeleton
                    title_suffix = f"Rigorously Pruned Minimal Skeleton ({len(highlight_indices)} points)"
                elif highlight_mode == "Minimal Cycle + Birth Neighbors":
                    highlight_indices = expanded_neighbors_list
                    title_suffix = f"Minimal Cycle + Birth Neighbors (dist <= {birth_dist:.4f}, {len(highlight_indices)} points)"
                elif highlight_mode == "Birth Edge":
                    highlight_indices = selected_cycle["birth_edge"]
                    title_suffix = f"Birth Edge at t={selected_cycle['birth']:.5f}"
                else:
                    highlight_indices = selected_cycle["death_edge"]
                    title_suffix = f"Death Edge at t={selected_cycle['death']:.5f}"
            else:
                if highlight_mode == "Full Cavity Shell (>= 6 pts)":
                    highlight_indices = selected_cycle["vertices"]
                    title_suffix = f"Full Cavity Shell ({len(highlight_indices)} points)"
                elif highlight_mode == "Rigorously Pruned Minimal Cavity (>= 6 pts)":
                    highlight_indices = rigorous_skeleton
                    title_suffix = f"Minimal Cavity Shell ({len(highlight_indices)} points)"
                elif highlight_mode == "Birth Edge":
                    highlight_indices = selected_cycle["birth_edge"]
                    title_suffix = f"Birth Edge at t={selected_cycle['birth']:.5f}"
                else:
                    highlight_indices = selected_cycle["death_edge"]
                    title_suffix = f"Death Edge at t={selected_cycle['death']:.5f}"
    
            for idx in highlight_indices:
                if idx < len(marker_colors):
                    marker_colors[idx] = "rgba(255, 0, 0, 0.95)"
                    marker_sizes[idx] = 10
                    marker_symbols[idx] = "diamond"
    
            if vis_dim_count == 2:
                fig_proj = go.Figure(
                    data=[
                        go.Scatter(
                            x=plot_coords[:, 0],
                            y=plot_coords[:, 1],
                            mode="lines",
                            marker=dict(size=marker_sizes, color=marker_colors, symbol=marker_symbols),
                            text=[
                                f"Array Index: {i}<br>Dataset Row: {pipeline.window_df.index.get_loc(pipeline.window_df.index[i])}<br>Time: {pipeline.window_df.index[i]}"
                                for i in range(pipeline.n_points)
                            ],
                            hoverinfo="text",
                        )
                    ]
                )
                fig_proj.update_layout(
                    title=f"2D Projection ({', '.join(plot_labels)}) — Highlighted: {title_suffix}",
                    xaxis_title=plot_labels[0],
                    yaxis_title=plot_labels[1],
                    margin=dict(l=0, r=0, b=0, t=40),
                    height=500,
                )
            else:
                fig_proj = go.Figure(
                    data=[
                        go.Scatter3d(
                            x=plot_coords[:, 0],
                            y=plot_coords[:, 1],
                            z=plot_coords[:, 2],
                            mode="lines",
                            marker=dict(size=marker_sizes, color=marker_colors, symbol=marker_symbols),
                            text=[
                                f"Array Index: {i}<br>Dataset Row: {pipeline.window_df.index.get_loc(pipeline.window_df.index[i])}<br>Time: {pipeline.window_df.index[i]}"
                                for i in range(pipeline.n_points)
                            ],
                            hoverinfo="text",
                        )
                    ]
                )
                fig_proj.update_layout(
                    title=f"3D Projection ({', '.join(plot_labels)}) — Highlighted: {title_suffix}",
                    scene=dict(
                        xaxis_title=plot_labels[0],
                        yaxis_title=plot_labels[1],
                        zaxis_title=plot_labels[2],
                    ),
                    margin=dict(l=0, r=0, b=0, t=40),
                    height=600,
                )
    
            st.plotly_chart(fig_proj, use_container_width=True)
    
            st.markdown("###  Traceback Index & Coordinate Tables")
    
           
            
    
            with st.expander(f" Feature Associated Points ({len(selected_cycle['vertices'])} points)", expanded=False):
                st.dataframe(build_traceback_dataframe(pipeline.window_df,selected_cycle["vertices"],selected_dims), use_container_width=True)
    
            if homology_dim == 0:
                with st.expander(
                    f" Immediate Pre-Death Neighbors ({len(selected_cycle['pre_death_neighbors'])} points)",
                    expanded=False,
                ):
                    st.dataframe(
                        build_traceback_dataframe(pipeline.window_df,selected_cycle["pre_death_neighbors"],selected_dims),
                        use_container_width=True,
                    )
            else:
                with st.expander(
                    f" Rigorously Pruned Minimal Skeleton ({len(rigorous_skeleton)} points)",
                    expanded=False,
                ):
                    st.dataframe(build_traceback_dataframe(pipeline.window_df,rigorous_skeleton,selected_dims), use_container_width=True)
    
            with st.expander(f" Birth Edge / Vertex ({len(selected_cycle['birth_edge'])} points)", expanded=False):
                st.dataframe(build_traceback_dataframe(pipeline.window_df,selected_cycle['birth_edge'],selected_dims), use_container_width=True)
    
            with st.expander(f" Pre-Death Edge (Last structural edge before collapse) ({len(selected_cycle['pre_death_edge'])} points)", expanded=False):
                st.dataframe(build_traceback_dataframe(pipeline.window_df,selected_cycle['pre_death_edge'],selected_dims), use_container_width=True)
    
            with st.expander(f" Actual Death Edge (Collapsing edge at t={selected_cycle['death']:.5f}) ({len(selected_cycle['death_edge'])} points)", expanded=False):
                st.dataframe(build_traceback_dataframe(pipeline.window_df,selected_cycle['death_edge'],selected_dims), use_container_width=True)
    
    # =============================================================================
    # SECTION 6: DIMENSION ABLATION ANALYSIS
    # =============================================================================
    st.markdown("---")
    st.header("6. Custom & Automated Dimension Ablation Analysis")
    
    ablation_mode = st.radio(
        "Ablation Mode", ["Automated Leave-One-Out Scan", "Manual Feature Selection"], horizontal=True
    )
    
    if ablation_mode == "Automated Leave-One-Out Scan":
        min_dims_for_ablation = 2 if homology_dim == 0 else (3 if homology_dim == 1 else 4)
        if len(selected_dims) < min_dims_for_ablation:
            st.warning(
                f" Automated Leave-One-Out Scan for $H_{homology_dim}$ requires at least **{min_dims_for_ablation} active dimensions**."
            )
        else:
            if st.button(" Run Automated Ablation Analysis Across All Dimensions"):
                if not cycles:
                    st.warning("No base features found to ablate.") 
                else:
                    selected_cycle_for_ablation = cycles[0]
                    rigorous_skeleton_ablation = pipeline.extract_minimal_skeleton(
                        selected_cycle_for_ablation, protect_critical=protect_critical_edges
                    )
    
                    st.subheader(" Automated Ablation Continuous Statistics")
                    scan_results = []
                    subspace_store = {}
    
                    for dim_to_drop in selected_dims:
                        sub_dims = [d for d in selected_dims if d != dim_to_drop]
                        if len(sub_dims) < (1 if homology_dim == 0 else (2 if homology_dim == 1 else 3)):
                            continue
                        sub_pipeline = TDAPipeline(
                            df=pipeline.df,
                            selected_dims=sub_dims,
                            start_idx=pipeline.start_idx,
                            window_size=pipeline.window_size,
                            pc_stride=int(pc_stride),
                            stride_agg_mode=stride_agg_mode,
                            window_direction=window_direction,
                            standardize=standardize_data,
                            homology_dim=homology_dim,
                        )
                        sub_cycles = sub_pipeline.compute_raw_cycles()
    
                        match_res = match_cycle_across_spaces(
                            selected_cycle_for_ablation,
                            rigorous_skeleton_ablation,
                            sub_pipeline,
                            sub_cycles,
                        )
    
                        scan_results.append(
                            {
                                "Dropped Dimension": dim_to_drop,
                                "Matched Feature ID": f"C{match_res['matched_cycle_id']}"
                                if match_res["matched_cycle_id"] is not None
                                else "None",
                                "Original Interval": match_res["orig_interval"],
                                "Ablated Interval": match_res["sub_interval"],
                                "Full Feature Overlap (%)": match_res["full_overlap"],
                                "Minimal Skeleton Overlap (%)": match_res["skeleton_overlap"],
                                "Interval Overlap (%)": match_res["interval_overlap_pct"],
                            }
                        )
    
                        if match_res["matched_cycle_id"] is not None:
                            subspace_store[dim_to_drop] = {
                                "sub_pipeline": sub_pipeline,
                                "matched_cycle": next(
                                    sc for sc in sub_cycles if sc["id"] == match_res["matched_cycle_id"]
                                ),
                            }
    
                    st.dataframe(pd.DataFrame(scan_results), use_container_width=True)
                    if subspace_store:
                        st.session_state["subspace_store"] = subspace_store
    
        if "subspace_store" in st.session_state and st.session_state["subspace_store"]:
            st.markdown("---")
            st.subheader(" Visualizing Subspaces")
            subspace_store = st.session_state["subspace_store"]
            selected_preserved_drop = st.selectbox(
                "Select Dropped Dimension for Subspace Plot:",
                options=list(subspace_store.keys()),
                key="preserved_drop_select",
            )
    
            sub_info = subspace_store[selected_preserved_drop]
            sub_pipe = sub_info["sub_pipeline"]
            sub_matched_cycle = sub_info["matched_cycle"]
    
            sub_plot_dims = (
                sub_pipe.selected_dims[:2]
                if len(sub_pipe.selected_dims) == 2
                else sub_pipe.selected_dims[:3]
            )
            sub_plot_pc = (
                sub_pipe.point_cloud
                if len(sub_plot_dims) == 3
                else sub_pipe.point_cloud[:, :2]
            )
    
            sub_colors = ["rgba(70, 130, 180, 0.85)"] * sub_pipe.n_points
            sub_sizes = [6] * sub_pipe.n_points
            sub_symbols = ["circle"] * sub_pipe.n_points
    
            for idx in sub_matched_cycle["vertices"]:
                if idx < len(sub_colors):
                    sub_colors[idx] = "rgba(255, 0, 0, 0.95)"
                    sub_sizes[idx] = 10
                    sub_symbols[idx] = "diamond"
    
            if len(sub_plot_dims) == 2:
                fig_sub = go.Figure(
                    data=[
                        go.Scatter(
                            x=sub_plot_pc[:, 0],
                            y=sub_plot_pc[:, 1],
                            mode="lines",
                            marker=dict(size=sub_sizes, color=sub_colors, symbol=sub_symbols),
                            text=[
                                f"Subspace Index: {i}<br>Dataset Row: {sub_pipe.window_df.index.get_loc(sub_pipe.window_df.index[i])}"
                                for i in range(sub_pipe.n_points)
                            ],
                            hoverinfo="text",
                        )
                    ]
                )
                fig_sub.update_layout(
                    title=f"Ablated Subspace (Dropped: {selected_preserved_drop}) — 2D Projection ({', '.join(sub_plot_dims)})",
                    xaxis_title=sub_plot_dims[0],
                    yaxis_title=sub_plot_dims[1],
                    height=500,
                )
            else:
                fig_sub = go.Figure(
                    data=[
                        go.Scatter3d(
                            x=sub_plot_pc[:, 0],
                            y=sub_plot_pc[:, 1],
                            z=sub_plot_pc[:, 2],
                            mode="lines",
                            marker=dict(size=sub_sizes, color=sub_colors, symbol=sub_symbols),
                            text=[
                                f"Subspace Index: {i}<br>Dataset Row: {sub_pipe.window_df.index.get_loc(sub_pipe.window_df.index[i])}"
                                for i in range(sub_pipe.n_points)
                            ],
                            hoverinfo="text",
                        )
                    ]
                )
                fig_sub.update_layout(
                    title=f"Ablated Subspace (Dropped: {selected_preserved_drop}) — 3D Projection ({', '.join(sub_plot_dims)})",
                    scene=dict(
                        xaxis_title=sub_plot_dims[0],
                        yaxis_title=sub_plot_dims[1],
                        zaxis_title=sub_plot_dims[2],
                    ),
                    height=600,
                )
            st.plotly_chart(fig_sub, use_container_width=True)
    
    else:
        dims_to_remove = st.multiselect(
            "Select Dimensions to Remove", options=selected_dims, default=[], key="manual_dims_remove"
        )
        if st.button("Run Manual Ablation Test"):
            if not dims_to_remove:
                st.warning("Please select at least one dimension to remove.")
            elif not cycles:
                st.warning("No features available for ablation.")
            else:
                selected_cycle_for_ablation = cycles[0]
                rigorous_skeleton_ablation = pipeline.extract_minimal_skeleton(
                    selected_cycle_for_ablation, protect_critical=protect_critical_edges
                )
                sub_dims = [d for d in selected_dims if d not in dims_to_remove]
                min_req_dims = 1 if homology_dim == 0 else (2 if homology_dim == 1 else 3)
                if len(sub_dims) < min_req_dims:
                    st.error(
                        f"Cannot run ablation: fewer than {min_req_dims} dimensions remaining."
                    )
                else:
                    sub_pipeline = TDAPipeline(
                        df=pipeline.df,
                        selected_dims=sub_dims,
                        start_idx=pipeline.start_idx,
                        window_size=pipeline.window_size,
                        pc_stride=int(pc_stride),
                        stride_agg_mode=stride_agg_mode,
                        window_direction=window_direction,
                        standardize=standardize_data,
                        homology_dim=homology_dim,
                    )
                    sub_cycles = sub_pipeline.compute_raw_cycles()
                    match_res = match_cycle_across_spaces(
                        selected_cycle_for_ablation,
                        rigorous_skeleton_ablation,
                        sub_pipeline,
                        sub_cycles,
                    )
    
                    res1, res2, res3, res4 = st.columns(4)
                    with res1:
                        st.metric(
                            "Best Match ID",
                            f"C{match_res['matched_cycle_id']}"
                            if match_res["matched_cycle_id"] is not None
                            else "None",
                        )
                    with res2:
                        st.metric("Full Overlap", f"{match_res['full_overlap']}%")
                    with res3:
                        st.metric("Skeleton Overlap", f"{match_res['skeleton_overlap']}%")
                    with res4:
                        st.metric("Interval Overlap", f"{match_res['interval_overlap_pct']}%")
    
                    st.info(
                        f" **Persistence Intervals:** Original Space: `{match_res['orig_interval']}` vs. Ablated Subspace: `{match_res['sub_interval']}`"
                    )
    
    # =============================================================================
    # SECTION 7: SUCCESSIVE WINDOW PERSISTENCE ANALYSIS
    # =============================================================================
    st.markdown("---")
    st.header("7. Successive Window Persistence Analysis & Distances")
    
    # Pull global landscape parameters from session state (initialized/configured in Section 3)
    active_layers = st.session_state.get("global_n_layers", 2)
    active_num_pts = st.session_state.get("global_num_pts", 200)
    
    st.markdown(
        f"Compute persistence homology, persistence images, pairwise Wasserstein distances (order $p$), and landscape $L_p$ norms computed with **{active_layers} layers** and **{active_num_pts} bins** across successive windows."
    )
    
    col_sw1, col_sw2, col_sw3 = st.columns(3)
    with col_sw1:
        n_successive_windows = st.number_input(
            "Number of Successive Windows", min_value=1, max_value=10, value=3, step=1
        )
    with col_sw2:
        min_sw_start = (
            0 if window_direction == "Future (Forward from Start Index)" else window_size - 1
        )
        max_sw_start = (
            max(0, max_total_rows - window_size)
            if window_direction == "Future (Forward from Start Index)"
            else max_total_rows - 1
        )
        successive_start_row = st.number_input(
            "Window Sequence Start Index",
            min_value=min_sw_start,
            max_value=max_sw_start,
            value=min_sw_start,
            step=1,
        )
    with col_sw3:
        successive_step_size = st.number_input(
            "Step Size / Window Offset", min_value=1, max_value=window_size, value=window_size, step=1
        )
    
    if st.button(" Run Successive Windows Persistence Scan"):
        successive_results = []
    
        for i in range(n_successive_windows):
            curr_start = successive_start_row + (i * successive_step_size)
            if (
                window_direction == "Future (Forward from Start Index)"
                and curr_start + window_size > len(raw_df)
            ):
                st.warning(f" Successive Window {i+1} exceeds dataset boundary. Stopping scan early.")
                break
            elif window_direction == "Past (Backward from Start Index)" and curr_start >= len(raw_df):
                st.warning(f" Successive Window {i+1} exceeds dataset boundary. Stopping scan early.")
                break
    
            sub_pipe = TDAPipeline(
                df=raw_df,
                selected_dims=selected_dims,
                start_idx=curr_start,
                window_size=window_size,
                pc_stride=int(pc_stride),
                stride_agg_mode=stride_agg_mode,
                window_direction=window_direction,
                standardize=standardize_data,
                homology_dim=homology_dim,
            )
            raw_c = sub_pipe.compute_raw_cycles()
    
            if raw_c:
                max_p_w = max([c["persistence"] for c in raw_c])
                eff_thresh_w = (
                    max_p_w * (rel_persistence_pct / 100.0)
                    if thresh_type == "Relative (% of Max Persistence)"
                    else min_persistence_thresh
                )
                w_cycles = [c for c in raw_c if c["persistence"] >= eff_thresh_w]
                w_diag = np.array(
                    [[c["birth"], c["death"]] for c in raw_c if not np.isinf(c["death"])]
                )
            else:
                w_cycles = []
                max_p_w = 0.0
                w_diag = np.empty((0, 2))
    
            w_matrix, w_b_bins, w_p_bins, _ = generate_persistence_image(
                w_cycles,
                res=pi_res,
                sigma=pi_sigma,
                filter_mode=pi_filter_mode,
                custom_expr=custom_filter_expr,
            )
    
            # Compute Lp Norm dynamically using global session state variables
            landscape_norm = compute_landscape_lp_norm(
                cycles=w_cycles, 
                num_pts=active_num_pts, 
                n_layers=active_layers,p=p
            )
    
            successive_results.append(
                {
                    "window_num": i + 1,
                    "start_idx": curr_start,
                    "n_cycles": len(w_cycles),
                    "max_persistence": max_p_w,
                    "landscape_lp_norm": landscape_norm,
                    "matrix": w_matrix,
                    "b_bins": w_b_bins,
                    "p_bins": w_p_bins,
                    "diagram": w_diag,
                }
            )
    
        if successive_results:
            st.subheader(" Persistence Images & Landscape $L_p$ Norms")
            cols = st.columns(min(3, len(successive_results)))
            for idx, res in enumerate(successive_results):
                col_target = cols[idx % len(cols)]
                with col_target:
                    fig_w = go.Figure(
                        data=go.Heatmap(
                            z=res["matrix"],
                            x=np.round(res["b_bins"], 5),
                            y=np.round(res["p_bins"][::-1], 5),
                            colorscale="Viridis",
                            showscale=(idx == 0),
                        )
                    )
                    fig_w.update_layout(
                        title=f"Window {res['window_num']} (Anchor Index {res['start_idx']})<br>Features: {res['n_cycles']} | $L_p$ Norm: {res['landscape_lp_norm']:.4f}",
                        xaxis_title="Birth ($b$)",
                        yaxis_title="Persistence",
                        height=350,
                        margin=dict(l=20, r=20, t=50, b=20),
                    )
                    st.plotly_chart(fig_w, use_container_width=True)
    
            if len(successive_results) > 1:
                st.markdown("---")
                st.subheader(
                    f" Pairwise {wasserstein_order}-Wasserstein Distances Between Successive Windows"
                )
    
                wasserstein_rows = []
                for i in range(len(successive_results) - 1):
                    w1 = successive_results[i]
                    w2 = successive_results[i + 1]
                    w_dist = compute_diagram_wasserstein(
                        w1["diagram"], w2["diagram"], order=wasserstein_order
                    )
    
                    wasserstein_rows.append(
                        {
                            "Comparison Pair": f"Window {w1['window_num']} vs Window {w2['window_num']}",
                            "Anchor Index A": w1["start_idx"],
                            "Anchor Index B": w2["start_idx"],
                            f"Wasserstein Distance (p={wasserstein_order})": round(w_dist, 5),
                        }
                    )
    
                st.dataframe(pd.DataFrame(wasserstein_rows), use_container_width=True)
       
def page_2_tda():
    # =============================================================================
    # SECTION 8: SLIDING WINDOW TDA FEATURE EXTRACTION & POWER SPECTRAL DENSITY
    # =============================================================================
    st.markdown("---")
    st.header("8. Sliding Window TDA Feature Extraction & Power Spectral Density")
    st.markdown("""
    Compute topological feature vectors across sliding windows using global parameters configured in Section 3, and analyze their **Power Spectral Density (PSD)**, **correlations (including interval-feature / lagged tests)**, and **ACF**.
    """)
    
    s8_n_layers = st.session_state.get(
    "landscape_n_layers", st.session_state.get("global_n_layers", 3)
    )
    s8_n_bins = st.session_state.get(
        "landscape_num_pts", st.session_state.get("global_num_pts", 200)
    )
    p = st.session_state.get("landscape_lp_order", 2.0)
    
    pi_res= st.session_state.get("pi_res",st.session_state.get("pi_res_input",25))
    pi_sigma =st.session_state.get("pi_sigma)",st.session_state.get("pi_sigma_input",0.15))
    
    if window_direction == "Future (Forward from Start Index)":
        s8_min_start = 0
        s8_max_start = max(0, max_total_rows - window_size)
        s8_default_start = 0
        s8_default_end = min(s8_default_start, s8_max_start)
    else:
        s8_min_start = min(window_size - 1, max_total_rows - 1)
        s8_max_start = max_total_rows - 1
        s8_default_start = s8_min_start
        s8_default_end = min(s8_default_start, s8_max_start)
    
    s8_mode = st.radio("Specify Range By:", ["Row Index (Integer)", "Timestamp Selection"], horizontal=True, key="s8_mode")
    
    if s8_mode == "Timestamp Selection" and hasattr(raw_df.index, "__len__") and len(raw_df.index) > 0:
        s8_ts_options = raw_df.index.tolist()
        col_s8_ts1, col_s8_ts2, col_s8_3 = st.columns(3)
        with col_s8_ts1:
            s8_ts_start = st.selectbox("Start Timestamp", options=s8_ts_options, index=0, key="s8_ts_start")
        with col_s8_ts2:
            s8_ts_end = st.selectbox("End Timestamp", options=s8_ts_options, index=min(len(s8_ts_options) - 1, 25), key="s8_ts_end")
        with col_s8_3:
            s8_step_stride = st.number_input("Row Stride (1 = Every Window)", min_value=1, max_value=max(1, window_size), value=1, step=1, key="s8_stride")
    
        s8_start_row = int(raw_df.index.get_loc(s8_ts_start))
        s8_end_row = int(raw_df.index.get_loc(s8_ts_end))
        if s8_start_row > s8_end_row:
            s8_start_row, s8_end_row = s8_end_row, s8_start_row
        s8_start_row = max(s8_min_start, min(s8_start_row, s8_max_start))
        s8_end_row = max(s8_min_start, min(s8_end_row, s8_max_start))
    else:
        col_s8_1, col_s8_2, col_s8_3 = st.columns(3)
        with col_s8_1:
            s8_start_row = st.number_input("Range Start Anchor Index", min_value=s8_min_start, max_value=s8_max_start, value=s8_default_start, step=1, key="s8_start")
        with col_s8_2:
            s8_end_row = st.number_input("Range End Anchor Index (Inclusive)", min_value=0, max_value=10000000, value=s8_default_end, step=1, key="s8_end")
        with col_s8_3:
            s8_step_stride = st.number_input("Row Stride (1 = Every Window)", min_value=1, max_value=max(1, window_size), value=1, step=1, key="s8_stride")
    
    s8_features_selected = st.multiselect(
        "Select Topological Features to Compute for Every Anchor Index:",
        options=[
            "Betti Curve (Binned)",
            "Persistence Landscape (Binned)",
            "Persistence Image (Flattened Vector)",
            "Landscape Lp Norm",
            "Persistence Entropy",
            "Total Persistence",
            "Mean Persistence",
            "Median Persistence",
            "Max Persistence",
            "Persistence Std Dev",
            "Number of Features Above Threshold",
            "Wasserstein Distance (vs Previous Window)",
            "Wasserstein Distance (vs Next Window)",
            "Maximum Betti Number",
        ],
        default=[
            "Landscape Lp Norm",
            "Persistence Entropy",
            "Total Persistence",
            "Mean Persistence",
            "Max Persistence",
            "Number of Features Above Threshold",
            "Wasserstein Distance (vs Previous Window)",
        ],
        key="s8_feat_select",
    )
    
    col_s8_p1, col_s8_p2 = st.columns(2)
    with col_s8_p1:
        st.info(f" **Global Landscape Layers ($k$):** `{s8_n_layers}` (Inherited from Sec. 3)")
    with col_s8_p2:
        st.info(f" **Global Filtration Bins:** `{s8_n_bins}` (Inherited from Sec. 3)")
    
    s8_thresh_mode = st.selectbox("Cutoff Mode", options=["Absolute (Fine Decimal)", "Relative (% of Max Persistence)"], key="s8_thresh_mode_input")
    
    if s8_thresh_mode == "Absolute (Fine Decimal)":
        s8_thresh_val = st.number_input("Min Persistence Cutoff (Absolute)", min_value=0.0, max_value=100.0, value=0.0005, format="%.5f", step=0.0001, key="s8_abs_val")
    else:
        s8_thresh_val = st.slider("Min Persistence Cutoff (% of Max Persistence across windows)", 0, 50, 5, step=1, key="s8_rel_val")
    
    if st.button(" Extract TDA Features Over Range", key="btn_s8_run"):
        if not s8_features_selected:
            st.warning("Select at least one feature type to compute.")
        else:
            st.session_state["s8_thresh_mode"] = s8_thresh_mode
            st.session_state["s8_thresh_val"] = float(s8_thresh_val)

            row_indices = list(range(int(s8_start_row), int(s8_end_row) + 1, int(s8_step_stride)))
            progress_bar = st.progress(0)

            window_records = []
            global_min_dist = float("inf")
            global_max_dist = 0.0
            global_b_min, global_b_max = float("inf"), 0.0
            global_p_min, global_p_max = float("inf"), 0.0

            df_vals_cache = raw_df.values
            dims_tuple = tuple(selected_dims)
            col_indices_tuple = tuple(raw_df.columns.get_loc(d) for d in selected_dims)

            for idx_pos, r_start in enumerate(row_indices):
                w_diag, w_min_d, w_max_d = compute_fast_diagram_only(
                    df_vals_cache,
                    col_indices_tuple,
                    dims_tuple,
                    r_start,
                    window_size,
                    int(pc_stride),
                    window_direction,
                    stride_agg_mode,
                    standardize_data,
                    homology_dim,
                    s8_thresh_mode,
                    s8_thresh_val,
                )

                global_min_dist = min(global_min_dist, w_min_d)
                global_max_dist = max(global_max_dist, w_max_d)

                if len(w_diag) > 0:
                    global_b_min = min(global_b_min, np.min(w_diag[:, 0]))
                    global_b_max = max(global_b_max, np.max(w_diag[:, 0]))
                    global_p_min = min(global_p_min, np.min(w_diag[:, 1] - w_diag[:, 0]))
                    global_p_max = max(global_p_max, np.max(w_diag[:, 1] - w_diag[:, 0]))

                w_cycles = []
                for d_idx, (b_val, d_val) in enumerate(w_diag):
                    p_val = float(d_val - b_val)
                    w_cycles.append({
                        "id": d_idx,
                        "birth": float(b_val),
                        "death": float(d_val),
                        "persistence": p_val,
                        "peak_time": float((b_val + d_val) / 2.0),
                        "peak_height": p_val / 2.0,
                    })

                window_records.append({
                    "anchor_row": r_start,
                    "timestamp": str(raw_df.index[r_start]),
                    "cycles": w_cycles,
                    "diagram": w_diag,
                    "min_dist": w_min_d,
                    "max_dist": w_max_d,
                })
                progress_bar.progress((idx_pos + 1) / len(row_indices))

            if global_min_dist == float("inf") or global_min_dist >= global_max_dist:
                global_min_dist, global_max_dist = 0.0, 1.0
            if global_b_min == float("inf"):
                global_b_min, global_b_max = 0.0, 1.0
            if global_p_min == float("inf"):
                global_p_min, global_p_max = 0.0, 1.0

            pi_fixed_bounds = (
                max(0.0, global_b_min - 0.005),
                global_b_max + 0.005,
                max(0.0, global_p_min - 0.005),
                global_p_max + 0.005,
            )
            dist_bounds = (global_min_dist, global_max_dist)

            output_rows = []
            for idx_pos, rec in enumerate(window_records):
                row_dict = {
                    "Window_Anchor_Row": rec["anchor_row"],
                    "Anchor_Timestamp": rec["timestamp"],
                    "Active_Features_Count": len(rec["cycles"]),
                }

                w_cycles = rec["cycles"]
                pers_arr = np.array([c["persistence"] for c in w_cycles]) if w_cycles else np.array([])

                if "Total Persistence" in s8_features_selected:
                    row_dict["total_persistence"] = round(float(np.sum(pers_arr)), 6) if len(pers_arr) > 0 else 0.0
                if "Mean Persistence" in s8_features_selected:
                    row_dict["mean_persistence"] = round(float(np.mean(pers_arr)), 6) if len(pers_arr) > 0 else 0.0
                if "Median Persistence" in s8_features_selected:
                    row_dict["median_persistence"] = round(float(np.median(pers_arr)), 6) if len(pers_arr) > 0 else 0.0
                if "Max Persistence" in s8_features_selected:
                    row_dict["max_persistence"] = round(float(np.max(pers_arr)), 6) if len(pers_arr) > 0 else 0.0
                if "Persistence Std Dev" in s8_features_selected:
                    row_dict["persistence_std"] = round(float(np.std(pers_arr)), 6) if len(pers_arr) > 0 else 0.0
                if "Number of Features Above Threshold" in s8_features_selected:
                    row_dict["features_above_threshold_count"] = int(len(w_cycles))

                if "Landscape Lp Norm" in s8_features_selected:
                    lp_val = compute_landscape_lp_norm(
                        w_cycles, num_pts=int(s8_n_bins), n_layers=int(s8_n_layers), t_bounds=dist_bounds, p=p
                    )
                    row_dict[f"landscape_Lp_norm_layers_1_to_{int(s8_n_layers)}"] = round(lp_val, 6)

                if "Persistence Entropy" in s8_features_selected:
                    ent_val = compute_persistence_entropy(w_cycles)
                    row_dict["persistence_entropy"] = round(ent_val, 6)

                if "Wasserstein Distance (vs Previous Window)" in s8_features_selected:
                    if idx_pos > 0:
                        prev_diag = window_records[idx_pos - 1]["diagram"]
                        w_dist_prev = compute_diagram_wasserstein(rec["diagram"], prev_diag, order=wasserstein_order)
                        row_dict["wasserstein_to_previous"] = round(w_dist_prev, 6)
                    else:
                        row_dict["wasserstein_to_previous"] = np.nan

                if "Wasserstein Distance (vs Next Window)" in s8_features_selected:
                    if idx_pos < len(window_records) - 1:
                        next_diag = window_records[idx_pos + 1]["diagram"]
                        w_dist_next = compute_diagram_wasserstein(rec["diagram"], next_diag, order=wasserstein_order)
                        row_dict["wasserstein_to_next"] = round(w_dist_next, 6)
                    else:
                        row_dict["wasserstein_to_next"] = np.nan

                if "Maximum Betti Number" in s8_features_selected:
                    _, betti_vec = compute_betti_curve(w_cycles, num_pts=int(s8_n_bins), t_bounds=dist_bounds)
                    max_betti = int(np.max(betti_vec)) if len(betti_vec) > 0 else 0
                    row_dict["max_betti_number"] = max_betti

                if "Betti Curve (Binned)" in s8_features_selected:
                    _, betti_vec = compute_betti_curve(w_cycles, num_pts=int(s8_n_bins), t_bounds=dist_bounds)
                    if len(betti_vec) != int(s8_n_bins):
                        betti_vec = np.zeros(int(s8_n_bins), dtype=int)
                    for b_idx, val in enumerate(betti_vec):
                        row_dict[f"betti_bin_{b_idx}"] = int(val)

                if "Persistence Landscape (Binned)" in s8_features_selected:
                    t_grid = np.linspace(dist_bounds[0], dist_bounds[1], int(s8_n_bins))
                    layers_mat = compute_landscape_layers(w_cycles, t_grid, n_layers=int(s8_n_layers))
                    for l_idx in range(int(s8_n_layers)):
                        for b_idx in range(int(s8_n_bins)):
                            row_dict[f"land_L{l_idx+1}_bin_{b_idx}"] = round(float(layers_mat[l_idx, b_idx]), 6)

                if "Persistence Image (Flattened Vector)" in s8_features_selected:
                    pi_mat, _, _, _ = generate_persistence_image(
                        w_cycles,
                        res=pi_res,
                        sigma=pi_sigma,
                        filter_mode=pi_filter_mode,
                        custom_expr=custom_filter_expr,
                        fixed_bounds=pi_fixed_bounds,
                    )
                    flat_pi = pi_mat.flatten()
                    for p_idx, val in enumerate(flat_pi):
                        row_dict[f"pi_pixel_{p_idx}"] = round(float(val), 6)

                output_rows.append(row_dict)

            s8_result_df = pd.DataFrame(output_rows)
            st.session_state["s8_result_df"] = s8_result_df
            st.session_state["s8_n_layers_used"] = int(s8_n_layers)
            st.session_state["s8_dist_bounds"] = dist_bounds

    
    if "s8_result_df" in st.session_state and st.session_state["s8_result_df"] is not None:
        s8_df = st.session_state["s8_result_df"]
        used_layers = st.session_state.get("s8_n_layers_used", int(s8_n_layers))
    
        st.success(f" Generated TDA Feature Dataset: `{s8_df.shape[0]}` rows × `{s8_df.shape[1]}` columns")
        st.dataframe(s8_df, use_container_width=True)
    
        csv_export = s8_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=" Download Extracted TDA Features CSV",
            data=csv_export,
            file_name=f"tda_H{homology_dim}_extracted_features.csv",
            mime="text/csv",
        )
    
       
        vol_df_source = (
            st.session_state.get("s1_all_volatility_df")
            if st.session_state.get("s1_all_volatility_df") is not None
            else st.session_state.get("s1_dim_volatility_df")
        )
        vol_cols = list(vol_df_source.columns) if isinstance(vol_df_source, pd.DataFrame) else []
        raw_numeric_cols_s8 = raw_df.select_dtypes(include=[np.number]).columns.tolist()
    

        available_series_map = {}
        for c in s8_df.columns:
            if c not in ["Window_Anchor_Row", "Anchor_Timestamp"] and pd.api.types.is_numeric_dtype(s8_df[c]):
                available_series_map[f"TDA | {c}"] = s8_df[c].astype(float)
    
        for c in vol_cols:
            aligned_v = _align_series_to_s8_anchors(vol_df_source[c], s8_df)
            available_series_map[f"Volatility | {c}"] = aligned_v
            s8_df[f"vol_{c}"] = aligned_v
    
        for c in raw_numeric_cols_s8:
            aligned_r = _align_series_to_s8_anchors(raw_df[c], s8_df)
            available_series_map[f"Raw | {c}"] = aligned_r
            s8_df[f"raw_{c}"] = aligned_r
        st.session_state["s8_df"]= s8_df
    
        # =========================================================================
        # INDEPENDENT TIME-SERIES PLOTS (INCLUDING VOLATILITY & RAW DATASET)
        # =========================================================================
        st.markdown("---")
        st.subheader(" Independent Time-Series Plots")
    
        col_plot_ctrl1, col_plot_ctrl2, col_plot_ctrl3, col_plot_ctrl4 = st.columns(4)
        with col_plot_ctrl1:
            show_lp_plot = st.checkbox("Plot Landscape Lp Norm", value=True, key="chk_plot_lp")
        with col_plot_ctrl2:
            show_ent_plot = st.checkbox("Plot Persistence Entropy", value=True, key="chk_plot_ent")
        with col_plot_ctrl3:
            show_wass_plot = st.checkbox("Plot Wasserstein Distance", value=True, key="chk_plot_wass")
        with col_plot_ctrl4:
            show_max_betti_plot = st.checkbox("Plot Max Betti Number", value=False, key="chk_plot_max_betti")
      
    
        p_opt1, p_opt2, p_opt3, p_opt4 = st.columns(4)
        with p_opt1:
            show_selected_vol_plot = st.checkbox(
                "Plot Volatility Dataset Column",
                value=bool(len(vol_cols) > 0),
                disabled=(len(vol_cols) == 0),
                key="chk_plot_s1_vol_col",
            )
        with p_opt2:
            selected_s1_vol_col = st.selectbox(
                "Volatility Dataset Column",
                options=vol_cols if vol_cols else ["(No Volatility DF in Session)"],
                disabled=(len(vol_cols) == 0),
                key="sel_s1_vol_col",
            )
        with p_opt3:
            show_selected_raw_plot = st.checkbox(
                "Plot Initial Raw Dataset Column",
                value=False,
                disabled=(len(raw_numeric_cols_s8) == 0),
                key="chk_plot_raw_col",
            )
        with p_opt4:
            selected_raw_col = st.selectbox(
                "Initial Raw Dataset Column",
                options=raw_numeric_cols_s8 if raw_numeric_cols_s8 else ["(No Numeric Raw Cols)"],
                disabled=(len(raw_numeric_cols_s8) == 0),
                key="sel_raw_col",
            )
    
    
    
        lp_col_name = f"landscape_Lp_norm_layers_1_to_{used_layers}"
    
        if show_lp_plot and lp_col_name in s8_df.columns:
            fig_lp = go.Figure()
            fig_lp.add_trace(go.Scatter(
                x=s8_df["Window_Anchor_Row"],
                y=s8_df[lp_col_name],
                mode="lines+markers",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=6),
                name="Landscape Lp Norm",
                text=s8_df["Anchor_Timestamp"],
                hovertemplate="Anchor Index: %{x}<br>Time: %{text}<br>Lp Norm: %{y:.5f}<extra></extra>",
            ))
    
            
    
            fig_lp.update_layout(
                title="Persistence Landscape Lp Norm",
                xaxis_title="Window Anchor Index",
                yaxis_title="Landscape Lp Norm",
                height=380,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_lp, use_container_width=True)
    
        if show_ent_plot and "persistence_entropy" in s8_df.columns:
            fig_ent = go.Figure(data=[go.Scatter(
                x=s8_df["Window_Anchor_Row"],
                y=s8_df["persistence_entropy"],
                mode="lines+markers",
                line=dict(color="#9467bd", width=2),
                marker=dict(size=6),
                text=s8_df["Anchor_Timestamp"],
                hovertemplate="Anchor Index: %{x}<br>Time: %{text}<br>Entropy: %{y:.5f}<extra></extra>",
            )])
            fig_ent.update_layout(title="Persistence Entropy Across Windows", xaxis_title="Window Anchor Index", yaxis_title="Entropy (Bits)", height=350)
            st.plotly_chart(fig_ent, use_container_width=True)
    
        if show_wass_plot:
            wass_col = (
                "wasserstein_to_previous"
                if "wasserstein_to_previous" in s8_df.columns
                else ("wasserstein_to_next" if "wasserstein_to_next" in s8_df.columns else None)
            )
            if wass_col and wass_col in s8_df.columns:
                valid_w_df = s8_df.dropna(subset=[wass_col])
                wass_title = (
                    f"Wasserstein Distance ($p={wasserstein_order}$) vs Previous Window"
                    if wass_col == "wasserstein_to_previous"
                    else f"Wasserstein Distance ($p={wasserstein_order}$) vs Next Window"
                )
                fig_wass = go.Figure(data=[go.Scatter(
                    x=valid_w_df["Window_Anchor_Row"],
                    y=valid_w_df[wass_col],
                    mode="lines+markers",
                    line=dict(color="#d62728", width=2),
                    marker=dict(size=6),
                    text=valid_w_df["Anchor_Timestamp"],
                    hovertemplate="Anchor Index %{{x}}<br>Time: %{{text}}<br>Wasserstein: %{{y:.5f}}<extra></extra>",
                )])
                fig_wass.update_layout(title=wass_title, xaxis_title="Window Anchor Index", yaxis_title="Wasserstein Distance", height=350)
                st.plotly_chart(fig_wass, use_container_width=True)
    
        if show_max_betti_plot and "max_betti_number" in s8_df.columns:
            fig_betti_max = go.Figure(data=[go.Scatter(
                x=s8_df["Window_Anchor_Row"],
                y=s8_df["max_betti_number"],
                mode="lines+markers",
                line=dict(color="#2ca02c", width=2),
                marker=dict(size=6),
                text=s8_df["Anchor_Timestamp"],
                hovertemplate="Anchor Index: %{x}<br>Time: %{text}<br>Max Betti: %{y}<extra></extra>",
            )])
            fig_betti_max.update_layout(title=f"Maximum Betti Number ($H_{homology_dim}$) Across Windows", xaxis_title="Window Anchor Index", yaxis_title="Max Betti Number", height=350)
            st.plotly_chart(fig_betti_max, use_container_width=True)
    
        if show_selected_vol_plot and selected_s1_vol_col in vol_cols:
            vol_series_aligned = available_series_map[f"Volatility | {selected_s1_vol_col}"]
            fig_s1_vol = go.Figure(data=[go.Scatter(
                x=s8_df["Window_Anchor_Row"],
                y=vol_series_aligned,
                mode="lines+markers",
                line=dict(color="#17becf", width=2),
                marker=dict(size=5),
                text=s8_df["Anchor_Timestamp"],
                hovertemplate="Anchor Index: %{x}<br>Time: %{text}<br>Volatility: %{y:.6f}<extra></extra>",
            )])
            fig_s1_vol.update_layout(
                title=f"Volatility Dataset Column: `{selected_s1_vol_col}`",
                xaxis_title="Window Anchor Index",
                yaxis_title=selected_s1_vol_col,
                height=350,
            )
            st.plotly_chart(fig_s1_vol, use_container_width=True)
    
        if show_selected_raw_plot and selected_raw_col in raw_numeric_cols_s8:
            raw_series_aligned = available_series_map[f"Raw | {selected_raw_col}"]
            fig_raw_s8 = go.Figure(data=[go.Scatter(
                x=s8_df["Window_Anchor_Row"],
                y=raw_series_aligned,
                mode="lines+markers",
                line=dict(color="#8c564b", width=2),
                marker=dict(size=5),
                text=s8_df["Anchor_Timestamp"],
                hovertemplate="Anchor Index: %{x}<br>Time: %{text}<br>Value: %{y:.6f}<extra></extra>",
            )])
            fig_raw_s8.update_layout(
                title=f"Initial Raw Dataset Column: `{selected_raw_col}`",
                xaxis_title="Window Anchor Index",
                yaxis_title=selected_raw_col,
                height=350,
            )
            st.plotly_chart(fig_raw_s8, use_container_width=True)
    
    
        # =========================================================================
        # POWER SPECTRAL DENSITY (PSD) ANALYSIS (Standard Welch)
        # =========================================================================
        st.markdown("###  Power Spectral Density (PSD) Analysis (Welch's Method)")
        st.markdown("Analyze the spectral frequency components and dominant periodicities of extracted topological time-series (e.g., Lp Norm, Entropy, Wasserstein distance).")
    
        numeric_s8_cols = [c for c in s8_df.columns if c not in ["Window_Anchor_Row", "Anchor_Timestamp"]]
        psd_target_col = st.selectbox("Select Topological Metric for PSD Analysis", options=numeric_s8_cols, key="s8_psd_col")
    
        if psd_target_col:
            series_data = s8_df[psd_target_col].dropna().values
            if len(series_data) > 4:
                fs_val = st.number_input("Sampling Frequency ($f_s$)", min_value=0.1, max_value=1000.0, value=1.0, step=0.1, key="s8_psd_fs")
                nperseg_val = st.slider("Welch Segment Length (`nperseg`)", min_value=4, max_value=max(8, len(series_data)), value=min(256, len(series_data)), step=1, key="s8_psd_nperseg")
    
                f_welch, pxx_welch = welch(series_data, fs=fs_val, nperseg=nperseg_val, window="hann")
    
                fig_psd = go.Figure(data=[go.Scatter(
                    x=f_welch,
                    y=pxx_welch,
                    mode="lines",
                    line=dict(color="#ff7f0e", width=2),
                    hovertemplate="Frequency: %{x:.4f} Hz<br>PSD Power: %{y:.5f}<extra></extra>",
                )])
                fig_psd.update_layout(
                    title=f"Power Spectral Density (PSD) of `{psd_target_col}`",
                    xaxis_title="Frequency (Hz)",
                    yaxis_title="PSD Power Spectral Density",
                    height=400,
                )
                st.plotly_chart(fig_psd, use_container_width=True)
    
    
    # =============================================================================
    # SECTION 9: ROLLING SPECTRAL WINDOW & SPECTROGRAM ANALYSIS
    # =============================================================================
    st.markdown("---")
    st.header("9. Rolling Spectral Window (Spectrogram & Frequency Band Power)")
    st.markdown("""
    Apply a secondary rolling lookback window over TDA features, volatility columns, or raw dataset columns
    to track rolling spectrograms, band power (including average low-frequency band power), secondary Welch PSD,
    ACF, and correlation / lagged interval-feature comparisons against raw or other series.
    """)
    
    
    
    s9_base_frame = None
    
    if (
        "s8_result_df" in st.session_state
        and st.session_state["s8_result_df"] is not None
        and len(st.session_state["s8_result_df"]) > 0
    ):
        s9_base_frame = st.session_state["s8_result_df"].copy()
    
    meta_cols = {"Window_Anchor_Row", "Anchor_Timestamp"}
    numeric_cols_s9 = (
        [
            c
            for c in s9_base_frame.columns
            if c not in meta_cols and pd.api.types.is_numeric_dtype(s9_base_frame[c])
        ]
        if s9_base_frame is not None
        else []
    )
    
    if s9_base_frame is None or len(s9_base_frame) == 0 or not numeric_cols_s9:
        st.warning("No numeric series available for rolling spectral analysis, compute in section 8")
    else:
        s9_target_key = st.selectbox(
            "Select Series for Rolling Spectral & Secondary PSD Analysis (TDA, Volatility, or Raw)",
            options=numeric_cols_s9,
            key="s9_target_col",
        )
    
        series_full_s9 = s9_base_frame[s9_target_key].to_numpy(dtype=float)
        valid_mask_s9 = np.isfinite(series_full_s9)
        series_data = series_full_s9[valid_mask_s9]
    
        if "Anchor_Timestamp" in s9_base_frame.columns:
            ts_all = s9_base_frame["Anchor_Timestamp"].astype(str).to_numpy()
        else:
            ts_all = s9_base_frame.index.astype(str).to_numpy()
        aligned_timestamps_s9 = ts_all[valid_mask_s9]
    
        if len(series_data) > 4:
            col_gk1, col_gk2, col_gk3 = st.columns(3)
            with col_gk1:
                gk_sec_window = st.number_input(
                    "Secondary Rolling Window Size (Rows)",
                    min_value=10,
                    max_value=len(series_data),
                    value=min(250, len(series_data)),
                    step=10,
                    key="gk_sec_window",
                )
            with col_gk2:
                gk_sec_stride = st.number_input(
                    "Secondary Window Stride",
                    min_value=1,
                    max_value=max(1, int(gk_sec_window)),
                    value=1,
                    step=1,
                    key="gk_sec_stride",
                )
            with col_gk3:
                gk_fs = st.number_input(
                    "Rolling Sampling Frequency ($f_s$)",
                    min_value=0.1,
                    max_value=1000.0,
                    value=1.0,
                    step=0.1,
                    key="gk_fs_val",
                )
    
            col_gk4, col_gk5, col_gk6, col_gk7 = st.columns(4)
            with col_gk4:
                gk_nperseg = st.slider(
                    "Rolling Welch `nperseg`",
                    min_value=4,
                    max_value=max(8, int(gk_sec_window)),
                    value=min(64, int(gk_sec_window)),
                    step=1,
                    key="gk_nperseg_val",
                )
            with col_gk5:
                gk_overlap_pct = st.slider(
                    "Welch Overlap %",
                    min_value=0,
                    max_value=90,
                    value=50,
                    step=5,
                    key="gk_overlap_pct",
                    help="Percentage of overlap between adjacent internal Welch segments.",
                )
            with col_gk6:
                nyquist_limit = float(gk_fs / 2.0)
                gk_max_freq = st.number_input(
                    "Max Display Frequency (Hz)",
                    min_value=0.001,
                    max_value=nyquist_limit,
                    value=min(0.5, nyquist_limit),
                    step=0.01,
                    format="%.4f",
                    key="gk_max_freq",
                )
            with col_gk7:
                analysis_mode = st.radio(
                    "Frequency Band Mode",
                    ["Single Band Range", "Multi-Interval Bands"],
                    horizontal=True,
                    key="s9_analysis_mode",
                )
    
            if analysis_mode == "Single Band Range":
                st.markdown("**Tracking Frequency Band (Hz)**")
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    f_min_target = st.number_input(
                        "Min Hz",
                        min_value=0.0,
                        max_value=nyquist_limit,
                        value=0.01,
                        step=0.0001,
                        format="%.4f",
                        key="s9_f_min_input",
                    )
                with col_f2:
                    f_max_target = st.number_input(
                        "Max Hz",
                        min_value=0.0,
                        max_value=nyquist_limit,
                        value=0.05,
                        step=0.0001,
                        format="%.4f",
                        key="s9_f_max_input",
                    )
    
                bands_config = [
                    (f_min_target, f_max_target, f"Band [{f_min_target:.4f} - {f_max_target:.4f} Hz]")
                ]
            else:
                st.markdown("**Multi-Interval Band Configuration**")
                col_mb1, col_mb2, col_mb3 = st.columns(3)
                with col_mb1:
                    multi_start_freq = st.number_input(
                        "Starting Frequency (Hz)",
                        min_value=0.0,
                        max_value=nyquist_limit,
                        value=0.005,
                        step=0.0001,
                        format="%.4f",
                        key="s9_multi_start",
                    )
                with col_mb2:
                    multi_interval_size = st.number_input(
                        "Interval Width (Hz)",
                        min_value=0.0001,
                        max_value=nyquist_limit,
                        value=0.01,
                        step=0.0001,
                        format="%.4f",
                        key="s9_multi_width",
                    )
                with col_mb3:
                    multi_num_intervals = st.number_input(
                        "Number of Intervals",
                        min_value=1,
                        max_value=10,
                        value=3,
                        step=1,
                        key="s9_multi_count",
                    )
    
                bands_config = []
                curr_f = multi_start_freq
                for i in range(int(multi_num_intervals)):
                    next_f = curr_f + multi_interval_size
                    bands_config.append((curr_f, next_f, f"Band {i+1}: [{curr_f:.4f} - {next_f:.4f} Hz]"))
                    curr_f = next_f
    
            low_freq_upper = st.number_input(
                "Low-Frequency Cutoff Upper Bound (Hz) for Dedicated Low-Freq Band Average",
                min_value=0.0001,
                max_value=nyquist_limit,
                value=min(0.05, nyquist_limit),
                step=0.001,
                format="%.4f",
                key="s9_low_freq_upper",
            )
    
            invalid_bands = [b for b in bands_config if b[0] >= b[1] or b[1] > nyquist_limit]
            if invalid_bands:
                st.error(
                    "One or more frequency bands have invalid bounds (Min must be < Max (and within Nyquist limit))."
                )
            else:
                st.markdown("---")
                st.markdown("### Range Filter for Plotting")
                sub_range_mode = st.radio(
                    "Isolate Plot Range By:",
                    ["Full Extracted Range", "Timestamp Selectors", "Row Index Range"],
                    horizontal=True,
                    key="s9_range_mode",
                )
    
                all_timestamps = aligned_timestamps_s9.tolist()
                sub_start_ts, sub_end_ts = all_timestamps[0], all_timestamps[-1]
                sub_start_row_idx, sub_end_row_idx = 0, len(series_data) - 1
    
                if sub_range_mode == "Timestamp Selectors":
                    col_sub_ts1, col_sub_ts2 = st.columns(2)
                    with col_sub_ts1:
                        sub_start_ts = st.selectbox(
                            "Plot Start Timestamp",
                            options=all_timestamps,
                            index=0,
                            key="s9_sub_start_ts",
                        )
                    with col_sub_ts2:
                        sub_end_ts = st.selectbox(
                            "Plot End Timestamp",
                            options=all_timestamps,
                            index=len(all_timestamps) - 1,
                            key="s9_sub_end_ts",
                        )
                elif sub_range_mode == "Row Index Range":
                    col_sub_r1, col_sub_r2 = st.columns(2)
                    with col_sub_r1:
                        sub_start_row_idx = st.number_input(
                            "Start Row Index",
                            min_value=0,
                            max_value=len(series_data) - 1,
                            value=0,
                            step=1,
                            key="s9_sub_start_row",
                        )
                    with col_sub_r2:
                        sub_end_row_idx = st.number_input(
                            "End Row Index",
                            min_value=0,
                            max_value=len(series_data) - 1,
                            value=len(series_data) - 1,
                            step=1,
                            key="s9_sub_end_row",
                        )
    
                st.markdown("")
                if st.button("Run Rolling Spectral Analysis", key="button9"):
                    st.session_state["ran_sec10"] = True
    
                if st.session_state.get("ran_sec10", False):
                    rolling_times = []
                    rolling_end_row_positions = []
                    rolling_freqs = None
                    spectrogram_matrix = []
                    multi_band_power_dict = {label: [] for _, _, label in bands_config}
                    low_freq_band_power_series = []
    
                    # Map valid series_data positions back to s9_base_frame row positions
                    valid_base_row_positions = np.flatnonzero(valid_mask_s9)
    
                    with st.spinner("Computing rolling spectrogram, band power, and low-frequency band average..."):
                        for start_idx in range(0, len(series_data) - int(gk_sec_window) + 1, int(gk_sec_stride)):
                            end_idx = start_idx + int(gk_sec_window)
                            chunk = series_data[start_idx:end_idx]
    
                            chunk_nperseg = min(int(gk_nperseg), len(chunk))
                            noverlap_val = int(chunk_nperseg * (gk_overlap_pct / 100.0))
                            if noverlap_val >= chunk_nperseg:
                                noverlap_val = chunk_nperseg - 1
    
                            chunk_nfft = max(2048, 8 * chunk_nperseg)
    
                            f_chunk, pxx_chunk = welch(
                                chunk,
                                fs=gk_fs,
                                nperseg=chunk_nperseg,
                                noverlap=noverlap_val,
                                window="hamming",
                                nfft=chunk_nfft,
                            )
    
                            freq_mask = f_chunk <= gk_max_freq #chunk size specified by chunk_nfft 
                            f_filtered = f_chunk[freq_mask] #gives list of frequencies
                            pxx_filtered = pxx_chunk[freq_mask] #gives discrete filtered spectrogram
    
                            if rolling_freqs is None:
                                rolling_freqs = f_filtered
    
                            spectrogram_matrix.append(pxx_filtered)
    
                            for b_min, b_max, label in bands_config:
                                band_mask = (f_filtered >= b_min) & (f_filtered <= b_max)
                                mean_band_power = float(np.mean(pxx_filtered[band_mask])) if np.any(band_mask) else 0.0
                                multi_band_power_dict[label].append(mean_band_power)
    
                            lf_mask = (f_chunk >= 0.0) & (f_chunk <= float(low_freq_upper))
                            lf_power = float(np.mean(pxx_chunk[lf_mask])) if np.any(lf_mask) else 0.0
                            low_freq_band_power_series.append(lf_power)
    
                            rolling_times.append(aligned_timestamps_s9[end_idx - 1])
                            rolling_end_row_positions.append(int(valid_base_row_positions[end_idx - 1]))
    
                    if len(rolling_times) > 0 and rolling_freqs is not None and len(rolling_freqs) > 0:
                        spec_arr = np.array(spectrogram_matrix).T
    
                        s9_df = pd.DataFrame({
                            "time": rolling_times,
                            "base_row_pos": rolling_end_row_positions,
                        })
                        s9_df["col_idx"] = np.arange(len(s9_df), dtype=int)
                        for label in multi_band_power_dict:
                            s9_df[label] = multi_band_power_dict[label]
    
                        band_cols_list = list(multi_band_power_dict.keys())
                        s9_df["Avg_Configured_Bands_Power"] = s9_df[band_cols_list].mean(axis=1)
                        lf_col_name = f"Avg_LowFreq_Band_Power_[0-{float(low_freq_upper):.4f}Hz]"
                        s9_df[lf_col_name] = low_freq_band_power_series
    
                        if sub_range_mode == "Timestamp Selectors":
                            s9_df = s9_df[(s9_df["time"] >= sub_start_ts) & (s9_df["time"] <= sub_end_ts)]
                        elif sub_range_mode == "Row Index Range":
                            s9_df = s9_df[
                                (s9_df["col_idx"] >= sub_start_row_idx) & (s9_df["col_idx"] <= sub_end_row_idx)
                            ]
    
                        if len(s9_df) == 0:
                            st.warning("The selected timestep filter range contains no rolling window points.")
                        else:
                            filtered_times = s9_df["time"].tolist()
                            filtered_col_indices = s9_df["col_idx"].to_numpy(dtype=int)
                            filtered_base_rows = s9_df["base_row_pos"].to_numpy(dtype=int)
                            filtered_spec_arr = spec_arr[:, filtered_col_indices]
                            st.session_state["s9_band_power_df"] = s9_df.copy()
                            
                                
                            # Save exact metadata Section 10 needs so indices & timestamps stay 1-to-1 with s9_df
                            st.session_state["s9_to_s10_context"] = {
                                "target_key": s9_target_key,
                                "filtered_times": filtered_times,
                                "filtered_base_rows": filtered_base_rows,
                                "band_cols_list": band_cols_list,
                                "lf_col_name": lf_col_name,
                                "gk_fs": float(gk_fs),
                            }
    
                            st.markdown("#### Rolling Spectrogram (Time vs. Frequency vs. Power)")
                            fig_spec = go.Figure(
                                data=go.Heatmap(
                                    z=filtered_spec_arr,
                                    x=filtered_times,
                                    y=rolling_freqs,
                                    colorscale="Viridis",
                                    colorbar=dict(title="PSD Power"),
                                )
                            )
                            fig_spec.update_layout(
                                title=f"Rolling Spectrogram of `{s9_target_key}` (Filtered Range)",
                                xaxis_title="Window End Timestamp",
                                yaxis_title="Frequency (Hz)",
                                height=400,
                            )
                            st.plotly_chart(fig_spec, use_container_width=True)
    
                            st.markdown("#### Band-Specific & Average Low-Frequency Spectral Power")
                            fig_multi_line = go.Figure()
                            colors = [
                                "#d62728",
                                "#1f77b4",
                                "#2ca02c",
                                "#ff7f0e",
                                "#9467bd",
                                "#8c564b",
                                "#e377c2",
                                "#7f7f7f",
                                "#bcbd22",
                                "#17becf",
                            ]
    
                            for idx, label in enumerate(band_cols_list):
                                fig_multi_line.add_trace(
                                    go.Scatter(
                                        x=filtered_times,
                                        y=s9_df[label].to_numpy(dtype=float),
                                        mode="lines",
                                        name=label,
                                        line=dict(width=2, color=colors[idx % len(colors)]),
                                    )
                                )
    
                            fig_multi_line.add_trace(
                                go.Scatter(
                                    x=filtered_times,
                                    y=s9_df["Avg_Configured_Bands_Power"].to_numpy(dtype=float),
                                    mode="lines",
                                    name="Avg Configured Bands Power",
                                    line=dict(width=2.5, dash="solid", color="#ff7f0e"),
                                )
                            )
                            fig_multi_line.add_trace(
                                go.Scatter(
                                    x=filtered_times,
                                    y=s9_df[lf_col_name].to_numpy(dtype=float),
                                    mode="lines",
                                    name=lf_col_name,
                                    line=dict(width=2.5, dash="solid", color="#e377c2"),
                                )
                            )
    
                            fig_multi_line.update_layout(
                                title="Comparative Spectral Power Across Configured & Low-Frequency Bands",
                                xaxis_title="Window End Timestamp",
                                yaxis_title="Mean PSD Power",
                                height=400,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            )
                            st.plotly_chart(fig_multi_line, use_container_width=True)
    
                            # =====================================================================
                            # SECONDARY FREQUENCY ANALYSIS (WELCH PSD ON BAND POWER SERIES)
                            # =====================================================================
                            st.markdown("---")
                            st.markdown("#### Secondary Frequency Analysis (Welch PSD on Band Power Series)")
                            max_allowed_nperseg = max(8, len(s9_df))
                            default_sec_nperseg = min(64, max_allowed_nperseg)
    
                            sec_welch_nperseg = st.slider(
                                "Secondary Welch `nperseg` (for Band Power PSD)",
                                min_value=4,
                                max_value=max_allowed_nperseg,
                                value=default_sec_nperseg,
                                step=2,
                                key="s9_sec_nperseg_slider",
                            )
    
                            sec_psd_candidates = band_cols_list + ["Avg_Configured_Bands_Power", lf_col_name]
                            for idx, p_col in enumerate(sec_psd_candidates):
                                power_series = s9_df[p_col].to_numpy(dtype=float)
                                if len(power_series) >= 4:
                                    effective_nperseg = min(len(power_series), int(sec_welch_nperseg))
                                    f_sub, pxx_sub = welch(
                                        power_series,
                                        fs=gk_fs,
                                        nperseg=effective_nperseg,
                                        window="flattop",
                                    )
                                    fig_sub_welch = go.Figure()
                                    fig_sub_welch.add_trace(
                                        go.Scatter(
                                            x=f_sub,
                                            y=pxx_sub,
                                            mode="lines",
                                            name=f"PSD of {p_col}",
                                            line=dict(width=2, color=colors[idx % len(colors)]),
                                        )
                                    )
                                    fig_sub_welch.update_layout(
                                        title=f"Internal Frequency Spectrum of `{p_col}` (nperseg={effective_nperseg})",
                                        xaxis_title="Frequency (Hz) of the Power Series",
                                        yaxis_title="Power Spectral Density",
                                        height=350,
                                        template="plotly_dark",
                                    )
                                    st.plotly_chart(fig_sub_welch, use_container_width=True)
    
    
    # =====================================================================
    # SECTION 10: CORRELATION & LAGGED INTERVAL-FEATURE ANALYSIS
    # =====================================================================
    st.markdown("---")
    st.markdown("## Section 10: Correlation & Lagged Interval-Feature Analysis")
    render_s9_metric_methodology_help(expanded=False)
    
    s9_df_s10 = st.session_state.get("s9_band_power_df")
    s8_df_s10 = st.session_state.get("s8_result_df")
    
    if s9_df_s10 is None or len(s9_df_s10) < 2:
        st.info("Run Rolling Spectral Analysis in Section 9 first to populate `s9_df`.")
    else:
        s9_df = s9_df_s10.reset_index(drop=True)
        n_pts = len(s9_df)
        filtered_times = (
            s9_df["time"].astype(str).tolist()
            if "time" in s9_df.columns
            else [str(i) for i in range(n_pts)]
        )
    
        skip_cols = {"time", "col_idx", "base_row_pos"}
        band_cols = [
            c for c in s9_df.columns
            if c not in skip_cols and pd.api.types.is_numeric_dtype(s9_df[c])
        ]
    
        meta_cols = {"Window_Anchor_Row", "Anchor_Timestamp"}
        has_s8 = isinstance(s8_df_s10, pd.DataFrame) and len(s8_df_s10) > 0
        base_cols = (
            [c for c in s8_df_s10.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(s8_df_s10[c])]
            if has_s8
            else []
        )
    
        if has_s8 and "base_row_pos" in s9_df.columns and base_cols:
            base_rows = np.clip(s9_df["base_row_pos"].to_numpy(dtype=int), 0, len(s8_df_s10) - 1)
        else:
            base_rows = None
    
        band_labels = [f"Band Power | {c}" for c in band_cols]
        corr_options = band_labels + base_cols
    
        def get_series_10(label: str) -> np.ndarray:
            prefix = "Band Power | "
            if label.startswith(prefix):
                return s9_df[label[len(prefix):]].to_numpy(dtype=float)
            if not has_s8 or label not in s8_df_s10.columns:
                return np.full(n_pts, np.nan, dtype=float)
            if base_rows is not None:
                return s8_df_s10[label].to_numpy(dtype=float)[base_rows]
            return s8_df_s10[label].to_numpy(dtype=float)[:n_pts]
    
        if len(corr_options) < 2:
            st.warning("Need at least 2 numeric series available in Section 10 for correlation and interval analysis.")
        else:
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                default_sx = 0
                for i_k, k_str in enumerate(corr_options):
                    if k_str.startswith("Band Power | Avg_LowFreq_Band_Power"):
                        default_sx = i_k
                        break
                s10_sx_key = st.selectbox(
                    "Series X",
                    options=corr_options,
                    index=default_sx,
                    key="s10_corr_x",
                )
            with sc2:
                default_sy = 1 if len(corr_options) > 1 else 0
                for i_k, k_str in enumerate(corr_options):
                    if k_str.startswith("raw_") or "Raw" in k_str:
                        default_sy = i_k
                        break
                s10_sy_key = st.selectbox(
                    "Series Y",
                    options=corr_options,
                    index=default_sy,
                    key="s10_corr_y",
                )
            with sc3:
                s10_corr_method = st.selectbox(
                    "Full-Series Correlation Test Type",
                    options=["pearson", "difference", "spearman"],
                    format_func=lambda m: {
                        "pearson": "Pearson Correlation",
                        "difference": "Difference Correlation (First Differences)",
                        "spearman": "Spearman Rank Correlation",
                    }[m],
                    key="s10_corr_method",
                )
    
            sx_arr = get_series_10(s10_sx_key)
            sy_arr = get_series_10(s10_sy_key)
    
            r_s10, p_s10, n_s10 = _compute_corr_metric(
                sx_arr, sy_arr, method=s10_corr_method, return_n=True
            )
            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("Series Correlation (r)", f"{r_s10:.5f}" if np.isfinite(r_s10) else "N/A")
            sm2.metric("p-value", f"{p_s10:.4e}" if np.isfinite(p_s10) else "N/A")
            sm3.metric("Test Method", s10_corr_method.capitalize())
            sm4.metric("Valid Pairs (n)", f"{n_s10:,}")
    
            st.markdown("##### Lagged Interval Feature Engineering & Correlation / Sign-Alignment(no p-value) Test")
            fe1, fe2, fe3 = st.columns(3)
            with fe1:
                s10_x_interval_agg = st.selectbox(
                    "Series X Interval Feature [t, t+k_x]",
                    options=["diff", "mean", "sum"],
                    format_func=lambda a: {
                        "diff": "Interval Difference (x_{t+k_x} - x_t)",
                        "mean": "Interval Average (mean of x[t : t+k_x])",
                        "sum": "Interval Sum (sum of x[t : t+k_x])",
                    }[a],
                    key="s10_x_interval_agg",
                )
            with fe2:
                s10_y_interval_agg = st.selectbox(
                    "Series Y Interval Feature [t+s, t+s+k_y]",
                    options=["diff", "mean", "sum"],
                    format_func=lambda a: {
                        "diff": "Interval Difference (y_{t+s+k_y} - y_{t+s})",
                        "mean": "Interval Average (mean of y[t+s : t+s+k_y])",
                        "sum": "Interval Sum (sum of y[t+s : t+s+k_y])",
                    }[a],
                    key="s10_y_interval_agg",
                )
            with fe3:
                s10_interval_score_metric = st.selectbox(
                    "Primary Interval Score Metric",
                    options=["uncentered_cosine", "sign_agreement", "pearson", "spearman"],
                    format_func=lambda m: {
                        "uncentered_cosine": "Uncentered Sign Alignment (Weighted by Magnitude)",
                        "sign_agreement": "Sign Agreement (Ignores Magnitude)",
                        "pearson": "Mean-Centered Pearson",
                        "spearman": "Mean-Centered Spearman",
                    }[m],
                    key="s10_interval_score_metric",
                )
    
            st1, st2, st3 = st.columns(3)
            with st1:
                s10_use_signed_interval_diff = st.checkbox(
                    "Keep signs of interval features (uncheck to take |feature| absolute magnitudes)",
                    value=True,
                    key="s10_ic_signed_mode",
                )
            with st2:
                s10_use_diff_of_diffs = st.checkbox(
                    "Also take first difference across intervals (np.diff on the interval feature series)",
                    value=False,
                    key="s10_ic_diff_of_diffs",
                )
            with st3:
                s10_dx_direction_filter = st.selectbox(
                    "X Interval Change Filter (based on x_{t+k_x} - x_t)",
                    options=["any", "up", "down"],
                    format_func=lambda d: {
                        "any": "Any Direction (Threshold Filter on |x_{t+k_x} - x_t|)",
                        "up": "Went Up Only (x_{t+k_x} - x_t > 0)",
                        "down": "Went Down Only (x_{t+k_x} - x_t < 0)",
                    }[d],
                    key="s10_ic_dx_direction",
                )
    
            k_col1, k_col2, k_col3, k_col4 = st.columns(4)
            with k_col1:
                s10_kx_min = st.number_input(
                    "Series X Min k (k_x_min)",
                    min_value=1,
                    max_value=max(1, len(sx_arr) - 1),
                    value=1,
                    step=1,
                    key="s10_ic_kx_min",
                )
            with k_col2:
                s10_kx_max = st.number_input(
                    "Series X Max k (k_x_max)",
                    min_value=int(s10_kx_min),
                    max_value=max(1, len(sx_arr) - 1),
                    value=min(max(int(s10_kx_min), 5), max(1, len(sx_arr) - 1)),
                    step=1,
                    key="s10_ic_kx_max",
                )
            with k_col3:
                s10_ky_min = st.number_input(
                    "Series Y Min k (k_y_min)",
                    min_value=1,
                    max_value=max(1, len(sy_arr) - 1),
                    value=1,
                    step=1,
                    key="s10_ic_ky_min",
                )
            with k_col4:
                s10_ky_max = st.number_input(
                    "Series Y Max k (k_y_max)",
                    min_value=int(s10_ky_min),
                    max_value=max(1, len(sy_arr) - 1),
                    value=min(max(int(s10_ky_min), 5), max(1, len(sy_arr) - 1)),
                    step=1,
                    key="s10_ic_ky_max",
                )
    
            sic1, sic2 = st.columns(2)
            with sic1:
                s10_lag_s = st.number_input(
                    "Time Lag / Shift (s)",
                    min_value=0,
                    max_value=max(0, len(sy_arr) - 1),
                    value=1,
                    step=1,
                    key="s10_ic_lag_s",
                )
            with sic2:
                s10_min_dx = st.number_input(
                    "Min Abs Change |x_{t+k_x} - x_t| (0 = include 0s)",
                    min_value=0.0,
                    max_value=1e9,
                    value=0.0,
                    step=0.001,
                    format="%.6f",
                    key="s10_ic_min_dx",
                )
    
            agg_short = {"diff": "Diff", "mean": "Mean", "sum": "Sum"}
            dir_tag_10 = {"any": "Any", "up": "UpOnly", "down": "DownOnly"}[s10_dx_direction_filter]
            sign_tag_10 = "Signed" if s10_use_signed_interval_diff else "Abs"
            score_short = {
                "uncentered_cosine": "UncenteredCos",
                "sign_agreement": "SignAgree",
                "pearson": "Pearson",
                "spearman": "Spearman",
            }[s10_interval_score_metric]
            feat_pair_label = (
                f"{score_short} | X:{agg_short[s10_x_interval_agg]}(k_x) "
                f"vs Y:{agg_short[s10_y_interval_agg]}(k_y) ({sign_tag_10})"
            )
            if s10_use_diff_of_diffs:
                feat_pair_label += " + np.diff"
            interval_mode_label_10 = f"{feat_pair_label} [{dir_tag_10}]"
            thresh_val_10 = float(s10_min_dx)
    
            kx_list = list(range(int(s10_kx_min), int(s10_kx_max) + 1))
            ky_list = list(range(int(s10_ky_min), int(s10_ky_max) + 1))
            n_steps_k = max(len(kx_list), len(ky_list))
    
            s10_ic_rows = []
            qualifying_x_segments_by_pair = {}
            qualifying_y_segments_by_pair = {}
            pair_labels_available = []
    
            for step_i in range(n_steps_k):
                kx_val = int(kx_list[min(step_i, len(kx_list) - 1)])
                ky_val = int(ky_list[min(step_i, len(ky_list) - 1)])
                pair_label = f"Step {step_i + 1}: k_x={kx_val}, k_y={ky_val}"
                pair_labels_available.append(pair_label)
            
                max_t = min(
                    len(sx_arr) - 1 - kx_val,
                    len(sy_arr) - 1 - int(s10_lag_s) - ky_val,
                )
                if max_t < 0:
                    qualifying_x_segments_by_pair[pair_label] = []
                    qualifying_y_segments_by_pair[pair_label] = []
                    s10_ic_rows.append({
                        "Step": step_i + 1,
                        "Pair_Label": pair_label,
                        "Interval_k_X": kx_val,
                        "Interval_k_Y": ky_val,
                        "Lag_s": int(s10_lag_s),
                        "Interval_Mode": interval_mode_label_10,
                        "Min_Abs_Change_X": thresh_val_10,
                        "Qualifying_Pairs": 0,
                        "NonZero_Sign_Pairs": 0,
                        "Total_Possible_Pairs": 0,
                        "Mean_Feature_X": np.nan,
                        "Mean_Feature_Y_lag_s": np.nan,
                        "Uncentered_Sign_Alignment": np.nan,
                        "Uncentered_Cos_p_value": np.nan,
                        "Sign_Agreement": np.nan,
                        "Sign_Agreement_p_value": np.nan,
                        "Centered_Corr": np.nan,
                        "Primary_Score": np.nan,
                        "p_value": np.nan,
                    })
                    continue
            
                t_vec = np.arange(0, max_t + 1, dtype=int)
                signed_dx_10 = sx_arr[t_vec + kx_val] - sx_arr[t_vec]
                abs_dx_10 = np.abs(signed_dx_10)
            
                feat_x_raw = _s9_extract_interval_feature(
                    sx_arr, t_vec, kx_val, agg_mode=s10_x_interval_agg
                )
                feat_y_raw = _s9_extract_interval_feature(
                    sy_arr, t_vec + int(s10_lag_s), ky_val, agg_mode=s10_y_interval_agg
                )
            
                fin_m = np.isfinite(feat_x_raw) & np.isfinite(feat_y_raw) & np.isfinite(signed_dx_10)
                thresh_m = abs_dx_10 >= 0.0 if thresh_val_10 <= 0.0 else abs_dx_10 > thresh_val_10
            
                if s10_dx_direction_filter == "up":
                    dir_m = signed_dx_10 > 0.0
                elif s10_dx_direction_filter == "down":
                    dir_m = signed_dx_10 < 0.0
                else:
                    dir_m = np.ones_like(signed_dx_10, dtype=bool)
            
                q_m = fin_m & thresh_m & dir_m
                qualifying_x_segments_by_pair[pair_label] = [
                    (int(t_val), int(t_val + kx_val), int(t_val), kx_val) for t_val in t_vec[q_m]
                ]
                qualifying_y_segments_by_pair[pair_label] = [
                    (
                        int(t_val + int(s10_lag_s)),
                        int(t_val + int(s10_lag_s) + ky_val),
                        int(t_val),
                        ky_val,
                    )
                    for t_val in t_vec[q_m]
                ]
            
                if s10_use_signed_interval_diff:
                    x_sel = feat_x_raw[q_m]
                    y_sel = feat_y_raw[q_m]
                else:
                    x_sel = np.abs(feat_x_raw[q_m])
                    y_sel = np.abs(feat_y_raw[q_m])
            
                if s10_use_diff_of_diffs:
                    if len(x_sel) >= 2 and len(y_sel) >= 2:
                        x_eval = np.diff(x_sel)
                        y_eval = np.diff(y_sel)
                    else:
                        x_eval = np.array([], dtype=float)
                        y_eval = np.array([], dtype=float)
                else:
                    x_eval = x_sel
                    y_eval = y_sel
            
                uncentered_cos_val, sign_agree_val, n_nz_sign, p_cos_val, p_sign_val = _s9_compute_directional_metrics(
                    x_eval, y_eval, return_p=True
                )
                centered_method_10 = "spearman" if s10_interval_score_metric == "spearman" else "pearson"
                r_cent_10, p_cent_10 = _interval_compute_corr_metric(
                    x_eval, y_eval, method=centered_method_10
                )
            
                if s10_interval_score_metric == "uncentered_cosine":
                    primary_score_10 = uncentered_cos_val
                    primary_p_10 = p_cos_val
                elif s10_interval_score_metric == "sign_agreement":
                    primary_score_10 = sign_agree_val
                    primary_p_10 = p_sign_val
                else:
                    primary_score_10 = r_cent_10
                    primary_p_10 = p_cent_10
            
                s10_ic_rows.append({
                    "Step": step_i + 1,
                    "Pair_Label": pair_label,
                    "Interval_k_X": kx_val,
                    "Interval_k_Y": ky_val,
                    "Lag_s": int(s10_lag_s),
                    "Interval_Mode": interval_mode_label_10,
                    "Min_Abs_Change_X": thresh_val_10,
                    "Qualifying_Pairs": int(np.sum(q_m)),
                    "NonZero_Sign_Pairs": n_nz_sign,
                    "Total_Possible_Pairs": int(np.sum(fin_m)),
                    "Mean_Feature_X": float(np.mean(x_sel)) if len(x_sel) > 0 else np.nan,
                    "Mean_Feature_Y_lag_s": float(np.mean(y_sel)) if len(y_sel) > 0 else np.nan,
                    "Uncentered_Sign_Alignment": uncentered_cos_val,
                    "Uncentered_Cos_p_value": p_cos_val,
                    "Sign_Agreement": sign_agree_val,
                    "Sign_Agreement_p_value": p_sign_val,
                    "Centered_Corr": r_cent_10,
                    "Primary_Score": primary_score_10,
                    "p_value": primary_p_10,
                })
            
            
            df_s10_ic = pd.DataFrame(s10_ic_rows)
            st.session_state["s10_interval_corr_df"] = df_s10_ic
            st.dataframe(df_s10_ic, use_container_width=True)
            df_plot_ic = df_s10_ic
            
            
            x_axis_labels_ic = [f"kX={r['Interval_k_X']}, kY={r['Interval_k_Y']}" for _, r in df_plot_ic.iterrows()]
            fig_s10_ic = go.Figure()
            fig_s10_ic.add_trace(go.Scatter(
                x=x_axis_labels_ic,
                y=df_plot_ic["Uncentered_Sign_Alignment"],
                mode="lines+markers",
                name="Uncentered Sign Alignment (Weighted)",
                line=dict(color="#1f77b4", width=2.5, dash="solid"),
            ))
            fig_s10_ic.add_trace(go.Scatter(
                x=x_axis_labels_ic,
                y=df_plot_ic["Sign_Agreement"],
                mode="lines+markers",
                name="Sign Agreement (Ignores Magnitude)",
                line=dict(color="#2ca02c", width=2, dash="solid"),
            ))
            fig_s10_ic.add_trace(go.Bar(
                x=x_axis_labels_ic,
                y=df_plot_ic["Qualifying_Pairs"],
                name="Qualifying Pairs",
                yaxis="y2",
                opacity=0.3,
            ))
            fig_s10_ic.update_layout(
                title=f"Lagged Interval Directional Alignment —  (s={int(s10_lag_s)}, threshold={thresh_val_10}, dir={dir_tag_10})",
                xaxis_title="Interval Pair (k_x, k_y)",
                yaxis=dict(title="Score [-1, +1]"),
                yaxis2=dict(title="Qualifying Pairs", overlaying="y", side="right", showgrid=False),
                height=360,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_s10_ic, use_container_width=True)
            # =================================================================
            # SERIES X & SERIES Y PLOTS WITH QUALIFYING / LAGGED INTERVALS
            # =================================================================
            st.markdown("##### Series X & Lagged Series Y Plots with Compared Intervals Highlighted")
            selected_highlight_pair = st.selectbox(
                "Interval Pair (k_x, k_y) to Highlight on Series X & Lagged Series Y",
                options=pair_labels_available,
                index=0,
                key="s10_highlight_k_sel",
            )
            x_segs_show = qualifying_x_segments_by_pair.get(selected_highlight_pair, [])
            y_segs_show = qualifying_y_segments_by_pair.get(selected_highlight_pair, [])
            sel_kx_disp = x_segs_show[0][3] if x_segs_show else int(kx_list[0])
            sel_ky_disp = y_segs_show[0][3] if y_segs_show else int(ky_list[0])
    
            # 1) Series X plot highlighting [t, t + k_x]
            x_plot_positions = np.arange(len(sx_arr), dtype=int)
            fig_s1_hi = go.Figure()
            fig_s1_hi.add_trace(go.Scatter(
                x=x_plot_positions,
                y=sx_arr,
                mode="lines+markers",
                name=f"Series X: {s10_sx_key}",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=4),
                text=filtered_times,
                hovertemplate="Index: %{x}<br>Time: %{text}<br>Value: %{y:.6f}<extra></extra>",
            ))
    
            hi_x_pos, hi_x_val, hi_x_txt = [], [], []
            for t_start_i, t_end_i, t_anchor, kx_cur in x_segs_show:
                if t_start_i < len(sx_arr) and t_end_i < len(sx_arr):
                    seg_x = sx_arr[t_start_i : t_end_i + 1]
                    if np.all(np.isfinite(seg_x)):
                        hi_x_pos.extend([t_start_i, t_end_i, None])
                        hi_x_val.extend([sx_arr[t_start_i], sx_arr[t_end_i], None])
                        dx_seg = sx_arr[t_end_i] - sx_arr[t_start_i]
                        if s10_x_interval_agg == "mean":
                            feat_x_val = float(np.mean(seg_x))
                        elif s10_x_interval_agg == "sum":
                            feat_x_val = float(np.sum(seg_x))
                        else:
                            feat_x_val = float(dx_seg)
                        lbl_x = (
                            f"X Interval t={t_anchor} [{t_start_i} -> {t_end_i}] (k_x={kx_cur})<br>"
                            f"X {agg_short[s10_x_interval_agg]} = {feat_x_val:+.6f} | Δx = {dx_seg:+.6f}"
                        )
                        hi_x_txt.extend([lbl_x, lbl_x, ""])
    
            if hi_x_pos:
                fig_s1_hi.add_trace(go.Scatter(
                    x=hi_x_pos,
                    y=hi_x_val,
                    mode="lines+markers",
                    name=f"Qualifying X Intervals [t, t+k_x] ({agg_short[s10_x_interval_agg]}, k_x={sel_kx_disp}, count={len(x_segs_show)})",
                    line=dict(color="#ff2e63", width=3.5),
                    marker=dict(size=7, color="#ff2e63", symbol="circle"),
                    text=hi_x_txt,
                    hovertemplate="%{text}<extra></extra>",
                    connectgaps=False,
                ))
    
            fig_s1_hi.update_layout(
                title=f"Series X (`{s10_sx_key}`) — Qualifying Intervals [t, t+k_x] ({agg_short[s10_x_interval_agg]}, k_x={sel_kx_disp}, dir={dir_tag_10}, thresh={thresh_val_10})",
                xaxis_title="Filtered Series Index",
                yaxis_title=s10_sx_key,
                height=370,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_s1_hi, use_container_width=True)
    
            # 2) Series Y plot right below highlighting lag-shifted intervals [t + s, t + s + k_y]
            y_plot_positions = np.arange(len(sy_arr), dtype=int)
            fig_s2_hi = go.Figure()
            fig_s2_hi.add_trace(go.Scatter(
                x=y_plot_positions,
                y=sy_arr,
                mode="lines+markers",
                name=f"Series Y: {s10_sy_key}",
                line=dict(color="#2ca02c", width=2),
                marker=dict(size=4),
                text=filtered_times,
                hovertemplate="Index: %{x}<br>Time: %{text}<br>Value: %{y:.6f}<extra></extra>",
            ))
    
            hi_y_pos, hi_y_val, hi_y_txt = [], [], []
            for y_start_i, y_end_i, t_anchor, ky_cur in y_segs_show:
                if y_start_i < len(sy_arr) and y_end_i < len(sy_arr):
                    seg_y = sy_arr[y_start_i : y_end_i + 1]
                    if np.all(np.isfinite(seg_y)):
                        hi_y_pos.extend([y_start_i, y_end_i, None])
                        hi_y_val.extend([sy_arr[y_start_i], sy_arr[y_end_i], None])
                        dy_seg = sy_arr[y_end_i] - sy_arr[y_start_i]
                        if s10_y_interval_agg == "mean":
                            feat_y_val = float(np.mean(seg_y))
                        elif s10_y_interval_agg == "sum":
                            feat_y_val = float(np.sum(seg_y))
                        else:
                            feat_y_val = float(dy_seg)
                        lbl_y = (
                            f"Lagged Y Interval for t={t_anchor}, s={int(s10_lag_s)} "
                            f"[{y_start_i} -> {y_end_i}] (k_y={ky_cur})<br>"
                            f"Y {agg_short[s10_y_interval_agg]} = {feat_y_val:+.6f} | Δy = {dy_seg:+.6f}"
                        )
                        hi_y_txt.extend([lbl_y, lbl_y, ""])
    
            if hi_y_pos:
                fig_s2_hi.add_trace(go.Scatter(
                    x=hi_y_pos,
                    y=hi_y_val,
                    mode="lines+markers",
                    name=f"Compared Y Intervals [t+s, t+s+k_y] ({agg_short[s10_y_interval_agg]}, s={int(s10_lag_s)}, k_y={sel_ky_disp}, count={len(y_segs_show)})",
                    line=dict(color="#ff7f0e", width=3.5),
                    marker=dict(size=7, color="#ff7f0e", symbol="diamond"),
                    text=hi_y_txt,
                    hovertemplate="%{text}<extra></extra>",
                    connectgaps=False,
                ))
    
            fig_s2_hi.update_layout(
                title=f"Series Y (`{s10_sy_key}`) — Lagged Intervals [t+s, t+s+k_y] ({agg_short[s10_y_interval_agg]}, s={int(s10_lag_s)}, k_y={sel_ky_disp})",
                xaxis_title="Filtered Series Index",
                yaxis_title=s10_sy_key,
                height=370,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_s2_hi, use_container_width=True)
    
        # =====================================================================
        # AUTOCORRELATION FUNCTION (ACF) TEST IN SECTION 10
        # =====================================================================
        st.markdown("---")
        st.markdown("#### Autocorrelation Function (ACF) Test")
        st.markdown(
            "Compute and inspect sample autocorrelation across lags for any band power, "
            "low-frequency average, TDA, volatility, or raw series in Section 10."
        )
    
        if not corr_options:
            st.warning("No numeric series available in `s9_df` or `s8_result_df` for ACF.")
        else:
            acf10_c1, acf10_c2 = st.columns(2)
            with acf10_c1:
                s10_acf_key = st.selectbox(
                    "Select Series for ACF Test",
                    options=corr_options,
                    key="s10_acf_series",
                )
            with acf10_c2:
                s10_acf_max_lag = st.number_input(
                    "Max Lag",
                    min_value=1,
                    max_value=max(2, n_pts - 1),
                    value=min(20, max(1, n_pts - 1)),
                    step=1,
                    key="s10_acf_max_lag",
                )
    
            s10_acf_arr = get_series_10(s10_acf_key)
            lags_raw,_,acf_raw,_ = _compute_acf( s10_acf_arr,
                max_lag=int(s10_acf_max_lag),
                criterion =  "First Zero Crossing",
                include_lag_zero = False,
                adaptive_lag = False )
            n_eff_10 = int(np.sum(np.isfinite(s10_acf_arr)))
            conf_bound_10 = 1.96 / np.sqrt(n_eff_10) if n_eff_10 > 0 else np.nan
            lags_10 = np.atleast_1d(np.asarray(lags_raw))
            acf_10 = np.atleast_1d(np.asarray(acf_raw, dtype=float))
            
            n_common = min(len(lags_10), len(acf_10))
            lags_10 = lags_10[:n_common]
            acf_10 = acf_10[:n_common]
    
            if len(lags_10) > 0:
                df_s10_acf = pd.DataFrame({
                    "Lag": lags_10,
                    "ACF": acf_10,
                    "Approx_95pct_Upper": np.full(len(lags_10), conf_bound_10),
                    "Approx_95pct_Lower": np.full(len(lags_10), -conf_bound_10),
                    "Significant_Approx_95pct": np.abs(acf_10) > conf_bound_10,
                })
                st.session_state["s10_acf_df"] = df_s10_acf
                st.dataframe(df_s10_acf, use_container_width=True)
    
                fig_s10_acf = go.Figure()
                fig_s10_acf.add_trace(
                    go.Scatter(
                        x=lags_10,
                        y=acf_10,
                        mode="lines+markers",
                        name="Sample ACF",
                        line=dict(color="#1f77b4"),
                    )
                )
                if np.isfinite(conf_bound_10):
                    fig_s10_acf.add_hline(
                        y=conf_bound_10,
                        line_dash="solid",
                        line_color="red",
                        annotation_text="+95% CI",
                    )
                    fig_s10_acf.add_hline(
                        y=-conf_bound_10,
                        line_dash="solid",
                        line_color="red",
                        annotation_text="-95% CI",
                    )
                fig_s10_acf.update_layout(
                    title=f"ACF Plot: `{s10_acf_key}`",
                    xaxis_title="Lag",
                    yaxis_title="Autocorrelation",
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_s10_acf, use_container_width=True)
            else:
                st.info("Not enough non-null observations to compute ACF.")

pg = st.navigation([
    st.Page(page_1_tda, title="1. Single point cloud analysis", default=True),
    st.Page(page_2_tda, title="2.Time-series: Specral Density & Correlation"),
])
pg.run()