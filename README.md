# TDA-Persistence-homology-time-series-analysis-and-feature-interpretability
Interactive Python/Streamlit workbench for persistent homology feature engineering &amp; interpretability ( dimension ablation, minimal cycles, and raw point tracebacks) and 1D time-delay / multivariate signal analysis (PSD, ACF, rolling windows, and custom lagged/signed interval correlation tests).

Key Capabilities
1. Persistent Homology Feature Engineering & Interpretability

    Topological Summaries & Vector Embeddings: Compute Betti curves, persistence landscapes, persistence images, LpLp​ landscape norms, persistence entropy, total/mean/median/max/std persistence, active feature counts, max Betti numbers, and window-to-window Wasserstein distances.

    Interpretability & Diagnostics: Inspect minimal cycles, run dimension ablations, and trace topological features back to their underlying point-cloud rows, timestamps, and coordinates.

    Sliding-Window Feature Extraction: Extract topological feature time series across configurable row-index or timestamp ranges with stride control, persistence cutoffs (absolute or relative), and CSV export.

2. Time-Delay Embeddings & Spectral / Correlation Diagnostics

    1D Time-Delay & Multivariate Workflows: Construct time-delay embeddings (with configurable delay direction, lag ττ, dimension, and window averaging) and analyze multivariate / volatility / raw series alongside extracted TDA features.

    Spectral & Autocorrelation Analysis: Compute Welch Power Spectral Density (PSD), rolling spectrograms, configurable frequency band power (including low-frequency band averages), secondary band-power PSDs, mutual-information lag selection, and sample autocorrelation (ACF) curves/lags.

    Standard & Custom Interval / Directional Correlation Tests: Evaluate Pearson, Spearman, and first-difference correlations, plus custom lagged interval comparisons and directional alignment metrics—uncentered cosine (with exact sign-flip enumeration for small samples and seeded Monte Carlo sign-flip pp-values for larger samples) and sign agreement (with exact two-sided binomial pp-values).

Quick Start

    Clone the repository and install dependencies:
    Bash

    git clone <your-repo-url>
    cd <your-repo-folder>
    pip install -r requirements.txt

    Launch the Streamlit app:
    Bash

    streamlit run app.py

    (Always use streamlit run rather than python app.py so Streamlit initializes an active ScriptRunContext.)

Typical Workflow

    Configure data & embedding/window settings: Load your dataset, choose dimensions or time-delay embedding parameters, and set global landscape/filtration parameters.

    Explore topology & trace features: Inspect persistence diagrams, landscapes, persistence images, minimal cycles, dimension ablations, and raw point tracebacks.

    Extract sliding-window TDA features: Run feature extraction over your target anchor range and export the resulting feature table as CSV.

    Analyze dynamics, spectra & correlations: Plot topological and aligned raw/volatility series, inspect Welch PSD, rolling spectrograms, band power, and ACF, and run standard or lagged/signed interval correlation tests.

Repository Structure (Customize to Match Your Layout)
Plaintext

├── app.py                  # Main Streamlit multi-section application
├── utils/                  # TDA helpers, embeddings, PSD/ACF, and correlation metrics
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
