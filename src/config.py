"""Experiment parameters.

Values are taken from the paper's main-text specification. Where a parameter
is used only to make the full configuration fit the replication's compute
budget, both the paper value and the executed value are kept side by side and
selected by an experiment ``--full`` flag.
"""
from __future__ import annotations

# =============================================================================
# Universe and sample period
# =============================================================================

# Paper universe: 346 large-capitalisation and liquid US stocks (Section 3.1).
UNIVERSE_SIZE_PAPER: int = 346
# Executed universe: top-N US stocks by market capitalisation at sample start.
# Reduced to fit the replication's compute budget while preserving the paper's
# ranking rule (large-cap, liquid).
UNIVERSE_SIZE_RUN: int = 149

# Full historical window needed to estimate 60-day betas from January 2001 and
# have a 5-year rolling window ready at the start of the trading period
# (Section 3.1).
DATA_START: str = "2000-10-01"  # calendar date (YYYY-MM-DD)
DATA_END: str = "2024-12-31"    # calendar date (YYYY-MM-DD)

# Out-of-sample trading period (Section 3.1).
OOS_START: str = "2006-01-03"
OOS_END: str = "2024-12-31"

# =============================================================================
# Factor model and residual construction
# =============================================================================

# Rolling window for Fama-French 5-factor beta estimation (Section 2.1.1).
BETA_WINDOW: int = 60  # trading days

# Number of Fama-French factors (Mkt-RF, SMB, HML, RMW, CMA).
N_FACTORS: int = 5

# =============================================================================
# Feature construction
# =============================================================================

# Lookback for cumulative residual process X_{t-1}^L (Section 2.1.2).
FEATURE_LOOKBACK: int = 30  # trading days

# Number of regimes for the regime-dependent OU model used to compute the
# regime features. The paper uses a non-parametric HDP-HMM (Appendix A); this
# replication substitutes a 3-state Gaussian HMM. The deviation is recorded in
# results/RESULTS.md.
HMM_N_REGIMES: int = 3

# Total number of trading-signal features in Table 1.
N_FEATURES: int = 18

# =============================================================================
# Model training schedule
# =============================================================================

# Rolling estimation window length (Section 3.1).
TRAIN_WINDOW_YEARS: int = 5
TRAIN_YEARS: int = 4
VALIDATION_YEARS: int = 1
RETRAIN_EVERY: int = 252  # trading days between model re-estimations

# =============================================================================
# Allocation models
# =============================================================================

# Transaction cost per unit turnover applied to the net Sharpe objective and to
# net return reporting (Section 2.1.3, equation (14)).
TRANSACTION_COST: float = 0.0005  # one-way, 5 basis points

# FFN architecture (Section 2.1.3): 4 hidden layers of sizes 32-16-8-4 with
# ReLU activations and a linear output.
FFN_HIDDEN_SIZES: tuple[int, ...] = (32, 16, 8, 4)
FFN_DROPOUT: float = 0.1          # not fully specified in the paper
FFN_LEARNING_RATE: float = 1e-3   # not fully specified in the paper
FFN_EPOCHS: int = 8               # not fully specified in the paper
FFN_BATCH_DAYS: int = 120         # number of days per mini-batch for the Sharpe loss

# XGB hyperparameters (not fully specified in the paper; defaults chosen from
# the ranges typical of boosted-tree pipelines on tabular financial data).
XGB_MAX_DEPTH: int = 4
XGB_LEARNING_RATE: float = 0.05
XGB_N_ESTIMATORS: int = 200
XGB_MIN_CHILD_WEIGHT: float = 1.0
XGB_SUBSAMPLE: float = 0.8
XGB_COLSAMPLE_BYTREE: float = 0.8

# =============================================================================
# Attribution and inference
# =============================================================================

# Rolling window for the attribution time-series plots (Section 3.2.2).
ATTRIBUTION_ROLL_DAYS: int = 63

# Number of coalitions sampled for the Monte Carlo Shapley estimator
# (Section 2.2.2, "more than 500 sampled feature coalitions"). The executed
# value is kept small to fit the compute budget.
SHAPLEY_N_COALITIONS_PAPER: int = 500
SHAPLEY_N_COALITIONS_RUN: int = 200

# Circular block bootstrap for Sharpe inference (Section 2.2.5).
BOOTSTRAP_B: int = 4999
BOOTSTRAP_BLOCK: int = 5
BOOTSTRAP_B_RUN: int = 999  # reduced for compute budget; use BOOTSTRAP_B with --full

# =============================================================================
# Intervention experiments
# =============================================================================

# Confidence parameter of the exposure multiplier (Section 2.3.2, eq. (42)).
INTERVENTION_ETA: float = 0.5
# Clipping range for the stabilised exposure multiplier.
INTERVENTION_LAMBDA_MIN: float = 0.0
INTERVENTION_LAMBDA_MAX: float = 3.0

# Transaction-cost sensitivity grid (Section 3.3.3).
TC_IS_GRID_BPS: tuple[int, ...] = (1, 3, 5, 7, 9)
TC_OOS_GRID_BPS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

# =============================================================================
# Feature name order (Table 1).
# =============================================================================

FEATURE_NAMES: tuple[str, ...] = (
    "kappa",            # OU mean-reversion speed
    "mu",               # OU long-run mean
    "sigma_ou",         # OU diffusion
    "cum_resid",        # cumulative residual X_{t-1}^L
    "r2",               # OU fit quality
    "std_signal",       # standardised OU signal
    "regime_entropy",   # posterior regime entropy
    "regime_duration",  # expected regime duration
    "regime_trans",     # regime transition probability
    "real_res",         # realised residual
    "pred_res",         # predicted residual (regime-conditional)
    "resid_std",        # residual rolling std
    "resid_skew",       # residual rolling skew
    "resid_kurt",       # residual rolling kurtosis
    "ent_res",          # entropy x residual interaction
    "adv",              # average daily dollar volume
    "log_mcap",         # log market capitalisation
    "turnover_stock",   # stock turnover (volume / shares outstanding)
)

# =============================================================================
# Random seeds for reproducibility
# =============================================================================

SEED: int = 20250101

# =============================================================================
# Results directory
# =============================================================================

RESULTS_DIR: str = "results"
