"""Tests for the Fama-French residual construction (Section 2.1.1)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.residuals import rolling_betas, residuals_and_projection
from src import config


def test_rolling_beta_window(synthetic_panel):
    returns, _, _, factors = synthetic_panel
    excess = returns.sub(factors["rf"], axis=0)
    betas = rolling_betas(excess, factors[["mkt_rf", "smb", "hml", "rmw", "cma"]])
    T, N, K = betas.shape
    assert T == len(excess) and N == excess.shape[1] and K == 5
    # Rows before the window must be NaN; the first full-window row must be finite.
    assert np.isnan(betas[config.BETA_WINDOW - 2]).all()
    assert np.isfinite(betas[config.BETA_WINDOW]).any()


def test_residuals_orthogonal_to_factors(synthetic_panel):
    """epsilon_t must be orthogonal to the factors used to project it (Eq. 3)."""
    returns, _, _, factors = synthetic_panel
    excess = returns.sub(factors["rf"], axis=0)
    f5 = factors[["mkt_rf", "smb", "hml", "rmw", "cma"]]
    betas = rolling_betas(excess, f5)
    eps, phi = residuals_and_projection(excess, f5, betas)
    # Pick a date well past the warm-up and verify that beta' * eps_t ~= 0.
    t_idx = 200
    B = betas[t_idx - 1]
    e = eps.iloc[t_idx].to_numpy()
    valid = np.isfinite(B).all(axis=1) & np.isfinite(e)
    inner = B[valid].T @ e[valid]
    # By construction epsilon_t is in the null-space of beta', so beta' eps ~ 0.
    assert np.allclose(inner, 0.0, atol=1e-8)
    # Phi should be idempotent on the valid subspace: Phi @ Phi == Phi.
    P = phi[t_idx]
    idx = np.where(np.isfinite(B).all(axis=1))[0]
    sub = P[np.ix_(idx, idx)]
    assert np.allclose(sub @ sub, sub, atol=1e-8)
