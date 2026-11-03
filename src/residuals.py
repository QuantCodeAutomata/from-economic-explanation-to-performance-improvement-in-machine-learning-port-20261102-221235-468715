"""Fama-French five-factor residuals (Section 2.1.1).

Daily factor loadings ``beta_{t-1}`` for every stock are estimated with a
rolling 60-day OLS regression of excess stock returns on the five Fama-French
factors (equation (2)). The projection matrix

    Phi_{t-1} = I - beta(beta' beta)^{-1} beta'

(equation (3)) extracts the tradable residual portfolio and the stock-space
residuals ``epsilon_t = Phi_{t-1} R_t`` (equation (4)).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import BETA_WINDOW, N_FACTORS


FACTOR_COLS = ("mkt_rf", "smb", "hml", "rmw", "cma")


def excess_returns(returns: pd.DataFrame, rf: pd.Series) -> pd.DataFrame:
    """Daily excess returns R_{n,t} = r_{n,t} - rf_t (equation (1))."""
    rf = rf.reindex(returns.index).fillna(0.0)
    return returns.subtract(rf, axis=0)


def rolling_betas(ex_rets: pd.DataFrame, factors: pd.DataFrame,
                   window: int = BETA_WINDOW) -> np.ndarray:
    """Rolling 60-day OLS betas for every stock.

    Returns an array of shape (T, N, K) where entry [t, n, :] is the beta
    estimated on dates ``t-window+1 ... t`` for stock ``n``. The leading
    ``window-1`` rows are NaN.
    """
    T, N = ex_rets.shape
    K = len(FACTOR_COLS)
    F = factors[list(FACTOR_COLS)].to_numpy(dtype=float)
    R = ex_rets.to_numpy(dtype=float)
    betas = np.full((T, N, K), np.nan, dtype=float)
    mask = np.isfinite(R)
    # Pre-compute rolling X'X with F only; however the mask is per-stock so we
    # loop by stock (vectorised across windows via linear algebra on slices).
    for n in range(N):
        y_all = R[:, n]
        valid = np.isfinite(y_all)
        for t in range(window - 1, T):
            lo = t - window + 1
            v = valid[lo: t + 1]
            if v.sum() < window - 5:
                continue
            Xs = F[lo: t + 1][v]
            ys = y_all[lo: t + 1][v]
            # Add intercept-free regression per the paper's equation (2).
            try:
                XtX = Xs.T @ Xs
                beta = np.linalg.solve(XtX, Xs.T @ ys)
            except np.linalg.LinAlgError:
                continue
            betas[t, n, :] = beta
    return betas


def residuals_and_projection(ex_rets: pd.DataFrame, factors: pd.DataFrame,
                              betas: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Daily residuals ``epsilon_t`` and projection matrices ``Phi_{t-1}``.

    ``betas`` must be the rolling window output of :func:`rolling_betas` and is
    lagged by one day so that only information up to ``t-1`` is used to form
    the residual at date ``t`` (equation (4)).
    """
    T, N, K = betas.shape
    F = factors[list(FACTOR_COLS)].to_numpy(dtype=float)
    R = ex_rets.to_numpy(dtype=float)
    eps = np.full_like(R, np.nan)
    phi = np.full((T, N, N), np.nan)
    I = np.eye(N)
    for t in range(1, T):
        B = betas[t - 1]  # shape (N, K); lagged
        valid = np.isfinite(B).all(axis=1) & np.isfinite(R[t])
        if valid.sum() < K + 2:
            continue
        idx = np.where(valid)[0]
        Bv = B[idx]  # (n_valid, K)
        try:
            G = np.linalg.solve(Bv.T @ Bv, Bv.T)   # (K, n_valid)
        except np.linalg.LinAlgError:
            continue
        f_hat = G @ R[t][idx]                      # (K,)
        eps_v = R[t][idx] - Bv @ f_hat             # (n_valid,)
        eps[t, idx] = eps_v
        phi_small = np.eye(len(idx)) - Bv @ G
        full = np.zeros((N, N))
        full[np.ix_(idx, idx)] = phi_small
        phi[t] = full
    eps_df = pd.DataFrame(eps, index=ex_rets.index, columns=ex_rets.columns)
    return eps_df, phi


__all__ = ["FACTOR_COLS", "excess_returns", "rolling_betas", "residuals_and_projection"]
