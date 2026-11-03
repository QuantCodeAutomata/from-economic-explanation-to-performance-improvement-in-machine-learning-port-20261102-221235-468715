"""Residual-to-stock weight mapping, turnover and gross/net returns.

Implements equations (12)-(14) of the paper:

    w^R_{t-1} = (w^{eps'}_{t-1} Phi_{t-1}) / || . ||_1,
    tau_t     = || w^R_{t-1} - w^R_{t-2} ||_1,
    R_{0,t}   = w^R_{t-1}' R_t,
    R_{c,t}   = R_{0,t} - c tau_t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def residual_to_stock_weights(w_eps: pd.DataFrame, phi: np.ndarray) -> pd.DataFrame:
    """Project residual weights onto stock weights and L1-normalise (eq. (12))."""
    T, N = w_eps.shape
    out = np.zeros_like(w_eps.values, dtype=float)
    W = w_eps.to_numpy(dtype=float)
    for t in range(T):
        we = W[t]
        if not np.isfinite(we).any():
            continue
        we = np.nan_to_num(we)
        proj = we @ phi[t]            # 1 x N
        norm = np.abs(proj).sum()
        if norm > 0:
            out[t] = proj / norm
    return pd.DataFrame(out, index=w_eps.index, columns=w_eps.columns)


def portfolio_turnover(w_stock: pd.DataFrame) -> pd.Series:
    """Daily portfolio turnover tau_t = ||w_{t-1} - w_{t-2}||_1 (eq. (13))."""
    diff = w_stock.diff().abs().sum(axis=1)
    return diff.fillna(0.0)


def gross_returns(w_stock: pd.DataFrame, ex_rets: pd.DataFrame) -> pd.Series:
    """Daily gross excess portfolio return R_{0,t} = w^R_{t-1}' R_t."""
    w_lag = w_stock.shift(1).reindex_like(ex_rets).fillna(0.0)
    r = ex_rets.fillna(0.0)
    return (w_lag * r).sum(axis=1)


def net_returns(gross: pd.Series, turnover: pd.Series, c: float) -> pd.Series:
    """Net excess portfolio return R_{c,t} = R_{0,t} - c tau_t."""
    return gross - c * turnover.reindex(gross.index).fillna(0.0)


def annualised_metrics(returns: pd.Series) -> dict:
    """Mean, volatility and Sharpe at annual frequency (252 trading days)."""
    r = returns.dropna()
    if len(r) == 0:
        return {"mean": np.nan, "vol": np.nan, "sharpe": np.nan}
    mu = float(r.mean()) * 252
    sd = float(r.std(ddof=0)) * np.sqrt(252)
    sharpe = mu / sd if sd > 0 else np.nan
    return {"mean": mu, "vol": sd, "sharpe": sharpe}


__all__ = [
    "residual_to_stock_weights",
    "portfolio_turnover",
    "gross_returns",
    "net_returns",
    "annualised_metrics",
]
