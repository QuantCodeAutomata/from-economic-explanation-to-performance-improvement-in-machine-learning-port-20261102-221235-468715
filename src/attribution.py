"""Local first-order attribution propagation (Section 2.2 of the paper).

This module propagates allocation-level feature attributions
``phi_i(w^eps_{n,t-1})`` (produced upstream by TreeSHAP for XGB and DeepSHAP
for FFN) through the chain

    residual weights -> stock weights -> turnover -> gross/net return
                     -> mean return, volatility, Sharpe ratio.

The implementation follows equations (24)-(32) of the paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Model-level attribution helpers (SHAP wrappers)
# ---------------------------------------------------------------------------

def tree_shap_values(model, X: np.ndarray) -> np.ndarray:
    """TreeSHAP values for an XGBoost regressor. Shape (n_samples, n_features)."""
    import shap
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(X)
    if isinstance(sv, list):
        sv = sv[0]
    return np.asarray(sv)


def deep_shap_values(torch_model, X: np.ndarray, background: np.ndarray) -> np.ndarray:
    """DeepSHAP values for a PyTorch FFN. Falls back to the simple linearisation
    phi_i(x) = (x_i - E[x_i]) * E[d f / d x_i] when SHAP's DeepExplainer cannot
    be constructed for the chosen framework version.
    """
    try:
        import shap
        import torch
        bg = torch.from_numpy(background.astype(np.float32))
        expl = shap.DeepExplainer(torch_model, bg)
        sv = expl.shap_values(torch.from_numpy(X.astype(np.float32)))
        if isinstance(sv, list):
            sv = sv[0]
        return np.asarray(sv)
    except Exception:
        return _linear_attribution_pytorch(torch_model, X, background)


def _linear_attribution_pytorch(model, X: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Gradient x (x - E[x]) backstop used when DeepExplainer is unavailable."""
    import torch
    model.eval()
    xb = torch.from_numpy(background.astype(np.float32)).requires_grad_(True)
    yb = model(xb).sum()
    yb.backward()
    g_mean = xb.grad.detach().mean(dim=0).numpy()
    x_mean = background.mean(axis=0)
    return (X - x_mean[None, :]) * g_mean[None, :]


# ---------------------------------------------------------------------------
# Propagation from residual to stock weights
# ---------------------------------------------------------------------------

def propagate_to_stock_weights(phi_eps: np.ndarray,
                                phi_mat: np.ndarray,
                                w_stock_norm: np.ndarray,
                                proj_num: np.ndarray) -> np.ndarray:
    """Equation (25).

    ``phi_eps``   shape (N, K): per-feature attribution to residual weights.
    ``phi_mat``   shape (N, N): projection matrix Phi_{t-1}.
    ``w_stock_norm`` shape (N,): realised stock weight after L1 normalisation.
    ``proj_num``  shape (N,): the unnormalised projection w^eps' Phi.
    Returns phi_i(w^R_{n}) of shape (N, K).
    """
    tilde_phi = phi_eps.T @ phi_mat          # (K, N)
    s = np.sign(proj_num)
    norm = np.abs(proj_num).sum()
    if norm == 0:
        return np.zeros_like(tilde_phi.T)
    # sum_m sign(proj_m) * tilde_phi[k, m] for each k.
    agg = (tilde_phi * s[None, :]).sum(axis=1)   # (K,)
    phi_R = (tilde_phi - w_stock_norm[None, :] * agg[:, None]) / norm  # (K, N)
    return phi_R.T  # (N, K)


# ---------------------------------------------------------------------------
# Portfolio-level propagation
# ---------------------------------------------------------------------------

@dataclass
class PortfolioAttribution:
    turnover: pd.DataFrame     # (date, feature)
    gross_ret: pd.DataFrame    # (date, feature)
    net_ret: pd.DataFrame      # (date, feature)
    mean_gross: pd.Series      # per feature (annualised)
    mean_net: pd.Series        # per feature (annualised)
    vol_gross: pd.Series       # per feature contribution to annualised vol
    vol_net: pd.Series
    sharpe_gross: pd.Series    # per feature contribution to annualised SR
    sharpe_net: pd.Series


def turnover_attribution(phi_stock_prev: np.ndarray,
                          phi_stock_prev2: np.ndarray,
                          w_prev: np.ndarray,
                          w_prev2: np.ndarray) -> np.ndarray:
    """Equation (26). ``phi_stock_prev`` shape (N, K)."""
    s = np.sign(w_prev - w_prev2)
    diff = phi_stock_prev - phi_stock_prev2
    return (s[:, None] * diff).sum(axis=0)        # (K,)


def gross_return_attribution(phi_stock_prev: np.ndarray,
                              ret_t: np.ndarray) -> np.ndarray:
    """Equation (27)."""
    return (phi_stock_prev * ret_t[:, None]).sum(axis=0)


def net_return_attribution(phi_gross: np.ndarray,
                            phi_turn: np.ndarray, c: float) -> np.ndarray:
    """Equation (28)."""
    return phi_gross - c * phi_turn


def mean_return_attribution(phi_ret_day: pd.DataFrame) -> pd.Series:
    """Equation (29). ``phi_ret_day`` indexed by date, columns are features."""
    return phi_ret_day.mean(axis=0) * 252.0


def vol_attribution(phi_ret_day: pd.DataFrame, ret_series: pd.Series) -> pd.Series:
    """Equation (31). Per-feature contribution to the annualised volatility.

    sigma_i = (1/sigma) * cov(phi_i(R), R) * sqrt(252)
    """
    sigma = float(ret_series.std(ddof=0))
    if sigma == 0:
        return pd.Series(0.0, index=phi_ret_day.columns)
    cov = phi_ret_day.sub(phi_ret_day.mean(axis=0)).mul(
        ret_series - ret_series.mean(), axis=0).mean(axis=0)
    return cov / sigma * np.sqrt(252.0)


def sharpe_attribution(mean_contrib: pd.Series, vol_contrib: pd.Series,
                        mu_total: float, sigma_total: float) -> pd.Series:
    """Equation (32). SR = mu/sigma -> phi_i(SR) = phi_i(mu)/sigma - mu * phi_i(sigma)/sigma^2."""
    if sigma_total == 0:
        return pd.Series(0.0, index=mean_contrib.index)
    return mean_contrib / sigma_total - mu_total * vol_contrib / (sigma_total ** 2)


def aggregate_portfolio_attribution(phi_ret_gross: pd.DataFrame,
                                     phi_ret_net: pd.DataFrame,
                                     phi_turn: pd.DataFrame,
                                     gross: pd.Series, net: pd.Series,
                                     ) -> PortfolioAttribution:
    mu_g = mean_return_attribution(phi_ret_gross)
    mu_n = mean_return_attribution(phi_ret_net)
    vol_g = vol_attribution(phi_ret_gross, gross)
    vol_n = vol_attribution(phi_ret_net, net)
    sg = sharpe_attribution(mu_g, vol_g, float(gross.mean() * 252),
                              float(gross.std(ddof=0) * np.sqrt(252)))
    sn = sharpe_attribution(mu_n, vol_n, float(net.mean() * 252),
                              float(net.std(ddof=0) * np.sqrt(252)))
    return PortfolioAttribution(
        turnover=phi_turn, gross_ret=phi_ret_gross, net_ret=phi_ret_net,
        mean_gross=mu_g, mean_net=mu_n, vol_gross=vol_g, vol_net=vol_n,
        sharpe_gross=sg, sharpe_net=sn,
    )


# ---------------------------------------------------------------------------
# High-level driver: build daily portfolio-level attributions
# ---------------------------------------------------------------------------

def build_daily_attributions(outputs, phi_mat: np.ndarray, ex: pd.DataFrame,
                               feature_names: list[str], c: float) -> dict:
    """Compute daily portfolio-level attribution tensors from pipeline outputs.

    Parameters
    ----------
    outputs : :class:`src.pipeline.ModelOutputs`
    phi_mat : np.ndarray of shape (T, N, N) — the per-day projection matrices
    ex      : excess-return DataFrame (T, N)
    feature_names : length-K feature names in the panel order
    c       : one-way transaction cost per unit turnover

    Returns
    -------
    dict with keys ``phi_turn`` (T, K), ``phi_gross`` (T, K),
    ``phi_net`` (T, K) all as DataFrames indexed by date.
    """
    from src.models import FFNAllocator, XGBAllocator  # local import

    dates = outputs.gross.index
    tickers = outputs.weights_eps.columns
    K = len(feature_names)
    N = len(tickers)

    phi_turn = pd.DataFrame(0.0, index=dates, columns=feature_names)
    phi_gross = pd.DataFrame(0.0, index=dates, columns=feature_names)
    phi_net = pd.DataFrame(0.0, index=dates, columns=feature_names)

    # Build a date -> (vintage_date, model) lookup.
    vintage_bounds = outputs.vintage_boundaries
    def _vintage_for(d):
        v = None
        for vb in vintage_bounds:
            if d >= vb:
                v = vb
        return v

    feat_df = outputs.features  # columns: date, ticker, <features>
    feat_df = feat_df.set_index(["date", "ticker"]).sort_index()

    all_dates = list(outputs.weights_eps.index)
    date_pos = {d: i for i, d in enumerate(all_dates)}

    prev_phi_stock = None  # (N, K)
    for d in dates:
        if d not in feat_df.index.get_level_values(0):
            prev_phi_stock = None
            continue
        day = feat_df.loc[d]
        if day.empty:
            prev_phi_stock = None
            continue
        day = day.reindex(tickers)
        X = day[feature_names].to_numpy(float)
        valid = np.isfinite(X).all(axis=1)
        X_valid = X[valid]
        if X_valid.shape[0] == 0:
            prev_phi_stock = None
            continue
        v = _vintage_for(d)
        model = outputs.vintages.get(v)
        if model is None:
            prev_phi_stock = None
            continue
        if isinstance(model, XGBAllocator):
            sv = tree_shap_values(model.model, X_valid)
        elif isinstance(model, FFNAllocator):
            vt = outputs.val_feature_tables.get(v)
            bg = vt[feature_names].dropna().to_numpy(float)[-256:] if vt is not None else X_valid[:32]
            sv = deep_shap_values(model.model, X_valid, bg)
        else:
            prev_phi_stock = None
            continue
        # Scatter sv back to full N x K.
        phi_eps = np.zeros((N, K))
        phi_eps[valid] = sv

        # proj_num and stock weights on day d.
        pi = date_pos[d]
        w_eps = outputs.weights_eps.iloc[pi].to_numpy(float)
        proj_num = w_eps @ phi_mat[pi]
        w_stock_norm = outputs.weights_stock.iloc[pi].to_numpy(float)
        phi_stock = propagate_to_stock_weights(phi_eps, phi_mat[pi], w_stock_norm, proj_num)

        # Daily returns on day d+1 realise the gross return R_{0, d+1}.
        if pi + 1 < len(all_dates):
            r_next = ex.iloc[pi + 1].to_numpy(float)
            r_next = np.nan_to_num(r_next)
        else:
            r_next = np.zeros(N)

        pg = gross_return_attribution(phi_stock, r_next)
        if prev_phi_stock is not None:
            w_prev = outputs.weights_stock.iloc[pi].to_numpy(float)
            w_prev2 = outputs.weights_stock.iloc[pi - 1].to_numpy(float)
            pt = turnover_attribution(phi_stock, prev_phi_stock, w_prev, w_prev2)
        else:
            pt = np.zeros(K)
        pn = net_return_attribution(pg, pt, c)
        phi_gross.loc[d] = pg
        phi_net.loc[d] = pn
        phi_turn.loc[d] = pt
        prev_phi_stock = phi_stock

    return {"phi_turn": phi_turn, "phi_gross": phi_gross, "phi_net": phi_net}


__all__ = [
    "tree_shap_values", "deep_shap_values",
    "propagate_to_stock_weights", "turnover_attribution",
    "gross_return_attribution", "net_return_attribution",
    "mean_return_attribution", "vol_attribution", "sharpe_attribution",
    "aggregate_portfolio_attribution", "PortfolioAttribution",
    "build_daily_attributions",
]
