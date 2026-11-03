"""Trading-signal features (Table 1, Section 2.1.2).

For each stock-date the following 18 features are built from the cumulative
residual process X_{t-1}^L (L = 30 trading days) and liquidity data:

    kappa, mu, sigma_ou, cum_resid, r2, std_signal,
    regime_entropy, regime_duration, regime_trans,
    real_res, pred_res,
    resid_std, resid_skew, resid_kurt, ent_res,
    adv, log_mcap, turnover_stock
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from src.config import FEATURE_LOOKBACK, FEATURE_NAMES, HMM_N_REGIMES
from src.hmm_regime import fit_regime_features


def _ou_fit(x: np.ndarray) -> tuple[float, float, float, float]:
    """Fit a discrete Ornstein-Uhlenbeck process to cumulative residuals.

    dX_t = kappa (mu - X_t) dt + sigma dW_t, discretised as an AR(1):
        X_t = a + b X_{t-1} + eta_t,
    with kappa = -log(b) / dt, mu = a / (1 - b),
    sigma_OU = sqrt(var(eta) * 2 kappa / (1 - b^2)).

    Returns (kappa, mu, sigma_ou, r2).
    """
    if np.isnan(x).any() or len(x) < 5:
        return np.nan, np.nan, np.nan, np.nan
    y = x[1:]
    xl = x[:-1]
    if np.std(xl) < 1e-12:
        return np.nan, np.nan, np.nan, np.nan
    b, a = np.polyfit(xl, y, 1)
    resid = y - (a + b * xl)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    if b <= 0 or b >= 1:
        return np.nan, np.nan, np.nan, r2
    kappa = -np.log(b)
    mu = a / (1 - b)
    var_eta = float(np.var(resid, ddof=1)) if len(resid) > 1 else np.nan
    denom = 1 - b * b
    sigma_ou = np.sqrt(var_eta * 2 * kappa / denom) if denom > 0 and var_eta > 0 else np.nan
    return float(kappa), float(mu), float(sigma_ou), float(r2)


def _rolling_window_stats(resid: np.ndarray, L: int = FEATURE_LOOKBACK) -> pd.DataFrame:
    """Compute per-date OU parameters, cumulative residual, R2 and moments.

    Vectorised: forms all rolling windows at once and solves the AR(1)
    regression X_t = a + b X_{t-1} + eta per window with closed-form OLS,
    which replaces an explicit Python/polyfit loop.
    """
    T = len(resid)
    cols = ["kappa", "mu", "sigma_ou", "cum_resid", "r2", "std_signal",
            "resid_std", "resid_skew", "resid_kurt"]
    out = {k: np.full(T, np.nan) for k in cols}
    if T <= L:
        return pd.DataFrame(out)
    r = resid.astype(float)
    # (T-L, L) rolling windows of raw residuals.
    idx = np.arange(T - L)[:, None] + np.arange(L)[None, :]
    W = r[idx]
    finite_mask = np.isfinite(W).all(axis=1)
    # Residual-level moments.
    out["resid_std"][L:] = W.std(axis=1, ddof=1)
    # scipy stats.skew/kurtosis along axis.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out["resid_skew"][L:] = stats.skew(W, axis=1, bias=False)
        out["resid_kurt"][L:] = stats.kurtosis(W, axis=1, bias=False)
    # Cumulative residual process X per window (zero-meaned).
    Wc = W - W.mean(axis=1, keepdims=True)
    X = np.cumsum(Wc, axis=1)                               # (M, L)
    cum = X[:, -1]
    out["cum_resid"][L:] = cum
    # AR(1) regression per window: y = a + b * xl.
    xl = X[:, :-1]
    y = X[:, 1:]
    xm = xl.mean(axis=1, keepdims=True)
    ym = y.mean(axis=1, keepdims=True)
    xd = xl - xm; yd = y - ym
    sxx = (xd * xd).sum(axis=1)
    sxy = (xd * yd).sum(axis=1)
    b = np.where(sxx > 1e-24, sxy / np.where(sxx == 0, 1.0, sxx), np.nan)
    a = ym.squeeze(1) - b * xm.squeeze(1)
    pred = a[:, None] + b[:, None] * xl
    resid_eta = y - pred
    ss_res = (resid_eta ** 2).sum(axis=1)
    ss_tot = ((y - ym) ** 2).sum(axis=1)
    r2 = np.where(ss_tot > 0, 1 - ss_res / np.where(ss_tot == 0, 1.0, ss_tot), np.nan)
    valid = finite_mask & (b > 0) & (b < 1) & np.isfinite(b)
    kappa = np.where(valid, -np.log(np.where(valid, b, 0.5)), np.nan)
    mu = np.where(valid, a / np.where(1 - b == 0, np.nan, 1 - b), np.nan)
    var_eta = resid_eta.var(axis=1, ddof=1)
    denom = 1 - b * b
    sigma_ou = np.where(valid & (var_eta > 0) & (denom > 0),
                         np.sqrt(np.where(valid, var_eta * 2 * np.maximum(kappa, 1e-12) /
                                            np.where(denom == 0, 1.0, denom), 0.0)),
                         np.nan)
    out["kappa"][L:] = np.where(valid, kappa, np.nan)
    out["mu"][L:] = np.where(valid, mu, np.nan)
    out["sigma_ou"][L:] = sigma_ou
    out["r2"][L:] = np.where(finite_mask, r2, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_eq = sigma_ou / np.sqrt(2 * kappa)
    out["std_signal"][L:] = np.where(np.isfinite(sigma_eq) & (sigma_eq > 0),
                                       (cum - mu) / np.where(sigma_eq == 0, 1.0, sigma_eq),
                                       np.nan)
    return pd.DataFrame(out)


def _liquidity_features(close: pd.Series, volume: pd.Series,
                         shares: pd.Series | None, L: int = FEATURE_LOOKBACK
                         ) -> pd.DataFrame:
    """Average daily dollar volume, log market cap, stock turnover (Table 1)."""
    dollar = (close * volume).astype(float)
    adv = dollar.rolling(L, min_periods=max(5, L // 2)).mean()
    if shares is None:
        # Proxy shares as a constant giving a reasonable log market cap scale;
        # the paper's values are not published so a stationary proxy keeps the
        # feature on a comparable scale across stocks.
        shares = pd.Series(1e8, index=close.index)
    mcap = (close * shares.reindex(close.index).ffill()).astype(float)
    log_mcap = np.log(mcap.replace(0, np.nan))
    turnover = (volume / shares.reindex(close.index).ffill()).astype(float)
    turnover_roll = turnover.rolling(L, min_periods=max(5, L // 2)).mean()
    return pd.DataFrame({"adv": np.log(adv.replace(0, np.nan)),
                          "log_mcap": log_mcap,
                          "turnover_stock": turnover_roll})


def build_feature_panel(residuals: pd.DataFrame, closes: pd.DataFrame,
                         volumes: pd.DataFrame,
                         shares_outstanding: pd.DataFrame | None = None,
                         L: int = FEATURE_LOOKBACK,
                         n_regimes: int = HMM_N_REGIMES,
                         ) -> pd.DataFrame:
    """Build the 18-feature stock-date panel (Table 1).

    Returns a long DataFrame with columns ``date, ticker`` plus the 18 feature
    columns in :data:`FEATURE_NAMES`.
    """
    rows: list[pd.DataFrame] = []
    for tic in residuals.columns:
        r = residuals[tic].astype(float).to_numpy()
        if np.isfinite(r).sum() < L + 60:
            continue
        base = _rolling_window_stats(r, L)
        base.index = residuals.index
        # Regime-conditional OU features from the HDP-HMM substitute.
        regime = fit_regime_features(r, residuals.index, n_regimes=n_regimes)
        close = closes[tic] if tic in closes.columns else pd.Series(np.nan, index=residuals.index)
        vol = volumes[tic] if tic in volumes.columns else pd.Series(np.nan, index=residuals.index)
        shares = None
        if shares_outstanding is not None and tic in shares_outstanding.columns:
            shares = shares_outstanding[tic]
        liq = _liquidity_features(close, vol, shares, L=L)
        feat = pd.concat([base, regime, liq], axis=1)
        feat["real_res"] = pd.Series(r, index=residuals.index)
        feat["ent_res"] = feat["regime_entropy"] * feat["real_res"]
        feat = feat.loc[:, list(FEATURE_NAMES)]
        feat["date"] = residuals.index
        feat["ticker"] = tic
        rows.append(feat)
    if not rows:
        raise RuntimeError("No stock produced valid features; check residual coverage.")
    out = pd.concat(rows, ignore_index=False).reset_index(drop=True)
    return out[["date", "ticker", *FEATURE_NAMES]]


def standardize_features(df: pd.DataFrame, feature_names: list[str],
                          fit_start: pd.Timestamp, fit_end: pd.Timestamp,
                          ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Cross-sectional time-pooled standardisation using the training window.

    Means and standard deviations are estimated on rows with ``fit_start <=
    date <= fit_end`` only, then applied to the whole frame; this prevents
    look-ahead at the feature level.
    """
    mask = (df["date"] >= fit_start) & (df["date"] <= fit_end)
    fit = df.loc[mask, feature_names].astype(float)
    mu = fit.mean()
    sd = fit.std(ddof=0).replace(0, 1.0)
    out = df.copy()
    out[feature_names] = (df[feature_names].astype(float) - mu) / sd
    out[feature_names] = out[feature_names].replace([np.inf, -np.inf], np.nan)
    return out, mu, sd


__all__ = ["build_feature_panel", "standardize_features"]
