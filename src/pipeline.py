"""End-to-end statistical-arbitrage pipeline.

This module glues together the loader, residual construction, feature panel,
rolling model training (XGB and FFN) and the portfolio evaluation. It produces,
for each allocation model:

    - daily residual weights ``w^eps_{t-1}`` per stock,
    - daily stock weights ``w^R_{t-1}``,
    - daily portfolio turnover ``tau_t`` and gross / net returns,
    - the per-stock standardised features used at prediction time, keyed by the
      model vintage that was active on that date.

Models are refit every :data:`src.config.RETRAIN_EVERY` trading days on a
5-year rolling window (4 years training + 1 year validation), as described in
Section 3.1 of the paper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from src.config import (FEATURE_NAMES, OOS_END, OOS_START, RETRAIN_EVERY, SEED,
                         TRAIN_YEARS, TRANSACTION_COST, VALIDATION_YEARS)
from src.features import build_feature_panel, standardize_features
from src.models import FFNAllocator, XGBAllocator, scores_to_weights
from src.portfolio import (annualised_metrics, gross_returns, net_returns,
                             portfolio_turnover, residual_to_stock_weights)
from src.residuals import excess_returns, residuals_and_projection, rolling_betas


# ---------------------------------------------------------------------------
# Pipeline outputs
# ---------------------------------------------------------------------------

@dataclass
class ModelOutputs:
    """Outputs produced by one allocation model over the full OOS window."""

    name: str
    weights_eps: pd.DataFrame                     # (date, ticker) residual weights
    weights_stock: pd.DataFrame                   # (date, ticker) stock weights
    turnover: pd.Series                           # daily portfolio turnover
    gross: pd.Series                              # daily gross excess returns
    net: pd.Series                                # daily net excess returns at c
    features: pd.DataFrame                        # standardised feature panel
    vintages: dict[pd.Timestamp, object] = field(default_factory=dict)
    vintage_boundaries: list[pd.Timestamp] = field(default_factory=list)
    val_feature_tables: dict[pd.Timestamp, pd.DataFrame] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_core(returns: pd.DataFrame, volumes: pd.DataFrame,
                  closes: pd.DataFrame, factors: pd.DataFrame,
                  cache_dir: str | None = "results"
                  ) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Compute excess returns, rolling betas, residuals and the feature panel.

    The HMM-based feature construction is expensive. When ``cache_dir`` is
    supplied the output is cached under ``<cache_dir>/_core_cache.pkl`` keyed
    on the shape and endpoints of the inputs so that subsequent experiments
    can reuse it without recomputing features.
    """
    key = None
    if cache_dir is not None:
        import hashlib
        import os
        import pickle

        sig = (
            returns.shape,
            str(returns.index.min()),
            str(returns.index.max()),
            tuple(returns.columns),
        )
        h = hashlib.md5(repr(sig).encode()).hexdigest()[:12]
        path = os.path.join(cache_dir, f"_core_cache_{h}.pkl")
        if os.path.exists(path):
            print(f"[pipeline] loading cached core features from {path}", flush=True)
            with open(path, "rb") as fh:
                return pickle.load(fh)
        key = path

    ex = excess_returns(returns, factors["rf"])
    betas = rolling_betas(ex, factors)
    eps, phi = residuals_and_projection(ex, factors, betas)
    feats = build_feature_panel(eps, closes, volumes)
    result = (ex, phi, eps, feats)

    if key is not None:
        import pickle
        os.makedirs(os.path.dirname(key) or ".", exist_ok=True)
        with open(key, "wb") as fh:
            pickle.dump(result, fh)
        print(f"[pipeline] cached core features to {key}", flush=True)
    return result


# ---------------------------------------------------------------------------
# Rolling training schedule
# ---------------------------------------------------------------------------

def _vintage_dates(dates: pd.DatetimeIndex, oos_start: pd.Timestamp,
                    oos_end: pd.Timestamp, retrain_every: int = RETRAIN_EVERY
                    ) -> list[pd.Timestamp]:
    oos_dates = dates[(dates >= oos_start) & (dates <= oos_end)]
    out = []
    for i in range(0, len(oos_dates), retrain_every):
        out.append(oos_dates[i])
    return out


def _split_windows(feats: pd.DataFrame, vintage_date: pd.Timestamp,
                    train_years: int = TRAIN_YEARS,
                    val_years: int = VALIDATION_YEARS,
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    val_end = vintage_date - pd.Timedelta(days=1)
    val_start = val_end - pd.DateOffset(years=val_years)
    train_end = val_start - pd.Timedelta(days=1)
    train_start = train_end - pd.DateOffset(years=train_years)
    train = feats[(feats["date"] >= train_start) & (feats["date"] <= train_end)]
    val = feats[(feats["date"] >= val_start) & (feats["date"] <= val_end)]
    return train, val


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_allocation(model_name: str,
                    feats: pd.DataFrame,
                    eps: pd.DataFrame,
                    phi: np.ndarray,
                    ex: pd.DataFrame,
                    oos_start: pd.Timestamp,
                    oos_end: pd.Timestamp,
                    retrain_every: int = RETRAIN_EVERY,
                    cost: float = TRANSACTION_COST,
                    exposure_lambda_lookup: Callable | None = None,
                    ) -> ModelOutputs:
    """Train the allocator with rolling vintages and build the portfolio.

    ``exposure_lambda_lookup`` is used by the intervention experiments: when
    supplied it must return a length-K vector of exposure multipliers for the
    date range (``vintage_start``, ``vintage_end``). The multipliers scale the
    standardised features before the model's forward pass; model parameters are
    not retrained (Section 2.3.4).
    """
    dates = eps.index
    tickers = eps.columns
    T = len(dates)
    N = len(tickers)
    feature_names = list(FEATURE_NAMES)

    vintages = _vintage_dates(dates, oos_start, oos_end, retrain_every)
    weights_eps = pd.DataFrame(0.0, index=dates, columns=tickers)
    collected_feats: list[pd.DataFrame] = []
    trained: dict[pd.Timestamp, object] = {}
    val_tables: dict[pd.Timestamp, pd.DataFrame] = {}
    # Precompute next-day residual return once (XGB target). Stacking inside
    # the vintage loop would otherwise rebuild it each iteration.
    _eps_next = eps.shift(-1).stack(dropna=False).rename("eps_next").reset_index()
    _eps_next.columns = ["date", "ticker", "eps_next"]

    for i, vdate in enumerate(vintages):
        print(f"[pipeline:{model_name}] vintage {i+1}/{len(vintages)} {vdate.date()}", flush=True)
        v_end = vintages[i + 1] - pd.Timedelta(days=1) if i + 1 < len(vintages) else oos_end
        train, val = _split_windows(feats, vdate)
        if train.empty or val.empty:
            continue
        combined = pd.concat([train, val], ignore_index=True)
        combined_std, mu, sd = standardize_features(
            combined, feature_names, train["date"].min(), train["date"].max())
        train_std = combined_std[combined_std["date"] <= train["date"].max()].dropna(subset=feature_names)
        val_std = combined_std[combined_std["date"] >= val["date"].min()].dropna(subset=feature_names)
        if train_std.empty or val_std.empty:
            continue
        merged = train_std.merge(_eps_next, on=["date", "ticker"],
                                   how="left").dropna(subset=["eps_next"])
        X = merged[feature_names].to_numpy(float)
        y = merged["eps_next"].to_numpy(float)

        if model_name == "XGB":
            model = XGBAllocator()
            model.fit(X, y, feature_names)
        elif model_name == "FFN":
            model = FFNAllocator()
            # Build feats_by_date for the training window.
            feats_by_date = []
            train_dates = sorted(train_std["date"].unique())
            for d in train_dates:
                day = train_std[train_std["date"] == d].sort_values("ticker")
                tic_idx = [tickers.get_loc(t) for t in day["ticker"] if t in tickers]
                z = day[feature_names].to_numpy(float)
                r_arr = np.zeros(len(day))
                try:
                    pos = dates.get_loc(d)
                except KeyError:
                    continue
                if pos + 1 >= T:
                    continue
                for j, t in enumerate(day["ticker"]):
                    if t in eps.columns:
                        r_arr[j] = eps.iloc[pos + 1][t] if np.isfinite(eps.iloc[pos + 1][t]) else 0.0
                m = np.isfinite(z).all(axis=1) & (np.abs(r_arr) < 1.0)
                feats_by_date.append((z, r_arr, m))
            model.fit(feats_by_date, feature_names)
        else:
            raise ValueError(f"Unknown model {model_name}")
        trained[vdate] = model
        val_tables[vdate] = val_std.copy()

        # Apply model to the OOS slice [vdate, v_end].
        oos_slice = combined_std[(combined_std["date"] >= vdate)
                                   & (combined_std["date"] <= v_end)].copy()
        if oos_slice.empty:
            # Standardisation above might not cover [vdate, v_end]; redo it on
            # the full feature panel using the training mu/sd.
            scope = feats[(feats["date"] >= vdate) & (feats["date"] <= v_end)].copy()
            scope[feature_names] = (scope[feature_names].astype(float) - mu) / sd
            oos_slice = scope
        oos_slice = oos_slice.dropna(subset=feature_names)
        if oos_slice.empty:
            continue
        if exposure_lambda_lookup is not None:
            lam = exposure_lambda_lookup(vdate)  # length-K np.ndarray
            oos_slice[feature_names] = oos_slice[feature_names].values * lam[np.newaxis, :]
        X_oos = oos_slice[feature_names].to_numpy(float)
        scores = model.predict_scores(X_oos)
        oos_slice = oos_slice.assign(_score=scores)
        wide = oos_slice.pivot_table(index="date", columns="ticker", values="_score")
        w = scores_to_weights(wide).reindex(index=dates, columns=tickers).fillna(0.0)
        mask = (dates >= vdate) & (dates <= v_end)
        weights_eps.loc[mask] = w.loc[mask].values
        collected_feats.append(oos_slice)

    weights_stock = residual_to_stock_weights(weights_eps, phi)
    tau = portfolio_turnover(weights_stock)
    gross = gross_returns(weights_stock, ex)
    net = net_returns(gross, tau, cost)

    oos_mask = (gross.index >= oos_start) & (gross.index <= oos_end)
    out = ModelOutputs(
        name=model_name,
        weights_eps=weights_eps,
        weights_stock=weights_stock,
        turnover=tau[oos_mask],
        gross=gross[oos_mask],
        net=net[oos_mask],
        features=pd.concat(collected_feats, ignore_index=True) if collected_feats else pd.DataFrame(),
        vintages=trained,
        vintage_boundaries=vintages,
        val_feature_tables=val_tables,
    )
    return out


def summarise(outputs: ModelOutputs) -> dict:
    """Table 2 metrics for a single model's OOS period."""
    g = annualised_metrics(outputs.gross)
    n = annualised_metrics(outputs.net)
    return {
        "gross_mean": g["mean"], "gross_vol": g["vol"], "gross_sharpe": g["sharpe"],
        "net_mean": n["mean"], "net_vol": n["vol"], "net_sharpe": n["sharpe"],
        "avg_turnover": float(outputs.turnover.mean()),
    }


__all__ = ["ModelOutputs", "prepare_core", "run_allocation", "summarise"]
