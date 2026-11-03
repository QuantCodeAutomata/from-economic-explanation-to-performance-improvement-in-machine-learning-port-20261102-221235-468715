"""Allocation models that map z_{n,t-1} to residual weights w^eps_{n,t-1}.

Two models are implemented:

* ``XGBAllocator`` - an :class:`xgboost.XGBRegressor` trained to predict the
  next-day residual return and whose cross-sectional predictions are rescaled
  into zero-net, unit-L1 weights at each date. The paper describes the XGB
  allocator as "a boosted tree ensemble" trained against the net Sharpe
  objective but does not report its exact loss specification; a cross-sectional
  rescaling of predicted residual returns is the standard long-short rule used
  for statistical arbitrage with boosted trees.
* ``FFNAllocator`` - a small PyTorch feed-forward network (32-16-8-4, ReLU,
  linear output) trained end-to-end against the portfolio-level differentiable
  net Sharpe loss by backpropagating through daily portfolio returns and
  turnover (Section 2.1.3).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import (FFN_BATCH_DAYS, FFN_DROPOUT, FFN_EPOCHS,
                         FFN_HIDDEN_SIZES, FFN_LEARNING_RATE, SEED,
                         TRANSACTION_COST, XGB_COLSAMPLE_BYTREE,
                         XGB_LEARNING_RATE, XGB_MAX_DEPTH,
                         XGB_MIN_CHILD_WEIGHT, XGB_N_ESTIMATORS,
                         XGB_SUBSAMPLE)

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _cross_sectional_rescale(scores: pd.DataFrame) -> pd.DataFrame:
    """Zero-net, L1-normalised cross-sectional weights."""
    centred = scores.sub(scores.mean(axis=1), axis=0)
    norm = centred.abs().sum(axis=1).replace(0, np.nan)
    weights = centred.div(norm, axis=0).fillna(0.0)
    return weights


# ---------------------------------------------------------------------------
# XGB allocator
# ---------------------------------------------------------------------------

@dataclass
class XGBAllocator:
    """Boosted-tree residual-return predictor with cross-sectional rescaling."""

    model: object = None
    feature_names: tuple[str, ...] = ()

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> None:
        import xgboost as xgb
        self.feature_names = tuple(feature_names)
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        self.model = xgb.XGBRegressor(
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            min_child_weight=XGB_MIN_CHILD_WEIGHT,
            subsample=XGB_SUBSAMPLE,
            colsample_bytree=XGB_COLSAMPLE_BYTREE,
            tree_method="hist",
            objective="reg:squarederror",
            random_state=SEED,
            n_jobs=4,
            verbosity=0,
        )
        self.model.fit(X[mask], y[mask])

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        mask = np.isfinite(X).all(axis=1)
        out = np.zeros(len(X), dtype=float)
        if mask.any():
            out[mask] = self.model.predict(X[mask])
        return out


# ---------------------------------------------------------------------------
# FFN allocator with portfolio-level Sharpe loss
# ---------------------------------------------------------------------------

if _HAS_TORCH:
    class _FFN(nn.Module):
        def __init__(self, in_dim: int,
                      hidden: tuple[int, ...] = FFN_HIDDEN_SIZES,
                      dropout: float = FFN_DROPOUT) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            d = in_dim
            for h in hidden:
                layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
                d = h
            layers.append(nn.Linear(d, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
            return self.net(x).squeeze(-1)
else:
    class _FFN:
        pass


@dataclass
class FFNAllocator:
    model: object = None
    feature_names: tuple[str, ...] = ()

    def fit(self, feats_by_date: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
            feature_names: list[str], c: float = TRANSACTION_COST,
            epochs: int = FFN_EPOCHS, lr: float = FFN_LEARNING_RATE,
            batch_days: int = FFN_BATCH_DAYS) -> None:
        """Fit the FFN by maximising the portfolio-level net Sharpe loss.

        ``feats_by_date`` is a list of ``(z_t, r_t, mask_t)`` triples where
        ``z_t`` has shape ``(N, K)`` and ``r_t`` has shape ``(N,)``. The loss
        back-propagates through the cross-sectional rescaling, next-day return
        and L1 turnover.
        """
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is not available; cannot fit FFNAllocator.")
        self.feature_names = tuple(feature_names)
        in_dim = len(feature_names)
        torch.manual_seed(SEED)
        self.model = _FFN(in_dim)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.model.train()
        T = len(feats_by_date)
        for ep in range(epochs):
            idx = np.random.default_rng(SEED + ep).permutation(T)
            for i in range(0, T, batch_days):
                batch_idx = idx[i: i + batch_days]
                net_terms = []
                prev_w = None
                # Process the batch in chronological order so that turnover
                # between consecutive dates is well defined.
                batch_idx = np.sort(batch_idx)
                for j, t in enumerate(batch_idx):
                    z, r, m = feats_by_date[t]
                    if m.sum() < 5:
                        prev_w = None
                        continue
                    z_t = torch.tensor(z[m], dtype=torch.float32)
                    r_t = torch.tensor(r[m], dtype=torch.float32)
                    scores = self.model(z_t)
                    scores = scores - scores.mean()
                    nrm = scores.abs().sum() + 1e-8
                    w = scores / nrm
                    gross = (w * r_t).sum()
                    if prev_w is not None and prev_w.shape == w.shape:
                        tau = (w - prev_w).abs().sum()
                    else:
                        tau = torch.zeros((), dtype=gross.dtype)
                    net_terms.append(gross - c * tau)
                    prev_w = w
                if len(net_terms) < 2:
                    continue
                net = torch.stack(net_terms)
                mu = net.mean()
                sd = net.std() + 1e-6
                loss = -mu / sd
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
        self.model.eval()

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        if not _HAS_TORCH or self.model is None:
            raise RuntimeError("FFN model is not initialised.")
        mask = np.isfinite(X).all(axis=1)
        out = np.zeros(len(X), dtype=float)
        with torch.no_grad():
            if mask.any():
                z = torch.tensor(X[mask], dtype=torch.float32)
                out[mask] = self.model(z).cpu().numpy()
        return out


# ---------------------------------------------------------------------------
# Shared cross-sectional allocation step
# ---------------------------------------------------------------------------

def scores_to_weights(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Residual weights w^eps_{t} from model scores (zero-net, L1-normalised)."""
    return _cross_sectional_rescale(scores_df)


__all__ = ["XGBAllocator", "FFNAllocator", "scores_to_weights"]
