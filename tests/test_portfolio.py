"""Tests for the residual -> stock weight mapping and performance accounting."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio import (residual_to_stock_weights, portfolio_turnover,
                             gross_returns, net_returns, annualised_metrics)


def _toy_setup():
    dates = pd.bdate_range("2020-01-02", periods=10)
    tickers = ["A", "B", "C"]
    w_eps = pd.DataFrame(
        np.array([[0.5, -0.3, 0.2]] * 10), index=dates, columns=tickers)
    phi = np.broadcast_to(np.eye(3), (10, 3, 3)).copy()
    rets = pd.DataFrame(
        np.array([[0.01, -0.02, 0.005]] * 10), index=dates, columns=tickers)
    return w_eps, phi, rets


def test_l1_normalisation():
    w_eps, phi, _ = _toy_setup()
    w_stock = residual_to_stock_weights(w_eps, phi)
    # Each day's L1 norm must be 1 (Eq. 12).
    assert np.allclose(w_stock.abs().sum(axis=1), 1.0)


def test_zero_turnover_when_constant_weights():
    w_eps, phi, _ = _toy_setup()
    w_stock = residual_to_stock_weights(w_eps, phi)
    tau = portfolio_turnover(w_stock)
    # Constant weights -> turnover is zero after the first day.
    assert np.allclose(tau.iloc[1:], 0.0)


def test_net_leq_gross_with_positive_cost():
    _, _, rets = _toy_setup()
    gross = pd.Series(np.linspace(0.01, 0.02, 10), index=rets.index)
    tau = pd.Series(np.ones(10) * 0.5, index=rets.index)
    net = net_returns(gross, tau, c=0.0005)
    assert (net <= gross).all()


def test_annualised_sharpe_sign():
    r = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 1000))
    m = annualised_metrics(r)
    assert "sharpe" in m and "mean" in m and "vol" in m
    assert m["vol"] > 0 and m["sharpe"] == pytest.approx(m["mean"] / m["vol"])
