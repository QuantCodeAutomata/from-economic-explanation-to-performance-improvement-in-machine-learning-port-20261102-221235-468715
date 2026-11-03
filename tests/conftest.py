"""Shared pytest fixtures.

Registers the ``market_data`` marker and provides a reduced OHLCV slice loaded
through ``data.data_loader.load_market_data`` for tests that need real data.
Skipped when ``DATALOADER_API_KEY`` is unset so the suite stays runnable
without network access.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "market_data: tests that require the QuantX Dataloader API key"
    )


@pytest.fixture(scope="session")
def market_slice(request):
    if not os.environ.get("DATALOADER_API_KEY"):
        pytest.skip("DATALOADER_API_KEY not set; skipping market-data test")
    from data.data_loader import load_market_data
    cache = request.config.cache.mkdir("market_slice")
    returns, volumes, closes, factors = load_market_data(
        universe_size=5, start="2010-01-01", end="2011-12-31")
    return returns, volumes, closes, factors


@pytest.fixture(scope="session")
def synthetic_panel():
    """Deterministic synthetic OHLCV + factor panel for unit tests."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2004-01-02", "2008-12-31")
    tickers = [f"T{i:02d}" for i in range(10)]
    T, N = len(dates), len(tickers)
    # Correlated returns through a 3-factor structure.
    factors = pd.DataFrame(
        rng.normal(0, 0.01, (T, 5)),
        index=dates, columns=["mkt_rf", "smb", "hml", "rmw", "cma"])
    factors["rf"] = 0.00005
    betas = rng.normal(0, 0.5, (N, 5))
    stock_rets = factors[["mkt_rf", "smb", "hml", "rmw", "cma"]].values @ betas.T
    stock_rets += rng.normal(0, 0.015, (T, N))
    returns = pd.DataFrame(stock_rets, index=dates, columns=tickers)
    closes = (1.0 + returns).cumprod() * 100.0
    volumes = pd.DataFrame(rng.integers(1_000_000, 5_000_000, (T, N)),
                             index=dates, columns=tickers).astype(float)
    return returns, volumes, closes, factors
