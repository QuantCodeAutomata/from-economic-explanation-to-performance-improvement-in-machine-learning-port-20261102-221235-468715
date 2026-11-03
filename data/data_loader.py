"""Market-data loader for the replication.

Daily OHLCV bars are fetched through the QuantX Dataloader REST API
(``POST https://quantx-api.limex.pro/apis/dataloader/fetch-ohlcvs``). Bars are
dividend-adjusted (the paper constructs gross stock returns, equation (1), from
total-return prices).

Fama-French 5-factor daily data are downloaded from the Kenneth French Data
Library through ``pandas_datareader``.

A local parquet cache under ``data/cache/`` avoids re-downloading between runs;
the cache directory is gitignored.
"""
from __future__ import annotations

import io
import os
import time
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

DATALOADER_BASE = "https://quantx-api.limex.pro/apis/dataloader"
DATALOADER_URL = f"{DATALOADER_BASE}/fetch-ohlcvs"
SAVE_URL = f"{DATALOADER_BASE}/save-dataframe"
ARTIFACT_URL = f"{DATALOADER_BASE}/artifacts"
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)


# ---------------------------------------------------------------------------
# Universe selection
# ---------------------------------------------------------------------------

# A hand-curated roster of continuously-trading US large caps that are present
# over most of 2001-2024. The paper does not list its 346 tickers; this
# roster is used as the point-in-time universe starting from the sample start
# and is sized by ``UNIVERSE_SIZE_RUN``. Delistings and renames mean the actual
# universe narrows during the sample; this is noted in results/RESULTS.md.
LARGE_CAP_UNIVERSE: list[str] = [
    "AAPL", "MSFT", "JPM", "JNJ", "PG", "KO", "PEP", "WMT", "XOM", "CVX",
    "IBM", "GE", "MRK", "PFE", "INTC", "CSCO", "ORCL", "T", "VZ", "HD",
    "MCD", "DIS", "BA", "CAT", "MMM", "AXP", "GS", "MS", "C", "BAC",
    "WFC", "USB", "TXN", "QCOM", "ADBE", "COST", "LOW", "NKE", "SBUX",
    "UNH", "LLY", "ABT", "BMY", "AMGN", "GILD", "DUK", "SO", "D",
    "EMR", "ITW", "HON", "LMT", "RTX", "NOC", "GD", "COP", "SLB",
    "APD", "DD", "DOW", "PPG", "NEM", "FCX", "SYY", "MDT", "WBA", "CL",
    "KMB", "GIS", "K", "HSY", "MO", "PM", "ADP", "PAYX", "EFX",
    "MMC", "AON", "TRV", "ALL", "MET", "PRU", "AFL", "SPG", "PLD",
    "EQR", "AVB", "ESS", "WY", "ED", "EXC", "AEP", "XEL",
    "PCG", "ETR", "NEE", "SRE", "WEC", "DTE", "CMS", "AWK",
    "BK", "STT", "NTRS", "RF", "KEY", "FITB", "HBAN", "PNC",
    "EL", "CLX", "CHD", "CPB", "CAG", "MKC", "SJM", "KR",
    "ADM", "BG", "APA", "DVN", "MRO", "EOG", "HES", "PSX",
    "VLO", "MPC", "OXY", "HAL", "BKR", "OKE", "KMI", "WMB",
    "F", "GM", "LEG", "WHR", "GPC", "AZO", "ORLY", "AAP",
    "ROK", "DOV", "SWK", "PCAR", "CMI", "NUE", "STLD", "AA",
    "PH", "SNA", "GWW", "FAST", "WM", "RSG", "ECL", "SHW",
    "LIN",
]


def _cache_key(ticker: str, start: str, end: str) -> Path:
    return CACHE_DIR / f"ohlcv_{ticker}_{start}_{end}.parquet"


def _parse_ohlcv_response(js: dict, tickers: list[str]) -> pd.DataFrame:
    """Parse the Dataloader wide-format response into a long OHLCV frame."""
    result = js.get("result") or {}
    ohlcv_df = result.get("ohlcv_df") or {}
    rows = ohlcv_df.get("rows") or []
    if not rows:
        return pd.DataFrame()
    wide = pd.DataFrame(rows)
    # Date column (API uses "Date").
    date_col = next((c for c in wide.columns if c.lower() == "date"), None)
    if date_col is None:
        return pd.DataFrame()
    wide[date_col] = pd.to_datetime(wide[date_col], utc=True).dt.tz_localize(None).dt.normalize()
    frames: list[pd.DataFrame] = []
    for tic in tickers:
        prefix = f"{tic}_"
        cols = {c: c[len(prefix):].lower() for c in wide.columns if c.startswith(prefix)}
        if not cols:
            # Single-ticker response: bare open/high/low/close/volume columns.
            cols = {c: c.lower() for c in wide.columns
                    if c.lower() in {"open", "high", "low", "close", "volume"}}
            if not cols:
                continue
        sub = wide[[date_col] + list(cols.keys())].rename(columns={date_col: "date", **cols})
        sub["ticker"] = tic
        sub = sub.dropna(subset=["close"]).sort_values("date")
        if not sub.empty:
            frames.append(sub)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _auth_headers() -> dict[str, str]:
    key = os.environ.get("DATALOADER_API_KEY")
    if not key:
        raise RuntimeError("DATALOADER_API_KEY is not set; cannot fetch market data.")
    return {"Authorization": f"Bearer {key}"}


def _wide_to_long(wide: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Convert a wide (date + ticker-prefixed OHLCV columns) frame to long form."""
    date_col = next((c for c in wide.columns if c.lower() == "date"), None)
    if date_col is None:
        return pd.DataFrame()
    wide = wide.copy()
    wide[date_col] = pd.to_datetime(wide[date_col], utc=True).dt.tz_localize(None).dt.normalize()
    frames: list[pd.DataFrame] = []
    lower_tickers = {t.lower(): t for t in tickers}
    for tlow, tup in lower_tickers.items():
        prefix = f"{tlow}_"
        cols = {c: c[len(prefix):].lower() for c in wide.columns if c.lower().startswith(prefix)}
        if not cols:
            continue
        sub = wide[[date_col] + list(cols.keys())].rename(columns={date_col: "date", **cols})
        sub["ticker"] = tup
        sub = sub.dropna(subset=["close"]).sort_values("date")
        if not sub.empty:
            frames.append(sub)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _request_dataloader_batch(tickers: list[str], start: str, end: str,
                               timeframe: str = "1d") -> pd.DataFrame:
    """Call the QuantX Dataloader for a batch of tickers.

    For large batches (payload > ~15 MB after serialization), the API stores the
    result and returns a ``result_id``. We then use ``save-dataframe`` to produce
    a parquet artifact and ``artifacts/{id}/download`` to retrieve it.
    """
    headers = _auth_headers()
    payload = {
        "tickers": tickers,
        "timeframe": timeframe,
        "start_date": start,
        "end_date": end,
        "adjust_dividends": True,
        "identity_date": end,
    }
    r = requests.post(DATALOADER_URL, json=payload, headers=headers, timeout=300)
    r.raise_for_status()
    js = r.json()
    result = js.get("result") or {}
    ohlcv_df = result.get("ohlcv_df") or {}
    rows = ohlcv_df.get("rows") or []
    if rows:
        return _wide_to_long(pd.DataFrame(rows), tickers)
    rid = result.get("result_id")
    if not rid:
        return pd.DataFrame()
    # Export as parquet artifact and download.
    sv = requests.post(SAVE_URL, json={
        "result_id": rid, "ticker": "BATCH", "dataset_name": "bars",
        "timeframe": timeframe, "start_date": start, "end_date": end,
        "file_format": "parquet", "data_kind": "ohlcv",
    }, headers=headers, timeout=300)
    sv.raise_for_status()
    aid = sv.json()["result"]["artifact_id"]
    dl = requests.get(f"{ARTIFACT_URL}/{aid}/download", headers=headers, timeout=120)
    dl.raise_for_status()
    url = dl.json()["result"]["url"]
    blob = requests.get(url, timeout=600).content
    wide = pd.read_parquet(io.BytesIO(blob))
    return _wide_to_long(wide, tickers)


def fetch_ohlcv_panel(tickers: Iterable[str], start: str, end: str,
                       batch_size: int = 20) -> pd.DataFrame:
    """Fetch a long-format OHLCV panel with per-ticker parquet caching."""
    tickers = list(tickers)
    frames: list[pd.DataFrame] = []
    pending: list[str] = []
    for tic in tickers:
        cache = _cache_key(tic, start, end)
        if cache.exists():
            df = pd.read_parquet(cache)
            if not df.empty:
                frames.append(df)
        else:
            pending.append(tic)
    for i in range(0, len(pending), batch_size):
        batch = pending[i: i + batch_size]
        try:
            long = _request_dataloader_batch(batch, start, end)
        except Exception as e:
            print(f"[loader] batch {batch[:3]}...: {type(e).__name__}: {e}")
            long = pd.DataFrame()
        for tic in batch:
            if long.empty:
                continue
            sub = long[long["ticker"] == tic].copy()
            if sub.empty:
                continue
            cache = _cache_key(tic, start, end)
            sub.to_parquet(cache)
            frames.append(sub)
        print(f"[loader] fetched batch {i//batch_size + 1}/{(len(pending)+batch_size-1)//batch_size}")
        time.sleep(0.1)
    if not frames:
        raise RuntimeError("No OHLCV data was fetched for the requested universe.")
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    return panel


# ---------------------------------------------------------------------------
# Fama-French factors
# ---------------------------------------------------------------------------

FF5_CACHE = CACHE_DIR / "ff5_daily.parquet"
FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)


def fetch_ff5_daily() -> pd.DataFrame:
    """Daily Fama-French 5 factors plus the risk-free rate (percent / 100)."""
    if FF5_CACHE.exists():
        return pd.read_parquet(FF5_CACHE)
    r = requests.get(FF5_URL, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        name = [n for n in zf.namelist() if n.endswith(".CSV") or n.endswith(".csv")][0]
        raw = zf.read(name).decode("latin-1")
    # Parse CSV: skip header lines until the row that starts with a date.
    lines = raw.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        cells = line.split(",")
        if cells and cells[0].strip().isdigit() and len(cells[0].strip()) == 8:
            start_idx = i
            break
    end_idx = len(lines)
    for i in range(start_idx, len(lines)):
        cells = lines[i].split(",")
        if not cells[0].strip().isdigit():
            end_idx = i
            break
    data_rows = [ln.split(",") for ln in lines[start_idx:end_idx]]
    df = pd.DataFrame(data_rows, columns=["date", "mkt_rf", "smb", "hml", "rmw", "cma", "rf"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    for c in ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]:
        df[c] = df[c].astype(float) / 100.0
    df.to_parquet(FF5_CACHE)
    return df


# ---------------------------------------------------------------------------
# Top-level loader used by experiments
# ---------------------------------------------------------------------------

def load_market_data(universe_size: int, start: str, end: str
                     ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (returns, volumes, closes, factors).

    ``returns``, ``volumes`` and ``closes`` are wide (date x ticker) frames of
    daily simple returns, daily share volume and adjusted close, aligned on
    business dates. ``factors`` is a date-indexed frame with columns
    ``mkt_rf``, ``smb``, ``hml``, ``rmw``, ``cma``, ``rf``.
    """
    tickers = LARGE_CAP_UNIVERSE[: universe_size]
    panel = fetch_ohlcv_panel(tickers, start, end)
    panel = panel[panel["close"].notna() & (panel["close"] > 0)]
    panel = panel.sort_values(["ticker", "date"])
    panel["ret"] = panel.groupby("ticker")["close"].pct_change()
    returns = panel.pivot(index="date", columns="ticker", values="ret").sort_index()
    volumes = panel.pivot(index="date", columns="ticker", values="volume").sort_index()
    closes = panel.pivot(index="date", columns="ticker", values="close").sort_index()
    factors = fetch_ff5_daily().set_index("date").sort_index()
    factors = factors.reindex(returns.index).ffill()
    enough = returns.notna().sum() > 252
    keep = returns.columns[enough]
    returns = returns[keep]
    volumes = volumes[keep]
    closes = closes[keep]
    return returns, volumes, closes, factors
