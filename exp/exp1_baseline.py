"""Experiment 1 — Baseline statistical-arbitrage portfolios.

Reproduces the baseline out-of-sample gross and net performance of the XGB and
FFN allocation models trained on the 18-feature trading-signal panel described
in Table 1 (Section 2.1 of the paper). Writes:

* ``results/exp1_summary.csv`` — gross and net annualised Sharpe, mean return,
  volatility and average portfolio turnover for XGB and FFN, next to the
  values reported in Table 2.
* ``results/exp1_cumulative.png`` — cumulative gross and net returns
  (Figure 1).
* ``results/exp1_turnover.png`` — daily portfolio turnover (Figure 2).
* ``results/exp1_outputs_<model>.pkl`` — the full :class:`src.pipeline.ModelOutputs`
  object cached for the downstream attribution experiments.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.data_loader import load_market_data
from src import config
from src.pipeline import prepare_core, run_allocation, summarise


# Paper's Table 2 reference numbers (p. 28).
PAPER_TABLE2 = {
    "XGB":  dict(gross_sharpe=1.0207, net_sharpe=0.6175, avg_turnover=0.1321),
    "FFN":  dict(gross_sharpe=1.3051, net_sharpe=0.5750, avg_turnover=0.2670),
}


def main(full: bool = False) -> None:
    results_dir = Path(config.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    universe = config.UNIVERSE_SIZE_PAPER if full else config.UNIVERSE_SIZE_RUN
    returns, volumes, closes, factors = load_market_data(
        universe, config.DATA_START, config.DATA_END)
    print(f"[exp1] loaded returns {returns.shape}, factors {factors.shape}")

    ex, phi, eps, feats = prepare_core(returns, volumes, closes, factors)
    print(f"[exp1] features {feats.shape}, residuals {eps.shape}")

    oos_start = pd.Timestamp(config.OOS_START)
    oos_end = pd.Timestamp(config.OOS_END)

    rows = []
    outputs = {}
    for name in ("XGB", "FFN"):
        print(f"[exp1] training {name}")
        out = run_allocation(name, feats, eps, phi, ex, oos_start, oos_end)
        s = summarise(out)
        outputs[name] = out
        rows.append({"model": name, **s,
                     "paper_gross_sharpe": PAPER_TABLE2[name]["gross_sharpe"],
                     "paper_net_sharpe":   PAPER_TABLE2[name]["net_sharpe"],
                     "paper_avg_turnover": PAPER_TABLE2[name]["avg_turnover"]})
        with open(results_dir / f"exp1_outputs_{name}.pkl", "wb") as fh:
            pickle.dump(out, fh)
        print(f"[exp1] {name}:", {k: round(float(v), 4) for k, v in s.items()})

    summary = pd.DataFrame(rows).set_index("model")
    summary.to_csv(results_dir / "exp1_summary.csv")
    print(summary)

    # Cumulative returns plot.
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, out in outputs.items():
        (1.0 + out.gross).cumprod().plot(ax=ax, label=f"{name} gross", lw=1.2)
        (1.0 + out.net).cumprod().plot(ax=ax, label=f"{name} net", lw=1.2, ls="--")
    ax.set_title("Cumulative gross and net excess returns (Figure 1)")
    ax.set_ylabel("Cumulative return (1 = start)")
    ax.set_xlabel("Date")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "exp1_cumulative.png", dpi=120)
    plt.close(fig)

    # Turnover plot.
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, out in outputs.items():
        out.turnover.rolling(21).mean().plot(ax=ax, label=f"{name} 21-day avg", lw=1.0)
    ax.set_title("Daily portfolio turnover, 21-day moving average (Figure 2)")
    ax.set_ylabel("tau_t")
    ax.set_xlabel("Date")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "exp1_turnover.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                     help="Use the paper's full configuration (346 stocks).")
    args = ap.parse_args()
    main(full=args.full)
