"""Experiment 2 — Propagate model-level attributions to portfolio-level metrics.

Loads the pipeline outputs saved by ``exp.exp1_baseline`` and computes
portfolio-level feature contributions (Section 2.2.4) to turnover, gross and
net Sharpe ratios. Writes:

* ``results/exp2_contributions_<model>.csv`` — annualised mean, volatility and
  gross/net Sharpe contributions per feature (Figures 3, 5, 6).
* ``results/exp2_rolling_<model>_<metric>.csv`` — 63-day rolling mean of the
  daily attribution for the four representative features ``prob_signal``,
  ``mu``, ``pred_res``, ``pre_res`` (Figures 4 and 7).
* ``results/exp2_summary.png`` — bar chart comparing gross-vs-net Sharpe
  contributions across the 18 features.
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
from src.attribution import (aggregate_portfolio_attribution,
                               build_daily_attributions)
from src.pipeline import prepare_core


def main(full: bool = False) -> None:
    results = Path(config.RESULTS_DIR)
    results.mkdir(parents=True, exist_ok=True)

    universe = config.UNIVERSE_SIZE_PAPER if full else config.UNIVERSE_SIZE_RUN
    returns, volumes, closes, factors = load_market_data(
        universe, config.DATA_START, config.DATA_END)
    ex, phi, eps, _ = prepare_core(returns, volumes, closes, factors)

    feature_names = list(config.FEATURE_NAMES)
    rep_features = ["std_signal", "mu", "pred_res", "real_res"]

    bars = {}
    for name in ("XGB", "FFN"):
        p = results / f"exp1_outputs_{name}.pkl"
        if not p.exists():
            print(f"[exp2] skipping {name}: {p} missing (run exp1 first)")
            continue
        with open(p, "rb") as fh:
            out = pickle.load(fh)
        print(f"[exp2] computing attributions for {name}")
        daily = build_daily_attributions(out, phi, ex, feature_names,
                                           config.TRANSACTION_COST)
        agg = aggregate_portfolio_attribution(
            daily["phi_gross"], daily["phi_net"], daily["phi_turn"],
            out.gross, out.net)
        contribs = pd.DataFrame({
            "mean_gross_bps_per_yr": agg.mean_gross * 1e4,
            "mean_net_bps_per_yr":   agg.mean_net * 1e4,
            "vol_gross_bps_per_yr":  agg.vol_gross * 1e4,
            "vol_net_bps_per_yr":    agg.vol_net * 1e4,
            "sharpe_gross":          agg.sharpe_gross,
            "sharpe_net":            agg.sharpe_net,
            "mean_turnover":         daily["phi_turn"].mean(axis=0) * 252.0,
        })
        contribs.to_csv(results / f"exp2_contributions_{name}.csv")
        bars[name] = contribs

        for metric, df in (("gross", daily["phi_gross"]),
                            ("net", daily["phi_net"]),
                            ("turnover", daily["phi_turn"])):
            roll = df[[c for c in rep_features if c in df.columns]].rolling(
                config.ATTRIBUTION_ROLL_DAYS, min_periods=10).mean()
            roll.to_csv(results / f"exp2_rolling_{name}_{metric}.csv")

    if bars:
        fig, axes = plt.subplots(1, len(bars), figsize=(6 * len(bars), 5), sharey=True)
        if len(bars) == 1:
            axes = [axes]
        for ax, (name, df) in zip(axes, bars.items()):
            df_sorted = df.sort_values("sharpe_net")
            idx = np.arange(len(df_sorted))
            ax.barh(idx - 0.2, df_sorted["sharpe_gross"], 0.4, label="gross")
            ax.barh(idx + 0.2, df_sorted["sharpe_net"], 0.4, label="net")
            ax.set_yticks(idx)
            ax.set_yticklabels(df_sorted.index)
            ax.set_title(f"{name}: feature contribution to Sharpe (Figure 5/6)")
            ax.axvline(0.0, color="k", lw=0.5)
            ax.legend()
        fig.tight_layout()
        fig.savefig(results / "exp2_summary.png", dpi=120)
        plt.close(fig)
        pickle.dump(
            {name: df.index.tolist() for name, df in bars.items()},
            open(results / "exp2_features_order.pkl", "wb"),
        )
    print("[exp2] done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    main(full=args.full)
