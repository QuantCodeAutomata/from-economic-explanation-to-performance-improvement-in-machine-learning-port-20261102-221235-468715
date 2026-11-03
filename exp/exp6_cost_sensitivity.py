"""Experiment 6 — Transaction-cost sensitivity of attribution-guided
interventions.

For each model, re-derives validation-period net-Sharpe attributions under
``c_IS in {1,3,5,7,9} bps``, applies the stabilised model-specific exposure
rule, and evaluates OOS annualised net Sharpe under
``c_OOS in {0,1,...,10} bps``. Writes the resulting grid to
``results/exp6_cost_grid_<model>.csv``.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from data.data_loader import load_market_data
from src import config
from src.attribution import build_daily_attributions
from src.pipeline import prepare_core, run_allocation


def _lambda_from_attr(T_vec: np.ndarray, eta: float) -> np.ndarray:
    sd = np.std(T_vec)
    if sd == 0 or not np.isfinite(sd):
        return np.ones_like(T_vec)
    return 1.0 + eta * np.tanh(T_vec / sd)


def _net_sharpe(gross: pd.Series, turnover: pd.Series, c: float) -> float:
    r = (gross - c * turnover).dropna()
    s = r.std(ddof=0)
    return float(r.mean() / s * np.sqrt(252.0)) if s > 0 else 0.0


def main(full: bool = False) -> None:
    results = Path(config.RESULTS_DIR)
    universe = config.UNIVERSE_SIZE_PAPER if full else config.UNIVERSE_SIZE_RUN
    returns, volumes, closes, factors = load_market_data(
        universe, config.DATA_START, config.DATA_END)
    ex, phi_mat, eps, feats = prepare_core(returns, volumes, closes, factors)
    feature_names = list(config.FEATURE_NAMES)
    K = len(feature_names)

    c_is_grid_bps = [1, 3, 5, 7, 9]
    c_oos_grid_bps = list(range(0, 11))
    oos_start = pd.Timestamp(config.OOS_START)
    oos_end = pd.Timestamp(config.OOS_END)

    for name in ("XGB", "FFN"):
        p = results / f"exp1_outputs_{name}.pkl"
        if not p.exists():
            print(f"[exp6] missing {name}")
            continue
        with open(p, "rb") as fh:
            base = pickle.load(fh)
        boundaries = base.vintage_boundaries

        grid = pd.DataFrame(index=c_is_grid_bps, columns=c_oos_grid_bps, dtype=float)
        grid.index.name = "c_IS_bps"; grid.columns.name = "c_OOS_bps"
        for c_is in c_is_grid_bps:
            c_is_rate = c_is * 1e-4
            print(f"[exp6] {name}: c_IS={c_is} bps -> recompute attribution")
            daily = build_daily_attributions(base, phi_mat, ex, feature_names, c_is_rate)
            phi_net = daily["phi_net"]

            # Stabilised model-specific lambdas per vintage.
            vintage_lambda = {}
            lam_prev = np.ones(K)
            for vb in boundaries:
                val_start = vb - pd.DateOffset(years=1)
                val_slice = phi_net.loc[(phi_net.index >= val_start) & (phi_net.index < vb)]
                if val_slice.empty:
                    vintage_lambda[vb] = lam_prev.copy()
                    continue
                T_vec = val_slice.sum(axis=0).values
                lam_un = _lambda_from_attr(T_vec, config.INTERVENTION_ETA)
                lam_st = np.clip(lam_prev * lam_un, config.INTERVENTION_LAMBDA_MIN,
                                   config.INTERVENTION_LAMBDA_MAX)
                vintage_lambda[vb] = lam_st
                lam_prev = lam_st

            def _lookup(vb, d=vintage_lambda):
                return d.get(vb, np.ones(K))

            out = run_allocation(name, feats, eps, phi_mat, ex, oos_start, oos_end,
                                   exposure_lambda_lookup=_lookup)
            for c_oos in c_oos_grid_bps:
                grid.loc[c_is, c_oos] = _net_sharpe(out.gross, out.turnover, c_oos * 1e-4)
        grid.to_csv(results / f"exp6_cost_grid_{name}.csv")

        # Baseline net SR curve.
        base_curve = pd.Series(
            {c: _net_sharpe(base.gross, base.turnover, c * 1e-4) for c in c_oos_grid_bps},
            name="baseline"
        )
        base_curve.to_csv(results / f"exp6_baseline_curve_{name}.csv")
    print("[exp6] done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    main(full=args.full)
