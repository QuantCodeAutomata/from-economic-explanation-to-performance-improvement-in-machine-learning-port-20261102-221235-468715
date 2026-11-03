"""Experiment 5 — Attribution-guided feature-exposure interventions.

Applies four intervention rules (model-specific unstabilised/stabilised,
cross-model unstabilised/stabilised) by scaling the standardised features of
each OOS year with multipliers derived from the preceding validation-period
net-Sharpe attribution. Models are not retrained.

Writes:

* ``results/exp5_summary.csv`` — baseline and intervention portfolio summary
  (Table 7).
* ``results/exp5_alpha.csv`` — Fama-French 5-factor alpha and Newey-West
  t-statistics (Table 8).
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from data.data_loader import load_market_data
from src import config
from src.attribution import build_daily_attributions
from src.pipeline import prepare_core, run_allocation, summarise


def _lambda_from_attr(attr_vec: np.ndarray, eta: float) -> np.ndarray:
    """Equation (42): f_eta(T) = 1 + eta * tanh(T / sd(T))."""
    sd = np.std(attr_vec)
    if sd == 0 or not np.isfinite(sd):
        return np.ones_like(attr_vec)
    return 1.0 + eta * np.tanh(attr_vec / sd)


def _bootstrap_sr_improvement(a: pd.Series, b: pd.Series, block: int,
                                B: int, rng: np.random.Generator) -> float:
    """One-sided block-bootstrap p-value for SR(b) > SR(a)."""
    a = a.dropna().values; b = b.dropna().values
    n = min(len(a), len(b))
    if n < block * 2:
        return np.nan
    a = a[-n:]; b = b[-n:]
    def sr(x):
        s = x.std(ddof=0)
        return x.mean() / s * np.sqrt(252.0) if s > 0 else 0.0
    diff = sr(b) - sr(a)
    k = n // block
    cnt = 0
    for _ in range(B):
        starts = rng.integers(0, n, size=k)
        idx = (starts[:, None] + np.arange(block)[None, :]).reshape(-1) % n
        ab = a[idx] - a.mean(); bb = b[idx] - b.mean()
        if sr(bb) - sr(ab) >= diff:
            cnt += 1
    return cnt / B


def _factor_alpha(ret: pd.Series, factors: pd.DataFrame) -> tuple[float, float]:
    """Annualised alpha and Newey-West t-statistic."""
    df = pd.concat([ret.rename("y"),
                      factors[["mkt_rf", "smb", "hml", "rmw", "cma"]]], axis=1).dropna()
    if df.empty:
        return (np.nan, np.nan)
    X = sm.add_constant(df.drop(columns=["y"]))
    model = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    alpha_daily = float(model.params["const"])
    t_alpha = float(model.tvalues["const"])
    return alpha_daily * 252.0, t_alpha


def main(full: bool = False) -> None:
    results = Path(config.RESULTS_DIR)
    results.mkdir(parents=True, exist_ok=True)

    universe = config.UNIVERSE_SIZE_PAPER if full else config.UNIVERSE_SIZE_RUN
    returns, volumes, closes, factors = load_market_data(
        universe, config.DATA_START, config.DATA_END)
    ex, phi_mat, eps, feats = prepare_core(returns, volumes, closes, factors)
    feature_names = list(config.FEATURE_NAMES)
    K = len(feature_names)

    baselines = {}
    for name in ("XGB", "FFN"):
        p = results / f"exp1_outputs_{name}.pkl"
        if not p.exists():
            print(f"[exp5] missing {p}; run exp1 first")
            return
        with open(p, "rb") as fh:
            baselines[name] = pickle.load(fh)

    # Compute validation-year attribution vectors per vintage for each model.
    attribs = {}
    for name, out in baselines.items():
        path = results / f"exp3_attribution_{name}.pkl"
        if path.exists():
            with open(path, "rb") as fh:
                attribs[name] = pickle.load(fh)
        else:
            attribs[name] = build_daily_attributions(
                out, phi_mat, ex, feature_names, config.TRANSACTION_COST)
            with open(path, "wb") as fh:
                pickle.dump(attribs[name], fh)

    # Build the yearly validation attribution signal for each model: sum of
    # phi_i(R_{c,t}) over the validation window preceding each vintage.
    vintage_lambda_specific = {name: {} for name in baselines}
    for name, out in baselines.items():
        boundaries = out.vintage_boundaries
        phi_net = attribs[name]["phi_net"]
        lam_prev = np.ones(K)
        for i, vb in enumerate(boundaries):
            val_start = vb - pd.DateOffset(years=1)
            val_slice = phi_net.loc[(phi_net.index >= val_start)
                                      & (phi_net.index < vb)]
            if val_slice.empty:
                vintage_lambda_specific[name][vb] = {"unstab": np.ones(K),
                                                      "stab":   np.ones(K)}
                continue
            T_vec = val_slice.sum(axis=0).values
            lam_un = _lambda_from_attr(T_vec, config.INTERVENTION_ETA)
            lam_st = np.clip(lam_prev * lam_un, config.INTERVENTION_LAMBDA_MIN,
                             config.INTERVENTION_LAMBDA_MAX)
            vintage_lambda_specific[name][vb] = {"unstab": lam_un, "stab": lam_st}
            lam_prev = lam_st

    # Cross-model lambdas.
    vintage_lambda_cross = {"unstab": {}, "stab": {}}
    all_vbs = sorted(set(sum(([vb for vb in d] for d in
                             [vintage_lambda_specific["XGB"],
                              vintage_lambda_specific["FFN"]]), [])))
    for vb in all_vbs:
        lx = vintage_lambda_specific["XGB"].get(vb, {"unstab": np.ones(K), "stab": np.ones(K)})
        lf = vintage_lambda_specific["FFN"].get(vb, {"unstab": np.ones(K), "stab": np.ones(K)})
        vintage_lambda_cross["unstab"][vb] = 0.5 * (lx["unstab"] + lf["unstab"])
        vintage_lambda_cross["stab"][vb] = 0.5 * (lx["stab"] + lf["stab"])

    def _lookup_factory(d):
        def _lookup(vb):
            return d.get(vb, np.ones(K))
        return _lookup

    rng = np.random.default_rng(config.SEED)
    B = min(999 if not full else config.BOOTSTRAP_B, 999)

    rows = []
    alpha_rows = []
    oos_start = pd.Timestamp(config.OOS_START)
    oos_end = pd.Timestamp(config.OOS_END)
    for name in ("XGB", "FFN"):
        base = baselines[name]
        s_base = summarise(base)
        a_g, t_g = _factor_alpha(base.gross, factors)
        a_n, t_n = _factor_alpha(base.net, factors)
        rows.append({"model": name, "rule": "baseline", **s_base})
        alpha_rows.append({"model": name, "rule": "baseline",
                             "alpha_gross": a_g, "t_gross": t_g,
                             "alpha_net": a_n, "t_net": t_n})
        for rule_name, source in (
            ("specific_unstab", {vb: d["unstab"] for vb, d in vintage_lambda_specific[name].items()}),
            ("specific_stab",   {vb: d["stab"]   for vb, d in vintage_lambda_specific[name].items()}),
            ("cross_unstab",     vintage_lambda_cross["unstab"]),
            ("cross_stab",       vintage_lambda_cross["stab"]),
        ):
            print(f"[exp5] running {name} / {rule_name}")
            out = run_allocation(name, feats, eps, phi_mat, ex, oos_start, oos_end,
                                   exposure_lambda_lookup=_lookup_factory(source))
            s = summarise(out)
            pval = _bootstrap_sr_improvement(base.net, out.net,
                                               config.BOOTSTRAP_BLOCK, B, rng)
            a_g, t_g = _factor_alpha(out.gross, factors)
            a_n, t_n = _factor_alpha(out.net, factors)
            rows.append({"model": name, "rule": rule_name,
                           **s, "p_sr_improvement_net": pval})
            alpha_rows.append({"model": name, "rule": rule_name,
                                 "alpha_gross": a_g, "t_gross": t_g,
                                 "alpha_net": a_n, "t_net": t_n})

    pd.DataFrame(rows).to_csv(results / "exp5_summary.csv", index=False)
    pd.DataFrame(alpha_rows).to_csv(results / "exp5_alpha.csv", index=False)
    print("[exp5] done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    main(full=args.full)
