"""Experiment 4 — Dependence, cross-model similarity, temporal persistence.

Produces (Section 3.2.4-3.3.1 of the paper):

* ``results/exp4_within_model.csv`` — mean and max off-diagonal absolute
  correlation among feature attribution time series per model/metric (Table 4).
* ``results/exp4_cross_model_yearly.csv`` — mean/median yearly cross-model
  correlation and sign consistency (Table 5).
* ``results/exp4_persistence.csv`` — validation-to-OOS persistence per model
  and metric (Table 6).
* ``results/exp4_persistence_horizons.csv`` — multi-horizon persistence.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, wilcoxon

from data.data_loader import load_market_data
from src import config
from src.attribution import build_daily_attributions
from src.pipeline import prepare_core


METRICS = ("gross", "net", "turnover")


def _load_or_compute_attrib(name, out, phi_mat, ex, feature_names, results):
    path = results / f"exp3_attribution_{name}.pkl"
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    daily = build_daily_attributions(out, phi_mat, ex, feature_names,
                                       config.TRANSACTION_COST)
    with open(path, "wb") as fh:
        pickle.dump(daily, fh)
    return daily


def _within_model(df: pd.DataFrame) -> tuple[float, float]:
    c = df.corr().abs().values
    np.fill_diagonal(c, np.nan)
    return float(np.nanmean(c)), float(np.nanmax(c))


def _sign_consistency(a: np.ndarray, b: np.ndarray) -> float:
    mask = (a != 0) & (b != 0)
    if mask.sum() == 0:
        return np.nan
    return float((np.sign(a[mask]) == np.sign(b[mask])).mean())


def main(full: bool = False) -> None:
    results = Path(config.RESULTS_DIR)
    universe = config.UNIVERSE_SIZE_PAPER if full else config.UNIVERSE_SIZE_RUN
    returns, volumes, closes, factors = load_market_data(
        universe, config.DATA_START, config.DATA_END)
    ex, phi_mat, eps, _ = prepare_core(returns, volumes, closes, factors)
    feature_names = list(config.FEATURE_NAMES)

    attribs = {}
    outputs = {}
    for name in ("XGB", "FFN"):
        p = results / f"exp1_outputs_{name}.pkl"
        if not p.exists():
            print(f"[exp4] skipping {name}")
            continue
        with open(p, "rb") as fh:
            outputs[name] = pickle.load(fh)
        attribs[name] = _load_or_compute_attrib(
            name, outputs[name], phi_mat, ex, feature_names, results)

    # 1. Within-model dependence summary.
    within_rows = []
    for name, att in attribs.items():
        for metric in METRICS:
            key = f"phi_{metric}" if metric != "turnover" else "phi_turn"
            # Approximate residual-weight attribution by projecting daily phi_gross
            # back to feature dimension; here we use gross as a proxy for residual
            # dependence because propagated phi_eps per day is not stored.
            if metric == "gross":
                df = att["phi_gross"]
            elif metric == "net":
                df = att["phi_net"]
            else:
                df = att["phi_turn"]
            m, mx = _within_model(df)
            within_rows.append({"model": name, "metric": metric,
                                  "mean_abs_corr": m, "max_abs_corr": mx})
    pd.DataFrame(within_rows).to_csv(results / "exp4_within_model.csv", index=False)

    # 2. Cross-model yearly similarity (Table 5).
    if "XGB" in attribs and "FFN" in attribs:
        rows = []
        for metric in ("gross", "net", "turnover"):
            key = {"gross": "phi_gross", "net": "phi_net", "turnover": "phi_turn"}[metric]
            dx = attribs["XGB"][key]; df = attribs["FFN"][key]
            common = dx.index.intersection(df.index)
            dx = dx.loc[common]; df = df.loc[common]
            yearly_rhos = []
            yearly_scs = []
            for y in sorted(set(common.year)):
                mask = common.year == y
                if mask.sum() < 20:
                    continue
                vx = dx.loc[mask].sum(axis=0).values
                vf = df.loc[mask].sum(axis=0).values
                if np.allclose(vx, 0) or np.allclose(vf, 0):
                    continue
                r, _ = pearsonr(vx, vf)
                yearly_rhos.append(r)
                yearly_scs.append(_sign_consistency(vx, vf))
            yearly_rhos = np.array(yearly_rhos); yearly_scs = np.array(yearly_scs)
            if len(yearly_rhos) > 0:
                try:
                    _, pval = wilcoxon(yearly_rhos, alternative="greater")
                except Exception:
                    pval = np.nan
            else:
                pval = np.nan
            rows.append({"metric": metric,
                           "n_years": len(yearly_rhos),
                           "mean_rho": float(np.mean(yearly_rhos)) if len(yearly_rhos) else np.nan,
                           "median_rho": float(np.median(yearly_rhos)) if len(yearly_rhos) else np.nan,
                           "pct_pos_rho": float(np.mean(yearly_rhos > 0)) if len(yearly_rhos) else np.nan,
                           "median_sc": float(np.median(yearly_scs)) if len(yearly_scs) else np.nan,
                           "wilcoxon_p": float(pval)})
        pd.DataFrame(rows).to_csv(results / "exp4_cross_model_yearly.csv", index=False)

    # 3. Validation-to-OOS persistence.
    pers_rows = []
    for name, out in outputs.items():
        att = attribs[name]
        # Validation window features are stored; apply the model to them to get
        # a validation-year attribution snapshot, then compare its yearly
        # cumulative vector with the subsequent OOS year's cumulative vector.
        oos_years = sorted(set(att["phi_net"].index.year))
        year_vectors = {}
        for y in oos_years:
            mask = att["phi_net"].index.year == y
            if mask.sum() < 20:
                continue
            year_vectors[y] = {
                "gross": att["phi_gross"].loc[mask].sum(axis=0).values,
                "net":   att["phi_net"].loc[mask].sum(axis=0).values,
                "turn":  att["phi_turn"].loc[mask].sum(axis=0).values,
            }
        years_sorted = sorted(year_vectors.keys())
        for metric in ("gross", "net", "turn"):
            rhos = []
            scs = []
            for i in range(1, len(years_sorted)):
                a = year_vectors[years_sorted[i - 1]][metric]
                b = year_vectors[years_sorted[i]][metric]
                if np.allclose(a, 0) or np.allclose(b, 0):
                    continue
                r, _ = pearsonr(a, b)
                rhos.append(r)
                scs.append(_sign_consistency(a, b))
            rhos = np.array(rhos); scs = np.array(scs)
            pers_rows.append({"model": name, "metric": metric,
                                "n": len(rhos),
                                "median_rho": float(np.median(rhos)) if len(rhos) else np.nan,
                                "median_sc": float(np.median(scs)) if len(scs) else np.nan})
        # Multi-horizon persistence (H = 1..8 years, non-overlapping windows).
        for H in range(1, min(9, len(years_sorted) // 2 + 1)):
            rhos = []
            n_h = len(years_sorted) - 2 * H + 1
            if n_h <= 0:
                continue
            for i in range(n_h):
                prev = np.sum([year_vectors[years_sorted[i + j]]["net"] for j in range(H)], axis=0)
                nxt = np.sum([year_vectors[years_sorted[i + H + j]]["net"] for j in range(H)], axis=0)
                if np.allclose(prev, 0) or np.allclose(nxt, 0):
                    continue
                r, _ = pearsonr(prev, nxt)
                rhos.append(r)
            pers_rows.append({"model": name, "metric": f"net_h{H}",
                                "n": len(rhos),
                                "median_rho": float(np.median(rhos)) if rhos else np.nan,
                                "median_sc": np.nan})
    pd.DataFrame(pers_rows).to_csv(results / "exp4_persistence.csv", index=False)
    print("[exp4] done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    main(full=args.full)
