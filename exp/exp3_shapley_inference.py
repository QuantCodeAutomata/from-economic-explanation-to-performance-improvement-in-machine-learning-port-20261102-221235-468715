"""Experiment 3 — Validate Sharpe attributions against Shapley values and
compute statistical inference via circular block bootstrap.

Writes:

* ``results/exp3_shapley_vs_firstorder.csv`` — correlations and RMSE between
  first-order Sharpe attributions and MC Shapley values per model and metric
  (Figures 8-9).
* ``results/exp3_significant_counts.csv`` — number of feature-year
  observations with Sharpe attribution significantly different from zero at
  the 5% level (Table 3).
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from data.data_loader import load_market_data
from src import config
from src.attribution import build_daily_attributions
from src.pipeline import prepare_core
from src.shapley import (circular_block_bootstrap_sharpe_attr,
                           monte_carlo_shapley, shapley_main_effects)


def _annualised_sharpe(x: np.ndarray) -> float:
    sd = x.std(ddof=0)
    if sd == 0:
        return 0.0
    return float(x.mean() / sd * np.sqrt(252.0))


def _value_fn_for_year(sub_phi_gross: pd.DataFrame, sub_phi_net: pd.DataFrame,
                        sub_r_gross: pd.Series, sub_r_net: pd.Series,
                        target: str):
    """Return v(S) = Sharpe of the sum of included feature contributions."""
    base_gross = sub_r_gross.to_numpy()
    base_net = sub_r_net.to_numpy()
    phi_g = sub_phi_gross.to_numpy()  # (T, K)
    phi_n = sub_phi_net.to_numpy()
    # The expected value of a coalition S is defined as the Sharpe ratio of the
    # explained portion sum_{i in S} phi_i(R_t).
    def fn(lam):
        if target == "gross":
            x = phi_g @ lam
        else:
            x = phi_n @ lam
        return _annualised_sharpe(x)
    return fn


def main(full: bool = False) -> None:
    results = Path(config.RESULTS_DIR)
    results.mkdir(parents=True, exist_ok=True)

    universe = config.UNIVERSE_SIZE_PAPER if full else config.UNIVERSE_SIZE_RUN
    returns, volumes, closes, factors = load_market_data(
        universe, config.DATA_START, config.DATA_END)
    ex, phi_mat, eps, _ = prepare_core(returns, volumes, closes, factors)
    feature_names = list(config.FEATURE_NAMES)
    K = len(feature_names)

    rng = np.random.default_rng(config.SEED)
    n_coal = (config.SHAPLEY_N_COALITIONS_PAPER if full
                else config.SHAPLEY_N_COALITIONS_RUN)
    B = config.BOOTSTRAP_B if full else config.BOOTSTRAP_B_RUN

    corr_rows: list[dict] = []
    sig_counts: list[dict] = []

    for name in ("XGB", "FFN"):
        p = results / f"exp1_outputs_{name}.pkl"
        if not p.exists():
            print(f"[exp3] skipping {name}: {p} missing")
            continue
        with open(p, "rb") as fh:
            out = pickle.load(fh)

        att_path = results / f"exp3_attribution_{name}.pkl"
        if att_path.exists():
            with open(att_path, "rb") as fh:
                daily = pickle.load(fh)
        else:
            print(f"[exp3] computing daily attributions for {name}")
            daily = build_daily_attributions(out, phi_mat, ex, feature_names,
                                               config.TRANSACTION_COST)
            with open(att_path, "wb") as fh:
                pickle.dump(daily, fh)

        phi_g = daily["phi_gross"]
        phi_n = daily["phi_net"]
        gross = out.gross
        net = out.net

        # Yearly loop: Shapley vs first-order per feature, per metric.
        years = sorted(set(phi_g.index.year))
        year_rows_g = []
        year_rows_n = []
        n_sig_g = 0; n_sig_n = 0
        total = 0
        for y in years:
            mask = phi_g.index.year == y
            if mask.sum() < 60:
                continue
            sub_pg = phi_g.loc[mask]; sub_pn = phi_n.loc[mask]
            sub_rg = gross.loc[mask]; sub_rn = net.loc[mask]
            # First-order Sharpe attribution per feature, from the aggregated
            # mean and vol decomposition applied to this year.
            mu_g = sub_pg.mean() * 252.0
            mu_n = sub_pn.mean() * 252.0
            sg_total = _annualised_sharpe(sub_rg.to_numpy())
            sn_total = _annualised_sharpe(sub_rn.to_numpy())
            vg = (sub_pg.sub(sub_pg.mean(axis=0)).mul(sub_rg - sub_rg.mean(), axis=0)
                  .mean(axis=0)) / sub_rg.std(ddof=0) * np.sqrt(252.0)
            vn = (sub_pn.sub(sub_pn.mean(axis=0)).mul(sub_rn - sub_rn.mean(), axis=0)
                  .mean(axis=0)) / sub_rn.std(ddof=0) * np.sqrt(252.0)
            sg_total_scale = float(sub_rg.std(ddof=0)) * np.sqrt(252.0)
            sn_total_scale = float(sub_rn.std(ddof=0)) * np.sqrt(252.0)
            if sg_total_scale > 0:
                fo_g = mu_g / sg_total_scale - sg_total * vg / sg_total_scale
            else:
                fo_g = pd.Series(0.0, index=feature_names)
            if sn_total_scale > 0:
                fo_n = mu_n / sn_total_scale - sn_total * vn / sn_total_scale
            else:
                fo_n = pd.Series(0.0, index=feature_names)

            # MC Shapley estimates on the explained-component Sharpe game.
            val_g = _value_fn_for_year(sub_pg, sub_pn, sub_rg, sub_rn, "gross")
            val_n = _value_fn_for_year(sub_pg, sub_pn, sub_rg, sub_rn, "net")
            psi_g = monte_carlo_shapley(val_g, K, n_coal, rng)
            psi_n = monte_carlo_shapley(val_n, K, n_coal, rng)

            year_rows_g.append(pd.DataFrame({"first_order": fo_g.values,
                                              "shapley": psi_g,
                                              "feature": feature_names,
                                              "year": y}))
            year_rows_n.append(pd.DataFrame({"first_order": fo_n.values,
                                              "shapley": psi_n,
                                              "feature": feature_names,
                                              "year": y}))

            # Bootstrap significance per feature-year on net Sharpe only to
            # bound compute; the paper reports both.
            for i, f in enumerate(feature_names):
                ci_g = circular_block_bootstrap_sharpe_attr(
                    sub_rg.to_numpy(), sub_pg[f].to_numpy(),
                    config.BOOTSTRAP_BLOCK, min(B, 999), rng)
                ci_n = circular_block_bootstrap_sharpe_attr(
                    sub_rn.to_numpy(), sub_pn[f].to_numpy(),
                    config.BOOTSTRAP_BLOCK, min(B, 999), rng)
                n_sig_g += int(ci_g.significant)
                n_sig_n += int(ci_n.significant)
                total += 1

        if year_rows_g:
            df_g = pd.concat(year_rows_g, ignore_index=True)
            df_n = pd.concat(year_rows_n, ignore_index=True)
            r_g, _ = pearsonr(df_g["first_order"], df_g["shapley"])
            r_n, _ = pearsonr(df_n["first_order"], df_n["shapley"])
            rmse_g = float(np.sqrt(((df_g["first_order"] - df_g["shapley"]) ** 2).mean()))
            rmse_n = float(np.sqrt(((df_n["first_order"] - df_n["shapley"]) ** 2).mean()))
            corr_rows.append({"model": name, "metric": "gross",
                                "pearson_r": r_g, "rmse": rmse_g,
                                "paper_r": 0.764 if name == "XGB" else 0.882})
            corr_rows.append({"model": name, "metric": "net",
                                "pearson_r": r_n, "rmse": rmse_n,
                                "paper_r": 0.765 if name == "XGB" else 0.916})
            df_g.to_csv(results / f"exp3_scatter_{name}_gross.csv", index=False)
            df_n.to_csv(results / f"exp3_scatter_{name}_net.csv", index=False)
        paper_sig = {"XGB": (28, 26), "FFN": (24, 33)}.get(name, (None, None))
        sig_counts.append({"model": name, "sig_gross": n_sig_g, "sig_net": n_sig_n,
                             "total_feature_years": total,
                             "paper_sig_gross": paper_sig[0],
                             "paper_sig_net": paper_sig[1]})

    pd.DataFrame(corr_rows).to_csv(results / "exp3_shapley_vs_firstorder.csv", index=False)
    pd.DataFrame(sig_counts).to_csv(results / "exp3_significant_counts.csv", index=False)
    print("[exp3] done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    main(full=args.full)
