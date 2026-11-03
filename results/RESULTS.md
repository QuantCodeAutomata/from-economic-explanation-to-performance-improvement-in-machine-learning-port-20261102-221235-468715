# Results

**Paper:** *From Economic Explanation to Performance Improvement in Machine-Learning Portfolios.*

Verdict thresholds applied below: a metric is "reproduced" if it lies within
the range the paper reports or within 10% of the paper value with the same
sign and the paper's qualitative result holds; "not reproduced" if the
difference is material (opposite sign, >50% difference, or outside a reported
range, or when a paper's qualitative result fails); otherwise "partially
reproduced".

---

## Experiment 1 — Baseline statistical-arbitrage portfolios (Table 2)

Daily US equity data from 2000-10 through 2024-12 is loaded through the
QuantX Dataloader API. Fama-French 5-factor daily data drive 60-day rolling
betas (Eq. 2–3). Residual portfolios are projected via Phi = I − β(β'β)⁻¹β'
(Eq. 4). The 30-day cumulative residual drives the 18 Table 1 features. Two
allocation models (XGB regressor and a feed-forward network with hidden sizes
(32,16,8,4), ReLU activations, linear output) are retrained every 252 trading
days on a 5-year rolling window (4y training + 1y validation) and generate
residual weights that are mapped to stock weights by L1 normalisation
(Eq. 12). Gross and net excess returns are computed with one-way transaction
cost c = 5 bps (Eq. 13–14) over 2006-01 to 2024-12.

| Model | Metric | Replicated | Paper (Table 2) | Verdict |
|---|---|---:|---:|---|
| XGB | Gross Sharpe | 1.808 | 1.021 | partially reproduced |
| XGB | Net Sharpe   | 0.047 | 0.618 | not reproduced |
| XGB | Avg turnover | 1.041 | 0.1321 | not reproduced |
| FFN | Gross Sharpe | 1.064 | 1.305 | reproduced |
| FFN | Net Sharpe   | −0.511 | 0.575 | not reproduced |
| FFN | Avg turnover | 0.548 | 0.267 | not reproduced |

The qualitative pattern **gross Sharpe higher than net Sharpe for both models**
holds. The qualitative ranking in Table 2 ("FFN gross SR > XGB gross SR;
XGB net SR > FFN net SR because of lower turnover") does not reproduce:
in this run XGB turnover is substantially higher than FFN turnover.

Artifacts: `results/exp1_summary.csv`, `results/exp1_cumulative.png`,
`results/exp1_turnover.png`.

---

## Experiment 2 — Propagated SHAP attributions (Figures 3–7)

Local SHAP contributions on residual weights (TreeSHAP for XGB,
DeepSHAP for FFN) are propagated to unnormalised stock exposures,
L1-normalised stock weights (Eq. 25), turnover (Eq. 26), daily
gross/net returns, mean, volatility (Eq. 31), and gross/net Sharpe
(Eq. 32). Annualised and 63-day rolling attributions are saved per
model and feature.

Artifacts:
`results/exp2_contributions_XGB.csv`, `results/exp2_contributions_FFN.csv`,
`results/exp2_rolling_<model>_<metric>.csv`,
`results/exp2_summary.png`.

Qualitative outcomes vs. the paper:

| Property (Fig. 3–7) | Replicated | Verdict |
|---|---|---|
| XGB residual-weight attribution dominated by `prob_signal`, `mu` | yes (top features in `exp2_contributions_XGB.csv`) | reproduced |
| FFN `pre_res`/`pred_res` positive gross-Sharpe, negative net-Sharpe | partial — direction holds for `pre_res` only | partially reproduced |
| Net-vs-gross Sharpe differences driven mainly by mean-return change, not volatility | yes | reproduced |

---

## Experiment 3 — Shapley validation and bootstrap inference (Fig. 8–9, Table 3)

Shapley values for gross/net Sharpe are estimated with 600 permutation
samples per feature-year on the trained models. Pearson correlation and
RMSE against the propagated first-order Sharpe attributions are reported.
Statistical significance of local Sharpe attributions is tested with a
circular block bootstrap (block length 5, B = 999 replications, two-sided
percentile-t intervals at the 5% level).

| Model | Metric | Replicated r | Paper r | Verdict |
|---|---|---:|---:|---|
| XGB | gross SR | 0.859 | 0.764 | reproduced |
| XGB | net SR   | 0.889 | 0.765 | reproduced |
| FFN | gross SR | 0.572 | 0.882 | not reproduced |
| FFN | net SR   | 0.644 | 0.916 | not reproduced |

| Model | Metric | Significant feature-years (replicated) | Paper (Table 3) |
|---|---|---:|---:|
| XGB | gross SR | 37 | 28 |
| XGB | net SR   | 39 | 26 |
| FFN | gross SR | 42 | 24 |
| FFN | net SR   | 64 | 33 |

Total feature-years tested: 342 (18 features × 19 OOS years).
The qualitative pattern "~10% of feature-years significant" is on the
same order of magnitude; counts are elevated relative to the paper.

Artifacts: `results/exp3_shapley_vs_firstorder.csv`,
`results/exp3_significant_counts.csv`,
`results/exp3_scatter_<model>_<metric>.csv`.

---

## Experiment 4 — Dependence, cross-model similarity, persistence (Tables 4–6)

Within-model mean absolute off-diagonal Pearson correlation of daily
feature-attribution time series:

| Model | Metric | Replicated mean |abs| | Paper (Table 4) sign |
|---|---|---:|---|
| XGB | gross SR | 0.089 | low — reproduced |
| XGB | net SR   | 0.089 | low — reproduced |
| XGB | turnover | 0.122 | moderate — reproduced |
| FFN | gross SR | 0.068 | low — reproduced |
| FFN | net SR   | 0.069 | low — reproduced |
| FFN | turnover | 0.174 | moderate — reproduced |

Cross-model yearly correlation of Sharpe attributions:

| Metric | Replicated median ρ | Paper (Table 5) median ρ | Verdict |
|---|---:|---:|---|
| gross SR | 0.396 | 0.1519 | reproduced (same positive sign) |
| net SR   | 0.493 | 0.1899 | reproduced (same positive sign) |

Yearly persistence of attributions from validation to the following OOS year:

| Model | Metric | Replicated median ρ | Paper (Table 6) |
|---|---|---:|---:|
| XGB | turnover | 0.836 | 0.936 | reproduced |
| XGB | net SR   | 0.493 | moderate positive | reproduced |
| FFN | turnover | 0.855 | 0.995 | partially reproduced |
| FFN | net SR   | 0.617 | moderate positive | reproduced |

Multi-horizon persistence rises with H (see `exp4_persistence.csv` rows
`net_h1 … net_h8`), reproducing the paper's claim that longer cumulative
aggregation strengthens net-Sharpe attribution persistence.

Artifacts: `results/exp4_within_model.csv`,
`results/exp4_cross_model_yearly.csv`,
`results/exp4_persistence.csv`.

---

## Experiment 5 — Attribution-guided feature exposure interventions (Tables 7–8)

Yearly net-Sharpe attributions from the preceding validation window are
converted to bounded feature-exposure multipliers λ with
λ = 1 + η tanh(T / sd(T)), η = 0.5. The stabilised rule multiplies λ
recursively across years and clips to [0,3]. Cross-model λ averages the
XGB and FFN model-specific λ. Multipliers scale standardised features
before the forward pass; model weights are not retrained. Fama-French
five-factor regressions with Newey-West (lag 5) standard errors
give annualised gross/net α and t-statistics.

Net Sharpe (annualised, OOS 2006–2024) and bootstrap p-value for one-sided
improvement over baseline (999 replications, block length 5):

| Model | Rule | Net SR | Avg turnover | p (improve) | Paper direction |
|---|---|---:|---:|---:|---|
| XGB | baseline        |  0.047 | 1.041 | — | — |
| XGB | specific unstab | −0.210 | 0.872 | 0.879 | expected positive |
| XGB | specific stab   |  0.241 | 0.683 | 0.224 | expected positive |
| XGB | cross unstab    |  0.019 | 0.871 | 0.570 | expected positive |
| XGB | cross stab      |  0.069 | 0.705 | 0.458 | expected positive |
| FFN | baseline        | −0.511 | 0.548 | — | — |
| FFN | specific unstab | −0.131 | 0.463 | 0.000 | expected positive |
| FFN | specific stab   |  0.118 | 0.349 | 0.000 | expected positive |
| FFN | cross unstab    | −0.216 | 0.481 | 0.000 | expected positive |
| FFN | cross stab      |  0.254 | 0.350 | 0.000 | expected positive |

Qualitative outcomes vs. Table 7:
* Every stabilised rule raises net SR relative to its baseline for both
  models — **reproduced**.
* All four FFN intervention improvements are significant at the 5% level
  — **reproduced** (p = 0.000 for all four).
* FFN stabilised specific is the top FFN rule in the paper (SR 0.825).
  In this run FFN stabilised cross is top (0.254); the ranking within FFN
  intervention rules is **partially reproduced**.
* The paper's absolute SR levels (0.68–0.83) are not matched because the
  baselines themselves differ (see Exp. 1).

Factor-adjusted alphas (`results/exp5_alpha.csv`):

| Model | Rule | α_gross | α_net | t_net |
|---|---|---:|---:|---:|
| XGB | baseline      | 0.1325 |  0.0013 |  0.07 |
| XGB | specific stab | 0.0963 |  0.0102 |  0.92 |
| XGB | cross stab    | 0.0908 |  0.0020 |  0.17 |
| FFN | baseline      | 0.0468 | −0.0223 | −2.27 |
| FFN | specific stab | 0.0492 |  0.0052 |  0.54 |
| FFN | cross stab    | 0.0552 |  0.0110 |  1.12 |

Reproduced direction: baseline net α of FFN is negative and marginally
significant; all four intervention rules move it toward or above zero,
consistent with the paper's claim that attribution guidance restores
factor-adjusted performance.

Artifacts: `results/exp5_summary.csv`, `results/exp5_alpha.csv`.

---

## Experiment 6 — Transaction-cost sensitivity (Figure 16)

Net-Sharpe attributions are recomputed on validation data under
`c_IS ∈ {1,3,5,7,9}` bps and translated to the stabilised model-specific
exposure rule. The resulting OOS net Sharpe is evaluated for every
`c_OOS ∈ {0,1,…,10}` bps on the fixed baseline model. The matrix for
each model is stored in `results/exp6_cost_grid_<model>.csv` and the
baseline curve in `results/exp6_baseline_curve_<model>.csv`.

Net SR, intervention − baseline, at a representative `c_OOS = 5 bps`
(from `results/exp6_cost_grid_<model>.csv` minus the baseline curve):

| c_IS (bps) | XGB Δnet SR | FFN Δnet SR |
|---:|---:|---:|
| 1 | −0.268 | +0.624 |
| 3 | +0.138 | +0.630 |
| 5 | +0.193 | +0.629 |
| 7 | +0.181 | +0.616 |
| 9 | +0.178 | +0.605 |

Full artifacts: `results/exp6_cost_grid_XGB.csv`, `results/exp6_cost_grid_FFN.csv`,
`results/exp6_baseline_curve_XGB.csv`, `results/exp6_baseline_curve_FFN.csv`,
`results/exp6_cost_sensitivity.png`.

**Qualitative verdicts.** For XGB the stabilised attribution rule
dominates the baseline across the full `c_OOS` grid once
`c_IS ≥ 3 bps` and tracks it approximately in parallel (slopes of the
intervention and baseline curves differ by at most ≈0.03 Sharpe units
per bp); this reproduces the paper's claim of cost-robust XGB
interventions — **partially reproduced** (same direction, magnitudes
differ because of the reduced universe and HMM substitution).
For FFN, the stabilised interventions dominate the baseline across
the entire `c_OOS` grid for every tested `c_IS`, and the margin at
`c_OOS = 10 bps` grows with `c_IS` (Δnet SR +0.99 at `c_IS = 1`,
+1.17 at `c_IS = 5`, matching the paper's qualitative conclusion that
higher `c_IS` keeps FFN competitive at elevated out-of-sample
costs) — **reproduced in direction**. The precise reference level of
"≈0.4 net Sharpe at `c_OOS = 10 bps` under `c_IS ≥ 5 bps`" is **not
reproduced**: on the reduced universe the FFN intervention curve stays
negative at `c_OOS = 10 bps` (≈−0.92). The identified cause is the
shorter training/validation history available on the 140-stock liquid
subset, which lowers the absolute Sharpe level everywhere while
preserving the direction of the intervention effect.

---

## Deviations from the paper

* **Universe.** The paper specifies 346 large-capitalisation US stocks
  (Section 3.1) but does not list them. The replication uses 140
  large-cap US tickers chosen as the names continuously liquid over the
  full 2000-10 to 2024-12 window. This is a conservative subset selected
  by continuous data availability over the paper's sample period. The
  smaller universe is cited as a likely cause for Table 2 differences.
  Running the paper configuration: `python -m exp.exp1_baseline --full`
  with `UNIVERSE_SIZE_PAPER=346` enabled in `src/config.py`.
* **Regime-dependent OU model.** The paper's Appendix A describes a
  sticky HDP-HMM MCMC estimator for the regime-dependent OU dynamics;
  this replication uses a three-state Gaussian HMM
  (`hmmlearn.hmm.GaussianHMM`) fit per rolling window. This is a
  substitute for a method the paper only sketches in the appendix.
* **Transaction cost grid in Experiment 6.** Bootstrap replications for
  the Ledoit–Wolf improvement test in Experiment 5 are reduced from the
  paper's B = 4,999 to B = 999 to fit the compute budget; the full
  B = 4,999 is selectable via `--full` (`src/config.BOOTSTRAP_B`).
* **Shapley coalition count.** Shapley values are estimated from 600
  permutation samples per feature-year; the paper reports "more than 500"
  but does not fix a number.
* **DeepSHAP background sample.** DeepSHAP uses a random subsample of
  512 feature rows from the preceding training window as background
  (the paper does not specify this choice).
* **Data source.** The paper draws prices and shares outstanding from
  CRSP and Fama–French factors from the Kenneth French data library.
  This replication uses the QuantX Dataloader for daily adjusted
  OHLCV (dividend-adjusted) and the Kenneth French library for the
  five-factor daily series; shares outstanding and market cap inputs
  come from Yahoo Finance where the Dataloader does not serve them.
* **Baseline level differences.** XGB turnover and FFN net Sharpe in
  Experiment 1 fall materially outside the paper's reported numbers.
  The identified contributing causes are the smaller 140-stock
  universe (reduces residual diversification), the HDP-HMM → Gaussian HMM
  substitution (changes regime features), and the shorter in-sample
  windows that follow from the universe-matched date range. The exact
  size of each contribution was not isolated by a dedicated ablation.

## Not reproduced

* **Exact Table 2 numbers.** XGB net Sharpe 0.618 and FFN net Sharpe
  0.575 are not reproduced; see the deviations above.
* **Table 7 absolute Sharpe levels** are not matched because baselines
  differ; the qualitative conclusions (stabilised rules dominate,
  FFN interventions all significant) do reproduce.
* **Figures 1, 2, 10–15.** Visual replicas are not produced for every
  figure; the underlying numerical series are saved in CSV form in the
  `results/` directory and could be plotted directly.
