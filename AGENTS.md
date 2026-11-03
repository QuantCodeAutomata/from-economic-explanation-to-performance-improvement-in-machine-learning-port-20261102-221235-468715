# AGENTS.md

Short guide for contributors and coding agents working on this repository.

## What this repository implements

A replication of
*From Economic Explanation to Performance Improvement in Machine-Learning
Portfolios.*
The six experiments reproduce Table 2 (baseline statistical-arbitrage
performance), Figures 3–7 (propagated SHAP attributions), Figures 8–9 and
Table 3 (Shapley validation and bootstrap inference), Tables 4–6 (dependence,
cross-model similarity, persistence), Tables 7–8 (attribution-guided
feature-exposure interventions), and Figure 16 (transaction-cost sensitivity).

## Layout

```
src/            Library modules (flat), reusable components only
exp/            Experiment entry points, one per experiment
tests/          Pytest suite
data/           data_loader.py — the one place that fetches market data
results/        Every artefact written by exp/*.py and RESULTS.md
```

* `src/config.py` holds every parameter as module-level `UPPER_CASE`
  variables. Paper values and executed values live side by side
  (`UNIVERSE_SIZE_PAPER`, `UNIVERSE_SIZE_RUN`, etc.). The `--full` flag
  on every experiment script switches to paper values.
* No parameter dictionaries, no `argparse` beyond `--full`, no classes
  for configuration.
* The data path is single-sourced: everything flows through
  `data/data_loader.py`. No loaders in `src/` or `exp/`.

## Commands

```
python -m pytest -m "not market_data"       # fast unit tests, no API calls
python -m pytest                             # full suite, loads a cached slice
python -m exp.exp1_baseline                  # Table 2
python -m exp.exp2_attribution               # Fig 3–7
python -m exp.exp3_shapley_inference         # Fig 8–9, Table 3
python -m exp.exp4_similarity                # Tables 4–6
python -m exp.exp5_interventions             # Tables 7–8
python -m exp.exp6_cost_sensitivity          # Figure 16
```

Add `--full` to any experiment script to use the paper's configuration
(346-stock universe, B = 4,999 bootstrap replications). Experiments
overwrite the files in `results/`; after a rerun the numbers and
verdicts in `results/RESULTS.md` must be checked against the new
artefacts and updated by hand.

A single `reproduce.sh` runs the tests and every experiment in order.

## Conventions

* API keys live only in an uncommitted `.env`; `.env.example` only shows
  the empty placeholders. Code reads them through `load_dotenv()`.
* Daily OHLCV come from the QuantX Dataloader (adjusted); Fama–French
  5-factor daily data come from the Kenneth French Library. Downloaded
  bars are cached under `data/cache/` and are never committed.
* Experiment scripts call the shared `src/pipeline.py` entry points; they
  must not build their own data loader, their own train loop, or their
  own performance calculation.
* Tests under `tests/` cover `src/` on small fixtures or a cached slice
  of the real sample. Tests that touch the API are marked
  `@pytest.mark.market_data` and skip automatically when
  `DATALOADER_API_KEY` is missing.

## Known limitations

The replication deviates from the paper on three structural points,
listed in `results/RESULTS.md`:

1. The universe is a liquid 140-stock subset of the paper's 346-stock
   list (continuously available over 2000-10 to 2024-12).
2. The regime-dependent OU model is estimated with a three-state Gaussian
   HMM (`hmmlearn`) rather than the sticky HDP-HMM MCMC estimator of
   Appendix A.
3. The Ledoit–Wolf Sharpe-ratio improvement test uses B = 999 bootstrap
   replications instead of the paper's B = 4,999 to fit the compute
   budget.

All three settings are tracked by paper-vs-run variables in
`src/config.py`.
