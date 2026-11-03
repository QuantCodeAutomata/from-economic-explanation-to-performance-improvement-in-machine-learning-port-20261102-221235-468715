# From Economic Explanation to Performance Improvement in Machine-Learning Portfolios

Replication of the empirical pipeline described in
*From Economic Explanation to Performance Improvement in Machine-Learning
Portfolios.*

The paper constructs residual portfolios from a Fama–French 5-factor model,
derives 18 trading-signal features (Table 1), trains an XGBoost and a
feed-forward neural network allocation model (Section 2.1.3), builds a local
first-order feature-attribution framework for portfolio-level metrics
(Section 2.2), validates it against Shapley values and bootstrap inference
(Sections 2.2.2 and 2.2.5), and uses the attributions to drive post-training
feature-exposure interventions (Section 2.3).

## Repository layout

```
.
├── README.md
├── AGENTS.md
├── requirements.txt
├── reproduce.sh
├── .env.example
├── data/
│   └── data_loader.py          # QuantX Dataloader + Fama–French loader
├── src/
│   ├── config.py               # all parameters, grouped by experiment
│   ├── residuals.py            # Fama–French 5-factor residual projection
│   ├── hmm_regime.py           # 3-state Gaussian HMM for regime features
│   ├── features.py             # 18 Table 1 features
│   ├── models.py               # XGB and FFN allocation functions
│   ├── pipeline.py             # train/validate/trade with 252-day retrain
│   ├── portfolio.py            # L1 stock weights, turnover, net return
│   ├── attribution.py          # first-order propagation Eq. 25–32
│   └── shapley.py              # Monte-Carlo Shapley + block bootstrap
├── exp/
│   ├── exp1_baseline.py        # Table 2 baseline
│   ├── exp2_attribution.py     # Fig. 3–7
│   ├── exp3_shapley_inference.py  # Fig. 8–9 and Table 3
│   ├── exp4_similarity.py      # Tables 4–6
│   ├── exp5_interventions.py   # Tables 7–8
│   └── exp6_cost_sensitivity.py   # Figure 16
├── tests/
│   ├── conftest.py             # cached market-data slice fixture
│   ├── test_residuals.py       # unit tests for src.residuals
│   └── test_portfolio.py       # unit tests for src.portfolio
└── results/
    ├── RESULTS.md              # paper comparison with verdicts
    └── exp*_*.{csv,png,log}    # artefacts written by each experiment
```

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill DATALOADER_API_KEY (QuantX)
```

`DATALOADER_API_KEY` is required for the QuantX Dataloader; a free key is
available at <https://quantx.limex.pro/>. Downloaded bars are cached under
`data/cache/` and are never committed. Fama–French 5-factor daily data are
pulled from the Kenneth French Data Library through `pandas_datareader`.

## Running the experiments

Every command is run from the repository root.

```
# Experiment 1 — baseline Table 2 portfolios (XGB and FFN).
python -m exp.exp1_baseline

# Experiment 2 — propagated SHAP attribution (requires exp1 outputs).
python -m exp.exp2_attribution

# Experiment 3 — Shapley validation and block bootstrap.
python -m exp.exp3_shapley_inference

# Experiment 4 — within/cross-model similarity and persistence.
python -m exp.exp4_similarity

# Experiment 5 — attribution-guided feature-exposure interventions.
python -m exp.exp5_interventions

# Experiment 6 — robustness over transaction-cost grid.
python -m exp.exp6_cost_sensitivity
```

Every script writes its artefacts into `results/`. Running an experiment
overwrites the files it produces; the numbers and verdicts in
`results/RESULTS.md` must then be checked against the new files and updated
by hand.

`reproduce.sh` runs the test suite and every experiment in order.

The paper configuration (universe of 346 names, B = 4,999 bootstrap
replications) is selectable by passing `--full` to any experiment script,
for example:

```
python -m exp.exp1_baseline --full
```

The default run uses a reduced liquid US large-cap universe (~140 names
continuously available over 2000-10 to 2024-12) and B = 999 bootstrap
replications so that the full six-experiment pipeline fits inside the
replication's compute budget. Estimated full-paper download alone exceeds
two hours on a cold cache; the reduction and its implications for the
Table 2 numbers are documented in `results/RESULTS.md`.

## Tests

```
python -m pytest -m "not market_data"   # unit tests only, no API calls
python -m pytest                          # include the cached market-data slice tests
```

The `market_data` marker tags tests that load a small cached slice through
`data/data_loader.py`; they skip automatically when `DATALOADER_API_KEY` is
not set.
