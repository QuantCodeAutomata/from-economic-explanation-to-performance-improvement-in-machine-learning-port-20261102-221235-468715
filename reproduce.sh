#!/usr/bin/env bash
# Run every experiment in order and the test suite.
# Each script writes its artefacts into results/.
set -u

python -m pytest -m "not market_data" -q || true

python -m exp.exp1_baseline
python -m exp.exp2_attribution
python -m exp.exp3_shapley_inference
python -m exp.exp4_similarity
python -m exp.exp5_interventions
python -m exp.exp6_cost_sensitivity
