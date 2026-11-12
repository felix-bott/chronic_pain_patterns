#!/bin/bash
# TRAINS: the two pain winners on the chronic back pain (CBP) and neuropathic pain (NP) subgroups.
# Same folds and normalisation as the main analysis, restricted to the subgroup's individuals.

# Load the shared settings and helper functions (train, run, log, ...)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

run "$PY" "$TRAIN/shared/make_analysis_tables.py" --subgroups   # CV_CBP.csv, CV_NP.csv, n_per_*.csv

for subgroup in CBP NP; do
  train "$WINNER_PAIN_DEFAULT" painVars "$subgroup"    # CPN, linear
  train "$WINNER_PAIN_BEST"    painVars "$subgroup"    # FBR, Schaefer-100, 2 s, ridge
done
