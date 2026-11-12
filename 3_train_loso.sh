#!/bin/bash
# TRAINS: all four winners with leave-one-study-out cross-validation: 8 folds, each holding out one
# dataset. Targets and EEG power are normalised within each dataset; features are computed once.

# Load the shared settings and helper functions (train, run, log, ...)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

run "$PY" "$TRAIN/shared/make_analysis_tables.py" --loso   # CV_LOSO.csv, normFactors_LOSO_*.csv

train "$WINNER_PAIN_DEFAULT" painVars LOSO    # CPN, linear
train "$WINNER_PAIN_BEST"    painVars LOSO    # FBR, Schaefer-100, 2 s, ridge
train "$WINNER_AGE_DEFAULT"  age      LOSO    # FBS, sensor, 2 s, ridge
train "$WINNER_AGE_BEST"     age      LOSO    # HCF, sensor, 2 s, linear
