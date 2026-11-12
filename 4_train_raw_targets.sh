#!/bin/bash
# TRAINS: all four winners on raw targets (age in years, pain as rated) on the main folds.
# Inputs stay harmonized, and harmonized age stays the confounder for pain.

# Load the shared settings and helper functions (train, run, log, ...)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

train "$WINNER_PAIN_DEFAULT" painVarsRAW RAW    # CPN, linear
train "$WINNER_PAIN_BEST"    painVarsRAW RAW    # FBR, Schaefer-100, 2 s, ridge
train "$WINNER_AGE_DEFAULT"  ageRAW      RAW    # FBS, sensor, 2 s, ridge
train "$WINNER_AGE_BEST"     ageRAW      RAW    # HCF, sensor, 2 s, linear
