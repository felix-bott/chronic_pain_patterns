#!/bin/bash
# Copy the tables, per-model summaries, figures and CV tables from CPP_results to CPP_results_light,
# unchanged (no training). Needs the full results. Tables include subfolders such as
# analysis_main/tables/age_associations/ from 9_misc_analyses.py, so run script 9 before this one.

# Load the shared settings and helper functions
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
LIGHT="$CPP/CPP_results_light"

mkdir -p "$LIGHT"
rm -rf "$LIGHT"/analysis_*
cd "$RES"

# Find the tables (with their subfolders), figures, summaries and CV tables (but not norm tables), and copy each one to
# $LIGHT. --parents keeps the file's folder structure in the copy.
find analysis_* -type f \( -path 'analysis_*/tables/*' -o -path 'analysis_*/figures/*' -o -path 'analysis_*/*/summaries/*' \
  -o \( -path 'analysis_*/inputs/CV*' ! -name '*normFactors*' \) \) \
  -exec cp --parents -t "$LIGHT" {} +
