#!/bin/bash
# Recreate the combined tables and tile figures from the per-model summaries in CPP_results_light
# (no training), compare the tables with the stored ones, and recompute the overall and per-fold
# metrics. Needs only this repository and the conda environment.
#
#   ./8_recreate_from_light.sh [output folder]     default: CPP_results_recreated in this repository
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HERE/CPP_results_recreated}"
mkdir -p "$OUT"
export CPP_RESULTS="$(cd "$OUT" && pwd)"   # all scripts below write here

# Load the shared settings and helper functions (analysis_dir, ...)
source "$HERE/common.sh"
LIGHT="$CPP/CPP_results_light"

# Analyses to recreate: analysis:target
ANALYSES=(
    "main:age"
    "main:pain"
    "CBP:pain"
    "NP:pain"
    "LOSO:age"
    "LOSO:pain"
    "RAW:ageRAW"
    "RAW:painRAW"
)

# Find the per-model summaries and copy each one to $CPP_RESULTS. --parents keeps the file's folder
# structure in the copy. The parentheses keep the change of folder from affecting the rest of this script.
(
    cd "$LIGHT"
    find analysis_* -type f -path 'analysis_*/*/summaries/*' -exec cp --parents -t "$CPP_RESULTS" {} +
)

for analysis_target in "${ANALYSES[@]}"; do
    IFS=":" read -r analysis target <<< "$analysis_target"
    "$PY" "$ANALYSIS/combine/combine_summaries.py" --target "$target" --analysis "$analysis"
done

mkdir -p "$(analysis_dir main)/figures"
for target in age pain; do
  "$PY" "$ANALYSIS/plots/plot_tiles.py"             --target "$target"
  "$PY" "$ANALYSIS/plots/plot_tiles_all_configs.py" --target "$target"
done

"$PY" "$ANALYSIS/check_light_recreation.py" --reference "$LIGHT" --recreated "$CPP_RESULTS"
"$PY" "$ANALYSIS/check_per_fold_metrics.py" --light_dir "$LIGHT"
