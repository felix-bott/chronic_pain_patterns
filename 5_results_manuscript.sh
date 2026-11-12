#!/bin/bash
# Results of the main analysis, from the stored predictions (no training): per-model summaries,
# combined tables, tile figures and KDE figures. The KDE figures need the EEG signals.

# Load the shared settings and helper functions (analysis_dir, ...)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
MAIN="$(analysis_dir main)"
N_JOBS="${N_JOBS:-8}"
mkdir -p "$MAIN/tables" "$MAIN/figures"

"$PY" "$ANALYSIS/write_dataset_aliases.py"

for target in age pain; do
  for experiment in 2025 2025GENERAL; do   # OTiS_EEG, OTiS_GEN
    "$PY" "$ANALYSIS/summaries/summarize_predictions_OTiS.py"             --experiment_ID $experiment --target $target
    "$PY" "$ANALYSIS/summaries/summarize_predictions_final_layer_OTiS.py" --experiment_ID $experiment --target $target
  done
  "$PY" "$ANALYSIS/summaries/summarize_predictions_LaBraM.py"             --target $target
  "$PY" "$ANALYSIS/summaries/summarize_predictions_final_layer_LaBraM.py" --target $target
  for m in CND CNS; do "$PY" "$ANALYSIS/summaries/summarize_convnets.py" --model_name $m --target $target; done
  for m in HCF FBR FBS; do "$PY" "$ANALYSIS/summaries/summarize_feature_models.py" --model_type $m --target $target; done
  "$PY" "$ANALYSIS/summaries/summarize_predictions_CPN.py" --target $target

  "$PY" "$ANALYSIS/combine/combine_summaries.py"     --target $target   # with Spearman rho and Bayes factors
  "$PY" "$ANALYSIS/plots/plot_tiles.py"              --target $target
  "$PY" "$ANALYSIS/plots/plot_tiles_all_configs.py"  --target $target
done

for variant in sensor_20 yeo_20 schaefer_20; do
  "$PY" "$ANALYSIS/plots/plot_kde_distributions.py" --signal_variant $variant --n_jobs "$N_JOBS"
done
"$PY" "$ANALYSIS/plots/plot_age_kde_with_legend.py"
