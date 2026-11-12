#!/bin/bash
# TRAINS: all 72 configurations of the benchmark, each for age and pain, on the main folds.
#
# Deep models run through their family's launcher (training/*/submit_*.sh), which holds that
# family's settings. Classical models run through the train helper in common.sh.
#
#   DRY_RUN=1 ./1_train_all_models.sh   print every job instead of running it
#
# Predictions and features go to CPP_results; checkpoints of the deep models to CPP_CHECKPOINTS.

# Load the shared settings and helper functions (train, run, log, config_paths, ...)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

export DRY_RUN="${DRY_RUN:-0}"

OTIS="OTiS/execution_scripts"


# ---- settings -----------------------------------------------------------------------------------

# OTiS stage S1, one sweep per signal variant:
# run_name|folder_name|hyperparams (learning_rate:batch_size:accum_iter:mask_ratio)
S1_VARIANTS=(
    "epoched_autoPreproc_10.0_0.0_250_Normalized|sensorSignalsNormalized|3e-4:8:2:0.75"
    "OrthoAggSchaefer100_freq-all_epochLength-10_overlap-0|sourceSignalsNormalized|1e-3:16:1:0.75 1e-4:16:1:0.75 3e-4:16:1:0.75"
    "Schaefer100_freq-all_epochLength-10_overlap-0|sourceSignalsNormalized|1e-3:8:2:0.75 3.01e-4:8:2:0.75 3e-4:8:2:0.75"
)

# OTiS stage S2: spatial_epoch|use_otis_general_preproc (n: OTiS_EEG, y: OTiS_GEN)
S2_VARIANTS=(
    "Sensor_2|n"
    "Sensor_20|n"
    "Schaefer100_2|n"
    "Schaefer100_20|n"
    "Yeo7_2|n"
    "Yeo7_20|n"

    "Sensor_2|y"
    "Schaefer100_2|y"
    "Yeo7_2|y"
)

# Final-layer models of OTiS and LaBraM: algorithm:aggregation
FINAL_LAYER_MODELS=(
    "linear:mean"
    "linear:none"
    "xgboost:mean"
)

# Classical models (16 configurations): model:spatial_epoch:algorithm
CLASSICAL=(
    "HCF:Yeo7_2:rf"
    "HCF:Sensor_2:rf"
    "HCF:Schaefer100_2:rf"
    "HCF:Yeo7_2:linear"
    "HCF:Sensor_2:linear"
    "HCF:Schaefer100_2:linear"
    "HCF:Yeo7_2:xgboost"
    "HCF:Sensor_2:xgboost"
    # HCF Schaefer-100 xgboost: out of memory
    "FBR:Sensor_2:ridge"
    "FBR:Schaefer100_2:ridge"
    "FBR:Sensor_2:xgboost"
    "FBR:Schaefer100_2:xgboost"
    "FBS:Sensor_2:ridge"
    "FBS:Sensor_2:xgboost"
    "CPN:Yeo7_2:linear"
    "CPN:Yeo7_2:xgboost"
)


# ---- functions ----------------------------------------------------------------------------------

# Launchers submit cluster jobs and return; later stages need the earlier stages' outputs.
wait_for_jobs() {
    if [ "$DRY_RUN" = 1 ]; then
        return
    fi

    while [ "$(squeue -u "$USER" --noheader | wc -l)" -gt 0 ]
    do
        sleep 600
    done
}

# Run a launcher from its own folder, as it calls ./job_*.sh
launch() {
    local folder=$1
    local launcher=$2

    log "launcher: $folder/$launcher"

    # The parentheses keep the change of folder from affecting the rest of this script
    (
        cd "$TRAIN/$folder"
        "./$launcher"
    )
}


# ---- OTiS, stage S1: pretraining on LEMON and TDBRAIN, one sweep per signal variant -------------
for variant in "${S1_VARIANTS[@]}"; do
    IFS="|" read -r run_name folder_name hyperparams <<< "$variant"

    # Settings written before a command apply to that command only; the launcher reads them
    RUN_NAME="$run_name" FOLDER_NAME="$folder_name" HYPERPARAMS="$hyperparams" launch "$OTIS" submit_pretrain_s1.sh
done
wait_for_jobs

# ---- OTiS, stage S2: pretraining on each fold's training data (also exports the embeddings) -----
for variant in "${S2_VARIANTS[@]}"; do
    IFS="|" read -r spatial use_otis_general_preproc <<< "$variant"

    # Sets RUN_NAME and SIGNALS for this variant
    config_paths "$spatial"

    # Settings written before a command apply to that command only; the launcher reads them
    RUN_NAME="$RUN_NAME" FOLDER_NAME="$SIGNALS" OTIS_GENERAL="$use_otis_general_preproc" launch "$OTIS" submit_pretrain_s2.sh
done
wait_for_jobs

# ---- OTiS, full finetuning (all variants, both targets) and final-layer models -------------------
launch "$OTIS" submit_finetune.sh

for experiment in 2025 2025GENERAL; do               # OTiS_EEG, OTiS_GEN
    if [ "$experiment" = "2025GENERAL" ]; then
        sigLen="2"
    else
        sigLen="2 20"
    fi

    for final_layer_model in "${FINAL_LAYER_MODELS[@]}"; do
        IFS=":" read -r algorithm aggregation <<< "$final_layer_model"

        # $sigLen has no quotes on purpose: each value becomes its own argument
        run "$PY" -u "$TRAIN/$OTIS/train_embedding_models.py" \
            --experiment_ID "$experiment" \
            --algorithm "$algorithm" \
            --aggregation "$aggregation" \
            --sigLen $sigLen
    done
done

# ---- LaBraM: full finetuning (also exports the embeddings), then final-layer models -------------
launch LaBraM submit_labram.sh
wait_for_jobs

for final_layer_model in "${FINAL_LAYER_MODELS[@]}"; do
    IFS=":" read -r algorithm aggregation <<< "$final_layer_model"

    run "$PY" -u "$TRAIN/LaBraM/train_labram_embedding_models.py" \
        --algorithm "$algorithm" \
        --aggregation "$aggregation" \
        --sigLen 2 20
done

# ---- ConvNetDeep, ConvNetShallow ---------------------------------------------------------------
launch MEEGBenchmark submit_convnets.sh

# ---- classical models (16 configurations) -------------------------------------------------------
for config in "${CLASSICAL[@]}"; do
    for target in age painVars; do
        train "$config" "$target" main
    done
done
wait_for_jobs
