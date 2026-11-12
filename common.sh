#!/bin/bash
# Shared by the numbered scripts: environment, paths, the four winners, and helper functions.
# Not meant to be run on its own; the numbered scripts source it.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# ---- environment -------------------------------------------------------------------------------
source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"
CPP="$CPP_CODE"
ROOT="$TURBO_ROOT"
PY="${CPP_PYTHON:-$HOME/.conda/envs/otis_benchmarks_new/bin/python}"
ENV_DIR="$(cd "$(dirname "$PY")/.." && pwd)"
export PATH="$ENV_DIR/bin:$PATH" R_HOME="$ENV_DIR/lib/R"   # rpy2 needs the env's R, which has BayesFactor
# The trainers start their own parallel workers; cap the threads each of them may use.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

# ---- paths -------------------------------------------------------------------------------------
FT="$ROOT/data/fineTuning"                     # signals and tables (read-only)
RES="$CPP_RESULTS"                             # everything this pipeline writes
TRAIN="$CPP/training"
ANALYSIS="$CPP/analysis"
CV_MAIN="$RES/analysis_main/inputs/CV70-20-10Table_MegaStrat.csv"
NORM_DIR="$RES/analysis_main/inputs"
MAIN_FOLDS="Fold1 Fold2 Fold3 Fold4 Fold5 Fold6 Fold7 Fold8 Fold9 Fold10"
LOSO_FOLDS="LOSO_Set_Munich1 LOSO_Set_Munich2 LOSO_Set_Otago1 LOSO_Set_Otago2 LOSO_Set_Brisbane LOSO_Set_Seattle LOSO_Set_Haifa LOSO_Set_Boulder"

# ---- the four winners (MODEL:SPATIAL_EPOCH:ALGORITHM) -------------------------------------------
WINNER_PAIN_DEFAULT="CPN:Yeo7_2:linear"            # best preset configuration, pain
WINNER_PAIN_BEST="FBR:Schaefer100_2:ridge"         # best configuration overall, pain
WINNER_AGE_DEFAULT="FBS:Sensor_2:ridge"            # best preset configuration, age
WINNER_AGE_BEST="HCF:Sensor_2:linear"              # best configuration overall, age

# ---- configuration -> signals and norm table (the one place that maps them) ---------------------
# Sets RUN_NAME, SIGNALS (folder under data/fineTuning), NORM_TABLE (stored), LOSO_NORM (name).
config_paths() {
    local spatial=$1

    if [ "$spatial" = "Sensor_2" ]; then
        RUN_NAME="epoched_autoPreproc_CUT2.0_CUT0.5_250"
        SIGNALS="sensorSignals"
        NORM_TABLE="CV70-20-10Table_Sensor1020_all_MegaStrat_normFactorsNew.csv"
        LOSO_NORM="Sensor"

    elif [ "$spatial" = "Sensor_20" ]; then
        RUN_NAME="epoched_autoPreproc_20.0_0.5_250"
        SIGNALS="sensorSignals"
        NORM_TABLE="CV70-20-10Table_Sensor1020_all_MegaStrat_normFactorsNew.csv"
        LOSO_NORM="Sensor"

    elif [ "$spatial" = "Schaefer100_2" ]; then
        RUN_NAME="Schaefer100_freq-all_epochLength-2_overlap-0.5"
        SIGNALS="sourceSignals"
        NORM_TABLE="CV70-20-10Table_Schaefer100_all_2.0_MegaStrat_normFactorsNew.csv"
        LOSO_NORM="Schaefer100_2"

    elif [ "$spatial" = "Schaefer100_20" ]; then
        RUN_NAME="Schaefer100_freq-all_epochLength-20_overlap-0.5"
        SIGNALS="sourceSignals"
        NORM_TABLE="CV70-20-10Table_Schaefer100_all_20.0_MegaStrat_normFactorsNew.csv"
        LOSO_NORM=""

    elif [ "$spatial" = "Yeo7_2" ]; then
        RUN_NAME="OrthoAggSchaefer100_freq-all_epochLength-2_overlap-0.5"
        SIGNALS="sourceSignals"
        NORM_TABLE="CV70-20-10Table_OrthoAggSchaefer100_all_2.0_MegaStrat_normFactorsNew.csv"
        LOSO_NORM=""

    elif [ "$spatial" = "Yeo7_20" ]; then
        RUN_NAME="OrthoAggSchaefer100_freq-all_epochLength-20_overlap-0.5"
        SIGNALS="sourceSignals"
        NORM_TABLE="CV70-20-10Table_OrthoAggSchaefer100_all_20.0_MegaStrat_normFactorsNew.csv"
        LOSO_NORM=""

    else
        echo "Unknown configuration: $spatial" >&2
        return 1
    fi
}

# ---- analysis -> CV table and folds -------------------------------------------------------------
# RAW: raw targets on the main folds; CBP, NP: main folds restricted to a subgroup; LOSO: one fold
# per held-out dataset.

# Prints the results folder of one analysis, for use as: folder="$(analysis_dir main)"
analysis_dir() {
    local analysis=$1

    if [ "$analysis" = "main" ]; then
        echo "$RES/analysis_main"
    elif [ "$analysis" = "CBP" ]; then
        echo "$RES/analysis_CBP-subgroup"
    elif [ "$analysis" = "NP" ]; then
        echo "$RES/analysis_NP-subgroup"
    elif [ "$analysis" = "LOSO" ]; then
        echo "$RES/analysis_LOSO"
    elif [ "$analysis" = "RAW" ]; then
        echo "$RES/analysis_rawTargets"
    else
        echo "Unknown analysis: $analysis" >&2
        return 1
    fi
}

# Sets CV_TABLE and FOLDS for one analysis.
analysis_tables() {
    local analysis=$1

    if [[ "$analysis" == "main" || "$analysis" == "RAW" ]]; then
        CV_TABLE="$CV_MAIN"
        FOLDS="$MAIN_FOLDS"
    elif [[ "$analysis" == "CBP" || "$analysis" == "NP" ]]; then
        CV_TABLE="$(analysis_dir "$analysis")/inputs/CV_$analysis.csv"
        FOLDS="$MAIN_FOLDS"
    elif [ "$analysis" = "LOSO" ]; then
        CV_TABLE="$(analysis_dir LOSO)/inputs/CV_LOSO.csv"
        FOLDS="$LOSO_FOLDS"
    else
        echo "Unknown analysis: $analysis" >&2
        return 1
    fi
}

# Prints a message with the current date and time.
log() {
    echo "[$(date '+%F %T')] $*"
}

# Runs the command given to it ("$@" is that command). With DRY_RUN=1 it only prints the command.
run() {
    if [ "${DRY_RUN:-0}" = 1 ]; then
        echo "DRY_RUN $*"
    else
        "$@"
    fi
}

# Succeeds if the prediction file of every fold in $FOLDS exists, as the trainers write them.
all_folds_done() {
    local predictions_dir=$1
    local fold

    for fold in $FOLDS
    do
        if [ ! -f "$predictions_dir/$fold.npz" ]; then
            return 1
        fi
    done
    return 0
}

# ---- train one configuration on one target in one analysis (TRAINS a model) ----------------------
# Example: train "CPN:Yeo7_2:linear" painVars main
train() {
    local config=$1      # MODEL:SPATIAL:ALGORITHM
    local target=$2      # age, painVars, ageRAW, painVarsRAW
    local analysis=$3    # main, CBP, NP, LOSO, RAW

    local model spatial algo
    IFS=":" read -r model spatial algo <<< "$config"

    # Sets CV_TABLE, FOLDS, RUN_NAME, SIGNALS, NORM_TABLE and LOSO_NORM
    analysis_tables "$analysis"
    config_paths "$spatial"

    local out
    out="$(analysis_dir "$analysis")/$model/$RUN_NAME"
    local main_features="$RES/analysis_main/$model/$RUN_NAME/features"

    # Finished runs are skipped
    if all_folds_done "$out/predictions/${algo}_${target}"; then
        log "skip (done): $model $spatial $algo $target $analysis"
        return
    fi
    log "train: $model $spatial $algo $target $analysis"

    if [ "$model" = "CPN" ]; then
        train_cpn "$target" "$algo" "$main_features" "$out"
    else
        train_feature_model "$model" "$spatial" "$target" "$algo" "$analysis" "$main_features" "$out"
    fi
}

train_cpn() {
    local target=$1
    local algo=$2
    local main_features=$3
    local out=$4

    # $FOLDS has no quotes on purpose: each fold becomes its own argument
    run "$PY" -u "$TRAIN/CPN/train_cpn_models.py" \
        --target "$target" \
        --algorithm "$algo" \
        --CVtable_path "$CV_TABLE" \
        --folds $FOLDS \
        --features_dir "$main_features" \
        --output "$out"
}

train_feature_model() {
    local model=$1
    local spatial=$2
    local target=$3
    local algo=$4
    local analysis=$5
    local main_features=$6
    local out=$7

    local script
    if [ "$model" = "FBR" ]; then
        script="$TRAIN/MEEGBenchmark/train_fbr.py"
    elif [ "$model" = "FBS" ]; then
        script="$TRAIN/MEEGBenchmark/train_fbs.py"
    elif [ "$model" = "HCF" ]; then
        script="$TRAIN/MEEGBenchmark/train_hcf.py"
    else
        echo "Unknown model: $model" >&2
        return 1
    fi

    # Optional flags, collected in one string
    local extra_args=""

    # Pain: harmonized age as confounder
    if [[ "$target" == "painVars"* ]]; then
        extra_args="$extra_args --confounder age"
    fi

    local norm features_dir
    if [ "$analysis" = "LOSO" ]; then
        if [ -z "$LOSO_NORM" ]; then
            echo "No LOSO norm table for $spatial" >&2
            return 1
        fi
        norm="$(analysis_dir LOSO)/inputs/normFactors_LOSO_$LOSO_NORM.csv"
        features_dir="$out/features"
        extra_args="$extra_args --features_fold LOSO"    # one feature set for all LOSO folds

        # Inverse operators do not depend on the fold
        if [ "$model" = "FBS" ]; then
            extra_args="$extra_args --invops_fold Fold1"
        fi
    else
        norm="$NORM_DIR/$NORM_TABLE"
        features_dir="$main_features"                    # shared by all analyses on the main folds
    fi

    # $FOLDS and $extra_args have no quotes on purpose: each item becomes its own argument
    run "$PY" -u "$script" \
        --target "$target" \
        --algorithm "$algo" \
        --data_path "$FT/$SIGNALS/$RUN_NAME/" \
        --CVtable_path "$CV_TABLE" \
        --NCtable_path "$norm" \
        --output "$out" \
        --features_dir "$features_dir" \
        --folds $FOLDS \
        $extra_args
}

# ---- summaries and combined tables for one analysis ----------------------------------------------
# Example: summarize LOSO "age pain"
summarize() {
    local analysis=$1
    local targets=$2     # as in the tables: age, pain, ageRAW, painRAW

    local target model_type
    for target in $targets
    do
        for model_type in FBR FBS HCF
        do
            "$PY" "$ANALYSIS/summaries/summarize_feature_models.py" \
                --model_type "$model_type" \
                --target "$target" \
                --analysis "$analysis"
        done
        "$PY" "$ANALYSIS/summaries/summarize_predictions_CPN.py" --target "$target" --analysis "$analysis"
        "$PY" "$ANALYSIS/combine/combine_summaries.py" --target "$target" --analysis "$analysis"
    done
}
