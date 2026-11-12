#!/bin/bash

#SBATCH --job-name=finetuning_hypertune_testcluster
#SBATCH --time=0-06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
: "${TURBO_ROOT:?Start this job through its submit_*.sh script}"

RES_ROOT="$CPP_RESULTS"
PRETRAINED_ROOT="${PRETRAINED_ROOT:-$CPP_CHECKPOINTS}"  # checkpoints of the previous stage
CVTABLE_PATH="${CVTABLE_PATH:-${RES_ROOT}/analysis_main/inputs/CV70-20-10Table_MegaStrat.csv}"
NORM_DIR="${NORM_DIR:-${RES_ROOT}/analysis_main/inputs}"

current_learning_rate=$1
current_batch_size=$2
current_weight_decay=$3
current_drop_path=$4
current_layer_decay=$5
fold=$6
run_name=$7
RUN_MODE=$8
outputdir=$9
modelGroup=${10}
continue_from_ckpnt=${11}
current_mask_ratio=${12}
patch_width=${13}
target=${14}
epochs=${15}
folder_name=${16}
accum_iterations=${17}
use_otis_general_preproc=${18}


echo $outputdir

sleep 30

# Load python module
ml python/anaconda3
source $(conda info --base)/etc/profile.d/conda.sh

# Activate corresponding environment
# If you launch your script from a terminal where your environment is already loaded, conda won't activate the environment. The following guards against that. Not necessary if you always run this script from a clean terminal
conda deactivate
conda deactivate # sometimes there seems to be a "nested" python environment, e.g., (base) on top of (otis). To ensure a clean slate, I use conda deactivate twice.

# If the following does not work, try 'source activate <env-name>'
conda activate otis
conda env list


# Function to find the latest checkpoint and set resume/start_epoch args

find_latest_checkpoint() {
    local outputdir="$1"
    local continue_from_ckpnt="${2:-n}"

    local checkpoint_files max_iter max_file checkpoint_path start_epoch

    if [[ "$continue_from_ckpnt" != "y" ]]; then
        echo "No resume requested. Starting training from scratch."
        declare -g resume_arg=""
        declare -g start_epoch_arg=""
        return
    fi

    # List candidate files and filter by pattern: checkpoint-<iter>-avg-<anything>.pth
    checkpoint_files=$(ls -1 "$outputdir" 2>/dev/null | grep -E '^checkpoint-[0-9]+-avg-.*\.pth$' || true)

    if [[ -z "$checkpoint_files" ]]; then
        echo "No checkpoint file found in $outputdir. Starting training from scratch."
        declare -g resume_arg=""
        declare -g start_epoch_arg=""
        return
    fi

    # Extract iteration numbers, sort numerically, take the largest
    max_iter=$(printf '%s\n' "$checkpoint_files" \
        | sed -E 's/^checkpoint-([0-9]+)-.*/\1/' \
        | sort -n \
        | tail -1)

    # Pick the file(s) that match that iteration; take the first
    max_file=$(printf '%s\n' "$checkpoint_files" \
        | grep -E "^checkpoint-${max_iter}-avg-.*\.pth$" \
        | head -1)

    if [[ -n "$max_file" ]]; then
        checkpoint_path="$outputdir/$max_file"
        echo "Checkpoint file found: $checkpoint_path with highest iteration: $max_iter"
        start_epoch=$((max_iter + 1))
        declare -g resume_arg="--resume $checkpoint_path"
        declare -g start_epoch_arg="--start_epoch $start_epoch"
    else
        echo "No matching checkpoint after parsing. Starting training from scratch."
        declare -g resume_arg=""
        declare -g start_epoch_arg=""
    fi
}


find_latest_checkpoint "$outputdir" "$continue_from_ckpnt"
echo $start_epoch_arg
echo $resume_arg

# Set norm_tab_name and pretrained_model_path based on run_name
modelGroupS2=${modelGroup/FT/PT_S2}
case "$run_name" in
    "OrthoAggSchaefer100_freq-all_epochLength-2_overlap-0.5")
        norm_tab_name="CV70-20-10Table_OrthoAggSchaefer100_all_2.0_MegaStrat_normFactorsNew.csv"
        pretrained_model_path="${PRETRAINED_ROOT}/${modelGroupS2}_mr_${current_mask_ratio}/${run_name}/${fold}"
        ;;
    "OrthoAggSchaefer100_freq-all_epochLength-20_overlap-0.5")
        norm_tab_name="CV70-20-10Table_OrthoAggSchaefer100_all_20.0_MegaStrat_normFactorsNew.csv"
        pretrained_model_path="${PRETRAINED_ROOT}/${modelGroupS2}_mr_${current_mask_ratio}/${run_name}/${fold}"
        ;;
    "Schaefer100_freq-all_epochLength-2_overlap-0.5")
        norm_tab_name="CV70-20-10Table_Schaefer100_all_2.0_MegaStrat_normFactorsNew.csv"
        pretrained_model_path="${PRETRAINED_ROOT}/${modelGroupS2}_mr_${current_mask_ratio}/${run_name}/${fold}"
        ;;
    "Schaefer100_freq-all_epochLength-20_overlap-0.5")
        norm_tab_name="CV70-20-10Table_Schaefer100_all_20.0_MegaStrat_normFactorsNew.csv"
        pretrained_model_path="${PRETRAINED_ROOT}/${modelGroupS2}_mr_${current_mask_ratio}/${run_name}/${fold}"
        ;;        
    "epoched_autoPreproc_20.0_0.5_250" | \
    "epoched_autoPreproc_CUT2.0_CUT0.5_250")
        norm_tab_name="CV70-20-10Table_Sensor1020_all_MegaStrat_normFactorsNew.csv"
        pretrained_model_path="${PRETRAINED_ROOT}/${modelGroupS2}_mr_${current_mask_ratio}/${run_name}/${fold}"
        ;;
    *)
        echo "Cannot find matching normalization factor table"
        exit 1
        ;;
esac


hardCodedPretrainedModel=""
if [[ "$hardCodedPretrainedModel" == "" ]]; then
    # Identify pre-trained model with the highest ncc-score
    max_ncc=-1
    max_ncc_file=""

    # Loop over all subfolders and search for files matching the pattern
    for file in $(find "$pretrained_model_path" -type f -name "checkpoint-*-ncc-*.pth"); do
        # Extract the float value after "ncc-"
        ncc_value=$(echo "$file" | grep -oP '(?<=ncc-)[0-9]+\.[0-9]+')
        
        # Check if this ncc_value is the highest found so far
        #if (( $(echo "$ncc_value > $max_ncc" | bc -l) )); then
        if awk "BEGIN {exit !($ncc_value > $max_ncc)}"; then
            max_ncc=$ncc_value
            max_ncc_file=$file
        fi
    done

    # Output the result
    if [[ -n "$max_ncc_file" ]]; then
        echo "File with the highest NCC score: $max_ncc_file"
        echo "Highest NCC score: $max_ncc"
    else
        echo "No matching files found."
    fi
else
    echo "Using hard-coded pretrained model: $hardCodedPretrainedModel"
    max_ncc_file=$hardCodedPretrainedModel
fi


# Set the base command based on RUN_MODE
if [ "$RUN_MODE" = "cluster" ]; then
    # Find a port within a specific range (49152-65535)
    function get_free_port() {
        while :; do
            PORT=$((49152 + RANDOM % 16384))
            ss -ltn | grep -q ":$PORT " || break  # Check if port is in use
        done
        echo "$PORT"
    }
    DYNAMIC_PORT=$(get_free_port)
    base_cmd=("torchrun" "--rdzv-endpoint=localhost:$DYNAMIC_PORT" "--nproc_per_node" "1" "--nnodes" "1" "--node_rank" "0")
else
    base_cmd=("python" "-u")
fi

# Set lower and upper bounds based on the target
if [[ "$target" == "age" ]]; then
    lower_bnd=1
    upper_bnd=2
elif [[ "$target" == "pain" ]]; then
    lower_bnd=0
    upper_bnd=1
else
    echo "Error: Unsupported target value '$target'."
    exit 1
fi

if [ "$use_otis_general_preproc" = "y" ]; then
    c2pwArg="--chop2patch_width_mult"
    timestepsArg="--time_steps 4992"
    # override patch_width to 24 for pretrained otis_baseDeep
    patch_width=24
else
    c2pwArg=""
    timestepsArg=""
fi

############################################
# 1) Base command (NO output_dir / log_dir here)
############################################
cmd_train_base=(
    "${base_cmd[@]}"
    $CPP_CODE/training/OTiS/main_finetune.py
    --data_path $TURBO_ROOT/data/fineTuning/"$folder_name"/"$run_name"/
    --labels_path ${CVTABLE_PATH}
    --CVtable_path ${CVTABLE_PATH}
    --NCtable_path ${NORM_DIR}/"$norm_tab_name"
    --fold "$fold"
    --finetune "$max_ncc_file"
    --downstream_task regression
    --lower_bnd "$lower_bnd"
    --upper_bnd "$upper_bnd"
    --nb_classes 1
    --target_weights 1
    --eval_criterion avg
    --wandb
    --wandb_entity "${WANDB_ENTITY:-}"
    --wandb_project finetuning_"$modelGroup"_"$run_name"
    --blr "$current_learning_rate"
    --batch_size "$current_batch_size"
    --epochs "$epochs"
    --warmup_epochs 5
    --weight_decay "$current_weight_decay"
    --accum_iter 1
    --drop_path "$current_drop_path"
    --layer_decay "$current_layer_decay"
    --crop_lower_bnd 1
    --crop_upper_bnd 1
    --ft_surr_phase_noise 0
    --rescaling_sigma 0
    --jitter_sigma 0
    --max_delta 0.00
    --patience 15
    --patch_width "$patch_width"
    --enable_MIL
    --bag_size 6
    --label_smoothing 0.0
    --MIL_agg_method "mean"
    --train_freqs "all"
    --test_freqs "all"
    --accum_iter "$accum_iterations"
    $c2pwArg
    $timestepsArg
)

############################################
# 2a) Normal training: add output_dir / log_dir
############################################
cmd_train=(
    "${cmd_train_base[@]}"
    --output_dir "$outputdir"
    --log_dir "$outputdir/logs"
    $start_epoch_arg
    $resume_arg
)

# Check if checkpoint for final epoch exists, skip training if so
final_checkpoint_pattern="checkpoint-$((epochs-1))-"
final_checkpoint_exists=$(ls "$outputdir" | grep -E "^$final_checkpoint_pattern.*" | head -n 1)
if [[ -n "$final_checkpoint_exists" ]]; then
    echo "Final checkpoint for epoch $((epochs-1)) exists: $final_checkpoint_exists. Skipping training."
else
    echo "Files in output directory ($outputdir):"
    ls -lh "$outputdir"
    echo "Running training command..."
    "${cmd_train[@]}"
fi

best_checkpoint=""
max_avg="-999999"
best_step=""
for file in $(ls "$outputdir" | grep -E 'checkpoint-[0-9]+-avg--?[0-9]+\.[0-9]+\.pth'); do
    step=$(echo "$file" | grep -oP '(?<=checkpoint-)[0-9]+')
    avg_value=$(echo "$file" | grep -oP '(?<=-avg-)-?[0-9]+\.[0-9]+')
    if awk "BEGIN {exit !($avg_value > $max_avg)}"; then
        max_avg=$avg_value
        best_checkpoint="$outputdir/$file"
        best_step=$step
    fi
done

if [[ -n "$best_checkpoint" ]]; then
    echo "Best checkpoint: $best_checkpoint"
    echo "Step: $best_step"
    echo "Avg: $max_avg"
else
    echo "No checkpoint file matching pattern found in $outputdir"
    exit 1
fi


############################################
# Evaluation: base + output_dir/log_dir + resume
############################################
cmd_eval_base=(
  "${base_cmd[@]}"
  $CPP_CODE/training/OTiS/main_finetune.py
  --data_path $TURBO_ROOT/data/fineTuning/"$folder_name"/"$run_name"/
  --labels_path ${CVTABLE_PATH}
  --CVtable_path ${CVTABLE_PATH}
  --NCtable_path ${NORM_DIR}/"$norm_tab_name"
  --fold "$fold"
  --downstream_task regression
  --lower_bnd "$lower_bnd"
  --upper_bnd "$upper_bnd"
  --nb_classes 1
  --target_weights 1
  --batch_size "$current_batch_size"
  --crop_lower_bnd 1
  --crop_upper_bnd 1
  --ft_surr_phase_noise 0
  --rescaling_sigma 0
  --jitter_sigma 0
  --max_delta 0.00
  --patch_width "$patch_width"
  --label_smoothing 0.0
  --save_logits
  --omit_optimizer_from_ckpnt
  --train_freqs "all"
  --test_freqs "all"
)


cmd_test_only=(
  "${cmd_eval_base[@]}"
  --output_dir "$outputdir"
  --log_dir "$outputdir/logs"
  --resume "$best_checkpoint"
  --test_only
)

if ls "$outputdir/logits"/logitsTEST*.csv 1> /dev/null 2>&1; then
    echo "Logits test file already exists in $outputdir/logits. Skipping test_only command."
else
    echo "Running test_only command..."
    printf '%q ' "${cmd_test_only[@]}"; echo
    "${cmd_test_only[@]}"
fi

############################################
# 2b) Retraining variant: add retrained dirs + extras
############################################
retrained_outputdir="$outputdir/retrained"
retrained_logdir="$retrained_outputdir/logs"
find_latest_checkpoint "$retrained_outputdir" "$continue_from_ckpnt"
echo $start_epoch_arg
echo $resume_arg
cmd_train_retrain=(
    "${cmd_train_base[@]}"
    --output_dir "$retrained_outputdir"
    --log_dir "$retrained_logdir"
    --max_epochs "$((best_step + 1))"
    --combineTrainVal
    --freeze_pos_embed_y
    $start_epoch_arg
    $resume_arg
)

final_checkpoint_pattern="checkpoint-$((best_step))-"
final_checkpoint_exists=$(ls "$retrained_outputdir" | grep -E "^$final_checkpoint_pattern.*" | head -n 1)
if [[ -n "$final_checkpoint_exists" ]]; then
    echo "Checkpoint for best step $best_step already exists in $retrained_outputdir. Skipping retraining."
else
    printf '%q ' "${cmd_train_retrain[@]}"; echo
    echo "Running retraining command..."
    "${cmd_train_retrain[@]}"
fi

# Evaluate retrained
retrained_best_checkpoint=$(ls "$retrained_outputdir" | grep -E "checkpoint-$best_step-avg--?[0-9]+\.[0-9]+\.pth" | head -n 1)
retrained_best_checkpoint="$retrained_outputdir/$retrained_best_checkpoint"
echo "Best checkpoint for retrained model: $retrained_best_checkpoint"
cmd_eval_retrained=(
  "${cmd_eval_base[@]}"
  --output_dir "$retrained_outputdir"
  --log_dir "$retrained_logdir"
  --resume "$retrained_best_checkpoint"
  --test_only
)

if ls "$outputdir/retrained/logits"/logitsTEST*.csv 1> /dev/null 2>&1; then
    echo "Logits test file already exists in $outputdir/logits. Skipping test_only command."
else
    echo "Running eval_retrained command..."
    "${cmd_eval_retrained[@]}"
fi

# Copy what the summaries read from the checkpoint folder to CPP_results
target_name=$target; [ "$target" = pain ] && target_name=painVars
case "$modelGroup" in *GENERAL*) otis_model=OTiS_GEN ;; *) otis_model=OTiS_EEG ;; esac
published="${RES_ROOT}/analysis_main/${otis_model}/${run_name}/predictions/full_${target_name}/${fold}/$(basename "$outputdir")"
mkdir -p "${published}/retrained/logits"
cp "$outputdir/log.txt" "$published/"
cp "$outputdir"/retrained/logits/logitsTEST*.csv "${published}/retrained/logits/"
