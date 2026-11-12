#!/bin/bash

#SBATCH --job-name=pretraining_hypertune
#SBATCH --time=0-7:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
: "${TURBO_ROOT:?Start this job through its submit_*.sh script}"

RES_ROOT="$CPP_RESULTS"
PRETRAINED_ROOT="${PRETRAINED_ROOT:-$CPP_CHECKPOINTS}"  # checkpoints of the previous stage
CVTABLE_PATH="${CVTABLE_PATH:-${RES_ROOT}/analysis_main/inputs/CV70-20-10Table_MegaStrat.csv}"
NORM_DIR="${NORM_DIR:-${RES_ROOT}/analysis_main/inputs}"

current_learning_rate=$1
current_batch_size=$2
current_accum_iter=$3
current_mask_ratio=$4
run_name=$5
RUN_MODE=$6
continue_from_ckpnt=$7
outputdir=$8
fold=$9
modelGroup=${10}
repretrain_off=${11}
epochs=${12}
epochsStop=${13}
patch_width=${14}
folder_name=${15}
use_otis_general_preproc=${16}

# Check if the script is running as an sbatch job
if [ "$RUN_MODE" = "cluster" ]; then
    echo "Running as an sbatch job (Job ID: $SLURM_JOB_ID)"
    device_arg=""
else
    device_arg="--device cuda:1"
fi


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


# Set norm_tab_name and pretrained_model_path based on run_name
modelGroupS1=${modelGroup/S2/S1}
case "$run_name" in
    "OrthoAggSchaefer100_freq-all_epochLength-2_overlap-0.5")
        norm_tab_name="CV70-20-10Table_OrthoAggSchaefer100_all_2.0_MegaStrat_normFactorsNew.csv"
        pretrained_model_path="${PRETRAINED_ROOT}/${modelGroupS1}_mr_${current_mask_ratio}/OrthoAggSchaefer100_freq-all_epochLength-10_overlap-0"
        ;;
    "OrthoAggSchaefer100_freq-all_epochLength-20_overlap-0.5")
        norm_tab_name="CV70-20-10Table_OrthoAggSchaefer100_all_20.0_MegaStrat_normFactorsNew.csv"
        pretrained_model_path="${PRETRAINED_ROOT}/${modelGroupS1}_mr_${current_mask_ratio}/OrthoAggSchaefer100_freq-all_epochLength-10_overlap-0"
        ;;
    "Schaefer100_freq-all_epochLength-2_overlap-0.5")
        norm_tab_name="CV70-20-10Table_Schaefer100_all_2.0_MegaStrat_normFactorsNew.csv"
        pretrained_model_path="${PRETRAINED_ROOT}/${modelGroupS1}_mr_${current_mask_ratio}/Schaefer100_freq-all_epochLength-10_overlap-0"
        ;;
    "Schaefer100_freq-all_epochLength-20_overlap-0.5")
        norm_tab_name="CV70-20-10Table_Schaefer100_all_20.0_MegaStrat_normFactorsNew.csv"
        pretrained_model_path="${PRETRAINED_ROOT}/${modelGroupS1}_mr_${current_mask_ratio}/Schaefer100_freq-all_epochLength-10_overlap-0"
        ;;        
    "epoched_autoPreproc_20.0_0.5_250" | \
    "epoched_autoPreproc_CUT2.0_CUT0.5_250")
        norm_tab_name="CV70-20-10Table_Sensor1020_all_MegaStrat_normFactorsNew.csv"
        pretrained_model_path="${PRETRAINED_ROOT}/${modelGroupS1}_mr_${current_mask_ratio}/epoched_autoPreproc_10.0_0.0_250_Normalized"
        ;;
    *)
        echo "Cannot find matching normalization factor table"
        exit 1
        ;;
esac

echo "Using norm_tab_name: $norm_tab_name"
if [ "$use_otis_general_preproc" = "y" ]; then
    echo "Using OTiS general pre-processing pretrained model path"
else
    echo "Using pretrained_model_path: $pretrained_model_path"
fi


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
# Extract the string between the last and second last slash from max_ncc_file
max_ncc_dir=$(dirname "$max_ncc_file")
max_ncc_folder=$(basename "$max_ncc_dir")

if [ "$use_otis_general_preproc" = "y" ]; then
    max_ncc_file="$OTIS_WEIGHTS"
    modelArg="--model otis_baseDeep_patchX_dec160d4b"
    ignoreYArg="--ignore_pos_embed_y"
    c2pwArg="--chop2patch_width_mult"
    repretrArg="--repretrain"
    timestepsArg="--time_steps 4992"
    trainonlypeyArg="--train_only_pos_embed_y"
    # override patch_width to 24 for pretrained otis_baseDeep
    patch_width=24
    echo "Using hard-coded pretrained model: $max_ncc_file, but hyperparameters derived from identified pretrained model with highest NCC score." 
else
    echo "Using identified pretrained model with highest NCC score and corresponding hyperparameters."
    modelArg=""
    ignoreYArg=""
    c2pwArg=""
    repretrArg=""
    timestepsArg=""
    trainonlypeyArg=""
fi


# Extract the substrings corresponding to learning rate, batch size, accum iter, and mask ratio
IFS='_' read -r _ blr batch_size accum_iter mask_ratio <<< "$max_ncc_folder"
# Replace current_batch_size, current_accum_iter, current_learning_rate, and current_mask_ratio if they are -1
if [ "$current_batch_size" -eq -1 ]; then
    current_batch_size=$batch_size
fi

if [ "$current_accum_iter" -eq -1 ]; then
    current_accum_iter=$accum_iter
fi

if [ "$current_learning_rate" == "-1" ]; then
    current_learning_rate=$blr
fi

if [ "$current_mask_ratio" == "-1" ]; then
    current_mask_ratio=$mask_ratio
fi

echo current_batch_size: $current_batch_size
echo current_accum_iter: $current_accum_iter
echo current_learning_rate: $current_learning_rate
echo current_mask_ratio: $current_mask_ratio

if [ "$use_otis_general_preproc" = "y" ]; then
    echo "Increasing exponent of learning rate by 1 for OTiS general preproc"

    # Extract coefficient and exponent using shell parameter expansion
    coef="${current_learning_rate%e*}"         # before 'e'
    exp="${current_learning_rate#*e}"          # after 'e'

    # Increment exponent by 1
    new_exp=$((exp + 1))

    # Reconstruct LR
    current_learning_rate="${coef}e${new_exp}"
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


# Check if there are any checkpoint files in the output directory
checkpoint_file=$(find "$outputdir" -type f -name "checkpoint-[0-9]*-ncc-*.pth" | sort -V | tail -n 1)

if [[ -n "$checkpoint_file" && "$continue_from_ckpnt" == "y" ]]; then
    echo "Found checkpoint file: $checkpoint_file. Continuing re-pretraining from this checkpoint."
    max_ncc_file=$checkpoint_file
    # Extract the checkpoint number from the checkpoint file name
    checkpoint_no=$(basename "$checkpoint_file" | grep -oP '(?<=checkpoint-)[0-9]+(?=-ncc)')

    echo "Checkpoint number: $checkpoint_no"
    start_epoch_arg="--start_epoch $checkpoint_no"
else
    echo "No checkpoint file found in the output directory."
    checkpoint_no=0
    start_epoch_arg=""
fi


cmd_pretrain=(
    "${base_cmd[@]}"
    $CPP_CODE/training/OTiS/main_pretrain.py
    --data_path $TURBO_ROOT/data/fineTuning/"$folder_name"/"$run_name"/
    --labels_path ${CVTABLE_PATH}
    --CVtable_path ${CVTABLE_PATH}
    --NCtable_path ${NORM_DIR}/"$norm_tab_name"
    --fold "$fold"
    --batch_size "$current_batch_size"
    --wandb
    --wandb_entity "${WANDB_ENTITY:-}"
    --wandb_project pretraining_"$modelGroup"_"$run_name"
    --accum_iter "$current_accum_iter"
    --blr "$current_learning_rate"
    --epochs "$epochs"
    --warmup_epochs $((epochs / 10))
    --output_dir "$outputdir"
    --mask_ratio "$current_mask_ratio"
    --ncc_weight 10
    --patch_width "$patch_width" # 100 for Schaefer100; 25 for OrthoAgg
    --resume $max_ncc_file
    --downstream_task regression
    --upper_bnd 2
    --omit_optimizer_from_ckpnt
    $device_arg
    $start_epoch_arg
    --max_epochs "$epochsStop"
    --train_freqs all
    --test_freqs all
    $modelArg
    $ignoreYArg
    $c2pwArg
    $timestepsArg
    $repretrArg
    $trainonlypeyArg
)


if [ "$checkpoint_no" -eq $((epochs - 1)) ]; then
    echo "Checkpoint number matches epochs-1. Skipping re-pretraining."
    hardCodedPretrainedModel2="" 
elif [ "$repretrain_off" == "y" ]; then
    echo "re-pretraining disabled"
    hardCodedPretrainedModel2=$max_ncc_file
else
    echo "Starting re-pretraining..."
    export WANDB_DIR="$TURBO_ROOT/wandb" && \
    export WANDB__SERVICE_WAIT=1800 && \
    "${cmd_pretrain[@]}"
    hardCodedPretrainedModel2="" 
fi


# identify best epoch for evaluation
if [[ "$hardCodedPretrainedModel2" == "" ]]; then
    # Identify pre-trained model with the highest ncc-score
    max_ncc=-1
    max_ncc_file=""

    # Loop over all subfolders and search for files matching the pattern
    for file in $(find "$outputdir" -type f -name "checkpoint-*-ncc-*.pth"); do
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
        echo "File with the highest NCC score (pretrain S2): $max_ncc_file"
        echo "Highest NCC score (pretrain S2): $max_ncc"
    else
        echo "No matching files found."
    fi
else
    echo "Using hard-coded pretrained model: $hardCodedPretrainedModel2"
    max_ncc_file=$hardCodedPretrainedModel2
fi

# Frequency identifiers
freq_identifiers=("all")

# Repeat the following for all frequency identifiers
for freq in "${freq_identifiers[@]}"; do
    echo "evaluating embeddings for frequency identifier: $freq"

    # Replace "-all" in run_name with the current frequency identifier
    current_run_name="${run_name/all/$freq}"

    echo $freq
    echo $current_run_name

    # Common arguments
    cmd_eval_base=(
        "${base_cmd[@]}"
        $CPP_CODE/training/OTiS/main_pretrain.py
        --data_path $TURBO_ROOT/data/fineTuning/"$folder_name"/"$current_run_name"/
        --labels_path ${CVTABLE_PATH}
        --CVtable_path ${CVTABLE_PATH}
        --NCtable_path ${NORM_DIR}/"$norm_tab_name"
        --fold "$fold"
        --output_dir "$outputdir"
        --patch_width "$patch_width"
        --mask_ratio 0.0
        --resume $max_ncc_file
        --save_embeddings
        --downstream_task regression
        --upper_bnd 2
        --freq_identifier $freq
        --train_freqs single
        --test_freqs single
        --batch_size 4
        $device_arg
        $modelArg
        $c2pwArg
        $timestepsArg
        $repretrArg
    )

    # Command for --eval
    cmd_eval=("${cmd_eval_base[@]}" --eval)

    # Command for --eval_train
    cmd_eval_train=("${cmd_eval_base[@]}" --eval_train)

    # Command for --eval_test
    cmd_eval_test=("${cmd_eval_base[@]}" --eval_test)

    # Run the commands
    echo "Running command for --eval"
    "${cmd_eval[@]}"

    echo "Running command for --eval_train"
    "${cmd_eval_train[@]}"

    echo "Running command for --eval_test"
    "${cmd_eval_test[@]}"

done

# Copy the embeddings to CPP_results, where the final-layer models read them
case "$modelGroup" in *GENERAL*) otis_model=OTiS_GEN ;; *) otis_model=OTiS_EEG ;; esac
embed_published="${RES_ROOT}/analysis_main/${otis_model}/${run_name}/features/${fold}"
mkdir -p "$embed_published"
rm -f "$embed_published/LOCATION.txt"  # new embeddings replace the redirect
cp "$outputdir"/embeddings/*.csv "$embed_published/"


# Copy the embeddings to CPP_results, where the final-layer models read them
case "$modelGroup" in *GENERAL*) otis_model=OTiS_GEN ;; *) otis_model=OTiS_EEG ;; esac
embed_published="${RES_ROOT}/analysis_main/${otis_model}/${run_name}/features/${fold}"
mkdir -p "$embed_published"
rm -f "$embed_published/LOCATION.txt"  # new embeddings replace the redirect
cp "$outputdir"/embeddings/*.csv "$embed_published/"
