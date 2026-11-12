#!/bin/bash

#SBATCH --job-name=pretraining_hypertune
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
: "${TURBO_ROOT:?Start this job through its submit_*.sh script}"

current_learning_rate=$1
current_batch_size=$2
current_accum_iter=$3
current_mask_ratio=$4
run_name=$5
RUN_MODE=$6
continue_from_ckpnt=$7
outputdir=$8
modelGroup=$9
epochs=${10}
epochsStop=${11}
patch_width=${12}
folder_name=${13}
shrink_data_arg=${14}

echo "Helloooooooo"

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Check if the script is running as an sbatch job
if [ "$RUN_MODE" = "cluster" ]; then
    echo "Running as an sbatch job (Job ID: $SLURM_JOB_ID)"
    device_arg=""
else
    device_arg="--device cuda:0"
fi


# Load python module
ml python/anaconda3
source $(conda info --base)/etc/profile.d/conda.sh

# Activate corresponding environment
# If you launch your script from a terminal where your environment is already loaded, conda won't activate the environment. The following guards against that. Not necessary if you always run this script from a clean terminal
conda deactivate
conda deactivate # sometimes there seems to be a "nested" python environment, e.g., (base) on top of (otis). To ensure a clean slate, I use conda deactivate twice.

conda activate otis
conda env list

# This script checks for an existing checkpoint file in the specified output directory if training should be continued from a previous checkpoint.
if [[ $continue_from_ckpnt == "y" ]]; then
    checkpoint_files=$(ls "$outputdir" | grep -E 'checkpoint-[0-9]+-ncc-[0-9]+\.[0-9]+\.pth')
else
    checkpoint_files=""
fi

# Initialize variables to store the file with the maximum checkpoint_iter
max_checkpoint_file=""
max_checkpoint_iter=-1

# Loop through each checkpoint file to find the one with the largest checkpoint_iter
for checkpoint_file in $checkpoint_files; do
    # Extract the integer after "checkpoint-"
    checkpoint_iter=$(echo "$checkpoint_file" | grep -oP '(?<=checkpoint-)\d+')
    
    # Check if this checkpoint_iter is the largest found so far
    if (( checkpoint_iter > max_checkpoint_iter )); then
        max_checkpoint_iter=$checkpoint_iter
        max_checkpoint_file=$checkpoint_file
    fi
done

# Check if we found any checkpoint files
if [[ -n "$max_checkpoint_file" ]]; then
    checkpoint_path="$outputdir/$max_checkpoint_file"
    echo "Checkpoint file found: $checkpoint_path with highest iteration: $max_checkpoint_iter"
    
    # Increment max_checkpoint_iter by 1 to set the start_epoch
    start_epoch=$((max_checkpoint_iter + 1))

    # Set the resume argument to use this checkpoint file
    resume_arg="--resume $checkpoint_path"
    start_epoch_arg="--start_epoch $start_epoch"
else
    echo "Starting training from scratch."
    resume_arg=""
    start_epoch_arg=""
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


# Common arguments
cmd=(
    "${base_cmd[@]}"
    $CPP_CODE/training/OTiS/main_pretrain.py
    --data_path $TURBO_ROOT/data/preTraining/"$folder_name"/"$run_name"/
    --CVtable_path $TURBO_ROOT/data/preTraining/resTables/CV5Table_manualEdit.csv
    --batch_size "$current_batch_size"
    --wandb
    --wandb_entity "${WANDB_ENTITY:-}"
    --wandb_project pretraining_"$modelGroup"_"$run_name"
    --accum_iter "$current_accum_iter"
    --blr "$current_learning_rate"
    --epochs "$epochs"
    --warmup_epochs $((epochs * 5 / 100))
    --output_dir "$outputdir"
    --mask_ratio "$current_mask_ratio"
    --ncc_weight 10
    --patch_width "$patch_width"
    --train_freqs all
    --test_freqs all
    --max_epochs "$epochsStop"
    $resume_arg
    $start_epoch_arg
    $device_arg
    $shrink_data_arg
)

# Run the command
export WANDB_DIR="$TURBO_ROOT/wandb" && \
"${cmd[@]}"
