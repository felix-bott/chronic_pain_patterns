#!/bin/bash

#SBATCH --job-name=finetuning_hypertune_testcluster
#SBATCH --time=0-04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
: "${TURBO_ROOT:?Start this job through its submit_*.sh script}"

RES_ROOT="$CPP_RESULTS"
CVTABLE_PATH="${CVTABLE_PATH:-${RES_ROOT}/analysis_main/inputs/CV70-20-10Table_MegaStrat.csv}"
NORM_DIR="${NORM_DIR:-${RES_ROOT}/analysis_main/inputs}"

output_dir=$1
fold=$2
lr=$3
batch_size=$4
update_freq=$5
signalLength=$6  # e.g., SHORT, LONG
input_size=$7    # e.g., 200, 400
lower_bnd=$8
upper_bnd=$9
use_envelope=${10}  # e.g., --use_envelope or empty 
RUN_MODE=${11}  # "local" or "cluster"
embed_dir=${12}
published=${13}

echo $outputdir

sleep 3

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
conda activate labram
conda env list


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


if [ -f "${output_dir}checkpoint-best.pth" ]; then
    echo "Checkpoint exists at ${output_dir}checkpoint-best.pth, skipping training for $fold with lr=$lr and batch_size=$batch_size."
else
    # Remove existing log.txt if it exists
    if [ -f "${output_dir}log.txt" ]; then
        rm "${output_dir}log.txt"
    fi

    "${base_cmd[@]}" \
    finetune_labram.py \
    --data_path $TURBO_ROOT/data/fineTuning/sensorSignals/${signalLength}/ \
    --labels_path ${CVTABLE_PATH} \
    --CVtable_path ${CVTABLE_PATH} \
    --NCtable_path ${NORM_DIR}/CV70-20-10Table_Sensor1020_all_MegaStrat_normFactorsNew.csv \
    --fold $fold \
    --lower_bnd $lower_bnd \
    --upper_bnd $upper_bnd \
    --nb_classes 1 \
    --input_size $input_size \
    --loss_type regression \
    --save_ckpt_freq 20 \
    --no_auto_resume \
    --output_dir $output_dir \
    --log_dir ${output_dir}logs \
    --model labram_base_patch200_200 \
    --finetune "$LABRAM_WEIGHTS" \
    --weight_decay 0.05 \
    --batch_size $batch_size \
    --lr $lr \
    --update_freq $update_freq \
    --warmup_epochs 5 \
    --epochs 50 \
    --layer_decay 0.65 \
    --drop_path 0.1 \
    --disable_rel_pos_bias \
    --abs_pos_emb \
    --dataset TUM \
    --disable_qkv_bias \
    --seed 0 \
    $use_envelope \
    #> $output_dir/run_train.log 2>&1

    wait
fi

if [ -f "${output_dir}eval_prediction_test.csv" ]; then
    echo "Evaluation results exist at ${output_dir}eval_prediction_test.csv, skipping evaluation for $fold with lr=$lr and batch_size=$batch_size."
else
    "${base_cmd[@]}" \ 
    finetune_labram.py \
    --data_path $TURBO_ROOT/data/fineTuning/sensorSignals/${signalLength}/ \
    --labels_path ${CVTABLE_PATH} \
    --CVtable_path ${CVTABLE_PATH} \
    --NCtable_path ${NORM_DIR}/CV70-20-10Table_Sensor1020_all_MegaStrat_normFactorsNew.csv \
    --fold $fold \
    --lower_bnd $lower_bnd \
    --upper_bnd $upper_bnd \
    --nb_classes 1 \
    --input_size $input_size \
    --loss_type regression \
    --no_auto_resume \
    --output_dir $output_dir \
    --log_dir ${output_dir}logs \
    --model labram_base_patch200_200 \
    --finetune "$LABRAM_WEIGHTS" \
    --disable_rel_pos_bias \
    --abs_pos_emb \
    --dataset TUM \
    --disable_qkv_bias \
    --seed 0 \
    --eval \
    --resume $output_dir/checkpoint-best.pth \
    $use_envelope \
    #> $output_dir/run_eval.log 2>&1
    wait
fi

# get best epoch
max_epoch=$(python get_best_epoch "$output_dir/log.txt")
echo "Best epoch for $fold with lr=$lr and batch_size=$batch_size is $max_epoch."

mkdir -p "${output_dir}retrained"
if [ -f "${output_dir}retrained/checkpoint-best.pth" ]; then
    echo "Checkpoint exists at $output_dir/retrained/checkpoint-best.pth, skipping retraining for $fold with lr=$lr and batch_size=$batch_size."
else
    # Remove existing log.txt if it exists
    if [ -f "${output_dir}retrained/log.txt" ]; then
        rm "${output_dir}retrained/log.txt"
    fi

    "${base_cmd[@]}" \
    finetune_labram.py \
    --data_path $TURBO_ROOT/data/fineTuning/sensorSignals/${signalLength}/ \
    --labels_path ${CVTABLE_PATH} \
    --CVtable_path ${CVTABLE_PATH} \
    --NCtable_path ${NORM_DIR}/CV70-20-10Table_Sensor1020_all_MegaStrat_normFactorsNew.csv \
    --fold $fold \
    --lower_bnd $lower_bnd \
    --upper_bnd $upper_bnd \
    --nb_classes 1 \
    --input_size $input_size \
    --loss_type regression \
    --save_ckpt_freq 20 \
    --no_auto_resume \
    --output_dir ${output_dir}retrained/ \
    --log_dir ${output_dir}retrained/logs \
    --model labram_base_patch200_200 \
    --finetune "$LABRAM_WEIGHTS" \
    --weight_decay 0.05 \
    --batch_size $batch_size \
    --lr $lr \
    --update_freq $update_freq \
    --warmup_epochs 5 \
    --epochs 50 \
    --layer_decay 0.65 \
    --drop_path 0.1 \
    --disable_rel_pos_bias \
    --abs_pos_emb \
    --dataset TUM \
    --disable_qkv_bias \
    --seed 0 \
    --fixed_epoch $max_epoch \
    --merge_train_val \
    $use_envelope \
    #> ${output_dir}retrained/run_train.log 2>&1

    wait
fi


if [ -f "${output_dir}retrained/eval_prediction_test.csv" ]; then
    echo "Evaluation results exist at ${output_dir}retrained/eval_prediction_test.csv, skipping reevaluation for $fold with lr=$lr and batch_size=$batch_size."
else
    "${base_cmd[@]}" \
    finetune_labram.py \
    --data_path $TURBO_ROOT/data/fineTuning/sensorSignals/${signalLength}/ \
    --labels_path ${CVTABLE_PATH} \
    --CVtable_path ${CVTABLE_PATH} \
    --NCtable_path ${NORM_DIR}/CV70-20-10Table_Sensor1020_all_MegaStrat_normFactorsNew.csv \
    --fold $fold \
    --lower_bnd $lower_bnd \
    --upper_bnd $upper_bnd \
    --nb_classes 1 \
    --input_size $input_size \
    --loss_type regression \
    --no_auto_resume \
    --output_dir ${output_dir}retrained/ \
    --log_dir ${output_dir}retrained/logs \
    --model labram_base_patch200_200 \
    --finetune "$LABRAM_WEIGHTS" \
    --disable_rel_pos_bias \
    --abs_pos_emb \
    --dataset TUM \
    --disable_qkv_bias \
    --seed 0 \
    --eval \
    --resume ${output_dir}retrained/checkpoint-best.pth \
    $use_envelope \
    #> ${output_dir}retrained/run_eval.log 2>&1

    wait
fi

if [ -f "${embed_dir}embedding_train.csv" ]; then
    echo "Embeddings already exist at ${embed_dir}embedding_train.csv, skipping reevaluation for $fold with lr=$lr and batch_size=$batch_size."
else
    rm -f "${embed_dir}LOCATION.txt"  # new embeddings replace the redirect
    "${base_cmd[@]}" \
    finetune_labram.py \
    --data_path $TURBO_ROOT/data/fineTuning/sensorSignals/${signalLength}/ \
    --labels_path ${CVTABLE_PATH} \
    --CVtable_path ${CVTABLE_PATH} \
    --NCtable_path ${NORM_DIR}/CV70-20-10Table_Sensor1020_all_MegaStrat_normFactorsNew.csv \
    --fold $fold \
    --lower_bnd $lower_bnd \
    --upper_bnd $upper_bnd \
    --nb_classes 1 \
    --input_size $input_size \
    --loss_type regression \
    --no_auto_resume \
    --output_dir ${embed_dir} \
    --log_dir ${output_dir}retrained/logs \
    --model labram_base_patch200_200 \
    --finetune "$LABRAM_WEIGHTS" \
    --disable_rel_pos_bias \
    --abs_pos_emb \
    --dataset TUM \
    --disable_qkv_bias \
    --seed 0 \
    --eval \
    --eval_all \
    --save_embeddings ${embed_dir} \
    $use_envelope \

    wait
fi

# Copy what the summaries read from the checkpoint folder to CPP_results
mkdir -p "${published}retrained"
cp "${output_dir}log.txt" "$published"
cp "${output_dir}retrained/eval_prediction_test.csv" "${published}retrained/"
