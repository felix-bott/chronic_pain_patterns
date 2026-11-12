#!/bin/bash

#SBATCH --job-name=pretraining_hypertune
#SBATCH --time=0-4:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
: "${TURBO_ROOT:?Start this job through its submit_*.sh script}"

RES_ROOT="$CPP_RESULTS"
CVTABLE_PATH="${CVTABLE_PATH:-${RES_ROOT}/analysis_main/inputs/CV70-20-10Table_MegaStrat.csv}"
NORM_DIR="${NORM_DIR:-${RES_ROOT}/analysis_main/inputs}"

model_type=$1
lr_factor=$2
target=$3
batch_size=$4
n_epochs=$5
outputdir=$6
run_name=$7
RUN_MODE=$8 

# Load python module
ml python/anaconda3
source $(conda info --base)/etc/profile.d/conda.sh

# Activate corresponding environment
# If you launch your script from a terminal where your environment is already loaded, conda won't activate the environment. The following guards against that. Not necessary if you always run this script from a clean terminal
conda deactivate
conda deactivate # sometimes there seems to be a "nested" python environment, e.g., (base) on top of (otis). To ensure a clean slate, I use conda deactivate twice.

# If the following does not work, try 'source activate <env-name>'
conda activate otis_benchmarks_new
conda env list

case "$run_name" in
    "OrthoAggSchaefer100_freq-all_epochLength-2_overlap-0.5")
        norm_tab_name="CV70-20-10Table_OrthoAggSchaefer100_all_2.0_MegaStrat_normFactorsNew.csv"
        folder_name="sourceSignals"
        upsample=2
        ;;
    "OrthoAggSchaefer100_freq-all_epochLength-20_overlap-0.5")
        norm_tab_name="CV70-20-10Table_OrthoAggSchaefer100_all_20.0_MegaStrat_normFactorsNew.csv"
        folder_name="sourceSignals"
        upsample=1
        ;;
    "Schaefer100_freq-all_epochLength-2_overlap-0.5")
        norm_tab_name="CV70-20-10Table_Schaefer100_all_2.0_MegaStrat_normFactorsNew.csv"
        folder_name="sourceSignals"
        upsample=2
        ;;
    "Schaefer100_freq-all_epochLength-20_overlap-0.5")
        norm_tab_name="CV70-20-10Table_Schaefer100_all_20.0_MegaStrat_normFactorsNew.csv"
        folder_name="sourceSignals"
        upsample=1
        ;;        
    "epoched_autoPreproc_20.0_0.5_250")
        norm_tab_name="CV70-20-10Table_Sensor1020_all_MegaStrat_normFactorsNew.csv"
        folder_name="sensorSignals"
        upsample=1
        ;;
    "epoched_autoPreproc_CUT2.0_CUT0.5_250")
        norm_tab_name="CV70-20-10Table_Sensor1020_all_MegaStrat_normFactorsNew.csv"
        folder_name="sensorSignals"
        upsample=2
        ;;
    *)
        echo "Cannot find matching normalization factor table"
        exit 1
        ;;
esac


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

# Construct the training command
cmd_train=(
    "${base_cmd[@]}"
    $CPP_CODE/training/MEEGBenchmark/train_convnets.py
    --model_type "$model_type"
    --lr_factor "$lr_factor"
    --target "$target"
    --batch_size "$batch_size"
    --n_epochs "$n_epochs"
    --data_path $TURBO_ROOT/data/fineTuning/"$folder_name"/"$run_name"/
    --CVtable_path ${CVTABLE_PATH}
    --NCtable_path ${NORM_DIR}/"$norm_tab_name"
    --output "$outputdir"
    --upsample "$upsample"
)

echo ${cmd_train[@]}

# Execute the training command
"${cmd_train[@]}"
