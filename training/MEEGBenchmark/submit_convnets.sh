#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/../../paths.sh"
CKPT_ROOT="$CPP_CHECKPOINTS"  # new checkpoints and logs, outside CPP_results
RES_ROOT="$CPP_RESULTS"


# Parameters
RUN_MODE="${RUN_MODE:-cluster}"
# DRY_RUN=1: print the jobs that would be submitted, without creating folders or submitting anything
if [ "${DRY_RUN:-0}" = 1 ]; then
    RUN_MODE="cluster"
    sbatch() { echo "DRY_RUN sbatch $*"; }
    mkdir() { :; }
    sleep() { :; }
    squeue() { :; }
fi
model_types=("deep" "shallow") #("deep" "shallow")
targets=("painVars" "age") # "painVars")
lr_factors=(0.1) #(1 3 0.33)
run_names=(
    "OrthoAggSchaefer100_freq-all_epochLength-20_overlap-0.5"
    "epoched_autoPreproc_20.0_0.5_250"
    "Schaefer100_freq-all_epochLength-20_overlap-0.5"
    "OrthoAggSchaefer100_freq-all_epochLength-2_overlap-0.5"
    "epoched_autoPreproc_CUT2.0_CUT0.5_250"
    "Schaefer100_freq-all_epochLength-2_overlap-0.5"
)
n_epochs=20

# Function to check the number of active jobs
check_active_jobs() {
    local max_jobs=8 # set maximum number of jobs to submit to cluster
    local active_jobs=$(squeue -u $USER --noheader | wc -l)

    # Wait until the number of active jobs is less than max_jobs
    while [ "$active_jobs" -ge "$max_jobs" ]; do
        echo "Waiting for jobs to finish... Currently $active_jobs active jobs."
        sleep 600
        active_jobs=$(squeue -u $USER --noheader | wc -l)
    done
}

# Iterate over all combinations of parameters
for target in "${targets[@]}"; do
    for model_type in "${model_types[@]}"; do
        for lr_factor in "${lr_factors[@]}"; do
            for run_name in "${run_names[@]}"; do
                
                if [[ "$run_name" == "epoched_autoPreproc_20.0_0.5_250" || "$run_name" == "Schaefer100_freq-all_epochLength-20_overlap-0.5" || "$run_name" == "Schaefer100_freq-all_epochLength-2_overlap-0.5" ]]; then
                    batch_size=64
                    echo "Setting batch_size=64 for run_name=$run_name to avoid CUDA memory issues."
                else
                    batch_size=128
                    echo "Setting batch_size=128 for run_name=$run_name."
                fi


                logdir=${CKPT_ROOT}/slurmOutputs/ConvNet${model_type}_${target}/${run_name}/all_folds_setting_${lr_factor}_${batch_size}
                model_folder=CND; [ "$model_type" = shallow ] && model_folder=CNS
                outputdir=${RES_ROOT}/analysis_main/${model_folder}/${run_name}/predictions/full_${target}/all_folds_setting_${lr_factor}_${batch_size}
                mkdir -p $logdir
                mkdir -p $outputdir

                # Skip if results already exist
                if [ -f "${outputdir}/all_fold_results.npz" ]; then
                    echo "Found ${outputdir}/all_fold_results.npz — skipping experiment with model_type=$model_type, lr_factor=$lr_factor, target=$target, batch_size=$batch_size, n_epochs=$n_epochs"
                    continue
                fi

                if [ "$RUN_MODE" = "cluster" ]; then
                    # Check and wait for fewer than a certain number of active jobs before submitting a new one
                    check_active_jobs

                    # Run on the cluster using sbatch
                    echo "Running experiment with model_type=$model_type, lr_factor=$lr_factor, target=$target, batch_size=$batch_size, n_epochs=$n_epochs"
                    sbatch --output="${logdir}/%A.out" --error="${logdir}/%A.err" ./job_convnets.sh \
                        "$model_type" \
                        "$lr_factor" \
                        "$target" \
                        "$batch_size" \
                        "$n_epochs" \
                        "$outputdir" \
                        "$run_name" \
                        "$RUN_MODE" &
                    sleep 3 # Wait for 3 seconds before submitting the next job


                elif [ "$RUN_MODE" = "local" ]; then
                    # Run locally (call the script directly)
                    echo "Running experiment with model_type=$model_type, lr_factor=$lr_factor, target=$target, batch_size=$batch_size, n_epochs=$n_epochs"
                    ./job_convnets.sh \
                        "$model_type" \
                        "$lr_factor" \
                        "$target" \
                        "$batch_size" \
                        "$n_epochs" \
                        "$outputdir" \
                        "$run_name" \
                        "$RUN_MODE" \
                        &> "${logdir}/local_run.log" &
                    wait # Wait for the previous command to complete before starting the next one
                else
                    echo "Error: Invalid RUN_MODE specified. Use 'local' or 'cluster'."
                    exit 1

                fi
            done
        done
    done
done


                        
                        
