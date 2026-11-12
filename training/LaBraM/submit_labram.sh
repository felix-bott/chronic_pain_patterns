#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/../../paths.sh"
CKPT_ROOT="$CPP_CHECKPOINTS"  # new checkpoints and logs, outside CPP_results
RES_ROOT="$CPP_RESULTS"


ml python/anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda deactivate
conda activate labram
conda env list

RUN_MODE="${RUN_MODE:-cluster}"
# DRY_RUN=1: print the jobs that would be submitted, without creating folders or submitting anything
if [ "${DRY_RUN:-0}" = 1 ]; then
    RUN_MODE="cluster"
    sbatch() { echo "DRY_RUN sbatch $*"; }
    mkdir() { :; }
    sleep() { :; }
    squeue() { :; }
fi
use_envelope='' #'--use_envelope'


# Function to check the number of active jobs
check_active_jobs() {
    local max_jobs=10 # set maximum number of jobs to submit to cluster
    local active_jobs=$(squeue -u $USER --noheader | wc -l)

    # Wait until the number of active jobs is less than max_jobs
    while [ "$active_jobs" -ge "$max_jobs" ]; do
        echo "Waiting for jobs to finish... Currently $active_jobs active jobs."
        sleep 600
        active_jobs=$(squeue -u $USER --noheader | wc -l)
    done
}


for sigLen in SHORT LONG
do
    if [ "$sigLen" == "LONG" ]; then
        signalLength="epoched_autoPreproc_20.0_0.5_250"
        hyperparams=(
            "3e-5:16:4" # # learning_rate:batch_size:update_freq(=accum_iter)
            "1e-4:16:4" #
            "3e-4:16:4" 
            "1e-5:16:4" 
        )
        input_size=4000
    elif [ "$sigLen" == "SHORT" ]; then
        signalLength="epoched_autoPreproc_CUT2.0_CUT0.5_250"
        hyperparams=(
            "3e-5:64:1" # # learning_rate:batch_size:update_freq(=accum_iter)
            "1e-4:64:1" #
            "3e-4:64:1" 
            "1e-5:64:1"
        )
        input_size=400
    else
        echo "Invalid signal length specified. Exiting."
        exit 1
    fi
    
    for target in age pain
    do
        if [ "$target" == "age" ]; then
            lower_bnd=1
            upper_bnd=2
        elif [ "$target" == "pain" ]; then
            lower_bnd=0
            upper_bnd=1
        fi

        # Iterate over each learning rate
        for param in "${hyperparams[@]}"; 
        do
            IFS=":" read -r lr batch_size update_freq <<< "$param"

            for fold in Fold{1..10}; do

                output_dir="${CKPT_ROOT}/LaBraM/FT_${target}_${sigLen}/${fold}/setting_${lr}_${batch_size}_${update_freq}/"
                embed_dir="${RES_ROOT}/analysis_main/LaBraM/${signalLength}/features/${fold}/"
                target_name=$target; [ "$target" = pain ] && target_name=painVars
                published="${RES_ROOT}/analysis_main/LaBraM/${signalLength}/predictions/full_${target_name}/${fold}/setting_${lr}_${batch_size}_${update_freq}/"
                mkdir -p "$output_dir"
                mkdir -p "$embed_dir"

                printf "Starting training for fold %s with learning rate %s, batch size %s, update freq %s\n" "$fold" "$lr" "$batch_size" "$update_freq"

                if [[ -f "${published}retrained/eval_prediction_test.csv" && ( -f "${embed_dir}/embeddings_train.csv" || -f "${embed_dir}LOCATION.txt" ) ]]; then
                    echo "skipping job: Logits file and ebedding exist already."
                else
                    # Submit job to cluster using sbatch
                    logdir="${output_dir}logs"
                    mkdir -p "$logdir"
                    echo "logdir: ${logdir}"

                    # Run based on the selected mode
                    if [ "$RUN_MODE" = "cluster" ]; then
                        # Check and wait for fewer than a certain number of active jobs before submitting a new one
                        check_active_jobs

                        # submit job to cluster passing the expected positional arguments to job_labram.sh
                        sbatch --output="${logdir}/%A.out" --error="${logdir}/%A.err" ./job_labram.sh \
                            "$output_dir" \
                            "$fold" \
                            "$lr" \
                            "$batch_size" \
                            "$update_freq" \
                            "$signalLength" \
                            "$input_size" \
                            "$lower_bnd" \
                            "$upper_bnd" \
                            "$use_envelope" \
                            "$RUN_MODE" \
                            "$embed_dir" \
                            "$published" &
                    
                        sleep 3 # Wait for 3 seconds before submitting the next job
                    elif [ "$RUN_MODE" = "local" ]; then
                        # Run locally (call the script directly)
                        start_time=$(date +%s)
                        ./job_labram.sh \
                            "$output_dir" \
                            "$fold" \
                            "$lr" \
                            "$batch_size" \
                            "$update_freq" \
                            "$signalLength" \
                            "$input_size" \
                            "$lower_bnd" \
                            "$upper_bnd" \
                            "$use_envelope" \
                            "$RUN_MODE" \
                            "$embed_dir" \
                            "$published" &> "${logdir}/local_run.log"
                        end_time=$(date +%s)
                        elapsed=$((end_time - start_time))
                        echo "Execution time for fold $fold: ${elapsed} seconds" | tee -a "${logdir}/local_run.log"
                        wait # Wait for the previous command to complete before starting the next one
                    else
                        echo "Error: Invalid RUN_MODE specified. Use 'local' or 'cluster'."
                        exit 1
                    fi
                fi


            done
        done    
    done
done
