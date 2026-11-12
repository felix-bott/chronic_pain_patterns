#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/../../../paths.sh"
CKPT_ROOT="$CPP_CHECKPOINTS"  # new checkpoints and logs, outside CPP_results
RES_ROOT="$CPP_RESULTS"

RUN_MODE="${RUN_MODE:-local}"
# DRY_RUN=1: print the jobs that would be submitted, without creating folders or submitting anything
if [ "${DRY_RUN:-0}" = 1 ]; then
    RUN_MODE="cluster"
    sbatch() { echo "DRY_RUN sbatch $*"; }
    mkdir() { :; }
    sleep() { :; }
    squeue() { :; }
fi
project_name="pretrain"
run_name="${RUN_NAME:-OrthoAggSchaefer100_freq-all_epochLength-10_overlap-0}"  # set RUN_NAME to pretrain another variant
#run_name="Schaefer100_freq-all_epochLength-10_overlap-0" #
#run_name="epoched_autoPreproc_10.0_0.0_250_Normalized"
folder_name="${FOLDER_NAME:-sourceSignalsNormalized}"
continue_from_ckpnt="y"


if [[ "$run_name" == *"OrthoAgg"* ]]; then
    hyperparams=(
        #"3e-4:16:1:0.5"  #learning_rate:batch_size:accum_iter:mask_ratio
        #"3e-4:16:1:0.75"
        #"1e-3:16:1:0.5" 
        #"1e-3:16:1:0.75"
        #"1e-4:16:1:0.5"
        #"1e-4:16:1:0.75"
        #"1e-3:64:1:0.5"
        #"1e-3:64:1:0.75"
        #"1e-4:64:1:0.5"
        #"1e-4:64:1:0.75"
        #"3e-4:64:1:0.5"
        #"3e-4:64:1:0.75"

        #"3e-4:16:1:0.875"
        "1e-3:16:1:0.875"
        #"1e-4:16:1:0.875"
        #"1e-3:64:1:0.875"
        #"1e-4:64:1:0.875"
        #"3e-4:64:1:0.875"
    )
    epochs=800
    epochsStop=9999 #200 9999
    patch_width=25
    shrink_data_arg=""

elif [[ "$run_name" == *"Schaefer100"* ]]; then
    hyperparams=(
        #"3e-3:8:2:0.875" # done
        #"1e-3:8:2:0.875" # done
        #"3e-4:8:2:0.875"  # done -> select -> nan -> repeat with epochs = 150 -> FINAL?
        #"3e-3:16:2:0.875" # nan
        #"1e-3:16:2:0.875" # done
        #"3e-4:16:2:0.875" # done
        #"1e-4:8:2:0.875" # in progress

        #"3e-3:8:2:0.5"  # nan
        #"3e-3:8:2:0.75" # nan
        #"1e-3:8:2:0.5"  # done -> select -> nan
        "1e-3:8:2:0.75" # done
        #"3e-4:8:2:0.5"  # done -> select as backup for "1e-3:8:2:0.5" -> FINAL
        #"3.01e-4:8:2:0.75" # done -> select -> FINAL
        #"3e-3:16:2:0.5"  # nan
        #"3e-3:16:2:0.75" # done
        #"1e-3:16:2:0.5"  # nan
        #"1e-3:16:2:0.75" # nan
        #"3e-4:16:2:0.5"  # done
        #"3e-4:16:2:0.75" # done
    )
    epochs=300 #300
    epochsStop=150
    patch_width=100
    shrink_data_arg="--shrink_data"

elif [[ "$run_name" == *"autoPreproc"* ]]; then
    hyperparams=(
        "3.01e-4:8:2:0.75" # done -> select -> FINAL
    )
    epochs=150
    epochsStop=9999
    patch_width=100
    shrink_data_arg="--shrink_data"

fi


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

experiment_ID="2025"
modelGroup="PT_S1_$experiment_ID" #finetunedModels

# Iterate over each learning rate
# HYPERPARAMS="blr:batch_size:accum_iter:mask_ratio ...": run these settings instead of the list above
if [ -n "${HYPERPARAMS:-}" ]; then read -r -a hyperparams <<< "$HYPERPARAMS"; fi
for param in "${hyperparams[@]}"; do
    IFS=":" read -r blr batch_size accum_iter mask_ratio <<< "$param"

    modelGroup_mr="${modelGroup}_mr_${mask_ratio}"
    logdir=${CKPT_ROOT}/slurmOutputs/${modelGroup_mr}/${run_name}/setting_${blr}_${batch_size}_${accum_iter}_${mask_ratio}
    outputdir=${CKPT_ROOT}/${modelGroup_mr}/${run_name}/setting_${blr}_${batch_size}_${accum_iter}_${mask_ratio}
    mkdir -p $logdir
    mkdir -p $outputdir

    if [ "$RUN_MODE" = "cluster" ]; then
        # Check and wait for fewer than a certain number of active jobs before submitting a new one
        check_active_jobs

        # Run on the cluster using sbatch
        echo "Running experiment with blr=$blr, batch_size=$batch_size, accum_iter=$accum_iter, mask_ratio=$mask_ratio"
        sbatch --output="${logdir}/%A.out" --error="${logdir}/%A.err" ./job_pretrain_s1.sh \
            $blr \
            $batch_size \
            $accum_iter \
            $mask_ratio \
            $run_name \
            $RUN_MODE \
            $continue_from_ckpnt \
            $outputdir \
            $modelGroup \
            $epochs \
            $epochsStop \
            $patch_width \
            $folder_name \
            $shrink_data_arg \
        sleep 3 # Wait for 3 seconds before submitting the next job

    elif [ "$RUN_MODE" = "local" ]; then
        
        # Run locally (call the script directly)
        echo "Running experiment with blr=$blr, batch_size=$batch_size, accum_iter=$accum_iter, mask_ratio=$mask_ratio"
        ./job_pretrain_s1.sh \
            $blr \
            $batch_size \
            $accum_iter \
            $mask_ratio \
            $run_name \
            $RUN_MODE \
            $continue_from_ckpnt \
            $outputdir \
            $modelGroup \
            $epochs \
            $epochsStop \
            $patch_width \
            $folder_name \
            $shrink_data_arg \
            &> "${logdir}/local_run.log" &
        wait # Wait for the previous command to complete before starting the next one

    else
        echo "Error: Invalid RUN_MODE specified. Use 'local' or 'cluster'."
        exit 1
    fi
done
