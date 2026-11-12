#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/../../../paths.sh"
CKPT_ROOT="$CPP_CHECKPOINTS"  # new checkpoints and logs, outside CPP_results
RES_ROOT="$CPP_RESULTS"

RUN_MODE="${RUN_MODE:-cluster}"
# DRY_RUN=1: print the jobs that would be submitted, without creating folders or submitting anything
if [ "${DRY_RUN:-0}" = 1 ]; then
    RUN_MODE="cluster"
    sbatch() { echo "DRY_RUN sbatch $*"; }
    mkdir() { :; }
    sleep() { :; }
    squeue() { :; }
fi
project_name="pretrain"
#run_name="Schaefer100_freq-all_epochLength-2_overlap-0.5" #
#run_name="Schaefer100_freq-all_epochLength-20_overlap-0.5"
run_name="${RUN_NAME:-epoched_autoPreproc_CUT2.0_CUT0.5_250}"  # set RUN_NAME to pretrain another variant
folder_name="${FOLDER_NAME:-sensorSignals}"
continue_from_ckpnt="y"
repretrain_off="n"
use_otis_general_preproc="${OTIS_GENERAL:-n}"  # y: OTiS_GEN (experiment 2025GENERAL)

if [[ "$run_name" == *"OrthoAgg"* ]]; then
    hyperparams=(
        #"3e-4:16:1:0.5"  #learning_rate:batch_size:accum_iter:mask_ratio
        #"3e-4:16:1:0.75"
        #"1e-3:16:1:0.5" 
        "1e-3:16:1:0.75"
        #"1e-4:16:1:0.5"
        #"1e-4:16:1:0.75"
        #"1e-3:64:1:0.5"
        #"1e-3:64:1:0.75"
        #"1e-4:64:1:0.5"
        #"1e-4:64:1:0.75"
        #"3e-4:64:1:0.5"
        #"3e-4:64:1:0.75"

        #"3e-4:16:1:0.875"
        #"1e-3:16:1:0.875"
        #"1e-4:16:1:0.875"
        #"1e-3:64:1:0.875"
        #"1e-4:64:1:0.875"
        #"3e-4:64:1:0.875"
    )
    epochs=25
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
    epochs=25
    epochsStop=200
    patch_width=100
    shrink_data_arg="--shrink_data"

elif [[ "$run_name" == *"autoPreproc"* ]]; then
    hyperparams=(
        "3.01e-4:8:2:0.75"
    )
    epochs=25
    epochsStop=9999
    patch_width=100
    shrink_data_arg="--shrink_data"

fi

# override patch width for use_otis_general_preproc
if [ "$use_otis_general_preproc" = "y" ]; then
    patch_width=24
    epochs=25
    if [[ ( "$run_name" == *"_20.0_"* || "$run_name" == *"-20_"* ) && "$run_name" != *"OrthoAgg"* ]]; then
        echo "This configuration (small patch_width, large segemt length) is not possible due to insufficient CUDA memory."
        exit 1
    fi
fi


# Function to check the number of active jobs
check_active_jobs() {
    local max_jobs=5 # set maximum number of jobs to submit to cluster
    local active_jobs=$(squeue -u $USER --noheader | wc -l)

    # Wait until the number of active jobs is less than max_jobs
    while [ "$active_jobs" -ge "$max_jobs" ]; 
    do
        echo "Waiting for jobs to finish... Currently $active_jobs active jobs."
        sleep 1000
        active_jobs=$(squeue -u $USER --noheader | wc -l)
    done
}

experiment_ID="2025"
if [[ "$use_otis_general_preproc" == "y" ]]; then
    experiment_ID="${experiment_ID}GENERAL"
fi
modelGroup="PT_S2_$experiment_ID" #finetunedModels
echo "Model Group: $modelGroup"

version=""

# Iterate over each learning rate
# HYPERPARAMS="blr:batch_size:accum_iter:mask_ratio ...": run these settings instead of the list above
if [ -n "${HYPERPARAMS:-}" ]; then read -r -a hyperparams <<< "$HYPERPARAMS"; fi
for param in "${hyperparams[@]}"; 
do
    IFS=":" read -r blr batch_size accum_iter mask_ratio <<< "$param"

    # Iterate over CV-folds
    for fold in Fold1 Fold2 Fold3 Fold4 Fold5 Fold6 Fold7 Fold8 Fold9 Fold10
    do 
        modelGroup_mr="${modelGroup}_mr_${mask_ratio}${version}"
        logdir=${CKPT_ROOT}/slurmOutputs/${modelGroup_mr}/${run_name}/${fold}/setting_${blr}_${batch_size}_${accum_iter}_${mask_ratio}
        outputdir=${CKPT_ROOT}/${modelGroup_mr}/${run_name}/${fold}/setting_${blr}_${batch_size}_${accum_iter}_${mask_ratio}
        mkdir -p $logdir
        mkdir -p $outputdir

        if [ "$RUN_MODE" = "cluster" ]; then
            # Check and wait for fewer than a certain number of active jobs before submitting a new one
            check_active_jobs

            # Run the command only if it was not run to completion before
            epoch=25
            checkpoint_file="checkpoint-$((epoch - 1))"
            #if ! ls "$outputdir" | grep -q "$checkpoint_file"; then
            if true; then
                # Run on the cluster using sbatch
                echo "Running experiment with blr=$blr, batch_size=$batch_size, accum_iter=$accum_iter, mask_ratio=$mask_ratio, fold=$fold"
                sbatch --output="${logdir}/%A.out" --error="${logdir}/%A.err" ./job_pretrain_s2.sh \
                    "$blr" \
                    "$batch_size" \
                    "$accum_iter" \
                    "$mask_ratio" \
                    "$run_name" \
                    "$RUN_MODE" \
                    "$continue_from_ckpnt" \
                    "$outputdir" \
                    "$fold" \
                    "$modelGroup" \
                    "$repretrain_off" \
                    "$epochs" \
                    "$epochsStop" \
                    "$patch_width" \
                    "$folder_name" \
                    "$use_otis_general_preproc"&
                sleep 3 # Wait for 3 seconds before submitting the next job
            else
                echo "Checkpoint file '$checkpoint_file' found in '$outputdir'. Skipping execution."
            fi

        elif [ "$RUN_MODE" = "local" ]; then
            # Run locally (call the script directly)
            echo "Running experiment with blr=$blr, batch_size=$batch_size, accum_iter=$accum_iter, mask_ratio=$mask_ratio"
            ./job_pretrain_s2.sh \
                "$blr" \
                "$batch_size" \
                "$accum_iter" \
                "$mask_ratio" \
                "$run_name" \
                "$RUN_MODE" \
                "$continue_from_ckpnt" \
                "$outputdir" \
                "$fold" \
                "$modelGroup" \
                "$repretrain_off" \
                "$epochs" \
                "$epochsStop" \
                "$patch_width" \
                "$folder_name" \
                "$use_otis_general_preproc" \
                &> "${logdir}/local_run.log" &
            wait # Wait for the previous command to complete before starting the next one

        else
            echo "Error: Invalid RUN_MODE specified. Use 'local' or 'cluster'."
            exit 1

        fi
    done
done
