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

#Variants


#target="pain" # "pain" OR "age"


VARIANTS=(
  "OrthoAggSchaefer100_freq-all_epochLength-20_overlap-0.5|sourceSignals|n"
  "epoched_autoPreproc_20.0_0.5_250|sensorSignals|n"
  "Schaefer100_freq-all_epochLength-20_overlap-0.5|sourceSignals|n"
  "OrthoAggSchaefer100_freq-all_epochLength-2_overlap-0.5|sourceSignals|n"
  "epoched_autoPreproc_CUT2.0_CUT0.5_250|sensorSignals|n"
  "Schaefer100_freq-all_epochLength-2_overlap-0.5|sourceSignals|n"

  "OrthoAggSchaefer100_freq-all_epochLength-2_overlap-0.5|sourceSignals|y"
  "epoched_autoPreproc_CUT2.0_CUT0.5_250|sensorSignals|y"
  "Schaefer100_freq-all_epochLength-2_overlap-0.5|sourceSignals|y"
)

TARGETS=("pain" "age")  # iterate over both targets


continue_from_ckpnt="y"

# Function to check the number of active jobs
check_active_jobs() {
    local max_jobs=11 # set maximum number of jobs to submit to cluster
    local active_jobs=$(squeue -u $USER --noheader | wc -l)

    # Wait until the number of active jobs is less than max_jobs
    while [ "$active_jobs" -ge "$max_jobs" ]; do
        echo "Waiting for jobs to finish... Currently $active_jobs active jobs."
        sleep 600
        active_jobs=$(squeue -u $USER --noheader | wc -l)
    done
}


for variant in "${VARIANTS[@]}"; do
    IFS="|" read -r run_name folder_name use_otis_general_preproc <<< "$variant"

    for target in "${TARGETS[@]}"; do

        if [[ "$run_name" == *"OrthoAgg"* ]]; then
            patch_width=25
        else
            patch_width=100
        fi

        # override patch width for use_otis_general_preproc
        if [ "$use_otis_general_preproc" = "y" ]; then
            patch_width=24
        fi

        # First (default) set of hyperparameters
        hyperparams=(
            #1e-5:0.2:0.1:0.75:0.75:25 #blr batch_size weight_decay drop_path layer_decay mask_ratio(during pretraining) epochs
            #3e-6:0.2:0.1:0.75:0.75:25
            #3e-5:0.2:0.1:0.75:0.75:25
            1e-4:0.2:0.1:0.75:0.75:25
        )

        # Select which set to use based on run_name
        if [[ ( "$run_name" == *"20.0"* || "$run_name" == *"-20_"* ) && "$run_name" != *"OrthoAgg"* && "$run_name" == *"Schaefer100"* ]]; then
            # this is to ensure that no CUDA memory issues occur when using larger data (20s epochs in sensor level or Schaefer100 configuration)
            accum_iterations=4
            batch_size=4
        elif [[ ( "$run_name" == *"20.0"* || "$run_name" == *"-20_"* ) && "$run_name" == *"epoched_autoPreproc"* ]]; then
            accum_iterations=2
            batch_size=8
        else
            accum_iterations=1
            batch_size=16
        fi


        experiment_ID="2025"
        if [[ "$use_otis_general_preproc" == "y" ]]; then
            experiment_ID="${experiment_ID}GENERAL"
        fi
        modelGroup="FT_$experiment_ID" #finetunedModels


        for param in "${hyperparams[@]}"; do
            IFS=":" read -r  blr weight_decay drop_path layer_decay mask_ratio epochs<<< "$param"

            # Iterate over CV-folds
            for fold in Fold1 Fold2 Fold3 Fold4 Fold5 Fold6 Fold7 Fold8 Fold9 Fold10 
            do  
                modelGroup_mr="${modelGroup}_mr_${mask_ratio}_${target}"
                logdir=${CKPT_ROOT}/slurmOutputs/${modelGroup_mr}/${run_name}/${fold}/setting_${blr}_${batch_size}_${weight_decay}_${drop_path}_${layer_decay}_${mask_ratio}
                outputdir=${CKPT_ROOT}/${modelGroup_mr}/${run_name}/${fold}/setting_${blr}_${batch_size}_${weight_decay}_${drop_path}_${layer_decay}_${mask_ratio}
                mkdir -p $logdir
                mkdir -p $outputdir

                echo "Running experiment with blr=$blr, batch_size=$batch_size, weight_decay=$weight_decay, drop_path=$drop_path, layer_decay=$layer_decay, fold=$fold, mr=$mask_ratio, target=$target", epochs=$epochs

                target_name=$target; [ "$target" = pain ] && target_name=painVars
                case "$modelGroup" in *GENERAL*) otis_model=OTiS_GEN ;; *) otis_model=OTiS_EEG ;; esac
                published="${RES_ROOT}/analysis_main/${otis_model}/${run_name}/predictions/full_${target_name}/${fold}/$(basename "$outputdir")"
                if ls "${published}/retrained/logits/logitsTEST"*.csv 1> /dev/null 2>&1; then
                    echo "Logits file exists in ${published}/retrained/logits/. Skipping this iteration."
                    continue
                fi

                # Run based on the selected mode
                if [ "$RUN_MODE" = "cluster" ]; then
                    # Check and wait for fewer than a certain number of active jobs before submitting a new one
                    check_active_jobs

                    # Submit job to cluster using sbatch
                    sbatch --output="${logdir}/%A.out" --error="${logdir}/%A.err" ./job_finetune.sh \
                        "$blr" \
                        "$batch_size" \
                        "$weight_decay" \
                        "$drop_path" \
                        "$layer_decay" \
                        "$fold" \
                        "$run_name" \
                        "$RUN_MODE" \
                        "$outputdir" \
                        "$modelGroup" \
                        "$continue_from_ckpnt" \
                        "$mask_ratio" \
                        "$patch_width" \
                        "$target" \
                        "$epochs" \
                        "$folder_name" \
                        "$accum_iterations" \
                        "$use_otis_general_preproc" &

                    sleep 3 # Wait for 3 seconds before submitting the next job
                elif [ "$RUN_MODE" = "local" ]; then
                    # Run locally (call the script directly)
                    start_time=$(date +%s)
                    ./job_finetune.sh \
                        "$blr" \
                        "$batch_size" \
                        "$weight_decay" \
                        "$drop_path" \
                        "$layer_decay" \
                        "$fold" \
                        "$run_name" \
                        "$RUN_MODE" \
                        "$outputdir" \
                        "$modelGroup" \
                        "$continue_from_ckpnt" \
                        "$mask_ratio" \
                        "$patch_width" \
                        "$target" \
                        "$epochs" \
                        "$folder_name" \
                        "$accum_iterations" \
                        "$use_otis_general_preproc" &> "${logdir}/local_run.log"
                    end_time=$(date +%s)
                    elapsed=$((end_time - start_time))
                    echo "Execution time for fold $fold: ${elapsed} seconds" | tee -a "${logdir}/local_run.log"
                    wait # Wait for the previous command to complete before starting the next one
                else
                    echo "Error: Invalid RUN_MODE specified. Use 'local' or 'cluster'."
                    exit 1
                fi
            done
            
            # Wait for all background processes (folds) to finish before moving to the next parameter setting
            echo "Waiting for all folds to complete for parameter setting..."
            wait
            echo "All folds completed for parameter setting."
        done
    done
done
