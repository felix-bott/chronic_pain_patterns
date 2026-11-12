import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import os
import json
import ast 
import sys


# --- CPP adaptation: paths and settings come from paths.py and the command line ---
import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from paths import CV_MAIN_TABLE, canonical_target, predictions_dir, summaries_dir
_parser = argparse.ArgumentParser()
_parser.add_argument("--experiment_ID", required=True, choices=['2025', '2025GENERAL'])
_parser.add_argument("--target", required=True, choices=['age', 'pain'])
args = _parser.parse_args()
# --- end CPP adaptation ---


val_metric = "test_avg" # Note: In OTiS code, "test" is used to describe the inner validation set, while "actual_test" is used for the held-out test set.

experiment_ID = args.experiment_ID
OUTPUT_FOLDER = {"2025": "OTiS_EEG", "2025GENERAL": "OTiS_GEN"}
target = args.target
mr = 0.75

if experiment_ID == "2025":
    summary_df = pd.DataFrame(
        [
            {"spatConfig": "Sensor",        "tempConfig": 2},
            {"spatConfig": "Sensor",        "tempConfig": 20}, # This does not work with 2025GENERAL due to memory constraints
            {"spatConfig": "Schaefer100",   "tempConfig": 2},
            {"spatConfig": "Schaefer100",   "tempConfig": 20},  # This does not work with 2025GENERAL due to memory constraints
            {"spatConfig": "Yeo7",          "tempConfig": 2}, 
            {"spatConfig": "Yeo7",          "tempConfig": 20},
        ]
    )
elif experiment_ID == "2025GENERAL":
    summary_df = pd.DataFrame(
        [
            {"spatConfig": "Sensor",        "tempConfig": 2},
            {"spatConfig": "Schaefer100",   "tempConfig": 2},
            {"spatConfig": "Yeo7",          "tempConfig": 2}, 
        ]
    )

# Initialize overall metric columns
summary_df["R"] = np.nan
summary_df["R2"] = np.nan
summary_df["MAE"] = np.nan
summary_df["RMSE"] = np.nan

# Initialize per-fold metric columns
for fold_idx in range(1, 11):
    summary_df[f"R_Fold{fold_idx}"] = np.nan
    summary_df[f"R2_Fold{fold_idx}"] = np.nan
    summary_df[f"MAE_Fold{fold_idx}"] = np.nan
    summary_df[f"RMSE_Fold{fold_idx}"] = np.nan

# Append target, mr, and Experiment_ID to each row
summary_df["target"] = target
summary_df["mr"] = mr
summary_df["Experiment_ID"] = experiment_ID

# Append predictions and observations columns
summary_df["predictions"] = None
summary_df["observations"] = None
summary_df["predictions"] = summary_df["predictions"].astype(object)
summary_df["observations"] = summary_df["observations"].astype(object)

# CV-table path
cv_table_path = CV_MAIN_TABLE
cv_table = pd.read_csv(cv_table_path)
cv_table = cv_table.set_index("participant_id")

# Iterate over each configuration
for i_config, row in summary_df.iterrows():
    spatConfig = row["spatConfig"]
    tempConfig = row["tempConfig"]
    mr = row["mr"]
    target = row["target"]
    experiment_ID = row["Experiment_ID"]

    config = f"{spatConfig}_{tempConfig}"
    match config:
        case "Sensor_2":
            run_name = "epoched_autoPreproc_CUT2.0_CUT0.5_250"
        case "Sensor_20":
            run_name = "epoched_autoPreproc_20.0_0.5_250"
        case "Schaefer100_2":
            run_name = "Schaefer100_freq-all_epochLength-2_overlap-0.5"
        case "Schaefer100_20":
            run_name = "Schaefer100_freq-all_epochLength-20_overlap-0.5"
        case "Yeo7_2":
            run_name = "OrthoAggSchaefer100_freq-all_epochLength-2_overlap-0.5"
        case "Yeo7_20":
            run_name = "OrthoAggSchaefer100_freq-all_epochLength-20_overlap-0.5"
        case _:
            print("Invalid configuration")    

    results_dir = predictions_dir("main", OUTPUT_FOLDER[experiment_ID], run_name, f"full_{canonical_target(target)}")
    output_csv_path = f'{summaries_dir("main", OUTPUT_FOLDER[experiment_ID])}FT_full_{target}_{mr}.csv' 
    os.makedirs(summaries_dir("main", OUTPUT_FOLDER[experiment_ID]), exist_ok=True)


    def load_log_to_dataframe(log_path):
        records = []
        with open(log_path, 'r') as f:
            for line in f:
                # Replace non-standard JSON values
                line = line.replace('NaN', 'null').replace('Infinity', 'null')
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError:
                    continue
        return pd.DataFrame(records)


    folderNames = os.listdir(f'{results_dir}Fold1')
    val_avg_all = {key: [] for key in folderNames}
    for fold_idx in range(1, 11):
        for iSetting in range(1, len(folderNames) + 1):
            currSetting = folderNames[iSetting - 1]
            log_path = f'{results_dir}Fold{fold_idx}/{currSetting}/log.txt'
            log_df = load_log_to_dataframe(log_path)
            val_avg_all[currSetting].extend([log_df[val_metric].max()])
    val_avg_all_df = pd.DataFrame(val_avg_all)
    # V1 Hyperparameter selection: individual hyper parameter setting based on max val_avg
    optim_settings = val_avg_all_df.idxmax(axis=1)
    # V2 Hyperparameter selection: single hyper parameter setting based on max val_avg - averaged across all folds


    print(val_avg_all_df.shape[1], "different hyper-parameter settings found.")

    # Combine results from Fold1 to Fold10
    dfs = []
    for fold_idx in range(1, 11):
        currSetting = optim_settings[fold_idx-1]  # Use the first setting for all folds
        csv_path = f'{results_dir}Fold{fold_idx}/{currSetting}/retrained/logits/logitsTEST_targets_freqs_MILgroup.csv'
        df = pd.read_csv(csv_path)
        df['Fold'] = fold_idx  # Optionally add a column to indicate the fold
        dfs.append(df)

        # Compute per-fold metrics (grouped by MIL_Group)
        df = df[['Logits', 'Targets', 'MIL_Group']].copy()
        df["Logits"] = df["Logits"].apply(lambda x: ast.literal_eval(x)[0] if isinstance(x, str) else x)
        df["Targets"] = df["Targets"].apply(lambda x: ast.literal_eval(x)[0] if isinstance(x, str) else x)
        # group by MIL_Group, average logits/targets and take the unique age value for each group
        fold_avg = df.groupby('MIL_Group', as_index=False).agg({
            'Logits': 'mean',
            'Targets': 'mean',
        })


        r_fold, _ = pearsonr(fold_avg['Targets'], fold_avg['Logits'])
        r2_fold = r2_score(fold_avg['Targets'], fold_avg['Logits'])
        fold_errors = fold_avg['Logits'] - fold_avg['Targets']
        mae_fold = np.mean(np.abs(fold_errors))
        rmse_fold = np.sqrt(np.mean(fold_errors ** 2))

        summary_df.loc[i_config, f"R_Fold{fold_idx}"] = r_fold
        summary_df.loc[i_config, f"R2_Fold{fold_idx}"] = r2_fold
        summary_df.loc[i_config, f"MAE_Fold{fold_idx}"] = mae_fold
        summary_df.loc[i_config, f"RMSE_Fold{fold_idx}"] = rmse_fold

    # Concatenate all DataFrames into one
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df[['Logits', 'Targets', 'MIL_Group']].copy()
    combined_df["Logits"] = combined_df["Logits"].apply(lambda x: ast.literal_eval(x)[0] if isinstance(x, str) else x)
    combined_df["Targets"] = combined_df["Targets"].apply(lambda x: ast.literal_eval(x)[0] if isinstance(x, str) else x)
    combined_df_avg = combined_df.groupby('MIL_Group', as_index=False).agg({
            'Logits': 'mean',
            'Targets': 'mean'
        })


    pearson_corr, _ = pearsonr(combined_df_avg['Targets'], combined_df_avg['Logits'])
    r2 = r2_score(combined_df_avg['Targets'], combined_df_avg['Logits'])
    errors = combined_df_avg['Logits'] - combined_df_avg['Targets']
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    # Store overall metrics in summary_df
    summary_df.loc[i_config, "R"] = pearson_corr
    summary_df.loc[i_config, "R2"] = r2
    summary_df.loc[i_config, "MAE"] = mae
    summary_df.loc[i_config, "RMSE"] = rmse

    summary_df.at[i_config, "predictions"] = json.dumps(combined_df_avg["Logits"].tolist())
    summary_df.at[i_config, "observations"] = json.dumps(combined_df_avg["Targets"].tolist())

    # Print metrics for this configuration
    print(f"Configuration: {config}")
    print(f"Pearson correlation: {pearson_corr:.4f}")
    print(f"Coefficient of determination (R^2): {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print("-" * 50)


# Save summary_df to CSV
summary_df.to_csv(output_csv_path, index=False)
print(f"Summary saved to {output_csv_path}")
