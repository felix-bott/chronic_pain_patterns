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
from paths import canonical_target, predictions_dir, summaries_dir
_parser = argparse.ArgumentParser()
_parser.add_argument("--target", required=True, choices=['age', 'pain'])
args = _parser.parse_args()
# --- end CPP adaptation ---


target = args.target

summary_df = pd.DataFrame(
    [
        {"tempConfig": 2},
        {"tempConfig": 20},
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

# Append target to each row
summary_df["target"] = target

# Append predictions and observations columns
summary_df["predictions"] = None
summary_df["observations"] = None
summary_df["predictions"] = summary_df["predictions"].astype(object)
summary_df["observations"] = summary_df["observations"].astype(object)

# Iterate over each configuration
for i_config, row in summary_df.iterrows():
    tempConfig = row["tempConfig"]
    target = row["target"]

    config = f"{target}_{tempConfig}"
    match config:
        case "pain_2":
            run_name = "epoched_autoPreproc_CUT2.0_CUT0.5_250"
        case "pain_20":
            run_name = "epoched_autoPreproc_20.0_0.5_250"
        case "age_2":
            run_name = "epoched_autoPreproc_CUT2.0_CUT0.5_250"
        case "age_20":
            run_name = "epoched_autoPreproc_20.0_0.5_250"
        case _:
            print("Invalid configuration")    

    results_dir = predictions_dir("main", "LaBraM", run_name, f"full_{canonical_target(target)}")
    output_csv_path = f'{summaries_dir("main", "LaBraM")}FT_full_{target}.csv' 
    os.makedirs(summaries_dir("main", "LaBraM"), exist_ok=True)


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
            log_path = f'{results_dir}/Fold{fold_idx}/{currSetting}/log.txt'
            log_df = load_log_to_dataframe(log_path)
            val_avg_all[currSetting].extend([log_df['val_avg'].max()])
    val_avg_all_df = pd.DataFrame(val_avg_all)
    # V1 Hyperparameter selection: individual hyper parameter setting based on max val_avg
    optim_settings = val_avg_all_df.idxmax(axis=1)
    # V2 Hyperparameter selection: single hyper parameter setting based on max val_avg - averaged across all folds

    print("number of hyper parameter settings evaluated:", val_avg_all_df.shape[1])

    # Combine results from Fold1 to Fold10
    dfs = []
    for fold_idx in range(1, 11):
        currSetting = optim_settings[fold_idx-1]  # Use the first setting for all folds
        csv_path = f'{results_dir}Fold{fold_idx}/{currSetting}/retrained/eval_prediction_test.csv'
        df = pd.read_csv(csv_path)
        df['Fold'] = fold_idx  # Optionally add a column to indicate the fold
        dfs.append(df)

        # Compute per-fold metrics (grouped by subject)
        df = df[['pred', 'true', 'subject']].copy()
        df["pred"] = df["pred"].apply(lambda x: ast.literal_eval(x)[0] if isinstance(x, str) else x)
        df["true"] = df["true"].apply(lambda x: ast.literal_eval(x)[0] if isinstance(x, str) else x)
        fold_avg = df.groupby('subject', as_index=False).mean()


        r_fold, _ = pearsonr(fold_avg['true'], fold_avg['pred'])
        r2_fold = r2_score(fold_avg['true'], fold_avg['pred'])
        fold_errors = fold_avg['pred'] - fold_avg['true']
        mae_fold = np.mean(np.abs(fold_errors))
        rmse_fold = np.sqrt(np.mean(fold_errors ** 2))

        summary_df.loc[i_config, f"R_Fold{fold_idx}"] = r_fold
        summary_df.loc[i_config, f"R2_Fold{fold_idx}"] = r2_fold
        summary_df.loc[i_config, f"MAE_Fold{fold_idx}"] = mae_fold
        summary_df.loc[i_config, f"RMSE_Fold{fold_idx}"] = rmse_fold

    # Concatenate all DataFrames into one
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df[['pred', 'true', 'subject']].copy()
    combined_df["pred"] = combined_df["pred"].apply(lambda x: ast.literal_eval(x)[0] if isinstance(x, str) else x)
    combined_df["true"] = combined_df["true"].apply(lambda x: ast.literal_eval(x)[0] if isinstance(x, str) else x)
    combined_df_avg = combined_df.groupby('subject', as_index=False).mean()

    pearson_corr, _ = pearsonr(combined_df_avg['true'], combined_df_avg['pred'])
    r2 = r2_score(combined_df_avg['true'], combined_df_avg['pred'])
    errors = combined_df_avg['pred'] - combined_df_avg['true']
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    # Store overall metrics in summary_df
    summary_df.loc[i_config, "R"] = pearson_corr
    summary_df.loc[i_config, "R2"] = r2
    summary_df.loc[i_config, "MAE"] = mae
    summary_df.loc[i_config, "RMSE"] = rmse

    summary_df.at[i_config, "predictions"] = json.dumps(combined_df_avg["pred"].tolist())
    summary_df.at[i_config, "observations"] = json.dumps(combined_df_avg["true"].tolist())

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
