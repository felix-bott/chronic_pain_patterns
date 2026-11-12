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
from paths import CV_MAIN_TABLE, ANALYSES, CPN_RUN, analysis_folds, canonical_target, fold_label, predictions_dir, summaries_dir
_parser = argparse.ArgumentParser()
_parser.add_argument("--target", required=True, choices=['age', 'ageRAW', 'pain', 'painRAW'])
_parser.add_argument("--analysis", default="main", choices=ANALYSES,
                     help="main (default), or an additional analysis of the winners")
args = _parser.parse_args()
# --- end CPP adaptation ---

analysis = args.analysis
folds = analysis_folds(analysis)

model_type = "CPN"
target = args.target

summary_df = pd.DataFrame(
    [
        {"algo": "linear"},
        {"algo": "xgboost"},
    ]
) 

# Initialize overall metric columns
summary_df["R"] = np.nan
summary_df["R2"] = np.nan
summary_df["MAE"] = np.nan
summary_df["RMSE"] = np.nan

# Initialize per-fold metric columns
for fold in folds:
    summary_df[f"R_{fold_label(fold)}"] = np.nan
    summary_df[f"R2_{fold_label(fold)}"] = np.nan
    summary_df[f"MAE_{fold_label(fold)}"] = np.nan
    summary_df[f"RMSE_{fold_label(fold)}"] = np.nan

# Append target, mr, and Experiment_ID to each row
summary_df["target"] = target

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
    algo = row["algo"]
    target = row["target"]

    results_dir = predictions_dir(analysis, model_type, CPN_RUN, f"{algo}_{canonical_target(target)}")
    if analysis != "main" and not os.path.isdir(results_dir):
        continue  # additional analyses cover only the winners
    output_csv_path = f'{summaries_dir(analysis, model_type)}Direct_{target}.csv' 
    os.makedirs(summaries_dir(analysis, model_type), exist_ok=True)

    # Combine results from Fold1 to Fold10
    dfs = []
    for fold in folds:
        npz_path = f'{results_dir}{fold}.npz'
        with np.load(npz_path, allow_pickle=True) as data:
            fold_results = data['fold_results'].item()

        df = pd.DataFrame({
            'y_test': fold_results['y_test'],
            'y_pred': fold_results['y_pred'],
            'subjects': fold_results['subjects'],
        })
        df['Fold'] = fold  # Optionally add a column to indicate the fold
        dfs.append(df)

        # Compute per-fold metrics (grouped by subjects)
        df = df[['y_pred', 'y_test', 'subjects']].copy()
        df["y_pred"] = df["y_pred"].apply(lambda x: ast.literal_eval(x)[0] if isinstance(x, str) else x)
        df["y_test"] = df["y_test"].apply(lambda x: ast.literal_eval(x)[0] if isinstance(x, str) else x)
        # group by subjects, average y_pred/y_test and take the unique age value for each group
        fold_avg = df.groupby('subjects', as_index=False).agg({
            'y_pred': 'mean',
            'y_test': 'mean',
        })


        r_fold, _ = pearsonr(fold_avg['y_test'], fold_avg['y_pred'])
        r2_fold = r2_score(fold_avg['y_test'], fold_avg['y_pred'])
        fold_errors = fold_avg['y_pred'] - fold_avg['y_test']
        mae_fold = np.mean(np.abs(fold_errors))
        rmse_fold = np.sqrt(np.mean(fold_errors ** 2))

        summary_df.loc[i_config, f"R_{fold_label(fold)}"] = r_fold
        summary_df.loc[i_config, f"R2_{fold_label(fold)}"] = r2_fold
        summary_df.loc[i_config, f"MAE_{fold_label(fold)}"] = mae_fold
        summary_df.loc[i_config, f"RMSE_{fold_label(fold)}"] = rmse_fold

    # Concatenate all DataFrames into one
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df[['y_pred', 'y_test', 'subjects']].copy()
    combined_df["y_pred"] = combined_df["y_pred"].apply(lambda x: ast.literal_eval(x)[0] if isinstance(x, str) else x)
    combined_df["y_test"] = combined_df["y_test"].apply(lambda x: ast.literal_eval(x)[0] if isinstance(x, str) else x)
    combined_df_avg = combined_df.groupby('subjects', as_index=False).agg({
            'y_pred': 'mean',
            'y_test': 'mean'
        })


    pearson_corr, _ = pearsonr(combined_df_avg['y_test'], combined_df_avg['y_pred'])
    r2 = r2_score(combined_df_avg['y_test'], combined_df_avg['y_pred'])
    errors = combined_df_avg['y_pred'] - combined_df_avg['y_test']
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    # Store overall metrics in summary_df
    summary_df.loc[i_config, "R"] = pearson_corr
    summary_df.loc[i_config, "R2"] = r2
    summary_df.loc[i_config, "MAE"] = mae
    summary_df.loc[i_config, "RMSE"] = rmse

    summary_df.at[i_config, "predictions"] = json.dumps(combined_df_avg["y_pred"].tolist())
    summary_df.at[i_config, "observations"] = json.dumps(combined_df_avg["y_test"].tolist())

    # Print metrics for this configuration
    print(f"Algo: {algo}")
    print(f"Pearson correlation: {pearson_corr:.4f}")
    print(f"Coefficient of determination (R^2): {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print("-" * 50)


# Save summary_df to CSV
if analysis != "main":
    summary_df = summary_df.dropna(subset=["R"]).reset_index(drop=True)  # only the configs that were rerun
    if summary_df.empty:
        print(f"No {model_type} predictions for {analysis} / {args.target}")
        sys.exit(0)
    output_csv_path = f'{summaries_dir(analysis, model_type)}Direct_{args.target}.csv'
summary_df.to_csv(output_csv_path, index=False)
print(f"Summary saved to {output_csv_path}")
