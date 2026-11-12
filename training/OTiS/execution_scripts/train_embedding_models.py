print('here0')
import numpy as np
import os
import pandas as pd
print('here1')
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import spearmanr
print('here2')
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "shared"))
from classical_models import train_model
from model_settings import extend_settings
from evaluation import pearson_corr, aggregate_by_ids
import argparse
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))  # CPP_code
from paths import CV_MAIN_TABLE, canonical_target, features_dir, predictions_dir, resolve_features

parser = argparse.ArgumentParser(description="Train the final-layer models on OTiS embeddings (exported by S2 pretraining).")
parser.add_argument("--experiment_ID", default="2025GENERAL", choices=["2025", "2025GENERAL"],
                    help="2025: OTiS_EEG; 2025GENERAL: OTiS_GEN")
parser.add_argument("--algorithm", default=None, choices=["linear", "xgboost"],
                    help="default: the experiment's setting below")
parser.add_argument("--aggregation", default=None, choices=["mean", "none"],
                    help="default: the experiment's setting below")
parser.add_argument("--sigLen", type=int, nargs="+", default=None, help="epoch lengths (default: the experiment's setting)")
parser.add_argument("--atlas", nargs="+", default=None, choices=["Sensor", "Schaefer100", "Yeo7"],
                    help="spatial setups (default: all three)")
parser.add_argument("--folds", nargs="+", default=[f"Fold{i}" for i in range(1, 11)],
                    help="folds; each needs embeddings exported by the S2 run of that fold. Splits and "
                         "targets come from those exports")
args = parser.parse_args()
from datetime import datetime
print('import section completed')




experiment_ID = args.experiment_ID

settings = [
    {
        "selected_target": 1,
        "selected_confounder": None
    },
    {
        "selected_target": 0,
        "selected_confounder": 1 
    }
]

if experiment_ID == "2025GENERAL":
    addSettings = {
        'sigLen': [2],
        'mr': [0.75],
        'atlas': ['Sensor', 'Schaefer100', 'Yeo7'],
        'model_type_all': [['linear']], #, ['xgboost'], ['linear']],
        'aggregation_method': ['none'], #, 'mean', 'none'], # 'none' (when using none, the range for the alpha hyper-parameter should be adjusted)
        'n_components': [[0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 9999]]# [[16, 32, 64,128, 256, 512, 9999]] #[[int(round(1.3**n)) for n in range(3, 28)]]  # Sequence round 1.5^n
    }
elif experiment_ID == "2025":
    addSettings = {
        'sigLen': [2, 20],
        'mr': [0.75],
        'atlas': ['Sensor', 'Schaefer100', 'Yeo7'],
        'model_type_all': [['xgboost']], #, ['xgboost'], ['linear']],
        'aggregation_method': ['mean'], #, 'mean', 'none'], # 'none' (when using none, the range for the alpha hyper-parameter should be adjusted)
        'n_components': [[0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 9999]]# [[16, 32, 64,128, 256, 512, 9999]] #[[int(round(1.3**n)) for n in range(3, 28)]]  # Sequence round 1.5^n
    }

for key, value in [('model_type_all', [[args.algorithm]] if args.algorithm else None),
                   ('aggregation_method', [args.aggregation] if args.aggregation else None),
                   ('sigLen', args.sigLen), ('atlas', args.atlas)]:
    if value is not None:
        addSettings[key] = value
settings = extend_settings(settings, addSettings)

all_freqs = ['all']

ncores = 32

cv_table_path = CV_MAIN_TABLE
cv_table = pd.read_csv(cv_table_path)

# Define parameter grids for each model type
grid_params = {
    'linear': {},
    'xgboost': {
        'max_depth': [3, 6],
        'min_child_weight': [1],
        'learning_rate': [0.03, 0.1, 0.3]
    },
}
include_val_in_train = True
numIter = 1000

# Get the current date and time as a string
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Current date and time: {current_datetime}")

def load_csv_as_numpy(file_path, return_ids=False):
    data = pd.read_csv(file_path)

    if return_ids:
        return data.iloc[:, 1:].values, data.iloc[:, 0].values  # Return features and IDs separately
    else:
        return data.iloc[:, 1:].values

settings_tested = []

print("starting setting iterations")
for iSetting in range(len(settings)):

    currSetting = settings[iSetting]
    mr = currSetting["mr"]
    atlas = currSetting["atlas"]
    sigLen = currSetting["sigLen"]
    currConfig = f"{atlas}_{sigLen}"
    match currConfig:
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

    model_folder = {"2025": "OTiS_EEG", "2025GENERAL": "OTiS_GEN"}[experiment_ID]

    model_type_all = currSetting["model_type_all"]
    n_components = currSetting["n_components"]
    aggregation_method = currSetting["aggregation_method"]
    selected_target = currSetting["selected_target"]
    selected_confounder = currSetting["selected_confounder"]

    if selected_target == 0:
        currTarget = "pain"
    elif selected_target == 1:
        currTarget = "age"

    currModels = "_".join(model_type_all)

    y_pred_all = []
    y_test_all = []
    fold_data = []
    print("#############")
    print("_".join([model_type_all[0], aggregation_method, currTarget, f"{mr}", currConfig]))
    print("#############")
    print("starting fold iterations")
    
    for curr_fold in args.folds:

        embed_path = resolve_features(f"{features_dir('main', model_folder, run_name)}{curr_fold}") + "/"
        preds_path = predictions_dir("main", model_folder, run_name, f"{currModels}_{aggregation_method}_{canonical_target(currTarget)}")
        os.makedirs(preds_path, exist_ok=True)


        X_train = []
        X_val = []
        X_test = []
        for i_freq, freq in enumerate(all_freqs):
            
            # Load training data
            X_train_f, train_ids = load_csv_as_numpy(embed_path+f'embeddings_{freq}_train.csv', return_ids=True)
            if(i_freq == 0):
                train_ids_master = np.unique(train_ids)
                yz_train = load_csv_as_numpy(embed_path+f'targets_{freq}_train.csv')
                yz_train, train_ids_reordered = aggregate_by_ids(yz_train, train_ids, train_ids_master, aggregation_method)
            X_train_f, _ = aggregate_by_ids(X_train_f, train_ids, train_ids_master, aggregation_method)

            # Load validation data
            X_val_f, val_ids = load_csv_as_numpy(embed_path+f'embeddings_{freq}_val.csv', return_ids=True)
            if(i_freq == 0):
                val_ids_master = np.unique(val_ids)
                yz_val = load_csv_as_numpy(embed_path+f'targets_{freq}_val.csv')
                yz_val, val_ids_reordered = aggregate_by_ids(yz_val, val_ids, val_ids_master, aggregation_method)
            X_val_f, _ = aggregate_by_ids(X_val_f, val_ids, val_ids_master, aggregation_method)

            # Load test data
            X_test_f, test_ids = load_csv_as_numpy(embed_path+f'embeddings_{freq}_test.csv', return_ids=True)
            if(i_freq == 0):
                test_ids_master = np.unique(test_ids)
                yz_test = load_csv_as_numpy(embed_path+f'targets_{freq}_test.csv')
                yz_test, test_ids_reordered = aggregate_by_ids(yz_test, test_ids, test_ids_master, aggregation_method)
            X_test_f, _ = aggregate_by_ids(X_test_f, test_ids, test_ids_master, aggregation_method)

            if(i_freq == 0):
                X_train = X_train_f
                X_val = X_val_f
                X_test = X_test_f
            else:
                X_train = np.hstack((X_train, X_train_f))
                X_val = np.hstack((X_val, X_val_f))
                X_test = np.hstack((X_test, X_test_f))

        if include_val_in_train:
            # Concatenate training and validation set
            X_train = np.vstack((X_train, X_val))
            yz_train = np.vstack((yz_train, yz_val))
            train_ids_reordered = np.hstack((train_ids_reordered, val_ids_reordered))


        y_train = yz_train[:,selected_target].reshape(-1, 1)
        y_val = yz_val[:,selected_target].reshape(-1, 1)
        y_test = yz_test[:,selected_target].reshape(-1, 1)

        if selected_confounder: 
            z_train = yz_train[:,selected_confounder].reshape(-1, 1)
            z_val = yz_val[:,selected_confounder].reshape(-1, 1)
            z_test = yz_test[:,selected_confounder].reshape(-1, 1)
        else:
            z_train = []
            z_val = []
            z_test = []
        
        # Define scoring functions
        pearson_scorer = make_scorer(pearson_corr, greater_is_better=True)
        mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

        # Train model using nested cross-validation
        print("starting model training")
        model, best_score, best_param = train_model(X_train, y_train, z_train, train_ids_reordered, n_components, model_type_all, grid_params, ncores, numIter, pearson_scorer)

        # perform confounder correction
        y_train = y_train - model.confounder_regressor_y.predict(z_train)
        y_val = y_val - model.confounder_regressor_y.predict(z_val)
        y_test = y_test - model.confounder_regressor_y.predict(z_test)

        # Evaluate on training sets
        y_train_pred = model.predict(X_train, z_train)
        train_ids_master = np.unique(train_ids_reordered)
        y_train_pred, _ = aggregate_by_ids(y_train_pred, train_ids_reordered, train_ids_master, 'mean')
        y_train, _ = aggregate_by_ids(y_train, train_ids_reordered, train_ids_master, 'mean')

        train_mse = mean_squared_error(y_train, y_train_pred)
        train_corr = np.corrcoef(y_train.squeeze(), y_train_pred.squeeze())[0, 1]

        # Evaluate on validation set
        y_val_pred = model.predict(X_val, z_val)
        val_ids_master = np.unique(val_ids_reordered)
        y_val_pred, _ = aggregate_by_ids(y_val_pred, val_ids_reordered, val_ids_master, 'mean')
        y_val, _ = aggregate_by_ids(y_val, val_ids_reordered, val_ids_master, 'mean')

        val_mse = mean_squared_error(y_val, y_val_pred)
        val_corr = np.corrcoef(y_val.squeeze(), y_val_pred.squeeze())[0, 1]

        # Evaluate on test set
        y_test_pred = model.predict(X_test, z_test)
        test_ids_master = np.unique(test_ids_reordered)
        y_test_pred, _ = aggregate_by_ids(y_test_pred, test_ids_reordered, test_ids_master, 'mean')
        y_test, test_ids_aggregated = aggregate_by_ids(y_test, test_ids_reordered, test_ids_master, 'mean')

        test_mse = mean_squared_error(y_test, y_test_pred)
        test_corr = np.corrcoef(y_test.squeeze(), y_test_pred.squeeze())[0, 1]

        # Save predictions and true values for the current fold
        y_test = y_test.squeeze()
        y_test_pred = y_test_pred.squeeze()
        fold_results = {
            "subjects": test_ids_aggregated,
            "sets": [cv_table.loc[cv_table['participant_id'] == sub, 'set'].values[0] for sub in test_ids_aggregated],
            "fold": curr_fold,
            "y_test": y_test,
            "y_pred": y_test_pred
        }
        fold_data.append(fold_results)
        
        # save results for this fold in a separate file
        fold_file = os.path.join(preds_path, f"{curr_fold}.npz")
        np.savez(fold_file, fold_results=fold_results)
        print(f"Saved results for fold {curr_fold} → {fold_file}")   

        # accumulate for all folds
        y_pred_all.extend(y_test_pred)
        y_test_all.extend(y_test)
        r2 = r2_score(y_test, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)
        if np.std(y_test) == 0 or np.std(y_test_pred) == 0:
            pcc = 0
        else:
            pcc = np.corrcoef(y_test, y_test_pred)[0, 1]
        print(f"{curr_fold} - R^2 score: {r2:.4f}")
        print(f"{curr_fold} - Mean Absolute Error: {mae:.4f}")
        print(f"{curr_fold} - Correlation: {pcc:,.4f}")
        print("\n")

        del model

    fold_results = {
        "subjects": [],
        "sets": [],
        "fold": "all_folds",
        "y_test": y_test_all,
        "y_pred": y_pred_all
    }
    
    # Save the results for the current fold
    fold_data.append(fold_results)
    all_fold_file = os.path.join(preds_path, f"all_fold_results.npz")
    np.savez(all_fold_file, results=fold_data)
    print(f"Results across all folds saved under {all_fold_file}.")


    print(f"\nAll folds - R^2 score: {r2_score(y_test_all, y_pred_all):.4f}")
    print(f"All folds - Mean Absolute Error: {mean_absolute_error(y_test_all, y_pred_all):.4f}")
    print(f"All folds - Correlation: {np.corrcoef(y_test_all, y_pred_all)[0, 1]:.4f}")
    rho, pval = spearmanr(y_test_all, y_pred_all)
    print(f"All folds - Rank correlation (Spearman ρ): {rho:.4f}")
    print("=="*50)
    print("=="*50)
    print("=="*50)
    
