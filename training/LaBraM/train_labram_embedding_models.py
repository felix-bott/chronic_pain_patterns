import numpy as np
import os
import sys
import pandas as pd
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_absolute_error
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "shared"))
from classical_models import train_model
from model_settings import extend_settings
from evaluation import pearson_corr, aggregate_by_ids
from targets import target_column
import argparse
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # CPP_code
from paths import CV_MAIN_TABLE, canonical_target, features_dir, predictions_dir, resolve_features

parser = argparse.ArgumentParser(description="Train the final-layer models on LaBraM embeddings.")
parser.add_argument("--sigLen", type=int, nargs="+", default=[2, 20], help="epoch lengths in seconds (2, 20)")
parser.add_argument("--algorithm", default="xgboost", choices=["linear", "xgboost"])
parser.add_argument("--aggregation", default="mean", choices=["mean", "none"],
                    help="mean: average embeddings per individual; none: one row per segment")
parser.add_argument("--CVtable_path", default=CV_MAIN_TABLE)
parser.add_argument("--folds", nargs="+", default=[f"Fold{i}" for i in range(1, 11)],
                    help="folds; each needs embeddings exported by the LaBraM run of that fold")
args = parser.parse_args()
from datetime import datetime
print('import section completed')



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

addSettings = {
    'sigLen': args.sigLen,
    'model_type_all': [[args.algorithm]], #[['xgboost'], ['linear']],
    'aggregation_method': [args.aggregation], #['none'], # 'none' (when using none, the range for the alpha hyper-parameter should be adjusted)
    'n_components': [[0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 9999]]# [[16, 32, 64,128, 256, 512, 9999]] #[[int(round(1.3**n)) for n in range(3, 28)]]  # Sequence round 1.5^n
}
settings = extend_settings(settings, addSettings)

ncores = 24

cv_table_path = args.CVtable_path
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

# Load the CSV files into numpy arrays
def load_csv_as_numpy(file_path, return_ids=False):
    data = pd.read_csv(file_path)
    if return_ids:
        return data.iloc[:, 1:].values, data.iloc[:, 0].values  # Return features and IDs separately
    else:
        return data.iloc[:, 1:].values

settings_tested = []

for iSetting in range(len(settings)):
    currSetting = settings[iSetting]
    match currSetting["sigLen"]:
        case 2:
            run_name = "epoched_autoPreproc_CUT2.0_CUT0.5_250"
        case 20:
            run_name = "epoched_autoPreproc_20.0_0.5_250"
        case _:
            print("Invalid configuration")

    selected_target = currSetting["selected_target"]
    selected_confounder = currSetting["selected_confounder"]
    
    if selected_target == 0:
        currTarget = "pain"
    elif selected_target == 1:
        currTarget = "age"

    model_type_all = currSetting["model_type_all"]
    currModels = "_".join(model_type_all)

    y_pred_all = []
    y_test_all = []
    fold_data = []
    print("#############")
    print("_".join([ currSetting['model_type_all'][0], currSetting['aggregation_method'], currTarget, f"{currSetting['sigLen']}" ]))
    print("#############")
    for curr_fold in args.folds:
        
        embed_path = resolve_features(os.path.join(features_dir("main", "LaBraM", run_name), curr_fold))
        preds_path = predictions_dir("main", "LaBraM", run_name, f"{currModels}_{currSetting['aggregation_method']}_{canonical_target(currTarget)}")
        os.makedirs(preds_path, exist_ok=True)

        # Load training features
        X_train, train_ids = load_csv_as_numpy(os.path.join(embed_path, 'embeddings_train.csv'), return_ids=True)
        train_ids_master = np.unique(train_ids)
        X_train, train_ids = aggregate_by_ids(X_train, train_ids, train_ids_master, currSetting['aggregation_method'])

        # Load validation features
        X_val, val_ids = load_csv_as_numpy(os.path.join(embed_path, 'embeddings_val.csv'), return_ids=True)
        val_ids_master = np.unique(val_ids)
        X_val, val_ids = aggregate_by_ids(X_val, val_ids, val_ids_master, currSetting['aggregation_method']) 

        # Load test features
        X_test, test_ids = load_csv_as_numpy(os.path.join(embed_path, 'embeddings_test.csv'), return_ids=True)
        test_ids_master = np.unique(test_ids)
        X_test, test_ids = aggregate_by_ids(X_test, test_ids, test_ids_master, currSetting['aggregation_method']) 

        # Load targets and confounders     
        cols = [target_column(curr_fold, 'painVars'), target_column(curr_fold, 'age')]
        yz_train = cv_table.set_index('participant_id').reindex(train_ids)[cols].values
        yz_train, _ = aggregate_by_ids(yz_train, train_ids, train_ids_master, currSetting['aggregation_method'])
        yz_val = cv_table.set_index('participant_id').reindex(val_ids)[cols].values
        yz_val, _ = aggregate_by_ids(yz_val, val_ids, val_ids_master, currSetting['aggregation_method'])
        yz_test = cv_table.set_index('participant_id').reindex(test_ids)[cols].values
        yz_test, _ = aggregate_by_ids(yz_test, test_ids, test_ids_master, currSetting['aggregation_method'])

        if include_val_in_train:
            # Concatenate training and validation set
            X_train = np.vstack((X_train, X_val))
            yz_train = np.vstack((yz_train, yz_val))
            train_ids = np.hstack((train_ids, val_ids))
            train_ids_master = np.hstack((train_ids_master, val_ids_master))

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
        model, best_score, best_param = train_model(X_train, y_train, z_train, train_ids, currSetting['n_components'], currSetting['model_type_all'], grid_params, ncores, numIter, pearson_scorer)

        # perform confounder correction
        y_train = y_train - model.confounder_regressor_y.predict(z_train)
        y_val = y_val - model.confounder_regressor_y.predict(z_val)
        y_test = y_test - model.confounder_regressor_y.predict(z_test)

        # Evaluate on training sets
        y_train_pred = model.predict(X_train, z_train)
        train_ids_master = np.unique(train_ids)
        y_train_pred, _ = aggregate_by_ids(y_train_pred, train_ids, train_ids_master, 'mean')
        y_train, _ = aggregate_by_ids(y_train, train_ids, train_ids_master, 'mean')

        train_mse = mean_squared_error(y_train, y_train_pred)
        train_corr = np.corrcoef(y_train.ravel(), y_train_pred.ravel())[0, 1]

        # Evaluate on validation set
        y_val_pred = model.predict(X_val, z_val)
        val_ids_master = np.unique(val_ids)
        y_val_pred, _ = aggregate_by_ids(y_val_pred, val_ids, val_ids_master, 'mean')
        y_val, _ = aggregate_by_ids(y_val, val_ids, val_ids_master, 'mean')

        val_mse = mean_squared_error(y_val, y_val_pred)
        val_corr = np.corrcoef(y_val.ravel(), y_val_pred.ravel())[0, 1]

        # Evaluate on test set
        y_test_pred = model.predict(X_test, z_test)
        test_ids_master = np.unique(test_ids)
        y_test_pred, _ = aggregate_by_ids(y_test_pred, test_ids, test_ids_master, 'mean')
        y_test, test_ids_aggregated = aggregate_by_ids(y_test, test_ids, test_ids_master, 'mean')

        test_mse = mean_squared_error(y_test, y_test_pred)
        test_corr = np.corrcoef(y_test.ravel(), y_test_pred.ravel())[0, 1]

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
    print("=="*50)
    print("=="*50)
    print("=="*50)

    

    
