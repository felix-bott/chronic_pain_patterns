import numpy as np
import os
import sys
import pandas as pd
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import spearmanr
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "shared"))
from classical_models import train_model
from evaluation import pearson_corr
from targets import target_column
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # CPP_code
from paths import resolve_features
print('import section completed')




def get_args_parser():
    parser = argparse.ArgumentParser(description="Train the ChronicPainNetworks (CPN) models.")
    parser.add_argument("--target", type=str, required=True,
                        choices=["age", "ageRAW", "painVars", "painVarsRAW"],
                        help="target; pain targets use age as confounder")
    parser.add_argument("--algorithm", type=str, required=True, choices=["linear", "xgboost"])
    parser.add_argument("--CVtable_path", type=str,
                        required=True,
                        help="CV table; may hold a subset of the individuals in the feature table")
    parser.add_argument("--folds", type=str, nargs="+", default=[f"Fold{i}" for i in range(1, 11)],
                        help="CV-table columns to use as folds (default: Fold1 ... Fold10)")
    parser.add_argument("--features_dir", type=str, required=True,
                        help="folder with the CPN feature table")
    parser.add_argument("--output", type=str, required=True,
                        help="predictions go to <output>/predictions/<algorithm>_<target>/")
    return parser

args = get_args_parser().parse_args()

currTarget = args.target
currConfounder = "age" if args.target.startswith("painVars") else None  # harmonized age, also for raw pain
model_type_all = [args.algorithm]
n_components = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 9999]

dat_dir = resolve_features(args.features_dir)
ncores = 32

cv_table = pd.read_csv(args.CVtable_path)

# harmonization settings
standardize_within_studies = False # as in the main analysis, also for LOSO

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


# Load the CSV files into numpy arrays
def load_csv_as_numpy(file_path, return_ids=False, measure='V3connMatV10c3', freq_bands=['all']):
    data = pd.read_csv(file_path)
    data = data[data['subtype'] == measure]  # Filter by measure
    cols = data.columns[7:]
    freqs = [[freq for freq in ['theta', 'alpha', 'beta'] if freq in col] for col in cols]
    cols_nofreq = [col.replace(freq[0], '') for col, freq in zip(cols, freqs)]
    cols_split = [col.split('_') for col in cols_nofreq] 
    cols_conn = [True if conn[0] != conn[1] else False for conn in cols_split] # select only off-diagonal entries of the connectivity matrix
    selected_cols = [True] * 7 + cols_conn  # Keep the first 7 columns and the connection columns
    data = data.loc[:, selected_cols]  # Select the relevant columns

    if return_ids:
        return data.iloc[:, 7:].values, data.iloc[:, 2].values  # Return features and IDs separately
    else:
        return data.iloc[:, 7:].values


# Load features
X_tr_va_te, IDs_tr_va_te = load_csv_as_numpy(os.path.join(dat_dir, "connTable_NETAEC_Wide_AnalysisID_CPN4benchmark.csv"), return_ids=True)

# Keep the individuals in the CV table, in the order of the feature table
in_cv = np.isin(IDs_tr_va_te, cv_table.participant_id.values)
X_tr_va_te, IDs_tr_va_te = X_tr_va_te[in_cv], IDs_tr_va_te[in_cv]
if set(cv_table.participant_id) != set(IDs_tr_va_te):
    raise ValueError("Some individuals in the CV table have no CPN features.")
target_tab = cv_table.set_index('participant_id').loc[IDs_tr_va_te].reset_index()
if not np.array_equal(target_tab.participant_id.values, IDs_tr_va_te):
    raise ValueError("Ordering of participant_id in target_tab does not match IDs_tr_va_te.")

currModels = "_".join(model_type_all)

y_pred_all = []
y_test_all = []
fold_data = []
print("#############")
print("_".join([model_type_all[0], currTarget]))
print("#############")
for curr_fold in args.folds:
    # extract targets and confounders
    y_tr_va_te = target_tab[target_column(curr_fold, args.target)].values
    if currConfounder:
        z_tr_va_te = target_tab[target_column(curr_fold, currConfounder)].values
    mask_tr = target_tab[curr_fold] == 'train'
    maks_va = target_tab[curr_fold] == 'val'
    mask_te = target_tab[curr_fold] == 'test'

    if include_val_in_train:
        mask_tr = mask_tr | maks_va  # Combine train and validation sets element-wise

    preds_path = os.path.join(args.output, "predictions", f"{currModels}_{currTarget}")
    os.makedirs(preds_path, exist_ok=True)

    X_train = X_tr_va_te[mask_tr]
    X_test = X_tr_va_te[mask_te]
    
    IDs_tr = IDs_tr_va_te[mask_tr]
    IDs_test = IDs_tr_va_te[mask_te]

    y_train = y_tr_va_te[mask_tr].reshape(-1, 1)
    y_test = y_tr_va_te[mask_te].reshape(-1, 1)

    if standardize_within_studies:
        sets_train = target_tab.set[mask_tr].values
        sets_test = target_tab.set[mask_te].values
    else:
        sets_train = None
        sets_test = None

    if currConfounder:
        z_train = z_tr_va_te[mask_tr].reshape(-1, 1)
        z_test = z_tr_va_te[mask_te].reshape(-1, 1)
    else:
        z_train = []
        z_test = []
    
    # Define scoring functions
    pearson_scorer = make_scorer(pearson_corr, greater_is_better=True)
    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

    # Train model using nested cross-validation
    model, best_score, best_param = train_model(X_train, y_train, z_train, IDs_tr , n_components, model_type_all, grid_params, ncores, numIter, pearson_scorer, \
                                                   sets_train=sets_train)

    # perform confounder correction
    y_train = y_train - model.confounder_regressor_y.predict(z_train)
    y_test = y_test - model.confounder_regressor_y.predict(z_test)

    # Evaluate on training sets
    y_train_pred = model.predict(X_train, z_train, sets_train)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_corr = np.corrcoef(y_train.ravel(), y_train_pred.ravel())[0, 1]

    # Evaluate on test set
    y_test_pred = model.predict(X_test, z_test, sets_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_corr = np.corrcoef(y_test.ravel(), y_test_pred.ravel())[0, 1]

    # Save predictions and true values for the current fold
    y_test = y_test.squeeze()
    y_test_pred = y_test_pred.squeeze()
    fold_results = {
        "subjects": IDs_test,
        "sets": [cv_table.loc[cv_table['participant_id'] == sub, 'set'].values[0] for sub in IDs_test],
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
all_y_corr_spearman, pval = spearmanr(y_test_all, y_pred_all)
print(f"All folds - Spearman correlation (p): {all_y_corr_spearman} ({pval})")
print("=="*50)
print("=="*50)
print("=="*50)
