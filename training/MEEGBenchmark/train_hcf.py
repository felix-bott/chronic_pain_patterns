import pandas as pd
import os
import numpy as np

import h5io
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # CPP_code
from paths import resolve_features

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import  GridSearchCV
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error

import argparse


from joblib import Parallel, delayed
import time

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "shared"))
from classical_models import train_model
from evaluation import pearson_corr, aggregate_by_ids
from targets import target_column
from eeg_common import load_epochs, compute_features_subject, aggregate_features, residualize

MAIN_FOLDS = [f'Fold{i}' for i in range(1, 11)]

def get_args_parser():
    parser = argparse.ArgumentParser(description="Run EEG deep learning model.")
    
    # Argument parsing 
    parser = argparse.ArgumentParser(description="Run EEG deep learning model.")

    parser.add_argument(
        "--target", type=str, choices=["age", "ageRAW", "painVars", "painVarsRAW"], 
        default="ageRAW", help="target variable (default: 'age')"
    )
    parser.add_argument(
        "--confounder", type=str, choices=["age", "painVars"], 
        default=None, help=""
    )
    parser.add_argument(
        "--data_path", type=str, 
        required=True,
        help=""
    )
    parser.add_argument(
        "--CVtable_path", type=str, 
        required=True, help=""
    )
    parser.add_argument(
        "--NCtable_path", type=str, 
        required=True,
        help=""
    )
    parser.add_argument(
        "--output", type=str, 
        required=True, help=""
    )
    parser.add_argument(
        "--aggregate", type=str, 
        default="features", help="predictions or features"
    )
    parser.add_argument(
        "--algorithm", type=str, 
        default="linear", help="rf, xgboost, 'linear'"
    )
    parser.add_argument(
        "--folds", type=str, nargs="+", default=MAIN_FOLDS,
        help="CV-table columns to use as folds (default: Fold1 ... Fold10)"
    )
    parser.add_argument(
        "--features_dir", type=str, default=None,
        help="folder with cached features (default: --output). Missing features are computed into it"
    )
    parser.add_argument(
        "--features_fold", type=str, default=None,
        help="use the features of this norm-table column for every fold, e.g. LOSO, where the "
             "normalisation is the same for all folds (default: each fold's own features)"
    )
    return parser

def process_subject(signals_path, norm_table, curr_subject, curr_set, curr_fold, splitSegments):
    epochs = load_epochs(signals_path, norm_table, curr_subject, curr_set, curr_fold, splitSegments)
    features = compute_features_subject(epochs, njobs=1)  
    return features

def get_data(features, cv_table, subjects, curr_fold, fold_type, currTarget, currConfounder=None):
    subjects_train = cv_table.query(f'{curr_fold} == "{fold_type}"').participant_id
    subjects_train = [sub for sub in subjects_train if sub in subjects.tolist()]
    colName = target_column(curr_fold, currTarget)
    norm_ages_train = [
        cv_table.loc[cv_table['participant_id'] == sub, colName].values[0]
        for sub in subjects_train
    ]
    X_train = [features[sub] for sub in subjects_train]
    y_train = norm_ages_train

    if currConfounder:
        colNameConf = target_column(curr_fold, currConfounder)
        norm_conf_train = [
            cv_table.loc[cv_table['participant_id'] == sub, colNameConf].values[0]
            for sub in subjects_train
        ]
        z_train = norm_conf_train
    else:
        z_train = None

    return X_train, y_train, z_train, subjects_train

def main(args):
    cv_table_path = args.CVtable_path
    norm_table_path = args.NCtable_path
    save_path = args.output
    data_path = args.data_path
    currTarget = args.target
    aggregate = args.aggregate
    algorithm = args.algorithm
    currConfounder = args.confounder
    requested_features_dir = args.features_dir or save_path
    features_dir = resolve_features(requested_features_dir)

    def features_file(curr_fold):
        return f"{features_dir}/features_handcrafted_{args.features_fold or curr_fold}.h5"

    # load CV-table
    cv_table = pd.read_csv(cv_table_path)

    # load normalization table
    norm_table = pd.read_csv(norm_table_path)

    # specify output path and check if file already exists
    os.makedirs(save_path, exist_ok=True)

    # some in-function settings
    njobs = 24
    splitSegments = False

    subjects = cv_table['participant_id']
    sets = cv_table['set']


    selected_folds = args.folds

    print("starting feature extraction")
    os.makedirs(features_dir, exist_ok=True)
    feature_folds = [args.features_fold] if args.features_fold else selected_folds
    for curr_fold in feature_folds:
        file_path = features_file(curr_fold)

        if os.path.exists(file_path):
            print(f'Features for {curr_fold} already exist. Skipping feature extraction.')
            continue

        if features_dir != requested_features_dir:
            raise FileNotFoundError(f"{file_path} is missing; LOCATION.txt in {requested_features_dir} points there")

        t0 = time.time()
        features = Parallel(n_jobs=njobs)(
            delayed(process_subject)(data_path, norm_table, sub, curr_set, curr_fold, splitSegments) for sub, curr_set in zip(subjects, sets))
        t1 = time.time()
        print(f"Execution time for {curr_fold}: {t1 - t0:.2f} seconds")
        out = {sub: ff for sub, ff in zip(subjects, features)}

        # Save the features
        h5io.write_hdf5(
            file_path,
            out,
            overwrite=True
        )
        print(f'Features for {curr_fold} saved under {file_path}')


    y_pred_all = []
    y_test_all = []
    fold_data = []
    
    preds_path = f"{save_path}/predictions/{algorithm}_{currTarget}/"
    os.makedirs(preds_path, exist_ok=True)

    print(f"starting training with algo={algorithm}, target={currTarget}, confounder={currConfounder}")
    for curr_fold in selected_folds:
        print(f"\nProcessing {curr_fold}...")
        
        fold_file = os.path.join(preds_path, f"{curr_fold}.npz")
        if os.path.exists(fold_file):
            fold_results = np.load(fold_file, allow_pickle=True)['fold_results'].item()
            y_pred_all.append(fold_results['y_pred'])
            y_test_all.append(fold_results['y_test'])
            fold_data.append(fold_results)
            print(f'Predictions for {curr_fold} already exist. Skipping training.')
            continue

        # get features and targets
        features = h5io.read_hdf5(features_file(curr_fold))

        X_train, y_train, z_train, sub_train = get_data(features, cv_table, subjects, curr_fold, 'train', currTarget, currConfounder=currConfounder)

        X_val, y_val, z_val, sub_val = get_data(features, cv_table, subjects, curr_fold, 'val', currTarget, currConfounder=currConfounder)

        X_train = X_train + X_val
        y_train = y_train + y_val
        sub_train = sub_train + sub_val
        if currConfounder:
            z_train = z_train + z_val

        X_test, y_test, z_test, sub_test = get_data(features, cv_table, subjects, curr_fold, 'test', currTarget, currConfounder=currConfounder)


        # aggregate settings
        if aggregate == "predictions":
            agg_arg = 'separate'
        elif aggregate == "features":
            agg_arg = 'mean'

        # aggregate features across segments
        agg_fun = FunctionTransformer(aggregate_features, kw_args={'func': agg_arg})
        X_train_feat = agg_fun.fit_transform(X_train, y_train)
        X_val_feat   = agg_fun.transform(X_val)
        X_test_feat  = agg_fun.transform(X_test)


        if currConfounder:
            Z_train = np.column_stack([np.ones(len(z_train)), z_train])  # (n_train, 2)
            B, _, _, _ = np.linalg.lstsq(Z_train, X_train_feat, rcond=None)

            # Residualize features in train, val, test
            X_train_feat = residualize(X_train_feat, z_train, B)
            X_val_feat = residualize(X_val_feat, z_val, B)
            X_test_feat = residualize(X_test_feat, z_test, B)

            y_train_arr = np.asarray(y_train, dtype=float)
            beta_y, _, _, _ = np.linalg.lstsq(Z_train, y_train_arr, rcond=None)  # (2,)

            # Residualize targets in train, val, test
            y_train = residualize(y_train, z_train, beta_y)
            y_val   = residualize(y_val,   z_val,   beta_y)
            y_test  = residualize(y_test,  z_test,  beta_y)


        # aggregate settings
        if aggregate == "predictions":
            y_train_new = []
            sub_train_new = []
            for i in range(len(X_train)):
                y_train_new = y_train_new + [y_train[i]] * X_train[i].shape[0]
                sub_train_new = sub_train_new + [sub_train[i]] * X_train[i].shape[0]
            sub_train = sub_train_new
            y_train = y_train_new

            y_test_new = []
            sub_test_new = []
            for i in range(len(X_test)):
                y_test_new = y_test_new + [y_test[i]] * X_test[i].shape[0]
                sub_test_new = sub_test_new + [sub_test[i]] * X_test[i].shape[0]
            sub_test = sub_test_new
            y_test = y_test_new
                


        # define model
        if algorithm == "rf":
            param_grid = {'max_depth': [4, 6, 8, 16, 32, None],
                        'max_features': ['log2', 'sqrt']}
            model = GridSearchCV(
                RandomForestRegressor(n_estimators=1000,
                                    random_state=42),
                param_grid=param_grid,
                cv=5,
                n_jobs=njobs)

            # fit the model
            time_0 = time.time()
            model.fit(X_train_feat, y_train)
            time_1 = time.time()
            print(f"Training time for {curr_fold}: {time_1 - time_0:.2f} seconds")

            # Predict and evaluate
            y_test_pred = model.predict(X_test_feat)
            y_test = np.array(y_test).reshape(-1,1)

        elif algorithm == "xgboost" or algorithm == "linear":

            # Define scoring functions
            pearson_scorer = make_scorer(pearson_corr, greater_is_better=True)

            n_components = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 9999]
            model_type_all = [algorithm]
            grid_params = {
                'linear': {},  
                'xgboost': {
                    'max_depth': [3, 6],
                    'min_child_weight': [1],
                    'learning_rate': [0.03, 0.1, 0.3]
                },
            }
            numIter = 1000
            y_train = np.array(y_train).reshape(-1,1)
            y_test = np.array(y_test).reshape(-1,1)
            # Train model using nested cross-validation
            model, _, _ = train_model(X_train_feat, y_train, [], sub_train, n_components, model_type_all, grid_params, njobs, numIter, pearson_scorer)

            # perform confounder correction
            y_test = y_test - model.confounder_regressor_y.predict(z_test)

            # Evaluate on test set
            y_test_pred = model.predict(X_test_feat, z_test)
            
        test_ids_master = np.unique(sub_test)
        y_test_pred, _ = aggregate_by_ids(y_test_pred, np.array(sub_test), test_ids_master, 'mean')
        y_test, test_ids_aggregated = aggregate_by_ids(y_test, np.array(sub_test), test_ids_master, 'mean')
        sub_test = test_ids_aggregated

        y_test = y_test.squeeze().tolist()
        y_test_pred = y_test_pred.squeeze().tolist()


        # Save predictions and true values for the current fold
        fold_results = {
            "subjects": sub_test,
            "sets": [cv_table.loc[cv_table['participant_id'] == sub, 'set'].values[0] for sub in sub_test],
            "fold": curr_fold,
            "y_test": y_test,
            "y_pred": y_test_pred
        }
        fold_data.append(fold_results)
        
        # save results for this fold in a separate file
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


if __name__ == "__main__":
    print("Starting script...")
    parser = get_args_parser()

    # If VSCode runs it with no args → defaults are used automatically
    args = parser.parse_args()

    main(args)
