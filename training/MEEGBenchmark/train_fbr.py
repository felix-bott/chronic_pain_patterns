import pandas as pd
import os
import numpy as np
import time
from joblib import Parallel, delayed
import h5io
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # CPP_code
from paths import resolve_features

from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error, mean_absolute_error

from scipy.stats import spearmanr

import coffeine # This apprears to work only with pyriemann v=0.7 and not with v=0.8, which is currently the latest

from eeg_common import load_epochs, residualize

import argparse

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "shared"))
from classical_models import train_model
from evaluation import pearson_corr, aggregate_by_ids
from targets import target_column

MAIN_FOLDS = [f'Fold{i}' for i in range(1, 11)]

def get_args_parser():
    # Argument parsing 
    parser = argparse.ArgumentParser(description="Run EEG deep learning model.")

    parser.add_argument(
        "--target", type=str, choices=["age", "ageRAW", "painVars", "painVarsRAW"], 
        default="age", help="target variable (default: 'age')"
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
        "--algorithm", type=str, 
        default="xgboost", help="ridge, xgboost"
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

def process_subject(signals_path, norm_table, curr_subject, curr_set, curr_fold, splitSegments, bench_cfg):
    epochs = load_epochs(signals_path, norm_table, curr_subject, curr_set, curr_fold, splitSegments)
    features = compute_fbr_features(epochs, bench_cfg, njobs=1) 
    return features


def compute_fbr_features(epochs, bench_cfg, njobs=1):
    features, meta_info = coffeine.compute_features(
        epochs, features=('covs',), n_fft=1024, n_overlap=512,
        fs=epochs.info['sfreq'], fmax=49, frequency_bands=bench_cfg['frequency_bands'])
    features['meta_info'] = meta_info
    return features


def get_data(features, cv_table, subjects, curr_fold, fold_type, currTarget, bench_cfg, currConfounder=None):
    subjects_train = cv_table.query(f'{curr_fold} == "{fold_type}"').participant_id
    subjects_train = [sub for sub in subjects_train if sub in subjects.tolist()]
    norm_ages_train = [
        cv_table.loc[cv_table['participant_id'] == sub, target_column(curr_fold, currTarget)].values[0]
        for sub in subjects_train
    ]
    X_train = [features[sub]['covs'] for sub in subjects_train]
    X_train = np.array(X_train)
    X_train_invisible = pd.DataFrame(
        {band: list(X_train[:, ii]) for ii, band in
            enumerate(bench_cfg['frequency_bands'])})
    y_train = norm_ages_train

    if currConfounder:
        norm_conf_train = [
            cv_table.loc[cv_table['participant_id'] == sub, target_column(curr_fold, currConfounder)].values[0]
            for sub in subjects_train
        ]
        z_train = norm_conf_train
    else:
        z_train = None

    return X_train_invisible, y_train, z_train, subjects_train


def main(args):
    cv_table_path = args.CVtable_path
    norm_table_path = args.NCtable_path
    save_path = args.output
    data_path = args.data_path
    currTarget = args.target
    currConfounder = args.confounder
    algorithm = args.algorithm
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
    njobs = 16
    splitSegments = False

    subjects = cv_table['participant_id']
    sets = cv_table['set']

    rank = "full"

    bench_cfg = { 
            'frequency_bands': {
                "delta": (1, 4),
                "theta": (4.0, 8.0),
                "alpha": (8.0, 15.0),
                "beta_low": (15.0, 26.0),
                "beta_mid": (26.0, 35.0),
                "beta_high": (35.0, 49)
                },
            'feature_map': 'fb_covs'}

    print("starting feature extraction")
    # filter-bank riemann features approach:
    selected_folds = args.folds
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
            delayed(process_subject)(data_path, norm_table, sub, curr_set, curr_fold, splitSegments, bench_cfg) for sub, curr_set in zip(subjects, sets))
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
    print("starting training")
    preds_path = f"{save_path}/predictions/{algorithm}_{currTarget}/"
    os.makedirs(preds_path, exist_ok=True)

    for curr_fold in selected_folds:
        print(f"\nProcessing {curr_fold}...")
        
        # get features and targets
        features = h5io.read_hdf5(features_file(curr_fold))

        X_train, y_train, z_train, sub_train = get_data(features, cv_table, subjects, curr_fold, 'train', currTarget, bench_cfg, currConfounder)
        X_val, y_val, z_val, sub_val = get_data(features, cv_table, subjects, curr_fold, 'val', currTarget, bench_cfg, currConfounder)

        X_train = pd.concat([X_train, X_val], axis=0, ignore_index=True)
        y_train = y_train + y_val
        sub_train = sub_train + sub_val
        if currConfounder:
            z_train = z_train + z_val

        X_test, y_test, z_test, sub_test = get_data(features, cv_table, subjects, curr_fold, 'test', currTarget, bench_cfg, currConfounder)

        # define transformation of covariance matrices
        filter_bank_transformer = coffeine.make_filter_bank_transformer(
            names=list(bench_cfg['frequency_bands']),
            method='riemann',
            projection_params=dict(scale='auto', n_compo=rank)
        )

        # transform covariance matrices
        X_train_feat = filter_bank_transformer.fit_transform(X_train, y_train)
        X_val_feat   = filter_bank_transformer.transform(X_val)
        X_test_feat  = filter_bank_transformer.transform(X_test)

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

        if algorithm == "ridge":
            model = make_pipeline(
                StandardScaler(),
                RidgeCV(alphas=np.logspace(-5, 10, 100))
                )

            # Fit the model
            t0 = time.time()
            model.fit(X_train_feat, y_train)
            t1 = time.time()
            print(f"Execution time for model fitting in fold {curr_fold}: {t1 - t0:.2f} seconds")

            # Predict and evaluate
            y_test_pred = model.predict(X_test_feat)
        elif algorithm == "xgboost":
            # Define scoring functions
            pearson_scorer = make_scorer(pearson_corr, greater_is_better=True)

            n_components = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 9999]
            model_type_all = ['xgboost']
            grid_params = {
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
        else:
            raise Exception(f"algorithm {algorithm} not implemented")

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


if __name__ == "__main__":
    print("Starting script...")
    parser = get_args_parser()

    # If VSCode runs it with no args → defaults are used automatically
    args = parser.parse_args()

    main(args)
