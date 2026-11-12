print("TOP OF FILE EXECUTED, __name__ =", __name__)
import pandas as pd
import os
import numpy as np
from joblib import Parallel, delayed

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.compose import TransformedTargetRegressor

from eeg_common import load_epochs

from deep_learning_utils import create_model, create_estimator
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "shared"))
from targets import target_column

from skorch.helper import SliceDataset

from braindecode.datasets import WindowsDataset, BaseConcatDataset

import torch
from torch.utils.data import Subset

import argparse

import gc
import glob

print("Starting script 0...")

def get_args_parser():
    parser = argparse.ArgumentParser(description="Run EEG deep learning model.")
    
    # Argument parsing
    parser = argparse.ArgumentParser(description="Run EEG deep learning model.")
    parser.add_argument(
        "--model_type", type=str, choices=["shallow", "deep"],
        default="deep", help="Choose model type: 'shallow' or 'deep' (default: 'shallow')"
    )
    parser.add_argument(
        "--lr_factor", type=float, 
        default=1, help="factor to be multiplied with the default learning rate (default: 1)"
    )
    parser.add_argument(
        "--target", type=str, choices=["age", "ageRAW", "painVars", "painVarsRAW"], 
        default="age", help="target variable (default: 'age')"
    )
    parser.add_argument(
        "--batch_size", type=int, 
        default=64, help="Batch size for training (default: 128)"
    )
    parser.add_argument(
        "--n_epochs", type=int, 
        default=5, help="Number of epochs for training (default: 25)"
    )
    parser.add_argument(
        "--data_path", type=str, 
        required=True, help=""
    )
    parser.add_argument(
        "--CVtable_path", type=str, 
        required=True, help=""
    )
    parser.add_argument(
        "--NCtable_path", type=str, 
        required=True, help=""
    )
    parser.add_argument(
        "--output", type=str, 
        required=True, help=""
    )
    parser.add_argument(
        "--upsample", type=int, 
        default=2, help=""
    )
    parser.add_argument(
        "--folds", type=str, nargs="+", default=[f"Fold{i}" for i in range(1, 11)],
        help="CV-table columns to use as folds (default: Fold1 ... Fold10)"
    )
    return parser


def create_windows_ds_from_epochs_obj(
        epochs,
        rec_i,
        age,
        target_name=None,
        transform=None
):
    """Create a braindecode WindowsDataset from a preloaded mne.Epochs object.

    Parameters
    ----------
    epochs: mne.Epochs
        The preloaded epochs object.
    rec_i: int
        The absolute id of the recording.
    age: int
        The age of the subject of this recording.
    target_name: str | None
        The name of the target. If not None, has to be an entry in description.
    transform: callable
        A transform to be applied to the data on __getitem__.

    Returns
    -------
    braindecode.datasets.WindowsDataset
        A braindecode WindowsDataset.
    """
    description = {'rec': rec_i, 'age': age}
    target = -1
    if target_name is not None:
        assert target_name in description, (
            f"If 'target_name' is provided, it must be a key in the description. "
            f"Got {target_name} but available keys are: {list(description)}"
        )
        target = description[target_name]

    metadata = np.array([
        list(range(len(epochs))),           # i_window_in_trial
        len(epochs) * [-1],                 # i_start_in_trial
        len(epochs) * [-1],                 # i_stop_in_trial
        len(epochs) * [target],             # target
    ])
    metadata = pd.DataFrame(
        data=metadata.T,
        columns=['i_window_in_trial', 'i_start_in_trial', 'i_stop_in_trial', 'target'],
    )
    epochs.metadata = metadata
    epochs.metadata.reset_index(drop=True, inplace=True)

    ds = WindowsDataset(
        windows=epochs,
        description=description,
        targets_from='metadata',
        transform=transform,
    )
    return ds

class DataScaler(object):
    """On call multiply x with scaling_factor."""
    def __init__(self, scaling_factor):
        self.scaling_factor = scaling_factor

    def __call__(self, x):
        return x * self.scaling_factor

def create_dataset_from_epochs(
        epochs_list, ages, scaling_factor=1, n_jobs=1, debug=False):
    """
    Convert a list of mne.Epochs objects to a braindecode dataset and add ages as targets.
    """

    def process_epochs(epochs, rec_i, age):
        try:
            return create_windows_ds_from_epochs_obj(
                epochs=epochs, rec_i=rec_i, age=age, target_name='age',
                transform=DataScaler(scaling_factor=scaling_factor)
            )
        except Exception as e:
            print(f"Failed to process recording {rec_i}: {e}")
            return None

    datasets = Parallel(n_jobs=n_jobs)(
        delayed(process_epochs)(epochs, rec_i, age)
        for rec_i, (epochs, age) in enumerate(zip(epochs_list, ages))
    )

    valid_epochs, valid_datasets = [], []
    for ep, ds in zip(epochs_list, datasets):
        if ds is not None:
            valid_epochs.append(ep)
            valid_datasets.append(ds)

    print(f'Processed {len(valid_epochs)} out of {len(epochs_list)} mne.Epochs objects successfully.')

    return BaseConcatDataset(valid_datasets), valid_epochs

def lowpassfilter_epochs(epochs):
    """Apply a lowpass filter to the epochs."""
    # Apply a low-pass filter to the epochs
    epochs._data = np.array(epochs._data, copy=True)
    epochs.filter(l_freq=None, h_freq=49., picks='eeg')
    return epochs

def main(args):
    model_name = args.model_type
    lr_factor = args.lr_factor
    currTarget = args.target
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    cv_table_path = args.CVtable_path
    norm_table_path = args.NCtable_path
    save_path = args.output
    data_path = args.data_path
    upsample = args.upsample

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # load CV-table
    cv_table = pd.read_csv(cv_table_path)

    # load normalization table
    norm_table = pd.read_csv(norm_table_path)

    # specify output path and check if file already exists
    os.makedirs(save_path, exist_ok=True)

    # some in-function settings
    njobs = 8
    lowpass_on = False
    splitSegments = False

    subjects = cv_table['participant_id']
    sets = cv_table['set']

    y_pred_all = []
    y_test_all = []
    fold_data = []

    selected_folds = args.folds
    print("starting training")
    for curr_fold in selected_folds:
        print(f"\nProcessing {curr_fold}...")

        # skip fold if results already exist
        fold_file_pattern = os.path.join(save_path, f"{curr_fold}_avg-*.npz")
        matching_files = glob.glob(fold_file_pattern)
        if len(matching_files) > 0:
            fold_file = matching_files[0]   # take the first match
            print(f"Fold {curr_fold}: found cached results -> {fold_file}")
            cached = np.load(fold_file, allow_pickle=True)
            fold_results = cached["fold_results"].item()
            fold_data.append(fold_results)
            y_pred_all.extend(fold_results["y_pred"])
            y_test_all.extend(fold_results["y_test"])
            continue

        # laod epochs
        print("starting data loading")
        epochs_list = Parallel(n_jobs=njobs)(
            delayed(load_epochs)(data_path, norm_table, sub, curr_set, curr_fold, splitSegments, upsample=upsample) for sub, curr_set in zip(subjects, sets))
        epochs_list = [epoch for epoch in epochs_list if epoch is not None]
        subjects_sel = [subject for epoch, subject in zip(epochs_list, subjects) if epoch is not None]
        print("data loaded")

        if lowpass_on:
            # Apply lowpass filter to epochs (see paper)
            epochs_list = Parallel(n_jobs=njobs)(
                delayed(lowpassfilter_epochs)(epochs) for epochs in epochs_list) 


        # ----training data----
        subjects_tr = cv_table.query(f'{curr_fold} == "train"').participant_id
        subjects_tr = [sub for sub in subjects_tr if sub in subjects_sel]

        y_train = [
            cv_table.loc[cv_table['participant_id'] == sub, target_column(curr_fold, currTarget)].values[0]
            for sub in subjects_tr
        ]

        sub_to_idx = {sub: i for i, sub in enumerate(subjects_sel)}
        subjects_tr_inds = [sub_to_idx[sub] for sub in subjects_tr]
        epoch_list_tr = [epochs_list[i] for i in subjects_tr_inds]

        print("creating datasets from epochs")
        ds_train, _ = create_dataset_from_epochs(
            epochs_list=epoch_list_tr,
            ages=y_train,
            n_jobs=njobs,
            scaling_factor=5,
        )

        print("training data compiled")

        # ----validation data----
        subjects_val = cv_table.query(f'{curr_fold} == "val"').participant_id
        subjects_val = [sub for sub in subjects_val if sub in subjects_sel]

        y_val = [
            cv_table.loc[cv_table['participant_id'] == sub, target_column(curr_fold, currTarget)].values[0]
            for sub in subjects_val
        ]

        subjects_val_inds = [sub_to_idx[sub] for sub in subjects_val]
        epoch_list_val = [epochs_list[i] for i in subjects_val_inds]

        ds_val, _ = create_dataset_from_epochs(
            epochs_list=epoch_list_val,
            ages=y_val,
            n_jobs=1,
            scaling_factor=5,
        )

        print("validation data compiled")

        # ----combine train and val datasets for fixed split in skorch----
        ds_combined = BaseConcatDataset([ds_train, ds_val])
        X_train_sd = SliceDataset(ds_combined, idx=0, indices=None)
        y_train_sd = SliceDataset(ds_combined, idx=1, indices=None)

        # indices: train = [0 .. len_tr-1], val = [len_tr .. len_tr+len_val-1]
        len_tr = len(ds_train)
        len_val = len(ds_val)
        valid_indices = np.arange(len_tr, len_tr + len_val)


        # ----define fixed validation split----
        def make_fixed_split(valid_indices):
            valid_indices = np.array(valid_indices)

            def train_split(X, y=None, **fit_params):
                all_idx = np.arange(len(X))
                train_idx = np.setdiff1d(all_idx, valid_indices)
                return Subset(X, train_idx), Subset(X, valid_indices)

            return train_split


        # ----test data----
        subjects_test = cv_table.query(f'{curr_fold} == "test"').participant_id
        subjects_test = [sub for sub in subjects_test if sub in subjects_sel]
        y_test = [
            cv_table.loc[cv_table['participant_id'] == sub, target_column(curr_fold, currTarget)].values[0]
            for sub in subjects_test
        ]
        subject_to_index = {sub: i for i, sub in enumerate(subjects_sel)}
        subjects_test_inds = [subject_to_index[sub] for sub in subjects_test]
        epoch_list_test = [epochs_list[i] for i in subjects_test_inds]

        ds_test, valid_fnames = create_dataset_from_epochs(
            epochs_list=epoch_list_test,
            ages=y_test,
            n_jobs=njobs,
            scaling_factor=5
        )
        X_test_sd = SliceDataset(ds_test, idx=0, indices=None)
        y_test_sd = SliceDataset(ds_test, idx=1, indices=None)

        print("test data compiled")

        # ----define and fit model----
        cropped = True
        seed = 20211022

        x, y, _ = ds_train[0]    # load a single window to get number of eeg channels and time points for model creation
        n_channels, window_size = x.shape
        mdl, lr, weight_decay = create_model(
            model_name=model_name,
            window_size=window_size,
            n_channels=n_channels,
            cropped=cropped,
            seed=seed,
        )
        estimator = create_estimator(
            model=mdl,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr*lr_factor,
            weight_decay=weight_decay,
            n_jobs=njobs,
            train_split=make_fixed_split(valid_indices) #predefined_split(ds_valid)
        )
        
        # use a StandardScaler to scale targets to zero mean unit variance fold
        # by fold to facilitate model training. in estimator.predict, the inverse
        # transform is applied, such that we can compute scores based on unscaled
        # targets
        model = TransformedTargetRegressor(
            regressor=estimator,
            transformer=StandardScaler(),
        )

        # Fit the model
        model.fit(X_train_sd, y_train_sd)

        # determine epoch with highest avg-score (mean of R2 and PCC)
        avg_score = np.mean(np.stack([model.regressor_.history[:,'valid_R2'], model.regressor_.history[:,'valid_PCC']], axis=1), axis=1)
        best_avg_score = np.max(avg_score)
        best_epoch = np.argmax(avg_score)
        print(f"Best epoch: {best_epoch} with avg score: {avg_score[best_epoch]:.4f}")

        # reinstantiate model and train on train+val for best_epoch epochs
        mdlRE, lr, weight_decay = create_model(
            model_name=model_name,
            window_size=window_size,
            n_channels=n_channels,
            cropped=cropped,
            seed=seed,
        )
        estimatorRE = create_estimator(
            model=mdlRE,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr*lr_factor,
            weight_decay=weight_decay,
            n_jobs=njobs,
            max_epochs=best_epoch+1
        )

        modelRE = TransformedTargetRegressor(
            regressor=estimatorRE,
            transformer=StandardScaler(),
        )

        # Refit the model
        modelRE.fit(X_train_sd, y_train_sd)


        # ----predict and evaluate----
        df = X_test_sd.dataset.get_metadata()
        df['predictions'] = modelRE.predict(X_test_sd)
        df['observations'] = y_test_sd
        df = df.groupby('rec').mean()
        y_pred = df['predictions'].values

        # cast to same type
        y_test = np.array(y_test, dtype=np.float64)
        y_pred = np.array(y_pred, dtype=np.float64)

        # ----save results----
        # Save predictions and true values for the current fold
        fold_results = {
            "subjects": subjects_test,
            "sets": [cv_table.loc[cv_table['participant_id'] == sub, 'set'].values[0] for sub in subjects_test],
            "fold": curr_fold,
            "y_test": y_test,
            "y_pred": y_pred
        }
        fold_data.append(fold_results)
        
        # save results for this fold in a separate file
        fold_file = os.path.join(save_path, f"{curr_fold}_avg-{best_avg_score:.4f}.npz")
        np.savez(fold_file, fold_results=fold_results)
        print(f"Saved results for fold {curr_fold} → {fold_file}")   

        # accumulate for all folds
        y_pred_all.extend(y_pred)
        y_test_all.extend(y_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        if np.std(y_test) == 0 or np.std(y_pred) == 0:
            pcc = 0
        else:
            pcc = np.corrcoef(y_test, y_pred)[0, 1]
        print(f"{curr_fold} - R^2 score: {r2:.4f}")
        print(f"{curr_fold} - Mean Absolute Error: {mae:.4f}")
        print(f"{curr_fold} - Correlation: {pcc:,.4f}")

        # ---- end of fold cleanup ----
        del model, modelRE, estimator, estimatorRE, mdl, mdlRE
        del ds_train, ds_val, ds_test, ds_combined
        del X_train_sd, y_train_sd, X_test_sd, y_test_sd
        del epochs_list, epoch_list_tr, epoch_list_val, epoch_list_test

        gc.collect()
        torch.cuda.empty_cache()


    fold_results = {
        "subjects": [],
        "sets": [],
        "fold": "all_folds",
        "y_test": y_test_all,
        "y_pred": y_pred_all,
        "model": None,
        "modelRE": None
    }
        
    # Save the results for the current fold
    fold_data.append(fold_results)
    all_fold_file = os.path.join(save_path, f"all_fold_results.npz")
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
