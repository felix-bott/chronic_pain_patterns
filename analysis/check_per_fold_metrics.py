"""Recompute overall and per-fold metrics from CPP_results_light and compare them with the stored ones.

The predictions and observations in a combined table are sorted by subject ID. Position i therefore
belongs to the i-th subject, in sorted order, of those tested in the analysis, and the CV table
gives that subject's fold.
"""
import argparse
import glob
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from paths import ANALYSIS_DIRS, analysis_folds, fold_label

MAX_DIFFERENCE = 1e-9
MAIN_CV = "analysis_main/inputs/CV70-20-10Table_MegaStrat.csv"
CV_TABLES = {"main": MAIN_CV, "RAW": MAIN_CV, "CBP": "analysis_CBP-subgroup/inputs/CV_CBP.csv",
             "NP": "analysis_NP-subgroup/inputs/CV_NP.csv", "LOSO": "analysis_LOSO/inputs/CV_LOSO.csv"}


def metrics(observed, predicted):
    errors = predicted - observed
    return {"R": pearsonr(observed, predicted)[0], "R2": r2_score(observed, predicted),
            "MAE": np.mean(np.abs(errors)), "RMSE": np.sqrt(np.mean(errors ** 2))}


def test_folds(cv_path, folds):
    """Fold of each tested subject, sorted by subject ID."""
    cv = pd.read_csv(cv_path).melt(id_vars="participant_id", value_vars=folds, var_name="fold", value_name="split")
    tested = cv[cv["split"] == "test"].sort_values("participant_id")
    if tested["participant_id"].duplicated().any():
        sys.exit(f"{cv_path}: a subject is in the test set of more than one fold")
    return tested["fold"].to_numpy()


def largest_difference(row, subject_folds, folds):
    predicted = np.array(json.loads(row["predictions"]))
    observed = np.array(json.loads(row["observations"]))
    if len(predicted) != len(subject_folds):
        sys.exit(f"{row['source_path']}: {len(predicted)} values for {len(subject_folds)} subjects")
    differences = [abs(value - row[name]) for name, value in metrics(observed, predicted).items()]
    for fold in folds:
        selected = subject_folds == fold
        differences += [abs(value - row[f"{name}_{fold_label(fold)}"])
                        for name, value in metrics(observed[selected], predicted[selected]).items()]
    return max(differences)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--light_dir", required=True)
    light_dir = parser.parse_args().light_dir
    for analysis, folder in ANALYSIS_DIRS.items():
        folds = analysis_folds(analysis)
        subject_folds = test_folds(os.path.join(light_dir, CV_TABLES[analysis]), folds)
        largest = 0.0
        for path in glob.glob(os.path.join(light_dir, folder, "tables", "combined_*_tables.csv")):
            for _, row in pd.read_csv(path).iterrows():
                largest = max(largest, largest_difference(row, subject_folds, folds))
        print(f"{analysis}: largest metric difference {largest:.1e}")
        if largest > MAX_DIFFERENCE:
            sys.exit(f"{analysis}: recomputed metrics differ from the stored ones")
    print("Overall and per-fold metrics match")


if __name__ == "__main__":
    main()
