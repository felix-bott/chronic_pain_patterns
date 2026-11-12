"""Normalisation and split tables for all analyses.

The CV table holds, per individual, which split they are in for each fold ("train", "val",
"test") and their normalised targets (`<fold>_NORM<target>`). The norm tables hold, per raw
`set`, the EEG power factor for each fold. This module is the one place where these values are
computed:

- target_zscore(): targets are z-scored per raw set, using the mean and SD (ddof=1) of the
  individuals the normalisation is fitted on.
- power_factor(): signals of a raw set are multiplied by c = sqrt(e / geomean(TV)), where TV is
  an individual's total variance (summed over channels, averaged over segments, after the
  sqrt(exp(-9)) rescale in eeg_common.process_segments).

Main analysis: fitted on each fold's training individuals. Main and subgroup analyses read the
stored tables.

LOSO: fitted on all individuals of each raw set, so the values do not depend on which dataset is
held out. RS2020 and LONG are normalised separately, as in the main analysis, and held out
together as Set_Munich1.

Subgroups: rows of the stored CV table, values unchanged.

Usage: make_analysis_tables.py [--subgroups] [--loso] [--n_jobs N]   (no flag: all)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))              # CPP_code: paths, dataset_names
sys.path.insert(0, str(HERE.parent / "MEEGBenchmark"))  # eeg_common
from paths import TURBO_ROOT, inputs_dir, tables_dir, CV_MAIN_TABLE
from dataset_names import DATASET_SETS, alias_for
from eeg_common import load_epochs

FT = Path(TURBO_ROOT) / "data" / "fineTuning"
CV_TABLE = Path(CV_MAIN_TABLE)

MAIN_FOLDS = [f"Fold{i}" for i in range(1, 11)]
TARGETS = ["painVars", "age", "depressionVars"]

# Stored norm table -> signals it was fitted on. Sensor factors come from the 20 s epochs,
# although they are applied to the 2 s signals.
NORM_TABLE_SIGNALS = {
    "CV70-20-10Table_Sensor1020_all_MegaStrat_normFactorsNew.csv": "sensorSignals/epoched_autoPreproc_20.0_0.5_250",
    "CV70-20-10Table_Schaefer100_all_2.0_MegaStrat_normFactorsNew.csv": "sourceSignals/Schaefer100_freq-all_epochLength-2_overlap-0.5",
    "CV70-20-10Table_Schaefer100_all_20.0_MegaStrat_normFactorsNew.csv": "sourceSignals/Schaefer100_freq-all_epochLength-20_overlap-0.5",
    "CV70-20-10Table_OrthoAggSchaefer100_all_2.0_MegaStrat_normFactorsNew.csv": "sourceSignals/OrthoAggSchaefer100_freq-all_epochLength-2_overlap-0.5",
    "CV70-20-10Table_OrthoAggSchaefer100_all_20.0_MegaStrat_normFactorsNew.csv": "sourceSignals/OrthoAggSchaefer100_freq-all_epochLength-20_overlap-0.5",
}
# LOSO power tables: name -> stored norm table whose signals they use
LOSO_NORM_TABLES = {
    "Schaefer100_2": "CV70-20-10Table_Schaefer100_all_2.0_MegaStrat_normFactorsNew.csv",  # FBR
    "Sensor": "CV70-20-10Table_Sensor1020_all_MegaStrat_normFactorsNew.csv",              # FBS, HCF
}

SUBGROUPS = {
    "CBP": ["CBP", "CLBP"],
    "NP": ["NP", "PHN", "PNP"],
}

SEED = 42


# ---------------------------------------------------------------- normalisation

def target_zscore(values, sets, fit_mask):
    """z-score `values` per raw set with mean and SD (ddof=1) of the individuals in `fit_mask`."""
    values, sets, fit_mask = pd.Series(values), pd.Series(sets), pd.Series(fit_mask, dtype=bool)
    out = pd.Series(np.nan, index=values.index)
    for s in sets.unique():
        in_set = sets == s
        fit = values[in_set & fit_mask]
        out[in_set] = (values[in_set] - fit.mean()) / fit.std(ddof=1)
    return out.values


def power_factor(total_variance, sets, fit_mask):
    """Per raw set: c = sqrt(e / geometric mean of the total variance of the individuals in `fit_mask`)."""
    tv, sets, fit_mask = pd.Series(total_variance), pd.Series(sets), pd.Series(fit_mask, dtype=bool)
    return {s: float(np.sqrt(np.e / np.exp(np.log(tv[(sets == s) & fit_mask]).mean())))
            for s in sets.unique()}


def _subject_total_variance(signals_path, subject, raw_set):
    unit_norm = pd.DataFrame({"set": [raw_set], "unit": [1.0]})
    epochs = load_epochs(signals_path, unit_norm, subject, raw_set, "unit", splitSegments=False)
    return epochs.get_data().var(axis=2).sum(axis=1).mean()


def total_variance(signal_folder, cv_table, n_jobs):
    """Total variance per individual, in CV-table order."""
    signals_path = str(FT / signal_folder)
    tv = Parallel(n_jobs=n_jobs)(
        delayed(_subject_total_variance)(signals_path, sub, s)
        for sub, s in zip(cv_table.participant_id, cv_table["set"]))
    return np.asarray(tv)


# ---------------------------------------------------------------- subgroups

def write_subgroup_tables(cv_table):
    for name, diagnoses in SUBGROUPS.items():
        out = Path(inputs_dir(name))
        out.mkdir(parents=True, exist_ok=True)
        cv_table[cv_table["diagnosis"].isin(diagnoses)].to_csv(out / f"CV_{name}.csv", index=False)


def write_counts(cv_table):
    per_dataset, per_fold = [], []
    for name, diagnoses in SUBGROUPS.items():
        sub = cv_table[cv_table["diagnosis"].isin(diagnoses)].assign(dataset=lambda d: d["set"].map(alias_for))
        for (dataset, raw_set), n in sub.groupby(["dataset", "set"]).size().items():
            per_dataset.append({"subgroup": name, "dataset": dataset, "set": raw_set, "n": n})
        for fold in MAIN_FOLDS:
            for (split, dataset), n in sub.groupby([fold, "dataset"]).size().items():
                per_fold.append({"subgroup": name, "fold": fold, "split": split, "dataset": dataset, "n": n})
    per_dataset = pd.DataFrame(per_dataset)
    per_fold = pd.DataFrame(per_fold)
    # every dataset of a subgroup in every fold and split, including zero counts
    full = []
    for name in SUBGROUPS:
        datasets = sorted(per_dataset.loc[per_dataset.subgroup == name, "dataset"].unique())
        for fold in MAIN_FOLDS:
            for split in ["train", "val", "test"]:
                for dataset in datasets:
                    full.append({"subgroup": name, "fold": fold, "split": split, "dataset": dataset})
    per_fold = pd.DataFrame(full).merge(per_fold, how="left").fillna({"n": 0}).astype({"n": int})
    for name in SUBGROUPS:
        out = Path(tables_dir(name))
        out.mkdir(parents=True, exist_ok=True)
        per_dataset[per_dataset.subgroup == name].to_csv(out / "n_per_dataset.csv", index=False)
        per_fold[per_fold.subgroup == name].to_csv(out / "n_per_fold.csv", index=False)


# ---------------------------------------------------------------- LOSO

def loso_folds():
    """Fold name -> raw sets held out, one fold per dataset."""
    return {f"LOSO_{dataset}": list(raw_sets) for dataset, raw_sets in DATASET_SETS.items()}


def write_loso_tables(cv_table, n_jobs):
    """LOSO CV table (splits and targets) and LOSO norm tables (power factors)."""
    out = Path(inputs_dir("LOSO"))
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    is_val = cv_table[MAIN_FOLDS].eq("val").to_numpy().sum()
    not_test = cv_table[MAIN_FOLDS].ne("test").to_numpy().sum()
    val_share = is_val / not_test  # share of val among non-test individuals in the main CV table

    keep = ["participant_id", "set", "group", "age", "sex", "diagnosis", "painVars", "depressionVars",
            "category_painVars", "category_age"]
    loso = cv_table[keep].copy()
    everyone = np.ones(len(loso), dtype=bool)
    targets = {t: target_zscore(loso[t], loso["set"], everyone) for t in TARGETS}

    folds = loso_folds()
    for fold, held_out in folds.items():
        split = np.where(loso["set"].isin(held_out), "test", "train").astype(object)
        # val: the main table's share, drawn within each raw training set and pain category
        for (_, _), idx in loso[split == "train"].groupby(["set", "category_painVars"]).groups.items():
            n_val = int(round(val_share * len(idx)))
            split[loso.index.get_indexer(rng.choice(np.asarray(idx), n_val, replace=False))] = "val"
        loso[fold] = split
    for target in TARGETS:
        for fold in folds:
            loso[f"{fold}_NORM{target}"] = targets[target]
    loso.to_csv(out / "CV_LOSO.csv", index=False)

    for name, stored_table in LOSO_NORM_TABLES.items():
        tv = total_variance(NORM_TABLE_SIGNALS[stored_table], cv_table, n_jobs)
        c = power_factor(tv, cv_table["set"], everyone)
        norm = pd.DataFrame({"set": list(c)})
        for fold in ["LOSO"] + list(folds):  # "LOSO": the column used for the shared features
            norm[fold] = norm["set"].map(c)
        norm.to_csv(out / f"normFactors_LOSO_{name}.csv", index=False)
    return val_share


# ----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--subgroups", action="store_true")
    parser.add_argument("--loso", action="store_true")
    parser.add_argument("--n_jobs", type=int, default=16)
    args = parser.parse_args()
    run_all = not (args.subgroups or args.loso)
    cv_table = pd.read_csv(CV_TABLE)

    if args.subgroups or run_all:
        write_subgroup_tables(cv_table)
        write_counts(cv_table)
        print("Subgroup tables and counts written")
    if args.loso or run_all:
        share = write_loso_tables(cv_table, args.n_jobs)
        print(f"LOSO tables written (val share {share:.3f})")


if __name__ == "__main__":
    main()
