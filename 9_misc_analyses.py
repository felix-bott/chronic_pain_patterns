"""Association of age with harmonized global signal power and with pain intensity (no training).

Addresses the question whether age-related variability, which the feature-based models remove by
residualisation, could drive the deep learning models, which receive the harmonized signals and
targets without residualisation. If harmonized global power and harmonized pain are only weakly
associated with harmonized age, age offers little shortcut to either.

Global signal power: an individual's total variance TV (summed over channels, averaged over
segments), computed exactly as for the power factors (make_analysis_tables.total_variance), on
the signals each stored norm table was fitted on. Harmonized power in fold f is c_{set,f}^2 * TV,
with c the stored factor of that fold; it is analysed on the log scale.

Associations (Pearson r with 95% CI, Spearman rho, R^2 = r^2):
  power_vs_age  harmonized: log harmonized power vs Fold<k>_NORMage, all individuals, per fold
                raw:        log TV vs age in years, pooled over datasets (before harmonization)
                per dataset: log TV vs age in years within each dataset (harmonization only
                            rescales a raw set, so within-set r equals that of harmonized values;
                            Set_Munich1 pools RS2020 and LONG, which are scaled separately, so
                            its rows use harmonized power averaged over folds)
  pain_vs_age   harmonized: Fold<k>_NORMpainVars vs Fold<k>_NORMage, per fold
                raw:        pain rating vs age in years, pooled
                per dataset: pain rating vs age in years within each dataset

Outputs in CPP_results/analysis_main/tables/age_associations/:
  total_variance_<signals>.csv   cached TV per individual (delete to recompute)
  age_associations_per_fold.csv  harmonized associations, one row per fold
  age_associations_summary.csv   harmonized (median and range over folds), raw and per dataset

Usage: 9_misc_analyses.py [--n_jobs N]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "training" / "shared"))
from paths import CV_MAIN_TABLE, MAIN_FOLDS, inputs_dir, tables_dir
from dataset_names import alias_for
from make_analysis_tables import NORM_TABLE_SIGNALS, total_variance

OUT = Path(tables_dir("main")) / "age_associations"


# ---------------------------------------------------------------- statistics

def correlate(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    r, p = stats.pearsonr(x, y)
    half = 1.959964 / np.sqrt(n - 3)  # Fisher z, 95%
    rho, p_rho = stats.spearmanr(x, y)
    return {"n": n, "r": r, "r_ci_low": np.tanh(np.arctanh(r) - half), "r_ci_high": np.tanh(np.arctanh(r) + half),
            "p": p, "R2": r ** 2, "rho": rho, "p_rho": p_rho}


def summarize_folds(per_fold):
    """Median and range over folds of the per-fold statistics."""
    rows = []
    for (assoc, signals), d in per_fold.groupby(["association", "signals"], sort=False):
        row = {"association": assoc, "signals": signals, "scope": "harmonized, all individuals",
               "dataset": "all", "n": int(d["n"].median())}
        for col in ["r", "R2", "rho"]:
            row[col] = d[col].median()
            row[f"{col}_min"], row[f"{col}_max"] = d[col].min(), d[col].max()
        row["r_ci_low"], row["r_ci_high"] = d["r_ci_low"].min(), d["r_ci_high"].max()  # widest over folds
        row["p"], row["p_rho"] = d["p"].median(), d["p_rho"].median()
        row["p_min"] = d["p"].min()
        rows.append(row)
    return rows


def raw_and_per_dataset(cv, x_col, y_col, assoc, signals):
    rows = [{"association": assoc, "signals": signals, "scope": "raw, pooled", "dataset": "all",
             **correlate(cv[x_col], cv[y_col])}]
    for dataset, d in cv.groupby("dataset"):
        rows.append({"association": assoc, "signals": signals, "scope": "raw, within dataset",
                     "dataset": dataset, **correlate(d[x_col], d[y_col])})
    return rows


# ---------------------------------------------------------------- global power

def cached_total_variance(norm_table, cv, n_jobs):
    signals = NORM_TABLE_SIGNALS[norm_table]
    path = OUT / f"total_variance_{Path(signals).name}.csv"
    if path.exists():
        tv = pd.read_csv(path)
    else:
        print(f"Computing total variance: {signals}", flush=True)
        tv = cv[["participant_id", "set"]].assign(total_variance=total_variance(signals, cv, n_jobs))
        tv.to_csv(path, index=False)
    return cv[["participant_id"]].merge(tv, on="participant_id", how="left")["total_variance"].to_numpy()


def power_vs_age(cv, n_jobs):
    per_fold, rows = [], []
    for norm_table, signals in NORM_TABLE_SIGNALS.items():
        name = Path(signals).name
        log_tv = np.log(cached_total_variance(norm_table, cv, n_jobs))
        factors = pd.read_csv(Path(inputs_dir("main")) / norm_table).set_index("set")
        harmonized = {}
        for fold in MAIN_FOLDS:
            c = cv["set"].map(factors[fold]).to_numpy()
            harmonized[fold] = log_tv + 2 * np.log(c)
            per_fold.append({"association": "power_vs_age", "signals": name, "fold": fold,
                             **correlate(harmonized[fold], cv[f"{fold}_NORMage"])})
        data = cv.assign(log_power=log_tv,
                         log_power_harmonized=np.mean([harmonized[f] for f in MAIN_FOLDS], axis=0))
        rows += raw_and_per_dataset(data, "log_power", "age", "power_vs_age", name)[:1]
        # within dataset: harmonized power, so Set_Munich1 (two raw sets, scaled separately) is correct
        rows += [dict(r, scope="harmonized, within dataset")
                 for r in raw_and_per_dataset(data, "log_power_harmonized", "age", "power_vs_age", name)[1:]]
    return per_fold, rows


# ---------------------------------------------------------------- pain

def pain_vs_age(cv):
    per_fold = [{"association": "pain_vs_age", "signals": "", "fold": fold,
                 **correlate(cv[f"{fold}_NORMpainVars"], cv[f"{fold}_NORMage"])} for fold in MAIN_FOLDS]
    return per_fold, raw_and_per_dataset(cv, "painVars", "age", "pain_vs_age", "")


# ----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n_jobs", type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    cv = pd.read_csv(CV_MAIN_TABLE)
    cv["dataset"] = cv["set"].map(alias_for)

    fold_pain, rows_pain = pain_vs_age(cv)
    fold_power, rows_power = power_vs_age(cv, args.n_jobs)

    per_fold = pd.DataFrame(fold_power + fold_pain)
    per_fold.to_csv(OUT / "age_associations_per_fold.csv", index=False)

    summary = pd.DataFrame(summarize_folds(per_fold) + rows_power + rows_pain)
    order = {"harmonized, all individuals": 0, "raw, pooled": 1, "harmonized, within dataset": 2,
             "raw, within dataset": 2}
    summary = summary.sort_values(["association", "signals", "scope"], key=lambda s: s.map(order) if s.name == "scope" else s,
                                  kind="stable")
    summary.to_csv(OUT / "age_associations_summary.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_rows", 200, "display.float_format", "{:.3f}".format):
        print(summary[["association", "signals", "scope", "dataset", "n", "r", "r_ci_low", "r_ci_high", "p", "R2", "rho"]]
              .to_string(index=False))
    print(f"\nWritten to {OUT}")


if __name__ == "__main__":
    main()
