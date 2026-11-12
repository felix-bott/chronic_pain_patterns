import pandas as pd
import os
import numpy as np
import json
from scipy.stats import spearmanr

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter

utils = importr("utils")
utils.chooseCRANmirror(ind=1)
BayesFactor = importr("BayesFactor")


# --- CPP adaptation: paths and settings come from paths.py and the command line ---
import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from paths import CPP_RESULTS, ANALYSES, summaries_dir, tables_dir
_parser = argparse.ArgumentParser()
_parser.add_argument("--target", required=True, choices=['age', 'ageRAW', 'pain', 'painRAW'])
_parser.add_argument("--analysis", default="main", choices=ANALYSES,
                     help="main (default), or an additional analysis of the winners")
args = _parser.parse_args()
# --- end CPP adaptation ---

analysis = args.analysis
out_dir = tables_dir(analysis)
os.makedirs(out_dir, exist_ok=True)
# Output name: combined_<target>_tables.csv for the main and raw-target analyses,
# combined_<target>_<analysis>_tables.csv otherwise (e.g. combined_pain_LOSO_tables.csv)
out_stem = f"combined_{args.target}" if analysis in ("main", "RAW") else f"combined_{args.target}_{analysis}"

target = args.target

if target == "age":
    table_names = {
        'CND': ['FT_full_age'],
        'CNS': ['FT_full_age'],
        'CPN': ['Direct_age'],
        'FBR': ['Direct_age'],
        'FBS': ['Direct_age'],
        'HCF': ['Direct_age'],
        'LaBraM': ['FT_finLay_age', 'FT_full_age'],
        'OTiS_EEG': ['FT_finLay_age_0.75', 'FT_full_age_0.75'],
        'OTiS_GEN': ['FT_finLay_age_0.75', 'FT_full_age_0.75'],
        }
elif target == "pain":
    table_names = {
        'CND': ['FT_full_painVars'],
        'CNS': ['FT_full_painVars'],
        'CPN': ['Direct_pain'],
        'FBR': ['Direct_painVars'],
        'FBS': ['Direct_painVars'],
        'HCF': ['Direct_painVars'],
        'LaBraM': ['FT_finLay_pain', 'FT_full_pain'],
        'OTiS_EEG': ['FT_finLay_pain_0.75', 'FT_full_pain_0.75'],
        'OTiS_GEN': ['FT_finLay_pain_0.75', 'FT_full_pain_0.75'],
        }

if analysis != "main":
    # Reruns exist only for the winners: CPN and the feature models, whichever were run
    rerun_files = {
        "age": {"CPN": "Direct_age", "feature": "Direct_age"},
        "pain": {"CPN": "Direct_pain", "feature": "Direct_painVars"},
        "ageRAW": {"CPN": "Direct_ageRAW", "feature": "Direct_ageRAW"},
        "painRAW": {"CPN": "Direct_painRAW", "feature": "Direct_painVarsRAW"},
    }[target]
    table_names = {}
    for model in ["CPN", "FBR", "FBS", "HCF"]:
        fname = rerun_files["CPN" if model == "CPN" else "feature"]
        if os.path.exists(f"{summaries_dir(analysis, model)}{fname}.csv"):
            table_names[model] = [fname]

def correlation_bayes_factor(y, x, rscale="medium", nullInterval=None):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    with localconverter(ro.default_converter + numpy2ri.converter):
        r_y = ro.conversion.py2rpy(y)
        r_x = ro.conversion.py2rpy(x)

        if nullInterval is None:
            bf_obj = BayesFactor.correlationBF(
                y=r_y,
                x=r_x,
                rscale=rscale
            )
        else:
            r_int = ro.conversion.py2rpy(
                np.asarray(nullInterval, dtype=float)
            )
            bf_obj = BayesFactor.correlationBF(
                y=r_y,
                x=r_x,
                rscale=rscale,
                nullInterval=r_int
            )

        # Extract scalar Bayes factor
        bf_val = np.array(ro.r("as.vector")(bf_obj))[0]

    return float(bf_val)

def load_and_append_csvs(folder_to_files: dict[str, list[str]]) -> pd.DataFrame:
    """
    Loads all CSVs described by folder_to_files from each model's summaries folder,
    concatenates them into one table, and union-aligns columns (missing -> NA).

    Adds provenance columns:
      - model_folder: key in folder_to_files
      - source_file:  csv filename
      - source_path:  full path
    """
    dfs = []

    for folder, files in folder_to_files.items():
        for fname in files:
            # Allow filenames given with/without ".csv"
            if not fname.lower().endswith(".csv"):
                fname_csv = f"{fname}.csv"
            else:
                fname_csv = fname

            path = os.path.join(summaries_dir(analysis, folder), fname_csv)

            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing CSV: {path}")

            df = pd.read_csv(path)
            df["predictions"] = df["predictions"].apply(json.loads)
            df["observations"] = df["observations"].apply(json.loads)

            # Provenance columns help keep meaning after appending
            df.insert(0, "model_folder", folder)
            df.insert(1, "source_file", fname_csv)
            df.insert(2, "source_path", os.path.relpath(path, CPP_RESULTS))

            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    # Union of columns across all tables; missing values become NA automatically
    out = pd.concat(dfs, axis=0, ignore_index=True, sort=False)

    # Optional: prefer pandas' nullable missing values consistently
    out = out.convert_dtypes()

    return out


if __name__ == "__main__":
    combined = load_and_append_csvs(table_names)

    # compute additional metrics
    combined["spearman_pred_obs"] = [
        spearmanr(p,o)[0]
        for p, o in zip(combined["predictions"], combined["observations"])
    ]
    combined["correlationBF"] = [
        correlation_bayes_factor(p, o, nullInterval=[0,1])
        for p, o in zip(combined["predictions"], combined["observations"])
    ]

    for col in ["spatConfig", "tempConfig", "algo", "agg"]:  # e.g. CPN has no spatial or epoch setting
        if col not in combined.columns:
            combined[col] = pd.NA
    combined_small = combined[[
        "model_folder",
        "source_file",
        "spatConfig",
        "tempConfig",
        "algo",
        "agg",
        "R",
        "correlationBF",
        "spearman_pred_obs",
        "R2"
    ]]


    # Write output
    out_path = os.path.join(out_dir, f"{out_stem}_tables.csv")
    combined.to_csv(out_path, index=False)

    out_path_small = os.path.join(out_dir, f"{out_stem}_tables_small.csv")
    combined_small.to_csv(out_path_small, index=False)

    print(f"Combined rows: {len(combined):,}")
