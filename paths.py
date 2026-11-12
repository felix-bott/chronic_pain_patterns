"""Central paths for the CPP pipeline.

Results are organised per analysis, then per model and configuration:
CPP_RESULTS/analysis_<name>/<model>/<config>/{features,predictions}.
"""
import os
from pathlib import Path

TURBO_ROOT = os.environ.get("TURBO_ROOT", str(Path(__file__).resolve().parents[1]))
CPP_RESULTS = os.environ.get("CPP_RESULTS", os.path.join(TURBO_ROOT, "CPP_results"))
FINETUNE_DATA = os.path.join(TURBO_ROOT, "data", "fineTuning")  # signals and tables (read-only)
INV_OPS_DIR = os.path.join(FINETUNE_DATA, "invOps")
TEMPLATES_DIR = os.path.join(FINETUNE_DATA, "templates")

ANALYSIS_DIRS = {
    "main": "analysis_main",
    "CBP": "analysis_CBP-subgroup",
    "NP": "analysis_NP-subgroup",
    "LOSO": "analysis_LOSO",
    "RAW": "analysis_rawTargets",
}
ANALYSES = list(ANALYSIS_DIRS)
MAIN_FOLDS = [f"Fold{i}" for i in range(1, 11)]
CPN_RUN = "OrthoAggSchaefer100_freq-all_epochLength-2_overlap-0.5"  # configuration of the CPN features


def analysis_dir(analysis):
    return os.path.join(CPP_RESULTS, ANALYSIS_DIRS[analysis])


def inputs_dir(analysis):
    return os.path.join(analysis_dir(analysis), "inputs") + "/"


def tables_dir(analysis):
    return os.path.join(analysis_dir(analysis), "tables") + "/"


def figures_dir(analysis):
    return os.path.join(analysis_dir(analysis), "figures") + "/"


def summaries_dir(analysis, model):
    return os.path.join(analysis_dir(analysis), model, "summaries") + "/"


def features_dir(analysis, model, run_name):
    return os.path.join(analysis_dir(analysis), model, run_name, "features") + "/"


def predictions_dir(analysis, model, run_name, name):
    """Folder of one prediction set, e.g. name = "ridge_painVars" or "full_age"."""
    return os.path.join(analysis_dir(analysis), model, run_name, "predictions", name) + "/"


def resolve_features(folder):
    """The folder holding the features: this one, or the one named in its LOCATION.txt."""
    pointer = os.path.join(folder, "LOCATION.txt")
    if not os.path.exists(pointer):
        return folder
    with open(pointer) as fh:
        target = os.path.join(TURBO_ROOT, fh.read().strip())
    if not os.path.isdir(target):
        raise FileNotFoundError(f"{pointer} names a folder that does not exist: {target}")
    return target


CV_MAIN_TABLE = os.path.join(inputs_dir("main"), "CV70-20-10Table_MegaStrat.csv")


def canonical_target(target):
    """Target as named in prediction folders: "pain" and "painRAW" become "painVars" and "painVarsRAW"."""
    return {"pain": "painVars", "painRAW": "painVarsRAW"}.get(target, target)


def analysis_folds(analysis):
    """Fold columns used by an analysis."""
    if analysis == "LOSO":
        from dataset_names import DATASET_SETS
        return [f"LOSO_{dataset}" for dataset in DATASET_SETS]
    return MAIN_FOLDS


def fold_label(fold):
    """Label in metric columns: R_Fold3, or R_Set_Haifa for the LOSO fold that holds out Set_Haifa."""
    return fold[len("LOSO_"):] if fold.startswith("LOSO_") else fold
