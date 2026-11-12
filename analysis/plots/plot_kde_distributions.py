import os
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.stats import gaussian_kde
from matplotlib.ticker import MaxNLocator

# --- CPP adaptation: paths and settings come from paths.py and the command line ---
import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from paths import TURBO_ROOT, CV_MAIN_TABLE, figures_dir, inputs_dir
_parser = argparse.ArgumentParser()
_parser.add_argument("--signal_variant", required=True, choices=['sensor_2', 'sensor_20', 'yeo_2', 'yeo_20', 'schaefer_2', 'schaefer_20'])
_parser.add_argument("--n_jobs", default=1, type=int)
args = _parser.parse_args()
# --- end CPP adaptation ---

signal_variant = args.signal_variant
n_jobs = args.n_jobs
curr_fold = "FoldAll"  # only used for naming outputs
bw_all = 0.4
if signal_variant == "sensor_2":
    signals_path = f"{TURBO_ROOT}/data/fineTuning/sensorSignals/epoched_autoPreproc_CUT2.0_CUT0.5_250/"
    norm_table_path = f"{inputs_dir('main')}CV70-20-10Table_Sensor1020_all_MegaStrat_normFactorsNew.csv"
elif signal_variant == "sensor_20":
    signals_path = f"{TURBO_ROOT}/data/fineTuning/sensorSignals/epoched_autoPreproc_20.0_0.5_250/"
    norm_table_path = f"{inputs_dir('main')}CV70-20-10Table_Sensor1020_all_MegaStrat_normFactorsNew.csv"
elif signal_variant == "yeo_2":
    signals_path = f"{TURBO_ROOT}/data/fineTuning/sourceSignals/OrthoAggSchaefer100_freq-all_epochLength-2_overlap-0.5/"
    norm_table_path = f"{inputs_dir('main')}CV70-20-10Table_OrthoAggSchaefer100_all_2.0_MegaStrat_normFactorsNew.csv"
elif signal_variant == "yeo_20":
    signals_path = f"{TURBO_ROOT}/data/fineTuning/sourceSignals/OrthoAggSchaefer100_freq-all_epochLength-2_overlap-0.5/"
    norm_table_path = f"{inputs_dir('main')}CV70-20-10Table_OrthoAggSchaefer100_all_20.0_MegaStrat_normFactorsNew.csv"
elif signal_variant == "schaefer_2":
    signals_path = f"{TURBO_ROOT}/data/fineTuning/sourceSignals/Schaefer100_freq-all_epochLength-2_overlap-0.5/"
    norm_table_path = f"{inputs_dir('main')}CV70-20-10Table_Schaefer100_all_2.0_MegaStrat_normFactorsNew.csv"
elif signal_variant == "schaefer_20":
    signals_path = f"{TURBO_ROOT}/data/fineTuning/sourceSignals/Schaefer100_freq-all_epochLength-20_overlap-0.5/"
    norm_table_path = f"{inputs_dir('main')}CV70-20-10Table_Schaefer100_all_20.0_MegaStrat_normFactorsNew.csv"
else:
    raise ValueError(f"Unknown signal_variant: {signal_variant}")

cv_table_path = CV_MAIN_TABLE
output_dir = figures_dir("main")

cv_table = pd.read_csv(cv_table_path)
norm_table = pd.read_csv(norm_table_path).set_index("set")

from dataset_names import SET_COLORS, alias_for

cv_table["set_alias"] = cv_table["set"].apply(alias_for)

def process_segments_with_norm(signals, norm_factor: float) -> tuple[float, float]:
    trials = signals["trial"][0, 0]
    n_epochs = trials.size
    rescale_factor = np.sqrt(np.exp(-9))

    epoch_var = []
    epoch_var_scaled = []

    for iSegment in range(n_epochs):
        segment = trials[0, iSegment].astype(np.float64, copy=True)  # (channels, time)
        segment -= np.mean(segment, axis=1, keepdims=True)

        segment_rescaled = segment * rescale_factor
        segment_scaled = segment_rescaled * norm_factor

        v = np.sum(np.mean(segment_rescaled ** 2, axis=1))
        vs = np.sum(np.mean(segment_scaled ** 2, axis=1))

        epoch_var.append(v)
        epoch_var_scaled.append(vs)

    return (float(np.mean(epoch_var)) if epoch_var else np.nan,
            float(np.mean(epoch_var_scaled)) if epoch_var_scaled else np.nan)

# ---- Index files once ----
all_files = [os.path.join(signals_path, fn) for fn in os.listdir(signals_path)]

def find_subject_file(curr_subject: str) -> str | None:
    for fp in all_files:
        if os.path.basename(fp).startswith(f"{curr_subject}__"):
            return fp
    return None

def compute_subject(curr_subject: str, curr_set: str, currSetAlias: str):
    if curr_set not in norm_table.index:
        return {"participant_id": curr_subject, "set": curr_set, "set_alias": currSetAlias, "error": "missing_norm_factor"}

    nf = float(np.mean(norm_table.loc[curr_set].to_numpy()))
    subject_file = find_subject_file(curr_subject)

    if subject_file is None or not os.path.exists(subject_file):
        return {"participant_id": curr_subject, "set": curr_set, "set_alias": currSetAlias, "error": "file_not_found"}

    try:
        dataSubject = scipy.io.loadmat(subject_file)
    except Exception as e:
        return {"participant_id": curr_subject, "set": curr_set, "set_alias": currSetAlias, "error": f"load_error: {e}", "file": subject_file}

    if "brainSignals" in dataSubject:
        signals = dataSubject["brainSignals"]
    elif "data" in dataSubject:
        signals = dataSubject["data"]
    else:
        return {"participant_id": curr_subject, "set": curr_set, "set_alias": currSetAlias, "error": "missing_signal_key", "file": subject_file}

    try:
        tv, tvs = process_segments_with_norm(signals, nf)
    except Exception as e:
        return {"participant_id": curr_subject, "set": curr_set, "set_alias": currSetAlias, "error": f"compute_error: {e}", "file": subject_file}

    return {
        "participant_id": curr_subject,
        "set": curr_set,
        "set_alias": currSetAlias,
        "file": subject_file,
        "totalVariance": tv,
        "totalVarianceScaled": tvs,
        "error": None
    }

tasks = list(zip(cv_table["participant_id"], cv_table["set"], cv_table["set_alias"]))

results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
    delayed(compute_subject)(subj, s, s_alias) for subj, s, s_alias in tasks
)

ok_df = pd.DataFrame([r for r in results if not r.get("error")])
err_df = pd.DataFrame([r for r in results if r.get("error")])

if len(err_df):
    print("Some errors (showing first 10):")
    cols = [c for c in ["participant_id", "set", "set_alias", "error", "file"] if c in err_df.columns]
    print(err_df.head(10)[cols])

tv_df = ok_df.dropna(subset=["totalVariance", "totalVarianceScaled"]).copy()
print(f"tv_df ready: {len(tv_df)} subjects")

aliases = sorted(tv_df["set_alias"].dropna().unique())
if len(aliases) == 0:
    raise ValueError("No set aliases found to plot.")

color_map = {a: SET_COLORS[a] for a in aliases}

def kde_fill(ax, values, label, xlim, color, alpha=0.4, bw=1):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return None

    kde = gaussian_kde(values, bw_method=bw)
    grid = np.linspace(xlim[0], xlim[1], 400)
    dens = kde(grid)

    ax.fill_between(
        grid,
        dens,
        color=color,
        alpha=alpha,
        label=label,
        zorder=1
    )

    ax.plot(
        grid,
        dens,
        color=color,
        linewidth=2.5,
        alpha = 0.8,
        zorder=2
    )
    return color

def fold_mean(df: pd.DataFrame, suffix: str) -> pd.Series | None:
    cols = [c for c in df.columns if c.startswith("Fold") and c.endswith(suffix)]
    if not cols:
        return None
    return df[cols].mean(axis=1)

def plot_by_alias_two_panel(
    df: pd.DataFrame,
    value_col_raw: str,
    value_col_norm: str | None,
    out_png: str,
    xlab_raw: str,
    xlab_norm: str,
    do_log_center: bool = False,
    bw: float = 0.2,
    use_same_xlim: bool = True,
    norm_is_series: bool = False,
    norm_series: pd.Series | None = None,
    xlim_raw = None,
    xlim_norm = None
):
    """
    df must contain 'set_alias' plus the raw column.
    norm can be either:
      - a column name (value_col_norm), OR
      - an explicit Series (norm_is_series=True, norm_series provided) aligned to df.index.
    """
    with plt.rc_context({
        'font.size': 16,
        'font.family': 'Helvetica'
    }):
        fig, axes = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(3*1.4, 4*1.4),
            constrained_layout=True
        )

    # ---------- RAW pooled range ----------
    raw_all = df[value_col_raw].to_numpy(dtype=float)
    raw_all = raw_all[np.isfinite(raw_all)]
    if do_log_center:
        raw_all = raw_all[raw_all > 0]
        raw_all = np.log10(raw_all)
        shift_raw = np.mean(raw_all)
        raw_all_plot = raw_all - shift_raw
    else:
        shift_raw = 0.0
        raw_all_plot = raw_all

    if raw_all_plot.size < 2:
        raise ValueError(f"Not enough values to plot raw: {value_col_raw}")

    if not xlim_raw:
        xlim_raw = (np.min(raw_all_plot), np.max(raw_all_plot))

    # ---------- NORM pooled range ----------
    if norm_is_series:
        if norm_series is None:
            raise ValueError("norm_is_series=True but norm_series=None")
        norm_all = norm_series.to_numpy(dtype=float)
    else:
        if value_col_norm is None:
            raise ValueError("value_col_norm is None but norm_is_series=False")
        norm_all = df[value_col_norm].to_numpy(dtype=float)

    norm_all = norm_all[np.isfinite(norm_all)]
    if do_log_center:
        norm_all = norm_all[norm_all > 0]
        norm_all = np.log10(norm_all)
        shift_norm = np.mean(norm_all)
        norm_all_plot = norm_all - shift_norm
    else:
        shift_norm = 0.0
        norm_all_plot = norm_all

    if norm_all_plot.size < 2:
        raise ValueError(f"Not enough values to plot normalized for: {value_col_raw}")

    if not xlim_norm:
        xlim_norm = (np.min(norm_all_plot), np.max(norm_all_plot))

    if use_same_xlim:
        # emulate your totalVariance behavior: same x-range in both panels
        xlim_norm = xlim_raw

    # ---------- Panel 1: raw ----------
    ax = axes[0]
    ax.set_xlim(*xlim_raw)
    for a in aliases:
        vals = df.loc[df["set_alias"] == a, value_col_raw].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if do_log_center:
            vals = vals[vals > 0]
            vals = np.log10(vals) if vals.size else np.array([])
            vals = vals - shift_raw
        if vals.size >= 2:
            c = color_map[a]
            kde_fill(ax, vals, f"{a} (n={len(vals)})", xlim_raw, color=c, bw=bw)
            ax.axvline(np.mean(vals), linestyle="--", linewidth=3.5, alpha=1.0, color=c)

    ax.set_xlabel(xlab_raw)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

    # ---------- Panel 2: normalized ----------
    ax2 = axes[1]
    ax2.set_xlim(*xlim_norm)
    for a in aliases:
        idx = df["set_alias"] == a
        if norm_is_series:
            vals = norm_series.loc[idx].to_numpy(dtype=float)
        else:
            vals = df.loc[idx, value_col_norm].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if do_log_center:
            vals = vals[vals > 0]
            vals = np.log10(vals) if vals.size else np.array([])
            vals = vals - shift_norm
        if vals.size >= 2:
            c = color_map[a]
            kde_fill(ax2, vals, f"{a} (n={len(vals)})", xlim_norm, color=c, bw=bw)
            ax2.axvline(np.mean(vals), linestyle="--", linewidth=3.5, alpha=1.0, color=c)

    ax2.set_xlabel(xlab_norm)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_ylim(bottom=0)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=5))

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {out_png}")

save_power = os.path.join(output_dir, f"KDE_totalVariance_allsets_{curr_fold}_{signal_variant}.png")
plot_by_alias_two_panel(
    df=tv_df,
    value_col_raw="totalVariance",
    value_col_norm="totalVarianceScaled",
    out_png=save_power,
    xlab_raw="log10(global power)",
    xlab_norm="log10(global power of harmonized signals)",
    do_log_center=True,
    bw=bw_all,
    use_same_xlim=True,
    norm_is_series=False
)

age_norm = fold_mean(cv_table, "NORMage")

save_age = os.path.join(output_dir, f"KDE_age_allsets_{curr_fold}.png")
plot_by_alias_two_panel(
    df=cv_table,
    value_col_raw="age",
    value_col_norm=None,
    out_png=save_age,
    xlab_raw="age",
    xlab_norm="harmonized age",
    do_log_center=False,
    bw=bw_all,
    use_same_xlim=False,        # raw age and normalized age typically differ in scale
    norm_is_series=True,
    norm_series=age_norm,
    xlim_raw = (0, 100),
    xlim_norm=(-4,4)
)

pain_norm = fold_mean(cv_table, "NORMpainVars")

save_pain = os.path.join(output_dir, f"KDE_painVars_allsets_{curr_fold}.png")
plot_by_alias_two_panel(
    df=cv_table,
    value_col_raw="painVars",
    value_col_norm=None,
    out_png=save_pain,
    xlab_raw="pain intensity",
    xlab_norm="harmonized pain intensity",
    do_log_center=False,
    bw=bw_all,
    use_same_xlim=False,        # raw pain and normalized pain typically differ in scale
    norm_is_series=True,
    norm_series=pain_norm,
    xlim_raw = (0, 10),
    xlim_norm=(-4,4)
)

print("Done. Saved:")
print(f"  - {save_power}")
print(f"  - {save_age}")
print(f"  - {save_pain}")
