import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

# --- CPP adaptation: paths and settings come from paths.py and the command line ---
import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from paths import figures_dir, tables_dir
_parser = argparse.ArgumentParser()
_parser.add_argument("--target", required=True, choices=['age', 'pain'])
args = _parser.parse_args()
# --- end CPP adaptation ---

# 1) Path to CSV on disk
target = args.target
CSV_PATH = tables_dir("main") + f"combined_{target}_tables_small.csv"

# 2) One preset config per model_folder (keys MUST match values in df["model_folder"])
PRESET_BY_MODEL = {
    "OTiS_GEN":     {"spatConfig": "Sensor",     "tempConfig": 2,       "algo": None,       "agg": None},
    "OTiS_EEG":            {"spatConfig": "Sensor",     "tempConfig": 2,       "algo": None,       "agg": None},
    "LaBraM":               {"spatConfig": "Sensor",     "tempConfig": 2,       "algo": None,       "agg": None},
    "CND":          {"spatConfig": "Sensor",     "tempConfig": 2,       "algo": None,       "agg": None},
    "CNS":       {"spatConfig": "Sensor",     "tempConfig": 2,       "algo": None,       "agg": None},
    "HCF":                  {"spatConfig": "Sensor",     "tempConfig": 2,       "algo": "rf",       "agg": "mean"},
    "FBS":                  {"spatConfig": "Sensor",     "tempConfig": 2,       "algo": "ridge",    "agg": "mean"},
    "FBR":                  {"spatConfig": "Sensor",     "tempConfig": 2,       "algo": "ridge",    "agg": "mean"},
    "CPN":                  {"spatConfig": "Yeo7",       "tempConfig": 2,       "algo": "linear",   "agg": "mean"},
}

MODEL_ALIASES = {
    "OTiS_GEN": "OTiS_GEN",
    "OTiS_EEG": "OTiS_EEG",
    "LaBraM": "LaBraM",
    "CND": "ConvNetDeep",
    "CNS": "ConvNetShallow",
    "HCF": "HandCraftedFeatures",
    "FBS": "FilterBankSource",
    "FBR": "FilterBankRiemann",
    "CPN": "ChronicPainNetworks"
}

CONFIG_ALIASES = {
    "spatConfig": {
        "Sensor": "Sen-64",
        "Yeo7": "Yeo-7",
        "Schaefer100": "Sch-100",
        None: "None"
    },
    "tempConfig": {
        2: "2s",
        20: "20s",
        None: "None"
    },
    "algo": {
        "ridge": "Lin",
        "rf": "Non-lin",
        "linear": "Lin",
        "xgboost": "Non-lin",
        None: "Full"
    },
    "agg": {
        "mean": "Feat",
        None: "Pred",
        "none": "Pred"
    }
}

complement_info = {
    "CPN": {'spatConfig': 'Yeo7', 'tempConfig': 2, 'algo': 'linear', 'agg': 'mean'},
    "LaBraM": {'spatConfig': 'Sensor'},
}

METRICS = ["R", "spearman_pred_obs", "R2"]
METRICS_LABELS = {"R": "R", "spearman_pred_obs": "ρ", "R2": "R²"}

if target == "age":
    METRIC_NORM = {
        "R": plt.Normalize(-0.5, 0.5),
        "spearman_pred_obs": plt.Normalize(-0.5, 0.5),
        "R2": mpl.colors.TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.25),
    }
elif target == "pain":
    METRIC_NORM = {
        "R": plt.Normalize(-0.25, 0.25),
        "spearman_pred_obs": plt.Normalize(-0.25, 0.25),
        "R2": mpl.colors.TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.05),
    }

# R² colorbar is asymmetric (white at 0): label both ends, the negative midpoint and 0
R2_TICKS = [METRIC_NORM["R2"].vmin, METRIC_NORM["R2"].vmin / 2, 0.0, METRIC_NORM["R2"].vmax]

BF_COL = "correlationBF"


def text_color(face):
    """Black or white text, whichever contrasts more with the tile color (WCAG relative luminance)."""
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in mpl.colors.to_rgb(face)]
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return "white" if lum < 0.18 else "black"

# ---------------- NEW: best-config columns + aliases ----------------
BEST_CFG_COLS = ["spatConfig", "tempConfig", "algo", "agg"]

# You can rename what is displayed in the plot here (defaults to the raw column name)
BEST_CFG_ALIASES = {
    "spatConfig": "spat",
    "tempConfig": "temp",
    "algo": "algo",
    "agg": "agg",
}


def same_cfg(col, a, b):
    def norm(v):
        if v is None or pd.isna(v):
            return None
        if isinstance(v, str):
            v = v.strip().lower()
            if col == "agg" and v == "none":
                return None
        return v
    return norm(a) == norm(b)

# -------------------------------------------------------------------

def match_row(sub, cfg):
    m = pd.Series(True, index=sub.index)
    for col, val in cfg.items():
        if val is None:
            m &= sub[col].isna()
        else:
            m &= (sub[col] == val)
    hit = sub[m]
    return None if hit.empty else hit.iloc[0]

# --- Load ---
df = pd.read_csv(CSV_PATH)

for model, info in complement_info.items():
    m = df["model_folder"] == model
    for col, val in info.items():
        df.loc[m & df[col].isna(), col] = val

df["tempConfig"] = pd.to_numeric(df["tempConfig"], errors="coerce")
for c in METRICS:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df[BF_COL] = pd.to_numeric(df[BF_COL], errors="coerce")

models = list(PRESET_BY_MODEL.keys())
n_models = len(models)

# --- Build data arrays ---
preset_vals = np.full((n_models, 3), np.nan)
best_vals   = np.full((n_models, 3), np.nan)
preset_bf   = np.full(n_models, np.nan)
best_bf     = np.full(n_models, np.nan)

# NEW: best config values per row (strings/objects)
best_cfg_vals = {c: [None] * n_models for c in BEST_CFG_COLS}

# NEW: per-model flags for formatting config tiles
cfg_no_variants = {c: [False] * n_models for c in BEST_CFG_COLS}   # -> wrap in [...]
cfg_underline   = {c: [False] * n_models for c in BEST_CFG_COLS}   # -> differs from preset


for i, model in enumerate(models):
    sub = df[df["model_folder"] == model]

    # Determine whether each config dimension has variants for this model
    for c in BEST_CFG_COLS:
        u = pd.Series(sub[c]).dropna().unique()
        cfg_no_variants[c][i] = (len(u) <= 1)

    preset_row = match_row(sub, PRESET_BY_MODEL[model])
    if preset_row is not None:
        preset_vals[i] = preset_row[METRICS].values
        preset_bf[i] = preset_row[BF_COL]

    if not sub.empty and sub["R"].notna().any():
        best_row = sub.loc[sub["R"].idxmax()]
        best_vals[i] = best_row[METRICS].values
        best_bf[i] = best_row[BF_COL]

        for c in BEST_CFG_COLS:
            bv = best_row.get(c, np.nan)
            bv = None if pd.isna(bv) else bv
            best_cfg_vals[c][i] = bv

            pv = PRESET_BY_MODEL[model].get(c, None)
            cfg_underline[c][i] = (not same_cfg(c, bv, pv))

# --- emphasis logic (R column only) ---
bold_preset_R = np.isfinite(preset_bf) & (preset_bf > 3)

bf_ratio = np.full(n_models, np.nan)
valid_ratio = np.isfinite(preset_bf) & np.isfinite(best_bf) & (preset_bf != 0)
bf_ratio[valid_ratio] = best_bf[valid_ratio] / preset_bf[valid_ratio]

underline_best_R = np.isfinite(bf_ratio) & (bf_ratio > 10)
bold_best_R      = np.isfinite(best_bf) & (best_bf > 10)

# --- Plot: ONE axis for labels + both blocks + cfg columns; separate colorbar column ---
size_factor = 1
fig = plt.figure(figsize=(12.0 * size_factor, 1.1 * size_factor * n_models))

gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.07], wspace=0.25)
ax = fig.add_subplot(gs[0, 0])

# Colorbar column: 3 stacked bars
cgs = gs[0, 1].subgridspec(3, 1, hspace=0.35)
cax_R   = fig.add_subplot(cgs[0, 0])
cax_rho = fig.add_subplot(cgs[1, 0])
cax_R2  = fig.add_subplot(cgs[2, 0])

# Colormap
cmap = LinearSegmentedColormap.from_list(
        "blue_white_rocket",
        # negative values: white -> blue (mirror of the white -> red ramp for positives)
        ["#3a64b0", "#a9c1e6", "#ffffff", "#f2c2a2", "#c43c35"]
    )

# ---- layout in "tile units" ----
label_w = 1.0

gap_label_to_preset = 0.30
gap_R_to_others     = 0.20
gap_block           = 0.60
gap_cfg             = 0.2   # space between config tiles (in tile units)
gap_best_to_cfg     = 0.6

cfg_tile_w = 1.5
cfg_cols_w = len(BEST_CFG_COLS) * cfg_tile_w + (len(BEST_CFG_COLS) - 1) * gap_cfg

def x_in_block(block_left, j):
    return block_left + j + (gap_R_to_others if j >= 1 else 0.0)

block_w = 3.0 + gap_R_to_others

x_label0  = 0.0
x_preset0 = x_label0 + label_w + gap_label_to_preset
x_best0   = x_preset0 + block_w + gap_block

# cfg columns start
x_cfg0 = x_best0 + block_w + gap_best_to_cfg

total_w = x_cfg0 + cfg_cols_w

ax.set_xlim(0, total_w)
ax.set_aspect("equal", adjustable="box")
ax.set_facecolor("white")

ax.set_yticks([])
ax.tick_params(left=False)
ax.tick_params(axis="x", bottom=False, labelbottom=False)

# Header space (titles + metric labels)
header_rows = 2.0
ax.set_ylim(n_models, -header_rows)

for spine in ax.spines.values():
    spine.set_visible(False)

# ---------- Manual headers ----------
y_titles  = -1.0
y_metrics = -0.5

ax.text(x_label0 + label_w/2,   y_titles,  "Model name",   ha="center", va="center", fontsize=11)
ax.text(x_preset0 + block_w/2,  y_titles,  "Preset",       ha="center", va="center", fontsize=11)
ax.text(x_best0 + block_w/2,    y_titles,  "Best (max R)", ha="center", va="center", fontsize=11)
ax.text(x_cfg0 + cfg_cols_w/2,  y_titles,  "Best config",  ha="center", va="center", fontsize=11)

def metric_centers(block_left):
    return [x_in_block(block_left, j) + 0.5 for j in range(3)]

ax.text(x_label0 + 0.5, y_metrics, "Model", ha="center", va="center", fontsize=10)

for x, lab in zip(metric_centers(x_preset0), [METRICS_LABELS[m] for m in METRICS]):
    ax.text(x, y_metrics, lab, ha="center", va="center", fontsize=10)

for x, lab in zip(metric_centers(x_best0), [METRICS_LABELS[m] for m in METRICS]):
    ax.text(x, y_metrics, lab, ha="center", va="center", fontsize=10)

# per-property headers using aliases
for k, col in enumerate(BEST_CFG_COLS):
    hdr = BEST_CFG_ALIASES.get(col, col)
    ax.text(
        x_cfg0 + k * (cfg_tile_w + gap_cfg) + cfg_tile_w / 2,
        y_metrics,
        hdr,
        ha="center",
        va="center",
        fontsize=10
    )

def draw_tile(ax, x, y, w=1.0, face=(1,1,1,1), text=None, ha="center", fontweight="normal",
              underline=False, underline_xpad_frac=0.18, underline_y=0.62, x_text_frac=0.5, fontsize=9):
    ax.add_patch(patches.Rectangle(
        (x, y), w, 1,
        facecolor=face,
        edgecolor="white",
        linewidth=2
    ))
    if text is not None:
        ax.text(x + w * x_text_frac, y + 0.5, text,
                ha=ha, va="center", fontsize=fontsize, fontweight=fontweight, color=text_color(face))
    if underline:
        ax.plot(
            [x + w * underline_xpad_frac, x + w * (1.0 - underline_xpad_frac)],
            [y + underline_y,            y + underline_y],
            linewidth=1, solid_capstyle="butt", color=text_color(face)
        )

def draw_cfg_tile(ax, x, y, w, face, top_text, bottom_text=None, fontsize=8.3, divider=True):
    """
    Config tile with optional two-line layout:
      - If bottom_text is provided: top (best) above divider, bottom (preset) below divider.
      - Otherwise: single centered line (top_text only).
    """
    ax.add_patch(patches.Rectangle(
        (x, y), w, 1,
        facecolor=face,
        edgecolor="white",
        linewidth=2
    ))

    if bottom_text is None:
        ax.text(x + w * 0.5, y + 0.5, top_text, ha="center", va="center", fontsize=fontsize)
        return

    # Divider line
    if divider:
        ax.plot([x + w * 0.12, x + w * 0.88], [y + 0.5, y + 0.5],
                linewidth=1, solid_capstyle="butt", color="black")

    # Two text bands
    ax.text(x + w * 0.5, y + 0.28, top_text,    ha="center", va="center", fontsize=fontsize)
    ax.text(x + w * 0.5, y + 0.74, bottom_text, ha="center", va="center", fontsize=fontsize)

def fmt_cfg(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
        return "—"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if float(v).is_integer():
            return str(int(v))
        return f"{float(v):.3g}"
    return str(v)

def fmt_cfg_alias(col, v):
    v = None if (v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v)) else v
    return CONFIG_ALIASES.get(col, {}).get(v, fmt_cfg(v))

# --- draw rows ---
for i, model in enumerate(models):
    # Label tile (right-aligned near right edge)
    draw_tile(
        ax,
        x=x_label0,
        y=i,
        w=label_w,
        face=(1, 1, 1, 1),
        text=MODEL_ALIASES[model],
        ha="right",
        x_text_frac=0.95
    )

    # Preset tiles
    for j, metric in enumerate(METRICS):
        val = preset_vals[i, j]
        face = (1, 1, 1, 1) if np.isnan(val) else cmap(METRIC_NORM[metric](val))
        is_bold = (j == 0 and bool(bold_preset_R[i]))
        draw_tile(
            ax,
            x=x_in_block(x_preset0, j),
            y=i,
            w=1.0,
            face=face,
            text=None if np.isnan(val) else f"{val:.2f}",
            fontweight="bold" if is_bold else "normal",
            underline=False
        )

    # Best tiles
    for j, metric in enumerate(METRICS):
        val = best_vals[i, j]
        face = (1, 1, 1, 1) if np.isnan(val) else cmap(METRIC_NORM[metric](val))
        is_bold = (j == 0 and bool(bold_best_R[i]))
        is_ul   = (j == 0 and bool(underline_best_R[i]))
        draw_tile(
            ax,
            x=x_in_block(x_best0, j),
            y=i,
            w=1.0,
            face=face,
            text=None if np.isnan(val) else f"{val:.2f}",
            fontweight="bold" if is_bold else "normal",
            underline=is_ul
        )

    # Config tiles: show best (top) + preset/standard (bottom) when best differs
    for k, col in enumerate(BEST_CFG_COLS):
        bv = best_cfg_vals[col][i]
        bv = None if pd.isna(bv) else bv
        pv = PRESET_BY_MODEL[model].get(col, None)

        best_shown   = fmt_cfg_alias(col, bv)
        preset_shown = fmt_cfg_alias(col, pv)

        # If there are no variants for this model on this config dimension -> wrap single entry in [...]
        if cfg_no_variants[col][i]:
            # In this case best == preset, so print a single value (as before), but still bracket it.
            shown = f"[{best_shown}]"
            draw_cfg_tile(
                ax,
                x=x_cfg0 + k * (cfg_tile_w + gap_cfg),
                y=i,
                w=cfg_tile_w,
                face=(0.95, 0.95, 0.95, 0.95),
                top_text=shown,
                bottom_text=None,
                fontsize=8.5,
                divider=False
            )
        else:
            # Variants exist:
            # If best differs from preset, show two-line tile with divider.
            # Otherwise, show a single-line tile (no divider).
            if cfg_underline[col][i]:
                draw_cfg_tile(
                    ax,
                    x=x_cfg0 + k * (cfg_tile_w + gap_cfg),
                    y=i,
                    w=cfg_tile_w,
                    face=(0.95, 0.95, 0.95, 0.95),
                    top_text=best_shown,
                    bottom_text=preset_shown,
                    fontsize=7.9,
                    divider=True
                )
            else:
                draw_cfg_tile(
                    ax,
                    x=x_cfg0 + k * (cfg_tile_w + gap_cfg),
                    y=i,
                    w=cfg_tile_w,
                    face=(0.95, 0.95, 0.95, 0.95),
                    top_text=best_shown,
                    bottom_text=None,
                    fontsize=8.5,
                    divider=False
                )

# --- Three separate colorbars: R, rho, R2 ---
sm_R   = mpl.cm.ScalarMappable(norm=METRIC_NORM["R"], cmap=cmap)
sm_rho = mpl.cm.ScalarMappable(norm=METRIC_NORM["spearman_pred_obs"], cmap=cmap)
sm_R2  = mpl.cm.ScalarMappable(norm=METRIC_NORM["R2"], cmap=cmap)
sm_R.set_array([]); sm_rho.set_array([]); sm_R2.set_array([])

cbR = fig.colorbar(sm_R, cax=cax_R, extend="both");     cbR.set_label("R")
cbr = fig.colorbar(sm_rho, cax=cax_rho, extend="both"); cbr.set_label("ρ")
cb2 = fig.colorbar(sm_R2, cax=cax_R2, extend="both");   cb2.set_label("R²")
cb2.set_ticks(R2_TICKS)

out = figures_dir("main") + f"tiles_{target}.pdf"
plt.savefig(out, bbox_inches="tight")
print(f"Saved: {out}")
