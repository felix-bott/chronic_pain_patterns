"""
Paged tile table for combined_age_table_small.csv (A4 PDF, multi-page)

Changes vs prior:
- Single column layout per page (wider, avoids overly narrow two-column table).
- Split across N pages (specifiable via N_PAGES; default 2).
- Algo column displays "Lin (ridge)" / "Non-lin (rf)" etc. when applicable.
- Model column is right-aligned and clipped to its tile, preventing overlap into config columns.

Output: multi-page PDF at OUT_PDF.

Requirements:
  pip install matplotlib pandas numpy
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages


# --- CPP adaptation: paths and settings come from paths.py and the command line ---
import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from paths import figures_dir, tables_dir
_parser = argparse.ArgumentParser()
_parser.add_argument("--target", required=True, choices=['age', 'pain'])
args = _parser.parse_args()
# --- end CPP adaptation ---

# ----------------------- User settings -----------------------
target = args.target
CSV_PATH = tables_dir("main") + f"combined_{target}_tables_small.csv"
OUT_PDF  = figures_dir("main") + f"tiles_{target}_all_models_all_configs_A4_multipage.pdf"

# Set to 2 or 3 (or any positive int). Rows are distributed approximately evenly.
N_PAGES = 3
PAGE_HEIGHT_FRACTION = 0.75  # e.g. use 55% of the page height

METRICS = ["R", "spearman_pred_obs", "R2"]
METRICS_LABELS = {"R": "R", "spearman_pred_obs": "ρ", "R2": "R²"}

MODEL_ALIASES = {
    "OTiS_GEN": "OTiS_GEN",
    "OTiS_EEG": "OTiS_EEG",
    "LaBraM": "LaBraM",
    "CND": "ConvNetDeep",
    "CNS": "ConvNetShallow",
    "HCF": "HandCraftedFeatures",
    "FBS": "FilterBankSource",
    "FBR": "FilterBankRiemann",
    "CPN": "ChronicPainNetworks",
}

# Base label mapping for algo column (Lin / Non-lin) while preserving exact algo in parentheses
ALGO_BASE = {
    "ridge": "Lin",
    "linear": "Lin",
    "rf": "Non-lin",
    "xgboost": "Non-lin",
    None: "Full",
    np.nan: "Full",
}

CONFIG_ALIASES = {
    "spatConfig": {
        "Sensor": "Sen-64",
        "Yeo7": "Yeo-7",
        "Schaefer100": "Sch-100",
        None: "—",
        np.nan: "—",
    },
    "tempConfig": {
        2: "2s",
        20: "20s",
        None: "—",
        np.nan: "—",
    },
    # for agg we keep compact labels
    "agg": {
        "mean": "Feat",
        None: "Pred",
        "none": "Pred",
        np.nan: "—",
    },
}

# Optional: fill missing config values for specific models
complement_info = {
    "CPN": {'spatConfig': 'Yeo7', 'tempConfig': 2, 'algo': 'linear', 'agg': 'mean'},
    "LaBraM": {'spatConfig': 'Sensor'},
}

if target == "age":
    METRIC_NORM = {
        "R": mpl.colors.Normalize(-0.5, 0.5),
        "spearman_pred_obs": mpl.colors.Normalize(-0.5, 0.5),
        "R2": mpl.colors.TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.25),
    }
else:
    METRIC_NORM = {
        "R": mpl.colors.Normalize(-0.25, 0.25),
        "spearman_pred_obs": mpl.colors.Normalize(-0.25, 0.25),
        "R2": mpl.colors.TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.05),
    }
# R² colorbar is asymmetric (white at 0): label both ends, the negative midpoint and 0
R2_TICKS = [METRIC_NORM["R2"].vmin, METRIC_NORM["R2"].vmin / 2, 0.0, METRIC_NORM["R2"].vmax]

# ------------------------------------------------------------


def fmt_cfg(v):
    if v is None or pd.isna(v):
        return None
    if isinstance(v, str):
        s = v.strip()
        if s.lower() == "none":
            return None
        return s
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        if float(v).is_integer():
            return int(v)
        return float(v)
    return v


def fmt_cfg_alias(col, v):
    v = fmt_cfg(v)
    return CONFIG_ALIASES.get(col, {}).get(v, "—" if v is None else str(v))


def model_display_name(model_folder: str) -> str:
    return MODEL_ALIASES.get(model_folder, model_folder)


def algo_display(v):
    """
    Return e.g.:
      - "Lin (ridge)"
      - "Non-lin (rf)"
      - "—"
    If algo is missing -> "—"
    """
    v = fmt_cfg(v)
    if v is None:
        return "Full"
    base = ALGO_BASE.get(v, "Other")
    return f"{base} ({v})"


def build_config_sort_key(row) -> str:
    """
    Sort key uses the *display* strings for spat/temp/algo/agg so ordering matches what is shown.
    """
    parts = [
        fmt_cfg_alias("spatConfig", row.get("spatConfig", None)),
        fmt_cfg_alias("tempConfig", row.get("tempConfig", None)),
        algo_display(row.get("algo", None)),
        fmt_cfg_alias("agg", row.get("agg", None)),
    ]
    return "|".join(parts)


def text_color(face):
    """Black or white text, whichever contrasts more with the tile color (WCAG relative luminance)."""
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in mpl.colors.to_rgb(face)]
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return "white" if lum < 0.18 else "black"


def draw_tile(ax, x, y, w, h, face, text=None, fontsize=7.8,
              fontweight="normal", ha="center", x_text_frac=0.5,
              clip_text_to_tile=False):
    rect = patches.Rectangle((x, y), w, h, facecolor=face, edgecolor="white", linewidth=1.6)
    ax.add_patch(rect)
    if text is not None:
        t = ax.text(
            x + w * x_text_frac,
            y + h * 0.5,
            text,
            ha=ha,
            va="center",
            fontsize=fontsize,
            fontweight=fontweight,
            color=text_color(face),
        )
        if clip_text_to_tile:
            # prevent overlap into neighboring columns
            t.set_clip_path(rect)


def draw_page(df_page, page_idx, n_pages, cmap, fig=None):
    """
    Single-page render on A4 portrait. Returns (fig).
    """
    # A4 portrait in inches
    if fig is None:
        fig = plt.figure(figsize=(8.27, 11.69), constrained_layout=False)
    fig.clf()

    # Use only a fraction of the vertical page height (top-aligned)
    left = 0.06
    width = 0.88
    height = 0.86 * PAGE_HEIGHT_FRACTION
    bottom = 0.08 + (0.86 - height)  # shifts axis upward so it stays top-aligned

    ax = fig.add_axes([left, bottom, width, height])

    # Columns: # | Model | spat | temp | algo | agg | R | ρ | R²
    w_count = 0.65
    w_model = 3.35
    w_spat  = 1.05
    w_temp  = 1.05
    w_algo  = 1.65
    w_agg   = 1.00
    w_m     = 0.85

    col_names = ["#", "Model", "spat", "temp", "algo", "agg", "R", "ρ", "R²"]
    col_widths = [w_count, w_model, w_spat, w_temp, w_algo, w_agg, w_m, w_m, w_m]
    x_edges = np.cumsum([0] + col_widths).tolist()
    total_w = x_edges[-1]

    # Row sizing
    h = 1.0
    header_h = 1.85  # title + header row
    n = len(df_page)

    ax.set_xlim(0, total_w)
    ax.set_ylim(header_h + n * h, 0)
    ax.axis("off")
    ax.set_facecolor("white")

    # Title
    ax.text(
        total_w * 0.5,
        0.55,
        f"Model performance tiles ({target}) — page {page_idx+1}/{n_pages}",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )

    # Header row labels
    y_hdr = 1.25
    for i, name in enumerate(col_names):
        x0, x1 = x_edges[i], x_edges[i + 1]
        ax.text((x0 + x1) * 0.5, y_hdr, name, ha="center", va="center", fontsize=9.0)

    # Draw rows
    cfg_face = (0.95, 0.95, 0.95, 1)

    for r, (_, row) in enumerate(df_page.iterrows()):
        y = header_h + r * h

        draw_tile(
            ax,
            x=x_edges[0],
            y=y,
            w=col_widths[0],
            h=h,
            face=(1, 1, 1, 1),
            text=str(r + 1 + page_idx * len(df_page)),
            fontsize=7.4,
        )

        # Model (right-aligned; clipped to its tile to avoid overlap)
        draw_tile(
            ax,
            x=x_edges[1],
            y=y,
            w=col_widths[1],
            h=h,
            face=(1, 1, 1, 1),
            text=row["_model_name"],
            fontsize=7.9,
            ha="center",
            x_text_frac=0.50,
            clip_text_to_tile=True,
        )

        # Config tiles
        draw_tile(ax, x_edges[2], y, col_widths[2], h, cfg_face, text=row["_spat"], fontsize=7.2)
        draw_tile(ax, x_edges[3], y, col_widths[3], h, cfg_face, text=row["_temp"], fontsize=7.2)
        draw_tile(ax, x_edges[4], y, col_widths[4], h, cfg_face, text=row["_algo"], fontsize=6.9)
        draw_tile(ax, x_edges[5], y, col_widths[5], h, cfg_face, text=row["_agg"],  fontsize=7.2)

        # Metrics
        for j, metric in enumerate(METRICS):
            val = row.get(metric, np.nan)
            x = x_edges[6 + j]
            w = col_widths[6 + j]
            if pd.isna(val):
                face = (1, 1, 1, 1)
                txt = None
            else:
                face = cmap(METRIC_NORM[metric](float(val)))
                txt = f"{float(val):.2f}"
            draw_tile(ax, x, y, w, h, face, text=txt, fontsize=7.6)

    # Colorbars (bottom area, horizontal, side by side so each keeps its own ticks and label)
    cax1 = fig.add_axes([0.08, 0.030, 0.26, 0.012])
    cax2 = fig.add_axes([0.39, 0.030, 0.26, 0.012])
    cax3 = fig.add_axes([0.70, 0.030, 0.26, 0.012])

    sm_R   = mpl.cm.ScalarMappable(norm=METRIC_NORM["R"], cmap=cmap); sm_R.set_array([])
    sm_rho = mpl.cm.ScalarMappable(norm=METRIC_NORM["spearman_pred_obs"], cmap=cmap); sm_rho.set_array([])
    sm_R2  = mpl.cm.ScalarMappable(norm=METRIC_NORM["R2"], cmap=cmap); sm_R2.set_array([])

    cb1 = fig.colorbar(sm_R,   cax=cax1, orientation="horizontal", extend="both"); cb1.set_label("R", fontsize=8)
    cb2 = fig.colorbar(sm_rho, cax=cax2, orientation="horizontal", extend="both"); cb2.set_label("ρ", fontsize=8)
    cb3 = fig.colorbar(sm_R2,  cax=cax3, orientation="horizontal", extend="both"); cb3.set_label("R²", fontsize=8)

    cb3.set_ticks(R2_TICKS)
    for cb in (cb1, cb2, cb3):
        cb.ax.tick_params(labelsize=7)

    return fig


def main():
    if N_PAGES < 1:
        raise ValueError("N_PAGES must be >= 1")

    df = pd.read_csv(CSV_PATH)

    # Optional patching for missing config values
    for model, info in complement_info.items():
        m = df["model_folder"] == model
        for col, val in info.items():
            if col in df.columns:
                df.loc[m & df[col].isna(), col] = val

    # Ensure required config cols exist
    for col in ["spatConfig", "tempConfig", "algo", "agg"]:
        if col not in df.columns:
            df[col] = np.nan

    # Coerce numerics
    if "tempConfig" in df.columns:
        df["tempConfig"] = pd.to_numeric(df["tempConfig"], errors="coerce")
    for c in METRICS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Display columns
    df["_model_name"] = df["model_folder"].astype(str).map(model_display_name)
    df["_spat"] = df.apply(lambda r: fmt_cfg_alias("spatConfig", r["spatConfig"]), axis=1)
    df["_temp"] = df.apply(lambda r: fmt_cfg_alias("tempConfig", r["tempConfig"]), axis=1)
    df["_algo"] = df.apply(lambda r: algo_display(r["algo"]), axis=1)
    df["_agg"]  = df.apply(lambda r: fmt_cfg_alias("agg", r["agg"]), axis=1)

    # Sorting: alphabetical by model_aliases and config_aliases (as displayed)
    df["_cfg_key"] = df.apply(build_config_sort_key, axis=1)
    df = df.sort_values(by=["_model_name", "_cfg_key"], ascending=[True, True]).reset_index(drop=True)

    # Split rows into N_PAGES approximately evenly
    n = len(df)
    rows_per_page = int(math.ceil(n / N_PAGES))

    # Colormap
    cmap = LinearSegmentedColormap.from_list(
        "blue_white_rocket",
        # negative values: white -> blue (mirror of the white -> red ramp for positives)
        ["#3a64b0", "#a9c1e6", "#ffffff", "#f2c2a2", "#c43c35"]
    )

    fig = plt.figure(figsize=(8.27, 11.69))
    with PdfPages(OUT_PDF) as pdf:
        for p in range(N_PAGES):
            start = p * rows_per_page
            end = min((p + 1) * rows_per_page, n)
            if start >= n:
                break
            df_page = df.iloc[start:end]
            draw_page(df_page, p, N_PAGES, cmap, fig=fig)
            pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
