"""NEW script (no original generator was found). Reconstructs LEGEND.png.

The original LEGEND.png is not a standalone legend. It is an earlier version of the
age KDE figure with legends and density axis labels enabled, and serves as the color key
for the legend-free KDE panels. This script rebuilds that figure from the CV table with
the same KDE logic as plot_kde_distributions.py. Figure size is estimated from the original image.
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from paths import CV_MAIN_TABLE, figures_dir
from dataset_names import SET_COLORS as COLORS, alias_for
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

BW = 0.4

cv = pd.read_csv(CV_MAIN_TABLE)
cv["set_alias"] = cv["set"].map(alias_for)
norm_cols = [c for c in cv.columns if c.startswith("Fold") and c.endswith("NORMage")]
cv["age_norm"] = cv[norm_cols].mean(axis=1)

fig, axes = plt.subplots(2, 1, figsize=(10.1, 8.1), constrained_layout=True)
for ax, col, xlim, xlab in [(axes[0], "age", (0, 100), "age"),
                            (axes[1], "age_norm", (-4, 4), "Fold-mean normalized age")]:
    grid = np.linspace(*xlim, 400)
    for alias in sorted(COLORS):
        vals = cv.loc[cv["set_alias"] == alias, col].dropna().to_numpy(float)
        dens = gaussian_kde(vals, bw_method=BW)(grid)
        ax.fill_between(grid, dens, color=COLORS[alias], alpha=0.4, label=f"{alias} (n={len(vals)})")
        ax.plot(grid, dens, color=COLORS[alias], alpha=0.4)
        ax.axvline(vals.mean(), linestyle="--", linewidth=3.5, color=COLORS[alias])
    ax.set_xlim(*xlim); ax.set_xlabel(xlab); ax.set_ylabel("Density"); ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
out = os.path.join(figures_dir("main"), "LEGEND.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print("Saved", out)
