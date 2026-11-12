"""Write CPP_results/DATASET_ALIASES.md: raw set values, the paper's names and sample sizes.

Fails if a `set` value in the CV table has no alias.
"""
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from dataset_names import DATASET_SETS, alias_for
from paths import CPP_RESULTS, CV_MAIN_TABLE

import pandas as pd

# Participants analysed per dataset, from Table 1 of the paper.
PAPER_N = {"Set_Munich1": 119, "Set_Munich2": 63, "Set_Brisbane": 60, "Set_Otago1": 57,
           "Set_Otago2": 77, "Set_Boulder": 68, "Set_Haifa": 47, "Set_Seattle": 123}


def main():
    cv = pd.read_csv(CV_MAIN_TABLE)
    aliases = cv["set"].map(alias_for)  # raises KeyError for an unmapped set
    counts = aliases.value_counts()
    missing = set(DATASET_SETS) - set(counts.index)
    if missing:
        sys.exit(f"Aliases with no participants in the CV table: {sorted(missing)}")

    lines = ["# Dataset names", "",
             "Names follow Bott et al. (2025), eBioMedicine 120:105955, doi:10.1016/j.ebiom.2025.105955. "
             "The mapping is defined in `CPP_code/dataset_names.py`.", "",
             "Raw `set` values remain in signal filenames, the CV table, the norm tables and stored predictions, "
             "because they are lookup keys.", "",
             "| Paper name | Raw `set` | n in this project | n in paper Table 1 |", "|---|---|---|---|"]
    for alias, raws in DATASET_SETS.items():
        lines.append(f"| {alias} | {' + '.join(raws)} | {counts[alias]} | {PAPER_N[alias]} |")
    lines += [f"| **Total** | | **{len(cv)}** | **{sum(PAPER_N.values())}** |", "",
              "The paper does not link raw names to its datasets. This mapping comes from the project's earlier plot "
              "scripts, and participant ID prefixes and diagnosis mix are consistent with it.", "",
              "Sample sizes differ between this project and the paper. The reason has not been established.", ""]
    out = os.path.join(CPP_RESULTS, "DATASET_ALIASES.md")
    with open(out, "w") as fh:
        fh.write("\n".join(lines))
    print("Saved", out)


if __name__ == "__main__":
    main()
