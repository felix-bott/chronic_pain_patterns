"""Compare recreated combined tables with reference tables, column by column.

Exits with an error if a table is missing, the rows differ, or a value differs by more than the
tolerance. Metrics are compared absolutely, Bayes factors relative to their size.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

METRIC_TOLERANCE = 1e-6
BAYES_FACTOR_TOLERANCE = 1e-3  # relative
LIST_COLUMNS = ("predictions", "observations")


def compare_column(name, reference, recreated):
    """Largest difference in a column, or None for a text column that matches."""
    if name in LIST_COLUMNS:
        pairs = [(np.array(json.loads(a)), np.array(json.loads(b))) for a, b in zip(reference, recreated)]
        if any(len(a) != len(b) for a, b in pairs):
            return np.inf
        return max(np.abs(a - b).max() for a, b in pairs)
    a, b = pd.to_numeric(reference, errors="coerce"), pd.to_numeric(recreated, errors="coerce")
    if a.notna().sum() == 0 and b.notna().sum() == 0:
        same = reference.fillna("").astype(str).to_numpy() == recreated.fillna("").astype(str).to_numpy()
        return None if same.all() else np.inf
    if (a.isna() != b.isna()).any():
        return np.inf
    difference = (a - b).abs()
    if name == "correlationBF":
        difference = difference / a.abs().clip(lower=1e-300)
    return float(difference.max())


def compare_tables(reference_path, recreated_path):
    reference, recreated = pd.read_csv(reference_path), pd.read_csv(recreated_path)
    if len(reference) != len(recreated):
        return [f"{len(reference)} rows against {len(recreated)}"], {}
    problems, largest = [], {}
    for column in reference.columns:
        if column not in recreated.columns:
            problems.append(f"column {column} missing")
            continue
        difference = compare_column(column, reference[column], recreated[column])
        if difference is None:
            continue
        family = "correlationBF" if column == "correlationBF" else (column if column in LIST_COLUMNS else "metrics")
        largest[family] = max(largest.get(family, 0.0), difference)
        tolerance = BAYES_FACTOR_TOLERANCE if column == "correlationBF" else METRIC_TOLERANCE
        if difference > tolerance:
            problems.append(f"{column} differs by {difference}")
    return problems, largest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="folder with analysis_*/tables")
    parser.add_argument("--recreated", required=True, help="folder with analysis_*/tables")
    args = parser.parse_args()
    paths = sorted(glob.glob(os.path.join(args.reference, "analysis_*", "tables", "combined_*.csv")))
    if not paths:
        sys.exit(f"No tables in {args.reference}")
    failed = 0
    for path in paths:
        relative = os.path.relpath(path, args.reference)
        recreated = os.path.join(args.recreated, relative)
        problems, largest = compare_tables(path, recreated) if os.path.exists(recreated) else (["missing"], {})
        summary = ", ".join(f"{k} {v:.1e}" for k, v in sorted(largest.items()))
        print(f"{'FAIL' if problems else 'ok  '} {relative}  max difference: {summary}")
        for problem in problems:
            print("     ", problem)
        failed += bool(problems)
    if failed:
        sys.exit(f"{failed} of {len(paths)} tables differ")
    print(f"All {len(paths)} tables match")


if __name__ == "__main__":
    main()
