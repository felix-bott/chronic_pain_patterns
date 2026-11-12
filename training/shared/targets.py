"""Which CV-table column holds the target for a given fold.

Targets ending in "RAW" are read unnormalised from their raw column ("ageRAW" -> "age",
"painVarsRAW" -> "painVars"). All other targets are read from the fold's normalised column,
e.g. ("Fold3", "age") -> "Fold3_NORMage".
"""


def target_column(fold, target):
    if target.endswith("RAW"):
        return target[: -len("RAW")]
    return f"{fold}_NORM{target}"
