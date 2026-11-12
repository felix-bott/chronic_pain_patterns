"""Central dataset names for the CPP pipeline.

Display names follow Bott et al. (2025), "Exploring electroencephalographic chronic pain
biomarkers: a mega-analysis", eBioMedicine 120:105955, doi:10.1016/j.ebiom.2025.105955.

Raw `set` values are lookup keys: they appear in signal filenames, the CV table, the norm
tables and stored predictions. Keep using them for lookups, and use these names for anything
a reader sees. Set_Munich1 combines two raw sets.
"""

DATASET_SETS = {
    "Set_Munich1": ("RS2020", "LONG"),
    "Set_Munich2": ("ZEBHAUSER",),
    "Set_Otago1": ("ADHIA",),
    "Set_Otago2": ("ADHIA2",),
    "Set_Brisbane": ("DAY",),
    "Set_Seattle": ("JENSEN",),
    "Set_Haifa": ("YARNITSKY",),
    "Set_Boulder": ("WAGER",),
}

SET_COLORS = {
    "Set_Munich1": "#4E79A7",  # blue
    "Set_Munich2": "#F28E2B",  # orange
    "Set_Otago1": "#59A14F",  # green
    "Set_Otago2": "#8CD17D",  # light green
    "Set_Brisbane": "#E15759",  # red
    "Set_Seattle": "#B07AA1",  # purple
    "Set_Haifa": "#76B7B2",  # teal
    "Set_Boulder": "#EDC948",  # gold
}

_RAW_TO_ALIAS = {raw: alias for alias, raws in DATASET_SETS.items() for raw in raws}


def alias_for(raw_set):
    """Return the paper's name for a raw `set` value. Raises KeyError for an unknown set."""
    return _RAW_TO_ALIAS[raw_set]
