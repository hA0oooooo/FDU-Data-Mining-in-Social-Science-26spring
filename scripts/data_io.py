"""Data loading helpers for raw CEPS files."""

import pandas as pd

from config import RAW_DIR


def find_baseline_student_file():
    """Find the baseline student .dta file without hard-coding the Chinese name."""
    for path in RAW_DIR.glob("*.dta"):
        if "\u5b66\u751f" in path.name and "\u57fa\u7ebf" in path.name:
            return path
    raise FileNotFoundError("Cannot find baseline student .dta file in dataset/raw")


def find_baseline_parent_file():
    """Find the baseline parent .dta file without hard-coding the Chinese name."""
    for path in RAW_DIR.glob("*.dta"):
        if "\u5bb6\u957f" in path.name and "\u57fa\u7ebf" in path.name:
            return path
    raise FileNotFoundError("Cannot find baseline parent .dta file in dataset/raw")


def load_matched_students():
    """Load baseline, follow-up, and parent files and align them by ids."""
    baseline_path = find_baseline_student_file()
    parent_path = find_baseline_parent_file()
    followup_path = RAW_DIR / "cepsw2studentCN.dta"
    baseline = pd.read_stata(baseline_path)
    parent = pd.read_stata(parent_path)
    followup = pd.read_stata(followup_path)

    matched_ids = sorted(
        set(baseline["ids"].dropna().astype(int))
        & set(followup["ids"].dropna().astype(int))
    )
    baseline_matched = baseline[baseline["ids"].astype(int).isin(matched_ids)].copy()
    followup_matched = followup[followup["ids"].astype(int).isin(matched_ids)].copy()

    baseline_matched = (
        baseline_matched.set_index(baseline_matched["ids"].astype(int))
        .sort_index()
        .loc[matched_ids]
    )
    followup_matched = (
        followup_matched.set_index(followup_matched["ids"].astype(int))
        .sort_index()
        .loc[matched_ids]
    )
    parent_matched = parent[parent["ids"].astype(int).isin(matched_ids)].copy()
    parent_matched = (
        parent_matched.set_index(parent_matched["ids"].astype(int))
        .sort_index()
        .loc[matched_ids]
    )
    return baseline, followup, baseline_matched, followup_matched, parent_matched, matched_ids
