"""Shared variable construction utilities."""

import numpy as np
import pandas as pd

from config import YES


def fix_label(label):
    """Decode CEPS labels when pandas exposes mojibake text."""
    if label is None:
        return ""
    try:
        return str(label).encode("latin-1").decode("gbk")
    except Exception:
        return str(label)


def cat_code(col, text):
    """Return categorical code whose decoded label contains text."""
    if not hasattr(col, "cat"):
        return None
    for code, label in enumerate(col.cat.categories):
        if text in fix_label(label):
            return code
    return None


def binary_yes(col):
    """Map a yes/no categorical variable to 1=yes, 0=not yes, NaN=missing."""
    yes = cat_code(col, YES)
    out = pd.Series(np.nan, index=col.index, dtype=float)
    observed = col.notna()
    out.loc[observed] = (col.cat.codes[observed] == yes).astype(float)
    return out


def no_tutoring_question_to_tutoring(col):
    """CEPS any-tutoring item is phrased as no tutoring: yes means no."""
    yes = cat_code(col, YES)
    out = pd.Series(np.nan, index=col.index, dtype=float)
    observed = col.notna()
    out.loc[observed] = (col.cat.codes[observed] != yes).astype(float)
    return out


def subject_or(df, variables):
    """Any positive response across subject-specific tutoring items."""
    items = [binary_yes(df[v]) for v in variables]
    mat = pd.concat(items, axis=1)
    out = mat.max(axis=1, skipna=True)
    out[mat.notna().sum(axis=1) == 0] = np.nan
    return out.astype(float)


def zscore(series):
    s = pd.to_numeric(series, errors="coerce").astype(float)
    return (s - s.mean()) / s.std()


def categorical_float(col):
    if hasattr(col, "cat"):
        out = col.cat.codes.astype(float)
        out[out < 0] = np.nan
        return out
    return pd.to_numeric(col, errors="coerce")
