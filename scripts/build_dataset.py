"""Build analysis panel and Fall 2013 sample."""

import pandas as pd

from config import HAN, MALE, SUBJECT_TUTORING_VARS
from variable_utils import (
    binary_yes,
    cat_code,
    categorical_float,
    no_tutoring_question_to_tutoring,
    subject_or,
    zscore,
)


def add_tutoring_variables(df, baseline, followup):
    df["any_bl"] = no_tutoring_question_to_tutoring(baseline["b1900"])
    df["any_fu"] = no_tutoring_question_to_tutoring(followup["w2b1100"])

    for subject, spec in SUBJECT_TUTORING_VARS.items():
        df[f"{subject}_bl"] = subject_or(baseline, spec["baseline"])
        df[f"{subject}_fu"] = subject_or(followup, spec["followup"])

    df["core_bl"] = df[["math_bl", "chinese_bl", "english_bl"]].max(axis=1, skipna=True)
    df["core_fu"] = df[["math_fu", "chinese_fu", "english_fu"]].max(axis=1, skipna=True)

    for base in ["any", "core", "math", "chinese", "english"]:
        df[f"new_{base}"] = ((df[f"{base}_bl"] == 0) & (df[f"{base}_fu"] == 1)).astype(float)
        df[f"never_{base}"] = ((df[f"{base}_bl"] == 0) & (df[f"{base}_fu"] == 0)).astype(float)
        missing = df[f"{base}_bl"].isna() | df[f"{base}_fu"].isna()
        df.loc[missing, [f"new_{base}", f"never_{base}"]] = pd.NA


def add_outcomes(df, baseline, followup):
    df["z_chinese_bl"] = (pd.to_numeric(baseline["stdchn"], errors="coerce") - 70) / 10
    df["z_math_bl"] = (pd.to_numeric(baseline["stdmat"], errors="coerce") - 70) / 10
    df["z_english_bl"] = (pd.to_numeric(baseline["stdeng"], errors="coerce") - 70) / 10

    for subject, raw, full in [
        ("chinese", "w2chn", "w2upchn"),
        ("math", "w2mat", "w2upmat"),
        ("english", "w2eng", "w2upeng"),
    ]:
        pct = (
            pd.to_numeric(followup[raw], errors="coerce")
            / pd.to_numeric(followup[full], errors="coerce")
            * 100
        )
        df[f"z_{subject}_fu"] = zscore(pct)

    df["z_total_bl"] = df[["z_chinese_bl", "z_math_bl", "z_english_bl"]].mean(axis=1)
    df["z_total_fu"] = df[["z_chinese_fu", "z_math_fu", "z_english_fu"]].mean(axis=1)
    for subject in ["chinese", "math", "english", "total"]:
        df[f"delta_{subject}"] = df[f"z_{subject}_fu"] - df[f"z_{subject}_bl"]

    df["z_cog_bl"] = pd.to_numeric(baseline["cog3pl"], errors="coerce")
    df["z_cog_fu"] = pd.to_numeric(followup["w2cog3pl"], errors="coerce")
    df["delta_cog"] = df["z_cog_fu"] - df["z_cog_bl"]

    baseline_emotion = pd.concat(
        [zscore(categorical_float(baseline[v])) for v in ["a1801", "a1802", "a1803"]],
        axis=1,
    )
    followup_emotion = pd.concat(
        [
            zscore(categorical_float(followup[v]))
            for v in ["w2c2501", "w2c2503", "w2c2506", "w2c2507", "w2c2508"]
        ],
        axis=1,
    )
    df["neg_emo_bl"] = baseline_emotion.mean(axis=1)
    df.loc[baseline_emotion.notna().sum(axis=1) < 2, "neg_emo_bl"] = pd.NA
    df["neg_emo_fu"] = followup_emotion.mean(axis=1)
    df.loc[followup_emotion.notna().sum(axis=1) < 3, "neg_emo_fu"] = pd.NA
    df["delta_neg_emo"] = df["neg_emo_fu"] - df["neg_emo_bl"]
    df["high_neg_emo_bl_25"] = (
        df["neg_emo_bl"] >= df["neg_emo_bl"].dropna().quantile(0.75)
    ).astype(float)
    df.loc[df["neg_emo_bl"].isna(), "high_neg_emo_bl_25"] = pd.NA
    df["high_neg_emo_fu_25"] = (
        df["neg_emo_fu"] >= df["neg_emo_fu"].dropna().quantile(0.75)
    ).astype(float)
    df.loc[df["neg_emo_fu"].isna(), "high_neg_emo_fu_25"] = pd.NA


def add_controls(df, baseline, parent):
    df["male"] = (baseline["a01"].cat.codes == cat_code(baseline["a01"], MALE)).astype(float)
    df.loc[baseline["a01"].isna(), "male"] = pd.NA
    df["age"] = 2013 - pd.to_numeric(baseline["a02a"], errors="coerce")
    df["han"] = (baseline["a03"].cat.codes == cat_code(baseline["a03"], HAN)).astype(float)
    df.loc[baseline["a03"].isna(), "han"] = pd.NA
    df["only_child"] = binary_yes(baseline["b01"])
    df["father_edu"] = categorical_float(baseline["b07"])
    df["mother_edu"] = categorical_float(baseline["b06"])
    df["family_econ"] = categorical_float(baseline["b09"])
    df["parent_income_relative"] = categorical_float(parent["be21"])
    df["parent_child_time"] = categorical_float(parent["ba13"])


def build_analysis_panel(baseline, followup, parent):
    fall_code = cat_code(baseline["fall"], "2013")
    fall = baseline["fall"].cat.codes == fall_code

    df = pd.DataFrame(index=baseline.index)
    df["ids"] = baseline.index
    df["schids"] = baseline["schids"].values
    df["fall2013"] = fall.astype(int)

    add_tutoring_variables(df, baseline, followup)
    add_outcomes(df, baseline, followup)
    add_controls(df, baseline, parent)
    return df, df[df["fall2013"] == 1].copy()
