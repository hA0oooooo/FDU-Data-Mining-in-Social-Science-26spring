"""Robustness checks for the tutoring analysis."""

import numpy as np
import pandas as pd

from models import clustered_regression, covariates_for_model


ROBUSTNESS_SPECS = [
    {
        "check": "sample_restriction",
        "analysis": "math_tutoring_to_math_score",
        "outcome": "delta_math",
        "treatment": "new_math",
        "control": "never_core",
        "sample_treatment": "new_core",
        "sample_control": "never_core",
        "extra_covariates": ["z_math_bl"],
        "treatment_controls": ["new_chinese", "new_english"],
    },
    {
        "check": "sample_restriction",
        "analysis": "core_tutoring_to_total_score",
        "outcome": "delta_total",
        "treatment": "new_core",
        "control": "never_core",
        "extra_covariates": [],
    },
    {
        "check": "sample_restriction",
        "analysis": "core_tutoring_to_cognition",
        "outcome": "delta_cog",
        "treatment": "new_core",
        "control": "never_core",
        "extra_covariates": [],
    },
    {
        "check": "sample_restriction",
        "analysis": "core_tutoring_to_negative_emotion",
        "outcome": "delta_neg_emo",
        "treatment": "new_core",
        "control": "never_core",
        "extra_covariates": ["neg_emo_bl"],
    },
]

VALUE_ADDED_SPECS = [
    {
        "check": "value_added",
        "analysis": "math_tutoring_to_math_score",
        "outcome": "z_math_fu",
        "treatment": "new_math",
        "control": "never_core",
        "sample_treatment": "new_core",
        "sample_control": "never_core",
        "extra_covariates": ["z_math_bl"],
        "treatment_controls": ["new_chinese", "new_english"],
    },
    {
        "check": "value_added",
        "analysis": "core_tutoring_to_total_score",
        "outcome": "z_total_fu",
        "treatment": "new_core",
        "control": "never_core",
        "extra_covariates": ["z_total_bl"],
    },
    {
        "check": "value_added",
        "analysis": "core_tutoring_to_cognition",
        "outcome": "z_cog_fu",
        "treatment": "new_core",
        "control": "never_core",
        "extra_covariates": ["z_cog_bl"],
    },
    {
        "check": "value_added",
        "analysis": "core_tutoring_to_negative_emotion",
        "outcome": "neg_emo_fu",
        "treatment": "new_core",
        "control": "never_core",
        "extra_covariates": ["neg_emo_bl"],
    },
]

PLACEBO_SPECS = [
    {
        "check": "cross_subject_placebo",
        "analysis": "math_tutoring_to_chinese_score",
        "outcome": "delta_chinese",
        "treatment": "new_math",
        "control": "never_core",
        "sample_treatment": "new_core",
        "sample_control": "never_core",
        "extra_covariates": ["z_chinese_bl"],
        "treatment_controls": ["new_chinese", "new_english"],
    },
    {
        "check": "cross_subject_placebo",
        "analysis": "math_tutoring_to_english_score",
        "outcome": "delta_english",
        "treatment": "new_math",
        "control": "never_core",
        "sample_treatment": "new_core",
        "sample_control": "never_core",
        "extra_covariates": ["z_english_bl"],
        "treatment_controls": ["new_chinese", "new_english"],
    },
]


def run_subject_overlap_checks(fall_sample):
    data = fall_sample.copy()
    rows = []
    data["new_math_only"] = (
        (data["new_math"] == 1)
        & (data["new_chinese"] != 1)
        & (data["new_english"] != 1)
    ).astype(float)
    math_only = data[(data["new_math_only"] == 1) | (data["never_core"] == 1)].copy()
    for outcome, baseline in [
        ("delta_math", "z_math_bl"),
        ("delta_chinese", "z_chinese_bl"),
        ("delta_english", "z_english_bl"),
    ]:
        row = run_one(
            math_only,
            "math_only_specificity",
            f"math_only_to_{outcome}",
            outcome,
            "new_math_only",
            "never_core",
            [baseline],
            "fall2013",
        )
        if row is not None:
            rows.append(row)

    data["new_subject_count"] = data[["new_math", "new_chinese", "new_english"]].sum(
        axis=1,
        skipna=False,
    )
    data["new_single_subject"] = (data["new_subject_count"] == 1).astype(float)
    data["new_multi_subject"] = (data["new_subject_count"] >= 2).astype(float)
    combo = data[(data["never_core"] == 1) | (data["new_core"] == 1)].copy()
    for outcome, baseline in [
        ("delta_total", "z_total_bl"),
        ("delta_math", "z_math_bl"),
        ("delta_chinese", "z_chinese_bl"),
        ("delta_english", "z_english_bl"),
    ]:
        covs = [] if outcome == "delta_total" else [baseline]
        for treatment in ["new_single_subject", "new_multi_subject"]:
            row = run_one(
                combo,
                "single_vs_multi_subject",
                f"{treatment}_to_{outcome}",
                outcome,
                treatment,
                "never_core",
                covs,
                "fall2013",
            )
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows)


def subject_combination_summary(fall_sample):
    data = fall_sample[(fall_sample["new_core"] == 1) | (fall_sample["never_core"] == 1)].copy()
    data["combo"] = "never_core"
    data.loc[
        (data["new_math"] == 1) & (data["new_chinese"] != 1) & (data["new_english"] != 1),
        "combo",
    ] = "math_only"
    data.loc[
        (data["new_math"] != 1) & (data["new_chinese"] == 1) & (data["new_english"] != 1),
        "combo",
    ] = "chinese_only"
    data.loc[
        (data["new_math"] != 1) & (data["new_chinese"] != 1) & (data["new_english"] == 1),
        "combo",
    ] = "english_only"
    data.loc[
        (data["new_math"] == 1) & (data["new_chinese"] == 1) & (data["new_english"] != 1),
        "combo",
    ] = "math_chinese"
    data.loc[
        (data["new_math"] == 1) & (data["new_chinese"] != 1) & (data["new_english"] == 1),
        "combo",
    ] = "math_english"
    data.loc[
        (data["new_math"] != 1) & (data["new_chinese"] == 1) & (data["new_english"] == 1),
        "combo",
    ] = "chinese_english"
    data.loc[
        (data["new_math"] == 1) & (data["new_chinese"] == 1) & (data["new_english"] == 1),
        "combo",
    ] = "all_three"

    new_core = data[data["new_core"] == 1]
    rows = []
    for combo, sub in data.groupby("combo"):
        rows.append(
            {
                "combo": combo,
                "n": len(sub),
                "share_of_core_sample": len(sub) / len(data),
                "share_of_new_core": len(sub) / len(new_core) if combo != "never_core" else pd.NA,
                "z_total_bl_mean": sub["z_total_bl"].mean(),
                "z_cog_bl_mean": sub["z_cog_bl"].mean(),
                "father_edu_mean": sub["father_edu"].mean(),
                "mother_edu_mean": sub["mother_edu"].mean(),
                "family_econ_mean": sub["family_econ"].mean(),
                "delta_total_mean": sub["delta_total"].mean(),
                "delta_math_mean": sub["delta_math"].mean(),
                "delta_chinese_mean": sub["delta_chinese"].mean(),
                "delta_english_mean": sub["delta_english"].mean(),
            }
        )
    return pd.DataFrame(rows)


def run_one(
    data,
    check,
    analysis,
    outcome,
    treatment,
    control,
    extra_covariates,
    sample,
    treatment_controls=None,
    sample_treatment=None,
    sample_control=None,
):
    sample_treatment = sample_treatment or treatment
    sample_control = sample_control or control
    treatment_controls = list(treatment_controls or [])
    model_data = data[
        (data[sample_treatment] == 1) | (data[sample_control] == 1)
    ].copy()
    result = clustered_regression(
        model_data,
        outcome,
        treatment,
        covariates_for_model(extra_covariates, treatment_controls),
        method="DID",
    )
    if result is None:
        return None
    return {
        "check": check,
        "analysis": analysis,
        "sample": sample,
        "outcome": outcome,
        "treatment": treatment,
        "treatment_controls": ";".join(treatment_controls),
        **result,
    }


def run_sample_restriction(panel, fall_sample):
    rows = []
    for spec in ROBUSTNESS_SPECS:
        rows.append(run_one(fall_sample, sample="fall2013", **spec))
        rows.append(run_one(panel, sample="all_matched", **spec))
    return [row for row in rows if row is not None]


def run_value_added(fall_sample):
    rows = []
    for spec in VALUE_ADDED_SPECS:
        rows.append(run_one(fall_sample, sample="fall2013", **spec))
    return [row for row in rows if row is not None]


def run_placebo(fall_sample):
    rows = []
    for spec in PLACEBO_SPECS:
        rows.append(run_one(fall_sample, sample="fall2013", **spec))
    return [row for row in rows if row is not None]


def benjamini_hochberg(main_results):
    did = main_results[main_results["method"] == "DID"].copy()
    did = did.sort_values("pvalue").reset_index(drop=True)
    m = len(did)
    q_values = did["pvalue"].to_numpy() * m / (np.arange(m) + 1)
    for i in range(m - 2, -1, -1):
        q_values[i] = min(q_values[i], q_values[i + 1])
    did["bh_qvalue"] = np.minimum(q_values, 1.0)
    return did[["analysis", "outcome", "treatment", "pvalue", "bh_qvalue"]]


def run_robustness(panel, fall_sample, main_results):
    rows = []
    rows.extend(run_sample_restriction(panel, fall_sample))
    rows.extend(run_value_added(fall_sample))
    rows.extend(run_placebo(fall_sample))
    robustness = pd.DataFrame(rows)
    multiple_testing = benjamini_hochberg(main_results)
    subject_overlap = run_subject_overlap_checks(fall_sample)
    combination_summary = subject_combination_summary(fall_sample)
    return robustness, multiple_testing, subject_overlap, combination_summary
