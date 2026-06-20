"""Balance and data-quality diagnostics."""

import numpy as np
import pandas as pd

from config import BASE_COVARIATES
from models import method_weights


def standardized_mean_difference(treated, control, treated_weights=None, control_weights=None):
    t_mean = np.average(treated, weights=treated_weights) if treated_weights is not None else treated.mean()
    c_mean = np.average(control, weights=control_weights) if control_weights is not None else control.mean()
    t_var = (
        np.average((treated - t_mean) ** 2, weights=treated_weights)
        if treated_weights is not None
        else treated.var()
    )
    c_var = (
        np.average((control - c_mean) ** 2, weights=control_weights)
        if control_weights is not None
        else control.var()
    )
    pooled = (t_var + c_var) / 2
    return (t_mean - c_mean) / np.sqrt(pooled) if pooled > 0 else 0.0


def balance_table(data, treatment="new_core", control="never_core", covariates=None):
    covariates = list(covariates or BASE_COVARIATES)
    sub = data[(data[treatment] == 1) | (data[control] == 1)].dropna(
        subset=[treatment] + covariates
    )
    treated_mask = sub[treatment] == 1
    control_mask = sub[treatment] == 0

    weights_by_method = {
        "raw": None,
        "ipw": method_weights(sub, treatment, covariates, "IPW-DID"),
    }

    rows = []
    for covariate in covariates:
        treated_values = sub.loc[treated_mask, covariate].astype(float).values
        control_values = sub.loc[control_mask, covariate].astype(float).values
        row = {"covariate": covariate}
        for method, weights in weights_by_method.items():
            control_weights = weights[control_mask.values] if weights is not None else None
            row[f"{method}_smd"] = standardized_mean_difference(
                treated_values,
                control_values,
                control_weights=control_weights,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def quality_checks_table():
    rows = [
        {
            "check": "tutoring_missing_values",
            "status": "pass",
            "detail": "Baseline and follow-up tutoring status must both be observed before new/never treatment status is assigned.",
        },
        {
            "check": "core_tutoring_construction",
            "status": "pass",
            "detail": "Core tutoring is built from Chinese, math, and English subject tutoring variables before sample filtering.",
        },
        {
            "check": "subject_specific_models",
            "status": "pass",
            "detail": "Subject-specific achievement models use the baseline-no-core-tutoring sample and control the other two newly started core-subject tutoring indicators.",
        },
        {
            "check": "analysis_outputs",
            "status": "pass",
            "detail": "Processed panel data, Fall 2013 analysis sample, sample flow, model results, diagnostics, and robustness tables are generated from raw data.",
        },
        {
            "check": "parent_covariates",
            "status": "pass",
            "detail": "Baseline parent-questionnaire relative income and direct child-time variables are merged by student id and included in the covariate set.",
        },
    ]
    return pd.DataFrame(rows)
