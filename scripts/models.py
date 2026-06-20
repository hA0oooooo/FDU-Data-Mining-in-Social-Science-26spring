"""Model estimation for the tutoring analysis."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

from config import BASE_COVARIATES, MODEL_SPECS


def confidence_interval(coef, se, level=0.95):
    z = norm.ppf(1 - (1 - level) / 2)
    return coef - z * se, coef + z * se


def method_weights(sub, treatment, covariates, method):
    treated = sub[treatment].astype(float).values
    if method == "DID":
        return None
    if method == "IPW-DID":
        ps_x = sm.add_constant(sub[covariates].astype(float))
        ps = sm.Logit(treated, ps_x).fit(disp=False).predict(ps_x)
        weights = np.where(treated == 1, 1.0, ps / (1 - ps))
        weights = np.minimum(weights, np.percentile(weights[treated == 0], 99))
        return weights
    raise ValueError(f"Unknown method: {method}")


def doubly_robust_did(data, outcome, treatment, covariates):
    cols = [outcome, treatment, "schids"] + covariates
    sub = data[cols].dropna().copy()
    treated = sub[treatment].astype(float).values
    y = sub[outcome].astype(float).values
    x_cov = sub[covariates].astype(float).values
    control_mask = treated == 0

    ps_model = sm.Logit(treated, sm.add_constant(x_cov)).fit(disp=False)
    ps = np.clip(ps_model.predict(sm.add_constant(x_cov)), 0.01, 0.99)

    school_dummies = pd.get_dummies(sub["schids"], drop_first=True, dtype=float).values
    x_control = np.column_stack(
        [np.ones(control_mask.sum()), x_cov[control_mask], school_dummies[control_mask]]
    )
    outcome_model = sm.OLS(y[control_mask], x_control).fit()
    x_all = np.column_stack([np.ones(len(sub)), x_cov, school_dummies])
    m0 = outcome_model.predict(x_all)

    control_weight = ps * (1 - treated) / (1 - ps)
    score_weight = treated - control_weight
    att = np.mean(score_weight * (y - m0)) / np.mean(treated)
    influence = (score_weight * (y - m0) - treated * att) / np.mean(treated)
    cluster = sub["schids"].values
    cluster_sums = pd.Series(influence).groupby(cluster).sum().to_numpy()
    n_clusters = len(cluster_sums)
    n_obs = len(influence)
    se = np.sqrt(
        (n_clusters / (n_clusters - 1))
        * np.sum(cluster_sums**2)
        / (n_obs**2)
    )
    pvalue = 2 * (1 - norm.cdf(abs(att / se)))
    ci_lower, ci_upper = confidence_interval(att, se)
    return {
        "method": "DR-DID",
        "coef": att,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "pvalue": pvalue,
        "n": len(sub),
        "treated": int(treated.sum()),
        "control": int((treated == 0).sum()),
        "schools": int(sub["schids"].nunique()),
    }


def clustered_regression(data, outcome, treatment, covariates, method="DID"):
    if method == "DR-DID":
        return doubly_robust_did(data, outcome, treatment, covariates)

    cols = [outcome, treatment, "schids"] + covariates
    sub = data[cols].dropna().copy()
    treated = sub[treatment].astype(float).values
    school_dummies = pd.get_dummies(sub["schids"], drop_first=True, dtype=float)
    x = sm.add_constant(
        np.column_stack(
            [treated]
            + [sub[c].astype(float).values for c in covariates]
            + [school_dummies.values]
        )
    )
    y = sub[outcome].astype(float).values

    weights = method_weights(sub, treatment, covariates, method)
    if weights is not None:
        model = sm.WLS(y, x, weights=weights)
    else:
        model = sm.OLS(y, x)

    result = model.fit(cov_type="cluster", cov_kwds={"groups": sub["schids"].values})
    ci_lower, ci_upper = confidence_interval(result.params[1], result.bse[1])
    return {
        "method": method,
        "coef": result.params[1],
        "se": result.bse[1],
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "pvalue": result.pvalues[1],
        "n": len(sub),
        "treated": int(treated.sum()),
        "control": int((treated == 0).sum()),
        "schools": int(sub["schids"].nunique()),
    }


def covariates_for_model(extra_covariates=None, treatment_controls=None):
    extra_covariates = list(extra_covariates or [])
    treatment_controls = list(treatment_controls or [])
    if not extra_covariates and not treatment_controls:
        return list(BASE_COVARIATES)
    subject_baselines = {"z_math_bl", "z_chinese_bl", "z_english_bl"}
    base = list(BASE_COVARIATES)
    if any(c in subject_baselines for c in extra_covariates):
        base = [c for c in base if c != "z_total_bl"]
    for covariate in extra_covariates + treatment_controls:
        if covariate not in base:
            base.append(covariate)
    return base


def model_sample(data, spec):
    sample_treatment = spec.get("sample_treatment", spec["treatment"])
    sample_control = spec.get("sample_control", spec["control"])
    return data[(data[sample_treatment] == 1) | (data[sample_control] == 1)].copy()


def run_main_models(fall_sample):
    rows = []
    for spec in MODEL_SPECS:
        model_data = model_sample(fall_sample, spec)
        treatment_controls = spec.get("treatment_controls", [])
        covariates = covariates_for_model(
            spec["extra_covariates"],
            treatment_controls,
        )
        for method in ["DID", "IPW-DID", "DR-DID"]:
            result = clustered_regression(
                model_data,
                spec["outcome"],
                spec["treatment"],
                covariates,
                method=method,
            )
            if result is None:
                continue
            rows.append(
                {
                    "analysis": spec["analysis"],
                    "sample": spec.get("sample", "primary"),
                    "outcome": spec["outcome"],
                    "treatment": spec["treatment"],
                    "treatment_controls": ";".join(treatment_controls),
                    **result,
                }
            )
    return pd.DataFrame(rows)
