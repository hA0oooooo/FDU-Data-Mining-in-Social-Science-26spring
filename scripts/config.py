"""Project configuration for the CEPS tutoring analysis."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "dataset" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

YES = "\u662f"
MALE = "\u7537"
HAN = "\u6c49"

SUBJECT_TUTORING_VARS = {
    "math": {
        "baseline": ["b1901", "b1902"],
        "followup": ["w2b1101", "w2b1102"],
    },
    "chinese": {
        "baseline": ["b1903"],
        "followup": ["w2b1103"],
    },
    "english": {
        "baseline": ["b1904"],
        "followup": ["w2b1104"],
    },
}

BASE_COVARIATES = [
    "z_total_bl",
    "z_cog_bl",
    "male",
    "age",
    "han",
    "only_child",
    "father_edu",
    "mother_edu",
    "family_econ",
    "parent_income_relative",
    "parent_child_time",
]

MODEL_SPECS = [
    {
        "analysis": "math_tutoring_to_math_score",
        "outcome": "delta_math",
        "treatment": "new_math",
        "control": "never_core",
        "sample_treatment": "new_core",
        "sample_control": "never_core",
        "sample": "baseline_no_core",
        "extra_covariates": ["z_math_bl"],
        "treatment_controls": ["new_chinese", "new_english"],
    },
    {
        "analysis": "english_tutoring_to_english_score",
        "outcome": "delta_english",
        "treatment": "new_english",
        "control": "never_core",
        "sample_treatment": "new_core",
        "sample_control": "never_core",
        "sample": "baseline_no_core",
        "extra_covariates": ["z_english_bl"],
        "treatment_controls": ["new_math", "new_chinese"],
    },
    {
        "analysis": "chinese_tutoring_to_chinese_score",
        "outcome": "delta_chinese",
        "treatment": "new_chinese",
        "control": "never_core",
        "sample_treatment": "new_core",
        "sample_control": "never_core",
        "sample": "baseline_no_core",
        "extra_covariates": ["z_chinese_bl"],
        "treatment_controls": ["new_math", "new_english"],
    },
    {
        "analysis": "any_extracurricular_to_total_score",
        "outcome": "delta_total",
        "treatment": "new_any",
        "control": "never_any",
        "extra_covariates": [],
    },
    {
        "analysis": "core_tutoring_to_total_score",
        "outcome": "delta_total",
        "treatment": "new_core",
        "control": "never_core",
        "extra_covariates": [],
    },
    {
        "analysis": "core_tutoring_to_cognition",
        "outcome": "delta_cog",
        "treatment": "new_core",
        "control": "never_core",
        "extra_covariates": [],
    },
    {
        "analysis": "core_tutoring_to_negative_emotion",
        "outcome": "delta_neg_emo",
        "treatment": "new_core",
        "control": "never_core",
        "extra_covariates": ["neg_emo_bl"],
    },
    {
        "analysis": "core_tutoring_to_high_negative_emotion",
        "outcome": "high_neg_emo_fu_25",
        "treatment": "new_core",
        "control": "never_core",
        "extra_covariates": ["high_neg_emo_bl_25"],
    },
]
