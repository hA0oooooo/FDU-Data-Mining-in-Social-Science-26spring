"""Output table construction."""

import pandas as pd

from config import OUTPUT_DIR, PROCESSED_DIR
from diagnostics import quality_checks_table


def sample_flow_table(baseline_total, matched_ids, fall_sample):
    any_valid = fall_sample["any_bl"].notna() & fall_sample["any_fu"].notna()
    core_valid = fall_sample["core_bl"].notna() & fall_sample["core_fu"].notna()
    rows = [
        ("baseline_total", baseline_total),
        ("baseline_grade7_matched", len(matched_ids)),
        ("fall2013_matched", int(fall_sample.shape[0])),
        (
            "fall2013_any_two_wave_nonmissing",
            int(any_valid.sum()),
        ),
        ("baseline_any_tutoring", int((fall_sample.loc[any_valid, "any_bl"] == 1).sum())),
        ("followup_any_tutoring", int((fall_sample.loc[any_valid, "any_fu"] == 1).sum())),
        (
            "fall2013_core_two_wave_nonmissing",
            int(core_valid.sum()),
        ),
        ("baseline_core_tutoring", int((fall_sample.loc[core_valid, "core_bl"] == 1).sum())),
        ("followup_core_tutoring", int((fall_sample.loc[core_valid, "core_fu"] == 1).sum())),
        ("new_any_tutoring", int(fall_sample["new_any"].sum())),
        ("never_any_tutoring", int(fall_sample["never_any"].sum())),
        ("new_core_tutoring", int(fall_sample["new_core"].sum())),
        ("never_core_tutoring", int(fall_sample["never_core"].sum())),
        ("new_math_tutoring", int(fall_sample["new_math"].sum())),
        ("new_chinese_tutoring", int(fall_sample["new_chinese"].sum())),
        ("new_english_tutoring", int(fall_sample["new_english"].sum())),
    ]
    return pd.DataFrame(rows, columns=["step", "n"])


def write_all_outputs(
    panel,
    fall_sample,
    baseline_total,
    matched_ids,
    model_results,
    balance_diagnostics,
    robustness_results,
    multiple_testing,
    subject_overlap,
    combination_summary,
):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    panel.to_parquet(PROCESSED_DIR / "tutoring_panel.parquet", index=False)
    fall_sample.to_parquet(PROCESSED_DIR / "tutoring_fall2013_sample.parquet", index=False)

    sample_flow_table(baseline_total, matched_ids, fall_sample).to_csv(
        OUTPUT_DIR / "sample_flow.csv", index=False, encoding="utf-8-sig"
    )
    model_results.to_csv(
        OUTPUT_DIR / "main_results.csv", index=False, encoding="utf-8-sig"
    )
    balance_diagnostics.to_csv(
        OUTPUT_DIR / "balance_diagnostics.csv", index=False, encoding="utf-8-sig"
    )
    robustness_results.to_csv(
        OUTPUT_DIR / "robustness_results.csv", index=False, encoding="utf-8-sig"
    )
    multiple_testing.to_csv(
        OUTPUT_DIR / "multiple_testing.csv", index=False, encoding="utf-8-sig"
    )
    subject_overlap.to_csv(
        OUTPUT_DIR / "subject_overlap_checks.csv", index=False, encoding="utf-8-sig"
    )
    combination_summary.to_csv(
        OUTPUT_DIR / "subject_combination_summary.csv", index=False, encoding="utf-8-sig"
    )
    quality_checks_table().to_csv(
        OUTPUT_DIR / "data_quality_checks.csv", index=False, encoding="utf-8-sig"
    )
