#!/usr/bin/env python3
"""Entry point for the CEPS tutoring analysis."""

from build_dataset import build_analysis_panel
from data_io import load_matched_students
from diagnostics import balance_table
from models import run_main_models
from outputs import write_all_outputs
from robustness import run_robustness


def main():
    baseline, _followup, baseline_matched, followup_matched, parent_matched, matched_ids = load_matched_students()
    panel, fall_sample = build_analysis_panel(baseline_matched, followup_matched, parent_matched)
    model_results = run_main_models(fall_sample)
    balance_diagnostics = balance_table(fall_sample)
    robustness_results, multiple_testing, subject_overlap, combination_summary = run_robustness(panel, fall_sample, model_results)
    write_all_outputs(
        panel=panel,
        fall_sample=fall_sample,
        baseline_total=len(baseline),
        matched_ids=matched_ids,
        model_results=model_results,
        balance_diagnostics=balance_diagnostics,
        robustness_results=robustness_results,
        multiple_testing=multiple_testing,
        subject_overlap=subject_overlap,
        combination_summary=combination_summary,
    )

    print("Wrote dataset/processed/tutoring_panel.parquet")
    print("Wrote dataset/processed/tutoring_fall2013_sample.parquet")
    print("Wrote outputs/sample_flow.csv")
    print("Wrote outputs/main_results.csv")
    print("Wrote outputs/balance_diagnostics.csv")
    print("Wrote outputs/robustness_results.csv")
    print("Wrote outputs/multiple_testing.csv")
    print("Wrote outputs/subject_overlap_checks.csv")
    print("Wrote outputs/subject_combination_summary.csv")
    print("Wrote outputs/data_quality_checks.csv")


if __name__ == "__main__":
    main()
