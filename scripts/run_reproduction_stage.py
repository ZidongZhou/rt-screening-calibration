"""Run one resumable reproduction stage.

This entry point is intentionally small so reviewers can rerun only the stage
needed for a manuscript claim instead of restarting the entire pipeline.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rt_screening.engine import (
    bootstrap_main_comparison,
    create_splits,
    fit_rt_residuals,
    generate_tables_figures,
    run_ablations,
    run_adaptive_simulation,
    run_baselines,
    run_leakage_tests,
    run_main_models,
    run_subgroup_robustness,
    validate_data,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=[
            "primary",
            "robustness",
            "publication",
            "additional",
            "selective",
            "checks",
        ],
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.force:
        os.environ["FORCE"] = "1"
    force = args.force

    def needed(relative: str) -> bool:
        return force or not (ROOT / relative).exists()

    if args.stage == "checks":
        runpy.run_path("scripts/check_expected_outputs.py", run_name="__main__")
        runpy.run_path("scripts/check_frontiers_submission_outputs.py", run_name="__main__")
        return

    if args.stage == "primary":
        if needed("results/tables/table1b_label_distribution.csv"):
            validate_data()
        if force or not all((ROOT / "data" / "splits" / f"{name}_ids.csv").exists() for name in ["train", "val", "test"]):
            create_splits()
    long = fit_rt_residuals()
    if args.stage == "primary":
        if needed("results/tables/table3_baseline_performance.csv"):
            run_baselines(long)
        if needed("results/tables/main_model_results.csv"):
            run_main_models(long)
        if needed("results/tables/table5_adaptive_budget_results.csv"):
            run_adaptive_simulation(long)
    elif args.stage == "robustness":
        if needed("results/tables/table4_ablation_results.csv"):
            run_ablations(long)
        if needed("results/tables/bootstrap_ci_main_comparison.csv"):
            bootstrap_main_comparison(n_boot=200)
        if needed("results/tables/table6_subgroup_robustness.csv"):
            run_subgroup_robustness(long)
        if needed("results/tables/table7_leakage_tests.csv"):
            run_leakage_tests(long)
    elif args.stage == "publication":
        generate_tables_figures(long)
        from scripts import additional_analyses as additional

        additional.generate_additional_figures(long)
        runpy.run_path("scripts/12_export_frontiers_submission_outputs.py", run_name="__main__")
    elif args.stage == "additional":
        from scripts import additional_analyses as additional

        jobs = [
            ("results/tables/table8_standard_calibration_baselines_19.csv", additional.run_standard_calibration_baselines),
            ("results/tables/table9_learned_maskaware_baselines.csv", additional.run_learned_maskaware_baselines),
            ("results/tables/table10_per_class_metrics_19.csv", additional.run_per_class_error_tables),
            ("results/tables/table12_high_risk_burden_19.csv", additional.run_high_risk_burden_table),
            ("results/tables/table11_rt_residual_sensitivity_19.csv", additional.run_rt_residual_sensitivity),
            ("results/tables/table14_calibration_grid_boundary_sensitivity_19.csv", additional.run_calibration_grid_boundary_sensitivity),
            ("results/tables/table7b_stepwise_leakage_audit.csv", additional.run_stepwise_leakage_audit),
        ]
        for output, function in jobs:
            if needed(output):
                function(long)
        if needed("results/figures/figure5_calibration_controls_comparison.png"):
            additional.generate_additional_figures(long)
    elif args.stage == "selective":
        from scripts import selective_auditing_analyses as selective

        jobs = [
            ("results/tables/table15_shortform_error_profile.csv", selective.run_shortform_error_profile),
            ("results/tables/table16_error_detection_models.csv", selective.run_shortform_error_detection),
            ("results/tables/table17_selective_fullform_escalation.csv", selective.run_selective_fullform_escalation),
            ("results/tables/table18_process_quality_strata.csv", selective.run_process_quality_strata),
            ("results/tables/table19_cross_scale_transfer.csv", selective.run_cross_scale_transfer),
            ("results/tables/table20_escalation_bootstrap_ci.csv", selective.run_escalation_bootstrap_ci),
            ("results/tables/table21_ocw_threshold_sensitivity.csv", selective.run_ocw_threshold_sensitivity),
            ("results/tables/table22_per_scale_selective_escalation.csv", selective.run_per_scale_selective_escalation),
        ]
        for output, function in jobs:
            if needed(output):
                function(long)


if __name__ == "__main__":
    main()
