"""Export publication-ready Frontiers tables and figures.

This script does not refit models. It converts the validated analysis outputs in
``results/tables`` and ``results/figures`` into the final numbering, labels, and
file formats used by the Frontiers in Psychology manuscript.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE_TABLES = ROOT / "results" / "tables"
SOURCE_FIGURES = ROOT / "results" / "figures"
OUT = ROOT / "results" / "frontiers_submission"
MAIN_TABLES = OUT / "main_tables"
SUPP_TABLES = OUT / "supplementary_tables"
MAIN_FIGURES = OUT / "main_figures"
SUPP_FIGURES = OUT / "supplementary_figures"

for directory in [MAIN_TABLES, SUPP_TABLES, MAIN_FIGURES, SUPP_FIGURES]:
    directory.mkdir(parents=True, exist_ok=True)


SCALE_ORDER = ["PHQ-9", "GAD-7", "PSS", "ISI"]


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(SOURCE_TABLES / name)


def save_table(df: pd.DataFrame, directory: Path, filename: str) -> None:
    df.to_csv(directory / filename, index=False)


def rounded(values: pd.Series, digits: int = 3) -> pd.Series:
    return values.astype(float).round(digits)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    # Matplotlib may emit RGBA TIFFs even with a white face color. Convert through
    # Pillow so the submission file is explicitly RGB at 300 dpi.
    with Image.open(stem.with_suffix(".png")) as image:
        image.convert("RGB").save(
            stem.with_suffix(".tiff"),
            format="TIFF",
            compression="tiff_lzw",
            dpi=(300, 300),
        )
    plt.close(fig)


def convert_existing_figure(source: Path, destination_stem: Path) -> None:
    image = Image.open(source).convert("RGB")
    image.save(
        destination_stem.with_suffix(".tiff"),
        format="TIFF",
        compression="tiff_lzw",
        dpi=(300, 300),
    )
    image.save(destination_stem.with_suffix(".png"), format="PNG", dpi=(300, 300))


def main_table_1() -> pd.DataFrame:
    label_distribution = read("table1b_label_distribution.csv")
    long_audit = read("long_format_audit.csv").iloc[0]
    score_check = read("score_recalculation_checks.csv")
    split_files = [ROOT / "data" / "splits" / f"{split}_ids.csv" for split in ["train", "val", "test"]]
    if all(path.exists() for path in split_files):
        split_counts = [len(pd.read_csv(path)) for path in split_files]
    else:
        split_summary = read("label_distribution_by_split.csv")
        split_counts = [
            int(split_summary.loc[split_summary["split"] == split, "n_participants"].iloc[0])
            for split in ["train", "val", "test"]
        ]
    rows = [
        ["Participants", f"{int(long_audit['n_participants']):,}", "Complete cases in the released CSV files"],
        ["Item-level records", f"{int(long_audit['n_item_records']):,}", "Participants × 37 questionnaire items"],
        ["Train / validation / test", " / ".join(f"{n:,}" for n in split_counts), "Disjoint participant split"],
    ]
    notes = {
        "PHQ-9": "Minimal to severe",
        "GAD-7": "Minimal to very severe",
        "PSS": "Low to very high",
        "ISI": "No clinically significant insomnia to severe",
    }
    for scale in SCALE_ORDER:
        values = label_distribution[label_distribution["scale"] == scale]["count"]
        rows.append([f"{scale} distribution", " / ".join(f"{int(v):,}" for v in values), notes[scale]])
    pss_mismatches = int(
        score_check.loc[score_check["scale"] == "PSS", "n_score_mismatches"].iloc[0]
    )
    rows.append(["Score reconstruction", f"PSS: {pss_mismatches} mismatch", "Released score retained as reference"])
    return pd.DataFrame(rows, columns=["Item", "Value", "Note"])


def main_table_2() -> pd.DataFrame:
    data = read("main_model_results.csv")
    mapping = {
        "responses": "Response-only",
        "responses_rt_calibration_only": "RT calibration-only",
        "responses_rt_state": "RT score-adjusted",
    }
    data = data[data["budget"].isin([11, 19, 26]) & data["variant"].isin(mapping)].copy()
    data["Variant"] = data["variant"].map(mapping)
    data = data.sort_values(["budget", "variant"], key=lambda col: col.map(
        {"responses": 0, "responses_rt_calibration_only": 1, "responses_rt_state": 2}
    ) if col.name == "variant" else col)
    return pd.DataFrame(
        {
            "Item budget": data["budget"].astype(int),
            "Variant": data["Variant"],
            "Macro-F1": rounded(data["mean_macro_f1"]),
            "Elevated-score recall": rounded(data["mean_high_risk_recall"]),
            "Brier score": rounded(data["mean_brier_score"]),
            "ECE*": rounded(data["mean_ece"]),
        }
    )


def main_table_3() -> pd.DataFrame:
    data = read("table8_standard_calibration_baselines_19.csv")
    methods = [
        "response_only_uncalibrated",
        "temperature_scaling",
        "one_vs_rest_isotonic",
        "constant_softening",
        "shuffled_rt_placebo",
        "true_rt_calibration_only",
        "rt_score_adjusted",
    ]
    data = data.set_index("calibration_method").loc[methods].reset_index()
    method_labels = {
        "response_only_uncalibrated": "Response-only, uncalibrated",
        "temperature_scaling": "Temperature scaling",
        "one_vs_rest_isotonic": "One-vs-rest isotonic calibration",
        "constant_softening": "Constant confidence softening",
        "shuffled_rt_placebo": "Shuffled-RT placebo calibration",
        "true_rt_calibration_only": "True RT calibration-only",
        "rt_score_adjusted": "RT score-adjusted",
    }
    return pd.DataFrame(
        {
            "Calibration method": data["calibration_method"].map(method_labels),
            "RT information": data["uses_true_rt"],
            "Labels fixed": data["label_preserved"],
            "Macro-F1": rounded(data["mean_macro_f1"]),
            "Brier score": rounded(data["mean_brier_score"]),
            "ECE-L": rounded(data["mean_ece"]),
            "ECE-A": rounded(data["mean_ece_argmax"]),
            "NLL": rounded(data["mean_nll"]),
        }
    )


def main_table_4() -> pd.DataFrame:
    data = read("main_model_per_scale_metrics.csv")
    mapping = {
        "responses": "Response-only",
        "responses_rt_calibration_only": "RT calibration-only",
        "responses_rt_state": "RT score-adjusted",
    }
    data = data[(data["budget"] == 19) & data["variant"].isin(mapping)].copy()
    data["Scale order"] = data["scale"].map({s: i for i, s in enumerate(SCALE_ORDER)})
    data["Variant order"] = data["variant"].map(
        {"responses": 0, "responses_rt_calibration_only": 1, "responses_rt_state": 2}
    )
    data = data.sort_values(["Scale order", "Variant order"])
    return pd.DataFrame(
        {
            "Scale": data["scale"],
            "Variant": data["variant"].map(mapping),
            "Macro-F1": rounded(data["macro_f1"]),
            "MAE": rounded(data["mae"]),
            "Elevated-score recall": rounded(data["high_risk_recall"]),
            "Brier score": rounded(data["brier_score"]),
            "ECE*": rounded(data["ece"]),
        }
    )


def main_table_5() -> pd.DataFrame:
    data = read("table12_high_risk_burden_19.csv")
    data = data[data["variant"] == "Response-only"].copy()
    data["Scale order"] = data["scale"].map({s: i for i, s in enumerate(SCALE_ORDER)})
    data = data.sort_values("Scale order")
    return pd.DataFrame(
        {
            "Scale": data["scale"],
            "True elevated-score cases": data["true_high_risk_n"].astype(int),
            "Flagged cases": data["flagged_n"].astype(int),
            "Positive predictive value": rounded(data["high_risk_precision_ppv"]),
            "Recall": rounded(data["high_risk_recall_sensitivity"]),
            "Specificity": rounded(data["specificity"]),
            "False-positive rate": rounded(data["false_positive_rate"]),
        }
    )


def main_table_6() -> pd.DataFrame:
    data = read("table15_shortform_error_profile.csv").copy()
    scale_labels = {"Mean / all scale-decisions": "All scale decisions"}
    return pd.DataFrame(
        {
            "Scale": data["scale"].replace(scale_labels),
            "Decisions": data["n_decisions"].astype(int),
            "Short-form disagreements, n (%)": [
                f"{int(n):,} ({100 * rate:.1f})"
                for n, rate in zip(data["n_errors"], data["shortform_error_rate"])
            ],
            "Elevated-score false negatives, n": data["n_high_risk_fn"].astype(int),
            "Overconfident wrong decisions, n (%)": [
                f"{int(n):,} ({100 * rate:.1f})"
                for n, rate in zip(
                    data["n_overconfident_wrong"], data["overconfident_wrong_rate"]
                )
            ],
        }
    )


def main_table_7() -> pd.DataFrame:
    data = read("table16_error_detection_models.csv")
    model_order = ["confidence_only", "rt_only", "confidence_plus_rt", "shuffled_rt_placebo"]
    outcome_labels = {
        "wrong": "Short-form disagreement",
        "overconfident_wrong": "Overconfident wrong decision",
    }
    predictor_labels = {
        "confidence_only": "Confidence only",
        "rt_only": "Response time only",
        "confidence_plus_rt": "Confidence + response time",
        "shuffled_rt_placebo": "Confidence + shuffled response time",
    }
    rows = []
    for target in ["wrong", "overconfident_wrong"]:
        for model in model_order:
            row = data[(data["target"] == target) & (data["model"] == model)].iloc[0]
            rows.append(
                [
                    f"{outcome_labels[target]} (n = {int(row['test_positive_n']):,})",
                    predictor_labels[model],
                    round(float(row["pr_auc"]), 3),
                    round(float(row["roc_auc"]), 3),
                ]
            )
    return pd.DataFrame(rows, columns=["Outcome", "Predictor set", "PR-AUC", "ROC-AUC"])


def main_table_8() -> pd.DataFrame:
    data = read("table17_selective_fullform_escalation.csv")
    order = [
        "random",
        "confidence_only",
        "rt_only",
        "rt_plus_conf_error",
        "rt_plus_conf_ocw",
        "rt_plus_conf_hfn",
        "shuffled_rt_placebo",
        "oracle",
    ]
    labels = {
        "random": "Random",
        "confidence_only": "Lowest confidence",
        "rt_only": "RT process score",
        "rt_plus_conf_error": "Confidence + RT error model",
        "rt_plus_conf_ocw": "Confidence + RT OCW model",
        "rt_plus_conf_hfn": "Confidence + RT ES-FN model",
        "shuffled_rt_placebo": "Confidence + shuffled RT",
        "oracle": "Oracle",
    }
    data = data[(data["escalation_rate"].round(3) == 0.2)].set_index("strategy").loc[order]
    return pd.DataFrame(
        {
            "Escalation ranking": [labels[s] for s in order],
            "All short-form disagreements captured (%)": rounded(100 * data["error_capture_rate"], 1),
            "OCW captured (%)": rounded(100 * data["overconfident_wrong_capture_rate"], 1),
            "Elevated-score false negatives captured (%)": rounded(100 * data["high_risk_fn_capture_rate"], 1),
        }
    )


def main_table_9() -> pd.DataFrame:
    data = read("table18_process_quality_strata.csv")
    order = ["fast_inconsistent", "slow_interrupted", "stable_typical", "mixed_unstable"]
    labels = {
        "fast_inconsistent": "Fast-inconsistent",
        "slow_interrupted": "Slow/interrupted",
        "stable_typical": "Stable-typical",
        "mixed_unstable": "Mixed pattern",
    }
    data = data[data["stratum_source"] == "true_rt"].set_index("process_quality_stratum").loc[order]
    return pd.DataFrame(
        {
            "Process-quality stratum": [
                f"{labels[s]} (n = {int(data.loc[s, 'n_participants']):,})" for s in order
            ],
            "Error %": rounded(100 * data["shortform_error_rate"], 1),
            "OCW %": rounded(100 * data["overconfident_error_rate"], 1),
            "ECE-L": rounded(data["ece_l"]),
            "R": rounded(data["mean_reliability"]),
        }
    )


def main_table_10() -> pd.DataFrame:
    adaptive = read("table5_adaptive_budget_results.csv")
    adaptive = adaptive[adaptive["budget"] == 19].copy()
    selected = [
        ("fixed_order", "responses", "Fixed-order truncation"),
        ("short_forms", "responses", "Established short-form item set"),
        ("train_selected_fixed", "responses", "Training-selected fixed subset"),
        ("random_repeated_mean", "responses", "Repeated random subsets (50)"),
        ("response_only_uncertainty", "responses", "Response-only adaptive selection"),
        ("rt_aware_uncertainty", "responses_rt_state", "RT-aware adaptive selection"),
    ]
    rows = []
    for policy, variant, label in selected:
        row = adaptive[(adaptive["policy"] == policy) & (adaptive["variant"] == variant)].iloc[0]
        rows.append(
            [label, row["mean_macro_f1"], row["mean_ece"], row["mean_high_risk_recall"], row["mean_brier_score"]]
        )
    learned = read("table9_learned_maskaware_baselines.csv")
    learned_labels = {
        "train_selected_fixed_maskaware_cart": "Training-selected CART",
        "train_selected_fixed_maskaware_logistic_l2": "Training-selected logistic regression",
        "short_forms_maskaware_cart": "Short-form CART",
        "short_forms_maskaware_logistic_l2": "Short-form logistic regression",
    }
    for _, row in learned.iterrows():
        rows.append(
            [
                learned_labels[row["policy"]],
                row["mean_macro_f1"],
                row["mean_ece"],
                row["mean_high_risk_recall"],
                row["mean_brier_score"],
            ]
        )
    out = pd.DataFrame(
        rows,
        columns=["Reduced-item policy", "Macro-F1", "ECE-L", "Elevated-score recall", "Brier score"],
    )
    for column in out.columns[1:]:
        out[column] = rounded(out[column])
    return out


def supplementary_table_1() -> pd.DataFrame:
    operations = [
        "Fit RT winsorization thresholds and residual models on training participants only.",
        "Select scale- and budget-specific calibration parameters on validation participants only.",
        "Initialize one training-informative warm-up item per scale.",
        "Update partial scores, uncertainty, coverage, and response-process consistency from observed items only.",
        "Score remaining candidates using training informativeness, uncertainty, coverage, redundancy, and fixed process-quality weights.",
        "Predict scale-specific severity categories and calibrated probabilities; compute scale-wise and mean metrics.",
    ]
    return pd.DataFrame({"Step": range(1, 7), "Operation": operations})


def supplementary_table_2() -> pd.DataFrame:
    data = read("table11_rt_residual_sensitivity_19.csv")
    return pd.DataFrame(
        {
            "Residual model": data["residual_model"],
            "Training R²": rounded(data["train_R2_log_rt"]),
            "Residual SD": rounded(data["train_residual_sd"]),
            "5th percentile": rounded(data["residual_q05"]),
            "Median": rounded(data["residual_q50"]),
            "95th percentile": rounded(data["residual_q95"]),
        }
    )


def supplementary_table_3() -> pd.DataFrame:
    stepwise = read("table7c_stepwise_leakage_summary.csv").set_index("perturbation")
    leakage = read("table7_leakage_tests.csv").set_index("test")
    rows = [
        ["Stepwise response perturbation", f"{int(stepwise.loc['future_response', 'prefixes_unchanged'])}/{int(stepwise.loc['future_response', 'prefixes_unchanged'])} prefix checks", stepwise.loc["future_response", "status"]],
        ["Stepwise RT perturbation", f"{int(stepwise.loc['future_rt', 'prefixes_unchanged'])}/{int(stepwise.loc['future_rt', 'prefixes_unchanged'])} prefix checks", stepwise.loc["future_rt", "status"]],
        ["Full response perturbation", leakage.loc["future responses excluded from actual policy", "details"], leakage.loc["future responses excluded from actual policy", "status"]],
        ["Full RT perturbation", leakage.loc["future RT excluded from actual policy", "details"], leakage.loc["future RT excluded from actual policy", "status"]],
        ["Split identifiers", leakage.loc["split disjointness", "details"], leakage.loc["split disjointness", "status"]],
        ["RT residual training only", leakage.loc["RT residual train-only fit", "details"], leakage.loc["RT residual train-only fit", "status"]],
        ["Fixed-prefix prediction invariance", leakage.loc["fixed-prefix prediction invariance", "details"], leakage.loc["fixed-prefix prediction invariance", "status"]],
    ]
    return pd.DataFrame(rows, columns=["Audit", "Evidence", "Status"])


def supplementary_table_4() -> pd.DataFrame:
    data = read("table19_cross_scale_transfer.csv")
    pivot = data.pivot(index="heldout_scale", columns="model", values="pr_auc").loc[SCALE_ORDER]
    return pd.DataFrame(
        {
            "Held-out scale": pivot.index,
            "Confidence only, PR-AUC": rounded(pivot["confidence_only"]),
            "Confidence + RT, PR-AUC": rounded(pivot["confidence_plus_rt"]),
            "Confidence + shuffled RT, PR-AUC": rounded(pivot["shuffled_rt_placebo"]),
        }
    )


def supplementary_table_5() -> pd.DataFrame:
    data = read("table20_escalation_bootstrap_ci.csv")
    labels = {
        "random": "Random",
        "confidence_only": "Lowest confidence",
        "rt_only": "RT process score",
        "rt_plus_conf_error": "Confidence + RT (any error)",
        "rt_plus_conf_ocw": "Confidence + RT (OCW)",
    }
    data = data[data["strategy"].isin(labels)].copy()
    data["Strategy"] = data["strategy"].map(labels)
    return pd.DataFrame(
        {
            "Strategy": data["Strategy"],
            "Error capture % (95% CI)": [
                f"{100*r.error_capture_rate:.1f} ({100*r.error_capture_rate_ci_low:.1f}-{100*r.error_capture_rate_ci_high:.1f})"
                for r in data.itertuples()
            ],
            "OCW capture % (95% CI)": [
                f"{100*r.overconfident_wrong_capture_rate:.1f} ({100*r.overconfident_wrong_capture_rate_ci_low:.1f}-{100*r.overconfident_wrong_capture_rate_ci_high:.1f})"
                if pd.notna(r.overconfident_wrong_capture_rate_ci_low) else f"{100*r.overconfident_wrong_capture_rate:.1f}"
                for r in data.itertuples()
            ],
            "ES-FN capture % (95% CI)": [
                f"{100*r.high_risk_fn_capture_rate:.1f} ({100*r.high_risk_fn_capture_rate_ci_low:.1f}-{100*r.high_risk_fn_capture_rate_ci_high:.1f})"
                if pd.notna(r.high_risk_fn_capture_rate_ci_low) else f"{100*r.high_risk_fn_capture_rate:.1f}"
                for r in data.itertuples()
            ],
        }
    )


def supplementary_table_6() -> pd.DataFrame:
    data = read("table21_ocw_threshold_sensitivity.csv")
    data = data[data["overconfidence_quantile"].round(2).isin([0.70, 0.75, 0.90])].copy()
    labels = {
        "confidence_only": "Confidence only",
        "rt_only": "RT only",
        "confidence_plus_rt": "Confidence + RT",
        "shuffled_rt_placebo": "Confidence + shuffled RT",
    }
    return pd.DataFrame(
        {
            "Validation quantile": rounded(data["overconfidence_quantile"], 2),
            "Model": data["model"].map(labels),
            "Test OCW, n": data["test_positive_n"].astype(int),
            "PR-AUC": rounded(data["pr_auc"]),
            "ROC-AUC": rounded(data["roc_auc"]),
            "Recall at top 10%": rounded(data["recall_at_top10pct"]),
            "Precision at top 10%": rounded(data["precision_at_top10pct"]),
        }
    )


def supplementary_table_7() -> pd.DataFrame:
    data = read("table22_per_scale_selective_escalation.csv")
    labels = {
        "confidence_only": "Lowest confidence",
        "rt_only": "RT process score",
        "rt_plus_conf_ocw": "Confidence + RT (OCW)",
    }
    return pd.DataFrame(
        {
            "Scale": data["scale"],
            "Strategy": data["strategy"].map(labels),
            "Errors, n": data["n_errors"].astype(int),
            "OCW, n": data["n_ocw"].astype(int),
            "Error capture %": rounded(100 * data["error_capture_rate"], 1),
            "OCW capture %": rounded(100 * data["overconfident_wrong_capture_rate"], 1),
        }
    )


def supplementary_table_8() -> pd.DataFrame:
    return read("table24_non_pss_selective_escalation.csv")


def supplementary_table_9() -> pd.DataFrame:
    return read("table25_detection_pr_auc_bootstrap.csv")


def supplementary_table_10() -> pd.DataFrame:
    return read("table26_internal_consistency.csv")


def supplementary_table_11() -> pd.DataFrame:
    return read("table27_score_level_sensitivity.csv")


def export_tables() -> None:
    main_tables = [
        main_table_1(),
        main_table_2(),
        main_table_3(),
        main_table_4(),
        main_table_5(),
        main_table_6(),
        main_table_7(),
        main_table_8(),
        main_table_9(),
        main_table_10(),
    ]
    main_names = [
        "Table_1_sample_characteristics.csv",
        "Table_2_classification_calibration_budgets.csv",
        "Table_3_calibration_controls.csv",
        "Table_4_scale_specific_performance.csv",
        "Table_5_high_risk_burden.csv",
        "Table_6_short_form_error_distribution.csv",
        "Table_7_error_detection.csv",
        "Table_8_selective_escalation.csv",
        "Table_9_process_quality_strata.csv",
        "Table_10_reduced_item_comparators.csv",
    ]
    for table, name in zip(main_tables, main_names):
        save_table(table, MAIN_TABLES, name)

    supp_tables = [
        supplementary_table_1(),
        supplementary_table_6(),
        supplementary_table_9(),
        supplementary_table_5(),
        supplementary_table_7(),
        supplementary_table_8(),
        supplementary_table_2(),
        supplementary_table_3(),
        supplementary_table_4(),
        supplementary_table_10(),
        supplementary_table_11(),
    ]
    supp_names = [
        "Supplementary_Table_S1_simulation_procedure.csv",
        "Supplementary_Table_S2_ocw_threshold_sensitivity.csv",
        "Supplementary_Table_S3_detection_pr_auc_bootstrap.csv",
        "Supplementary_Table_S4_escalation_bootstrap_ci.csv",
        "Supplementary_Table_S5_per_scale_escalation.csv",
        "Supplementary_Table_S6_non_pss_escalation.csv",
        "Supplementary_Table_S7_rt_residualization.csv",
        "Supplementary_Table_S8_leakage_checks.csv",
        "Supplementary_Table_S9_cross_scale_transfer.csv",
        "Supplementary_Table_S10_internal_consistency.csv",
        "Supplementary_Table_S11_score_level_sensitivity.csv",
    ]
    for table, name in zip(supp_tables, supp_names):
        save_table(table, SUPP_TABLES, name)


def export_figures() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    budget = read("table5_adaptive_budget_results.csv")
    policy_specs = [
        ("fixed_order", "responses", "Fixed order", "#4C78A8", "o"),
        ("short_forms", "responses", "Short form", "#F58518", "^"),
        ("response_only_uncertainty", "responses", "Response-only", "#54A24B", "s"),
        ("rt_aware_uncertainty", "responses_rt_state", "RT-aware", "#E45756", "D"),
        ("random_repeated_mean", "responses", "Random mean", "#7A5195", "x"),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
    for policy, variant, label, color, marker in policy_specs:
        subset = budget[(budget["policy"] == policy) & (budget["variant"] == variant)].sort_values("budget")
        ax.plot(subset["budget"], subset["mean_macro_f1"], label=label, color=color, marker=marker, linewidth=2.2)
    ax.set_xlabel("Item budget")
    ax.set_ylabel("Mean macro-F1")
    ax.set_xticks([11, 19, 26])
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    ax.legend(frameon=False)
    save_figure(fig, MAIN_FIGURES / "Figure_1_macro_F1_across_item_budgets")

    primary = read("main_model_results.csv")
    variant_specs = [
        ("responses", "Response-only", "#4C78A8", "s"),
        ("responses_rt_calibration_only", "RT calibration-only", "#F58518", "o"),
        ("responses_rt_state", "RT score-adjusted", "#54A24B", "D"),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
    for variant, label, color, marker in variant_specs:
        subset = primary[primary["variant"] == variant].sort_values("budget")
        ax.plot(subset["budget"], subset["mean_ece"], label=label, color=color, marker=marker, linewidth=2.2)
    ax.set_xlabel("Item budget")
    ax.set_ylabel("Mean ECE*")
    ax.set_xticks([11, 19, 26])
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    ax.legend(frameon=False)
    save_figure(fig, MAIN_FIGURES / "Figure_2_ECE_star_across_item_budgets")

    escalation = read("table17_selective_fullform_escalation.csv")
    strategy_specs = [
        ("random", "Random", "#4C78A8", "o"),
        ("confidence_only", "Lowest confidence", "#F58518", "s"),
        ("rt_only", "RT process score", "#54A24B", "^"),
        ("rt_plus_conf_error", "Confidence + RT (any error)", "#E45756", "D"),
        ("rt_plus_conf_ocw", "Confidence + RT (overconfident error)", "#7A5195", "P"),
        ("shuffled_rt_placebo", "Confidence + shuffled RT", "#9D755D", "X"),
        ("oracle", "Oracle upper bound", "#D65DB1", "*"),
    ]
    fig, ax = plt.subplots(figsize=(9.0, 5.5), constrained_layout=True)
    for strategy, label, color, marker in strategy_specs:
        subset = escalation[escalation["strategy"] == strategy].sort_values("avg_items_used")
        ax.plot(
            subset["avg_items_used"],
            100 * subset["error_capture_rate"],
            label=label,
            color=color,
            marker=marker,
            linewidth=2.2,
            markersize=6.5,
        )
    ax.set_xlabel("Average number of items completed")
    ax.set_ylabel("Short-form errors captured (%)")
    ax.set_xticks([20, 21, 22, 23, 24])
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    ax.legend(loc="upper left", ncol=2, frameon=False)
    save_figure(fig, MAIN_FIGURES / "Figure_3_selective_full_form_escalation_frontier")

    strata = read("table18_process_quality_strata.csv")
    strata = strata[strata["stratum_source"] == "true_rt"].set_index("process_quality_stratum")
    order = ["fast_inconsistent", "slow_interrupted", "stable_typical", "mixed_unstable"]
    labels = ["Fast-\ninconsistent", "Slow /\ninterrupted", "Stable-\ntypical", "Mixed\npattern"]
    values = [100 * strata.loc[name, "shortform_error_rate"] for name in order]
    fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
    bars = ax.bar(labels, values, color=["#D95F59", "#E69F55", "#59A14F", "#4E79A7"])
    ax.set_ylabel("Short-form error rate (%)")
    ax.set_ylim(0, max(values) + 4)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.5, f"{value:.1f}%", ha="center")
    save_figure(fig, MAIN_FIGURES / "Figure_4_process_quality_strata")

    supplementary_sources = [
        ("figure1_rt_residualization.png", "Supplementary_Figure_S1_rt_residualization"),
        ("figure4_per_scale_reliability_diagrams.png", "Supplementary_Figure_S2_per_scale_reliability"),
        ("figure5_calibration_controls_comparison.png", "Supplementary_Figure_S3_calibration_controls"),
        ("figure6_subgroup_fnr.png", "Supplementary_Figure_S4_response_speed_subgroups"),
    ]
    for source, destination in supplementary_sources:
        convert_existing_figure(SOURCE_FIGURES / source, SUPP_FIGURES / destination)


def write_manifest() -> None:
    text = """# Frontiers submission output map

This directory contains the final display numbering used in the Frontiers in
Psychology manuscript.

- `main_tables/`: Tables 1-10, in manuscript citation order.
- `supplementary_tables/`: Supplementary Tables S1-S11.
- `main_figures/`: Figures 1-4 as 300 dpi RGB PNG and TIFF files.
- `supplementary_figures/`: Supplementary Figures S1-S4 as 300 dpi RGB PNG and TIFF files.

The underlying full-resolution analysis outputs remain in `results/tables` and
`results/figures`. This export layer changes presentation labels and numbering;
it does not alter model estimates.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    export_tables()
    export_figures()
    write_manifest()
    print(f"Frontiers submission outputs written to {OUT}")
