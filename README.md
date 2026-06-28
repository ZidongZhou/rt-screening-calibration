# Response-Time Process Signals for Selective Escalation in Reduced-Item Mental-Health Symptom Screening

Code and analysis outputs for the manuscript:

> Response-Time Process Signals for Selective Escalation in Reduced-Item Mental-Health Symptom Screening

The analysis uses the public response time questionnaire dataset for PHQ-9, GAD-7, PSS-14, and ISI. Outcomes are score-derived questionnaire categories. They are not clinical diagnoses.

Raw participant CSV files are not included in this repository. To rerun the pipeline from the public data source:

```bash
python scripts/download_data.py
```

The script downloads the five Zenodo CSV files into `data/raw/` and checks their MD5 hashes.

## What is in the repository

`src/rt_screening/` contains the reusable pipeline code. The numbered scripts in `scripts/` run validation, split generation, response time residualization, model fitting, calibration checks, selective escalation analyses, and manuscript table or figure export.

The submitted aggregate outputs are kept under `results/`. The display items used by the manuscript are in `results/frontiers_submission/`:

- `main_tables/`: Tables 1 to 10
- `supplementary_tables/`: Supplementary Tables S1-S11.
- `main_figures/`: Figures 1 to 4
- `supplementary_figures/`: Supplementary Figures S1 to S4

The files in `results/tables/`, `results/figures/`, and `results/supplementary_figures/` are pipeline outputs. Some filenames keep the older internal numbering used while the analysis was being assembled.

## Main result files

The main selective escalation checks are:

- `results/tables/table15_shortform_error_profile.csv`
- `results/tables/table16_error_detection_models.csv`
- `results/tables/table17_selective_fullform_escalation.csv`
- `results/tables/table18_process_quality_strata.csv`
- `results/tables/table19_cross_scale_transfer.csv`
- `results/tables/table20_escalation_bootstrap_ci.csv`
- `results/tables/table21_ocw_threshold_sensitivity.csv`
- `results/tables/table22_per_scale_selective_escalation.csv`
- `results/tables/table24_non_pss_selective_escalation.csv`
- `results/tables/table25_detection_pr_auc_bootstrap.csv`
- `results/tables/table26_internal_consistency.csv`
- `results/tables/table27_score_level_sensitivity.csv`

At the 19-item budget, response-only screening reached macro-F1 = 0.689 and ECE-L = 0.098. RT calibration-only kept the issued labels fixed and reduced ECE-L to 0.083. RT score adjustment lowered argmax ECE to 0.079, but macro-F1 dropped to 0.676. One-vs-rest isotonic calibration remained the stronger probability calibration control, with ECE-L = 0.038.

The selective escalation analyses therefore focus on a narrower question: whether observed-prefix response time helps flag short-form decisions that should be completed with the remaining questionnaire items.

## Run checks

For a quick check of the packaged repository:

```bash
bash run_smoke.sh
python scripts/check_expected_outputs.py
python scripts/check_frontiers_submission_outputs.py
```

Without raw CSV files, tests that require participant-level recomputation are skipped. After downloading the source CSVs, run the full test suite with:

```bash
python -m pytest -q
```

## Reproduce the analysis

Primary pipeline:

```bash
FORCE=1 bash run_all.sh
```

Additional calibration, leakage, residualization, and selective escalation analyses:

```bash
FORCE_ADDITIONAL=1 bash run_additional_analyses.sh
```

Manuscript-facing tables and figures:

```bash
bash reproduce_all_manuscript_results.sh
python scripts/12_export_frontiers_submission_outputs.py
```

On Windows, the same stages can be run one at a time:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_stage.ps1 primary
powershell -ExecutionPolicy Bypass -File .\run_stage.ps1 robustness
powershell -ExecutionPolicy Bypass -File .\run_stage.ps1 additional
powershell -ExecutionPolicy Bypass -File .\run_stage.ps1 selective
python scripts/12_export_frontiers_submission_outputs.py
powershell -ExecutionPolicy Bypass -File .\run_stage.ps1 checks
```

## Data note

This repository does not redistribute raw participant-level CSV files, processed participant tables, model files, or participant split identifier CSVs. The source dataset is cited in the manuscript and can be downloaded with `scripts/download_data.py`.

Escalation in this code means replacing a reduced-item questionnaire category with the complete questionnaire-derived category after more items are completed. It is an offline simulation, not a clinical diagnostic model or a deployed adaptive test.

## Environment

The final checks were run with Python 3.11.8. `requirements.txt` gives the ordinary install requirements. `requirements-lock.txt` records the direct package versions used for the submitted run.
