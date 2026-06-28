#!/usr/bin/env bash
set -euo pipefail
# Additional manuscript analyses. The clean release ships generated outputs;
# set FORCE_ADDITIONAL=1 to recompute them.
need(){ [[ "${FORCE_ADDITIONAL:-0}" == "1" || ! -f "$1" ]]; }
run_stage(){
python - <<PY
from scripts import additional_analyses as r
long = r.fit_rt_residuals()
print('[analysis-stage] $1', flush=True)
r.$2(long)
print('[analysis-stage] $1 done', flush=True)
PY
}

# Some additional calibration tables require row-level prediction files from
# the primary pipeline.  When this script is called on a clean compact release
# with FORCE_ADDITIONAL=1, fail early with a clear instruction rather than a
# downstream Python traceback.
if [[ "${FORCE_ADDITIONAL:-0}" == "1" && ! -f results/tables/predictions_rt_aware_uncertainty_b19_responses_rt_state.csv ]]; then
  echo "[analysis-stage] Missing row-level prediction files. Run FORCE=1 bash run_all.sh before FORCE_ADDITIONAL=1 bash run_additional_analyses.sh." >&2
  exit 2
fi

need results/tables/table8_standard_calibration_baselines_19.csv && run_stage standard_calibration run_standard_calibration_baselines || echo '[analysis-stage] standard_calibration already present'
need results/tables/table9_learned_maskaware_baselines.csv && run_stage learned_maskaware run_learned_maskaware_baselines || echo '[analysis-stage] learned_maskaware already present'
need results/tables/table10_per_class_metrics_19.csv && run_stage per_class run_per_class_error_tables || echo '[analysis-stage] per_class already present'
need results/tables/table12_high_risk_burden_19.csv && run_stage high_risk_burden run_high_risk_burden_table || echo '[analysis-stage] high_risk_burden already present'
need results/tables/table11_rt_residual_sensitivity_19.csv && run_stage rt_residual_diagnostics run_rt_residual_sensitivity || echo '[analysis-stage] rt_residual_diagnostics already present'
need results/tables/table14_calibration_grid_boundary_sensitivity_19.csv && run_stage calibration_grid_boundary run_calibration_grid_boundary_sensitivity || echo '[analysis-stage] calibration_grid_boundary already present'
need results/tables/table7b_stepwise_leakage_audit.csv && run_stage stepwise_leakage run_stepwise_leakage_audit || echo '[analysis-stage] stepwise_leakage already present'
need results/figures/figure4_per_scale_reliability_diagrams.png && run_stage additional_figures generate_additional_figures || echo '[analysis-stage] additional_figures already present'
# Deployment-oriented selective auditing analyses. Kept separate from engine.py.
selective_need=0
for f in \
  results/tables/table15_shortform_error_profile.csv \
  results/tables/table16_error_detection_models.csv \
  results/tables/table17_selective_fullform_escalation.csv \
  results/tables/table18_process_quality_strata.csv \
  results/tables/table19_cross_scale_transfer.csv \
  results/tables/table20_escalation_bootstrap_ci.csv \
  results/tables/table21_ocw_threshold_sensitivity.csv \
  results/tables/table22_per_scale_selective_escalation.csv \
  results/figures/figure7_escalation_burden_frontier.png \
  results/figures/figure8_process_quality_error_rates.png; do
  if [[ "${FORCE_ADDITIONAL:-0}" == "1" || ! -f "$f" ]]; then
    selective_need=1
  fi
done
if [[ "$selective_need" == "1" ]]; then
python - <<'PYSEL'
from scripts import selective_auditing_analyses as s
long = s.fit_rt_residuals()
print('[analysis-stage] selective_auditing', flush=True)
s.run_all_selective_auditing(long)
print('[analysis-stage] selective_auditing done', flush=True)
PYSEL
else
  echo '[analysis-stage] selective_auditing already present'
fi
