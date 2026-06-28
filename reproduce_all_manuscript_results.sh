#!/usr/bin/env bash
set -euo pipefail
FORCE=1 bash run_all.sh
FORCE_ADDITIONAL=1 bash run_additional_analyses.sh
python scripts/check_expected_outputs.py
python scripts/12_export_frontiers_submission_outputs.py
python scripts/check_frontiers_submission_outputs.py
