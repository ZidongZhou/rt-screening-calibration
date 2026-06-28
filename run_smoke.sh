#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
python -m compileall -q src scripts tests
python -m pytest tests -q
python scripts/check_expected_outputs.py
python scripts/check_frontiers_submission_outputs.py
