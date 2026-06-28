#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
python scripts/run_all_pipeline.py
