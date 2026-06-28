import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rt_screening.core import run_subgroup_robustness, run_leakage_tests
run_subgroup_robustness()
run_leakage_tests()
