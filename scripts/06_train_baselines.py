import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rt_screening.core import fit_rt_residuals, run_baselines

if __name__ == "__main__":
    print("[6/11] evaluating baseline item-budget policies...", flush=True)
    long = fit_rt_residuals()
    run_baselines(long)
