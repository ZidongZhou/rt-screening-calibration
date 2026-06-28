from pathlib import Path
import subprocess, sys, os
ROOT = Path(__file__).resolve().parents[1]
env = os.environ.copy(); env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
for b in [11, 19, 26]:
    subprocess.check_call([sys.executable, "scripts/evaluate_policy.py", "--policy", "response_only_uncertainty", "--budget", str(b), "--variant", "responses", "--tag", f"response_only_uncertainty_b{b}_responses"], cwd=ROOT, env=env)
    subprocess.check_call([sys.executable, "scripts/evaluate_policy.py", "--policy", "rt_aware_uncertainty", "--budget", str(b), "--variant", "responses_rt_state", "--tag", f"rt_aware_uncertainty_b{b}_responses_rt_state"], cwd=ROOT, env=env)
subprocess.check_call([sys.executable, "scripts/combine_outputs.py"], cwd=ROOT, env=env)
