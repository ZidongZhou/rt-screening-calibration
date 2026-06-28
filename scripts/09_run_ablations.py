from pathlib import Path
import subprocess, sys, os
ROOT = Path(__file__).resolve().parents[1]
env = os.environ.copy(); env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
for ab, variant, policy in [
    ("partial_responses_only", "responses", "response_only_uncertainty"),
    ("rt_only", "rt_only", "rt_aware_uncertainty"),
    ("partial_responses_plus_raw_rt", "responses_rawrt", "rt_aware_uncertainty"),
    ("partial_responses_plus_content_rt_residual", "responses_resid_content", "rt_aware_uncertainty"),
    ("partial_responses_plus_response_conditioned_rt_residual", "responses_resid_conditioned", "rt_aware_uncertainty"),
    ("partial_responses_plus_rt_residual_and_online_state", "responses_rt_state", "rt_aware_uncertainty"),
]:
    subprocess.check_call([sys.executable, "scripts/evaluate_policy.py", "--policy", policy, "--budget", "19", "--variant", variant, "--tag", f"ablation_{ab}"], cwd=ROOT, env=env)
subprocess.check_call([sys.executable, "scripts/combine_outputs.py"], cwd=ROOT, env=env)
