import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import argparse
import pandas as pd
from rt_screening.core import evaluate_policy, fit_rt_residuals, load_ids, root_path

parser = argparse.ArgumentParser()
parser.add_argument('--policy', required=True)
parser.add_argument('--budget', type=int, required=True)
parser.add_argument('--variant', required=True)
parser.add_argument('--tag', required=True)
parser.add_argument('--seed', type=int, default=None)
args = parser.parse_args()
long = fit_rt_residuals()
ids = load_ids('test')
agg, per, pred = evaluate_policy(args.policy, args.budget, args.variant, ids, 'test', long=long, seed=args.seed)
agg_df = pd.DataFrame([agg])
agg_df.to_csv(root_path('results/tables', f'eval_{args.tag}_aggregate.csv'), index=False)
per.to_csv(root_path('results/tables', f'eval_{args.tag}_per_scale.csv'), index=False)
pred.to_csv(root_path('results/tables', f'predictions_{args.tag}.csv'), index=False)
