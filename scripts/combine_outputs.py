import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from pathlib import Path
from rt_screening.core import root_path, load_config, bootstrap_main_comparison, run_subgroup_robustness, run_leakage_tests, generate_tables_figures, fit_rt_residuals

tables = root_path('results/tables')
# Baselines
base_tags = []
for policy in ['fixed_order','short_forms','train_selected_fixed']:
    for b in load_config()['budgets']:
        base_tags.append(f'{policy}_b{b}_responses')
for b in load_config()['budgets']:
    for r in range(load_config().get('random_repeats', 5)):
        base_tags.append(f'random_b{b}_r{r}_responses')
base_rows=[]; base_per=[]
for tag in base_tags:
    f=tables/f'eval_{tag}_aggregate.csv'
    if f.exists(): base_rows.append(pd.read_csv(f))
    pf=tables/f'eval_{tag}_per_scale.csv'
    if pf.exists(): base_per.append(pd.read_csv(pf))
base=pd.concat(base_rows, ignore_index=True) if base_rows else pd.DataFrame()
# summarize random repeats
if not base.empty:
    fixed=base[base['policy']!='random'].copy()
    random=base[base['policy']=='random'].copy()
    summaries=[]
    for b,g in random.groupby('budget'):
        row={'policy':'random_repeated_mean','budget':b,'variant':'responses','split':'test','n':int(g['n'].iloc[0]),'avg_items_used':b}
        for c in g.select_dtypes(include='number').columns:
            if c not in ['budget','n','avg_items_used']:
                row[c]=g[c].mean()
        summaries.append(row)
    oracle={'policy':'full_scale_scoring_upper_bound','budget':37,'variant':'oracle','split':'test','n':int(fixed['n'].iloc[0] if len(fixed) else 0),'avg_items_used':37,'mean_macro_f1':1.0,'mean_weighted_f1':1.0,'mean_mae':0.0,'mean_quadratic_weighted_kappa':1.0,'mean_high_risk_recall':1.0,'mean_false_negative_rate':0.0,'mean_auroc':1.0,'mean_pr_auc':1.0,'mean_brier_score':0.0,'mean_ece':0.0}
    out=pd.concat([pd.DataFrame([oracle]), fixed, pd.DataFrame(summaries)], ignore_index=True, sort=False)
    out.to_csv(tables/'table3_baseline_performance.csv', index=False)
if base_per:
    pd.concat(base_per, ignore_index=True).to_csv(tables/'table3b_baseline_per_scale_metrics.csv', index=False)
# Main
main_tags=[]
for b in load_config()['budgets']:
    main_tags += [f'response_only_uncertainty_b{b}_responses', f'rt_aware_uncertainty_b{b}_responses_rt_state']
main_rows=[]; main_per=[]
for tag in main_tags:
    f=tables/f'eval_{tag}_aggregate.csv'
    if f.exists(): main_rows.append(pd.read_csv(f))
    pf=tables/f'eval_{tag}_per_scale.csv'
    if pf.exists(): main_per.append(pd.read_csv(pf))
if main_rows:
    pd.concat(main_rows, ignore_index=True).to_csv(tables/'main_model_results.csv', index=False)
if main_per:
    pd.concat(main_per, ignore_index=True).to_csv(tables/'main_model_per_scale_metrics.csv', index=False)
# Adaptive budget table
parts=[]
for f in [tables/'table3_baseline_performance.csv', tables/'main_model_results.csv']:
    if f.exists(): parts.append(pd.read_csv(f))
if parts:
    df=pd.concat(parts, ignore_index=True, sort=False)
    rows=[]
    for b in load_config()['budgets']:
        ro=df[(df['policy']=='response_only_uncertainty') & (df['budget']==b)]
        if ro.empty: continue
        ro=ro.iloc[0]
        for _,r in df[df['budget']==b].iterrows():
            d=r.to_dict(); d['delta_macro_f1_vs_response_only']=d.get('mean_macro_f1',float('nan'))-ro['mean_macro_f1']; d['delta_ece_vs_response_only']=d.get('mean_ece',float('nan'))-ro['mean_ece']; rows.append(d)
    pd.DataFrame(rows).to_csv(tables/'table5_adaptive_budget_results.csv', index=False)
# Ablations
abl_tags=['ablation_partial_responses_only','ablation_rt_only','ablation_partial_responses_plus_raw_rt','ablation_partial_responses_plus_content_rt_residual','ablation_partial_responses_plus_response_conditioned_rt_residual','ablation_partial_responses_plus_rt_residual_and_online_state']
abl=[]
for tag in abl_tags:
    f=tables/f'eval_{tag}_aggregate.csv'
    if f.exists():
        row=pd.read_csv(f); row['ablation']=tag.replace('ablation_',''); abl.append(row)
if abl:
    out=pd.concat(abl, ignore_index=True)
    out=pd.concat([out, pd.DataFrame([{'ablation':'posthoc_full_gmm_not_used','policy':'analysis_only','budget':37,'variant':'not_used_for_prediction'}])], ignore_index=True, sort=False)
    out.to_csv(tables/'table4_ablation_results.csv', index=False)
# downstream tests/figures
long=fit_rt_residuals()
bootstrap_main_comparison(n_boot=100)
run_subgroup_robustness(long)
run_leakage_tests(long)
generate_tables_figures(long)
