"""Compact, deterministic synthesis of frozen production results; no science rerun."""
from pathlib import Path
from fractions import Fraction
import argparse, hashlib, json, math
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/nuisance_learning'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def value(x):return Fraction(int(x['numerator']),int(x['denominator']))
def directed(x,digits,upper=False):
    f=value(x);scale=10**digits
    n=-((-f.numerator*scale)//f.denominator) if upper else (f.numerator*scale)//f.denominator
    return f'{n//scale}.{n%scale:0{digits}d}'
def derive():
    names=['RESULTS.json','EVALUATION.json','GRADIENT/RESULTS.json','GRADIENT/COUNTS.json',
           'GRADIENT/GH_REFERENCE.json','ACTIVE_TARGET/RESULTS.json','ACTIVE_TARGET/EXPECTED_RISKS.json',
           'ACTIVE_TARGET/ACTUAL_TARGETS.json','ACTIVE_TARGET/RB_COMPARISONS.json','ACTIVE_TARGET/QUALIFICATION.json']
    sources={f'artifacts/nuisance_learning/{name}':sha(OUT/name) for name in names}
    r=read(OUT/'RESULTS.json');e=read(OUT/'EVALUATION.json')['paired_cells']
    cells=[]
    for n in [0,254]:
      for N in [64,256,1024]:
       for B in [32,512,8192]:
        for K in [2,4]:
         rows=[x for x in e if (x['n'],x['N'],x['B'],x['K'])==(n,N,B,K)]
         assert len(rows)==16
         cells.append(dict(n=n,N=N,B=B,K=K,streams=16,
           **{key:math.fsum(x[key] for x in rows)/16 for key in ['raw_reference_KL','RB_reference_KL','table_MLE_reference_KL']},
           raw_endpoint_count=sum(abs(x['raw_grid_coordinate'])==5 for x in rows),
           analytic_paired_gain_count=sum(x['sufficient_true_KL_gain_gt_001'] for x in rows),
           descriptive_not_new_confidence_family=True))
    bounds=[]
    for x in r['endpoint_CP_family']:
        bounds.append(dict(K=x['K'],successes=x['successes'],trials=x['trials'],
          CP_lower=x['CP_lower'],raw_expected_KL_lower=x['true_raw_expected_KL_lower'],
          table_MLE_expected_KL_upper=x['true_table_MLE_expected_KL_upper'],
          expected_difference_lower=x['true_expected_raw_minus_table_MLE_lower'],
          raw_lower_display=directed(x['true_raw_expected_KL_lower'],6),
          MLE_upper_display=directed(x['true_table_MLE_expected_KL_upper'],6,True),
          gap_lower_display=directed(x['true_expected_raw_minus_table_MLE_lower'],6),
          outward_rounding=True))
    g=read(OUT/'GRADIENT/RESULTS.json');by={(x['n'],x['method']):x for x in g['estimators']}
    grad=dict(n=4094,old_banks_reused=65536,independent_confirmation=False,
       variance={method:by[(4094,method)]['variance_ddof1'] for method in ['Price','raw_path','noise_CV']},
       raw_over_Price_variance=by[(4094,'raw_path')]['variance_ddof1']/by[(4094,'Price')]['variance_ddof1'],
       B4096=[x for x in g['outer_B'] if x['n']==4094 and x['outer_B']==4096],
       n0_CV_exceeds_raw_variance=by[(0,'noise_CV')]['variance_ddof1']>by[(0,'raw_path')]['variance_ddof1'],
       GH_reference=read(OUT/'GRADIENT/GH_REFERENCE.json'))
    q=read(OUT/'ACTIVE_TARGET/QUALIFICATION.json');tar=read(OUT/'ACTIVE_TARGET/ACTUAL_TARGETS.json')['rows']
    comp=read(OUT/'ACTIVE_TARGET/RB_COMPARISONS.json')['rows'];ex=read(OUT/'ACTIVE_TARGET/EXPECTED_RISKS.json')['rows']
    active=[]
    for x in ex:
      if x['variant']=='primary':
        rows=[t for t in tar if (t['N'],t['K'])==(x['N'],x['K'])];assert len(rows)==16
        active.append(dict(N=x['N'],K=x['K'],reference_expected_risk_min=x['exact_reference_expected_KL_min'],
          reference_expected_risk_max=x['exact_reference_expected_KL_max'],
          analytic_upper=x['true_expected_KL_upper_for_any_argmin_selection'],
          table_MLE_reference_expected_risk=x['table_MLE']['exact_reference_expected_KL'],
          actual_count_target_reference_risk_min_mean=math.fsum(z['target_reference_risk_min'] for z in rows)/16,
          actual_count_target_reference_risk_max_mean=math.fsum(z['target_reference_risk_max'] for z in rows)/16,
          absolute_target_agreement_by_B={str(B):sum(z['target_absolute_rho_membership'] for z in comp if (z['N'],z['K'],z['B'])==(x['N'],x['K'],B)) for B in [32,512,8192]},
          post_hoc=True,strict_infinite_bank_target_certificate=False))
    return dict(source_sha256=sha(__file__),inputs=sources,learning_ledger=r['ledger'],
       primary=r['primary_sign_test'],CP_bounds=bounds,reference_cells=cells,
       gradient=grad,gradient_ledger=read(OUT/'GRADIENT/COUNTS.json'),
       active_target=active,active_target_ledger=read(OUT/'ACTIVE_TARGET/RESULTS.json')['ledger'],
       active_target_qualified=q['all_passed'],active_target_qualification_rows=len(q['count_state_rows'])+len(q['K2_null_cross_rows']),
       active_target_max_curve_spread=max(x['max_pointwise_curve_difference'] for x in q['count_state_rows']),
       new_scientific_computation=False)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['write','check']);a=p.parse_args();result=derive()
    path=OUT/'DERIVED_SUMMARY.json'
    if a.action=='write':
      with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    else:assert read(path)==result
    print(json.dumps(dict(passed=True,new_scientific_computation=False,primary_successes=result['primary']['successes'],
       CP_gap_lower_displays=[x['gap_lower_display'] for x in result['CP_bounds']],
       active_target_qualification_rows=result['active_target_qualification_rows'])))
