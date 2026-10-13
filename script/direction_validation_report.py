"""Author readout and static comparison from the sealed-feasibility tables.

No candidate generation, sampling, optimization, confidence realization or
outcome-based choice of the original twelve objects.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/direction_validation_v1_20261007'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def n_for(plan,key):return plan['minimum_n'][key]['eb_proxy']['n']

def facts():
    s=read(OUT/'SUMMARY.json')['derived'];a=read(OUT/'AUDIT.json')
    assert a['passed'] and a['numerical_qualification']['GL_all_qualified']
    rows=s['rows'];assert len(rows)==12
    eligible=[r for r in rows if r['curvature']['moments']['mean']>1e-4]
    summary=[]
    for r in eligible:
        c=r['curvature'];fs=r['finite_gain'];bs=r['bridge'];tr=r['transfer']
        summary.append(dict(unit=r['unit'],training_N=r['N'],curvature_support_mean=c['moments']['mean'],score_SD=c['moments']['std'],
            curvature_EB_proxy_n_for_margin_1e_minus4=n_for(c['planning'],'useful'),
            per_t=[dict(t=b['t'],direct_gain=f['moments']['mean'],normalized_score_correlation=b['correlation']['value'],
                SD_ratio=b['SD_ratio_Xt_over_Xc']['value'],SNR_ratio=b['SNR_ratio_Xt_over_Xc']['value'],
                direct_matched_margin_n=n_for(f['planning'],'matched_curvature'),
                direct_over_curvature_matched_n=b['matched_margin_required_n']['eb_proxy']['finite_over_curvature']['value'],
                direct_n_gain_1e_minus5=n_for(f['planning'],'gain_1e_5'),direct_n_gain_1e_minus4=n_for(f['planning'],'gain_1e_4'),
                e_minus=t['main_e_minus'],transfer_floor_with_known_curvature_mean=t['normalized_mean_gain_lower_from_curvature'],
                transfer_zero_support_absent=t['thresholds']['zero']['transfer_gives_no_support_even_with_unlimited_data'])
                for f,b,t in zip(fs,bs,tr)]))
    gains=[f['moments']['mean'] for r in rows for f in r['finite_gain']]
    record=dict(source_summary_sha256=sha(OUT/'SUMMARY.json'),source_audit_sha256=sha(OUT/'AUDIT.json'),
        fixed_objects=12,fixed_candidates=36,objects_with_curvature_support_above_1e_minus4=len(eligible),positive_finite_gains=sum(x>0 for x in gains),
        finite_gains_above_1e_minus5=sum(x>1e-5 for x in gains),finite_gains_above_1e_minus4=sum(x>1e-4 for x in gains),
        maximum_fixed_candidate_gain=max(gains),positive_object_table=summary,
        zero_oracle_EB_proxy_useful_margin_at_n16384=not any(row['target_checks']['useful']['mean_minus_eb_proxy_exceeds_target'] for r in rows for row in r['curvature']['planning']['rows'] if row['n']==16384),
        zero_oracle_EB_proxy_useful_margin_at_n65536=not any(row['target_checks']['useful']['mean_minus_eb_proxy_exceeds_target'] for r in rows for row in r['curvature']['planning']['rows'] if row['n']==65536),
        oracle_filter=s['oracle_retrospective']['all_units'],new_validation_samples=0,new_training=False,
        planning_n_is_not_necessary_sample_size=True,planning_is_not_observed_CI_or_power=True,
        statistical_method_is_preexisting=True,curvature_transfer_failure_not_candidate_failure=True,
        no_claim_against_all_confidence_methods=True)
    return record

def figure(record):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'svg.hashsalt':'direction-validation-v1'})
    table=record['positive_object_table'];fig,ax=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    colors=['#267D9D','#AC5B34','#648247']
    for r,color in zip(table,colors):
        ts=[q['t'] for q in r['per_t']];ratio=[q['direct_over_curvature_matched_n'] for q in r['per_t']]
        ax[0].plot(ts,ratio,'o-',color=color,label=r['unit'])
    ax[0].axhline(1,color='#777',linewidth=.8);ax[0].set(xlabel='Fixed angle amplitude t',ylabel='Direct / curvature planning n',
        title='Matched claims: small-step data needs converge');ax[0].legend(fontsize=8,loc='upper left')
    labels=[r['unit'].replace('D3_','') for r in table];x=np.arange(len(table));direct=[r['per_t'][2]['direct_gain']*1e5 for r in table]
    transfer=[r['per_t'][2]['transfer_floor_with_known_curvature_mean']*1e5 for r in table]
    ax[1].bar(x-.18,direct,.34,color='#267D9D',label='Direct candidate gain')
    ax[1].bar(x+.18,transfer,.34,color='#AC5B34',label='Curvature transfer floor at known mean')
    ax[1].axhline(0,color='#777',linewidth=.8);ax[1].set_xticks(x,labels)
    ax[1].set(ylabel='Finite risk gain (units of 1e-5 nat)',title='t = 0.25: positive gains, negative transfer floors');ax[1].legend(fontsize=8,loc='upper right')
    for axis in ax:axis.grid(alpha=.18);axis.set_axisbelow(True)
    fig.suptitle('Oracle EB planning for negative-curvature objects; all 12 retained, no validation data',fontsize=11)
    for ext in ('png','svg'):
        path=OUT/('feasibility_comparison_final.'+ext)
        if path.exists():raise FileExistsError(path)
        fig.savefig(path,dpi=180)
    plt.close(fig)

def main(check):
    f=facts();path=OUT/'AUTHOR_READOUT_FINAL.json'
    if check:
        r=read(path);assert r['facts']==f and r['source_sha256']==sha(__file__)
        for name,h in r['figures'].items():assert sha(ROOT/name)==h
    else:
        if path.exists():raise FileExistsError(path)
        figure(f);save(path,dict(source_sha256=sha(__file__),facts=f,
            figures={(OUT/('feasibility_comparison_final.'+ext)).relative_to(ROOT).as_posix():sha(OUT/('feasibility_comparison_final.'+ext)) for ext in ('png','svg')}))
    print(json.dumps(dict(passed=True,check=check,new_PMF_calls=0,new_validation_samples=0,old_objects=12)))
if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--run',action='store_true');g.add_argument('--check',action='store_true');a=p.parse_args();main(a.check)
