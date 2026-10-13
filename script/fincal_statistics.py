"""Frozen paired development readout, conditional on one shared CAL model."""
from pathlib import Path
import argparse,hashlib,json,math
import numpy as np
from scipy.stats import t as student_t

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/finite_calibration/TRAIN'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8'))
def save(p,x):
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def freeze():
    assert not (OUT/'STATISTICS_PROTOCOL.json').exists()
    f=read(OUT/'FREEZE.json')
    save(OUT/'STATISTICS_PROTOCOL.json',dict(source_sha256=sha(__file__),freeze_sha256=sha(OUT/'FREEZE.json'),
        model_sha256=f['reference_sha256'],family=f['risk_family'],mesh_gate=f['mesh_discretization_gate'],
        descriptive_intervals='two-sided df15 t at alpha=.05; not simultaneous and not distribution-free',
        primary_intervals='two-sided df15 t at alpha=.05/4 per contrast; conditional on common frozen CAL model',
        independent_units=16,shared_teacher_and_N_prefixes=True,no_outcome_based_extra_comparisons=True,
        degenerate_interval_policy='zero observed SE is reported and flagged, not called a population equality proof'))
    print('Statistics protocol frozen')


def summary(values,critical):
    x=np.asarray(values,dtype=float);mean=float(x.mean());se=float(x.std(ddof=1)/math.sqrt(len(x)))
    return dict(n=len(x),values=x.tolist(),mean=mean,SE=se,lower=mean-critical*se,upper=mean+critical*se,
                zero_observed_SE=bool(se==0),interval_interpretation='approximate paired/stream t interval conditional on frozen calibration')


def run():
    protocol=read(OUT/'STATISTICS_PROTOCOL.json'); result=read(OUT/'RESULTS.json')
    assert protocol['source_sha256']==sha(__file__)
    assert protocol['freeze_sha256']==sha(OUT/'FREEZE.json')
    assert result['reference_sha256']==protocol['model_sha256']
    assert len(result['records'])==16
    rows=[x for rec in result['records'] for x in rec['outputs']]
    assert len(rows)==result['output_count']==672
    lookup={(x['stream'],x['theta'],x['N'],x['method']):x for x in rows}
    tc=float(student_t.ppf(.975,15));family_tc=float(student_t.ppf(1-.05/(2*4),15))
    descriptive=[]
    for N in [1024,4096]:
        for theta in [0.,.3,.6]:
            for method in result['methods']:
                selected=[lookup[(i,theta,N,method)] for i in range(16)]
                r=summary([x['reference_joint_KL'] for x in selected],tc)
                r.update(theta=theta,N=N,method=method,rhos=[x['rho'] for x in selected],
                         mean_rho=float(np.mean([x['rho'] for x in selected])),
                         endpoint_count=sum(abs(x['rho'])==.75 for x in selected))
                descriptive.append(r)
    primary=[]
    gain=[lookup[(i,.3,4096,'raw')]['reference_joint_KL']-lookup[(i,.3,4096,'finitecal')]['reference_joint_KL'] for i in range(16)]
    r=summary(gain,family_tc);r.update(id='theta03_finite_improves_raw',theta=.3,N=4096,
        contrast='raw_minus_finite',threshold=.001,pass_=bool(r['lower']>.001))
    primary.append(r)
    for theta in [0.,.3,.6]:
        excess=[lookup[(i,theta,4096,'finitecal')]['reference_joint_KL']-lookup[(i,theta,4096,'exact')]['reference_joint_KL'] for i in range(16)]
        r=summary(excess,family_tc);r.update(id=f'theta{theta:g}_finite_noninferior_MLE',theta=theta,N=4096,
            contrast='finite_minus_exact',threshold=.0002,pass_=bool(r['upper']<.0002))
        primary.append(r)
    certificates=[]
    for row in rows:
        if 'continuous_objective_regret_bound_real_arithmetic' in row:
            value=row['continuous_objective_regret_bound_real_arithmetic']
            certificates.append(dict(stream=row['stream'],theta=row['theta'],N=row['N'],method=row['method'],
                bound=value,pass_=bool(value<=protocol['mesh_gate'])))
    assert len(certificates)==480
    # Fixed observed signals, not estimated anew from these training outputs.
    signals=[]
    for theta in [0.,.3,.6]:
        t=lookup[(0,theta,4096,'exact')]['teacher_r']
        for N in [1024,4096]:signals.append(dict(theta=theta,N=N,r_theta=t,sqrtN_r=math.sqrt(N)*t,
            signal_over_pair_standard_error=math.sqrt(N)*t/math.sqrt(1-t*t)))
    save(OUT/'STATISTICS.json',dict(source_sha256=sha(__file__),protocol_sha256=sha(OUT/'STATISTICS_PROTOCOL.json'),
        results_sha256=sha(OUT/'RESULTS.json'),reference_sha256=protocol['model_sha256'],
        descriptive=descriptive,primary=primary,primary_passed=sum(r['pass_'] for r in primary),primary_count=4,
        t_critical_descriptive=tc,t_critical_family=family_tc,conditional_on_one_frozen_calibration=True,
        paired_t_is_development_approximation=True,certificates=certificates,
        mesh_passed=sum(x['pass_'] for x in certificates),mesh_count=480,signals=signals,
        counts=dict(independent_base_streams=16,outputs=672,grid_near_continuous_outputs=480,
                    numerical_continuous_MLE_outputs=96,zero_outputs=96),new_scientific_samples=0))
    print(json.dumps(dict(primary_passed=sum(r['pass_'] for r in primary),mesh_passed=sum(x['pass_'] for x in certificates),
        primary=[{k:r[k] for k in ['id','mean','lower','upper','pass_']} for r in primary],
        highN_means=[{k:r[k] for k in ['theta','method','mean','mean_rho']} for r in descriptive if r['N']==4096])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','run']);a=p.parse_args()
    freeze() if a.mode=='freeze' else run()
