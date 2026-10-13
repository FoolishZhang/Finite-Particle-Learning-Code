"""REC-v2 independent arithmetic and fixed 8-model symmetric-root GH audit.

Does not import the new producer. --check replays saved arithmetic and small
1D marginal checks only, never the main MC bank generation or 4D integrations.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import itertools
import json
import math
import time
import traceback
import numpy as np
from scipy.stats import t
from rec1_verify import independent_pmf

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/rec_scale_v2_20261005'
OLD=ROOT/'artifacts/rec_v1_20261004/REC1'
Y=np.array(list(itertools.product((0,1),repeat=4)))
PAIRS=tuple(itertools.combinations(range(4),2))
ARMS=('NEAR_FREE','UNIT_FREE')
STATES=('BASE','MU_SAFE','SCALE2')
MODES=('retained','eliminated')
TARGETS=('F_D','F_P')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def close(a,b,tolerance=2e-12):
    error=float(np.max(abs(np.asarray(a)-np.asarray(b))))
    assert error<=tolerance,(error,tolerance)
    return error


def risk(P,Q):
    return math.fsum(float(p)*(math.log(float(p))-math.log(float(q))) for p,q in zip(P,Q))


def inputs():
    f=read(OUT/'FREEZE.json')
    for group in ('sources','inputs'):
        for path,digest in f[group].items():assert sha(ROOT/path)==digest,path
    assert f['qualification_sha256']==sha(OUT/'QUALIFICATION.json')
    return f


def arithmetic():
    inputs()
    results=read(OUT/'RESULTS.json')
    assert results['freeze_sha256']==sha(OUT/'FREEZE.json')
    truth=read(ROOT/'artifacts/active_scale_boundary/REFERENCE/RESULTS.json')
    P=np.array(next(r for r in truth['records'] if r['name']=='mixed')['teacher_pmf'])
    close(P,results['teacher_P']);p=P@Y
    marginal_x,marginal_w=np.polynomial.hermite.hermgauss(96)
    marginal_x*=math.sqrt(2);marginal_w/=math.sqrt(math.pi)
    bank_list=[];all_models=[]
    max_risk_error=max_score_error=max_marginal_error=0.
    root_calls=0;feasible=0
    for stream,record in enumerate(results['records']):
        assert record['stream']==stream
        assert sha(ROOT/record['model_file'])==record['model_sha256']
        assert sha(ROOT/record['score_file'])==record['score_sha256']
        saved=read(ROOT/record['model_file']);old=read(OLD/f'models_{stream:02}.json')
        source=read(ROOT/f'artifacts/joint_scale_learning/TRAIN/stream_{stream:02}.json')
        with np.load(ROOT/source['input_file'],allow_pickle=False) as data:
            labels=data['active_labels']
            empirical=np.array([np.count_nonzero(np.all(labels==y,axis=1))/4096 for y in Y])
        close(empirical,saved['empirical']);assert saved['reused_baseline']==old['baseline']
        assert len(saved['records'])==6
        for arm in ARMS:
            original=next(r for r in old['records'] if r['arm']==arm and r['state']=='BASE')
            base=next(r for r in saved['records'] if r['arm']==arm and r['state']=='BASE')
            safe=next(r for r in saved['records'] if r['arm']==arm and r['state']=='MU_SAFE')
            scale=next(r for r in saved['records'] if r['arm']==arm and r['state']=='SCALE2')
            assert base['model']==original['model']
            g=np.array(original['construction']['directions']['MU']['gradient'])
            close(safe['model']['mu'],np.clip(np.array(base['model']['mu'])-4*g,-2,2))
            for key in ('s','rho','tau'):close(safe['model'][key],base['model'][key],0.)
            for key in ('rho','tau'):close(scale['model'][key],base['model'][key],0.)
            meta=base['construction'];feasible+=meta['scale_feasible']
            root_calls+=sum(r['calls'] for r in meta['roots'])
            assert all(r['calls']<=102 for r in meta['roots'])
            assert not meta['teacher_used'] and not meta['evaluation_bank_used']
            if meta['scale_feasible']:
                close(scale['model']['s'],np.minimum(2.,2*np.array(base['model']['s'])))
            else:
                assert scale['model']==base['model']
            independent_marginals=[]
            for item in (base,scale):
                m=item['model']
                vector=np.array([marginal_w@(1/(1+np.exp(-mu-s*marginal_x))) for mu,s in zip(m['mu'],m['s'])])
                independent_marginals.append(vector)
            max_marginal_error=max(max_marginal_error,close(*independent_marginals,tolerance=5e-11))
            close(independent_marginals[0],meta['marginal_base96'])
            close(independent_marginals[1],meta['marginal_scale96'])
            for item in (base,scale):
                lower=[]
                for row,(i,j) in zip(item['capacity']['pairs'],PAIRS):
                    cov=4*(float(P@(Y[:,i]*Y[:,j]))-p[i]*p[j])
                    cap=.9*item['model']['s'][i]*item['model']['s'][j]/4
                    gap=max(0.,abs(cov)-cap)
                    close(row['teacher_spin_covariance'],cov)
                    close(row['capacity_upper_bound'],cap)
                    close(row['gap'],gap)
                    close(row['pair_KL_lower_bound'],gap*gap/8)
                    assert row['excludes_exact_pair_match']==(gap>1e-12)
                    lower.append(gap*gap/8)
                close(item['capacity']['maximum_KL_lower_bound'],max(lower))
        for row in saved['records']:
            original=next(r for r in old['records'] if r['arm']==row['arm'] and r['state']=='BASE')
            Q=np.array(row['main_pmf']);assert min(Q)>0;close(Q.sum(),1.)
            R=risk(P,Q);max_risk_error=max(max_risk_error,close(row['risk'],R))
            close(row['delta_risk'],R-original['risk'])
            nll=-float(empirical@np.log(Q))
            close(row['empirical_NLL'],nll)
            close(row['empirical_NLL_change'],nll-original['empirical_exact_NLL'])
            assert row['qualified']==(all(v['passed'] for v in row['qualification']) and row['marginal_passed'] and row['mean_descent_passed'])
        all_models+=saved['records']
        with np.load(ROOT/record['score_file'],allow_pickle=False) as data:banks=data['reduced_scores'].copy()
        assert banks.shape==(2048,2,3,2,2) and np.isfinite(banks).all();bank_list.append(banks)
        for row in record['stats']:
            ai,si,mi,ti=ARMS.index(row['arm']),STATES.index(row['state']),MODES.index(row['mode']),TARGETS.index(row['target'])
            values=banks[:,ai,si,mi,ti];difference=values-banks[:,ai,0,mi,ti]
            close(row['mean'],values.mean());close(row['delta'],difference.mean());close(row['bank_variance'],difference.var(ddof=1))
        if stream in (0,4,8,12):
            ra=np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([202610051210,stream,0])))
            rn=np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([202610051210,stream,1])))
            a=ra.standard_normal((32,100,4));n=rn.standard_normal((32,100,256))
            assert hashlib.sha256(a.tobytes()).hexdigest()==record['bank_hashes'][0]['active']
            assert hashlib.sha256(n.tobytes()).hexdigest()==record['bank_hashes'][0]['nuisance']
            for row in saved['records']:
                m=row['model'];z=a[0]@np.linalg.cholesky(m['covariance']).T+m['mu']
                probability=1/(1+np.exp(-z));noise=np.prod(1/(1+np.exp(-m['tau']*n[0])),axis=1)
                score=np.zeros((2,16))
                for j,y in enumerate(Y):
                    likelihood=np.prod(np.where(y,probability,1-probability),axis=1)
                    score[0,j]=-math.log(math.fsum(likelihood*noise)/100)-256*math.log(2)
                    score[1,j]=-math.log(math.fsum(likelihood)/100)
                expected=score@np.stack((empirical,P),axis=1)
                actual=banks[0,ARMS.index(row['arm']),STATES.index(row['state'])]
                max_score_error=max(max_score_error,close(expected,actual))
    critical=float(t.ppf(1-.05/32,2047));assert len(results['family'])==16
    for row in results['family']:
        ai,si,mi,ti=ARMS.index(row['arm']),STATES.index(row['candidate']),MODES.index(row['mode']),TARGETS.index(row['target'])
        differences=np.stack([b[:,ai,si,mi,ti]-b[:,ai,0,mi,ti] for b in bank_list])
        mean=float(differences.mean(axis=1).mean());se=float(np.sqrt(differences.var(axis=1,ddof=1).sum()/2048)/16)
        lo,hi=mean-critical*se,mean+critical*se
        close(row['delta'],mean);close(row['SE'],se);close(row['interval'],[lo,hi])
        assert row['direction']==('decrease' if hi<0 else 'increase' if lo>0 else 'unresolved')
        assert row['precision_passed']==(se<=2e-5)
    for row in results['risk']:
        selected=[r for r in all_models if r['arm']==row['arm'] and r['state']==row['state']]
        close(row['mean_risk'],np.mean([r['risk'] for r in selected]))
        close(row['paired_delta'],[r['delta_risk'] for r in selected])
        close(row['mean_delta'],np.mean([r['delta_risk'] for r in selected]))
        close(row['empirical_NLL_mean_change'],np.mean([r['empirical_NLL_change'] for r in selected]))
    assert feasible==results['feasible_scale_candidates'] and root_calls==results['marginal_root_calls']
    return dict(passed=True,source_sha256=sha(__file__),result_sha256=sha(OUT/'RESULTS.json'),
        models=96,risk_recomputations=96,capacity_pairs_recomputed=384,primary_MC_cells=16,
        max_risk_error=max_risk_error,max_direct_score_error=max_score_error,max_marginal_error=max_marginal_error,
        direct_pattern_scores=768,replayed_Gaussian=3328000,independent_1D_GH96_calls=256,
        scope='All saved arithmetic plus prescribed marginal and first-bank checks; no new scientific data or banks; not external expert reproduction')


def integrate():
    inputs();tick=time.perf_counter();result=read(OUT/'RESULTS.json');P=np.array(result['teacher_P'])
    save(OUT/'GH_AUDIT_EXECUTION.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),freeze_sha256=sha(OUT/'FREEZE.json'),automatic_retry=False))
    try:
        rows=[]
        for stream in (0,4,8,12):
            models=read(OUT/f'models_{stream:02}.json')['records']
            for arm in ARMS:
                row=next(r for r in models if r['arm']==arm and r['state']=='SCALE2');m=row['model']
                q=independent_pmf(m['mu'],np.array(m['covariance']),44)
                error=float(np.max(abs(np.array(q['pmf'])-row['main_pmf'])));risk_error=abs(risk(P,q['pmf'])-row['risk'])
                ok=error<=2e-7 and risk_error<=2e-6 and q['mass_error']<=2e-12
                rows.append(dict(stream=stream,arm=arm,state='SCALE2',independent=q,max_PMF_error=error,KL_error=risk_error,passed=ok))
                print(json.dumps(dict(stream=stream,arm=arm,passed=ok)),flush=True)
        save(OUT/'GH_AUDIT.json',dict(passed=all(r['passed'] for r in rows),records=rows,nodes=8*44**4,
            result_sha256=sha(OUT/'RESULTS.json'),source_sha256=sha(__file__),seconds=time.perf_counter()-tick))
    except Exception:
        save(OUT/'GH_AUDIT_ERROR.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('arithmetic','integrate','check'))
    command=parser.parse_args().command
    if command=='integrate':integrate()
    else:
        result=arithmetic()
        if command=='arithmetic':save(OUT/'ARITHMETIC_AUDIT.json',result)
        else:assert result==read(OUT/'ARITHMETIC_AUDIT.json')
        print(json.dumps(result))
