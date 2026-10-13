"""Independent REC-v3 arithmetic, symmetric-root moments and public first-bank demo.

The demo reads only this script plus published panel_00.json and stream_00.json.
It never reads ignored arrays or the original observations.
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
from scipy.special import expit
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/rec_k_v3_20261005'
Y=np.array(list(itertools.product((0,1),repeat=4)))
ARMS=('NEAR_FREE','UNIT_FREE');STATES=('BASE','SCALE2');TARGETS=('F_D','F_P');KS=(100,400,1600)


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(p,value):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def close(a,b,tol=2e-12):
    error=float(np.max(abs(np.asarray(a)-np.asarray(b))))
    assert error<=tol,(error,tol)
    return error


def inputs():
    f=read(OUT/'FREEZE.json')
    for group in ('sources','inputs'):
        for p,digest in f[group].items():assert sha(ROOT/p)==digest,p
    assert sha(OUT/'QUALIFICATION.json')==f['qualification_sha256']


def first_bank(stream):
    row=read(OUT/f'stream_{stream:02}.json');panel=read(ROOT/row['panel_file'])
    assert sha(ROOT/row['panel_file'])==row['panel_sha256']
    rng=np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([202610051510,stream,0])))
    epsilon=rng.standard_normal((16,1600,4))
    assert hashlib.sha256(epsilon.tobytes()).hexdigest()==row['bank_hashes'][0]['Gaussian']
    output=np.empty((2,2,3,2));weights=np.array([panel['empirical'],panel['teacher_P']]).T
    for ai,arm in enumerate(ARMS):
        for si,state in enumerate(STATES):
            m=next(r['model'] for r in panel['models'] if r['arm']==arm and r['state']==state)
            z=epsilon[0]@np.linalg.cholesky(m['covariance']).T+m['mu']
            scores=np.empty((3,16))
            for j,y in enumerate(Y):
                likelihood=np.prod(expit((2*y-1)*z),axis=1)
                for ki,k in enumerate(KS):scores[ki,j]=-math.log(math.fsum(likelihood[:k])/k)
            output[ai,si]=scores@weights
    error=close(output,row['first_bank_scores'])
    return dict(passed=True,stream=stream,max_score_error=error,replayed_Gaussian=102400,
        pattern_scores=192,scope='One existing bank, not a population preference estimate',
        candidate_minus_base_F_P={arm:(output[ai,1,:,1]-output[ai,0,:,1]).tolist() for ai,arm in enumerate(ARMS)})


def arithmetic():
    inputs();result=read(OUT/'RESULTS.json')
    assert result['freeze_sha256']==sha(OUT/'FREEZE.json')
    truth=read(ROOT/'artifacts/active_scale_boundary/REFERENCE/RESULTS.json')
    P=np.array(next(r for r in truth['records'] if r['name']=='mixed')['teacher_pmf'])
    banks_all=[];panels=[];max_coefficient_error=0.;max_direct_error=0.
    for stream,record in enumerate(result['records']):
        assert record['stream']==stream
        assert sha(ROOT/record['panel_file'])==record['panel_sha256']
        assert sha(ROOT/record['score_file'])==record['score_sha256']
        panel=read(ROOT/record['panel_file']);old=read(ROOT/panel['source'])
        assert sha(ROOT/panel['source'])==panel['source_sha256']
        close(panel['teacher_P'],P);close(panel['empirical'],old['empirical'])
        T=np.array(panel['empirical']);panels.append(panel)
        for row in panel['models']:
            prior=next(r for r in old['records'] if r['arm']==row['arm'] and r['state']==row['state'])
            assert row['model']==prior['model']
            close(row['reference_q'],prior['main_pmf'],0.)
            Q=np.array(row['reference_q']);M=np.array(row['moments'][-1]['second'])
            assert [v['order'] for v in row['moments']]==[24,32,40]
            coeff=[.5*math.fsum(float(w)*(float(m)/float(q)**2-1) for w,m,q in zip(weights,M,Q)) for weights in (T,P)]
            max_coefficient_error=max(max_coefficient_error,close(coeff,row['coefficients']))
            close(row['exact_losses'],[-T@np.log(Q),-P@np.log(Q)])
            close(row['risk'],math.fsum(float(p)*(math.log(float(p))-math.log(float(q))) for p,q in zip(P,Q)))
            assert row['moments_qualified']==(all(v['passed'] for v in row['qualification']) and row['min_relative_variance']>=-2e-9)
        with np.load(ROOT/record['score_file'],allow_pickle=False) as f:banks=f['reduced_scores'].copy()
        assert banks.shape==(1024,2,2,3,2) and np.isfinite(banks).all()
        close(banks[0],record['first_bank_scores']);banks_all.append(banks)
        assert [h['start'] for h in record['bank_hashes']]==list(range(0,1024,16))
        for row in record['stats']:
            ai=ARMS.index(row['arm']);ki=KS.index(row['K']);ti=TARGETS.index(row['target'])
            delta=banks[:,ai,1,ki,ti]-banks[:,ai,0,ki,ti]
            if row['kind']=='K_contrast':delta-=banks[:,ai,1,0,ti]-banks[:,ai,0,0,ti]
            close(row['mean'],delta.mean());close(row['variance'],delta.var(ddof=1))
        if stream in (0,4,8,12):max_direct_error=max(max_direct_error,first_bank(stream)['max_score_error'])
    critical=float(t.ppf(1-.05/32,1023));assert len(result['family'])==16
    for row in result['family']:
        ai=ARMS.index(row['arm']);ki=KS.index(row['K']);ti=TARGETS.index(row['target'])
        deltas=[]
        for banks in banks_all:
            d=banks[:,ai,1,ki,ti]-banks[:,ai,0,ki,ti]
            if row['kind']=='K_contrast':d=d-(banks[:,ai,1,0,ti]-banks[:,ai,0,0,ti])
            deltas.append(d)
        deltas=np.stack(deltas);mean=float(deltas.mean(axis=1).mean());se=float(np.sqrt(deltas.var(axis=1,ddof=1).sum()/1024)/16)
        lo,hi=mean-critical*se,mean+critical*se
        close(row['mean'],mean);close(row['SE'],se);close(row['interval'],[lo,hi])
        assert row['precision_passed']==(se<=1e-5)
        assert row['direction']==('decrease' if hi<0 else 'increase' if lo>0 else 'unresolved')
    for row in result['exact']:
        loss=[];coef=[]
        for panel in panels:
            base=next(x for x in panel['models'] if x['arm']==row['arm'] and x['state']=='BASE')
            scale=next(x for x in panel['models'] if x['arm']==row['arm'] and x['state']=='SCALE2')
            loss.append(np.array(scale['exact_losses'])-base['exact_losses']);coef.append(np.array(scale['coefficients'])-base['coefficients'])
        close(row['exact_loss_delta'],np.mean(loss,axis=0));close(row['coefficient_delta'],np.mean(coef,axis=0))
    for row in result['bias']:
        exact=next(x for x in result['exact'] if x['arm']==row['arm']);ti=TARGETS.index(row['target'])
        cell=next(x for x in result['family'] if (x['kind'],x['arm'],x['K'],x['target'])==('endpoint',row['arm'],row['K'],row['target']))
        close(row['exact_delta'],exact['exact_loss_delta'][ti]);close(row['MC_delta'],cell['mean'])
        close(row['bias_increment'],cell['mean']-row['exact_delta'])
        close(row['predicted_delta'],row['exact_delta']+exact['coefficient_delta'][ti]/row['K'])
        close(row['K_times_bias'],row['K']*row['bias_increment'])
    return dict(passed=True,source_sha256=sha(__file__),result_sha256=sha(OUT/'RESULTS.json'),
        fixed_models=64,primary_MC_cells=16,max_coefficient_difference=max_coefficient_error,
        max_direct_score_difference=max_direct_error,pattern_scores_replayed=768,Gaussian_replayed=409600,
        scope='All saved paired-bank arithmetic, moment coefficient arithmetic and prescribed direct-score replay; no new scientific banks or endpoints')


def independent_moments(model,order):
    eig,U=np.linalg.eigh(model['covariance']);assert eig.min()>0
    root=(U*np.sqrt(eig))@U.T
    x,w=np.polynomial.hermite.hermgauss(order);x*=math.sqrt(2);w/=math.sqrt(math.pi)
    first=np.zeros(16);second=np.zeros(16)
    for begin in range(0,order**4,65536):
        ids=np.array(np.unravel_index(np.arange(begin,min(begin+65536,order**4)),(order,)*4)).T
        z=x[ids]@root.T+model['mu'];weight=np.prod(w[ids],axis=1)
        for j,y in enumerate(Y):
            a=np.prod(expit((2*y-1)*z),axis=1)
            first[j]+=np.dot(weight,a);second[j]+=np.dot(weight,a*a)
    return dict(order=order,root='symmetric_numpy_hermite',first=first.tolist(),second=second.tolist(),mass_error=abs(float(first.sum())-1),nodes=order**4)


def integrate():
    inputs();start=time.perf_counter()
    save(OUT/'MOMENT_AUDIT_EXECUTION.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),freeze_sha256=sha(OUT/'FREEZE.json'),automatic_retry=False))
    try:
        rows=[]
        for stream in (0,4,8,12):
            panel=read(OUT/f'panel_{stream:02}.json');weights=np.array([panel['empirical'],panel['teacher_P']]).T
            for arm in ARMS:
                row=next(x for x in panel['models'] if x['arm']==arm and x['state']=='SCALE2')
                own=independent_moments(row['model'],44);q=np.array(row['reference_q'])
                second_error=float(np.max(abs(np.array(own['second'])-row['moments'][-1]['second'])))
                B=.5*(np.array(own['second'])/q**2-1)@weights
                coefficient_error=float(np.max(abs(B-row['coefficients'])))
                first_error=float(np.max(abs(np.array(own['first'])-q)))
                ok=second_error<=2e-7 and coefficient_error<=2e-6 and first_error<=2e-7 and own['mass_error']<=2e-12
                rows.append(dict(stream=stream,arm=arm,state='SCALE2',independent=own,second_error=second_error,
                                 coefficient_error=coefficient_error,first_error=first_error,qualified=ok))
                print(json.dumps(dict(stream=stream,arm=arm,qualified=ok)),flush=True)
        save(OUT/'MOMENT_AUDIT.json',dict(completed=True,qualified=all(r['qualified'] for r in rows),records=rows,
             result_sha256=sha(OUT/'RESULTS.json'),source_sha256=sha(__file__),nodes=8*44**4,seconds=time.perf_counter()-start))
    except Exception:
        save(OUT/'MOMENT_AUDIT_ERROR.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('demo','arithmetic','check','integrate'))
    command=parser.parse_args().command
    if command=='demo':print(json.dumps(first_bank(0),indent=2))
    elif command=='integrate':integrate()
    else:
        result=arithmetic()
        if command=='arithmetic':save(OUT/'ARITHMETIC_AUDIT.json',result)
        else:assert result==read(OUT/'ARITHMETIC_AUDIT.json')
        print(json.dumps(result))
