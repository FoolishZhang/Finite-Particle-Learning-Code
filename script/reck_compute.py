"""REC-v3 active-K comparison at immutable BASE/SCALE2 endpoints.

No private arrays are scientific inputs. New scores require a pre-execution
freeze; no endpoint is fitted or selected by this module.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import math
import platform
import subprocess
import time
import traceback
import numpy as np
import scipy
from scipy.special import expit, logsumexp, roots_hermitenorm
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/rec_k_v3_20261005'
DOC=ROOT/'docs/08_research_研究工作/REC_v3_20261005'
OLD=ROOT/'artifacts/rec_scale_v2_20261005'
TRUTH=ROOT/'artifacts/active_scale_boundary/REFERENCE/RESULTS.json'
Y=np.array(list(itertools.product((0,1),repeat=4)))
ARMS=('NEAR_FREE','UNIT_FREE')
STATES=('BASE','SCALE2')
TARGETS=('F_D','F_P')
KS=(100,400,1600)
BASE='3b48251b88cf3332a4f6be7e241d094015a29e9c'
HANDOFF='853358fe26a70335d6a5fb627372433694de9e410ebbccf0d6f9e49623e0c531'
CONFIG=dict(K=list(KS),streams=16,banks=1024,chunk=16,seed=202610051510,rng='PCG64DXSM',
    workers=4,threads=1,orders=[24,32,40],main_order=40,audit_streams=[0,4,8,12],audit_order=44,
    moment_tolerance=2e-7,coefficient_tolerance=2e-6,mass_tolerance=2e-12,relative_variance_floor=-2e-9,
    family=16,alpha=.05,df=1023,SE_limit=1e-5,mode='nuisance analytically eliminated; subtract 256 log 2')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def rel(path):return Path(path).relative_to(ROOT).as_posix()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def scores(model,epsilon,ks=KS):
    assert max(ks)<=epsilon.shape[1] and min(ks)>=1
    z=epsilon@np.linalg.cholesky(model['covariance']).T+model['mu']
    loga=z@Y.T-np.logaddexp(0.,z).sum(axis=2)[:,:,None]
    result=np.stack([math.log(k)-logsumexp(loga[:,:k],axis=1) for k in ks],axis=1)
    assert np.isfinite(result).all()
    return result


def moments(model,order):
    x,w=roots_hermitenorm(order);w=w/math.sqrt(2*math.pi)
    L=np.linalg.cholesky(model['covariance'])
    first=np.zeros(16,dtype=np.longdouble);second=np.zeros(16,dtype=np.longdouble)
    for start in range(0,order**4,65536):
        ids=np.array(np.unravel_index(np.arange(start,min(start+65536,order**4)),(order,)*4)).T
        z=x[ids]@L.T+model['mu'];weight=np.prod(w[ids],axis=1)
        plus,minus=expit(z),expit(-z)
        for j,y in enumerate(Y):
            a=np.prod(np.where(y,plus,minus),axis=1)
            first[j]+=np.sum(weight*a,dtype=np.longdouble)
            second[j]+=np.sum(weight*a*a,dtype=np.longdouble)
    return dict(order=order,root='cholesky',first=np.asarray(first,dtype=float).tolist(),
                second=np.asarray(second,dtype=float).tolist(),mass_error=abs(float(first.sum())-1),nodes=order**4)


def coefficients(moment,q,weights):
    relative=np.array(moment['second'])/np.array(q)**2-1
    return .5*relative@weights,relative


def qualify():
    tick=time.perf_counter()
    try:
        import rec1_diagnostics as legacy
        model=dict(mu=[-.4,-.1,.2,.5],covariance=np.diag(np.array([.2,.3,.4,.5])**2).tolist(),tau=0.)
        epsilon=np.sin(np.arange(3*7*4).reshape(3,7,4)*.37)
        actual=scores(model,epsilon,(1,3,7))
        direct=[];legacy_errors=[]
        for ki,k in enumerate((1,3,7)):
            z=epsilon[:,:k]@np.linalg.cholesky(model['covariance']).T+model['mu']
            p=1/(1+np.exp(-z))
            manual=np.stack([-np.log(np.prod(np.where(y,p,1-p),axis=2).mean(axis=1)) for y in Y],axis=1)
            direct.append(float(np.max(abs(manual-actual[:,ki]))))
            old=legacy.scores(model,epsilon[:,:k],np.zeros((3,k,1)))[:,1]
            legacy_errors.append(float(np.max(abs(old-actual[:,ki]))))
        constant=scores(model,np.zeros((2,7,4)),(1,3,7))
        constant_error=float(np.max(abs(constant-constant[:,0:1])))
        assert max(direct+legacy_errors+[constant_error])<2e-12
        made=moments(model,24)
        x,w=np.polynomial.hermite.hermgauss(96);x*=math.sqrt(2);w/=math.sqrt(math.pi)
        first=[];second=[]
        for y in Y:
            one=[];two=[]
            for i in range(4):
                p=1/(1+np.exp(-(model['mu'][i]+math.sqrt(model['covariance'][i][i])*x)))
                a=p if y[i] else 1-p
                one.append(float(w@a));two.append(float(w@(a*a)))
            first.append(math.prod(one));second.append(math.prod(two))
        q_error=float(np.max(abs(np.array(first)-made['first'])))
        m2_error=float(np.max(abs(np.array(second)-made['second'])))
        assert q_error<2e-12 and m2_error<2e-12 and made['mass_error']<2e-12
        assert sum(made['second'])<1 and np.min(np.array(made['second'])/np.array(first)**2-1)>0
        result=dict(passed=True,source_sha256=sha(__file__),audit_sha256=sha(ROOT/'script/reck_audit.py'),
            legacy_score_source_sha256=sha(legacy.__file__),direct_prefix_errors=direct,legacy_prefix_errors=legacy_errors,
            constant_particle_error=constant_error,first_moment_error=q_error,second_moment_error=m2_error,
            second_moments_not_normalized=True,seconds=time.perf_counter()-tick,
            environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
            scope='Deterministic manufactured arrays, no saved scientific endpoints or scientific RNG')
        save(OUT/'QUALIFICATION.json',result);print(json.dumps(result))
    except Exception:
        attempt=len(list(OUT.glob('QUALIFICATION_FAILURE_*.json')))
        save(OUT/f'QUALIFICATION_FAILURE_{attempt:02}.json',dict(source_sha256=sha(__file__),traceback=traceback.format_exc()))
        snapshot=OUT/f'QUALIFICATION_SOURCE_{attempt:02}.py'
        with snapshot.open('xb') as f:f.write(Path(__file__).read_bytes())
        raise


def source_inputs():
    sources=[Path(__file__),ROOT/'script/reck_audit.py',ROOT/'script/rec1_diagnostics.py',DOC/'CONTRACT.md',DOC/'THEORY.md']
    inputs=[TRUTH,OLD/'RESULTS.json',OLD/'DELIVERY_MANIFEST.json',OLD/'HANDOFF_INPUT.md']
    inputs += [OLD/f'models_{j:02}.json' for j in range(16)]
    return {rel(p):sha(p) for p in sources},{rel(p):sha(p) for p in inputs}


def freeze():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==BASE
    assert sha(ROOT/'HANDOFF.md')==HANDOFF
    q=read(OUT/'QUALIFICATION.json')
    assert q['passed'] and q['source_sha256']==sha(__file__) and q['audit_sha256']==sha(ROOT/'script/reck_audit.py')
    sources,inputs=source_inputs()
    for p in inputs:subprocess.check_call(['git','ls-files','--error-unmatch',p],cwd=ROOT,stdout=subprocess.DEVNULL)
    save(OUT/'FREEZE.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),authorized=True,baseline=BASE,
         config=CONFIG,sources=sources,inputs=inputs,qualification_sha256=sha(OUT/'QUALIFICATION.json'),
         feedback=dict(sha256=HANDOFF,reviewed_commit='9bdb4d532d187d9d9195d106663eb04d9d8f6185',use='reference; unchanged, no new feedback'),
         environment=q['environment'],inputs_all_previously_tracked=True))
    print(json.dumps(dict(frozen=True,sha256=sha(OUT/'FREEZE.json'))))


def validate():
    f=read(OUT/'FREEZE.json');assert f['config']==CONFIG and f['authorized']
    for group in ('sources','inputs'):
        for p,digest in f[group].items():assert sha(ROOT/p)==digest,p
    assert sha(OUT/'QUALIFICATION.json')==f['qualification_sha256']


def stream_run(stream):
    tick=time.perf_counter()
    try:
        old=read(OLD/f'models_{stream:02}.json');T=np.array(old['empirical'])
        truth=next(r for r in read(TRUTH)['records'] if r['name']=='mixed');P=np.array(truth['teacher_pmf'])
        weights=np.stack((T,P),axis=1);models=[]
        for arm,state in itertools.product(ARMS,STATES):
            row=next(r for r in old['records'] if r['arm']==arm and r['state']==state)
            assert row['qualified']
            variants=[moments(row['model'],n) for n in (24,32,40)]
            q=np.array(row['main_pmf']);main=variants[-1];B,rv=coefficients(main,q,weights)
            tests=[]
            for v in variants:
                c,_=coefficients(v,q,weights)
                first_error=float(np.max(abs(np.array(v['first'])-q)))
                moment_error=float(np.max(abs(np.array(v['second'])-main['second'])))
                coefficient_error=float(np.max(abs(c-B)))
                ok=first_error<=2e-7 and moment_error<=2e-7 and coefficient_error<=2e-6 and v['mass_error']<=2e-12
                tests.append(dict(order=v['order'],first_error=first_error,second_error=moment_error,coefficient_error=coefficient_error,mass_error=v['mass_error'],passed=bool(ok)))
            models.append(dict(arm=arm,state=state,model=row['model'],reference_q=q.tolist(),risk=row['risk'],
                exact_losses=[float(-T@np.log(q)),float(-P@np.log(q))],moments=variants,coefficients=B.tolist(),
                min_relative_variance=float(rv.min()),qualification=tests,
                moments_qualified=bool(all(v['passed'] for v in tests) and rv.min()>=-2e-9)))
        panel=OUT/f'panel_{stream:02}.json'
        save(panel,dict(stream=stream,source=rel(OLD/f'models_{stream:02}.json'),source_sha256=sha(OLD/f'models_{stream:02}.json'),
                       empirical=T.tolist(),teacher_P=P.tolist(),models=models))
        rng=np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([CONFIG['seed'],stream,0])))
        banks=np.empty((1024,2,2,3,2));hashes=[]
        for start in range(0,1024,16):
            epsilon=rng.standard_normal((16,1600,4))
            hashes.append(dict(start=start,Gaussian=hashlib.sha256(epsilon.tobytes()).hexdigest()))
            for ai,arm in enumerate(ARMS):
                for si,state in enumerate(STATES):
                    model=next(r['model'] for r in models if r['arm']==arm and r['state']==state)
                    banks[start:start+16,ai,si]=scores(model,epsilon)@weights
        scores_path=OUT/f'scores_{stream:02}.npz';assert not scores_path.exists()
        np.savez_compressed(scores_path,reduced_scores=banks)
        stats=[]
        for ai,arm in enumerate(ARMS):
            for ki,k in enumerate(KS):
                for ti,target in enumerate(TARGETS):
                    delta=banks[:,ai,1,ki,ti]-banks[:,ai,0,ki,ti]
                    stats.append(dict(kind='endpoint',arm=arm,K=k,target=target,mean=float(delta.mean()),variance=float(delta.var(ddof=1))))
            for ki,k in enumerate(KS[1:],start=1):
                high=banks[:,ai,1,ki,1]-banks[:,ai,0,ki,1]
                low=banks[:,ai,1,0,1]-banks[:,ai,0,0,1]
                difference=high-low
                stats.append(dict(kind='K_contrast',arm=arm,K=k,reference_K=100,target='F_P',mean=float(difference.mean()),variance=float(difference.var(ddof=1))))
        record=dict(stream=stream,panel_file=rel(panel),panel_sha256=sha(panel),score_file=rel(scores_path),score_sha256=sha(scores_path),
                    first_bank_scores=banks[0].tolist(),bank_hashes=hashes,stats=stats,
                    moments_qualified=all(r['moments_qualified'] for r in models),seconds=time.perf_counter()-tick)
        save(OUT/f'stream_{stream:02}.json',record)
        return record
    except Exception:
        save(OUT/f'ERROR_stream_{stream:02}.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise


def summary(records):
    family=[];critical=float(t.ppf(1-.05/32,1023))
    for prototype in records[0]['stats']:
        key=tuple(prototype[k] for k in ('kind','arm','K','target'))
        selected=[next(r for r in stream['stats'] if tuple(r[k] for k in ('kind','arm','K','target'))==key) for stream in records]
        mean=float(np.mean([r['mean'] for r in selected]));se=float(math.sqrt(sum(r['variance']/1024 for r in selected))/16)
        interval=[mean-critical*se,mean+critical*se]
        family.append(dict(kind=key[0],arm=key[1],K=key[2],target=key[3],mean=mean,SE=se,interval=interval,
            direction='decrease' if interval[1]<0 else 'increase' if interval[0]>0 else 'unresolved',precision_passed=se<=1e-5))
    panels=[read(ROOT/r['panel_file']) for r in records];exact=[];bias=[]
    for arm in ARMS:
        differences=[];coefficient_differences=[]
        for panel in panels:
            base=next(r for r in panel['models'] if r['arm']==arm and r['state']=='BASE')
            scale=next(r for r in panel['models'] if r['arm']==arm and r['state']=='SCALE2')
            differences.append(np.array(scale['exact_losses'])-base['exact_losses'])
            coefficient_differences.append(np.array(scale['coefficients'])-base['coefficients'])
        delta=np.mean(differences,axis=0);dB=np.mean(coefficient_differences,axis=0)
        exact.append(dict(arm=arm,exact_loss_delta=delta.tolist(),coefficient_delta=dB.tolist()))
        for ki,k in enumerate(KS):
            for ti,target in enumerate(TARGETS):
                cell=next(r for r in family if (r['kind'],r['arm'],r['K'],r['target'])==('endpoint',arm,k,target))
                increment=cell['mean']-float(delta[ti]);prediction=float(delta[ti]+dB[ti]/k)
                bias.append(dict(arm=arm,K=k,target=target,exact_delta=float(delta[ti]),MC_delta=cell['mean'],
                    bias_increment=increment,K_times_bias=k*increment,coefficient_delta=float(dB[ti]),
                    predicted_delta=prediction,MC_minus_prediction=cell['mean']-prediction,
                    interpretation='Prespecified descriptive second-order approximation, not a finite-K remainder bound or extra inference family'))
    return dict(config=CONFIG,freeze_sha256=sha(OUT/'FREEZE.json'),records=records,family=family,exact=exact,bias=bias,
        moments_qualified=all(r['moments_qualified'] for r in records),
        accounting=dict(new_observations=0,new_training=0,new_candidates=0,new_baseline_fits=0,fixed_models=64,
            moment_integrals=192,moment_nodes=64*sum(n**4 for n in (24,32,40)),audit_moment_integrals=8,audit_moment_nodes=8*44**4,
            prior_vectors=16*1024*1600,Gaussian=16*1024*1600*4,active_logits=16*1024*1600*4*4,
            pattern_scores=16*1024*4*3*16,particle_label_reductions=16*1024*4*sum(KS)*16,
            audit_replayed_Gaussian=4*16*1600*4,demo_replayed_Gaussian=16*1600*4),
        scope='Fixed endpoints and saved empirical weights; active-only mathematical finite objective; conditional MC inference, no training-population inference')


def run():
    validate();start=time.perf_counter()
    save(OUT/'EXECUTION.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),freeze_sha256=sha(OUT/'FREEZE.json'),automatic_retry=False))
    try:
        records=[]
        with ProcessPoolExecutor(max_workers=4) as pool:
            for future in as_completed([pool.submit(stream_run,j) for j in range(16)]):
                row=future.result();records.append(row)
                print(json.dumps(dict(stream=row['stream'],moments_qualified=row['moments_qualified'],seconds=row['seconds'])),flush=True)
        records.sort(key=lambda r:r['stream']);validate();result=summary(records);result['seconds']=time.perf_counter()-start
        save(OUT/'RESULTS.json',result)
        print(json.dumps(dict(completed=True,moments_qualified=result['moments_qualified'],seconds=result['seconds'])),flush=True)
    except Exception:
        save(OUT/'ERROR.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('qualify','freeze','run'))
    globals()[parser.parse_args().command]()
