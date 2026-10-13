"""Independent expected-score evaluation of mechanically enumerated outputs."""
from __future__ import annotations
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import argparse,hashlib,json,time,traceback,math
import numpy as np
from scipy.special import expit
import scbm6eval_mc as helper
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/scbm_budget_allocation';OUT=BASE/'EVAL'
HELPER_SHA='ced8d96024d95c2b81abf50076bba9901ffae8639cbcdf3fbde4fe0e20e1edd1'
NATIVE_INDICES=[0,1,2,3,34,35,66,67,98,99]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x,new=True):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x' if new else 'w',encoding='utf-8',newline='\n') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def contract(derived=True):
    assert sha(helper.__file__)==HELPER_SHA
    freeze=read(BASE/'FREEZE.json');assert freeze['run_authorized']
    for p,h in freeze['inputs'].items():assert sha(ROOT/p)==h,p
    assert freeze['inputs']['script/scbmalloc_eval.py']==sha(__file__)
    if derived:
        f=read(BASE/'DERIVED_FREEZE.json');assert f['run_authorized'] and f['root_freeze_sha256']==sha(BASE/'FREEZE.json')
        for p,h in f['inputs'].items():assert sha(ROOT/p)==h,p
    cfg=read(BASE/'CONFIG.json')['evaluation'];assert cfg['groups']==32 and cfg['banks_per_group']==1024 and cfg['candidates']==138
    return cfg
def selftest():
    assert not (BASE/'FREEZE.json').exists();assert sha(helper.__file__)==HELPER_SHA
    data=read(BASE/'DATA.json');mu=data['mu'];a=np.sin(np.arange(2*7*4)*.03).reshape(2,7,4);n=np.cos(np.arange(2*7*3)*.041).reshape(2,7,3)
    matrices=np.stack([np.eye(4),np.eye(4)*.8+.2*np.ones((4,4)),np.eye(4)*1.1-.1*np.ones((4,4))]);lowers=np.linalg.cholesky(matrices)
    values,_=helper.evaluate_block(a,n,lowers,mu);direct=np.empty_like(values)
    for b in range(2):
        for j in range(3):
            z=a[b]@lowers[j].T+mu
            for y,bits in enumerate(helper.PATTERNS):
                p=expit((2*bits-1)*z).prod(axis=1);u=expit(-n[b]).prod(axis=1)
                direct[b,0,j,y]=-math.log(float((p*u).mean()));direct[b,1,j,y]=3*math.log(2)-math.log(float(p.mean()))
    error=float(np.max(abs(values-direct)));assert error<2e-14
    save(OUT/'SELFTEST.json',dict(passed=True,source_sha256=sha(__file__),helper_sha256=HELPER_SHA,config_sha256=sha(BASE/'CONFIG.json'),max_error=error,new_RNG=0,new_model_integrals=0,new_training=0))
    print(json.dumps(dict(passed=True,max_error=error)))
def run_group(g):
    cfg=contract();c=read(BASE/'CANDIDATES.json');records=c['candidates'];assert len(records)==138
    matrices=np.array([x['covariance'] for x in records]);lowers=np.linalg.cholesky(matrices);mu=read(BASE/'DATA.json')['mu']
    rng={k:np.random.Generator(np.random.PCG64(np.random.SeedSequence([cfg['seed_base'],g,v]))) for k,v in cfg['roles'].items()}
    initial={k:x.bit_generator.state for k,x in rng.items()};digest={k:hashlib.sha256() for k in rng};blocks=[];prefix={};controls=None;beg=time.perf_counter()
    for block in range(16):
        a=rng['active'].standard_normal((64,100,4));n=rng['noise'].standard_normal((64,100,256));digest['active'].update(a.tobytes());digest['noise'].update(n.tobytes())
        values,_=helper.evaluate_block(a,n,lowers,mu);blocks.append(values.mean(axis=0))
        if block==0:
            prefix=dict(active_eps=a[:2].copy(),noise_eps=n[:2].copy(),individual_total_score=values[:2].copy())
            if g==0:controls=helper.native_controls(a[:2],n[:2],values[:2],matrices,mu,NATIVE_INDICES,1e-4)
    prefix['block_means']=np.stack(blocks);means=prefix['block_means'].mean(axis=0);assert means.shape==(2,138,16)
    p=OUT/f'group_{g:02}_prefix.npz'
    with p.open('xb') as f:np.savez_compressed(f,**prefix)
    row=dict(group=g,means=means.tolist(),prefix_npz=p.relative_to(ROOT).as_posix(),prefix_sha256=sha(p),prefix_array_sha256={k:ah(v) for k,v in prefix.items()},role_seed_sequences={k:[cfg['seed_base'],g,v] for k,v in cfg['roles'].items()},RNG_initial_states=initial,RNG_final_states={k:x.bit_generator.state for k,x in rng.items()},role_input_sha256={k:x.hexdigest() for k,x in digest.items()},consumed_banks=1024,completed_blocks=16,numerical_controls=controls,source_sha256=sha(__file__),freeze_sha256=sha(BASE/'FREEZE.json'),derived_freeze_sha256=sha(BASE/'DERIVED_FREEZE.json'),seconds=time.perf_counter()-beg)
    save(OUT/f'group_{g:02}.json',row);return row
def run():
    cfg=contract();test=read(OUT/'SELFTEST.json');assert test['passed'] and test['source_sha256']==sha(__file__)
    assert not (OUT/'EXECUTION.json').exists() and not list(OUT.glob('group_*.json'))
    save(OUT/'EXECUTION.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),freeze_sha256=sha(BASE/'FREEZE.json'),derived_freeze_sha256=sha(BASE/'DERIVED_FREEZE.json'),automatic_retry=False))
    beg=time.perf_counter();rows={};active={};nxt=0;pool=ProcessPoolExecutor(max_workers=2)
    try:
        while nxt<2:active[pool.submit(run_group,nxt)]=nxt;nxt+=1
        while active:
            done,_=wait(active,return_when=FIRST_COMPLETED)
            fresh=[f.result() for f in done]
            for f in done:del active[f]
            for row in fresh:rows[row['group']]=row;print(json.dumps(dict(completed_group=row['group'])),flush=True)
            save(OUT/'PROGRESS.json',dict(completed=sorted(rows),running=sorted(active.values())),False)
            while len(active)<2 and nxt<32:active[pool.submit(run_group,nxt)]=nxt;nxt+=1
        contract();ordered=[rows[g] for g in range(32)];qual=ordered[0]['numerical_controls'];assert qual['official_API_calls']==320
        result=dict(source_sha256=sha(__file__),helper_sha256=HELPER_SHA,freeze_sha256=sha(BASE/'FREEZE.json'),derived_freeze_sha256=sha(BASE/'DERIVED_FREEZE.json'),candidates_sha256=sha(BASE/'CANDIDATES.json'),groups=ordered,numerical_qualification=qual,profile_order=['raw','FACT'],accounting=dict(groups=32,independent_observation_banks=32768,prior_vectors=3276800,Gaussian_scalars=851968000,mathematical_bank_pattern_scores=32768*2*138*16,original_API_calls=320,new_observations=0,new_training=0,new_model_integrals=0),seconds=time.perf_counter()-beg,scope='Mathematical64 finite-K scores, conditional on fixed trained candidates. Per-bank inner K then outer averaging; no pooled K. Native32 qualification is restricted to first2 banks/group0, ten matrices.')
        save(OUT/'RESULTS.json',result);save(OUT/'STATE.json',dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),results_sha256=sha(OUT/'RESULTS.json')))
        print(json.dumps(dict(complete=True,seconds=result['seconds'],native_passed=qual['passed'],results_sha256=sha(OUT/'RESULTS.json'))))
    except BaseException:
        for f in active:f.cancel()
        save(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),completed=sorted(rows),already_active=sorted(active.values()),automatic_retry=False));raise
    finally:pool.shutdown(wait=True,cancel_futures=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['check','run']);a=p.parse_args();selftest() if a.mode=='check' else run()
