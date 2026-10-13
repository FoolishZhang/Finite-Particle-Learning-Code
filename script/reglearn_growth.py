"""Frozen growing-K population score panel, no observed-data training.

Four active observation patterns share one canonical nuisance draw per bank.
Gaussian symmetry makes this a population estimator, not an empirical-data substitute.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/regularization_learning/GROWTH'
RHO=np.linspace(-.75,.75,21)
NS=np.array([14,62,254],dtype=int)
SIGNS=np.array([[-1,-1],[-1,1],[1,-1],[1,1]],dtype=int)
STREAMS=16
BANKS=1024
BLOCK=64
KMAX=DMAX=256
SEED_BASE=202610020700


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p,data):
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/'FREEZE.json'
    assert not p.exists(), 'freeze already exists'
    plan=ROOT/'docs/08_research_研究工作/统一正则与真实相关学习_20261002/PLAN.md'
    assert sha(plan)=='b1f1363b7238e34fa7acafd0be6772f4c2cddc313352e1e31a1161e09657e04a'
    save(p,dict(source_sha256=sha(__file__),plan_sha256=sha(plan),
        ns=NS.tolist(),ks=(NS+2).tolist(),rho=RHO.tolist(),signs=SIGNS.tolist(),
        streams=STREAMS,banks_per_stream=BANKS,block=BLOCK,
        rng='numpy Generator(PCG64(seed_base+stream)); standard_normal((64,256,256)) sequential blocks',
        seed_base=SEED_BASE,new_Gaussian_scalars=STREAMS*BANKS*KMAX*DMAX,
        unique_prior_vectors=STREAMS*BANKS*KMAX, independent_streams=STREAMS,
        new_observed_samples=0,new_data_fits=0,precision_SE_absolute=5e-4,
        all_finite_curves_fixed_candidates_not_continuous_globopt=True))
    print(json.dumps({'frozen':True,'gaussians':STREAMS*BANKS*KMAX*DMAX}))


def score_block(z):
    size=len(z)
    raw=np.empty((size,3,21,4));marg=np.empty_like(raw)
    z1,z2=z[:,:,0],z[:,:,1]
    first={s:-np.logaddexp(0.,-s*z1) for s in [-1,1]}
    for ni,n in enumerate(NS):
        k=n+2
        noise=-np.logaddexp(0.,-z[:,:k,2:n+2]).sum(axis=2)
        for ri,rho in enumerate(RHO):
            x2=rho*z1[:,:k]+np.sqrt(1-rho*rho)*z2[:,:k]
            second={s:-np.logaddexp(0.,-s*x2) for s in [-1,1]}
            for si,(s1,s2) in enumerate(SIGNS):
                active=first[int(s1)][:,:k]+second[int(s2)]
                raw[:,ni,ri,si]=np.log(k)-logsumexp(noise+active,axis=1)
                marg[:,ni,ri,si]=np.log(k)-logsumexp(active,axis=1)
    # Common rho=0 comparison reduces floating-point cancellation later.
    raw-=raw[:,:,10:11,:].copy()
    marg-=marg[:,:,10:11,:].copy()
    return raw,marg


def run():
    frozen=json.loads((OUT/'FREEZE.json').read_text('utf-8'))
    assert sha(__file__)==frozen['source_sha256']
    assert not (OUT/'RESULTS.json').exists(), 'results already exist'
    start=time.perf_counter();records=[]
    for stream in range(STREAMS):
        dest=OUT/f'stream_{stream:02}.npz'
        assert not dest.exists(),dest
        rng=np.random.Generator(np.random.PCG64(SEED_BASE+stream))
        raw=np.empty((BANKS,3,21,4));marg=np.empty_like(raw)
        prior_hash=hashlib.sha256()
        first_hash=None
        for offset in range(0,BANKS,BLOCK):
            z=rng.standard_normal((BLOCK,KMAX,DMAX))
            prior_hash.update(z.tobytes())
            if offset==0:first_hash=hashlib.sha256(z.tobytes()).hexdigest()
            r,m=score_block(z)
            raw[offset:offset+BLOCK]=r;marg[offset:offset+BLOCK]=m
        assert np.isfinite(raw).all() and np.isfinite(marg).all()
        np.savez_compressed(dest,raw=raw,marg=marg,rho=RHO,ns=NS,signs=SIGNS)
        records.append(dict(stream=stream,seed=SEED_BASE+stream,file=dest.relative_to(ROOT).as_posix(),
            file_sha256=sha(dest),full_prior_sha256=prior_hash.hexdigest(),first64_prior_sha256=first_hash,
            raw_mean=raw.mean(axis=0).tolist(),marg_mean=marg.mean(axis=0).tolist()))
        save(OUT/'PROGRESS.json',dict(completed_streams=len(records),elapsed_seconds=time.perf_counter()-start))
        print(json.dumps({'stream':stream,'elapsed_seconds':round(time.perf_counter()-start,2)}),flush=True)
    save(OUT/'RESULTS.json',dict(source_sha256=sha(__file__),freeze_sha256=sha(OUT/'FREEZE.json'),
        records=records,rho=RHO.tolist(),ns=NS.tolist(),signs=SIGNS.tolist(),
        elapsed_seconds=time.perf_counter()-start,science_ledger=frozen,
        claim='population fixed-candidate contrasts; no empirical-data training or continuous optimum certificate'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','run']);a=p.parse_args()
    freeze() if a.mode=='freeze' else run()
