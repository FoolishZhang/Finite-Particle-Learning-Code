"""Frozen-output evaluation kernel: deterministic development only in this file.

Canonical nuisance labels are valid for a fresh-prior expectation, including the
expectation conditional on each old observation. This coupling does not describe
the old training realization. Price and path gradients agree in expectation,
not bank by bank. No quadrature, scientific RNG or fitting is executed here.
"""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name]='1'
from pathlib import Path
import argparse
import hashlib
import json
import math
import time
import numpy as np
from scipy.special import expit, logsumexp
import torch
import scbminv_kernel as official_adapter

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/scbm_frozen_evaluation/KERNEL'
SOURCE=Path(__file__).resolve()
ADAPTER_SHA='c6712abbb4e05ef0c25c688a05dad106034c41888040592424f50362f04129ad'
MU=np.array([-.6,-.2,.2,.6],dtype=np.float64)
C0=np.array([[1.,.25,-.15,.10],[.25,1.,.10,-.15],[-.15,.10,1.,.25],[.10,-.15,.25,1.]],dtype=np.float64)
D=C0-np.eye(4)
PATTERNS=((np.arange(16)[:,None]>>np.array([3,2,1,0]))&1).astype(np.int8)
PAIRS=[(i,j) for i in range(4) for j in range(i+1,4)]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save_new(name,value):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/name).open('x',encoding='utf8',newline='\n') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write('\n')


def provenance():
    assert sha(official_adapter.__file__)==ADAPTER_SHA
    return dict(source_sha256=sha(SOURCE),official_adapter_sha256=ADAPTER_SHA,
        official=official_adapter.provenance(),dtype='float64',active_mean=MU.tolist(),
        active_covariance=C0.tolist(),direction=D.tolist(),active_patterns=PATTERNS.tolist(),
        scope='Fresh-prior expectation evaluator at frozen scales, with exact active-pattern enumeration. Canonical nuisance labels apply only in expectation; no old per-row training coupling is claimed.',
        gradient_scope='Price is a Gaussian covariance derivative estimator; path is the fixed-bank Cholesky derivative. They are not identical bankwise.')


def cholesky_and_derivative(t):
    assert math.isfinite(t) and -.75<=t<=.75
    L=np.linalg.cholesky(np.eye(4)+(1.+t)*D)
    inverse=np.linalg.solve(L,np.eye(4))
    transformed=inverse@D@inverse.T
    lower=np.tril(transformed)
    lower[np.diag_indices(4)]*=.5
    return L,L@lower


def prepare_noise(nuisance_eps,scales,alphas=(1.,.5,0.)):
    """Compute nuisance logweights once, centered by their n*log(2) constant."""
    nu=np.asarray(nuisance_eps,dtype=np.float64)
    tau=np.asarray(scales,dtype=np.float64)
    assert nu.ndim==3 and tau.shape==(nu.shape[2],)
    assert np.isfinite(nu).all() and np.isfinite(tau).all() and (tau>=0).all()
    assert len(set(alphas))==len(alphas) and all(math.isfinite(a) and a>=0 for a in alphas)
    cache={}
    for alpha in alphas:
        if alpha==0. or nu.shape[2]==0:
            cache[float(alpha)]=np.zeros(nu.shape[:2],dtype=np.float64)
        else:
            logits=nu*(float(alpha)*tau)
            cache[float(alpha)]=np.sum(math.log(2.)-np.logaddexp(0.,logits),axis=2)
    return dict(n=nu.shape[2],centered_logweights=cache)


def active_terms(active_eps,t,need_path=True):
    eps=np.asarray(active_eps,dtype=np.float64)
    assert eps.ndim==3 and eps.shape[2]==4 and np.isfinite(eps).all()
    L,Lprime=cholesky_and_derivative(t)
    x=eps@L.T+MU
    # log p(y|x)=y.x-sum softplus(x); no product of probabilities is formed.
    logactive=x@PATTERNS.T-np.logaddexp(0.,x).sum(axis=2)[:,:,None]
    u=PATTERNS[None,None,:,:]-expit(x)[:,:,None,:]
    price_cross=np.zeros(logactive.shape,dtype=np.float64)
    for i,j in PAIRS:price_cross+=D[i,j]*u[:,:,:,i]*u[:,:,:,j]
    path_mark=None
    if need_path:
        xprime=eps@Lprime.T
        path_mark=np.einsum('bkpi,bki->bkp',u,xprime,optimize=True)
    return dict(logits=x,logactive=logactive,price_cross=price_cross,path_mark=path_mark)


def from_terms(terms,centered_noise,n):
    """Return arrays [bank, pattern]; gradients have no n*log2 constant."""
    noise=np.asarray(centered_noise,dtype=np.float64)
    la=terms['logactive']
    assert noise.shape==la.shape[:2] and np.isfinite(noise).all()
    logweights=la+noise[:,:,None]
    normalizer=logsumexp(logweights,axis=1)
    pi=np.exp(logweights-normalizer[:,None,:])
    centered=math.log(la.shape[1])-normalizer
    price=-np.sum(pi*(1.-pi)*terms['price_cross'],axis=1)
    result=dict(centered_score=centered,score=centered+n*math.log(2.),price_t=price)
    if terms['path_mark'] is not None:result['path_t']=-np.sum(pi*terms['path_mark'],axis=1)
    return result


def evaluate(active_eps,noise_cache,t,need_path=True):
    terms=active_terms(active_eps,t,need_path)
    return {alpha:from_terms(terms,weight,noise_cache['n'])
            for alpha,weight in noise_cache['centered_logweights'].items()}


def weight_patterns(pattern_arrays,weights):
    """Same bank-level arrays, separately weighted by empirical or true PMFs.

    `weights` may contain {'conditional': counts/N, 'population': frozen PMF}.
    No renormalization of supplied probabilities is silently performed.
    """
    result={}
    for name,values in weights.items():
        p=np.asarray(values,dtype=np.float64)
        assert p.shape==(16,) and np.isfinite(p).all() and (p>=0).all()
        assert abs(float(p.sum())-1.)<2e-12
        result[name]={key:array@p for key,array in pattern_arrays.items()}
    return result


def official32_pattern_means(active_eps,nuisance_eps,scales,t,alpha):
    """16 genuine API calls, one pattern per batch of independent banks.

    The API returns only a batch mean. We do not pretend it yields per-row or
    per-pattern values after a single flattened B*16 call.
    """
    assert sha(official_adapter.__file__)==ADAPTER_SHA
    eps=torch.as_tensor(np.asarray(active_eps),dtype=torch.float32)
    nu=torch.as_tensor(np.asarray(nuisance_eps),dtype=torch.float32)
    tau=torch.as_tensor(np.asarray(scales),dtype=torch.float32)
    B,K,_=eps.shape;n=nu.shape[2]
    assert nu.shape[:2]==(B,K) and tau.shape==(n,)
    eye=torch.eye(4,dtype=torch.float32)
    cov=eye+(1.+torch.tensor(t,dtype=torch.float32))*(torch.tensor(C0,dtype=torch.float32)-eye)
    active=eps@torch.linalg.cholesky(cov).T+torch.tensor(MU,dtype=torch.float32)
    if alpha==0.:
        logits=active
    else:
        logits=torch.cat((active,nu*(torch.tensor(alpha,dtype=torch.float32)*tau)),dim=2)
    probabilities=torch.sigmoid(logits.permute(0,2,1))
    answers=[]
    with torch.no_grad():
        for pattern in PATTERNS:
            y=np.broadcast_to(pattern,(B,4)).copy()
            labels=torch.as_tensor(y,dtype=torch.int64)
            if alpha!=0.:labels=torch.cat((labels,torch.zeros((B,n),dtype=torch.int64)),dim=1)
            value=official_adapter.official_loss(K).compute_concept_loss(probabilities,labels)
            if alpha==0.:value=value+n*math.log(2.)
            answers.append(float(value))
    return dict(pattern_means=np.asarray(answers),official_API_calls=16,
        rows_per_call=B,K=K,alpha=float(alpha),
        probability_evaluations=16*probabilities.numel(),
        sigmoid_zero_evaluations=16*int((probabilities==0).sum()),
        sigmoid_one_evaluations=16*int((probabilities==1).sum()),
        maximum_absolute_logit=float(logits.abs().max()),
        scope='Each mean is from one actual official float32 API call over B banks; alpha0 uses active API plus analytical nlog2. Control differences include input/parameter rounding and expression changes.')


def deterministic_inputs(B=3,K=7,n=5):
    active=1.1*np.sin(np.arange(B*K*4,dtype=np.float64)*.113).reshape(B,K,4)
    nuisance=.9*np.cos(np.arange(B*K*n,dtype=np.float64)*.017).reshape(B,K,n)
    scales=.20+.01*np.sin(np.arange(n,dtype=np.float64)*.19)
    return active,nuisance,scales


def check():
    prov=provenance();active,noise,tau=deterministic_inputs();cache=prepare_noise(noise,tau)
    t=.13;answer=evaluate(active,cache,t)
    empty=prepare_noise(noise[:,:,:0],tau[:0],alphas=(0.,))
    fact=evaluate(active,empty,t)[0.]
    identity=[]
    for key in ('centered_score','price_t','path_t'):
        error=float(np.max(abs(answer[0.][key]-fact[key])))
        assert error==0.
        identity.append(dict(field=key,maximum_error=error))
    constant_error=float(np.max(abs(answer[0.]['score']-fact['score']-5*math.log(2.))))
    assert constant_error<2e-15
    fd_checks=[];h=2e-4
    for alpha in (1.,.5,0.):
        values=[evaluate(active,cache,t+m*h,need_path=False)[alpha]['centered_score'] for m in (-2,-1,1,2)]
        fd=(values[0]-8*values[1]+8*values[2]-values[3])/(12*h)
        error=float(np.max(abs(fd-answer[alpha]['path_t'])))
        assert error<2e-9,(alpha,error)
        fd_checks.append(dict(alpha=alpha,step=h,maximum_path_FD_error=error,tolerance=2e-9))
    # Algebraic Price check: finite-difference mixed Hessians of the fixed-bank
    # score with respect to a particle's active logits. This is not a Gaussian
    # expectation calculation or a bankwise comparison against the path score.
    x=active_terms(active,t)['logits'];w=cache['centered_logweights'][1.]
    def score_at_logits(z):
        loga=z@PATTERNS.T-np.logaddexp(0.,z).sum(axis=2)[:,:,None]
        return math.log(z.shape[1])-logsumexp(loga+w[:,:,None],axis=1)
    hessian=np.zeros((x.shape[0],16));hh=.002
    for k in range(x.shape[1]):
        for i,j in PAIRS:
            values=[]
            for si,sj in ((1,1),(1,-1),(-1,1),(-1,-1)):
                z=x.copy();z[:,k,i]+=si*hh;z[:,k,j]+=sj*hh
                values.append(score_at_logits(z))
            hessian+=D[i,j]*(values[0]-values[1]-values[2]+values[3])/(4*hh*hh)
    price_error=float(np.max(abs(hessian-answer[1.]['price_t'])))
    assert price_error<2e-7,price_error
    difference=float(np.max(abs(answer[1.]['price_t']-answer[1.]['path_t'])))
    assert difference>1e-6,'Do not accidentally identify Price and path bankwise'
    empirical=np.arange(1,17,dtype=np.float64);empirical/=empirical.sum()
    weighted=weight_patterns(answer[1.],dict(conditional=empirical,population=np.full(16,1/16)))
    assert np.allclose(weighted['conditional']['score'],np.sum(answer[1.]['score']*empirical,axis=1),rtol=0,atol=2e-15)
    # Actual production shape, still deterministic bounded tensors; no MC draw.
    big_active,big_noise,big_tau=deterministic_inputs(B=4,K=100,n=256)
    big=evaluate(big_active,prepare_noise(big_noise,big_tau),t)
    api_checks=[]
    for alpha in (1.,.5,0.):
        api=official32_pattern_means(big_active,big_noise,big_tau,t,alpha)
        expected=big[alpha]['score'].mean(axis=0)
        delta=api['pattern_means']-expected
        maximum=float(np.max(abs(delta)))
        assert maximum<1e-4,(alpha,maximum)
        api_checks.append(dict(alpha=alpha,maximum_pattern_score_error=maximum,
            per_pattern_difference=delta.tolist(),tolerance=1e-4,
            conditional_weighted_error=float(delta@empirical),uniform_weighted_error=float(delta.mean()),
            official_API_calls=api['official_API_calls'],rows_per_call=api['rows_per_call'],
            sigmoid_zero_evaluations=api['sigmoid_zero_evaluations'],sigmoid_one_evaluations=api['sigmoid_one_evaluations']))
    result=dict(passed=True,**prov,factorized_identity=identity,factorized_constant_error=constant_error,
        path_FD_checks=fd_checks,Price_Hessian_check=dict(step=hh,maximum_error=price_error,tolerance=2e-7,
            evaluated_particle_pair_hessians=x.shape[1]*len(PAIRS),new_Gaussian_integrals=0),
        Price_path_bankwise_difference=difference,weighted_pattern_check=True,official32_checks=api_checks,
        new_scientific_RNG_draws=0,new_model_integrals=0,new_fits=0,
        test_scope='Only deterministic sine/cosine tensors, algebraic Hessian identity and finite differences; no Price expectation certificate or sampled precision assessment.')
    save_new('SELFTEST.json',result);print(json.dumps(result,indent=2))


def bench():
    prov=provenance();B,K,n=64,100,256
    active,noise,tau=deterministic_inputs(B,K,n)
    grid=np.linspace(-.75,.75,21)
    # Fixed artificial weights exercise both F_D and F_P aggregation without
    # selecting a scientific dataset or measuring any experimental outcome.
    p=np.arange(1,17,dtype=np.float64);p/=p.sum()
    weights=dict(conditional=p,population=np.full(16,1/16))
    trials=[]
    for repeat in range(4):
        start=time.perf_counter();cache=prepare_noise(noise,tau)
        preparation=time.perf_counter()-start;tick=time.perf_counter();checksum=0.
        for t in grid:
            for arrays in evaluate(active,cache,float(t),need_path=True).values():
                weighted=weight_patterns(arrays,weights)
                checksum+=sum(float(r['centered_score'].sum()) for r in weighted.values())
        evaluation=time.perf_counter()-tick
        if repeat:trials.append(dict(noise_preparation_seconds=preparation,
            grid_score_Price_path_weighting_seconds=evaluation,total_seconds=preparation+evaluation,checksum=checksum))
    assert len({r['checksum'] for r in trials})==1
    median=float(np.median([r['total_seconds'] for r in trials]))
    result=dict(**prov,banks_per_block=B,K=K,n=n,t_grid=grid.tolist(),alphas=[1.,.5,0.],
        trial_seconds=trials,median_seconds_per_scale_model=median,
        projected_minutes_16_streams_one_frozen_scale_model={str(banks):median*(banks/B)*16/60 for banks in (4096,8192)},
        projected_minutes_16_streams_TAIL_and_LAST={str(banks):median*(banks/B)*16*2/60 for banks in (4096,8192)},
        accounting_scope='Noise logweights cached once per scale model and input block. Active terms shared across alpha1/.5/0. Projection conservatively runs separate TAIL/LAST grids; further legal cache reuse is not assumed.',
        timing_scope='Float64 21-candidate score, Price, path and two weighting vectors; excludes scientific RNG, IO and independent official32 controls.',
        new_scientific_RNG_draws=0,new_model_integrals=0,new_fits=0)
    save_new('BENCHMARK.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['check','bench','provenance'])
    phase=parser.parse_args().phase
    if phase=='provenance':print(json.dumps(provenance(),indent=2))
    else:globals()[phase]()
