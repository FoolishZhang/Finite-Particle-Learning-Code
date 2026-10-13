"""Unmodified SCBM concept-loss API with explicit, model-compatible fixtures.

The primary path really invokes the frozen SCBLoss.compute_concept_loss.
The separately named stable_logits64 path is a numerical control, not an
official float64 API call (the official method forcibly casts labels to float32).
No sampler, encoder, task head, precision penalty, or neural training is claimed.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(1)

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/scbm_invariance/KERNEL"
SOURCE=Path(__file__).resolve()
COMMIT="58c3e602cb11fb4ea9106226e0f9e58f50d5d10a"
SNAPSHOT=ROOT/"reproducibility_snapshots/cbm_public_sources/SCBM"/COMMIT
OFFICIAL_LOSSES=SNAPSHOT/"models/losses.py"
OFFICIAL_MODELS=SNAPSHOT/"models/models.py"
FIXTURE=dict(mu=[-.6,-.2,.2,.6],C0=[[1.,.25,-.15,.10],[.25,1.,.10,-.15],
    [-.15,.10,1.,.25],[.10,-.15,.25,1.]],candidate_t=[-.75,0.,.75],a_nuisance=1.)
PAIRS=[(i,j) for i in range(4) for j in range(i+1,4)]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def ah(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def save_new(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x",encoding="utf-8",newline="\n") as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")

@lru_cache(maxsize=1)
def official_module():
    name="scbminv_official_losses_"+COMMIT
    spec=importlib.util.spec_from_file_location(name,OFFICIAL_LOSSES)
    assert spec is not None and spec.loader is not None
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    origin=Path(inspect.getsourcefile(module.SCBLoss.compute_concept_loss)).resolve()
    assert origin==OFFICIAL_LOSSES.resolve()
    return module

@lru_cache(maxsize=None)
def official_loss(K):
    module=official_module()
    config=SimpleNamespace(reg_precision="l1",reg_weight=1.,num_monte_carlo=int(K),training_mode="joint")
    return module.SCBLoss(num_classes=2,alpha=1.,config=config)

def provenance():
    module=official_module()
    return dict(repository="https://github.com/mvandenhi/SCBM",commit=COMMIT,
        official_losses=rel(OFFICIAL_LOSSES),official_losses_sha256=sha(OFFICIAL_LOSSES),
        official_models=rel(OFFICIAL_MODELS),official_models_sha256=sha(OFFICIAL_MODELS),
        default_config=rel(SNAPSHOT/"configs/model/model_defaults.yaml"),
        default_config_sha256=sha(SNAPSHOT/"configs/model/model_defaults.yaml"),
        class_name=module.SCBLoss.__name__,method_name=module.SCBLoss.compute_concept_loss.__name__,
        method_origin=rel(Path(inspect.getsourcefile(module.SCBLoss.compute_concept_loss))),
        torch_version=torch.__version__,torch_file=str(Path(torch.__file__).resolve()),device="cpu",
        primary_dtype="float32",torch_threads=torch.get_num_threads(),
        scope="Direct concept-loss API invocation only, not a full SCBM.forward, optimizer, head, precision-regularization or original-environment reproduction")

def fixture_tensors(t,dtype,mu_delta=None,cov_delta=None):
    mu=torch.tensor(FIXTURE["mu"],dtype=dtype)
    C0=torch.tensor(FIXTURE["C0"],dtype=dtype)
    D=C0-torch.eye(4,dtype=dtype)
    if mu_delta is not None:mu=mu+mu_delta
    cov=C0+float(t)*D
    if cov_delta is not None:
        basis=torch.zeros((6,4,4),dtype=dtype)
        for k,(i,j) in enumerate(PAIRS):basis[k,i,j]=basis[k,j,i]=1.
        cov=cov+torch.einsum("p,pij->ij",cov_delta,basis)
    lower=torch.linalg.cholesky(cov)
    return mu,cov,lower,D

def evaluate(active_eps,noise_eps,active_labels,noise_labels,t,padding,
             path="official32",factorized=False,need_gradient=False,nuisance_scale=1.,
             perturb_mu=None,perturb_cov=None):
    """K,B,C fixture tensors; returns actual batch-mean loss and gradients.

    Covariance coordinates perturb both symmetric entries. Therefore D dot
    cov_offdiag needs one D_ij per i<j, without another factor of two.
    """
    assert path in ("official32","stable_logits64")
    dtype=torch.float32 if path=="official32" else torch.float64
    eps=np.asarray(active_eps);nu=np.asarray(noise_eps)
    y=np.asarray(active_labels);yn=np.asarray(noise_labels)
    assert eps.ndim==3 and eps.shape[2]==4 and nu.shape[:2]==eps.shape[:2]
    K,B,_=eps.shape
    assert 0<=padding<=nu.shape[2] and y.shape==(B,4) and yn.shape==(B,nu.shape[2])
    assert np.isin(y,[0,1]).all() and np.isin(yn,[0,1]).all()
    mu_delta=torch.tensor(np.zeros(4) if perturb_mu is None else perturb_mu,dtype=dtype,requires_grad=need_gradient)
    cov_delta=torch.tensor(np.zeros(6) if perturb_cov is None else perturb_cov,dtype=dtype,requires_grad=need_gradient)
    scale=torch.tensor(float(nuisance_scale),dtype=dtype,requires_grad=need_gradient)
    mu,cov,lower,D=fixture_tensors(t,dtype,mu_delta,cov_delta)
    base=torch.as_tensor(np.ascontiguousarray(eps),dtype=dtype)
    # Same K,B,C layout as MultivariateNormal.rsample([K]); sampling itself is
    # supplied explicitly for valid prefix coupling, never claimed byte-equal.
    active_logits=torch.matmul(lower,base.unsqueeze(-1)).squeeze(-1)+mu
    if factorized or padding==0:
        logits=active_logits
        labels=torch.as_tensor(np.ascontiguousarray(y),dtype=torch.int64)
    else:
        nuisance_logits=scale*torch.as_tensor(np.ascontiguousarray(nu[:,:,:padding]),dtype=dtype)
        logits=torch.cat((active_logits,nuisance_logits),dim=-1)
        labels=torch.as_tensor(np.concatenate((y,yn[:,:padding]),axis=1),dtype=torch.int64)
    by_concept=logits.movedim(0,-1) # B,C,K as in the official forward output.
    diagnostics=dict(path=path,padding=int(padding),factorized=bool(factorized),K=int(K),rows=int(B),
        loss_input_shape=list(by_concept.shape),sampling_layout="explicit eps[K,B,C], then movedim(0,-1)",
        covariance_min_eigenvalue=float(torch.linalg.eigvalsh(cov.detach()).min()),
        maximum_absolute_logit=float(by_concept.detach().abs().max()),official_API_calls=0)
    if path=="official32":
        probabilities=torch.sigmoid(by_concept)
        obj=official_loss(K)
        # This exact official bound method is the production scoring call.
        loss=obj.compute_concept_loss(probabilities,labels)
        diagnostics.update(official_API_calls=1,probability_evaluations=probabilities.numel(),
            sigmoid_zero_count=int((probabilities.detach()==0).sum()),
            sigmoid_one_count=int((probabilities.detach()==1).sum()),
            probability_min=float(probabilities.detach().min()),probability_max=float(probabilities.detach().max()))
    else:
        targets=labels.to(dtype).unsqueeze(-1).expand_as(by_concept)
        bce=F.binary_cross_entropy_with_logits(by_concept,targets,reduction="none")
        loss=-torch.logsumexp(-bce.sum(dim=1),dim=1).mean()+math.log(K)
        diagnostics.update(control_scope="Independent stable-logit float64 formula, including higher-precision input/parameter transforms; not official API and not a BCE-only error decomposition")
    if factorized:loss=loss+padding*math.log(2.)
    answer=dict(value=float(loss.detach()),diagnostics=diagnostics)
    if need_gradient:
        grads=torch.autograd.grad(loss,(mu_delta,cov_delta,scale),allow_unused=True)
        gmu=grads[0];gcov=grads[1]
        ga=0. if grads[2] is None else float(grads[2].detach())
        coefficients=torch.stack([D[i,j] for i,j in PAIRS])
        directional=float(torch.dot(gcov,coefficients).detach())
        answer["gradient"]=dict(mu=gmu.detach().cpu().numpy().tolist(),cov_offdiag=gcov.detach().cpu().numpy().tolist(),
            nuisance_scale=ga,correlation_direction=directional,
            nuisance_scale_per_coordinate=ga/padding if padding else 0.,covariance_pairs=[list(p) for p in PAIRS])
    return answer

def deterministic_inputs(K=100,B=7,n=256):
    active=1.1*np.sin(np.arange(K*B*4,dtype=float)*.113).reshape(K,B,4)
    nuisance=.9*np.cos(np.arange(K*B*n,dtype=float)*.017).reshape(K,B,n)
    yy=np.arange(B*4).reshape(B,4)
    labels=((3*yy+yy//4)%5<2).astype(np.int8)
    nn=np.arange(B*n).reshape(B,n)
    noise_labels=((7*nn+nn//max(1,n))%11<5).astype(np.int8)
    return active,nuisance,labels,noise_labels

def flatten_gradient(result):
    g=result["gradient"]
    return np.asarray(g["mu"]+g["cov_offdiag"]+[g["nuisance_scale"]],float)

def official_dtype_probe():
    labels=torch.tensor([[0,1,0,1]],dtype=torch.int64)
    source=official_loss(100)
    values={}
    for dtype in (torch.float32,torch.float64):
        p=torch.full((1,4,100),.4,dtype=dtype,requires_grad=True)
        try:
            loss=source.compute_concept_loss(p,labels)
            loss.backward()
            values[str(dtype)]=dict(supported=True,value=float(loss.detach()),maximum_probability_gradient=float(p.grad.abs().max()))
        except RuntimeError as error:
            values[str(dtype)]=dict(supported=False,error=str(error))
    assert values["torch.float32"]["supported"] and not values["torch.float64"]["supported"]
    return values

def check():
    data=deterministic_inputs();active,noise,y,yn=data
    pd=[]
    for t in FIXTURE["candidate_t"]:
        _,C,_,_=fixture_tensors(t,torch.float64)
        pd.append(dict(t=t,eigenvalues=torch.linalg.eigvalsh(C).tolist()))
    assert min(min(v["eigenvalues"]) for v in pd)>0
    api=evaluate(*data,t=0.,padding=3,need_gradient=True)
    control=evaluate(*data,t=0.,padding=3,path="stable_logits64",need_gradient=True)
    # Official float32 finite differences need a scale appropriate to float32
    # loss evaluation; the separate stable64 check uses a finer step.
    differences={}
    for path,step,tolerance in (("official32",.01,3e-4),("stable_logits64",2e-4,3e-8)):
        truth=evaluate(*data,t=0.,padding=3,path=path,need_gradient=True)
        expected=flatten_gradient(truth);fd=[]
        for component in range(11):
            values=[]
            for multiple in (-2.,-1.,1.,2.):
                mu=np.zeros(4);cov=np.zeros(6);a=1.
                delta=multiple*step
                if component<4:mu[component]=delta
                elif component<10:cov[component-4]=delta
                else:a+=delta
                values.append(evaluate(*data,t=0.,padding=3,path=path,perturb_mu=mu,perturb_cov=cov,nuisance_scale=a)["value"])
            fd.append((values[0]-8*values[1]+8*values[2]-values[3])/(12*step))
        error=float(np.max(np.abs(np.asarray(fd)-expected)));assert error<tolerance,(path,error)
        differences[path]=dict(step=step,maximum_absolute_gradient_difference=error,tolerance=tolerance)
    identities=[]
    base=evaluate(*data,t=0.,padding=0,need_gradient=True)
    for n in (0,64,256):
        fact=evaluate(*data,t=0.,padding=n,factorized=True,need_gradient=True)
        const=evaluate(active,np.zeros_like(noise),y,yn,t=0.,padding=n,need_gradient=True,nuisance_scale=0.)
        value_error=abs(fact["value"]-base["value"]-n*math.log(2))
        fixed_value=abs(const["value"]-fact["value"])
        gradient_error=float(np.max(abs(flatten_gradient(fact)[:10]-flatten_gradient(base)[:10])))
        const_grad_error=float(np.max(abs(flatten_gradient(const)[:10]-flatten_gradient(base)[:10])))
        assert value_error<3e-5 and fixed_value<4e-5 and gradient_error==0 and const_grad_error<2e-6
        identities.append(dict(n=n,factorized_centered_value_error=value_error,constant_half_API_value_error=fixed_value,
            factorized_active_gradient_error=gradient_error,constant_half_active_gradient_error=const_grad_error,
            note="a=0 is only a constant-probability algebra control, not an attained original softplus-floor model parameter"))
    floor=evaluate(*data,t=0.,padding=256,need_gradient=True,nuisance_scale=1e-6)
    floor_fact=evaluate(*data,t=0.,padding=256,factorized=True,need_gradient=True)
    results=dict(passed=True,source_sha256=sha(SOURCE),official=provenance(),fixture=FIXTURE,SPD=pd,
        dtype_probe=official_dtype_probe(),gradient_checks=differences,factorization_checks=identities,
        standard_float32_vs_stable64=dict(value_difference=api["value"]-control["value"],
            maximum_gradient_difference=float(np.max(abs(flatten_gradient(api)-flatten_gradient(control))))),
        small_scale_control=dict(scale=1e-6,value_difference_from_factorized=floor["value"]-floor_fact["value"],
            active_gradient_difference=float(np.max(abs(flatten_gradient(floor)[:10]-flatten_gradient(floor_fact)[:10])))),
        new_scientific_RNG_draws=0,new_model_integrals=0,new_data_fits=0,
        scope="Deterministic model-compatible tensors and genuine official API calls; not empirical scientific outcomes or a full model-forward reproduction")
    save_new(OUT/"SELFTEST.json",results);print(json.dumps(results),flush=True)

def bench():
    cases=[]
    for K in (100,400):
        data=deterministic_inputs(K=K,B=64,n=256)
        start=time.perf_counter();values=[]
        for padding in (0,64,256):
            for t in FIXTURE["candidate_t"]:
                values.append(evaluate(*data,t=t,padding=padding,need_gradient=t==0.))
        elapsed=time.perf_counter()-start
        start=time.perf_counter();factor=[]
        for t in FIXTURE["candidate_t"]:
            factor.append(evaluate(*data,t=t,padding=0,factorized=True,need_gradient=t==0.))
        factor_seconds=time.perf_counter()-start
        start=time.perf_counter()
        for padding in (0,64,256):
            for t in FIXTURE["candidate_t"]:
                evaluate(*data,t=t,padding=padding,path="stable_logits64",need_gradient=t==0.)
        reference_seconds=time.perf_counter()-start
        cases.append(dict(K=K,rows=64,official_raw_seconds=elapsed,factorized_active_seconds=factor_seconds,
            numerical64_first_batch_seconds=reference_seconds,
            official_probability_saturation=sum(v["diagnostics"]["sigmoid_zero_count"]+v["diagnostics"]["sigmoid_one_count"] for v in values),
            checksum=sum(v["value"] for v in values)+sum(v["value"] for v in factor)))
    main=next(c for c in cases if c["K"]==100);aux=next(c for c in cases if c["K"]==400)
    projected=16*(64*(main["official_raw_seconds"]+main["factorized_active_seconds"])+
        16*(aux["official_raw_seconds"]+aux["factorized_active_seconds"])+
        main["numerical64_first_batch_seconds"]+aux["numerical64_first_batch_seconds"])
    out=dict(source_sha256=sha(SOURCE),official=provenance(),cases=cases,
        projected_serial_seconds_excluding_RNG_and_storage=projected,
        new_scientific_RNG_draws=0,new_model_integrals=0,new_data_fits=0,
        scope="Bounded deterministic shapes only; official calls are genuine, timings are estimates not scientific outcomes")
    save_new(OUT/"BENCHMARK.json",out);print(json.dumps(out),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=["check","bench","provenance"])
    phase=parser.parse_args().phase
    if phase=="provenance":print(json.dumps(provenance()),flush=True)
    else:globals()[phase]()
