"""Six freely learned active correlations through the untouched SCBM API.

The parameter constructor and update map know no true covariance or radial
direction. Scientific sampling belongs to a separately frozen runner.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_key]='1'
from pathlib import Path
from dataclasses import dataclass
import argparse,hashlib,itertools,json,math,time
import numpy as np
import torch
import torch.nn.functional as F
import scbminv_kernel as official_adapter

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/scbm_multicorrelation/KERNEL'
SOURCE=Path(__file__).resolve()
ADAPTER_SHA='c6712abbb4e05ef0c25c688a05dad106034c41888040592424f50362f04129ad'
PAIRS=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
MU=[-.6,-.2,.2,.6]
FLOOR=1e-6;EIGEN_FLOOR=.1;REPAIR_TARGET=.1001
METHODS=['FIXED','FREE','FACT'];DTYPE=torch.float32
torch.set_num_threads(1)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save_new(name,value):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/name).open('x',encoding='utf8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def basis(dtype=DTYPE):
    value=torch.zeros((6,4,4),dtype=dtype)
    for p,(i,j) in enumerate(PAIRS):value[p,i,j]=value[p,j,i]=1.
    return value


def covariance_numpy(rho):
    rho=np.asarray(rho,dtype=np.float64);assert rho.shape==(6,) and np.isfinite(rho).all()
    value=np.eye(4)
    for r,(i,j) in zip(rho,PAIRS):value[i,j]=value[j,i]=r
    return value


def provenance():
    assert sha(official_adapter.__file__)==ADAPTER_SHA
    return dict(source_sha256=sha(SOURCE),official_adapter_sha256=ADAPTER_SHA,
        official=official_adapter.provenance(),pair_order=[list(p) for p in PAIRS],fixed_mu=MU,
        initial_rho=[0.]*6,fixed_active_diagonal=[1.]*4,eigen_floor=EIGEN_FLOOR,
        repair_target=REPAIR_TARGET,scale_floor=FLOOR,
        scope='Six physical symmetric off-diagonal correlations from identity; fixed means/diagonal and zero active-nuisance cross covariance. Actual concept-loss API only, not full SCBM network training.',
        feasibility_scope='After Adam, infeasible proposals shrink toward identity. This is not the nearest-correlation projection or a convergence guarantee; the safety margin need not define a continuous mathematical retraction.')


@dataclass(frozen=True)
class Batch:
    active_eps:torch.Tensor
    nuisance_eps:torch.Tensor
    active_labels:torch.Tensor
    nuisance_labels:torch.Tensor


def prepare_batch(active_eps,nuisance_eps,active_labels,nuisance_labels):
    e=torch.as_tensor(active_eps,dtype=DTYPE).contiguous()
    nu=torch.as_tensor(nuisance_eps,dtype=DTYPE).contiguous()
    y=torch.as_tensor(active_labels);yn=torch.as_tensor(nuisance_labels)
    assert e.ndim==nu.ndim==3 and e.shape[2]==4 and e.shape[:2]==nu.shape[:2]
    assert y.shape==(e.shape[1],4) and yn.shape==(e.shape[1],nu.shape[2])
    assert bool(torch.isfinite(e).all()) and bool(torch.isfinite(nu).all())
    assert bool(((y==0)|(y==1)).all()) and bool(((yn==0)|(yn==1)).all())
    return Batch(e,nu,y.to(torch.int64).contiguous(),yn.to(torch.int64).contiguous())


class ScoreProgram(torch.nn.Module):
    def __init__(self,method,K=100,n=256,scale0=1.):
        super().__init__();assert method in METHODS and K>=1 and n>=1 and scale0>FLOOR
        assert sha(official_adapter.__file__)==ADAPTER_SHA
        self.method,self.K,self.n=method,K,n
        self.rho=torch.nn.Parameter(torch.zeros(6,dtype=DTYPE))
        self.register_buffer('mu',torch.tensor(MU,dtype=DTYPE))
        self.register_buffer('identity',torch.eye(4,dtype=DTYPE))
        self.register_buffer('symmetric_basis',basis())
        self.register_buffer('fixed_scales',torch.full((n,),scale0,dtype=DTYPE))
        if method=='FREE':
            x=scale0-FLOOR;raw=x+math.log(-math.expm1(-x))
            self.raw_scale=torch.nn.Parameter(torch.full((n,),raw,dtype=DTYPE))
        else:self.register_parameter('raw_scale',None)

    def covariance(self):return self.identity+torch.einsum('p,pij->ij',self.rho,self.symmetric_basis)
    def scales(self):return F.softplus(self.raw_scale)+FLOOR if self.raw_scale is not None else self.fixed_scales

    def forward(self,batch,diagnostics=None):
        assert batch.active_eps.shape[0]==self.K and batch.nuisance_eps.shape[2]==self.n
        L=torch.linalg.cholesky(self.covariance())
        active=torch.matmul(L,batch.active_eps.unsqueeze(-1)).squeeze(-1)+self.mu
        if self.method=='FACT':logits,labels=active,batch.active_labels
        else:
            logits=torch.cat((active,batch.nuisance_eps*self.scales()),dim=-1)
            labels=torch.cat((batch.active_labels,batch.nuisance_labels),dim=1)
        probabilities=torch.sigmoid(logits.movedim(0,-1))
        loss=official_adapter.official_loss(self.K).compute_concept_loss(probabilities,labels)
        if self.method=='FACT':loss=loss+self.n*math.log(2.)
        if diagnostics is not None:
            p=probabilities.detach()
            diagnostics.update(official_API_calls=1,probability_evaluations=p.numel(),
                sigmoid_zero_count=int((p==0).sum()),sigmoid_one_count=int((p==1).sum()),
                maximum_absolute_logit=float(logits.detach().abs().max()))
        return loss

    @torch.no_grad()
    def restore_feasibility(self):
        proposed=self.rho.detach().numpy().astype(np.float64)
        assert np.isfinite(proposed).all(),'Nonfinite proposal: do not retry input'
        before=float(np.linalg.eigvalsh(covariance_numpy(proposed)).min())
        alpha=1.
        if before<EIGEN_FLOOR:
            alpha=(1.-REPAIR_TARGET)/(1.-before)
            assert 0.<alpha<1.
            self.rho.copy_(torch.as_tensor((alpha*proposed).astype(np.float32)))
        after=float(np.linalg.eigvalsh(covariance_numpy(self.rho.detach().numpy())).min())
        assert after>=EIGEN_FLOOR,(before,alpha,after)
        _,info=torch.linalg.cholesky_ex(self.covariance())
        assert int(info)==0,'Post-repair native float32 Cholesky failed'
        return dict(rho_unprojected=proposed.tolist(),projection_alpha=alpha,
                    proposal_eigmin=before,eigmin=after,projected=alpha!=1.,cholesky_success=True)


def make_optimizer(program,lr_rho=.02,lr_scale=.02):
    groups=[dict(params=[program.rho],lr=lr_rho)]
    if program.raw_scale is not None:groups.append(dict(params=[program.raw_scale],lr=lr_scale))
    return torch.optim.Adam(groups,betas=(.9,.999),eps=1e-8,weight_decay=0.,amsgrad=False,foreach=False,fused=False)


def make_programs(K=100,n=256,scale0=1.,lr_rho=.02,lr_scale=.02):
    models={m:ScoreProgram(m,K,n,scale0) for m in METHODS}
    return models,{m:make_optimizer(p,lr_rho,lr_scale) for m,p in models.items()}


def gradients(program,batch):
    loss=program(batch);values=torch.autograd.grad(loss,tuple(program.parameters()))
    return loss,{name:g for (name,_),g in zip(program.named_parameters(),values)}


def optimizer_step(program,optimizer,batch,diagnostics=False):
    optimizer.zero_grad(set_to_none=True);detail={} if diagnostics else None
    before=program.rho.detach().numpy().copy();loss=program(batch,detail)
    assert bool(torch.isfinite(loss));loss.backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in program.parameters())
    gradient=program.rho.grad.detach().numpy().copy()
    optimizer.step();repair=program.restore_feasibility()
    scales=program.scales().detach()
    assert all(bool(torch.isfinite(p).all()) for p in program.parameters())
    assert bool(torch.isfinite(scales).all()) and bool((scales>=FLOOR).all())
    return dict(loss=float(loss.detach()),rho_before=before.tolist(),rho_gradient=gradient.tolist(),
        rho_after=program.rho.detach().numpy().tolist(),**repair,
        scale_min=float(scales.min()),scale_mean=float(scales.mean()),scale_max=float(scales.max()),diagnostics=detail)


def stable64(program,active,noise,y,yn):
    """Independent stable-logit expression, not an official double API call."""
    rho=program.rho.detach().to(torch.float64).requires_grad_()
    raw=program.raw_scale.detach().to(torch.float64).requires_grad_() if program.raw_scale is not None else None
    C=torch.eye(4,dtype=torch.float64)+torch.einsum('p,pij->ij',rho,basis(torch.float64))
    x=torch.matmul(torch.linalg.cholesky(C),torch.as_tensor(active,dtype=torch.float64).unsqueeze(-1)).squeeze(-1)+torch.tensor(MU,dtype=torch.float64)
    if program.method=='FACT':logits=x;labels=torch.as_tensor(y,dtype=torch.float64)
    else:
        scales=F.softplus(raw)+FLOOR if raw is not None else program.fixed_scales.detach().to(torch.float64)
        logits=torch.cat((x,torch.as_tensor(noise,dtype=torch.float64)*scales),dim=-1)
        labels=torch.as_tensor(np.concatenate((y,yn),axis=1),dtype=torch.float64)
    z=logits.movedim(0,-1);target=labels.unsqueeze(-1).expand_as(z)
    loss=-torch.logsumexp(-F.binary_cross_entropy_with_logits(z,target,reduction='none').sum(dim=1),dim=1).mean()+math.log(program.K)
    if program.method=='FACT':loss=loss+program.n*math.log(2.)
    g=torch.autograd.grad(loss,(rho,raw) if raw is not None else (rho,))
    return dict(loss=float(loss.detach()),rho_gradient=g[0].detach().numpy().tolist(),
                raw_gradient=None if raw is None else g[1].detach().numpy().tolist())


def deterministic(K=100,B=7,n=256):
    arrays=official_adapter.deterministic_inputs(K,B,n)
    return prepare_batch(*arrays),arrays


def set_test_rho(program,values):
    with torch.no_grad():program.rho.copy_(torch.tensor(values,dtype=DTYPE))


def check():
    prov=provenance();batch,arrays=deterministic();compat=[]
    # True covariance appears only in this explicit compatibility diagnostic,
    # never in the model constructor or scientific update map.
    diagnostic_truth=np.asarray(official_adapter.FIXTURE['C0'])
    truth_rho=np.array([diagnostic_truth[i,j] for i,j in PAIRS])
    old=official_adapter.evaluate(*arrays,t=0.,padding=256,need_gradient=True)
    for method in METHODS:
        program=ScoreProgram(method);assert torch.equal(program.rho.detach(),torch.zeros(6))
        set_test_rho(program,truth_rho);loss,g=gradients(program,batch)
        ref=official_adapter.evaluate(*arrays,t=0.,padding=256,factorized=True,need_gradient=True) if method=='FACT' else old
        value_error=abs(float(loss.detach())-ref['value'])
        error=float(np.max(abs(g['rho'].detach().numpy()-np.array(ref['gradient']['cov_offdiag']))))
        assert value_error<4e-5 and error<3e-6
        compat.append(dict(method=method,value_error=value_error,six_gradient_error=error))
    small,small_arrays=deterministic(n=5);fd=[]
    for method in METHODS:
        program=ScoreProgram(method,n=5);set_test_rho(program,[.11,-.07,.05,.09,-.06,.13])
        _,g=gradients(program,small);control=stable64(program,*small_arrays)
        native=[];finite=[];h=.02
        for name,count in [('rho',6)]+([('raw_scale',5)] if method=='FREE' else []):
            parameter=getattr(program,name);original=parameter.detach().clone()
            for i in range(count):
                values=[]
                for multiplier in (-2,-1,1,2):
                    with torch.no_grad():parameter.copy_(original);parameter[i]+=multiplier*h;values.append(float(program(small)))
                with torch.no_grad():parameter.copy_(original)
                native.append(float(g[name][i]));finite.append((values[0]-8*values[1]+8*values[2]-values[3])/(12*h))
        err=float(np.max(abs(np.asarray(native)-finite)));assert err<3e-4,(method,err)
        rhoerr=float(np.max(abs(g['rho'].detach().numpy()-control['rho_gradient'])));assert rhoerr<2e-5
        fd.append(dict(method=method,maximum_FD_error=err,FD_step=h,native_vs_stable64_six_gradient_error=rhoerr))
    geometry=[]
    for signs in itertools.product((-1.,1.),repeat=6):
        program=ScoreProgram('FIXED');set_test_rho(program,.95*np.asarray(signs))
        before=program.rho.detach().numpy().astype(float);result=program.restore_feasibility()
        after=program.rho.detach().numpy().astype(float)
        assert np.max(abs(after-before*result['projection_alpha']))<4e-8
        geometry.append(result['eigmin'])
    retained=[]
    for t in (-.75,0.,.75):
        p=ScoreProgram('FIXED');set_test_rho(p,(1+t)*truth_rho);before=p.rho.detach().clone()
        result=p.restore_feasibility();assert result['projection_alpha']==1. and torch.equal(before,p.rho.detach())
        retained.append(dict(old_t=t,eigmin=result['eigmin']))
    adam=[];models,optimizers=make_programs()
    for method,program in models.items():
        _,g=gradients(program,batch);expected={}
        for name,p in program.named_parameters():
            q=p.detach().numpy().astype(float);gg=g[name].detach().numpy().astype(float)
            expected[name]=q-.02*gg/(abs(gg)+1e-8)
        result=optimizer_step(program,optimizers[method],batch,True)
        assert result['projection_alpha']==1.
        errors={name:float(np.max(abs(p.detach().numpy()-expected[name]))) for name,p in program.named_parameters()}
        assert max(errors.values())<2e-7
        adam.append(dict(method=method,parameter_errors=errors,step=result))
    forced=ScoreProgram('FREE');set_test_rho(forced,np.full(6,.89998));opt=make_optimizer(forced)
    forced.rho.grad=-torch.ones(6);forced.raw_scale.grad=torch.ones(256);opt.step()
    moments={key:opt.state[forced.rho][key].clone() for key in ('exp_avg','exp_avg_sq')}
    raw_before=forced.raw_scale.detach().clone();repair=forced.restore_feasibility()
    assert repair['projected'] and torch.equal(raw_before,forced.raw_scale.detach())
    assert all(torch.equal(opt.state[forced.rho][key],v) for key,v in moments.items())
    fact=ScoreProgram('FACT')
    fact_loss,_=gradients(fact,batch)
    zeros=ScoreProgram('FIXED')
    with torch.no_grad():zeros.fixed_scales.zero_()
    zloss,zg=gradients(zeros,batch);_,fg=gradients(fact,batch)
    assert abs(float((zloss-fact_loss).detach()))<4e-5 and float((zg['rho']-fg['rho']).abs().max())<3e-6
    custom=ScoreProgram('FIXED',n=5,scale0=.2)
    custom_loss,custom_grad=gradients(custom,small);custom64=stable64(custom,*small_arrays)
    custom_value_error=abs(float(custom_loss.detach())-custom64['loss'])
    custom_gradient_error=float(np.max(abs(custom_grad['rho'].detach().numpy()-custom64['rho_gradient'])))
    assert custom_value_error<1e-4 and custom_gradient_error<2e-5
    output=dict(passed=True,**prov,old_kernel_compatibility=compat,finite_differences=fd,
        repaired_synthetic_sign_vertices=64,minimum_repaired_eigenvalue=min(geometry),
        old_radial_diagnostic_matrices_retained=retained,Adam_first_step_checks=adam,
        forced_feasibility_update=repair,Adam_moments_preserved=True,raw_scales_not_projected=True,
        factorized_constant_half_identity=dict(loss_error=abs(float((zloss-fact_loss).detach())),rho_gradient_error=float((zg['rho']-fg['rho']).abs().max())),
        fixed_scale_point2_control=dict(loss_error=custom_value_error,rho_gradient_error=custom_gradient_error),
        new_scientific_RNG_draws=0,new_model_integrals=0,new_scientific_fits=0,
        test_scope='Deterministic tensors, six-coordinate finite differences, algebraic feasibility checks and isolated Adam updates. No training pilot or scientific outcome evaluation.')
    save_new('SELFTEST.json',output);print(json.dumps(output,indent=2))


def bench():
    batch,_=deterministic(B=64);times={m:[] for m in METHODS};checksums={m:[] for m in METHODS}
    for repeat in range(9):
        models,optimizers=make_programs()
        for method in METHODS:
            start=time.perf_counter();result=optimizer_step(models[method],optimizers[method],batch,True)
            elapsed=time.perf_counter()-start
            if repeat>=2:times[method].append(elapsed);checksums[method].append(result['loss'])
    assert all(len(set(v))==1 for v in checksums.values())
    medians={m:float(np.median(v)) for m,v in times.items()};total=sum(medians.values())
    result=dict(**provenance(),batch=64,K=100,nuisance=256,method_seconds=times,method_median_seconds=medians,
        sum_of_method_medians=total,projected_16streams_512steps_compute_seconds=total*16*512,
        projected_compute_minutes=total*16*512/60,new_scientific_RNG_draws=0,new_model_integrals=0,new_scientific_fits=0,
        timing_scope='Every measured step starts anew from identity/scale1 with empty private Adam states. Includes forward/backward, Adam, feasibility eigensolves and full probability diagnostics; excludes RNG, storage, reference scoring and multiprocessing contention.')
    save_new('BENCHMARK.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['check','bench','provenance'])
    phase=parser.parse_args().phase
    if phase=='provenance':print(json.dumps(provenance(),indent=2))
    else:globals()[phase]()
