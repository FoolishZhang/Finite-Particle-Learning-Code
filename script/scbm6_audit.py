"""Independent optimizer/input audit. No producer evaluator is imported.

The official loss is loaded unchanged for limited native replay; NumPy supplies
the independent mathematical gradient and full saved-gradient Adam arithmetic.
No new reference integration or independent scientific inputs are generated.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
import argparse,hashlib,importlib.util,itertools,json,math,time,traceback
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import expit,logsumexp
from scipy.stats import t as student_t
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1];SOURCE=Path(__file__).resolve()
BASE=ROOT/'artifacts/scbm_multicorrelation';TRAIN=BASE/'TRAIN';OUT=BASE/'AUDIT'
DOC=ROOT/'docs/08_research_研究工作/六相关自由学习与方向机制_20261003'
OFF=ROOT/'reproducibility_snapshots/cbm_public_sources/SCBM/58c3e602cb11fb4ea9106226e0f9e58f50d5d10a/models/losses.py'
MU=np.array([-.6,-.2,.2,.6]);C0=np.array([[1.,.25,-.15,.1],[.25,1.,.1,-.15],[-.15,.1,1.,.25],[.1,-.15,.25,1.]])
PAIRS=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3));PATTERNS=np.array(list(itertools.product((0,1),repeat=4)));METHODS=('FIXED','FREE','FACT');STEPS=(0,31,127,511);NUMSTEPS=STEPS;AUDIT_STREAMS=(0,15)
ROLES=dict(data_gaussian=0,data_active_uniform=1,data_noise_uniform=2,prior_active_gaussian=3,prior_noise_gaussian=4)
TOL=dict(api_loss=5e-5,api_rho=3e-6,api_raw=5e-7,mathematical=1e-10,arithmetic=2e-11,
         full_parameter=1e-4,one_step_parameter=3e-6,moment_abs=2e-6,moment_rel=2e-5,tau_abs=2e-7,tau_rel=2e-7,maxlogit=2e-6,repair=1e-12,repair_parameter=2e-7,direction=1e-10)
_LOSS=None
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def require(ok,why):
    if not ok:raise RuntimeError(why)
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def now():return datetime.now(timezone.utc).isoformat()
def near(a,b,tol=TOL['arithmetic'],why='numeric'):
    a,b=np.asarray(a,float),np.asarray(b,float);require(a.shape==b.shape,why+' shape')
    err=float(np.max(abs(a-b))) if a.size else 0.
    require(np.isfinite(a).all() and np.isfinite(b).all() and np.all(abs(a-b)<=tol),why+f' max error {err}')
    return err
def official():
    global _LOSS
    if _LOSS is None:
        spec=importlib.util.spec_from_file_location('scbm6_independent_official',OFF);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        _LOSS=m.SCBLoss(alpha=1.,config=SimpleNamespace(reg_precision='l1',reg_weight=1.,num_monte_carlo=100,training_mode='joint'))
    return _LOSS

def covariance(rho):
    out=np.eye(4)
    for x,(i,j) in zip(rho,PAIRS):out[i,j]=out[j,i]=float(x)
    return out

def repair(rho):
    proposed=np.asarray(rho,np.float32).astype(float);eig=float(np.linalg.eigvalsh(covariance(proposed))[0])
    alpha=(1.-.1001)/(1.-eig) if eig<.1 else 1.
    result=(alpha*proposed).astype(np.float32)
    return result,alpha,eig,float(np.linalg.eigvalsh(covariance(result))[0])

def native(method,rho,raw,e,nu,y,yn):
    rrho=torch.tensor(rho,dtype=torch.float32,requires_grad=True)
    rawp=torch.tensor(raw,dtype=torch.float32,requires_grad=True) if method=='FREE' else None
    rows=[]
    for i in range(4):
        row=[]
        for j in range(4):
            row.append(torch.tensor(1.,dtype=torch.float32) if i==j else rrho[PAIRS.index(tuple(sorted((i,j))))])
        rows.append(torch.stack(row))
    lower=torch.linalg.cholesky(torch.stack(rows))
    x=(lower@torch.tensor(e,dtype=torch.float32).unsqueeze(-1)).squeeze(-1)+torch.tensor(MU,dtype=torch.float32)
    yy=y
    if method!='FACT':
        tau=F.softplus(rawp)+1e-6 if rawp is not None else torch.ones(256,dtype=torch.float32)
        x=torch.cat([x,torch.tensor(nu,dtype=torch.float32)*tau],dim=-1);yy=np.concatenate([y,yn],axis=1)
    prob=torch.sigmoid(x.movedim(0,-1));loss=official().compute_concept_loss(prob,torch.tensor(yy,dtype=torch.int64))
    if method=='FACT':loss=loss+256*math.log(2)
    grad=torch.autograd.grad(loss,(rrho,rawp) if rawp is not None else (rrho,))
    return dict(loss=float(loss.detach()),rho_gradient=grad[0].detach().numpy(),raw_gradient=None if rawp is None else grad[1].detach().numpy(),zero=int((prob.detach()==0).sum()),one=int((prob.detach()==1).sum()),evaluations=prob.numel(),maxlogit=float(x.detach().abs().max()))

def math_score(method,rho,raw,e,nu,y,yn):
    lower=np.linalg.cholesky(covariance(rho));x=e@lower.T+MU;sign=(2*y.astype(float)-1.)[None];signed=x*sign
    logs=-np.logaddexp(0.,-signed).sum(axis=-1);sn=tau=rawfactor=None
    if method!='FACT':
        if method=='FREE':
            rr=np.asarray(raw,float);tau=np.where(rr>20,rr,np.logaddexp(0.,rr))+1e-6;rawfactor=np.where(rr>20,1.,expit(rr))
        else:tau=np.ones(256)
        sn=nu*(2*yn.astype(float)-1.)[None];logs+=-np.logaddexp(0.,-sn*tau).sum(axis=-1)
    norm=logsumexp(logs,axis=0);weights=np.exp(logs-norm[None]);loss=float(np.mean(math.log(100)-norm))
    if method=='FACT':loss+=256*math.log(2)
    gradients=[]
    for i,j in PAIRS:
        change=np.zeros((4,4));change[i,j]=change[j,i]=1.
        left=np.linalg.solve(lower,change);center=np.linalg.solve(lower,left.T).T
        phi=np.tril(center);phi[np.diag_indices(4)]*=.5;dx=e@(lower@phi).T
        gradients.append(float(-np.sum(weights[:,:,None]*sign*expit(-signed)*dx)/len(y)))
    gr=None if method!='FREE' else -np.sum(weights[:,:,None]*sn*expit(-sn*tau),axis=(0,1))/len(y)*rawfactor
    return dict(loss=loss,rho_gradient=np.array(gradients),raw_gradient=gr)


def adam(m,v,g,step):
    """Independent scalar/elementwise float32 Adam expression, no optimizer call."""
    g=np.asarray(g,np.float32)
    m=(m+np.float32(.1)*(g-m)).astype(np.float32)
    v=(v*np.float32(.999)+np.float32(.001)*g*g).astype(np.float32)
    den=(np.sqrt(v)/np.float32(math.sqrt(1-.999**step))+np.float32(1e-8)).astype(np.float32)
    delta=(np.float32(-.02/(1-.9**step))*m/den).astype(np.float32)
    return m,v,delta

def artificial():
    e=np.sin(np.arange(100*3*4).reshape(100,3,4)*.13);nu=np.cos(np.arange(100*3*256).reshape(100,3,256)*.019)
    y=(np.arange(12).reshape(3,4)%3==0).astype(np.int8);yn=(np.arange(768).reshape(3,256)%5<2).astype(np.int8)
    raw=np.full(256,math.log(math.expm1(1-1e-6)));rho=np.array([.11,-.07,.05,.09,-.06,.13]);h=1e-5
    answer=math_score('FREE',rho,raw,e,nu,y,yn);errors=[]
    for q in range(6):
        plus=rho.copy();minus=rho.copy();plus[q]+=h;minus[q]-=h
        fd=(math_score('FREE',plus,raw,e,nu,y,yn)['loss']-math_score('FREE',minus,raw,e,nu,y,yn)['loss'])/(2*h)
        errors.append(near(fd,answer['rho_gradient'][q],2e-8,'synthetic rho FD'))
    for q in (0,127,255):
        plus=raw.copy();minus=raw.copy();plus[q]+=h;minus[q]-=h
        fd=(math_score('FREE',rho,plus,e,nu,y,yn)['loss']-math_score('FREE',rho,minus,e,nu,y,yn)['loss'])/(2*h)
        errors.append(near(fd,answer['raw_gradient'][q],2e-8,'synthetic raw FD'))
    parameter=torch.nn.Parameter(torch.zeros(274,dtype=torch.float32));optimizer=torch.optim.Adam([parameter],lr=.02,betas=(.9,.999),eps=1e-8,foreach=False,fused=False)
    m=np.zeros(274,np.float32);v=m.copy();localmax=0.
    for step in range(512):
        g=(.05*np.sin((np.arange(274)+1)*(step+1)*.013)).astype(np.float32);before=parameter.detach().numpy().copy()
        parameter.grad=torch.tensor(g);optimizer.step();m,v,delta=adam(m,v,g,step+1)
        localmax=max(localmax,near(before+delta,parameter.detach().numpy(),TOL['one_step_parameter'],'synthetic Adam'))
    for value,key in [(m,'exp_avg'),(v,'exp_avg_sq')]:near(value,optimizer.state[parameter][key].numpy(),TOL['moment_abs']+TOL['moment_rel']*abs(optimizer.state[parameter][key].numpy()),'synthetic moments')
    for signs in itertools.product((-1,1),repeat=6):
        result,alpha,_,after=repair(.95*np.array(signs));require(after>=.1 and 0<alpha<1,'synthetic repair')
    native_errors=[]
    for method in METHODS:
        aa=native(method,rho,raw if method=='FREE' else None,e,nu,y,yn)
        mm=math_score(method,rho,raw if method=='FREE' else None,e,nu,y,yn)
        native_errors.append(near(aa['loss'],mm['loss'],1e-4,'constant native loss'))
        near(aa['rho_gradient'],mm['rho_gradient'],2e-5,'constant native six gradient')
        if method=='FREE':near(aa['raw_gradient'],mm['raw_gradient'],2e-6,'constant native raw')
    return dict(passed=True,maximum_FD_error=max(errors),Adam_one_step_error=localmax,synthetic_repair_vertices=64,native_constant_checks=3,maximum_constant_native_loss_error=max(native_errors),scientific_RNG=0,model_integrals=0)


def freeze():
    paths=[SOURCE,DOC/'AUDIT_PLAN.md',DOC/'EXPERIMENT_DESIGN.md',TRAIN/'FREEZE.json',TRAIN/'RESULTS.json',BASE/'DIRECTION/FREEZE.json',BASE/'DIRECTION/RESULTS.json',ROOT/'script/scbm6_run.py',ROOT/'script/scbm6_kernel.py',ROOT/'script/scbm6_reference.py',ROOT/'script/scbm6_direction.py',ROOT/'script/scbmeval_kernel.py',ROOT/'script/scbminv_reference.py',OFF,ROOT/'artifacts/scbm_invariance/REFERENCE/MODEL.json']
    for extra in [BASE/'REFERENCE/FREEZE.json',BASE/'REFERENCE/RESULTS.json']:
        if extra.exists():paths.append(extra)
    save(OUT/'PROTOCOL.json',dict(created_utc=now(),inputs={rel(p):sha(p) for p in paths},tolerances=TOL,streams=list(AUDIT_STREAMS),steps=list(STEPS),methods=list(METHODS),direction_prefix_banks=4,deterministic_check=artificial(),numpy_version=np.__version__,torch_version=torch.__version__,new_model_integrals=0,automatic_retry=False,scope='Whole input and saved-arithmetic replay; only24 native states/8 raw vectors and128 direction banks at bottom level. Not blind to ROOT status summaries.'))
    print(json.dumps(dict(source_sha256=sha(SOURCE),protocol_sha256=sha(OUT/'PROTOCOL.json'))),flush=True)

def bind():
    protocol=read(OUT/'PROTOCOL.json')
    for path,h in protocol['inputs'].items():require(sha(ROOT/path)==h,'audit input changed '+path)
    return protocol


def replay(stream):
    rng={name:np.random.Generator(np.random.PCG64(np.random.SeedSequence([202610031100,stream,r]))) for name,r in ROLES.items()}
    initial={name:r.bit_generator.state for name,r in rng.items()}
    z=rng['data_gaussian'].standard_normal((4096,4));u=rng['data_active_uniform'].random((4096,4));un=rng['data_noise_uniform'].random((4096,256))
    y=(u<expit(z@np.linalg.cholesky(C0).T+MU)).astype(np.int8);yn=(un<.5).astype(np.int8)
    hashes={name:ah(a) for name,a in [('data_gaussian',z),('data_active_uniform',u),('data_noise_uniform',un)]}
    hh={name:hashlib.sha256() for name in ['prior_active_gaussian','prior_noise_gaussian']};first={};chosen={}
    for step in range(512):
        e=rng['prior_active_gaussian'].standard_normal((100,64,4));nu=rng['prior_noise_gaussian'].standard_normal((100,64,256))
        for name,a in [('prior_active_gaussian',e),('prior_noise_gaussian',nu)]:
            b=a.tobytes();hh[name].update(b)
            if step==0:first[name]=hashlib.sha256(b).hexdigest()
        if step==0 or (stream in AUDIT_STREAMS and step in STEPS):chosen[step]=(e.copy(),nu.copy())
    hashes.update({name:h.hexdigest() for name,h in hh.items()})
    data=dict(active_labels=y,nuisance_labels=yn,data_active_gaussian=z,data_active_uniform=u,first_prior_active=chosen[0][0],first_prior_nuisance=chosen[0][1])
    return data,chosen,hashes,first,initial,{name:r.bit_generator.state for name,r in rng.items()}

def trajectory(tape,record):
    rho=np.asarray(tape['rho']);raw=np.asarray(tape['free_raw']);tau=np.asarray(tape['free_tau'])
    require(rho.shape==(3,513,6) and raw.shape==tau.shape==(513,256),'trajectory dimensions')
    require(np.array_equal(rho[:,0],np.zeros((3,6),np.float32)),'identity initialization')
    initial=np.float32((1-1e-6)+math.log(-math.expm1(-(1-1e-6))))
    require(np.array_equal(raw[0],np.full(256,initial,np.float32)),'raw initialization')
    require(np.array_equal(tape['batch_start'],np.arange(512)*64%4096),'cyclic batches')
    physical=np.where(raw.astype(float)>20,raw.astype(float),np.logaddexp(0.,raw.astype(float)))+1e-6
    tauerr=near(physical,tau,TOL['tau_abs']+TOL['tau_rel']*abs(tau),'all physical scales')
    require(np.isfinite(rho).all() and np.isfinite(raw).all() and (tau>=np.float32(1e-6)).all(),'finite/floor')
    require(tape['finite'].all() and tape['feasible'].all() and tape['cholesky_success'].all(),'saved qualification')
    gradients=np.concatenate([tape['rho_gradient'].transpose(1,0,2).reshape(512,18),tape['free_raw_gradient']],axis=1).astype(np.float32)
    require(gradients.shape==(512,274) and np.isfinite(gradients).all(),'gradient dimensions')
    m=np.zeros(274,np.float32);v=m.copy();expected_raw=raw[0].copy();maxlocal=maxraw=maxrepair=0.
    for step in range(512):
        m,v,delta=adam(m,v,gradients[step],step+1);before=np.r_[rho[:,step].ravel(),raw[step]].astype(np.float32);proposal=before+delta
        maxlocal=max(maxlocal,near(proposal[:18].reshape(3,6),tape['rho_unprojected'][:,step],TOL['one_step_parameter'],'six-rho proposal'))
        maxlocal=max(maxlocal,near(proposal[18:],raw[step+1],TOL['one_step_parameter'],'raw update'))
        expected_raw+=delta[18:];maxraw=max(maxraw,near(expected_raw,raw[step+1],TOL['full_parameter'],'cumulative raw'))
        for mi in range(3):
            expected,alpha,eig0,eig1=repair(tape['rho_unprojected'][mi,step])
            maxrepair=max(maxrepair,near(expected,rho[mi,step+1],TOL['repair_parameter'],'cast32 repair'))
            near(alpha,tape['projection_alpha'][mi,step],TOL['repair'],'repair alpha');near(eig0,tape['proposal_eigmin'][mi,step],TOL['repair'],'proposal eigenvalue')
            near(eig1,tape['eigmin'][mi,step+1],TOL['repair'],'repaired eigenvalue')
            require(tape['repaired'][mi,step]==(eig0<.1) and eig1>=.1,'repair trigger/feasibility');np.linalg.cholesky(covariance(rho[mi,step+1]))
    final_m=np.r_[tape['final_rho_adam_m'].ravel(),tape['final_free_raw_adam_m']];final_v=np.r_[tape['final_rho_adam_v'].ravel(),tape['final_free_raw_adam_v']]
    merr=near(m,final_m,TOL['moment_abs']+TOL['moment_rel']*abs(final_m),'final m');verr=near(v,final_v,TOL['moment_abs']+TOL['moment_rel']*abs(final_v),'final v')
    near(tape['eigmin'][:,0],np.ones(3),0.,'initial eig')
    for mi,method in enumerate(METHODS):
        for kind in ['TAIL','LAST']:
            r=rho[mi,449:].astype(float).mean(axis=0) if kind=='TAIL' else rho[mi,512].astype(float)
            scales=tau[449:].astype(float).mean(axis=0) if kind=='TAIL' else tau[512].astype(float)
            if method!='FREE':scales=np.full(256,1. if method=='FIXED' else 0.)
            output=record['outputs'][method][kind];near(r,output['rho'],why='physical rho endpoint');near(covariance(r),output['covariance'],why='endpoint covariance');near(scales,output['scales'],why='physical scale endpoint')
            eig=float(np.linalg.eigvalsh(covariance(r))[0]);require(eig>=.1,'TAIL convex feasibility');near(eig,output['eigmin'],TOL['repair'],'endpoint eig')
        d=record['diagnostics'][method]
        for key,value in [('feasibility_repair_count',int(tape['repaired'][mi].sum())),('sigmoid_zero_count',int(tape['sigmoid_zero_count'][mi].sum())),('sigmoid_one_count',int(tape['sigmoid_one_count'][mi].sum())),('probability_evaluations',int(tape['probability_evaluations'][mi].sum()))]:require(d[key]==value,'diagnostic '+key)
        for key,value in [('minimum_eigenvalue',tape['eigmin'][mi].min()),('minimum_proposal_eigenvalue',tape['proposal_eigmin'][mi].min()),('minimum_repair_alpha',tape['projection_alpha'][mi].min()),('maximum_absolute_logit',tape['maximum_absolute_logit'][mi].max())]:near(value,d[key],why='diagnostic '+key)
        require(d['all_finite'] and d['all_feasible'] and d['all_Cholesky_success'],'diagnostic qualifications')
        require(np.array_equal(tape['probability_evaluations'][mi],np.full(512,100*64*(4 if method=='FACT' else 260))),'probability evaluation count')
    return dict(tau_error=tauerr,Adam_one_step_error=maxlocal,raw_cumulative_error=maxraw,repair_error=maxrepair,final_m_error=merr,final_v_error=verr)


def prefix_math(data,inputs,tape,record):
    errors=dict(API_loss=0.,API_rho=0.,API_raw=0.,stable_loss=0.,stable_rho=0.,stable_raw=0.)
    require({(r['preupdate_step'],r['method']) for r in record['numerical_controls']}=={(s,m) for s in NUMSTEPS for m in METHODS},'NUM scope')
    for step in STEPS:
        begin=64*step%4096;y=data['active_labels'][begin:begin+64];yn=data['nuisance_labels'][begin:begin+64];e,nu=inputs[step]
        for mi,method in enumerate(METHODS):
            tt=tape['rho'][mi,step];raw=tape['free_raw'][step] if method=='FREE' else None
            api=native(method,tt,raw,e,nu,y,yn);mat=math_score(method,tt,raw,e,nu,y,yn)
            errors['API_loss']=max(errors['API_loss'],near(api['loss'],tape['loss'][mi,step],TOL['api_loss'],'prefix original API loss'))
            errors['API_rho']=max(errors['API_rho'],near(api['rho_gradient'],tape['rho_gradient'][mi,step],TOL['api_rho'],'prefix original API t gradient'))
            if method=='FREE':errors['API_raw']=max(errors['API_raw'],near(api['raw_gradient'],tape['free_raw_gradient'][step],TOL['api_raw'],'prefix original API raw gradient'))
            require(api['zero']==tape['sigmoid_zero_count'][mi,step] and api['one']==tape['sigmoid_one_count'][mi,step] and api['evaluations']==tape['probability_evaluations'][mi,step],'prefix probability counters')
            near(api['maxlogit'],tape['maximum_absolute_logit'][mi,step],TOL['maxlogit'],'prefix maxlogit')
            if step in NUMSTEPS:
                row=next(r for r in record['numerical_controls'] if r['preupdate_step']==step and r['method']==method)
                near(row['official_loss'],tape['loss'][mi,step],0.,'NUM tape loss');near(row['official_rho_gradient'],tape['rho_gradient'][mi,step],0.,'NUM tape t gradient')
                errors['stable_loss']=max(errors['stable_loss'],near(mat['loss'],row['stable64_loss'],TOL['mathematical'],'NumPy stable loss'))
                errors['stable_rho']=max(errors['stable_rho'],near(mat['rho_gradient'],row['stable64_rho_gradient'],TOL['mathematical'],'NumPy Cholesky t gradient'))
                le=row['official_loss']-row['stable64_loss'];te=np.asarray(row['official_rho_gradient'])-row['stable64_rho_gradient']
                near(le,row['loss_difference'],why='NUM loss difference');near(te,row['rho_gradient_difference'],why='NUM t difference')
                require(row['loss_passed']==(abs(le)<=1e-4) and row['rho_gradient_passed']==(np.max(abs(te))<=2e-5),'NUM loss/t gate')
                if method=='FREE':
                    near(row['official_raw_gradient'],tape['free_raw_gradient'][step],0.,'NUM raw tape')
                    errors['stable_raw']=max(errors['stable_raw'],near(mat['raw_gradient'],row['stable64_raw_gradient'],TOL['mathematical'],'NumPy raw path gradient'))
                    diff=np.array(row['official_raw_gradient'])-np.array(row['stable64_raw_gradient']);mx=float(max(abs(diff)))
                    near(diff,row['raw_gradient_difference'],why='NUM raw differences');near(mx,row['maximum_raw_gradient_difference'],why='NUM raw maximum')
                    require(row['raw_gradient_passed']==(mx<=2e-6),'NUM raw gate')
                require(row['passed']==(row['loss_passed'] and row['rho_gradient_passed'] and row.get('raw_gradient_passed',True)),'NUM joint gate')
    return errors

def num_summary(rows):
    for row in rows:
        le=row['official_loss']-row['stable64_loss'];diff=np.asarray(row['official_rho_gradient'])-row['stable64_rho_gradient'];mx=float(np.max(abs(diff)))
        near(le,row['loss_difference'],why='all NUM loss difference');near(diff,row['rho_gradient_difference'],why='all NUM rho diff');near(mx,row['maximum_rho_gradient_difference'],why='all NUM rho max')
        require(row['loss_passed']==(abs(le)<=1e-4) and row['rho_gradient_passed']==(mx<=2e-5),'all NUM gates')
        if row['method']=='FREE':
            rd=np.asarray(row['official_raw_gradient'])-row['stable64_raw_gradient'];rm=float(np.max(abs(rd)))
            near(rd,row['raw_gradient_difference'],why='all NUM raw difference');near(rm,row['maximum_raw_gradient_difference'],why='all NUM raw max');require(row['raw_gradient_passed']==(rm<=2e-6),'all NUM raw gate')
        require(row['passed']==(row['loss_passed'] and row['rho_gradient_passed'] and row.get('raw_gradient_passed',True)),'all NUM joint')
    return dict(checks=len(rows),passed=sum(r['passed'] for r in rows),loss_passed=sum(r['loss_passed'] for r in rows),rho_gradient_checks=len(rows),rho_gradient_components=6*len(rows),rho_gradient_passed=sum(r['rho_gradient_passed'] for r in rows),raw_gradient_checks=sum('raw_gradient_passed' in r for r in rows),raw_gradient_passed=sum(r.get('raw_gradient_passed',False) for r in rows),maximum_loss_difference=max(abs(r['loss_difference']) for r in rows),maximum_rho_gradient_difference=max(r['maximum_rho_gradient_difference'] for r in rows),maximum_raw_gradient_difference=max(r.get('maximum_raw_gradient_difference',0) for r in rows))


def run():
    bind();prod=read(TRAIN/'RESULTS.json');f=read(TRAIN/'FREEZE.json')
    require(prod['freeze_sha256']==sha(TRAIN/'FREEZE.json') and prod['source_sha256']==sha(ROOT/'script/scbm6_run.py') and prod['config']==f['config'],'TRAIN bindings')
    require(prod['config']['steps']==512 and prod['config']['numerical_preupdate_steps']==list(NUMSTEPS),'TRAIN scope')
    save(OUT/'TRAIN_EXECUTION.json',dict(start_utc=now(),source_sha256=sha(SOURCE),protocol_sha256=sha(OUT/'PROTOCOL.json'),training_results_sha256=sha(TRAIN/'RESULTS.json'),automatic_retry=False))
    start=time.perf_counter();records=[]
    require(len(prod['records'])==16 and {r['stream'] for r in prod['records']}==set(range(16)),'streams')
    for stream in range(16):
        tick=time.perf_counter();rec=next(r for r in prod['records'] if r['stream']==stream)
        data,inputs,hashes,first,initial,final=replay(stream)
        require(rec['role_input_sha256']==hashes and rec['first_prior_sha256']==first,'full five-role byte hash')
        require(rec['RNG_initial_states']==initial and rec['RNG_final_states']==final,'initial/final PCG64 states')
        require(rec['role_seed_sequences']=={name:[202610031100,stream,r] for name,r in ROLES.items()},'role seed metadata')
        require(rec['label_sha256']==dict(active_labels=ah(data['active_labels']),nuisance_labels=ah(data['nuisance_labels'])),'full label hashes')
        require(sha(ROOT/rec['input_file'])==rec['input_sha256'] and sha(ROOT/rec['output_file'])==rec['output_sha256'],'input/trajectory bytes')
        require(rec['source_sha256']==prod['source_sha256'] and rec['freeze_sha256']==prod['freeze_sha256'] and rec['completed_steps']==512,'stream binding/steps')
        with np.load(ROOT/rec['input_file']) as a:
            require(set(a.files)==set(data),'input field scope')
            for name,x in data.items():require(np.array_equal(x,a[name]),'saved input '+name)
        with np.load(ROOT/rec['output_file']) as a:tape={name:a[name].copy() for name in a.files}
        updates=trajectory(tape,rec)
        require({(r['preupdate_step'],r['method']) for r in rec['numerical_controls']}=={(step,method) for step in NUMSTEPS for method in METHODS},'all NUM scope')
        for row in rec['numerical_controls']:
            mi=METHODS.index(row['method']);step=row['preupdate_step']
            near(row['official_loss'],tape['loss'][mi,step],0.,'all NUM tape loss')
            near(row['official_rho_gradient'],tape['rho_gradient'][mi,step],0.,'all NUM tape gradient')
            if row['method']=='FREE':near(row['official_raw_gradient'],tape['free_raw_gradient'][step],0.,'all NUM raw tape')
        errors=prefix_math(data,inputs,tape,rec) if stream in AUDIT_STREAMS else {'bottom_level_not_recomputed':True}
        require(num_summary(rec['numerical_controls'])==rec['numerical_qualification'],'stream NUM summary')
        rr=dict(stream=stream,passed=True,complete_RNG_hash_states_labels=True,trajectory=updates,prefix=errors,seconds=time.perf_counter()-tick)
        records.append(rr);save(OUT/f'stream_{stream:02}.json',rr)
        print(json.dumps(dict(audit_stream=stream,passed=True,seconds=rr['seconds'],Adam_error=updates['raw_cumulative_error'],prefix=errors)),flush=True)
    rows=[r for rec in prod['records'] for r in rec['numerical_controls']];require(num_summary(rows)==prod['numerical_qualification'],'all NUM summary')
    ledger=dict(independent_streams=16,new_complete_observations=65536,data_Gaussian_scalars=262144,data_Uniform_scalars=17039360,new_prior_vectors=52428800,prior_Gaussian_scalars=13631488000,prior_active_Gaussian_scalars=209715200,prior_noise_Gaussian_scalars=13421772800,observation_presentations=524288,controlled_optimization_programs=48,TAIL_outputs=48,LAST_outputs=48,official_training_API_calls=24576,stable64_control_calls=192,checked_rho_gradient_components=1152,new_neural_network_fits=0,new_model_integrals=0)
    require(prod['ledger']==ledger,'production training ledger')
    save(OUT/'TRAIN_REPORT.json',dict(passed=True,source_sha256=sha(SOURCE),protocol_sha256=sha(OUT/'PROTOCOL.json'),training_results_sha256=sha(TRAIN/'RESULTS.json'),records=records,numerical_qualification=prod['numerical_qualification'],seconds=time.perf_counter()-start,
         accounting=dict(replayed_prior_Gaussian=13631488000,replayed_data_Gaussian=262144,replayed_data_Uniform=17039360,new_independent_scientific_draws=0,new_model_integrals=0,unique_step_inputs=8,official_API_calls=24,NumPy_mathematical_scores=24,NumPy_FREE_gradient_vectors=8,Adam_program_steps=24576),
         scope='Full input/state replay and all saved-gradient Adam arithmetic; bottom-level only streams0/15 at steps0/31/127/511; rho updates checked locally against every saved state/proposal; cumulative raw/moments independently rebuilt. Not a complete independent training rerun.'))


SCALARS=('score','centered_score','overlap','mean_ess','mean_inverse_ess')
VECTORS=('gradient','mean_mark_term','coupling_term','markmc')
METRICS=SCALARS+VECTORS

def truth_reference():
    model=read(ROOT/'artifacts/scbm_invariance/REFERENCE/MODEL.json');require(model['patterns']==PATTERNS.tolist() and model['qualification_passed'],'truth reference')
    p=np.asarray(next(r['main_pmf'] for r in model['records'] if r['t']==0.));spin=2.*PATTERNS-1.;means=p@spin
    cov=np.array([p@(spin[:,i]*spin[:,j])-means[i]*means[j] for i,j in PAIRS])
    return p,np.stack([cov/4,cov/2])

def direction_math(active,noise,p,centers):
    all_values={k:[] for k in METRICS};spins=2.*PATTERNS-1.
    for ci,C in enumerate([np.eye(4),C0]):
        x=active@np.linalg.cholesky(C).T+MU;signed=x[:,:,None,:]*spins[None,None,:,:]
        loga=-np.logaddexp(0.,-signed).sum(axis=-1)
        residual=spins[None,None,:,:]*expit(-signed)
        marks=np.stack([residual[...,i]*residual[...,j] for i,j in PAIRS],axis=-1)
        profiles={k:[] for k in METRICS}
        for scale in (1.,.2,0.):
            lognu=np.zeros(active.shape[:2]) if scale==0 else -np.logaddexp(0.,noise*scale).sum(axis=2)+noise.shape[2]*math.log(2)
            logw=loga+lognu[:,:,None];maximum=logw.max(axis=1,keepdims=True);w=np.exp(logw-maximum);den=w.sum(axis=1,keepdims=True);pi=w/den
            inverse=(pi*pi).sum(axis=1);overlap=1-inverse;gamma=pi*(1-pi)
            centered=math.log(active.shape[1])-(maximum+np.log(den)).squeeze(axis=1)
            gradient=-np.einsum('bkp,bkpq->bpq',gamma,marks)
            mean=-overlap[:,:,None]*centers[ci];coupling=-np.einsum('bkp,bkpq->bpq',gamma,marks-centers[ci])
            values=dict(score=centered+noise.shape[2]*math.log(2),centered_score=centered,overlap=overlap,mean_ess=1/inverse,mean_inverse_ess=inverse,gradient=gradient,mean_mark_term=mean,coupling_term=coupling,markmc=marks.mean(axis=1))
            for k in METRICS:profiles[k].append(values[k])
        for k in METRICS:all_values[k].append(np.stack(profiles[k]))
    return {k:np.stack(v) for k,v in all_values.items()}

def direction():
    bind();base=BASE/'DIRECTION';prod=read(base/'RESULTS.json');f=read(base/'FREEZE.json');p,centers=truth_reference()
    require(prod['source_sha256']==sha(ROOT/'script/scbm6_direction.py') and prod['freeze_sha256']==sha(base/'FREEZE.json'),'DIRECTION binding')
    for name,h in f['inputs'].items():require(sha(ROOT/name)==h,'DIRECTION frozen input')
    require(prod['config']==f['config'] and prod['config']['family_size']==18 and prod['config']['SE_threshold']==5e-5,'DIRECTION scope')
    near(p,prod['reference']['weights'],0.,'reference weights');near(centers,prod['reference']['mean_mark_reference'],why='mean mark center')
    save(OUT/'DIRECTION_EXECUTION.json',dict(start_utc=now(),source_sha256=sha(SOURCE),protocol_sha256=sha(OUT/'PROTOCOL.json'),results_sha256=sha(base/'RESULTS.json'),automatic_retry=False))
    tick=time.perf_counter();records=[];group={k:[] for k in METRICS}
    require([r['replicate'] for r in prod['records']]==list(range(32)),'replicate order')
    for index,r in enumerate(prod['records']):
        rng={name:np.random.Generator(np.random.PCG64(np.random.SeedSequence([202610031200,index,role]))) for name,role in [('active',0),('nuisance',1)]}
        initial={k:v.bit_generator.state for k,v in rng.items()};digests={k:hashlib.sha256() for k in rng};prefix={}
        for block in range(8):
            for key,d in [('active',4),('nuisance',256)]:
                z=rng[key].standard_normal((64,100,d));digests[key].update(z.tobytes(order='C'))
                if block==0:prefix[key]=z[:4].copy()
        require(r['role_entropy']=={name:[202610031200,index,role] for name,role in [('active',0),('nuisance',1)]},'DIRECTION seeds')
        require(r['RNG_initial_states']==initial and r['RNG_final_states']=={k:v.bit_generator.state for k,v in rng.items()},'DIRECTION RNG states')
        require(r['role_input_sha256']=={k:v.hexdigest() for k,v in digests.items()},'DIRECTION all input hashes')
        require(r['consumed_banks']==512 and r['completed_blocks']==8,'DIRECTION budget')
        require(r['source_sha256']==prod['source_sha256'] and r['freeze_sha256']==prod['freeze_sha256'],'replicate binding')
        path=ROOT/r['arrays_path'];require(sha(path)==r['arrays_sha256'],'DIRECTION NPZ hash')
        values=direction_math(prefix['active'],prefix['nuisance'],p,centers);maximum=0.;weighted={}
        with np.load(path) as a:
            require(set(a.files)=={'active','nuisance'}|{'pattern_'+k for k in METRICS}|{'block_'+k for k in METRICS},'DIRECTION NPZ fields')
            for key in ['active','nuisance']:
                require(np.array_equal(a[key],prefix[key]) and ah(prefix[key])==r['prefix_'+key+'_sha256'],'DIRECTION prefix input')
            for key in METRICS:
                maximum=max(maximum,near(values[key],a['pattern_'+key],TOL['direction'],'DIRECTION prefix '+key))
                block=np.asarray(a['block_'+key]);require(block.shape==((8,2,3,16) if key in SCALARS else (8,2,3,16,6)),'DIRECTION block axes')
                mean=block.mean(axis=0);near(mean,r['pattern_means'][key],why='block aggregation '+key)
                weighted[key]=np.tensordot(mean,p,axes=(2,0));near(weighted[key],r['means'][key],why='pattern weighting '+key);group[key].append(weighted[key])
        for matrix in range(2):
            near(weighted['markmc'][matrix,0],weighted['markmc'][matrix,1],why='mark profile independence');near(weighted['markmc'][matrix,0],weighted['markmc'][matrix,2],why='mark FACT independence')
        decomp=float(np.max(abs(weighted['gradient']-weighted['mean_mark_term']-weighted['coupling_term'])))
        overlap=float(np.max(abs(weighted['overlap']+weighted['mean_inverse_ess']-p.sum())))
        near(decomp,r['maximum_decomposition_error'],why='decomposition aggregate');require(decomp<=1e-10 and overlap<=1e-10,'decomposition qualifications')
        ids=r['overlap_identities']
        for key,value in [('raw_pattern_weight_mass',float(p.sum())),('mass_minus_mean_inverse_ESS',p.sum()-weighted['mean_inverse_ess']),('one_minus_mean_inverse_ESS',1-weighted['mean_inverse_ess']),('one_minus_inverse_mean_ESS_different_quantity',1-1/weighted['mean_ess']),('maximum_actual_identity_error',overlap)]:near(value,ids[key],why='ESS identity '+key)
        rr=dict(replicate=index,passed=True,all_input_hash_state_replay=True,prefix_banks=4,maximum_prefix_error=maximum);records.append(rr);save(OUT/f'direction_{index:02}.json',rr)
    group={k:np.asarray(v) for k,v in group.items()};gradient=group['gradient'];sign=np.sign([C0[i,j] for i,j in PAIRS]);critical=float(student_t.ppf(1-.05/36,31));count=0
    definitions=[('signed_scale1',gradient[:,1,0]*sign,'negative'),('signed_scale1_minus_FACT',(gradient[:,1,0]-gradient[:,1,2])*sign,'negative'),('signed_scale02_minus_scale1',(gradient[:,1,1]-gradient[:,1,0])*sign,'positive')]
    for name,values,direction_name in definitions:
        for q,pair in enumerate(PAIRS):
            row=prod['primary_family'][count];count+=1;v=values[:,q];mean=math.fsum(v)/32;se=math.sqrt(math.fsum((v-mean)**2)/31/32);lo,hi=mean-critical*se,mean+critical*se
            require(row['name']==name and row['pair']==list(pair) and row['truth_sign']==int(sign[q]) and row['expected']==direction_name and row['replicates']==32 and row['df']==31,'DIRECTION primary identity')
            for key,value in [('replicate_values',v),('mean',mean),('SE',se),('lower',lo),('upper',hi),('critical_t',critical),('SE_threshold',5e-5)]:near(value,row[key],why='DIRECTION primary '+key)
            require(row['effect_passed']==(hi<0 if direction_name=='negative' else lo>0) and row['precision_passed']==(se<=5e-5),'DIRECTION gates')
    require(len(prod['primary_family'])==18 and prod['primary_effects_passed']==sum(x['effect_passed'] for x in prod['primary_family']) and prod['primary_precision_passed']==sum(x['precision_passed'] for x in prod['primary_family']),'DIRECTION gate counts')
    for key,value in group.items():
        near(value.mean(axis=0),prod['descriptive'][key]['mean'],why='DIRECTION descriptive mean');near(value.std(axis=0,ddof=1)/math.sqrt(32),prod['descriptive'][key]['SE'],why='DIRECTION descriptive SE')
    residual=gradient[:,1,1]-gradient[:,1,2];near(residual.mean(axis=0),prod['truth_scale02_minus_FACT_descriptive']['mean'],why='small-scale residual mean');near(residual.std(axis=0,ddof=1)/math.sqrt(32),prod['truth_scale02_minus_FACT_descriptive']['SE'],why='small-scale residual SE')
    ledger=dict(independent_MC_groups=32,new_base_banks=16384,new_prior_vectors=1638400,Gaussian_scalars=425984000,active_Gaussian_scalars=6553600,nuisance_Gaussian_scalars=419430400,new_observations=0,new_fits=0,new_model_integrals=0,main_official_API_calls=0,saved_prefix_complete_banks=128,saved_block_pattern_mean_arrays=256)
    require(prod['ledger']==ledger,'DIRECTION ledger')
    combined=sum(x['effect_passed'] and x['precision_passed'] for x in prod['primary_family'])
    save(OUT/'DIRECTION_REPORT.json',dict(passed=True,source_sha256=sha(SOURCE),protocol_sha256=sha(OUT/'PROTOCOL.json'),results_sha256=sha(base/'RESULTS.json'),records=records,science_gate_counts=dict(effect=prod['primary_effects_passed'],precision=prod['primary_precision_passed'],combined=combined,total=18,whole_family_combined=combined==18),seconds=time.perf_counter()-tick,new_independent_scientific_draws=0,new_model_integrals=0,scope='Complete two-role replay and saved arithmetic; only first4 banks per32groups independently scored, all16patterns/6coordinates/2matrices/3profiles.'))

def statistics():
    bind();require(read(OUT/'TRAIN_REPORT.json')['passed'],'TRAIN audit incomplete');ref=read(BASE/'REFERENCE/RESULTS.json');train=read(TRAIN/'RESULTS.json');p,unused=truth_reference()
    require(ref['source_sha256']==sha(ROOT/'script/scbm6_reference.py') and ref['training_results_sha256']==sha(TRAIN/'RESULTS.json') and ref['freeze_sha256']==sha(BASE/'REFERENCE/FREEZE.json'),'REFERENCE bindings')
    for name,h in read(BASE/'REFERENCE/FREEZE.json')['inputs'].items():require(sha(ROOT/name)==h,'REFERENCE frozen input')
    save(OUT/'STATISTICS_EXECUTION.json',dict(start_utc=now(),source_sha256=sha(SOURCE),reference_results_sha256=sha(BASE/'REFERENCE/RESULTS.json'),new_integrals=0))
    key_of=lambda r:'|'.join(float(x).hex() for x in r)
    expected={};unique={key_of(np.zeros(6))}
    for row in train['records']:
        for method in METHODS:
            for kind in ('TAIL','LAST'):
                r=row['outputs'][method][kind]['rho'];expected[row['stream'],method,kind]=r;unique.add(key_of(r))
    bykey={r['key']:r for r in ref['records']};require(set(bykey)==unique and len(ref['endpoints'])==96,'REFERENCE endpoints/dedup')
    risks={}
    for row in ref['records']:
        require(row['key']==key_of(row['rho']),'rho hex identity');C=covariance(row['rho']);near(C,row['covariance'],0.,'complete covariance')
        eig=float(np.linalg.eigvalsh(C)[0]);near(eig,row['minimum_eigenvalue'],1e-12,'reference eigenvalue');main=np.asarray(row['main_pmf'])
        require([v['order'] for v in row['variants']]==[24,32,40],'reference fixed orders');near(main,next(v['pmf'] for v in row['variants'] if v['order']==32),0.,'main PMF')
        for v,q in zip(row['variants'],row['qualifications']):
            value=np.asarray(v['pmf']);change=float(np.max(abs(value-main)));norm=abs(float(value.sum())-1.);positive=bool((value>0).all());passed=change<=5e-8 and norm<=2e-12 and positive
            near(change,q['maximum_change'],why='reference order change');near(norm,q['normalization_error'],why='reference mass');require(q['positive']==positive and q['passed']==passed,'reference qualification')
        risk=math.fsum(float(a)*math.log(float(a/b)) for a,b in zip(p,main));near(risk,row['reference_joint_KL'],why='stored PMF KL');risks[row['key']]=risk
        require(row['passed']==(eig>=.09999 and all(q['passed'] for q in row['qualifications'])),'reference matrix qualification')
    observed={}
    for row in ref['endpoints']:
        ident=row['stream'],row['method'],row['kind'];require(row['rho']==expected[ident] and row['key']==key_of(row['rho']),'endpoint identity')
        near(row['reference_joint_KL'],risks[row['key']],why='endpoint risk');observed[ident]=row['reference_joint_KL']
    require(ref['identity_reference']==bykey[key_of(np.zeros(6))],'identity reference')
    crit=float(student_t.ppf(1-.05/6,15));near(crit,ref['critical_t'],why='TRAIN risk critical')
    definitions=[('FIXED_minus_FREE','FIXED','FREE','positive',0.,.0002),('FIXED_minus_FACT','FIXED','FACT','positive',0.,.0002),('FREE_minus_FACT_noninferiority','FREE','FACT','upper_below_margin',.0002,.00005)]
    require(len(ref['primary_family'])==3,'risk family size')
    for row,(name,left,right,direction_name,margin,gate) in zip(ref['primary_family'],definitions):
        v=np.array([observed[i,left,'TAIL']-observed[i,right,'TAIL'] for i in range(16)]);mean=math.fsum(v)/16;se=math.sqrt(math.fsum((v-mean)**2)/15/16);lo,hi=mean-crit*se,mean+crit*se
        require(row['name']==name and row['kind']=='TAIL' and row['expected']==direction_name and row['margin']==margin and row['SE_threshold']==gate,'TRAIN risk comparison')
        for key,value in [('stream_values',v),('mean',mean),('SE',se),('lower',lo),('upper',hi)]:near(value,row[key],why='TRAIN risk '+key)
        require(row['effect_passed']==(lo>0 if direction_name=='positive' else hi<margin) and row['precision_passed']==(se<=gate),'TRAIN risk gates')
    for row in ref['descriptive']:
        method,kind=row['method'],row['kind'];v=np.asarray([observed[i,method,kind] for i in range(16)]);r=np.asarray([expected[i,method,kind] for i in range(16)])
        for key,value in [('mean_joint_KL',v.mean()),('SE_joint_KL',v.std(ddof=1)/4),('mean_rho',r.mean(axis=0)),('min_rho',r.min(axis=0)),('max_rho',r.max(axis=0)),('endpoint_rho',r)]:near(value,row[key],why='risk descriptive '+key)
    qual=all(r['passed'] for r in ref['records']);require(ref['qualification_passed']==ref['primary_interpretation_qualified']==qual,'qualified risk interpretation')
    require(ref['primary_effects_passed']==sum(x['effect_passed'] for x in ref['primary_family']) and ref['primary_precision_passed']==sum(x['precision_passed'] for x in ref['primary_family']),'risk counts')
    count=len(unique);ledger=dict(unique_covariances=count,trained_outputs=96,identity_baselines=1,PMF_integrations=3*count,scalar_pattern_integrals=48*count,Gaussian_nodes=count*sum(i**4 for i in [24,32,40]),new_truth_integrals=0,new_random_draws=0,new_parameter_fits=0);require(ref['accounting']==ledger,'REFERENCE accounting')
    save(OUT/'REFERENCE_ARITHMETIC.json',dict(passed=True,source_sha256=sha(SOURCE),protocol_sha256=sha(OUT/'PROTOCOL.json'),reference_results_sha256=sha(BASE/'REFERENCE/RESULTS.json'),checked_outputs=96,checked_covariances=count,reference_qualified=qual,science_gate_counts=dict(effect=ref['primary_effects_passed'],precision=ref['primary_precision_passed'],combined=sum(x['effect_passed'] and x['precision_passed'] for x in ref['primary_family']),total=3),new_model_integrals=0,scope='All saved PMF/qualification/risk and primary arithmetic; independent9PMF integrals separately assigned.'))

def report():
    bind();names=['TRAIN_REPORT.json','DIRECTION_REPORT.json','REFERENCE_ARITHMETIC.json'];parts={name:read(OUT/name) for name in names};require(all(x['passed'] for x in parts.values()),'audit incomplete')
    save(OUT/'REPORT.json',dict(passed=True,source_sha256=sha(SOURCE),protocol_sha256=sha(OUT/'PROTOCOL.json'),phase_receipts={name:sha(OUT/name) for name in names},TRAIN_science_gate_counts=parts['REFERENCE_ARITHMETIC.json']['science_gate_counts'],DIRECTION_science_gate_counts=parts['DIRECTION_REPORT.json']['science_gate_counts'],new_independent_scientific_draws=0,new_model_integrals=0,scope='Complete original-input replay and saved-gradient/repair/statistical arithmetic;24 native states,8 fullraw gradients,128 direction banks only at bottom level. Independent reference integrals are another receipt.'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['check','freeze','run','direction','statistics','report']);args=parser.parse_args()
    if args.action=='check':print(json.dumps(artificial()),flush=True)
    elif args.action=='freeze':freeze()
    else:
        try:globals()[args.action]()
        except BaseException:
            fail=OUT/(args.action.upper()+'_FAILURE.json')
            if not fail.exists():save(fail,dict(utc=now(),traceback=traceback.format_exc(),automatic_retry=False))
            raise
