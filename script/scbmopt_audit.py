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
import argparse,hashlib,importlib.util,json,math,time,traceback
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import expit,logsumexp
from scipy.stats import t as student_t
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1];SOURCE=Path(__file__).resolve()
BASE=ROOT/'artifacts/scbm_scale_optimization';TRAIN=BASE/'TRAIN';OUT=BASE/'AUDIT'
DOC=ROOT/'docs/08_research_研究工作/可学习无关尺度与优化路径_20261003'
OFF=ROOT/'reproducibility_snapshots/cbm_public_sources/SCBM/58c3e602cb11fb4ea9106226e0f9e58f50d5d10a/models/losses.py'
MU=np.array([-.6,-.2,.2,.6]);C0=np.array([[1.,.25,-.15,.1],[.25,1.,.1,-.15],[-.15,.1,1.,.25],[.1,-.15,.25,1.]])
D=C0-np.eye(4);METHODS=('FIXED','FREE','FACT');STEPS=(0,1,31,127,511);NUMSTEPS=(0,31,127,511)
ROLES=dict(data_gaussian=0,data_active_uniform=1,data_noise_uniform=2,prior_active_gaussian=3,prior_noise_gaussian=4)
TOL=dict(api_loss=5e-5,api_t=3e-6,api_raw=5e-7,mathematical=1e-10,arithmetic=2e-11,
         full_parameter=1e-4,one_step_parameter=3e-6,moment_abs=2e-6,moment_rel=2e-5,tau_abs=2e-7,tau_rel=2e-7,maxlogit=2e-6)
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
        spec=importlib.util.spec_from_file_location('scbmopt_independent_official',OFF);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        _LOSS=m.SCBLoss(alpha=1.,config=SimpleNamespace(reg_precision='l1',reg_weight=1.,num_monte_carlo=100,training_mode='joint'))
    return _LOSS

def native(method,t,raw,e,nu,y,yn):
    tt=torch.tensor(float(t),dtype=torch.float32,requires_grad=True)
    rr=torch.tensor(raw,dtype=torch.float32,requires_grad=True) if method=='FREE' else None
    eye=torch.eye(4,dtype=torch.float32);dd=torch.tensor(C0,dtype=torch.float32)-eye
    lower=torch.linalg.cholesky(eye+(1+tt)*dd)
    x=(lower@torch.tensor(e,dtype=torch.float32).unsqueeze(-1)).squeeze(-1)+torch.tensor(MU,dtype=torch.float32)
    yy=y
    if method!='FACT':
        tau=F.softplus(rr)+1e-6 if rr is not None else torch.ones(256,dtype=torch.float32)
        x=torch.cat([x,torch.tensor(nu,dtype=torch.float32)*tau],dim=-1);yy=np.concatenate([y,yn],axis=1)
    p=torch.sigmoid(x.movedim(0,-1));loss=official().compute_concept_loss(p,torch.tensor(yy,dtype=torch.int64))
    if method=='FACT':loss=loss+256*math.log(2)
    g=torch.autograd.grad(loss,(tt,rr) if rr is not None else (tt,))
    return dict(loss=float(loss.detach()),t_gradient=float(g[0]),raw_gradient=None if rr is None else g[1].detach().numpy(),
                zero=int((p.detach()==0).sum()),one=int((p.detach()==1).sum()),evaluations=p.numel(),maxlogit=float(x.detach().abs().max()))

def math_score(method,t,raw,e,nu,y,yn):
    lower=np.linalg.cholesky(np.eye(4)+(1+float(t))*D)
    x=e@lower.T+MU;s=(2*y.astype(float)-1.)[None];signed=x*s
    logs=-np.logaddexp(0.,-signed).sum(axis=-1)
    sn=None;tau=None;rawfactor=None
    if method!='FACT':
        if method=='FREE':
            rr=np.asarray(raw,dtype=float);tau=np.where(rr>20,rr,np.logaddexp(0.,rr))+1e-6
            rawfactor=np.where(rr>20,1.,expit(rr))
        else:tau=np.ones(256)
        sn=nu*(2*yn.astype(float)-1.)[None];logs+=-np.logaddexp(0.,-sn*tau).sum(axis=-1)
    norm=logsumexp(logs,axis=0);weights=np.exp(logs-norm[None]);loss=float(np.mean(math.log(100)-norm))
    if method=='FACT':loss+=256*math.log(2)
    left=np.linalg.solve(lower,D);a=np.linalg.solve(lower,left.T).T
    phi=np.tril(a);phi[np.diag_indices(4)]*=.5;dl=lower@phi;dx=e@dl.T
    gt=float(-np.sum(weights[:,:,None]*s*expit(-signed)*dx)/len(y))
    gr=None if method!='FREE' else -np.sum(weights[:,:,None]*sn*expit(-sn*tau),axis=(0,1))/len(y)*rawfactor
    return dict(loss=loss,t_gradient=gt,raw_gradient=gr)

def adam(m,v,g,step):
    """Independent scalar/elementwise float32 Adam expression, no optimizer call."""
    g=np.asarray(g,np.float32)
    m=(m+np.float32(.1)*(g-m)).astype(np.float32)
    v=(v*np.float32(.999)+np.float32(.001)*g*g).astype(np.float32)
    den=(np.sqrt(v)/np.float32(math.sqrt(1-.999**step))+np.float32(1e-8)).astype(np.float32)
    delta=(np.float32(-.02/(1-.9**step))*m/den).astype(np.float32)
    return m,v,delta

def artificial():
    # Mathematical finite differences at fixed nonrandom arrays, n=256.
    e=np.sin(np.arange(100*3*4).reshape(100,3,4)*.13)
    nu=np.cos(np.arange(100*3*256).reshape(100,3,256)*.019)
    y=(np.arange(12).reshape(3,4)%3==0).astype(np.int8);yn=(np.arange(768).reshape(3,256)%5<2).astype(np.int8)
    raw=np.full(256,math.log(math.expm1(1-1e-6)));t=.11
    answer=math_score('FREE',t,raw,e,nu,y,yn);h=1e-5
    fd=(math_score('FREE',t+h,raw,e,nu,y,yn)['loss']-math_score('FREE',t-h,raw,e,nu,y,yn)['loss'])/(2*h)
    terr=near(fd,answer['t_gradient'],2e-8,'artificial t FD');rerr=0.
    for i in (0,127,255):
        plus=raw.copy();minus=raw.copy();plus[i]+=h;minus[i]-=h
        value=(math_score('FREE',t,plus,e,nu,y,yn)['loss']-math_score('FREE',t,minus,e,nu,y,yn)['loss'])/(2*h)
        rerr=max(rerr,near(value,answer['raw_gradient'][i],2e-8,'artificial raw FD'))
    # Prescribed gradients test arithmetic, not a sampled optimization problem.
    p=torch.nn.Parameter(torch.tensor(np.r_[np.zeros(3),np.full(256,raw[0])],dtype=torch.float32))
    opt=torch.optim.Adam([p],lr=.02,betas=(.9,.999),eps=1e-8,foreach=False,fused=False)
    candidate=p.detach().numpy().copy();m=np.zeros(259,np.float32);v=m.copy();maxerr=0.;localerr=0.
    for s0 in range(512):
        g=(.05*np.sin((np.arange(259)+1)*(s0+1)*.013)+.01*np.cos((s0+1)*.07)).astype(np.float32)
        old=p.detach().numpy().copy();p.grad=torch.tensor(g);opt.step()
        with torch.no_grad():p[:3].clamp_(-.75,.75)
        m,v,delta=adam(m,v,g,s0+1);candidate+=delta;candidate[:3]=np.clip(candidate[:3],-.75,.75)
        one=old+delta;one[:3]=np.clip(one[:3],-.75,.75)
        maxerr=max(maxerr,near(candidate,p.detach().numpy(),TOL['full_parameter'],'artificial Adam cumulative'))
        localerr=max(localerr,near(one,p.detach().numpy(),TOL['one_step_parameter'],'artificial Adam local'))
    for got,key in [(m,'exp_avg'),(v,'exp_avg_sq')]:near(got,opt.state[p][key].numpy(),TOL['moment_abs']+TOL['moment_rel']*abs(opt.state[p][key].numpy()),'artificial moments')
    return dict(passed=True,t_FD_error=terr,raw_FD_error=rerr,Adam_cumulative_error=maxerr,Adam_local_error=localerr,scientific_RNG=0,model_integrals=0)

def freeze():
    paths=[SOURCE,DOC/'AUDIT_PLAN.md',DOC/'EXPERIMENT_DESIGN.md',TRAIN/'FREEZE.json',ROOT/'script/scbmopt_run.py',ROOT/'script/scbmopt_kernel.py',ROOT/'script/scbmopt_reference.py',ROOT/'script/scbminv_kernel.py',ROOT/'script/scbminv_reference.py',OFF,ROOT/'artifacts/scbm_invariance/REFERENCE/MODEL.json']
    save(OUT/'PROTOCOL.json',dict(created_utc=now(),inputs={rel(p):sha(p) for p in paths},tolerances=TOL,steps=list(STEPS),methods=list(METHODS),deterministic_check=artificial(),numpy_version=np.__version__,torch_version=torch.__version__,new_model_integrals=0,automatic_retry=False))
    print(json.dumps(dict(source_sha256=sha(SOURCE),protocol_sha256=sha(OUT/'PROTOCOL.json'))),flush=True)
def bind():
    p=read(OUT/'PROTOCOL.json')
    for path,h in p['inputs'].items():require(sha(ROOT/path)==h,'audit input changed '+path)
    return p

def replay(stream):
    rng={name:np.random.Generator(np.random.PCG64(np.random.SeedSequence([202610030200,stream,r]))) for name,r in ROLES.items()}
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
        if step in STEPS:chosen[step]=(e.copy(),nu.copy())
    hashes.update({name:h.hexdigest() for name,h in hh.items()})
    data=dict(active_labels=y,nuisance_labels=yn,data_active_gaussian=z,data_active_uniform=u,first_prior_active=chosen[0][0],first_prior_nuisance=chosen[0][1])
    return data,chosen,hashes,first,initial,{name:r.bit_generator.state for name,r in rng.items()}

def trajectory(tape,record):
    t=np.asarray(tape['t']);raw=np.asarray(tape['free_raw']);tau=np.asarray(tape['free_tau'])
    require(t.shape==(3,513) and raw.shape==tau.shape==(513,256),'trajectory shape')
    require(np.array_equal(t[:,0],np.zeros(3,np.float32)),'initial t')
    initial_raw=np.float32((1-1e-6)+math.log(-math.expm1(-(1-1e-6))))
    require(np.array_equal(raw[0],np.full(256,initial_raw,np.float32)),'initial raw')
    require(np.array_equal(tape['batch_start'],np.arange(512)*64%4096),'cyclic fixed data indices')
    physical=np.where(raw.astype(float)>20,raw.astype(float),np.logaddexp(0.,raw.astype(float)))+1e-6
    tauerr=near(physical,tau,TOL['tau_abs']+TOL['tau_rel']*abs(tau),'all raw to tau')
    require(np.isfinite(t).all() and np.isfinite(raw).all() and np.all(t>=-.75) and np.all(t<=.75),'parameters finite and feasible')
    require(np.all(tau>=np.float32(1e-6)) and np.array_equal(tape['boundary'],abs(t)==.75),'boundary/floor')
    require(tape['feasible'].shape==(3,513) and tape['feasible'].all() and tape['finite'].shape==(3,512) and tape['finite'].all(),'stored feasibility')
    require(np.array_equal(t[:,1:],np.clip(tape['t_unprojected'],-.75,.75)),'stored t projection')
    require(np.array_equal(tape['projected'],tape['t_unprojected']!=t[:,1:]),'stored projection flag')
    gradients=np.concatenate([tape['t_gradient'].T,tape['free_raw_gradient']],axis=1).astype(np.float32)
    require(gradients.shape==(512,259) and np.isfinite(gradients).all(),'gradient tape')
    expected=np.r_[t[:,0],raw[0]].astype(np.float32);m=np.zeros(259,np.float32);v=m.copy();maxglobal=0.;maxlocal=0.
    for s in range(512):
        m,v,delta=adam(m,v,gradients[s],s+1)
        before=np.r_[t[:,s],raw[s]].astype(np.float32);local=(before+delta).astype(np.float32)
        near(local[:3],tape['t_unprojected'][:,s],TOL['one_step_parameter'],'independent unprojected t')
        local[:3]=np.clip(local[:3],-.75,.75);expected+=delta;expected[:3]=np.clip(expected[:3],-.75,.75)
        after=np.r_[t[:,s+1],raw[s+1]]
        maxlocal=max(maxlocal,near(local,after,TOL['one_step_parameter'],'all one-step Adam'))
        maxglobal=max(maxglobal,near(expected,after,TOL['full_parameter'],'all cumulative Adam'))
    final_m=np.r_[tape['final_t_adam_m'],tape['final_free_raw_adam_m']];final_v=np.r_[tape['final_t_adam_v'],tape['final_free_raw_adam_v']]
    merr=near(m,final_m,TOL['moment_abs']+TOL['moment_rel']*abs(final_m),'final first moments')
    verr=near(v,final_v,TOL['moment_abs']+TOL['moment_rel']*abs(final_v),'final second moments')
    for mi,method in enumerate(METHODS):
        for kind in ['TAIL','LAST']:
            tt=float(t[mi,449:].astype(float).mean()) if kind=='TAIL' else float(t[mi,512])
            scales=tau[449:].astype(float).mean(axis=0) if kind=='TAIL' else tau[512].astype(float)
            if method!='FREE':scales=np.full(256,1. if method=='FIXED' else 0.)
            near(tt,record['outputs'][method][kind]['t'],why='TAIL/LAST t')
            near(scales,record['outputs'][method][kind]['scales'],why='TAIL/LAST physical scales')
        diag=record['diagnostics'][method]
        for key,got in [('boundary_postupdate_count',int(tape['boundary'][mi,1:].sum())),('projected_count',int(tape['projected'][mi].sum())),('sigmoid_zero_count',int(tape['sigmoid_zero_count'][mi].sum())),('sigmoid_one_count',int(tape['sigmoid_one_count'][mi].sum())),('probability_evaluations',int(tape['probability_evaluations'][mi].sum()))]:require(diag[key]==got,'trajectory diagnostic '+key)
        require(diag['all_finite'] and diag['all_feasible'],'diagnostic finite')
        near(tape['maximum_absolute_logit'][mi].max(),diag['maximum_absolute_logit'],why='maxlogit summary')
        require(np.array_equal(tape['probability_evaluations'][mi],np.full(512,100*64*(4 if method=='FACT' else 260))),'all probability evaluation counts')
    return dict(tau_error=tauerr,Adam_cumulative_error=maxglobal,Adam_one_step_error=maxlocal,Adam_final_m_error=merr,Adam_final_v_error=verr)

def prefix_math(data,inputs,tape,record):
    errors=dict(API_loss=0.,API_t=0.,API_raw=0.,stable_loss=0.,stable_t=0.,stable_raw=0.)
    require({(r['preupdate_step'],r['method']) for r in record['numerical_controls']}=={(s,m) for s in NUMSTEPS for m in METHODS},'NUM scope')
    for step in STEPS:
        begin=64*step%4096;y=data['active_labels'][begin:begin+64];yn=data['nuisance_labels'][begin:begin+64];e,nu=inputs[step]
        for mi,method in enumerate(METHODS):
            tt=float(tape['t'][mi,step]);raw=tape['free_raw'][step] if method=='FREE' else None
            api=native(method,tt,raw,e,nu,y,yn);mat=math_score(method,tt,raw,e,nu,y,yn)
            errors['API_loss']=max(errors['API_loss'],near(api['loss'],tape['loss'][mi,step],TOL['api_loss'],'prefix original API loss'))
            errors['API_t']=max(errors['API_t'],near(api['t_gradient'],tape['t_gradient'][mi,step],TOL['api_t'],'prefix original API t gradient'))
            if method=='FREE':errors['API_raw']=max(errors['API_raw'],near(api['raw_gradient'],tape['free_raw_gradient'][step],TOL['api_raw'],'prefix original API raw gradient'))
            require(api['zero']==tape['sigmoid_zero_count'][mi,step] and api['one']==tape['sigmoid_one_count'][mi,step] and api['evaluations']==tape['probability_evaluations'][mi,step],'prefix probability counters')
            near(api['maxlogit'],tape['maximum_absolute_logit'][mi,step],TOL['maxlogit'],'prefix maxlogit')
            if step in NUMSTEPS:
                row=next(r for r in record['numerical_controls'] if r['preupdate_step']==step and r['method']==method)
                near(row['official_loss'],tape['loss'][mi,step],0.,'NUM tape loss');near(row['official_t_gradient'],tape['t_gradient'][mi,step],0.,'NUM tape t gradient')
                errors['stable_loss']=max(errors['stable_loss'],near(mat['loss'],row['stable64_loss'],TOL['mathematical'],'NumPy stable loss'))
                errors['stable_t']=max(errors['stable_t'],near(mat['t_gradient'],row['stable64_t_gradient'],TOL['mathematical'],'NumPy Cholesky t gradient'))
                le=row['official_loss']-row['stable64_loss'];te=row['official_t_gradient']-row['stable64_t_gradient']
                near(le,row['loss_difference'],why='NUM loss difference');near(te,row['t_gradient_difference'],why='NUM t difference')
                require(row['loss_passed']==(abs(le)<=1e-4) and row['t_gradient_passed']==(abs(te)<=2e-5),'NUM loss/t gate')
                if method=='FREE':
                    near(row['official_raw_gradient'],tape['free_raw_gradient'][step],0.,'NUM raw tape')
                    errors['stable_raw']=max(errors['stable_raw'],near(mat['raw_gradient'],row['stable64_raw_gradient'],TOL['mathematical'],'NumPy raw path gradient'))
                    diff=np.array(row['official_raw_gradient'])-np.array(row['stable64_raw_gradient']);mx=float(max(abs(diff)))
                    near(diff,row['raw_gradient_difference'],why='NUM raw differences');near(mx,row['maximum_raw_gradient_difference'],why='NUM raw maximum')
                    require(row['raw_gradient_passed']==(mx<=2e-6),'NUM raw gate')
                require(row['passed']==(row['loss_passed'] and row['t_gradient_passed'] and row.get('raw_gradient_passed',True)),'NUM joint gate')
            else:
                # This additional predetermined first-prefix step has no saved
                # stable64 row; compare mathematical/native within original NUM gates.
                near(api['loss'],mat['loss'],1e-4,'step1 native/math loss')
                near(api['t_gradient'],mat['t_gradient'],2e-5,'step1 native/math t')
                if method=='FREE':near(api['raw_gradient'],mat['raw_gradient'],2e-6,'step1 native/math raw')
    return errors

def num_summary(rows):
    return dict(checks=len(rows),passed=sum(r['passed'] for r in rows),loss_passed=sum(r['loss_passed'] for r in rows),t_gradient_passed=sum(r['t_gradient_passed'] for r in rows),raw_gradient_checks=sum('raw_gradient_passed' in r for r in rows),raw_gradient_passed=sum(r.get('raw_gradient_passed',False) for r in rows),maximum_loss_difference=max(abs(r['loss_difference']) for r in rows),maximum_t_gradient_difference=max(abs(r['t_gradient_difference']) for r in rows),maximum_raw_gradient_difference=max(r.get('maximum_raw_gradient_difference',0) for r in rows))

def run():
    bind();prod=read(TRAIN/'RESULTS.json');f=read(TRAIN/'FREEZE.json')
    require(prod['freeze_sha256']==sha(TRAIN/'FREEZE.json') and prod['source_sha256']==sha(ROOT/'script/scbmopt_run.py') and prod['config']==f['config'],'TRAIN bindings')
    require(prod['config']['steps']==512 and prod['config']['numerical_preupdate_steps']==list(NUMSTEPS),'TRAIN scope')
    save(OUT/'TRAIN_EXECUTION.json',dict(start_utc=now(),source_sha256=sha(SOURCE),protocol_sha256=sha(OUT/'PROTOCOL.json'),training_results_sha256=sha(TRAIN/'RESULTS.json'),automatic_retry=False))
    start=time.perf_counter();records=[]
    require(len(prod['records'])==16 and {r['stream'] for r in prod['records']}==set(range(16)),'streams')
    for stream in range(16):
        tick=time.perf_counter();rec=next(r for r in prod['records'] if r['stream']==stream)
        data,inputs,hashes,first,initial,final=replay(stream)
        require(rec['role_input_sha256']==hashes and rec['first_prior_sha256']==first,'full five-role byte hash')
        require(rec['RNG_initial_states']==initial and rec['RNG_final_states']==final,'initial/final PCG64 states')
        require(rec['role_seed_sequences']=={name:[202610030200,stream,r] for name,r in ROLES.items()},'role seed metadata')
        require(rec['label_sha256']==dict(active_labels=ah(data['active_labels']),nuisance_labels=ah(data['nuisance_labels'])),'full label hashes')
        require(sha(ROOT/rec['input_file'])==rec['input_sha256'] and sha(ROOT/rec['output_file'])==rec['output_sha256'],'input/trajectory bytes')
        require(rec['source_sha256']==prod['source_sha256'] and rec['freeze_sha256']==prod['freeze_sha256'] and rec['completed_steps']==512,'stream binding/steps')
        with np.load(ROOT/rec['input_file']) as a:
            require(set(a.files)==set(data),'input field scope')
            for name,x in data.items():require(np.array_equal(x,a[name]),'saved input '+name)
        with np.load(ROOT/rec['output_file']) as a:tape={name:a[name].copy() for name in a.files}
        updates=trajectory(tape,rec);errors=prefix_math(data,inputs,tape,rec)
        require(num_summary(rec['numerical_controls'])==rec['numerical_qualification'],'stream NUM summary')
        rr=dict(stream=stream,passed=True,complete_RNG_hash_states_labels=True,trajectory=updates,prefix=errors,seconds=time.perf_counter()-tick)
        records.append(rr);save(OUT/f'stream_{stream:02}.json',rr)
        print(json.dumps(dict(audit_stream=stream,passed=True,seconds=rr['seconds'],Adam_error=updates['Adam_cumulative_error'],prefix=errors)),flush=True)
    rows=[r for rec in prod['records'] for r in rec['numerical_controls']];require(num_summary(rows)==prod['numerical_qualification'],'all NUM summary')
    ledger=dict(independent_streams=16,new_complete_observations=65536,data_Gaussian_scalars=262144,data_Uniform_scalars=17039360,new_prior_vectors=52428800,prior_Gaussian_scalars=13631488000,prior_active_Gaussian_scalars=209715200,prior_noise_Gaussian_scalars=13421772800,observation_presentations=524288,controlled_optimization_programs=48,TAIL_outputs=48,LAST_outputs=48,official_training_API_calls=24576,stable64_control_calls=192,new_neural_network_fits=0,new_model_integrals=0)
    require(prod['ledger']==ledger,'production training ledger')
    save(OUT/'TRAIN_REPORT.json',dict(passed=True,source_sha256=sha(SOURCE),protocol_sha256=sha(OUT/'PROTOCOL.json'),training_results_sha256=sha(TRAIN/'RESULTS.json'),records=records,numerical_qualification=prod['numerical_qualification'],seconds=time.perf_counter()-start,
         accounting=dict(replayed_prior_Gaussian=13631488000,replayed_data_Gaussian=262144,replayed_data_Uniform=17039360,new_independent_scientific_draws=0,new_model_integrals=0,unique_step_inputs=80,official_API_calls=240,NumPy_mathematical_scores=240,NumPy_FREE_gradient_vectors=80,Adam_program_steps=24576),
         scope='Full input/state replay and all saved-gradient Adam arithmetic; bottom-level only steps0/1/31/127/511 per stream. Not a complete independent training rerun.'))

def statistics():
    bind();require(read(OUT/'TRAIN_REPORT.json')['passed'],'TRAIN audit incomplete')
    p=BASE/'REFERENCE/RESULTS.json';ref=read(p);train=read(TRAIN/'RESULTS.json')
    require(ref['source_sha256']==sha(ROOT/'script/scbmopt_reference.py') and ref['training_results_sha256']==sha(TRAIN/'RESULTS.json'),'risk reference bindings')
    require(ref['freeze_sha256']==sha(BASE/'REFERENCE/FREEZE.json'),'risk reference freeze binding')
    for path,h in read(BASE/'REFERENCE/FREEZE.json')['inputs'].items():require(sha(ROOT/path)==h,'risk frozen input '+path)
    save(OUT/'STATISTICS_EXECUTION.json',dict(start_utc=now(),source_sha256=sha(SOURCE),reference_results_sha256=sha(p),new_model_integrals=0))
    truthmodel=read(ROOT/'artifacts/scbm_invariance/REFERENCE/MODEL.json');truth=np.array(next(r['main_pmf'] for r in truthmodel['records'] if r['t']==0.))
    require(ref['truth_reference_sha256']==sha(ROOT/'artifacts/scbm_invariance/REFERENCE/MODEL.json'),'truth bytes')
    expected={}
    for r in train['records']:
        for method in METHODS:
            for kind in ['TAIL','LAST']:expected[r['stream'],method,kind]=r['outputs'][method][kind]['t']
    require(len(ref['endpoints'])==96 and len(ref['records'])==len({float(t).hex() for t in expected.values()}),'endpoint/dedup count')
    bykey={r['key']:r for r in ref['records']};risks={}
    for rr in ref['records']:
        require(rr['key']==float(rr['t']).hex(),'binary t key');main=np.array(rr['main_pmf'])
        require([v['order'] for v in rr['variants']]==[24,32,40],'fixed quadrature orders')
        require(np.array_equal(main,next(v['pmf'] for v in rr['variants'] if v['order']==32)),'main PMF')
        for v,q in zip(rr['variants'],rr['qualifications']):
            a=np.array(v['pmf']);delta=float(max(abs(a-main)));norm=abs(float(a.sum())-1.)
            near(delta,q['maximum_change'],why='PMF resolution');near(norm,q['normalization_error'],why='PMF normalization')
            passed=delta<=5e-8 and norm<=2e-12 and bool(np.all(a>0))
            require(q['passed']==passed and q['positive']==bool(np.all(a>0)),'PMF qualification gate')
        risk=math.fsum(float(x)*math.log(float(x/y)) for x,y in zip(truth,main));near(risk,rr['reference_joint_KL'],why='endpoint KL arithmetic')
        require(rr['passed']==all(x['passed'] for x in rr['qualifications']),'endpoint gate')
    for r in ref['endpoints']:
        ident=r['stream'],r['method'],r['kind'];require(r['t']==expected[ident] and r['key']==float(r['t']).hex(),'endpoint identity')
        near(r['reference_joint_KL'],bykey[r['key']]['reference_joint_KL'],why='endpoint risk lookup');risks[ident]=r['reference_joint_KL']
    critical=float(student_t.ppf(1-.05/6,15));near(critical,ref['critical_t'],why='risk t critical')
    expected_comps=[('FIXED_minus_FREE','FIXED','FREE','positive',0.,.0002),('FIXED_minus_FACT','FIXED','FACT','positive',0.,.0002),('FREE_minus_FACT_noninferiority','FREE','FACT','upper_below_margin',.0002,.00005)]
    require(len(ref['primary_family'])==3,'risk primary count')
    for row,(name,left,right,direction,margin,gate) in zip(ref['primary_family'],expected_comps):
        require(row['name']==name and row['kind']=='TAIL' and row['expected']==direction and row['margin']==margin and row['SE_threshold']==gate,'risk primary identity')
        x=np.array([risks[s,left,'TAIL']-risks[s,right,'TAIL'] for s in range(16)]);mean=float(math.fsum(x)/16);se=math.sqrt(math.fsum((x-mean)**2)/15/16);low,high=mean-critical*se,mean+critical*se
        for key,v in [('stream_values',x),('mean',mean),('SE',se),('lower',low),('upper',high)]:near(v,row[key],why='risk primary '+key)
        require(row['effect_passed']==(low>0 if direction=='positive' else high<margin) and row['precision_passed']==(se<=gate),'risk primary gate')
    require(len(ref['descriptive'])==6,'descriptive risk scope')
    for row in ref['descriptive']:
        x=np.array([risks[s,row['method'],row['kind']] for s in range(16)]);ts=[expected[s,row['method'],row['kind']] for s in range(16)]
        for key,v in [('mean_joint_KL',float(x.mean())),('SE_joint_KL',float(x.std(ddof=1)/4)),('mean_t',float(np.mean(ts))),('min_t',min(ts)),('max_t',max(ts)),('endpoint_t',ts)]:near(v,row[key],why='descriptive '+key)
    count=len(bykey);require(ref['accounting']==dict(unique_endpoints=count,final_outputs=96,PMF_integrations=3*count,scalar_pattern_integrals=48*count,Gaussian_nodes=count*sum(x**4 for x in [24,32,40]),new_truth_integrals=0,new_random_draws=0,new_parameter_fits=0),'reference accounting')
    qual=all(r['passed'] for r in ref['records']);require(ref['qualification_passed']==qual and ref['primary_interpretation_qualified']==qual,'reference qualification status')
    require(ref['primary_effects_passed']==sum(r['effect_passed'] for r in ref['primary_family']) and ref['primary_precision_passed']==sum(r['precision_passed'] for r in ref['primary_family']),'risk gate counts')
    save(OUT/'REPORT.json',dict(passed=True,source_sha256=sha(SOURCE),protocol_sha256=sha(OUT/'PROTOCOL.json'),training_audit_sha256=sha(OUT/'TRAIN_REPORT.json'),reference_results_sha256=sha(p),all96_outputs_and_risk_statistics_passed=True,science_gate_counts=dict(effects_passed=ref['primary_effects_passed'],precision_passed=ref['primary_precision_passed'],family_size=3),new_model_integrals=0,new_independent_scientific_draws=0,reference_integral_audit='Separately assigned; this phase checks stored PMF arithmetic, not new quadrature.'))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['check','freeze','run','statistics']);args=ap.parse_args()
    if args.action=='check':print(json.dumps(artificial()),flush=True)
    elif args.action=='freeze':freeze()
    else:
        try:globals()[args.action]()
        except BaseException:
            fail=OUT/(args.action.upper()+'_FAILURE.json')
            if not fail.exists():save(fail,dict(utc=now(),traceback=traceback.format_exc(),automatic_retry=False))
            raise
