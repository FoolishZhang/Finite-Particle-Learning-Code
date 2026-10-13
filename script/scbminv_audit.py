"""Frozen independent SCBM RUN audit; never imports producer kernel/run.

Only the original official loss module is loaded for source-identical API32
replay. Mathematical scores and Cholesky path derivatives are NumPy/SciPy.
No reference integrator or new independent scientific sampler is executed.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import numpy as np
from scipy.special import expit,logsumexp
from scipy.stats import t as student_t
import torch
torch.set_num_threads(1)

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/scbm_invariance'
RUN=BASE/'RUN';OUT=BASE/'AUDIT';SOURCE=Path(__file__).resolve()
DOC=ROOT/'docs/08_research_研究工作/真实联合评分接口与无关扩展_20261003'
OFF=ROOT/'reproducibility_snapshots/cbm_public_sources/SCBM/58c3e602cb11fb4ea9106226e0f9e58f50d5d10a/models/losses.py'
MU=np.array([-.6,-.2,.2,.6])
C0=np.array([[1.,.25,-.15,.10],[.25,1.,.10,-.15],[-.15,.10,1.,.25],[.10,-.15,.25,1.]])
D=C0-np.eye(4);PAIRS=[(i,j) for i in range(4) for j in range(i+1,4)]
DC=np.array([D[i,j] for i,j in PAIRS]);TS=(-.75,0.,.75);NS=(0,64,256)
ROLES=dict(data_gaussian=0,data_active_uniform=1,data_nuisance_uniform=2,prior_active=3,prior_nuisance=4)
TOLS=dict(api_score=5e-5,api_gradient=5e-6,math=1e-10,arithmetic=2e-11,diagnostic=2e-6,direction=5e-7)
_LOSSES={}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def require(ok,message):
    if not ok:raise RuntimeError(message)
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def now():return datetime.now(timezone.utc).isoformat()
def near(a,b,tol=TOLS['arithmetic'],name='numeric'):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    require(a.shape==b.shape,name+' shape');err=float(np.max(np.abs(a-b))) if a.size else 0.
    require(np.isfinite(a).all() and np.isfinite(b).all() and err<=tol,f'{name}: {err} > {tol}')
    return err
def normalized(g,n):
    g=np.array(g,dtype=float,copy=True)
    if n:g[...,10]/=n
    return g
def gradient_vector(row):return np.array(row['mu']+row['cov_offdiag']+[row['nuisance_scale']])

def official(K):
    if K not in _LOSSES:
        spec=importlib.util.spec_from_file_location('independent_scbm_official_loss',OFF)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        cfg=SimpleNamespace(num_monte_carlo=K,training_mode='joint',reg_precision='l1',reg_weight=1.)
        _LOSSES[K]=mod.SCBLoss(num_classes=2,alpha=1.,config=cfg)
    return _LOSSES[K]

def api32(e,nu,y,yn,t,n,a=1.,gradient=False):
    """Fresh independent graph, original public scoring method unchanged."""
    K,B=e.shape[:2];dtype=torch.float32
    delta_mu=torch.zeros(4,dtype=dtype,requires_grad=gradient)
    delta_cov=torch.zeros(6,dtype=dtype,requires_grad=gradient)
    aa=torch.tensor(a,dtype=dtype,requires_grad=gradient)
    base=torch.tensor(C0,dtype=dtype);cc=base+t*(base-torch.eye(4,dtype=dtype))
    hh=np.zeros((6,4,4))
    for q,(i,j) in enumerate(PAIRS):hh[q,i,j]=hh[q,j,i]=1.
    cc=cc+torch.sum(delta_cov[:,None,None]*torch.tensor(hh,dtype=dtype),dim=0)
    ll=torch.linalg.cholesky(cc)
    zz=torch.tensor(np.ascontiguousarray(e),dtype=dtype)
    xx=(ll@zz.unsqueeze(-1)).squeeze(-1)+torch.tensor(MU,dtype=dtype)+delta_mu
    yy=y
    if n:
        xx=torch.cat([xx,aa*torch.tensor(np.ascontiguousarray(nu[:,:,:n]),dtype=dtype)],dim=-1)
        yy=np.concatenate([y,yn[:,:n]],axis=1)
    logits=xx.permute(1,2,0);pp=torch.sigmoid(logits)
    value=official(K).compute_concept_loss(pp,torch.tensor(yy,dtype=torch.int64))
    diag=dict(sigmoid_zero_count=int((pp.detach()==0).sum()),sigmoid_one_count=int((pp.detach()==1).sum()),
              probability_evaluations=pp.numel(),maximum_absolute_logit=float(logits.detach().abs().max()),
              probability_min=float(pp.detach().min()),probability_max=float(pp.detach().max()),
              covariance_min_eigenvalue=float(torch.linalg.eigvalsh(cc.detach()).min()),loss_input_shape=list(logits.shape))
    g=None
    if gradient:
        gg=torch.autograd.grad(value,(delta_mu,delta_cov,aa),allow_unused=True)
        g=np.r_[gg[0].detach().numpy(),gg[1].detach().numpy(),0. if gg[2] is None else float(gg[2].detach())]
    return float(value.detach()),g,diag

def mathematical(e,nu,y,yn,t,n,a=1.,gradient=False,mu=None,cov=None):
    """Exact path formulas at supplied double arrays; no Price estimator."""
    K,B=e.shape[:2];cc=C0+t*D if cov is None else cov
    ll=np.linalg.cholesky(cc);xx=e@ll.T+(MU if mu is None else mu)
    signs=2*y.astype(float)-1.;signed=xx*signs[None]
    logs=-np.logaddexp(0.,-signed).sum(axis=2)
    ds=None
    if n:
        z=nu[:,:,:n]*(2*yn[:,:n].astype(float)-1.)[None]
        logs+=-np.logaddexp(0.,-a*z).sum(axis=2)
        if gradient:ds=np.sum(z*expit(-a*z),axis=2)
    norm=logsumexp(logs,axis=0);value=float(np.mean(math.log(K)-norm))
    if not gradient:return value,None
    weights=np.exp(logs-norm[None]);adj=-weights[:,:,None]*signs[None]*expit(-signed)/B
    gmu=adj.sum(axis=(0,1));gcov=[]
    for i,j in PAIRS:
        h=np.zeros((4,4));h[i,j]=h[j,i]=1.
        left=np.linalg.solve(ll,h);sym=np.linalg.solve(ll,left.T).T
        phi=np.tril(sym);phi[np.diag_indices(4)]*=.5
        dl=ll@phi;dx=e@dl.T;gcov.append(float(np.sum(adj*dx)))
    ga=0. if n==0 else float(-np.sum(weights*ds)/B)
    return value,np.r_[gmu,gcov,ga]

def selftest():
    e=np.sin(np.arange(7*5*4).reshape(7,5,4)*.13);nu=np.cos(np.arange(7*5*6).reshape(7,5,6)*.07)
    y=((np.arange(20).reshape(5,4)*3)%7<3).astype(np.int8);yn=(np.arange(30).reshape(5,6)%3==0).astype(np.int8)
    v,g=mathematical(e,nu,y,yn,0.,6,gradient=True);fd=[];step=1e-5
    for q in range(11):
        values=[]
        for sign in [-1,1]:
            mu=MU.copy();cov=C0.copy();aa=1.
            if q<4:mu[q]+=sign*step
            elif q<10:
                i,j=PAIRS[q-4];cov[i,j]+=sign*step;cov[j,i]+=sign*step
            else:aa+=sign*step
            values.append(mathematical(e,nu,y,yn,0.,6,aa,mu=mu,cov=cov)[0])
        fd.append((values[1]-values[0])/(2*step))
    err=near(fd,g,2e-9,'deterministic Cholesky path FD')
    value,gg,_=api32(e,nu,y,yn,0.,6,gradient=True)
    ae=near(normalized(gg,6),normalized(g,6),5e-6,'artificial API/math')
    plain=mathematical(e,nu,y,yn,0.,0)[0];zero=mathematical(e,nu,y,yn,0.,6,a=0.)[0]
    near(zero-plain,6*math.log(2),2e-14,'zero-scale factorization')
    return dict(passed=True,FD_error=err,API_math_gradient_error=ae,scientific_RNG=0,model_integrals=0)

def freeze():
    paths=[SOURCE,DOC/'AUDIT_PLAN.md',OFF,ROOT/'script/scbminv_kernel.py',ROOT/'script/scbminv_run.py',
           ROOT/'script/scbminv_reference.py',RUN/'FREEZE.json',RUN/'RESULTS.json',BASE/'REFERENCE/MODEL.json',
           DOC/'EXPERIMENT_DESIGN.md',DOC/'SOURCE_CONTRACT.md']
    save(OUT/'PROTOCOL.json',dict(created_utc=now(),inputs={rel(p):sha(p) for p in paths},tolerances=TOLS,
         selftest=selftest(),prefix_rows=64,Ks=[100,400],paddings=list(NS),candidates=list(TS),
         new_model_integrals=0,reference_audit_assigned_elsewhere=True,no_automatic_retry=True,
         torch_version=torch.__version__,numpy_version=np.__version__))
    print(json.dumps(dict(frozen=True,source_sha256=sha(SOURCE),protocol_sha256=sha(OUT/'PROTOCOL.json'))),flush=True)

def replay(stream):
    rng={name:np.random.Generator(np.random.PCG64(np.random.SeedSequence([202610030100,stream,role]))) for name,role in ROLES.items()}
    z=rng['data_gaussian'].standard_normal((4096,4));u=rng['data_active_uniform'].random((4096,4));un=rng['data_nuisance_uniform'].random((4096,256))
    y=(u<expit(z@np.linalg.cholesky(C0).T+MU)).astype(np.int8);yn=(un<.5).astype(np.int8)
    hashes={name:ah(a) for name,a in [('data_gaussian',z),('data_active_uniform',u),('data_nuisance_uniform',un),('active_labels',y),('nuisance_labels',yn)]}
    hh={name:hashlib.sha256() for name in ['prior_active','prior_nuisance']};first={};prefix={}
    for bi in range(64):
        k=400 if bi<16 else 100
        e=rng['prior_active'].standard_normal((k,64,4));nu=rng['prior_nuisance'].standard_normal((k,64,256))
        for name,a in [('prior_active',e),('prior_nuisance',nu)]:
            raw=a.tobytes();hh[name].update(raw)
            if bi==0:first[name]=hashlib.sha256(raw).hexdigest()
        if bi==0:prefix=dict(first_prior_active=e.copy(),first_prior_nuisance=nu.copy())
    return dict(data_active_gaussian=z,data_active_uniform=u,active_labels=y,nuisance_labels=yn,**prefix),hashes,{k:h.hexdigest() for k,h in hh.items()},first

def mean_fields(data,prefix,nb):
    keys=['raw_values','raw_gradient','raw_correlation_direction','factorized_active_values','factorized_active_gradient']
    s={k:np.mean(data[prefix+'_'+k][:nb],axis=0) for k in keys}
    v=s['raw_values'];fa=s['factorized_active_values']
    s.update(raw_low_minus_truth=v[:,0]-v[:,1],raw_high_minus_truth=v[:,2]-v[:,1],
         factorized_low_minus_truth=float(fa[0]-fa[1]),factorized_high_minus_truth=float(fa[2]-fa[1]),
         factorized_score_with_analytic_nlog2=fa[None]+np.array(NS)[:,None]*math.log(2),rows=64*nb)
    return s

def diagnostic(got,want):
    for key in ['sigmoid_zero_count','sigmoid_one_count','probability_evaluations','loss_input_shape']:require(got[key]==want[key],'API diagnostic '+key)
    for key in ['maximum_absolute_logit','probability_min','probability_max','covariance_min_eigenvalue']:near(got[key],want[key],TOLS['diagnostic'],'API diagnostic '+key)

def check_prefix(inputs,data,record):
    y=inputs['active_labels'][:64];yn=inputs['nuisance_labels'][:64]
    maxima=dict(api_score=0.,api_gradient=0.,math_score=0.,math_gradient=0.)
    require({(r['K'],r['padding'],r['t']) for r in record['numerical64']}=={(k,n,t) for k in (100,400) for n in NS for t in TS},'complete NUM scope')
    require({(r['K'],r['padding'],r['t'],r['scale']) for r in record['scale_controls']}=={(k,n,t,a) for k in (100,400) for n in (64,256) for t in TS for a in (0.,1e-6)},'complete control scope')
    for K in (100,400):
        e=inputs['first_prior_active'][:K];nu=inputs['first_prior_nuisance'][:K];cached={}
        for ni,n in enumerate(NS):
            for ti,t in enumerate(TS):
                score,g,diag=api32(e,nu,y,yn,t,n,gradient=t==0.);stable,sg=mathematical(e,nu,y,yn,t,n,gradient=t==0.)
                cached[n,t]=(score,g,diag)
                maxima['api_score']=max(maxima['api_score'],near(score,data[f'K{K}_raw_values'][0,ni,ti],TOLS['api_score'],'prefix API score'))
                for key,field in [('sigmoid_zero_count','sigmoid_zero_counts'),('sigmoid_one_count','sigmoid_one_counts'),('probability_evaluations','probability_evaluations')]:require(diag[key]==int(data[f'K{K}_{field}'][0,ni,ti]),'prefix API count')
                near(diag['maximum_absolute_logit'],data[f'K{K}_maximum_absolute_logits'][0,ni,ti],TOLS['diagnostic'],'prefix max logit')
                num=next(r for r in record['numerical64'] if r['K']==K and r['padding']==n and r['t']==t)
                near(score,num['official32_score'],TOLS['api_score'],'NUM API score')
                maxima['math_score']=max(maxima['math_score'],near(stable,num['stable64_score'],TOLS['math'],'NumPy stable score'))
                diff=num['official32_score']-num['stable64_score'];near(diff,num['score_difference'],name='NUM recorded score difference')
                require(num['score_pass']==(abs(diff)<=5e-4),'NUM score gate')
                if t==0.:
                    maxima['api_gradient']=max(maxima['api_gradient'],near(normalized(g,n),normalized(data[f'K{K}_raw_gradient'][0,ni],n),TOLS['api_gradient'],'prefix native gradient'))
                    og=gradient_vector(num['official32_gradient']);mg=gradient_vector(num['stable64_gradient'])
                    near(normalized(g,n),normalized(og,n),TOLS['api_gradient'],'NUM native gradient')
                    maxima['math_gradient']=max(maxima['math_gradient'],near(normalized(sg,n),normalized(mg,n),TOLS['math'],'NumPy Cholesky path gradient'))
                    near(float(DC@sg[4:10]),num['stable64_gradient']['correlation_direction'],TOLS['math'],'NumPy D direction')
                    ng=normalized(og,n)-normalized(mg,n)
                    near(ng,num['normalized_gradient_difference'],name='NUM normalized difference');mx=float(max(abs(ng)))
                    near(mx,num['maximum_normalized_gradient_difference'],name='NUM max gradient difference');require(num['gradient_pass']==(mx<=5e-5),'NUM gradient gate')
                    near(num['correlation_direction_difference'],num['official32_gradient']['correlation_direction']-num['stable64_gradient']['correlation_direction'],name='NUM D difference')
                    for row in [num['official32_gradient'],num['stable64_gradient']]:
                        require(row['covariance_pairs']==[list(p) for p in PAIRS],'gradient coordinate pairs')
                        near(row['nuisance_scale_per_coordinate'],row['nuisance_scale']/n if n else 0.,name='normalized nuisance field')
        for control in [r for r in record['scale_controls'] if r['K']==K]:
            n,t,a=control['padding'],control['t'],control['scale']
            require(n in (64,256) and a in (0.,1e-6) and t in TS,'control schema')
            value,g,diag=api32(e,nu,y,yn,t,n,a=a,gradient=t==0.)
            maxima['api_score']=max(maxima['api_score'],near(value,control['official32_score'],TOLS['api_score'],'control API score'))
            diagnostic(diag,control['diagnostics'])
            # Qualification arithmetic uses saved native values, independently
            # tied to native prefix above; do not silently redefine its gate.
            fact=float(data[f'K{K}_factorized_active_values'][0,TS.index(t)])+n*math.log(2)
            near(fact,control['factorized_score_from_active_API'],name='control factorized score')
            diff=control['official32_score']-fact;near(diff,control['value_difference'],name='control difference')
            require(control['value_qualified']==(abs(diff)<=5e-4),'control value gate')
            if t==0.:
                fg=data[f'K{K}_factorized_active_gradient'][0,:10]
                err=float(max(abs(g[:10]-fg)))
                near(err,control['maximum_active_gradient_difference'],TOLS['api_gradient'],'control active gradient')
                require(control['active_gradient_qualified']==(control['maximum_active_gradient_difference']<=5e-5),'control gradient gate')
                near(g[10],control['raw_scale_gradient_at_control'],TOLS['api_gradient']*n,'control nuisance derivative')
    require(len(record['numerical64'])==18 and len(record['scale_controls'])==24,'prefix NUM/control counts')
    return maxima

def stats(x):
    a=np.asarray(x,float);mean=float(math.fsum(a)/len(a));se=math.sqrt(math.fsum((a-mean)**2)/(len(a)-1)/len(a))
    return dict(mean=mean,SE=se,streams=len(a))
def check_stat(x,row):
    out=stats(x)
    for k,v in out.items():near(v,row[k],name='statistics '+k)
    return out

def all_statistics(summaries,production):
    crit=float(student_t.ppf(1-.05/8,15));near(crit,production['critical_t'],name='critical t')
    require(production['family_size']==4 and production['alpha']==.05 and production['df']==15,'family definition')
    require(len(production['primary_family'])==4,'primary family count')
    vals=[[],[],[],[]]
    for scopes in summaries:
        s=scopes['K100_all'];vals[0].append(s['raw_high_minus_truth'][0]);vals[1].append(s['raw_high_minus_truth'][2])
        vals[2].append(s['raw_correlation_direction'][2]-s['raw_correlation_direction'][0]);vals[3].append(s['raw_gradient'][2,10]/256)
    signs=['positive','negative','negative','positive'];thresholds=[5e-4,1e-3,2e-3,5e-4]
    names=['K100_base_high_minus_truth','K100_pad256_high_minus_truth','K100_padding_correlation_gradient_shift','K100_pad256_nuisance_scale_gradient_per_coordinate'];primary=[]
    for i,r in enumerate(production['primary_family']):
        require(r['name']==names[i] and r['expected_sign']==signs[i] and r['SE_threshold']==thresholds[i],'primary identity')
        near(vals[i],r['stream_values'],name='primary input values');st=check_stat(vals[i],r)
        lo,hi=st['mean']-crit*st['SE'],st['mean']+crit*st['SE'];near([lo,hi],[r['lower'],r['upper']],name='primary interval')
        effect=lo>0 if signs[i]=='positive' else hi<0
        require(r['effect_passed']==effect and r['precision_passed']==(st['SE']<=thresholds[i]),'primary gates')
        primary.append(dict(name=names[i],**st,lower=lo,upper=hi,effect_passed=bool(effect),precision_passed=st['SE']<=thresholds[i]))
    require(len(production['descriptive_score_tables'])==27 and len(production['descriptive_gradient_tables'])==9,'table counts')
    for r in production['descriptive_score_tables']:
        ni,ti=NS.index(r['padding']),TS.index(r['t']);s=[x[r['scope']] for x in summaries]
        require(r['rows_per_stream']==s[0]['rows'],'score table rows')
        for key,values in [('raw_score',[x['raw_values'][ni,ti] for x in s]),('raw_contrast_to_truth',[x['raw_values'][ni,ti]-x['raw_values'][ni,1] for x in s]),('factorized_score',[x['factorized_score_with_analytic_nlog2'][ni,ti] for x in s]),('factorized_contrast_to_truth',[x['factorized_active_values'][ti]-x['factorized_active_values'][1] for x in s])]:check_stat(values,r[key])
    for r in production['descriptive_gradient_tables']:
        n=r['padding'];ni=NS.index(n);s=[x[r['scope']] for x in summaries];g=np.array([x['raw_gradient'][ni] for x in s])
        for j,row in enumerate(r['mu_components']+r['cov_components']):check_stat(g[:,j],row)
        check_stat(g[:,10],r['nuisance_scale']);check_stat(g[:,10]/n if n else np.zeros(16),r['nuisance_scale_per_coordinate'])
        check_stat([x['raw_correlation_direction'][ni] for x in s],r['correlation_direction'])
    num=[dict(stream=r['stream'],**x) for r in production['records'] for x in r['numerical64']]
    controls=[dict(stream=r['stream'],**x) for r in production['records'] for x in r['scale_controls']]
    require(num==production['numerical64'] and controls==production['scale_controls'],'full NUM/control aggregation')
    nq=dict(score_count=len(num),score_passed=sum(r['score_pass'] for r in num),gradient_count=sum('gradient_pass' in r for r in num),gradient_passed=sum(r.get('gradient_pass',False) for r in num),maximum_score_difference=max(abs(r['score_difference']) for r in num),maximum_normalized_gradient_difference=max(r.get('maximum_normalized_gradient_difference',0) for r in num))
    cq=dict(value_count=len(controls),value_passed=sum(r['value_qualified'] for r in controls),active_gradient_count=sum('active_gradient_qualified' in r for r in controls),active_gradient_passed=sum(r.get('active_gradient_qualified',False) for r in controls))
    require(nq==production['numerical_qualification'] and cq==production['scale_control_qualification'],'NUM/control qualification counts')
    return dict(primary=primary,numerical_qualification=nq,scale_control_qualification=cq,all_descriptive_tables_passed=True)

def run():
    protocol=read(OUT/'PROTOCOL.json')
    for path,h in protocol['inputs'].items():require(sha(ROOT/path)==h,'Frozen audit input changed '+path)
    prod=read(RUN/'RESULTS.json');frozen=read(RUN/'FREEZE.json');ref=read(BASE/'REFERENCE/MODEL.json')
    require(prod['freeze_sha256']==sha(RUN/'FREEZE.json') and prod['reference_sha256']==sha(BASE/'REFERENCE/MODEL.json'),'production bindings')
    require(prod['source_sha256']==sha(ROOT/'script/scbminv_run.py')==frozen['source_sha256'] and prod['kernel_sha256']==sha(ROOT/'script/scbminv_kernel.py')==frozen['kernel_sha256'],'producer source binding')
    require(prod['config']==frozen['config'] and prod['fixture']==frozen['fixture'],'production configuration binding')
    require(prod['fixture']['mu']==MU.tolist() and prod['fixture']['C0']==C0.tolist(),'fixture')
    for path,h in frozen['official_inputs'].items():require(sha(ROOT/path)==h,'official source changed')
    require(sha(ROOT/frozen['design_path'])==frozen['design_sha256'],'design binding')
    require(str(torch.__version__)==prod['official']['torch_version'],'Torch runtime version')
    save(OUT/'EXECUTION.json',dict(start_utc=now(),source_sha256=sha(SOURCE),protocol_sha256=sha(OUT/'PROTOCOL.json'),production_results_sha256=sha(RUN/'RESULTS.json'),automatic_retry=False))
    started=time.perf_counter();records=[];summaries=[]
    require(len(prod['records'])==16 and {r['stream'] for r in prod['records']}==set(range(16)),'stream IDs')
    for stream in range(16):
        tick=time.perf_counter();rec=next(r for r in prod['records'] if r['stream']==stream)
        require(rec['source_sha256']==prod['source_sha256'],'stream source binding')
        inputs,hashes,priors,first=replay(stream)
        require(hashes==rec['data_hashes'] and priors==rec['prior_hashes'] and first==rec['first_prior_batch_hashes'],'complete 5-role RNG replay')
        require(rec['role_seed_sequences']=={name:[202610030100,stream,role] for name,role in ROLES.items()},'role seed metadata')
        require(sha(ROOT/rec['input_file'])==rec['input_sha256'] and sha(ROOT/rec['output_file'])==rec['output_sha256'],'NPZ file bytes')
        with np.load(ROOT/rec['input_file']) as saved:
            require(set(saved.files)==set(inputs),'input schema')
            for k,a in inputs.items():require(np.array_equal(a,saved[k]),'full input '+k)
        with np.load(ROOT/rec['output_file']) as saved:data={k:saved[k].copy() for k in saved.files}
        fields=['raw_values','raw_gradient','raw_correlation_direction','factorized_active_values','factorized_active_gradient','sigmoid_zero_counts','sigmoid_one_counts','probability_evaluations','maximum_absolute_logits']
        require(set(data)=={f'K{k}_{field}' for k in (100,400) for field in fields},'output schema')
        for K,nb in [(100,64),(400,16)]:
            shapes=dict(raw_values=(nb,3,3),raw_gradient=(nb,3,11),raw_correlation_direction=(nb,3),factorized_active_values=(nb,3),factorized_active_gradient=(nb,11),sigmoid_zero_counts=(nb,3,3),sigmoid_one_counts=(nb,3,3),probability_evaluations=(nb,3,3),maximum_absolute_logits=(nb,3,3))
            for field,shape in shapes.items():require(data[f'K{K}_{field}'].shape==shape and np.isfinite(data[f'K{K}_{field}']).all(),'output shape/finite '+field)
            require(np.array_equal(data[f'K{K}_factorized_active_values'],data[f'K{K}_raw_values'][:,0]),'all factorized values')
            require(np.array_equal(data[f'K{K}_factorized_active_gradient'],data[f'K{K}_raw_gradient'][:,0]),'all factorized gradients')
            direction=np.sum(data[f'K{K}_raw_gradient'][:,:,4:10].astype(np.float32)*DC.astype(np.float32),axis=2,dtype=np.float32)
            near(direction,data[f'K{K}_raw_correlation_direction'],TOLS['direction'],'all D direction arithmetic')
            pd=rec['probability_diagnostics'][str(K)]
            require(pd['scope']==('all4096' if K==100 else 'prefix1024'),'probability scope')
            for field in ['sigmoid_zero_counts','sigmoid_one_counts','probability_evaluations']:
                a=data[f'K{K}_{field}'];require(np.issubdtype(a.dtype,np.integer) and np.all(a>=0),'nonnegative count')
                require(np.array_equal(a.sum(axis=0),pd[field]),'all probability diagnostic sums')
            expected=np.broadcast_to(np.array([K*64*(n+4) for n in NS])[None,:,None],(nb,3,3))
            require(np.array_equal(expected,data[f'K{K}_probability_evaluations']),'all probability counts')
            near(data[f'K{K}_maximum_absolute_logits'].max(axis=0),pd['maximum_absolute_logits'],name='all maximum logits')
        independent={name:mean_fields(data,pref,nb) for name,pref,nb in [('K100_all','K100',64),('K100_prefix1024','K100',16),('K400_prefix1024','K400',16)]}
        for name,s in independent.items():
            for key,value in s.items():near(value,rec['summaries'][name][key],name='all summary '+name+'/'+key)
        y=inputs['active_labels'];codes=y@np.array([8,4,2,1],dtype=np.int64)
        for name,N in [('all',4096),('prefix1024',1024)]:
            counts=np.bincount(codes[:N],minlength=16);require(counts.tolist()==rec['pattern_counts'][name],'data pattern counts')
            for row,rr in zip(rec['numerical_reference_scores'][name],ref['records']):
                require(row['t']==rr['t'],'reference candidate')
                nll=-math.fsum(float(c)*math.log(p) for c,p in zip(counts,rr['main_pmf']))/N
                near(nll,row['numerical_reference_NLL'],name='reference empirical NLL')
        maxima=check_prefix(inputs,data,rec)
        require(rec['official_API_calls']==984 and rec['stable64_control_calls']==18,'per stream call count')
        result=dict(stream=stream,passed=True,all_five_RNG_roles_and_labels_exact=True,first_prior_batch_exact=True,
                    maximum_prefix_errors=maxima,all_saved_summaries_and_diagnostics_passed=True,seconds=time.perf_counter()-tick)
        records.append(result);summaries.append(independent);save(OUT/f'stream_{stream:02}.json',result)
        print(json.dumps(dict(audit_stream=stream,passed=True,seconds=result['seconds'],errors=maxima)),flush=True)
    analysis=all_statistics(summaries,prod)
    ledger=dict(independent_streams=16,new_complete_observations=65536,main_prior_K=100,auxiliary_K=400,auxiliary_observations_per_stream=1024,prior_vectors=11468800,prior_Gaussian_scalars=2981888000,prior_active_Gaussian_scalars=45875200,prior_nuisance_Gaussian_scalars=2936012800,data_Gaussian_scalars=262144,data_active_uniforms=262144,data_nuisance_uniforms=16777216,data_uniforms=17039360,new_neural_network_fits=0,new_parameter_optimizations=0,official_API_calls=15744,stable64_control_calls=288)
    require(prod['ledger']==ledger,'production ledger')
    truth=next(r['main_pmf'] for r in ref['records'] if r['t']==0.)
    for r in ref['records']:
        near(sum(truth[i]*math.log(truth[i]/r['main_pmf'][i]) for i in range(16)),r['reference_joint_KL'],name='reference KL arithmetic')
    save(OUT/'REPORT.json',dict(passed=True,source_sha256=sha(SOURCE),protocol_sha256=sha(OUT/'PROTOCOL.json'),production_results_sha256=sha(RUN/'RESULTS.json'),records=records,statistics=analysis,
          scope='All data/prior RNG replay; only first64 observations/stream at K100/400 bottom-level API/math reconstruction; all1280 saved block arithmetic/statistics. Not complete production API rerun.',
          accounting=dict(replayed_prior_Gaussian_scalars=2981888000,replayed_data_Gaussian_scalars=262144,replayed_data_Uniform_scalars=17039360,new_independent_scientific_draws=0,new_model_integrals=0,official_prefix_API_calls=672,NumPy_prefix_mathematical_scores=288,NumPy_prefix_gradients=96,checked_NUM_scores=288,checked_NUM_gradients=96,checked_scale_values=384,checked_scale_active_gradients=128,saved_blocks=1280),
          reference_integral_audit='Assigned to independent spectral agent; no new integrals in this script',seconds=time.perf_counter()-started))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['check','freeze','run']);a=p.parse_args()
    if a.action=='check':print(json.dumps(selftest()),flush=True)
    elif a.action=='freeze':freeze()
    else:
        try:run()
        except BaseException:
            fp=OUT/'FAILURE.json'
            if not fp.exists():save(fp,dict(utc=now(),traceback=traceback.format_exc(),automatic_retry=False))
            raise
