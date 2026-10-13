"""REC-v4: fresh active-only learning with 400 inside versus 4 x 100 outside log."""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_key]='1'
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,hashlib,json,math,subprocess,time,traceback
import numpy as np
import torch
from scipy.special import expit,roots_hermitenorm
from scipy.optimize import brentq
from scipy.stats import t
import jointscale_kernel as kernel
import jointscale_compute as reference

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/rec_learning_v4_20261005';DOC=ROOT/'docs/08_research_研究工作/REC_v4_learning_20261005'
BASE='f35a50628d43eab458bcb23a0660adf2986e2ec1'
CASES=('NULL','MIXED');ARMS=('OUTER100x4','INNER400')
CONFIG=dict(seed=202610054010,reps=32,cases=list(CASES),observations=8192,steps=512,batch=64,total_prior=400,
    K=[100,400],blocks=4,lr=.01,tail=64,dtype='float64 stable math',initial_s=1.,initial_C='I',
    orders=[32,40,48],main_order=40,pmf_tolerance=2e-7,risk_tolerance=2e-6,mass_tolerance=2e-12,
    audit_reps=[0,16],audit_steps=[0,255,511],audit_reference_reps=[0,8,16,24],audit_order=44,
    alpha=.05,family=6,df=31,SE_limit=.0002,practical_epsilon=.0001,workers=4,threads=1)

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rel(p):return Path(p).relative_to(ROOT).as_posix()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def ah(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def default(x):
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,np.ndarray):return x.tolist()
    raise TypeError(type(x).__name__)
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);s=json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=default)+'\n'
    with p.open('x',encoding='utf-8',newline='\n') as f:f.write(s)
def vector(model):return np.concatenate([p.detach().numpy().ravel() for p in model.parameters()])
def adam_vector(model,opt,key):
    return np.concatenate([opt.state.get(p,{}).get(key,torch.zeros_like(p)).detach().numpy().ravel() for p in model.parameters()])
def model_from(v,K):
    model=kernel.ScoreProgram('ZERO',v[:4],v[4:8],K=K,n=1,tau0=0.,path='math64')
    model.load_physical_state(dict(mu=list(v[:4]),s=list(v[4:8]),rho=list(v[8:14]),tau=0.))
    return model
def loss(model,eps,labels,outer):
    blocks=[eps[k:k+100] for k in range(0,400,100)] if outer else [eps]
    values=[]
    for e in blocks:
        batch=kernel.prepare_batch(e,np.zeros((len(e),len(labels),1)),labels,np.zeros((len(labels),1),dtype=int),path='math64')
        values.append(model(batch)-math.log(2))
    return torch.stack(values).mean()
def initial_mu(p):
    x,w=roots_hermitenorm(64);w=w/math.sqrt(2*math.pi)
    q=lambda mu:float(w@expit(mu+x))
    lo,hi=q(-2),q(2);targets=np.clip(p,lo,hi)
    values=[brentq(lambda mu:q(mu)-target,-2,2,xtol=2e-14) for target in targets]
    return np.array(values),dict(clipped=(targets!=p).tolist(),maximum_marginal_error=float(np.max(abs(np.array([q(m) for m in values])-p))))
def tail(vectors):
    mu=vectors[:,:4].mean(axis=0);cov=[]
    for v in vectors:
        C=kernel.correlation_numpy(v[8:14]);cov.append(v[4:8,None]*C*v[None,4:8])
    cov=np.mean(cov,axis=0);s=np.sqrt(np.diag(cov));C=cov/np.outer(s,s)
    return dict(mu=mu.tolist(),covariance=cov.tolist(),s=s.tolist(),correlation=C.tolist(),rho=[float(C[i,j]) for i,j in kernel.PAIRS])

def stream_run(ci,rep):
    folder=OUT/CASES[ci]/f'rep_{rep:02}';folder.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    try:
        rng={r:np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([CONFIG['seed'],ci,rep,r]))) for r in range(3)}
        z=rng[0].standard_normal((8192,4));u=rng[1].random((8192,4));C=np.eye(4) if ci==0 else reference.DATA_C
        labels=(u<expit(z@np.linalg.cholesky(C).T+reference.DATA_MU)).astype(np.int64)
        p=(labels.sum(axis=0)+.5)/8193.;mu,init=initial_mu(p)
        models=[kernel.ScoreProgram('ZERO',mu,1.,K=k,n=1,tau0=0.,path='math64') for k in (100,400)]
        opts=[kernel.make_optimizer(m,.01) for m in models]
        tape={key:np.empty((2,513,14)) for key in ('params','adam_m','adam_v')}
        tape.update({key:np.empty((2,512,14)) for key in ('gradient','proposal')})
        tape['loss']=np.empty((2,512));tape['alpha']=np.empty((2,512))
        for j,m in enumerate(models):tape['params'][j,0]=vector(m);tape['adam_m'][j,0]=0;tape['adam_v'][j,0]=0
        hashes=[];audit_prior={};arm_seconds=np.zeros(2);rng_seconds=0.
        for step in range(512):
            tick=time.perf_counter();eps=rng[2].standard_normal((400,64,4));rng_seconds+=time.perf_counter()-tick
            hashes.append(ah(eps));begin=(step*64)%8192;batch=labels[begin:begin+64]
            if rep in (0,16) and step in (0,255,511):audit_prior[f'prior_{step}']=eps.copy()
            for j,(m,opt) in enumerate(zip(models,opts)):
                tick=time.perf_counter();opt.zero_grad();value=loss(m,eps,batch,j==0);value.backward()
                g=np.concatenate([v.grad.detach().numpy().ravel() for v in m.parameters()]);assert np.isfinite(g).all()
                tape['loss'][j,step]=float(value.detach());tape['gradient'][j,step]=g
                opt.step();tape['proposal'][j,step]=vector(m);repair=m.restore_feasibility();tape['alpha'][j,step]=repair['projection_alpha']
                tape['params'][j,step+1]=vector(m);tape['adam_m'][j,step+1]=adam_vector(m,opt,'exp_avg');tape['adam_v'][j,step+1]=adam_vector(m,opt,'exp_avg_sq')
                arm_seconds[j]+=time.perf_counter()-tick
        arrays=folder/'TRACE.npz';np.savez_compressed(arrays,labels=labels.astype(np.uint8),**tape,**audit_prior)
        P,teacher_p,product=reference.teacher();P=product if ci==0 else P
        patterns=reference.PATTERNS;ind=np.prod(np.where(patterns,p,1-p),axis=1);ind_risk=float(P@np.log(P/ind));rows=[]
        for j,arm in enumerate(ARMS):
            state=tail(tape['params'][j,-64:]);variants=[reference.pmf(np.array(state['mu']),np.array(state['covariance']),order) for order in (32,40,48)]
            q=np.array(variants[1]['pmf']);risk=float(P@np.log(P/q));checks=[]
            for v in variants:
                err=float(np.max(abs(np.array(v['pmf'])-q)));re=abs(float(P@np.log(P/np.array(v['pmf'])))-risk)
                checks.append(dict(order=v['order'],max_PMF_error=err,risk_error=re,mass_error=v['mass_error'],passed=bool(err<=2e-7 and re<=2e-6 and v['mass_error']<=2e-12 and min(v['pmf'])>0)))
            marginal=q@patterns;own=np.prod(np.where(patterns,marginal,1-marginal),axis=1)
            rows.append(dict(arm=arm,model=state,main_pmf=q.tolist(),risk=risk,versus_independent=risk-ind_risk,
                own_product_risk=float(P@np.log(P/own)),dependence_gain=float(P@np.log(P/own))-risk,
                variants=variants,qualification=checks,qualified=all(x['passed'] for x in checks)))
        result=dict(case=CASES[ci],rep=rep,trace_file=rel(arrays),trace_sha256=sha(arrays),data_hashes=dict(Gaussian=ah(z),Uniform=ah(u),labels=ah(labels.astype(np.uint8))),
            data_counts=labels.sum(axis=0).tolist(),initial_mu=mu.tolist(),initialization=init,independent_p=p.tolist(),independent_risk=ind_risk,
            teacher_P=P.tolist(),prior_hashes=hashes,rows=rows,qualified=all(r['qualified'] for r in rows),
            arm_seconds=arm_seconds.tolist(),prior_generation_seconds=rng_seconds,seconds=time.perf_counter()-start,
            source_sha256=sha(__file__),freeze_sha256=sha(OUT/'FREEZE.json'))
        save(folder/'RESULT.json',result);return result
    except Exception:save(folder/'ERROR.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise

def summarize(records):
    family=[];means=[];crit=float(t.ppf(1-.05/12,31))
    for case in CASES:
        rows=[r for r in records if r['case']==case];risk=np.array([[a['risk'] for a in r['rows']] for r in rows]);ind=np.array([r['independent_risk'] for r in rows])
        for name,values in [('INNER-OUTER',risk[:,1]-risk[:,0]),('INNER-IND',risk[:,1]-ind),('OUTER-IND',risk[:,0]-ind)]:
            mean=float(values.mean());se=float(values.std(ddof=1)/math.sqrt(32));ci=[mean-crit*se,mean+crit*se]
            family.append(dict(case=case,contrast=name,paired_values=values.tolist(),mean=mean,SE=se,interval=ci,
                precision_passed=se<=.0002,improvement=bool(ci[1]<0),material_improvement=bool(ci[1]<0 and mean<=-.0001)))
        means.append(dict(case=case,OUTER=float(risk[:,0].mean()),INNER=float(risk[:,1].mean()),IND=float(ind.mean()),
            dependence_gain=np.mean([[a['dependence_gain'] for a in r['rows']] for r in rows],axis=0).tolist()))
    return dict(config=CONFIG,freeze_sha256=sha(OUT/'FREEZE.json'),records=records,family=family,means=means,qualified=all(r['qualified'] for r in records),
        accounting=dict(new_observations=524288,new_independent_fits=64,optimizers=128,updates=65536,prior_vectors=838860800,prior_Gaussian=3355443200,
            data_Gaussian=2097152,data_Uniform=2097152,score_forwards=163840,reference_nodes=1141374976,audit_reference_nodes=29984768),
        scope='Two fixed generating laws, new paired data/prior streams; active-only math64 finite programs, not full raw SCBM or isolated noise-free target causality')

def freeze():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==BASE
    q=read(OUT/'QUALIFICATION.json');assert q['passed'] and q['producer_sha256']==sha(__file__) and q['audit_sha256']==sha(ROOT/'script/reclearn_audit.py')
    sources=[Path(__file__),ROOT/'script/reclearn_audit.py',Path(kernel.__file__),Path(reference.__file__),DOC/'CONTRACT.md']
    inputs=[reference.TRUTH,ROOT/'artifacts/use_selection_v2_20261005/HANDOFF_INPUT.md']
    save(OUT/'FREEZE.json',dict(authorized=True,created_utc=datetime.now(timezone.utc).isoformat(),baseline=BASE,config=CONFIG,
        sources={rel(p):sha(p) for p in sources},inputs={rel(p):sha(p) for p in inputs},qualification_sha256=sha(OUT/'QUALIFICATION.json'),environment=q['environment']))
    print(json.dumps(dict(frozen=True,sha256=sha(OUT/'FREEZE.json'))))

def validate():
    f=read(OUT/'FREEZE.json');assert f['config']==CONFIG and f['authorized']
    for group in ('sources','inputs'):
        for p,h in f[group].items():assert sha(ROOT/p)==h,p
    assert sha(OUT/'QUALIFICATION.json')==f['qualification_sha256']

def run():
    validate();tick=time.perf_counter();save(OUT/'EXECUTION.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),automatic_retry=False))
    try:
        rows=[]
        with ProcessPoolExecutor(max_workers=4) as pool:
            for future in as_completed([pool.submit(stream_run,ci,rep) for ci in range(2) for rep in range(32)]):
                r=future.result();rows.append(r);print(json.dumps(dict(case=r['case'],rep=r['rep'],qualified=r['qualified'],seconds=r['seconds'])),flush=True)
        rows.sort(key=lambda r:(CASES.index(r['case']),r['rep']));validate();result=summarize(rows);result['seconds']=time.perf_counter()-tick
        save(OUT/'RESULTS.json',result);print(json.dumps(dict(completed=True,qualified=result['qualified'],seconds=result['seconds'])),flush=True)
    except Exception:save(OUT/'ERROR.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('freeze','run'));globals()[parser.parse_args().command]()
