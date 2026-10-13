"""USE-v3: latent-trait selection on fixed UCI writer-disjoint train/test files."""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_key]='1'
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import argparse,hashlib,json,math,platform,subprocess,time,traceback
import numpy as np
import scipy
from scipy.special import expit,logsumexp,roots_hermitenorm
from scipy.optimize import minimize
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/use_digits_v3_20261005';DOC=ROOT/'docs/08_research_研究工作/USE_v3_20261005'
BASE='f35a50628d43eab458bcb23a0660adf2986e2ec1'
NAMES=('IND','L001','L010','L100');METHODS=('RAW100','OUT16','INNER1600','JVI100','AGH')
CONFIG=dict(seed=202610052710,reps=32,dimension=64,latent=2,fit_rows=3058,val_rows=765,test_rows=1797,
    threshold=8,train_order=24,lambdas=[.001,.01,.1],starts=2,bias_bounds=[-10.,10.],loading_bounds=[-3.,3.],
    optimizer=dict(maxiter=120,maxfun=260,maxls=20,ftol=1e-10,gtol=1e-6),hard_calls=300,
    reference_orders=[12,20,32,48,64],base_reference=32,mode_tol=1e-8,mode_maxiter=60,mode_maxbacktrack=20,
    reference_tolerance=1e-5,coarse_reference_tolerance=1e-4,audit_reps=[0,8,16,24],audit_orders=[64,80],
    audit_profiles=64,audit_tolerance=1e-4,K=100,total_prior=1600,alpha=.05,family=5,df=31,
    SE_limit=.025,practical_epsilon=.05,workers=4,threads=1)

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rel(p):return Path(p).relative_to(ROOT).as_posix()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def encoder(x):
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,np.ndarray):return x.tolist()
    raise TypeError(type(x).__name__)
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=encoder)+'\n'
    with p.open('x',encoding='utf-8',newline='\n') as f:f.write(text)

@lru_cache(None)
def rule(order):
    x,w=roots_hermitenorm(order);lw=np.log(w)-.5*math.log(2*math.pi)
    nodes=np.stack(np.meshgrid(x,x,indexing='ij'),axis=-1).reshape(-1,2)
    return nodes,(lw[:,None]+lw[None,:]).ravel()

def unpack(theta,d):return np.asarray(theta[:d]),np.asarray(theta[d:]).reshape(d,2)

def objective(theta,profiles,frequency,penalty,order=24):
    b,W=unpack(theta,profiles.shape[1]);z,lw=rule(order);eta=z@W.T+b
    loga=eta@profiles.T-np.logaddexp(0.,eta).sum(axis=1)[:,None]
    joint=loga+lw[:,None];logq=logsumexp(joint,axis=0)
    R=np.exp(joint-logq[None,:])*frequency[None,:]
    residual=R@profiles-R.sum(axis=1)[:,None]*expit(eta)
    gradient=np.concatenate((-residual.sum(axis=0),(-residual.T@z+penalty*W).ravel()))
    value=-float(frequency@logq)+penalty*float(np.sum(W*W))/2
    assert np.isfinite(value) and np.isfinite(gradient).all()
    return value,gradient

def logprob_prior(theta,profiles,order):
    b,W=unpack(theta,profiles.shape[1]);z,lw=rule(order);total=np.full(len(profiles),-np.inf)
    for start in range(0,len(z),256):
        eta=z[start:start+256]@W.T+b
        loga=eta@profiles.T-np.logaddexp(0.,eta).sum(axis=1)[:,None]
        total=np.logaddexp(total,logsumexp(loga+lw[start:start+256,None],axis=0))
    return total

def modes(theta,profiles):
    b,W=unpack(theta,profiles.shape[1]);z=np.zeros((len(profiles),2));backtracks=0
    for iteration in range(60):
        eta=z@W.T+b;p=expit(eta);g=z-(profiles-p)@W
        active=np.max(abs(g),axis=1)>1e-8
        if not active.any():break
        h=np.eye(2)[None]+np.einsum('nd,di,dj->nij',p*(1-p),W,W)
        step=np.linalg.solve(h,g[...,None])[...,0];dot=np.sum(g*step,axis=1)
        before=.5*np.sum(z*z,axis=1)+np.logaddexp(0.,eta).sum(axis=1)-np.sum(profiles*eta,axis=1)
        alpha=active.astype(float)
        for bt in range(20):
            proposed=z-alpha[:,None]*step;v=proposed@W.T+b
            after=.5*np.sum(proposed*proposed,axis=1)+np.logaddexp(0.,v).sum(axis=1)-np.sum(profiles*v,axis=1)
            bad=active&(after>before-.25*alpha*dot+1e-12)
            if not bad.any():break
            alpha[bad]*=.5;backtracks+=int(bad.sum())
        assert not bad.any(),'Posterior-mode backtracking budget'
        z=proposed
    eta=z@W.T+b;p=expit(eta);g=z-(profiles-p)@W
    error=float(np.max(abs(g)));assert error<=1e-8,'Posterior-mode convergence'
    h=np.eye(2)[None]+np.einsum('nd,di,dj->nij',p*(1-p),W,W)
    L=np.linalg.cholesky(np.linalg.inv(h));logdet=np.linalg.slogdet(h)[1]
    return z,L,logdet,dict(iterations=iteration+1,max_gradient=error,backtracks=backtracks)

def agh_at(theta,profiles,mode,L,logdet,order):
    b,W=unpack(theta,profiles.shape[1]);u,lw=rule(order);values=np.empty(len(profiles))
    for start in range(0,len(profiles),8):
        y=profiles[start:start+8];z=mode[start:start+8,None,:]+np.einsum('nij,mj->nmi',L[start:start+8],u)
        eta=z@W.T+b
        loga=np.einsum('nmd,nd->nm',eta,y)-np.logaddexp(0.,eta).sum(axis=2)
        ratio=loga-.5*np.sum(z*z,axis=2)+.5*np.sum(u*u,axis=1)[None,:]-.5*logdet[start:start+8,None]
        values[start:start+8]=logsumexp(ratio+lw[None,:],axis=1)
    assert np.isfinite(values).all() and max(values)<=1e-8
    return values

def reference(theta,profiles):
    mode,L,logdet,info=modes(theta,profiles);variants={n:agh_at(theta,profiles,mode,L,logdet,n) for n in (12,20,32)}
    e12=float(np.max(abs(variants[12]-variants[32])));e20=float(np.max(abs(variants[20]-variants[32])))
    order=32;qualified=e12<=1e-4 and e20<=1e-5;errors=dict(coarse=e12,adjacent=e20)
    if not qualified:
        for n,previous in ((48,32),(64,48)):
            variants[n]=agh_at(theta,profiles,mode,L,logdet,n);error=float(np.max(abs(variants[n]-variants[previous])))
            order=n;errors[str(n)]=error;qualified=error<=1e-5
            if qualified:break
    return variants[order],variants,dict(main_order=order,qualified=bool(qualified),errors=errors,mode=info,
                                       profiles=len(profiles),nodes=len(profiles)*sum(n*n for n in variants))

def score_patterns(theta,profiles,prior):
    b,W=unpack(theta,profiles.shape[1]);eta=prior@W.T+b
    loga=eta@profiles.T-np.logaddexp(0.,eta).sum(axis=1)[:,None]
    blocks=math.log(100)-logsumexp(loga.reshape(16,100,len(profiles)),axis=1)
    low=blocks[0];outside=blocks.mean(axis=0);inside=math.log(1600)-logsumexp(loga,axis=0)
    a=loga[:100];left=np.vstack((np.full((1,len(profiles)),-np.inf),np.logaddexp.accumulate(a,axis=0)[:-1]))
    right=np.vstack((np.logaddexp.accumulate(a[::-1],axis=0)[::-1][1:],np.full((1,len(profiles)),-np.inf)))
    loo=math.log(99)-np.logaddexp(left,right);jvi=100*low-99*loo.mean(axis=0)
    result=np.stack((low,outside,inside,jvi));assert np.isfinite(result).all();return result

def fit(train,random_start):
    patterns,counts=np.unique(train,axis=0,return_counts=True);freq=counts/len(train);d=train.shape[1]
    p=(train.sum(axis=0)+.5)/(len(train)+1);bias=np.clip(np.log(p/(1-p)),-10,10)
    centered=train-train.mean(axis=0);cov=centered.T@centered/len(train)
    eigen,U=np.linalg.eigh(cov);ix=np.argsort(eigen)[-2:][::-1]
    loading=np.clip(4*U[:,ix]*np.sqrt(np.maximum(eigen[ix],0.)),-1,1)
    starts=[np.concatenate((bias,loading.ravel())),np.concatenate((bias,random_start.ravel()))]
    candidates=[dict(name='IND',p=p.tolist())];fits=[];tapes={}
    for ci,penalty in enumerate(CONFIG['lambdas'],start=1):
        ends=[]
        for si,x0 in enumerate(starts):
            xs=[];fs=[];gs=[]
            def fun(x):
                assert len(fs)<300,'Training call budget'
                value,g=objective(x,patterns,freq,penalty);xs.append(x.copy());fs.append(value);gs.append(g.copy());return value,g
            tick=time.perf_counter();r=minimize(fun,x0.copy(),method='L-BFGS-B',jac=True,bounds=[(-10,10)]*d+[(-3,3)]*(2*d),options=CONFIG['optimizer'])
            assert np.isfinite(r.x).all() and np.isfinite(r.fun)
            key=f'{NAMES[ci]}_start{si}';tapes[key+'_x']=np.array(xs);tapes[key+'_loss']=np.array(fs);tapes[key+'_gradient']=np.array(gs)
            row=dict(name=NAMES[ci],penalty=penalty,start=si,theta=r.x.tolist(),objective=float(r.fun),gradient=r.jac.tolist(),
                nfev=int(r.nfev),nit=int(r.nit),success=bool(r.success),message=str(r.message),seconds=time.perf_counter()-tick)
            assert r.nfev==len(fs);fits.append(row);ends.append(row)
        chosen=min(range(2),key=lambda j:(ends[j]['objective'],j))
        candidates.append(dict(name=NAMES[ci],penalty=penalty,theta=ends[chosen]['theta'],selected_start=chosen))
    return candidates,fits,tapes

def load_data():
    a=np.loadtxt(OUT/'DATA_SOURCE/optdigits.tra',delimiter=',',dtype=np.int64)
    b=np.loadtxt(OUT/'DATA_SOURCE/optdigits.tes',delimiter=',',dtype=np.int64)
    assert a.shape==(3823,65) and b.shape==(1797,65)
    return (a[:,:64]>=8).astype(float),(b[:,:64]>=8).astype(float)

def select(row):return int(np.flatnonzero(row<=np.min(row)+1e-12)[0])

def stream_run(rep):
    folder=OUT/f'rep_{rep:02}';folder.mkdir(parents=True,exist_ok=False);tick=time.perf_counter()
    try:
        data,test=load_data();rng={i:np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([CONFIG['seed'],rep,i]))) for i in range(3)}
        permutation=rng[0].permutation(len(data));ti=permutation[:3058];vi=permutation[3058:]
        train=data[ti];validation=data[vi];initial=.15*rng[1].standard_normal((64,2))
        candidates,fits,tapes=fit(train,initial)
        path=folder/'TRAINING.npz';np.savez_compressed(path,**tapes)
        trained=dict(rep=rep,train_indices=ti.tolist(),validation_indices=vi.tolist(),random_start=initial.tolist(),candidates=candidates,fits=fits,
            trace_file=rel(path),trace_sha256=sha(path),train_sha256=ah(train),validation_sha256=ah(validation),test_sha256=ah(test),
            source_sha256=sha(__file__),freeze_sha256=sha(OUT/'FREEZE.json'),locked_before_validation=True)
        save(folder/'TRAINED.json',trained)
        vp,vc=np.unique(validation,axis=0,return_counts=True);vf=vc/len(validation);prior=rng[2].standard_normal((1600,2))
        matrix=np.empty((5,4));reference_arrays={};refmeta=[];score_arrays={}
        for j,c in enumerate(candidates):
            if j==0:
                p=np.array(c['p']);logq=vp@np.log(p)+(1-vp)@np.log1p(-p);scores=np.broadcast_to(-logq,(4,len(vp))).copy()
                variants={0:logq};meta=dict(qualified=True,main_order=0,nodes=0)
            else:
                logq,variants,meta=reference(c['theta'],vp);scores=score_patterns(c['theta'],vp,prior)
            matrix[:4,j]=scores@vf;matrix[4,j]=-logq@vf;score_arrays[c['name']]=scores
            for n,values in variants.items():reference_arrays[f'val_{c["name"]}_{n}']=values
            refmeta.append(dict(name=c['name'],**meta))
        choices=[select(row) for row in matrix]
        save(folder/'SELECTION.json',dict(rep=rep,trained_sha256=sha(folder/'TRAINED.json'),scores=matrix.tolist(),choices=choices,
             methods=list(METHODS),candidates=list(NAMES),validation_references=refmeta,prior_sha256=ah(prior),locked_before_test=True))
        tp,tc=np.unique(test,axis=0,return_counts=True);tf=tc/len(test);test_losses=[];testmeta=[]
        for j,c in enumerate(candidates):
            if j==0:
                p=np.array(c['p']);logq=tp@np.log(p)+(1-tp)@np.log1p(-p);variants={0:logq};meta=dict(qualified=True,main_order=0,nodes=0)
            else:logq,variants,meta=reference(c['theta'],tp)
            test_losses.append(float(-logq@tf));testmeta.append(dict(name=c['name'],**meta))
            for n,values in variants.items():reference_arrays[f'test_{c["name"]}_{n}']=values
        evaluation=folder/'EVALUATION.npz';np.savez_compressed(evaluation,prior=prior,val_profiles=vp,val_frequency=vf,test_profiles=tp,test_frequency=tf,
              **reference_arrays,**{f'score_{name}':v for name,v in score_arrays.items()})
        selected=np.array(test_losses)[choices]
        result=dict(rep=rep,folder=rel(folder),trained_sha256=sha(folder/'TRAINED.json'),selection_sha256=sha(folder/'SELECTION.json'),
            evaluation_file=rel(evaluation),evaluation_sha256=sha(evaluation),candidate_test_NLL=test_losses,selected_test_NLL=selected.tolist(),
            choices=choices,test_best_candidate=int(np.argmin(test_losses)),test_relative_best=(selected-min(test_losses)).tolist(),
            test_reference=testmeta,numerically_qualified=bool(all(x['qualified'] for x in refmeta+testmeta)),
            reference_nodes=sum(x['nodes'] for x in refmeta+testmeta),training_calls=sum(f['nfev'] for f in fits),
            converged_fits=sum(f['success'] for f in fits),seconds=time.perf_counter()-tick)
        save(folder/'RESULT.json',result);return result
    except Exception:
        save(folder/'ERROR.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise

def summary(records):
    x=np.array([r['selected_test_NLL'] for r in records]);ind=np.array([r['candidate_test_NLL'][0] for r in records]);critical=float(t.ppf(1-.05/10,31))
    family=[]
    for name,values in zip(('RAW100-AGH','OUT16-INNER1600','INNER1600-AGH','JVI100-AGH','AGH-IND'),
                           (x[:,0]-x[:,4],x[:,1]-x[:,2],x[:,2]-x[:,4],x[:,3]-x[:,4],x[:,4]-ind)):
        mean=float(values.mean());se=float(values.std(ddof=1)/math.sqrt(32));interval=[mean-critical*se,mean+critical*se]
        family.append(dict(contrast=name,paired_values=values.tolist(),mean=mean,SE=se,interval=interval,precision_passed=se<=.025,
            material_excess=bool(interval[0]>0 and mean>=.05),below_practical_excess=bool(interval[1]<.05),useful_joint_gain=bool(interval[1]<-.05)))
    return dict(config=CONFIG,freeze_sha256=sha(OUT/'FREEZE.json'),records=records,family=family,methods=list(METHODS),candidates=list(NAMES),
        mean_test_NLL=x.mean(axis=0).tolist(),mean_candidate_test_NLL=np.mean([r['candidate_test_NLL'] for r in records],axis=0).tolist(),
        mean_test_relative_best=np.mean([r['test_relative_best'] for r in records],axis=0).tolist(),
        choice_counts={m:np.bincount([r['choices'][i] for r in records],minlength=4).tolist() for i,m in enumerate(METHODS)},
        numerical_qualification=all(r['numerically_qualified'] for r in records),training_calls=sum(r['training_calls'] for r in records),
        converged_fits=sum(r['converged_fits'] for r in records),reference_nodes=sum(r['reference_nodes'] for r in records),
        scope='Conditional on the fixed UCI official datasets; independent randomized workflow runs, not 32 independent natural data sets or population confidence coverage')

def source_inputs():
    sources=[Path(__file__),ROOT/'script/udigits_audit.py',DOC/'CONTRACT.md',DOC/'QUALIFICATION.md']
    inputs=list((OUT/'DATA_SOURCE').glob('*'))
    return {rel(p):sha(p) for p in sources},{rel(p):sha(p) for p in inputs if p.is_file()}

def freeze():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==BASE
    q=read(OUT/'NUMERICAL_QUALIFICATION.json');assert q['passed'] and q['producer_sha256']==sha(__file__) and q['audit_sha256']==sha(ROOT/'script/udigits_audit.py')
    sources,inputs=source_inputs();save(OUT/'FREEZE.json',dict(authorized=True,created_utc=datetime.now(timezone.utc).isoformat(),baseline=BASE,
        config=CONFIG,sources=sources,inputs=inputs,qualification_sha256=sha(OUT/'NUMERICAL_QUALIFICATION.json'),environment=q['environment']))
    print(json.dumps(dict(frozen=True,sha256=sha(OUT/'FREEZE.json'))))

def validate():
    f=read(OUT/'FREEZE.json');assert f['authorized'] and f['config']==CONFIG
    for group in ('sources','inputs'):
        for p,digest in f[group].items():assert sha(ROOT/p)==digest,p
    assert sha(OUT/'NUMERICAL_QUALIFICATION.json')==f['qualification_sha256']

def run():
    validate();start=time.perf_counter();save(OUT/'EXECUTION.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),freeze_sha256=sha(OUT/'FREEZE.json'),automatic_retry=False))
    try:
        records=[]
        with ProcessPoolExecutor(max_workers=4) as pool:
            for future in as_completed([pool.submit(stream_run,j) for j in range(32)]):
                r=future.result();records.append(r);print(json.dumps(dict(rep=r['rep'],qualified=r['numerically_qualified'],converged=r['converged_fits'],calls=r['training_calls'],seconds=r['seconds'])),flush=True)
        records.sort(key=lambda r:r['rep']);validate();r=summary(records);r['seconds']=time.perf_counter()-start
        save(OUT/'RESULTS.json',r);print(json.dumps(dict(completed=True,qualified=r['numerical_qualification'],seconds=r['seconds'])),flush=True)
    except Exception:save(OUT/'ERROR.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('freeze','run'));globals()[parser.parse_args().command]()
