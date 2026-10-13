"""Frozen non-oracle loading learning and independent population evaluation."""
from pathlib import Path
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse, hashlib, itertools, json, math, platform, sys, time, traceback
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from scipy.stats import t
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from script import compensation_learning_core as core
from script import covariance_boundary_profile as profile
OUT = ROOT/'artifacts/compensation_learning/RUN'
DOC = core.DOC
CFG = dict(dimensions=[16,64], streams=8, Ns=[256,4096], B=64, confirm_B=512,
           seed=10520261001, methods=['MOM','quasiMLE','raw','col'], bounds=[.5,2.5],
           starts=[.5,1.5,2.5], grid=[.5,1.,1.5,2.,2.5], gtol=1e-7, ftol=1e-12,
           maxiter=200, maxfun=400, pg_gate=1e-6, fd_gate=5e-6, grid_gate=1e-6,
           fd_steps=[1e-4,5e-5], risk_alpha=.025, risk_family=24, risk_se=.002,
           score_alpha=.025, score_family=16, score_se=.003, K='d', chunk=128)
read, write, sha, rel, require, now = core.read, core.write, core.sha, core.rel, core.require, core.now
MOM_PATH = ROOT/'artifacts/objective_strengthening/N/MOMENTS.json'

def moments():
    r = read(MOM_PATH)
    require(r['qualified'], 'unqualified alpha input')
    return r['gh'][-1]['values']['1.0']

def hist(y):
    m = y.shape[1]//2
    n = (y > 0).reshape(len(y),2,m).sum(2)
    h = np.zeros((m+1,m+1))
    np.add.at(h,(n[:,0],n[:,1]),1/len(y))
    return h

def mom(y):
    d = y.shape[1]; m = d//2
    s = y.reshape(len(y),2,m).sum(2)
    pair = (np.mean(s.astype(float)**2,axis=0)-m)/(m*(m-1))
    w2 = (d-1)*pair/moments()['alpha']**2
    return np.sqrt(np.clip(w2,.25,6.25)), dict(pair=pair, untruncated_squared_loading=w2)

class Objective:
    def __init__(self, method, d, y, z, e, raw_cache=None):
        self.method,self.d,self.y,self.z,self.e = method,d,y,z,e
        self.h = hist(y); self.cache={}; self.requests=0; self.actual=0
        m=d//2
        self.S = y.reshape(len(y),2,m).sum(2)*.5
        self.T = raw_cache
        if method=='raw' and self.T is None:
            self.T = [(.5*y[:,g*m:(g+1)*m]) @ e[:,:,g*m:(g+1)*m].reshape(-1,m).T
                      for g in range(2)]
            self.T = [v.reshape(len(y),*z.shape) for v in self.T]
    def __call__(self,w):
        self.requests+=1
        key=tuple(map(float,w))
        if key not in self.cache:
            self.cache[key]=self.compute(np.asarray(w,dtype=float)); self.actual+=1
        return self.cache[key]
    def compute(self,w):
        d=self.d; m=d//2; r=1/math.sqrt(d-1)
        if self.method=='quasiMLE':
            q=core.reference(d,w)
            return -float(np.sum(self.h*q['log_pattern'])), -np.sum(self.h[:,:,None]*q['grad_log_pattern'],axis=(0,1))
        if self.method=='col':
            lp,lq,gp,gq=core.log_h(self.z,d,w)
            return col_scores(self.h,lp,lq,gp,gq,mean=True)
        lam=w*r; private=np.sqrt(1-lam*lam)
        C=[]; Cp=[]
        for g in range(2):
            ee=self.e[:,:,g*m:(g+1)*m]
            eta=lam[g]*self.z[:,:,None]+private[g]*ee
            C.append(np.logaddexp(eta*.5,-eta*.5).sum(-1))
            Cp.append((.5*np.tanh(eta*.5)*r*(self.z[:,:,None]-lam[g]/private[g]*ee)).sum(-1))
        loss=0.; grad=np.zeros(2); n=len(self.y); B,K=self.z.shape
        for start in range(0,n,CFG['chunk']):
            sl=slice(start,min(start+CFG['chunk'],n))
            v=-C[0]-C[1]
            v=v[None]+sum(lam[g]*self.S[sl,g,None,None]*self.z+private[g]*self.T[g][sl] for g in range(2))
            lse=logsumexp(v,axis=2)
            pp=np.exp(v-lse[:,:,None])
            loss+=float(np.sum(-lse+math.log(K)))
            for g in range(2):
                dv=r*self.S[sl,g,None,None]*self.z-r*lam[g]/private[g]*self.T[g][sl]-Cp[g]
                grad[g]-=float(np.sum(pp*dv))
        return loss/(n*B),grad/(n*B)

def col_scores(prob,lp,lq,gp=None,gq=None,mean=False):
    m=prob.shape[0]-1; B,K=lp.shape[1:]
    score=np.zeros(B); gradient=np.zeros((B,2))
    for a,b in zip(*np.nonzero(prob)):
        v=a*lp[0]+(m-a)*lq[0]+b*lp[1]+(m-b)*lq[1]
        lse=logsumexp(v,axis=1)
        score+=prob[a,b]*(-lse+math.log(K))
        if gp is not None:
            pp=np.exp(v-lse[:,None])
            gradient[:,0]-=prob[a,b]*np.sum(pp*(a*gp[0]+(m-a)*gq[0]),axis=1)
            gradient[:,1]-=prob[a,b]*np.sum(pp*(b*gp[1]+(m-b)*gq[1]),axis=1)
    return (float(score.mean()),gradient.mean(0)) if mean else score

def finite_gradient(obj,w,step):
    out=[]
    for g in range(2):
        if w[g]-2*step>=.5 and w[g]+2*step<=2.5:
            offsets=[-2,-1,1,2]; coeff=[1,-8,8,-1]
        elif w[g]+4*step<=2.5:
            offsets=[0,1,2,3,4]; coeff=[-25,48,-36,16,-3]
        else:
            offsets=[0,-1,-2,-3,-4]; coeff=[25,-48,36,-16,3]
        values=[]
        for off in offsets:
            p=np.array(w,copy=True);p[g]+=off*step;values.append(obj(p)[0])
        out.append(float(np.dot(coeff,values)/(12*step)))
    return np.asarray(out)

def fit(obj):
    starts=[]
    for w in itertools.product(CFG['starts'],repeat=2):
        r=minimize(obj,np.asarray(w),jac=True,method='L-BFGS-B',bounds=[(.5,2.5)]*2,
                   options={k:CFG[k] for k in ('gtol','ftol','maxiter','maxfun')})
        starts.append(dict(start=w,w=r.x,objective=float(r.fun),gradient=r.jac,success=bool(r.success),
                           status=int(r.status),message=str(r.message),nit=int(r.nit),nfev=int(r.nfev)))
    minimum=min(x['objective'] for x in starts)
    index=next(i for i,x in enumerate(starts) if x['objective']<=minimum+1e-12)
    best=starts[index];w=np.asarray(best['w']);f,g=obj(w)
    pg=float(np.max(abs(w-np.clip(w-g,.5,2.5))))
    fd=[dict(step=h,gradient=finite_gradient(obj,w,h)) for h in CFG['fd_steps']]
    error=max(float(np.max(abs(x['gradient']-g))) for x in fd)
    grid=[dict(w=v,objective=obj(v)[0]) for v in itertools.product(CFG['grid'],repeat=2)]
    gap=max(0.,f-min(x['objective'] for x in grid))
    checks=dict(projected_gradient=pg<=CFG['pg_gate'],finite_difference=error<=CFG['fd_gate'],grid=gap<=CFG['grid_gate'])
    return dict(w=w,objective=f,gradient=g,starts=starts,selected_start=index,pg=pg,fd=fd,fd_error=error,
                diagnostic_grid=grid,grid_improvement=gap,optimization_checks=checks,
                optimization_qualified=all(checks.values()),requests=obj.requests,actual=obj.actual,
                global_optimality_certified=False)

def algebra():
    d=4;B=2;K=3
    y=np.array(list(itertools.product([-1,1],repeat=d)),dtype=np.int8)
    z=(np.arange(B*K).reshape(B,K)-2)/7
    e=(np.arange(B*K*d).reshape(B,K,d)-11)/13
    w=np.array([.6,.8]);o=Objective('raw',d,y,z,e)
    f,g=o(w);lam=np.repeat(w,2)/math.sqrt(3)
    eta=lam*z[:,:,None]+np.sqrt(1-lam*lam)*e
    lp=-np.logaddexp(0.,-y[:,None,None,:]*eta).sum(-1)
    direct=float(np.mean(-logsumexp(lp,axis=2)+math.log(K)))
    fd=[]
    for j in range(2):
        a=w.copy();b=w.copy();a[j]+=1e-5;b[j]-=1e-5
        fd.append((o(a)[0]-o(b)[0])/2e-5)
    raw_gradient_error=float(max(abs(g-np.array(fd))))
    require(abs(f-direct)<1e-12 and raw_gradient_error<1e-8,'raw algebra')
    errors={}
    yy=np.where((np.arange(7*16).reshape(7,16)%5)<2,1,-1).astype(np.int8)
    for method in ('col','quasiMLE'):
        oo=Objective(method,16,yy,z,None);ww=np.array([.8,1.7]);ff,gg=oo(ww)
        fd=[]
        for j in range(2):
            aa=ww.copy();bb=ww.copy();aa[j]+=1e-5;bb[j]-=1e-5
            fd.append((oo(aa)[0]-oo(bb)[0])/2e-5)
        errors[method]=float(max(abs(gg-np.array(fd))))
        require(errors[method]<1e-8,method+' algebra gradient')
        if method=='col':
            lp,lq,_,_=core.log_h(z,16,ww)
            fullp=np.repeat(lp,8,axis=0).transpose(1,2,0)
            fullq=np.repeat(lq,8,axis=0).transpose(1,2,0)
            vv=np.where(yy[:,None,None,:]>0,fullp,fullq).sum(-1)
            errors['col_full_coordinate_value']=abs(ff-float(np.mean(-logsumexp(vv,axis=2)+math.log(K))))
            require(errors['col_full_coordinate_value']<1e-12,'col count identity')
    return dict(passed=True,raw_value_error=abs(f-direct),raw_gradient_error=raw_gradient_error,
                other_errors=errors,scientific_RNG_calls=0,labels_generated=0,
                deterministic_fixture_only=True,new_candidate_reference_calls=5)

def freeze():
    require(not (OUT/'FREEZE.json').exists(),'already frozen')
    require(read(core.OUT/'STATE.json')['status']=='complete','REF incomplete')
    paths=[Path(__file__),DOC/'PLAN_RUN.md',ROOT/'script/compensation_learning_core.py',
           core.OUT/'FREEZE.json',core.OUT/'RESULTS_FREEZE.json',MOM_PATH,
           ROOT/'script/covariance_boundary_profile.py',ROOT/'script/covariance_boundary_numeric.py']
    paths += [core.OLD/f'REFERENCE_d{d}.json' for d in CFG['dimensions']]
    require(sha(ROOT/'HANDOFF.md')==core.HANDOFF_SHA,'HANDOFF changed')
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'HANDOFF_SNAPSHOT.md').write_bytes((ROOT/'HANDOFF.md').read_bytes())
    write(OUT/'FREEZE.json',dict(utc=now(),base=core.BASE,config=CFG,files={rel(p):sha(p) for p in paths},
          handoff_sha256=core.HANDOFF_SHA,python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
          platform=platform.platform(),seed_scheme='SeedSequence([seed,d,stream,phase]); phase 0 train, 1 confirm'))
    write(OUT/'ALGEBRA.json',algebra())
    print('RUN frozen; algebra passed',flush=True)

def training_unit(d,s):
    path=OUT/f'd{d}_s{s}';path.mkdir(exist_ok=True)
    rng=np.random.default_rng(np.random.SeedSequence([CFG['seed'],d,s,0]))
    N=max(CFG['Ns']);m=d//2;rho=.9/(d-1)
    zt=rng.standard_normal((N,1));et=rng.standard_normal((N,d))
    eta=(math.sqrt(rho)*zt+math.sqrt(1-rho)*et)*np.repeat([6.,2.],m)
    y=(2*(rng.random((N,d))<expit(eta))-1).astype(np.int8)
    z=rng.standard_normal((CFG['B'],d));e=rng.standard_normal((CFG['B'],d,d))
    require(np.max(abs(z))<=12,'common factor outside qualification domain')
    np.savez_compressed(path/'TRAIN.npz',y=y,z=z,e=e)
    full=Objective('raw',d,y,z,e);cache=full.T
    rows=[]
    for N in CFG['Ns']:
        w,extra=mom(y[:N]);rows.append(dict(d=d,stream=s,N=N,method='MOM',w=w,moment=extra))
        for method in CFG['methods'][1:]:
            start=time.monotonic()
            o=Objective(method,d,y[:N],z,e,[v[:N] for v in cache] if method=='raw' else None)
            row=dict(d=d,stream=s,N=N,method=method,**fit(o),seconds=time.monotonic()-start)
            rows.append(row)
            write(path/f'SELECT_N{N}_{method}.json',row)
            print(f'train d{d} s{s} N{N} {method}: w={row["w"]} qualified={row["optimization_qualified"]}',flush=True)
    write(path/'SELECTIONS.json',rows)
    return rows

def evaluate_unit(d,s,rows):
    path=OUT/f'd{d}_s{s}';data=np.load(path/'TRAIN.npz');z=data['z']
    teacher=core.teacher_reference(d);p=teacher['pmf'];m=d//2
    unique={tuple(r['w']):None for r in rows};numeric=[]
    for w in unique:
        q,receipt=core.qualify_reference(d,w,teacher_pmf=p)
        kernel=core.qualify_log_h(z,d,w)
        nr=dict(w=w,reference=receipt,kernel=kernel)
        numeric.append(nr)
        write(path/f'NUMERIC_{len(numeric)-1}.json',nr)
        require(receipt['qualified'] and kernel['qualified'],'endpoint numerical qualification failed')
        unique[w]=q['pmf']
    counts=np.add.outer(np.arange(m+1),np.arange(m+1));scaled=(2*counts-d)/math.sqrt(d)
    masks=[abs(scaled)<=1,scaled<-1,scaled>1]
    risks=[]
    for row in rows:
        q=unique[tuple(row['w'])]
        risks.append({k:row[k] for k in ('d','stream','N','method','w')}|dict(
            joint_KL=core.stable_kl(p,q),events=[dict(P=float(p[mask].sum()),Q=float(q[mask].sum())) for mask in masks],
            optimization_qualified=row.get('optimization_qualified',True)))
    write(path/'RISKS.json',risks)
    old=read(core.OLD/f'REFERENCE_d{d}.json')
    oracle=tuple(np.asarray(old['params'][old['names'].index('comp')]['lambda'])*math.sqrt(d-1))
    params=list(unique)
    if oracle not in params:params.append(oracle)
    rng=np.random.default_rng(np.random.SeedSequence([CFG['seed'],d,s,1]))
    zz=rng.standard_normal((CFG['confirm_B'],d));ee=rng.standard_normal((CFG['confirm_B'],d,d))
    require(np.max(abs(zz))<=12,'confirm common domain')
    np.savez_compressed(path/'CONFIRM_INPUT.npz',z=zz,e=ee)
    raw=np.empty((len(params),CFG['confirm_B']));col=np.empty_like(raw)
    for i,w in enumerate(params):
        lam=np.repeat(w,m)/math.sqrt(d-1)
        for b in range(0,CFG['confirm_B'],8):
            eta=lam*zz[b:b+8,:,None]+np.sqrt(1-lam*lam)*ee[b:b+8]
            raw[i,b:b+8]=profile.scores(eta,p,[d])[0][:,0]+d*moments()['ell']
        lp,lq,_,_=core.log_h(zz,d,w)
        col[i]=col_scores(p,lp,lq)
        receipt=core.qualify_log_h(zz,d,w)
        require(receipt['qualified'],'confirm kernel qualification')
        numeric.append(dict(confirm=True,w=w,kernel=receipt))
    np.savez_compressed(path/'CONFIRM.npz',w=np.asarray(params),raw=raw,col=col)
    write(path/'EVALUATION.json',dict(numerical=numeric,unique_learned_references=len(unique),
          reference_variant_integrations=3*len(unique),confirmation_candidates=len(params),oracle_comp=oracle,
          no_selection_after_risk=True,strict_error_certificate=False))
    print(f'evaluation d{d} s{s}: {len(unique)} learned references, {len(params)} confirm candidates',flush=True)
    return risks

def interval(values,alpha,family,se_gate):
    x=np.asarray(values);se=float(x.std(ddof=1)/math.sqrt(len(x)));mean=float(x.mean())
    half=float(t.ppf(1-alpha/(2*family),len(x)-1)*se)
    return dict(values=x,mean=mean,SE=se,low=mean-half,high=mean+half,precision=se<=se_gate,
                alpha=alpha,family=family,df=len(x)-1,approximate_t=True)

def collect():
    allrisk=[];allselect=[];riskstats=[];scorestats=[];riskgates=[];scoregates=[]
    for d in CFG['dimensions']:
        rr=[];ss=[];cc=[]
        for s in range(CFG['streams']):
            path=OUT/f'd{d}_s{s}'
            rr.append(read(path/'RISKS.json'));ss.append(read(path/'SELECTIONS.json'))
            with np.load(path/'CONFIRM.npz') as a:cc.append({k:a[k].copy() for k in a.files})
        allrisk.extend(itertools.chain.from_iterable(rr));allselect.extend(itertools.chain.from_iterable(ss))
        for N in CFG['Ns']:
            for a,b in [('raw','MOM'),('raw','col'),('raw','quasiMLE'),('MOM','quasiMLE'),('col','quasiMLE'),('MOM','col')]:
                v=[next(x['joint_KL'] for x in r if x['N']==N and x['method']==a)-next(x['joint_KL'] for x in r if x['N']==N and x['method']==b) for r in rr]
                row=dict(d=d,N=N,pair=f'{a}-{b}',**interval(v,CFG['risk_alpha'],24,CFG['risk_se']))
                if N==4096 and (a,b) in [('raw','MOM'),('raw','col'),('MOM','quasiMLE'),('col','quasiMLE')]:
                    row['main_gate']=bool(row['precision'] and (row['low']>.002 if a=='raw' else row['high']<.001));riskgates.append(row['main_gate'])
                riskstats.append(row)
            for objective,a,b in [('raw','raw','MOM'),('raw','raw','quasiMLE'),('col','col','MOM'),('col','col','quasiMLE')]:
                v=[]
                for s in range(CFG['streams']):
                    def score(method):
                        w=next(x['w'] for x in ss[s] if x['N']==N and x['method']==method)
                        idx=next(i for i,q in enumerate(cc[s]['w']) if np.array_equal(q,w))
                        return float(cc[s][objective][idx].mean())
                    v.append(score(a)-score(b))
                row=dict(d=d,N=N,objective=objective,pair=f'{a}-{b}',**interval(v,CFG['score_alpha'],16,CFG['score_se']))
                if N==4096 and objective=='raw' and b=='MOM':
                    row['main_gate']=bool(row['precision'] and row['high']<-.001);scoregates.append(row['main_gate'])
                scorestats.append(row)
    numerical=[read(OUT/f'd{d}_s{s}'/'EVALUATION.json') for d in CFG['dimensions'] for s in range(8)]
    result=dict(risks=allrisk,risk_contrasts=riskstats,score_contrasts=scorestats,risk_main_gates=riskgates,score_main_gates=scoregates,
      optimization_qualified=sum(x.get('optimization_qualified',False) for x in allselect),continuous_procedures=96,MOM_outputs=32,
      optimizer_invocations=864,objective_requests=sum(x.get('requests',0) for x in allselect),actual_objective_evaluations=sum(x.get('actual',0) for x in allselect),
      endpoint_reference_integrations=sum(x['reference_variant_integrations'] for x in numerical),
      new_training_vectors=65536,new_training_bits=2621440,training_prior_vectors=40960,confirm_prior_vectors=327680,total_prior_vectors=368640,
      independent_streams=16,new_deployment_labels=0,new_NN=0,selection_seal_sha256=sha(OUT/'SELECTION_FREEZE.json'))
    write(OUT/'RESULTS.json',result)

def run():
    f=read(OUT/'FREEZE.json');core.verify(f['files']);require(f['config']==CFG,'config changed')
    require(not (OUT/'STATE.json').exists(),'RUN already started')
    write(OUT/'STATE.json',dict(status='running',utc=now(),stage='training'))
    start=time.monotonic()
    try:
        for d in CFG['dimensions']:
            for s in range(CFG['streams']):training_unit(d,s)
        files={rel(p):sha(p) for p in OUT.glob('d*/**/*') if p.is_file()}
        write(OUT/'SELECTION_FREEZE.json',dict(utc=now(),files=files,all_128_selected_before_risk=True))
        write(OUT/'STATE.json',dict(status='running',utc=now(),stage='evaluation'),True)
        for d in CFG['dimensions']:
            for s in range(CFG['streams']):evaluate_unit(d,s,read(OUT/f'd{d}_s{s}'/'SELECTIONS.json'))
        collect()
        write(OUT/'STATE.json',dict(status='complete',utc=now(),seconds=time.monotonic()-start,active_jobs=[]),True)
        files={rel(p):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='RESULTS_FREEZE.json'}
        write(OUT/'RESULTS_FREEZE.json',dict(source_freeze_sha256=sha(OUT/'FREEZE.json'),files=files))
    except Exception:
        write(OUT/'FAILURE.json',dict(utc=now(),traceback=traceback.format_exc()))
        write(OUT/'STATE.json',dict(status='failed',utc=now(),active_jobs=[]),True)
        raise

def check(workspace=False):
    f=read(OUT/'FREEZE.json');core.verify(f['files'])
    seal=read(OUT/'RESULTS_FREEZE.json');require(seal['source_freeze_sha256']==sha(OUT/'FREEZE.json'),'source seal')
    files={k:v for k,v in seal['files'].items() if workspace or not k.endswith('.npz')}
    core.verify(files);require(read(OUT/'STATE.json')['status']=='complete','incomplete')
    require(read(OUT/'RESULTS.json')['continuous_procedures']==96,'fit count')
    print('RUN byte/receipt check passed; no scientific replay',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['algebra','freeze','run','check']);p.add_argument('--workspace',action='store_true');a=p.parse_args()
    if a.action=='algebra':print(json.dumps(algebra()))
    elif a.action=='check':check(a.workspace)
    else:globals()[a.action]()
