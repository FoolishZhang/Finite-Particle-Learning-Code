"""Frozen low-dimensional raw-prior objectives; separate from group scoring."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,time,traceback
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import numpy as np
import scipy
from scipy.special import expit,logsumexp,xlogy
from scipy.stats import beta,t
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/objective_strengthening/R'
DOC=ROOT/'docs/08_research_研究工作/总体目标塌缩与同模型修复_20261001'
CFG=dict(dimensions=[16,64,256],K_by_d={'16':[10],'64':[10,100],'256':[10]},streams=8,c=.9,
         truth=[6.,2.],a=1.,b=8.,coarse=[1.,1.5,2.,3.,4.,6.,8.],refine_radius=.5,refine_step=.125,
         n_select=1024,n_confirm=4096,prior_chunk=128,label_chunk=256,event_chunk=1024,event_N=65536,
         event_family=38,event_alpha=.025,score_family=4,score_alpha=.025,SE_per_dim=.002,
         improvement_per_dim=.001,brier_gate=.001,near=.005,tie=1e-12,seed=7316100100)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return Path(p).relative_to(ROOT).as_posix()
def read(p):return json.loads(Path(p).read_text('utf-8'))
def now():return datetime.now(timezone.utc).isoformat()
def write(p,v,replace=False):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w' if replace else 'x',encoding='utf-8',newline='\n') as f:
        json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def verify(m):
    for n,h in m.items():assert sha(ROOT/n)==h,n
def rng(seed):return np.random.Generator(np.random.PCG64(seed))
def transform(z,d):
    p=np.sqrt(1+CFG['c']/(d-1));q=np.sqrt(1-CFG['c'])
    return p*z+(q-p)*z.mean(axis=-1,keepdims=True)
def labels(d,uv,n,seed,chunk):
    g=rng(seed);out=np.empty((n,d),np.int8);scale=np.r_[np.full(d//2,uv[0]),np.full(d//2,uv[1])]
    for start in range(0,n,chunk):
        size=min(chunk,n-start);z=transform(g.standard_normal((size,d)),d)
        out[start:start+size]=2*(g.random((size,d))<expit(z*scale)).astype(np.int8)-1
    return out
def freeze():
    assert not OUT.exists();OUT.mkdir(parents=True)
    (OUT/'HANDOFF_INPUT.md').write_bytes((ROOT/'HANDOFF.md').read_bytes())
    write(OUT/'CONFIG.json',CFG)
    write(OUT/'ENVIRONMENT.json',dict(numpy=np.__version__,scipy=scipy.__version__,threads=2))
    paths=[Path(__file__),DOC/'PLAN_R.md',OUT/'CONFIG.json',OUT/'ENVIRONMENT.json',OUT/'HANDOFF_INPUT.md']
    write(OUT/'FREEZE.json',dict(utc=now(),before_scientific_rng=True,files={rel(p):sha(p) for p in paths}))
    write(OUT/'STATE.json',dict(status='frozen',active_jobs=[]));print('R frozen',flush=True)
def smoke():
    d=4;z=np.arange(24,dtype=float).reshape(2,3,4)/11-1;y=np.array([[1,-1,1,1],[-1,1,1,-1]])
    z=transform(z,d);uv=(2.,1.)
    eta=z*np.array([2.,2.,1.,1.]);direct=-np.logaddexp(0,-eta*y[:,None,:]).sum(-1)
    a=(z[:,:,:2]*y[:,None,:2]).sum(-1)/2;b=(z[:,:,2:]*y[:,None,2:]).sum(-1)/2
    grouped=2*a+b-np.logaddexp(z[:,:,:2],-z[:,:,:2]).sum(-1)-np.logaddexp(z[:,:,2:]/2,-z[:,:,2:]/2).sum(-1)
    err=float(np.max(abs(direct-grouped)));assert err<1e-12
    A=transform(np.eye(d),d);R=(1+CFG['c']/(d-1))*np.eye(d)-CFG['c']/(d-1)*np.ones((d,d))
    cov=float(np.max(abs(A@A.T-R)));assert cov<1e-12
    write(OUT/'SMOKE.json',dict(expression_error=err,covariance_error=cov,RNG_calls=0))
def evaluate(d,Y,seed,Ks,params):
    params=sorted(set(tuple(map(float,p)) for p in params));us=sorted({p[0] for p in params});vs=sorted({p[1] for p in params})
    g=rng(seed);n=len(Y);kmax=max(Ks);result=np.empty((len(Ks),n,len(params)));maxerr=0.;started=time.perf_counter()
    for start in range(0,n,CFG['prior_chunk']):
        size=min(CFG['prior_chunk'],n-start);z=transform(g.standard_normal((size,kmax,d)),d)
        za,zb=z[:,:,:d//2],z[:,:,d//2:];y=Y[start:start+size]
        ta=(za*y[:,None,:d//2]).sum(-1)/2;tb=(zb*y[:,None,d//2:]).sum(-1)/2
        ca={u:np.logaddexp(u*za/2,-u*za/2).sum(-1) for u in us}
        cb={v:np.logaddexp(v*zb/2,-v*zb/2).sum(-1) for v in vs}
        for j,(u,v) in enumerate(params):
            logp=u*ta+v*tb-ca[u]-cb[v]
            for ki,K in enumerate(Ks):result[ki,start:start+size,j]=-logsumexp(logp[:,:K],axis=1)+np.log(K)
            if start==0 and (u,v) in [(6.,2.),(1.,1.),(3.,1.)]:
                eta=z[0,0]*np.r_[np.full(d//2,u),np.full(d//2,v)]
                direct=-np.logaddexp(0,-eta*y[0]).sum();maxerr=max(maxerr,abs(float(direct-logp[0,0])))
    assert np.isfinite(result).all() and maxerr<=1e-10
    return params,result,dict(seconds=time.perf_counter()-started,expression_error=maxerr,
                              prior_particles=n*kmax,particle_candidate_evaluations=n*kmax*len(params),
                              prior_seed=seed,N=n,Ks=Ks,parameters=len(params))
def choose(params,values):
    means=values.mean(0);best=means.min();i=next(i for i,x in enumerate(means) if x<=best+CFG['tie'])
    near=np.flatnonzero(means<=best+CFG['near']).tolist()
    return dict(scales=list(params[i]),index=i,objective=float(means[i]),near_indices=near,curve=means.tolist())
def refine(uv):
    axes=[]
    for x in uv:
        lo=max(CFG['a'],x-CFG['refine_radius']);hi=min(CFG['b'],x+CFG['refine_radius'])
        axes.append(np.arange(round(lo*8),round(hi*8)+1)/8)
    return [(float(u),float(v)) for u in axes[0] for v in axes[1]]
def unit_seed(di,s):return CFG['seed']+100000*di+1000*s
def select(di,s):
    d=CFG['dimensions'][di];Ks=CFG['K_by_d'][str(d)];seed=unit_seed(di,s)
    Y=labels(d,CFG['truth'],CFG['n_select'],seed,CFG['label_chunk'])
    coarse=[(u,v) for u in CFG['coarse'] for v in CFG['coarse']]
    p,loss,receipt=evaluate(d,Y,seed+1,Ks,coarse);cs=[choose(p,loss[i]) for i in range(len(Ks))]
    fine_sets=[refine(x['scales']) for x in cs];union=sorted(set(x for f in fine_sets for x in f))
    p2,loss2,receipt2=evaluate(d,Y,seed+1,Ks,union)
    fs=[]
    for ki,fset in enumerate(fine_sets):
        ids=[p2.index(x) for x in fset];fs.append(choose(fset,loss2[ki][:,ids]))
    path=OUT/'local'/f'select_d{d}_s{s}.npz';path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path,Y=Y,coarse_losses=loss,fine_losses=loss2)
    record=dict(d=d,stream=s,Ks=Ks,coarse_parameters=p,fine_union=p2,coarse=cs,fine=fs,
                receipt=receipt,refinement_replay=receipt2,array=rel(path),array_sha256=sha(path),
                refinement_uses_same_random_inputs=True)
    write(OUT/'selection'/f'd{d}_s{s}.json',record)
    print(json.dumps(dict(stage='selection',d=d,stream=s,scales=[v['scales'] for v in fs])),flush=True)
    return record
def confirm(di,s,sel):
    d=sel['d'];Ks=sel['Ks'];seed=unit_seed(di,s);params={(6.,2.),(1.,1.),(3.,1.)}
    for co,fi in zip(sel['coarse'],sel['fine']):
        params.add(tuple(co['scales']));u,v=fi['scales'];params.add((u,v))
        for du,dv in [(-.125,0),(.125,0),(0,-.125),(0,.125)]:params.add((min(8,max(1,u+du)),min(8,max(1,v+dv))))
    Y=labels(d,CFG['truth'],CFG['n_confirm'],seed+2,CFG['label_chunk'])
    p,loss,receipt=evaluate(d,Y,seed+3,Ks,params)
    path=OUT/'local'/f'confirm_d{d}_s{s}.npz';np.savez_compressed(path,losses=loss)
    r=dict(d=d,stream=s,Ks=Ks,parameters=p,means=loss.mean(1).tolist(),receipt=receipt,array=rel(path),array_sha256=sha(path))
    write(OUT/'confirmation'/f'd{d}_s{s}.json',r);return r
def interval(values,d):
    x=np.asarray(values);mean=float(x.mean());se=float(x.std(ddof=1)/np.sqrt(8));tc=float(t.ppf(1-CFG['score_alpha']/(2*CFG['score_family']),7))
    return dict(values=x.tolist(),mean=mean,SE=se,interval=[mean-tc*se,mean+tc*se],
                precision_pass=bool(se/d<=CFG['SE_per_dim']),t_is_approximation=True)
def event_eval(state,i):
    d,u,v=state;seed=CFG['seed']+9000000+i
    Y=labels(d,(u,v),CFG['event_N'],seed,CFG['event_chunk']);counts=((Y.sum(1,dtype=np.int64)+d)//2).astype(np.uint16)
    x=int((((2*counts.astype(np.int64)-d)**2)<=d).sum());n=len(counts);a=CFG['event_alpha']/CFG['event_family']
    ci=[0. if not x else float(beta.ppf(a/2,x,n-x+1)),1. if x==n else float(beta.ppf(1-a/2,x+1,n-x))]
    path=OUT/'local'/f'event_{i:03}.npz';np.savez_compressed(path,counts=counts)
    return dict(state=list(state),seed=seed,N=n,event_count=x,probability=x/n,interval=ci,array=rel(path),array_sha256=sha(path))
def rectangle(p,q):
    lo,hi=p[0]-q[1],p[1]-q[0];a=0. if lo<=0<=hi else min(abs(lo),abs(hi))
    def kl(x,y):return float(xlogy(x,x/y)+xlogy(1-x,(1-x)/(1-y)))
    lk=0. if lo<=0<=hi else (kl(p[0],q[1]) if lo>0 else kl(p[1],q[0]))
    return dict(difference=[lo,hi],brier=[a*a,max(abs(lo),abs(hi))**2],logloss=[lk,max(kl(x,y) for x in p for y in q)])
def run():
    verify(read(OUT/'FREEZE.json')['files']);assert read(OUT/'STATE.json')['status']=='frozen';smoke()
    write(OUT/'START.json',dict(utc=now(),freeze_sha256=sha(OUT/'FREEZE.json')));sels={}
    write(OUT/'STATE.json',dict(status='running',phase='selection',pid=os.getpid()),True)
    for di,d in enumerate(CFG['dimensions']):
        for s in range(8):
            write(OUT/'STATE.json',dict(status='running',phase='selection',d=d,stream=s,pid=os.getpid()),True)
            sels[d,s]=select(di,s)
    write(OUT/'SELECTION_SEAL.json',dict(utc=now(),before_confirmation=True,files={rel(p):sha(p) for p in (OUT/'selection').glob('*.json')}))
    conf={}
    for di,d in enumerate(CFG['dimensions']):
        for s in range(8):
            write(OUT/'STATE.json',dict(status='running',phase='confirmation',d=d,stream=s,pid=os.getpid()),True)
            conf[d,s]=confirm(di,s,sels[d,s])
        print(json.dumps(dict(stage='confirmation',d=d)),flush=True)
    states={(d,6.,2.) for d in CFG['dimensions']}|{(d,1.,1.) for d in CFG['dimensions']}
    states|={(d,*f['scales']) for (d,s),r in sels.items() for f in r['fine']}
    assert len(states)<=38
    write(OUT/'EVENT_DESIGN_SEAL.json',dict(utc=now(),before_event_RNG=True,states=sorted(states),selection_seal_sha256=sha(OUT/'SELECTION_SEAL.json')))
    events=[event_eval(x,i) for i,x in enumerate(sorted(states))];lookup={tuple(r['state']):r for r in events}
    write(OUT/'EVENTS.json',events);results=[]
    for d in CFG['dimensions']:
        for ki,K in enumerate(CFG['K_by_d'][str(d)]):
            diffs=[];risks=[];models=[]
            for s in range(8):
                sel=sels[d,s]['fine'][ki]['scales'];models.append(sel);r=conf[d,s];p=[tuple(x) for x in r['parameters']]
                vals=r['means'][ki];diffs.append(vals[p.index((6.,2.))]-vals[p.index(tuple(sel))])
                tr=lookup[d,6.,2.];pr=lookup[(d,*sel)];bounds=rectangle(tr['interval'],pr['interval'])
                risks.append(dict(scales=sel,probability=pr['probability'],brier_plugin=(tr['probability']-pr['probability'])**2,**bounds))
            stats=interval(diffs,d);blo=float(np.mean([r['brier'][0] for r in risks]))
            results.append(dict(d=d,K=K,models=models,floor_choices=sum(x==[1.,1.] for x in models),confirmation_truth_minus_selected=stats,
              realized_model_event_risks=risks,mean_event_brier_plugin=float(np.mean([r['brier_plugin'] for r in risks])),
              mean_event_brier_interval=[blo,float(np.mean([r['brier'][1] for r in risks]))],
              primary_score_pass=bool(stats['precision_pass'] and stats['interval'][0]/d>CFG['improvement_per_dim']),primary_event_pass=bool(blo>CFG['brier_gate'])))
    write(OUT/'RESULTS.json',results)
    count=dict(new_teacher_vectors=122880,unique_prior_particles=4915200,selection_prior_particles=983040,confirmation_prior_particles=3932160,
       fitted_procedures=32,coarse_intermediate_selections=32,grid_search_stages=64,
       event_models=len(events),event_observations=CFG['event_N']*len(events),new_neural_models=0,
       primary_score_pass=sum(x['primary_score_pass'] for x in results),primary_event_pass=sum(x['primary_event_pass'] for x in results),
       particle_candidate_evaluations=sum(r['receipt']['particle_candidate_evaluations']+r['refinement_replay']['particle_candidate_evaluations'] for r in sels.values())+sum(r['receipt']['particle_candidate_evaluations'] for r in conf.values()),
       covariance_structure_known=True,continuous_global_optimality_certified=False)
    write(OUT/'COUNTS.json',count)
    files=[p for p in OUT.rglob('*') if p.is_file() and 'local' not in p.parts and p.name!='STATE.json']
    write(OUT/'RESULTS_FREEZE.json',dict(utc=now(),source_freeze_sha256=sha(OUT/'FREEZE.json'),files={rel(p):sha(p) for p in files},local_arrays={rel(p):sha(p) for p in (OUT/'local').glob('*.npz')}))
    write(OUT/'STATE.json',dict(status='complete',active_jobs=[]),True);print(json.dumps(count),flush=True)
def check(workspace):
    verify(read(OUT/'FREEZE.json')['files']);r=read(OUT/'RESULTS_FREEZE.json');verify(r['files'])
    assert r['source_freeze_sha256']==sha(OUT/'FREEZE.json') and read(OUT/'STATE.json')['status']=='complete'
    if workspace:verify(r['local_arrays'])
    assert len(read(OUT/'RESULTS.json'))==4
    print(json.dumps(dict(passed=True,workspace=workspace,science_not_reexecuted=True)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','run','check']);p.add_argument('--workspace',action='store_true');a=p.parse_args()
    try:
        if a.phase=='freeze':freeze()
        elif a.phase=='run':run()
        else:check(a.workspace)
    except Exception as exc:
        if a.phase=='run' and not (OUT/'RESULTS_FREEZE.json').exists():
            write(OUT/f'FAILURE_{time.time_ns()}.json',dict(error=repr(exc),traceback=traceback.format_exc()))
            write(OUT/'STATE.json',dict(status='stopped_failure',active_jobs=[]),True)
        raise
