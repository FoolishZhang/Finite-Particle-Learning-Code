"""Frozen K4 null-teacher population measurements with bounded Price statistics."""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import argparse,hashlib,itertools,json,math,platform,sys,time,traceback
from datetime import datetime,timezone
import numpy as np
import scipy
from scipy.special import expit,log_expit,logsumexp
ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/08_research_研究工作/无关维度与固定噪声总体偏好_20261001'
OUT=ROOT/'artifacts/nuisance_dimensions/MC'
BASE='04f6f0e09e3aff396d0b3651ae3d158fc4ed4662'
HANDOFF='ebd22608c6c192bef4596e5cab5fd3ef9aa62f6120d39e9d1f83c9575291d8ee'
CFG=dict(K=4,a=1.,R=.75,theta=0.,ns=[0,14,62,254,1022,4094],rhos=[.3,.75],
         streams=8,banks_per_stream=8192,chunk=64,seed=202610010401,
         roles={'active':0,'uniform_path':1,'nuisance':2},raw_noise_prefix_banks=2,
         alpha=.05,family=18,price_abs_bound=.25,gap_abs_bound=.1875,
         numerical_bound_tolerance=1e-12,new_labels=0,new_data_fits=0)
LABELS=np.array(list(itertools.product([0.,1.],repeat=2)))
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return Path(p).relative_to(ROOT).as_posix()
def read(p):return json.loads(Path(p).read_text('utf-8'))
def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)
def write(p,x,replace=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w' if replace else 'x',encoding='utf-8',newline='\n') as f:json.dump(x,f,default=serial,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def require(x,msg):
    if not x:raise RuntimeError(msg)
def verify(files):
    for p,h in files.items():require(sha(ROOT/p)==h,'changed '+p)

def bank_statistics(z,s,rho):
    """z=(B,K,2), s=(B,K), scalar rho or independent per-bank rho."""
    rr=np.asarray(rho,dtype=float)
    if rr.ndim==0:rr=np.full(len(z),float(rr))
    require(rr.shape==(len(z),) and np.max(abs(rr))<=CFG['R'],'rho domain')
    rr=rr[:,None];q=np.sqrt(1-rr*rr)
    x1=z[:,:,0];x2=rr*x1+q*z[:,:,1]
    p1,p2=expit(x1),expit(x2)
    plus1,minus1=log_expit(x1),log_expit(-x1)
    plus2,minus2=log_expit(x2),log_expit(-x2)
    velocity=x1-rr/q*z[:,:,1]
    centered=s-s.max(axis=1,keepdims=True)
    noise_pi=np.exp(centered-logsumexp(centered,axis=1,keepdims=True))
    price=np.zeros(len(z));path=np.zeros(len(z));controlled=np.zeros(len(z))
    for c1,c2 in LABELS:
        active=c1*plus1+(1-c1)*minus1+c2*plus2+(1-c2)*minus2
        h=centered+active;lse=logsumexp(h,axis=1,keepdims=True);pi=np.exp(h-lse)
        r1,r2=c1-p1,c2-p2
        price-=.25*np.sum(pi*(1-pi)*r1*r2,axis=1)
        derivative=r2*velocity
        path+=.25*np.sum((noise_pi-pi)*derivative,axis=1)
        # This includes a candidate-independent centering constant; only differences are used.
        controlled+=.25*(-lse[:,0]+math.log(CFG['K'])+np.sum(noise_pi*active,axis=1))
    return price,path,controlled

def artificial():
    z=(np.arange(3*4*2).reshape(3,4,2)-11)/9
    s=np.array([[.2,-.1,-.4,.1],[-.5,-.2,.1,.6],[0,0,0,0]])
    rho=.3;price,path,value=bank_statistics(z,s,rho)
    direct=[]
    for b in range(len(z)):
        x=np.c_[z[b,:,0],rho*z[b,:,0]+math.sqrt(1-rho*rho)*z[b,:,1]]
        out=0.
        for label in LABELS:
            probs=expit(x);logp=np.sum(label*log_expit(x)+(1-label)*log_expit(-x),axis=1)
            pi=np.exp(s[b]+logp-logsumexp(s[b]+logp))
            out-=.25*sum(pi[k]*(1-pi[k])*(label[0]-probs[k,0])*(label[1]-probs[k,1]) for k in range(4))
        direct.append(out)
    h=1e-5
    fd=(bank_statistics(z,s,rho+h)[2]-bank_statistics(z,s,rho-h)[2])/(2*h)
    err=float(np.max(abs(price-direct)));fd_err=float(np.max(abs(path-fd)))
    require(err<1e-12 and fd_err<1e-8,'artificial algebra')
    require(np.max(abs(price))<=.25,'bound fixture')
    random_rho=np.array([.1,.4,.7]);hvalues=.75*bank_statistics(z,s,random_rho)[0]
    require(np.max(abs(hvalues))<=.1875,'path integration bound fixture')
    return dict(passed=True,explicit_Price_error=err,path_FD_error=fd_err,
                Price_not_equated_to_path_on_artificial_nodes=True,scientific_RNG_calls=0,new_labels=0)

def freeze():
    require(not (OUT/'FREEZE.json').exists(),'already frozen')
    require(sha(ROOT/'HANDOFF.md')==HANDOFF,'HANDOFF changed')
    old=ROOT/'artifacts/weak_emission/NUMERIC/GRID.json'
    paths=[Path(__file__),DOC/'PLAN_MC.md',DOC/'MC_DESIGN.md',old]
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'HANDOFF_INPUT.md').write_bytes((ROOT/'HANDOFF.md').read_bytes())
    write(OUT/'FREEZE.json',dict(base=BASE,utc=now(),config=CFG,files={rel(p):sha(p) for p in paths},
         handoff_sha256=HANDOFF,python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()))
    write(OUT/'ARTIFICIAL_QA.json',artificial())
    print('MC frozen; artificial algebra passed',flush=True)

def stream(index):
    path=OUT/f's{index}';path.mkdir(exist_ok=True)
    make=lambda role:np.random.default_rng(np.random.SeedSequence([CFG['seed'],index,role]))
    B=CFG['banks_per_stream'];K=CFG['K'];ns=CFG['ns'];maxn=max(ns)
    active=make(0).standard_normal((B,K,2));uniform=make(1).random(B);rng=make(2)
    sums=np.zeros((B,len(ns),K));prefix=None
    for start in range(0,B,CFG['chunk']):
        count=min(CFG['chunk'],B-start)
        noise=rng.standard_normal((count,K,maxn))
        if start==0:prefix=noise[:CFG['raw_noise_prefix_banks']].copy()
        cumulative=np.cumsum(log_expit(noise),axis=2)
        for j,n in enumerate(ns):
            if n:sums[start:start+count,j,:]=cumulative[:,:,n-1]
    np.savez_compressed(path/'INPUT.npz',active=active,uniform=uniform,nuisance_sums=sums,noise_prefix=prefix)
    price=np.empty((B,len(ns),2));path_gradient=np.empty_like(price);gap=np.empty((B,len(ns)))
    for j,n in enumerate(ns):
        for g,rho in enumerate(CFG['rhos']):
            price[:,j,g],path_gradient[:,j,g],_=bank_statistics(active,sums[:,j],rho)
        gap[:,j]=CFG['R']*bank_statistics(active,sums[:,j],CFG['R']*uniform)[0]
    tol=CFG['numerical_bound_tolerance']
    require(np.isfinite(price).all() and np.isfinite(path_gradient).all() and np.isfinite(gap).all(),'nonfinite')
    require(np.max(abs(price))<=CFG['price_abs_bound']+tol,'Price sample range violated')
    require(np.max(abs(gap))<=CFG['gap_abs_bound']+tol,'gap sample range violated')
    np.savez_compressed(path/'STATISTICS.npz',price=price,path_gradient=path_gradient,gap=gap)
    write(path/'SUMMARY.json',dict(stream=index,price_mean=price.mean(0),path_mean=path_gradient.mean(0),gap_mean=gap.mean(0),
          max_abs_price=float(np.max(abs(price))),max_abs_gap=float(np.max(abs(gap))),
          files={rel(p):sha(p) for p in [path/'INPUT.npz',path/'STATISTICS.npz']}))
    print(f'MC stream {index} complete',flush=True)

def interval(values,bound):
    x=np.asarray(values,dtype=np.longdouble);M=len(x);mean=float(x.mean());var=float(np.sum((x-x.mean())**2)/(M-1))
    log=math.log(4*CFG['family']/CFG['alpha']);width=2*bound
    half=math.sqrt(2*var*log/M)+7*width*log/(3*(M-1))
    return dict(mean=mean,variance=var,samples=M,absolute_bound=bound,halfwidth=half,low=mean-half,high=mean+half,
                family=CFG['family'],alpha=CFG['alpha'],formula='Maurer-Pontil two tails and finite family',t_approximation=False)

def old_null_references():
    rows=read(ROOT/'artifacts/weak_emission/NUMERIC/GRID.json')['rows']
    grid={x['rho']:x for x in rows if x['a']==1 and x['variant']=='primary'}
    j=grid[0.]['K'].index(CFG['K']);out={}
    for rho in CFG['rhos']:
        out[str(rho)]=.5*(grid[rho]['path_gradient'][j]-grid[-rho]['path_gradient'][j])
    gap=.5*(grid[CFG['R']]['score_minus_rho0'][j]+grid[-CFG['R']]['score_minus_rho0'][j])
    return dict(gradient=out,gap=gap,numerically_qualified_not_strict_integral_certificate=True,
                known_during_design_not_new_discovery=True)

def collect():
    arrays=[]
    for s in range(CFG['streams']):
        with np.load(OUT/f's{s}/STATISTICS.npz') as data:arrays.append({k:data[k].copy() for k in data.files})
    price=np.concatenate([x['price'] for x in arrays]);gap=np.concatenate([x['gap'] for x in arrays]);path=np.concatenate([x['path_gradient'] for x in arrays])
    records=[];gates=[];certificates=[];diagnostics=[]
    for j,n in enumerate(CFG['ns']):
        scale=math.sqrt(max(1,n))
        for g,rho in enumerate(CFG['rhos']):
            r=dict(kind='gradient',n=n,rho=rho,**interval(price[:,j,g],CFG['price_abs_bound']))
            r.update(scaled_mean=scale*r['mean'],scaled_low=scale*r['low'],scaled_high=scale*r['high'])
            if n in [0,max(CFG['ns'])]:
                r['main_gate']=bool(r['low']>0 if n==0 else r['high']<0);gates.append(r['main_gate'])
            records.append(r)
            diff=path[:,j,g]-price[:,j,g]
            diagnostics.append(dict(n=n,rho=rho,Price_mean=float(price[:,j,g].mean()),path_mean=float(path[:,j,g].mean()),
                                    path_minus_Price_mean=float(diff.mean()),path_minus_Price_SE=float(diff.std(ddof=1)/math.sqrt(len(diff))),
                                    descriptive_only_no_new_gate=True))
        r=dict(kind='gap_R_minus_0',n=n,rho=CFG['R'],**interval(gap[:,j],CFG['gap_abs_bound']))
        r.update(scaled_mean=scale*r['mean'],scaled_low=scale*r['low'],scaled_high=scale*r['high'])
        if n in [0,max(CFG['ns'])]:
            r['main_gate']=bool(r['low']>0 if n==0 else r['high']<0);gates.append(r['main_gate'])
        records.append(r)
        eta=max(0.,-r['high']);kappa=1/6
        certificates.append(dict(n=n,eta=eta,exact_min_joint_KL_lower=kappa*kappa*eta*eta/(2*CFG['price_abs_bound']**2),
            near_min_formula='kappa^2 max(0,eta-epsilon)^2/(2 A^2)',sampling_event_only=True,
            not_the_asymptotic_endpoint_bound=True,numerical_rounding_not_interval_certified=True))
    M=CFG['streams']*CFG['banks_per_stream'];require(len(records)==18 and len(gates)==6,'ledger mismatch')
    write(OUT/'RESULTS.json',dict(records=records,main_gates=gates,main_passed=sum(gates),main_total=6,
          finite_n_risk_certificates=certificates,estimator_diagnostics=diagnostics,old_n0_references=old_null_references(),
          ledger=dict(independent_banks=M,seed_groups=8,role_rng_streams=24,prior_vectors=M*4,
            Gaussian_scalars=M*4*(2+max(CFG['ns'])),uniform_path_parameters=M,
            nested_n_bank_evaluations=M*len(CFG['ns']),new_labels=0,new_fits=0,new_NN=0),
          CI_scope='finite-sample iid bounded sampling, not floating-point or PRNG certificate',
          score_contract='F integrates stochastic K-particle log score; reported model KL refers to integrated Q'))

def run():
    freeze=read(OUT/'FREEZE.json');verify(freeze['files']);require(freeze['config']==CFG,'config')
    require(not (OUT/'STATE.json').exists(),'already started')
    start=time.monotonic();write(OUT/'STATE.json',dict(status='running',utc=now()))
    try:
        for s in range(CFG['streams']):stream(s)
        collect()
        write(OUT/'STATE.json',dict(status='complete',utc=now(),seconds=time.monotonic()-start,active_jobs=[]),True)
        files={rel(p):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='RESULTS_FREEZE.json'}
        write(OUT/'RESULTS_FREEZE.json',dict(source_freeze_sha256=sha(OUT/'FREEZE.json'),files=files))
    except Exception:
        write(OUT/'FAILURE.json',dict(utc=now(),traceback=traceback.format_exc()))
        write(OUT/'STATE.json',dict(status='failed',utc=now(),active_jobs=[]),True);raise

def check(workspace=False):
    f=read(OUT/'FREEZE.json');verify(f['files']);r=read(OUT/'RESULTS_FREEZE.json')
    require(r['source_freeze_sha256']==sha(OUT/'FREEZE.json'),'source chain')
    verify({p:h for p,h in r['files'].items() if workspace or not p.endswith('.npz')})
    require(read(OUT/'STATE.json')['status']=='complete','incomplete')
    v=read(OUT/'RESULTS.json');require(len(v['records'])==18 and len(v['main_gates'])==6,'count')
    print(json.dumps(dict(passed=True,workspace=workspace,read_only=True,no_RNG_or_estimator_replay=True)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['artificial','freeze','run','check']);p.add_argument('--workspace',action='store_true');a=p.parse_args()
    if a.action=='artificial':print(json.dumps(artificial()))
    elif a.action=='check':check(a.workspace)
    else:globals()[a.action]()
