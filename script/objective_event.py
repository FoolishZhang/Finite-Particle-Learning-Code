"""Frozen event consequences: deterministic limits and finite-D oracle baselines."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, math, os, platform, time, traceback
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import scipy
from scipy.integrate import quad
from scipy.special import expit, roots_hermitenorm, xlogy
from scipy.stats import norm, beta, binom

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/objective_strengthening/N'
DOC=ROOT/'docs/08_research_研究工作/总体目标塌缩与同模型修复_20261001'
CFG=dict(c=.9,truth=[6.,2.],floor=[1.,1.],a=1.,b=8.,scales=[1.,2.,6.,8.],
         orders=[128,256,512,1024,2048],quad_bound=12.,quadrature_gate=1e-10,
         dimensions=[16,64,256,1024],signs=[-1,1],N=65536,chunk=1024,
         seed=7306100100,family=16,alpha=.05,brier_threshold=.001)

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
def kl(p,q):return float(xlogy(p,p/q)+xlogy(1-p,(1-p)/(1-q)))
def cp(x,n):
    a=CFG['alpha']/CFG['family']
    return [0. if x==0 else float(beta.ppf(a/2,x,n-x+1)),
            1. if x==n else float(beta.ppf(1-a/2,x+1,n-x))]
def event(count,d):return (2*count.astype(np.int64)-d)**2<=d
def risk_rectangle(p,q):
    lo,hi=p[0]-q[1],p[1]-q[0]
    distance=0. if lo<=0<=hi else min(abs(lo),abs(hi))
    lowkl=0. if lo<=0<=hi else (kl(p[0],q[1]) if lo>0 else kl(p[1],q[0]))
    return dict(difference=[lo,hi],brier=[distance**2,max(abs(lo),abs(hi))**2],
                logloss=[lowkl,max(kl(x,y) for x in p for y in q)])

def freeze():
    assert not OUT.exists(),'new package only'
    OUT.mkdir(parents=True)
    (OUT/'HANDOFF_INPUT.md').write_bytes((ROOT/'HANDOFF.md').read_bytes())
    write(OUT/'CONFIG.json',CFG)
    write(OUT/'ENVIRONMENT.json',dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()))
    paths=[Path(__file__),DOC/'PLAN_N.md',OUT/'CONFIG.json',OUT/'ENVIRONMENT.json',OUT/'HANDOFF_INPUT.md']
    write(OUT/'FREEZE.json',dict(time_utc=now(),before_integrals_and_random_draws=True,files={rel(p):sha(p) for p in paths}))
    write(OUT/'STATE.json',dict(status='frozen',active_jobs=[]))
    print('N frozen',flush=True)

def moments():
    rows=[];gaussian={}
    for order in CFG['orders']:
        z,w=roots_hermitenorm(order);w=w/np.sqrt(2*np.pi)
        values={}
        for s in CFG['scales']:
            values[str(s)]=dict(alpha=float(w@(z*np.tanh(s*z/2))),
                                ell=float(w@(np.logaddexp(s*z/2,-s*z/2))))
        rows.append(dict(order=order,values=values));gaussian[order]=values
    independent={}
    for s in CFG['scales']:
        vals={}
        for name,fn in [('alpha',lambda z,s=s:z*np.tanh(s*z/2)),
                        ('ell',lambda z,s=s:np.logaddexp(s*z/2,-s*z/2))]:
            value,error,details=quad(lambda z:2*fn(z)*norm.pdf(z),0,CFG['quad_bound'],epsabs=1e-13,epsrel=1e-13,full_output=1)[:3]
            vals[name]=dict(value=value,reported_error=error,evaluations=details['neval'])
        independent[str(s)]=vals
    final=rows[-1]['values'];previous=rows[-2]['values']
    error=max(abs(final[s][k]-previous[s][k]) for s in final for k in ['alpha','ell'])
    cross=max(abs(final[s][k]-independent[s][k]['value']) for s in final for k in ['alpha','ell'])
    quad_error=max(independent[s][k]['reported_error'] for s in final for k in ['alpha','ell'])
    return dict(gh=rows,quad=independent,last_order_difference=error,independent_difference=cross,
                quad_reported_error_max=quad_error,qualified=max(error,cross,quad_error)<=CFG['quadrature_gate'],interval_certified=False,
                GH_scalar_integrand_evaluations=len(CFG['scales'])*sum(CFG['orders'])*2)

def derived(m):
    v=m['gh'][-1]['values'];av=(v['6.0']['alpha']+v['2.0']['alpha'])/2;aa=v['1.0']['alpha']
    rows=[]
    for sign in CFG['signs']:
        vt=1+sign*CFG['c']*av**2;va=1+sign*CFG['c']*aa**2
        p=float(2*norm.cdf(1/np.sqrt(vt))-1);q=float(2*norm.cdf(1/np.sqrt(va))-1)
        q0=float(2*norm.cdf(1)-1)
        rows.append(dict(sign=sign,teacher_variance=vt,floor_variance=va,teacher_event=p,floor_event=q,
                         independent_event=q0,teacher_minus_floor=p-q,brier_excess=(p-q)**2,
                         event_logloss_excess=kl(p,q),teacher_minus_independent=p-q0,
                         independent_brier=(p-q0)**2))
    slope=aa/2;lam=CFG['b']**2*(1+CFG['c']/127)
    coarse=2/slope*np.sqrt(2*lam*np.log(100)/128)
    return dict(limits=rows,coarse_bound_d128_K100=float(coarse),trivial_max_scale_excess=14.,
                coarse_bound_informative=bool(coarse<14),coarse_lambda=lam,ell_prime_floor=slope)

def simulate(di,si):
    d=CFG['dimensions'][di];sign=CFG['signs'][si];n=CFG['N']
    seed=CFG['seed']+100*di+si;rng=np.random.Generator(np.random.PCG64(seed))
    state0=rng.bit_generator.state
    counts=np.empty((n,2),dtype=np.uint16)
    rho=CFG['c']/(d-1);perp=np.sqrt(1-sign*rho);common=np.sqrt(1+sign*CFG['c'])
    scale=np.r_[np.full(d//2,CFG['truth'][0]),np.full(d//2,CFG['truth'][1])]
    for start in range(0,n,CFG['chunk']):
        size=min(CFG['chunk'],n-start)
        z=rng.standard_normal((size,d));u=rng.random((size,d))
        g=perp*z+(common-perp)*z.mean(axis=1,keepdims=True)
        counts[start:start+size,0]=(u<expit(g*scale)).sum(axis=1)
        counts[start:start+size,1]=(u<expit(g*CFG['a'])).sum(axis=1)
    path=OUT/'local'/f'd{d}_sign{sign}.npz';path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path,counts=counts)
    events=[int(event(counts[:,j],d).sum()) for j in range(2)]
    intervals=[cp(x,n) for x in events]
    rectangle=risk_rectangle(*intervals)
    direction=rectangle['difference'][0]>0 if sign<0 else rectangle['difference'][1]<0
    lower=math.ceil((d-math.sqrt(d))/2);upper=math.floor((d+math.sqrt(d))/2)
    return dict(d=d,sign=sign,N=n,seed=seed,state_initial=state0,state_final=rng.bit_generator.state,
                events=events,probabilities=[x/n for x in events],probability_intervals=intervals,
                risk_rectangle=rectangle,primary_pass=bool(direction and rectangle['brier'][0]>CFG['brier_threshold']),
                independent_binomial_event=float(binom.cdf(upper,d,.5)-binom.cdf(lower-1,d,.5)),
                count_histograms=[np.bincount(counts[:,j],minlength=d+1).tolist() for j in range(2)],
                array=rel(path),array_sha256=sha(path),shared_randomness_between_two_laws=True)

def run():
    verify(read(OUT/'FREEZE.json')['files']);assert read(OUT/'STATE.json')['status']=='frozen'
    write(OUT/'START.json',dict(time_utc=now(),freeze_sha256=sha(OUT/'FREEZE.json')))
    m=moments();write(OUT/'MOMENTS.json',m)
    if not m['qualified']:
        write(OUT/'STATE.json',dict(status='stopped_numerical_gate',active_jobs=[]),True)
        print('N quadrature gate failed; no RNG',flush=True);return
    write(OUT/'LIMITS.json',derived(m));rows=[]
    for di,d in enumerate(CFG['dimensions']):
        for si,sign in enumerate(CFG['signs']):
            row=simulate(di,si);rows.append(row);write(OUT/f'd{d}_sign{sign}.json',row)
            print(json.dumps(dict(d=d,sign=sign,p=row['probabilities'],passed=row['primary_pass'])),flush=True)
    write(OUT/'RESULTS.json',rows)
    write(OUT/'COUNTS.json',dict(base_vector_draws=524288,coupled_law_observations=1048576,
                               label_bit_outputs=356515840,new_fits=0,new_neural_models=0,
                               full_bit_arrays_saved=False,per_example_count_arrays_saved=True,
                               binomial_probability_family=16,primary_pass=sum(r['primary_pass'] for r in rows),primary_total=8))
    files=[p for p in OUT.glob('*.json') if p.name not in ['FREEZE.json','CONFIG.json','ENVIRONMENT.json','STATE.json']]
    write(OUT/'RESULTS_FREEZE.json',dict(time_utc=now(),source_freeze_sha256=sha(OUT/'FREEZE.json'),
              files={rel(p):sha(p) for p in files},local_arrays={r['array']:r['array_sha256'] for r in rows}))
    write(OUT/'STATE.json',dict(status='complete',active_jobs=[]),True)

def check(workspace):
    verify(read(OUT/'FREEZE.json')['files']);seal=read(OUT/'RESULTS_FREEZE.json');verify(seal['files'])
    assert seal['source_freeze_sha256']==sha(OUT/'FREEZE.json')
    assert read(OUT/'STATE.json')['status']=='complete'
    assert derived(read(OUT/'MOMENTS.json'))==read(OUT/'LIMITS.json')
    rows=read(OUT/'RESULTS.json');assert len(rows)==8
    for r in rows:
        for j,hist in enumerate(r['count_histograms']):
            assert sum(hist)==r['N']
            x=sum(v for i,v in enumerate(hist) if (2*i-r['d'])**2<=r['d'])
            assert x==r['events'][j] and cp(x,r['N'])==r['probability_intervals'][j]
        assert risk_rectangle(*r['probability_intervals'])==r['risk_rectangle']
    if workspace:
        verify(seal['local_arrays'])
        for r in rows:
            with np.load(ROOT/r['array']) as f:
                a=f['counts'];assert a.shape==(CFG['N'],2)
                assert [int(event(a[:,j],r['d']).sum()) for j in range(2)]==r['events']
    print(json.dumps(dict(passed=True,workspace=workspace,new_fits=0,random_draws_reexecuted=0,integrals_reexecuted=0,primary_pass=sum(r['primary_pass'] for r in rows))))

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
