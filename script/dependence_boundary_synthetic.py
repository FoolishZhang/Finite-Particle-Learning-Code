"""Frozen population MC-likelihood boundary experiment; no natural labels."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, math, time
import numpy as np
from scipy.special import expit, ndtri, roots_hermitenorm
from scipy.stats import qmc
from scipy.optimize import minimize_scalar

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'artifacts/dependence_boundary_v1/S'
DOC=ROOT/'docs/08_research_研究工作/边际依赖接口_20260929'
CFG=dict(rhos=[0.,.05,.108,.25],budgets=[2,10,100],samplers=['iid','antithetic'],streams=8,
    fit_power=12,eval_power=15,seed=20269401,t_bounds=[0.,16.],grid_positive=80,
    primary_rho=.108,primary_K=10,material_nats=.005)

def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def frozen(name,paths,**other):write(RUN/name,dict(time=now(),files={str(p.relative_to(ROOT).as_posix()):sha(p) for p in paths},**other))
def check(name):
    for p,h in read(RUN/name)['files'].items():assert sha(ROOT/p)==h,p
def freeze():
    assert not (RUN/'FREEZE.json').exists()
    write(RUN/'CONFIG.json',CFG)
    frozen('FREEZE.json',[Path(__file__),DOC/'PLAN_BOUNDARY.md',RUN/'CONFIG.json'],baseline='9b4b1e905489001c616e3d486f9c467f10e5d34d')
def draws(K,sampler,power,stream,eval=False):
    d=K if sampler=='iid' else K//2
    u=qmc.Sobol(d,scramble=True,seed=CFG['seed']+stream+(10000 if eval else 0)).random_base2(power)
    z=ndtri(np.clip(u,1e-15,1-1e-15))
    return z if sampler=='iid' else np.concatenate([z,-z],axis=1)
def components(t,z,grad=False):
    # -log likelihood = A(t) + rho * B(t); exact population sum over four states.
    if t==0:
        a=np.full(len(z),math.log(4));b=np.zeros(len(z))
        # For each random set: analytic right derivative after exact label averaging.
        vv=np.mean(z*z,axis=1);mm=z.mean(1)**2
        da=.25*mm;db=.25*(mm-vv)
    else:
        p=expit(math.sqrt(t)*z);pp=(p*p).mean(1);nn=((1-p)**2).mean(1);pn=(p*(1-p)).mean(1)
        a=-.25*(np.log(pp)+np.log(nn))-.5*np.log(pn)
        b=-.25*(np.log(pp)+np.log(nn))+.5*np.log(pn)
        dp=p*(1-p)*z/(2*math.sqrt(t));dpp=(2*p*dp).mean(1);dnn=(-2*(1-p)*dp).mean(1);dpn=((1-2*p)*dp).mean(1)
        da=-.25*(dpp/pp+dnn/nn)-.5*dpn/pn
        db=-.25*(dpp/pp+dnn/nn)+.5*dpn/pn
    return np.array([a.mean(),b.mean(),da.mean(),db.mean()])
def observed(t,order=240):
    z,w=roots_hermitenorm(order);w=w/math.sqrt(2*math.pi)
    return float(np.dot(w,np.tanh(math.sqrt(t)*z/2)**2))
def risks(rho,t):
    rp=observed(t);rc=observed(t,160)
    cond=-.5*(1+rho)*math.log((1+rp)/2)-.5*(1-rho)*math.log((1-rp)/2)
    oracle=-.5*(1+rho)*math.log((1+rho)/2)-.5*(1-rho)*math.log((1-rho)/2)
    return dict(model_correlation=rp,conditional_NLL=cond,joint_NLL=cond+math.log(2),KL=cond-oracle,
                feedback_gain=math.log(2)-cond,quadrature_delta=abs(rp-rc))
def fit():
    check('FREEZE.json');assert not (RUN/'FITS_FREEZE.json').exists()
    fitted=[];traces=[];derivatives=[];start=time.perf_counter()
    for K in CFG['budgets']:
        for sampler in CFG['samplers']:
            zz=[draws(K,sampler,CFG['fit_power'],s) for s in range(CFG['streams'])];cache={}
            def comp(t):
                key=float(t)
                if key not in cache:cache[key]=np.stack([components(key,z) for z in zz])
                return cache[key].mean(0)
            grid=np.r_[0.,np.geomspace(1e-5,16,CFG['grid_positive'])];vv=np.array([comp(t) for t in grid])
            for r in CFG['rhos']:
                vals=vv[:,0]+r*vv[:,1];candidate=[(float(vals[i]),float(t),'grid') for i,t in enumerate(grid)]
                for i in range(1,len(grid)-1):
                    if vals[i]<=vals[i-1] and vals[i]<=vals[i+1]:
                        opt=minimize_scalar(lambda t:comp(t)[0]+r*comp(t)[1],bounds=(grid[i-1],grid[i+1]),method='bounded',options={'xatol':1e-10})
                        assert opt.success;candidate.append((float(opt.fun),float(opt.x),'bounded'))
                loss,t,method=min(candidate)
                fitted.append(dict(K=K,sampler=sampler,rho=r,t=t,fit_loss=loss,selected_by=method,**risks(r,t)))
                true_d=-r/4 if sampler=='antithetic' else (1-(K-1)*r)/(4*K)
                moments=cache[0.][:,2]+r*cache[0.][:,3]
                f0=math.log(4);h=1e-4
                fh=np.stack([components(h,z) for z in zz]);f2h=np.stack([components(2*h,z) for z in zz])
                estimate=(4*(fh[:,0]+r*fh[:,1]-f0)-(f2h[:,0]+r*f2h[:,1]-f0))/(2*h)
                derivatives.append(dict(K=K,sampler=sampler,rho=r,analytic=true_d,sample_moment_mean=float(moments.mean()),
                    sample_moment_SE=float(moments.std(ddof=1)/math.sqrt(len(moments))),richardson_mean=float(estimate.mean()),
                    max_richardson_minus_same_samples=float(np.max(abs(estimate-moments)))))
            for t,arr in cache.items():traces.append(dict(K=K,sampler=sampler,t=t,stream_components=arr.tolist()))
            print('fitted population',K,sampler,[(r['rho'],round(r['t'],6),round(r['feedback_gain'],7)) for r in fitted if r['K']==K and r['sampler']==sampler],round(time.perf_counter()-start,1),flush=True)
    write(RUN/'FITS.json',fitted);write(RUN/'FIT_INTEGRALS.json',traces);write(RUN/'DERIVATIVES.json',derivatives)
    frozen('FITS_FREEZE.json',[RUN/'FITS.json',RUN/'FIT_INTEGRALS.json',RUN/'DERIVATIVES.json'])
def evaluate():
    check('FREEZE.json');check('FITS_FREEZE.json');assert not (RUN/'COMPLETE.json').exists()
    fitted=read(RUN/'FITS.json');rows=[]
    for f in fitted:
        r=f['rho'];c=np.stack([components(f['t'],draws(f['K'],f['sampler'],CFG['eval_power'],s,True)) for s in range(CFG['streams'])])
        losses=c[:,0]+r*c[:,1];grads=c[:,2]+r*c[:,3];g=float(grads.mean());se=float(grads.std(ddof=1)/math.sqrt(len(grads)))
        pg=max(0.,-g) if f['t']==0 else (max(0.,g) if f['t']==16 else abs(g))
        rows.append(dict(**f,new_stream_losses=losses.tolist(),new_stream_gradients=grads.tolist(),population_objective=float(losses.mean()),
            objective_SE=float(losses.std(ddof=1)/math.sqrt(len(losses))),gradient=g,gradient_SE=se,projected_gradient=pg,
            fit_precision_pass=bool(pg+3*se<.002 and losses.std(ddof=1)/math.sqrt(len(losses))<1e-4 and f['quadrature_delta']<1e-8)))
        print('evaluated',f['K'],f['sampler'],r,f['t'],rows[-1]['fit_precision_pass'],flush=True)
    a=next(f for f in rows if f['rho']==.108 and f['K']==10 and f['sampler']=='iid')
    b=next(f for f in rows if f['rho']==.108 and f['K']==10 and f['sampler']=='antithetic')
    diff=a['conditional_NLL']-b['conditional_NLL'];ders=read(RUN/'DERIVATIVES.json')
    decision=dict(primary_difference=diff,material_pass=bool(diff>.005 and a['fit_precision_pass'] and b['fit_precision_pass']),
        all_fit_precision=all(x['fit_precision_pass'] for x in rows),derivative_max_same_sample_error=max(x['max_richardson_minus_same_samples'] for x in ders),
        finite_K_global_unbiased_claim=False,global_optimizer_certificate=False,natural_data_used=False)
    write(RUN/'RESULTS.json',rows);write(RUN/'DECISION.json',decision)
    frozen('RESULTS_FREEZE.json',[RUN/'RESULTS.json',RUN/'DECISION.json']);write(RUN/'COMPLETE.json',dict(time=now(),status='complete',decision=decision))
    print(decision,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','fit','evaluate']);args=p.parse_args();globals()[args.phase]()
