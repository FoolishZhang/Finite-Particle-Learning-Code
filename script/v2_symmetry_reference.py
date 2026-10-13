"""Frozen small-dimensional decomposition of symmetry bias and particle cost."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,hashlib,json,math,os,platform,sys,time,traceback,itertools
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
import numpy as np
import scipy
from scipy.special import expit,log_expit,logsumexp,ndtri,gammaln
from scipy.stats import qmc,t
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/structure_tradeoff_v2/R'
DOC=ROOT/'docs/08_research_研究工作/结构预算与对称化代价_v2_20260930'
BASE='ce1aa8340028dbdfafc42187876bfc44999e2813'
CFG=dict(d=[8,16],profiles={'null':[4.,4.],'mild':[4.5,3.5],'contrast':[6.,2.]},c=.9,
         ref_scrambles=16,ref_exponents=[16,18],sobol_bits=52,ref_batch=4096,ref_seed=20281101,
         K=[1,10,100],streams=8,banks=256,bank_batch=16,bank_seed=20291101,
         ref_l1_gate=5e-5,ref_function_difference_gate=1e-5,ref_function_SE_gate=1e-5,
         propagated_reference_gate=1e-4,mass_gate=1e-12,identity_gate=1e-10,
         raw_SE_gate=1.,group_single_SE_gate=.01,paired_SE_gate=.005,
         alpha=.05,contrasts=72,material=.001,primary_d=16,primary_K=10,
         primary_profiles=['mild','contrast'],fits=0,observed_label_samples=0)
NAMES=['raw_excess','group_excess','single_excess','single_minus_group']
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x,exclusive=True):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x' if exclusive else 'w',encoding='utf-8',newline='\n') as f:f.write(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvwrite(p,rows):
    with p.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def verify(ok,message):
    if not ok:raise RuntimeError(message)
def freeze():
    assert not OUT.exists()
    OUT.mkdir(parents=True);(OUT/'local').mkdir();(OUT/'source_inputs').mkdir()
    (OUT/'.gitignore').write_text('/local/\n',encoding='utf-8',newline='\n')
    (OUT/'source_inputs/HANDOFF.md').write_bytes((ROOT/'HANDOFF.md').read_bytes())
    write(OUT/'CONFIG.json',CFG)
    write(OUT/'ENVIRONMENT.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()))
    paths=[Path(__file__),DOC/'PLAN_R.md',OUT/'CONFIG.json',OUT/'ENVIRONMENT.json',OUT/'source_inputs/HANDOFF.md',
           ROOT/'docs/08_research_研究工作/非交换适用域_v2_20260930/THEORY.md',ROOT/'artifacts/nonexchange_v2/MANIFEST.json']
    write(OUT/'FREEZE.json',dict(time_utc=now(),base=BASE,files={p.relative_to(ROOT).as_posix():sha(p) for p in paths}))
    write(OUT/'STATE.json',dict(status='frozen',time_utc=now(),active_jobs=[]))
    print(json.dumps({'freeze':sha(OUT/'FREEZE.json')}),flush=True)
def checkfreeze():
    assert read(OUT/'CONFIG.json')==CFG
    for n,h in read(OUT/'FREEZE.json')['files'].items():assert sha(ROOT/n)==h,n
def geometry(d):
    m=d//2;a,b=np.meshgrid(np.arange(m+1),np.arange(m+1),indexing='ij');a=a.ravel();b=b.ravel();n=a+b
    lg=(gammaln(m+1)-gammaln(a+1)-gammaln(m-a+1)+gammaln(m+1)-gammaln(b+1)-gammaln(m-b+1))
    lt=gammaln(d+1)-gammaln(n+1)-gammaln(d-n+1)
    labels=np.concatenate([(np.arange(m)[None,:]<a[:,None]),(np.arange(m)[None,:]<b[:,None])],axis=1).astype(float)
    indicator=np.eye(d+1)[n]
    return a,b,n,lg,lt,labels,indicator
def eta_from_z(z,scales):
    d=z.shape[-1];rho=CFG['c']/(d-1)
    u=np.sqrt(1+rho)*z-(np.sqrt(1+rho)-np.sqrt(1-CFG['c']))*z.mean(-1,keepdims=True)
    return u*np.repeat(scales,d//2)
def pb(eta):
    p=expit(eta);q=expit(-eta);mass=np.zeros((len(eta),eta.shape[1]+1));mass[:,0]=1
    for j in range(eta.shape[1]):
        mass[:,1:j+2]=mass[:,1:j+2]*q[:,j,None]+mass[:,:j+1]*p[:,j,None]
        mass[:,0]*=q[:,j]
    return mass
def joint_mass(eta):
    m=eta.shape[-1]//2;aa=pb(eta[:,:m]);bb=pb(eta[:,m:])
    return (aa[:,:,None]*bb[:,None,:]).reshape(len(eta),-1)
def functionals(p,d):
    _,_,n,lg,lt,_,ind=geometry(d);p=np.asarray(p);verify(np.all(p>0),'nonpositive reference mass')
    pn=p@ind;ps=pn[n]*np.exp(lg-lt)
    H=-float(p@(np.log(p)-lg));D=float(p@np.log(p/ps))
    return H,D,pn,ps
def smoke():
    errors=[]
    for d in [8,16]:
        rho=CFG['c']/(d-1);a=np.sqrt(1+rho);b=a-np.sqrt(1-CFG['c'])
        factor=a*np.eye(d)-b*np.ones((d,d))/d
        target=(1+rho)*np.eye(d)-rho*np.ones((d,d))
        errors.append(float(np.max(abs(factor@factor.T-target))))
    eta=np.array([[-2.7,-.3,.8,1.2],[.2,.2,.2,.2]])
    labels=np.array(list(itertools.product([0.,1.],repeat=4)));a=labels[:,:2].sum(1).astype(int);b=labels[:,2:].sum(1).astype(int)
    weights=np.exp(log_expit((2*labels[None,:,:]-1)*eta[:,None,:]).sum(-1))
    joint=joint_mass(eta);whole=pb(eta);_,_,_,_,_,_,ind=geometry(4)
    for j in range(2):
        explicit=np.bincount(3*a+b,weights=weights[j],minlength=9)
        errors.append(float(np.max(abs(explicit-joint[j]))))
    errors.append(float(np.max(abs(joint@ind-whole))))
    uniform=np.bincount(3*a+b,minlength=9)/16
    H,D,_,_=functionals(uniform,4);errors.extend([abs(H-4*math.log(2)),abs(D)])
    verify(max(errors)<1e-12,'deterministic probability smoke failure')
    write(OUT/'SMOKE.json',dict(passed=True,maximum_error=max(errors),enumerated_labels=16,new_scientific_observations=0))
def reference(profile,d,idx):
    start=time.perf_counter();scales=CFG['profiles'][profile];all_low=[];all_high=[];checks=[]
    for r in range(CFG['ref_scrambles']):
        seed=CFG['ref_seed']+10000*d+100*idx+r
        engine=qmc.Sobol(d,scramble=True,bits=52,seed=seed)
        uniform=engine.random_base2(CFG['ref_exponents'][1])
        verify(np.all((uniform>0)&(uniform<1)),'Sobol endpoint: no clipping permitted')
        z=ndtri(uniform);sums=[np.zeros((d//2+1)**2),np.zeros((d//2+1)**2)];err=0.
        for offset in range(0,len(z),CFG['ref_batch']):
            mass=joint_mass(eta_from_z(z[offset:offset+CFG['ref_batch']],scales))
            err=max(err,float(np.max(abs(mass.sum(1)-1))))
            sums[1]+=mass.sum(0)
            if offset<2**CFG['ref_exponents'][0]:sums[0]+=mass.sum(0)
        verify(err<CFG['mass_gate'],'reference normalization gate')
        probs=[sums[j]/(2**CFG['ref_exponents'][j]) for j in range(2)]
        maxnorm=max(abs(float(p.sum())-1) for p in probs);verify(maxnorm<CFG['mass_gate'],'integrated mass gate')
        probs=[p/p.sum() for p in probs] # Only roundoff-level normalization, recorded above.
        all_low.append(probs[0]);all_high.append(probs[1]);checks.append(dict(seed=seed,component_mass_error=err,integrated_mass_error=maxnorm))
    low=np.array(all_low);high=np.array(all_high);p=high.mean(0);p/=p.sum();pl=low.mean(0);pl/=pl.sum()
    fh=np.array([functionals(x,d)[:2] for x in high]);hl,dl,_,_=functionals(pl,d);hh,dd,pn,ps=functionals(p,d)
    l1=float(np.abs(p-pl).sum());diff=max(abs(hh-hl),abs(dd-dl));se=float(np.max(fh.std(0,ddof=1)/math.sqrt(16)))
    ok=l1<=CFG['ref_l1_gate'] and diff<=CFG['ref_function_difference_gate'] and se<=CFG['ref_function_SE_gate']
    result=dict(profile=profile,d=d,scales=scales,low=low.tolist(),high=high.tolist(),mean=p.tolist(),mean_low=pl.tolist(),
                H=hh,model_KL_single=dd,total_p=pn.tolist(),symmetrized_group_p=ps.tolist(),
                low_high_L1=l1,low_high_max_function_difference=diff,max_function_SE=se,numerical_pass=ok,
                RQMC_not_exact_or_distribution_free_certificate=True,integration_nodes=len(z)*16,checks=checks,
                elapsed_seconds=time.perf_counter()-start)
    write(OUT/'reference'/f'{profile}_d{d}.json',result)
    print(json.dumps({'phase':'reference','profile':profile,'d':d,'l1':l1,'function_diff':diff,'SE':se,'pass':ok}),flush=True)
    verify(ok,'Reference numerical gate failed; stop without expanding nodes')
def scores_from_logs(logs,p,d):
    H,D,_,_=functionals(p,d)
    v=np.einsum('...qm,m->...q',logs,p)-H
    return np.concatenate([v,(v[...,2]-v[...,1])[...,None]],axis=-1)
def bank_stream(profile,d,idx,stream):
    ref=read(OUT/'reference'/f'{profile}_d{d}.json');verify(ref['numerical_pass'],'bad reference')
    p=np.array(ref['mean']);pl=np.array(ref['mean_low']);refs=np.array(ref['high'])
    H,D,pn,_=functionals(p,d);a,b,n,lg,lt,labels,indicator=geometry(d)
    seed=CFG['bank_seed']+10000*d+100*idx+stream;rng=np.random.Generator(np.random.PCG64(seed))
    initial=rng.bit_generator.state;out=[];logmeans=np.zeros((3,3,len(p)));totalrisk=[]
    maxidentity=0.;normerror=0.;formulaerror=0.;start=time.perf_counter()
    for offset in range(0,CFG['banks'],CFG['bank_batch']):
        nb=CFG['bank_batch'];eta=eta_from_z(rng.standard_normal((nb*100,d)),CFG['profiles'][profile])
        j=joint_mass(eta);pnt=pb(eta)
        maxidentity=max(maxidentity,float(np.max(abs(j@indicator-pnt))))
        normerror=max(normerror,float(np.max(abs(j.sum(1)-1))),float(np.max(abs(pnt.sum(1)-1))))
        rawlog=eta@labels.T-np.logaddexp(0.,eta).sum(1)[:,None]
        if offset==0:
            other=log_expit((2*labels[None,:,:]-1)*eta[:4,None,:]).sum(-1)
            formulaerror=max(formulaerror,float(np.max(abs(other-rawlog[:4]))))
        rawlog=rawlog.reshape(nb,100,-1);j=j.reshape(nb,100,-1);pnt=pnt.reshape(nb,100,d+1)
        vals=[];tvals=[]
        for ki,k in enumerate(CFG['K']):
            qg=j[:,:k].mean(1);qn=pnt[:,:k].mean(1)
            verify(np.all(qg>0) and np.all(qn>0),'zero likelihood; no floors')
            logs=np.stack([-logsumexp(rawlog[:,:k],axis=1)+math.log(k),-np.log(qg)+lg,
                           -np.log(qn[:,n])+lt],axis=1)
            vals.append(scores_from_logs(logs,p,d));logmeans[ki]+=logs.sum(0)
            tvals.append(-np.log(qn)@pn)
            qs_group=qn[:,n]*np.exp(lg-lt)
            maxidentity=max(maxidentity,float(np.max(abs(qg@indicator-qn))),float(np.max(abs(qs_group@indicator-qn))))
        out.append(np.stack(vals,axis=1));totalrisk.append(np.stack(tvals,axis=1))
    losses=np.concatenate(out);totals=np.concatenate(totalrisk);logmeans/=CFG['banks']
    mean=losses.mean(0);low=scores_from_logs(logmeans,pl,d);high=scores_from_logs(logmeans,p,d)
    reference_readings=np.array([scores_from_logs(logmeans,r,d) for r in refs])
    refse=reference_readings.std(0,ddof=1)/4
    refD=np.array([functionals(r,d)[1] for r in refs]);lowD=functionals(pl,d)[1]
    jensen_single_reference=reference_readings[:,:,2]-refD[:,None]
    jensen_sens=float(max(np.max(abs((high[:,2]-D)-(low[:,2]-lowD))),
                          np.max(3*jensen_single_reference.std(0,ddof=1)/4)))
    sens=float(max(np.max(abs(high-low)),np.max(3*refse),jensen_sens))
    ok=maxidentity<CFG['identity_gate'] and normerror<CFG['mass_gate'] and formulaerror<CFG['identity_gate'] and sens<=CFG['propagated_reference_gate']
    target=OUT/'local'/f'{profile}_d{d}_s{stream}.npz'
    np.savez_compressed(target,losses=losses,total_count_loss=totals,mean_log_pattern_loss=logmeans,reference_readings=reference_readings)
    result=dict(profile=profile,d=d,stream=stream,seed=seed,initial_RNG=initial,final_RNG=rng.bit_generator.state,
                means=mean.tolist(),total_count_means=totals.mean(0).tolist(),bank_count=256,prior_particles=25600,
                count_identity_max_error=maxidentity,normalization_error=normerror,raw_formula_error=formulaerror,
                propagated_reference_sensitivity=sens,single_Jensen_reference_sensitivity=jensen_sens,numerical_pass=ok,array_sha256=sha(target),
                canonical_raw_only_population_equivalence=True,elapsed_seconds=time.perf_counter()-start)
    write(OUT/'streams'/f'{profile}_d{d}_s{stream}.json',result)
    print(json.dumps({'phase':'banks','profile':profile,'d':d,'stream':stream,'numerical_pass':ok,'reference_sensitivity':sens}),flush=True)
    verify(ok,'Bank numerical/reference gate failed; stop')
def collect():
    rows=[];decision=[];stream_rows=[];crit=float(t.ppf(1-CFG['alpha']/(2*72),7))
    for d in CFG['d']:
        for profile in CFG['profiles']:
            records=[read(OUT/'streams'/f'{profile}_d{d}_s{s}.json') for s in range(8)]
            means=np.array([r['means'] for r in records]);D=read(OUT/'reference'/f'{profile}_d{d}.json')['model_KL_single']
            for ki,k in enumerate(CFG['K']):
                for q,name in enumerate(NAMES):
                    values=means[:,ki,q];avg=float(values.mean());se=float(values.std(ddof=1)/math.sqrt(8));lo=avg-crit*se;hi=avg+crit*se
                    gate=1. if q==0 else (.005 if q==3 else .01);prec=se<=gate
                    row=dict(profile=profile,d=d,K=k,reading=name,mean=avg,SE=se,lower=lo,upper=hi,precision_pass=prec,
                             reference_qualified=True,confidence_family=72,confidence=.95,
                             model_KL_single_reference=D,derived_single_Jensen=avg-D if q==2 else None)
                    rows.append(row)
                    stream_rows.extend(dict(profile=profile,d=d,K=k,reading=name,stream=s,value=float(v)) for s,v in enumerate(values))
                    if d==16 and k==10 and profile in CFG['primary_profiles'] and name=='single_minus_group':
                        decision.append(dict(profile=profile,wrong_symmetry_preferred=prec and hi<-CFG['material'],mean=avg,upper=hi,
                                             model_change_strict_positive_from_theory=True,model_KL_reference=D))
    assert len(rows)==72 and len(decision)==2
    csvwrite(OUT/'PRIMARY.csv',rows);csvwrite(OUT/'STREAM_SCORES.csv',stream_rows)
    write(OUT/'DECISION.json',dict(primary=decision,all_primary_pass=all(x['wrong_symmetry_preferred'] for x in decision),
          precision_pass=sum(x['precision_pass'] for x in rows),contrasts=72,streams=48,prior_banks=12288,
          prior_particles=1228800,reference_integration_nodes=6*16*2**18,new_fits=0,new_observed_labels=0,
          no_claim_of_exact_QMC_reference_or_new_calibration_method=True))
    files=list((OUT/'reference').glob('*.json'))+list((OUT/'streams').glob('*.json'))+[OUT/'PRIMARY.csv',OUT/'STREAM_SCORES.csv',OUT/'DECISION.json']
    write(OUT/'RESULTS_FREEZE.json',dict(time_utc=now(),files={p.relative_to(ROOT).as_posix():sha(p) for p in files}))
    write(OUT/'STATE.json',dict(status='complete',time_utc=now(),active_jobs=[]),False)
def run():
    checkfreeze();assert read(OUT/'STATE.json')['status']=='frozen','No silent restart'
    smoke()
    write(OUT/'STATE.json',dict(status='running_reference',pid=os.getpid(),time_utc=now()),False)
    for d in CFG['d']:
        for idx,profile in enumerate(CFG['profiles']):reference(profile,d,idx)
    for d in CFG['d']:
        for idx,profile in enumerate(CFG['profiles']):
            for stream in range(8):
                write(OUT/'STATE.json',dict(status='running_banks',d=d,profile=profile,stream=stream,pid=os.getpid(),time_utc=now()),False)
                bank_stream(profile,d,idx,stream)
    collect()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','run']);args=p.parse_args()
    try:
        if args.phase=='freeze':freeze()
        else:run()
    except Exception as ex:
        if (OUT/'FREEZE.json').exists():
            write(OUT/f'FAILURE_{time.time_ns()}.json',dict(time_utc=now(),phase=args.phase,error=repr(ex),traceback=traceback.format_exc()))
            write(OUT/'STATE.json',dict(status='stopped_failure',time_utc=now(),active_jobs=[]),False)
        raise
