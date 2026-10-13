"""Fixed-candidate raw population score profiles; no data fitting or teacher sampling."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,sys,time,traceback
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
import numpy as np
import scipy
from scipy.special import logsumexp
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from script import covariance_boundary_numeric as core
DOC=ROOT/'docs/08_research_研究工作/相关结构补偿与发射噪声_20261001'
OUT=ROOT/'artifacts/covariance_boundary/PROFILE';NUM=ROOT/'artifacts/covariance_boundary/NUMERIC'
MOM=ROOT/'artifacts/objective_strengthening/N/MOMENTS.json'
BASE='6fcb333f7a0be02f568bbaccf38d28af2de407fd'
NAMES=['teacher','fixedRfloor','comp','half','over','independent']
CFG=dict(dimensions=[16,64],streams=8,banks=2048,seed=9516100100,bank_chunk=8,
    K_rule='[1,10,d]',names=NAMES,alpha=.05,family=16,SE_gate=.003,score_margin=.001,
    risk_margin=.001,formula_gate=1e-10,K1_gate=1e-12,mean_zero=True,
    nu=.1,Lambda=121.6,prefix_banks_saved=4)
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return Path(p).relative_to(ROOT).as_posix()
def write(p,x,replace=False):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w' if replace else 'x',encoding='utf-8',newline='\n') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def now():return datetime.now(timezone.utc).isoformat()
def verify(m):
    for n,h in m.items():assert sha(ROOT/n)==h,n
def psi(d,params):
    old=read(MOM);assert old['qualified'];values=old['gh'][-1]['values']
    return np.array([d/2*sum(values[str(float(s))]['ell'] for s in p['scales']) for p in params])

def scores(eta,prob,Ks):
    b,k,d=eta.shape;m=d//2
    one=np.concatenate([np.zeros((b,k,1)),np.cumsum(eta[:,:,:m],axis=-1)],axis=-1)
    two=np.concatenate([np.zeros((b,k,1)),np.cumsum(eta[:,:,m:],axis=-1)],axis=-1)
    logp=one[:,:,:,None]+two[:,:,None,:]-np.logaddexp(0.,eta).sum(-1)[:,:,None,None]
    out=np.empty((b,len(Ks)));raw=np.empty_like(out);base=np.empty_like(out)
    p=np.asarray(prob)
    for j,K in enumerate(Ks):
        loss=-logsumexp(logp[:,:K],axis=1)+np.log(K)
        single=-logp[:,:K].mean(1)
        raw[:,j]=(loss*p).sum((1,2));base[:,j]=(single*p).sum((1,2))
        out[:,j]=((loss-single)*p).sum((1,2))
        if K==1:assert np.max(abs(out[:,j]))<=CFG['K1_gate']
    assert np.isfinite(out).all() and np.max(out)<=1e-12
    return out,raw,base,logp.nbytes

def smoke():
    eta=(np.arange(2*4*4).reshape(2,4,4)-13)/9
    p=np.arange(1,10,dtype=float).reshape(3,3);p/=p.sum()
    out,raw,base,_=scores(eta,p,[1,3,4]);direct=np.zeros_like(raw);single=np.zeros_like(raw)
    for a in range(3):
        for b in range(3):
            y=np.r_[np.ones(a),-np.ones(2-a),np.ones(b),-np.ones(2-b)]
            lp=-np.logaddexp(0.,-eta*y).sum(-1)
            for j,K in enumerate([1,3,4]):
                direct[:,j]+=p[a,b]*(-logsumexp(lp[:,:K],axis=1)+np.log(K))
                single[:,j]+=p[a,b]*(-lp[:,:K].mean(1))
    error=max(float(np.max(abs(direct-raw))),float(np.max(abs(single-base))),float(np.max(abs(direct-single-out))))
    assert error<CFG['formula_gate']
    write(OUT/'SMOKE.json',dict(passed=True,formula_error=error,scientific_RNG_calls=0,
        tests='canonical prefix versus explicit full-coordinate labels, control variate, K1'))

def unit(d,s):
    ref=core.reference(d);assert ref['names']==NAMES
    params=ref['params'];p=np.asarray(ref['pmf'][0]);assert abs(p.sum()-1)<1e-12
    for q in params:
        scales=np.asarray(q['scales']);lam=np.asarray(q['lambda'])
        assert scales.shape==lam.shape==(2,) and np.max(abs(lam))<1
    Ps=psi(d,params);Ks=[1,10,d];B=CFG['banks'];sd=CFG['seed']+100000*d+1000*s
    gc=np.random.Generator(np.random.PCG64(sd));ge=np.random.Generator(np.random.PCG64(sd+1))
    initial=[gc.bit_generator.state,ge.bit_generator.state]
    corrected=np.empty((B,3,6));uncontrolled=np.empty_like(corrected);baseline=np.empty_like(corrected)
    prefix_c=prefix_e=None;peak=0;started=time.perf_counter()
    for begin in range(0,B,CFG['bank_chunk']):
        size=min(CFG['bank_chunk'],B-begin);common=gc.standard_normal((size,d));private=ge.standard_normal((size,d,d))
        if begin==0:prefix_c=common[:4].copy();prefix_e=private[:4].copy()
        for j,q in enumerate(params):
            scales=np.repeat(q['scales'],d//2);lam=np.repeat(q['lambda'],d//2)
            eta=scales*(common[:,:,None]*lam+private*np.sqrt(1-lam**2))
            gap,raw,base,mem=scores(eta,p,Ks);peak=max(peak,mem)
            corrected[begin:begin+size,:,j]=Ps[j]+gap
            uncontrolled[begin:begin+size,:,j]=raw;baseline[begin:begin+size,:,j]=base
    assert np.max(abs(corrected[:,0]-Ps))<=CFG['K1_gate']
    path=OUT/'local'/f'd{d}_s{s}.npz';path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path,scores=corrected,raw=uncontrolled,single=baseline,
        common_prefix=prefix_c,private_prefix=prefix_e)
    row=dict(d=d,stream=s,Ks=Ks,names=NAMES,psi=Ps.tolist(),means=corrected.mean(0).tolist(),
        raw_means=uncontrolled.mean(0).tolist(),single_means=baseline.mean(0).tolist(),
        seed_common=sd,seed_private=sd+1,RNG_initial=initial,RNG_final=[gc.bit_generator.state,ge.bit_generator.state],
        banks=B,unique_base_prior_vectors=B*d,base_gaussian_scalars=B*d*(d+1),
        transformed_candidate_prior_vectors=6*B*d,
        count_particle_candidate_terms=6*B*d*(d//2+1)**2,
        logsumexp_terms=6*B*sum(Ks)*(d//2+1)**2,
        peak_working_array_bytes=peak,seconds=time.perf_counter()-started,
        array=rel(path),array_sha256=sha(path),all_prior_inputs_saved=False,
        prefix_prior_inputs_saved=4*d,control_variate_preserves_population_target=True)
    write(OUT/'streams'/f'd{d}_s{s}.json',row)
    print(json.dumps(dict(d=d,stream=s,seconds=row['seconds'],
        over_minus_comp=(corrected[:,1:,4]-corrected[:,1:,2]).mean(0).tolist())),flush=True)
    return row

def interval(v):
    v=np.asarray(v);mean=float(v.mean());se=float(v.std(ddof=1)/np.sqrt(8))
    rad=float(t.ppf(1-CFG['alpha']/(2*CFG['family']),7))*se
    return dict(values=v.tolist(),mean=mean,SE=se,lower=mean-rad,upper=mean+rad,
        precision_pass=bool(se<=CFG['SE_gate']),t_is_approximation=True)

def collect(rows):
    contrasts=[];main=[];references={}
    comparisons=[('comp_minus_fixedRfloor',2,1),('over_minus_comp',4,2),('half_minus_comp',3,2),('independent_minus_comp',5,2)]
    for d in CFG['dimensions']:
        rs=[r for r in rows if r['d']==d];assert len(rs)==8
        ref=core.reference(d);risk=list(map(float,ref['risk_to_teacher']));references[d]=dict(names=NAMES,risk_to_teacher=risk)
        means=np.array([r['means'] for r in rs])
        for ki,K in enumerate([1,10,d]):
            if K==1:continue
            for name,left,right in comparisons:
                stat=interval(means[:,ki,left]-means[:,ki,right])
                row=dict(d=d,K=K,comparison=name,**stat,
                    reference_risk_difference=risk[left]-risk[right])
                contrasts.append(row)
                if name=='over_minus_comp':
                    main.append(dict(d=d,K=K,passed=bool(stat['precision_pass'] and stat['upper'] < -CFG['score_margin'] and risk[left]-risk[right]>CFG['risk_margin']),
                        score_upper=stat['upper'],risk_difference=risk[left]-risk[right],
                        claim='only the prespecified pair, not all covariance optimizers'))
    assert len(contrasts)==16 and len(main)==4
    fields=['banks','unique_base_prior_vectors','base_gaussian_scalars','transformed_candidate_prior_vectors','count_particle_candidate_terms','logsumexp_terms']
    counts={f:sum(r[f] for r in rows) for f in fields}
    assert counts['banks']==32768 and counts['unique_base_prior_vectors']==1310720
    counts.update(independent_streams=16,new_teacher_labels=0,new_fits=0,new_neural_models=0,
        new_integrals=0,candidates=6,analytic_K1_scores=True)
    write(OUT/'CONTRASTS.json',contrasts);write(OUT/'REFERENCE_RISKS.json',references)
    write(OUT/'DECISION.json',dict(primary=main,precision_pass=sum(x['precision_pass'] for x in contrasts),total=16,
        continuous_global_minimum_claimed=False,optimizer_selected_candidate_claimed=False,
        numerical_reference_conditional=True,all_failures_preserved=True))
    write(OUT/'COUNTS.json',counts);return main

def freeze():
    assert not OUT.exists();refseal=read(NUM/'RESULTS_FREEZE.json');verify(refseal['files'])
    assert read(NUM/'STATE.json')['status']=='complete'
    for d in CFG['dimensions']:
        r=core.reference(d);assert r['names']==NAMES
        assert np.asarray(r['pmf']).shape==(6,d//2+1,d//2+1)
        for q in r['params']:
            s=np.repeat(q['scales'],d//2);lam=np.repeat(q['lambda'],d//2)
            covariance=np.diag(s*s*(1-lam*lam))+np.outer(s*lam,s*lam)
            spectrum=np.linalg.eigvalsh(covariance)
            assert spectrum[0]>=CFG['nu']-1e-10 and spectrum[-1]<=CFG['Lambda']+1e-10
    OUT.mkdir(parents=True)
    (OUT/'HANDOFF_INPUT.md').write_bytes((ROOT/'HANDOFF.md').read_bytes())
    write(OUT/'CONFIG.json',CFG);write(OUT/'ENVIRONMENT.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,threads=2))
    paths=[Path(__file__),ROOT/'script/covariance_boundary_numeric.py',DOC/'PLAN_PROFILE.md',DOC/'PROFILE_IDENTITY.md',
        MOM,NUM/'RESULTS_FREEZE.json',OUT/'CONFIG.json',OUT/'ENVIRONMENT.json',OUT/'HANDOFF_INPUT.md']
    write(OUT/'FREEZE.json',dict(utc=now(),baseline_commit=BASE,before_scientific_RNG=True,
        files={rel(p):sha(p) for p in paths},reference_files=refseal['files']))
    write(OUT/'STATE.json',dict(status='frozen',active_jobs=[]));print('PROFILE frozen',flush=True)

def run():
    f=read(OUT/'FREEZE.json');verify(f['files']);verify(f['reference_files'])
    assert read(OUT/'STATE.json')['status']=='frozen';smoke()
    write(OUT/'START.json',dict(utc=now(),source_freeze_sha256=sha(OUT/'FREEZE.json')))
    rows=[]
    for d in CFG['dimensions']:
        for s in range(8):
            write(OUT/'STATE.json',dict(status='running',d=d,stream=s,pid=os.getpid()),True)
            rows.append(unit(d,s))
    main=collect(rows)
    files=[p for p in OUT.rglob('*') if p.is_file() and 'local' not in p.parts and p.name!='STATE.json']
    write(OUT/'RESULTS_FREEZE.json',dict(utc=now(),source_freeze_sha256=sha(OUT/'FREEZE.json'),
        files={rel(p):sha(p) for p in files},local_arrays={rel(p):sha(p) for p in (OUT/'local').glob('*.npz')}))
    write(OUT/'STATE.json',dict(status='complete',active_jobs=[]),True)
    print(json.dumps(dict(complete=True,primary=main)),flush=True)

def check(workspace):
    source=read(OUT/'FREEZE.json');verify(source['files']);verify(source['reference_files'])
    result=read(OUT/'RESULTS_FREEZE.json');verify(result['files'])
    assert result['source_freeze_sha256']==sha(OUT/'FREEZE.json') and read(OUT/'STATE.json')['status']=='complete'
    if workspace:verify(result['local_arrays'])
    print(json.dumps(dict(passed=True,workspace=workspace,no_science_reexecution=True)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','run','check']);p.add_argument('--workspace',action='store_true');a=p.parse_args()
    try:{'freeze':freeze,'run':run,'check':lambda:check(a.workspace)}[a.phase]()
    except Exception as e:
        if a.phase=='run' and OUT.exists() and not (OUT/'RESULTS_FREEZE.json').exists():
            write(OUT/f'FAILURE_{time.time_ns()}.json',dict(error=repr(e),traceback=traceback.format_exc()))
            write(OUT/'STATE.json',dict(status='stopped_failure',active_jobs=[]),True)
        raise
