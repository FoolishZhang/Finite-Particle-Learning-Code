"""Separately frozen larger-K comparison for the fixed comp/over pair."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,sys,time,traceback
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
import numpy as np
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from script import covariance_boundary_numeric as core
from script.covariance_boundary_profile import scores,psi
DOC=ROOT/'docs/08_research_研究工作/相关结构补偿与发射噪声_20261001'
OUT=ROOT/'artifacts/covariance_boundary/HIGHK';PARENT=OUT.parent/'PROFILE';NUM=OUT.parent/'NUMERIC'
CFG=dict(dimensions=[16,64],Ks=[256,4096],streams=8,banks=64,seed=10616100100,
    names=['comp','over'],alpha=.05,family=4,SE_gate=.003,score_margin=.001,risk_margin=.001)
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return Path(p).relative_to(ROOT).as_posix()
def write(p,x,replace=False):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w' if replace else 'x',encoding='utf-8',newline='\n') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def verify(m):
    for n,h in m.items():assert sha(ROOT/n)==h,n
def now():return datetime.now(timezone.utc).isoformat()
def freeze():
    assert not OUT.exists() and read(PARENT/'STATE.json')['status']=='complete'
    verify(read(PARENT/'RESULTS_FREEZE.json')['files']);OUT.mkdir(parents=True)
    (OUT/'HANDOFF_INPUT.md').write_bytes((ROOT/'HANDOFF.md').read_bytes());write(OUT/'CONFIG.json',CFG)
    files=[Path(__file__),ROOT/'script/covariance_boundary_profile.py',ROOT/'script/covariance_boundary_numeric.py',
        DOC/'PLAN_HIGHK.md',DOC/'PROFILE_IDENTITY.md',PARENT/'RESULTS_FREEZE.json',NUM/'RESULTS_FREEZE.json',
        ROOT/'artifacts/objective_strengthening/N/MOMENTS.json',OUT/'CONFIG.json',OUT/'HANDOFF_INPUT.md']
    write(OUT/'FREEZE.json',dict(utc=now(),before_scientific_RNG=True,
        proposed_after_original_PROFILE_results=True,files={rel(p):sha(p) for p in files},
        reference_files=read(NUM/'RESULTS_FREEZE.json')['files']))
    write(OUT/'STATE.json',dict(status='frozen',active_jobs=[]))
def unit(d,s):
    ref=core.reference(d);ids=[ref['names'].index(n) for n in CFG['names']];params=[ref['params'][j] for j in ids]
    p=np.asarray(ref['pmf'][0]);Ps=psi(d,params);B=CFG['banks'];K=max(CFG['Ks'])
    sd=CFG['seed']+100000*d+1000*s
    gc=np.random.Generator(np.random.PCG64(sd));ge=np.random.Generator(np.random.PCG64(sd+1))
    initial=[gc.bit_generator.state,ge.bit_generator.state];values=np.empty((B,2,2));peak=0;began=time.perf_counter()
    for b in range(B):
        common=gc.standard_normal((1,K));private=ge.standard_normal((1,K,d))
        if b==0:c0=common.copy();e0=private.copy()
        for j,q in enumerate(params):
            scale=np.repeat(q['scales'],d//2);lam=np.repeat(q['lambda'],d//2)
            eta=scale*(common[:,:,None]*lam+private*np.sqrt(1-lam*lam))
            gap,_,_,mem=scores(eta,p,CFG['Ks']);values[b,:,j]=Ps[j]+gap[0];peak=max(peak,mem)
    path=OUT/'local'/f'd{d}_s{s}.npz';path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path,scores=values,common_prefix=c0,private_prefix=e0)
    record=dict(d=d,stream=s,Ks=CFG['Ks'],names=CFG['names'],means=values.mean(0).tolist(),
        seed_common=sd,seed_private=sd+1,RNG_initial=initial,RNG_final=[gc.bit_generator.state,ge.bit_generator.state],
        banks=B,unique_base_prior_vectors=B*K,base_gaussian_scalars=B*K*(d+1),
        count_particle_candidate_terms=2*B*K*(d//2+1)**2,
        logsumexp_terms=2*B*sum(CFG['Ks'])*(d//2+1)**2,
        seconds=time.perf_counter()-began,peak_working_array_bytes=peak,array=rel(path),array_sha256=sha(path),
        prefix_banks_saved=1,all_prior_inputs_saved=False)
    write(OUT/'streams'/f'd{d}_s{s}.json',record)
    print(json.dumps(dict(d=d,stream=s,seconds=record['seconds'],over_minus_comp=(values[:,:,1]-values[:,:,0]).mean(0).tolist())),flush=True)
    return record
def collect(rows):
    contrasts=[];main=[]
    for d in CFG['dimensions']:
        x=np.array([r['means'] for r in rows if r['d']==d]);ref=core.reference(d)
        risk=float(ref['risk_to_teacher'][4]-ref['risk_to_teacher'][2])
        for ki,K in enumerate(CFG['Ks']):
            v=x[:,ki,1]-x[:,ki,0];mean=float(v.mean());se=float(v.std(ddof=1)/np.sqrt(8));rad=float(t.ppf(1-CFG['alpha']/(2*CFG['family']),7))*se
            row=dict(d=d,K=K,comparison='over_minus_comp',values=v.tolist(),mean=mean,SE=se,lower=mean-rad,upper=mean+rad,
                precision_pass=bool(se<=CFG['SE_gate']),reference_risk_difference=risk,t_is_approximation=True)
            contrasts.append(row)
            if K==4096:main.append(dict(d=d,K=K,passed=bool(row['precision_pass'] and row['lower']>.001 and risk>.001),score_lower=row['lower'],risk_difference=risk))
    write(OUT/'CONTRASTS.json',contrasts);write(OUT/'DECISION.json',dict(primary=main,precision_pass=sum(r['precision_pass'] for r in contrasts),total=4,
        independent_followup_not_original_PROFILE=True,all_failures_preserved=True,global_minimum_claim=False))
    count={f:sum(r[f] for r in rows) for f in ['banks','unique_base_prior_vectors','base_gaussian_scalars','count_particle_candidate_terms','logsumexp_terms']}
    assert count['banks']==1024 and count['unique_base_prior_vectors']==4194304
    count.update(independent_streams=16,new_teacher_labels=0,new_fits=0,new_integrals=0,new_neural_models=0)
    write(OUT/'COUNTS.json',count);return main
def run():
    f=read(OUT/'FREEZE.json');verify(f['files']);verify(f['reference_files']);assert read(OUT/'STATE.json')['status']=='frozen'
    write(OUT/'START.json',dict(utc=now(),source_freeze_sha256=sha(OUT/'FREEZE.json')));rows=[]
    for d in CFG['dimensions']:
        for s in range(8):
            write(OUT/'STATE.json',dict(status='running',d=d,stream=s,pid=os.getpid()),True);rows.append(unit(d,s))
    main=collect(rows);paths=[p for p in OUT.rglob('*') if p.is_file() and 'local' not in p.parts and p.name!='STATE.json']
    write(OUT/'RESULTS_FREEZE.json',dict(utc=now(),source_freeze_sha256=sha(OUT/'FREEZE.json'),files={rel(p):sha(p) for p in paths},
        local_arrays={rel(p):sha(p) for p in (OUT/'local').glob('*.npz')}))
    write(OUT/'STATE.json',dict(status='complete',active_jobs=[]),True);print(json.dumps(dict(complete=True,primary=main)),flush=True)
def check(workspace):
    f=read(OUT/'FREEZE.json');verify(f['files']);verify(f['reference_files']);r=read(OUT/'RESULTS_FREEZE.json');verify(r['files'])
    assert r['source_freeze_sha256']==sha(OUT/'FREEZE.json') and read(OUT/'STATE.json')['status']=='complete'
    if workspace:verify(r['local_arrays'])
    print(json.dumps(dict(passed=True,workspace=workspace,no_science_reexecution=True)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','run','check']);p.add_argument('--workspace',action='store_true');a=p.parse_args()
    try:{'freeze':freeze,'run':run,'check':lambda:check(a.workspace)}[a.phase]()
    except Exception as e:
        if a.phase=='run' and OUT.exists() and not (OUT/'RESULTS_FREEZE.json').exists():
            write(OUT/f'FAILURE_{time.time_ns()}.json',dict(error=repr(e),traceback=traceback.format_exc()))
            write(OUT/'STATE.json',dict(status='stopped_failure',active_jobs=[]),True)
        raise
