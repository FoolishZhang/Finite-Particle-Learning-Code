"""Frozen exact-reference A and finite-data B experiments; no legacy mutation."""
import os
for _key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[_key]='2'
from pathlib import Path
import argparse, hashlib, json, math, platform, subprocess, sys, time
from datetime import datetime, timezone
import numpy as np
from scipy.special import logsumexp, expit
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v3_query'
DOC=ROOT/'docs/08_research_研究工作/v3表示与条件查询_20260930'
FAMILIES=['product','cardinality','ising','linear_ar','count_ar']
BASE='e24f8bedf5844d51ca6e6355bb1ee0f191991f43'
HANDOFF='1138fe2f2eb358e1af917a5380221517543aaa25a9d8c80d71b32c8c6b284b24'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p): return Path(p).relative_to(ROOT).as_posix()
def dump(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def read(p):return json.loads(Path(p).read_text('utf-8'))
def journal(package,row):
    with (OUT/f'{package}_FIT_LOG.jsonl').open('a',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
def normalize(lp):return np.exp(lp-logsumexp(lp))
def state_matrix(d=8):return ((np.arange(1<<d)[:,None]>>np.arange(d))&1).astype(float)
def kl(p,q):return float(np.sum(p*(np.log(p)-np.log(q))))
def teachers():
    c=state_matrix();s=2*c-1;n=c.sum(1).astype(int)
    p=normalize(.7*np.sum(s[:,::2]*s[:,1::2],axis=1))
    pn=np.bincount(n,weights=p,minlength=9)
    return dict(indep=normalize(c@np.array([-.8,-.4,0,.4,.8,.2,-.2,.6])),
        count=normalize(-.35*(n-4)**2),allocation=p,
        allocation_sym=pn[n]/np.array([math.comb(8,int(j)) for j in n]),
        parity=normalize(.8*(-1.)**n))
def conditional_teachers():
    c=state_matrix();s=2*c-1;n=c.sum(1)
    beta=np.array([.25,.35,.45,.55,.65,.75,.85,.95])
    dep=[];ind=[]
    for x in [-1,1]:
        p=normalize(-.35*(n-4)**2+x*(s@beta))
        m=p@c
        dep.append(p);ind.append(normalize(c@np.log(m)+(1-c)@np.log1p(-m)))
    return dict(dependent=dep,independent=ind)

def freeze(package):
    OUT.mkdir(parents=True,exist_ok=True)
    assert sha(ROOT/'HANDOFF.md')==HANDOFF
    snapshot=OUT/'HANDOFF_INPUT.md'
    if not snapshot.exists():snapshot.write_bytes((ROOT/'HANDOFF.md').read_bytes())
    assert sha(snapshot)==HANDOFF
    paths=[Path(__file__).resolve(),ROOT/'script/v3_query_core.py',DOC/f'PLAN_{package}.md',snapshot]
    assert all(p.exists() for p in paths)
    dump(OUT/f'{package}_FREEZE.json',dict(package=package,base_commit=BASE,
        frozen_utc=datetime.now(timezone.utc).isoformat(),scientific_runs_started=False,
        inputs={rel(p):sha(p) for p in paths},python=sys.version,numpy=np.__version__,platform=platform.platform(),
        handoff_mode='reference; concise framework explicitly accepted by user',handoff_sha256=HANDOFF))
    print(f'{package} frozen',flush=True)

def verify_freeze(package):
    f=read(OUT/f'{package}_FREEZE.json')
    for name,h in f['inputs'].items():assert sha(ROOT/name)==h,name
    return f

def eval_fit(p,family):
    from v3_query_core import fit,query_metrics
    start=time.perf_counter();r=fit(p,family);elapsed=time.perf_counter()-start
    q=np.asarray(r['q']);assert q.shape==p.shape and np.min(q)>0 and abs(q.sum()-1)<1e-11
    return q,dict(family=family,params=r['params'],objective=float(r['objective']),
        gap=float(r['gap']),qualified=bool(r['gap']<=1e-8),fit_seconds=elapsed,
        optimizer_receipt=r['optimizer_receipt'])

def risk(p,q):
    from v3_query_core import query_metrics
    return dict(joint_KL=kl(p,q),queries={str(k):query_metrics(p,q,k) for k in [3,7]},
                marginal_max_error=float(np.max(abs(p@state_matrix()-q@state_matrix()))))

def run_a():
    verify_freeze('A');assert not (OUT/'A_RESULTS.json').exists()
    dump(OUT/'A_STARTED.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),freeze_sha256=sha(OUT/'A_FREEZE.json')))
    rows=[];pmfs={};index=np.arange(256)
    for name,p0 in teachers().items():
        for code,mask in [('native',0),('flip21',21)]:
            p=p0[index^mask];pmfs[f'truth_{name}_{code}']=p
            for family in FAMILIES:
                q,row=eval_fit(p,family);row.update(teacher=name,encoding=code,**risk(p,q))
                journal('A',row)
                rows.append(row);pmfs[f'fit_{name}_{code}_{family}']=q
                print('A',name,code,family,'KL',row['joint_KL'],'gap',row['gap'],flush=True)
    out=OUT/'A_PMFS.npz';np.savez_compressed(out,**pmfs)
    dump(OUT/'A_PMFS.json',{name:value.tolist() for name,value in pmfs.items()})
    dump(OUT/'A_RESULTS.json',dict(rows=rows,population_fits=len(rows),random_observations=0,
        qualified=sum(r['qualified'] for r in rows),array_file=rel(out),array_sha256=sha(out),
        freeze_sha256=sha(OUT/'A_FREEZE.json')))

def run_b():
    verify_freeze('B');assert not (OUT/'B_RESULTS.json').exists()
    dump(OUT/'B_STARTED.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),freeze_sha256=sha(OUT/'B_FREEZE.json')))
    rows=[];pop=[];pooled=[];arrays={};index=np.arange(256)
    for w,(world,ps) in enumerate(conditional_teachers().items()):
        for code,mask in [('native',0),('flip21',21)]:
            for x,p0 in enumerate(ps):
                p=p0[index^mask]
                for family in FAMILIES:
                    q,row=eval_fit(p,family);row.update(world=world,encoding=code,x=x,**risk(p,q));pop.append(row);journal('B',dict(kind='population',**row))
            pp=sum(ps)/2;pp=pp[index^mask]
            for family in FAMILIES:
                q,row=eval_fit(pp,family);row.update(world=world,encoding=code,**risk(pp,q));pooled.append(row);journal('B',dict(kind='pooled',**row))
        for rep in range(16):
            counts=[]
            for x,p in enumerate(ps):
                rng=np.random.default_rng(2026093000+10000*w+100*rep+x)
                counts.append(rng.multinomial(2048,p))
                arrays[f'counts_{world}_{rep}_{x}']=counts[-1]
            for code,mask in [('native',0),('flip21',21)]:
                for x,p0 in enumerate(ps):
                    p=p0[index^mask];target=(counts[x][index^mask]+1/256)/2049
                    for family in FAMILIES:
                        q,row=eval_fit(target,family)
                        row.update(world=world,encoding=code,x=x,rep=rep,**risk(p,q));rows.append(row)
                        journal('B',dict(kind='finite',**row))
                        arrays[f'q_{world}_{rep}_{code}_{x}_{family}']=q
            print('B',world,'rep',rep,'fits',len(rows),flush=True)
    contrasts=[];alpha=.05/32;critical=float(t.ppf(1-alpha/2,15))
    def metric(r,k):return r['queries'][str(k)]['conditional_excess_KL']
    for world in conditional_teachers():
        for code in ['native','flip21']:
            for family in FAMILIES[1:]:
                for k in [3,7]:
                    vals=[]
                    contrast_qualified=True
                    for rep in range(16):
                        subset=[r for r in rows if r['world']==world and r['encoding']==code and r['rep']==rep]
                        contrast_qualified=contrast_qualified and all(r['qualified'] for r in subset if r['family'] in ['product',family])
                        vals.append(sum(metric(r,k) for r in subset if r['family']=='product')/2-
                                    sum(metric(r,k) for r in subset if r['family']==family)/2)
                    mean=float(np.mean(vals));se=float(np.std(vals,ddof=1)/4)
                    contrasts.append(dict(world=world,encoding=code,family=family,k=k,
                        product_minus_model=mean,se=se,simultaneous_t_interval=[mean-critical*se,mean+critical*se],
                        contrast_qualified=contrast_qualified,
                        practical_gain=bool(contrast_qualified and mean-critical*se>1e-3),replicate_values=vals))
    out=OUT/'B_ARRAYS.npz';np.savez_compressed(out,**arrays)
    dump(OUT/'B_COUNTS.json',{name:value.tolist() for name,value in arrays.items() if name.startswith('counts_')})
    dump(OUT/'B_RESULTS.json',dict(rows=rows,population_rows=pop,pooled_rows=pooled,contrasts=contrasts,
        finite_fits=len(rows),population_fits=len(pop)+len(pooled),distinct_training_observations=2*16*2*2048,
        fitted_conditional_distributions=len(rows),fitted_joint_models_with_X=len(rows)//2,
        qualified=sum(r['qualified'] for r in rows+pop+pooled),total_fits=len(rows+pop+pooled),
        array_file=rel(out),array_sha256=sha(out),freeze_sha256=sha(OUT/'B_FREEZE.json'),
        inference='32 simultaneous Bonferroni t development intervals; normal approximation across 16 training replicates; not finite-sample theorem'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freezeA','runA','freezeB','runB'])
    action=parser.parse_args().action
    if action.startswith('freeze'):freeze(action[-1])
    elif action=='runA':run_a()
    else:run_b()
