"""A: frozen population/finite-data structure selection; no old science edits."""
import os
for _k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[_k]='2'
from pathlib import Path
from datetime import datetime,timezone
from functools import lru_cache
import argparse,hashlib,itertools,json,math,platform,sys,time,traceback
import numpy as np
import scipy
from scipy.special import logsumexp
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/v3_select/A'
DOC=ROOT/'docs/08_research_研究工作/v3结构选择与查询诊断_20260930'
METHODS=['product','CAR','RC_MI','RC_residual','RC_gain','RC_agreement','ChowLiu','Ising','histogram']
H='1138fe2f2eb358e1af917a5380221517543aaa25a9d8c80d71b32c8c6b284b24'
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return Path(p).relative_to(ROOT).as_posix()
def read(p):return json.loads(Path(p).read_text('utf-8'))
def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [serial(v) for v in x]
    return x
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(serial(obj),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def append(row):
    with (OUT/'FIT_LOG.jsonl').open('a',encoding='utf-8',newline='\n') as f:f.write(json.dumps(serial(row),ensure_ascii=False,allow_nan=False)+'\n')
@lru_cache(None)
def states(d=8):return ((np.arange(1<<d)[:,None]>>np.arange(d))&1).astype(float)
def normalize(z):return np.exp(z-logsumexp(z))
def teachers():
    c=states();s=2*c-1;n=c.sum(1)
    h=np.array([-.6,.4,-.3,.7,-.5,.2,.6,-.4]);edges=[(0,5),(1,6),(2,4),(3,7)]
    pair=s@h+sum(J*s[:,i]*s[:,j] for (i,j),J in zip(edges,[.35,.5,.65,.8]))
    contrast=np.prod(c[:,::2]-c[:,1::2],axis=1)
    return dict(indep=normalize(c@np.array([-.8,-.4,0,.4,.8,.2,-.2,.6])),count=normalize(-.35*(n-4)**2),
                pair=normalize(pair),pair_count=normalize(pair-.35*(n-4)**2),blindspot=(1+.9*contrast)/256)
@lru_cache(None)
def query_labels(d,k):
    c=states(d).astype(int);labels=[]
    for obs in itertools.combinations(range(d),k):
        e=c[:,obs]@(2**np.arange(k))
        for j in range(d):
            if j not in obs:labels.append(2*e+c[:,j])
    return labels
def score_table(q,k):
    d=round(math.log2(len(q)));score=np.zeros(len(q))
    for labels in query_labels(d,k):
        masses=np.bincount(labels,weights=q,minlength=2**(k+1)).reshape(-1,2)
        lp=np.log(masses)-np.log(masses.sum(1)[:,None])
        score-=lp.ravel()[labels]
    return score/len(query_labels(d,k))
def risks(p,q,tables):
    return dict(joint_KL=float(p@(np.log(p)-np.log(q))),query7=float(p@(tables[7]-score_table(p,7))),
                query3=float(p@(tables[3]-score_table(p,3))))
def count_params(params):
    if 'theta' in params:return len(params['theta'])
    return 255 if params.get('family')=='histogram' else 15

def fit_models(p,uid):
    import v3_select_core as core
    records=[];cache={};models={};traces={}
    def call(kind,fn,*args):
        start=time.perf_counter();raw=fn(p,*args);elapsed=time.perf_counter()-start
        q=np.asarray(raw['q']);assert q.shape==(256,) and q.min()>0 and abs(q.sum()-1)<1e-11
        result=dict(id=f'{uid}/m{len(records):02}',fit_kind=kind,params=serial(raw['params']),
                    objective=float(raw['objective']),gap=float(raw['gap']),qualified=bool(raw['gap']<=1e-8),
                    parameter_count=count_params(raw['params']),fit_seconds=elapsed,
                    receipt=serial(raw.get('receipt',raw.get('optimizer_receipt',{}))))
        record=(q,result);records.append(record);append(result);return record
    models['product']=call('product',core.fit_product)
    rootnodes=[list(range(8))];car=call('RC',core.fit_rc,rootnodes);models['CAR']=car
    cache[tuple([tuple(range(8))])]=car
    for method,kind in [('RC_MI','mi'),('RC_residual','residual'),('RC_gain','gain')]:
        start=time.perf_counter();weights=core.edge_scores(p,kind,q=(car[0] if kind in ['residual','gain'] else None));score_s=time.perf_counter()-start
        start=time.perf_counter();pairs=core.matching(8,weights,b=4);match_s=time.perf_counter()-start
        nodes=[list(e) for e in pairs]+rootnodes;key=tuple(sorted(tuple(sorted(e)) for e in nodes))
        if key not in cache:cache[key]=call('RC',core.fit_rc,[list(e) for e in key])
        models[method]=cache[key]
        traces[method]=dict(kind=kind,weights=serial(weights),nodes=[list(e) for e in key],score_seconds=score_s,matching_seconds=match_s)
    start=time.perf_counter();nodes=core.agglomerative_tree(p);agg_s=time.perf_counter()-start
    key=tuple(sorted(tuple(sorted(e)) for e in nodes))
    if key not in cache:cache[key]=call('RC',core.fit_rc,[list(e) for e in key])
    models['RC_agreement']=cache[key];traces['RC_agreement']=dict(nodes=[list(e) for e in key],structure_seconds=agg_s)
    models['ChowLiu']=call('ChowLiu',core.fit_chow_liu);models['Ising']=call('Ising',core.fit_ising)
    models['histogram']=call('histogram',core.fit_histogram)
    for method in METHODS:
        q,r=models[method]
        traces.setdefault(method,{})['model_id']=r['id']
        traces[method]['pipeline_qualified']=r['qualified'] and (car[1]['qualified'] if method in ['RC_residual','RC_gain'] else True)
    return models,records,traces

def frozen_inputs():return [Path(__file__).resolve(),ROOT/'script/v3_select_core.py',ROOT/'script/v3_query_core.py',DOC/'PLAN_A.md',OUT/'HANDOFF_INPUT.md']
def freeze():
    OUT.mkdir(parents=True,exist_ok=True);assert sha(ROOT/'HANDOFF.md')==H
    snap=OUT/'HANDOFF_INPUT.md';assert not snap.exists();snap.write_bytes((ROOT/'HANDOFF.md').read_bytes())
    write(OUT/'FREEZE.json',dict(frozen_utc=now(),base_commit='391714d0b75bf3b65170a22d4c9c763e69255eb8',
        inputs={rel(p):sha(p) for p in frozen_inputs()},python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
        platform=platform.platform(),training_observations=655360,validation_observations=40960,scientific_started=False))
    print('A selection frozen',flush=True)
def checkfreeze():
    for name,h in read(OUT/'FREEZE.json')['inputs'].items():assert sha(ROOT/name)==h,name
def best(scores,models,field):
    val=min(scores[m][field] for m in METHODS)
    tied=[m for m in METHODS if scores[m][field]<=val+1e-12]
    return min(tied,key=lambda m:(models[m][1]['parameter_count'],METHODS.index(m)))
def select(models,traces,validation,tables):
    start=time.perf_counter();scores={m:dict(joint=float(validation@(-np.log(models[m][0]))),query=float(validation@tables[m][7])) for m in METHODS};scoretime=time.perf_counter()-start
    j=best(scores,models,'joint');q=best(scores,models,'query')
    bank_qual=all(traces[m]['pipeline_qualified'] for m in METHODS)
    return scores,j,q,bank_qual,scoretime
def run():
    checkfreeze();write(OUT/'START.json',dict(first_scientific_utc=now(),freeze_sha256=sha(OUT/'FREEZE.json')))
    fits=[];population=[];units=[];counts_public={};arrays={};truths=teachers();ix=np.arange(256);spin=2*states()-1
    for ti,(teacher,p0) in enumerate(truths.items()):
        for code,mask in [('native',0),('flip21',21)]:
            p=p0[ix^mask];uid=f'population/{teacher}/{code}'
            models,recs,traces=fit_models(p,uid)
            rows={}
            for m,(q,record) in models.items():
                tables={k:score_table(q,k) for k in [3,7]};rows[m]=dict(model_id=record['id'],qualified=traces[m]['pipeline_qualified'],**risks(p,q,tables))
            fits.extend(r for _,r in recs);population.append(dict(teacher=teacher,encoding=code,candidates=rows,structure=traces))
        for rep in range(16):
            seed=5026093000+1000000*ti+10000*rep
            small=np.random.default_rng(seed).multinomial(512,p0);large=small+np.random.default_rng(seed+1).multinomial(7680,p0)
            vs=np.random.default_rng(seed+2).multinomial(128,p0);vl=vs+np.random.default_rng(seed+3).multinomial(384,p0)
            for name,v in [('train512',small),('train8192',large),('val128',vs),('val512',vl)]:counts_public[f'{teacher}/{rep}/{name}']=v.tolist()
            for code,mask in [('native',0),('flip21',21)]:
                p=p0[ix^mask]
                for n,count in [(512,small),(8192,large)]:
                    target=(count[ix^mask]+1/256)/(n+1);uid=f'finite/{teacher}/{code}/{rep}/{n}'
                    models,recs,traces=fit_models(target,uid)
                    tables={m:{k:score_table(q,k) for k in [3,7]} for m,(q,_) in models.items()}
                    selections=[]
                    for nv,vc in [(128,vs),(512,vl)]:
                        validation=vc[ix^mask]/nv
                        scores,j,q,bqual,scoretime=select(models,traces,validation,tables)
                        alarm_start=time.perf_counter();vm=spin.T@(validation[:,None]*spin);qm=spin.T@(models['CAR'][0][:,None]*spin)
                        residual=max(abs(vm[i,j]-qm[i,j]) for i in range(8) for j in range(i+1,8))
                        tau=math.sqrt(2*math.log(2*28*8/.05)/nv);alarm=bool(residual>tau);alarmtime=time.perf_counter()-alarm_start
                        chosen='RC_gain' if alarm else 'CAR';cqual=traces[chosen]['pipeline_qualified'] and traces['CAR']['pipeline_qualified']
                        selections.append(dict(nval=nv,scores=scores,joint_holdout=dict(selected=j,qualified=bqual),
                            query_holdout=dict(selected=q,qualified=bqual),moment_alarm=dict(selected=chosen,qualified=cqual,
                            alarm=alarm,max_pair_discrepancy=float(residual),threshold=tau),validation_score_seconds=scoretime,alarm_seconds=alarmtime))
                    # Teacher risk is used only after all model and validation decisions above.
                    rows={m:dict(model_id=r['id'],qualified=traces[m]['pipeline_qualified'],**risks(p,q,tables[m])) for m,(q,r) in models.items()}
                    for selection in selections:
                        for policy in ['joint_holdout','query_holdout','moment_alarm']:
                            choice=selection[policy];choice['risk']=dict(rows[choice['selected']])
                    units.append(dict(teacher=teacher,encoding=code,rep=rep,ntrain=n,candidates=rows,structure=traces,selections=selections,
                                      unique_fits=len(recs),bank_fit_seconds=sum(r['fit_seconds'] for _,r in recs)))
                    for q,r in recs:arrays[r['id']]=q
                    fits.extend(r for _,r in recs)
            print('A',teacher,rep,'units',len(units),'unique fits',len(fits),flush=True)
    contrasts=[];critical=float(t.ppf(1-.05/(2*40),15))
    for teacher in truths:
        for code in ['native','flip21']:
            for name in ['MI_minus_residual','gain_minus_residual','joint_minus_query_holdout','alarm_minus_query_holdout']:
                vals=[];qualified=True
                for unit in units:
                    if (unit['teacher'],unit['encoding'],unit['ntrain'])!=(teacher,code,8192):continue
                    c=unit['candidates'];s=next(s for s in unit['selections'] if s['nval']==512)
                    if name=='MI_minus_residual':a,b=c['RC_MI'],c['RC_residual']
                    elif name=='gain_minus_residual':a,b=c['RC_gain'],c['RC_residual']
                    elif name=='joint_minus_query_holdout':a,b=s['joint_holdout'],s['query_holdout']
                    else:a,b=s['moment_alarm'],s['query_holdout']
                    qualified=qualified and a['qualified'] and b['qualified']
                    vals.append(a.get('risk',a)['query7']-b.get('risk',b)['query7'])
                assert len(vals)==16;mean=float(np.mean(vals));se=float(np.std(vals,ddof=1)/4);ci=[mean-critical*se,mean+critical*se]
                contrasts.append(dict(teacher=teacher,encoding=code,contrast=name,values=vals,mean=mean,se=se,interval=ci,
                                      qualified=qualified,practical_positive=bool(qualified and ci[0]>.001)))
    np.savez_compressed(OUT/'PMFS.npz',**arrays)
    write(OUT/'COUNTS.json',counts_public);write(OUT/'FITS.json',fits)
    write(OUT/'RESULTS.json',dict(population=population,units=units,contrasts=contrasts,
        candidate_assignments=90+2880,unique_model_fits=len(fits),qualified_model_fits=sum(r['qualified'] for r in fits),
        failed_model_fits=[r['id'] for r in fits if not r['qualified']],training_observations=655360,validation_observations=40960,
        array_sha256=sha(OUT/'PMFS.npz'),ended_utc=now(),freeze_sha256=sha(OUT/'FREEZE.json')))
    print('A selection complete',len(fits),'unique fits',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','run']);a=p.parse_args()
    if a.phase=='freeze':freeze()
    else:
        try:run()
        except Exception as error:
            if (OUT/'START.json').exists():
                write(OUT/f'FAILURE_{time.time_ns()}.json',dict(time_utc=now(),error=repr(error),traceback=traceback.format_exc(),
                    partial_fit_log='FIT_LOG.jsonl',scientific_restart_allowed=False))
            raise
