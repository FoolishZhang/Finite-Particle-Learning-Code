"""B: classical bipartite structure budgets under an exogenous query workload."""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/v3_select'
OUT = BASE/'B'
SEAL = BASE/'B_FREEZE.json'
PLAN = ROOT/'docs/08_research_研究工作/v3结构选择与查询诊断_20260930/PLAN_B.md'
CFG = dict(dimensions=[16, 64], train_sizes=[64, 1024], validation_sizes=[128, 512],
           budgets=[1, 4], repetitions=32, target=0, weak_J=.35, strong_J=.8,
           seed_base=4026093000, dimension_stride=100000, repetition_stride=1000,
           seed_offsets=dict(matching=0, flip=1, train=2, validation=3),
           candidate_order=['PRODUCT', 'JOINT_MI', 'TARGET_MI'],
           deployed_methods=['PRODUCT', 'JOINT_MI', 'TARGET_MI', 'HOLDOUT_JOINT', 'HOLDOUT_QUERY'],
           contrasts=[['JOINT_MI','TARGET_MI'], ['JOINT_MI','HOLDOUT_QUERY'],
                      ['HOLDOUT_JOINT','HOLDOUT_QUERY'], ['PRODUCT','TARGET_MI']],
           interval_family=64, alpha=.05, primary_gain=.001, tie_tolerance=1e-12,
           numeric_gate=2e-12, pseudo_total=1., known_bipartition=True,
           empirical_MI_unsmoothed=True, target_secondary_joint_MI=True,
           distinct_train_vectors=65536, distinct_validation_vectors=32768,
           scientific_repetitions=64, iterative_optimizations=0)


def now(): return datetime.now(timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def native(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, np.generic): return obj.item()
    raise TypeError(type(obj).__name__)
def write(path, value, replace=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w' if replace else 'x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, default=native, allow_nan=False)
        f.write('\n')
def hashfiles(paths): return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
def verify(mapping):
    for name, digest in mapping.items(): assert sha(ROOT/name) == digest, name
def csvwrite(path, rows):
    with path.open('x', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def seedset(di, rep):
    base=CFG['seed_base']+CFG['dimension_stride']*di+CFG['repetition_stride']*rep
    return {key: base+off for key, off in CFG['seed_offsets'].items()}


def exact_budget_matching(weights, budget, left=None, right=None):
    """Float64 optimal assignment reduction; no greedy edge selection."""
    weights=np.asarray(weights, dtype=float); n=len(weights)
    assert weights.shape == (n,n) and np.isfinite(weights).all() and 0 <= budget <= n
    left=list(range(n)) if left is None else list(left)
    right=list(range(n,2*n)) if right is None else list(right)
    assert len(left)==len(right)==n
    if budget==0: return [], 0.
    count=2*n-budget; W=np.zeros((count,count))
    W[:n,:n]=weights
    W[n:,n:]=-(1+float(np.sum(abs(weights))))
    rows, cols=linear_sum_assignment(W, maximize=True)
    assert not np.any((rows>=n)&(cols>=n))
    selected=sorted((left[int(i)],right[int(j)]) for i,j in zip(rows,cols) if i<n and j<n)
    assert len(selected)==budget and len({a for a,b in selected})==budget and len({b for a,b in selected})==budget
    value=math.fsum(float(weights[i,j]) for i,j in zip(rows,cols) if i<n and j<n)
    return selected, value


def target_matching(mi, budget):
    n=len(mi); start=time.perf_counter()
    partner=int(np.argmax(mi[0])); first=(0,n+partner)
    primary_seconds=time.perf_counter()-start
    secondary_seconds=0.; edges=[first]
    if budget>1:
        left=list(range(1,n)); cols=[j for j in range(n) if j!=partner]
        start=time.perf_counter()
        rest,_=exact_budget_matching(mi[np.ix_(left,cols)], budget-1, left, [n+j for j in cols])
        secondary_seconds=time.perf_counter()-start
        edges+=rest
    return sorted(edges), dict(primary_seconds=primary_seconds, secondary_seconds=secondary_seconds)


def empirical_tables(data):
    start=time.perf_counter(); data=np.asarray(data,dtype=np.int64); N,d=data.shape; n=d//2
    assert d%2==0 and np.isin(data,[0,1]).all()
    ones=data.sum(0); unary_seconds=time.perf_counter()-start
    start=time.perf_counter(); both=data[:,:n].T@data[:,n:]
    counts=np.empty((n,n,2,2),dtype=np.int64)
    counts[:,:,1,1]=both; counts[:,:,1,0]=ones[:n,None]-both
    counts[:,:,0,1]=ones[None,n:]-both
    counts[:,:,0,0]=N-ones[:n,None]-ones[None,n:]+both
    p=counts/N; rows=p.sum(3); cols=p.sum(2); mi=np.zeros((n,n))
    for a in range(2):
        for b in range(2):
            part=p[:,:,a,b]; active=part>0
            mi[active]+=part[active]*(np.log(part[active])-np.log(rows[:,:,a][active])-np.log(cols[:,:,b][active]))
    assert np.min(mi)>=-1e-12
    corrections=int(np.sum(mi<0)); correction_max=float(max(0.,-np.min(mi)))
    mi=np.maximum(mi,0.)
    pair_seconds=time.perf_counter()-start
    return dict(N=N,d=d,ones=ones,counts=counts,MI=mi,
                unary_seconds=unary_seconds,pair_MI_seconds=pair_seconds,
                MI_roundoff_corrections=corrections, MI_roundoff_max=correction_max)


def make_model(stats, edges):
    N,d=stats['N'],stats['d']; n=d//2
    p1=(np.asarray(stats['ones'])+.5)/(N+1)
    tables=[(np.asarray(stats['counts'])[i,j-n]+.25)/(N+1) for i,j in edges]
    return dict(d=d,edges=[list(e) for e in edges],p1=p1.tolist(),pair_tables=[v.tolist() for v in tables])


def score_data(model, data, conditional=False):
    """Only the fitted candidate and observed validation vectors are accepted."""
    data=np.asarray(data,dtype=np.int8); p1=np.asarray(model['p1']); d=model['d']
    assert data.shape[1]==d
    if conditional:
        for edge,values in zip(model['edges'],model['pair_tables']):
            i,j=edge
            if i==0:
                table=np.asarray(values)
                return -(np.log(table[data[:,0],data[:,j]])-np.log(table.sum(0)[data[:,j]]))
        return -np.where(data[:,0],np.log(p1[0]),np.log1p(-p1[0]))
    logp=np.zeros(len(data)); used=set()
    for (i,j),values in zip(model['edges'],model['pair_tables']):
        table=np.asarray(values); logp+=np.log(table[data[:,i],data[:,j]]); used.update((i,j))
    for i in range(d):
        if i not in used: logp+=np.where(data[:,i],np.log(p1[i]),np.log1p(-p1[i]))
    return -logp


def choose_holdout(scores):
    """Frozen tolerance/priority; scores contain no teacher risk."""
    best=min(scores.values())
    return next(name for name in CFG['candidate_order'] if scores[name]<=best+CFG['tie_tolerance'])


def make_teacher(d, sd):
    n=d//2
    matching=np.random.Generator(np.random.PCG64(sd['matching'])).permutation(np.arange(n,d))
    flips=np.random.Generator(np.random.PCG64(sd['flip'])).integers(0,2,size=d)
    strength=np.full(n,CFG['strong_J']); strength[0]=CFG['weak_J']
    signs=(1-2*flips[:n])*(1-2*flips[matching])
    corr=signs*np.tanh(strength)
    return dict(d=d,right_by_left=matching.tolist(),flips=flips.tolist(),correlations=corr.tolist())


def sample_teacher(teacher, seed, N):
    rng=np.random.Generator(np.random.PCG64(seed)); d=teacher['d']; n=d//2
    # Sampling the recoded zero-field law directly; original signs never reach learners.
    left=rng.integers(0,2,size=(N,n),dtype=np.int8)
    same=rng.random((N,n))<(1+np.asarray(teacher['correlations']))/2
    data=np.empty((N,d),dtype=np.int8); data[:,:n]=left
    data[:,teacher['right_by_left']]=np.where(same,left,1-left)
    return data


def true_pair(teacher,i,j):
    assert i<teacher['d']//2<=j
    rho=teacher['correlations'][i] if teacher['right_by_left'][i]==j else 0.
    return np.array([[1+rho,1-rho],[1-rho,1+rho]])/4


def teacher_entropy(teacher):
    return -math.fsum(float(np.sum((p:=true_pair(teacher,i,j))*np.log(p))) for i,j in enumerate(teacher['right_by_left']))


def risks(model,teacher):
    """Post-selection analytic evaluation, never a selector input."""
    used=set(); ce=0.; query_ce=None
    for (i,j),values in zip(model['edges'],model['pair_tables']):
        p=true_pair(teacher,i,j); q=np.asarray(values)
        ce-=float(np.sum(p*np.log(q))); used.update((i,j))
        if i==0: query_ce=-float(np.sum(p*(np.log(q)-np.log(q.sum(0))[None,:])))
    p1=np.asarray(model['p1'])
    for i in range(model['d']):
        if i not in used:
            unary=-.5*float(np.log(p1[i])+np.log1p(-p1[i])); ce+=unary
            if i==0: query_ce=unary
    rho=abs(teacher['correlations'][0]); hcond=-((1+rho)/2*math.log((1+rho)/2)+(1-rho)/2*math.log((1-rho)/2))
    result=dict(joint_KL=ce-teacher_entropy(teacher),query_KL=query_ce-hcond,
                target_partner_recovered=any(i==0 and j==teacher['right_by_left'][0] for i,j in model['edges']))
    assert min(result['joint_KL'],result['query_KL'])>=-CFG['numeric_gate']
    return result


def oracle_model(teacher,budget,query):
    left=([0]+list(range(1,budget))) if query else list(range(1,budget+1))
    edges=[(i,teacher['right_by_left'][i]) for i in left]
    return dict(d=teacher['d'],edges=edges,p1=[.5]*teacher['d'],
                pair_tables=[true_pair(teacher,i,j).tolist() for i,j in edges])


def toyqa():
    """Deterministic arithmetic only, no RNG or scientific teacher sampling."""
    match_error=risk_error=marginal_error=0.
    for n in (2,3,4):
        W=np.array([[((i+2)*(j+3)%11)/17 for j in range(n)] for i in range(n)])
        for b in range(n+1):
            selected,value=exact_budget_matching(W,b)
            best=max((sum(W[i,j] for i,j in zip(rows,cols))
                for rows in itertools.combinations(range(n),b)
                for cols in itertools.permutations(range(n),b)),default=0.)
            match_error=max(match_error,abs(value-best))
    d=4; states=((np.arange(16)[:,None]>>np.arange(d))&1).astype(np.int8)
    teacher=dict(d=4,right_by_left=[3,2],flips=[0,1,1,0],correlations=[.3,-.6])
    p=np.ones(16)
    for i,j in enumerate(teacher['right_by_left']):p*=true_pair(teacher,i,j)[states[:,i],states[:,j]]
    stats=empirical_tables(np.repeat(states,np.arange(1,17),axis=0))
    for edges in ([],[(0,2)],[(0,3)],[(0,2),(1,3)],[(0,3),(1,2)]):
        model=make_model(stats,edges); q=np.exp(-score_data(model,states)); assert abs(q.sum()-1)<1e-12
        marginal_error=max(marginal_error,float(np.max(abs(q@states-np.asarray(model['p1'])))))
        analytic=risks(model,teacher)
        exact_joint=float(np.sum(p*np.log(p/q)))
        context=np.arange(16)>>1
        pmass=np.bincount(context,weights=p); qmass=np.bincount(context,weights=q)
        exact_query=float(np.sum(p*(np.log(p/pmass[context])-np.log(q/qmass[context]))))
        risk_error=max(risk_error,abs(exact_joint-analytic['joint_KL']),abs(exact_query-analytic['query_KL']))
        explicit=-np.log(q/qmass[context]); actual=score_data(model,states,True)
        assert np.max(abs(explicit-actual))<CFG['numeric_gate']
    assert choose_holdout(dict(PRODUCT=1.,JOINT_MI=1.-.5e-12,TARGET_MI=1.))=='PRODUCT'
    assert choose_holdout(dict(PRODUCT=2.,JOINT_MI=1.,TARGET_MI=1.-2e-12))=='TARGET_MI'
    assert max(match_error,risk_error,marginal_error)<CFG['numeric_gate']
    return dict(passed=True,matching_enumeration_max_error=match_error,risk_enumeration_max_error=risk_error,
                model_marginal_max_error=marginal_error,new_scientific_draws=0,iterative_fits=0)


def freeze():
    assert not OUT.exists() and not SEAL.exists()
    OUT.mkdir(parents=True); (OUT/'local').mkdir(); (OUT/'.gitignore').write_text('/local/\n',encoding='utf-8')
    (OUT/'source_inputs').mkdir(); (OUT/'source_inputs/HANDOFF.md').write_bytes((ROOT/'HANDOFF.md').read_bytes())
    write(OUT/'CONFIG.json',CFG)
    write(OUT/'ENVIRONMENT.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
          platform=platform.platform(),threads=2,timing_is_descriptive=True))
    files=[Path(__file__).resolve(),PLAN,OUT/'CONFIG.json',OUT/'ENVIRONMENT.json',OUT/'source_inputs/HANDOFF.md']
    write(SEAL,dict(time_utc=now(),files=hashfiles(files),source_sha256=sha(__file__),before_scientific_RNG=True))
    write(OUT/'STATE.json',dict(status='frozen',time_utc=now()))
    print(json.dumps(dict(frozen=True,source_sha256=sha(__file__),freeze_sha256=sha(SEAL))),flush=True)


def checkfreeze():
    assert read(OUT/'CONFIG.json')==CFG; verify(read(SEAL)['files'])


def run_rep(di,rep):
    d=CFG['dimensions'][di]; sd=seedset(di,rep); start=time.perf_counter()
    teacher=make_teacher(d,sd); train=sample_teacher(teacher,sd['train'],1024)
    validation=sample_teacher(teacher,sd['validation'],512); generation_seconds=time.perf_counter()-start
    path=OUT/'local'/f'd{d}_r{rep:02}.npz'; assert not path.exists()
    np.savez_compressed(path,train=train,validation=validation)
    levels=[]
    for N in CFG['train_sizes']:
        stats=empirical_tables(train[:N]); start=time.perf_counter(); product=make_model(stats,[])
        product_seconds=time.perf_counter()-start
        budgets=[]
        for b in CFG['budgets']:
            start=time.perf_counter(); je,jvalue=exact_budget_matching(stats['MI'],b)
            joint_seconds=time.perf_counter()-start
            te,target_time=target_matching(stats['MI'],b)
            start=time.perf_counter(); jm=make_model(stats,je); joint_fit=time.perf_counter()-start
            start=time.perf_counter(); tm=make_model(stats,te); target_fit=time.perf_counter()-start
            models=dict(PRODUCT=product,JOINT_MI=jm,TARGET_MI=tm)
            base_cost=stats['unary_seconds']+stats['pair_MI_seconds']
            pipeline=dict(PRODUCT=stats['unary_seconds']+product_seconds,
                JOINT_MI=base_cost+joint_seconds+joint_fit,
                TARGET_MI=base_cost+sum(target_time.values())+target_fit)
            all_cost=base_cost+product_seconds+joint_seconds+joint_fit+sum(target_time.values())+target_fit
            val_records=[]
            # All selections below use candidates and independent validation only.
            for nv in CFG['validation_sizes']:
                full_scores={}; query_scores={}; ft={}; qt={}
                for name in CFG['candidate_order']:
                    start=time.perf_counter(); full_scores[name]=float(score_data(models[name],validation[:nv]).mean()); ft[name]=time.perf_counter()-start
                    start=time.perf_counter(); query_scores[name]=float(score_data(models[name],validation[:nv],True).mean()); qt[name]=time.perf_counter()-start
                start=time.perf_counter(); hj=choose_holdout(full_scores); jt=time.perf_counter()-start
                start=time.perf_counter(); hq=choose_holdout(query_scores); qtime=time.perf_counter()-start
                val_records.append(dict(Nval=nv,joint_scores=full_scores,query_scores=query_scores,
                    HOLDOUT_JOINT=hj,HOLDOUT_QUERY=hq,joint_scoring_seconds=ft,query_scoring_seconds=qt,
                    joint_selection_seconds=jt,query_selection_seconds=qtime,
                    HOLDOUT_JOINT_pipeline_seconds=all_cost+sum(ft.values())+jt,
                    HOLDOUT_QUERY_pipeline_seconds=all_cost+sum(qt.values())+qtime))
            budgets.append(dict(b=b,models=models,
                joint_MI_objective=jvalue,target_MI_primary=float(stats['MI'][0,te[0][1]-d//2]),
                target_MI_secondary=float(sum(stats['MI'][i,j-d//2] for i,j in te if i!=0)),
                validation=val_records,cost=dict(joint_assignment_seconds=joint_seconds,
                    target_primary_seconds=target_time['primary_seconds'],target_secondary_assignment_seconds=target_time['secondary_seconds'],
                    joint_parameter_seconds=joint_fit,target_parameter_seconds=target_fit,
                    direct_pipeline_seconds=pipeline,all_candidates_training_seconds=all_cost)))
        levels.append(dict(Ntrain=N,statistics=stats,product_parameter_seconds=product_seconds,budgets=budgets))
    # All N/b/validation selectors in this repetition finish before truth scoring.
    for level in levels:
        for budget in level['budgets']:
            start=time.perf_counter()
            budget['risks']={name:risks(model,teacher) for name,model in budget['models'].items()}
            budget['oracle_diagnostics']={name:risks(oracle_model(teacher,budget['b'],isquery),teacher)
                for name,isquery in [('ORACLE_JOINT',False),('ORACLE_QUERY',True)]}
            budget['cost']['teacher_risk_evaluation_seconds']=time.perf_counter()-start
    write(OUT/'repetitions'/f'd{d}_r{rep:02}.json',dict(d=d,repetition=rep,seeds=sd,teacher=teacher,
        generation_seconds=generation_seconds,levels=levels,local_file=path.relative_to(ROOT).as_posix(),local_sha256=sha(path),
        source_sha256=sha(__file__),freeze_sha256=sha(SEAL)))
    print(json.dumps(dict(d=d,repetition=rep,completed=True)),flush=True)


def records():
    return [read(OUT/'repetitions'/f'd{d}_r{rep:02}.json') for d in CFG['dimensions'] for rep in range(32)]


def collect():
    scores=[]
    for rec in records():
        for level in rec['levels']:
            for budget in level['budgets']:
                for validation in budget['validation']:
                    for method in CFG['deployed_methods']:
                        selected=validation[method] if method.startswith('HOLDOUT') else method
                        seconds=validation[method+'_pipeline_seconds'] if method.startswith('HOLDOUT') else budget['cost']['direct_pipeline_seconds'][method]
                        scores.append(dict(d=rec['d'],repetition=rec['repetition'],Ntrain=level['Ntrain'],b=budget['b'],Nval=validation['Nval'],
                            method=method,selected_candidate=selected,**budget['risks'][selected],pipeline_seconds=seconds))
    assert len(scores)==2560
    csvwrite(OUT/'SCORES.csv',scores)
    summary=[]; intervals=[]; critical=float(student_t.ppf(1-.05/(2*64),31))
    for d,N,b,nv in itertools.product(CFG['dimensions'],CFG['train_sizes'],CFG['budgets'],CFG['validation_sizes']):
        subset=[r for r in scores if (r['d'],r['Ntrain'],r['b'],r['Nval'])==(d,N,b,nv)]
        common=dict(d=d,Ntrain=N,b=b,Nval=nv)
        for method in CFG['deployed_methods']:
            rows=[r for r in subset if r['method']==method]
            summary.append(dict(**common,method=method,mean_query_KL=float(np.mean([r['query_KL'] for r in rows])),
                mean_joint_KL=float(np.mean([r['joint_KL'] for r in rows])),target_partner_recovery=float(np.mean([r['target_partner_recovered'] for r in rows])),
                selected_PRODUCT=sum(r['selected_candidate']=='PRODUCT' for r in rows),
                selected_JOINT_MI=sum(r['selected_candidate']=='JOINT_MI' for r in rows),
                selected_TARGET_MI=sum(r['selected_candidate']=='TARGET_MI' for r in rows),
                mean_pipeline_seconds=float(np.mean([r['pipeline_seconds'] for r in rows]))))
        for worse,better in CFG['contrasts']:
            x=np.array([next(r['query_KL'] for r in subset if r['repetition']==rep and r['method']==worse)-
                        next(r['query_KL'] for r in subset if r['repetition']==rep and r['method']==better) for rep in range(32)])
            mean=float(x.mean()); se=float(x.std(ddof=1)/math.sqrt(32)); lo=mean-critical*se; hi=mean+critical*se
            primary=bool(N==1024 and nv==512 and worse=='JOINT_MI' and better=='TARGET_MI')
            intervals.append(dict(**common,contrast=worse+'-minus-'+better,mean_gain=mean,SE=se,lower=lo,upper=hi,
                primary=primary,practical_gain=bool(lo>.001),confidence_family=64,t_approximation=True))
    csvwrite(OUT/'SUMMARY.csv',summary); csvwrite(OUT/'INTERVALS.csv',intervals)
    write(OUT/'DECISION.json',dict(primary=[r for r in intervals if r['primary']],primary_passed=sum(r['primary'] and r['practical_gain'] for r in intervals),primary_total=4,
        all_prespecified_intervals=64,no_new_method_claim=True))
    write(OUT/'COUNTS.json',dict(teacher_realizations=64,train_vectors=65536,validation_vectors=32768,
        train_prefix_datasets=128,product_parameter_fits=128,joint_parameter_fits=256,target_parameter_fits=256,
        candidate_parameter_fits=640,iterative_optimizations=0,joint_assignment_calls=256,target_secondary_assignment_calls=128,
        target_primary_partner_choices=256,validation_decisions=1024,formal_deployment_records=2560,
        population_oracle_diagnostic_records=512,independent_repetitions_per_cell=32,interval_family=64,
        shared_data_and_parameters_reused=True))


def audit(workspace=False):
    checkfreeze(); maxerr=0.; nrows=0; allscore=[]
    for rec in records():
        d=rec['d']; n=d//2; teacher=rec['teacher']
        assert rec['seeds']==seedset(CFG['dimensions'].index(d),rec['repetition'])
        assert rec['source_sha256']==sha(__file__) and rec['freeze_sha256']==sha(SEAL)
        if workspace:
            assert sha(ROOT/rec['local_file'])==rec['local_sha256']
            data=np.load(ROOT/rec['local_file']); train=data['train']; validation=data['validation']
        for level in rec['levels']:
            stats=level['statistics']; N=level['Ntrain']; cnt=np.asarray(stats['counts']); ones=np.asarray(stats['ones'])
            assert cnt.shape==(n,n,2,2) and np.all(cnt.sum((2,3))==N)
            assert np.all(cnt[:,:,1,:].sum(2)==ones[:n,None]) and np.all(cnt[:,:,:,1].sum(2)==ones[None,n:])
            if workspace:
                assert np.array_equal(train[:N].sum(0),ones)
                for i in range(n):
                    for j in range(n):
                        test=np.bincount(2*train[:N,i]+train[:N,n+j],minlength=4).reshape(2,2)
                        assert np.array_equal(test,cnt[i,j])
            independent_mi=np.zeros((n,n))
            for i in range(n):
                for j in range(n):
                    joint=cnt[i,j]/N; outer=np.outer(joint.sum(1),joint.sum(0))
                    independent_mi[i,j]=max(0.,math.fsum(float(joint[a,b])*math.log(float(joint[a,b]/outer[a,b])) for a,b in itertools.product(range(2),repeat=2) if joint[a,b]>0))
            maxerr=max(maxerr,float(np.max(abs(independent_mi-np.asarray(stats['MI'])))))
            for budget in level['budgets']:
                b=budget['b']; mi=np.asarray(stats['MI']); je,val=exact_budget_matching(mi,b); te,_=target_matching(mi,b)
                assert [list(e) for e in je]==budget['models']['JOINT_MI']['edges']
                assert [list(e) for e in te]==budget['models']['TARGET_MI']['edges']
                for name,m in budget['models'].items():
                    recreated=make_model(stats,[tuple(e) for e in m['edges']]); assert recreated==m
                    quality=risks(m,teacher)
                    for key in ('joint_KL','query_KL'):maxerr=max(maxerr,abs(quality[key]-budget['risks'][name][key]))
                information=[math.log(2)+sum(p*math.log(p) for p in ((1+abs(rho))/2,(1-abs(rho))/2)) for rho in teacher['correlations']]
                expected_oracles=dict(ORACLE_JOINT=dict(joint_KL=sum(information)-sum(information[1:b+1]),query_KL=information[0]),
                    ORACLE_QUERY=dict(joint_KL=sum(information)-sum(information[:b]),query_KL=0.))
                for name,quality in expected_oracles.items():
                    for key,value in quality.items():maxerr=max(maxerr,abs(value-budget['oracle_diagnostics'][name][key]))
                for vr in budget['validation']:
                    for key,conditional in [('joint',False),('query',True)]:
                        assert choose_holdout(vr[key+'_scores'])==vr['HOLDOUT_'+key.upper()]
                        if workspace:
                            for name,m in budget['models'].items():
                                val=float(score_data(m,validation[:vr['Nval']],conditional).mean())
                                maxerr=max(maxerr,abs(val-vr[key+'_scores'][name]))
                    for method in CFG['deployed_methods']:
                        selected=vr[method] if method.startswith('HOLDOUT') else method
                        allscore.append(dict(d=d,Ntrain=N,b=b,Nval=vr['Nval'],repetition=rec['repetition'],method=method,
                            selected_candidate=selected,**budget['risks'][selected]))
                        nrows+=1
        if workspace:data.close()
    assert nrows==2560 and maxerr<CFG['numeric_gate']
    with (OUT/'SCORES.csv').open(encoding='utf-8',newline='') as f: published=list(csv.DictReader(f))
    assert len(published)==len(allscore)
    for a,b in zip(published,allscore):
        for key in ('d','Ntrain','b','Nval','repetition'):assert int(a[key])==b[key]
        for key in ('method','selected_candidate'):assert a[key]==b[key]
        for key in ('joint_KL','query_KL'):maxerr=max(maxerr,abs(float(a[key])-b[key]))
    with (OUT/'SUMMARY.csv').open(encoding='utf-8',newline='') as f: published=list(csv.DictReader(f))
    assert len(published)==80
    for row in published:
        subset=[r for r in allscore if all(r[key]==int(row[key]) for key in ('d','Ntrain','b','Nval')) and r['method']==row['method']]
        assert len(subset)==32
        for key in ('joint_KL','query_KL'):maxerr=max(maxerr,abs(float(row['mean_'+key])-math.fsum(r[key] for r in subset)/32))
        maxerr=max(maxerr,abs(float(row['target_partner_recovery'])-sum(r['target_partner_recovered'] for r in subset)/32))
        for name in CFG['candidate_order']:assert int(row['selected_'+name])==sum(r['selected_candidate']==name for r in subset)
    with (OUT/'INTERVALS.csv').open(encoding='utf-8',newline='') as f: table=list(csv.DictReader(f))
    assert len(table)==64; critical=float(student_t.ppf(1-.05/(2*64),31))
    for row in table:
        worse,better=row['contrast'].split('-minus-'); common={k:int(row[k]) for k in ('d','Ntrain','b','Nval')}
        subset=[r for r in allscore if all(r[k]==v for k,v in common.items())]
        delta=[next(r['query_KL'] for r in subset if r['repetition']==rep and r['method']==worse)-next(r['query_KL'] for r in subset if r['repetition']==rep and r['method']==better) for rep in range(32)]
        mean=math.fsum(delta)/32; se=math.sqrt(math.fsum((x-mean)**2 for x in delta)/(31*32))
        for key,value in [('mean_gain',mean),('SE',se),('lower',mean-critical*se),('upper',mean+critical*se)]:
            maxerr=max(maxerr,abs(float(row[key])-value))
        assert (row['practical_gain']=='True')==(mean-critical*se>.001)
    decision=read(OUT/'DECISION.json')
    assert decision['primary_total']==4 and decision['all_prespecified_intervals']==64
    assert decision['primary_passed']==sum(r['primary']=='True' and r['practical_gain']=='True' for r in table)
    assert maxerr<CFG['numeric_gate']
    return dict(passed=True,workspace=workspace,time_utc=now(),maximum_arithmetic_error=maxerr,
        teacher_realizations=64,candidate_parameter_fits_replayed=640,formal_records=2560,intervals=64,
        raw_train_validation_replayed=workspace,selection_risk_routines_reused=True,
        scope='independent pair-count/MI expressions and scalar statistics; deterministic selection/fit/risk replay; toy enumeration supplies independent risk and matching check',
        new_scientific_draws=0,new_iterative_fits=0,timing_not_replayed=True)


def run():
    checkfreeze(); assert read(OUT/'STATE.json')['status']=='frozen'
    write(OUT/'TOY_QA.json',toyqa())
    started=now(); write(OUT/'START.json',dict(first_scientific_start_utc=started,source_sha256=sha(__file__),freeze_sha256=sha(SEAL)))
    for di,d in enumerate(CFG['dimensions']):
        for rep in range(32):
            write(OUT/'STATE.json',dict(status='running',d=d,repetition=rep,first_scientific_start_utc=started,pid=os.getpid()),replace=True)
            run_rep(di,rep)
    collect(); write(OUT/'AUDIT.json',audit(workspace=True))
    paths=list((OUT/'repetitions').glob('*.json'))+[OUT/name for name in ['TOY_QA.json','START.json','SCORES.csv','SUMMARY.csv','INTERVALS.csv','DECISION.json','COUNTS.json','AUDIT.json']]
    write(OUT/'RESULTS_FREEZE.json',dict(time_utc=now(),source_freeze_sha256=sha(SEAL),files=hashfiles(paths)))
    write(OUT/'STATE.json',dict(status='complete',time_utc=now(),first_scientific_start_utc=started,active_jobs=[]),replace=True)
    print(json.dumps(dict(complete=True,**read(OUT/'DECISION.json'))),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['toyqa','freeze','run','check']);p.add_argument('--workspace',action='store_true');args=p.parse_args()
    try:
        if args.phase=='toyqa': print(json.dumps(toyqa()))
        elif args.phase=='freeze':freeze()
        elif args.phase=='run':run()
        else:
            verify(read(OUT/'RESULTS_FREEZE.json')['files']); assert read(OUT/'STATE.json')['status']=='complete'
            print(json.dumps(audit(args.workspace)))
    except Exception as exc:
        if args.phase=='run' and SEAL.exists() and not (OUT/'RESULTS_FREEZE.json').exists():
            write(OUT/f'FAILURE_{time.time_ns()}.json',dict(time_utc=now(),error=repr(exc),traceback=traceback.format_exc(),state=read(OUT/'STATE.json')))
            old=read(OUT/'STATE.json');old.update(status='stopped_failure',active_jobs=[]);write(OUT/'STATE.json',old,replace=True)
        raise


if __name__=='__main__': main()
