"""B2: reuse sealed B labels for fixed classical rules at equal total label cost."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import argparse
import csv
import itertools
import json
import math
import platform
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import scipy
from scipy.stats import t as student_t
import v3_select_budget as B

ROOT=B.ROOT
OUT=ROOT/'artifacts/v3_select/B2'
SEAL=OUT/'FREEZE.json'
PLAN=ROOT/'docs/08_research_研究工作/v3结构选择与查询诊断_20260930/PLAN_B2.md'
CFG=dict(dimensions=[16,64],Ntrain=[64,1024],Nval=[128,512],budgets=[1,4],repetitions=32,
         interval_family=8,alpha=.05,primary=dict(d=64,Ntrain=64,Nval=512,b=1,threshold=.001),
         new_random_draws=0,pools=256,closed_form_parameter_fits=1280,
         methods=['PRODUCT','JOINT_MI','TARGET_MI'],known_bipartition=True,
         developed_after_B=True,no_validation_labels_remaining_after_pool=True)


def toyqa():
    train=np.arange(24).reshape(6,4)%2; val=(np.arange(16).reshape(4,4)//3)%2
    for n,v in [(2,1),(2,4),(6,1),(6,4)]:
        joined=np.concatenate((train[:n],val[:v]),axis=0)
        assert len(joined)==n+v and np.array_equal(joined.sum(0),train[:n].sum(0)+val[:v].sum(0))
        assert np.array_equal(joined[:n],train[:n]) and np.array_equal(joined[n:],val[:v])
    return dict(passed=True,pool_prefix_cases=4,new_random_draws=0,base_deterministic_qa=B.toyqa())


def verify_B():
    B.checkfreeze(); B.verify(B.read(B.OUT/'RESULTS_FREEZE.json')['files'])
    assert B.read(B.OUT/'STATE.json')['status']=='complete'


def freeze():
    assert not OUT.exists(); verify_B(); OUT.mkdir(parents=True)
    inputs=[]
    for rec in B.records():
        assert B.sha(ROOT/rec['local_file'])==rec['local_sha256']
        inputs.append(dict(d=rec['d'],repetition=rec['repetition'],local_file=rec['local_file'],sha256=rec['local_sha256']))
    B.write(OUT/'B_ARRAY_INPUTS.json',inputs); B.write(OUT/'CONFIG.json',CFG)
    B.write(OUT/'ENVIRONMENT.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),threads=2))
    paths=[Path(__file__).resolve(),PLAN,Path(B.__file__).resolve(),B.SEAL,B.OUT/'RESULTS_FREEZE.json',OUT/'B_ARRAY_INPUTS.json',OUT/'CONFIG.json',OUT/'ENVIRONMENT.json']
    B.write(SEAL,dict(time_utc=B.now(),files=B.hashfiles(paths),before_any_pooled_scientific_fit=True,
        source_sha256=B.sha(__file__),original_B_source_sha256=B.sha(B.__file__)))
    B.write(OUT/'STATE.json',dict(status='frozen',time_utc=B.now()))
    print(json.dumps(dict(frozen=True,source_sha256=B.sha(__file__),freeze_sha256=B.sha(SEAL))),flush=True)


def checkfreeze():
    B.verify(B.read(SEAL)['files']); assert B.read(OUT/'CONFIG.json')==CFG; verify_B()


def run_rep(rec):
    assert B.sha(ROOT/rec['local_file'])==rec['local_sha256']
    with np.load(ROOT/rec['local_file']) as data:train=data['train']; val=data['validation']
    units=[]
    for n,v in itertools.product(CFG['Ntrain'],CFG['Nval']):
        start=time.perf_counter(); pooled=np.concatenate((train[:n],val[:v]),axis=0); pooling_seconds=time.perf_counter()-start
        stats=B.empirical_tables(pooled); start=time.perf_counter(); product=B.make_model(stats,[])
        product_fit_seconds=time.perf_counter()-start; budgets=[]
        for b in CFG['budgets']:
            start=time.perf_counter(); je,jobj=B.exact_budget_matching(stats['MI'],b); jselect=time.perf_counter()-start
            te,tcost=B.target_matching(stats['MI'],b)
            start=time.perf_counter(); jm=B.make_model(stats,je); jfit=time.perf_counter()-start
            start=time.perf_counter(); tm=B.make_model(stats,te); tfit=time.perf_counter()-start
            basic=pooling_seconds+stats['unary_seconds']+stats['pair_MI_seconds']
            budgets.append(dict(b=b,models=dict(PRODUCT=product,JOINT_MI=jm,TARGET_MI=tm),joint_MI_objective=jobj,
                cost=dict(joint_assignment_seconds=jselect,target_primary_seconds=tcost['primary_seconds'],
                    target_secondary_assignment_seconds=tcost['secondary_seconds'],joint_parameter_seconds=jfit,target_parameter_seconds=tfit,
                    pipeline_seconds=dict(PRODUCT=pooling_seconds+stats['unary_seconds']+product_fit_seconds,
                        JOINT_MI=basic+jselect+jfit,TARGET_MI=basic+sum(tcost.values())+tfit))))
        units.append(dict(Ntrain=n,Nval=v,Npool=n+v,statistics=stats,pooling_seconds=pooling_seconds,
                          product_parameter_seconds=product_fit_seconds,budgets=budgets))
    # Fitting all pooled candidates ends before reading teacher evaluation metadata.
    teacher=rec['teacher']
    for unit in units:
        oldlevel=next(level for level in rec['levels'] if level['Ntrain']==unit['Ntrain'])
        for budget in unit['budgets']:
            start=time.perf_counter(); budget['risks']={name:B.risks(m,teacher) for name,m in budget['models'].items()}
            budget['teacher_evaluation_seconds']=time.perf_counter()-start
            oldbudget=next(bb for bb in oldlevel['budgets'] if bb['b']==budget['b'])
            oldval=next(vv for vv in oldbudget['validation'] if vv['Nval']==unit['Nval'])
            selected=oldval['HOLDOUT_QUERY']
            budget['original_HOLDOUT_QUERY']=dict(selected_candidate=selected,**oldbudget['risks'][selected],
                pipeline_seconds=oldval['HOLDOUT_QUERY_pipeline_seconds'])
    path=OUT/'repetitions'/f"d{rec['d']}_r{rec['repetition']:02}.json"
    B.write(path,dict(d=rec['d'],repetition=rec['repetition'],units=units,
        original_B_rep_file=f"artifacts/v3_select/B/repetitions/d{rec['d']}_r{rec['repetition']:02}.json",
        original_B_rep_sha256=B.sha(B.OUT/'repetitions'/f"d{rec['d']}_r{rec['repetition']:02}.json"),
        original_local_file=rec['local_file'],original_local_sha256=rec['local_sha256'],source_sha256=B.sha(__file__),freeze_sha256=B.sha(SEAL)))
    print(json.dumps(dict(d=rec['d'],repetition=rec['repetition'],pools=4,parameter_fits=20,complete=True)),flush=True)


def records():
    return [B.read(OUT/'repetitions'/f'd{d}_r{rep:02}.json') for d in CFG['dimensions'] for rep in range(32)]


def score_rows():
    rows=[]
    for rec in records():
        for unit in rec['units']:
            for budget in unit['budgets']:
                common=dict(d=rec['d'],repetition=rec['repetition'],Ntrain=unit['Ntrain'],Nval=unit['Nval'],Npool=unit['Npool'],b=budget['b'])
                for name in CFG['methods']:
                    rows.append(dict(**common,method='POOLED_'+name,**budget['risks'][name],pipeline_seconds=budget['cost']['pipeline_seconds'][name]))
                rows.append(dict(**common,method='ORIGINAL_HOLDOUT_QUERY',**{key:budget['original_HOLDOUT_QUERY'][key] for key in ('joint_KL','query_KL','target_partner_recovered','pipeline_seconds')}))
    assert len(rows)==2048
    return rows


def collect():
    rows=score_rows(); B.csvwrite(OUT/'SCORES.csv',rows); summary=[]; intervals=[]
    critical=float(student_t.ppf(1-.05/(2*8),31))
    for d,n,v,b in itertools.product(CFG['dimensions'],CFG['Ntrain'],CFG['Nval'],CFG['budgets']):
        subset=[r for r in rows if (r['d'],r['Ntrain'],r['Nval'],r['b'])==(d,n,v,b)]
        for method in ['POOLED_'+name for name in CFG['methods']]+['ORIGINAL_HOLDOUT_QUERY']:
            selected=[r for r in subset if r['method']==method]
            summary.append(dict(d=d,Ntrain=n,Nval=v,Npool=n+v,b=b,method=method,
                mean_query_KL=float(np.mean([r['query_KL'] for r in selected])),mean_joint_KL=float(np.mean([r['joint_KL'] for r in selected])),
                target_partner_recovery=float(np.mean([r['target_partner_recovered'] for r in selected])),
                mean_pipeline_seconds=float(np.mean([r['pipeline_seconds'] for r in selected]))))
        if b!=1:continue
        values=np.array([next(r['query_KL'] for r in subset if r['repetition']==rep and r['method']=='ORIGINAL_HOLDOUT_QUERY')-
                         next(r['query_KL'] for r in subset if r['repetition']==rep and r['method']=='POOLED_TARGET_MI') for rep in range(32)])
        mean=float(values.mean());se=float(values.std(ddof=1)/math.sqrt(32));lower=mean-critical*se;upper=mean+critical*se
        primary=(d,n,v)==(64,64,512)
        intervals.append(dict(d=d,Ntrain=n,Nval=v,Npool=n+v,b=1,contrast='ORIGINAL_HQ-minus-POOLED_TARGET_MI',
            mean_gain=mean,SE=se,lower=lower,upper=upper,primary=primary,practical_gain=bool(lower>.001),
            confidence_family=8,t_approximation=True))
    B.csvwrite(OUT/'SUMMARY.csv',summary); B.csvwrite(OUT/'INTERVALS.csv',intervals)
    B.write(OUT/'DECISION.json',dict(primary=next(r for r in intervals if r['primary']),all_intervals=8,developed_after_B=True,no_new_method_claim=True))
    B.write(OUT/'COUNTS.json',dict(pooled_datasets=256,new_product_parameter_fits=256,new_joint_parameter_fits=512,
        new_target_parameter_fits=512,new_closed_form_fits=1280,new_iterative_optimizations=0,new_random_draws=0,
        joint_assignment_calls=512,target_primary_choices=512,target_secondary_assignment_calls=256,
        new_deployment_records=1536,original_HQ_references=512,formal_intervals=8,independent_repetitions_per_cell=32,
        old_B_fits_unchanged=640,original_train_vectors_reused=65536,original_validation_vectors_reused=32768,
        no_independent_validation_remaining=True,b4_not_independent_query_confirmation=True))


def audit(workspace=False):
    checkfreeze(); maximum=0.; models=0
    for rec in records():
        old=B.read(ROOT/rec['original_B_rep_file']); assert B.sha(ROOT/rec['original_B_rep_file'])==rec['original_B_rep_sha256']
        assert rec['source_sha256']==B.sha(__file__) and rec['freeze_sha256']==B.sha(SEAL)
        if workspace:
            assert B.sha(ROOT/rec['original_local_file'])==rec['original_local_sha256']
            data=np.load(ROOT/rec['original_local_file']); train=data['train']; val=data['validation']
        for unit in rec['units']:
            stats=unit['statistics']; n,v=unit['Ntrain'],unit['Nval']; count=np.asarray(stats['counts']); ones=np.asarray(stats['ones'])
            assert stats['N']==unit['Npool']==n+v
            half=rec['d']//2
            assert np.all(count.sum((2,3))==n+v)
            if workspace:
                assert np.array_equal(ones,train[:n].sum(0)+val[:v].sum(0))
                for i in range(half):
                    for j in range(half):
                        direct=np.bincount(2*train[:n,i]+train[:n,half+j],minlength=4)+np.bincount(2*val[:v,i]+val[:v,half+j],minlength=4)
                        assert np.array_equal(direct.reshape(2,2),count[i,j])
            mi=np.zeros((half,half))
            for i in range(half):
                for j in range(half):
                    p=count[i,j]/(n+v); outer=np.outer(p.sum(1),p.sum(0))
                    mi[i,j]=max(0.,math.fsum(float(p[a,b])*math.log(float(p[a,b]/outer[a,b])) for a,b in itertools.product(range(2),repeat=2) if p[a,b]>0))
            maximum=max(maximum,float(np.max(abs(mi-np.asarray(stats['MI'])))))
            oldlevel=next(level for level in old['levels'] if level['Ntrain']==n)
            for budget in unit['budgets']:
                b=budget['b']; joint,_=B.exact_budget_matching(np.asarray(stats['MI']),b); target,_=B.target_matching(np.asarray(stats['MI']),b)
                assert budget['models']['JOINT_MI']['edges']==[list(e) for e in joint]
                assert budget['models']['TARGET_MI']['edges']==[list(e) for e in target]
                for method,model in budget['models'].items():
                    assert B.make_model(stats,[tuple(e) for e in model['edges']])==model
                    risks=B.risks(model,old['teacher'])
                    for key in ('joint_KL','query_KL'):maximum=max(maximum,abs(risks[key]-budget['risks'][method][key]))
                    if method!='PRODUCT' or b==1:models+=1
                oldbudget=next(bb for bb in oldlevel['budgets'] if bb['b']==b)
                oldval=next(vv for vv in oldbudget['validation'] if vv['Nval']==v)
                selected=oldval['HOLDOUT_QUERY']; assert budget['original_HOLDOUT_QUERY']['selected_candidate']==selected
                for key in ('joint_KL','query_KL'):maximum=max(maximum,abs(budget['original_HOLDOUT_QUERY'][key]-oldbudget['risks'][selected][key]))
        if workspace:data.close()
    assert models==1280
    rows=score_rows()
    with (OUT/'SCORES.csv').open(encoding='utf-8',newline='') as f:public=list(csv.DictReader(f))
    assert len(public)==2048
    for a,b in zip(public,rows):
        for key in ('d','Ntrain','Nval','b','repetition'):assert int(a[key])==b[key]
        assert a['method']==b['method']
        for key in ('joint_KL','query_KL'):maximum=max(maximum,abs(float(a[key])-b[key]))
    with (OUT/'SUMMARY.csv').open(encoding='utf-8',newline='') as f:summary=list(csv.DictReader(f))
    assert len(summary)==64
    for row in summary:
        subset=[r for r in rows if all(r[key]==int(row[key]) for key in ('d','Ntrain','Nval','b')) and r['method']==row['method']]
        assert len(subset)==32
        for key in ('joint_KL','query_KL'):maximum=max(maximum,abs(float(row['mean_'+key])-math.fsum(r[key] for r in subset)/32))
    with (OUT/'INTERVALS.csv').open(encoding='utf-8',newline='') as f:intervals=list(csv.DictReader(f))
    assert len(intervals)==8;critical=float(student_t.ppf(1-.05/(2*8),31))
    for row in intervals:
        subset=[r for r in rows if all(r[key]==int(row[key]) for key in ('d','Ntrain','Nval','b'))]
        delta=[next(r['query_KL'] for r in subset if r['repetition']==rep and r['method']=='ORIGINAL_HOLDOUT_QUERY')-
               next(r['query_KL'] for r in subset if r['repetition']==rep and r['method']=='POOLED_TARGET_MI') for rep in range(32)]
        mean=math.fsum(delta)/32;se=math.sqrt(math.fsum((x-mean)**2 for x in delta)/(31*32))
        for key,value in [('mean_gain',mean),('SE',se),('lower',mean-critical*se),('upper',mean+critical*se)]:maximum=max(maximum,abs(float(row[key])-value))
        assert (row['practical_gain']=='True')==(mean-critical*se>.001)
    decision=B.read(OUT/'DECISION.json')['primary']; primary=next(r for r in intervals if r['primary']=='True')
    assert all(decision[key]==int(primary[key]) for key in ('d','Ntrain','Nval','b'))
    assert decision['practical_gain']==(primary['practical_gain']=='True')
    assert maximum<B.CFG['numeric_gate']
    return dict(passed=True,time_utc=B.now(),workspace=workspace,maximum_arithmetic_error=maximum,
        pooled_closed_form_fits_replayed=1280,old_HQ_references=512,intervals=8,new_random_draws=0,new_fits=0,
        raw_labels_replayed=workspace,production_selection_and_risk_functions_reused=True,
        scope='independent pooled integer-count/MI formulas and scalar confidence readouts; frozen B deterministic selection/fit/risk replay',timing_not_replayed=True)


def run():
    checkfreeze();assert B.read(OUT/'STATE.json')['status']=='frozen'
    B.write(OUT/'TOY_QA.json',toyqa());started=B.now()
    B.write(OUT/'START.json',dict(first_scientific_pooled_fit_utc=started,source_sha256=B.sha(__file__),freeze_sha256=B.sha(SEAL),new_random_draws=0))
    for rec in B.records():
        B.write(OUT/'STATE.json',dict(status='running',d=rec['d'],repetition=rec['repetition'],first_scientific_pooled_fit_utc=started,pid=os.getpid()),replace=True)
        run_rep(rec)
    collect();B.write(OUT/'AUDIT.json',audit(workspace=True))
    paths=list((OUT/'repetitions').glob('*.json'))+[OUT/name for name in ['TOY_QA.json','START.json','SCORES.csv','SUMMARY.csv','INTERVALS.csv','DECISION.json','COUNTS.json','AUDIT.json']]
    B.write(OUT/'RESULTS_FREEZE.json',dict(time_utc=B.now(),source_freeze_sha256=B.sha(SEAL),files=B.hashfiles(paths)))
    B.write(OUT/'STATE.json',dict(status='complete',time_utc=B.now(),first_scientific_pooled_fit_utc=started,active_jobs=[]),replace=True)
    print(json.dumps(dict(complete=True,decision=B.read(OUT/'DECISION.json'))),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['toyqa','freeze','run','check']);parser.add_argument('--workspace',action='store_true');args=parser.parse_args()
    try:
        if args.phase=='toyqa':print(json.dumps(toyqa()))
        elif args.phase=='freeze':freeze()
        elif args.phase=='run':run()
        else:
            B.verify(B.read(OUT/'RESULTS_FREEZE.json')['files']);assert B.read(OUT/'STATE.json')['status']=='complete';print(json.dumps(audit(args.workspace)))
    except Exception as error:
        if args.phase=='run' and SEAL.exists() and not (OUT/'RESULTS_FREEZE.json').exists():
            B.write(OUT/f'FAILURE_{time.time_ns()}.json',dict(time_utc=B.now(),error=repr(error),traceback=traceback.format_exc(),state=B.read(OUT/'STATE.json')))
            old=B.read(OUT/'STATE.json');old.update(status='stopped_failure',active_jobs=[]);B.write(OUT/'STATE.json',old,replace=True)
        raise


if __name__=='__main__':main()
