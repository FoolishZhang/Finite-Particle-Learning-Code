"""Frozen four-dimensional PMFs for unrestricted six-correlation outputs."""
from pathlib import Path
from datetime import datetime,timezone
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,json,time,traceback
import numpy as np
from scipy.stats import t as student_t
import scbminv_reference as helper

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/scbm_multicorrelation/REFERENCE'
TRAIN=ROOT/'artifacts/scbm_multicorrelation/TRAIN/RESULTS.json'
DOC=ROOT/'docs/08_research_研究工作/六相关自由学习与方向机制_20261003'
TRUTH=ROOT/'artifacts/scbm_invariance/REFERENCE/MODEL.json'
HELPER_SHA='79c911d1601ae9740b0e045a5d879394cfd4a82aaa9e48f72070fc46b5098a41'
METHODS=['FIXED','FREE','FACT'];KINDS=['TAIL','LAST']
PAIRS=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
CFG=dict(orders=[24,32,40],main_order=32,root='cholesky',maximum_PMF_change=5e-8,
    normalization_tolerance=2e-12,minimum_eigenvalue=.09999,streams=16,
    family_alpha=.05,family_size=3,noninferiority_margin=.0002,
    improvement_SE=.0002,noninferiority_SE=.00005,identity_reference=True,
    maximum_unique_covariances=97,maximum_PMF_integrations=291)
PROGRESS=dict(completed_PMF_integrations=0,completed_Gaussian_nodes=0,completed_unique_covariances=0,in_flight=None)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf8'))
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()


def save(p,x,new=True):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x' if new else 'w',encoding='utf8',newline='\n') as f:
        json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def covariance(rho):
    r=np.asarray(rho,dtype=np.float64);assert r.shape==(6,) and np.isfinite(r).all()
    C=np.eye(4)
    for v,(i,j) in zip(r,PAIRS):C[i,j]=C[j,i]=v
    return C


def key_of(rho):return '|'.join(float(x).hex() for x in rho)


def enumerate_outputs(train):
    identity=[0.]*6;unique={key_of(identity):identity};endpoints=[]
    assert [r['stream'] for r in train['records']]==list(range(16))
    for record in train['records']:
        for method in METHODS:
            for kind in KINDS:
                output=record['outputs'][method][kind];rho=[float(x) for x in output['rho']]
                C=covariance(rho)
                assert np.array_equal(C,np.asarray(output['covariance']))
                key=key_of(rho);unique[key]=rho
                endpoints.append(dict(stream=record['stream'],method=method,kind=kind,rho=rho,key=key))
    assert len(endpoints)==96 and len(unique)<=97
    return endpoints,unique


def primary_statistics(endpoints):
    lookup={(r['stream'],r['method'],r['kind']):r for r in endpoints}
    critical=float(student_t.ppf(1-.05/(2*3),15));primary=[]
    for name,left,right,direction,margin,threshold in [
        ('FIXED_minus_FREE','FIXED','FREE','positive',0.,.0002),
        ('FIXED_minus_FACT','FIXED','FACT','positive',0.,.0002),
        ('FREE_minus_FACT_noninferiority','FREE','FACT','upper_below_margin',.0002,.00005)]:
        values=np.asarray([lookup[s,left,'TAIL']['reference_joint_KL']-lookup[s,right,'TAIL']['reference_joint_KL'] for s in range(16)])
        mean=float(values.mean());se=float(values.std(ddof=1)/4);lower=mean-critical*se;upper=mean+critical*se
        primary.append(dict(name=name,kind='TAIL',stream_values=values.tolist(),mean=mean,SE=se,
            lower=lower,upper=upper,expected=direction,margin=margin,SE_threshold=threshold,
            effect_passed=bool(lower>0 if direction=='positive' else upper<margin),precision_passed=bool(se<=threshold)))
    descriptive=[]
    for kind in KINDS:
        for method in METHODS:
            rows=[lookup[s,method,kind] for s in range(16)]
            values=np.asarray([r['reference_joint_KL'] for r in rows]);rho=np.asarray([r['rho'] for r in rows])
            descriptive.append(dict(method=method,kind=kind,mean_joint_KL=float(values.mean()),SE_joint_KL=float(values.std(ddof=1)/4),
                mean_rho=rho.mean(axis=0).tolist(),min_rho=rho.min(axis=0).tolist(),max_rho=rho.max(axis=0).tolist(),endpoint_rho=rho.tolist()))
    return dict(primary_family=primary,critical_t=critical,primary_effects_passed=sum(r['effect_passed'] for r in primary),
                primary_precision_passed=sum(r['precision_passed'] for r in primary),descriptive=descriptive)


def freeze():
    assert sha(helper.__file__)==HELPER_SHA
    train=read(TRAIN);assert len(train['records'])==16
    train_freeze=read(TRAIN.parent/'FREEZE.json')
    assert train_freeze['reference_source_sha256']==sha(__file__)
    files=[Path(__file__),Path(helper.__file__),TRAIN,TRAIN.parent/'FREEZE.json',TRUTH,DOC/'EXPERIMENT_DESIGN.md']
    save(OUT/'FREEZE.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),helper_sha256=HELPER_SHA,
        config=CFG,inputs={rel(p):sha(p) for p in files},integrals_authorized=True,
        quadrature_policy_fixed_before_training=True,
        endpoint_policy='All96 TAIL/LAST rho vectors plus identity, deduplicated by ordered six float.hex values, without risk-based selection.'))


def run():
    frozen=read(OUT/'FREEZE.json')
    assert frozen['source_sha256']==sha(__file__) and frozen['config']==CFG and frozen['integrals_authorized']
    assert sha(helper.__file__)==frozen['helper_sha256']==HELPER_SHA
    for p,h in frozen['inputs'].items():assert sha(ROOT/p)==h,p
    save(OUT/'EXECUTION.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
        freeze_sha256=sha(OUT/'FREEZE.json'),automatic_retry=False))
    start=time.perf_counter();train=read(TRAIN);old_truth=read(TRUTH)
    assert old_truth['qualification_passed']
    truth=np.asarray(next(r['main_pmf'] for r in old_truth['records'] if r['t']==0.))
    endpoints,unique=enumerate_outputs(train);records=[]
    for key,rho in unique.items():
        C=covariance(rho);eigmin=float(np.linalg.eigvalsh(C).min());assert eigmin>0.
        variants=[]
        for order in CFG['orders']:
            PROGRESS['in_flight']=dict(key=key,rho=rho,order=order,partial_nodes_unknown_if_interrupted=True)
            save(OUT/'PROGRESS.json',PROGRESS,False)
            variants.append(helper.pmf(C,order,'cholesky'))
            PROGRESS['completed_PMF_integrations']+=1;PROGRESS['completed_Gaussian_nodes']+=order**4
            PROGRESS['in_flight']=None;save(OUT/'PROGRESS.json',PROGRESS,False)
        main=np.asarray(next(v['pmf'] for v in variants if v['order']==32));quals=[]
        for variant in variants:
            p=np.asarray(variant['pmf']);change=float(np.max(abs(p-main)));norm=abs(float(p.sum())-1.)
            quals.append(dict(order=variant['order'],maximum_change=change,normalization_error=norm,positive=bool(np.all(p>0)),
                passed=bool(change<=CFG['maximum_PMF_change'] and norm<=CFG['normalization_tolerance'] and np.all(p>0))))
        risk=float(np.sum(truth*np.log(truth/main)))
        row=dict(key=key,rho=rho,covariance=C.tolist(),minimum_eigenvalue=eigmin,main_pmf=main.tolist(),
            reference_joint_KL=risk,variants=variants,qualifications=quals,
            passed=bool(eigmin>=CFG['minimum_eigenvalue'] and all(q['passed'] for q in quals)))
        records.append(row);save(OUT/f'covariance_{len(records)-1:03}.json',row)
        PROGRESS['completed_unique_covariances']=len(records);save(OUT/'PROGRESS.json',PROGRESS,False)
    table={r['key']:r for r in records}
    for row in endpoints:row['reference_joint_KL']=table[row['key']]['reference_joint_KL']
    qualified=all(r['passed'] for r in records)
    result=dict(source_sha256=sha(__file__),helper_sha256=HELPER_SHA,freeze_sha256=sha(OUT/'FREEZE.json'),
        config=CFG,truth_reference_sha256=sha(TRUTH),training_results_sha256=sha(TRAIN),pair_order=[list(p) for p in PAIRS],
        records=records,endpoints=endpoints,identity_reference=table[key_of([0.]*6)],
        qualification_passed=qualified,primary_interpretation_qualified=qualified,**primary_statistics(endpoints),
        accounting=dict(unique_covariances=len(records),trained_outputs=96,identity_baselines=1,
            PMF_integrations=3*len(records),scalar_pattern_integrals=48*len(records),
            Gaussian_nodes=len(records)*sum(o**4 for o in CFG['orders']),new_truth_integrals=0,new_random_draws=0,new_parameter_fits=0),
        seconds=time.perf_counter()-start,
        scope='Direct general four-variable PMFs for six-correlation outputs, not interpolation on the old true ray. Numerical qualification, not a rigorous quadrature-remainder certificate. Identity/LAST are descriptive; original new TAIL three-comparison family only.')
    save(OUT/'RESULTS.json',result)
    print(json.dumps({k:result[k] for k in ('qualification_passed','primary_effects_passed','primary_precision_passed','accounting','seconds')},indent=2))


def check():
    # Synthetic output enumeration/linear algebra only; the quadrature helper
    # is not called. This check is safe before scientific freeze.
    records=[]
    for s in range(16):
        outputs={}
        for mi,m in enumerate(METHODS):
            outputs[m]={}
            for ki,k in enumerate(KINDS):
                rho=[.001*(s+1)+.003*mi+.0001*ki+.00001*j for j in range(6)]
                outputs[m][k]=dict(rho=rho,covariance=covariance(rho).tolist(),scales=[1.]*256)
        records.append(dict(stream=s,outputs=outputs))
    endpoints,unique=enumerate_outputs(dict(records=records));assert len(endpoints)==96 and len(unique)==97
    assert key_of([0.]*6)!=key_of([-0.]+[0.]*5)
    for row in endpoints:row['reference_joint_KL']={'FIXED':.001,'FREE':.0004,'FACT':.0003}[row['method']]+row['stream']*1e-6
    stats=primary_statistics(endpoints);assert len(stats['primary_family'])==3 and len(stats['descriptive'])==6
    save(OUT/'SELFTEST.json',dict(passed=True,source_sha256=sha(__file__),helper_sha256=sha(helper.__file__),
        unique_synthetic_covariances=len(unique),synthetic_outputs=len(endpoints),pair_order=[list(p) for p in PAIRS],
        new_model_integrals=0,new_random_draws=0,new_parameter_fits=0,
        scope='Only deterministic schema, exact coordinate-key handling and stored-number statistics; no production output or quadrature evaluated.'))
    print(json.dumps(dict(passed=True,source_sha256=sha(__file__),new_integrals=0)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['schema','check','freeze','run']);mode=parser.parse_args().mode
    if mode=='schema':print(json.dumps(dict(source_sha256=sha(__file__),helper_sha256=sha(helper.__file__),config=CFG)))
    else:
        try:globals()[mode]()
        except BaseException:
            if mode=='run' and not (OUT/'FAILURE.json').exists():save(OUT/'FAILURE.json',dict(utc=datetime.now(timezone.utc).isoformat(),traceback=traceback.format_exc(),progress=PROGRESS,automatic_retry=False))
            raise
