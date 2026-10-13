"""Prespecified likelihood-reference curves, using frozen PMFs where available."""
from pathlib import Path
from datetime import datetime, timezone
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse, hashlib, json, time, traceback
import numpy as np
import scbminv_reference as helper

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/scbm_frozen_evaluation/REFERENCE'
SEL=ROOT/'artifacts/scbm_frozen_evaluation/MODEL_SELECTION.json'
DOC=ROOT/'docs/08_research_研究工作/冻结表示的条件与总体目标_20261003'
OLD=ROOT/'artifacts/scbm_invariance/REFERENCE/MODEL.json'
OPT=ROOT/'artifacts/scbm_scale_optimization/REFERENCE/RESULTS.json'
CFG=dict(orders=[24,32,40],main_order=32,max_PMF_change=5e-8,
         normalization_tolerance=2e-12,minimum_eigenvalue=.12,new_grid_indices=[i for i in range(21) if i not in (0,10,20)])
PROGRESS=dict(completed_PMF_integrations=0,completed_nodes=0,in_flight=None)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf8'))
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def save(p,x,exclusive=True):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x' if exclusive else 'w',encoding='utf8',newline='\n') as f:
        json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def freeze():
    files=[Path(__file__),Path(helper.__file__),SEL,OLD,OPT,DOC/'EXPERIMENT_DESIGN.md']
    save(OUT/'FREEZE.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__),config=CFG,inputs={rel(p):sha(p) for p in files},
        integrals_authorized=True,old_science_unchanged=True))
def run():
    f=read(OUT/'FREEZE.json');assert f['config']==CFG and f['source_sha256']==sha(__file__) and f['integrals_authorized']
    for p,h in f['inputs'].items():assert sha(ROOT/p)==h,p
    save(OUT/'EXECUTION.json',dict(start_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__),freeze_sha256=sha(OUT/'FREEZE.json'),automatic_retry=False))
    start=time.perf_counter();sel=read(SEL);old=read(OLD);opt=read(OPT)
    assert sel['grid_t'][10]==0. and sel['grid_t'][0]==-.75 and sel['grid_t'][20]==.75
    table={};records=[];qual=[]
    for i in (0,10,20):
        t=sel['grid_t'][i];row=next(x for x in old['records'] if x['t']==t)
        table[float(t).hex()]=dict(t=t,pmf=row['main_pmf'],origin='old_scbminv',source_sha256=sha(OLD))
    for i in CFG['new_grid_indices']:
        t=sel['grid_t'][i];cov=np.eye(4)+(1+t)*(helper.COV-np.eye(4))
        assert np.linalg.eigvalsh(cov).min()>CFG['minimum_eigenvalue']
        variants=[]
        for order in CFG['orders']:
            PROGRESS['in_flight']=dict(grid_index=i,t=t,order=order,partial_nodes_unknown_if_interrupted=True)
            save(OUT/'PROGRESS.json',PROGRESS,False)
            variants.append(helper.pmf(cov,order,'cholesky'))
            PROGRESS['completed_PMF_integrations']+=1;PROGRESS['completed_nodes']+=order**4
            PROGRESS['in_flight']=None;save(OUT/'PROGRESS.json',PROGRESS,False)
        main=next(x['pmf'] for x in variants if x['order']==32)
        checks=[]
        for v in variants:
            p=np.asarray(v['pmf']);change=float(np.max(abs(p-main)));norm=abs(float(p.sum())-1.)
            checks.append(dict(grid_index=i,order=v['order'],max_change=change,normalization_error=norm,
                passed=bool(change<=CFG['max_PMF_change'] and norm<=CFG['normalization_tolerance'] and np.all(p>0))))
        record=dict(grid_index=i,t=t,main_pmf=main,variants=variants,qualifications=checks,passed=all(x['passed'] for x in checks))
        records.append(record);qual.extend(checks);save(OUT/f'grid_{i:02}.json',record)
        table[float(t).hex()]=dict(t=t,pmf=main,origin='new_grid',grid_index=i)
    actual_reused=[]
    for model in sel['models']:
        for kind in ['TAIL','LAST']:
            t=model[kind]['t'];key=float(t).hex()
            if key not in table:
                row=next(x for x in opt['records'] if x['key']==key)
                assert row['passed']
                table[key]=dict(t=t,pmf=row['main_pmf'],origin='old_scbmopt',source_sha256=sha(OPT))
            actual_reused.append(dict(stream=model['stream'],kind=kind,t=t,key=key,origin=table[key]['origin']))
    pop=np.asarray(sel['population_weights']);profiles=[('TAIL','learned'),('TAIL','half'),('TAIL','fact'),('LAST','learned'),('LAST','fact')]
    CE=np.empty((16,5,22,2));KL=np.empty((16,5,22))
    for m,model in enumerate(sel['models']):
        empirical=np.asarray(model['empirical_weights'])
        for c,(kind,_) in enumerate(profiles):
            ts=sel['grid_t']+[model[kind]['t']]
            for j,t in enumerate(ts):
                pmf=np.asarray(table[float(t).hex()]['pmf'])
                CE[m,c,j]=[-np.sum(pop*np.log(pmf)),-np.sum(empirical*np.log(pmf))]
                KL[m,c,j]=np.sum(pop*np.log(pop/pmf))
    relative=CE-CE[:,:,10:11,:]
    result=dict(source_sha256=sha(__file__),freeze_sha256=sha(OUT/'FREEZE.json'),config=CFG,
        selection_sha256=sha(SEL),records=records,PMF_table=table,actual_reused=actual_reused,
        profiles=profiles,targets=['P','D'],cross_entropy=CE.tolist(),relative_cross_entropy=relative.tolist(),
        reference_joint_KL=KL.tolist(),qualifications=qual,qualification_passed=all(q['passed'] for q in qual),
        accounting=dict(new_PMF_integrations=54,new_scalar_pattern_integrals=864,new_Gaussian_nodes=70926336,
            reused_grid_queries=3,reused_actual_queries=32,new_random_draws=0,new_fits=0),
        seconds=time.perf_counter()-start,
        scope='Qualified numerical likelihood reference, not exact integration or interpolated gradients. All profiles have the same integrated observation law at each t; actual points and three grid PMFs reuse old frozen results.')
    assert PROGRESS['completed_PMF_integrations']==54 and PROGRESS['completed_nodes']==70926336
    save(OUT/'RESULTS.json',result)
    print(json.dumps(dict(qualification_passed=result['qualification_passed'],accounting=result['accounting'],seconds=result['seconds'])),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['schema','freeze','run']);a=p.parse_args()
    if a.mode=='schema':print(json.dumps(dict(config=CFG,source_sha256=sha(__file__),helper_sha256=sha(helper.__file__))))
    else:
        try:globals()[a.mode]()
        except BaseException:
            if a.mode=='run' and not (OUT/'FAILURE.json').exists():
                save(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),progress=PROGRESS,automatic_retry=False))
            raise
