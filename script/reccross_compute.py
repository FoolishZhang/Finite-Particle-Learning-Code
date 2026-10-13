"""REC-v5 independent cross-objective evaluation of all immutable REC-v4 TAILs."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,hashlib,itertools,json,math,platform,subprocess,time,traceback
import numpy as np
import scipy
from scipy.special import logsumexp
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/rec_cross_v5_20261005'
OLD=ROOT/'artifacts/rec_learning_v4_20261005'
DOC=ROOT/'docs/08_research_研究工作/REC_v5_cross_20261005'
BASE='786e9061c83fcb13297428b04d17827892347385'
CASES=('NULL','MIXED');ARMS=('OUTER100x4','INNER400');TARGETS=('F_D','F_P')
Y=np.array(list(itertools.product((0,1),repeat=4)),dtype=int)
CONFIG=dict(seed=202610055010,banks=4096,chunk=32,particles=400,blocks=4,K=[100,400],
            reps=32,cases=list(CASES),workers=4,threads=1,alpha=.05,family=12,df=4095,
            SE_limit=1e-5,reference_tolerance=2e-6,jensen_tolerance=2e-12,
            audit_reps=[0,8,16,24],audit_banks=2)

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rel(p):return Path(p).relative_to(ROOT).as_posix()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    def default(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    with p.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False,default=default);f.write('\n')

def scores(model,eps):
    z=eps@np.linalg.cholesky(model['covariance']).T+np.asarray(model['mu'])
    loga=z@Y.T-np.logaddexp(0.,z).sum(axis=2)[:,:,None]
    block=math.log(100)-logsumexp(loga.reshape(len(eps),4,100,16),axis=2)
    inner=math.log(400)-logsumexp(loga,axis=1)
    result=np.stack((block.mean(axis=1),inner),axis=1)
    assert np.isfinite(result).all()
    assert float(np.min(result[:,0]-result[:,1]))>=-2e-12
    return result

def old_paths():
    for case in CASES:
        for rep in range(32):yield case,rep,OLD/case/f'rep_{rep:02}'/'RESULT.json'

def prepare():
    rows=[]
    for case,rep,path in old_paths():
        r=read(path);assert r['qualified']
        trace=ROOT/r['trace_file'];assert sha(trace)==r['trace_sha256']
        with np.load(trace,allow_pickle=False) as f:labels=f['labels']
        assert labels.shape==(8192,4) and ah(labels)==r['data_hashes']['labels']
        assert labels.sum(axis=0).tolist()==r['data_counts']
        counts=np.bincount(labels.astype(int)@np.array([8,4,2,1]),minlength=16)
        T=counts/8192;P=np.array(r['teacher_P']);weights=np.stack((T,P),axis=1);models=[]
        for arm,source in zip(ARMS,r['rows']):
            assert source['arm']==arm and source['qualified']
            q=np.array(source['main_pmf']);loss=-np.log(q)@weights
            changes=[float(np.max(abs(-np.log(v['pmf'])@weights-loss))) for v in source['variants']]
            assert max(changes)<=2e-6
            assert abs(float(P@np.log(P/q))-source['risk'])<2e-12
            models.append(dict(arm=arm,model=source['model'],pmf=q,exact_losses=loss,
                               risk=source['risk'],reference_loss_changes=changes))
        rows.append(dict(case=case,rep=rep,source=rel(path),source_sha256=sha(path),
                         trace_file=r['trace_file'],trace_sha256=r['trace_sha256'],
                         labels_sha256=r['data_hashes']['labels'],counts=counts,empirical=T,teacher_P=P,models=models))
    save(OUT/'INPUT_PANEL.json',dict(rows=rows,scope='Derived only from sealed models/PMFs and original labels; no new data or quadrature'))
    print(json.dumps(dict(prepared=True,streams=64,models=128)))

def freeze():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==BASE
    q=read(OUT/'QUALIFICATION.json');assert q['passed']
    assert q['producer_sha256']==sha(__file__) and q['audit_sha256']==sha(ROOT/'script/reccross_audit.py')
    sources=[Path(__file__),ROOT/'script/reccross_audit.py',DOC/'CONTRACT.md',ROOT/'script/jointscale_kernel.py']
    inputs=[OUT/'INPUT_PANEL.json',ROOT/'artifacts/research_advance_20261005/DELIVERY_MANIFEST.json',
            ROOT/'artifacts/use_margin_v4_20261005/HANDOFF_INPUT.md']+[p for _,_,p in old_paths()]
    panel=read(OUT/'INPUT_PANEL.json')
    save(OUT/'FREEZE.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),baseline=BASE,config=CONFIG,
        sources={rel(p):sha(p) for p in sources},inputs={rel(p):sha(p) for p in inputs},
        private_inputs={r['trace_file']:r['trace_sha256'] for r in panel['rows']},
        qualification_sha256=sha(OUT/'QUALIFICATION.json'),environment=q['environment'],
        feedback='Unchanged HANDOFF, reference as already recorded; latest user explicitly authorizes planned next step'))
    print(json.dumps(dict(frozen=True,sha256=sha(OUT/'FREEZE.json'))))

def validate(public_only=False):
    f=read(OUT/'FREEZE.json');assert f['config']==CONFIG
    for group in (('sources','inputs') if public_only else ('sources','inputs','private_inputs')):
        for name,h in f[group].items():assert sha(ROOT/name)==h,name
    assert sha(OUT/'QUALIFICATION.json')==f['qualification_sha256']

def one(ci,rep):
    tick=time.perf_counter();name=f'{CASES[ci]}_{rep:02}'
    try:
        row=next(r for r in read(OUT/'INPUT_PANEL.json')['rows'] if r['case']==CASES[ci] and r['rep']==rep)
        weights=np.stack((row['empirical'],row['teacher_P']),axis=1)
        rng=np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([CONFIG['seed'],ci,rep,0])))
        pattern=np.empty((4096,2,2,16));hashes=[]
        for begin in range(0,4096,32):
            eps=rng.standard_normal((32,400,4));hashes.append(dict(start=begin,sha256=ah(eps)))
            if begin==0:prefix_hash=ah(eps[:2])
            for arm in range(2):pattern[begin:begin+32,arm]=scores(row['models'][arm]['model'],eps)
        weighted=pattern@weights
        path=OUT/(name+'.npz');assert not path.exists();np.savez_compressed(path,pattern_scores=pattern,weighted_scores=weighted)
        delta=weighted[:,1]-weighted[:,0]
        result=dict(case=CASES[ci],rep=rep,score_file=rel(path),score_sha256=sha(path),bank_hashes=hashes,
                    first_two_bank_prior_sha256=prefix_hash,
                    first_bank_pattern_scores=pattern[0],means=weighted.mean(axis=0),
                    delta_means=delta.mean(axis=0),delta_SE=delta.std(axis=0,ddof=1)/64,
                    minimum_Jensen_gap=float(np.min(pattern[:,:,0]-pattern[:,:,1])),seconds=time.perf_counter()-tick)
        save(OUT/(name+'.json'),result);return result
    except Exception:save(OUT/('ERROR_'+name+'.json'),dict(traceback=traceback.format_exc(),automatic_retry=False));raise

def mc_cell(data):
    assert len(data)==4096
    critical=float(t.ppf(1-.05/24,4095))
    mean=float(data.mean());se=float(data.std(ddof=1)/64);ci=[mean-critical*se,mean+critical*se]
    return dict(mean=mean,SE=se,interval=ci,precision_passed=bool(se<=1e-5),
                direction='INNER' if ci[1]<0 else 'OUTER' if ci[0]>0 else 'unresolved')

def family_from_arrays(records):
    family=[];exact=[];classification=[];crit=float(t.ppf(1-.05/24,4095))
    panel=read(OUT/'INPUT_PANEL.json')['rows']
    for case in CASES:
        group=[r for r in records if r['case']==case];values=[]
        for row in group:
            with np.load(ROOT/row['score_file'],allow_pickle=False) as f:
                values.append(f['weighted_scores'][:,1]-f['weighted_scores'][:,0])
        stack=np.stack(values);panel_delta=stack.mean(axis=0)
        for target in range(2):
            for name,data in [('K100',panel_delta[:,0,target]),('K400',panel_delta[:,1,target]),
                              ('K400-K100',panel_delta[:,1,target]-panel_delta[:,0,target])]:
                family.append(dict(case=case,target=TARGETS[target],contrast=name,**mc_cell(data)))
            cells=[next(r for r in family if r['case']==case and r['target']==TARGETS[target] and r['contrast']==name) for name in ('K100','K400')]
            signs=tuple(x['direction'] for x in cells)
            label={('OUTER','INNER'):'aligned_target_split',('INNER','OUTER'):'reverse_target_split',
                   ('OUTER','OUTER'):'both_prefer_OUTER',('INNER','INNER'):'both_prefer_INNER'}.get(signs,'unresolved')
            if not all(x['precision_passed'] for x in cells):label='unresolved_precision'
            classification.append(dict(case=case,target=TARGETS[target],result=label,
                point_sign_counts=dict(both_OUTER=int(np.sum(np.all(stack.mean(axis=1)[:,:,target]>0,axis=1))),
                                       both_INNER=int(np.sum(np.all(stack.mean(axis=1)[:,:,target]<0,axis=1))),
                                       aligned_split=int(np.sum((stack.mean(axis=1)[:,0,target]>0)&(stack.mean(axis=1)[:,1,target]<0))),
                                       reverse_split=int(np.sum((stack.mean(axis=1)[:,0,target]<0)&(stack.mean(axis=1)[:,1,target]>0))))))
        differences=[np.array(r['models'][1]['exact_losses'])-r['models'][0]['exact_losses'] for r in panel if r['case']==case]
        risks=[r['models'][1]['risk']-r['models'][0]['risk'] for r in panel if r['case']==case]
        assert abs(float(np.mean(differences,axis=0)[1])-float(np.mean(risks)))<2e-12
        exact.append(dict(case=case,per_stream_exact_deltas=np.array(differences),mean_exact_deltas=np.mean(differences,axis=0),
                          per_stream_risk_deltas=risks,mean_risk_delta=float(np.mean(risks))))
    return family,exact,classification

def run():
    validate();tick=time.perf_counter();save(OUT/'EXECUTION.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),automatic_retry=False))
    try:
        rows=[]
        with ProcessPoolExecutor(max_workers=4) as pool:
            for f in as_completed([pool.submit(one,ci,rep) for ci in range(2) for rep in range(32)]):
                row=f.result();rows.append(row);print(json.dumps(dict(case=row['case'],rep=row['rep'],seconds=row['seconds'])),flush=True)
        rows.sort(key=lambda r:(CASES.index(r['case']),r['rep']));validate();family,exact,classification=family_from_arrays(rows)
        save(OUT/'RESULTS.json',dict(config=CONFIG,freeze_sha256=sha(OUT/'FREEZE.json'),records=rows,family=family,
            exact=exact,classification=classification,seconds=time.perf_counter()-tick,
            accounting=dict(new_observations=0,new_training=0,new_models=0,new_quadrature_nodes=0,
                            prior_vectors=104857600,Gaussian=419430400,model_particle_evaluations=209715200,
                            scalar_logits=838860800,pattern_scores=16777216,audit_replay_vectors=6400,audit_replay_Gaussian=25600),
            scope='Conditional MC inference on frozen REC-v4 endpoints and weights; no new training-population inference'))
        print(json.dumps(dict(completed=True,seconds=time.perf_counter()-tick)),flush=True)
    except Exception:save(OUT/'ERROR.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','freeze','run'))
    globals()[p.parse_args().command]()
