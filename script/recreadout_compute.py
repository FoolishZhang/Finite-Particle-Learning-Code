"""REC-v6: prespecified LAST/TAIL readouts of fixed REC-v4 trajectories."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,math,subprocess,time,traceback
import numpy as np
from scipy.stats import t
import reccross_compute as cross

ROOT=cross.ROOT;OUT=ROOT/'artifacts/rec_readout_v6_20261005';DOC=ROOT/'docs/08_research_研究工作/REC_v6_readout_20261005'
OLD=ROOT/'artifacts/rec_learning_v4_20261005';BASE='4d09c6b7572487aa7cec7609cc88b18bf507e5d1'
CASES=('NULL','MIXED');ARMS=('OUTER100x4','INNER400');READOUTS=('LAST','TAIL');TARGETS=('F_D','F_P')
PAIRS=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))
CONFIG=dict(seed=202610056010,reps=32,banks=4096,chunk=32,particles=400,K=[100,400],readouts=list(READOUTS),
    reference_orders=[32,40,48],main_order=40,pmf_tolerance=2e-7,loss_tolerance=2e-6,mass_tolerance=2e-12,
    alpha=.05,family=24,df=4095,SE_limit=1e-5,audit_reps=[0,8,16,24],audit_banks=2,
    audit_reference_reps=[0,16],audit_order=44,workers=4,threads=1)
read=cross.read;save=cross.save;sha=cross.sha;rel=cross.rel;ah=cross.ah

def physical(vector):
    v=np.asarray(vector,dtype=float);assert v.shape==(14,)
    C=np.eye(4)
    for value,(i,j) in zip(v[8:],PAIRS):C[i,j]=C[j,i]=value
    s=v[4:8];cov=s[:,None]*C*s[None,:]
    assert np.min(np.linalg.eigvalsh(C))>=.1-1e-12
    assert np.max(abs(v[:4]))<=2+1e-12 and min(s)>=.001-1e-12 and max(s)<=2+1e-12
    return dict(mu=v[:4].tolist(),s=s.tolist(),rho=v[8:].tolist(),correlation=C.tolist(),covariance=cov.tolist())

def average(vectors):
    states=[physical(v) for v in vectors]
    return dict(mu=np.mean([m['mu'] for m in states],axis=0).tolist(),
                covariance=np.mean([m['covariance'] for m in states],axis=0).tolist())

def prepare():
    rows=[]
    for ci,case in enumerate(CASES):
        for rep in range(32):
            source=OLD/case/f'rep_{rep:02}'/'RESULT.json';old=read(source)
            assert old['qualified'];trace=ROOT/old['trace_file'];assert sha(trace)==old['trace_sha256']
            with np.load(trace,allow_pickle=False) as f:labels=f['labels'];params=f['params']
            assert params.shape==(2,513,14) and labels.shape==(8192,4)
            assert ah(labels)==old['data_hashes']['labels'] and labels.sum(axis=0).tolist()==old['data_counts']
            counts=np.bincount(labels.astype(int)@np.array([8,4,2,1]),minlength=16)
            models=[[],[]]
            for ai,arm in enumerate(ARMS):
                tail=old['rows'][ai];assert tail['arm']==arm
                reconstructed=average(params[ai,-64:])
                for key in ('mu','covariance'):assert np.max(abs(np.array(reconstructed[key])-tail['model'][key]))<=2e-14
                models[0].append(dict(arm=arm,readout='LAST',model=physical(params[ai,512]),physical_vector=params[ai,512].tolist(),reference=None))
                models[1].append(dict(arm=arm,readout='TAIL',model=tail['model'],reference=dict(pmf=tail['main_pmf'],variants=tail['variants'],risk=tail['risk'])))
            rows.append(dict(case=case,rep=rep,source=rel(source),source_sha256=sha(source),trace_file=old['trace_file'],
                trace_sha256=old['trace_sha256'],labels_sha256=old['data_hashes']['labels'],counts=counts,empirical=counts/8192,
                teacher_P=old['teacher_P'],models=models))
    save(OUT/'INPUT_PANEL.json',dict(rows=rows,scope='Fixed LAST and original TAIL extraction; LAST has no scientific evaluation before freeze'))
    print('{"prepared":true,"readout_models":256}')

def freeze():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==BASE
    q=read(OUT/'QUALIFICATION.json');assert q['passed'] and q['producer_sha256']==sha(__file__) and q['audit_sha256']==sha(ROOT/'script/recreadout_audit.py')
    sources=[Path(__file__),ROOT/'script/recreadout_audit.py',DOC/'CONTRACT.md',ROOT/'script/reccross_compute.py',
             ROOT/'script/reccross_audit.py',ROOT/'script/jointscale_compute.py',ROOT/'script/jointscale_kernel.py',ROOT/'script/rec1_verify.py']
    panel=read(OUT/'INPUT_PANEL.json')['rows']
    inputs=[OUT/'INPUT_PANEL.json',ROOT/'artifacts/rec_cross_v5_20261005/DELIVERY_MANIFEST.json',ROOT/'artifacts/use_margin_v4_20261005/HANDOFF_INPUT.md']+[ROOT/r['source'] for r in panel]
    save(OUT/'FREEZE.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),baseline=BASE,config=CONFIG,
        sources={rel(p):sha(p) for p in sources},inputs={rel(p):sha(p) for p in inputs},private_inputs={r['trace_file']:r['trace_sha256'] for r in panel},
        qualification_sha256=sha(OUT/'QUALIFICATION.json'),environment=q['environment'],feedback='Unchanged HANDOFF; current user authorizes planned readout sensitivity'))
    print('{"frozen":true,"sha256":"'+sha(OUT/'FREEZE.json')+'"}')

def validate(public_only=False):
    f=read(OUT/'FREEZE.json');assert f['config']==CONFIG
    for group in (('sources','inputs') if public_only else ('sources','inputs','private_inputs')):
        for name,h in f[group].items():assert sha(ROOT/name)==h,name
    assert sha(OUT/'QUALIFICATION.json')==f['qualification_sha256']

def reference_model(entry,weights):
    if entry['readout']=='LAST':
        import jointscale_compute as legacy
        model=entry['model'];variants=[legacy.pmf(np.array(model['mu']),np.array(model['covariance']),n) for n in (32,40,48)]
        q=np.array(variants[1]['pmf'])
    else:variants=entry['reference']['variants'];q=np.array(entry['reference']['pmf'])
    losses=-np.log(q)@weights;P=weights[:,1];risk=float(P@np.log(P/q));checks=[]
    for v in variants:
        value=np.array(v['pmf']);pe=float(np.max(abs(value-q)));le=float(np.max(abs(-np.log(value)@weights-losses)))
        re=abs(float(P@np.log(P/value))-risk)
        checks.append(dict(order=v['order'],pmf_error=pe,loss_error=le,risk_error=re,mass_error=v['mass_error'],
            passed=bool(pe<=2e-7 and le<=2e-6 and re<=2e-6 and v['mass_error']<=2e-12 and min(value)>0)))
    return dict(arm=entry['arm'],readout=entry['readout'],pmf=q,exact_losses=losses,risk=risk,
                variants=variants,qualification=checks,qualified=all(v['passed'] for v in checks))

def one(ci,rep):
    tick=time.perf_counter();stem=f'{CASES[ci]}_{rep:02}'
    try:
        row=next(r for r in read(OUT/'INPUT_PANEL.json')['rows'] if (r['case'],r['rep'])==(CASES[ci],rep))
        weights=np.stack((row['empirical'],row['teacher_P']),axis=1)
        references=[[reference_model(m,weights) for m in entries] for entries in row['models']]
        rng=np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([CONFIG['seed'],ci,rep,0])))
        pattern=np.empty((4096,2,2,2,16));hashes=[]
        for begin in range(0,4096,32):
            eps=rng.standard_normal((32,400,4));hashes.append(dict(start=begin,sha256=ah(eps)))
            if begin==0:prefix=ah(eps[:2])
            for ri in range(2):
                for ai in range(2):pattern[begin:begin+32,ri,ai]=cross.scores(row['models'][ri][ai]['model'],eps)
        weighted=pattern@weights;delta=weighted[:,:,1]-weighted[:,:,0]
        path=OUT/(stem+'.npz');assert not path.exists();np.savez_compressed(path,pattern_scores=pattern,weighted_scores=weighted)
        result=dict(case=CASES[ci],rep=rep,references=references,qualified=all(m['qualified'] for x in references for m in x),
            score_file=rel(path),score_sha256=sha(path),bank_hashes=hashes,first_two_bank_prior_sha256=prefix,
            first_bank_pattern_scores=pattern[0],means=weighted.mean(axis=0),delta_means=delta.mean(axis=0),
            delta_SE=delta.std(axis=0,ddof=1)/64,minimum_Jensen_gap=float(np.min(pattern[:,:,:,0]-pattern[:,:,:,1])),seconds=time.perf_counter()-tick)
        save(OUT/(stem+'.json'),result);return result
    except Exception:save(OUT/('ERROR_'+stem+'.json'),dict(traceback=traceback.format_exc(),automatic_retry=False));raise

def mc_cell(values):
    assert len(values)==4096;mean=float(np.mean(values));se=float(np.std(values,ddof=1)/64)
    critical=float(t.ppf(1-.05/48,4095));ci=[mean-critical*se,mean+critical*se]
    return dict(mean=mean,SE=se,interval=ci,precision_passed=bool(se<=1e-5),
                direction='negative' if ci[1]<0 else 'positive' if ci[0]>0 else 'unresolved')

def summarize(records):
    family=[];exact=[];classification=[]
    for case in CASES:
        group=[r for r in records if r['case']==case];values=[]
        for row in group:
            with np.load(ROOT/row['score_file'],allow_pickle=False) as f:values.append(f['weighted_scores'][:,:,1]-f['weighted_scores'][:,:,0])
        values=np.stack(values);average_delta=values.mean(axis=0)
        for ti,target in enumerate(TARGETS):
            for ki,k in enumerate((100,400)):
                for label,data in [('LAST',average_delta[:,0,ki,ti]),('TAIL',average_delta[:,1,ki,ti]),
                                   ('INTERACTION',average_delta[:,0,ki,ti]-average_delta[:,1,ki,ti])]:
                    family.append(dict(case=case,target=target,K=k,contrast=label,**mc_cell(data)))
            for ri,label in enumerate(READOUTS):
                cells=[next(r for r in family if (r['case'],r['target'],r['K'],r['contrast'])==(case,target,k,label)) for k in (100,400)]
                signs=tuple(r['direction'] for r in cells)
                name={('positive','positive'):'both_prefer_OUTER',('negative','negative'):'both_prefer_INNER',
                      ('positive','negative'):'aligned_target_split',('negative','positive'):'reverse_target_split'}.get(signs,'unresolved')
                if not all(r['precision_passed'] for r in cells):name='unresolved_precision'
                estimates=values.mean(axis=1)[:,ri,:,ti]
                counts=dict(both_OUTER=int(np.sum(np.all(estimates>0,axis=1))),both_INNER=int(np.sum(np.all(estimates<0,axis=1))),
                    aligned_split=int(np.sum((estimates[:,0]>0)&(estimates[:,1]<0))),reverse_split=int(np.sum((estimates[:,0]<0)&(estimates[:,1]>0))))
                classification.append(dict(case=case,target=target,readout=label,result=name,point_sign_counts=counts))
        losses=np.array([[[m['exact_losses'] for m in x] for x in r['references']] for r in group])
        risks=np.array([[[m['risk'] for m in x] for x in r['references']] for r in group])
        exact.append(dict(case=case,mean_losses=losses.mean(axis=0),mean_risks=risks.mean(axis=0),
            arm_loss_deltas=(losses[:,:,1]-losses[:,:,0]).mean(axis=0),arm_risk_deltas=(risks[:,:,1]-risks[:,:,0]).mean(axis=0),
            readout_loss_deltas=(losses[:,0]-losses[:,1]).mean(axis=0),readout_risk_deltas=(risks[:,0]-risks[:,1]).mean(axis=0)))
    return dict(config=CONFIG,freeze_sha256=sha(OUT/'FREEZE.json'),records=records,family=family,exact=exact,classification=classification,
        qualified=all(r['qualified'] for r in records),
        accounting=dict(new_observations=0,new_training=0,new_optimizations=0,fixed_readout_models=256,new_LAST_references=128,
            prior_vectors=104857600,Gaussian=419430400,model_particle_evaluations=419430400,scalar_logits=1677721600,
            pattern_scores=33554432,reference_nodes=1141374976,audit_reference_nodes=29984768,audit_replay_vectors=6400,audit_replay_Gaussian=25600),
        scope='Fixed old trajectories; new LAST quadrature and independent MC readout sensitivity, not a new training population study')

def run():
    validate();tick=time.perf_counter();save(OUT/'EXECUTION.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),automatic_retry=False))
    try:
        rows=[]
        with ProcessPoolExecutor(max_workers=4) as pool:
            for future in as_completed([pool.submit(one,ci,rep) for ci in range(2) for rep in range(32)]):
                r=future.result();rows.append(r);print('{"case":"'+r['case']+'","rep":'+str(r['rep'])+',"qualified":'+str(r['qualified']).lower()+',"seconds":'+str(r['seconds'])+'}',flush=True)
        rows.sort(key=lambda r:(CASES.index(r['case']),r['rep']));validate();result=summarize(rows);result['seconds']=time.perf_counter()-tick
        save(OUT/'RESULTS.json',result);print('{"completed":true,"seconds":'+str(result['seconds'])+'}',flush=True)
    except Exception:save(OUT/'ERROR.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','freeze','run'));globals()[p.parse_args().command]()
