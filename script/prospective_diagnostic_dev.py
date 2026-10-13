"""Frozen independent development panel; no formal teacher or formal RNG use.

Stages are explicit and globally locked: freeze -> generate -> diagnose ->
followup -> evaluate. Existing scientific outputs are never overwritten.
"""
from __future__ import annotations
import os
for _thread_key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_thread_key] = '1'
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import numpy as np
import scipy
from scipy.special import expit
import joint_dependence_kernel as joint

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT/'docs/08_research_研究工作/ProspectiveDiagnosticDev_v1_20261007'
PLAN = ROOT/'docs/08_research_研究工作/ProspectiveDiagnostic_v2_20261007'
OUT = ROOT/'artifacts/prospective_diagnostic_dev_v1_20261007'
SEED = 202610071601
HANDOFF_SHA = '90982b0613b5dc549e20dc7aed23392d16129f8a45929352ebe6b5d724684d08'

def now(): return datetime.now(timezone.utc).isoformat()
def rel(path): return Path(path).resolve().relative_to(ROOT).as_posix()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text('utf-8'))
def save(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')
def binding(path): return dict(path=rel(path),sha256=sha(path),bytes=Path(path).stat().st_size)

def persist(stem,value):
    """Preserve arbitrary numeric result trees without pickle or nonfinite JSON."""
    arrays={}
    def convert(x):
        if isinstance(x,np.ndarray):
            key=f'array_{len(arrays):06d}'; arrays[key]=x.copy()
            assert not x.dtype.hasobject
            return {'__ndarray__':key,'dtype':str(x.dtype),'shape':list(x.shape)}
        if isinstance(x,np.generic): return convert(x.item())
        if isinstance(x,float) and not math.isfinite(x): return {'__nonfinite_float__':repr(x)}
        if isinstance(x,dict): return {str(k):convert(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)): return [convert(v) for v in x]
        return x
    stem=Path(stem); stem.parent.mkdir(parents=True,exist_ok=True)
    tree=convert(value); raw=stem.with_suffix('.npz')
    with raw.open('xb') as f: np.savez_compressed(f,**arrays)
    save(stem.with_suffix('.json'),dict(tree=tree,raw=binding(raw),created_utc=now()))

def restore(stem):
    r=read(Path(stem).with_suffix('.json')); raw=ROOT/r['raw']['path']
    assert sha(raw)==r['raw']['sha256']
    with np.load(raw,allow_pickle=False) as archive:
        def convert(x):
            if isinstance(x,dict) and '__ndarray__' in x:
                a=archive[x['__ndarray__']].copy(); assert list(a.shape)==x['shape'] and str(a.dtype)==x['dtype']; return a
            if isinstance(x,dict) and '__nonfinite_float__' in x: return float(x['__nonfinite_float__'])
            if isinstance(x,dict): return {k:convert(v) for k,v in x.items()}
            if isinstance(x,list): return [convert(v) for v in x]
            return x
        return convert(r['tree'])

def units():
    return [dict(id=f'D{d+1}_N{n:04d}_r{rep}',teacher=d,N=n,rep=rep)
            for d in range(3) for n in (1024,4096) for rep in range(2)]
def folder(unit): return OUT/'units'/unit['id']
def rng(unit,role):
    return np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([SEED,0,unit['teacher'],unit['N'],unit['rep'],role])))
def environment():
    return dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
                threads={k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')})
def source_files():
    return [Path(__file__),ROOT/'script/prospective_diagnostic_core.py',ROOT/'script/prospective_diagnostic_core_test.py',
            ROOT/'script/rankone_stationary_kernel.py',ROOT/'script/rankone_stationary_compute.py',ROOT/'script/joint_dependence_kernel.py',
            DOC/'CONTRACT.md',PLAN/'DESIGN.md',PLAN/'DECISION_RULES.json']

def freeze():
    import prospective_diagnostic_core as core
    assert not (OUT/'FREEZE.json').exists()
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/research'],cwd=ROOT,text=True).split()[0]
    assert head==remote,'The first requested authoring/plan push must complete before research freezing'
    v161='artifacts/submission_candidate_v1_6_1_20261007/DELIVERY_MANIFEST.json'
    subprocess.check_call(['git','cat-file','-e',head+':'+v161],cwd=ROOT)
    assert sha(ROOT/'HANDOFF.md')==HANDOFF_SHA
    assert (OUT/'CORE_MANUFACTURED_CHECK.json').is_file(), 'Complete manufactured implementation checks first'
    plan=read(PLAN/'DECISION_RULES.json')
    config=dict(schema='prospective_dev_config1',stage='independent_development_not_formal',seed=SEED,
        core_default_configuration=core.configuration(),legacy_qualification_gates=dict(core.stationary.GATES),
        teachers=plan['candidate_development_teachers'],units=units(),configuration_number=1,
        maximum_independent_units=12,maximum_development_configurations=2,formal_units_generated=0,
        training=dict(steps=600,banks=4,low_K=8,high_K=64,kick=.15,learning_rates=[.01,.002],switch_step=400),
        risk_orders=[40,60,80],rotated_order=72,rotation=math.pi/7,main_order=60,
        mass_tolerance=2e-12,PMF_tolerance=2e-8,paired_risk_tolerance=2e-6,
        real_teacher_parameters_are_only_used_by_generator_and_post_lock_evaluation=True,
        global_locks=['INPUTS_LOCK','DIAGNOSIS_LOCK','ENDPOINTS_LOCK','EVALUATION_LOCK'])
    save(OUT/'CONFIG.json',config)
    (OUT/'HANDOFF_INPUT.md').write_bytes((ROOT/'HANDOFF.md').read_bytes())
    save(OUT/'FREEZE.json',dict(created_utc=now(),baseline_commit=head,authoring_plan_push_verified=True,
         feedback_sha256=HANDOFF_SHA,environment=environment(),config=binding(OUT/'CONFIG.json'),
         sources={rel(p):binding(p) for p in source_files()},
         implementation_check=binding(OUT/'CORE_MANUFACTURED_CHECK.json'),
         implementation_sources_are_bound_working_tree_files_not_claimed_in_baseline_commit=True,
         new_scientific_execution_started=False))
    print(json.dumps(dict(stage='frozen',units=12,baseline=head)),flush=True)

def verify():
    f=read(OUT/'FREEZE.json')
    for row in [f['config'],f['implementation_check'],*f['sources'].values()]: assert sha(ROOT/row['path'])==row['sha256'],row['path']
    return read(OUT/'CONFIG.json')
def lock(name,stems):
    paths=[Path(s).with_suffix(ext) for s in stems for ext in ('.json','.npz')]
    save(OUT/(name+'.json'),dict(created_utc=now(),freeze_sha256=sha(OUT/'FREEZE.json'),files={rel(p):binding(p) for p in paths}))
def require_lock(name):
    r=read(OUT/(name+'.json')); assert r['freeze_sha256']==sha(OUT/'FREEZE.json')
    for row in r['files'].values(): assert sha(ROOT/row['path'])==row['sha256'],row['path']

def generate():
    config=verify(); assert not (OUT/'INPUTS_LOCK.json').exists()
    if not (OUT/'EXECUTION_STARTED.json').exists(): save(OUT/'EXECUTION_STARTED.json',dict(created_utc=now(),stage='development_configuration_1',formal_execution=False))
    stems=[]
    for u in units():
        stem=folder(u)/'inputs'; stems.append(stem)
        if stem.with_suffix('.json').exists(): restore(stem); continue
        t=time.perf_counter(); teacher=config['teachers'][u['teacher']]
        b=np.array(teacher['b']); a=np.array(teacher['a']); phi=np.array(teacher['phi']); W=a[:,None]*np.c_[np.cos(phi),np.sin(phi)]
        z=rng(u,0).normal(size=(u['N'],2)); uniforms=rng(u,1).uniform(size=(u['N'],8))
        y=(uniforms<expit(b+z@W.T)).astype(np.int8)
        indices=y@2**np.arange(7,-1,-1); counts=np.bincount(indices,minlength=256); q=counts/u['N']
        data_seconds=time.perf_counter()-t
        tick=time.perf_counter(); xi=rng(u,2).normal(size=8); kick_generation_seconds=time.perf_counter()-tick
        tick=time.perf_counter(); prior=rng(u,3).normal(size=(600,4,64,2)); prior_generation_seconds=time.perf_counter()-tick
        persist(stem,dict(unit=u,latent=z,uniforms=uniforms,labels=y,counts=counts,q=q,xi=xi,prior=prior,
             seconds=time.perf_counter()-t,data_generation_seconds=data_seconds,kick_generation_seconds=kick_generation_seconds,
             prior_generation_seconds=prior_generation_seconds,Gaussian_scalars=2*u['N']+8+prior.size,
             uniform_scalars=8*u['N'],prior_vectors_generated=600*4*64))
        print(json.dumps(dict(stage='generated',unit=u['id'])),flush=True)
    lock('INPUTS_LOCK',stems)

def diagnose():
    import prospective_diagnostic_core as core
    verify(); require_lock('INPUTS_LOCK'); assert not (OUT/'DIAGNOSIS_LOCK.json').exists(); stems=[]
    for u in units():
        data=restore(folder(u)/'inputs'); anchor_path=folder(u)/'anchor'; diag_path=folder(u)/'diagnosis'; start_path=folder(u)/'start'
        if not anchor_path.with_suffix('.json').exists(): persist(anchor_path,core.fit_anchor(data['q'],u['N']))
        anchor=restore(anchor_path)
        if not diag_path.with_suffix('.json').exists(): persist(diag_path,core.diagnose(anchor,data['q']))
        if not start_path.with_suffix('.json').exists(): persist(start_path,core.neutral_kick(anchor,data['xi']))
        stems.extend([anchor_path,diag_path,start_path]); diag=restore(diag_path)
        print(json.dumps(dict(stage='diagnosed',unit=u['id'],level=diag['full_level'],A=diag['full_raise'],B=diag['ref_raise'])),flush=True)
    lock('DIAGNOSIS_LOCK',stems)

def followup():
    import prospective_diagnostic_core as core
    verify(); require_lock('INPUTS_LOCK'); require_lock('DIAGNOSIS_LOCK'); assert not (OUT/'ENDPOINTS_LOCK.json').exists(); stems=[]
    for u in units():
        data=restore(folder(u)/'inputs'); start=restore(folder(u)/'start')
        for arm in (['LOW','HIGH'] if u['rep']==0 else ['HIGH','LOW'])+['REF']:
            stem=folder(u)/arm; stems.append(stem)
            if not stem.with_suffix('.json').exists():
                result=core.fit_reference(data['q'],start) if arm=='REF' else core.train_arm(data['q'],start,data['prior'],8 if arm=='LOW' else 64)
                persist(stem,result)
            result=restore(stem)
            print(json.dumps(dict(stage='trained',unit=u['id'],arm=arm,status=result['status'])),flush=True)
    lock('ENDPOINTS_LOCK',stems)

def evaluate():
    c=verify(); require_lock('INPUTS_LOCK'); require_lock('DIAGNOSIS_LOCK'); require_lock('ENDPOINTS_LOCK')
    assert not (OUT/'EVALUATION_LOCK.json').exists(); stems=[]
    rules=[(n,0.) for n in c['risk_orders']]+[(c['rotated_order'],c['rotation'])]
    teacher_P={}
    for i,t in enumerate(c['teachers']):
        stem=OUT/f'teacher_D{i+1}'; stems.append(stem)
        if not stem.with_suffix('.json').exists():
            before=time.perf_counter(); a=np.array(t['a']); phi=np.array(t['phi']); W=a[:,None]*np.c_[np.cos(phi),np.sin(phi)]
            pmfs=np.array([joint.model_pmf(t['b'],W,n,angle) for n,angle in rules]); persist(stem,dict(pmfs=pmfs,rules=rules,seconds=time.perf_counter()-before))
        teacher_P[i]=restore(stem)['pmfs']
    for u in units():
        stem=folder(u)/'risk'; stems.append(stem)
        if stem.with_suffix('.json').exists(): restore(stem); continue
        before=time.perf_counter(); data=restore(folder(u)/'inputs'); target=teacher_P[u['teacher']]; pmfs={}; kl={}; nll={}; qualified={}; errors={}
        for arm in ('LOW','HIGH','REF'):
            fit=restore(folder(u)/arm); P=np.array([joint.model_pmf(fit['b'],fit['W'],n,angle) for n,angle in rules]); pmfs[arm]=P
            assert np.all(P>0) and np.all(target>0),'Nonpositive numerical PMF: preserve stage exception'
            kl[arm]=np.sum(target*(np.log(target)-np.log(P)),axis=1)
            nll[arm]=-np.log(P)@data['q']
            mass=max(float(np.max(abs(P.sum(axis=1)-1))),float(np.max(abs(target.sum(axis=1)-1))))
            diff=max(float(np.max(abs(P-P[1]))),float(np.max(abs(target-target[1]))))
            errors[arm]=dict(mass_error=mass,pmf_max_difference=diff)
            qualified[arm]=bool(mass<=c['mass_tolerance'] and diff<=c['PMF_tolerance'])
        paired=np.sum(target*(np.log(pmfs['HIGH'])-np.log(pmfs['LOW'])),axis=1)
        discrepancy=float(np.max(abs(paired-paired[1]))); identity=float(np.max(abs(paired-(kl['LOW']-kl['HIGH']))))
        ok=qualified['LOW'] and qualified['HIGH'] and discrepancy<=c['paired_risk_tolerance']
        persist(stem,dict(pmfs=pmfs,KL=kl,empirical_NLL=nll,paired_gain=paired,main_index=1,rules=rules,
             qualified_by_arm=qualified,errors=errors,paired_numerical_discrepancy=discrepancy,paired_identity_error=identity,
             LOW_HIGH_risk_qualified=ok,seconds=time.perf_counter()-before,
             PMF_calls=12,PMF_label_node_evaluations=3*256*sum(n*n for n,_ in rules)))
        print(json.dumps(dict(stage='evaluated',unit=u['id'],qualified=ok,gain=float(paired[1]))),flush=True)
    lock('EVALUATION_LOCK',stems)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['freeze','generate','diagnose','followup','evaluate']); args=parser.parse_args()
    tick=time.perf_counter(); started=now(); status='completed'
    try: globals()[args.stage]()
    except BaseException as error:
        status='failed'
        OUT.mkdir(parents=True,exist_ok=True)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        save(OUT/f'EXECUTION_EXCEPTION_{args.stage}_{stamp}.json',dict(created_utc=now(),stage=args.stage,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc()))
        raise
    finally:
        OUT.mkdir(parents=True,exist_ok=True)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        save(OUT/f'STAGE_TIMING_{args.stage}_{stamp}.json',dict(started_utc=started,finished_utc=now(),stage=args.stage,status=status,
             wall_seconds=time.perf_counter()-tick,includes_stage_import_verification_and_archival_IO=True,
             excludes_initial_driver_process_import=True,not_counterfactual_deployment_cost=True))

if __name__=='__main__': main()
