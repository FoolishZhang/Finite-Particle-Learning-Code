"""Deterministic continuation after NumPy-bool JSON failure; never retrains a model."""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_key]='1'
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse,json,hashlib,time,traceback
import numpy as np
import uselect_compute as core

ROOT=core.ROOT;OUT=core.OUT
DOC=ROOT/'docs/08_research_研究工作/USE_v2_20261005/EXECUTION_REPAIR.md'

def prefix(path):
    text=Path(path).read_text(encoding='utf-8');decoder=json.JSONDecoder();result={}
    for key in ('case','rep','trained_sha256','validation_frequency','score_matrix','methods','candidate_names','choices'):
        token=json.dumps(key)+':';start=text.index(token)+len(token)
        result[key]=decoder.raw_decode(text[start:].lstrip())[0]
    return result

def freeze():
    core.validate();sources={core.rel(Path(__file__)):core.sha(__file__),core.rel(DOC):core.sha(DOC)}
    inputs={core.rel(OUT/p):core.sha(OUT/p) for p in ('FREEZE.json','TEACHERS.json','ERROR.json')};moves=[]
    for case in core.CASES:
        for rep in range(32):
            folder=OUT/case/f'rep_{rep:02}';row=core.read(folder/'TRAINED.json')
            assert not (folder/'RESULT.json').exists()
            error=core.read(folder/'ERROR.json');assert 'not JSON serializable' in error['traceback']
            partial=folder/'SELECTION.json';archive=folder/'SELECTION.partial_001.json'
            assert not archive.exists();p=prefix(partial)
            assert p['trained_sha256']==core.sha(folder/'TRAINED.json')
            assert (p['case'],p['rep'])==(case,rep)
            for path in (folder/'TRAINED.json',folder/'ERROR.json',ROOT/row['data_file'],ROOT/row['trace_file']):inputs[core.rel(path)]=core.sha(path)
            moves.append(dict(original=core.rel(partial),archive=core.rel(archive),sha256=core.sha(partial)))
    # The correction is a scalar representation change, not coercion of truth values.
    assert json.loads(json.dumps({'qualified':bool(np.bool_(False))}))['qualified'] is False
    assert json.loads(json.dumps({'qualified':bool(np.bool_(True))}))['qualified'] is True
    core.save(OUT/'REPAIR_FREEZE.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),sources=sources,inputs=inputs,
        partial_moves=moves,science_rules_unchanged=True,training_retries=0,streams=96,original_error='NumPy bool JSON serialization'))
    print(json.dumps(dict(repair_frozen=True,trained_streams=96,sha256=core.sha(OUT/'REPAIR_FREEZE.json'))))

def validate(archives=True):
    core.validate();f=core.read(OUT/'REPAIR_FREEZE.json')
    for group in ('sources','inputs'):
        for path,digest in f[group].items():assert core.sha(ROOT/path)==digest,path
    if archives:
        for move in f['partial_moves']:assert core.sha(ROOT/move['archive'])==move['sha256']
    return f

def one(ci,rep):
    tick=time.perf_counter();folder=OUT/core.CASES[ci]/f'rep_{rep:02}'
    try:
        trained=core.read(folder/'TRAINED.json');original=prefix(folder/'SELECTION.partial_001.json')
        with np.load(ROOT/trained['data_file'],allow_pickle=False) as data:
            validation=data['validation'].astype(np.int64);test=data['test'].astype(np.int64)
        freq=core.frequency(validation);references=[];candidates=trained['candidates']
        for c in candidates:
            if c['name']=='IND':
                p=np.array(c['p']);q=np.prod(np.where(core.Y,p,1-p),axis=1)
                references.append(dict(name='IND',main_pmf=q.tolist(),variants=[],qualified=True))
            else:
                variants=[core.pmf(c['theta'],n) for n in (40,60,80)];q=np.array(variants[1]['pmf'])
                errors=[float(np.max(abs(np.array(v['pmf'])-q))) for v in variants]
                qualified=bool(max(errors)<=2e-7 and max(v['mass_error'] for v in variants)<=2e-12 and min(q)>0)
                references.append(dict(name=c['name'],main_pmf=q.tolist(),variants=variants,pmf_errors=errors,qualified=qualified))
        rng=np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([core.CONFIG['seed'],ci,rep,8])))
        prior=rng.standard_normal((1600,2));score=np.empty((5,4));patterns=np.empty((4,4,256))
        for j,c in enumerate(candidates):
            q=np.array(references[j]['main_pmf'])
            per_pattern=np.broadcast_to(-np.log(q),(4,256)).copy() if j==0 else core.validation_scores(c['theta'],prior)
            patterns[:,j]=per_pattern;score[:4,j]=per_pattern@freq;score[4,j]=-freq@np.log(q)
        choices=[core.select(row) for row in score]
        score_error=float(np.max(abs(score-np.array(original['score_matrix']))))
        assert score_error<=2e-12 and choices==original['choices']
        assert freq.tolist()==original['validation_frequency']
        selection=dict(case=core.CASES[ci],rep=rep,trained_sha256=core.sha(folder/'TRAINED.json'),validation_frequency=freq.tolist(),
            score_matrix=score.tolist(),methods=list(core.METHODS),candidate_names=list(core.CANDIDATES),choices=choices,
            references=references,validation_prior_sha256=core.ah(prior),locked_before_test_and_teacher_risk=True,
            repair_freeze_sha256=core.sha(OUT/'REPAIR_FREEZE.json'),original_partial_sha256=core.sha(folder/'SELECTION.partial_001.json'),
            original_choice_match=True,original_score_max_difference=score_error)
        core.save(folder/'SELECTION.json',selection)
        teacher=core.read(OUT/'TEACHERS.json')['records'][ci];perm=np.array(trained['permutation'])
        mapping=(core.Y[:,np.argsort(perm)]@core.POWERS).astype(int);P=np.array(teacher['main_pmf'])[mapping]
        test_freq=core.frequency(test);risks=[];test_losses=[];errors=[]
        for reference in references:
            q=np.array(reference['main_pmf']);R=float(P@np.log(P/q));risks.append(R)
            test_losses.append(float(-test_freq@np.log(q)))
            errors.append(max([abs(float(P@np.log(P/np.array(v['pmf'])))-R) for v in reference['variants']],default=0.))
        selected=np.array(risks)[choices];oracle=int(np.argmin(risks));path=folder/'EVALUATION.npz';assert not path.exists()
        np.savez_compressed(path,pattern_scores=patterns,prior=prior)
        result=dict(case=core.CASES[ci],rep=rep,folder=core.rel(folder),trained_sha256=core.sha(folder/'TRAINED.json'),
            selection_sha256=core.sha(folder/'SELECTION.json'),evaluation_file=core.rel(path),evaluation_sha256=core.sha(path),
            candidate_risks=risks,candidate_test_NLL=test_losses,oracle_candidate=oracle,choices=choices,
            selected_risks=selected.tolist(),selection_regret=(selected-min(risks)).tolist(),
            selected_test_NLL=np.array(test_losses)[choices].tolist(),test_frequency=test_freq.tolist(),teacher_P=P.tolist(),
            risk_order_changes=errors,numerically_qualified=bool(all(r['qualified'] for r in references) and max(errors)<=2e-6),
            fits_converged=sum(f['success'] for f in trained['fits']),fit_calls=sum(f['saved_calls'] for f in trained['fits']),
            seconds=time.perf_counter()-tick,seconds_scope='continuation evaluation only; training already completed once',
            original_score_max_difference=score_error,original_choice_match=True)
        core.save(folder/'RESULT.json',result);return result
    except Exception:
        core.save(folder/'REPAIR_ERROR.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise

def run():
    f=validate(False);start=time.perf_counter()
    core.save(OUT/'REPAIR_EXECUTION.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),repair_freeze_sha256=core.sha(OUT/'REPAIR_FREEZE.json'),training_retries=0))
    for move in f['partial_moves']:
        original=(ROOT/move['original']).resolve();archive=(ROOT/move['archive']).resolve()
        assert original.is_relative_to(OUT.resolve()) and archive.is_relative_to(OUT.resolve())
        assert core.sha(original)==move['sha256'] and not archive.exists()
        original.rename(archive);assert core.sha(archive)==move['sha256']
    validate();records=[]
    try:
        with ProcessPoolExecutor(max_workers=4) as pool:
            for future in as_completed([pool.submit(one,ci,rep) for ci in range(3) for rep in range(32)]):
                row=future.result();records.append(row)
                print(json.dumps(dict(case=row['case'],rep=row['rep'],qualified=row['numerically_qualified'],original_choices_match=True)),flush=True)
        records.sort(key=lambda r:(core.CASES.index(r['case']),r['rep']));validate()
        result=core.summary(records);result['seconds']=time.perf_counter()-start
        result['seconds_scope']='Continuation only; original failed stage has individual training timings but no normal completion timer'
        result['execution_repair']=dict(failed_serializations=96,training_retries=0,data_regenerations=0,validation_prior_replay_vectors=153600,
            duplicated_validation_Gaussian=307200,duplicated_reference_nodes=288*11600,
            all_original_choices_matched=True,max_original_score_difference=max(r['original_score_max_difference'] for r in records),
            repair_freeze_sha256=core.sha(OUT/'REPAIR_FREEZE.json'),source_sha256=core.sha(__file__))
        core.save(OUT/'RESULTS.json',result)
        print(json.dumps(dict(completed=True,qualified=result['numerically_qualified'],seconds=result['seconds'])),flush=True)
    except Exception:
        core.save(OUT/'REPAIR_ERROR.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('freeze','run','check'))
    command=parser.parse_args().command
    if command=='check':validate();print(json.dumps({'repair_bindings_passed':True}))
    else:globals()[command]()
