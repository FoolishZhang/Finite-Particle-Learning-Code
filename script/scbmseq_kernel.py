"""Frozen original updates with four-bank sequential and FACT controls.

IN4 and M0..M3 retain the old ordered arithmetic. SEQ4 performs four
private Adam/repair updates at successive states; its macro means are
diagnostics, never an applied averaged gradient. No scientific RNG here.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor
import argparse,hashlib,json,multiprocessing as mp,time
import numpy as np
import torch
import scbmavg_kernel as previous
import scbmalloc_kernel as allocation

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'artifacts/scbm_sequential_generalization';OUT=PACKAGE/'KERNEL';SOURCE=Path(__file__).resolve()
PREVIOUS_SHA='66d7e9d537172e10e1da6ce8d1317b782ef4b6df2410d9635d07089c45a57e6c'
PATHS=['IN4','M0','M1','M2','M3','SEQ4','FACT']
MEMBERS=['M0','M1','M2','M3'];STRATEGIES=['IN4','OUT4','SEQ4','FACT']
old=previous.old
torch.set_num_threads(1)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8',newline='\n') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def provenance():
    assert sha(previous.__file__)==PREVIOUS_SHA
    return dict(source_sha256=sha(SOURCE),averaging_helper_path=rel(previous.__file__),
        averaging_helper_sha256=PREVIOUS_SHA,original=previous.provenance(),
        reused=['bank_backward','ordered_average','apply_average','numeric_control'],
        path_order=PATHS,strategy_order=STRATEGIES,member_order=MEMBERS,
        scope='IN4/OUT4/SEQ4 share four original K100 calls, with1/4/4 Adam/repair updates. SEQ4 order0..3 uses updated states and lr.02 eachmicrostep. FACT uses bank0 active-only and is a quality reference, not cost matched.')
def environment():return previous.environment()
def make_models(K=100,n=256,lr=.02):
    assert sha(previous.__file__)==PREVIOUS_SHA
    models={p:old.ScoreProgram('FACT' if p=='FACT' else 'FIXED',K=K,n=n,scale0=1.) for p in PATHS}
    return models,{p:old.make_optimizer(m,lr_rho=lr) for p,m in models.items()}

def step_quartet(models,optimizers,bank_inputs,y,yn,do_numerical=False):
    state={p:models[p].rho.detach().clone() for p in PATHS}
    losses={p:[] for p in PATHS};grads={p:[] for p in PATHS};details={p:[] for p in PATHS}
    stable={p:[] for p in PATHS};inputs={p:[] for p in PATHS};controls={};micro=[]
    timing={s:dict(native_forward_backward_seconds=0.,optimizer_and_repair_seconds=0.,
        original_API_calls=0,optimizer_steps=0,repair_calls=0) for s in STRATEGIES}
    shared_prepare=0.;numeric_seconds=0.;count=0;start=time.perf_counter()
    for bank_index,(active,noise) in enumerate(bank_inputs):
        assert bank_index<4
        tick=time.perf_counter();batch=old.prepare_batch(active,noise,y,yn);shared_prepare+=time.perf_counter()-tick
        for path in ['IN4',f'M{bank_index}','SEQ4']+(['FACT'] if bank_index==0 else []):
            strategy='OUT4' if path in MEMBERS else path
            if path!='SEQ4':assert torch.equal(models[path].rho.detach(),state[path])
            tick=time.perf_counter();loss,gradient,detail=previous.bank_backward(models[path],batch)
            timing[strategy]['native_forward_backward_seconds']+=time.perf_counter()-tick
            timing[strategy]['original_API_calls']+=1
            losses[path].append(loss);grads[path].append(gradient);details[path].append(detail);inputs[path].append(bank_index)
            if do_numerical:
                tick=time.perf_counter();stable[path].append(old.stable64(models[path],active,noise,y,yn));numeric_seconds+=time.perf_counter()-tick
            if path=='SEQ4':
                if do_numerical:
                    tick=time.perf_counter();control=previous.numeric_control([loss],[gradient],[stable[path][-1]])
                    control['input_bank_indices']=[bank_index];control['microstep']=bank_index
                    control['per_bank_controls'][0]['input_bank_index']=bank_index
                    controls[f'SEQ4_micro{bank_index}']=control;numeric_seconds+=time.perf_counter()-tick
                tick=time.perf_counter();row=previous.apply_average(models[path],optimizers[path],[loss],[gradient],[detail])
                timing[strategy]['optimizer_and_repair_seconds']+=time.perf_counter()-tick
                timing[strategy]['optimizer_steps']+=1;timing[strategy]['repair_calls']+=1
                adam=optimizers[path].state[models[path].rho]
                row.update(input_bank_index=bank_index,adam_m=adam['exp_avg'].detach().numpy().tolist(),
                    adam_v=adam['exp_avg_sq'].detach().numpy().tolist(),adam_step=int(adam['step']))
                micro.append(row)
        count+=1
    assert count==4 and [len(losses[p]) for p in PATHS]==[4,1,1,1,1,4,1]
    results={}
    for path in ['IN4',*MEMBERS,'FACT']:
        if do_numerical:
            tick=time.perf_counter();control=previous.numeric_control(losses[path],grads[path],stable[path])
            control['input_bank_indices']=inputs[path]
            for item,index in zip(control['per_bank_controls'],inputs[path]):item['input_bank_index']=index
            controls[path]=control;numeric_seconds+=time.perf_counter()-tick
        strategy='OUT4' if path in MEMBERS else path;tick=time.perf_counter()
        results[path]=previous.apply_average(models[path],optimizers[path],losses[path],grads[path],details[path])
        timing[strategy]['optimizer_and_repair_seconds']+=time.perf_counter()-tick
        timing[strategy]['optimizer_steps']+=1;timing[strategy]['repair_calls']+=1
        results[path]['input_bank_indices']=inputs[path]
    # Descriptive macro summaries from four different preupdate states.
    results['SEQ4']=dict(loss=float(previous.ordered_average(losses['SEQ4'])),
        rho_gradient=previous.ordered_average(grads['SEQ4']).numpy().tolist(),
        bank_loss=torch.stack(losses['SEQ4']).numpy().tolist(),bank_gradient=torch.stack(grads['SEQ4']).numpy().tolist(),
        bank_diagnostics=details['SEQ4'],input_bank_indices=[0,1,2,3],microsteps=micro,
        diagnostics=dict(official_API_calls=4,probability_evaluations=sum(d['probability_evaluations'] for d in details['SEQ4']),
            sigmoid_zero_count=sum(d['sigmoid_zero_count'] for d in details['SEQ4']),
            sigmoid_one_count=sum(d['sigmoid_one_count'] for d in details['SEQ4']),
            maximum_absolute_logit=max(d['maximum_absolute_logit'] for d in details['SEQ4'])),
        macro_summary_is_applied_update=False)
    assert [timing[s]['original_API_calls'] for s in STRATEGIES]==[4,4,4,1]
    assert [timing[s]['optimizer_steps'] for s in STRATEGIES]==[1,4,4,1]
    return results,controls,dict(strategies=timing,shared_float32_preparation_seconds=shared_prepare,
        numerical_control_seconds=numeric_seconds,shared_step_elapsed_seconds=time.perf_counter()-start,
        scope='Shared-execution call timers exclude RNG/hash/IO and shared preparation/controls; not separate end-to-end runtimes.')

def deterministic_inputs(K=100,B=7,n=5):return previous.deterministic_inputs(K=K,B=B,n=n)
def _compare(program,opt,reference,refopt):
    assert torch.equal(program.rho.detach(),reference.rho.detach())
    for key in ('exp_avg','exp_avg_sq','step'):assert torch.equal(opt.state[program.rho][key],refopt.state[reference.rho][key])
def selftest():
    assert not (PACKAGE/'FREEZE.json').exists()
    prov=provenance();banks,y,yn=deterministic_inputs();models,opts=make_models(n=5)
    prior,popts=allocation.make_models(n=5)
    seq=old.ScoreProgram('FIXED',n=5);seqopt=old.make_optimizer(seq)
    fact=old.ScoreProgram('FACT',n=5);factopt=old.make_optimizer(fact)
    controls=[];calls=0
    for macro in range(2):
        reference,_,_=allocation.step_quartet(prior,popts,iter(banks),y,yn,False);calls+=8
        seqrows=[old.optimizer_step(seq,seqopt,old.prepare_batch(*bank,y,yn),True) for bank in banks];calls+=4
        factrow=old.optimizer_step(fact,factopt,old.prepare_batch(*banks[0],y,yn),True);calls+=1
        result,numeric,_=step_quartet(models,opts,iter(banks),y,yn,True);calls+=13
        for path in allocation.PATHS:
            _compare(models[path],opts[path],prior[path],popts[path])
            assert result[path]['loss']==reference[path]['loss'] and result[path]['rho_gradient']==reference[path]['rho_gradient']
        _compare(models['SEQ4'],opts['SEQ4'],seq,seqopt);_compare(models['FACT'],opts['FACT'],fact,factopt)
        for bank,row in enumerate(seqrows):
            for key in ('rho_before','rho_after','rho_gradient','rho_unprojected','loss','projection_alpha'):
                assert result['SEQ4']['microsteps'][bank][key]==row[key]
            assert result['SEQ4']['microsteps'][bank]['adam_step']==macro*4+bank+1
        assert result['FACT']['loss']==factrow['loss']
        assert all(c['passed'] for c in numeric.values());controls.extend(numeric.values())
    big,by,byn=deterministic_inputs(B=64,n=256);bm,bo=make_models()
    _,bc,bt=step_quartet(bm,bo,iter(big),by,byn,True);calls+=13
    assert all(c['passed'] for c in bc.values());controls.extend(bc.values())
    assert len(controls)==30 and sum(c['stable64_calls'] for c in controls)==39
    save(OUT/'SELFTEST.json',dict(passed=True,created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(SOURCE),
        provenance=prov,environment=environment(),all_old_paths_match_frozen_reference_updates_bitwise=True,
        sequential_and_FACT_match_original_L1_updates_and_moments_bitwise=True,
        sequential_microstates_checked=True,production_shape_controls=bc,production_shape_timing=bt,
        deterministic_original_API_calls=calls,deterministic_stable64_calls=39,
        new_scientific_RNG=0,new_observations=0,new_model_integrals=0,new_training_fits=0,
        scope='Two deterministic tiny integration steps plus one isolated production-shape identity step; no sampled convergence pilot.'))
    print(json.dumps(dict(passed=True,source_sha256=sha(SOURCE),deterministic_original_API_calls=calls)))
def _initializer(barrier):
    global _barrier
    _barrier=barrier
def deterministic_worker(index):
    banks,y,yn=deterministic_inputs(B=64,n=256);_barrier.wait(timeout=90);durations=[]
    for _ in range(2):
        models,opts=make_models();tick=time.perf_counter();result,_,timing=step_quartet(models,opts,iter(banks),y,yn,False);durations.append(time.perf_counter()-tick)
    digest=hashlib.sha256(np.asarray([models[p].rho.detach().numpy() for p in PATHS],dtype=np.float32).tobytes()).hexdigest()
    return dict(index=index,pid=os.getpid(),durations=durations,parameter_sha256=digest,environment=environment(),
        final_step_timing=timing,deterministic_original_API_calls=26,input_bytes=sum(a.nbytes+b.nbytes for a,b in banks))
def benchmark():
    assert not (PACKAGE/'FREEZE.json').exists()
    banks,y,yn=deterministic_inputs(B=64,n=256);serial=[]
    for _ in range(3):
        models,opts=make_models();tick=time.perf_counter();step_quartet(models,opts,iter(banks),y,yn,False);serial.append(time.perf_counter()-tick)
    context=mp.get_context('spawn');tick=time.perf_counter()
    with ProcessPoolExecutor(max_workers=8,mp_context=context,initializer=_initializer,initargs=(context.Barrier(8),)) as pool:
        workers=[f.result() for f in [pool.submit(deterministic_worker,j) for j in range(8)]]
    wall=time.perf_counter()-tick
    assert len({w['pid'] for w in workers})==8 and len({w['parameter_sha256'] for w in workers})==1
    assert all(w['environment']['torch_threads']==1 for w in workers)
    slow=max(float(np.median(w['durations'])) for w in workers);steps=16*2*512
    save(OUT/'BENCHMARK.json',dict(passed=True,source_sha256=sha(SOURCE),provenance=provenance(),
        serial_seconds_per_quartet_step=serial,eight_worker_results=workers,startup_and_test_wall_seconds=wall,
        full_experiment_quartet_steps=steps,serial_compute_projection_seconds=steps*float(np.median(serial)),
        measured_contention_eight_worker_compute_projection_seconds=steps/8*slow,
        deterministic_original_API_calls=247,new_scientific_RNG=0,new_observations=0,new_model_integrals=0,new_training_fits=0,
        scope='Three serial and two isolated identity steps per8 synchronized workers. Projection excludes full-run RNG/hash/IO/NUM and long-run contention.'))
    print(json.dumps(dict(passed=True,serial_step_median=float(np.median(serial)),eight_worker_step_median_max=slow,projected_seconds=steps/8*slow)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['check','bench']);a=p.parse_args();{'check':selftest,'bench':benchmark}[a.mode]()
