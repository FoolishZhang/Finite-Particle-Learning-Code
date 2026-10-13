"""Score-budget-matched inner-gradient versus outer-parameter averaging.

Five private FIXED models share four independent K100 inputs. IN4 averages
four original gradients before one update. M0..M3 each update once from their
own input; OUT4 is constructed from their physical TAIL parameters by the runner.
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

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'artifacts/scbm_budget_allocation';OUT=PACKAGE/'KERNEL';SOURCE=Path(__file__).resolve()
PREVIOUS_SHA='66d7e9d537172e10e1da6ce8d1317b782ef4b6df2410d9635d07089c45a57e6c'
PATHS=['IN4','M0','M1','M2','M3'];STRATEGIES=['IN4','OUT4']
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
    return dict(source_sha256=sha(SOURCE),averaging_helper_path=rel(previous.__file__),averaging_helper_sha256=PREVIOUS_SHA,
        reused=['bank_backward','ordered_average','apply_average','numeric_control'],original=previous.provenance(),
        path_order=PATHS,strategy_order=STRATEGIES,
        scope='Four original K100 score/backward calls per strategy per step. IN4 has1 optimizer/repair, OUT4 comprises4 private member optimizers/repairs. Same expected finite-K target; no pooling or prediction mixture.')


def environment():return previous.environment()


def make_models(K=100,n=256,lr=.02):
    assert sha(previous.__file__)==PREVIOUS_SHA
    models={name:old.ScoreProgram('FIXED',K=K,n=n,scale0=1.) for name in PATHS}
    return models,{name:old.make_optimizer(model,lr_rho=lr) for name,model in models.items()}


def step_quartet(models,optimizers,bank_inputs,y,yn,do_numerical=False):
    state={name:models[name].rho.detach().clone() for name in PATHS}
    losses={name:[] for name in PATHS};grads={name:[] for name in PATHS}
    details={name:[] for name in PATHS};stable={name:[] for name in PATHS}
    inputs={name:[] for name in PATHS}
    timing={strategy:dict(native_forward_backward_seconds=0.,optimizer_and_repair_seconds=0.,
                         original_API_calls=0,optimizer_steps=0,repair_calls=0) for strategy in STRATEGIES}
    shared_prepare=0.;numeric_seconds=0.;count=0;start=time.perf_counter()
    for bank_index,(active,noise) in enumerate(bank_inputs):
        assert bank_index<4
        tick=time.perf_counter();batch=old.prepare_batch(active,noise,y,yn);shared_prepare+=time.perf_counter()-tick
        for name in ['IN4',f'M{bank_index}']:
            strategy='IN4' if name=='IN4' else 'OUT4'
            assert torch.equal(models[name].rho.detach(),state[name]),'Path updated before gradients were collected'
            tick=time.perf_counter();loss,gradient,detail=previous.bank_backward(models[name],batch)
            timing[strategy]['native_forward_backward_seconds']+=time.perf_counter()-tick
            timing[strategy]['original_API_calls']+=1
            losses[name].append(loss);grads[name].append(gradient);details[name].append(detail);inputs[name].append(bank_index)
            if do_numerical:
                tick=time.perf_counter();stable[name].append(old.stable64(models[name],active,noise,y,yn));numeric_seconds+=time.perf_counter()-tick
        count+=1
    assert count==4 and [len(losses[name]) for name in PATHS]==[4,1,1,1,1]
    controls={}
    if do_numerical:
        tick=time.perf_counter()
        for name in PATHS:
            control=previous.numeric_control(losses[name],grads[name],stable[name])
            control['input_bank_indices']=inputs[name]
            for item,index in zip(control['per_bank_controls'],inputs[name]):item['input_bank_index']=index
            controls[name]=control
        numeric_seconds+=time.perf_counter()-tick
    results={}
    for name in PATHS:
        strategy='IN4' if name=='IN4' else 'OUT4';tick=time.perf_counter()
        results[name]=previous.apply_average(models[name],optimizers[name],losses[name],grads[name],details[name])
        timing[strategy]['optimizer_and_repair_seconds']+=time.perf_counter()-tick
        timing[strategy]['optimizer_steps']+=1;timing[strategy]['repair_calls']+=1
        results[name]['input_bank_indices']=inputs[name]
    assert [timing[s]['original_API_calls'] for s in STRATEGIES]==[4,4]
    assert [timing[s]['optimizer_steps'] for s in STRATEGIES]==[1,4]
    return results,controls,dict(strategies=timing,shared_float32_preparation_seconds=shared_prepare,
        numerical_control_seconds=numeric_seconds,shared_step_elapsed_seconds=time.perf_counter()-start,
        scope='Strategy timers measure calls inside shared execution, excluding RNG/hash/IO, shared float32 preparation and numerical controls. They are not independently executed end-to-end runtimes or a speedup claim.')


def deterministic_inputs(K=100,B=7,n=5):return previous.deterministic_inputs(K=K,B=B,n=n)


def selftest():
    assert not (PACKAGE/'FREEZE.json').exists()
    prov=provenance();banks,y,yn=deterministic_inputs();models,opts=make_models(n=5)
    pair,pair_opts=previous.make_pair(n=5)
    prior_result,_=previous.step_pair(pair,pair_opts,iter(banks),y,yn,False)
    references={}
    for j in range(1,4):
        program=old.ScoreProgram('FIXED',n=5);optimizer=old.make_optimizer(program)
        result=old.optimizer_step(program,optimizer,old.prepare_batch(*banks[j],y,yn),True)
        references[f'M{j}']=(program,optimizer,result)
    result,controls,timing=step_quartet(models,opts,iter(banks),y,yn,True)
    comparisons=[('IN4',pair['L4'],pair_opts['L4'],prior_result['L4']),('M0',pair['L1'],pair_opts['L1'],prior_result['L1'])]
    comparisons += [(name,*items) for name,items in references.items()]
    for name,program,optimizer,record in comparisons:
        assert torch.equal(models[name].rho.detach(),program.rho.detach())
        assert result[name]['loss']==record['loss'] and result[name]['rho_gradient']==record['rho_gradient']
        for key in ('exp_avg','exp_avg_sq','step'):
            assert torch.equal(opts[name].state[models[name].rho][key],optimizer.state[program.rho][key])
    for j in range(4):
        assert result['IN4']['bank_gradient'][j]==result[f'M{j}']['bank_gradient'][0]
    assert all(c['passed'] for c in controls.values())
    assert all(b['loss_passed'] and b['gradient_passed'] for c in controls.values() for b in c['per_bank_controls'])
    big,by,byn=deterministic_inputs(B=64,n=256);bm,bo=make_models()
    _,bc,bt=step_quartet(bm,bo,iter(big),by,byn,True)
    assert all(c['passed'] for c in bc.values())
    assert all(b['loss_passed'] and b['gradient_passed'] for c in bc.values() for b in c['per_bank_controls'])
    save(OUT/'SELFTEST.json',dict(passed=True,created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(SOURCE),
        provenance=prov,environment=environment(),all_five_paths_match_frozen_reference_updates_bitwise=True,
        initial_shared_bank_gradients_match_bitwise=True,one_optimizer_step_per_private_path=True,
        production_shape_controls=bc,production_shape_timing=bt,
        deterministic_original_API_calls=24,deterministic_stable64_calls=16,
        new_scientific_RNG=0,new_observations=0,new_model_integrals=0,new_training_fits=0,
        scope='Two isolated deterministic steps at separate fixtures. IN4 matches the old L4 kernel; each member matches the original L1 update, including Adam moments. No convergence pilot.'))
    print(json.dumps(dict(passed=True,source_sha256=sha(SOURCE),deterministic_original_API_calls=24)))


def _initializer(barrier):
    global _barrier
    _barrier=barrier


def deterministic_worker(index):
    banks,y,yn=deterministic_inputs(B=64,n=256);_barrier.wait(timeout=90);durations=[]
    for _ in range(2):
        models,opts=make_models();start=time.perf_counter()
        result,_,timing=step_quartet(models,opts,iter(banks),y,yn,False);durations.append(time.perf_counter()-start)
    digest=hashlib.sha256(np.asarray([result[n]['rho_after'] for n in PATHS],dtype=np.float32).tobytes()).hexdigest()
    return dict(index=index,pid=os.getpid(),durations=durations,parameter_sha256=digest,environment=environment(),
        final_step_timing=timing,deterministic_original_API_calls=16,input_bytes=sum(a.nbytes+b.nbytes for a,b in banks))


def benchmark():
    assert not (PACKAGE/'FREEZE.json').exists()
    banks,y,yn=deterministic_inputs(B=64,n=256);serial=[]
    for _ in range(3):
        models,opts=make_models();start=time.perf_counter();step_quartet(models,opts,iter(banks),y,yn,False);serial.append(time.perf_counter()-start)
    context=mp.get_context('spawn');start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=8,mp_context=context,initializer=_initializer,initargs=(context.Barrier(8),)) as pool:
        futures=[pool.submit(deterministic_worker,j) for j in range(8)];workers=[f.result() for f in futures]
    wall=time.perf_counter()-start
    assert len({w['pid'] for w in workers})==8 and len({w['parameter_sha256'] for w in workers})==1
    assert all(w['environment']['torch_threads']==1 for w in workers)
    slow=max(float(np.median(w['durations'])) for w in workers);steps=4*16*512
    save(OUT/'BENCHMARK.json',dict(passed=True,source_sha256=sha(SOURCE),provenance=provenance(),
        serial_seconds_per_quartet_step=serial,eight_worker_results=workers,startup_and_test_wall_seconds=wall,
        full_experiment_quartet_steps=steps,serial_compute_projection_seconds=steps*float(np.median(serial)),
        measured_contention_eight_worker_compute_projection_seconds=steps/8*slow,
        deterministic_original_API_calls=152,new_scientific_RNG=0,new_model_integrals=0,new_training_fits=0,
        scope='Three serial and two isolated identity steps per each of8 synchronized workers. No random input or convergence pilot. Projection excludes full-run RNG/hash/IO/NUM and long-run contention.'))
    print(json.dumps(dict(passed=True,serial_step_median=float(np.median(serial)),eight_worker_step_median_max=slow,
        projected_seconds=steps/8*slow,distinct_worker_pids=[w['pid'] for w in workers])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['check','bench']);a=p.parse_args()
    {'check':selftest,'bench':benchmark}[a.mode]()
