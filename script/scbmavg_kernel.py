"""Outer averaging of independent K100 gradients, with the old FIXED update.

No stochastic generator belongs to this module. Every bank retains its own
original concept loss and unscaled six-component gradient. Parameters change
only after all required backwards have completed.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor
import argparse,hashlib,json,math,multiprocessing as mp,platform,time
import numpy as np
import torch
import scbm6_kernel as old

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'artifacts/scbm_bank_averaging'
OUT=PACKAGE/'KERNEL';SOURCE=Path(__file__).resolve()
OLD_SHA='40ad5e8139229fe624a5d888c77768a81b6addecab018b18b824a2c634573d13'
ARMS=['L1','L4'];BANKS=[1,4]
torch.set_num_threads(1)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def read(p):return json.loads(Path(p).read_text('utf8'))


def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8',newline='\n') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def provenance():
    assert sha(old.__file__)==OLD_SHA
    return dict(source_sha256=sha(SOURCE),old_kernel_path=rel(old.__file__),old_kernel_sha256=OLD_SHA,
                reused=['ScoreProgram FIXED','prepare_batch','make_optimizer','restore_feasibility','stable64'],
                original=old.provenance(),scope='Independent outer-bank gradient average before one unchanged Adam/repair update. No pooled K*L particles, no FREE branch or target change.')


def environment():
    result=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,
                torch_threads=torch.get_num_threads(),thread_environment={k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')})
    try:
        from threadpoolctl import threadpool_info
        result['threadpool_info']=threadpool_info();result['actual_BLAS_threads_measured']=True
    except ImportError:
        result['threadpool_info']=None;result['actual_BLAS_threads_measured']=False
    try:
        import psutil
        info=psutil.Process().memory_info()
        result['resident_bytes']=info.rss;result['peak_working_set_bytes']=getattr(info,'peak_wset',None)
    except ImportError:
        result['resident_bytes']=None;result['peak_working_set_bytes']=None
    return result


def make_pair(K=100,n=256,lr=.02):
    assert sha(old.__file__)==OLD_SHA
    models={a:old.ScoreProgram('FIXED',K=K,n=n,scale0=1.) for a in ARMS}
    return models,{a:old.make_optimizer(m,lr_rho=lr) for a,m in models.items()}


def bank_backward(program,batch):
    program.zero_grad(set_to_none=True)
    detail={};loss=program(batch,detail)
    assert bool(torch.isfinite(loss));loss.backward()
    gradient=program.rho.grad.detach().clone()
    assert bool(torch.isfinite(gradient).all())
    return loss.detach().clone(),gradient,detail


def ordered_average(values):
    assert len(values) in (1,4)
    # Copy the first value, then add the remaining unscaled values in bank order.
    # For L1 this is bit-identical to the original single-bank gradient.
    total=values[0].clone()
    for value in values[1:]:total.add_(value)
    total.mul_(1./len(values))
    return total


def apply_average(program,optimizer,losses,gradients,details):
    loss=ordered_average(losses);gradient=ordered_average(gradients)
    before=program.rho.detach().numpy().copy()
    program.rho.grad=gradient.clone()
    optimizer.step()
    repair=program.restore_feasibility()
    assert bool(torch.isfinite(program.rho).all())
    return dict(loss=float(loss),rho_before=before.tolist(),rho_gradient=gradient.numpy().tolist(),
        rho_after=program.rho.detach().numpy().tolist(),bank_loss=torch.stack(losses).numpy().tolist(),
        bank_gradient=torch.stack(gradients).numpy().tolist(),**repair,
        diagnostics=dict(official_API_calls=sum(d['official_API_calls'] for d in details),
            probability_evaluations=sum(d['probability_evaluations'] for d in details),
            sigmoid_zero_count=sum(d['sigmoid_zero_count'] for d in details),
            sigmoid_one_count=sum(d['sigmoid_one_count'] for d in details),
            maximum_absolute_logit=max(d['maximum_absolute_logit'] for d in details)),bank_diagnostics=details)


def numeric_control(losses,gradients,controls,loss_tolerance=1e-4,gradient_tolerance=2e-5):
    rows=[]
    for index,(loss,gradient,stable) in enumerate(zip(losses,gradients,controls)):
        native_gradient=gradient.numpy().astype(float)
        delta=native_gradient-np.asarray(stable['rho_gradient'])
        loss_delta=float(loss)-stable['loss']
        rows.append(dict(bank_index=index,native_loss=float(loss),stable64_loss=stable['loss'],
            loss_difference=loss_delta,native_gradient=native_gradient.tolist(),stable64_gradient=stable['rho_gradient'],
            gradient_difference=delta.tolist(),maximum_gradient_difference=float(abs(delta).max()),
            loss_passed=abs(loss_delta)<=loss_tolerance,gradient_passed=bool(abs(delta).max()<=gradient_tolerance)))
    loss=float(ordered_average(losses));gradient=ordered_average(gradients).numpy().astype(float)
    stable_loss=float(np.mean([x['loss'] for x in controls]));stable_gradient=np.mean([x['rho_gradient'] for x in controls],axis=0)
    delta=gradient-stable_gradient;loss_delta=loss-stable_loss
    return dict(native_loss=loss,stable64_loss=stable_loss,loss_difference=loss_delta,
        native_gradient=gradient.tolist(),stable64_gradient=stable_gradient.tolist(),gradient_difference=delta.tolist(),
        maximum_gradient_difference=float(abs(delta).max()),loss_passed=abs(loss_delta)<=loss_tolerance,
        gradient_passed=bool(abs(delta).max()<=gradient_tolerance),
        passed=bool(abs(loss_delta)<=loss_tolerance and abs(delta).max()<=gradient_tolerance),
        per_bank_controls=rows,stable64_calls=len(controls),loss_tolerance=loss_tolerance,gradient_tolerance=gradient_tolerance)


def step_pair(models,optimizers,bank_inputs,y,yn,do_numerical=False):
    """Consume exactly four input banks, shared first bank, then update each arm once."""
    state={a:models[a].rho.detach().clone() for a in ARMS}
    losses={a:[] for a in ARMS};gradients={a:[] for a in ARMS}
    details={a:[] for a in ARMS};stable={a:[] for a in ARMS};count=0
    for bank_index,(active,noise) in enumerate(bank_inputs):
        assert bank_index<4
        batch=old.prepare_batch(active,noise,y,yn)
        for arm in (ARMS if bank_index==0 else ['L4']):
            program=models[arm]
            assert torch.equal(program.rho.detach(),state[arm]),'Parameters changed before outer averaging finished'
            loss,gradient,detail=bank_backward(program,batch)
            losses[arm].append(loss);gradients[arm].append(gradient);details[arm].append(detail)
            if do_numerical:stable[arm].append(old.stable64(program,active,noise,y,yn))
        count+=1
    assert count==4 and [len(losses[a]) for a in ARMS]==BANKS
    numerical={a:numeric_control(losses[a],gradients[a],stable[a]) for a in ARMS} if do_numerical else {}
    results={a:apply_average(models[a],optimizers[a],losses[a],gradients[a],details[a]) for a in ARMS}
    return results,numerical


def deterministic_inputs(K=100,B=7,n=5):
    banks=[]
    active_index=np.arange(K*B*4,dtype=np.float64).reshape(K,B,4)
    noise_index=np.arange(K*B*n,dtype=np.float64).reshape(K,B,n)
    for bank in range(4):
        banks.append((1.1*np.sin(active_index*.071+bank*.47),.95*np.cos(noise_index*.013+bank*.29)))
    y=((np.arange(B*4).reshape(B,4)*7+3)%11<5).astype(np.int8)
    yn=((np.arange(B*n).reshape(B,n)*3+1)%7<3).astype(np.int8)
    return banks,y,yn


def selftest():
    assert not (PACKAGE/'FREEZE.json').exists()
    prov=provenance();banks,y,yn=deterministic_inputs();models,optimizers=make_pair(n=5)
    legacy=old.ScoreProgram('FIXED',n=5);legacy_opt=old.make_optimizer(legacy)
    legacy_result=old.optimizer_step(legacy,legacy_opt,old.prepare_batch(*banks[0],y,yn),True)
    manual=old.ScoreProgram('FIXED',n=5);manual_opt=old.make_optimizer(manual)
    joint_losses=[manual(old.prepare_batch(*bank,y,yn)) for bank in banks]
    joint=torch.stack(joint_losses).mean();joint.backward();joint_gradient=manual.rho.grad.detach().clone()
    manual_opt.step();manual.restore_feasibility()
    result,controls=step_pair(models,optimizers,iter(banks),y,yn,True)
    assert torch.equal(models['L1'].rho.detach(),legacy.rho.detach())
    assert result['L1']['loss']==legacy_result['loss'] and result['L1']['rho_gradient']==legacy_result['rho_gradient']
    for key in ('exp_avg','exp_avg_sq','step'):
        assert torch.equal(optimizers['L1'].state[models['L1'].rho][key],legacy_opt.state[legacy.rho][key])
    error=float(np.max(abs(joint_gradient.numpy()-np.asarray(result['L4']['rho_gradient']))))
    update_error=float(np.max(abs(manual.rho.detach().numpy()-models['L4'].rho.detach().numpy())))
    assert error<2e-7 and update_error<3e-7
    assert all(row['passed'] for row in controls.values())
    assert all(int(optimizers[a].state[models[a].rho]['step'])==1 for a in ARMS)
    # A forced feasibility repair must retain the single aggregate update's moments.
    probe=old.ScoreProgram('FIXED');probe_opt=old.make_optimizer(probe)
    with torch.no_grad():probe.rho.fill_(.89998)
    synthetic=[-torch.ones(6) for _ in range(4)];probe.rho.grad=ordered_average(synthetic);probe_opt.step()
    moments={k:probe_opt.state[probe.rho][k].clone() for k in ('exp_avg','exp_avg_sq','step')}
    repair=probe.restore_feasibility();assert repair['projected']
    assert all(torch.equal(v,probe_opt.state[probe.rho][k]) for k,v in moments.items())
    big,by,byn=deterministic_inputs(B=64,n=256);bm,bo=make_pair()
    br,bc=step_pair(bm,bo,iter(big),by,byn,True)
    assert all(c['passed'] for c in bc.values())
    assert all(b['loss_passed'] and b['gradient_passed'] for c in bc.values() for b in c['per_bank_controls'])
    save(OUT/'SELFTEST.json',dict(passed=True,created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(SOURCE),provenance=prov,environment=environment(),
        L1_legacy_loss_gradient_parameters_and_moments_bitwise_equal=True,
        L4_joint_objective_gradient_max_error=error,L4_joint_objective_parameter_max_error=update_error,
        one_Adam_step_per_arm=True,forced_repair_moments_retained=True,production_shape_controls=bc,
        deterministic_official_API_calls=15,deterministic_stable64_calls=10,
        new_scientific_RNG=0,new_model_integrals=0,new_training_fits=0,
        scope='One small joint-objective comparison and one full-shaped isolated step. Joint four-loss autograd independently verifies outer averaging; no model integration or convergence pilot.'))
    print(json.dumps(dict(passed=True,source_sha256=sha(SOURCE),gradient_error=error,parameter_error=update_error)))


def _worker_initializer(barrier):
    global _barrier
    _barrier=barrier


def deterministic_worker(index):
    banks,y,yn=deterministic_inputs(B=64,n=256)
    _barrier.wait(timeout=90)
    start=time.perf_counter();durations=[]
    for _ in range(2):
        models,optimizers=make_pair();tick=time.perf_counter()
        result,_=step_pair(models,optimizers,iter(banks),y,yn,False)
        durations.append(time.perf_counter()-tick)
    checksum=hashlib.sha256(np.asarray([result[a]['rho_after'] for a in ARMS],dtype=np.float32).tobytes()).hexdigest()
    return dict(index=index,pid=os.getpid(),durations=durations,seconds=time.perf_counter()-start,
        parameter_sha256=checksum,environment=environment(),deterministic_official_API_calls=10,
        input_bytes=sum(a.nbytes+b.nbytes for a,b in banks))


def benchmark():
    assert not (PACKAGE/'FREEZE.json').exists()
    banks,y,yn=deterministic_inputs(B=64,n=256);serial=[]
    for _ in range(3):
        models,optimizers=make_pair();start=time.perf_counter()
        step_pair(models,optimizers,iter(banks),y,yn,False);serial.append(time.perf_counter()-start)
    context=mp.get_context('spawn');start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=8,mp_context=context,initializer=_worker_initializer,initargs=(context.Barrier(8),)) as pool:
        futures=[pool.submit(deterministic_worker,i) for i in range(8)];workers=[f.result() for f in futures]
    wall=time.perf_counter()-start
    assert len({w['pid'] for w in workers})==8
    assert len({w['parameter_sha256'] for w in workers})==1
    assert all(w['environment']['torch_threads']==1 for w in workers)
    median=float(np.median(serial));parallel=max(float(np.median(w['durations'])) for w in workers)
    steps=4*16*512
    save(OUT/'BENCHMARK.json',dict(passed=True,source_sha256=sha(SOURCE),provenance=provenance(),
        serial_seconds_per_pair_step=serial,eight_worker_results=workers,eight_worker_startup_and_test_wall_seconds=wall,
        full_experiment_pair_steps=steps,serial_compute_projection_seconds=steps*median,
        measured_contention_eight_worker_compute_projection_seconds=(steps/8)*parallel,
        deterministic_official_API_calls=95,new_scientific_RNG=0,new_model_integrals=0,new_training_fits=0,
        scope='Three serial and two isolated identity steps per each of8 synchronized workers, all deterministic inputs. Projection excludes RNG generation, byte hashing, NPZ compression, numeric controls and prolonged memory contention. No learning trajectory or outcomes used.'))
    print(json.dumps(dict(passed=True,serial_pair_step_median=median,eight_worker_step_median_max=parallel,
        projected_seconds=steps/8*parallel,distinct_worker_pids=[w['pid'] for w in workers])))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['check','bench']);args=parser.parse_args()
    {'check':selftest,'bench':benchmark}[args.mode]()
