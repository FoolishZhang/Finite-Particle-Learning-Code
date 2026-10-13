"""Frozen controlled parameter optimization; no automatic scientific retry.

schema/check are RNG-free. freeze and run are separate explicit commands;
neither is performed by self-tests. Risk integrals are a separate package phase.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import argparse
import hashlib
import json
import math
import platform
import sys
import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import expit
import scbmopt_kernel as kernel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/scbm_scale_optimization/TRAIN'
SOURCE = Path(__file__).resolve()
DESIGN = ROOT / 'docs/08_research_研究工作/可学习无关尺度与优化路径_20261003/EXPERIMENT_DESIGN.md'
REFERENCE_SOURCE = ROOT / 'script/scbmopt_reference.py'
BASE = '4b37603e981851f8db317690f700e69a1820b05f'
KERNEL_SHA = 'f292b26b2f65fa7adb5204453345b744fd30e01ba86a6aaaf1b8f66cc2e1840b'
METHODS = ['FIXED', 'FREE', 'FACT']
CONFIG = dict(streams=16, observations=4096, nuisance=256, K=100, batch_size=64,
    steps=512, tail_postupdates=64, methods=METHODS, t0=0., scale0=1.,
    lr_t=.02, lr_scale=.02, betas=[.9,.999], eps=1e-8, weight_decay=0.,
    t_bounds=[-.75,.75], floor=1e-6, shuffle=False, gradient_clip=False,
    learning_rate_schedule=False, early_stop=False, workers=2, torch_threads=1,
    seed_base=202610030200,
    roles=dict(data_gaussian=0, data_active_uniform=1, data_noise_uniform=2,
               prior_active_gaussian=3, prior_noise_gaussian=4),
    numerical_preupdate_steps=[0,31,127,511], numerical_loss_tolerance=1e-4,
    numerical_t_gradient_tolerance=2e-5, numerical_raw_gradient_tolerance=2e-6,
    displayed_postupdate_steps=[0,32,128,512], primary_output='TAIL',
    primary_family_size=3, primary_alpha=.05, primary_SE_thresholds=[.0002,.0002,.00005],
    noninferiority_margin=.0002)


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p): return Path(p).resolve().relative_to(ROOT).as_posix()
def read(p): return json.loads(Path(p).read_text('utf8'))
def ah(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def save(p, obj, exclusive=True):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x' if exclusive else 'w', encoding='utf8', newline='\n') as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write('\n')


def environment():
    try:
        from threadpoolctl import threadpool_info
        pools=threadpool_info()
        available=True
        assert all(p['num_threads']==1 for p in pools),pools
    except ImportError:
        pools=[]
        available=False
    assert torch.get_num_threads()==1
    return dict(python=platform.python_version(), executable=str(Path(sys.executable).resolve()),
                numpy=np.__version__, torch=torch.__version__, device='cpu',
                torch_threads=torch.get_num_threads(),
                threadpoolctl_available=available, actual_threadpools=pools,
                thread_environment={k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')})


def schema():
    p = kernel.provenance()
    assert sha(kernel.__file__) == KERNEL_SHA
    return dict(base_commit=BASE, source_sha256=sha(SOURCE), kernel_sha256=KERNEL_SHA,
        legacy_kernel_sha256=kernel.LEGACY_SHA, design_path=rel(DESIGN), design_sha256=sha(DESIGN),
        reference_source_path=rel(REFERENCE_SOURCE), reference_source_sha256=sha(REFERENCE_SOURCE),
        environment=environment(), config=CONFIG, fixture=kernel.legacy.FIXTURE,
        official_inputs={p['official']['official_losses']:p['official']['official_losses_sha256'],
                        p['official']['official_models']:p['official']['official_models_sha256'],
                        p['official']['default_config']:p['official']['default_config_sha256']},
        RNG_order='Independent PCG64(SeedSequence([base,stream,role])). Roles0/1/2 once: float64 Gaussian[N,4], Uniform[N,4], Uniform[N,256]. Then each step role3 Gaussian[K,64,4] and role4 Gaussian[K,64,256], C contiguous bytes hashed in step order; one conversion to float32 shared by all methods.',
        outputs_schema='records[].outputs[method][TAIL/LAST] = {t:float,scales:list[256]}; FACT scales0 is the analytical factorization convention, not a trained or floor-feasible physical scale.',
        trajectory_schema=dict(t='float32[3,513], method order FIXED/FREE/FACT; index0 initial',
            free_raw='float32[513,256]', free_tau='float32[513,256]',
            loss='float32[3,512], preupdate', t_gradient='float32[3,512], preupdate',
            free_raw_gradient='float32[512,256], preupdate',
            t_unprojected='float32[3,512]', projected='bool[3,512]', boundary='bool[3,513]',
            sigmoid_zero_count='int64[3,512]', sigmoid_one_count='int64[3,512]',
            probability_evaluations='int64[3,512]', maximum_absolute_logit='float32[3,512]',
            finite='bool[3,512]', feasible='bool[3,513]', batch_start='int64[512]',
            final_t_adam_m='float32[3]', final_t_adam_v='float32[3]',
            final_free_raw_adam_m='float32[256]', final_free_raw_adam_v='float32[256]'),
        numeric_control_scope='Original float64 eps and current float32 parameter values promoted to float64; includes input/parameter representation and expression changes, not a BCE-only decomposition.')


def freeze():
    assert not (OUT/'FREEZE.json').exists() and not (OUT/'EXECUTION.json').exists()
    assert not list(OUT.glob('stream_*.npz'))
    check = read(OUT/'SELFTEST.json')
    assert check['passed'] and check['source_sha256'] == sha(SOURCE)
    kcheck = read(kernel.OUT/'SELFTEST.json')
    assert kcheck['passed'] and kcheck['source_sha256'] == KERNEL_SHA
    value = schema()
    value.update(created_utc=datetime.now(timezone.utc).isoformat(), run_authorized=True,
                 selftest_sha256=sha(OUT/'SELFTEST.json'), kernel_selftest_sha256=sha(kernel.OUT/'SELFTEST.json'))
    save(OUT/'FREEZE.json', value)
    print(json.dumps(dict(freeze_sha256=sha(OUT/'FREEZE.json'), **value), ensure_ascii=False, indent=2))


def validate_freeze():
    f = read(OUT/'FREEZE.json')
    assert f['run_authorized'] is True
    current = schema()
    for key, value in current.items(): assert f[key] == value, key
    assert f['selftest_sha256'] == sha(OUT/'SELFTEST.json')
    assert f['kernel_selftest_sha256'] == sha(kernel.OUT/'SELFTEST.json')
    assert read(OUT/'SELFTEST.json')['passed'] and read(kernel.OUT/'SELFTEST.json')['passed']
    for name, digest in f['official_inputs'].items(): assert sha(ROOT/name) == digest, name
    return f


def stable64(program, active, noise, y, yn):
    """Independent high-precision logit expression, never called official API."""
    t = program.t.detach().to(torch.float64).requires_grad_()
    raw = program.raw_scale.detach().to(torch.float64).requires_grad_() if program.raw_scale is not None else None
    mu = torch.tensor(kernel.legacy.FIXTURE['mu'], dtype=torch.float64)
    eye = torch.eye(4,dtype=torch.float64)
    D = torch.tensor(kernel.legacy.FIXTURE['C0'], dtype=torch.float64)-eye
    L = torch.linalg.cholesky(eye+(1+t)*D)
    active_logits = torch.matmul(L,torch.as_tensor(active,dtype=torch.float64).unsqueeze(-1)).squeeze(-1)+mu
    if program.method == 'FACT':
        logits, labels = active_logits, torch.as_tensor(y,dtype=torch.float64)
    else:
        tau = F.softplus(raw)+CONFIG['floor'] if raw is not None else torch.ones(program.n,dtype=torch.float64)
        logits = torch.cat((active_logits,torch.as_tensor(noise,dtype=torch.float64)*tau),dim=-1)
        labels = torch.as_tensor(np.concatenate((y,yn),axis=1),dtype=torch.float64)
    by_concept = logits.movedim(0,-1)
    target = labels.unsqueeze(-1).expand_as(by_concept)
    loss = -torch.logsumexp(-F.binary_cross_entropy_with_logits(by_concept,target,reduction='none').sum(dim=1),dim=1).mean()+math.log(program.K)
    if program.method == 'FACT': loss = loss+program.n*math.log(2.)
    gradients = torch.autograd.grad(loss,(t,raw) if raw is not None else (t,))
    return dict(loss=float(loss.detach()), t_gradient=float(gradients[0]),
                raw_gradient=None if raw is None else gradients[1].detach().numpy().tolist())


def numeric_row(stream, step, method, result, program, stable):
    loss_error = result['loss']-stable['loss']
    t_error = result['t_gradient']-stable['t_gradient']
    row = dict(stream=stream, preupdate_step=step, method=method,
        official_loss=result['loss'], stable64_loss=stable['loss'], loss_difference=loss_error,
        official_t_gradient=result['t_gradient'], stable64_t_gradient=stable['t_gradient'],
        t_gradient_difference=t_error,
        loss_passed=abs(loss_error)<=CONFIG['numerical_loss_tolerance'],
        t_gradient_passed=abs(t_error)<=CONFIG['numerical_t_gradient_tolerance'])
    if method == 'FREE':
        gradient = program.raw_scale.grad.detach().numpy().astype(np.float64)
        difference = gradient-np.asarray(stable['raw_gradient'])
        row.update(official_raw_gradient=gradient.tolist(),stable64_raw_gradient=stable['raw_gradient'],
                   raw_gradient_difference=difference.tolist(), maximum_raw_gradient_difference=float(np.max(abs(difference))),
                   raw_gradient_passed=bool(np.max(abs(difference))<=CONFIG['numerical_raw_gradient_tolerance']))
    row['passed'] = bool(row['loss_passed'] and row['t_gradient_passed'] and row.get('raw_gradient_passed',True))
    return row


def allocate(steps=512,n=256):
    tape = {name:np.full(shape,np.nan,dtype=np.float32) for name,shape in dict(
        t=(3,steps+1),free_raw=(steps+1,n),free_tau=(steps+1,n),loss=(3,steps),
        t_gradient=(3,steps),free_raw_gradient=(steps,n),t_unprojected=(3,steps),
        maximum_absolute_logit=(3,steps)).items()}
    for name in ('projected','finite'):tape[name]=np.zeros((3,steps),dtype=bool)
    for name in ('boundary','feasible'):tape[name]=np.zeros((3,steps+1),dtype=bool)
    for name in ('sigmoid_zero_count','sigmoid_one_count','probability_evaluations'):
        tape[name]=np.zeros((3,steps),dtype=np.int64)
    tape['batch_start']=np.full(steps,-1,dtype=np.int64)
    return tape


def snapshot(tape, models, index):
    for mi,method in enumerate(METHODS):
        model=models[method];t=float(model.t.detach())
        tape['t'][mi,index]=t
        tape['boundary'][mi,index]=t in CONFIG['t_bounds']
        scales=model.scales().detach()
        finite=bool(torch.isfinite(scales).all()) and math.isfinite(t)
        feasible=finite and CONFIG['t_bounds'][0]<=t<=CONFIG['t_bounds'][1] and bool((scales>=kernel.FLOOR).all())
        tape['feasible'][mi,index]=feasible
        assert feasible,(method,index,'infeasible')
    tape['free_raw'][index]=models['FREE'].raw_scale.detach().numpy()
    tape['free_tau'][index]=models['FREE'].scales().detach().numpy()


def advance(tape, models, optimizers, step, arrays, begin, stream, do_numerical):
    active,noise,y,yn=arrays
    batch=kernel.prepare_batch(*arrays)
    tape['batch_start'][step]=begin
    rows=[]
    for mi,method in enumerate(METHODS):
        model=models[method]
        control=stable64(model,*arrays) if do_numerical else None
        result=kernel.optimizer_step(model,optimizers[method],batch,diagnostics=True)
        for name,value in [('loss',result['loss']),('t_gradient',result['t_gradient']),
                           ('t_unprojected',result['t_unprojected']),('projected',result['projected'])]:
            tape[name][mi,step]=value
        for name in ('sigmoid_zero_count','sigmoid_one_count','probability_evaluations','maximum_absolute_logit'):
            tape[name][mi,step]=result['diagnostics'][name]
        tape['finite'][mi,step]=True # optimizer_step has checked loss, all gradients and parameters.
        if method=='FREE':tape['free_raw_gradient'][step]=model.raw_scale.grad.detach().numpy()
        if do_numerical:rows.append(numeric_row(stream,step,method,result,model,control))
    snapshot(tape,models,step+1)
    return rows


def final_moments(tape,models,optimizers):
    tape['final_t_adam_m']=np.asarray([float(optimizers[m].state[models[m].t]['exp_avg']) for m in METHODS],dtype=np.float32)
    tape['final_t_adam_v']=np.asarray([float(optimizers[m].state[models[m].t]['exp_avg_sq']) for m in METHODS],dtype=np.float32)
    state=optimizers['FREE'].state[models['FREE'].raw_scale]
    tape['final_free_raw_adam_m']=state['exp_avg'].numpy().copy()
    tape['final_free_raw_adam_v']=state['exp_avg_sq'].numpy().copy()


def outputs(tape,tail=64):
    result={}
    for mi,method in enumerate(METHODS):
        result[method]={}
        for label in ('TAIL','LAST'):
            t=float(tape['t'][mi,-tail:].astype(np.float64).mean()) if label=='TAIL' else float(tape['t'][mi,-1])
            if method=='FREE':
                scales=tape['free_tau'][-tail:].astype(np.float64).mean(axis=0) if label=='TAIL' else tape['free_tau'][-1].astype(np.float64)
            else:scales=np.full(tape['free_tau'].shape[1],1. if method=='FIXED' else 0.)
            result[method][label]=dict(t=t,scales=scales.tolist())
    return result


def numerical_summary(rows):
    return dict(checks=len(rows),passed=sum(r['passed'] for r in rows),
        loss_passed=sum(r['loss_passed'] for r in rows),t_gradient_passed=sum(r['t_gradient_passed'] for r in rows),
        raw_gradient_checks=sum('raw_gradient_passed' in r for r in rows),
        raw_gradient_passed=sum(r.get('raw_gradient_passed',False) for r in rows),
        maximum_loss_difference=max(abs(r['loss_difference']) for r in rows),
        maximum_t_gradient_difference=max(abs(r['t_gradient_difference']) for r in rows),
        maximum_raw_gradient_difference=max(r.get('maximum_raw_gradient_difference',0) for r in rows))


def run_stream(stream):
    freeze=validate_freeze();tick=time.perf_counter();step=-1;complete_steps=0
    input_path=OUT/f'input_{stream:02}.npz';output_path=OUT/f'stream_{stream:02}.npz'
    record_path=OUT/f'stream_{stream:02}.json'
    assert not any(p.exists() for p in (input_path,output_path,record_path))
    rng={name:np.random.Generator(np.random.PCG64(np.random.SeedSequence([CONFIG['seed_base'],stream,role]))) for name,role in CONFIG['roles'].items()}
    initial_states={name:r.bit_generator.state for name,r in rng.items()}
    role_hashes={};digests={name:hashlib.sha256() for name in ('prior_active_gaussian','prior_noise_gaussian')}
    tape=allocate();models,optimizers=kernel.make_programs();snapshot(tape,models,0)
    numerical=[]
    try:
        z=rng['data_gaussian'].standard_normal((4096,4));u=rng['data_active_uniform'].random((4096,4));un=rng['data_noise_uniform'].random((4096,256))
        role_hashes.update(data_gaussian=ah(z),data_active_uniform=ah(u),data_noise_uniform=ah(un))
        logits=z@np.linalg.cholesky(np.asarray(kernel.legacy.FIXTURE['C0'])).T+np.asarray(kernel.legacy.FIXTURE['mu'])
        labels=(u<expit(logits)).astype(np.int8);noise_labels=(un<.5).astype(np.int8)
        data_inputs=dict(active_labels=labels,nuisance_labels=noise_labels,data_active_gaussian=z,data_active_uniform=u)
        label_hashes=dict(active_labels=ah(labels),nuisance_labels=ah(noise_labels))
        del logits,un
        first_hashes={}
        for step in range(CONFIG['steps']):
            begin=(64*step)%4096;end=begin+64
            active=rng['prior_active_gaussian'].standard_normal((100,64,4))
            noise=rng['prior_noise_gaussian'].standard_normal((100,64,256))
            for name,a in (('prior_active_gaussian',active),('prior_noise_gaussian',noise)):
                b=np.ascontiguousarray(a).tobytes();digests[name].update(b)
                if step==0:first_hashes[name]=hashlib.sha256(b).hexdigest()
            if step==0:
                np.savez_compressed(input_path,**data_inputs,first_prior_active=active,first_prior_nuisance=noise)
                del data_inputs,z,u
            numerical.extend(advance(tape,models,optimizers,step,(active,noise,labels[begin:end],noise_labels[begin:end]),
                                     begin,stream,step in CONFIG['numerical_preupdate_steps']))
            complete_steps=step+1
        final_moments(tape,models,optimizers)
        np.savez_compressed(output_path,**tape)
        role_hashes.update({name:h.hexdigest() for name,h in digests.items()})
        result=dict(stream=stream,source_sha256=sha(SOURCE),freeze_sha256=sha(OUT/'FREEZE.json'),
            input_file=rel(input_path),input_sha256=sha(input_path),output_file=rel(output_path),output_sha256=sha(output_path),
            role_seed_sequences={name:[CONFIG['seed_base'],stream,role] for name,role in CONFIG['roles'].items()},
            RNG_initial_states=initial_states,RNG_final_states={name:r.bit_generator.state for name,r in rng.items()},
            role_input_sha256=role_hashes,label_sha256=label_hashes,first_prior_sha256=first_hashes,
            outputs=outputs(tape),output_convention='TAIL = mean of postupdates 449..512 physical t and tau. FACT scales0 denotes analytical elimination, not learned parameters or a realizable softplus+floor scale.',
            completed_steps=complete_steps,numerical_controls=numerical,numerical_qualification=numerical_summary(numerical),
            diagnostics={method:dict(boundary_postupdate_count=int(tape['boundary'][mi,1:].sum()),
                projected_count=int(tape['projected'][mi].sum()),sigmoid_zero_count=int(tape['sigmoid_zero_count'][mi].sum()),
                sigmoid_one_count=int(tape['sigmoid_one_count'][mi].sum()),maximum_absolute_logit=float(tape['maximum_absolute_logit'][mi].max()),
                probability_evaluations=int(tape['probability_evaluations'][mi].sum()),all_finite=bool(tape['finite'][mi].all()),
                all_feasible=bool(tape['feasible'][mi].all())) for mi,method in enumerate(METHODS)},
            seconds=time.perf_counter()-tick)
        validate_freeze();save(record_path,result)
        return result
    except BaseException as error:
        partial=OUT/f'partial_stream_{stream:02}.npz'
        if not partial.exists():np.savez_compressed(partial,**tape)
        save(OUT/f'ERROR_stream_{stream:02}.json',dict(stream=stream,error_type=type(error).__name__,error=str(error),
            traceback=traceback.format_exc(),current_step=step,completed_steps=complete_steps,
            RNG_initial_states=initial_states,RNG_current_states={name:r.bit_generator.state for name,r in rng.items()},
            role_hashes_so_far={**role_hashes,**{name:h.hexdigest() for name,h in digests.items()}},
            source_sha256=sha(SOURCE),freeze_sha256=sha(OUT/'FREEZE.json'),automatic_retry=False))
        raise


def run():
    f=validate_freeze()
    assert not (OUT/'EXECUTION.json').exists() and not list(OUT.glob('input_*.npz')) and not list(OUT.glob('stream_*.npz')), 'No scientific retry'
    save(OUT/'EXECUTION.json',dict(start_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(SOURCE),
        freeze_sha256=sha(OUT/'FREEZE.json'),environment=environment(),workers=2,automatic_retry=False))
    start=time.perf_counter();records={};next_stream=0
    pool=ProcessPoolExecutor(max_workers=2)
    active={}
    try:
        while next_stream<2:
            active[pool.submit(run_stream,next_stream)]=next_stream;next_stream+=1
        while active:
            done,_=wait(active,return_when=FIRST_COMPLETED)
            # Inspect all completed futures before dispatching more work. A
            # failure never triggers replacement seeds or an automatic retry.
            new_records=[future.result() for future in done]
            for future in done:del active[future]
            for record in new_records:
                records[record['stream']]=record
                print(json.dumps(dict(completed_stream=record['stream'],seconds=record['seconds'])),flush=True)
            save(OUT/'PROGRESS.json',dict(completed_streams=sorted(records),count=len(records),
                running_streams=sorted(active.values()),seconds=time.perf_counter()-start),exclusive=False)
            while len(active)<2 and next_stream<16:
                active[pool.submit(run_stream,next_stream)]=next_stream;next_stream+=1
        ordered=[records[i] for i in range(16)]
        numerical=[row for r in ordered for row in r['numerical_controls']]
        result=dict(source_sha256=sha(SOURCE),kernel_sha256=KERNEL_SHA,freeze_sha256=sha(OUT/'FREEZE.json'),
            design_sha256=f['design_sha256'],config=CONFIG,fixture=f['fixture'],environment=f['environment'],
            records=ordered,numerical_qualification=numerical_summary(numerical),
            ledger=dict(independent_streams=16,new_complete_observations=65536,data_Gaussian_scalars=262144,
                data_Uniform_scalars=17039360,new_prior_vectors=52428800,prior_Gaussian_scalars=13631488000,
                prior_active_Gaussian_scalars=209715200,prior_noise_Gaussian_scalars=13421772800,
                observation_presentations=524288,controlled_optimization_programs=48,TAIL_outputs=48,LAST_outputs=48,
                official_training_API_calls=24576,stable64_control_calls=192,new_neural_network_fits=0,new_model_integrals=0),
            seconds=time.perf_counter()-start,
            scope='Frozen low-dimensional controlled optimization; shared inputs across methods, not independent method streams. No encoder/task head/precision penalty/full SCBM training. TAIL risk reference and main statistical family are computed separately.',
            numerical_scope=schema()['numeric_control_scope'])
        validate_freeze();save(OUT/'RESULTS.json',result)
        save(OUT/'STATE.json',dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),
            source_sha256=sha(SOURCE),results_sha256=sha(OUT/'RESULTS.json'),seconds=result['seconds']))
        print(json.dumps(dict(completed=True,seconds=result['seconds'],numerical_qualification=result['numerical_qualification'])),flush=True)
    except BaseException as error:
        # The other already active stream may finish and preserve its output;
        # no further stream is dispatched after this exception.
        for future in active:future.cancel()
        save(OUT/'ERROR.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
            completed_streams=sorted(records),active_streams=sorted(active.values()),automatic_retry=False))
        raise
    finally:
        pool.shutdown(wait=True,cancel_futures=True)


def check():
    # Two deterministic steps exercise changing states and the exact save-array
    # schema. These are not sampled observations and no convergence is tested.
    models,optimizers=kernel.make_programs();tape=allocate(steps=2);snapshot(tape,models,0)
    arrays=kernel.legacy.deterministic_inputs(K=100,B=64,n=256)
    rows=[]
    for step in range(2):rows.extend(advance(tape,models,optimizers,step,arrays,64*step,-1,True))
    final_moments(tape,models,optimizers)
    out=outputs(tape,tail=2)
    assert all(np.isfinite(a).all() for a in tape.values())
    assert all(r['passed'] for r in rows),numerical_summary(rows)
    assert all(out['FACT'][key]['scales']==[0.]*256 for key in ('TAIL','LAST'))
    assert all(out['FIXED'][key]['scales']==[1.]*256 for key in ('TAIL','LAST'))
    assert out['FREE']['TAIL']['t']==float(tape['t'][1,1:].astype(np.float64).mean())
    assert len(rows)==6 and len([r for r in rows if r['method']=='FREE'])==2
    frozen_schema=schema()
    result=dict(passed=True,source_sha256=sha(SOURCE),kernel_sha256=KERNEL_SHA,
        shapes={k:list(v.shape) for k,v in tape.items()},dtypes={k:str(v.dtype) for k,v in tape.items()},
        numerical_qualification=numerical_summary(rows),numerical_controls=rows,
        outputs_schema_checked=True,private_optimizers_checked=True,config=frozen_schema['config'],
        new_scientific_RNG_draws=0,new_model_integrals=0,new_scientific_fits=0,
        scope='Deterministic fixed-array two-step integration/schema check; not a sampled training or convergence pilot')
    save(OUT/'SELFTEST.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='numerical_controls'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['schema','check','freeze','run'])
    phase=parser.parse_args().phase
    if phase=='schema':print(json.dumps(schema(),ensure_ascii=False,indent=2))
    else:globals()[phase]()
