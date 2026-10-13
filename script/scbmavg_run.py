"""Paired L1/L4 FIXED training on four frozen datasets and sixteen new priors."""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import argparse,hashlib,json,time,traceback
import numpy as np
import torch
import scbmavg_kernel as kernel

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'artifacts/scbm_bank_averaging';OUT=PACKAGE/'TRAIN';SOURCE=Path(__file__).resolve()
ARMS=kernel.ARMS


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def read(p):return json.loads(Path(p).read_text('utf8'))


def save(p,obj,exclusive=True):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x' if exclusive else 'w',encoding='utf8',newline='\n') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def load_contract():
    config=read(PACKAGE/'CONFIG.json');data=read(PACKAGE/'DATA.json')
    required=dict(data_ids=[0,1,2,3],repeats=16,arms=ARMS,outer_banks=[1,4],K=100,nuisance=256,
        observations=4096,batch_size=64,steps=512,tail_postupdates=64,initial_rho=[0.]*6,scale=1.,
        lr_rho=.02,betas=[.9,.999],eps=1e-8,weight_decay=0.,eigen_floor=.1,repair_target=.1001,
        shuffle=False,gradient_clip=False,learning_rate_schedule=False,early_stop=False,workers=8,
        torch_threads=1,seed_base=202610034000,roles={'active':0,'noise':1},
        numerical_preupdate_steps=[0,31,127,511],numerical_loss_tolerance=1e-4,numerical_gradient_tolerance=2e-5)
    assert all(config[k]==v for k,v in required.items()),'Unexpected frozen training contract'
    assert [d['data_id'] for d in data['data']]==config['data_ids']
    return config,data


def load_labels(data_row):
    p=ROOT/data_row['path'];assert sha(p)==data_row['sha256']
    with np.load(p,allow_pickle=False) as z:
        y=z['active_labels'].copy();yn=z['nuisance_labels'].copy()
    assert y.shape==(4096,4) and yn.shape==(4096,256)
    assert ah(y)==data_row['active_labels_sha256'] and ah(yn)==data_row['nuisance_labels_sha256']
    assert np.isin(y,[0,1]).all() and np.isin(yn,[0,1]).all()
    counts=np.bincount(y@np.array([8,4,2,1]),minlength=16)
    assert counts.tolist()==data_row['counts']
    return y,yn


def allocate(steps=512):
    shapes=dict(rho=(2,steps+1,6),aggregate_loss=(2,steps),aggregate_gradient=(2,steps,6),
        rho_unprojected=(2,steps,6),maximum_absolute_logit=(2,steps))
    for arm,L in zip(ARMS,[1,4]):
        shapes['bank_loss_'+arm]=(steps,L);shapes['bank_gradient_'+arm]=(steps,L,6)
        shapes['bank_maximum_absolute_logit_'+arm]=(steps,L)
    tape={key:np.full(shape,np.nan,dtype=np.float32) for key,shape in shapes.items()}
    for key,shape in [('projection_alpha',(2,steps)),('proposal_eigmin',(2,steps)),('eigmin',(2,steps+1))]:
        tape[key]=np.full(shape,np.nan,dtype=np.float64)
    for key in ('repaired','finite'):tape[key]=np.zeros((2,steps),dtype=bool)
    for key in ('feasible','cholesky_success'):tape[key]=np.zeros((2,steps+1),dtype=bool)
    for key in ('sigmoid_zero_count','sigmoid_one_count','probability_evaluations'):
        tape[key]=np.zeros((2,steps),dtype=np.int64)
        for arm,L in zip(ARMS,[1,4]):tape['bank_'+key+'_'+arm]=np.zeros((steps,L),dtype=np.int64)
    tape['batch_start']=np.full(steps,-1,dtype=np.int64)
    return tape


def snapshot(tape,models,index):
    for ai,arm in enumerate(ARMS):
        model=models[arm];rho=model.rho.detach().numpy();cov=kernel.old.covariance_numpy(rho)
        eig=float(np.linalg.eigvalsh(cov).min())
        with torch.no_grad():_,info=torch.linalg.cholesky_ex(model.covariance())
        assert eig>=.1 and int(info)==0 and np.isfinite(rho).all()
        assert torch.equal(model.fixed_scales,torch.ones_like(model.fixed_scales))
        tape['rho'][ai,index]=rho;tape['eigmin'][ai,index]=eig
        tape['feasible'][ai,index]=True;tape['cholesky_success'][ai,index]=True


def advance(tape,models,optimizers,step,banks,y,yn,begin,data_id,repeat,numerical):
    result,controls=kernel.step_pair(models,optimizers,banks,y,yn,numerical)
    tape['batch_start'][step]=begin
    rows=[]
    for ai,arm in enumerate(ARMS):
        row=result[arm]
        tape['aggregate_loss'][ai,step]=row['loss'];tape['aggregate_gradient'][ai,step]=row['rho_gradient']
        for key in ('rho_unprojected','projection_alpha','proposal_eigmin'):tape[key][ai,step]=row[key]
        tape['repaired'][ai,step]=row['projected'];tape['finite'][ai,step]=True
        tape['bank_loss_'+arm][step]=row['bank_loss'];tape['bank_gradient_'+arm][step]=row['bank_gradient']
        for key in ('maximum_absolute_logit','sigmoid_zero_count','sigmoid_one_count','probability_evaluations'):
            tape[key][ai,step]=row['diagnostics'][key]
            tape['bank_'+key+'_'+arm][step]=[d[key] for d in row['bank_diagnostics']]
        if numerical:rows.append(dict(data_id=data_id,repeat=repeat,preupdate_step=step,arm=arm,**controls[arm]))
    snapshot(tape,models,step+1)
    return rows


def final_moments(tape,models,optimizers):
    for key,letter in [('exp_avg','m'),('exp_avg_sq','v')]:
        tape['final_rho_adam_'+letter]=np.stack([optimizers[a].state[models[a].rho][key].detach().numpy() for a in ARMS])
    tape['final_adam_step']=np.array([int(optimizers[a].state[models[a].rho]['step']) for a in ARMS],dtype=np.int64)


def outputs(tape,tail=64):
    result={}
    for ai,arm in enumerate(ARMS):
        result[arm]={}
        for kind in ('TAIL','LAST'):
            rho=tape['rho'][ai,-tail:].astype(float).mean(axis=0) if kind=='TAIL' else tape['rho'][ai,-1].astype(float)
            covariance=kernel.old.covariance_numpy(rho);eig=float(np.linalg.eigvalsh(covariance).min());assert eig>=.1
            result[arm][kind]=dict(rho=rho.tolist(),covariance=covariance.tolist(),scales=[1.]*256,eigmin=eig)
    return result


def numerical_summary(rows):
    bank=[b for r in rows for b in r['per_bank_controls']]
    return dict(aggregate_checks=len(rows),aggregate_passed=sum(r['passed'] for r in rows),
        aggregate_loss_passed=sum(r['loss_passed'] for r in rows),aggregate_gradient_passed=sum(r['gradient_passed'] for r in rows),
        bank_checks=len(bank),bank_loss_passed=sum(b['loss_passed'] for b in bank),
        bank_gradient_passed=sum(b['gradient_passed'] for b in bank),stable64_calls=sum(r['stable64_calls'] for r in rows),
        maximum_aggregate_loss_difference=max(abs(r['loss_difference']) for r in rows),
        maximum_aggregate_gradient_difference=max(r['maximum_gradient_difference'] for r in rows),
        maximum_bank_loss_difference=max(abs(b['loss_difference']) for b in bank),
        maximum_bank_gradient_difference=max(b['maximum_gradient_difference'] for b in bank),
        passed=bool(all(r['passed'] for r in rows) and all(b['loss_passed'] and b['gradient_passed'] for b in bank)))


def schema():
    config,data=load_contract();sample=allocate(steps=2)
    return dict(source_sha256=sha(SOURCE),kernel_sha256=sha(kernel.__file__),config_sha256=sha(PACKAGE/'CONFIG.json'),
        data_sha256=sha(PACKAGE/'DATA.json'),provenance=kernel.provenance(),environment=kernel.environment(),config=config,
        array_schema={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in sample.items()},
        shape_note='Schema sample has2 steps; scientific trace has512 steps and513 rho states. Final moments shape[2,6], final_adam_step[2].',
        input_arrays={'first_prior_active':'float64[4,100,64,4]','first_prior_noise':'float64[4,100,64,256]'},
        output_contract='records sorted(data_id,repeat); outputs[arm L1/L4][TAIL/LAST]={rho[6],covariance[4,4],scales[256] all1,eigmin}',
        RNG_order='8 independent PCG64 streams per pair. Keys bank_0_active,bank_0_noise,...,bank_3_noise; SeedSequence([base,data_id,repeat,bank_index,role]). For each step, bank0..3: active[K,B,4] then noise[K,B,256], float64 C-order role hashes before float32 conversion. Bank0 shared across arms.',
        labels='Original complete active and nuisance labels are loaded unchanged from DATA paths. No canonicalization or new observation generation.',
        no_self_authorized_freeze=True)


def validate_freeze():
    assert not (PACKAGE/'VERSION_LOCK.json').exists()
    frozen=read(PACKAGE/'FREEZE.json');assert frozen['run_authorized'] is True
    for path,digest in frozen['inputs'].items():assert sha(ROOT/path)==digest,path
    for p in (SOURCE,Path(kernel.__file__),PACKAGE/'CONFIG.json',PACKAGE/'DATA.json'):
        assert frozen['inputs'][rel(p)]==sha(p)
    for path,source in [(kernel.OUT/'SELFTEST.json',Path(kernel.__file__)),
                        (kernel.OUT/'BENCHMARK.json',Path(kernel.__file__)),(OUT/'SELFTEST.json',SOURCE)]:
        test=read(path);assert test['passed'] and test['source_sha256']==sha(source)
    kernel.provenance();load_contract()
    return frozen


def run_pair(data_id,repeat):
    validate_freeze();config,data=load_contract()
    assert data_id in config['data_ids'] and 0<=repeat<config['repeats']
    name=f'd{data_id:02}_r{repeat:02}'
    inp=OUT/f'input_{name}.npz';out=OUT/f'trace_{name}.npz';record_path=OUT/f'pair_{name}.json'
    assert not any(p.exists() for p in (inp,out,record_path,OUT/f'ERROR_{name}.json'))
    source_data=next(d for d in data['data'] if d['data_id']==data_id);labels,noise_labels=load_labels(source_data)
    entropy={f'bank_{bank}_{role}':[config['seed_base'],data_id,repeat,bank,role_id]
             for bank in range(4) for role,role_id in config['roles'].items()}
    rng={key:np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed))) for key,seed in entropy.items()}
    initial={k:g.bit_generator.state for k,g in rng.items()};digests={k:hashlib.sha256() for k in rng}
    models,optimizers=kernel.make_pair();tape=allocate();snapshot(tape,models,0)
    numerical=[];completed=0;step=-1;consumed=0;first=[];start=time.perf_counter()
    try:
        for step in range(config['steps']):
            begin=(config['batch_size']*step)%config['observations'];end=begin+config['batch_size']
            def banks():
                nonlocal consumed
                for bank in range(4):
                    active=rng[f'bank_{bank}_active'].standard_normal((100,64,4),dtype=np.float64)
                    noise=rng[f'bank_{bank}_noise'].standard_normal((100,64,256),dtype=np.float64)
                    digests[f'bank_{bank}_active'].update(active.tobytes(order='C'))
                    digests[f'bank_{bank}_noise'].update(noise.tobytes(order='C'))
                    consumed+=64
                    if step==0:first.append((active.copy(),noise.copy()))
                    yield active,noise
            numerical.extend(advance(tape,models,optimizers,step,banks(),labels[begin:end],noise_labels[begin:end],
                                     begin,data_id,repeat,step in config['numerical_preupdate_steps']))
            completed=step+1
            if step==0:
                with inp.open('xb') as handle:
                    np.savez_compressed(handle,first_prior_active=np.stack([p[0] for p in first]),
                                        first_prior_noise=np.stack([p[1] for p in first]))
                first.clear()
        final_moments(tape,models,optimizers)
        assert np.array_equal(tape['final_adam_step'],[512,512])
        assert all(np.isfinite(v).all() for v in tape.values())
        with out.open('xb') as handle:np.savez_compressed(handle,**tape)
        record=dict(id=name,data_id=data_id,repeat=repeat,source_sha256=sha(SOURCE),kernel_sha256=sha(kernel.__file__),
            freeze_sha256=sha(PACKAGE/'FREEZE.json'),input_file=rel(inp),input_sha256=sha(inp),
            output_file=rel(out),output_sha256=sha(out),old_data_file=source_data['path'],old_data_sha256=source_data['sha256'],
            label_sha256=dict(active=ah(labels),noise=ah(noise_labels)),role_seed_sequences=entropy,
            RNG_initial_states=initial,RNG_final_states={k:g.bit_generator.state for k,g in rng.items()},
            role_input_sha256={k:h.hexdigest() for k,h in digests.items()},outputs=outputs(tape),
            completed_steps=completed,independent_observation_banks_consumed=consumed,
            numerical_controls=numerical,numerical_qualification=numerical_summary(numerical),
            diagnostics={arm:dict(feasibility_repair_count=int(tape['repaired'][ai].sum()),minimum_eigenvalue=float(tape['eigmin'][ai].min()),
                minimum_proposal_eigenvalue=float(tape['proposal_eigmin'][ai].min()),minimum_repair_alpha=float(tape['projection_alpha'][ai].min()),
                maximum_absolute_logit=float(tape['maximum_absolute_logit'][ai].max()),
                sigmoid_zero_count=int(tape['sigmoid_zero_count'][ai].sum()),sigmoid_one_count=int(tape['sigmoid_one_count'][ai].sum()),
                probability_evaluations=int(tape['probability_evaluations'][ai].sum()),all_finite=True,all_feasible=True,all_Cholesky_success=True)
                for ai,arm in enumerate(ARMS)},seconds=time.perf_counter()-start,
            scope='New prior pair conditional on one fixed old complete dataset. L4 is an average of four original K100 losses/gradients; all bank gradients are unscaled originals, one Adam step per arm.')
        validate_freeze();save(record_path,record);return record
    except BaseException:
        partial=OUT/f'partial_{name}.npz'
        if not partial.exists():
            with partial.open('xb') as handle:np.savez_compressed(handle,**tape)
        save(OUT/f'ERROR_{name}.json',dict(data_id=data_id,repeat=repeat,current_step=step,completed_steps=completed,
            consumed_observation_banks=consumed,traceback=traceback.format_exc(),RNG_initial_states=initial,
            RNG_current_states={k:g.bit_generator.state for k,g in rng.items()},role_hashes_so_far={k:h.hexdigest() for k,h in digests.items()},automatic_retry=False))
        raise


def run():
    validate_freeze();config,data=load_contract();OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'EXECUTION.json').exists() and not list(OUT.glob('pair_*.json')) and not list(OUT.glob('input_*.npz'))
    assert not list(OUT.glob('trace_*.npz')) and not list(OUT.glob('ERROR*.json'))
    save(OUT/'EXECUTION.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(SOURCE),
        kernel_sha256=sha(kernel.__file__),freeze_sha256=sha(PACKAGE/'FREEZE.json'),workers=8,environment=kernel.environment(),automatic_retry=False))
    # Scheduling order is fixed repeat-major; final serialization is data-major.
    tasks=[(d,r) for r in range(16) for d in config['data_ids']]
    records={};active={};next_task=0;start=time.perf_counter();pool=ProcessPoolExecutor(max_workers=8)
    try:
        while next_task<8:
            pair=tasks[next_task];active[pool.submit(run_pair,*pair)]=pair;next_task+=1
        while active:
            done,_=wait(active,return_when=FIRST_COMPLETED);finished=[f.result() for f in done]
            for f in done:del active[f]
            for row in finished:
                records[(row['data_id'],row['repeat'])]=row
                print(json.dumps(dict(completed_data_id=row['data_id'],repeat=row['repeat'],seconds=row['seconds'])),flush=True)
            save(OUT/'PROGRESS.json',dict(completed=[list(k) for k in sorted(records)],running=[list(k) for k in active.values()],seconds=time.perf_counter()-start),False)
            while len(active)<8 and next_task<len(tasks):
                pair=tasks[next_task];active[pool.submit(run_pair,*pair)]=pair;next_task+=1
        ordered=[records[(d,r)] for d in config['data_ids'] for r in range(16)]
        controls=[c for record in ordered for c in record['numerical_controls']]
        assert len(controls)==512 and sum(c['stable64_calls'] for c in controls)==1280
        result=dict(source_sha256=sha(SOURCE),kernel_sha256=sha(kernel.__file__),freeze_sha256=sha(PACKAGE/'FREEZE.json'),
            config_sha256=sha(PACKAGE/'CONFIG.json'),data_sha256=sha(PACKAGE/'DATA.json'),config=config,records=ordered,
            numerical_qualification=numerical_summary(controls),
            accounting=dict(fixed_old_datasets=4,old_unique_observations=16384,new_observations=0,
                prior_repeats_per_dataset=16,independent_prior_pairs=64,controlled_optimization_programs=128,
                TAIL_outputs=128,LAST_descriptive_outputs=128,independent_observation_banks=8388608,
                unique_prior_vectors=838860800,prior_Gaussian_scalars=218103808000,
                active_prior_Gaussian_scalars=3355443200,noise_prior_Gaussian_scalars=214748364800,
                paired_observation_presentations=2097152,arm_observation_presentations=4194304,
                original_training_API_calls=163840,aggregate_numeric_controls=512,stable64_calls=1280,
                new_neural_network_fits=0,new_model_integrals=0,scientific_retries=0),
            seconds=time.perf_counter()-start,
            scope='Conditional experiment on four fixed old datasets,16 independent prior-repeat columns. Same expected finite-K criterion and original FIXED parameterization; only outer prior-gradient averaging changes. No claim of a new training-data population or guaranteed endpoint improvement.')
        validate_freeze();save(OUT/'RESULTS.json',result)
        save(OUT/'STATE.json',dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),
            results_sha256=sha(OUT/'RESULTS.json'),source_sha256=sha(SOURCE),seconds=result['seconds']))
        print(json.dumps(dict(complete=True,seconds=result['seconds'],results_sha256=sha(OUT/'RESULTS.json'),
                             numerical_qualification=result['numerical_qualification'])),flush=True)
    except BaseException:
        for f in active:f.cancel()
        save(OUT/'ERROR.json',dict(traceback=traceback.format_exc(),completed_pairs=[list(k) for k in records],
            already_active_pairs=[list(k) for k in active.values()],automatic_retry=False))
        raise
    finally:pool.shutdown(wait=True,cancel_futures=True)


def selftest():
    assert not (PACKAGE/'FREEZE.json').exists()
    config,data=load_contract();models,optimizers=kernel.make_pair();tape=allocate(steps=2);snapshot(tape,models,0)
    banks,y,yn=kernel.deterministic_inputs(B=64,n=256);rows=[]
    for step in range(2):rows.extend(advance(tape,models,optimizers,step,iter(banks),y,yn,64*step,-1,-1,True))
    final_moments(tape,models,optimizers);result=outputs(tape,tail=2)
    assert all(np.isfinite(v).all() for v in tape.values())
    assert np.array_equal(tape['rho'][:,0],np.zeros((2,6))) and np.array_equal(tape['final_adam_step'],[2,2])
    for ai,arm in enumerate(ARMS):
        for step in range(2):
            values=torch.tensor(tape['bank_gradient_'+arm][step]);expected=kernel.ordered_average(list(values)).numpy()
            assert np.array_equal(expected,tape['aggregate_gradient'][ai,step])
        assert np.array_equal(result[arm]['TAIL']['rho'],tape['rho'][ai,1:].astype(float).mean(axis=0))
    summary=numerical_summary(rows);assert summary['passed'] and summary['aggregate_checks']==4 and summary['stable64_calls']==10
    save(OUT/'SELFTEST.json',dict(passed=True,source_sha256=sha(SOURCE),kernel_sha256=sha(kernel.__file__),
        config_sha256=sha(PACKAGE/'CONFIG.json'),data_sha256=sha(PACKAGE/'DATA.json'),
        array_schema={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in tape.items()},
        numerical_qualification=summary,per_bank_gradients_explicitly_saved_and_averaged=True,
        physical_tail_and_identity_checked=True,deterministic_original_API_calls=10,deterministic_stable64_calls=10,
        new_scientific_RNG=0,new_observations=0,new_model_integrals=0,new_training_fits=0,
        scope='Two deterministic integration steps and tape arithmetic/schema checks, not a sampled trajectory or convergence pilot.'))
    save(OUT/'SCHEMA.json',schema())
    print(json.dumps(dict(passed=True,source_sha256=sha(SOURCE),kernel_sha256=sha(kernel.__file__),numerical=summary)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['schema','check','run']);args=parser.parse_args()
    if args.mode=='schema':print(json.dumps(schema(),ensure_ascii=True,indent=2))
    else:{'check':selftest,'run':run}[args.mode]()
