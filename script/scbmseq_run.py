"""New-data quartet training; scientific RNG requires root and data freezes."""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_key]='1'
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import argparse,hashlib,json,time,traceback
import numpy as np
import torch
import scbmseq_kernel as kernel
ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'artifacts/scbm_sequential_generalization';OUT=PACKAGE/'TRAIN';SOURCE=Path(__file__).resolve()
PATHS=kernel.PATHS;STRATEGIES=kernel.STRATEGIES;MEMBERS=kernel.MEMBERS
BANK_COUNTS=[4,1,1,1,1,4,1]
DIAGNOSTICS=['maximum_absolute_logit','sigmoid_zero_count','sigmoid_one_count','probability_evaluations']
COUNTERS=['original_API_calls','optimizer_steps','repair_calls']
TIMERS=['native_forward_backward_seconds','optimizer_and_repair_seconds']
SHARED=['shared_float32_preparation_seconds','numerical_control_seconds','shared_step_elapsed_seconds','input_generation_and_hash_seconds']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def read(p):return json.loads(Path(p).read_text('utf8'))
def save(p,obj,exclusive=True):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x' if exclusive else 'w',encoding='utf8',newline='\n') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def load_contract(require_data=True):
    config=read(PACKAGE/'CONFIG.json')
    required=dict(data_ids=list(range(16)),repeats=2,arms=STRATEGIES,strategies=STRATEGIES,programs=PATHS,
        members=MEMBERS,outer_banks=4,K=100,nuisance=256,observations=4096,batch_size=64,steps=512,
        tail_postupdates=64,initial_rho=[0.]*6,scale=1.,lr_rho=.02,betas=[.9,.999],eps=1e-8,
        weight_decay=0.,eigen_floor=.1,repair_target=.1001,shuffle=False,gradient_clip=False,
        learning_rate_schedule=False,early_stop=False,workers=8,torch_threads=1,
        seed_base=202610040100,roles={'active':0,'noise':1},
        numerical_preupdate_steps=[0,31,127,511],numerical_loss_tolerance=1e-4,numerical_gradient_tolerance=2e-5)
    assert all(config.get(k)==v for k,v in required.items()),'Unexpected frozen training contract'
    assert config['data_generation']['seed_base']==202610040000
    assert config.get('seq_bank_order',[0,1,2,3])==[0,1,2,3]
    assert config.get('seq_lr_rho',.02)==.02
    data=read(PACKAGE/'DATA.json') if require_data else None
    if data is not None:assert [d['data_id'] for d in data['data']]==config['data_ids']
    return config,data
def load_labels(row):
    path=ROOT/row['path'];assert sha(path)==row['sha256']
    with np.load(path,allow_pickle=False) as z:y=z['active_labels'].copy();yn=z['nuisance_labels'].copy()
    assert y.shape==(4096,4) and yn.shape==(4096,256)
    assert np.isin(y,[0,1]).all() and np.isin(yn,[0,1]).all()
    assert np.bincount(y@np.array([8,4,2,1]),minlength=16).tolist()==row['counts']
    if 'active_labels_sha256' in row:assert ah(y)==row['active_labels_sha256']
    if 'nuisance_labels_sha256' in row:assert ah(yn)==row['nuisance_labels_sha256']
    return y,yn
def allocate(steps=512):
    shapes=dict(rho=(7,steps+1,6),rho_adam_m=(7,steps+1,6),rho_adam_v=(7,steps+1,6),
        aggregate_loss=(7,steps),aggregate_gradient=(7,steps,6),rho_unprojected=(7,steps,6),
        maximum_absolute_logit=(7,steps),seq_micro_rho=(steps,5,6),seq_micro_loss=(steps,4),
        seq_micro_gradient=(steps,4,6),seq_micro_rho_unprojected=(steps,4,6),
        seq_micro_adam_m=(steps,4,6),seq_micro_adam_v=(steps,4,6),seq_micro_maximum_absolute_logit=(steps,4))
    for path,L in zip(PATHS,BANK_COUNTS):
        shapes['bank_loss_'+path]=(steps,L);shapes['bank_gradient_'+path]=(steps,L,6)
        shapes['bank_maximum_absolute_logit_'+path]=(steps,L)
    tape={k:np.full(s,np.nan,np.float32) for k,s in shapes.items()}
    for k,s in [('projection_alpha',(7,steps)),('proposal_eigmin',(7,steps)),('eigmin',(7,steps+1)),
                ('seq_micro_projection_alpha',(steps,4)),('seq_micro_proposal_eigmin',(steps,4)),('seq_micro_eigmin',(steps,4))]:
        tape[k]=np.full(s,np.nan,np.float64)
    for k in ['repaired','finite']:tape[k]=np.zeros((7,steps),bool)
    for k in ['feasible','cholesky_success']:tape[k]=np.zeros((7,steps+1),bool)
    for k in ['seq_micro_repaired','seq_micro_cholesky_success']:tape[k]=np.zeros((steps,4),bool)
    for k in DIAGNOSTICS[1:]:
        tape[k]=np.zeros((7,steps),np.int64);tape['seq_micro_'+k]=np.zeros((steps,4),np.int64)
        for path,L in zip(PATHS,BANK_COUNTS):tape['bank_'+k+'_'+path]=np.zeros((steps,L),np.int64)
    tape['batch_start']=np.full(steps,-1,np.int64)
    tape['adam_step']=np.zeros((7,steps+1),np.int64);tape['seq_micro_adam_step']=np.zeros((steps,4),np.int64)
    for k in TIMERS:tape[k]=np.zeros((4,steps),np.float64)
    for k in COUNTERS:tape[k]=np.zeros((4,steps),np.int64)
    for k in SHARED:tape[k]=np.zeros(steps,np.float64)
    return tape
def snapshot(tape,models,optimizers,index):
    for ai,path in enumerate(PATHS):
        model=models[path];rho=model.rho.detach().numpy();eig=float(np.linalg.eigvalsh(kernel.old.covariance_numpy(rho)).min())
        with torch.no_grad():_,info=torch.linalg.cholesky_ex(model.covariance())
        assert eig>=.1 and int(info)==0 and np.isfinite(rho).all()
        assert torch.equal(model.fixed_scales,torch.ones_like(model.fixed_scales))
        tape['rho'][ai,index]=rho;tape['eigmin'][ai,index]=eig
        tape['feasible'][ai,index]=True;tape['cholesky_success'][ai,index]=True
        state=optimizers[path].state.get(model.rho,{})
        for key,letter in [('exp_avg','m'),('exp_avg_sq','v')]:
            tape['rho_adam_'+letter][ai,index]=state[key].detach().numpy() if key in state else np.zeros(6,np.float32)
        tape['adam_step'][ai,index]=int(state.get('step',0))
def advance(tape,models,optimizers,step,banks,y,yn,begin,data_id,repeat,numerical):
    result,controls,timing=kernel.step_quartet(models,optimizers,banks,y,yn,numerical)
    tape['batch_start'][step]=begin;rows=[]
    for ai,path in enumerate(PATHS):
        row=result[path];update=row['microsteps'][-1] if path=='SEQ4' else row
        tape['aggregate_loss'][ai,step]=row['loss'];tape['aggregate_gradient'][ai,step]=row['rho_gradient']
        for key in ['rho_unprojected','projection_alpha','proposal_eigmin']:tape[key][ai,step]=update[key]
        tape['repaired'][ai,step]=update['projected'];tape['finite'][ai,step]=True
        tape['bank_loss_'+path][step]=row['bank_loss'];tape['bank_gradient_'+path][step]=row['bank_gradient']
        for key in DIAGNOSTICS:
            tape[key][ai,step]=row['diagnostics'][key];tape['bank_'+key+'_'+path][step]=[d[key] for d in row['bank_diagnostics']]
    for microstep,row in enumerate(result['SEQ4']['microsteps']):
        if microstep==0:tape['seq_micro_rho'][step,0]=row['rho_before']
        else:assert np.array_equal(tape['seq_micro_rho'][step,microstep],row['rho_before'])
        tape['seq_micro_rho'][step,microstep+1]=row['rho_after']
        for key,source in [('loss','loss'),('gradient','rho_gradient'),('rho_unprojected','rho_unprojected'),
                           ('projection_alpha','projection_alpha'),('proposal_eigmin','proposal_eigmin'),('eigmin','eigmin'),
                           ('repaired','projected'),('cholesky_success','cholesky_success'),('adam_m','adam_m'),('adam_v','adam_v'),('adam_step','adam_step')]:
            tape['seq_micro_'+key][step,microstep]=row[source]
        for key in DIAGNOSTICS:tape['seq_micro_'+key][step,microstep]=row['diagnostics'][key]
    if numerical:
        for path in PATHS:
            names=[f'SEQ4_micro{j}' for j in range(4)] if path=='SEQ4' else [path]
            for key in names:rows.append(dict(data_id=data_id,repeat=repeat,preupdate_step=step,program=path,**controls[key]))
    for si,strategy in enumerate(STRATEGIES):
        for key,value in timing['strategies'][strategy].items():tape[key][si,step]=value
    for key in SHARED[:3]:tape[key][step]=timing[key]
    snapshot(tape,models,optimizers,step+1)
    assert np.array_equal(tape['seq_micro_rho'][step,0],tape['rho'][5,step])
    assert np.array_equal(tape['seq_micro_rho'][step,4],tape['rho'][5,step+1])
    return rows
def final_moments(tape,models,optimizers):
    for letter in ['m','v']:tape['final_rho_adam_'+letter]=tape['rho_adam_'+letter][:,-1].copy()
    tape['final_adam_step']=tape['adam_step'][:,-1].copy()
def outputs(tape,tail=64):
    paths={}
    for ai,path in enumerate(PATHS):
        paths[path]={}
        for kind in ['TAIL','LAST']:
            rho=tape['rho'][ai,-tail:].astype(float).mean(axis=0) if kind=='TAIL' else tape['rho'][ai,-1].astype(float)
            C=kernel.old.covariance_numpy(rho);eig=float(np.linalg.eigvalsh(C).min());assert eig>=.1
            paths[path][kind]=dict(rho=rho.tolist(),covariance=C.tolist(),scales=[1.]*256,eigmin=eig)
    result={p:paths[p] for p in ['IN4','SEQ4','FACT']};result['OUT4']={};members={p:paths[p] for p in MEMBERS}
    for kind in ['TAIL','LAST']:
        rho=np.stack([np.asarray(members[p][kind]['rho'],float) for p in MEMBERS]).mean(axis=0)
        C=kernel.old.covariance_numpy(rho);eig=float(np.linalg.eigvalsh(C).min());assert eig>=.1
        result['OUT4'][kind]=dict(rho=rho.tolist(),covariance=C.tolist(),scales=[1.]*256,eigmin=eig)
    return {p:result[p] for p in STRATEGIES},members
def numerical_summary(rows):
    bank=[b for r in rows for b in r['per_bank_controls']]
    return dict(aggregate_checks=len(rows),aggregate_passed=sum(r['passed'] for r in rows),
        aggregate_loss_passed=sum(r['loss_passed'] for r in rows),aggregate_gradient_passed=sum(r['gradient_passed'] for r in rows),
        bank_checks=len(bank),bank_loss_passed=sum(b['loss_passed'] for b in bank),bank_gradient_passed=sum(b['gradient_passed'] for b in bank),
        stable64_calls=sum(r['stable64_calls'] for r in rows),maximum_aggregate_loss_difference=max(abs(r['loss_difference']) for r in rows),
        maximum_aggregate_gradient_difference=max(r['maximum_gradient_difference'] for r in rows),
        maximum_bank_loss_difference=max(abs(b['loss_difference']) for b in bank),maximum_bank_gradient_difference=max(b['maximum_gradient_difference'] for b in bank),
        passed=bool(all(r['passed'] for r in rows) and all(b['loss_passed'] and b['gradient_passed'] for b in bank)))
def schema():
    config,_=load_contract(False);sample=allocate(steps=2)
    return dict(source_sha256=sha(SOURCE),kernel_sha256=sha(kernel.__file__),config_sha256=sha(PACKAGE/'CONFIG.json'),
        provenance=kernel.provenance(),environment=kernel.environment(),config=config,
        array_schema={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in sample.items()},
        path_order=PATHS,strategy_order=STRATEGIES,member_order=MEMBERS,
        shape_note='Sample2steps; scientific512macrosteps,513states. SEQ micro_rho[512,5,6] includesmacrostart and4ends; allmacro/micro Adam moments explicit.',
        sequential_semantics='aggregate_loss/gradient forSEQ4 are descriptive ordered means at4differentstates, NEVER applied updates. Macro rho_unprojected/projection_alpha/proposal_eigmin/repaired forSEQ4 describeONLY itslastmicrostep. Fullactualupdates use seq_micro_* arrays.',
        input_arrays={'first_prior_active':'float64[4,100,64,4]','first_prior_noise':'float64[4,100,64,256]'},
        output_contract='records sorted(data_id,repeat); outputs[IN4/OUT4/SEQ4/FACT][TAIL/LAST] +members[M0..3]. TAILlast64macroendphysicalstates449..512; OUT4float64 membermean. FACTanalyticallyeliminatesnuisance; savedscales1stateonly.',
        RNG_order='PCG64 SeedSequence([202610040100,data_id,repeat,bank_index,role0active/1noise]). Eachstepbank0..3 float64[K,B,coordinate]Cbyteshashedbeforeoncefloat32prepare; samebanksharedIN4/Mj/SEQ4; FACTbank0.',
        labels='16entirelynewdatasetsgeneratedonceunderdataseed202610040000,twopriorquartetsperdata; completedatalabelsloadedunchanged.',no_self_authorized_freeze=True)
def validate_freeze():
    assert not (PACKAGE/'VERSION_LOCK.json').exists()
    frozen=read(PACKAGE/'FREEZE.json');assert frozen['run_authorized'] is True
    for path,digest in frozen['inputs'].items():assert sha(ROOT/path)==digest,path
    for p in [SOURCE,Path(kernel.__file__),PACKAGE/'CONFIG.json']:assert frozen['inputs'][rel(p)]==sha(p)
    datafreeze=read(PACKAGE/'DATA_FREEZE.json');assert datafreeze['run_authorized'] is True
    assert datafreeze['root_freeze_sha256']==sha(PACKAGE/'FREEZE.json')
    for path,digest in datafreeze['inputs'].items():assert sha(ROOT/path)==digest,path
    assert datafreeze['inputs'][rel(PACKAGE/'DATA.json')]==sha(PACKAGE/'DATA.json')
    _,data=load_contract()
    for row in data['data']:assert datafreeze['inputs'][row['path']]==row['sha256']
    for path,source in [(kernel.OUT/'SELFTEST.json',Path(kernel.__file__)),(kernel.OUT/'BENCHMARK.json',Path(kernel.__file__)),(OUT/'SELFTEST.json',SOURCE)]:
        test=read(path);assert test['passed'] and test['source_sha256']==sha(source)
    kernel.provenance();return frozen
def run_pair(data_id,repeat):
    validate_freeze();config,data=load_contract();assert data_id in config['data_ids'] and 0<=repeat<2
    job_id=f'd{data_id:02}_r{repeat:02}';inp=OUT/f'input_{job_id}.npz';out=OUT/f'trace_{job_id}.npz';record_path=OUT/f'quartet_{job_id}.json'
    assert not any(p.exists() for p in [inp,out,record_path,OUT/f'ERROR_{job_id}.json',OUT/f'partial_{job_id}.npz'])
    source_data=next(d for d in data['data'] if d['data_id']==data_id);labels,noise_labels=load_labels(source_data)
    entropy={f'bank_{bank}_{role}':[config['seed_base'],data_id,repeat,bank,role_id] for bank in range(4) for role,role_id in config['roles'].items()}
    rng={k:np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed))) for k,seed in entropy.items()}
    initial={k:g.bit_generator.state for k,g in rng.items()};digests={k:hashlib.sha256() for k in rng}
    models,opts=kernel.make_models();tape=allocate();snapshot(tape,models,opts,0)
    numerical=[];completed=0;step=-1;consumed=0;first=[];tick=time.perf_counter()
    try:
        for step in range(512):
            begin=64*step%4096
            def banks():
                nonlocal consumed
                for bank in range(4):
                    draw=time.perf_counter();active=rng[f'bank_{bank}_active'].standard_normal((100,64,4),dtype=np.float64)
                    noise=rng[f'bank_{bank}_noise'].standard_normal((100,64,256),dtype=np.float64)
                    digests[f'bank_{bank}_active'].update(active.tobytes(order='C'));digests[f'bank_{bank}_noise'].update(noise.tobytes(order='C'));consumed+=64
                    if step==0:first.append((active.copy(),noise.copy()))
                    tape['input_generation_and_hash_seconds'][step]+=time.perf_counter()-draw;yield active,noise
            numerical.extend(advance(tape,models,opts,step,banks(),labels[begin:begin+64],noise_labels[begin:begin+64],begin,data_id,repeat,step in config['numerical_preupdate_steps']))
            completed=step+1
            if step==0:
                with inp.open('xb') as handle:np.savez_compressed(handle,first_prior_active=np.stack([b[0] for b in first]),first_prior_noise=np.stack([b[1] for b in first]))
                first.clear()
        final_moments(tape,models,opts)
        assert np.array_equal(tape['final_adam_step'],[512]*5+[2048,512])
        assert np.array_equal(tape['original_API_calls'],np.repeat([[4],[4],[4],[1]],512,axis=1))
        for key in ['optimizer_steps','repair_calls']:assert np.array_equal(tape[key],np.repeat([[1],[4],[4],[1]],512,axis=1))
        assert all(np.isfinite(v).all() for v in tape.values())
        with out.open('xb') as handle:np.savez_compressed(handle,**tape)
        result,members=outputs(tape)
        diagnostics={}
        for ai,path in enumerate(PATHS):
            prefix='seq_micro_' if path=='SEQ4' else ''
            repaired=tape[prefix+'repaired'] if prefix else tape['repaired'][ai]
            proposal=tape[prefix+'proposal_eigmin'] if prefix else tape['proposal_eigmin'][ai]
            alpha=tape[prefix+'projection_alpha'] if prefix else tape['projection_alpha'][ai]
            diagnostics[path]=dict(feasibility_repair_count=int(repaired.sum()),minimum_eigenvalue=float(min(tape['eigmin'][ai].min(),tape['seq_micro_eigmin'].min())) if prefix else float(tape['eigmin'][ai].min()),
                minimum_proposal_eigenvalue=float(proposal.min()),minimum_repair_alpha=float(alpha.min()),maximum_absolute_logit=float(tape['maximum_absolute_logit'][ai].max()),
                sigmoid_zero_count=int(tape['sigmoid_zero_count'][ai].sum()),sigmoid_one_count=int(tape['sigmoid_one_count'][ai].sum()),probability_evaluations=int(tape['probability_evaluations'][ai].sum()),all_finite=True,all_feasible=True,all_Cholesky_success=True)
        assert job_id==f'd{data_id:02}_r{repeat:02}'
        record=dict(id=job_id,data_id=data_id,repeat=repeat,source_sha256=sha(SOURCE),kernel_sha256=sha(kernel.__file__),freeze_sha256=sha(PACKAGE/'FREEZE.json'),data_freeze_sha256=sha(PACKAGE/'DATA_FREEZE.json'),
            input_file=rel(inp),input_sha256=sha(inp),output_file=rel(out),output_sha256=sha(out),data_file=source_data['path'],data_sha256=source_data['sha256'],
            label_sha256=dict(active=ah(labels),noise=ah(noise_labels)),role_seed_sequences=entropy,RNG_initial_states=initial,RNG_final_states={k:g.bit_generator.state for k,g in rng.items()},role_input_sha256={k:h.hexdigest() for k,h in digests.items()},
            outputs=result,members=members,completed_steps=completed,independent_observation_banks_consumed=consumed,numerical_controls=numerical,numerical_qualification=numerical_summary(numerical),diagnostics=diagnostics,
            strategy_accounting={s:{k:(float(tape[k][si].sum()) if k in TIMERS else int(tape[k][si].sum())) for k in COUNTERS+TIMERS} for si,s in enumerate(STRATEGIES)},
            shared_timing={k:float(tape[k].sum()) for k in SHARED},seconds=time.perf_counter()-tick,
            timing_scope='Shared-execution call timers, not separately timed end-to-end runs.',scope='New iid dataset, twoindependentpriorquartetsperdata; main3strategies4scorecalls,1/4/4updates; FACTqualityonlybank0.')
        validate_freeze();save(record_path,record);return record
    except BaseException:
        partial=OUT/f'partial_{job_id}.npz'
        if not partial.exists():
            with partial.open('xb') as handle:np.savez_compressed(handle,**tape)
        save(OUT/f'ERROR_{job_id}.json',dict(data_id=data_id,repeat=repeat,current_step=step,completed_steps=completed,consumed_observation_banks=consumed,traceback=traceback.format_exc(),RNG_initial_states=initial,RNG_current_states={k:g.bit_generator.state for k,g in rng.items()},role_hashes_so_far={k:h.hexdigest() for k,h in digests.items()},automatic_retry=False));raise
def run():
    validate_freeze();config,_=load_contract();OUT.mkdir(parents=True,exist_ok=True)
    assert not any(list(OUT.glob(pattern)) for pattern in ['EXECUTION.json','quartet_*.json','input_*.npz','trace_*.npz','partial_*.npz','ERROR*.json'])
    save(OUT/'EXECUTION.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(SOURCE),kernel_sha256=sha(kernel.__file__),freeze_sha256=sha(PACKAGE/'FREEZE.json'),data_freeze_sha256=sha(PACKAGE/'DATA_FREEZE.json'),workers=8,environment=kernel.environment(),automatic_retry=False))
    tasks=[(d,r) for r in range(2) for d in config['data_ids']];records={};active={};next_task=0;tick=time.perf_counter();pool=ProcessPoolExecutor(max_workers=8)
    try:
        while next_task<8:
            pair=tasks[next_task];active[pool.submit(run_pair,*pair)]=pair;next_task+=1
        while active:
            done,_=wait(active,return_when=FIRST_COMPLETED);finished=[f.result() for f in done]
            for f in done:del active[f]
            for row in finished:
                records[(row['data_id'],row['repeat'])]=row;print(json.dumps(dict(completed_data_id=row['data_id'],repeat=row['repeat'],seconds=row['seconds'])),flush=True)
            save(OUT/'PROGRESS.json',dict(completed=[list(k) for k in sorted(records)],running=[list(k) for k in active.values()],seconds=time.perf_counter()-tick),False)
            while len(active)<8 and next_task<len(tasks):
                pair=tasks[next_task];active[pool.submit(run_pair,*pair)]=pair;next_task+=1
        ordered=[records[(d,r)] for d in config['data_ids'] for r in range(2)];controls=[c for row in ordered for c in row['numerical_controls']]
        assert len(controls)==1280 and sum(c['stable64_calls'] for c in controls)==1664
        result=dict(source_sha256=sha(SOURCE),kernel_sha256=sha(kernel.__file__),freeze_sha256=sha(PACKAGE/'FREEZE.json'),data_freeze_sha256=sha(PACKAGE/'DATA_FREEZE.json'),config_sha256=sha(PACKAGE/'CONFIG.json'),data_sha256=sha(PACKAGE/'DATA.json'),config=config,records=ordered,numerical_qualification=numerical_summary(controls),
            accounting=dict(new_datasets=16,new_observations=65536,data_Gaussian_scalars=262144,data_Uniform_scalars=17039360,prior_repeats_per_dataset=2,independent_prior_quartets=32,controlled_optimization_programs=224,
                main_strategy_TAIL_outputs=128,main_strategy_LAST_descriptive_outputs=128,private_program_TAIL_outputs=224,private_program_LAST_outputs=224,derived_OUT4_TAIL_outputs=32,independent_observation_banks=4194304,
                unique_prior_vectors=419430400,prior_Gaussian_scalars=109051904000,active_prior_Gaussian_scalars=1677721600,noise_prior_Gaussian_scalars=107374182400,
                shared_quartet_observation_presentations=1048576,score_call_observation_presentations=13631488,optimizer_call_observation_presentations=10485760,
                original_training_API_calls=212992,optimizer_calls=163840,repair_calls=163840,strategy_original_API_calls=dict(IN4=65536,OUT4=65536,SEQ4=65536,FACT=16384),strategy_optimizer_calls=dict(IN4=16384,OUT4=65536,SEQ4=65536,FACT=16384),strategy_repair_calls=dict(IN4=16384,OUT4=65536,SEQ4=65536,FACT=16384),aggregate_numeric_controls=1280,stable64_calls=1664,new_neural_network_fits=0,new_model_integrals=0,scientific_retries=0),
            seconds=time.perf_counter()-tick,strategy_timing_totals={s:{k:sum(row['strategy_accounting'][s][k] for row in ordered) for k in COUNTERS+TIMERS} for s in STRATEGIES},
            timing_scope='Shared-execution timers are not independent end-to-end runtimes; main3equal4scorecallsbut1/4/4optimizer/repaircalls; FACTunmatched.',scope='16entirelynewiid datasets with2independentpriorquartetseach; privateSEQ4fixedbankorder0..3. No universaloptimizerclaim.')
        validate_freeze();save(OUT/'RESULTS.json',result);save(OUT/'STATE.json',dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),results_sha256=sha(OUT/'RESULTS.json'),source_sha256=sha(SOURCE),seconds=result['seconds']))
        print(json.dumps(dict(complete=True,seconds=result['seconds'],results_sha256=sha(OUT/'RESULTS.json'),numerical_qualification=result['numerical_qualification'])),flush=True)
    except BaseException:
        for f in active:f.cancel()
        save(OUT/'ERROR.json',dict(traceback=traceback.format_exc(),completed_pairs=[list(k) for k in records],already_active_pairs=[list(k) for k in active.values()],automatic_retry=False));raise
    finally:pool.shutdown(wait=True,cancel_futures=True)
def selftest():
    assert not (PACKAGE/'FREEZE.json').exists()
    config,_=load_contract(False);models,opts=kernel.make_models();tape=allocate(steps=2);snapshot(tape,models,opts,0)
    banks,y,yn=kernel.deterministic_inputs(B=64,n=256);rows=[]
    for step in range(2):rows.extend(advance(tape,models,opts,step,iter(banks),y,yn,64*step,-1,-1,True))
    final_moments(tape,models,opts);result,members=outputs(tape,tail=2)
    assert all(np.isfinite(v).all() for v in tape.values())
    assert np.array_equal(tape['rho'][:,0],np.zeros((7,6))) and np.array_equal(tape['final_adam_step'],[2]*5+[8,2])
    assert np.array_equal(tape['seq_micro_adam_step'],np.arange(1,9).reshape(2,4))
    for ai,path in enumerate(PATHS):
        for step in range(2):
            expected=kernel.previous.ordered_average(list(torch.tensor(tape['bank_gradient_'+path][step]))).numpy()
            assert np.array_equal(expected,tape['aggregate_gradient'][ai,step])
        endpoint=(members if path in MEMBERS else result)[path]
        assert np.array_equal(endpoint['TAIL']['rho'],tape['rho'][ai,1:].astype(float).mean(axis=0))
    assert np.array_equal(result['OUT4']['TAIL']['rho'],np.stack([members[p]['TAIL']['rho'] for p in MEMBERS]).mean(axis=0))
    assert np.array_equal(tape['original_API_calls'],np.repeat([[4],[4],[4],[1]],2,axis=1))
    assert np.array_equal(tape['optimizer_steps'],np.repeat([[1],[4],[4],[1]],2,axis=1))
    summary=numerical_summary(rows);assert summary['passed'] and summary['aggregate_checks']==20 and summary['stable64_calls']==26
    save(OUT/'SELFTEST.json',dict(passed=True,source_sha256=sha(SOURCE),kernel_sha256=sha(kernel.__file__),config_sha256=sha(PACKAGE/'CONFIG.json'),array_schema={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in tape.items()},numerical_qualification=summary,
        per_bank_gradients_explicitly_saved=True,sequential_microstate_and_Adam_boundaries_checked=True,physical_tail_and_OUT4_nested_mean_checked=True,
        deterministic_original_API_calls=26,deterministic_stable64_calls=26,new_scientific_RNG=0,new_observations=0,new_model_integrals=0,new_training_fits=0,
        scope='Two deterministic integrationsteps andtapeschema/arithmetictests; nosampledtrajectory orconvergencepilot. No DATA.json neededbefore newdatafreeze.'))
    save(OUT/'SCHEMA.json',schema());print(json.dumps(dict(passed=True,source_sha256=sha(SOURCE),kernel_sha256=sha(kernel.__file__),numerical=summary)))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['schema','check','run']);a=parser.parse_args()
    if a.mode=='schema':print(json.dumps(schema(),ensure_ascii=True,indent=2))
    else:{'check':selftest,'run':run}[a.mode]()
