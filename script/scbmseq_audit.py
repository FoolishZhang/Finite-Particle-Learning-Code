"""Independent, bounded audit of the new-data sequential-update experiment.

Uses only the frozen independent scbm6 audit cores at bottom level. Producer
evaluators, optimizers, RNG generators, and statistical helpers are not imported
by the scientific audit. Deterministic selftests draw no scientific randomness.
All output names retain an immutable (data_id, repeat) job identifier.
"""
from __future__ import annotations
import os
for _thread_key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_thread_key] = '1'
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse, hashlib, itertools, json, math, sys, time, traceback
import numpy as np
import torch
from scipy.special import expit, logsumexp
from scipy.stats import t as student
import scbm6_audit as core

torch.set_num_threads(1)
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve()
BASE = ROOT / 'artifacts/scbm_sequential_generalization'
OUT = BASE / 'AUDIT'
ARMS = ['IN4', 'OUT4', 'SEQ4', 'FACT']
PROGRAMS = ['IN4', 'M0', 'M1', 'M2', 'M3', 'SEQ4', 'FACT']
MEMBERS = ['M0', 'M1', 'M2', 'M3']
BITS = np.array(list(itertools.product((0, 1), repeat=4)))
PAIRS = list(itertools.combinations(range(4), 2))
STEPS = [0, 31, 127, 511]
BOTTOM_DATA = [0, 5, 10, 15]
RISK_CONTRASTS = [('OUT4', 'IN4'), ('SEQ4', 'OUT4'), ('OUT4', 'I'),
                  ('SEQ4', 'I'), ('FACT', 'I')]
SCORE_CONTRASTS = RISK_CONTRASTS[:2]
TOL = dict(arithmetic=2e-10, native_loss=5e-5, native_gradient=3e-6,
           math64_loss=1e-10, math64_gradient=1e-10, proposal=3e-6,
           repair_parameter=2e-7, repair=1e-12, moment_abs=2e-6,
           moment_rel=2e-5, eval_score=2e-10, reference_PMF=5e-8)
SCOPE = dict(training_replay_workers=8, training_pairs=32,
    training_RNG_roles=256, training_replayed_Gaussian_scalars=109051904000,
    training_native_aggregate_states=160, training_native_bank_calls=208,
    training_bottom_data_ids=BOTTOM_DATA, training_bottom_repeats=[0],
    training_bottom_steps=STEPS, full_saved_Adam_repair_replay=True,
    full_SEQ4_microtrace_replay=True, full_retraining=False,
    new_datasets=16, observations_per_dataset=4096, data_RNG_roles=48,
    evaluation_outer_datasets=16, evaluation_groups_per_dataset=8,
    evaluation_banks_per_group=512, evaluation_prefix_banks=2,
    evaluation_prefix_score_cells=81920, evaluation_native_calls=640,
    evaluation_replayed_banks=65536,
    evaluation_replayed_Gaussian_scalars=1703936000,
    reference_data_ids=[0, 4, 8, 12], reference_repeats=[0],
    reference_arms=ARMS, reference_order=36, reference_PMFs=16,
    reference_nodes=26873856, new_independent_random_draws=0,
    no_automatic_retry=True, tolerances=TOL)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ah(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save(path, value, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if exclusive else 'w', encoding='utf8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def near(a, b, tol=TOL['arithmetic']):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    error = np.abs(a - b)
    assert np.all(error <= tol), (float(error.max()), tol)
    return float(error.max()) if error.size else 0.


def cov(rho):
    value = np.eye(4)
    for x, (i, j) in zip(rho, PAIRS):
        value[i, j] = value[j, i] = x
    return value


def seqmean(values):
    values = np.asarray(values, np.float32)
    total = values[0].copy()
    for value in values[1:]:
        total = np.asarray(total + value, np.float32)
    return np.asarray(total * np.float32(1 / len(values)), np.float32)


def moments(values):
    values = [float(value) for value in values]
    mean = math.fsum(values) / len(values)
    se = math.sqrt(math.fsum((value - mean)**2 for value in values) /
                   (len(values) * (len(values) - 1)))
    return mean, se


def job_paths(data_id, repeat):
    """Private job ID is constructed once, never from a program loop variable."""
    job_id = f'd{data_id:02}_r{repeat:02}'
    return job_id, OUT / 'TRAIN' / f'{job_id}.json', OUT / 'TRAIN' / f'ERROR_{job_id}.json'


def verify_manifest(path):
    value = read(path)
    assert value['run_authorized'] is True
    for name, digest in value['inputs'].items():
        assert sha(ROOT / name) == digest, name
    return value


def verify_freeze(derived=True):
    """Staged provenance: ROOT cannot bind not-yet-created DATA/derived output."""
    root = verify_manifest(BASE / 'FREEZE.json')
    for path in (SOURCE, Path(core.__file__), core.OFF, BASE / 'CONFIG.json', OUT / 'SELFTEST.json'):
        assert root['inputs'][rel(path)] == sha(path), path
    assert rel(BASE / 'DATA.json') not in root['inputs']
    assert rel(BASE / 'DATA_FREEZE.json') not in root['inputs']
    assert rel(BASE / 'DERIVED_FREEZE.json') not in root['inputs']
    data = verify_manifest(BASE / 'DATA_FREEZE.json')
    assert data['root_freeze_sha256'] == sha(BASE / 'FREEZE.json')
    assert data['inputs'][rel(BASE / 'DATA.json')] == sha(BASE / 'DATA.json')
    assert rel(BASE / 'DERIVED_FREEZE.json') not in data['inputs']
    if derived:
        outputs = verify_manifest(BASE / 'DERIVED_FREEZE.json')
        assert outputs['root_freeze_sha256'] == sha(BASE / 'FREEZE.json')
        assert outputs['data_freeze_sha256'] == sha(BASE / 'DATA_FREEZE.json')
        assert outputs['inputs'][rel(BASE / 'CANDIDATES.json')] == sha(BASE / 'CANDIDATES.json')
    assert not (BASE / 'VERSION_LOCK.json').exists()
    return root, data


def data_check():
    cfg, data, spec = [read(BASE/name) for name in ('CONFIG.json', 'DATA.json', 'SPEC.json')]
    required = dict(data_ids=list(range(16)), repeats=2, arms=ARMS, strategies=ARMS,
                    programs=PROGRAMS, members=MEMBERS, outer_banks=4, K=100,
                    nuisance=256, observations=4096, batch_size=64, steps=512,
                    tail_postupdates=64, initial_rho=[0.]*6, lr_rho=.02,
                    seq_lr_rho=.02, seq_bank_order=[0,1,2,3], scale=1.,
                    betas=[.9,.999], eps=1e-8, eigen_floor=.1, repair_target=.1001,
                    shuffle=False, gradient_clip=False, learning_rate_schedule=False,
                    early_stop=False, seed_base=202610040100, roles={'active':0,'noise':1})
    assert all(cfg[key] == value for key,value in required.items())
    assert (cfg['training_family']['alpha'],cfg['training_family']['size'],cfg['training_family']['df']) == (.025,5,15)
    assert cfg['training_family']['SE_thresholds'] == [.0005,.0005,.0002,.0002,.0002]
    assert (cfg['evaluation']['alpha'],cfg['evaluation']['family_size'],cfg['evaluation']['df']) == (.025,2,15)
    assert data['root_freeze_sha256'] == sha(BASE/'FREEZE.json')
    for key in ('patterns','population_weights','mu','truth_rho','pair_order','truth_covariance','identity_risk'):
        assert data[key] == spec[key], key
    assert data['patterns'] == BITS.tolist() and data['pair_order'] == [list(p) for p in PAIRS]
    assert data['mu'] == core.MU.tolist() and data['truth_covariance'] == core.C0.tolist()
    for path,digest in data['inputs'].items():
        assert sha(ROOT/path) == digest, path
    assert [row['data_id'] for row in data['data']] == list(range(16))
    assert len({row['path'] for row in data['data']}) == 16
    records=[]; generation=cfg['data_generation']
    for row in data['data']:
        d=row['data_id']; path=ROOT/row['path']; assert sha(path) == row['sha256']
        with np.load(path,allow_pickle=False) as archive:
            arrays={key:archive[key] for key in archive.files}
        assert set(arrays) == {'active_normal','active_uniform','nuisance_uniform','active_labels','nuisance_labels'}
        for key,value in arrays.items():
            assert ah(value) == row['array_sha256'][key], (d,key)
        for role,shape in [('active_normal',(4096,4)),('active_uniform',(4096,4)),('nuisance_uniform',(4096,256))]:
            seed=[generation['seed_base'],d,generation['roles'][role]]
            assert row['role_seed_sequences'][role] == seed
            rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
            assert rng.bit_generator.state == row['RNG_initial_states'][role]
            value=rng.standard_normal(shape,dtype=np.float64) if role=='active_normal' else rng.random(shape,dtype=np.float64)
            assert np.array_equal(value,arrays[role]) and ah(value) == row['array_sha256'][role]
            assert rng.bit_generator.state == row['RNG_final_states'][role]
        latent=arrays['active_normal']@np.linalg.cholesky(np.array(data['truth_covariance'])).T+data['mu']
        expected=(arrays['active_uniform']<expit(latent)).astype(np.int8)
        expected_noise=(arrays['nuisance_uniform']<.5).astype(np.int8)
        assert arrays['active_labels'].dtype == arrays['nuisance_labels'].dtype == np.int8
        assert np.array_equal(expected,arrays['active_labels']) and np.array_equal(expected_noise,arrays['nuisance_labels'])
        assert ah(expected)==row['active_labels_sha256'] and ah(expected_noise)==row['nuisance_labels_sha256']
        counts=[int(np.all(expected==pattern,axis=1).sum()) for pattern in BITS]
        assert counts==row['counts'] and row['weights']==[count/4096 for count in counts]
        records.append(dict(data_id=d,path=row['path'],sha256=row['sha256'],roles=3,passed=True))
    assert data['accounting']==dict(new_observations=65536,data_Gaussian=262144,data_Uniform=17039360)
    save(OUT/'DATA.json',dict(passed=True,datasets=16,new_observations=65536,replayed_RNG_roles=48,
         replayed_Gaussian_scalars=262144,replayed_uniform_scalars=17039360,records=records,
         all_input_hashes_initial_final_states_labels_counts_weights_match=True,new_independent_draws=0))
    return cfg,data


def check_repair(proposal, actual, alpha, eig0, eig1, repaired):
    restored,a,b,c=core.repair(proposal)
    error=near(restored,actual,TOL['repair_parameter'])
    near([a,b,c],[alpha,eig0,eig1],TOL['repair'])
    assert repaired == (b<.1) and c>=.1
    np.linalg.cholesky(cov(actual))
    return error


def check_moment(a,b):
    return near(a,b,TOL['moment_abs']+TOL['moment_rel']*abs(b))


def replay_trajectory(tape, steps=512):
    """All saved gradients, local Adam proposals, moments and physical repairs.

    Independent moments are cumulative. Parameter comparisons start at each
    saved before-state, so this is not an independently relearned trajectory.
    SEQ4 gradients are used individually; its macro mean is never applied.
    """
    t=tape
    assert t['rho'].shape==(7,steps+1,6) and np.array_equal(t['rho'][:,0],np.zeros((7,6),np.float32))
    assert t['rho_adam_m'].shape==t['rho_adam_v'].shape==(7,steps+1,6)
    assert np.array_equal(t['rho_adam_m'][:,0],np.zeros((7,6)))
    assert np.array_equal(t['rho_adam_v'][:,0],np.zeros((7,6)))
    assert np.array_equal(t['adam_step'][:,0],np.zeros(7))
    assert t['seq_micro_rho'].shape==(steps,5,6)
    assert t['seq_micro_gradient'].shape==(steps,4,6)
    assert np.array_equal(t['batch_start'],np.arange(steps)*64%4096)
    assert all(np.isfinite(value).all() for value in t.values())
    for key in ('finite','feasible','cholesky_success'):
        assert t[key].all(), key
    m=np.zeros((7,6),np.float32); v=m.copy()
    max_proposal=max_repair=max_moment=0.
    for step in range(steps):
        for ai,program in enumerate(PROGRAMS):
            losses=t['bank_loss_'+program][step]; gradients=t['bank_gradient_'+program][step]
            assert len(losses)==len(gradients)==([4,1,1,1,1,4,1][ai])
            near(seqmean(losses),t['aggregate_loss'][ai,step],0.)
            near(seqmean(gradients),t['aggregate_gradient'][ai,step],0.)
            for key in ('sigmoid_zero_count','sigmoid_one_count','probability_evaluations'):
                assert t[key][ai,step]==t['bank_'+key+'_'+program][step].sum()
            near(t['maximum_absolute_logit'][ai,step],t['bank_maximum_absolute_logit_'+program][step].max(),0.)
            if program!='SEQ4':
                m[ai],v[ai],delta=core.adam(m[ai],v[ai],t['aggregate_gradient'][ai,step],step+1)
                max_proposal=max(max_proposal,near(t['rho'][ai,step]+delta,t['rho_unprojected'][ai,step],TOL['proposal']))
                max_repair=max(max_repair,check_repair(t['rho_unprojected'][ai,step],t['rho'][ai,step+1],
                    t['projection_alpha'][ai,step],t['proposal_eigmin'][ai,step],t['eigmin'][ai,step+1],t['repaired'][ai,step]))
                assert t['adam_step'][ai,step+1]==step+1
            else:
                near(t['seq_micro_rho'][step,0],t['rho'][ai,step],0.)
                near(t['seq_micro_rho'][step,4],t['rho'][ai,step+1],0.)
                near(t['seq_micro_gradient'][step],gradients,0.)
                near(t['seq_micro_loss'][step],losses,0.)
                for bank_index in range(4):
                    iteration=4*step+bank_index+1
                    before=t['seq_micro_rho'][step,bank_index]
                    m[ai],v[ai],delta=core.adam(m[ai],v[ai],gradients[bank_index],iteration)
                    proposal=t['seq_micro_rho_unprojected'][step,bank_index]
                    max_proposal=max(max_proposal,near(before+delta,proposal,TOL['proposal']))
                    max_repair=max(max_repair,check_repair(proposal,t['seq_micro_rho'][step,bank_index+1],
                        t['seq_micro_projection_alpha'][step,bank_index],t['seq_micro_proposal_eigmin'][step,bank_index],
                        t['seq_micro_eigmin'][step,bank_index],t['seq_micro_repaired'][step,bank_index]))
                    max_moment=max(max_moment,check_moment(m[ai],t['seq_micro_adam_m'][step,bank_index]),
                                   check_moment(v[ai],t['seq_micro_adam_v'][step,bank_index]))
                    assert t['seq_micro_adam_step'][step,bank_index]==iteration
                    assert t['seq_micro_cholesky_success'][step,bank_index]
                for key in ('maximum_absolute_logit','sigmoid_zero_count','sigmoid_one_count','probability_evaluations'):
                    near(t['seq_micro_'+key][step],t['bank_'+key+'_SEQ4'][step],0.)
                for macrokey,microkey in [('rho_unprojected','seq_micro_rho_unprojected'),('projection_alpha','seq_micro_projection_alpha'),
                                          ('proposal_eigmin','seq_micro_proposal_eigmin'),('repaired','seq_micro_repaired')]:
                    near(t[macrokey][ai,step],t[microkey][step,3],0.)
                assert t['adam_step'][ai,step+1]==4*(step+1)
            max_moment=max(max_moment,check_moment(m[ai],t['rho_adam_m'][ai,step+1]),
                           check_moment(v[ai],t['rho_adam_v'][ai,step+1]))
        if step==0:
            for bank_index in range(4):
                near(t['bank_gradient_IN4'][0,bank_index],t['bank_gradient_M'+str(bank_index)][0,0],0.)
            near(t['bank_gradient_IN4'][0,0],t['bank_gradient_SEQ4'][0,0],0.)
    near(m,t['final_rho_adam_m'],TOL['moment_abs']+TOL['moment_rel']*abs(t['final_rho_adam_m']))
    near(v,t['final_rho_adam_v'],TOL['moment_abs']+TOL['moment_rel']*abs(t['final_rho_adam_v']))
    assert np.array_equal(t['final_adam_step'],[steps]*5+[steps*4,steps])
    assert np.array_equal(t['original_API_calls'],np.repeat([[4],[4],[4],[1]],steps,axis=1))
    for key in ('optimizer_steps','repair_calls'):
        assert np.array_equal(t[key],np.repeat([[1],[4],[4],[1]],steps,axis=1))
    return dict(local_proposal_max_error=max_proposal,repair_max_error=max_repair,
                moment_max_error=max_moment,all_saved_states_checked=True,seq_microsteps=steps*4,
                scope='Cumulative saved-gradient Adam moments and local saved-before-state proposals/repairs; no full relearning.')


def training_pair(row):
    data_id,repeat=row['data_id'],row['repeat']
    job_id,success_path,error_path=job_paths(data_id,repeat)
    start=time.perf_counter(); step=-1
    try:
        cfg=read(BASE/'CONFIG.json'); data=read(BASE/'DATA.json')['data'][data_id]
        assert sha(ROOT/row['input_file'])==row['input_sha256']
        assert sha(ROOT/row['output_file'])==row['output_sha256']
        with np.load(ROOT/row['input_file'],allow_pickle=False) as archive:
            initial={key:archive[key] for key in archive.files}
        with np.load(ROOT/row['output_file'],allow_pickle=False) as archive:
            tape={key:archive[key] for key in archive.files}
        with np.load(ROOT/data['path'],allow_pickle=False) as archive:
            y=archive['active_labels']; yn=archive['nuisance_labels']
        assert row['data_file']==data['path'] and row['data_sha256']==data['sha256']
        assert sha(ROOT/row['data_file'])==row['data_sha256']
        assert row['label_sha256']==dict(active=ah(y),noise=ah(yn))
        arithmetic=replay_trajectory(tape)
        generators={}; digests={}
        for bank_index in range(4):
            for role,role_id in [('active',0),('noise',1)]:
                role_key=f'bank_{bank_index}_{role}'; seed=[cfg['seed_base'],data_id,repeat,bank_index,role_id]
                assert row['role_seed_sequences'][role_key]==seed
                generator=np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
                assert generator.bit_generator.state==row['RNG_initial_states'][role_key]
                generators[role_key]=generator; digests[role_key]=hashlib.sha256()
        native_checks=[]
        for step in range(512):
            begin=step*64%4096
            for bank_index in range(4):
                active=generators[f'bank_{bank_index}_active'].standard_normal((100,64,4),dtype=np.float64)
                noise=generators[f'bank_{bank_index}_noise'].standard_normal((100,64,256),dtype=np.float64)
                for role,value in [('active',active),('noise',noise)]:
                    digests[f'bank_{bank_index}_{role}'].update(value.tobytes(order='C'))
                if step==0:
                    assert np.array_equal(active,initial['first_prior_active'][bank_index])
                    assert np.array_equal(noise,initial['first_prior_noise'][bank_index])
                if data_id in BOTTOM_DATA and repeat==0 and step in STEPS:
                    tasks=[('IN4',0,bank_index,None), (f'M{bank_index}',bank_index+1,0,None),
                           ('SEQ4',5,bank_index,bank_index)]
                    if bank_index==0:
                        tasks.append(('FACT',6,0,None))
                    for program,ai,local_bank,microstep in tasks:
                        rho=tape['seq_micro_rho'][step,bank_index] if microstep is not None else tape['rho'][ai,step]
                        method='FACT' if program=='FACT' else 'FIXED'
                        native=core.native(method,rho,None,active,noise,y[begin:begin+64],yn[begin:begin+64])
                        math64=core.math_score(method,rho,None,active,noise,y[begin:begin+64],yn[begin:begin+64])
                        controls=[x for x in row['numerical_controls'] if x['program']==program and x['preupdate_step']==step and
                                  (program!='SEQ4' or x['microstep']==microstep)]
                        assert len(controls)==1
                        control=controls[0]; expected=list(range(4)) if program=='IN4' else [bank_index]
                        assert control['input_bank_indices']==expected
                        control_bank=control['per_bank_controls'][local_bank if program=='IN4' else 0]
                        assert control_bank['input_bank_index']==bank_index
                        nl=near(native['loss'],tape['bank_loss_'+program][step,local_bank],TOL['native_loss'])
                        ng=near(native['rho_gradient'],tape['bank_gradient_'+program][step,local_bank],TOL['native_gradient'])
                        ml=near(math64['loss'],control_bank['stable64_loss'],TOL['math64_loss'])
                        mg=near(math64['rho_gradient'],control_bank['stable64_gradient'],TOL['math64_gradient'])
                        for name,key in [('zero','sigmoid_zero_count'),('one','sigmoid_one_count'),('evaluations','probability_evaluations')]:
                            assert native[name]==tape['bank_'+key+'_'+program][step,local_bank]
                        near(native['maxlogit'],tape['bank_maximum_absolute_logit_'+program][step,local_bank],2e-6)
                        native_checks.append(dict(data_id=data_id,repeat=repeat,step=step,program=program,microstep=microstep,
                            input_bank_index=bank_index,native_loss_error=nl,native_gradient_error=ng,
                            math64_loss_error=ml,math64_gradient_error=mg))
        for role_key,generator in generators.items():
            assert digests[role_key].hexdigest()==row['role_input_sha256'][role_key]
            assert generator.bit_generator.state==row['RNG_final_states'][role_key]
        assert row['completed_steps']==512 and row['independent_observation_banks_consumed']==512*4*64
        paths={}
        for ai,program in enumerate(PROGRAMS):
            paths[program]={}
            for kind in ('TAIL','LAST'):
                endpoint=tape['rho'][ai,449:].astype(float).mean(axis=0) if kind=='TAIL' else tape['rho'][ai,-1].astype(float)
                stored=row['members'][program][kind] if program in MEMBERS else row['outputs'][program][kind]
                near(endpoint,stored['rho']); near(cov(endpoint),stored['covariance'])
                near(np.linalg.eigvalsh(cov(endpoint))[0],stored['eigmin'],1e-12)
                assert stored['scales']==[1.]*256 and stored['eigmin']>=.1
                paths[program][kind]=endpoint
            diagnostic=row['diagnostics'][program]
            for key in ('sigmoid_zero_count','sigmoid_one_count','probability_evaluations'):
                assert int(tape[key][ai].sum())==diagnostic[key]
            repairs=int(tape['seq_micro_repaired'].sum()) if program=='SEQ4' else int(tape['repaired'][ai].sum())
            assert repairs==diagnostic['feasibility_repair_count']
            source_min_eig=tape['seq_micro_eigmin'].min() if program=='SEQ4' else tape['eigmin'][ai].min()
            source_min_proposal=tape['seq_micro_proposal_eigmin'].min() if program=='SEQ4' else tape['proposal_eigmin'][ai].min()
            source_min_alpha=tape['seq_micro_projection_alpha'].min() if program=='SEQ4' else tape['projection_alpha'][ai].min()
            for key,value in [('minimum_eigenvalue',min(1.,source_min_eig)),('minimum_proposal_eigenvalue',source_min_proposal),
                              ('minimum_repair_alpha',source_min_alpha),('maximum_absolute_logit',tape['maximum_absolute_logit'][ai].max())]:
                near(value,diagnostic[key])
        assert set(row['outputs'])==set(ARMS) and set(row['members'])==set(MEMBERS)
        for kind in ('TAIL','LAST'):
            endpoint=np.stack([paths[member][kind] for member in MEMBERS]).mean(axis=0)
            stored=row['outputs']['OUT4'][kind]
            near(endpoint,stored['rho'],0.); near(cov(endpoint),stored['covariance'],0.)
            assert stored['eigmin']>=.1 and stored['scales']==[1.]*256
        for ai,strategy in enumerate(ARMS):
            for key in ('original_API_calls','optimizer_steps','repair_calls','native_forward_backward_seconds','optimizer_and_repair_seconds'):
                near(tape[key][ai].sum(),row['strategy_accounting'][strategy][key])
        for key,value in row['shared_timing'].items():
            near(tape[key].sum(),value)
        assert len(row['numerical_controls'])==40
        for control in row['numerical_controls']:
            program=control['program']; s=control['preupdate_step']; assert s in STEPS
            expected=[control['microstep']] if program=='SEQ4' else list(range(4)) if program=='IN4' else [0] if program=='FACT' else [MEMBERS.index(program)]
            assert control['input_bank_indices']==expected
            for local_index,(bank_index,bank) in enumerate(zip(expected,control['per_bank_controls'])):
                assert bank['input_bank_index']==bank_index and bank['bank_index']==local_index
                tape_index=control['microstep'] if program=='SEQ4' else local_index
                near(bank['native_loss'],tape['bank_loss_'+program][s,tape_index],0.)
                near(bank['native_gradient'],tape['bank_gradient_'+program][s,tape_index],0.)
        assert numerical_rows(row['numerical_controls'])==row['numerical_qualification']
        result=dict(passed=True,data_id=data_id,repeat=repeat,job_id=job_id,
            input_sha256=row['input_sha256'],output_sha256=row['output_sha256'],RNG_roles=8,
            replayed_Gaussian_scalars=3407872000,native_checks=native_checks,
            **arithmetic,seconds=time.perf_counter()-start)
        save(success_path,result)
        return result
    except BaseException:
        save(error_path,dict(data_id=data_id,repeat=repeat,job_id=job_id,last_step=step,
             traceback=traceback.format_exc(),automatic_retry=False))
        raise


def risk_vectors(kl, identity_risk):
    """Independent units are data sets; repeat means precede outer statistics."""
    kl = np.asarray(kl, float)
    assert kl.shape == (16, 2, 4)
    arm = {name: ARMS.index(name) for name in ARMS}
    values = []
    for a, b in RISK_CONTRASTS:
        rows = []
        for d in range(16):
            left = math.fsum(kl[d, r, arm[a]] for r in range(2)) / 2
            right = identity_risk if b == 'I' else math.fsum(kl[d, r, arm[b]] for r in range(2)) / 2
            rows.append(left - right)
        values.append(rows)
    return np.asarray(values)


def eval_vectors(means, data, local_indices):
    """8 MC groups and2 repeats averaged inside each of16 outer data units."""
    means = np.asarray(means, float)
    assert means.shape == (16, 8, 2, 10, 16)
    p = data['population_weights']
    D = np.zeros((2, 2, 16)); P = D.copy()
    for ci, (a, b) in enumerate(SCORE_CONTRASTS):
        for profile, d in itertools.product(range(2), range(16)):
            differences = []
            for g, r in itertools.product(range(8), range(2)):
                ia = local_indices[d][r][ARMS.index(a)]
                ib = local_indices[d][r][ARMS.index(b)]
                differences.append(means[d, g, profile, ia] - means[d, g, profile, ib])
            averaged = np.array([math.fsum(x[yi] for x in differences)/16 for yi in range(16)])
            D[ci, profile, d] = math.fsum(w*x for w, x in zip(data['data'][d]['weights'], averaged))
            P[ci, profile, d] = math.fsum(w*x for w, x in zip(p, averaged))
    return D, P


def summary(values, family_size, threshold, expected=None):
    mean, se = moments(values)
    critical = float(student.isf(.025 / (2 * family_size), len(values) - 1))
    lo, hi = mean-critical*se, mean+critical*se
    return dict(values=np.asarray(values).tolist(), mean=mean, SE=se, critical_t=critical,
                lower=lo, upper=hi, df=len(values)-1, alpha_family=.025,
                family_size=family_size, SE_threshold=threshold,
                direction='lower' if hi < 0 else 'higher' if lo > 0 else 'unresolved',
                resolved=lo > 0 or hi < 0, precision_passed=se <= threshold,
                expected=expected, expected_supported=hi < 0 if expected == 'lower' else None)


def verify_summary(record, values, threshold, family_size, expected=None):
    own = summary(values, family_size, threshold, expected)
    for key in ('values', 'mean', 'SE', 'critical_t', 'lower', 'upper'):
        near(record[key], own[key])
    for key in ('df', 'alpha_family', 'family_size', 'SE_threshold', 'direction',
                'resolved', 'precision_passed', 'expected', 'expected_supported'):
        assert record[key] == own[key], key


def eval_scores(active, noise, rows, mu):
    B, K, _ = active.shape
    values = np.empty((B, 2, len(rows), 16))
    nuisance = -np.logaddexp(0., noise).sum(axis=-1)
    for j, row in enumerate(rows):
        x = active @ np.linalg.cholesky(np.array(row['covariance'])).T + mu
        for yi, label in enumerate(BITS):
            la = -np.logaddexp(0., -(2*label-1)*x).sum(axis=-1)
            values[:, 0, j, yi] = math.log(K) - logsumexp(la+nuisance, axis=1)
            values[:, 1, j, yi] = 256*math.log(2.) + math.log(K) - logsumexp(la, axis=1)
    return values


def numerical_rows(rows):
    banks = []
    for record in rows:
        per_bank = record['per_bank_controls']
        for bank in per_bank:
            loss = bank['native_loss']-bank['stable64_loss']
            gradient = np.array(bank['native_gradient'])-bank['stable64_gradient']
            near(loss, bank['loss_difference']); near(gradient, bank['gradient_difference'])
            near(np.max(abs(gradient)), bank['maximum_gradient_difference'])
            assert bank['loss_passed'] == (abs(loss) <= 1e-4)
            assert bank['gradient_passed'] == (np.max(abs(gradient)) <= 2e-5)
            banks.append(bank)
        near(seqmean([b['native_loss'] for b in per_bank]), record['native_loss'], 0.)
        near(seqmean([b['native_gradient'] for b in per_bank]), record['native_gradient'], 0.)
        near(math.fsum(b['stable64_loss'] for b in per_bank)/len(per_bank), record['stable64_loss'])
        near(np.mean([b['stable64_gradient'] for b in per_bank], axis=0), record['stable64_gradient'])
        loss = record['native_loss']-record['stable64_loss']
        gradient = np.array(record['native_gradient'])-record['stable64_gradient']
        near(loss, record['loss_difference']); near(gradient, record['gradient_difference'])
        near(np.max(abs(gradient)), record['maximum_gradient_difference'])
        assert record['loss_passed'] == (abs(loss) <= 1e-4)
        assert record['gradient_passed'] == (np.max(abs(gradient)) <= 2e-5)
        assert record['passed'] == (record['loss_passed'] and record['gradient_passed'])
    return dict(aggregate_checks=len(rows), aggregate_passed=sum(r['passed'] for r in rows),
        aggregate_loss_passed=sum(r['loss_passed'] for r in rows),
        aggregate_gradient_passed=sum(r['gradient_passed'] for r in rows), bank_checks=len(banks),
        bank_loss_passed=sum(b['loss_passed'] for b in banks),
        bank_gradient_passed=sum(b['gradient_passed'] for b in banks),
        stable64_calls=sum(r['stable64_calls'] for r in rows),
        maximum_aggregate_loss_difference=max(abs(r['loss_difference']) for r in rows),
        maximum_aggregate_gradient_difference=max(r['maximum_gradient_difference'] for r in rows),
        maximum_bank_loss_difference=max(abs(b['loss_difference']) for b in banks),
        maximum_bank_gradient_difference=max(b['maximum_gradient_difference'] for b in banks),
        passed=bool(all(r['passed'] for r in rows) and all(b['loss_passed'] and b['gradient_passed'] for b in banks)))


def independent_pmf(task):
    index, row, mu, expected = task
    q = 36
    nodes, weights = np.polynomial.hermite.hermgauss(q)
    nodes = np.sqrt(2.)*nodes; weights = weights/math.sqrt(math.pi)
    eig, U = np.linalg.eigh(np.array(row['covariance']))
    root = (U*np.sqrt(eig))@U.T
    total = np.zeros(16, np.longdouble)
    for start in range(0, q**4, 65536):
        sequence = np.arange(start, min(start+65536, q**4))
        ids = np.column_stack([(sequence//q**p) % q for p in (3, 2, 1, 0)])
        probabilities = expit(nodes[ids]@root.T + mu); w = weights[ids].prod(axis=1)
        for yi, y in enumerate(BITS):
            total[yi] += np.sum(w*np.where(y, probabilities, 1-probabilities).prod(axis=1), dtype=np.longdouble)
    values = total.astype(float)
    error = float(np.max(abs(values-np.array(expected))))
    result = dict(index=index, pmf=values.tolist(), maximum_difference=error,
        normalization_error=abs(math.fsum(values)-1), order=36, root='symmetric',
        nodes=q**4, passed=error <= TOL['reference_PMF'] and abs(math.fsum(values)-1) <= 2e-12)
    save(OUT / 'REFERENCE' / f'candidate_{index:03}.json', result)
    return result


def training_totals(train):
    records=train['records']
    assert [(r['data_id'],r['repeat']) for r in records]==list(itertools.product(range(16),range(2)))
    expected_API=dict(IN4=65536,OUT4=65536,SEQ4=65536,FACT=16384)
    expected_update=dict(IN4=16384,OUT4=65536,SEQ4=65536,FACT=16384)
    for strategy in ARMS:
        for key,expected in [('original_API_calls',expected_API[strategy]),
                             ('optimizer_steps',expected_update[strategy]),('repair_calls',expected_update[strategy])]:
            assert sum(r['strategy_accounting'][strategy][key] for r in records)==expected
        for key,value in train['strategy_timing_totals'][strategy].items():
            near(math.fsum(r['strategy_accounting'][strategy][key] for r in records),value,2e-8)
    accounting=train['accounting']
    expected=dict(new_datasets=16,new_observations=65536,data_Gaussian_scalars=262144,data_Uniform_scalars=17039360,
        prior_repeats_per_dataset=2,independent_prior_quartets=32,controlled_optimization_programs=224,
        main_strategy_TAIL_outputs=128,main_strategy_LAST_descriptive_outputs=128,
        private_program_TAIL_outputs=224,private_program_LAST_outputs=224,derived_OUT4_TAIL_outputs=32,
        independent_observation_banks=4194304,unique_prior_vectors=419430400,
        prior_Gaussian_scalars=109051904000,active_prior_Gaussian_scalars=1677721600,
        noise_prior_Gaussian_scalars=107374182400,shared_quartet_observation_presentations=1048576,
        score_call_observation_presentations=13631488,optimizer_call_observation_presentations=10485760,
        original_training_API_calls=212992,optimizer_calls=163840,repair_calls=163840,
        strategy_original_API_calls=expected_API,strategy_optimizer_calls=expected_update,
        strategy_repair_calls=expected_update,aggregate_numeric_controls=1280,stable64_calls=1664,
        new_neural_network_fits=0,new_model_integrals=0,scientific_retries=0)
    assert accounting==expected
    return dict(all_counts_checked=True,new_datasets=16,new_observations=65536,
                optimization_programs=224,score_calls=212992,optimizer_calls=163840,
                repair_calls=163840,paired_unique_inputs_not_double_counted=True)


def check_candidates(train,candidates,data):
    rows=candidates['candidates']
    assert [row['index'] for row in rows]==list(range(130))
    near(rows[0]['rho'],data['truth_rho'],0.)
    near(rows[1]['rho'],np.zeros(6),0.)
    old_truth=read(ROOT/'artifacts/scbm_invariance/REFERENCE/MODEL.json')
    truth_pmf=next(row['main_pmf'] for row in old_truth['records'] if row['t']==0)
    identity=read(ROOT/'artifacts/scbm_multicorrelation/REFERENCE/RESULTS.json')['identity_reference']
    assert rows[0]['kind']==rows[1]['kind']=='old_reference'
    assert rows[0]['reused_pmf']==truth_pmf and rows[1]['reused_pmf']==identity['main_pmf']
    assert [(r['data_id'],r['repeat']) for r in train['records']]==list(itertools.product(range(16),range(2)))
    for record in train['records']:
        d,r=record['data_id'],record['repeat']
        for ai,arm in enumerate(ARMS):
            row=rows[2+8*d+4*r+ai]
            assert (row['data_id'],row['repeat'],row['arm'],row['kind'])==(d,r,arm,'TAIL')
            near(row['rho'],record['outputs'][arm]['TAIL']['rho'],0.)
    for row in rows:
        near(cov(row['rho']),row['covariance'],0.)
        assert np.linalg.eigvalsh(cov(row['rho']))[0]>=.09999
    return rows


def eval_native(active,noise,values,controls,rows,mu):
    checks=[]
    assert len(controls['checks'])==20
    assert {(item['candidate_index'],item['profile']) for item in controls['checks']}==set(itertools.product(range(10),('raw','FACT')))
    for item in controls['checks']:
        local=item['candidate_index']; ci=rows[local]['index']
        assert item['global_candidate_index']==ci
        profile=item['profile']; pi=['raw','FACT'].index(profile)
        C=torch.tensor(rows[local]['covariance'],dtype=torch.float32)
        x=torch.tensor(active,dtype=torch.float32)@torch.linalg.cholesky(C).T+torch.tensor(mu,dtype=torch.float32)
        logits=torch.cat([x,torch.tensor(noise,dtype=torch.float32)],dim=2) if pi==0 else x
        prob=torch.sigmoid(logits.permute(0,2,1)); actual=[]
        with torch.no_grad():
            for label in BITS:
                labels=np.tile(label,(2,1))
                if pi==0:
                    labels=np.column_stack([labels,np.zeros((2,256),dtype=np.int64)])
                loss=core.official().compute_concept_loss(prob,torch.tensor(labels,dtype=torch.int64))
                if pi==1:
                    loss=loss+256*math.log(2.)
                actual.append(float(loss))
        error=near(actual,item['native_total_score'],2e-6)
        mean=values[:,pi,local,:].mean(axis=0)
        near(mean,item['mathematical_total_score'])
        near(np.array(actual)-mean,item['difference'],2e-6)
        near(max(abs(x) for x in item['difference']),item['maximum_absolute_difference'])
        assert item['tolerance']==1e-4 and item['official_API_calls']==16
        assert item['passed']==(max(abs(x) for x in item['difference'])<=1e-4)
        checks.append(dict(candidate=ci,profile=profile,maximum_native_error=error))
    assert len(checks)*16==320
    assert controls['official_API_calls']==320 and controls['pattern_means_checked']==320
    near(controls['maximum_absolute_difference'],max(item['maximum_absolute_difference'] for item in controls['checks']))
    assert controls['passed']==all(item['passed'] for item in controls['checks'])
    assert controls['tolerance']==1e-4
    return dict(calls=320,checks=checks)


def audit_eval(evaluation,rows,data,cfg):
    maximum=mean_error=0.; native=[]; results=[]
    groups=evaluation['groups']
    assert [(record['data_id'],record['group']) for record in groups]==list(itertools.product(range(16),range(8)))
    for record in groups:
        d,g=record['data_id'],record['group']; indices=[0,1]+list(range(2+8*d,10+8*d))
        assert record['candidate_indices']==indices
        assert record['consumed_banks']==512 and record['completed_blocks']==8
        assert record['profile_order']==['raw','FACT']
        local=[rows[index] for index in indices]
        assert sha(ROOT/record['prefix_npz'])==record['prefix_sha256']
        with np.load(ROOT/record['prefix_npz'],allow_pickle=False) as archive:
            z={key:archive[key] for key in archive.files}
        for key,value in z.items():
            assert ah(value)==record['prefix_array_sha256'][key]
        for role,dim in [('active',4),('noise',256)]:
            seed=[cfg['seed_base'],d,g,cfg['roles'][role]]
            assert record['role_seed_sequences'][role]==seed
            rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
            assert rng.bit_generator.state==record['RNG_initial_states'][role]
            digest=hashlib.sha256()
            for block in range(8):
                value=rng.standard_normal((64,100,dim),dtype=np.float64)
                digest.update(value.tobytes(order='C'))
                if block==0:
                    assert np.array_equal(value[:2],z[role+'_eps'])
            assert digest.hexdigest()==record['role_input_sha256'][role]
            assert rng.bit_generator.state==record['RNG_final_states'][role]
        value=eval_scores(z['active_eps'],z['noise_eps'],local,np.array(data['mu']))
        assert value.shape==(2,2,10,16) and z['block_means'].shape==(8,2,10,16)
        maximum=max(maximum,near(value,z['individual_total_score'],TOL['eval_score']))
        means=np.array([math.fsum(v)/8 for v in z['block_means'].reshape(8,-1).T]).reshape(2,10,16)
        mean_error=max(mean_error,near(means,record['means']))
        if d in [0,15] and g==0:
            native.append(dict(data_id=d,group=g,**eval_native(z['active_eps'],z['noise_eps'],value,record['numerical_controls'],local,np.array(data['mu']))))
        else:
            assert record['numerical_controls'] is None
        result=dict(passed=True,data_id=d,group=g,prefix_sha256=record['prefix_sha256'],score_cells=640)
        save(OUT/'EVAL'/f'd{d:02}_group_{g:02}.json',result); results.append(result)
    assert len(native)==2 and sum(n['calls'] for n in native)==640
    controls=[dict(data_id=row['data_id'],group=row['group'],**row['numerical_controls'])
              for row in groups if row['numerical_controls'] is not None]
    qualification=dict(records=controls,checks=[c for q in controls for c in q['checks']],
        official_API_calls=sum(q['official_API_calls'] for q in controls),
        pattern_means_checked=sum(q['pattern_means_checked'] for q in controls),
        maximum_absolute_difference=max(q['maximum_absolute_difference'] for q in controls),
        passed=all(q['passed'] for q in controls),tolerance=1e-4)
    assert evaluation['numerical_qualification']==qualification
    assert evaluation['profile_order']==['raw','FACT']
    assert evaluation['accounting']==dict(data_sets=16,groups=128,groups_per_data=8,
        independent_observation_banks=65536,prior_vectors=6553600,Gaussian_scalars=1703936000,
        mathematical_bank_pattern_scores=20971520,original_API_calls=640,new_observations=0,
        new_training=0,new_model_integrals=0,new_neural_network_fits=0,automatic_retries=0)
    return dict(passed=True,groups=128,outer_datasets=16,groups_per_dataset=8,
        all_inputs_replayed=True,replayed_banks=65536,replayed_Gaussian_scalars=1703936000,
        prefix_score_cells=81920,maximum_prefix_error=maximum,maximum_block_mean_error=mean_error,
        native=native,native_calls=640,all_bottom_score_cells_not_replayed=True)


def reference_arithmetic(reference,data,rows):
    p=data['population_weights']; all_flags=[]; deltas=[]; nodes=0
    assert [row['index'] for row in reference['records']]==list(range(130))
    for record in reference['records']:
        index=record['index']; pmf=record['main_pmf']
        assert len(pmf)==16 and all(math.isfinite(value) and value>0 for value in pmf)
        near(record['reference_joint_KL'],math.fsum(w*math.log(w/v) for w,v in zip(p,pmf)))
        near(record['rho'],rows[index]['rho'],0.)
        near(record['reference_cross_entropy_P'],-math.fsum(w*math.log(v) for w,v in zip(p,pmf)))
        near(record['reference_cross_entropy_D'],[-math.fsum(w*math.log(v) for w,v in zip(d['weights'],pmf)) for d in data['data']])
        if index<2:
            assert record['PMF_reused'] is True and record['variants']==[] and pmf==rows[index]['reused_pmf']
            q=record['qualifications'][0]; normalization=abs(math.fsum(pmf)-1)
            near(normalization,q['normalization_error'])
            assert q['positive'] and q['passed']==(normalization<=2e-12)
        else:
            assert record['PMF_reused'] is False and [v['order'] for v in record['variants']]==[24,32,40]
            assert pmf==record['variants'][1]['pmf']
            for variant,qualification in zip(record['variants'],record['qualifications']):
                values=variant['pmf']; delta=max(abs(x-y) for x,y in zip(values,pmf)); normalization=abs(math.fsum(values)-1)
                positive=len(values)==16 and all(math.isfinite(x) and x>0 for x in values)
                near([delta,normalization],[qualification['maximum_change'],qualification['normalization_error']])
                assert qualification['positive']==positive
                assert qualification['passed']==(positive and normalization<=2e-12 and delta<=5e-8)
                assert variant['gaussian_nodes']==variant['order']**4
                nodes+=variant['gaussian_nodes']; deltas.append(delta)
                near(variant['covariance'],rows[index]['covariance'],0.)
        assert record['passed']==all(q['passed'] for q in record['qualifications'])
        all_flags.append(record['passed'])
    assert reference['qualification_passed']==all(all_flags)
    near(reference['maximum_PMF_change'],max(deltas))
    expected_nodes=128*sum(q**4 for q in (24,32,40))
    assert nodes==reference['accounting']['Gaussian_nodes']==expected_nodes
    assert reference['accounting']==dict(candidates=130,reused_PMFs=2,new_candidates=128,
        PMF_integrations=384,scalar_pattern_integrals=6144,Gaussian_nodes=expected_nodes,
        duplicate_candidates_skipped=0,new_truth_or_identity_integrals=0,new_random_draws=0,
        new_observations=0,new_fits=0,additional_orders_or_retries=0,LAST_integrations=0,
        member_integrations=0,dataset_arm_mean_integrations=0)
    near(reference['records'][1]['reference_joint_KL'],data['identity_risk'],2e-13)
    return dict(candidates=130,risks_reconstructed=130,production_PMFs=384,production_nodes=nodes,
                qualification_flags_independently_rebuilt=True)


def statistics(result,train,evaluation,reference,data):
    rho=np.array([[record['outputs'][arm]['TAIL']['rho'] for arm in ARMS] for record in train['records']]).reshape(16,2,4,6)
    members=np.array([[record['members'][member]['TAIL']['rho'] for member in MEMBERS] for record in train['records']]).reshape(16,2,4,6)
    kl=np.array([record['reference_joint_KL'] for record in reference['records'][2:]]).reshape(16,2,4)
    vectors=risk_vectors(kl,data['identity_risk'])
    assert len(result['training_primary'])==5 and len(result['evaluation_primary'])==2
    for index,values in enumerate(vectors):
        verify_summary(result['training_primary'][index],values,.0005 if index<2 else .0002,5)
    near(result['rho_by_data_repeat_arm'],rho)
    near(result['member_rho_by_data_repeat_member'],members)
    near(result['KL_by_data_repeat_arm'],kl)
    near(members.mean(axis=2),rho[:,:,1],2e-15)
    near(result['mean_KL_by_arm'],[math.fsum(kl[:,:,a].ravel())/32 for a in range(4)])
    means=np.array([record['means'] for record in evaluation['groups']]).reshape(16,8,2,10,16)
    local=[[[2+4*r+a for a in range(4)] for r in range(2)] for d in range(16)]
    D,P=eval_vectors(means,data,local)
    for index,values in enumerate(P[:,0]):
        verify_summary(result['evaluation_primary'][index],values,.0005,2)
    near(result['risk_outer_values'],vectors.T)
    near(result['raw_P_outer_values'],P[:,0].T)
    near(result['identity_reference_risk'],reference['records'][1]['reference_joint_KL'])
    # Rebuild score aggregation with explicit loops and fsum, not producer einsum.
    Ps=np.zeros((16,8,2,4)); Ds=Ps.copy()
    Pbase=np.zeros((16,8,2,2)); Dbase=Pbase.copy()
    for d,g,profile in itertools.product(range(16),range(8),range(2)):
        for a in range(4):
            for output,weights in ((Ps,data['population_weights']),(Ds,data['data'][d]['weights'])):
                output[d,g,profile,a]=math.fsum(weights[y]*means[d,g,profile,2+4*r+a,y]
                    for r,y in itertools.product(range(2),range(16)))/2
        for model in range(2):
            for output,weights in ((Pbase,data['population_weights']),(Dbase,data['data'][d]['weights'])):
                output[d,g,profile,model]=math.fsum(weights[y]*means[d,g,profile,model,y] for y in range(16))
    Pm=Ps.mean(axis=1); Dm=Ds.mean(axis=1)
    score_group=np.stack([Ps[:,:,0,ARMS.index(a)]-Ps[:,:,0,ARMS.index(b)] for a,b in SCORE_CONTRASTS],axis=-1)
    near(result['raw_P_group_values'],score_group)
    se=np.array([[moments(score_group[d,:,j])[1] for j in range(2)] for d in range(16)])
    near(result['within_data_MC_SE'],se)
    near(result['population_score_by_data_profile_arm'],Pm)
    near(result['data_score_by_data_profile_arm'],Dm)
    near(result['population_baseline_score_by_data_profile_model'],Pbase.mean(axis=1))
    near(result['data_baseline_score_by_data_profile_model'],Dbase.mean(axis=1))
    entropy=-math.fsum(w*math.log(w) for w in data['population_weights'])+256*math.log(2.)
    near(result['entropy_reference'],entropy)
    gap=Pm[:,0,:]-entropy-kl.mean(axis=1)
    near(result['raw_J_P_by_data_arm'],gap)
    assert len(result['gap_derived'])==2
    for j,record in enumerate(result['gap_derived']):
        values=P[j,0]-vectors[j]
        a,b=SCORE_CONTRASTS[j]
        near(values,gap[:,ARMS.index(a)]-gap[:,ARMS.index(b)])
        risk=summary(vectors[j],5,.0005); score=summary(P[j,0],2,.0005)
        near(record['values'],values)
        near([record['mean'],record['lower'],record['upper']],
             [math.fsum(values)/16,score['lower']-risk['upper'],score['upper']-risk['lower']])
        assert record['source_primary_indices']==dict(evaluation=j,training=j)
        assert 'SE' not in record and 'family_size' not in record
    assert len(result['parameter_descriptive'])==64
    truth=np.array(data['truth_rho']); maximum_identity_error=0.
    for d,a in itertools.product(range(16),range(4)):
        record=result['parameter_descriptive'][4*d+a]
        assert (record['data_id'],record['arm'])==(d,ARMS[a])
        x,y=rho[d,0,a]-truth,rho[d,1,a]-truth
        half=math.fsum((u-v)**2 for u,v in zip(x,y))/2
        cross=math.fsum(u*v for u,v in zip(x,y))
        mse=(math.fsum(u*u for u in x)+math.fsum(v*v for v in y))/2
        maximum_identity_error=max(maximum_identity_error,abs(half+cross-mse))
        near(record['mean_rho'],rho[d,:,a].mean(axis=0))
        near([record[k] for k in ['mean_KL','half_squared_distance','cross_product_error','parameter_MSE',
                                 'raw_P','raw_D','FACT_P','FACT_D','raw_J_P']],
             [math.fsum(kl[d,:,a])/2,half,cross,mse,Pm[d,0,a],Dm[d,0,a],Pm[d,1,a],Dm[d,1,a],gap[d,a]])
    near(result['parameter_identity_max_error'],maximum_identity_error,2e-14)
    assert maximum_identity_error<1e-12
    assert result['processing_accounting']==dict(new_RNG=0,new_model_integrals=0,new_training=0)
    assert result['reference_qualification']==reference['qualification_passed']
    assert result['training_numerical_qualification']==train['numerical_qualification']
    assert result['evaluation_numerical_qualification']==evaluation['numerical_qualification']
    return dict(passed=True,training_units=16,training_repeats_averaged=2,training_df=15,
        training_family_alpha=.025,training_family_size=5,evaluation_units=16,
        evaluation_groups_averaged=8,evaluation_repeats_averaged=2,evaluation_df=15,
        evaluation_family_alpha=.025,evaluation_family_size=2,primary_families=2,
        primary_contrasts=7,all_randomness_outer_units=True,parameters_D_FACT_descriptive=True,
        parameter_descriptive_rows=64,derived_gap_intervals=2,derived_gap_new_tests=0,
        parameter_identity_error=maximum_identity_error,new_tests=0)


def run():
    verify_freeze()
    test=read(OUT/'SELFTEST.json')
    assert test['passed'] and test['source_sha256']==sha(SOURCE)
    assert not (OUT/'EXECUTION.json').exists() and not (OUT/'RESULTS.json').exists()
    train,reference,evaluation,result,candidates=[read(BASE/path) for path in
        ['TRAIN/RESULTS.json','REFERENCE/RESULTS.json','EVAL/RESULTS.json','RESULTS.json','CANDIDATES.json']]
    for obj,name in [(train,'run'),(reference,'reference'),(evaluation,'eval'),(result,'results')]:
        assert obj['source_sha256']==sha(ROOT/f'script/scbmseq_{name}.py')
        assert obj['freeze_sha256']==sha(BASE/'FREEZE.json')
        assert obj['data_freeze_sha256']==sha(BASE/'DATA_FREEZE.json')
    for obj in [reference,evaluation,result]:
        assert obj['derived_freeze_sha256']==sha(BASE/'DERIVED_FREEZE.json')
    for obj in [reference,evaluation]:
        assert obj['candidates_sha256']==sha(BASE/'CANDIDATES.json')
    assert reference['training_results_sha256']==sha(BASE/'TRAIN/RESULTS.json')
    assert train['kernel_sha256']==sha(ROOT/'script/scbmseq_kernel.py')
    files=[BASE/path for path in ['FREEZE.json','DATA_FREEZE.json','DERIVED_FREEZE.json',
        'CONFIG.json','SPEC.json','DATA.json','TRAIN/RESULTS.json','REFERENCE/RESULTS.json',
        'EVAL/RESULTS.json','RESULTS.json','CANDIDATES.json']]
    inputs={rel(path):sha(path) for path in files}
    for row in train['records']:
        inputs[row['input_file']]=row['input_sha256']; inputs[row['output_file']]=row['output_sha256']
    for row in evaluation['groups']:
        inputs[row['prefix_npz']]=row['prefix_sha256']
    save(OUT/'EXECUTION.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(SOURCE),
        freeze_sha256=sha(BASE/'FREEZE.json'),data_freeze_sha256=sha(BASE/'DATA_FREEZE.json'),
        derived_freeze_sha256=sha(BASE/'DERIVED_FREEZE.json'),input_files=inputs,scope=SCOPE,automatic_retry=False))
    start=time.perf_counter(); completed=[]
    try:
        cfg,data=data_check()
        rows=check_candidates(train,candidates,data)
        with ProcessPoolExecutor(max_workers=8) as pool:
            futures=[pool.submit(training_pair,row) for row in train['records']]
            try:
                for future in as_completed(futures):
                    row=future.result(); completed.append(row)
                    save(OUT/'PROGRESS.json',dict(completed_training_pairs=sorted((r['data_id'],r['repeat']) for r in completed),
                        seconds=time.perf_counter()-start),False)
                    print(json.dumps(dict(audited_data_id=row['data_id'],repeat=row['repeat'],seconds=row['seconds'])),flush=True)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        completed.sort(key=lambda row:(row['data_id'],row['repeat']))
        controls=[control for row in train['records'] for control in row['numerical_controls']]
        assert numerical_rows(controls)==train['numerical_qualification']
        native=[check for row in completed for check in row['native_checks']]
        assert len(native)==208
        assert {(r['data_id'],r['repeat'],r['step']) for r in native}==set(itertools.product(BOTTOM_DATA,[0],STEPS))
        assert len({(r['data_id'],r['repeat'],r['step'],r['program'],r['microstep']) for r in native})==160
        totals=training_totals(train)
        ev=audit_eval(evaluation,rows,data,cfg['evaluation'])
        arithmetic=reference_arithmetic(reference,data,rows)
        reference_indices=[2+8*d+ai for d in [0,4,8,12] for ai in range(4)]
        tasks=[(index,rows[index],np.array(data['mu']),reference['records'][index]['main_pmf']) for index in reference_indices]
        with ProcessPoolExecutor(max_workers=4) as pool:
            reference_checks=list(pool.map(independent_pmf,tasks))
        stat=statistics(result,train,evaluation,reference,data)
        verify_freeze()
        for path,digest in inputs.items():
            assert sha(ROOT/path)==digest,path
        passed=(all(record['passed'] for record in reference_checks) and reference['qualification_passed'] and
                train['numerical_qualification']['passed'] and evaluation['numerical_qualification']['passed'])
        output=dict(passed=passed,remaining_required_fixes=[] if passed else ['A prescribed numerical qualification failed; no automatic refinement.'],
            source_sha256=sha(SOURCE),freeze_sha256=sha(BASE/'FREEZE.json'),data_freeze_sha256=sha(BASE/'DATA_FREEZE.json'),
            derived_freeze_sha256=sha(BASE/'DERIVED_FREEZE.json'),input_files=inputs,
            data_audit=read(OUT/'DATA.json'),
            training_audit=dict(pairs=32,RNG_roles=256,replayed_Gaussian_scalars=109051904000,
                all_hashes_states_match=True,all_saved_averages_Adam_and_repairs_checked=True,
                all_SEQ4_microstates_and_private_moments_checked=True,SEQ4_microsteps=65536,
                local_proposal_max_error=max(r['local_proposal_max_error'] for r in completed),
                repair_max_error=max(r['repair_max_error'] for r in completed),
                moment_max_error=max(r['moment_max_error'] for r in completed),
                native_aggregate_states=160,native_bank_calls=208,native_checks=native,accounting=totals,
                native_loss_max_error=max(x['native_loss_error'] for x in native),
                native_gradient_max_error=max(x['native_gradient_error'] for x in native),
                math64_loss_max_error=max(x['math64_loss_error'] for x in native),
                math64_gradient_max_error=max(x['math64_gradient_error'] for x in native)),
            evaluation_audit=ev,reference_audit=dict(saved_arithmetic=arithmetic,independent_checks=reference_checks,
                independent_PMFs=16,independent_four_dimensional_nodes=26873856),statistics_audit=stat,
            seconds=time.perf_counter()-start,new_independent_scientific_draws=0,new_training=0,
            new_audit_model_integrals=dict(PMFs=16,pattern_integrals=256,four_dimensional_nodes=26873856),
            reused_independent_cores=dict(path=rel(Path(core.__file__)),sha256=sha(core.__file__),
                functions=['math_score','native','adam','repair']),
            scope='All16 new-data inputs, original prior inputs and saved arithmetic replayed; bounded native/math prefixes and all saved statistics. No independent full relearning or complete EVAL rescoring; independent GH36 qualification, not a rigorous integral certificate.')
        save(OUT/'RESULTS.json',output)
        print(json.dumps(dict(passed=passed,seconds=output['seconds'],results_sha256=sha(OUT/'RESULTS.json'))),flush=True)
    except BaseException:
        save(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),completed_training_pairs=
            [[r['data_id'],r['repeat']] for r in completed],seconds=time.perf_counter()-start,scope=SCOPE,
            automatic_retry=False,all_existing_outputs_retained=True))
        raise


def deterministic_selftest():
    assert not (BASE/'FREEZE.json').exists()
    assert not (OUT/'EXECUTION.json').exists()
    successes = []; failures = []
    for data_id, repeat in itertools.product(range(16), range(2)):
        job_id, success, error = job_paths(data_id, repeat)
        for step in STEPS:
            for program in PROGRAMS:
                pass
        assert program == 'FACT'
        assert job_id == f'd{data_id:02}_r{repeat:02}'
        successes.append(success.name); failures.append(error.name)
    assert len(set(successes)) == len(set(failures)) == 32
    assert set(successes).isdisjoint(failures)
    kl = np.empty((16, 2, 4))
    for d, r, a in itertools.product(range(16), range(2), range(4)):
        kl[d, r, a] = .002+(d+1)*(r+2)*(a+3)/17731 + .0003*math.sin((d+1)*(a+1))
    rv = risk_vectors(kl, .002711)
    for ci, (a, b) in enumerate(RISK_CONTRASTS):
        for d in range(16):
            literal = (kl[d, 0, ARMS.index(a)]+kl[d, 1, ARMS.index(a)])/2
            literal -= .002711 if b == 'I' else (kl[d, 0, ARMS.index(b)]+kl[d, 1, ARMS.index(b)])/2
            near(rv[ci, d], literal, 2e-17)
        own = summary(rv[ci], 5, .0005 if ci < 2 else .0002)
        assert own['df'] == 15 and own['family_size'] == 5
        near(own['SE'], np.std(rv[ci], ddof=1)/4, 2e-17)
    local = [[[2+4*r+a for a in range(4)] for r in range(2)] for d in range(16)]
    means = np.empty((16, 8, 2, 10, 16))
    for d, g, p, c, yi in itertools.product(range(16), range(8), range(2), range(10), range(16)):
        means[d, g, p, c, yi] = 180 + (d+1)*(g+2)*(c+3)*(yi+1)/17731 + p*.0002*c*c
    weights = np.arange(1, 17, dtype=float)/136
    data = dict(population_weights=weights.tolist(), data=[dict(weights=np.roll(weights, d).tolist()) for d in range(16)])
    D, P = eval_vectors(means, data, local)
    literal_error = 0.
    for ci, (a, b) in enumerate(SCORE_CONTRASTS):
        for profile, d in itertools.product(range(2), range(16)):
            for array, w in ((D, np.roll(weights, d)), (P, weights)):
                cells = [w[yi]*(means[d,g,profile,local[d][r][ARMS.index(a)],yi]-
                                    means[d,g,profile,local[d][r][ARMS.index(b)],yi])
                         for g,r,yi in itertools.product(range(8), range(2), range(16))]
                literal_error = max(literal_error, near(array[ci,profile,d], math.fsum(cells)/16, 2e-13))
    # Existing independent stable64 core receives deterministic trigonometric fixtures.
    eps = np.sin(np.arange(100*3*4)*.03).reshape(100,3,4)
    noise = np.cos(np.arange(100*3*256)*.007).reshape(100,3,256)
    y = (np.arange(12).reshape(3,4)%2).astype(np.int8)
    yn = (np.arange(768).reshape(3,256)%3 == 0).astype(np.int8)
    rho = np.array([.11,-.07,.05,.09,-.06,.13]); finite_difference_errors=[]
    for method in ('FIXED', 'FACT'):
        answer = core.math_score(method,rho,None,eps,noise,y,yn)
        for coordinate in range(6):
            plus=rho.copy(); minus=rho.copy(); plus[coordinate]+=1e-5; minus[coordinate]-=1e-5
            fd=(core.math_score(method,plus,None,eps,noise,y,yn)['loss']-
                core.math_score(method,minus,None,eps,noise,y,yn)['loss'])/2e-5
            finite_difference_errors.append(near(fd,answer['rho_gradient'][coordinate],2e-8))
    core_check = core.artificial()
    # Producer runner is imported only here for a deterministic integration
    # fixture/schema comparison; scientific run() does not import it.
    import importlib.util
    runner_spec=importlib.util.spec_from_file_location('_scbmseq_audit_fixture_runner',ROOT/'script/scbmseq_run.py')
    runner=importlib.util.module_from_spec(runner_spec); runner_spec.loader.exec_module(runner)
    fixture_tape=runner.allocate(steps=2)
    fixture_models,fixture_optimizers=runner.kernel.make_models()
    runner.snapshot(fixture_tape,fixture_models,fixture_optimizers,0)
    fixture_banks,fixture_y,fixture_yn=runner.kernel.deterministic_inputs(B=64,n=256)
    fixture_controls=[]
    for fixture_step in range(2):
        fixture_controls.extend(runner.advance(fixture_tape,fixture_models,fixture_optimizers,fixture_step,
            iter(fixture_banks),fixture_y,fixture_yn,64*fixture_step,-1,-1,True))
    runner.final_moments(fixture_tape,fixture_models,fixture_optimizers)
    fixture_trajectory=replay_trajectory(fixture_tape,steps=2)
    assert numerical_rows(fixture_controls)==runner.numerical_summary(fixture_controls)
    # This validator must reject a detached SEQ4 starting state even when the
    # rest of the tape still contains valid Adam arithmetic.
    corrupted={key:value.copy() for key,value in fixture_tape.items()}
    corrupted['seq_micro_rho'][1,0,0]+=np.float32(.001)
    rejected=False
    try:
        replay_trajectory(corrupted,steps=2)
    except AssertionError:
        rejected=True
    assert rejected
    # Explicit private SEQ4 state/moments are advanced four times per macro.
    parameter = torch.nn.Parameter(torch.zeros(6, dtype=torch.float32))
    optimizer = torch.optim.Adam([parameter], lr=.02, betas=(.9,.999), eps=1e-8, foreach=False, fused=False)
    m=np.zeros(6,np.float32); v=m.copy(); micro_errors=[]; visits=[]
    for macro in range(8):
        for bank_index in range(4):
            micro_index=4*macro+bank_index; before=parameter.detach().numpy().copy()
            gradient=(.03*np.sin((np.arange(6)+1)*(micro_index+1)*.17)+.01*before).astype(np.float32)
            parameter.grad=torch.tensor(gradient); optimizer.step()
            m,v,delta=core.adam(m,v,gradient,micro_index+1)
            micro_errors.append(near(before+delta,parameter.detach().numpy(),TOL['proposal']))
            visits.append((macro,bank_index,micro_index+1))
    assert int(optimizer.state[parameter]['step']) == 32
    near(m,optimizer.state[parameter]['exp_avg'].numpy(),TOL['moment_abs'])
    near(v,optimizer.state[parameter]['exp_avg_sq'].numpy(),TOL['moment_abs'])
    # Interval subtraction is a derived enclosure, never a new test/family.
    rs=summary(rv[0],5,.0005); ps=summary(P[0,0],2,.0005)
    gap=[ps['lower']-rs['upper'],ps['upper']-rs['lower']]
    assert gap[0] <= ps['mean']-rs['mean'] <= gap[1]
    result=dict(passed=True,source_sha256=sha(SOURCE), independent_helper_sha256=sha(core.__file__),
        official_source_sha256=sha(core.OFF), scope=SCOPE,
        checks=['32 unique result/error paths survive all program loops',
                '16 nonexchangeable outer datasets,2-repeat q5 risk averaging',
                'independent 16x8 evaluation pools and local 10-candidate q2 averaging',
                '12 deterministic six-coordinate finite differences',
                '32 ordered private sequential Adam microsteps and moments',
                'derived raw-score minus risk interval subtraction'],
        maximum_FD_error=max(finite_difference_errors), maximum_eval_fixture_error=literal_error,
        maximum_sequential_Adam_error=max(micro_errors), reused_core_check=core_check,
        deterministic_trace_integration=fixture_trajectory,deterministic_trace_macrosteps=2,
        deterministic_trace_native_calls=26,deterministic_trace_stable64_calls=26,
        detached_SEQ4_boundary_rejected=True,
        deterministic_comparison_source_sha256=sha(ROOT/'script/scbmseq_run.py'),
        task_filenames=successes,error_filenames=failures, private_micro_visits=visits,
        new_scientific_RNG=0,new_observations=0,new_model_integrals=0,new_training=0,
        source_selftest_precedes_ROOT_freeze=True)
    save(OUT/'SELFTEST.json', result)
    print(json.dumps(dict(passed=True,source_sha256=sha(SOURCE))),flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf8'); sys.stderr.reconfigure(encoding='utf8')
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['check','run'])
    args=parser.parse_args()
    if args.mode == 'check':
        deterministic_selftest()
    else:
        run()
