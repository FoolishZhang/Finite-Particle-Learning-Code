"""Independent-bank scores at 35 frozen covariance candidates.

Main scores use mathematical float64, not native API32. The latter is called
only in the fixed numerical qualification. No gradient, training, model
integration, candidate selection, or statistical testing is performed here.
"""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name]='1'
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import platform
import time
import traceback
import numpy as np
from scipy.special import expit, logsumexp
import torch
import scbminv_kernel as official

torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(__file__).resolve()
PACKAGE=ROOT/'artifacts/scbm_six_frozen_targets'
OUT=PACKAGE/'MC'
KERNEL=PACKAGE/'KERNEL'
ADAPTER_SHA='c6712abbb4e05ef0c25c688a05dad106034c41888040592424f50362f04129ad'
PATTERNS=((np.arange(16)[:,None]>>np.array([3,2,1,0]))&1).astype(np.int8)
PAIRS=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
PROFILES=['raw','FACT']


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array_sha(value): return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
def rel(path): return Path(path).resolve().relative_to(ROOT).as_posix()
def read(path): return json.loads(Path(path).read_text('utf8'))


def save(path,obj,exclusive=True):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x' if exclusive else 'w',encoding='utf8',newline='\n') as handle:
        json.dump(obj,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write('\n')


def verify(inputs):
    for name,digest in inputs.items():
        p=(ROOT/name).resolve()
        assert p.is_relative_to(ROOT) and sha(p)==digest,name


def environment():
    result=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,
                torch_threads=torch.get_num_threads(),device='cpu',
                thread_environment={k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')})
    try:
        from threadpoolctl import threadpool_info
        result['threadpool_info']=threadpool_info();result['actual_BLAS_threads_measured']=True
    except ImportError:
        result['threadpool_info']=None;result['actual_BLAS_threads_measured']=False
    return result


def provenance():
    assert sha(official.__file__)==ADAPTER_SHA
    return dict(source=rel(SOURCE),source_sha256=sha(SOURCE),official_adapter=rel(official.__file__),
                official_adapter_sha256=ADAPTER_SHA,official=official.provenance(),
                main_dtype='float64',main_official_API_calls=0,
                canonical_scope='All nuisance labels zero is justified only for fresh-prior expectation under independent fair nuisance coordinates. It does not reproduce old realized training noise labels or certify native32 sign symmetry.')


def load_contract():
    config=read(PACKAGE/'CONFIG.json');candidates=read(PACKAGE/'CANDIDATES.json')
    expected={'groups':32,'banks_per_group':1024,'K':100,'nuisance':256,'chunk':64,
              'seed_base':202610032100,'saved_prefix_banks':4,'numerical_tolerance':1e-4,
              'workers':2,'dtype':'float64','canonical_nuisance_label':0}
    assert all(config[k]==v for k,v in expected.items())
    assert config['roles']=={'active':0,'noise':1} and config['profiles']==PROFILES
    assert config['native_control_candidate_indices']==[0,1,2,18,17,33]
    assert candidates['pair_order']==[list(p) for p in PAIRS]
    assert candidates['patterns']==PATTERNS.tolist()
    assert candidates['mean']==candidates['mu']==[-.6,-.2,.2,.6]
    rows=candidates['candidates'];assert len(rows)==35
    assert [r['index'] for r in rows]==list(range(35))
    names=['truth','high']+[f'FIXED_TAIL_{i:02}' for i in range(16)]+[f'FACT_TAIL_{i:02}' for i in range(16)]+['MEAN_FIXED']
    assert [r['name'] for r in rows]==names
    matrices=np.array([r['covariance'] for r in rows],dtype=np.float64)
    assert matrices.shape==(35,4,4) and np.isfinite(matrices).all()
    assert np.array_equal(matrices,matrices.transpose(0,2,1))
    assert np.array_equal(np.diagonal(matrices,axis1=1,axis2=2),np.ones((35,4)))
    for row,cov in zip(rows,matrices):
        assert np.array_equal(np.array(row['rho']),np.array([cov[i,j] for i,j in PAIRS]))
    assert np.linalg.eigvalsh(matrices).min()>=.09999
    lowers=np.linalg.cholesky(matrices)
    return config,candidates,matrices,lowers


def active_log_probabilities(active_eps,lower,mu):
    """[bank,K,16] log products for the original bit-pattern ordering."""
    x=np.asarray(active_eps,dtype=np.float64)@np.asarray(lower,dtype=np.float64).T+np.asarray(mu,dtype=np.float64)
    values=x@PATTERNS.T-np.logaddexp(0.,x).sum(axis=2)[:,:,None]
    return values,x


def evaluate_block(active_eps,noise_eps,lowers,mu):
    """Return total_score[bank,profile,candidate,pattern], never pool banks.

    raw: -log(mean_k(active_likelihood * nuisance_likelihood)).
    FACT: n*log2 -log(mean_k(active_likelihood)).
    One shared base input is transformed by each candidate; no new draw occurs.
    """
    active=np.asarray(active_eps,dtype=np.float64);noise=np.asarray(noise_eps,dtype=np.float64)
    lowers=np.asarray(lowers,dtype=np.float64);mu=np.asarray(mu,dtype=np.float64)
    assert active.ndim==noise.ndim==3 and active.shape[2]==4 and noise.shape[:2]==active.shape[:2]
    assert lowers.ndim==3 and lowers.shape[1:]==(4,4) and mu.shape==(4,)
    assert np.isfinite(active).all() and np.isfinite(noise).all() and np.isfinite(lowers).all()
    B,K,_=active.shape;n=noise.shape[2]
    noise_log=-np.logaddexp(0.,noise).sum(axis=2)
    output=np.empty((B,2,len(lowers),16),dtype=np.float64)
    max_active=0.
    for ci,lower in enumerate(lowers):
        logactive,x=active_log_probabilities(active,lower,mu)
        output[:,0,ci,:]=math.log(K)-logsumexp(logactive+noise_log[:,:,None],axis=1)
        output[:,1,ci,:]=n*math.log(2.)+math.log(K)-logsumexp(logactive,axis=1)
        max_active=max(max_active,float(np.max(abs(x))))
    assert np.isfinite(output).all()
    return output,dict(maximum_absolute_active_logit=max_active,
                       maximum_absolute_noise_logit=float(np.max(abs(noise))) if noise.size else 0.,
                       finite=True,inner_reduction_axis='K within each bank',output_shape=list(output.shape))


def official_pattern_means(active_eps,noise_eps,covariance,mu,profile):
    """16 actual official API32 calls, one pattern per batch of B banks."""
    assert profile in PROFILES and sha(official.__file__)==ADAPTER_SHA
    eps=torch.tensor(np.asarray(active_eps),dtype=torch.float32)
    noise=torch.tensor(np.asarray(noise_eps),dtype=torch.float32)
    covariance=torch.tensor(np.asarray(covariance),dtype=torch.float32)
    mean=torch.tensor(np.asarray(mu),dtype=torch.float32)
    B,K,_=eps.shape;n=noise.shape[2]
    assert noise.shape[:2]==(B,K)
    with torch.no_grad():
        x=eps@torch.linalg.cholesky(covariance).T+mean
        logits=torch.cat((x,noise),dim=2) if profile=='raw' else x
        probabilities=torch.sigmoid(logits.permute(0,2,1))
        values=[]
        for pattern in PATTERNS:
            labels=torch.tensor(np.broadcast_to(pattern,(B,4)).copy(),dtype=torch.int64)
            if profile=='raw': labels=torch.cat((labels,torch.zeros((B,n),dtype=torch.int64)),dim=1)
            value=official.official_loss(K).compute_concept_loss(probabilities,labels)
            if profile=='FACT': value=value+n*math.log(2.)
            values.append(float(value))
    return np.asarray(values,dtype=np.float64),dict(
        official_API_calls=16,rows_per_call=B,K=K,profile=profile,
        probability_evaluations=16*probabilities.numel(),
        sigmoid_zero_evaluations=16*int((probabilities==0).sum()),
        sigmoid_one_evaluations=16*int((probabilities==1).sum()),
        maximum_absolute_logit=float(logits.abs().max()),
        scope='Actual original float32 API batch means; FACT uses active API plus nlog2 in float32. Differences include all input/covariance rounding and mathematical expression differences, not only BCE.')


def native_controls(active,noise,math_scores,matrices,mu,indices,tolerance):
    checks=[]
    for ci in indices:
        for pi,profile in enumerate(PROFILES):
            native,diagnostic=official_pattern_means(active,noise,matrices[ci],mu,profile)
            expected=math_scores[:,pi,ci,:].mean(axis=0)
            delta=native-expected
            checks.append(dict(candidate_index=int(ci),
                native_total_score=native.tolist(),mathematical_total_score=expected.tolist(),
                difference=delta.tolist(),maximum_absolute_difference=float(np.max(abs(delta))),
                tolerance=tolerance,passed=bool(np.max(abs(delta))<=tolerance),**diagnostic))
    return dict(checks=checks,official_API_calls=sum(c['official_API_calls'] for c in checks),
                pattern_means_checked=sum(len(c['difference']) for c in checks),
                maximum_absolute_difference=max(c['maximum_absolute_difference'] for c in checks),
                passed=all(c['passed'] for c in checks),tolerance=tolerance)


def deterministic_inputs(B,K,n):
    active=1.2*np.sin(np.arange(B*K*4,dtype=np.float64)*.071).reshape(B,K,4)
    noise=.95*np.cos(np.arange(B*K*n,dtype=np.float64)*.013).reshape(B,K,n)
    return active,noise


def selftest():
    assert not (PACKAGE/'FREEZE.json').exists(),'Do not rerun development after scientific freeze'
    config,candidates,matrices,lowers=load_contract();prov=provenance()
    mu=np.array(candidates['mu'])
    active,noise=deterministic_inputs(3,7,5)
    covs=matrices[[0,1,2,17,34]];L=np.linalg.cholesky(covs)
    calculated,_=evaluate_block(active,noise,L,mu)
    direct=np.empty_like(calculated)
    for b in range(3):
        nuisance=np.prod(expit(-noise[b]),axis=1)
        for ci,lower in enumerate(L):
            x=active[b]@lower.T+mu
            for yi,pattern in enumerate(PATTERNS):
                likelihood=np.prod(expit((2*pattern-1)*x),axis=1)
                direct[b,0,ci,yi]=-math.log(float(np.mean(likelihood*nuisance)))
                direct[b,1,ci,yi]=5*math.log(2.)-math.log(float(np.mean(likelihood)))
    direct_error=float(np.max(abs(direct-calculated)));assert direct_error<2e-14
    zero,_=evaluate_block(active,np.zeros_like(noise),L,mu)
    identity_error=float(np.max(abs(zero[:,0]-zero[:,1])));assert identity_error<2e-14
    absent,_=evaluate_block(active,noise[:,:,:0],L,mu)
    factor_error=float(np.max(abs(calculated[:,1]-absent[:,1]-5*math.log(2.))));assert factor_error<2e-14
    # Taking log only after pooling all particles from all banks is a different
    # estimator. This deliberately uneven deterministic fixture distinguishes it.
    pooled=calculated.copy()
    pool_value=-logsumexp(-pooled[:,0,0,0])+math.log(3)
    mean_value=float(pooled[:,0,0,0].mean())
    assert abs(pool_value-mean_value)>1e-8
    active_big,noise_big=deterministic_inputs(4,100,256)
    main,_=evaluate_block(active_big,noise_big,lowers,mu)
    controls=native_controls(active_big,noise_big,main,matrices,mu,
                             config['native_control_candidate_indices'],config['numerical_tolerance'])
    assert controls['passed'] and controls['official_API_calls']==192
    save(KERNEL/'SELFTEST.json',dict(passed=True,created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(SOURCE),config_sha256=sha(PACKAGE/'CONFIG.json'),candidates_sha256=sha(PACKAGE/'CANDIDATES.json'),
        provenance=prov,environment=environment(),direct_product_max_error=direct_error,
        zero_noise_factorization_max_error=identity_error,factorized_constant_max_error=factor_error,
        bank_mean_distinguished_from_pooled_particle_log=True,native_controls=controls,
        new_scientific_RNG=0,new_model_integrals=0,new_fits=0,
        scope='Fixed sin/cos tensors only, no convergence or statistical pilot. Direct tiny products independently verify all patterns, profiles, bank axes and candidate transforms.'))
    print(json.dumps(dict(passed=True,source_sha256=sha(SOURCE),direct_error=direct_error,
                         numerical_max_difference=controls['maximum_absolute_difference'],new_scientific_RNG=0)))


def benchmark():
    assert not (PACKAGE/'FREEZE.json').exists()
    config,candidates,_,lowers=load_contract()
    active,noise=deterministic_inputs(64,100,256)
    elapsed=[]
    for _ in range(3):
        start=time.perf_counter();values,_=evaluate_block(active,noise,lowers,candidates['mu']);elapsed.append(time.perf_counter()-start)
    blocks=config['groups']*config['banks_per_group']//config['chunk']
    save(KERNEL/'BENCHMARK.json',dict(source_sha256=sha(SOURCE),config_sha256=sha(PACKAGE/'CONFIG.json'),
        candidates_sha256=sha(PACKAGE/'CANDIDATES.json'),seconds_per_fixed_block=elapsed,
        production_shape=list(values.shape),production_blocks=blocks,
        serial_compute_projection_seconds=float(np.median(elapsed))*blocks,
        ideal_two_worker_compute_projection_seconds=float(np.median(elapsed))*blocks/2,
        environment=environment(),new_scientific_RNG=0,new_model_integrals=0,new_fits=0,
        scope='Three timings on the same deterministic full-shaped input, no target values printed or used for design. Projection excludes random generation, hashing, IO, native controls and parallel overhead.'))
    print(json.dumps(dict(seconds_per_block=elapsed,serial_projection_seconds=float(np.median(elapsed))*blocks,
                         ideal_two_worker_projection_seconds=float(np.median(elapsed))*blocks/2)))


def schema():
    config,candidates,_,_=load_contract()
    return dict(source_sha256=sha(SOURCE),profile_order=PROFILES,candidate_names=[r['name'] for r in candidates['candidates']],
        pattern_order=PATTERNS.tolist(),means='float64[2,35,16] total score including common nlog2',
        npz=dict(active_eps='float64[4,100,4]',noise_eps='float64[4,100,256]',
                 individual_total_score='float64[4,2,35,16]',block_means='float64[16,2,35,16]'),
        RNG=dict(bit_generator='PCG64',seed_sequence='[seed_base,group,role]',roles=config['roles'],
            generation_order='16 chunks; within each chunk active[64,100,4] then noise[64,100,256]; each role hashed independently in generation order, float64 C-order bytes',
            coupling='Same generated base inputs across all35 covariance candidates and both profiles. No independent counts are multiplied by reused candidates, patterns or profiles.'),
        output='MC/RESULTS.json groups[32] ordered by group; each contains means, prefix_npz/prefix_sha256 and complete input/RNG ledger',
        native_scope='Only group0 first4 banks,6 candidate indices times2 profiles times16 patterns =192 original API32 calls; no gradients or main API32 claim',
        new_budget=dict(independent_groups=32,independent_banks=32768,prior_vectors=3276800,Gaussian_scalars=851968000),
        no_P_or_D_weighting_here=True,no_statistical_tests_here=True)


def validate_freeze():
    assert not (PACKAGE/'VERSION_LOCK.json').exists(),'Scientific package sealed'
    frozen=read(PACKAGE/'FREEZE.json')
    assert frozen['run_authorized'] is True
    verify(frozen['inputs'])
    for p in (SOURCE,PACKAGE/'CONFIG.json',PACKAGE/'CANDIDATES.json'):
        assert frozen['inputs'][rel(p)]==sha(p)
    for filename in ('SELFTEST.json','BENCHMARK.json','SCHEDULING_SELFTEST.json'):
        test=read(KERNEL/filename)
        assert test['source_sha256']==sha(SOURCE)
        if 'passed' in test: assert test['passed']
    provenance();load_contract()
    return frozen


def draw_block(generators,B,K,n):
    """Scientific draws. Called only after the run's root-freeze validation."""
    return (generators['active'].standard_normal((B,K,4),dtype=np.float64),
            generators['noise'].standard_normal((B,K,n),dtype=np.float64))


def run_group(group):
    validate_freeze()
    config,candidates,matrices,lowers=load_contract()
    assert 0<=group<config['groups']
    record_path=OUT/f'group_{group:02}.json';npz=OUT/f'group_{group:02}_prefix.npz'
    assert not record_path.exists() and not npz.exists() and not (OUT/f'ERROR_group_{group:02}.json').exists()
    generators={name:np.random.Generator(np.random.PCG64(np.random.SeedSequence([config['seed_base'],group,role])))
                for name,role in config['roles'].items()}
    initial={k:g.bit_generator.state for k,g in generators.items()};digests={k:hashlib.sha256() for k in generators}
    consumed=0;blocks=[];prefix={};controls=None;diagnostics=[];start=time.perf_counter()
    try:
        for block in range(config['banks_per_group']//config['chunk']):
            active,noise=draw_block(generators,config['chunk'],config['K'],config['nuisance'])
            digests['active'].update(active.tobytes(order='C'));digests['noise'].update(noise.tobytes(order='C'))
            consumed+=len(active)
            values,diagnostic=evaluate_block(active,noise,lowers,candidates['mu'])
            blocks.append(values.mean(axis=0));diagnostics.append(diagnostic)
            if block==0:
                prefix={'active_eps':active[:4].copy(),'noise_eps':noise[:4].copy(),
                        'individual_total_score':values[:4].copy()}
                if group==0:
                    controls=native_controls(prefix['active_eps'],prefix['noise_eps'],prefix['individual_total_score'],
                        matrices,candidates['mu'],config['native_control_candidate_indices'],config['numerical_tolerance'])
        prefix['block_means']=np.stack(blocks)
        means=prefix['block_means'].mean(axis=0)
        assert means.shape==(2,35,16) and consumed==config['banks_per_group'] and np.isfinite(means).all()
        with npz.open('xb') as handle: np.savez_compressed(handle,**prefix)
        record=dict(group=group,source_sha256=sha(SOURCE),freeze_sha256=sha(PACKAGE/'FREEZE.json'),
            config_sha256=sha(PACKAGE/'CONFIG.json'),candidates_sha256=sha(PACKAGE/'CANDIDATES.json'),
            means=means.tolist(),prefix_npz=rel(npz),prefix_sha256=sha(npz),
            role_seed_sequences={name:[config['seed_base'],group,role] for name,role in config['roles'].items()},
            RNG_initial_states=initial,RNG_final_states={name:g.bit_generator.state for name,g in generators.items()},
            role_input_sha256={name:h.hexdigest() for name,h in digests.items()},
            prefix_array_sha256={name:array_sha(a) for name,a in prefix.items()},
            consumed_banks=consumed,completed_blocks=len(blocks),numerical_controls=controls,
            maximum_absolute_active_logit=max(d['maximum_absolute_active_logit'] for d in diagnostics),
            maximum_absolute_noise_logit=max(d['maximum_absolute_noise_logit'] for d in diagnostics),
            finite=True,environment=environment(),seconds=time.perf_counter()-start)
        validate_freeze();save(record_path,record)
        return record
    except BaseException:
        partial=OUT/f'partial_group_{group:02}.npz'
        if blocks and not partial.exists():
            with partial.open('xb') as handle: np.savez_compressed(handle,block_means=np.stack(blocks),**{k:v for k,v in prefix.items() if k!='block_means'})
        save(OUT/f'ERROR_group_{group:02}.json',dict(group=group,traceback=traceback.format_exc(),
            consumed_banks=consumed,completed_blocks=len(blocks),RNG_initial_states=initial,
            RNG_current_states={k:g.bit_generator.state for k,g in generators.items()},
            input_hashes_so_far={k:h.hexdigest() for k,h in digests.items()},source_sha256=sha(SOURCE),automatic_retry=False))
        raise


def scheduling_initializer(barrier):
    global _scheduling_barrier
    _scheduling_barrier=barrier


def deterministic_worker(index):
    _scheduling_barrier.wait(timeout=30)
    active,noise=deterministic_inputs(2,7,3)
    scores,_=evaluate_block(active,noise,np.eye(4)[None,:,:],np.zeros(4))
    return dict(index=index,pid=os.getpid(),score_sha256=array_sha(scores),environment=environment())


def scheduling_check():
    assert not (PACKAGE/'FREEZE.json').exists()
    context=mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=2,mp_context=context,initializer=scheduling_initializer,
                             initargs=(context.Barrier(2),)) as pool:
        futures=[pool.submit(deterministic_worker,i) for i in range(2)]
        results=[f.result() for f in futures]
    assert results[0]['score_sha256']==results[1]['score_sha256']
    assert all(x['environment']['torch_threads']==1 for x in results)
    assert len({x['pid'] for x in results})==2
    save(KERNEL/'SCHEDULING_SELFTEST.json',dict(passed=True,source_sha256=sha(SOURCE),
        workers_requested=2,results=results,unique_worker_pids=sorted({x['pid'] for x in results}),
        new_scientific_RNG=0,new_model_integrals=0,new_fits=0,
        scope='Two deterministic tiny-array tasks through ProcessPoolExecutor; scientific generator and run_group were not called.'))
    print(json.dumps(dict(passed=True,worker_pids=[x['pid'] for x in results])))


def run():
    frozen=validate_freeze();config,candidates,_,_=load_contract()
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'EXECUTION.json').exists() and not list(OUT.glob('group_*.json')) and not list(OUT.glob('group_*_prefix.npz'))
    assert not list(OUT.glob('ERROR*.json')) and not (OUT/'RESULTS.json').exists()
    save(OUT/'EXECUTION.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(SOURCE),
        freeze_sha256=sha(PACKAGE/'FREEZE.json'),automatic_retry=False,environment=environment()))
    start=time.perf_counter();records={};active={};next_group=0
    pool=ProcessPoolExecutor(max_workers=config['workers'])
    try:
        while next_group<config['workers']:
            active[pool.submit(run_group,next_group)]=next_group;next_group+=1
        while active:
            completed,_=wait(active,return_when=FIRST_COMPLETED)
            fresh=[future.result() for future in completed]
            for future in completed:del active[future]
            for record in fresh:
                records[record['group']]=record
                print(json.dumps(dict(completed_group=record['group'],seconds=record['seconds'])),flush=True)
            save(OUT/'PROGRESS.json',dict(completed=sorted(records),running=sorted(active.values()),seconds=time.perf_counter()-start),False)
            while len(active)<config['workers'] and next_group<config['groups']:
                active[pool.submit(run_group,next_group)]=next_group;next_group+=1
        ordered=[records[i] for i in range(config['groups'])]
        assert all(r['consumed_banks']==1024 and r['completed_blocks']==16 for r in ordered)
        qualification=ordered[0]['numerical_controls'];assert qualification['official_API_calls']==192
        assert all(r['numerical_controls'] is None for r in ordered[1:])
        result=dict(source_sha256=sha(SOURCE),freeze_sha256=sha(PACKAGE/'FREEZE.json'),
            config_sha256=sha(PACKAGE/'CONFIG.json'),candidates_sha256=sha(PACKAGE/'CANDIDATES.json'),
            config=config,profile_order=PROFILES,candidate_names=[r['name'] for r in candidates['candidates']],
            patterns=PATTERNS.tolist(),groups=ordered,numerical_qualification=qualification,
            accounting=dict(independent_groups=32,independent_banks=32768,new_prior_vectors=3276800,
                Gaussian_scalars=851968000,active_Gaussian_scalars=13107200,noise_Gaussian_scalars=838860800,
                base_chunks=512,covariance_bank_evaluations=32768*35,
                profile_candidate_pattern_bank_scores=32768*2*35*16,
                official_numerical_API_calls=192,new_observations=0,new_fits=0,new_model_integrals=0,new_selections=0),
            seconds=time.perf_counter()-start,
            scope='All means are total mathematical float64 scores. Each bank has one inner K100 arithmetic likelihood average before log; outer bank averages are separate. No weighted population/data statistics or new tests are computed in this runner. Numerical failures are retained without expanding or retrying the scientific budget.')
        validate_freeze();save(OUT/'RESULTS.json',result)
        save(OUT/'STATE.json',dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),
            source_sha256=sha(SOURCE),results_sha256=sha(OUT/'RESULTS.json'),seconds=result['seconds']))
        print(json.dumps(dict(complete=True,seconds=result['seconds'],native_qualification_passed=qualification['passed'],
                             results_sha256=sha(OUT/'RESULTS.json'))),flush=True)
    except BaseException:
        for future in active:future.cancel()
        save(OUT/'ERROR.json',dict(traceback=traceback.format_exc(),completed_groups=sorted(records),
            already_active_groups=sorted(active.values()),automatic_retry=False))
        raise
    finally:
        pool.shutdown(wait=True,cancel_futures=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['schema','check','bench','schedule-check','run']);args=parser.parse_args()
    if args.mode=='schema':print(json.dumps(schema(),ensure_ascii=True,indent=2))
    else:{'check':selftest,'bench':benchmark,'schedule-check':scheduling_check,'run':run}[args.mode]()
