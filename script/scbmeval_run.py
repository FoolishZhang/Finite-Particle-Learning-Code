"""Independent frozen-model scoring; schema/check never generate scientific RNG.

Main arrays evaluate the ideal logistic score numerically in float64, with
canonical nuisance signs and two fixed active-pattern weight vectors. Original
float32 API calls are limited numerical controls, not the primary target.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_key]='1'
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import argparse,hashlib,json,math,platform,sys,time,traceback
import numpy as np
import torch
import scbmeval_kernel as kernel

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/scbm_frozen_evaluation/EVAL'
SOURCE=Path(__file__).resolve()
SELECTION=OUT.parent/'MODEL_SELECTION.json'
DESIGN=ROOT/'docs/08_research_研究工作/冻结表示的条件与总体目标_20261003/EXPERIMENT_DESIGN.md'
SUMMARY_SOURCE=ROOT/'script/scbmeval_summary.py'
REFERENCE_SOURCE=ROOT/'script/scbmeval_reference.py'
REFERENCE_HELPER=ROOT/'script/scbminv_reference.py'
REFERENCE_HELPER_SHA='79c911d1601ae9740b0e045a5d879394cfd4a82aaa9e48f72070fc46b5098a41'
BASE='9c792ef18f858a19fd85342e1a65d74e4cf9e5bd'
KERNEL_SHA='9dc59d1436abeb1ce5eaf4c70a098a6f34c4ced1d797492c0c56ff5eee8e64d1'
PROFILES=[['TAIL','learned'],['TAIL','half'],['TAIL','fact'],['LAST','learned'],['LAST','fact']]
TARGETS=['P','D'];METRICS=['centered_score','price_t','path_t']
CONFIG=dict(replicates=32,banks_per_replicate=512,chunk_banks=64,K=100,nuisance=256,
    frozen_models=16,profiles=PROFILES,targets=TARGETS,metrics=METRICS,
    grid_points=21,actual_t_index=21,points_per_profile=22,workers=2,torch_threads=1,
    seed_base=202610030400,roles=dict(active=0,nuisance=1),stored_prefix_banks=8,
    numerical_prefix_banks=4,numerical_model_indices=[0,15],numerical_contexts=['TAIL','LAST'],
    numerical_profile='learned',numerical_score_tolerance=1e-4,numerical_path_gradient_tolerance=2e-5,
    score_constant_removed='nuisance * log(2)',silent_weight_normalization=False,
    selection_or_significance_in_runner=False)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def read(p):return json.loads(Path(p).read_text('utf8'))
def ah(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def save(p,x,new=True):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x' if new else 'w',encoding='utf8',newline='\n') as handle:
        json.dump(x,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write('\n')


def environment():
    try:
        from threadpoolctl import threadpool_info
        pools=threadpool_info();available=True
        assert all(p['num_threads']==1 for p in pools),pools
    except ImportError:pools=[];available=False
    assert torch.get_num_threads()==1
    return dict(python=platform.python_version(),executable=str(Path(sys.executable).resolve()),
        numpy=np.__version__,torch=torch.__version__,torch_threads=torch.get_num_threads(),
        threadpoolctl_available=available,actual_threadpools=pools,
        thread_environment={k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')})


def load_selection(check_old_arrays=False):
    value=read(SELECTION)
    assert sha(ROOT/value['training_result_path'])==value['training_result_sha256']
    assert sha(ROOT/value['truth_reference_path'])==value['truth_reference_sha256']
    truth=read(ROOT/value['truth_reference_path'])
    assert value['population_weights']==next(r['main_pmf'] for r in truth['records'] if r['t']==0.)
    assert value['patterns']==kernel.PATTERNS.tolist()
    assert value['grid_t']==np.linspace(-.75,.75,21).tolist()
    assert [r['stream'] for r in value['models']]==list(range(16))
    old=read(ROOT/value['training_result_path'])
    for model in value['models']:
        assert model['N']==4096 and sum(model['empirical_counts'])==4096
        assert np.array_equal(np.asarray(model['empirical_counts'])/4096,model['empirical_weights'])
        previous=old['records'][model['stream']]
        for context in ('TAIL','LAST'):
            assert model[context]==previous['outputs']['FREE'][context]
        assert value['old_input_NPZ'][previous['input_file']]==previous['input_sha256']
        if check_old_arrays:
            p=ROOT/previous['input_file'];assert sha(p)==previous['input_sha256']
            with np.load(p,allow_pickle=False) as data:
                counts=np.bincount(data['active_labels']@np.array([8,4,2,1]),minlength=16)
                assert np.array_equal(counts,model['empirical_counts'])
    return value


def prefix_labels(selection):
    old=read(ROOT/selection['training_result_path']);answer={}
    for index in CONFIG['numerical_model_indices']:
        row=old['records'][index];p=ROOT/row['input_file']
        assert sha(p)==row['input_sha256']==selection['old_input_NPZ'][row['input_file']]
        with np.load(p,allow_pickle=False) as data:
            answer[index]=(data['active_labels'][:4].copy(),data['nuisance_labels'][:4].copy())
    return answer


def schema():
    assert sha(kernel.__file__)==KERNEL_SHA
    assert sha(REFERENCE_HELPER)==REFERENCE_HELPER_SHA
    selection=load_selection()
    official=kernel.provenance()['official']
    return dict(base_commit=BASE,source_sha256=sha(SOURCE),kernel_sha256=KERNEL_SHA,
        selection_path=rel(SELECTION),selection_sha256=sha(SELECTION),
        design_path=rel(DESIGN),design_sha256=sha(DESIGN),
        summary_source_path=rel(SUMMARY_SOURCE),summary_source_sha256=sha(SUMMARY_SOURCE),
        reference_source_path=rel(REFERENCE_SOURCE),reference_source_sha256=sha(REFERENCE_SOURCE),
        reference_helper_path=rel(REFERENCE_HELPER),reference_helper_sha256=REFERENCE_HELPER_SHA,
        official_adapter_path=rel(Path(kernel.official_adapter.__file__)),official_adapter_sha256=kernel.ADAPTER_SHA,
        official_inputs={official['official_losses']:official['official_losses_sha256'],official['official_models']:official['official_models_sha256'],official['default_config']:official['default_config_sha256']},
        config=CONFIG,environment=environment(),old_input_NPZ=selection['old_input_NPZ'],
        array_schema=dict(means='Each metric float64[16,5,22,2]; model,profile,point,target',
            block_summaries='block_<metric> float64[8,16,5,22,2]',
            block_pattern_summaries='block_pattern_<metric> float64[8,16,5,22,16]',
            saved_prefix='active float64[8,100,4], nuisance float64[8,100,256]',
            profiles=PROFILES,targets=TARGETS,metrics=METRICS,
            point_index='0..20 fixed grid;21 is that model/context original physical t, even if numerically equal to a grid point'),
        RNG_order='Independent PCG64 SeedSequence([202610030400,rep,role]); for each of 8 chunks draw role0 normal[64,100,4], then role1 normal[64,100,256], float64 C-order, concatenate raw bytes into separate role hashes.',
        score_scope='centered_score removes the common nlog2 from each pattern loss before P/D weighting, without silently normalizing the supplied numerical PMF. Only differences and gradients enter the planned inference.',
        numerical_scope='Original API32 per-pattern batch means minus nlog2 and path gradients, separately weighted P/D, are compared with canonical mathematical64 arrays. Canonical/sign-flip API32 equality is only numerically checked, not assumed exact.')


def freeze():
    assert not (OUT/'FREEZE.json').exists() and not (OUT/'EXECUTION.json').exists()
    assert not list(OUT.glob('rep_*.npz'))
    test=read(OUT/'SELFTEST.json');assert test['passed'] and test['source_sha256']==sha(SOURCE)
    assert read(kernel.OUT/'SELFTEST.json')['passed'] and read(kernel.OUT/'SELFTEST.json')['source_sha256']==KERNEL_SHA
    load_selection(check_old_arrays=True)
    value=schema();value.update(created_utc=datetime.now(timezone.utc).isoformat(),run_authorized=True,
        selftest_sha256=sha(OUT/'SELFTEST.json'),kernel_selftest_sha256=sha(kernel.OUT/'SELFTEST.json'))
    save(OUT/'FREEZE.json',value)
    print(json.dumps(dict(frozen=True,freeze_sha256=sha(OUT/'FREEZE.json'),source_sha256=sha(SOURCE))))


def validate_freeze():
    f=read(OUT/'FREEZE.json');assert f['run_authorized'] is True
    for key,value in schema().items():assert f[key]==value,key
    assert f['selftest_sha256']==sha(OUT/'SELFTEST.json')
    assert f['kernel_selftest_sha256']==sha(kernel.OUT/'SELFTEST.json')
    for name,digest in f['official_inputs'].items():assert sha(ROOT/name)==digest,name
    return f


def mathematical_block(active,noise,selection):
    """All patterns first averaged within a block, then fixed P/D weighting."""
    pattern={k:np.empty((16,5,22,16),dtype=np.float64) for k in METRICS}
    mapping={'centered_score':'centered_score','price_t':'price_t','path_t':'path_t'}
    contexts=[]
    for mi,model in enumerate(selection['models']):
        for context,alphas,profiles in [('TAIL',(1.,.5),[0,1]),('LAST',(1.,),[3])]:
            cache=kernel.prepare_noise(noise,model[context]['scales'],alphas)
            contexts.append((mi,context,cache,profiles))
    zero=np.zeros(active.shape[:2],dtype=np.float64)
    for ti,t in enumerate(selection['grid_t']):
        terms=kernel.active_terms(active,float(t),need_path=True)
        fact=kernel.from_terms(terms,zero,256)
        for metric in METRICS:
            mean=fact[mapping[metric]].mean(axis=0)
            pattern[metric][:,2,ti]=mean;pattern[metric][:,4,ti]=mean
        for mi,context,cache,profiles in contexts:
            for pi,w in zip(profiles,cache['centered_logweights'].values()):
                result=kernel.from_terms(terms,w,256)
                for metric in METRICS:pattern[metric][mi,pi,ti]=result[mapping[metric]].mean(axis=0)
    for mi,context,cache,profiles in contexts:
        terms=kernel.active_terms(active,float(selection['models'][mi][context]['t']),need_path=True)
        fact_index=2 if context=='TAIL' else 4
        for pi,w in [(fact_index,zero),*zip(profiles,cache['centered_logweights'].values())]:
            result=kernel.from_terms(terms,w,256)
            for metric in METRICS:pattern[metric][mi,pi,21]=result[mapping[metric]].mean(axis=0)
    weighted={k:np.empty((16,5,22,2),dtype=np.float64) for k in METRICS}
    for mi,model in enumerate(selection['models']):
        weights=np.column_stack((selection['population_weights'],model['empirical_weights']))
        assert np.all(abs(weights.sum(axis=0)-1.)<2e-12)
        for metric in METRICS:weighted[metric][mi]=pattern[metric][mi]@weights
    return weighted,pattern


def api_score_path(active,noise,tau,t,labels,noise_labels):
    """One real official API batch mean and its physical t path derivative."""
    tt=torch.tensor(t,dtype=torch.float32,requires_grad=True)
    eye=torch.eye(4,dtype=torch.float32)
    cov=eye+(1.+tt)*(torch.tensor(kernel.C0,dtype=torch.float32)-eye)
    x=torch.as_tensor(active,dtype=torch.float32)@torch.linalg.cholesky(cov).T+torch.tensor(kernel.MU,dtype=torch.float32)
    nuisance=torch.as_tensor(noise,dtype=torch.float32)*torch.as_tensor(tau,dtype=torch.float32)
    logits=torch.cat((x,nuisance),dim=2)
    probabilities=torch.sigmoid(logits.permute(0,2,1))
    y=torch.as_tensor(np.concatenate((labels,noise_labels),axis=1),dtype=torch.int64)
    loss=kernel.official_adapter.official_loss(active.shape[1]).compute_concept_loss(probabilities,y)
    gradient=torch.autograd.grad(loss,tt)[0]
    return dict(total_score=float(loss.detach()),centered_score=float(loss.detach())-256*math.log(2.),
        path_t=float(gradient),official_API_calls=1,
        sigmoid_zero_evaluations=int((probabilities.detach()==0).sum()),
        sigmoid_one_evaluations=int((probabilities.detach()==1).sum()),
        probability_evaluations=probabilities.numel(),maximum_absolute_logit=float(logits.detach().abs().max()))


def numerical_controls(active,noise,selection,labels,replicate):
    active=active[:4];noise=noise[:4]
    core=[];canonical=[]
    for mi in CONFIG['numerical_model_indices']:
        model=selection['models'][mi]
        weights=np.column_stack((selection['population_weights'],model['empirical_weights']))
        old_y,old_yn=labels[mi]
        for context in CONFIG['numerical_contexts']:
            t=model[context]['t'];tau=np.asarray(model[context]['scales'],dtype=float)
            terms=kernel.active_terms(active,t,need_path=True)
            cache=kernel.prepare_noise(noise,tau,alphas=(1.,))
            mathematical=kernel.from_terms(terms,cache['centered_logweights'][1.],256)
            native=[]
            for bits in kernel.PATTERNS:
                yy=np.broadcast_to(bits,(4,4)).copy()
                native.append(api_score_path(active,noise,tau,t,yy,np.zeros((4,256),dtype=np.int8)))
            api_score=np.array([r['centered_score'] for r in native]);api_path=np.array([r['path_t'] for r in native])
            math_score=mathematical['centered_score'].mean(axis=0);math_path=mathematical['path_t'].mean(axis=0)
            sd=(api_score-math_score)@weights;gd=(api_path-math_path)@weights
            core.append(dict(replicate=replicate,model_index=mi,context=context,t=t,
                official_pattern_total_score=[r['total_score'] for r in native],official_pattern_centered_score=api_score.tolist(),
                official_pattern_path_t=api_path.tolist(),mathematical_pattern_centered_score=math_score.tolist(),
                mathematical_pattern_path_t=math_path.tolist(),targets=TARGETS,
                weighted_centered_score_difference=sd.tolist(),weighted_path_t_difference=gd.tolist(),
                score_passed=(abs(sd)<=CONFIG['numerical_score_tolerance']).tolist(),
                path_passed=(abs(gd)<=CONFIG['numerical_path_gradient_tolerance']).tolist(),
                passed=bool(np.all(abs(sd)<=CONFIG['numerical_score_tolerance']) and np.all(abs(gd)<=CONFIG['numerical_path_gradient_tolerance'])),
                official_API_calls=16,sigmoid_zero_evaluations=sum(r['sigmoid_zero_evaluations'] for r in native),
                sigmoid_one_evaluations=sum(r['sigmoid_one_evaluations'] for r in native)))
            # For each old complete observation, transform the fresh Gaussian
            # noise signs so its signed logits equal the all-zero-label case.
            canonical_api=api_score_path(active,noise,tau,t,old_y,np.zeros_like(old_yn))
            original_api=api_score_path(active,noise*(1.-2.*old_yn[:,None,:]),tau,t,old_y,old_yn)
            score_error=original_api['centered_score']-canonical_api['centered_score']
            path_error=original_api['path_t']-canonical_api['path_t']
            canonical.append(dict(replicate=replicate,model_index=mi,context=context,t=t,
                canonical=canonical_api,original_labels_sign_flipped_noise=original_api,
                centered_score_difference=score_error,path_t_difference=path_error,
                score_passed=abs(score_error)<=CONFIG['numerical_score_tolerance'],
                path_passed=abs(path_error)<=CONFIG['numerical_path_gradient_tolerance'],
                passed=bool(abs(score_error)<=CONFIG['numerical_score_tolerance'] and abs(path_error)<=CONFIG['numerical_path_gradient_tolerance']),
                official_API_calls=2,scope='Numerical float32 check at four old complete label rows, not a bankwise claim about the original training realization or an exact native API symmetry theorem.'))
    return core,canonical


def qualification(core,canonical):
    return dict(core_cases=len(core),core_cases_passed=sum(r['passed'] for r in core),
        weighted_score_checks=2*len(core),weighted_score_passed=sum(sum(r['score_passed']) for r in core),
        weighted_path_checks=2*len(core),weighted_path_passed=sum(sum(r['path_passed']) for r in core),
        maximum_weighted_score_difference=max(abs(v) for r in core for v in r['weighted_centered_score_difference']),
        maximum_weighted_path_difference=max(abs(v) for r in core for v in r['weighted_path_t_difference']),
        canonical_cases=len(canonical),canonical_cases_passed=sum(r['passed'] for r in canonical),
        maximum_canonical_score_difference=max(abs(r['centered_score_difference']) for r in canonical),
        maximum_canonical_path_difference=max(abs(r['path_t_difference']) for r in canonical))


def run_replicate(replicate):
    f=validate_freeze();selection=load_selection();labels=prefix_labels(selection)
    tick=time.perf_counter();block=-1
    inp=OUT/f'input_{replicate:02}.npz';output=OUT/f'rep_{replicate:02}.npz';record_path=OUT/f'rep_{replicate:02}.json'
    assert not any(p.exists() for p in (inp,output,record_path))
    rng={name:np.random.Generator(np.random.PCG64(np.random.SeedSequence([CONFIG['seed_base'],replicate,role]))) for name,role in CONFIG['roles'].items()}
    initial={name:r.bit_generator.state for name,r in rng.items()};digests={name:hashlib.sha256() for name in rng}
    blocks={k:np.full((8,16,5,22,2),np.nan) for k in METRICS}
    pattern_blocks={k:np.full((8,16,5,22,16),np.nan) for k in METRICS}
    core=[];canonical=[];prefix_hashes={}
    try:
        for block in range(8):
            active=rng['active'].standard_normal((64,100,4));noise=rng['nuisance'].standard_normal((64,100,256))
            for name,value in (('active',active),('nuisance',noise)):
                digests[name].update(np.ascontiguousarray(value).tobytes())
                if block==0:prefix_hashes[name]=ah(value[:8])
            if block==0:
                np.savez_compressed(inp,active=active[:8],nuisance=noise[:8])
                core,canonical=numerical_controls(active,noise,selection,labels,replicate)
            weighted,patterns=mathematical_block(active,noise,selection)
            for key in METRICS:blocks[key][block]=weighted[key];pattern_blocks[key][block]=patterns[key]
        means={k:v.mean(axis=0) for k,v in blocks.items()}
        arrays={**{'block_'+k:v for k,v in blocks.items()},**{'block_pattern_'+k:v for k,v in pattern_blocks.items()},**means}
        assert all(np.isfinite(x).all() for x in arrays.values())
        np.savez_compressed(output,**arrays)
        record=dict(replicate=replicate,source_sha256=sha(SOURCE),kernel_sha256=KERNEL_SHA,freeze_sha256=sha(OUT/'FREEZE.json'),
            input_file=rel(inp),input_sha256=sha(inp),output_file=rel(output),output_sha256=sha(output),
            role_seed_sequences={name:[CONFIG['seed_base'],replicate,role] for name,role in CONFIG['roles'].items()},
            RNG_initial_states=initial,RNG_final_states={name:r.bit_generator.state for name,r in rng.items()},
            role_input_sha256={name:d.hexdigest() for name,d in digests.items()},first8_input_sha256=prefix_hashes,
            means={k:v.tolist() for k,v in means.items()},completed_chunks=8,base_banks=512,
            numerical_controls=core,canonical_controls=canonical,numerical_qualification=qualification(core,canonical),
            seconds=time.perf_counter()-tick)
        validate_freeze();save(record_path,record);return record
    except BaseException as error:
        partial=OUT/f'partial_rep_{replicate:02}.npz'
        if not partial.exists():np.savez_compressed(partial,**{'block_'+k:v for k,v in blocks.items()},**{'block_pattern_'+k:v for k,v in pattern_blocks.items()})
        save(OUT/f'ERROR_rep_{replicate:02}.json',dict(replicate=replicate,current_chunk=block,error=str(error),
            traceback=traceback.format_exc(),RNG_initial_states=initial,RNG_current_states={name:r.bit_generator.state for name,r in rng.items()},
            role_hashes_so_far={name:d.hexdigest() for name,d in digests.items()},automatic_retry=False))
        raise


def run():
    f=validate_freeze()
    assert not (OUT/'EXECUTION.json').exists() and not list(OUT.glob('input_*.npz')) and not list(OUT.glob('rep_*.npz')),'No automatic scientific retry'
    save(OUT/'EXECUTION.json',dict(start_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(SOURCE),
        freeze_sha256=sha(OUT/'FREEZE.json'),environment=environment(),workers=2,automatic_retry=False))
    start=time.perf_counter();records={};next_rep=0;active={};pool=ProcessPoolExecutor(max_workers=2)
    try:
        while next_rep<2:active[pool.submit(run_replicate,next_rep)]=next_rep;next_rep+=1
        while active:
            done,_=wait(active,return_when=FIRST_COMPLETED)
            finished=[future.result() for future in done]
            for future in done:del active[future]
            for row in finished:
                records[row['replicate']]=row
                print(json.dumps(dict(completed_replicate=row['replicate'],seconds=row['seconds'])),flush=True)
            save(OUT/'PROGRESS.json',dict(completed_replicates=sorted(records),count=len(records),
                running_replicates=sorted(active.values()),seconds=time.perf_counter()-start),new=False)
            while len(active)<2 and next_rep<32:active[pool.submit(run_replicate,next_rep)]=next_rep;next_rep+=1
        ordered=[records[i] for i in range(32)]
        core=[c for r in ordered for c in r['numerical_controls']];canonical=[c for r in ordered for c in r['canonical_controls']]
        result=dict(source_sha256=sha(SOURCE),kernel_sha256=KERNEL_SHA,freeze_sha256=sha(OUT/'FREEZE.json'),
            selection_sha256=f['selection_sha256'],config=CONFIG,array_schema=f['array_schema'],records=ordered,
            numerical_qualification=qualification(core,canonical),seconds=time.perf_counter()-start,
            ledger=dict(independent_evaluation_replicates=32,new_base_banks=16384,new_prior_vectors=1638400,
                Gaussian_scalars=425984000,active_Gaussian_scalars=6553600,nuisance_Gaussian_scalars=419430400,
                new_observations=0,new_fits=0,new_model_integrals=0,
                shared_active_grid_terms_per_chunk=21,actual_t_terms_per_chunk=32,
                common_FACT_grid_scores_per_chunk=21,learned_or_half_grid_scores_per_chunk=1008,
                actual_t_score_cases_per_chunk=80,score_kernel_calls=283904,
                reused_bank_pattern_score_cases=290717696,
                original_API_pattern_control_calls=2048,original_API_canonical_control_calls=256,
                original_API_control_calls=2304),
            scope='All 16 historical models are fixed and paired within each of 32 new evaluation replicates. No model selection, risk fitting or significance calculation is performed by this runner.',
            score_scope=f['score_scope'],numerical_scope=f['numerical_scope'])
        validate_freeze();save(OUT/'RESULTS.json',result)
        save(OUT/'STATE.json',dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),
            results_sha256=sha(OUT/'RESULTS.json'),source_sha256=sha(SOURCE),seconds=result['seconds']))
        print(json.dumps(dict(completed=True,seconds=result['seconds'],numerical_qualification=result['numerical_qualification'])),flush=True)
    except BaseException as error:
        for future in active:future.cancel()
        save(OUT/'ERROR.json',dict(error=str(error),traceback=traceback.format_exc(),completed_replicates=sorted(records),
            active_replicates=sorted(active.values()),automatic_retry=False));raise
    finally:pool.shutdown(wait=True,cancel_futures=True)


def artificial_selection():
    models=[]
    for i in range(16):
        counts=np.full(16,256)
        tau=.20+.01*np.sin(np.arange(256)*.19+i*.1)
        models.append(dict(stream=i,N=4096,empirical_counts=counts.tolist(),empirical_weights=(counts/4096).tolist(),
            TAIL=dict(t=(i-7)*.017,scales=tau.tolist()),LAST=dict(t=(i-7)*.019+.001,scales=(tau*.97).tolist())))
    return dict(models=models,population_weights=(np.arange(1,17)/136).tolist(),grid_t=np.linspace(-.75,.75,21).tolist())


def check():
    selection=artificial_selection();active,noise,_=kernel.deterministic_inputs(B=4,K=100,n=256)
    labels={}
    for i in (0,15):
        y=((np.arange(16).reshape(4,4)+i)%3==0).astype(np.int8)
        yn=((np.arange(1024).reshape(4,256)+i)%5<2).astype(np.int8)
        labels[i]=(y,yn)
    core,canonical=numerical_controls(active,noise,selection,labels,-1)
    weighted,patterns=mathematical_block(active,noise,selection)
    assert all(v.shape==(16,5,22,2) and np.isfinite(v).all() for v in weighted.values())
    assert all(v.shape==(16,5,22,16) and np.isfinite(v).all() for v in patterns.values())
    for metric in METRICS:
        assert np.array_equal(patterns[metric][:,2,:21],patterns[metric][:,4,:21])
        for mi in range(16):assert np.array_equal(patterns[metric][mi,2,:21],patterns[metric][0,2,:21])
        weights=np.column_stack((selection['population_weights'],selection['models'][0]['empirical_weights']))
        assert np.allclose(weighted[metric][0],patterns[metric][0]@weights,rtol=0,atol=1e-14)
    q=qualification(core,canonical)
    assert q['core_cases_passed']==4 and q['canonical_cases_passed']==4,q
    output=dict(passed=True,source_sha256=sha(SOURCE),kernel_sha256=KERNEL_SHA,config=CONFIG,
        numerical_qualification=q,numerical_controls=core,canonical_controls=canonical,
        weighted_shapes={k:list(v.shape) for k,v in weighted.items()},pattern_shapes={k:list(v.shape) for k,v in patterns.items()},
        shared_FACT_identity_passed=True,full_model_weighting_passed=True,
        new_scientific_RNG_draws=0,new_observations=0,new_model_integrals=0,new_fits=0,
        scope='Deterministic sine/cosine inputs and artificial model/label/weight arrays only. Not evaluation of scientific outcomes or a precision pilot.')
    save(OUT/'SELFTEST.json',output)
    print(json.dumps({k:v for k,v in output.items() if k not in ('numerical_controls','canonical_controls')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['schema','check','freeze','run'])
    phase=parser.parse_args().phase
    if phase=='schema':print(json.dumps(schema(),ensure_ascii=False,indent=2))
    else:globals()[phase]()
