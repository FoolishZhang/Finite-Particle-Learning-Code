"""Six-correlation training from identity; scientific run requires explicit freeze."""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_key]='1'
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import argparse,hashlib,json,math,platform,sys,time,traceback
import numpy as np
import torch
from scipy.special import expit
import scbm6_kernel as kernel

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/scbm_multicorrelation/TRAIN';SOURCE=Path(__file__).resolve()
DESIGN=ROOT/'docs/08_research_研究工作/六相关自由学习与方向机制_20261003/EXPERIMENT_DESIGN.md'
REFERENCE_SOURCE=ROOT/'script/scbm6_reference.py'
REFERENCE_HELPER=ROOT/'script/scbminv_reference.py'
REFERENCE_HELPER_SHA='79c911d1601ae9740b0e045a5d879394cfd4a82aaa9e48f72070fc46b5098a41'
KERNEL_SHA='40ad5e8139229fe624a5d888c77768a81b6addecab018b18b824a2c634573d13'
BASE='1edce5d5481f192815ff072a7b49a07f14e47f80'
METHODS=['FIXED','FREE','FACT']
# Used only to generate observed data. It is never passed to the learner's
# parameter constructor, optimizer, or covariance feasibility restoration.
DATA_COV=np.array([[1.,.25,-.15,.10],[.25,1.,.10,-.15],[-.15,.10,1.,.25],[.10,-.15,.25,1.]])
CONFIG=dict(streams=16,observations=4096,nuisance=256,K=100,batch_size=64,steps=512,
    tail_postupdates=64,methods=METHODS,pair_order=[list(p) for p in kernel.PAIRS],initial_rho=[0.]*6,
    scale0=1.,lr_rho=.02,lr_scale=.02,betas=[.9,.999],eps=1e-8,weight_decay=0.,
    eigen_floor=.1,repair_target=.1001,scale_floor=1e-6,shuffle=False,gradient_clip=False,
    learning_rate_schedule=False,early_stop=False,workers=2,torch_threads=1,
    seed_base=202610031100,roles=dict(data_gaussian=0,data_active_uniform=1,data_noise_uniform=2,
        prior_active_gaussian=3,prior_noise_gaussian=4),
    numerical_preupdate_steps=[0,31,127,511],numerical_loss_tolerance=1e-4,
    numerical_rho_gradient_tolerance=2e-5,numerical_raw_gradient_tolerance=2e-6,
    displayed_postupdate_steps=[0,32,128,512],primary_output='TAIL',
    primary_family_size=3,primary_alpha=.05,primary_SE_thresholds=[.0002,.0002,.00005],noninferiority_margin=.0002)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def read(p):return json.loads(Path(p).read_text('utf8'))
def ah(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def save(p,value,new=True):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x' if new else 'w',encoding='utf8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def environment():
    try:
        from threadpoolctl import threadpool_info
        pools=threadpool_info();available=True;assert all(p['num_threads']==1 for p in pools),pools
    except ImportError:pools=[];available=False
    assert torch.get_num_threads()==1
    return dict(python=platform.python_version(),executable=str(Path(sys.executable).resolve()),numpy=np.__version__,
        torch=torch.__version__,torch_threads=torch.get_num_threads(),device='cpu',
        threadpoolctl_available=available,actual_threadpools=pools,
        thread_environment={k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')})


def schema():
    assert sha(kernel.__file__)==KERNEL_SHA and sha(REFERENCE_HELPER)==REFERENCE_HELPER_SHA
    p=kernel.provenance()['official']
    return dict(base_commit=BASE,source_sha256=sha(SOURCE),kernel_sha256=KERNEL_SHA,
        official_adapter_sha256=kernel.ADAPTER_SHA,design_path=rel(DESIGN),design_sha256=sha(DESIGN),
        reference_source_path=rel(REFERENCE_SOURCE),reference_source_sha256=sha(REFERENCE_SOURCE),
        reference_helper_path=rel(REFERENCE_HELPER),reference_helper_sha256=REFERENCE_HELPER_SHA,
        environment=environment(),config=CONFIG,data_model=dict(mu=kernel.MU,covariance=DATA_COV.tolist()),
        official_inputs={p['official_losses']:p['official_losses_sha256'],p['official_models']:p['official_models_sha256'],p['default_config']:p['default_config_sha256']},
        RNG_order='Independent PCG64 SeedSequence([202610031100,stream,role]). Roles0/1/2 once: Gaussian[4096,4], Uniform[4096,4], Uniform[4096,256]. Each step roles3/4: Gaussian[100,64,4], Gaussian[100,64,256]. Float64 C-order bytes hashed, then one float32 conversion shared by three methods.',
        array_schema=dict(rho='float32[3,513,6], initial index0; method FIXED/FREE/FACT, pairs01/02/03/12/13/23',
            rho_gradient='float32[3,512,6], preupdate',rho_unprojected='float32[3,512,6]',
            projection_alpha='float64[3,512]',proposal_eigmin='float64[3,512]',eigmin='float64[3,513]',
            repaired='bool[3,512]',cholesky_success='bool[3,513]',finite='bool[3,512]',feasible='bool[3,513]',
            free_raw='float32[513,256]',free_tau='float32[513,256]',free_raw_gradient='float32[512,256]',
            loss='float32[3,512]',maximum_absolute_logit='float32[3,512]',
            sigmoid_zero_count='int64[3,512]',sigmoid_one_count='int64[3,512]',probability_evaluations='int64[3,512]',batch_start='int64[512]',
            final_rho_adam_m='float32[3,6]',final_rho_adam_v='float32[3,6]',
            final_free_raw_adam_m='float32[256]',final_free_raw_adam_v='float32[256]'),
        outputs_schema='records[].outputs[method][TAIL/LAST]={rho[6],covariance[4,4],scales[256],eigmin}; TAIL means physical postupdates449..512. FACT scales0 is an analytical-elimination convention, not a learned floor value.',
        numerical_scope='The original float64 Gaussian arrays and current float32 parameter values promoted to float64 enter a separate stable-logit expression. Differences include input/parameter rounding and expression changes, not only BCE.')


def freeze():
    assert not (OUT/'FREEZE.json').exists() and not (OUT/'EXECUTION.json').exists() and not list(OUT.glob('stream_*.npz'))
    check=read(OUT/'SELFTEST.json');assert check['passed'] and check['source_sha256']==sha(SOURCE)
    kc=read(kernel.OUT/'SELFTEST.json');assert kc['passed'] and kc['source_sha256']==KERNEL_SHA
    rc=read(OUT.parent/'REFERENCE/SELFTEST.json');assert rc['passed'] and rc['source_sha256']==sha(REFERENCE_SOURCE)
    value=schema();value.update(created_utc=datetime.now(timezone.utc).isoformat(),run_authorized=True,
        selftest_sha256=sha(OUT/'SELFTEST.json'),kernel_selftest_sha256=sha(kernel.OUT/'SELFTEST.json'),
        reference_selftest_sha256=sha(OUT.parent/'REFERENCE/SELFTEST.json'))
    save(OUT/'FREEZE.json',value);print(json.dumps(dict(frozen=True,source_sha256=sha(SOURCE),freeze_sha256=sha(OUT/'FREEZE.json'))))


def validate_freeze():
    f=read(OUT/'FREEZE.json');assert f['run_authorized'] is True
    for key,value in schema().items():assert f[key]==value,key
    for key,p in [('selftest_sha256',OUT/'SELFTEST.json'),('kernel_selftest_sha256',kernel.OUT/'SELFTEST.json'),
                  ('reference_selftest_sha256',OUT.parent/'REFERENCE/SELFTEST.json')]:assert f[key]==sha(p) and read(p)['passed']
    for name,digest in f['official_inputs'].items():assert sha(ROOT/name)==digest,name
    return f


def allocate(steps=512,n=256):
    shape=dict(rho=(3,steps+1,6),rho_gradient=(3,steps,6),rho_unprojected=(3,steps,6),
        free_raw=(steps+1,n),free_tau=(steps+1,n),free_raw_gradient=(steps,n),loss=(3,steps),maximum_absolute_logit=(3,steps))
    tape={k:np.full(v,np.nan,dtype=np.float32) for k,v in shape.items()}
    for k,s in [('projection_alpha',(3,steps)),('proposal_eigmin',(3,steps)),('eigmin',(3,steps+1))]:tape[k]=np.full(s,np.nan)
    for k in ('repaired','finite'):tape[k]=np.zeros((3,steps),dtype=bool)
    for k in ('cholesky_success','feasible'):tape[k]=np.zeros((3,steps+1),dtype=bool)
    for k in ('sigmoid_zero_count','sigmoid_one_count','probability_evaluations'):tape[k]=np.zeros((3,steps),dtype=np.int64)
    tape['batch_start']=np.full(steps,-1,dtype=np.int64);return tape


def snapshot(tape,models,index):
    for mi,method in enumerate(METHODS):
        model=models[method];rho=model.rho.detach().numpy();C=kernel.covariance_numpy(rho)
        eig=float(np.linalg.eigvalsh(C).min());scales=model.scales().detach()
        with torch.no_grad():_,info=torch.linalg.cholesky_ex(model.covariance())
        feasible=np.isfinite(rho).all() and eig>=.1 and int(info)==0 and bool(torch.isfinite(scales).all()) and bool((scales>=kernel.FLOOR).all())
        assert feasible,(method,index,eig)
        tape['rho'][mi,index]=rho;tape['eigmin'][mi,index]=eig
        tape['cholesky_success'][mi,index]=int(info)==0;tape['feasible'][mi,index]=feasible
    tape['free_raw'][index]=models['FREE'].raw_scale.detach().numpy()
    tape['free_tau'][index]=models['FREE'].scales().detach().numpy()


def numeric_row(stream,step,method,result,program,stable):
    le=result['loss']-stable['loss'];difference=np.asarray(result['rho_gradient'])-stable['rho_gradient'];maximum=float(np.max(abs(difference)))
    row=dict(stream=stream,preupdate_step=step,method=method,official_loss=result['loss'],stable64_loss=stable['loss'],loss_difference=le,
        official_rho_gradient=result['rho_gradient'],stable64_rho_gradient=stable['rho_gradient'],rho_gradient_difference=difference.tolist(),
        maximum_rho_gradient_difference=maximum,loss_passed=abs(le)<=1e-4,rho_gradient_passed=maximum<=2e-5)
    if method=='FREE':
        raw=program.raw_scale.grad.detach().numpy().astype(float);diff=raw-np.asarray(stable['raw_gradient']);mx=float(np.max(abs(diff)))
        row.update(official_raw_gradient=raw.tolist(),stable64_raw_gradient=stable['raw_gradient'],raw_gradient_difference=diff.tolist(),
                   maximum_raw_gradient_difference=mx,raw_gradient_passed=mx<=2e-6)
    row['passed']=bool(row['loss_passed'] and row['rho_gradient_passed'] and row.get('raw_gradient_passed',True));return row


def advance(tape,models,optimizers,step,arrays,begin,stream,do_numerical):
    batch=kernel.prepare_batch(*arrays);rows=[];tape['batch_start'][step]=begin
    for mi,method in enumerate(METHODS):
        model=models[method];control=kernel.stable64(model,*arrays) if do_numerical else None
        result=kernel.optimizer_step(model,optimizers[method],batch,True)
        for key in ('loss','rho_gradient','rho_unprojected','projection_alpha','proposal_eigmin'):tape[key][mi,step]=result[key]
        tape['repaired'][mi,step]=result['projected'];tape['finite'][mi,step]=True
        for key in ('sigmoid_zero_count','sigmoid_one_count','probability_evaluations','maximum_absolute_logit'):tape[key][mi,step]=result['diagnostics'][key]
        if method=='FREE':tape['free_raw_gradient'][step]=model.raw_scale.grad.detach().numpy()
        if do_numerical:rows.append(numeric_row(stream,step,method,result,model,control))
    snapshot(tape,models,step+1);return rows


def final_moments(tape,models,optimizers):
    for moment,label in [('exp_avg','m'),('exp_avg_sq','v')]:
        tape['final_rho_adam_'+label]=np.stack([optimizers[m].state[models[m].rho][moment].detach().numpy() for m in METHODS])
        tape['final_free_raw_adam_'+label]=optimizers['FREE'].state[models['FREE'].raw_scale][moment].detach().numpy().copy()


def outputs(tape,tail=64):
    result={}
    for mi,method in enumerate(METHODS):
        result[method]={}
        for kind in ('TAIL','LAST'):
            rho=tape['rho'][mi,-tail:].astype(float).mean(axis=0) if kind=='TAIL' else tape['rho'][mi,-1].astype(float)
            C=kernel.covariance_numpy(rho);eigmin=float(np.linalg.eigvalsh(C).min());assert eigmin>=.1
            if method=='FREE':tau=tape['free_tau'][-tail:].astype(float).mean(axis=0) if kind=='TAIL' else tape['free_tau'][-1].astype(float)
            else:tau=np.full(tape['free_tau'].shape[1],1. if method=='FIXED' else 0.)
            result[method][kind]=dict(rho=rho.tolist(),covariance=C.tolist(),scales=tau.tolist(),eigmin=eigmin)
    return result


def numerical_summary(rows):
    return dict(checks=len(rows),passed=sum(r['passed'] for r in rows),loss_passed=sum(r['loss_passed'] for r in rows),
        rho_gradient_checks=len(rows),rho_gradient_components=6*len(rows),rho_gradient_passed=sum(r['rho_gradient_passed'] for r in rows),
        raw_gradient_checks=sum('raw_gradient_passed' in r for r in rows),raw_gradient_passed=sum(r.get('raw_gradient_passed',False) for r in rows),
        maximum_loss_difference=max(abs(r['loss_difference']) for r in rows),
        maximum_rho_gradient_difference=max(r['maximum_rho_gradient_difference'] for r in rows),
        maximum_raw_gradient_difference=max(r.get('maximum_raw_gradient_difference',0) for r in rows))


def run_stream(stream):
    validate_freeze();tick=time.perf_counter();step=-1;completed=0
    inp=OUT/f'input_{stream:02}.npz';output=OUT/f'stream_{stream:02}.npz';record_path=OUT/f'stream_{stream:02}.json'
    assert not any(p.exists() for p in (inp,output,record_path))
    rng={name:np.random.Generator(np.random.PCG64(np.random.SeedSequence([CONFIG['seed_base'],stream,role]))) for name,role in CONFIG['roles'].items()}
    initial={name:r.bit_generator.state for name,r in rng.items()};hashes={}
    digests={name:hashlib.sha256() for name in ('prior_active_gaussian','prior_noise_gaussian')}
    tape=allocate();models,optimizers=kernel.make_programs();snapshot(tape,models,0);numerical=[]
    try:
        z=rng['data_gaussian'].standard_normal((4096,4));u=rng['data_active_uniform'].random((4096,4));un=rng['data_noise_uniform'].random((4096,256))
        hashes.update(data_gaussian=ah(z),data_active_uniform=ah(u),data_noise_uniform=ah(un))
        labels=(u<expit(z@np.linalg.cholesky(DATA_COV).T+np.asarray(kernel.MU))).astype(np.int8)
        noise_labels=(un<.5).astype(np.int8);del un
        label_hashes=dict(active_labels=ah(labels),nuisance_labels=ah(noise_labels));first_hashes={}
        for step in range(512):
            begin=(64*step)%4096;end=begin+64
            active=rng['prior_active_gaussian'].standard_normal((100,64,4));noise=rng['prior_noise_gaussian'].standard_normal((100,64,256))
            for name,value in [('prior_active_gaussian',active),('prior_noise_gaussian',noise)]:
                b=np.ascontiguousarray(value).tobytes();digests[name].update(b)
                if step==0:first_hashes[name]=hashlib.sha256(b).hexdigest()
            if step==0:
                np.savez_compressed(inp,active_labels=labels,nuisance_labels=noise_labels,data_active_gaussian=z,data_active_uniform=u,
                                    first_prior_active=active,first_prior_nuisance=noise)
                del z,u
            numerical.extend(advance(tape,models,optimizers,step,(active,noise,labels[begin:end],noise_labels[begin:end]),begin,stream,step in CONFIG['numerical_preupdate_steps']))
            completed=step+1
        final_moments(tape,models,optimizers);np.savez_compressed(output,**tape)
        hashes.update({name:d.hexdigest() for name,d in digests.items()})
        record=dict(stream=stream,source_sha256=sha(SOURCE),freeze_sha256=sha(OUT/'FREEZE.json'),
            input_file=rel(inp),input_sha256=sha(inp),output_file=rel(output),output_sha256=sha(output),
            role_seed_sequences={name:[CONFIG['seed_base'],stream,role] for name,role in CONFIG['roles'].items()},
            RNG_initial_states=initial,RNG_final_states={name:r.bit_generator.state for name,r in rng.items()},
            role_input_sha256=hashes,label_sha256=label_hashes,first_prior_sha256=first_hashes,
            outputs=outputs(tape),completed_steps=completed,numerical_controls=numerical,numerical_qualification=numerical_summary(numerical),
            diagnostics={method:dict(feasibility_repair_count=int(tape['repaired'][mi].sum()),minimum_eigenvalue=float(tape['eigmin'][mi].min()),
                minimum_proposal_eigenvalue=float(tape['proposal_eigmin'][mi].min()),minimum_repair_alpha=float(tape['projection_alpha'][mi].min()),
                sigmoid_zero_count=int(tape['sigmoid_zero_count'][mi].sum()),sigmoid_one_count=int(tape['sigmoid_one_count'][mi].sum()),
                probability_evaluations=int(tape['probability_evaluations'][mi].sum()),maximum_absolute_logit=float(tape['maximum_absolute_logit'][mi].max()),
                all_finite=bool(tape['finite'][mi].all()),all_feasible=bool(tape['feasible'][mi].all()),all_Cholesky_success=bool(tape['cholesky_success'][mi].all())) for mi,method in enumerate(METHODS)},
            seconds=time.perf_counter()-tick,
            output_scope='Physical six-correlation and scale averages for TAIL. Convex feasibility is checked. FACT scale0 is only analytical elimination; no true ray is used by the learner.')
        validate_freeze();save(record_path,record);return record
    except BaseException as error:
        partial=OUT/f'partial_stream_{stream:02}.npz'
        if not partial.exists():np.savez_compressed(partial,**tape)
        save(OUT/f'ERROR_stream_{stream:02}.json',dict(stream=stream,current_step=step,completed_steps=completed,error=str(error),traceback=traceback.format_exc(),
            RNG_initial_states=initial,RNG_current_states={name:r.bit_generator.state for name,r in rng.items()},
            role_hashes_so_far={**hashes,**{name:d.hexdigest() for name,d in digests.items()}},automatic_retry=False));raise


def run():
    f=validate_freeze()
    assert not (OUT/'EXECUTION.json').exists() and not list(OUT.glob('input_*.npz')) and not list(OUT.glob('stream_*.npz')),'No automatic scientific retry'
    save(OUT/'EXECUTION.json',dict(start_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(SOURCE),freeze_sha256=sha(OUT/'FREEZE.json'),environment=environment(),workers=2,automatic_retry=False))
    start=time.perf_counter();records={};next_stream=0;active={};pool=ProcessPoolExecutor(max_workers=2)
    try:
        while next_stream<2:active[pool.submit(run_stream,next_stream)]=next_stream;next_stream+=1
        while active:
            done,_=wait(active,return_when=FIRST_COMPLETED);finished=[future.result() for future in done]
            for future in done:del active[future]
            for row in finished:
                records[row['stream']]=row;print(json.dumps(dict(completed_stream=row['stream'],seconds=row['seconds'])),flush=True)
            save(OUT/'PROGRESS.json',dict(completed_streams=sorted(records),count=len(records),running_streams=sorted(active.values()),seconds=time.perf_counter()-start),False)
            while len(active)<2 and next_stream<16:active[pool.submit(run_stream,next_stream)]=next_stream;next_stream+=1
        ordered=[records[i] for i in range(16)];rows=[x for r in ordered for x in r['numerical_controls']]
        result=dict(source_sha256=sha(SOURCE),kernel_sha256=KERNEL_SHA,freeze_sha256=sha(OUT/'FREEZE.json'),design_sha256=f['design_sha256'],
            config=CONFIG,data_model=f['data_model'],environment=f['environment'],records=ordered,numerical_qualification=numerical_summary(rows),
            ledger=dict(independent_streams=16,new_complete_observations=65536,data_Gaussian_scalars=262144,data_Uniform_scalars=17039360,
                new_prior_vectors=52428800,prior_Gaussian_scalars=13631488000,prior_active_Gaussian_scalars=209715200,prior_noise_Gaussian_scalars=13421772800,
                observation_presentations=524288,controlled_optimization_programs=48,TAIL_outputs=48,LAST_outputs=48,
                official_training_API_calls=24576,stable64_control_calls=192,checked_rho_gradient_components=1152,
                new_neural_network_fits=0,new_model_integrals=0),seconds=time.perf_counter()-start,
            scope='New six-correlation programs from identity on new datasets, with fixed means and active/nuisance block structure. No known true direction in parameter construction or updates. No causal attribution of new-versus-old risk differences; endpoint reference and original new q3 family are separate.')
        validate_freeze();save(OUT/'RESULTS.json',result)
        save(OUT/'STATE.json',dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(SOURCE),results_sha256=sha(OUT/'RESULTS.json'),seconds=result['seconds']))
        print(json.dumps(dict(completed=True,seconds=result['seconds'],numerical_qualification=result['numerical_qualification'])),flush=True)
    except BaseException as error:
        for future in active:future.cancel()
        save(OUT/'ERROR.json',dict(error=str(error),traceback=traceback.format_exc(),completed_streams=sorted(records),active_streams=sorted(active.values()),automatic_retry=False));raise
    finally:pool.shutdown(wait=True,cancel_futures=True)


def check():
    models,optimizers=kernel.make_programs();tape=allocate(steps=2);snapshot(tape,models,0)
    _,arrays=kernel.deterministic(B=64);rows=[]
    for step in range(2):rows.extend(advance(tape,models,optimizers,step,arrays,64*step,-1,True))
    final_moments(tape,models,optimizers);result=outputs(tape,tail=2)
    assert all(np.isfinite(v).all() for v in tape.values()) and all(r['passed'] for r in rows)
    assert np.array_equal(tape['rho'][:,0],np.zeros((3,6)))
    assert all(result['FACT'][kind]['scales']==[0.]*256 for kind in ('TAIL','LAST'))
    assert np.array_equal(result['FREE']['TAIL']['rho'],tape['rho'][1,1:].astype(float).mean(axis=0))
    value=dict(passed=True,source_sha256=sha(SOURCE),kernel_sha256=KERNEL_SHA,
        shapes={k:list(v.shape) for k,v in tape.items()},dtypes={k:str(v.dtype) for k,v in tape.items()},
        numerical_qualification=numerical_summary(rows),numerical_controls=rows,outputs_schema_checked=True,identity_initialization_checked=True,
        config=CONFIG,new_scientific_RNG_draws=0,new_model_integrals=0,new_scientific_fits=0,
        scope='Deterministic fixed arrays and two isolated integration steps, not sampled learning outcomes or a convergence pilot.')
    save(OUT/'SELFTEST.json',value)
    print(json.dumps({k:v for k,v in value.items() if k!='numerical_controls'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['schema','check','freeze','run']);phase=parser.parse_args().phase
    if phase=='schema':print(json.dumps(schema(),ensure_ascii=False,indent=2))
    else:globals()[phase]()
