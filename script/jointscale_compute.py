"""Frozen joint-scale training and prespecified TAIL-only quadrature.

No experiment is run on import.  A root FREEZE.json must match every source and
the exact config before either training RNG or reference integrations execute.
"""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_key]="1"
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import argparse, hashlib, itertools, json, math, time, traceback
import numpy as np
import torch
from scipy.special import expit, roots_hermitenorm, logsumexp
from scipy.stats import t as student_t
import jointscale_kernel as kernel

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/joint_scale_learning'
DOC=ROOT/'docs/08_research_研究工作/两组尺度共同学习与恢复边界_20261004'
TRUTH=ROOT/'artifacts/active_scale_boundary/REFERENCE/RESULTS.json'
BASE='af707cd80bb0c1173a01cb9e93b21c8df63aa4e0'
ARMS=[f'{start}_{method}' for start in ('NEAR','UNIT') for method in kernel.METHODS]
ROLES=dict(data_gaussian=0,data_active_uniform=1,data_nuisance_uniform=2,prior_active_gaussian=3,prior_nuisance_gaussian=4)
CONFIG=dict(base_commit=BASE,streams=16,observations=4096,K=100,nuisance=256,batch_size=64,
    steps=512,tail_postupdates=64,arms=ARMS,methods=list(kernel.METHODS),starts=['NEAR','UNIT'],
    active_initial_scales=[.1,1.],initial_rho=[0.]*6,initial_tau=[1.,1.,0.],
    initial_mu='clip(logit((per-coordinate data count + .5)/(N+1)),-2,2); common to both starts and all methods',
    mean_bounds=[-2.,2.],scale_bounds=[.001,2.],tau_bounds=[0.,2.],eigen_floor=.1,repair_target=.1001,
    lr=.02,betas=[.9,.999],eps=1e-8,weight_decay=0.,shuffle=False,gradient_clip=False,
    learning_rate_schedule=False,early_stop=False,workers=4,torch_threads=1,
    seed_base=202610041500,roles=ROLES,preupdate_numerical_steps=[0,31,127,511],
    numerical_loss_tolerance=1e-4,numerical_gradient_tolerance=4e-5,
    displayed_postupdate_steps=[0,32,128,512],primary_output='TAIL',family_size=6,family_alpha=.05,
    SE_threshold=.0002,reference_orders=[32,40,48],main_reference_order=40,
    reference_max_pmf_change=2e-7,reference_max_KL_change=2e-6,reference_mass_tolerance=2e-12,
    reference_chunk=65536,reference_scope='96 TAIL models only; checkpoint and LAST parameters have no new risk integrals')
DATA_MU=[-.6,-.2,.2,.6]
DATA_C=np.asarray([[1.,.25,-.15,.10],[.25,1.,.10,-.15],[-.15,.10,1.,.25],[.10,-.15,.25,1.]])
PATTERNS=np.asarray(list(itertools.product((0,1),repeat=4)),dtype=int)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def read(p):return json.loads(Path(p).read_text('utf8'))
def save(p,x,new=True):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x' if new else 'w',encoding='utf8',newline='\n') as f:
        json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def schema():
    return dict(config=CONFIG,source_sha256=sha(__file__),kernel_sha256=sha(kernel.__file__),
        teacher_reference_path=rel(TRUTH),teacher_reference_sha256=sha(TRUTH),
        data_model=dict(mu=DATA_MU,covariance=DATA_C.tolist()),kernel=kernel.provenance(),
        RNG_order='PCG64 SeedSequence([seed_base,stream,role]); roles0/1/2 once Gaussian[N,4], Uniform[N,4], Uniform[N,256]; roles3/4 every step Gaussian[K,B,4], Gaussian[K,B,256]. Float64 C-order bytes hashed, then one common float32 conversion reused by six arms.',
        tape_schema=dict(params='float32[6,513,15] physical mu4,s4,rho6,tau; initial index0',
            gradient='float32[6,512,15]; fixed tau slot0',proposal='float32[6,512,15] before all repairs',
            adam_m='float32[6,513,15]; fixed tau slot0',adam_v='float32[6,513,15]; fixed tau slot0',
            projection_alpha='float64[6,512]',proposal_eigmin='float64[6,512]',eigmin='float64[6,513]',
            loss='float32[6,512]',projected='bool[6,512,4] mu,s,rho,tau',
            nuisance_chi='float64[6,512] batch mean using current preupdate nuisance tau, descriptive',
            batch_start='int64[512]',maximum_absolute_logit='float32[6,512]',
            sigmoid_zero_count='int64[6,512]',sigmoid_one_count='int64[6,512]',probability_evaluations='int64[6,512]'),
        output_scope='TAIL means last64 postupdate mu and physical covariance diag(s)Cdiag(s); reconstruct s=sqrt(diagSigma), C=D_s^-1 Sigma D_s^-1. Tau averaged separately. Single Gaussian output, not prediction mixture.')

def validate_freeze():
    f=read(OUT/'FREEZE.json')
    assert f.get('run_authorized') is True
    assert f['schema']==schema(),'Frozen contract/source changed'
    for path,digest in f.get('inputs',{}).items():assert sha(ROOT/path)==digest,path
    return f

def state_vector(state):return np.asarray(state['mu']+state['s']+state['rho']+[state['tau']],dtype=np.float32)
def flat_grad(grad):return np.asarray(grad['mu']+grad['s']+grad['rho']+[float(grad.get('tau',0.))],dtype=np.float32)
def models_for(mu):
    models={};opts={}
    for start,scale in zip(CONFIG['starts'],CONFIG['active_initial_scales']):
        for method in kernel.METHODS:
            name=f'{start}_{method}';models[name]=kernel.ScoreProgram(method,mu,scale)
            opts[name]=kernel.make_optimizer(models[name],CONFIG['lr'])
    return models,opts

def allocate():
    tape={k:np.full((6,513,15),np.nan,np.float32) for k in ('params','adam_m','adam_v')}
    tape.update({k:np.full((6,512,15),np.nan,np.float32) for k in ('gradient','proposal')})
    for k in ('loss','maximum_absolute_logit'):tape[k]=np.full((6,512),np.nan,np.float32)
    for k in ('projection_alpha','proposal_eigmin','nuisance_chi'):tape[k]=np.full((6,512),np.nan,np.float64)
    tape['eigmin']=np.full((6,513),np.nan,np.float64)
    tape['projected']=np.zeros((6,512,4),bool)
    for k in ('sigmoid_zero_count','sigmoid_one_count','probability_evaluations'):tape[k]=np.zeros((6,512),np.int64)
    tape['batch_start']=np.full(512,-1,np.int64)
    return tape

def snapshot(tape,models,opts,index):
    slices={'mu':slice(0,4),'s':slice(4,8),'rho':slice(8,14),'tau':slice(14,15)}
    for ai,arm in enumerate(ARMS):
        model=models[arm];model.assert_feasible();tape['params'][ai,index]=model.physical_vector()
        tape['eigmin'][ai,index]=np.linalg.eigvalsh(kernel.correlation_numpy(model.rho.detach().numpy())).min()
        for tape_key,opt_key in (('adam_m','exp_avg'),('adam_v','exp_avg_sq')):
            tape[tape_key][ai,index]=0.
            for name,p in model.named_parameters():
                state=opts[arm].state.get(p,{})
                if opt_key in state:tape[tape_key][ai,index,slices[name]]=state[opt_key].detach().numpy()

def output_model(vectors):
    vectors=np.asarray(vectors,dtype=float).reshape(-1,15)
    mu=vectors[:,:4].mean(axis=0)
    sigmas=[]
    for v in vectors:
        C=kernel.correlation_numpy(v[8:14]);sigmas.append(v[4:8,None]*C*v[None,4:8])
    covariance=np.mean(sigmas,axis=0);s=np.sqrt(np.diag(covariance));C=covariance/s[:,None]/s[None,:]
    eig=float(np.linalg.eigvalsh(C).min());assert eig>=kernel.EIGEN_FLOOR-1e-12
    return dict(mu=mu.tolist(),covariance=covariance.tolist(),s=s.tolist(),
        rho=[float(C[i,j]) for i,j in kernel.PAIRS],correlation=C.tolist(),tau=float(vectors[:,14].mean()),eigmin=eig,
        averaging='mu and full physical covariance separately; single Gaussian model')

def outputs(tape):
    return {arm:dict(TAIL=output_model(tape['params'][ai,-64:]),LAST=output_model(tape['params'][ai,-1:]),
        checkpoints={str(step):output_model(tape['params'][ai,step:step+1]) for step in CONFIG['displayed_postupdate_steps']})
        for ai,arm in enumerate(ARMS)}

def chi_for(noise,yn,tau):
    if tau==0.:return 1./CONFIG['K']
    logits=noise*float(tau);sign=2.*yn[None,:,:]-1.
    logw=-np.logaddexp(0.,-sign*logits).sum(axis=2)
    logpi=logw-logsumexp(logw,axis=0,keepdims=True)
    return float(np.exp(2.*logpi).sum(axis=0).mean())

def numerical_control(model,arrays):
    native=kernel.clone_to_path(model,'official32');math64=kernel.clone_to_path(model,'math64')
    nl,ng=kernel.gradients(native,kernel.prepare_batch(*arrays));ml,mg=kernel.gradients(math64,kernel.prepare_batch(*arrays,path='math64'))
    nflat=flat_grad({k:v.detach().numpy().tolist() for k,v in ng.items()}).astype(float)
    mflat=np.asarray(mg['mu'].detach().numpy().tolist()+mg['s'].detach().numpy().tolist()+mg['rho'].detach().numpy().tolist()+[float(mg.get('tau',0.))])
    ld=float(nl.detach())-float(ml.detach());gd=nflat-mflat
    return dict(official_loss=float(nl.detach()),math64_loss=float(ml.detach()),loss_difference=ld,
        official_gradient=nflat.tolist(),math64_gradient=mflat.tolist(),gradient_difference=gd.tolist(),
        maximum_gradient_difference=float(np.max(abs(gd))),maximum_math64_gradient=float(np.max(abs(mflat))),
        relative_maximum_gradient_difference=float(np.max(abs(gd))/max(float(np.max(abs(mflat))),1e-300)),
        passed=bool(abs(ld)<=CONFIG['numerical_loss_tolerance'] and np.max(abs(gd))<=CONFIG['numerical_gradient_tolerance']))

def run_stream(stream):
    validate_freeze();start=time.perf_counter();folder=OUT/'TRAIN';folder.mkdir(exist_ok=True)
    inp=folder/f'input_{stream:02}.npz';target=folder/f'stream_{stream:02}.npz';record=folder/f'stream_{stream:02}.json'
    assert not any(p.exists() for p in (inp,target,record)), 'No automatic retry'
    rng={name:np.random.Generator(np.random.PCG64(np.random.SeedSequence([CONFIG['seed_base'],stream,role]))) for name,role in ROLES.items()}
    initial={k:r.bit_generator.state for k,r in rng.items()};digests={k:hashlib.sha256() for k in ('prior_active_gaussian','prior_nuisance_gaussian')}
    hashes={};tape=allocate();numerical=[];step=-1
    try:
        z=rng['data_gaussian'].standard_normal((4096,4));u=rng['data_active_uniform'].random((4096,4));un=rng['data_nuisance_uniform'].random((4096,256))
        hashes.update(data_gaussian=ah(z),data_active_uniform=ah(u),data_nuisance_uniform=ah(un))
        labels=(u<expit(z@np.linalg.cholesky(DATA_C).T+DATA_MU)).astype(np.int8);yn=(un<.5).astype(np.int8);del un
        p=(labels.sum(axis=0)+.5)/4097.;mu=np.clip(np.log(p/(1.-p)),-2.,2.)
        models,opts=models_for(mu);snapshot(tape,models,opts,0)
        first_hashes={}
        for step in range(512):
            begin=(step*64)%4096;tape['batch_start'][step]=begin
            active=rng['prior_active_gaussian'].standard_normal((100,64,4));noise=rng['prior_nuisance_gaussian'].standard_normal((100,64,256))
            for name,array in (('prior_active_gaussian',active),('prior_nuisance_gaussian',noise)):
                b=np.ascontiguousarray(array).tobytes();digests[name].update(b)
                if step==0:first_hashes[name]=hashlib.sha256(b).hexdigest()
            if step==0:
                np.savez_compressed(inp,active_labels=labels,nuisance_labels=yn,data_active_gaussian=z,
                    data_active_uniform=u,first_prior_active=active,first_prior_nuisance=noise)
                del z,u
            arrays=(active,noise,labels[begin:begin+64],yn[begin:begin+64]);batch=kernel.prepare_batch(*arrays)
            common_chi={0.:.01,1.:chi_for(noise,arrays[3],1.)}
            for ai,arm in enumerate(ARMS):
                model=models[arm];tau=float(model.nuisance_scale().detach())
                if tau not in common_chi:common_chi[tau]=chi_for(noise,arrays[3],tau)
                tape['nuisance_chi'][ai,step]=common_chi[tau]
                if step in CONFIG['preupdate_numerical_steps']:
                    numerical.append(dict(stream=stream,arm=arm,preupdate_step=step,**numerical_control(model,arrays)))
                row=kernel.optimizer_step(model,opts[arm],batch,True)
                tape['loss'][ai,step]=row['loss'];tape['gradient'][ai,step]=flat_grad(row['gradient'])
                tape['proposal'][ai,step]=state_vector(row['repair']['proposal'])
                tape['projection_alpha'][ai,step]=row['repair']['projection_alpha'];tape['proposal_eigmin'][ai,step]=row['repair']['proposal_eigmin']
                tape['projected'][ai,step]=[row['repair']['projected'][k] for k in ('mu','s','rho','tau')]
                for key in ('maximum_absolute_logit','sigmoid_zero_count','sigmoid_one_count','probability_evaluations'):
                    tape[key][ai,step]=row['diagnostics'][key]
            snapshot(tape,models,opts,step+1)
        np.savez_compressed(target,**tape)
        hashes.update({k:v.hexdigest() for k,v in digests.items()})
        row=dict(stream=stream,source_sha256=sha(__file__),freeze_sha256=sha(OUT/'FREEZE.json'),
            input_file=rel(inp),input_sha256=sha(inp),output_file=rel(target),output_sha256=sha(target),
            role_seed_sequences={k:[CONFIG['seed_base'],stream,v] for k,v in ROLES.items()},
            RNG_initial_states=initial,RNG_final_states={k:r.bit_generator.state for k,r in rng.items()},
            role_input_sha256=hashes,first_prior_sha256=first_hashes,
            label_sha256=dict(active_labels=ah(labels),nuisance_labels=ah(yn)),data_counts=labels.sum(axis=0).tolist(),
            initial_mu=mu.tolist(),outputs=outputs(tape),completed_steps=512,numerical_controls=numerical,
            numerical_qualification=dict(checks=len(numerical),passed=sum(x['passed'] for x in numerical),
                maximum_loss_difference=max(abs(x['loss_difference']) for x in numerical),
                maximum_gradient_difference=max(max(abs(v) for v in x['gradient_difference']) for x in numerical)),
            diagnostics={arm:dict(repair_counts=tape['projected'][ai].sum(axis=0).tolist(),
                minimum_eigenvalue=float(tape['eigmin'][ai].min()),maximum_absolute_logit=float(tape['maximum_absolute_logit'][ai].max()),
                sigmoid_zero_count=int(tape['sigmoid_zero_count'][ai].sum()),sigmoid_one_count=int(tape['sigmoid_one_count'][ai].sum()),
                probability_evaluations=int(tape['probability_evaluations'][ai].sum())) for ai,arm in enumerate(ARMS)},
            seconds=time.perf_counter()-start)
        validate_freeze();save(record,row);return row
    except BaseException as error:
        np.savez_compressed(folder/f'partial_stream_{stream:02}.npz',**tape)
        save(folder/f'ERROR_stream_{stream:02}.json',dict(stream=stream,step=step,error=str(error),traceback=traceback.format_exc(),
            automatic_retry=False,RNG_current_states={k:r.bit_generator.state for k,r in rng.items()}));raise

def train():
    validate_freeze();folder=OUT/'TRAIN';folder.mkdir(exist_ok=True)
    assert not (folder/'EXECUTION.json').exists() and not list(folder.glob('stream_*.npz'))
    save(folder/'EXECUTION.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),freeze_sha256=sha(OUT/'FREEZE.json'),automatic_retry=False))
    tick=time.perf_counter();rows=[]
    with ProcessPoolExecutor(max_workers=CONFIG['workers']) as pool:
        futures={pool.submit(run_stream,stream):stream for stream in range(16)}
        for future in as_completed(futures):
            row=future.result();rows.append(row)
            print(json.dumps(dict(completed_stream=row['stream'],seconds=row['seconds'],numerical_passed=row['numerical_qualification']['passed'])),flush=True)
            save(folder/'PROGRESS.json',dict(completed_streams=sorted(r['stream'] for r in rows),seconds=time.perf_counter()-tick),False)
    rows.sort(key=lambda r:r['stream']);result=dict(config=CONFIG,records=rows,seconds=time.perf_counter()-tick,
        accounting=dict(new_observations=65536,optimizations=96,main_API_calls=49152,numerical_API_calls=384,
            prior_vectors=16*512*100*64,prior_Gaussian=16*512*100*64*260,
            data_Gaussian=16*4096*4,data_Uniform=16*4096*260,new_NN=0),
        numerical_qualification_passed=all(r['numerical_qualification']['passed']==24 for r in rows))
    save(folder/'RESULTS.json',result);print(json.dumps(dict(training_complete=True,seconds=result['seconds'],numerical_qualification_passed=result['numerical_qualification_passed'])),flush=True)

def pmf(mu,cov,order):
    nodes,weights=roots_hermitenorm(order);weights=weights/np.sqrt(2*np.pi)
    lower=np.linalg.cholesky(cov);total=np.zeros(16,dtype=np.longdouble)
    for start in range(0,order**4,CONFIG['reference_chunk']):
        index=np.arange(start,min(start+CONFIG['reference_chunk'],order**4),dtype=np.int64)
        ids=np.stack(np.unravel_index(index,(order,)*4),axis=1)
        x=nodes[ids]@lower.T+mu;w=np.prod(weights[ids],axis=1);plus=expit(x);minus=expit(-x)
        for j,y in enumerate(PATTERNS):total[j]+=np.sum(w*np.prod(np.where(y,plus,minus),axis=1),dtype=np.longdouble)
    raw=np.asarray(total,dtype=float);mass=float(raw.sum());normalized=raw/mass
    return dict(order=order,root='cholesky',raw_pmf=raw.tolist(),raw_mass=mass,mass_error=abs(mass-1.),
        pmf=normalized.tolist(),normalization_applied=True,Gaussian_nodes=order**4)

def teacher():
    source=read(TRUTH);record=next(r for r in source['records'] if r['name']=='mixed')
    P=np.asarray(record['teacher_pmf']);p=np.asarray(record['derived']['p'])
    product=np.prod(np.where(PATTERNS,p,1.-p),axis=1)
    return P,p,product

def reference_stream(stream):
    validate_freeze();P,p,product=teacher();record=read(OUT/'TRAIN'/f'stream_{stream:02}.json');rows=[]
    assert record['output_sha256']==sha(ROOT/record['output_file'])
    for arm in ARMS:
        model=record['outputs'][arm]['TAIL'];mu=np.asarray(model['mu']);cov=np.asarray(model['covariance'])
        variants=[pmf(mu,cov,order) for order in CONFIG['reference_orders']]
        main=np.asarray(variants[1]['pmf']);risk=float(np.sum(P*np.log(P/main)))
        qualification=[]
        for variant in variants:
            q=np.asarray(variant['pmf']);KL=float(np.sum(P*np.log(P/q)));change=float(np.max(abs(q-main)))
            passed=bool(change<=CONFIG['reference_max_pmf_change'] and abs(KL-risk)<=CONFIG['reference_max_KL_change']
                and variant['mass_error']<=CONFIG['reference_mass_tolerance'] and np.all(q>0))
            qualification.append(dict(order=variant['order'],max_pmf_change=change,KL_change=KL-risk,
                mass_error=variant['mass_error'],positive=bool(np.all(q>0)),passed=passed))
        rows.append(dict(stream=stream,arm=arm,kind='TAIL',model=model,main_pmf=main.tolist(),reference_joint_KL=risk,
            variants=variants,qualifications=qualification,qualification_passed=all(q['passed'] for q in qualification)))
    result=dict(stream=stream,rows=rows,qualification_passed=all(r['qualification_passed'] for r in rows))
    save(OUT/'REFERENCE'/f'stream_{stream:02}.json',result);return result

def reference():
    validate_freeze();train_result=read(OUT/'TRAIN/RESULTS.json');assert len(train_result['records'])==16
    folder=OUT/'REFERENCE';folder.mkdir(exist_ok=True);assert not (folder/'EXECUTION.json').exists()
    save(folder/'EXECUTION.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),freeze_sha256=sha(OUT/'FREEZE.json'),
        train_results_sha256=sha(OUT/'TRAIN/RESULTS.json'),automatic_retry=False))
    tick=time.perf_counter();records=[]
    with ProcessPoolExecutor(max_workers=CONFIG['workers']) as pool:
        for future in as_completed([pool.submit(reference_stream,s) for s in range(16)]):
            record=future.result();records.append(record)
            print(json.dumps(dict(reference_stream=record['stream'],qualification_passed=record['qualification_passed'])),flush=True)
    records.sort(key=lambda r:r['stream']);rows=[r for record in records for r in record['rows']]
    P,p,product=teacher();product_risk=float(np.sum(P*np.log(P/product)))
    look={(r['stream'],r['arm']):r['reference_joint_KL'] for r in rows}
    critical=float(student_t.ppf(1.-.05/(2*6),15));primary=[]
    for start in CONFIG['starts']:
        for left,right in (('FREE','FIXED'),('FREE','PRODUCT'),('ZERO','PRODUCT')):
            values=np.asarray([look[s,f'{start}_{left}']-(product_risk if right=='PRODUCT' else look[s,f'{start}_{right}']) for s in range(16)])
            mean=float(values.mean());se=float(values.std(ddof=1)/4)
            primary.append(dict(name=f'{start}_{left}_minus_{right}',start=start,left=left,right=right,
                stream_values=values.tolist(),mean=mean,SE=se,lower=mean-critical*se,upper=mean+critical*se,
                expected='negative',effect_passed=bool(mean+critical*se<0.),precision_passed=bool(se<=.0002)))
    result=dict(config=CONFIG,records=rows,teacher_reference_path=rel(TRUTH),teacher_reference_sha256=sha(TRUTH),
        teacher_pmf=P.tolist(),teacher_marginals=p.tolist(),truth_product_pmf=product.tolist(),truth_product_risk=product_risk,
        primary=primary,critical_t=critical,qualification_passed=all(r['qualification_passed'] for r in rows),
        primary_effects_passed=sum(r['effect_passed'] for r in primary),primary_precision_passed=sum(r['precision_passed'] for r in primary),
        descriptive={arm:dict(mean_joint_KL=float(np.mean([look[s,arm] for s in range(16)])),
            risks=[look[s,arm] for s in range(16)]) for arm in ARMS},
        accounting=dict(new_PMF_integrations=96*3,Gaussian_nodes=96*sum(o**4 for o in CONFIG['reference_orders']),
            new_observations=0,new_prior=0,new_fits=0),seconds=time.perf_counter()-tick,
        interval_scope='Bonferroni t development approximation for 16 independent new data/training streams; fixed numerical teacher and quadrature. Integration error not included.')
    save(folder/'RESULTS.json',result);print(json.dumps(dict(reference_complete=True,qualification_passed=result['qualification_passed'],
        primary_effects_passed=result['primary_effects_passed'],primary_precision_passed=result['primary_precision_passed'])),flush=True)

def check():
    """One isolated deterministic update per arm; no sampler or integration."""
    models,opts=models_for([-.35,-.1,.1,.35]);tape=allocate();snapshot(tape,models,opts,0)
    batch,arrays=kernel.deterministic_batch(B=7)
    numerical=[]
    for ai,arm in enumerate(ARMS):
        model=models[arm];numerical.append(dict(arm=arm,**numerical_control(model,arrays)))
        result=kernel.optimizer_step(model,opts[arm],batch,True)
        tape['gradient'][ai,0]=flat_grad(result['gradient']);tape['proposal'][ai,0]=state_vector(result['repair']['proposal'])
    snapshot(tape,models,opts,1)
    assert np.isfinite(tape['params'][:,:2]).all() and np.isfinite(tape['adam_m'][:,:2]).all()
    assert np.isfinite(tape['adam_v'][:,:2]).all()
    assert all(np.all(tape['adam_m'][ai,:2,-1]==0) for ai,arm in enumerate(ARMS) if not arm.endswith('_FREE'))
    averaged=[]
    for ai,arm in enumerate(ARMS):
        output=output_model(tape['params'][ai,:2]);physical=tape['params'][ai,:2].astype(float)
        expected=np.mean([np.diag(v[4:8])@kernel.correlation_numpy(v[8:14])@np.diag(v[4:8]) for v in physical],axis=0)
        covariance_error=float(np.max(abs(expected-np.asarray(output['covariance']))))
        assert covariance_error<1e-14 and output['eigmin']>=.1
        assert np.max(abs(np.asarray(output['mu'])-physical[:,:4].mean(axis=0)))==0.
        averaged.append(dict(arm=arm,covariance_averaging_error=covariance_error,eigmin=output['eigmin']))
    P,p,product=teacher();assert abs(P.sum()-1.)<1e-12 and abs(product.sum()-1.)<1e-12
    assert np.max(abs(P@PATTERNS-p))<1e-12
    chi0=chi_for(arrays[1],arrays[3],0.);chi1=chi_for(arrays[1],arrays[3],1.)
    assert chi0==.01 and .01<=chi1<=1.
    return dict(passed=all(row['passed'] for row in numerical),structural_passed=True,
        numerical_controls_passed=sum(row['passed'] for row in numerical),numerical_controls_total=len(numerical),
        source_sha256=sha(__file__),kernel_sha256=sha(kernel.__file__),
        numerical_controls=numerical,TAIL_covariance_averaging=averaged,
        product_reference_normalization=float(product.sum()),deterministic_chi=[chi0,chi1],
        new_scientific_RNG_draws=0,new_model_integrals=0,new_scientific_fits=0,
        scope='Deterministic schema/output checks and six isolated first updates; no scientific trajectory or pilot')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('schema','check','train','reference'))
    phase=parser.parse_args().phase
    if phase in ('schema','check'):print(json.dumps(globals()[phase](),ensure_ascii=False,indent=2))
    else:globals()[phase]()
