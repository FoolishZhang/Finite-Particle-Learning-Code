"""Independent finite-K objective witness on a sealed M2 exchangeable teacher."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import time
import traceback

import numpy as np
from scipy.special import expit, log_expit, logsumexp, gammaln
from scipy.stats import binom, t

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/constraint_boundary_v2/T'
PLAN=ROOT/'docs/08_research_研究工作/全局约束与采样门槛_20260929/PLAN_T.md'
METHODS=['original_IID10','original_IID100','original_ANTI100','count_RB_IID1','count_RB_IID10']
CONFIG=dict(d=[24,128],sigma=4.,c=.9,families=['NEG','POS_RANK1'],streams=8,observations=4096,batch=64,
            seed_base=20271901,original_K=[10,100],anti_K=100,RB_K=[1,10],alpha=.05,contrasts=20,
            material=.005,NEG_original_SE_gate=1.,other_SE_gate=.003,identity_gate=1e-12,normalization_gate=1e-12)

def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x,exclusive=True):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x' if exclusive else 'w',encoding='utf-8',newline='\n') as f:f.write(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvwrite(p,rows):
    with p.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def logchoose(d,n):return gammaln(d+1)-gammaln(n+1)-gammaln(d-n+1)
def neg_eta(z):
    d=z.shape[-1];m=z.mean(axis=-1,keepdims=True)
    return 4.*(math.sqrt(1+.9/(d-1))*(z-m)+math.sqrt(1-.9)*m)
def poisson_binomial(probabilities):
    """Full float64 PMF by coefficient recursion; no probability clipping."""
    p=np.asarray(probabilities,dtype=np.float64);assert p.ndim==2 and np.all((p>=0)&(p<=1))
    n,d=p.shape;mass=np.zeros((n,d+1),dtype=np.float64);mass[:,0]=1.
    for j in range(d):
        q=p[:,j,None]
        mass[:,1:j+2]=mass[:,1:j+2]*(1-q)+mass[:,:j+1]*q
        mass[:,0]*=1-p[:,j]
    assert np.isfinite(mass).all() and np.all(mass>=0)
    return mass
def original_neg(eta,y):
    primary=log_expit((2*y[:,None,:]-1)*eta).sum(-1)
    alternate=(y[:,None,:]*eta-np.logaddexp(0.,eta)).sum(-1)
    return primary,float(np.max(abs(primary-alternate)))
def original_rank1(eta,count,d):
    primary=count[:,None]*log_expit(eta)+(d-count[:,None])*log_expit(-eta)
    alternate=count[:,None]*eta-d*np.logaddexp(0.,eta)
    return primary,float(np.max(abs(primary-alternate)))
def rank1_RB(eta,count,d):
    lc=logchoose(d,count)[:,None]
    original,error=original_rank1(eta,count,d)
    # Independent binomial probability expression, retaining logchoose.
    conditional_count=binom.logpmf(count[:,None],d,expit(eta))
    rb=conditional_count-lc
    return rb,max(error,float(np.max(abs(rb-original)))),float(np.exp(conditional_count).min())
def neg_RB(eta,count,d):
    n,k,_=eta.shape;mass=poisson_binomial(expit(eta).reshape(-1,d))
    normalization=float(np.max(abs(mass.sum(1)-1)))
    selected=mass[np.arange(n*k),np.repeat(count,k)].reshape(n,k)
    assert np.all(selected>0),'Zero selected PB mass: preserve failure, never floor'
    answer=np.log(selected)-logchoose(d,count)[:,None]
    return answer,normalization,float(selected.min())
def loss(loglikelihood):
    return -logsumexp(loglikelihood,axis=1)+np.log(loglikelihood.shape[1])

def smoke():
    assert not (OUT/'SMOKE.json').exists()
    etas=np.array([[-1.2,.7,2.1,-.4],[.2,.2,.2,.2],[-2.,1.5,.3,-.8]])
    labels=np.array(list(itertools.product([0.,1.],repeat=4)));counts=labels.sum(1).astype(int)
    permutations=list(itertools.permutations(range(4)))
    mass=poisson_binomial(expit(etas));errors=[];formula_error=0.
    for j,eta in enumerate(etas):
        exact=np.exp(log_expit((2*labels-1)*eta).sum(1))
        enumerated=np.bincount(counts,weights=exact,minlength=5)
        errors.append(float(np.max(abs(enumerated-mass[j]))))
        for y,n in zip(labels,counts):
            averaged=np.mean([np.exp(log_expit((2*y-1)*eta[list(order)]).sum()) for order in permutations])
            errors.append(abs(float(averaged-mass[j,n]/math.comb(4,int(n)))))
        _,e=original_neg(np.broadcast_to(eta,(16,1,4)),labels);formula_error=max(formula_error,e)
    shared=np.tile(np.array([-.8,.1,1.4]),(16,1));rb,negative_control,_=rank1_RB(shared,counts,4)
    assert np.isfinite(loss(rb)).all()
    norm=float(np.max(abs(mass.sum(1)-1)))
    maximum=max(errors+[formula_error,negative_control,norm]);assert maximum<1e-12
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'.gitignore').write_text('/scores/\n',encoding='utf-8',newline='\n')
    write(OUT/'SMOKE.json',dict(time_utc=now(),passed=True,dimension=4,enumerated_labels=16,coordinate_permutations=24,
          PB_count_and_permutation_average_max_error=max(errors),PB_normalization_max_error=norm,
          original_two_formula_max_error=formula_error,POS_particle_identity_max_error=negative_control,
          finite_losses=True,source_sha256=sha(__file__),plan_sha256=sha(PLAN)))
    print(read(OUT/'SMOKE.json'),flush=True)

def verify_seal(seal):
    record=read(seal);assert isinstance(record.get('files'),dict),'Parameter seal must expose files hashes'
    for name,h in record['files'].items():assert sha(ROOT/name)==h,name
    return record
def freeze(parameters,parameter_seal):
    assert not (OUT/'FREEZE.json').exists();sm=read(OUT/'SMOKE.json');assert sm['passed'] and sm['source_sha256']==sha(__file__)
    p=Path(parameters).resolve();seal=Path(parameter_seal).resolve();m2=(ROOT/'artifacts/constraint_boundary_v2/M2').resolve()
    assert p.is_relative_to(m2) and seal.is_relative_to(m2),'Only passed M2 parameters are permitted'
    stamp=verify_seal(seal);assert p.relative_to(ROOT).as_posix() in stamp['files']
    m2config=read(m2/'CONFIG.json');assert m2config['sigma']==CONFIG['sigma'] and m2config['c']==CONFIG['c']
    bank=read(p);assert bank['all_numerical_pass'] is True
    rows=[r for r in bank['rows'] if r['d'] in CONFIG['d']]
    assert sorted(r['d'] for r in rows)==CONFIG['d'] and all(r['numerical_pass'] for r in rows)
    assert all(0<r['rank1_variance']<16 for r in rows)
    assert all(abs(r['latent_rho_positive']-CONFIG['c']/(r['d']-1))<1e-15 for r in rows)
    write(OUT/'INPUT_PARAMETERS.json',dict(parameters_file=p.relative_to(ROOT).as_posix(),parameter_seal=seal.relative_to(ROOT).as_posix(),rows=rows))
    paths=[Path(__file__),PLAN,OUT/'SMOKE.json',OUT/'INPUT_PARAMETERS.json',p,seal]
    write(OUT/'FREEZE.json',dict(time_utc=now(),config=CONFIG,files={x.relative_to(ROOT).as_posix():sha(x) for x in paths},
          M2_parameter_audit_passed=True,old_M_parameters_used=False,before_population_simulation=True))
    write(OUT/'ENVIRONMENT.json',dict(time_utc=now(),packages={x:importlib.metadata.version(x) for x in ['numpy','scipy']},
          dtype='float64',threads=2,RNG='PCG64',probability_floor=None))
def check_freeze():
    record=read(OUT/'FREEZE.json');assert record['config']==CONFIG
    for name,h in record['files'].items():assert sha(ROOT/name)==h,name

def one_stream(family,d,v,stream):
    seed=CONFIG['seed_base']+100000*CONFIG['families'].index(family)+100*d+stream
    teacher=np.random.default_rng(seed);prior=np.random.default_rng(seed+10000000);rbprior=np.random.default_rng(seed+20000000)
    outputs=[];allcounts=[];start=time.perf_counter();teacher_time=original_time=rb_time=0.
    formula_max=PB_norm_max=POS_identity_max=0.;minimum_PB=1.;normal_draws_teacher=normal_draws_original=normal_draws_RB=0
    for begin in range(0,4096,64):
        b=min(64,4096-begin);tick=time.perf_counter()
        if family=='NEG':
            eta=neg_eta(teacher.normal(size=(b,d)));y=(teacher.random((b,d))<expit(eta)).astype(float);count=y.sum(1).astype(int)
            normal_draws_teacher+=b*d
        else:
            p=expit(math.sqrt(v)*teacher.normal(size=b));count=teacher.binomial(d,p);normal_draws_teacher+=b
        teacher_time+=time.perf_counter()-tick;tick=time.perf_counter()
        if family=='NEG':
            full=neg_eta(prior.normal(size=(b,100,d)));logs,e=original_neg(full,y);opposite,ee=original_neg(-full[:,:50],y)
            normal_draws_original+=b*100*d
        else:
            full=math.sqrt(v)*prior.normal(size=(b,100));logs,e=original_rank1(full,count,d);opposite,ee=original_rank1(-full[:,:50],count,d)
            # Same-particle original versus RB, using the original bank as well.
            _,match,_=rank1_RB(full,count,d);POS_identity_max=max(POS_identity_max,match);normal_draws_original+=b*100
        formula_max=max(formula_max,e,ee)
        losses=[loss(logs[:,:10]),loss(logs),loss(np.concatenate([logs[:,:50],opposite],axis=1))]
        original_time+=time.perf_counter()-tick;tick=time.perf_counter()
        if family=='NEG':
            eta=neg_eta(rbprior.normal(size=(b,10,d)));rb,norm,minimum=neg_RB(eta,count,d)
            PB_norm_max=max(PB_norm_max,norm);normal_draws_RB+=b*10*d
        else:
            eta=math.sqrt(v)*rbprior.normal(size=(b,10));rb,match,minimum=rank1_RB(eta,count,d)
            POS_identity_max=max(POS_identity_max,match);normal_draws_RB+=b*10
        minimum_PB=min(minimum_PB,minimum);losses.extend([loss(rb[:,:1]),loss(rb)]);rb_time+=time.perf_counter()-tick
        values=np.stack(losses,axis=1);assert np.isfinite(values).all()
        assert formula_max<1e-12 and PB_norm_max<1e-12 and POS_identity_max<1e-12
        outputs.append(values);allcounts.append(count)
    values=np.concatenate(outputs);counts=np.concatenate(allcounts);baseline=d*np.log(2.);difference=values-baseline
    target=OUT/'scores'/f'{family}_d{d}_stream{stream}.npz';target.parent.mkdir(exist_ok=True)
    assert not target.exists();np.savez_compressed(target,losses=values,differences=difference,count=counts)
    rows=[]
    for j,method in enumerate(METHODS):
        K=[10,100,100,1,10][j]
        rows.append(dict(family=family,d=d,stream=stream,seed=seed,method=method,observations=4096,K=K,
                         mean_F=float(values[:,j].mean()),F0=float(baseline),mean_F_minus_F0=float(difference[:,j].mean()),
                         within_stream_SD=float(difference[:,j].std(ddof=1)),minimum=float(difference[:,j].min()),maximum=float(difference[:,j].max()),
                         logical_Bernoulli_factors=4096*K*d if j<3 else 0,
                         logical_PB_DP_state_updates=4096*K*d*(d+3)//2 if j>=3 and family=='NEG' else 0))
    receipt=dict(family=family,d=d,stream=stream,seed=seed,seconds=time.perf_counter()-start,teacher_seconds=teacher_time,
                 original_shared_bank_seconds=original_time,RB_shared_bank_seconds=rb_time,
                 normal_draws_teacher=normal_draws_teacher,normal_draws_original=normal_draws_original,normal_draws_RB=normal_draws_RB,
                 actual_original_unique_particle_likelihoods=4096*150,
                 actual_original_formula_crosschecks=4096*(150 if family=='NEG' else 260),
                 actual_POS_conditional_binomial_PMFs=4096*110 if family=='POS_RANK1' else 0,
                 actual_NEG_PB_state_updates=4096*10*d*(d+3)//2 if family=='NEG' else 0,
                 POS_conditional_Binomial_shortcut=family=='POS_RANK1',
                 original_formula_error=formula_max,PB_normalization_error=PB_norm_max,POS_particle_identity_error=POS_identity_max,
                 minimum_selected_count_probability=minimum_PB,count_histogram=np.bincount(counts,minlength=d+1).tolist(),
                 arrays_sha256=sha(target),finite_losses=True)
    return rows,receipt

def run():
    check_freeze();parameters=read(OUT/'INPUT_PARAMETERS.json')['rows'];stream_rows=[];receipts=[]
    for family in CONFIG['families']:
        for d in CONFIG['d']:
            v=next(r['rank1_variance'] for r in parameters if r['d']==d)
            for stream in range(8):
                checkpoint=OUT/'streams'/f'{family}_d{d}_stream{stream}.json'
                if checkpoint.exists():
                    saved=read(checkpoint);rows=saved['rows'];receipt=saved['receipt']
                    assert sha(OUT/'scores'/f'{family}_d{d}_stream{stream}.npz')==receipt['arrays_sha256']
                else:
                    write(OUT/'RUN_STATE.json',dict(status='running',family=family,d=d,stream=stream,pid=os.getpid(),time_utc=now()),exclusive=False)
                    rows,receipt=one_stream(family,d,v,stream);write(checkpoint,dict(time_utc=now(),rows=rows,receipt=receipt))
                stream_rows.extend(rows);receipts.append(receipt)
                print(family,d,stream,{r['method']:round(r['mean_F_minus_F0'],6) for r in rows},round(receipt['seconds'],2),flush=True)
    assert len(stream_rows)==160 and len(receipts)==32
    csvwrite(OUT/'STREAM_LOSSES.csv',stream_rows);write(OUT/'TIMING_AND_PRIMITIVES.json',receipts)
    critical=float(t.ppf(1-.05/(2*20),df=7));primary=[]
    for family,d,method in itertools.product(CONFIG['families'],CONFIG['d'],METHODS):
        values=np.array([r['mean_F_minus_F0'] for r in stream_rows if (r['family'],r['d'],r['method'])==(family,d,method)])
        assert len(values)==8;mean=float(values.mean());se=float(values.std(ddof=1)/np.sqrt(8));lo=mean-critical*se;hi=mean+critical*se
        gate=1. if family=='NEG' and method.startswith('original') else .003
        primary.append(dict(family=family,d=d,method=method,n_streams=8,mean=mean,SE=se,tcrit=critical,lower=lo,upper=hi,
                            SE_gate=gate,precision_pass=bool(se<=gate),true_theta_worse_than_independent=bool(se<=gate and lo>.005),
                            true_theta_better_than_independent=bool(se<=gate and hi<-.005)))
    assert len(primary)==20;csvwrite(OUT/'PRIMARY.csv',primary)
    get=lambda f,m:next(r for r in primary if (r['family'],r['d'],r['method'])==(f,128,m))
    bad=get('NEG','original_IID100');repaired=get('NEG','count_RB_IID10');positive=get('POS_RANK1','original_IID100')
    decision=dict(time_utc=now(),status='complete_fixed_population_objective_witness',primary_contrasts=20,
                  NEG_d128_independent_beats_true_original100=bad['true_theta_worse_than_independent'],
                  NEG_d128_true_beats_independent_RB10=repaired['true_theta_better_than_independent'],
                  POS_d128_true_beats_independent_original100=positive['true_theta_better_than_independent'],
                  main_witness_pass=bool(bad['true_theta_worse_than_independent'] and repaired['true_theta_better_than_independent'] and positive['true_theta_better_than_independent']),
                  precision_pass_count=sum(r['precision_pass'] for r in primary),fitted_models=0,
                  same_Q_before_after_RB=True,logchoose_retained=True,not_same_wallclock=True,
                  global_optimality_claim=False,natural_data_confidence_interval=False,
                  numerical_original_formula_max_error=max(r['original_formula_error'] for r in receipts),
                  numerical_PB_normalization_max_error=max(r['PB_normalization_error'] for r in receipts),
                  numerical_POS_particle_identity_max_error=max(r['POS_particle_identity_error'] for r in receipts))
    write(OUT/'DECISION.json',decision);check_freeze()
    write(OUT/'RUN_STATE.json',dict(status='complete',active_jobs=[],time_utc=now()),exclusive=False)
    files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.relative_to(OUT).parts[0]!='scores')
    write(OUT/'COMPLETE.json',dict(time_utc=now(),files={p.relative_to(ROOT).as_posix():sha(p) for p in files}))
    print(decision,flush=True)
def check():
    check_freeze()
    for name,h in read(OUT/'COMPLETE.json')['files'].items():assert sha(ROOT/name)==h,name
    for p in (OUT/'streams').glob('*.json'):
        receipt=read(p)['receipt'];name=f"{receipt['family']}_d{receipt['d']}_stream{receipt['stream']}.npz"
        assert sha(OUT/'scores'/name)==receipt['arrays_sha256']
    print('pass_T_frozen_M2_parameters_complete_streams_and_decisions')
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['smoke','freeze','run','check'])
    parser.add_argument('--parameters');parser.add_argument('--parameter-seal');args=parser.parse_args()
    try:
        if args.phase=='freeze':freeze(args.parameters,args.parameter_seal)
        elif args.phase=='run':
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=2):run()
        else:globals()[args.phase]()
    except Exception:
        if OUT.exists():
            stamp=time.time_ns();snapshot=OUT/f'FAILURE_{stamp}_source.py';snapshot.write_bytes(Path(__file__).read_bytes())
            write(OUT/f'FAILURE_{stamp}.json',dict(time_utc=now(),phase=args.phase,traceback=traceback.format_exc(),source_sha256=sha(__file__),source_snapshot=snapshot.name))
        raise
