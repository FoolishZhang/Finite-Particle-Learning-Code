"""USE-v2: new-data latent-trait model selection, with locked train/validation/test roles."""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import argparse, hashlib, itertools, json, math, platform, subprocess, time, traceback
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, roots_hermitenorm
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/use_selection_v2_20261005'
DOC=ROOT/'docs/08_research_研究工作/USE_v2_20261005'
BASE='f35a50628d43eab458bcb23a0660adf2986e2ec1'
HANDOFF='dc239cbb0d92ba9e3df1efc05d5918086e2cd1db2df2c5ede5b86c8ed6d2d869'
Y=np.array(list(itertools.product((0,1),repeat=8)),dtype=np.float64)
POWERS=2**np.arange(7,-1,-1)
CASES=('NULL','WEAK','STRONG')
CANDIDATES=('IND','L0','L002','L020')
METHODS=('RAW100','OUT16','INNER1600','JVI100','GH')
LAMBDAS=(0.,.002,.02)
MU=np.array([-.6,-.2,.2,.6,-.4,.4,0.,0.])
LOADING=np.array([[1.,0.],[.8,.3],[.8,-.3],[0.,1.],[.3,.8],[-.3,.8],[0.,0.],[0.,0.]])
CONFIG=dict(seed=202610052010,cases=list(CASES),strengths=[0.,.65,1.],reps=32,
    train=8192,validation=8192,test=8192,dimension=8,latent=2,train_order=20,
    reference_orders=[40,60,80],main_reference=60,K=100,total_prior=1600,groups=16,
    lambdas=list(LAMBDAS),starts=2,bias_bounds=[-2.5,2.5],loading_bounds=[-1.5,1.5],
    optimizer=dict(maxiter=180,maxfun=400,maxls=25,ftol=1e-12,gtol=1e-7),hard_calls=450,
    pmf_tolerance=2e-7,risk_tolerance=2e-6,mass_tolerance=2e-12,tie_tolerance=1e-12,
    audit_reps=[0,8,16,24],audit_order=72,gradient_audit_reps=[0],alpha=.05,family=15,df=31,
    practical_epsilon=.0002,SE_limit=.0002,workers=4,threads=1)


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rel(p):return Path(p).relative_to(ROOT).as_posix()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


@lru_cache(None)
def rule(order):
    x,w=roots_hermitenorm(order);w=w/math.sqrt(2*math.pi)
    nodes=np.array(list(itertools.product(x,x)));weights=np.outer(w,w).ravel()
    return nodes,weights


def unpack(theta):return np.asarray(theta[:8]),np.asarray(theta[8:]).reshape(8,2)


def objective(theta,frequency,penalty,order=20):
    b,W=unpack(theta);z,w=rule(order);logits=z@W.T+b
    loga=logits@Y.T-np.logaddexp(0.,logits).sum(axis=1)[:,None]
    logjoint=loga+np.log(w)[:,None];logq=logsumexp(logjoint,axis=0)
    posterior=np.exp(logjoint-logq[None,:])*np.asarray(frequency)[None,:]
    residual=posterior@Y-posterior.sum(axis=1)[:,None]*expit(logits)
    gradient=np.concatenate((-residual.sum(axis=0),(-residual.T@z+penalty*W).ravel()))
    nll=-float(np.asarray(frequency)@logq)
    value=nll+penalty*float(np.sum(W*W))/2
    assert np.isfinite(value) and np.isfinite(gradient).all()
    return value,gradient


def pmf(theta,order):
    b,W=unpack(theta);z,w=rule(order);logits=z@W.T+b
    loga=logits@Y.T-np.logaddexp(0.,logits).sum(axis=1)[:,None]
    q=np.exp(logsumexp(loga+np.log(w)[:,None],axis=0));mass=float(q.sum())
    return dict(order=order,pmf=(q/mass).tolist(),raw_mass=mass,mass_error=abs(mass-1),nodes=order**2)


def frequency(labels):return np.bincount(labels@POWERS,minlength=256).astype(float)/len(labels)


def validation_scores(theta,prior):
    b,W=unpack(theta);logits=prior@W.T+b
    loga=logits@Y.T-np.logaddexp(0.,logits).sum(axis=1)[:,None]
    blocks=math.log(100)-logsumexp(loga.reshape(16,100,256),axis=1)
    low=blocks[0];outside=blocks.mean(axis=0);inside=math.log(1600)-logsumexp(loga,axis=0)
    small=loga[:100]
    prefix=np.logaddexp.accumulate(small,axis=0)
    suffix=np.logaddexp.accumulate(small[::-1],axis=0)[::-1]
    left=np.vstack((np.full((1,256),-np.inf),prefix[:-1]))
    right=np.vstack((suffix[1:],np.full((1,256),-np.inf)))
    loo=math.log(99)-np.logaddexp(left,right)
    jackknife=100*low-99*loo.mean(axis=0)
    result=np.stack((low,outside,inside,jackknife))
    assert np.isfinite(result).all()
    return result


def fit_candidates(train,random_start):
    freq=frequency(train);means=train.mean(axis=0);p=(train.sum(axis=0)+.5)/(len(train)+1)
    bias=np.clip(np.log(p/(1-p)),-2.5,2.5)
    covariance=train.astype(float).T@train/len(train)-np.outer(means,means)
    d=p*(1-p);matrix=covariance/np.outer(d,d);np.fill_diagonal(matrix,0.)
    eig,U=np.linalg.eigh(matrix);indices=np.argsort(eig)[-2:][::-1]
    spectral=np.clip(U[:,indices]*np.sqrt(np.maximum(eig[indices],0.)), -.8,.8)
    starts=[np.concatenate((bias,spectral.ravel())),np.concatenate((bias,random_start.ravel()))]
    candidates=[dict(name='IND',p=p.tolist(),fit='Jeffreys smoothed independent marginals')]
    traces={};fit_rows=[]
    bounds=[(-2.5,2.5)]*8+[(-1.5,1.5)]*16
    for ci,penalty in enumerate(LAMBDAS,start=1):
        endings=[]
        for si,initial in enumerate(starts):
            xs=[];fs=[];gs=[]
            def fun(x):
                assert len(fs)<450,'Frozen objective-call budget exceeded'
                value,gradient=objective(x,freq,penalty)
                xs.append(x.copy());fs.append(value);gs.append(gradient.copy())
                return value,gradient
            tick=time.perf_counter()
            result=minimize(fun,initial.copy(),jac=True,method='L-BFGS-B',bounds=bounds,options=CONFIG['optimizer'])
            assert np.isfinite(result.x).all() and np.isfinite(result.fun)
            assert np.max(abs(result.x[:8]))<=2.5+1e-12 and np.max(abs(result.x[8:]))<=1.5+1e-12
            key=f'{CANDIDATES[ci]}_start{si}'
            traces[key+'_x']=np.asarray(xs);traces[key+'_loss']=np.asarray(fs);traces[key+'_gradient']=np.asarray(gs)
            row=dict(name=CANDIDATES[ci],penalty=penalty,start=si,initial=initial.tolist(),theta=result.x.tolist(),
                objective=float(result.fun),gradient=np.asarray(result.jac).tolist(),success=bool(result.success),
                status=int(result.status),message=str(result.message),nit=int(result.nit),nfev=int(result.nfev),
                saved_calls=len(fs),seconds=time.perf_counter()-tick)
            assert result.nfev==len(fs)
            endings.append(row);fit_rows.append(row)
        chosen=min(range(2),key=lambda i:(endings[i]['objective'],i))
        candidates.append(dict(name=CANDIDATES[ci],theta=endings[chosen]['theta'],penalty=penalty,
                               selected_start=chosen,training_objective=endings[chosen]['objective']))
    return candidates,fit_rows,traces,freq


def teacher_reference():
    rows=[]
    for ci,name in enumerate(CASES):
        theta=np.concatenate((MU,(CONFIG['strengths'][ci]*LOADING).ravel()))
        variants=[pmf(theta,n) for n in (40,60,80)];P=np.array(variants[1]['pmf'])
        errors=[float(np.max(abs(np.array(v['pmf'])-P))) for v in variants]
        kl_errors=[abs(float(np.array(v['pmf'])@np.log(np.array(v['pmf'])/P))) for v in variants]
        assert max(errors)<=2e-7 and max(kl_errors)<=2e-6 and max(v['mass_error'] for v in variants)<=2e-12
        rows.append(dict(case=name,theta=theta.tolist(),main_pmf=P.tolist(),variants=variants,
                         pmf_errors=errors,KL_errors=kl_errors,qualified=True))
    result=dict(records=rows,source_sha256=sha(__file__),freeze_sha256=sha(OUT/'FREEZE.json'))
    save(OUT/'TEACHERS.json',result)
    return result


def select(scores):
    minimum=float(np.min(scores));return int(np.flatnonzero(np.asarray(scores)<=minimum+1e-12)[0])


def stream_run(ci,rep):
    start=time.perf_counter();folder=OUT/CASES[ci]/f'rep_{rep:02}';folder.mkdir(parents=True,exist_ok=False)
    try:
        rng={role:np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([CONFIG['seed'],ci,rep,role]))) for role in range(9)}
        perm=rng[0].permutation(8);b=MU[perm];W=(CONFIG['strengths'][ci]*LOADING)[perm]
        datasets={};draw_hashes={}
        for name,rz,ru in [('train',1,2),('validation',3,4),('test',5,6)]:
            z=rng[rz].standard_normal((8192,2));u=rng[ru].random((8192,8))
            datasets[name]=(u<expit(z@W.T+b)).astype(np.int64)
            draw_hashes[name]=dict(Gaussian=ah(z),Uniform=ah(u),labels=ah(datasets[name].astype(np.uint8)))
        random_start=.15*rng[7].standard_normal((8,2))
        datafile=folder/'DATA.npz';np.savez_compressed(datafile,**{name:labels.astype(np.uint8) for name,labels in datasets.items()})
        candidates,fit_rows,traces,train_freq=fit_candidates(datasets['train'],random_start)
        tracefile=folder/'TRAINING_TRACE.npz';np.savez_compressed(tracefile,**traces)
        trained=dict(case=CASES[ci],rep=rep,permutation=perm.tolist(),random_start=random_start.tolist(),
            random_start_sha256=ah(random_start),draw_hashes=draw_hashes,train_frequency=train_freq.tolist(),
            candidates=candidates,fits=fit_rows,data_file=rel(datafile),data_sha256=sha(datafile),
            trace_file=rel(tracefile),trace_sha256=sha(tracefile),source_sha256=sha(__file__),
            freeze_sha256=sha(OUT/'FREEZE.json'),locked_before_validation=True)
        save(folder/'TRAINED.json',trained)
        # Only fitted models and independent validation data enter these selectors.
        val_freq=frequency(datasets['validation']);references=[]
        for candidate in candidates:
            if candidate['name']=='IND':
                p=np.array(candidate['p']);q=np.prod(np.where(Y,p,1-p),axis=1)
                references.append(dict(name='IND',main_pmf=q.tolist(),variants=[],qualified=True))
            else:
                variants=[pmf(candidate['theta'],n) for n in (40,60,80)]
                q=np.array(variants[1]['pmf']);errors=[float(np.max(abs(np.array(v['pmf'])-q))) for v in variants]
                qualified=max(errors)<=2e-7 and max(v['mass_error'] for v in variants)<=2e-12 and min(q)>0
                references.append(dict(name=candidate['name'],main_pmf=q.tolist(),variants=variants,pmf_errors=errors,qualified=qualified))
        prior=rng[8].standard_normal((1600,2));score_matrix=np.empty((5,4));by_pattern=np.empty((4,4,256))
        for j,candidate in enumerate(candidates):
            q=np.array(references[j]['main_pmf'])
            pattern=np.broadcast_to(-np.log(q),(4,256)).copy() if j==0 else validation_scores(candidate['theta'],prior)
            by_pattern[:,j]=pattern;score_matrix[:4,j]=pattern@val_freq;score_matrix[4,j]=-val_freq@np.log(q)
        choices=[select(row) for row in score_matrix]
        selection=dict(case=CASES[ci],rep=rep,trained_sha256=sha(folder/'TRAINED.json'),validation_frequency=val_freq.tolist(),
             score_matrix=score_matrix.tolist(),methods=list(METHODS),candidate_names=list(CANDIDATES),choices=choices,
             references=references,validation_prior_sha256=ah(prior),locked_before_test_and_teacher_risk=True)
        save(folder/'SELECTION.json',selection)
        # Teacher and test are first used for reporting AFTER selection is locked.
        teacher=read(OUT/'TEACHERS.json')['records'][ci]
        mapping=Y[:,np.argsort(perm)]@POWERS
        P=np.array(teacher['main_pmf'])[mapping.astype(int)]
        test_freq=frequency(datasets['test'])
        risks=[];test_losses=[];risk_checks=[]
        for reference in references:
            q=np.array(reference['main_pmf']);risk=float(P@np.log(P/q));risks.append(risk)
            test_losses.append(float(-test_freq@np.log(q)))
            changes=[abs(float(P@np.log(P/np.array(v['pmf'])))-risk) for v in reference['variants']]
            risk_checks.append(max(changes,default=0.))
        oracle=int(np.argmin(risks));selected_risks=np.array(risks)[choices]
        scoresfile=folder/'EVALUATION.npz';np.savez_compressed(scoresfile,pattern_scores=by_pattern,prior=prior)
        result=dict(case=CASES[ci],rep=rep,folder=rel(folder),trained_sha256=sha(folder/'TRAINED.json'),
            selection_sha256=sha(folder/'SELECTION.json'),evaluation_file=rel(scoresfile),evaluation_sha256=sha(scoresfile),
            candidate_risks=risks,candidate_test_NLL=test_losses,oracle_candidate=oracle,choices=choices,
            selected_risks=selected_risks.tolist(),selection_regret=(selected_risks-min(risks)).tolist(),
            selected_test_NLL=np.array(test_losses)[choices].tolist(),test_frequency=test_freq.tolist(),teacher_P=P.tolist(),
            risk_order_changes=risk_checks,numerically_qualified=bool(all(x['qualified'] for x in references) and max(risk_checks)<=2e-6),
            fits_converged=sum(x['success'] for x in fit_rows),fit_calls=sum(x['saved_calls'] for x in fit_rows),
            seconds=time.perf_counter()-start)
        save(folder/'RESULT.json',result)
        return result
    except Exception:
        save(folder/'ERROR.json',dict(case=CASES[ci],rep=rep,traceback=traceback.format_exc(),automatic_retry=False));raise


def summary(records):
    critical=float(t.ppf(1-.05/(2*15),31));families=[];descriptive=[]
    for case in CASES:
        rows=[r for r in records if r['case']==case];assert len(rows)==32
        risk=np.array([r['selected_risks'] for r in rows]);ind=np.array([r['candidate_risks'][0] for r in rows])
        differences=[risk[:,0]-risk[:,4],risk[:,1]-risk[:,2],risk[:,2]-risk[:,4],risk[:,3]-risk[:,4],risk[:,4]-ind]
        for label,values in zip(('RAW100-GH','OUT16-INNER1600','INNER1600-GH','JVI100-GH','GH-IND'),differences):
            mean=float(values.mean());se=float(values.std(ddof=1)/math.sqrt(32));interval=[mean-critical*se,mean+critical*se]
            families.append(dict(case=case,contrast=label,paired_values=values.tolist(),mean=mean,SE=se,interval=interval,
                precision_passed=se<=.0002,positive_material_loss=bool(interval[0]>0 and mean>=.0002),
                below_practical_excess=bool(interval[1]<.0002),useful_joint_gain=bool(interval[1]<-.0002)))
        descriptive.append(dict(case=case,mean_candidate_risks=np.mean([r['candidate_risks'] for r in rows],axis=0).tolist(),
            mean_selected_risks=risk.mean(axis=0).tolist(),mean_regrets=np.mean([r['selection_regret'] for r in rows],axis=0).tolist(),
            mean_test_NLL=np.mean([r['selected_test_NLL'] for r in rows],axis=0).tolist(),
            choice_counts={method:np.bincount([r['choices'][j] for r in rows],minlength=4).tolist() for j,method in enumerate(METHODS)},
            oracle_counts=np.bincount([r['oracle_candidate'] for r in rows],minlength=4).tolist()))
    return dict(config=CONFIG,freeze_sha256=sha(OUT/'FREEZE.json'),records=records,family=families,descriptive=descriptive,
        methods=list(METHODS),candidates=list(CANDIDATES),numerically_qualified=all(r['numerically_qualified'] for r in records),
        converged_fits=sum(r['fits_converged'] for r in records),actual_training_calls=sum(r['fit_calls'] for r in records),
        accounting=dict(new_independent_streams=96,new_observations=96*3*8192,data_Gaussian=96*3*8192*2,data_Uniform=96*3*8192*8,
            permutations=96,optimizer_runs=576,trained_joint_candidates=288,closed_form_independent_fits=96,
            validation_prior_vectors=96*1600,validation_Gaussian=96*1600*2,main_reference_nodes=(3+288)*sum(n*n for n in (40,60,80)),
            independent_reference_nodes=36*72**2,training_nodes_actual=sum(r['fit_calls'] for r in records)*400),
        scope='New-data controlled latent-trait selection workflow; development t intervals across 32 independent streams per generating law. Not full SCBM or a natural-data application.')


def source_inputs():
    sources=[Path(__file__),ROOT/'script/uselect_audit.py',DOC/'CONTRACT.md',DOC/'QUALIFICATION.md',DOC/'HANDOFF_DISPOSITION.md']
    inputs=[OUT/'HANDOFF_INPUT.md',OUT/'SOURCES.json']
    return {rel(p):sha(p) for p in sources},{rel(p):sha(p) for p in inputs}


def freeze():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==BASE
    assert sha(ROOT/'HANDOFF.md')==HANDOFF
    q=read(OUT/'NUMERICAL_QUALIFICATION.json')
    assert q['passed'] and q['producer_sha256']==sha(__file__) and q['audit_sha256']==sha(ROOT/'script/uselect_audit.py')
    sources,inputs=source_inputs()
    save(OUT/'FREEZE.json',dict(authorized=True,created_utc=datetime.now(timezone.utc).isoformat(),baseline=BASE,
         config=CONFIG,sources=sources,inputs=inputs,qualification_sha256=sha(OUT/'NUMERICAL_QUALIFICATION.json'),
         handoff_sha256=HANDOFF,handoff_use='reference',environment=q['environment']))
    print(json.dumps(dict(frozen=True,sha256=sha(OUT/'FREEZE.json'))))


def validate():
    f=read(OUT/'FREEZE.json');assert f['authorized'] and f['config']==CONFIG
    for group in ('sources','inputs'):
        for p,digest in f[group].items():assert sha(ROOT/p)==digest,p
    assert sha(OUT/'NUMERICAL_QUALIFICATION.json')==f['qualification_sha256']


def run():
    validate();tick=time.perf_counter()
    save(OUT/'EXECUTION.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),freeze_sha256=sha(OUT/'FREEZE.json'),automatic_retry=False))
    try:
        teacher_reference();records=[]
        with ProcessPoolExecutor(max_workers=4) as pool:
            for future in as_completed([pool.submit(stream_run,ci,rep) for ci in range(3) for rep in range(32)]):
                r=future.result();records.append(r)
                print(json.dumps(dict(case=r['case'],rep=r['rep'],qualified=r['numerically_qualified'],converged=r['fits_converged'],calls=r['fit_calls'],seconds=r['seconds'])),flush=True)
        records.sort(key=lambda r:(CASES.index(r['case']),r['rep']));validate()
        result=summary(records);result['seconds']=time.perf_counter()-tick
        save(OUT/'RESULTS.json',result);print(json.dumps(dict(completed=True,qualified=result['numerically_qualified'],seconds=result['seconds'])),flush=True)
    except Exception:
        save(OUT/'ERROR.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('freeze','run'))
    globals()[parser.parse_args().command]()
