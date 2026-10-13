"""Frozen finite-label / shared-bank raw-versus-collapsed learning experiment."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, os, sys, time, traceback
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '2'
import numpy as np
import scipy
from scipy.special import logsumexp, expit, xlogy
from scipy.stats import t, beta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from script import finite_learning_col as core
DOC = ROOT/'docs/08_research_研究工作/有限样本学习与均值自由度_20261001'
OUT = ROOT/'artifacts/finite_learning/RUN'
REF = ROOT/'artifacts/finite_learning/REF'
BASE = '994765369015e1ce2dd925ec2a28dc2fa7a875cc'
CFG = dict(dimensions=[16,64], streams=8, c=.9, truth=[6.,2.],
    axis=[1.,1.5,2.,3.,4.,6.,8.], N=[256,4096], B=[16,128],
    raw_K_multiplier=1, col_K_multipliers=[1,4], confirm_N=1024, confirm_B=128,
    seed=8416100100, label_chunk=256, bank_chunk=4,
    tie=1e-12, near=.001, risk_alpha=.025, risk_family=32, risk_SE=.001,
    score_alpha=.025, score_family=24, raw_score_SE_per_d=.01, col_score_SE=.001,
    floor_alpha=.05, floor_family=2,
    material_improvement=.002, noninferiority_margin=.001,
    formula_gate=1e-10, reference_risk_roundoff_gate=1e-10)
PARAMS = [(u,v) for u in CFG['axis'] for v in CFG['axis']]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p): return Path(p).relative_to(ROOT).as_posix()
def read(p): return json.loads(Path(p).read_text('utf-8'))
def now(): return datetime.now(timezone.utc).isoformat()
def write(p, x, replace=False):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w' if replace else 'x', encoding='utf-8', newline='\n') as f:
        json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def verify(mapping):
    for name,h in mapping.items(): assert sha(ROOT/name)==h,name
def rng(seed): return np.random.Generator(np.random.PCG64(seed))
def seed(d,s,offset=0): return CFG['seed']+1000000*d+1000*s+offset

def labels(d,n,sd):
    g=rng(sd);z=g.standard_normal((n,1));e=g.standard_normal((n,d))
    rho=CFG['c']/(d-1)
    eta=(np.sqrt(rho)*z+np.sqrt(1-rho)*e)*np.repeat(CFG['truth'],d//2)
    return 2*(g.random((n,d))<expit(eta)).astype(np.int8)-1

def banks(d,b,sd):
    common=rng(sd).standard_normal((b,4*d))
    private=rng(sd+1).standard_normal((b,d,d))
    rho=CFG['c']/(d-1)
    z=np.sqrt(rho)*common[:,:d,None]+np.sqrt(1-rho)*private
    return common,private,z

def raw_bank_means(Y,z,params,ns):
    began=time.perf_counter();n,d=Y.shape;b,k,_=z.shape;m=d//2
    answer=np.zeros((len(ns),b,len(params)));peak=0
    for bs in range(0,b,CFG['bank_chunk']):
        zz=z[bs:bs+CFG['bank_chunk']];bb=len(zz)
        za=zz[:,:,:m];zb=zz[:,:,m:]
        ca={u:np.logaddexp(u*za/2,-u*za/2).sum(-1) for u in {x[0] for x in params}}
        cb={v:np.logaddexp(v*zb/2,-v*zb/2).sum(-1) for v in {x[1] for x in params}}
        for st in range(0,n,CFG['label_chunk']):
            yy=Y[st:st+CFG['label_chunk']];nn=len(yy)
            ta=(yy[:,:m]@za.reshape(bb*k,m).T/2).reshape(nn,bb,k)
            tb=(yy[:,m:]@zb.reshape(bb*k,m).T/2).reshape(nn,bb,k)
            for j,(u,v) in enumerate(params):
                lp=u*ta+v*tb-ca[u]-cb[v]
                loss=-logsumexp(lp,axis=-1)+np.log(k)
                peak=max(peak,lp.nbytes)
                for ni,N in enumerate(ns):
                    upto=min(nn,max(0,N-st))
                    if upto: answer[ni,bs:bs+bb,j]+=loss[:upto].sum(0)
    answer/=np.asarray(ns)[:,None,None]
    assert np.isfinite(answer).all()
    return answer,dict(seconds=time.perf_counter()-began,prior_vectors=b*k,
        likelihood_particle_candidate_terms=n*b*k*len(params),peak_working_array_bytes=peak)

def count_weights(Y,ns):
    d=Y.shape[1];m=d//2
    counts=np.column_stack(((Y[:,:m]+1).sum(1)//2,(Y[:,m:]+1).sum(1)//2))
    states,inv=np.unique(counts,axis=0,return_inverse=True)
    weights=np.array([np.bincount(inv[:N],minlength=len(states))/N for N in ns])
    return states,weights,counts

def col_bank_means(Y,common,params,ns):
    began=time.perf_counter();d=Y.shape[1];m=d//2;b,kmax=common.shape
    states,weights,_=count_weights(Y,ns);lp,lq=core.log_h(common,d)
    assert lp.shape==lq.shape==(7,b,kmax)
    result=np.zeros((2,len(ns),b,len(params)));peak=0
    ca,cb=states[:,0][None,None,:],states[:,1][None,None,:]
    for bs in range(0,b,CFG['bank_chunk']):
        be=min(b,bs+CFG['bank_chunk'])
        for j,(u,v) in enumerate(params):
            ui,vi=CFG['axis'].index(u),CFG['axis'].index(v)
            logp=ca*lp[ui,bs:be,:,None]+(m-ca)*lq[ui,bs:be,:,None]
            logp+=cb*lp[vi,bs:be,:,None]+(m-cb)*lq[vi,bs:be,:,None]
            peak=max(peak,logp.nbytes)
            for ki,K in enumerate([d,4*d]):
                loss=-logsumexp(logp[:,:K],axis=1)+np.log(K)
                result[ki,:,bs:be,j]=weights@loss.T
    assert np.isfinite(result).all()
    return result,dict(seconds=time.perf_counter()-began,common_particles=b*kmax,
        unique_count_states=len(states),kernel_count_particle_candidate_terms=len(states)*b*kmax*len(params),
        logsumexp_terms=len(states)*b*(d+4*d)*len(params),peak_working_array_bytes=peak)

def mle_curves(Y,ns):
    d=Y.shape[1];states,weights,_=count_weights(Y,ns)
    logp=np.asarray(core.reference(d)['log_pattern'])
    return -weights@logp[:,states[:,0],states[:,1]].T

def choice(curve):
    curve=np.asarray(curve);minimum=float(curve.min())
    index=int(np.flatnonzero(curve<=minimum+CFG['tie'])[0])
    return dict(index=index,scales=list(PARAMS[index]),objective=float(curve[index]),
        near_indices=np.flatnonzero(curve<=minimum+CFG['near']).tolist(),curve=curve.tolist())

def select(d,s):
    began=time.perf_counter();Y=labels(d,max(CFG['N']),seed(d,s))
    common,private,z=banks(d,max(CFG['B']),seed(d,s,10))
    raw,rr=raw_bank_means(Y,z,PARAMS,CFG['N'])
    col,cr=col_bank_means(Y,common,PARAMS,CFG['N']);mle=mle_curves(Y,CFG['N'])
    selected=[]
    for ni,N in enumerate(CFG['N']):
        selected.append(dict(method='MLE',N=N,B=None,K=None,**choice(mle[ni])))
        for B in CFG['B']:
            selected.append(dict(method='raw',N=N,B=B,K=d,**choice(raw[ni,:B].mean(0))))
            for ki,K in enumerate([d,4*d]):
                selected.append(dict(method='col',N=N,B=B,K=K,**choice(col[ki,ni,:B].mean(0))))
    assert len(selected)==14
    path=OUT/'local'/f'train_d{d}_s{s}.npz';path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path,Y=Y,common=common,private=private,raw=raw,col=col,mle=mle)
    record=dict(d=d,stream=s,selected=selected,raw_receipt=rr,col_receipt=cr,
        array=rel(path),array_sha256=sha(path),elapsed_seconds=time.perf_counter()-began)
    write(OUT/'training'/f'd{d}_s{s}.json',record)
    print(json.dumps(dict(stage='training',d=d,stream=s,seconds=record['elapsed_seconds'],
        highN_highB=[(r['method'],r['K'],r['scales']) for r in selected if r['N']==4096 and r['B'] in [None,128]])),flush=True)
    return record

def confirm(d,s,selected):
    indices=sorted({r['index'] for r in selected}|{PARAMS.index((6.,2.)),0})
    params=[PARAMS[i] for i in indices];Y=labels(d,CFG['confirm_N'],seed(d,s,100))
    common,private,z=banks(d,CFG['confirm_B'],seed(d,s,110))
    raw,rr=raw_bank_means(Y,z,params,[len(Y)])
    col,cr=col_bank_means(Y,common,params,[len(Y)])
    path=OUT/'local'/f'confirm_d{d}_s{s}.npz'
    np.savez_compressed(path,Y=Y,common=common,private=private,raw=raw,col=col)
    record=dict(d=d,stream=s,indices=indices,raw_means=raw[0].mean(0).tolist(),
        col_means=col[:,0].mean(1).tolist(),raw_receipt=rr,col_receipt=cr,
        array=rel(path),array_sha256=sha(path))
    write(OUT/'confirmation'/f'd{d}_s{s}.json',record)
    print(json.dumps(dict(stage='confirmation',d=d,stream=s,candidates=len(indices))),flush=True)
    return record

def interval(values,family,alpha,se_gate):
    x=np.asarray(values);mean=float(x.mean());se=float(x.std(ddof=1)/np.sqrt(len(x)))
    rad=float(t.ppf(1-alpha/(2*family),len(x)-1))*se
    return dict(values=x.tolist(),mean=mean,SE=se,lower=mean-rad,upper=mean+rad,
        precision_pass=bool(se<=se_gate),SE_gate=se_gate,t_is_approximation=True)

def risk_tables():
    rows={}
    for d in CFG['dimensions']:
        p=np.asarray(core.reference(d)['pmf']);assert p.shape==(49,d//2+1,d//2+1)
        truth=p[PARAMS.index((6.,2.))]
        risk=np.sum(xlogy(truth[None],truth[None]/p),axis=(1,2))
        assert np.min(risk)>=-CFG['reference_risk_roundoff_gate']
        total=np.add.outer(np.arange(d//2+1),np.arange(d//2+1));spin=2*total-d
        events=[spin**2<=d,spin>np.sqrt(d),spin < -np.sqrt(d)]
        probs=np.array([p[:,a].sum(1) for a in events]).T
        pstar=probs[PARAMS.index((6.,2.))];brier=(probs-pstar)**2
        logloss=xlogy(pstar,pstar/probs)+xlogy(1-pstar,(1-pstar)/(1-probs))
        rows[d]=dict(joint_KL=risk.tolist(),event_probabilities=probs.tolist(),
            event_brier=brier.tolist(),event_logloss=logloss.tolist(),teacher_events=pstar.tolist())
    return rows

def find(record,method,N,B=None,K=None):
    return next(x for x in record['selected'] if (x['method'],x['N'],x['B'],x['K'])==(method,N,B,K))

def collect(training,confirmation):
    tables=risk_tables();contrasts=[];scores=[];modelrows=[]
    for (d,s),r in training.items():
        for q in r['selected']:
            j=q['index'];modelrows.append(dict(d=d,stream=s,**q,
                reference_joint_KL=tables[d]['joint_KL'][j],event_probabilities=tables[d]['event_probabilities'][j],
                event_brier=tables[d]['event_brier'][j],event_logloss=tables[d]['event_logloss'][j]))
    for d in CFG['dimensions']:
        risks=np.asarray(tables[d]['joint_KL'])
        for N in CFG['N']:
            for B in CFG['B']:
                values={k:[] for k in ['raw_minus_col_d','col_d_minus_MLE','col_4d_minus_MLE','col_d_minus_col_4d']}
                fv={k:[] for k in ['raw_truth_minus_selected','col_d_selected_minus_truth','col_4d_selected_minus_truth']}
                for s in range(8):
                    tr=training[d,s];cf=confirmation[d,s]
                    ix=[find(tr,'raw',N,B,d)['index'],find(tr,'col',N,B,d)['index'],
                        find(tr,'col',N,B,4*d)['index'],find(tr,'MLE',N)['index']]
                    a,b,c,m=risks[ix]
                    for name,v in zip(values,[a-b,b-m,c-m,b-c]):values[name].append(float(v))
                    truth=cf['indices'].index(PARAMS.index((6.,2.)))
                    for name,curve,j,direction in [('raw_truth_minus_selected',cf['raw_means'],ix[0],-1),
                        ('col_d_selected_minus_truth',cf['col_means'][0],ix[1],1),
                        ('col_4d_selected_minus_truth',cf['col_means'][1],ix[2],1)]:
                        fv[name].append(direction*(curve[cf['indices'].index(j)]-curve[truth]))
                for name,v in values.items():
                    contrasts.append(dict(d=d,N=N,B=B,comparison=name,
                        **interval(v,CFG['risk_family'],CFG['risk_alpha'],CFG['risk_SE'])))
                for name,v in fv.items():
                    gate=d*CFG['raw_score_SE_per_d'] if name.startswith('raw') else CFG['col_score_SE']
                    scores.append(dict(d=d,N=N,B=B,comparison=name,
                        **interval(v,CFG['score_family'],CFG['score_alpha'],gate)))
    assert len(contrasts)==32 and len(scores)==24 and len(modelrows)==224
    primaries=[]
    for r in contrasts:
        if r['N']==4096 and r['B']==128 and r['comparison']!='col_d_minus_col_4d':
            passed=r['precision_pass'] and (r['lower']>CFG['material_improvement'] if r['comparison']=='raw_minus_col_d' else r['upper']<CFG['noninferiority_margin'])
            primaries.append(dict(d=r['d'],comparison=r['comparison'],passed=bool(passed),
                kind='material_improvement' if r['comparison']=='raw_minus_col_d' else 'one_sided_noninferiority_not_equivalence'))
    floor_bounds=[]
    for d in CFG['dimensions']:
        k=sum(find(training[d,s],'raw',4096,128,d)['index']==0 for s in range(8))
        lo=float(beta.ppf(CFG['floor_alpha']/CFG['floor_family'],k,9-k)) if k else 0.
        risks=np.asarray(tables[d]['joint_KL']);rmin=float(risks.min())
        bound=rmin+lo*(float(risks[0])-rmin)
        floor_bounds.append(dict(d=d,N=4096,B=128,K=d,n=8,floor_selections=k,
            probability_lower=lo,reference_minimum=rmin,floor_reference_risk=float(risks[0]),
            expected_fixed_reference_risk_lower=bound,passed=bool(bound>CFG['material_improvement']),
            preregistered=True,post_hoc=False,binomial_family_alpha=CFG['floor_alpha'],binomial_family_size=2,
            strict_Gaussian_quadrature_error_certificate=False))
    count=dict(new_training_teacher_vectors=65536,new_confirmation_teacher_vectors=16384,
        new_teacher_vectors=81920,teacher_bit_outputs=3276800,
        training_raw_prior_vectors=81920,confirmation_raw_prior_vectors=81920,
        training_common_factor_draws=327680,confirmation_common_factor_draws=327680,
        common_factor_prefix_shared_with_raw_per_phase=81920,
        final_selection_endpoints=224,streams=16,new_neural_models=0,new_deployment_samples=0,
        raw_likelihood_particle_candidate_terms=sum(r['raw_receipt']['likelihood_particle_candidate_terms'] for r in list(training.values())+list(confirmation.values())),
        col_count_particle_candidate_terms=sum(r['col_receipt']['kernel_count_particle_candidate_terms'] for r in list(training.values())+list(confirmation.values())),
        col_logsumexp_terms=sum(r['col_receipt']['logsumexp_terms'] for r in list(training.values())+list(confirmation.values())))
    write(OUT/'RISK_TABLES.json',tables);write(OUT/'MODEL_RESULTS.json',modelrows)
    write(OUT/'RISK_CONTRASTS.json',contrasts);write(OUT/'SCORE_CONTRASTS.json',scores)
    write(OUT/'FLOOR_PROBABILITY_BOUNDS.json',floor_bounds)
    write(OUT/'DECISION.json',dict(primary_readouts=primaries,
        additional_preregistered_floor_readouts=floor_bounds,
        risk_precision_pass=sum(r['precision_pass'] for r in contrasts),risk_total=32,
        score_precision_pass=sum(r['precision_pass'] for r in scores),score_total=24,
        all_failures_retained=True,continuous_global_optimality_certified=False,
        strict_numerical_integration_certificate=False,reference_conditional=True))
    write(OUT/'COUNTS.json',count);return primaries

def smoke():
    d=4;b=2;k=4;z=(np.arange(b*k*d).reshape(b,k,d)-13)/9
    Y=np.array([[1,-1,1,-1],[-1,-1,1,1],[1,1,-1,1]],dtype=np.int8)
    params=[(1.,1.),(2.,1.)];actual,_=raw_bank_means(Y,z,params,[1,3])
    expected=np.empty_like(actual)
    for j,(u,v) in enumerate(params):
        eta=z*np.array([u,u,v,v]);lp=-np.logaddexp(0,-Y[:,None,None,:]*eta).sum(-1)
        loss=-logsumexp(lp,axis=-1)+np.log(k)
        for ni,N in enumerate([1,3]):expected[ni,:,j]=loss[:N].mean(0)
    err=float(np.max(abs(actual-expected)));assert err<CFG['formula_gate']
    write(OUT/'SMOKE.json',dict(raw_formula_error=err,new_scientific_samples=0,RNG_calls=0))

def freeze():
    assert not OUT.exists()
    ref=read(REF/'RESULTS_FREEZE.json');verify(ref['files'])
    assert read(REF/'STATE.json')['status']=='complete'
    assert list(core.AXIS)==CFG['axis'] and core.CFG['c']==CFG['c']
    assert read(REF/'QUALIFICATION.json')['all_passed']
    for d in CFG['dimensions']:
        reference=core.reference(d)
        assert [tuple(p) for p in reference['parameters']]==PARAMS
        assert np.asarray(reference['pmf']).shape==(49,d//2+1,d//2+1)
    OUT.mkdir(parents=True)
    (OUT/'HANDOFF_INPUT.md').write_bytes((ROOT/'HANDOFF.md').read_bytes())
    write(OUT/'CONFIG.json',CFG);write(OUT/'ENVIRONMENT.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,threads=2))
    files=[Path(__file__),ROOT/'script/finite_learning_col.py',DOC/'PLAN_RUN.md',
        OUT/'HANDOFF_INPUT.md',OUT/'CONFIG.json',OUT/'ENVIRONMENT.json',REF/'RESULTS_FREEZE.json']
    write(OUT/'FREEZE.json',dict(utc=now(),baseline_commit=BASE,before_scientific_RNG=True,
        files={rel(p):sha(p) for p in files},reference_files=ref['files']))
    write(OUT/'STATE.json',dict(status='frozen',active_jobs=[]));print('RUN frozen',flush=True)

def run():
    freeze_data=read(OUT/'FREEZE.json');verify(freeze_data['files']);verify(freeze_data['reference_files'])
    assert read(OUT/'STATE.json')['status']=='frozen';smoke()
    write(OUT/'START.json',dict(utc=now(),source_freeze_sha256=sha(OUT/'FREEZE.json')))
    training={};confirmation={}
    for d in CFG['dimensions']:
        for s in range(8):
            write(OUT/'STATE.json',dict(status='running',phase='training',d=d,stream=s,pid=os.getpid()),True)
            training[d,s]=select(d,s)
    write(OUT/'SELECTION_SEAL.json',dict(utc=now(),before_confirmation_RNG=True,
        files={rel(p):sha(p) for p in (OUT/'training').glob('*.json')},
        local_arrays={rel(p):sha(p) for p in (OUT/'local').glob('train_*.npz')}))
    for d in CFG['dimensions']:
        for s in range(8):
            write(OUT/'STATE.json',dict(status='running',phase='confirmation',d=d,stream=s,pid=os.getpid()),True)
            confirmation[d,s]=confirm(d,s,training[d,s]['selected'])
    primary=collect(training,confirmation)
    files=[p for p in OUT.rglob('*') if p.is_file() and 'local' not in p.parts and p.name!='STATE.json']
    write(OUT/'RESULTS_FREEZE.json',dict(utc=now(),source_freeze_sha256=sha(OUT/'FREEZE.json'),
        files={rel(p):sha(p) for p in files},local_arrays={rel(p):sha(p) for p in (OUT/'local').glob('*.npz')}))
    write(OUT/'STATE.json',dict(status='complete',active_jobs=[]),True)
    print(json.dumps(dict(completed=True,primary=primary)),flush=True)

def check(workspace):
    source=read(OUT/'FREEZE.json');verify(source['files']);verify(source['reference_files'])
    results=read(OUT/'RESULTS_FREEZE.json');verify(results['files'])
    assert results['source_freeze_sha256']==sha(OUT/'FREEZE.json')
    assert read(OUT/'STATE.json')['status']=='complete'
    if workspace:verify(results['local_arrays'])
    print(json.dumps(dict(passed=True,workspace=workspace,science_not_reexecuted=True)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','run','check']);p.add_argument('--workspace',action='store_true');a=p.parse_args()
    try:
        {'freeze':freeze,'run':run,'check':lambda:check(a.workspace)}[a.phase]()
    except Exception as exc:
        if a.phase=='run' and OUT.exists() and not (OUT/'RESULTS_FREEZE.json').exists():
            write(OUT/f'FAILURE_{time.time_ns()}.json',dict(error=repr(exc),traceback=traceback.format_exc()))
            write(OUT/'STATE.json',dict(status='stopped_failure',active_jobs=[]),True)
        raise
