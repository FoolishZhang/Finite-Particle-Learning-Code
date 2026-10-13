"""Independent audit of frozen-panel evaluation; no producer import/evaluate.

Replays existing inputs only. Mathematical first-block subsets are compared to
saved block-pattern means; actual API32 controls are separately re-invoked.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
import argparse,hashlib,importlib.util,itertools,json,math,time,traceback
import numpy as np
import torch
from scipy.special import expit,logsumexp
from scipy.stats import t as student_t
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1];SOURCE=Path(__file__).resolve()
BASE=ROOT/'artifacts/scbm_frozen_evaluation';EVAL=BASE/'EVAL';OUT=BASE/'AUDIT'
DOC=ROOT/'docs/08_research_研究工作/冻结表示的条件与总体目标_20261003'
OFF=ROOT/'reproducibility_snapshots/cbm_public_sources/SCBM/58c3e602cb11fb4ea9106226e0f9e58f50d5d10a/models/losses.py'
MU=np.array([-.6,-.2,.2,.6]);C0=np.array([[1.,.25,-.15,.1],[.25,1.,.1,-.15],[-.15,.1,1.,.25],[.1,-.15,.25,1.]])
D=C0-np.eye(4);PAT=np.array(list(itertools.product([0,1],repeat=4)),np.int8)
PROFILES=[('TAIL',1.),('TAIL',.5),('TAIL',0.),('LAST',1.),('LAST',0.)]
METRICS=['centered_score','price_t','path_t'];POINTS=[0,10,14,20,21]
TOL=dict(mathematical=2e-10,API_score=5e-5,API_path=3e-6,arithmetic=2e-11)
_LOSS=None
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def require(ok,why):
    if not ok:raise RuntimeError(why)
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def now():return datetime.now(timezone.utc).isoformat()
def near(a,b,tol=TOL['arithmetic'],why='numeric'):
    a,b=np.asarray(a,float),np.asarray(b,float);require(a.shape==b.shape,why+' shape')
    err=float(np.max(abs(a-b))) if a.size else 0.
    require(np.isfinite(a).all() and np.isfinite(b).all() and err<=tol,why+f': {err} > {tol}')
    return err
def official(K):
    global _LOSS
    if _LOSS is None:
        spec=importlib.util.spec_from_file_location('scbmeval_independent_official',OFF);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        _LOSS=m.SCBLoss(alpha=1.,config=SimpleNamespace(reg_precision='l1',reg_weight=1.,num_monte_carlo=K,training_mode='joint'))
    return _LOSS

def active_math(e,t):
    l=np.linalg.cholesky(np.eye(4)+(1+t)*D);left=np.linalg.solve(l,D);a=np.linalg.solve(l,left.T).T
    phi=np.tril(a);phi[np.diag_indices(4)]*=.5;dl=l@phi
    x=e@l.T+MU;dx=e@dl.T
    logp=[];cross=[];path=[]
    for y in PAT:
        s=2*y.astype(float)-1.;signed=x*s
        logp.append(-np.logaddexp(0.,-signed).sum(axis=-1))
        mark=s*expit(-signed)
        cross.append(sum(D[i,j]*mark[:,:,i]*mark[:,:,j] for i in range(4) for j in range(i+1,4)))
        path.append(np.sum(mark*dx,axis=-1))
    return dict(logp=np.stack(logp,axis=-1),cross=np.stack(cross,axis=-1),path=np.stack(path,axis=-1))
def noise_math(noise,scale,alpha):
    if alpha==0:return np.zeros(noise.shape[:2])
    return noise.shape[2]*math.log(2)-np.logaddexp(0.,noise*(alpha*np.asarray(scale))).sum(axis=-1)
def score_math(terms,lognoise):
    logs=terms['logp']+lognoise[:,:,None];norm=logsumexp(logs,axis=1);p=np.exp(logs-norm[:,None,:])
    # Different algebra from the producer's pi*(1-pi) form.
    price=np.sum(p*p*terms['cross'],axis=1)-np.sum(p*terms['cross'],axis=1)
    return dict(centered_score=math.log(logs.shape[1])-norm,price_t=price,path_t=-np.sum(p*terms['path'],axis=1))
def native(e,noise,scale,t,y,yn):
    tt=torch.tensor(t,dtype=torch.float32,requires_grad=True);eye=torch.eye(4,dtype=torch.float32)
    cov=eye+(1+tt)*(torch.tensor(C0,dtype=torch.float32)-eye)
    x=torch.tensor(e,dtype=torch.float32)@torch.linalg.cholesky(cov).T+torch.tensor(MU,dtype=torch.float32)
    z=torch.tensor(noise,dtype=torch.float32)*torch.tensor(scale,dtype=torch.float32)
    p=torch.sigmoid(torch.cat([x,z],dim=-1).permute(0,2,1))
    labels=torch.tensor(np.concatenate([y,yn],axis=1),dtype=torch.int64)
    loss=official(e.shape[1]).compute_concept_loss(p,labels);gradient=torch.autograd.grad(loss,tt)[0]
    return dict(total_score=float(loss.detach()),centered_score=float(loss.detach())-noise.shape[2]*math.log(2),path_t=float(gradient),
                sigmoid_zero_evaluations=int((p.detach()==0).sum()),sigmoid_one_evaluations=int((p.detach()==1).sum()),
                probability_evaluations=p.numel(),maximum_absolute_logit=float(torch.cat([x,z],dim=-1).detach().abs().max()))

def check():
    e=np.sin(np.arange(2*3*4).reshape(2,3,4)*.17);nu=np.cos(np.arange(2*3*5).reshape(2,3,5)*.11);tau=np.full(5,.2)
    t=.12;noise=noise_math(nu,tau,1.);at=active_math(e,t);value=score_math(at,noise);h=1e-5
    fd=(score_math(active_math(e,t+h),noise)['centered_score']-score_math(active_math(e,t-h),noise)['centered_score'])/(2*h)
    patherr=near(fd,value['path_t'],2e-9,'deterministic path FD')
    # Independent mixed Hessian of fixed-particle log score, not path=Price.
    l=np.linalg.cholesky(np.eye(4)+(1+t)*D);x=e@l.T+MU;hs=.001;hh=np.zeros((2,16))
    def loss_at(z):
        logs=np.stack([-np.logaddexp(0.,-z*(2*y-1)).sum(axis=-1) for y in PAT],axis=-1)+noise[:,:,None]
        return math.log(3)-logsumexp(logs,axis=1)
    for k in range(3):
        for i in range(4):
            for j in range(i+1,4):
                v=[]
                for si,sj in [(1,1),(1,-1),(-1,1),(-1,-1)]:
                    z=x.copy();z[:,k,i]+=si*hs;z[:,k,j]+=sj*hs;v.append(loss_at(z))
                hh+=D[i,j]*(v[0]-v[1]-v[2]+v[3])/(4*hs*hs)
    pe=near(hh,value['price_t'],2e-7,'deterministic mixed Hessian Price')
    a=np.arange(32*16*5*22*2).reshape(32,16,5,22,2);chosen=(np.arange(16)*3+1)%22
    direct=np.array([[a[r,m,3,chosen[m],1] for m in range(16)] for r in range(20,32)])
    require(np.array_equal(a[20:,np.arange(16),3,chosen,1],direct),'unequal selection axes')
    return dict(passed=True,path_FD_error=patherr,Price_Hessian_error=pe,selection_axes_passed=True,new_scientific_draws=0,new_model_integrals=0)

def freeze():
    paths=[SOURCE,DOC/'AUDIT_PLAN.md',DOC/'EXPERIMENT_DESIGN.md',EVAL/'FREEZE.json',BASE/'MODEL_SELECTION.json',
           ROOT/'script/scbmeval_kernel.py',ROOT/'script/scbmeval_run.py',ROOT/'script/scbmeval_reference.py',ROOT/'script/scbmeval_summary.py',ROOT/'script/scbminv_reference.py',OFF]
    sel=read(BASE/'MODEL_SELECTION.json')
    paths.extend(ROOT/sel[k] for k in ['training_result_path','truth_reference_path'])
    paths.append(ROOT/'artifacts/scbm_scale_optimization/REFERENCE/RESULTS.json')
    save(OUT/'PROTOCOL.json',dict(created_utc=now(),inputs={rel(p):sha(p) for p in paths},tolerances=TOL,
         first_complete_block_banks=64,model_indices=[0,15],profiles=list(range(5)),points=POINTS,
         official_NUM_prefix_banks=4,official_API_replays=2304,deterministic_check=check(),new_model_integrals=0,automatic_retry=False))
    print(json.dumps(dict(source_sha256=sha(SOURCE),protocol_sha256=sha(OUT/'PROTOCOL.json'))),flush=True)
def bind():
    p=read(OUT/'PROTOCOL.json')
    for path,h in p['inputs'].items():require(sha(ROOT/path)==h,'audit frozen input '+path)
    return p

def old_panel():
    sel=read(BASE/'MODEL_SELECTION.json');train=read(ROOT/sel['training_result_path']);truth=read(ROOT/sel['truth_reference_path'])
    require(sha(ROOT/sel['training_result_path'])==sel['training_result_sha256'] and sha(ROOT/sel['truth_reference_path'])==sel['truth_reference_sha256'],'old source binding')
    require(sel['patterns']==PAT.tolist() and sel['population_weights']==next(x['main_pmf'] for x in truth['records'] if x['t']==0.),'pattern/population weights')
    require(sel['zero_index']==10 and sel['contrast_index']==14 and sel['grid_t']==np.linspace(-.75,.75,21).tolist(),'grid identity')
    labels={};require([m['stream'] for m in sel['models']]==list(range(16)),'all old models')
    for m in sel['models']:
        rec=next(r for r in train['records'] if r['stream']==m['stream']);p=ROOT/rec['input_file']
        require(sha(p)==rec['input_sha256']==sel['old_input_NPZ'][rec['input_file']],'old INPUT hash')
        for kind in ['TAIL','LAST']:require(m[kind]==rec['outputs']['FREE'][kind],'old physical model')
        with np.load(p,allow_pickle=False) as inp:
            y=inp['active_labels'];counts=[int(np.all(y==pattern,axis=1).sum()) for pattern in PAT]
            require(len(y)==m['N']==4096 and counts==m['empirical_counts'],'independent old pattern counts')
            require(np.array_equal(np.array(counts)/4096,m['empirical_weights']),'empirical weights')
            if m['stream'] in (0,15):labels[m['stream']]=(y[:4].copy(),inp['nuisance_labels'][:4].copy())
    return sel,labels

def replay(rep):
    rng={name:np.random.Generator(np.random.PCG64(np.random.SeedSequence([202610030400,rep,role]))) for name,role in [('active',0),('nuisance',1)]}
    initial={name:r.bit_generator.state for name,r in rng.items()};dig={name:hashlib.sha256() for name in rng};first=None
    for block in range(8):
        e=rng['active'].standard_normal((64,100,4));nu=rng['nuisance'].standard_normal((64,100,256))
        dig['active'].update(e.tobytes());dig['nuisance'].update(nu.tobytes())
        if block==0:first=(e.copy(),nu.copy())
    return first,{k:h.hexdigest() for k,h in dig.items()},initial,{k:r.bit_generator.state for k,r in rng.items()}

def bottom_math(e,nu,sel,data):
    term={};noise={};errors={k:0. for k in METRICS}
    for mi in (0,15):
        model=sel['models'][mi]
        for pi,(kind,alpha) in enumerate(PROFILES):
            key=(mi,kind,alpha)
            noise[key]=noise_math(nu,model[kind]['scales'],alpha)
            for point in POINTS:
                t=float(sel['grid_t'][point] if point<21 else model[kind]['t'])
                if t not in term:term[t]=active_math(e,t)
                got=score_math(term[t],noise[key])
                for name in METRICS:
                    errors[name]=max(errors[name],near(got[name].mean(axis=0),data['block_pattern_'+name][0,mi,pi,point],TOL['mathematical'],'first-block stored '+name))
    return errors

def api_controls(e,nu,sel,labels,record):
    e=e[:4];nu=nu[:4];scoremax=0.;gradmax=0.;mathmax=0.
    require({(x['model_index'],x['context']) for x in record['numerical_controls']}=={(m,k) for m in (0,15) for k in ['TAIL','LAST']},'NUM cases')
    require({(x['model_index'],x['context']) for x in record['canonical_controls']}=={(m,k) for m in (0,15) for k in ['TAIL','LAST']},'canonical cases')
    for mi in (0,15):
        model=sel['models'][mi];weights=np.column_stack([sel['population_weights'],model['empirical_weights']]);yold,ynold=labels[mi]
        for kind in ['TAIL','LAST']:
            t=model[kind]['t'];tau=np.array(model[kind]['scales']);mathval=score_math(active_math(e,t),noise_math(nu,tau,1.))
            core=next(r for r in record['numerical_controls'] if r['model_index']==mi and r['context']==kind)
            require(core['t']==t and core['targets']==['P','D'] and core['official_API_calls']==16,'NUM identity')
            calls=[native(e,nu,tau,t,np.tile(pattern,(4,1)),np.zeros((4,256),np.int8)) for pattern in PAT]
            for field in ['total_score','centered_score','path_t']:
                tol=TOL['API_path'] if field=='path_t' else TOL['API_score']
                error=near([r[field] for r in calls],core['official_pattern_'+field],tol,'official pattern '+field)
                if field=='path_t':gradmax=max(gradmax,error)
                else:scoremax=max(scoremax,error)
            for name in ['centered_score','path_t']:
                mathmax=max(mathmax,near(mathval[name].mean(axis=0),core['mathematical_pattern_'+name],TOL['mathematical'],'NUM mathematical pattern '+name))
            sd=(np.array(core['official_pattern_centered_score'])-np.array(core['mathematical_pattern_centered_score']))@weights
            gd=(np.array(core['official_pattern_path_t'])-np.array(core['mathematical_pattern_path_t']))@weights
            near(sd,core['weighted_centered_score_difference'],why='weighted NUM score');near(gd,core['weighted_path_t_difference'],why='weighted NUM path')
            require(core['score_passed']==(abs(sd)<=1e-4).tolist() and core['path_passed']==(abs(gd)<=2e-5).tolist(),'NUM original gates')
            require(core['passed']==bool(np.all(abs(sd)<=1e-4) and np.all(abs(gd)<=2e-5)),'NUM case gate')
            for field in ['sigmoid_zero_evaluations','sigmoid_one_evaluations']:require(core[field]==sum(r[field] for r in calls),'NUM saturation')
            control=next(r for r in record['canonical_controls'] if r['model_index']==mi and r['context']==kind)
            can=native(e,nu,tau,t,yold,np.zeros_like(ynold));orig=native(e,nu*(1.-2.*ynold[:,None,:]),tau,t,yold,ynold)
            for got,stored in [(can,control['canonical']),(orig,control['original_labels_sign_flipped_noise'])]:
                for key in ['total_score','centered_score','path_t']:
                    tol=TOL['API_path'] if key=='path_t' else TOL['API_score'];error=near(got[key],stored[key],tol,'signflip '+key)
                    if key=='path_t':gradmax=max(gradmax,error)
                    else:scoremax=max(scoremax,error)
                for key in ['sigmoid_zero_evaluations','sigmoid_one_evaluations','probability_evaluations']:require(got[key]==stored[key],'signflip counters')
            se=control['original_labels_sign_flipped_noise']['centered_score']-control['canonical']['centered_score'];ge=control['original_labels_sign_flipped_noise']['path_t']-control['canonical']['path_t']
            near(se,control['centered_score_difference'],why='signflip score diff');near(ge,control['path_t_difference'],why='signflip path diff')
            require(control['score_passed']==(abs(se)<=1e-4) and control['path_passed']==(abs(ge)<=2e-5) and control['passed']==(abs(se)<=1e-4 and abs(ge)<=2e-5),'signflip original gates')
    return dict(API_score=scoremax,API_path=gradmax,mathematical_NUM=mathmax)

def qualification(core,can):
    return dict(core_cases=len(core),core_cases_passed=sum(r['passed'] for r in core),weighted_score_checks=2*len(core),weighted_score_passed=sum(sum(r['score_passed']) for r in core),weighted_path_checks=2*len(core),weighted_path_passed=sum(sum(r['path_passed']) for r in core),maximum_weighted_score_difference=max(abs(v) for r in core for v in r['weighted_centered_score_difference']),maximum_weighted_path_difference=max(abs(v) for r in core for v in r['weighted_path_t_difference']),canonical_cases=len(can),canonical_cases_passed=sum(r['passed'] for r in can),maximum_canonical_score_difference=max(abs(r['centered_score_difference']) for r in can),maximum_canonical_path_difference=max(abs(r['path_t_difference']) for r in can))

def run():
    bind();prod=read(EVAL/'RESULTS.json');f=read(EVAL/'FREEZE.json');sel,labels=old_panel()
    require(prod['source_sha256']==sha(ROOT/'script/scbmeval_run.py') and prod['kernel_sha256']==sha(ROOT/'script/scbmeval_kernel.py') and prod['freeze_sha256']==sha(EVAL/'FREEZE.json'),'production bindings')
    require(prod['config']==f['config'] and prod['selection_sha256']==sha(BASE/'MODEL_SELECTION.json'),'config/selection')
    save(OUT/'EVAL_EXECUTION.json',dict(start_utc=now(),source_sha256=sha(SOURCE),protocol_sha256=sha(OUT/'PROTOCOL.json'),evaluation_results_sha256=sha(EVAL/'RESULTS.json'),automatic_retry=False))
    start=time.perf_counter();records=[];allmeans={k:[] for k in METRICS}
    require(len(prod['records'])==32 and {r['replicate'] for r in prod['records']}==set(range(32)),'all replicates')
    for rep in range(32):
        tick=time.perf_counter();r=next(x for x in prod['records'] if x['replicate']==rep)
        (e,nu),hashes,initial,final=replay(rep)
        require(r['role_input_sha256']==hashes and r['RNG_initial_states']==initial and r['RNG_final_states']==final,'full RNG hashes/states')
        require(r['role_seed_sequences']=={name:[202610030400,rep,i] for name,i in [('active',0),('nuisance',1)]},'RNG seed metadata')
        require(r['first8_input_sha256']==dict(active=ah(e[:8]),nuisance=ah(nu[:8])),'first8 hashes')
        require(sha(ROOT/r['input_file'])==r['input_sha256'] and sha(ROOT/r['output_file'])==r['output_sha256'],'NPZ byte hashes')
        with np.load(ROOT/r['input_file'],allow_pickle=False) as a:require(np.array_equal(e[:8],a['active']) and np.array_equal(nu[:8],a['nuisance']),'stored first8')
        with np.load(ROOT/r['output_file'],allow_pickle=False) as a:data={k:a[k].copy() for k in a.files}
        require(set(data)=={p+k for p in ['', 'block_', 'block_pattern_'] for k in METRICS},'output fields')
        weighted_errors=[]
        for name in METRICS:
            patterns=data['block_pattern_'+name];block=data['block_'+name];require(patterns.shape==(8,16,5,22,16) and block.shape==(8,16,5,22,2),'block shape')
            rebuilt=np.empty_like(block)
            for mi,m in enumerate(sel['models']):
                for target,w in enumerate([sel['population_weights'],m['empirical_weights']]):rebuilt[:,mi,:,:,target]=np.sum(patterns[:,mi]*np.asarray(w)[None,None,None,:],axis=-1)
            weighted_errors.append(near(rebuilt,block,why='all block pattern weighting'))
            mean=rebuilt.mean(axis=0);near(mean,data[name],why='all block means');near(mean,r['means'][name],why='rep JSON means')
            require(np.array_equal(patterns[:,:,2,:21],patterns[:,:,4,:21]),'FACT context identity')
            for mi in range(16):require(np.array_equal(patterns[:,mi,2,:21],patterns[:,0,2,:21]),'FACT old-model identity')
            allmeans[name].append(mean)
        bm=bottom_math(e,nu,sel,data);num=api_controls(e,nu,sel,labels,r)
        require(qualification(r['numerical_controls'],r['canonical_controls'])==r['numerical_qualification'],'rep qualification arithmetic')
        require(r['completed_chunks']==8 and r['base_banks']==512,'rep budget')
        rec=dict(replicate=rep,passed=True,full_RNG_states_hashes=True,all_saved_weighting_max_error=max(weighted_errors),first_block_subset_math=bm,official_controls=num,seconds=time.perf_counter()-tick)
        records.append(rec);save(OUT/f'rep_{rep:02}.json',rec)
        print(json.dumps(dict(audit_replicate=rep,passed=True,seconds=rec['seconds'],math=bm,API=num)),flush=True)
    core=[x for r in prod['records'] for x in r['numerical_controls']];can=[x for r in prod['records'] for x in r['canonical_controls']]
    require(qualification(core,can)==prod['numerical_qualification'],'overall NUM arithmetic')
    expected=dict(independent_evaluation_replicates=32,new_base_banks=16384,new_prior_vectors=1638400,Gaussian_scalars=425984000,active_Gaussian_scalars=6553600,nuisance_Gaussian_scalars=419430400,new_observations=0,new_fits=0,new_model_integrals=0,shared_active_grid_terms_per_chunk=21,actual_t_terms_per_chunk=32,common_FACT_grid_scores_per_chunk=21,learned_or_half_grid_scores_per_chunk=1008,actual_t_score_cases_per_chunk=80,score_kernel_calls=283904,reused_bank_pattern_score_cases=290717696,original_API_pattern_control_calls=2048,original_API_canonical_control_calls=256,original_API_control_calls=2304)
    require(prod['ledger']==expected,'production ledger')
    np.savez_compressed(OUT/'RECONSTRUCTED_MEANS.npz',**{k:np.array(v) for k,v in allmeans.items()})
    save(OUT/'EVAL_REPORT.json',dict(passed=True,source_sha256=sha(SOURCE),protocol_sha256=sha(OUT/'PROTOCOL.json'),evaluation_results_sha256=sha(EVAL/'RESULTS.json'),reconstructed_means_sha256=sha(OUT/'RECONSTRUCTED_MEANS.npz'),records=records,numerical_qualification=prod['numerical_qualification'],seconds=time.perf_counter()-start,
        accounting=dict(replayed_Gaussian_scalars=425984000,old_input_NPZ_checked=16,new_NPZ_checked=64,first_complete_block_banks_per_rep=64,limited_math_cases=1600,official_API_replays=2304,new_independent_scientific_draws=0,new_model_integrals=0),scope='All RNG/state and saved block/weight arithmetic; direct mathematical bottom level only first64 banks, oldmodels0/15, fiveprofiles, fivepoints. Not full main evaluator rerun.'))

def ci(x,m=4):
    x=np.asarray(x,float);n=len(x);mean=float(math.fsum(x)/n);se=math.sqrt(math.fsum((x-mean)**2)/(n-1)/n);q=float(student_t.ppf(1-.05/(2*m),n-1))
    return dict(replicates=n,df=n-1,mean=mean,SE=se,critical_t=q,lower=mean-q*se,upper=mean+q*se)

def reference_arithmetic(sel):
    r=read(BASE/'REFERENCE/RESULTS.json');require(r['source_sha256']==sha(ROOT/'script/scbmeval_reference.py') and r['selection_sha256']==sha(BASE/'MODEL_SELECTION.json'),'reference binding')
    require(r['freeze_sha256']==sha(BASE/'REFERENCE/FREEZE.json'),'reference freeze')
    for p,h in read(BASE/'REFERENCE/FREEZE.json')['inputs'].items():require(sha(ROOT/p)==h,'reference input '+p)
    require(len(r['records'])==18 and [x['grid_index'] for x in r['records']]==[i for i in range(21) if i not in (0,10,20)],'reference new grid')
    for row in r['records']:
        main=np.array(row['main_pmf']);require([x['order'] for x in row['variants']]==[24,32,40],'reference fixed orders')
        near(main,next(x['pmf'] for x in row['variants'] if x['order']==32),0.,'reference main PMF')
        for v,q in zip(row['variants'],row['qualifications']):
            p=np.array(v['pmf']);delta=float(max(abs(p-main)));norm=abs(float(p.sum())-1.)
            near(delta,q['max_change'],why='reference PMF variation');near(norm,q['normalization_error'],why='reference norm')
            require(q['passed']==(delta<=5e-8 and norm<=2e-12 and bool(np.all(p>0))),'reference qualification')
    old=read(ROOT/sel['truth_reference_path']);opt=read(ROOT/'artifacts/scbm_scale_optimization/REFERENCE/RESULTS.json')
    table=r['PMF_table']
    for key,row in table.items():
        require(key==float(row['t']).hex(),'PMF key')
        if row['origin']=='old_scbminv':want=next(x['main_pmf'] for x in old['records'] if x['t']==row['t'])
        elif row['origin']=='old_scbmopt':want=next(x['main_pmf'] for x in opt['records'] if x['key']==key)
        else:want=next(x['main_pmf'] for x in r['records'] if x['grid_index']==row['grid_index'])
        require(row['pmf']==want,'PMF table reuse/source')
    ce=np.empty((16,5,22,2));kl=np.empty((16,5,22));pop=np.array(sel['population_weights'])
    for mi,m in enumerate(sel['models']):
        for pi,(kind,_) in enumerate(PROFILES):
            for point,t in enumerate(sel['grid_t']+[m[kind]['t']]):
                p=np.array(table[float(t).hex()]['pmf'])
                for wi,w in enumerate([pop,m['empirical_weights']]):ce[mi,pi,point,wi]=-math.fsum(float(a)*math.log(float(b)) for a,b in zip(w,p))
                kl[mi,pi,point]=math.fsum(float(a)*math.log(float(a/b)) for a,b in zip(pop,p))
    near(ce,r['cross_entropy'],why='all reference CE');near(ce-ce[:,:,10:11],r['relative_cross_entropy'],why='all reference contrast');near(kl,r['reference_joint_KL'],why='all reference KL')
    expected=dict(new_PMF_integrations=54,new_scalar_pattern_integrals=864,new_Gaussian_nodes=70926336,reused_grid_queries=3,reused_actual_queries=32,new_random_draws=0,new_fits=0)
    require(r['accounting']==expected and r['qualification_passed']==all(x['passed'] for x in r['qualifications']),'reference aggregate')
    return dict(passed=True,qualification_passed=r['qualification_passed'],reference_results_sha256=sha(BASE/'REFERENCE/RESULTS.json'),new_audit_integrals=0)

def statistics():
    bind();er=read(OUT/'EVAL_REPORT.json');require(er['passed'],'EVAL audit incomplete')
    require(sha(OUT/'RECONSTRUCTED_MEANS.npz')==er['reconstructed_means_sha256'],'independent means bytes')
    st=read(BASE/'STATISTICS/RESULTS.json');sel,_=old_panel()
    require(st['source_sha256']==sha(ROOT/'script/scbmeval_summary.py') and st['evaluation_results_sha256']==sha(EVAL/'RESULTS.json'),'statistics binding')
    require(st['freeze_sha256']==sha(BASE/'STATISTICS/FREEZE.json'),'statistics freeze')
    for p,h in read(BASE/'STATISTICS/FREEZE.json')['inputs'].items():require(sha(ROOT/p)==h,'statistic input '+p)
    save(OUT/'STATISTICS_EXECUTION.json',dict(start_utc=now(),source_sha256=sha(SOURCE),statistics_results_sha256=sha(BASE/'STATISTICS/RESULTS.json'),automatic_retry=False))
    with np.load(OUT/'RECONSTRUCTED_MEANS.npz') as z:arrays={k:z[k].copy() for k in METRICS}
    a=arrays['centered_score'];candidate_order=[21]+list(range(21));discovery=np.mean(a[:16,:,3,:,1],axis=0)
    chosen=np.array([min(candidate_order,key=lambda j:discovery[m,j]) for m in range(16)],dtype=int)
    holdout=np.array([[a[rep,m,3,21,1]-a[rep,m,3,chosen[m],1] for m in range(16)] for rep in range(16,32)])
    chosen_t=[sel['models'][m]['LAST']['t'] if j==21 else sel['grid_t'][j] for m,j in enumerate(chosen)]
    selected=st['selected'];require(selected['candidate_order']==candidate_order and selected['selection_replicates']==list(range(16)) and selected['validation_replicates']==list(range(16,32)),'honest split')
    near(chosen,selected['selected_point_indices'],0.,'independent selected index');near(chosen_t,selected['selected_t'],0.,'selected t');near(discovery,selected['discovery_means'],why='selection means');near(holdout,selected['validation_per_model_gains'],why='heldout gains')
    c=a[:,:,:,14,0]-a[:,:,:,10,0]
    values=[c[:,:,0].mean(axis=1),(c[:,:,0]-c[:,:,2]).mean(axis=1),(c[:,:,1]-c[:,:,0]).mean(axis=1),holdout.mean(axis=1)]
    names=['P1_population_wrong_preference','P2_nuisance_excess_tilt','P3_half_scale_upward_shift','P4_LAST_conditional_selected_improvement'];directions=['negative','negative','positive','positive']
    primary=[]
    require(len(st['primary_family'])==4,'four primary rows')
    for x,row,name,direction in zip(values,st['primary_family'],names,directions):
        require(row['name']==name and row['expected']==direction and row['SE_threshold']==5e-5,'primary identity');near(x,row['replicate_values'],why='primary values')
        result=ci(x)
        for k,v in result.items():near(v,row[k],why='primary '+k)
        effect=result['upper']<0 if direction=='negative' else result['lower']>0
        require(row['effect_passed']==effect and row['precision_passed']==(result['SE']<=5e-5),'primary original gates');primary.append(dict(name=name,**result,effect_passed=effect,precision_passed=result['SE']<=5e-5))
    require(st['primary_effects_passed']==sum(r['effect_passed'] for r in primary) and st['primary_precision_passed']==sum(r['precision_passed'] for r in primary),'primary pass counts')
    relscore=a-a[:,:,:,10:11,:];q=float(student_t.ppf(1-.05/(2*7480),31))
    for name,base in [('relative_score',relscore),('price_t',arrays['price_t'])]:
        panel=np.concatenate([base,base.mean(axis=1,keepdims=True)],axis=1);mean=panel.mean(axis=0);se=panel.std(axis=0,ddof=1)/math.sqrt(32)
        require(mean.shape==(17,5,22,2),'band shape')
        for k,v in [('mean',mean),('SE',se),('lower',mean-q*se),('upper',mean+q*se),('critical_t',q)]:near(v,st['diagnostic_bands'][name][k],why='all 7480 bands '+k)
    extra=st['descriptive']
    for key,v in [('tail_population_learned_contrast',c[:,:,0].mean(axis=1)),('tail_population_half_contrast',c[:,:,1].mean(axis=1)),('tail_population_fact_contrast',c[:,:,2].mean(axis=1)),('tail_population_half_excess',(c[:,:,1]-c[:,:,2]).mean(axis=1)),('tail_D_minus_P_relative',(relscore[:,:,:3,:,1]-relscore[:,:,:3,:,0]).mean(axis=1)),('price_minus_path_panel',(arrays['price_t']-arrays['path_t']).mean(axis=1))]:near(v,extra[key],why='descriptive '+key)
    for key,profile,target in [('LAST_validation_population_gains',3,0),('LAST_validation_factorized_D_gains',4,1)]:
        direct=[[a[rep,m,profile,21,target]-a[rep,m,profile,chosen[m],target] for m in range(16)] for rep in range(16,32)]
        near(direct,extra[key],why='explicit validation '+key)
    ref=reference_arithmetic(sel)
    # Derived interval arithmetic from the original joint family, no new test.
    half=dict(mean=primary[0]['mean']+primary[2]['mean'],lower=primary[0]['lower']+primary[2]['lower'],upper=primary[0]['upper']+primary[2]['upper'],scope='Post hoc sum of two already jointly adjusted primary intervals; pairwise half contrast only, not full recovery or a new t test.')
    near(values[0]+values[2],extra['tail_population_half_contrast'],why='P1+P3 exact half identity')
    save(OUT/'REPORT.json',dict(passed=True,source_sha256=sha(SOURCE),protocol_sha256=sha(OUT/'PROTOCOL.json'),evaluation_audit_sha256=sha(OUT/'EVAL_REPORT.json'),statistics_results_sha256=sha(BASE/'STATISTICS/RESULTS.json'),reference_arithmetic=ref,primary=primary,science_gate_counts=dict(effect_passed=sum(r['effect_passed'] for r in primary),precision_passed=sum(r['precision_passed'] for r in primary),family_size=4),all_7480_bands_checked=True,selection_validation_rebuilt_by_explicit_loops=True,derived_half_interval=half,new_model_integrals=0,new_independent_scientific_draws=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['check','freeze','run','statistics']);args=p.parse_args()
    if args.action=='check':print(json.dumps(check()),flush=True)
    elif args.action=='freeze':freeze()
    else:
        try:globals()[args.action]()
        except BaseException:
            path=OUT/(args.action.upper()+'_FAILURE.json')
            if not path.exists():save(path,dict(utc=now(),traceback=traceback.format_exc(),automatic_retry=False))
            raise
