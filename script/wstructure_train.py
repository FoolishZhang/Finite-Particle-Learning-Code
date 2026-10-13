"""Fresh paired complete-row training across observationally equivalent nuisances.

Only check/bench are RNG-free. The scientific run requires a root-authored
source-bound TRAIN/FREEZE.json. Old imports below supply pure arithmetic only.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse, json, hashlib, os, time, traceback
from fractions import Fraction
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import numpy as np
from scipy.special import expit, log_expit, logsumexp
from scipy.interpolate import PchipInterpolator
from scipy.stats import t as student_t
import fincal_train as old
import bankvar_train as bank
import curvcal_train as cross

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/nuisance_structure/TRAIN'
DOC=ROOT/'docs/08_research_研究工作/无关权重结构与方差匹配_20261002'
FAMILIES=('diffuse','boundedhetero','concentrated')
METHODS=FAMILIES+('marg','exact')
CFG=dict(a=1.,R=.75,n=254,K=256,B=128,N=4096,streams=16,
         teachers=[0.,.6],seed_base=202610023200,grid_points=801,
         bank_block=32,rho_batch=16,workers=2,chebyshev_degree=192,
         ellipse_q_numerator=28,ellipse_q_denominator=25,active_radius_bound=7.,
         family_alpha=.05,primary_family=3,real_arithmetic_regret_threshold=3e-5)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def read(p):return json.loads(Path(p).read_text('utf8'))
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def now():return datetime.now(timezone.utc).isoformat()

def scaled_prior(z,scales):
    scales=np.asarray(scales,dtype=float)
    assert scales.shape==(z.shape[2]-2,) and np.all(scales>0)
    out=z.copy();out[:,:,2:]*=scales[None,None,:]
    assert np.array_equal(out[:,:,:2],z[:,:,:2])
    return out

def raw_nodes(weights,active,y2,nodes):
    """Actual full-row score minus a candidate-independent offset, B x theta x rho."""
    N=weights.shape[1];out=np.empty((len(weights),len(y2),len(nodes)))
    indicator=(y2>0).astype(float)[None,:,:]
    for j in range(0,len(nodes),CFG['rho_batch']):
        rr=nodes[j:j+CFG['rho_batch']]
        lp=-np.log(old.second_probabilities_batch(weights,active,rr))
        minus,plus=lp[...,0],lp[...,1]
        out[:,:,j:j+len(rr)]=minus.mean(1)[:,None,:]+np.matmul(indicator,plus-minus)/N
    return out

def direct_full(z,y1,yn,y2,rhos):
    out=np.empty((len(z),len(y2),len(rhos)))
    for b in range(len(z)):
        nuisance=log_expit(yn[:,None,:]*z[b,None,:,2:]).sum(-1)
        first=log_expit(y1[:,None]*z[b,None,:,0])
        for j,rho in enumerate(rhos):
            x=rho*z[b,:,0]+np.sqrt(1-rho*rho)*z[b,:,1]
            for ti,second in enumerate(y2):
                ll=nuisance+first+log_expit(second[:,None]*x[None,:])
                out[b,ti,j]=(np.log(z.shape[1])-logsumexp(ll,axis=1)).mean()
    return out

def check():
    z,y1,yn,yall=old.deterministic_inputs(11,3,7,8);y2=yall[[0,2]]
    rr=np.array([-.75,-.41,0.,.3,.6,.75]);checks=[]
    scales=[np.ones(8),np.tile([.5,1.5],4),np.r_[3.5,np.full(7,1/8)]]
    for family,s in zip(FAMILIES,scales):
        zz=scaled_prior(z,s);w,off=old.build_weights(zz,y1,yn)
        cached=raw_nodes(w,zz[:,:,:2],y2,rr)+off.mean(1)[:,None,None]
        direct=direct_full(zz,y1,yn,y2,rr)
        error=float(np.max(np.abs(cached-direct)));assert error<3e-12
        h=2e-5;q=.3
        fd=raw_nodes(w,zz[:,:,:2],y2,q+np.array([-2,-1,1,2])*h)@np.array([1,-8,8,-1])/(12*h)
        grad,_=bank.path_gradients(w,zz[:,:,:2],y1,y2,np.array([q,q]))
        ge=float(np.max(np.abs(fd-grad)));assert ge<3e-10
        checks.append(dict(family=family,cache_full_loss_error=error,path_gradient_FD_error=ge))
    ref=old.marg_bank_losses_batch(z[:,:,:2],y1,y2,[len(y1)],rr)[:,:,0]
    assert all(np.array_equal(ref,old.marg_bank_losses_batch(scaled_prior(z,s)[:,:,:2],y1,y2,[len(y1)],rr)[:,:,0]) for s in scales)
    return dict(passed=True,checks=checks,marginalization_identical=True,scientific_RNG_calls=0,
                old_interpolation_and_outward_qa=bank.deterministic_check())

def benchmark():
    checks=check();z,y1,yn,ya=old.deterministic_inputs(CFG['N'],CFG['bank_block'],CFG['K'],CFG['n']);y2=ya[[0,2]]
    # Arbitrary deterministic positive scales exercise the real shapes; no variance matching is inferred.
    scales=np.r_[20.,np.full(CFG['n']-1,1/CFG['n'])]
    start=time.perf_counter();zz=scaled_prior(z,scales);w,off=old.build_weights(zz,y1,yn);wsec=time.perf_counter()-start
    nodes=bank.chebyshev_nodes()[::6];start=time.perf_counter();values=raw_nodes(w,zz[:,:,:2],y2,nodes);sec=time.perf_counter()-start
    return dict(passed=True,checks=checks,weight_seconds=wsec,node_seconds=sec,nodes=len(nodes),
        projected_serial_training_seconds=CFG['streams']*3*(CFG['B']/CFG['bank_block'])*(wsec+sec*193/len(nodes)),
        deterministic_only=True,scientific_RNG_calls=0,score_checksum=float(values.sum()))

def context(expected=None):
    p=OUT/'FREEZE.json';f=read(p)
    if expected:assert sha(p)==expected
    assert f['config']==CFG and f['run_authorized']
    for name,h in f['inputs'].items():assert sha(ROOT/name)==h,name
    assert f['trigger']['passed']
    ref=read(ROOT/f['known_model_reference']);ref=ref.get('known_active_reference',ref);grid=np.asarray(ref['rho801'])
    assert len(grid)==801 and grid[0]==-.75 and grid[-1]==.75 and grid[400]==0
    rfun=PchipInterpolator(grid,np.asarray(ref['r801']),extrapolate=False)
    scales={name:np.asarray(x) for name,x in f['nuisance_scales'].items()}
    assert set(scales)==set(FAMILIES) and all(x.shape==(CFG['n'],) and np.all(x>0) for x in scales.values())
    return f,grid,rfun,scales

def inputs(stream):
    data=np.random.default_rng(np.random.SeedSequence([CFG['seed_base'],stream,0]))
    un=data.random((CFG['N'],CFG['n']));yn=np.where(un<.5,-1,1).astype(np.int8)
    g=data.standard_normal((CFG['N'],2));u=data.random((CFG['N'],2))
    y1=np.where(u[:,0]<expit(g[:,0]),1,-1).astype(np.int8)
    y2=np.stack([np.where(u[:,1]<expit(t*g[:,0]+np.sqrt(1-t*t)*g[:,1]),1,-1) for t in CFG['teachers']]).astype(np.int8)
    prior=np.random.default_rng(np.random.SeedSequence([CFG['seed_base'],stream,1]))
    z=prior.standard_normal((CFG['B'],CFG['K'],CFG['n']+2))
    hashes={k:old.array_sha(v) for k,v in dict(noise_uniform=un,active_gaussian=g,active_uniform=u,prior=z,y1=y1,y2=y2,ynoise=yn).items()}
    return z,y1,yn,y2,hashes

def stream_run(stream,freeze_hash):
    f,grid,rfun,scales=context(freeze_hash);start=time.perf_counter()
    z,y1,yn,y2,hashes=inputs(stream)
    gate=bank.active_gate(z[:,:,:2]);nodes=bank.chebyshev_nodes() if gate['pass_all'] else grid
    matrix=bank.interpolation_matrix(nodes,grid) if gate['pass_all'] else None
    eps=bank.interpolation_error_upper(CFG) if gate['pass_all'] else 0.
    mle,boundary,pairs=bank.mle_estimates(y1,y2,rfun,CFG['R'])
    stem=OUT/f'stream_{stream:02}';arrays=dict(y1=y1,ynoise=yn,y2=y2,prior_active=z[:,:,:2],rho=grid,nodes=nodes,
        mle=mle,mle_boundary=boundary,empirical_pair=pairs)
    D=old.curvature_bound(z[:,:,:2],1.,CFG['R']);h=old.mesh_width_upper(grid)
    mesh=old.mesh_regret_upper(D,h);regret=old.outward_float(Fraction.from_float(mesh)+2*Fraction.from_float(eps))
    rows=[]
    for family in FAMILIES:
        allnodes=np.empty((CFG['B'],len(y2),len(nodes)));offset=np.empty((CFG['B'],CFG['N']))
        for begin in range(0,CFG['B'],CFG['bank_block']):
            end=min(begin+CFG['bank_block'],CFG['B']);zz=scaled_prior(z[begin:end],scales[family])
            w,off=old.build_weights(zz,y1,yn)
            allnodes[begin:end]=raw_nodes(w,zz[:,:,:2],y2,nodes);offset[begin:end]=off
            del w
        values=allnodes@matrix.T if matrix is not None else allnodes.copy()
        values-=values[:,:,400:401].copy()
        arrays[family+'_nodes']=allnodes;arrays[family+'_scores']=values;arrays[family+'_offset']=offset
    margnodes=old.marg_bank_losses_batch(z[:,:,:2],y1,y2,[CFG['N']],nodes)[:,:,0]
    marg=margnodes@matrix.T if matrix is not None else margnodes.copy();marg-=marg[:,:,400:401].copy()
    arrays['marg_nodes']=margnodes;arrays['marg_scores']=marg
    for ti,theta in enumerate(CFG['teachers']):
        rt=float(rfun(theta))
        for method in METHODS:
            row=dict(stream=stream,method=method,theta=theta,N=CFG['N'],B=CFG['B'],K=CFG['K'],n=CFG['n'],empirical_pair=float(pairs[ti]),data_MLE=float(mle[ti]),MLE_boundary=bool(boundary[ti]))
            if method=='exact':rho=float(mle[ti])
            else:
                curve=arrays[method+'_scores'].mean(0)[ti];j=int(np.argmin(curve));rho=float(grid[j])
                row.update(grid_index=j,selected_score=float(curve[j]),curvature_bound=D,mesh_step=h,
                    known_real_arithmetic_regret_bound=regret,precision_pass=bool(regret<=CFG['real_arithmetic_regret_threshold']))
            rhat=float(rfun(rho));row.update(rho=rho,observed_r=rhat,reference_joint_KL=old.observed_kl(rt,rhat),
                 absolute_agreement_error=abs(rhat-rt)/2,squared_discrepancy_to_data_MLE=(rho-float(mle[ti]))**2)
            rows.append(row)
    np.savez_compressed(str(stem)+'.npz',**arrays)
    record=dict(stream=stream,input_hashes=hashes,active_gate=gate,evaluation='Chebyshev193' if gate['pass_all'] else 'direct801',
        output_file=rel(Path(str(stem)+'.npz')),output_sha256=sha(str(stem)+'.npz'),outputs=rows,seconds=time.perf_counter()-start)
    save(str(stem)+'.json',record)
    print(json.dumps(dict(completed_stream=stream,seconds=record['seconds'])),flush=True)
    return record

def summarize(records):
    rows=[r for f in records for r in f['outputs']];q=student_t.ppf(1-CFG['family_alpha']/(2*CFG['primary_family']),CFG['streams']-1)
    index={(r['stream'],r['theta'],r['method']):r for r in rows};comparisons=[]
    for theta in CFG['teachers']:
        values=[index[i,theta,'diffuse']['reference_joint_KL']-index[i,theta,'concentrated']['reference_joint_KL'] for i in range(CFG['streams'])]
        comparisons.append(dict(name='diffuse_minus_concentrated_KL',theta=theta,values=values))
    values=[float(np.mean([index[i,t,'diffuse']['squared_discrepancy_to_data_MLE']-index[i,t,'concentrated']['squared_discrepancy_to_data_MLE'] for t in CFG['teachers']])) for i in range(CFG['streams'])]
    comparisons.append(dict(name='theta_pooled_MLE_squared_discrepancy',values=values))
    for row in comparisons:
        x=np.asarray(row['values']);mean=float(x.mean());se=float(x.std(ddof=1)/np.sqrt(len(x)));low,high=mean-q*se,mean+q*se
        row.update(mean=mean,SE=se,low=float(low),high=float(high),passed=bool(low>0),critical=float(q),development_t_approximation=True)
    tables=[]
    for theta in CFG['teachers']:
        for method in METHODS:
            rr=[r for r in rows if r['theta']==theta and r['method']==method]
            tables.append(dict(theta=theta,method=method,mean_rho=float(np.mean([r['rho'] for r in rr])),
                mean_KL=float(np.mean([r['reference_joint_KL'] for r in rr])),
                mean_MLE_MSE=float(np.mean([r['squared_discrepancy_to_data_MLE'] for r in rr])),
                endpoint_count=sum(abs(r['rho'])==CFG['R'] for r in rr),
                mean_absolute_agreement_error=float(np.mean([r['absolute_agreement_error'] for r in rr]))))
    return dict(records=records,comparisons=comparisons,tables=tables,outputs=len(rows),
        primary_passed=sum(c['passed'] for c in comparisons),primary_total=3,
        grid_precision_passed=sum(r.get('precision_pass',False) for r in rows),grid_outputs=128,
        known_regret_max=max(r.get('known_real_arithmetic_regret_bound',0) for r in rows),
        independent_streams=16,ledger=dict(new_base_observations=65536,shared_input_teacher_vectors=131072,
        data_Gaussian=131072,data_Uniform=16777216,prior_banks=2048,prior_vectors=524288,prior_Gaussian=134217728,
        grid_outputs=128,numerical_MLE_outputs=32,new_correction_banks=0,new_model_integrals=0))

def run():
    freeze=OUT/'FREEZE.json';context();assert not (OUT/'STATE.json').exists(),'Already started; retain failure, no silent restart'
    start=time.perf_counter();save(OUT/'STATE.json',dict(status='running',started_utc=now()))
    try:
        with ProcessPoolExecutor(max_workers=CFG['workers']) as pool:
            jobs=[pool.submit(stream_run,i,sha(freeze)) for i in range(CFG['streams'])]
            records=[job.result() for job in as_completed(jobs)]
        records.sort(key=lambda r:r['stream']);save(OUT/'RESULTS.json',summarize(records))
        save(OUT/'STATE.json',dict(status='complete',finished_utc=now(),seconds=time.perf_counter()-start,active_jobs=[]))
    except BaseException:
        save(OUT/'FAILURE.json',dict(utc=now(),traceback=traceback.format_exc()))
        save(OUT/'STATE.json',dict(status='failed',active_jobs=[]));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['check','bench','run']);a=p.parse_args()
    if a.action=='run':run()
    else:
        result=check() if a.action=='check' else benchmark()
        save(OUT/('ARTIFICIAL_QA.json' if a.action=='check' else 'BENCHMARK.json'),dict(source_sha256=sha(__file__),**result))
        print(json.dumps(result),flush=True)
