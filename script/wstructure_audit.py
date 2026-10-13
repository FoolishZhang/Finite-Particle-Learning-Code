"""Independent nuisance-structure audit; no production evaluator is imported.

check/freeze only inspect source or deterministic artificial arrays. Reference
executes fourteen new component integrals; MC/TRAIN replay prior inputs only.
Every executed phase has a unique receipt and never retries automatically.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import numpy as np
import mpmath as mp
from scipy.interpolate import PchipInterpolator
from scipy.special import expit, logsumexp
from scipy.stats import t as student_t
import bankvar_audit_train as ind

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(__file__).resolve()
BASE=ROOT/'artifacts/nuisance_structure'
OUT=BASE/'AUDIT'
DOC=ROOT/'docs/08_research_研究工作/无关权重结构与方差匹配_20261002'
FAMILIES=('diffuse','boundedhetero','concentrated')
TEACHERS=(0.,.6)
TARGETS=np.array([-.75,-.6,-.3,-.03,0.,.03,.3,.6,.75])
SIGNS=((-1,-1),(-1,1),(1,-1),(1,1))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def require(ok,message):
    if not ok:raise RuntimeError(message)
def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write('\n')
def now():return datetime.now(timezone.utc).isoformat()
def close(got,expected,tol=2e-12,message='Numeric mismatch'):
    a,b=np.asarray(got,dtype=float),np.asarray(expected,dtype=float)
    require(a.shape==b.shape,message+' shape')
    error=float(np.max(np.abs(a-b))) if a.size else 0.
    require(np.isfinite(a).all() and np.isfinite(b).all() and error<=tol,message+f': {error} > {tol}')
    return error

def model():
    m=read(BASE/'REFERENCE/MODEL.json');require(m['qualification_passed'],'Reference qualification')
    ref=m['known_active_reference'];grid=np.array(ref['rho801'])
    rf=PchipInterpolator(grid,np.array(ref['r801']),extrapolate=False)
    scales={(p['family'],p['n']):np.array(p['scales']) for p in m['profiles']}
    return m,grid,rf,scales

def direct_rows(prior,scales,y1,yn,y2,queries):
    """Full N rows through disjoint sign masks, not the production cache."""
    B,K=prior.shape[:2];values=np.empty((B,len(y2),len(queries)));offset=np.empty((B,len(y1)))
    pos,neg=(yn>0).astype(float),(yn<0).astype(float)
    for b,z in enumerate(prior):
        nuisance=z[:,2:]*scales[None,:]
        plus,minus=-np.logaddexp(0.,-nuisance),-np.logaddexp(0.,nuisance)
        base=pos@plus.T+neg@minus.T-np.logaddexp(0.,-y1[:,None]*z[None,:,0])
        offset[b]=math.log(K)-logsumexp(base,axis=1)
        for j,rho in enumerate(queries):
            x=rho*z[:,0]+math.sqrt(1-rho*rho)*z[:,1]
            for ti,second in enumerate(y2):
                full=base-np.logaddexp(0.,-second[:,None]*x[None,:])
                values[b,ti,j]=np.mean(math.log(K)-logsumexp(full,axis=1))
    return values,offset

def check():
    result=ind.selftest()
    z=np.cos(np.arange(2*5*6,dtype=float).reshape(2,5,6)/7)
    y1=np.array([-1,1,-1,1],dtype=np.int8)
    yn=np.array([[-1,1,-1,1],[1,-1,-1,1],[1,1,1,-1],[-1,-1,1,-1]],dtype=np.int8)
    y2=np.array([y1,-y1]);scales=np.array([3.,.2,.5,1.4]);rho=np.array([-.7,0.,.6])
    got,_=direct_rows(z,scales,y1,yn,y2,rho);expected=np.empty_like(got)
    for b in range(2):
        for ti,row in enumerate(y2):
            for qi,q in enumerate(rho):
                total=0.
                for j in range(4):
                    factors=[]
                    for k in range(5):
                        x2=q*z[b,k,0]+math.sqrt(1-q*q)*z[b,k,1]
                        p=float(expit(y1[j]*z[b,k,0])*expit(row[j]*x2))
                        for i in range(4):p*=float(expit(yn[j,i]*scales[i]*z[b,k,i+2]))
                        factors.append(p)
                    total-=math.log(math.fsum(factors)/5)
                expected[b,ti,qi]=total/4
    err=close(got,expected,2e-13,'Artificial direct rows')
    cases=[dict(n=n,K=k,family=fam) for n in (2,4) for k in (3,5) for fam in FAMILIES]
    fixtures={(fam,n):np.ones(n) for n in (2,4) for fam in FAMILIES}
    wz=np.array([[(1+t*s1*s2)/4 for s1,s2 in SIGNS] for t in (0.,.1)])
    mz=direct_mc(dict(active=np.zeros((2,5,2)),nuisance=np.zeros((2,5,4)),uniform=np.array([.2,.8])),cases,fixtures,wz)
    expected_price=np.array([[-s1*s2*(1-1/c['K'])/4 for s1,s2 in SIGNS] for c in cases])
    close(mz['price_modes'],np.broadcast_to(expected_price[None,:,None,:],(2,12,4,4)),2e-14,'MC zero-bank Price')
    require(np.max(abs(mz['score_difference_modes']))==0 and np.max(abs(mz['path_modes']))==0,'MC zero-bank score/path')
    return dict(passed=True,old_independent_helpers=result,scaled_full_row_error=err,
                independent_MC_zero_bank_passed=True,scientific_RNG=0,model_integrals=0)

def freeze():
    paths=[SOURCE,Path(ind.__file__),DOC/'AUDIT_PLAN.md',DOC/'REFERENCE_PLAN.md',DOC/'MC_DESIGN.md',DOC/'TRAIN_DESIGN.md',
           ROOT/'script/wstructure_reference.py',ROOT/'script/wstructure_score.py',ROOT/'script/wstructure_train.py',
           ROOT/'script/fincal_train.py',ROOT/'script/bankvar_train.py',ROOT/'script/curvcal_train.py',
           BASE/'REFERENCE/FREEZE.json',BASE/'REFERENCE/MODEL.json',BASE/'MC/FREEZE.json',ROOT/'artifacts/finite_calibration/CAL/REFERENCE.json']
    for p in paths:require(p.exists(),'Missing frozen source/input '+str(p))
    receipt=dict(schema='wstructure-independent-audit-v1',created_utc=now(),inputs={rel(p):sha(p) for p in paths},
                 check=check(),raw_MC_banks_per_stream=4,raw_TRAIN_banks_per_stream=2,raw_TRAIN_points=TARGETS.tolist(),
                 reference_digits=50,new_reference_component_integrals=14,no_new_independent_RNG=True,
                 automatic_retry=False)
    save(OUT/'PROTOCOL.json',receipt)
    print(json.dumps(dict(frozen=True,source_sha256=sha(SOURCE),protocol_sha256=sha(OUT/'PROTOCOL.json'))),flush=True)

def start_phase(name):
    p=read(OUT/'PROTOCOL.json')
    for path,h in p['inputs'].items():require(sha(ROOT/path)==h,'Protocol input changed: '+path)
    phase=name.upper();folder=BASE/phase;result=folder/('MODEL.json' if phase=='REFERENCE' else 'RESULTS.json')
    require(result.exists(),'Production incomplete: '+str(result))
    frozen=read(folder/'FREEZE.json')
    for path,h in frozen.get('inputs',{}).items():
        if isinstance(h,str):require(sha(ROOT/path)==h,'Production frozen input changed: '+path)
    data=read(result)
    source=ROOT/'script'/('wstructure_reference.py' if phase=='REFERENCE' else 'wstructure_score.py' if phase=='MC' else 'wstructure_train.py')
    if phase!='TRAIN':
        require(frozen['source_sha256']==sha(source)==data['source_sha256'],'Production source binding')
        require(data['freeze_sha256']==sha(folder/'FREEZE.json'),'Production freeze result binding')
        require(sha(ROOT/frozen['plan_path'])==frozen['plan_sha256'],'Production plan binding')
    if phase=='MC':
        require(frozen['run_authorized'] and frozen['reference_sha256']==sha(BASE/'REFERENCE/MODEL.json')==data['reference_sha256'],'MC reference binding')
        require(data['config']==frozen['config'],'MC configuration binding')
    if phase=='TRAIN':
        require(frozen['run_authorized'] and frozen['trigger']['passed'],'TRAIN trigger authorization')
        require(rel(source) in frozen['inputs'] and frozen['inputs'][rel(source)]==sha(source),'TRAIN source binding')
    save(OUT/(phase+'_EXECUTION.json'),dict(start_utc=now(),audit_source_sha256=sha(SOURCE),
         protocol_sha256=sha(OUT/'PROTOCOL.json'),production_freeze_sha256=sha(folder/'FREEZE.json'),
         production_result_sha256=sha(result),automatic_retry=False))
    return frozen,data

def reference():
    _,m=start_phase('REFERENCE');started=time.perf_counter();mp.mp.dps=50
    scales=sorted({float(a) for p in m['profiles'] for a in p['scales']});require(len(scales)==7,'Expected seven final scales')
    moments={};rows=[]
    for af in scales:
        a=mp.mpf(af);knots=sorted(set([mp.mpf(0),min(mp.mpf(1),1/a),mp.mpf(1),mp.mpf(4),mp.mpf(12)]))+[mp.inf]
        norm=mp.sqrt(2*mp.pi)
        def logs(x):return -mp.log1p(mp.exp(-a*x)),-mp.log1p(mp.exp(a*x))
        def first(x):
            lp,lm=logs(x);return mp.exp(-x*x/2)/norm*(lp+lm)
        mu=mp.quad(first,knots)
        def central(x):
            lp,lm=logs(x);return mp.exp(-x*x/2)/norm*((lp-mu)**2+(lm-mu)**2)
        variance=mp.quad(central,knots);moments[af]=variance
        production=next(v for v in m['validation'] if v['a']==af)
        err=abs(float(variance)-production['main_variance']);tol=1e-13+1e-11*abs(production['main_variance'])
        require(err<=tol,'Independent reference variance failed')
        muerr=abs(float(mu)-production['tight']['mean_logsigmoid']);require(muerr<=tol,'Independent reference mean failed')
        rows.append(dict(a=af,mean_logsigmoid=str(mu),variance=str(variance),variance_absolute_error=err,
                         mean_absolute_error=muerr,tolerance=tol,component_integrals=2,
                         subintervals_per_component=len(knots)-1,passed=True))
        print(json.dumps(dict(reference_a=af,passed=True)),flush=True)
    profiles=[]
    for p in m['profiles']:
        total=mp.fsum(moments[float(a)] for a in p['scales']);target=p['n']*moments[1.]
        residual=float((total-target)/target);require(abs(residual)<=1e-10,'Variance matching failed')
        cv=np.array([float(moments[float(a)]) for a in p['scales']])
        close(cv,p['component_variances'],1e-13+1e-11*max(cv),'Profile components')
        close(float(total),p['total_variance'],1e-13+1e-11*float(total),'Profile sum')
        close(float(max(cv)/float(total)),p['maximum_variance_share'],2e-12,'Maximum variance share')
        profiles.append(dict(n=p['n'],family=p['family'],independent_variance=str(total),relative_matching_residual=residual,passed=True))
    old=read(ROOT/'artifacts/finite_calibration/CAL/REFERENCE.json');main=next(v for v in old['orders'] if v['order']==old['main_order'])
    ar=m['known_active_reference'];require(ar['rho801']==old['rho801'] and ar['r801']==main['r801'] and ar['H801']==main['H801'],'Known active reference changed')
    out=dict(passed=True,scales=rows,profiles=profiles,known_active_reference_exact_copy=True,
             accounting=dict(new_independent_draws=0,new_component_model_integrals=14,new_scale_roots=0,new_r_integrals=0),
             scope='Independent high-precision numerical qualification, not rigorous quadrature remainder certification.',
             elapsed_seconds=time.perf_counter()-started)
    save(OUT/'REFERENCE_RESULT.json',out)

def replay_train(stream):
    gen=np.random.default_rng(np.random.SeedSequence([202610023200,stream,0]))
    un=gen.random((4096,254));yn=np.where(un<.5,-1,1).astype(np.int8)
    g=gen.standard_normal((4096,2));u=gen.random((4096,2))
    y1=np.where(u[:,0]<expit(g[:,0]),1,-1).astype(np.int8)
    y2=np.stack([np.where(u[:,1]<expit(t*g[:,0]+math.sqrt(1-t*t)*g[:,1]),1,-1) for t in TEACHERS]).astype(np.int8)
    hashes={k:arrsha(v) for k,v in dict(noise_uniform=un,active_gaussian=g,active_uniform=u,y1=y1,y2=y2,ynoise=yn).items()}
    gen=np.random.default_rng(np.random.SeedSequence([202610023200,stream,1]));active=[];hh=hashlib.sha256();prefix=None
    for begin in range(0,128,32):
        z=gen.standard_normal((32,256,256));hh.update(z.tobytes());active.append(z[:,:,:2].copy())
        if begin==0:prefix=z[:2].copy()
    hashes['prior']=hh.hexdigest()
    return y1,yn,y2,np.concatenate(active),prefix,hashes

def train():
    f,prod=start_phase('TRAIN');started=time.perf_counter();_,grid,rfun,scales=model()
    require(f['config']['n']==254 and f['config']['K']==256 and f['config']['B']==128 and f['config']['N']==4096,'TRAIN frozen dimensions')
    require(f['config']['seed_base']==202610023200 and f['config']['teachers']==list(TEACHERS) and f['config']['streams']==16,'TRAIN frozen seeds/teachers')
    require(sha(ROOT/f['known_model_reference'])==sha(BASE/'REFERENCE/MODEL.json'),'TRAIN reference identity')
    for fam in FAMILIES:require(np.array_equal(f['nuisance_scales'][fam],scales[fam,254]),'TRAIN frozen scales')
    idx=np.array([np.argmin(abs(grid-x)) for x in TARGETS]);close(grid[idx],TARGETS,1e-14,'Audit grid queries')
    h=ind.up(max(F.from_float(float(b))-F.from_float(float(a)) for a,b in zip(grid[:-1],grid[1:])))
    eps_cheb=ind.up(F(700)*F(25,28)**192);records=[];outputs=[]
    require(len(prod['records'])==16,'TRAIN stream count')
    for stream in range(16):
        tick=time.perf_counter();rec=next(r for r in prod['records'] if r['stream']==stream)
        y1,yn,y2,active,prefix,hashes=replay_train(stream)
        require(hashes==rec['input_hashes'],'Complete TRAIN RNG hash mismatch')
        require(sha(ROOT/rec['output_file'])==rec['output_sha256'],'TRAIN NPZ bytes changed')
        radius,gate=ind.ellipsoid_gate(active);require(radius==rec['active_gate']['max_squared_radius_upper'] and gate==rec['active_gate']['pass_all'],'TRAIN active gate')
        eps=eps_cheb if gate else 0.;D=ind.curvature(active)
        mesh=ind.up(F.from_float(D)*F.from_float(h)**2/8);regret=ind.up(F.from_float(mesh)+2*F.from_float(eps))
        mle,boundary,pair=ind.bisect_mle(y1,y2,rfun)
        audit=dict(stream=stream,full_RNG_replay_passed=True,gate=gate,max_squared_radius_upper=radius,families=[])
        with np.load(ROOT/rec['output_file']) as data:
            for name,a in [('y1',y1),('ynoise',yn),('y2',y2),('prior_active',active),('rho',grid)]:
                require(np.array_equal(a,data[name]),'TRAIN saved input '+name)
            close(mle,data['mle'],2e-11,'TRAIN MLE');require(np.array_equal(boundary,data['mle_boundary']),'MLE boundary')
            close(pair,data['empirical_pair'],0.,'Empirical pair');nodes=data['nodes'].copy()
            if gate:
                positive=.75*np.cos(np.pi*np.arange(96)/192);expected_nodes=np.r_[-positive,0.,positive[::-1]];expected_nodes[[0,-1]]=[-.75,.75]
            else:expected_nodes=grid
            require(np.array_equal(nodes,expected_nodes),'Node gate output')
            curves={}
            for fam in FAMILIES:
                values=data[fam+'_scores'].copy();rebuilt=ind.reconstruct_interpolation(nodes,data[fam+'_nodes'],grid)
                rebuilt-=rebuilt[:,:,400:401].copy();ie=close(rebuilt,values,3e-11,'TRAIN saved interpolation')
                direct,off=direct_rows(prefix,scales[fam,254],y1,yn,y2,grid[idx]);direct-=direct[:,:,4:5].copy()
                rawerr=close(direct,np.take(values[:2],idx,axis=2),2*eps+1e-9,'TRAIN limited raw')
                oe=close(off,data[fam+'_offset'][:2],2e-10,'TRAIN raw offsets')
                curves[fam]=values.mean(axis=0)
                audit['families'].append(dict(family=fam,saved_interpolation_error=ie,limited_full_row_error=rawerr,offset_error=oe))
            dm=ind.marg_rebuild(active,y1,y2,nodes);me=close(dm,data['marg_nodes'],2e-12,'TRAIN all marg nodes')
            mi=ind.reconstruct_interpolation(nodes,dm,grid);mi-=mi[:,:,400:401].copy()
            mie=close(mi,data['marg_scores'],3e-11,'TRAIN marg interpolation');curves['marg']=data['marg_scores'].mean(axis=0)
        require(len(rec['outputs'])==10,'TRAIN outputs per stream')
        for row in rec['outputs']:
            ti=TEACHERS.index(row['theta']);method=row['method'];require(method in (*FAMILIES,'marg','exact'),'Unknown method')
            close(row['empirical_pair'],pair[ti],0.,'Output pair');close(row['data_MLE'],mle[ti],2e-11,'Output MLE')
            require(row['MLE_boundary']==bool(boundary[ti]),'Output boundary')
            if method=='exact':rho=float(mle[ti])
            else:
                curve=curves[method][ti];j=min(range(len(grid)),key=lambda x:curve[x]);rho=float(grid[j])
                require(j==row['grid_index'],'TRAIN selected index');close(row['selected_score'],curve[j],3e-12,'Selected score')
                for k,v in [('curvature_bound',D),('mesh_step',h),('known_real_arithmetic_regret_bound',regret)]:close(row[k],v,0.,k)
                require(row['precision_pass']==(regret<=3e-5),'Precision gate')
            rh=float(rfun(rho));rt=float(rfun(row['theta']));kl=ind.observed_kl(rt,rh);disc=(rho-float(mle[ti]))**2
            for k,v,tol in [('rho',rho,2e-11),('observed_r',rh,2e-12),('reference_joint_KL',kl,3e-14),('absolute_agreement_error',abs(rh-rt)/2,2e-12),('squared_discrepancy_to_data_MLE',disc,2e-11)]:close(row[k],v,tol,k)
            outputs.append(dict(stream=stream,theta=row['theta'],method=method,rho=rho,reference_joint_KL=kl,
                           squared_discrepancy_to_data_MLE=disc,absolute_agreement_error=abs(rh-rt)/2))
        audit.update(passed=True,all_marg_node_error=me,all_marg_interpolation_error=mie,curvature_bound=D,
                     known_regret_bound=regret,elapsed_seconds=time.perf_counter()-tick)
        records.append(audit);save(OUT/f'TRAIN_stream_{stream:02}.json',audit)
        print(json.dumps(dict(TRAIN_audit_stream=stream,passed=True,seconds=audit['elapsed_seconds'])),flush=True)
    stats=audit_train_statistics(outputs,prod)
    save(OUT/'TRAIN_RESULT.json',dict(passed=True,records=records,statistics=stats,
         accounting=dict(replayed_base_observations=65536,replayed_prior_banks=2048,replayed_prior_Gaussian=134217728,
                         replayed_data_Gaussian=131072,replayed_data_Uniform=16777216,new_independent_draws=0,
                         full_row_recomputed_unique_banks=32,full_row_recomputed_family_banks=96,
                         raw_points_per_teacher=9,all_saved_selections=128,numerical_MLE=32,new_model_integrals=0),
         elapsed_seconds=time.perf_counter()-started))

def audit_train_statistics(rows,prod):
    require(len(rows)==160 and len({(r['stream'],r['theta'],r['method']) for r in rows})==160,'TRAIN complete unique rows')
    index={(r['stream'],r['theta'],r['method']):r for r in rows};q=float(student_t.ppf(1-.05/6,15));out=[]
    for t in TEACHERS:
        x=np.array([index[i,t,'diffuse']['reference_joint_KL']-index[i,t,'concentrated']['reference_joint_KL'] for i in range(16)])
        out.append(dict(name='diffuse_minus_concentrated_KL',theta=t,values=x))
    x=np.array([sum(index[i,t,'diffuse']['squared_discrepancy_to_data_MLE']-index[i,t,'concentrated']['squared_discrepancy_to_data_MLE'] for t in TEACHERS)/2 for i in range(16)])
    out.append(dict(name='theta_pooled_MLE_squared_discrepancy',values=x))
    for r,p in zip(out,prod['comparisons']):
        require(r['name']==p['name'] and r.get('theta')==p.get('theta'),'TRAIN comparison identity')
        x=r['values'];mean=float(math.fsum(x)/16);se=float(np.sqrt(math.fsum((x-mean)**2)/15/16));lo,hi=mean-q*se,mean+q*se
        for k,v in [('values',x),('mean',mean),('SE',se),('low',lo),('high',hi),('critical',q)]:close(p[k],v,2e-12,'TRAIN '+k)
        require(p['passed']==(lo>0),'TRAIN statistical gate');r.update(values=x.tolist(),mean=mean,SE=se,low=lo,high=hi,passed=lo>0)
    require(prod['primary_passed']==sum(r['passed'] for r in out) and prod['primary_total']==3,'TRAIN family pass count')
    for p in prod['tables']:
        rr=[r for r in rows if r['method']==p['method'] and r['theta']==p['theta']];require(len(rr)==16,'TRAIN table rows')
        for k,field in [('mean_rho','rho'),('mean_KL','reference_joint_KL'),('mean_MLE_MSE','squared_discrepancy_to_data_MLE'),('mean_absolute_agreement_error','absolute_agreement_error')]:close(p[k],math.fsum(r[field] for r in rr)/16,2e-12,'TRAIN table '+k)
        require(p['endpoint_count']==sum(abs(r['rho'])==.75 for r in rr),'TRAIN endpoint count')
    require(prod['outputs']==160 and prod['grid_outputs']==128 and prod['independent_streams']==16,'TRAIN total counts')
    allprod=[r for rec in prod['records'] for r in rec['outputs']]
    require(prod['grid_precision_passed']==sum(r.get('precision_pass',False) for r in allprod),'TRAIN precision count')
    close(prod['known_regret_max'],max(r.get('known_real_arithmetic_regret_bound',0.) for r in allprod),0.,'TRAIN max regret')
    ledger=dict(new_base_observations=65536,shared_input_teacher_vectors=131072,data_Gaussian=131072,data_Uniform=16777216,
                prior_banks=2048,prior_vectors=524288,prior_Gaussian=134217728,grid_outputs=128,numerical_MLE_outputs=32,new_correction_banks=0,new_model_integrals=0)
    require(prod['ledger']==ledger,'TRAIN ledger')
    return dict(passed=True,comparisons=out,all_tables_checked=True,ledger_checked=True)

def replay_mc(stream):
    rng={name:np.random.Generator(np.random.PCG64(np.random.SeedSequence([202610023100,stream,i])))
         for name,i in [('active',0),('uniform',1),('nuisance',2)]}
    digests={name:hashlib.sha256() for name in rng};first={};prefix=None
    all_uniform=[]
    for begin in range(0,4096,64):
        values=dict(active=rng['active'].standard_normal((64,256,2)),uniform=rng['uniform'].random(64),
                    nuisance=rng['nuisance'].standard_normal((64,256,254)))
        for name,a in values.items():
            raw=a.tobytes();digests[name].update(raw)
            if begin==0:first[name]=hashlib.sha256(raw).hexdigest()
        if begin==0:prefix={name:a[:4].copy() for name,a in values.items()}
        all_uniform.append(values['uniform'])
    return prefix,{name:h.hexdigest() for name,h in digests.items()},first,np.concatenate(all_uniform)

def direct_mc(prefix,cases,scales,weights):
    active,noise,u=prefix['active'],prefix['nuisance'],prefix['uniform'];B=len(active)
    shape=(B,12,4,4);out=dict(score_difference_modes=np.empty(shape),price_modes=np.empty(shape),path_modes=np.empty(shape),
           random_price_modes=np.empty((B,12,2,4)),random_rho=np.empty((B,2)),gap=np.empty((B,12,2)),
           nuisance_ESS=np.empty((B,12)),nuisance_max_weight=np.empty((B,12)),nuisance_score_offset=np.empty((B,12)))
    for b in range(B):
        for ci,c in enumerate(cases):
            n,k=c['n'],c['K'];za=active[b,:k];scaled=noise[b,:k,:n]*scales[c['family'],n]
            nuisance=-np.logaddexp(0.,-scaled).sum(axis=1);peak=float(nuisance.max())
            nu=np.exp(nuisance-peak);den=math.fsum(nu);q=nu/den
            out['nuisance_ESS'][b,ci]=1/math.fsum(q*q)
            out['nuisance_max_weight'][b,ci]=float(max(q))
            out['nuisance_score_offset'][b,ci]=math.log(k)-peak-math.log(den)
            def evaluate(rho):
                x=za[:,0]*rho+za[:,1]*math.sqrt(1-rho*rho)
                dx=za[:,0]-za[:,1]*rho/math.sqrt(1-rho*rho)
                loss=[];price=[];path=[]
                for s1,s2 in SIGNS:
                    logs=nuisance-np.logaddexp(0.,-s1*za[:,0])-np.logaddexp(0.,-s2*x)
                    m=float(logs.max());qq=np.exp(logs-m);total=math.fsum(qq);p=qq/total
                    comp=expit(-s1*za[:,0])*expit(-s2*x)
                    loss.append(math.log(k)-m-math.log(total))
                    price.append(-s1*s2*(math.fsum(p*comp)-math.fsum(p*p*comp)))
                    path.append(-s2*math.fsum(p*expit(-s2*x)*dx))
                return np.array(loss),np.array(price),np.array(path)
            anchor=evaluate(0.)[0]
            for ri,rho in enumerate([0.,.3,.6,.75]):
                loss,pd,pathd=evaluate(rho)
                out['score_difference_modes'][b,ci,ri]=loss-anchor
                out['price_modes'][b,ci,ri]=pd;out['path_modes'][b,ci,ri]=pathd
            for ti,t in enumerate(TEACHERS):
                rho=t+(.75-t)*u[b];pd=evaluate(rho)[1]
                out['random_rho'][b,ti]=rho;out['random_price_modes'][b,ci,ti]=pd
                out['gap'][b,ci,ti]=(.75-t)*math.fsum(pd*weights[ti])
    return out

def mc():
    f,prod=start_phase('MC');started=time.perf_counter();_,grid,rfun,scales=model()
    cases=[dict(n=n,K=k,family=fam) for n in (62,254) for k in (64,256) for fam in FAMILIES]
    require(prod['cases']==cases and len(prod['records'])==16,'MC cases/streams')
    moments=[0.,float(rfun(grid[np.argmin(abs(grid-.6))]))]
    weights=np.array([[(1+t*s1*s2)/4 for s1,s2 in SIGNS] for t in moments]);close(weights,prod['teacher_weights'],0.,'Teacher weights')
    require(prod['mode_signs']==[list(x) for x in SIGNS],'MC modes')
    records=[];means_by_stream=[];gaps=[]
    for stream in range(16):
        tick=time.perf_counter();rec=next(r for r in prod['records'] if r['stream']==stream)
        prefix,hashes,first,uniform=replay_mc(stream)
        require(hashes==rec['input_sha256'] and first==rec['first_block_sha256'],'Complete MC input replay')
        require(rec['role_seed_sequences']=={name:[202610023100,stream,i] for name,i in [('active',0),('uniform',1),('nuisance',2)]},'MC role seeds')
        require(sha(ROOT/rec['audit_input_file'])==rec['audit_input_sha256'] and sha(ROOT/rec['file'])==rec['file_sha256'],'MC saved bytes')
        with np.load(ROOT/rec['audit_input_file']) as inp:
            for name,a in prefix.items():require(np.array_equal(a,inp[name]),'MC prefix '+name)
            require(np.array_equal(inp['signs'],SIGNS),'MC prefix signs')
        direct=direct_mc(prefix,cases,scales,weights);errors={};means={}
        with np.load(ROOT/rec['file']) as values:
            require(np.array_equal(values['signs'],SIGNS) and np.array_equal(values['rhos'],[0.,.3,.6,.75]) and np.array_equal(values['thetas'],TEACHERS),'MC metadata')
            require(set(values.files)==set(direct)|{'signs','rhos','thetas'},'MC complete field schema')
            for name,a in direct.items():errors[name]=close(a,values[name][:4],2e-10,'MC limited '+name)
            random_expected=np.array([t+(.75-t)*uniform for t in TEACHERS]).T
            close(values['random_rho'],random_expected,0.,'MC all random points')
            recomputed_gap=np.einsum('bctm,tm->bct',values['random_price_modes'],weights)*np.array([.75,.15])[None,None,:]
            close(values['gap'],recomputed_gap,2e-12,'MC all gap arithmetic')
            for name in direct:
                if name=='random_rho':continue
                a=values[name];require(len(a)==4096 and np.isfinite(a).all(),'MC array shape/finite')
                means[name]=np.sum(a,axis=0,dtype=np.longdouble).astype(float)/4096
                close(means[name],rec['means'][name],2e-12,'MC stream mean '+name)
            g=values['gap'].copy();gv=np.sum((g-means['gap'])**2,axis=0,dtype=np.longdouble).astype(float)/4095
            close(gv,rec['within_stream_gap_sample_variance'],2e-12,'MC stream variance');gaps.append(g)
            require(np.max(abs(values['price_modes']))<=1+2e-14,'MC Price bounded')
            require(np.max(abs(g[:,:,0]))<=.75+2e-14 and np.max(abs(g[:,:,1]))<=.15+2e-14,'MC gap bounded')
        means_by_stream.append(means)
        row=dict(stream=stream,passed=True,complete_role_RNG_replay=True,first_block_and_saved_prefix_passed=True,
                 limited_bottom_level_errors=errors,all_saved_means_and_gap_arithmetic_passed=True,elapsed_seconds=time.perf_counter()-tick)
        records.append(row);save(OUT/f'MC_stream_{stream:02}.json',row)
        print(json.dumps(dict(MC_audit_stream=stream,passed=True,seconds=row['elapsed_seconds'])),flush=True)
    allgap=np.concatenate(gaps);mean=np.sum(allgap,axis=0,dtype=np.longdouble).astype(float)/65536
    variance=np.sum((allgap-mean)**2,axis=0,dtype=np.longdouble).astype(float)/65535
    close(mean,prod['all_gap_mean'],2e-12,'MC pooled mean');close(variance,prod['all_gap_sample_variance'],2e-12,'MC pooled variance')
    primary=[];logterm=math.log(960.)
    require(len(prod['primary_family'])==12,'MC primary count')
    for row in prod['primary_family']:
        ci=cases.index({k:row[k] for k in ['n','K','family']});ti=TEACHERS.index(row['theta']);require(row['n']==254,'MC primary n')
        L=.75-row['theta'];hw=math.sqrt(2*variance[ci,ti]*logterm/65536)+14*L*logterm/(3*65535)
        lo,hi=mean[ci,ti]-hw,mean[ci,ti]+hw;expected='positive' if row['family']=='concentrated' else 'negative'
        for name,a in [('mean',mean[ci,ti]),('sample_variance',variance[ci,ti]),('L',L),('logterm',logterm),('half_width',hw),('lower',lo),('upper',hi)]:close(row[name],a,2e-12,'MC EB '+name)
        passed=bool(lo>0 if expected=='positive' else hi<0)
        require(row['M']==65536 and row['expected_sign']==expected and row['passed']==passed,'MC EB gate')
        primary.append(dict(n=254,K=row['K'],family=row['family'],theta=row['theta'],lower=float(lo),upper=float(hi),passed=passed))
    require(len({(r['K'],r['family'],r['theta']) for r in primary})==12,'MC complete family')
    for name,field,threshold in [('price','price_modes',2e-4),('score_difference','score_difference_modes',5e-4),('path','path_modes',None)]:
        stack=np.array([m[field] for m in means_by_stream]);tm=np.einsum('scrm,tm->scrt',stack,weights)
        mm=tm.mean(axis=0);se=np.sqrt(np.sum((tm-mm)**2,axis=0)/(15*16));p=prod['precision'][name]
        close(mm,p['mean'],2e-12,'MC precision mean');close(se,p['stream_SE'],2e-12,'MC precision SE')
        mask=np.ones_like(se,dtype=bool)
        if name=='score_difference':mask[:,0,:]=False
        require(np.array_equal(mask,p['assessed_mask']) and p['threshold']==threshold,'MC precision assessment')
        if threshold is not None:
            require(np.array_equal(se<=threshold,p['pass_mask']) and p['assessed_count']==int(mask.sum()) and p['pass_count']==int(((se<=threshold)&mask).sum()),'MC precision gates')
        else:require(p['pass_mask'] is None and p['pass_count'] is None and p['assessed_count']==0,'MC path descriptive scope')
    ledger=dict(independent_banks=65536,independent_seed_groups=16,prior_vectors=16777216,Gaussian_scalars=4294967296,
                active_Gaussian_scalars=33554432,nuisance_Gaussian_scalars=4261412864,uniform_scalars=65536,new_observations=0,new_data_fits=0,new_neural_models=0)
    require(prod['accounting']==ledger,'MC accounting')
    save(OUT/'MC_RESULT.json',dict(passed=True,records=records,primary=primary,all_statistics_and_accounting_passed=True,
         accounting=dict(replayed_banks=65536,replayed_Gaussian_scalars=4294967296,replayed_uniforms=65536,
                         independently_scored_banks=64,case_prefixes_per_bank=12,new_independent_draws=0,new_model_integrals=0),
         elapsed_seconds=time.perf_counter()-started))

def report():
    phases={name:read(OUT/(name+'_RESULT.json')) for name in ('REFERENCE','MC','TRAIN') if (OUT/(name+'_RESULT.json')).exists()}
    require(set(phases)=={'REFERENCE','MC','TRAIN'},'All three audits must complete')
    save(OUT/'REPORT.json',dict(passed=all(x['passed'] for x in phases.values()),audit_source_sha256=sha(SOURCE),
         protocol_sha256=sha(OUT/'PROTOCOL.json'),phase_result_sha256={name:sha(OUT/(name+'_RESULT.json')) for name in phases},
         accounting={name:x['accounting'] for name,x in phases.items()},
         scope='Complete input replays; raw bottom-level MC first 4 banks/stream, TRAIN first 2 banks/stream/all rows; all saved statistics. Not a full bottom-level production rerun.',
         new_reference_audit_component_integrals=14,new_independent_scientific_draws=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['check','freeze','reference','mc','train','report']);a=p.parse_args()
    if a.action=='check':print(json.dumps(check()),flush=True)
    elif a.action in ('freeze','report'):globals()[a.action]()
    else:
        try:globals()[a.action]()
        except BaseException:
            failure=OUT/(a.action.upper()+'_FAILURE.json')
            if not failure.exists():save(failure,dict(utc=now(),traceback=traceback.format_exc(),automatic_retry=False))
            raise
