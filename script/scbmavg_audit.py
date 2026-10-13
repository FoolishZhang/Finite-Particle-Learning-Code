"""Bounded independent audit of the outer-bank averaging experiment.

Reuses the previously frozen *independent audit* math/native/Adam/repair cores.
No producer optimizer, scoring evaluator, statistical routine, or run is called
by the scientific audit. check() only compares deterministic statistic fixtures.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,hashlib,importlib.util,itertools,json,math,sys,time,traceback
import numpy as np
import torch
from scipy.special import expit,logsumexp
from scipy.stats import t as student
import scbm6_audit as core

torch.set_num_threads(1)
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/scbm_bank_averaging';OUT=BASE/'AUDIT'
DOC=ROOT/'docs/08_research_研究工作/同目标下的粒子梯度平均_20261003'
ARMS=['L1','L4'];BITS=np.array(list(itertools.product([0,1],repeat=4)))
PAIRS=list(itertools.combinations(range(4),2));STEPS=[0,31,127,511]
TOL=dict(arithmetic=2e-10,gradient_mean=0.,native_loss=5e-5,native_gradient=3e-6,
         math64_loss=1e-10,math64_gradient=1e-10,proposal=3e-6,
         repair_parameter=2e-7,repair=1e-12,moment_abs=2e-6,moment_rel=2e-5,
         eval_score=2e-10,reference_PMF=5e-8)
SCOPE=dict(training_replay_workers=8,training_pairs=64,
    training_replayed_Gaussian_scalars=218103808000,
    training_native_aggregate_states=32,training_native_bank_calls=80,
    training_bottom_data_ids=[0,1,2,3],training_bottom_repeats=[0],training_bottom_steps=STEPS,
    evaluation_groups=32,evaluation_prefix_banks=2,evaluation_prefix_score_cells=282624,
    evaluation_replayed_Gaussian_scalars=851968000,evaluation_native_calls=320,
    reference_indices=[2,3,34,35,66,67,98,99]+list(range(130,138)),reference_order=36,
    reference_PMFs=16,reference_nodes=26873856,new_independent_random_draws=0,
    no_automatic_retry=True,tolerances=TOL)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,obj,exclusive=True):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x' if exclusive else 'w',encoding='utf8',newline='\n') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def near(a,b,tol=TOL['arithmetic']):
    a,b=np.asarray(a,float),np.asarray(b,float)
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    e=np.abs(a-b);assert np.all(e<=tol),(float(e.max()),tol)
    return float(e.max()) if e.size else 0.
def cov(rho):
    a=np.eye(4)
    for x,(i,j) in zip(rho,PAIRS):a[i,j]=a[j,i]=x
    return a
def seqmean(a):
    a=np.asarray(a,np.float32);v=a[0].copy()
    for x in a[1:]:v=np.asarray(v+x,np.float32)
    return np.asarray(v*np.float32(1/len(a)),np.float32)
def moments(values):
    v=[float(x) for x in values];m=math.fsum(v)/len(v)
    se=math.sqrt(math.fsum((x-m)**2 for x in v)/(len(v)*(len(v)-1)))
    return m,se


def verify_freeze(derived=True):
    root=read(BASE/'FREEZE.json');assert root['run_authorized'] is True
    require=[Path(__file__),Path(core.__file__),core.OFF,BASE/'CONFIG.json',BASE/'DATA.json',DOC/'PLAN.md']
    for p in require:assert root['inputs'][rel(p)]==sha(p),p
    records=[root]
    if derived:
        x=read(BASE/'DERIVED_FREEZE.json');assert x['run_authorized'] is True and x['root_freeze_sha256']==sha(BASE/'FREEZE.json');records.append(x)
    for record in records:
        for p,h in record['inputs'].items():assert sha(ROOT/p)==h,p
    assert not (BASE/'VERSION_LOCK.json').exists()


def data_check():
    c,d=read(BASE/'CONFIG.json'),read(BASE/'DATA.json')
    assert c['data_ids']==[0,1,2,3] and c['repeats']==16 and c['outer_banks']==[1,4]
    assert d['patterns']==BITS.tolist() and d['pair_order']==[list(x) for x in PAIRS]
    for p,h in d['inputs'].items():assert sha(ROOT/p)==h,p
    for row in d['data']:
        assert sha(ROOT/row['path'])==row['sha256']
        with np.load(ROOT/row['path'],allow_pickle=False) as f:y=f['active_labels'];yn=f['nuisance_labels']
        assert y.shape==(4096,4) and yn.shape==(4096,256)
        assert ah(y)==row['active_labels_sha256'] and ah(yn)==row['nuisance_labels_sha256']
        counts=[int(np.all(y==bits,axis=1).sum()) for bits in BITS]
        assert counts==row['counts'] and row['weights']==[x/4096 for x in counts]
    old=read(ROOT/'artifacts/scbm_invariance/REFERENCE/MODEL.json')
    assert d['population_weights']==next(x['main_pmf'] for x in old['records'] if x['t']==0)
    return c,d


def numerical_rows(rows):
    aggregates=[];banks=[]
    for r in rows:
        for b in r['per_bank_controls']:
            dl=b['native_loss']-b['stable64_loss'];dg=np.array(b['native_gradient'])-b['stable64_gradient']
            near(dl,b['loss_difference']);near(dg,b['gradient_difference']);near(np.max(abs(dg)),b['maximum_gradient_difference'])
            assert b['loss_passed']==(abs(dl)<=1e-4) and b['gradient_passed']==(np.max(abs(dg))<=2e-5)
            banks.append(b)
        near(seqmean(np.array([b['native_loss'] for b in r['per_bank_controls']])),r['native_loss'],0.)
        near(seqmean(np.array([b['native_gradient'] for b in r['per_bank_controls']])),r['native_gradient'],0.)
        near(math.fsum(b['stable64_loss'] for b in r['per_bank_controls'])/len(r['per_bank_controls']),r['stable64_loss'])
        near(np.mean([b['stable64_gradient'] for b in r['per_bank_controls']],axis=0),r['stable64_gradient'])
        dl=r['native_loss']-r['stable64_loss'];dg=np.array(r['native_gradient'])-r['stable64_gradient']
        near(dl,r['loss_difference']);near(dg,r['gradient_difference']);near(np.max(abs(dg)),r['maximum_gradient_difference'])
        assert r['loss_passed']==(abs(dl)<=1e-4) and r['gradient_passed']==(np.max(abs(dg))<=2e-5)
        assert r['passed']==(r['loss_passed'] and r['gradient_passed']);aggregates.append(r)
    return dict(aggregate_checks=len(aggregates),aggregate_passed=sum(r['passed'] for r in aggregates),
        aggregate_loss_passed=sum(r['loss_passed'] for r in aggregates),aggregate_gradient_passed=sum(r['gradient_passed'] for r in aggregates),
        bank_checks=len(banks),bank_loss_passed=sum(b['loss_passed'] for b in banks),bank_gradient_passed=sum(b['gradient_passed'] for b in banks),
        stable64_calls=sum(r['stable64_calls'] for r in aggregates),
        maximum_aggregate_loss_difference=max(abs(r['loss_difference']) for r in aggregates),
        maximum_aggregate_gradient_difference=max(r['maximum_gradient_difference'] for r in aggregates),
        maximum_bank_loss_difference=max(abs(b['loss_difference']) for b in banks),maximum_bank_gradient_difference=max(b['maximum_gradient_difference'] for b in banks),
        passed=bool(all(r['passed'] for r in aggregates) and all(b['loss_passed'] and b['gradient_passed'] for b in banks)))


def training_pair(row):
    """One independent task; all512steps x4roles reproduced, no relearning."""
    start=time.perf_counter();d,r=row['data_id'],row['repeat'];name=f'd{d:02}_r{r:02}';step=-1
    try:
        assert sha(ROOT/row['input_file'])==row['input_sha256'] and sha(ROOT/row['output_file'])==row['output_sha256']
        with np.load(ROOT/row['input_file'],allow_pickle=False) as f:initial_inputs={k:f[k] for k in f.files}
        with np.load(ROOT/row['output_file'],allow_pickle=False) as f:t={k:f[k] for k in f.files}
        data=read(BASE/'DATA.json')['data'][d];assert data['path']==row['old_data_file'] and data['sha256']==row['old_data_sha256']
        with np.load(ROOT/data['path'],allow_pickle=False) as f:y=f['active_labels'];yn=f['nuisance_labels']
        assert row['label_sha256']==dict(active=ah(y),noise=ah(yn))
        assert t['rho'].shape==(2,513,6) and np.array_equal(t['rho'][:,0],np.zeros((2,6)))
        assert np.array_equal(t['batch_start'],np.arange(512)*64%4096)
        assert np.array_equal(t['final_adam_step'],[512,512])
        assert all(np.isfinite(v).all() for v in t.values())
        for key in ['finite','feasible','cholesky_success']:assert t[key].all()
        native_checks=[];max_proposal=max_repair=0.;m=np.zeros((2,6),np.float32);v=m.copy()
        generators={};digests={}
        for bank in range(4):
            for role,ri in [('active',0),('noise',1)]:
                key=f'bank_{bank}_{role}';seed=[202610034000,d,r,bank,ri]
                assert row['role_seed_sequences'][key]==seed
                g=np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
                assert g.bit_generator.state==row['RNG_initial_states'][key]
                generators[key]=g;digests[key]=hashlib.sha256()
        for step in range(512):
            for ai,arm in enumerate(ARMS):
                near(seqmean(t['bank_gradient_'+arm][step]),t['aggregate_gradient'][ai,step],0.)
                near(seqmean(t['bank_loss_'+arm][step]),t['aggregate_loss'][ai,step],0.)
                for key in ['sigmoid_zero_count','sigmoid_one_count','probability_evaluations']:
                    assert t[key][ai,step]==t['bank_'+key+'_'+arm][step].sum()
                near(t['maximum_absolute_logit'][ai,step],t['bank_maximum_absolute_logit_'+arm][step].max(),0.)
            m,v,delta=core.adam(m,v,t['aggregate_gradient'][:,step],step+1)
            max_proposal=max(max_proposal,near(t['rho'][:,step]+delta,t['rho_unprojected'][:,step],TOL['proposal']))
            for ai in range(2):
                restored,alpha,e0,e1=core.repair(t['rho_unprojected'][ai,step])
                max_repair=max(max_repair,near(restored,t['rho'][ai,step+1],TOL['repair_parameter']))
                near([alpha,e0,e1],[t['projection_alpha'][ai,step],t['proposal_eigmin'][ai,step],t['eigmin'][ai,step+1]],TOL['repair'])
                assert t['repaired'][ai,step]==(e0<.1) and e1>=.1
                np.linalg.cholesky(cov(t['rho'][ai,step+1]))
            begin=(step*64)%4096
            for bank in range(4):
                active=generators[f'bank_{bank}_active'].standard_normal((100,64,4),dtype=np.float64)
                noise=generators[f'bank_{bank}_noise'].standard_normal((100,64,256),dtype=np.float64)
                digests[f'bank_{bank}_active'].update(active.tobytes(order='C'));digests[f'bank_{bank}_noise'].update(noise.tobytes(order='C'))
                if step==0:
                    assert np.array_equal(active,initial_inputs['first_prior_active'][bank])
                    assert np.array_equal(noise,initial_inputs['first_prior_noise'][bank])
                if r==0 and step in STEPS:
                    for ai in ([0,1] if bank==0 else [1]):
                        arm=ARMS[ai];rho=t['rho'][ai,step]
                        native=core.native('FIXED',rho,None,active,noise,y[begin:begin+64],yn[begin:begin+64])
                        math64=core.math_score('FIXED',rho,None,active,noise,y[begin:begin+64],yn[begin:begin+64])
                        nl=near(native['loss'],t['bank_loss_'+arm][step,bank],TOL['native_loss'])
                        ng=near(native['rho_gradient'],t['bank_gradient_'+arm][step,bank],TOL['native_gradient'])
                        control=next(x for x in row['numerical_controls'] if x['arm']==arm and x['preupdate_step']==step)['per_bank_controls'][bank]
                        ml=near(math64['loss'],control['stable64_loss'],TOL['math64_loss']);mg=near(math64['rho_gradient'],control['stable64_gradient'],TOL['math64_gradient'])
                        assert native['zero']==t['bank_sigmoid_zero_count_'+arm][step,bank] and native['one']==t['bank_sigmoid_one_count_'+arm][step,bank]
                        assert native['evaluations']==t['bank_probability_evaluations_'+arm][step,bank]
                        near(native['maxlogit'],t['bank_maximum_absolute_logit_'+arm][step,bank],2e-6)
                        native_checks.append(dict(data_id=d,repeat=r,step=step,arm=arm,bank=bank,native_loss_error=nl,native_gradient_error=ng,math64_loss_error=ml,math64_gradient_error=mg))
        for key,g in generators.items():
            assert digests[key].hexdigest()==row['role_input_sha256'][key]
            assert g.bit_generator.state==row['RNG_final_states'][key]
        assert row['completed_steps']==512 and row['independent_observation_banks_consumed']==512*4*64
        near(m,t['final_rho_adam_m'],TOL['moment_abs']+TOL['moment_rel']*abs(t['final_rho_adam_m']))
        near(v,t['final_rho_adam_v'],TOL['moment_abs']+TOL['moment_rel']*abs(t['final_rho_adam_v']))
        for ai,arm in enumerate(ARMS):
            for kind in ['TAIL','LAST']:
                rho=t['rho'][ai,449:].astype(float).mean(axis=0) if kind=='TAIL' else t['rho'][ai,-1].astype(float)
                endpoint=row['outputs'][arm][kind];near(rho,endpoint['rho']);near(cov(rho),endpoint['covariance']);assert endpoint['scales']==[1.]*256
                near(np.linalg.eigvalsh(cov(rho))[0],endpoint['eigmin'],1e-12)
            diagnostic=row['diagnostics'][arm]
            for key in ['sigmoid_zero_count','sigmoid_one_count','probability_evaluations']:assert int(t[key][ai].sum())==diagnostic[key]
            assert int(t['repaired'][ai].sum())==diagnostic['feasibility_repair_count']
            for key,value in [('minimum_eigenvalue',t['eigmin'][ai].min()),('minimum_proposal_eigenvalue',t['proposal_eigmin'][ai].min()),('minimum_repair_alpha',t['projection_alpha'][ai].min()),('maximum_absolute_logit',t['maximum_absolute_logit'][ai].max())]:near(value,diagnostic[key])
        assert numerical_rows(row['numerical_controls'])==row['numerical_qualification']
        result=dict(passed=True,data_id=d,repeat=r,input_sha256=row['input_sha256'],output_sha256=row['output_sha256'],RNG_roles=8,
            replayed_Gaussian_scalars=3407872000,local_proposal_max_error=max_proposal,repair_max_error=max_repair,
            native_checks=native_checks,seconds=time.perf_counter()-start,
            scope='All saved-gradient moments rebuilt; local proposal checked from each saved before-state, then repair from saved proposal. No claim of an independently relearned cumulative rho trajectory.')
        save(OUT/'TRAIN'/f'{name}.json',result);return result
    except BaseException:
        save(OUT/'TRAIN'/f'ERROR_{name}.json',dict(data_id=d,repeat=r,last_step=step,traceback=traceback.format_exc(),automatic_retry=False));raise


def check_candidates(train,candidates):
    rows=candidates['candidates'];assert [x['index'] for x in rows]==list(range(138))
    old=read(ROOT/'artifacts/scbm_invariance/REFERENCE/MODEL.json')
    for j,t in enumerate([0.,.75]):
        source=next(x for x in old['records'] if x['t']==t)
        assert rows[j]['reused_pmf']==source['main_pmf'] and rows[j]['kind']=='old_reference'
        near(rows[j]['rho'],(1+t)*np.array([.25,-.15,.1,.1,-.15,.25]),0.)
    assert [(x['data_id'],x['repeat']) for x in train['records']]==[(d,r) for d in range(4) for r in range(16)]
    for d in range(4):
        for r in range(16):
            record=train['records'][d*16+r]
            for ai,arm in enumerate(ARMS):
                row=rows[2+(d*16+r)*2+ai]
                near(row['rho'],record['outputs'][arm]['TAIL']['rho'],0.)
                assert (row['data_id'],row['repeat'],row['arm'],row['kind'])==(d,r,arm,'TAIL')
        for ai,arm in enumerate(ARMS):
            point=np.mean([x['outputs'][arm]['TAIL']['rho'] for x in train['records'][d*16:(d+1)*16]],axis=0)
            near(point,rows[130+d*2+ai]['rho'],2e-15)
    for row in rows:
        near(cov(row['rho']),row['covariance'],0.);assert np.linalg.eigvalsh(cov(row['rho']))[0]>=.09999
    return rows


def eval_scores(active,noise,rows,mu):
    B,K,_=active.shape;out=np.empty((B,2,len(rows),16));nu=-np.logaddexp(0.,noise).sum(axis=-1)
    for j,row in enumerate(rows):
        x=active@np.linalg.cholesky(np.array(row['covariance'])).T+mu
        for yi,y in enumerate(BITS):
            la=-np.logaddexp(0.,-(2*y-1)*x).sum(axis=-1)
            out[:,0,j,yi]=math.log(K)-logsumexp(la+nu,axis=1)
            out[:,1,j,yi]=256*math.log(2.)+math.log(K)-logsumexp(la,axis=1)
    return out


def eval_native(active,noise,values,controls,rows,mu):
    checks=[]
    for item in controls['checks']:
        ci=item['candidate_index'];profile=item['profile'];pi=['raw','FACT'].index(profile)
        eps=torch.tensor(active,dtype=torch.float32);C=torch.tensor(rows[ci]['covariance'],dtype=torch.float32)
        x=eps@torch.linalg.cholesky(C).T+torch.tensor(mu,dtype=torch.float32)
        logits=torch.cat([x,torch.tensor(noise,dtype=torch.float32)],dim=2) if pi==0 else x
        prob=torch.sigmoid(logits.permute(0,2,1));actual=[]
        with torch.no_grad():
            for y in BITS:
                labels=np.tile(y,(2,1))
                if pi==0:labels=np.column_stack([labels,np.zeros((2,256),dtype=np.int64)])
                loss=core.official().compute_concept_loss(prob,torch.tensor(labels,dtype=torch.int64))
                if pi==1:loss=loss+256*math.log(2.)
                actual.append(float(loss))
        error=near(actual,item['native_total_score'],2e-6);mean=values[:,pi,ci,:].mean(axis=0)
        near(mean,item['mathematical_total_score']);near(np.array(actual)-mean,item['difference'],2e-6)
        assert item['passed']==(max(abs(x) for x in item['difference'])<=1e-4)
        checks.append(dict(candidate=ci,profile=profile,maximum_native_error=error))
    assert len(checks)*16==320
    return dict(calls=320,checks=checks)


def audit_eval(evaluation,rows,data,cfg):
    maximum=meanerr=0.;native=None
    assert [x['group'] for x in evaluation['groups']]==list(range(32))
    for group,record in enumerate(evaluation['groups']):
        assert sha(ROOT/record['prefix_npz'])==record['prefix_sha256']
        with np.load(ROOT/record['prefix_npz'],allow_pickle=False) as f:z={k:f[k] for k in f.files}
        for k,v in z.items():assert ah(v)==record['prefix_array_sha256'][k]
        for role,dim in [('active',4),('noise',256)]:
            seed=[cfg['seed_base'],group,cfg['roles'][role]];assert record['role_seed_sequences'][role]==seed
            rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)));assert rng.bit_generator.state==record['RNG_initial_states'][role];digest=hashlib.sha256()
            for block in range(16):
                x=rng.standard_normal((64,100,dim),dtype=np.float64);digest.update(x.tobytes(order='C'))
                if block==0:assert np.array_equal(x[:2],z[role+'_eps'])
            assert digest.hexdigest()==record['role_input_sha256'][role] and rng.bit_generator.state==record['RNG_final_states'][role]
        value=eval_scores(z['active_eps'],z['noise_eps'],rows,np.array(data['mu']))
        maximum=max(maximum,near(value,z['individual_total_score'],TOL['eval_score']))
        means=np.array([math.fsum(v)/16 for v in z['block_means'].reshape(16,-1).T]).reshape(2,138,16)
        meanerr=max(meanerr,near(means,record['means']))
        if group==0:native=eval_native(z['active_eps'],z['noise_eps'],value,record['numerical_controls'],rows,np.array(data['mu']))
        else:assert record['numerical_controls'] is None
        save(OUT/'EVAL'/f'group_{group:02}.json',dict(passed=True,group=group,prefix_sha256=record['prefix_sha256'],score_cells=2*2*138*16))
    assert evaluation['numerical_qualification']==evaluation['groups'][0]['numerical_controls']
    return dict(passed=True,groups=32,replayed_Gaussian_scalars=851968000,prefix_score_cells=282624,
                maximum_prefix_error=maximum,maximum_block_mean_error=meanerr,native=native)


def independent_pmf(task):
    index,row,mu,expected=task;q=36;nodes,weights=np.polynomial.hermite.hermgauss(q)
    nodes=np.sqrt(2.)*nodes;weights=weights/math.sqrt(math.pi)
    eig,U=np.linalg.eigh(np.array(row['covariance']));S=(U*np.sqrt(eig))@U.T;total=np.zeros(16,np.longdouble)
    for start in range(0,q**4,65536):
        n=np.arange(start,min(start+65536,q**4));ids=np.column_stack([(n//q**p)%q for p in [3,2,1,0]])
        prob=expit(nodes[ids]@S.T+mu);w=weights[ids].prod(axis=1)
        for yi,y in enumerate(BITS):total[yi]+=np.sum(w*np.where(y,prob,1-prob).prod(axis=1),dtype=np.longdouble)
    values=total.astype(float);error=float(np.max(abs(values-np.array(expected))))
    result=dict(index=index,pmf=values.tolist(),maximum_difference=error,normalization_error=abs(math.fsum(values)-1),
        order=36,root='symmetric',nodes=q**4,passed=error<=5e-8 and abs(math.fsum(values)-1)<=2e-12)
    save(OUT/'REFERENCE'/f'candidate_{index:03}.json',result);return result


def reference_arithmetic(reference,data,rows):
    p=data['population_weights'];all_flags=[];deltas=[];nodes=0
    assert [x['index'] for x in reference['records']]==list(range(138))
    for row in reference['records']:
        j=row['index'];pmf=row['main_pmf'];near(row['reference_joint_KL'],math.fsum(w*math.log(w/v) for w,v in zip(p,pmf)))
        assert pmf and all(math.isfinite(v) and v>0 for v in pmf)
        near(row['rho'],rows[j]['rho'],0.)
        near(row['reference_cross_entropy_P'],-math.fsum(w*math.log(v) for w,v in zip(p,pmf)))
        near(row['reference_cross_entropy_D'],[-math.fsum(w*math.log(v) for w,v in zip(d['weights'],pmf)) for d in data['data']])
        if j<2:
            assert row['PMF_reused'] is True and row['variants']==[] and pmf==rows[j]['reused_pmf']
            q=row['qualifications'][0];norm=abs(math.fsum(pmf)-1)
            near(norm,q['normalization_error']);assert q['positive'] and q['passed']==(norm<=2e-12)
        else:
            assert row['PMF_reused'] is False and [v['order'] for v in row['variants']]==[24,32,40]
            assert pmf==row['variants'][1]['pmf']
            for v,q in zip(row['variants'],row['qualifications']):
                values=v['pmf'];delta=max(abs(x-y) for x,y in zip(values,pmf));norm=abs(math.fsum(values)-1)
                positive=len(values)==16 and all(math.isfinite(x) and x>0 for x in values)
                near([delta,norm],[q['maximum_change'],q['normalization_error']])
                assert q['positive']==positive and q['passed']==(positive and norm<=2e-12 and delta<=5e-8)
                assert v['gaussian_nodes']==v['order']**4;nodes+=v['gaussian_nodes'];deltas.append(delta)
                near(v['covariance'],rows[j]['covariance'],0.)
        assert row['passed']==all(q['passed'] for q in row['qualifications'])
        all_flags.append(row['passed'])
    assert reference['qualification_passed']==all(all_flags)
    near(reference['maximum_PMF_change'],max(deltas))
    assert nodes==reference['accounting']['Gaussian_nodes']==535887872
    assert reference['accounting']['PMF_integrations']==408 and reference['accounting']['LAST_integrations']==0
    return dict(candidates=138,risks_reconstructed=138,production_PMFs=408,production_nodes=nodes,
                qualification_flags_independently_rebuilt=True)


def train_stats_vectors(rho,kl):
    risk=[math.fsum(kl[d,r,1]-kl[d,r,0] for d in range(4))/4 for r in range(16)]
    distances=np.zeros((4,8,2))
    for d,j,a in itertools.product(range(4),range(8),range(2)):
        distances[d,j,a]=math.fsum((rho[d,2*j,a,k]-rho[d,2*j+1,a,k])**2 for k in range(6))/2
    dispersion=[math.fsum(distances[d,j,1]-distances[d,j,0] for d in range(4))/4 for j in range(8)]
    return np.array(risk),np.array(dispersion),distances


def eval_stats_vectors(means,data):
    p=data['population_weights'];D=np.empty((len(means),2));P=D.copy()
    for g,profile in itertools.product(range(len(means)),range(2)):
        dv=[];pv=[]
        for d,r in itertools.product(range(4),range(16)):
            i=2+(d*16+r)*2;delta=means[g,profile,i+1]-means[g,profile,i]
            dv.append(math.fsum(w*x for w,x in zip(data['data'][d]['weights'],delta)))
            pv.append(math.fsum(w*x for w,x in zip(p,delta)))
        D[g,profile]=math.fsum(dv)/64;P[g,profile]=math.fsum(pv)/64
    return D,P


def verify_summary(row,values,threshold,expected=None):
    m,se=moments(values);df=len(values)-1;crit=float(student.isf(.025/4,df));lo=m-crit*se;hi=m+crit*se
    near(values,row['values']);near([m,se,crit,lo,hi],[row[k] for k in ['mean','SE','critical_t','lower','upper']])
    assert row['df']==df and row['alpha_family']==.025 and row['family_size']==2 and row['SE_threshold']==threshold
    assert row['direction']==('lower' if hi<0 else 'higher' if lo>0 else 'unresolved')
    assert row['resolved']==(lo>0 or hi<0) and row['precision_passed']==(se<=threshold)
    assert row['expected']==expected and row['expected_supported']==(hi<0 if expected=='lower' else None)


def statistics(result,train,evaluation,reference,data):
    rho=np.array([[r['outputs'][a]['TAIL']['rho'] for a in ARMS] for r in train['records']]).reshape(4,16,2,6)
    kl=np.array([r['reference_joint_KL'] for r in reference['records'][2:130]]).reshape(4,16,2)
    risk,disp,distances=train_stats_vectors(rho,kl)
    verify_summary(result['training_primary'][0],risk,.0005);verify_summary(result['training_primary'][1],disp,.05,'lower')
    near(result['rho_by_data_repeat_arm'],rho);near(result['KL_by_data_repeat_arm'],kl)
    near(result['mean_KL_by_arm'],[math.fsum(kl[:,:,a].ravel())/64 for a in range(2)])
    near(result['training_mean_risk_difference'],math.fsum(risk)/16)
    near(result['mean_disjoint_conditional_dispersion_by_arm'],distances.mean(axis=(0,1)))
    truth=np.array(data['truth_rho'])
    for row in result['parameter_descriptive']:
        d=row['data_id'];a=ARMS.index(row['arm']);points=rho[d,:,a]
        mean=np.array([math.fsum(points[:,k])/16 for k in range(6)])
        S=math.fsum(float(x)**2 for x in (points-mean).ravel())/15
        bias=math.fsum((mean-truth)**2)-S/16;mse=math.fsum(float(x)**2 for x in (points-truth).ravel())/16
        near(row['mean_rho'],mean);near([row[k] for k in ['sample_trace_covariance_unbiased','debiased_conditional_mean_parameter_error','sample_parameter_MSE','identity_error']],[S,bias,mse,mse-S-bias])
        near(row['mean_KL'],math.fsum(kl[d,:,a])/16);near(row['mean_matrix_KL'],reference['records'][130+d*2+a]['reference_joint_KL'])
        near(row['disjoint_pair_half_squared_distances'],distances[d,:,a])
    means=np.array([r['means'] for r in evaluation['groups']]);D,P=eval_stats_vectors(means,data)
    verify_summary(result['evaluation_primary'][0],D[:,0],.0002);verify_summary(result['evaluation_primary'][1],P[:,0],.0002)
    for name,vec in [('D',D[:,1]),('P',P[:,1])]:
        m,se=moments(vec);near(m,result['evaluation_FACT_descriptive'][name+'_mean']);near(se,result['evaluation_FACT_descriptive'][name+'_SE'])
    for row in result['score_descriptive']:
        d=row['data_id'];a=ARMS.index(row['arm']);profile=['raw','FACT'].index(row['profile'])
        w=data['population_weights'] if row['weights']=='P' else data['data'][d]['weights'];z=[];jm=[]
        for group in means:
            z.append(math.fsum(math.fsum(w[y]*group[profile,2+(d*16+r)*2+a,y] for y in range(16)) for r in range(16))/16)
            jm.append(math.fsum(w[y]*group[profile,130+d*2+a,y] for y in range(16)))
        m,se=moments(z);near(m,row['mean_score']);near(se,row['MC_SE']);near(math.fsum(x-y for x,y in zip(z,jm))/32,row['finite_panel_Jensen_mean'])
    for d,row in enumerate(result['initial_gradient_descriptive']):
        banks=[]
        for rec in train['records'][16*d:16*(d+1)]:
            with np.load(ROOT/rec['output_file'],allow_pickle=False) as f:b=f['bank_gradient_L4'][0].astype(float);one=f['bank_gradient_L1'][0,0].astype(float)
            assert np.array_equal(one,b[0]);banks.append(b)
        b=np.array(banks);single=b.reshape(64,6);average=b.mean(axis=1)
        variance=lambda x:math.fsum(float(y)**2 for y in (x-x.mean(axis=0)).ravel())/(len(x)-1)
        vs,va=variance(single),variance(average)
        near([vs,va,va/vs],[row[k] for k in ['single_trace_variance','average_trace_variance','ratio']])
    assert result['reference_qualification']==reference['qualification_passed']
    assert result['training_numerical_qualification']==train['numerical_qualification']
    assert result['evaluation_numerical_qualification']==evaluation['numerical_qualification']
    return dict(passed=True,training_units=[16,8],training_df=[15,7],evaluation_units=32,
                primary_families=2,primary_contrasts=4,parameter_descriptions=8,score_descriptions=32,
                negative_debiased_estimates_retained=True,new_tests=0)


def check():
    assert not (BASE/'FREEZE.json').exists();cfg,data=data_check()
    spec=importlib.util.spec_from_file_location('_scbmavg_stats_compare',ROOT/'script/scbmavg_results.py');prod=importlib.util.module_from_spec(spec);spec.loader.exec_module(prod)
    rho=np.empty((4,16,2,6));kl=np.empty((4,16,2))
    for d,r,a,k in itertools.product(range(4),range(16),range(2),range(6)):rho[d,r,a,k]=.1*math.sin((d+1)*(r+2)*(a+3)*(k+4)*.17)
    for d,r,a in itertools.product(range(4),range(16),range(2)):kl[d,r,a]=.03+(d+1)*(r+1)*(a+2)/10000
    x=train_stats_vectors(rho,kl);y=prod.train_vectors(rho,kl)
    for xx,yy in zip(x,y):near(xx,yy,2e-14)
    values=np.empty((3,2,138,16))
    for g,h,j,y in itertools.product(range(3),range(2),range(138),range(16)):values[g,h,j,y]=(g+1)*(j+2)*(y+3)/177+(h+1)*j*j/993
    xx=eval_stats_vectors(values,data);yy=prod.eval_vectors(values,data)
    for x,y in zip(xx,yy):near(x,y,2e-12)
    # Reused analytic helper checked locally against finite differences, no RNG.
    eps=np.sin(np.arange(100*3*4)*.03).reshape(100,3,4);noise=np.cos(np.arange(100*3*256)*.007).reshape(100,3,256)
    yl=(np.arange(12).reshape(3,4)%2).astype(np.int8);yn=(np.arange(768).reshape(3,256)%3==0).astype(np.int8);rr=np.array([.11,-.07,.05,.09,-.06,.13]);answer=core.math_score('FIXED',rr,None,eps,noise,yl,yn)
    for j in range(6):
        plus=rr.copy();minus=rr.copy();plus[j]+=1e-5;minus[j]-=1e-5
        fd=(core.math_score('FIXED',plus,None,eps,noise,yl,yn)['loss']-core.math_score('FIXED',minus,None,eps,noise,yl,yn)['loss'])/2e-5
        near(fd,answer['rho_gradient'][j],2e-8)
    save(OUT/'SELFTEST.json',dict(passed=True,source_sha256=sha(__file__),independent_helper_sha256=sha(core.__file__),
        config_sha256=sha(BASE/'CONFIG.json'),data_sha256=sha(BASE/'DATA.json'),scope=SCOPE,
        checks=['four old datasets/hash/labels','nonexchangeable two-level statistic fixtures','six-coordinate deterministic finite differences'],
        new_scientific_RNG=0,new_model_integrals=0,new_training=0))
    print(json.dumps(dict(passed=True,source_sha256=sha(__file__))),flush=True)


def run():
    verify_freeze();cfg,data=data_check();test=read(OUT/'SELFTEST.json');assert test['passed'] and test['source_sha256']==sha(__file__)
    assert not (OUT/'EXECUTION.json').exists() and not (OUT/'RESULTS.json').exists()
    train,reference,evaluation,result,candidates=[read(BASE/p) for p in ['TRAIN/RESULTS.json','REFERENCE/RESULTS.json','EVAL/RESULTS.json','RESULTS.json','CANDIDATES.json']]
    rows=check_candidates(train,candidates)
    files=[BASE/p for p in ['FREEZE.json','DERIVED_FREEZE.json','CONFIG.json','DATA.json','TRAIN/RESULTS.json','REFERENCE/RESULTS.json','EVAL/RESULTS.json','RESULTS.json','CANDIDATES.json']]
    inputs={rel(p):sha(p) for p in files}
    for row in train['records']:
        inputs[row['input_file']]=row['input_sha256'];inputs[row['output_file']]=row['output_sha256']
    for row in evaluation['groups']:inputs[row['prefix_npz']]=row['prefix_sha256']
    save(OUT/'EXECUTION.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
        freeze_sha256=sha(BASE/'FREEZE.json'),derived_freeze_sha256=sha(BASE/'DERIVED_FREEZE.json'),input_files=inputs,scope=SCOPE,automatic_retry=False))
    start=time.perf_counter();completed=[]
    try:
        with ProcessPoolExecutor(max_workers=8) as pool:
            futures=[pool.submit(training_pair,row) for row in train['records']]
            try:
                for future in as_completed(futures):
                    row=future.result();completed.append(row)
                    save(OUT/'PROGRESS.json',dict(completed_training_pairs=sorted((r['data_id'],r['repeat']) for r in completed),seconds=time.perf_counter()-start),False)
                    print(json.dumps(dict(audited_data_id=row['data_id'],repeat=row['repeat'],seconds=row['seconds'])),flush=True)
            except BaseException:
                for future in futures:future.cancel()
                raise
        completed.sort(key=lambda r:(r['data_id'],r['repeat']))
        controls=[x for r in train['records'] for x in r['numerical_controls']]
        assert numerical_rows(controls)==train['numerical_qualification']
        native=[x for row in completed for x in row['native_checks']];assert len(native)==80
        ev=audit_eval(evaluation,rows,data,cfg['evaluation'])
        refarith=reference_arithmetic(reference,data,rows)
        tasks=[(i,rows[i],np.array(data['mu']),reference['records'][i]['main_pmf']) for i in SCOPE['reference_indices']]
        with ProcessPoolExecutor(max_workers=4) as pool:refchecks=list(pool.map(independent_pmf,tasks))
        stat=statistics(result,train,evaluation,reference,data)
        verify_freeze()
        for p,h in inputs.items():assert sha(ROOT/p)==h,p
        passed=all(r['passed'] for r in refchecks)
        out=dict(passed=passed,remaining_required_fixes=[] if passed else ['Independent PMF tolerance failure; no automatic refinement.'],
            source_sha256=sha(__file__),freeze_sha256=sha(BASE/'FREEZE.json'),derived_freeze_sha256=sha(BASE/'DERIVED_FREEZE.json'),input_files=inputs,
            training_audit=dict(pairs=64,RNG_roles=512,replayed_Gaussian_scalars=218103808000,
                all_hashes_states_match=True,all_saved_averages_Adam_and_repairs_checked=True,
                local_proposal_max_error=max(r['local_proposal_max_error'] for r in completed),repair_max_error=max(r['repair_max_error'] for r in completed),
                native_aggregate_states=32,native_bank_calls=80,native_checks=native,
                native_loss_max_error=max(x['native_loss_error'] for x in native),native_gradient_max_error=max(x['native_gradient_error'] for x in native),
                math64_loss_max_error=max(x['math64_loss_error'] for x in native),math64_gradient_max_error=max(x['math64_gradient_error'] for x in native)),
            evaluation_audit=ev,reference_audit=dict(saved_arithmetic=refarith,independent_checks=refchecks,
                independent_PMFs=16,independent_four_dimensional_nodes=26873856),statistics_audit=stat,
            seconds=time.perf_counter()-start,new_independent_scientific_draws=0,new_training=0,
            new_audit_model_integrals=dict(PMFs=16,pattern_integrals=256,four_dimensional_nodes=26873856),
            reused_independent_cores=dict(path=rel(Path(core.__file__)),sha256=sha(core.__file__),functions=['math_score','native','adam','repair']),
            scope='Full original inputs replay and saved arithmetic, bounded native/math prefix, all statistics. Not independent full relearning or complete EVAL rescoring. Reuses previous independent audit cores, never producer evaluators.')
        save(OUT/'RESULTS.json',out);print(json.dumps(dict(passed=passed,seconds=out['seconds'],results_sha256=sha(OUT/'RESULTS.json'))),flush=True)
    except BaseException:
        save(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),completed_training_pairs=[(r['data_id'],r['repeat']) for r in completed],
            seconds=time.perf_counter()-start,scope=SCOPE,automatic_retry=False));raise


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf8');sys.stderr.reconfigure(encoding='utf8')
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['check','run']);a=p.parse_args();check() if a.mode=='check' else run()
