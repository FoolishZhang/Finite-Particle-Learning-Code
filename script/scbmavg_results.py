"""Prespecified two-level conditional inference; no new draws or quadratures."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,math
import numpy as np
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/scbm_bank_averaging'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def verify():
    root=read(BASE/'FREEZE.json');derived=read(BASE/'DERIVED_FREEZE.json');assert root['run_authorized'] and derived['run_authorized'] and derived['root_freeze_sha256']==sha(BASE/'FREEZE.json')
    for record in [root,derived]:
        for p,h in record['inputs'].items():assert sha(ROOT/p)==h,p
def summary(name,values,alpha,q,precision,expected=None):
    v=np.asarray(values,dtype=float);df=len(v)-1;critical=float(t.ppf(1-alpha/(2*q),df));mean=float(v.mean());se=float(v.std(ddof=1)/np.sqrt(len(v)));lo=mean-critical*se;hi=mean+critical*se
    return dict(name=name,values=v.tolist(),mean=mean,SE=se,df=df,critical_t=critical,lower=lo,upper=hi,direction='lower' if hi<0 else 'higher' if lo>0 else 'unresolved',resolved=bool(lo>0 or hi<0),SE_threshold=precision,precision_passed=bool(se<=precision),expected=expected,expected_supported=(bool(hi<0) if expected=='lower' else None),alpha_family=alpha,family_size=q,scope='Two-sided Bonferroni development t interval; not an exact finite-sample certificate.')
def train_vectors(rho,kl):
    rho=np.asarray(rho);kl=np.asarray(kl);assert rho.shape==(4,16,2,6) and kl.shape==(4,16,2)
    risk=(kl[:,:,1]-kl[:,:,0]).mean(axis=0)
    pair_distance=.5*((rho[:,0::2]-rho[:,1::2])**2).sum(axis=3)
    dispersion=(pair_distance[:,:,1]-pair_distance[:,:,0]).mean(axis=0)
    return risk,dispersion,pair_distance
def eval_vectors(means,data):
    a=np.asarray(means);assert a.shape[1:]==(2,138,16)
    tail=a[:,:,2:130].reshape(a.shape[0],2,4,16,2,16)
    delta=tail[:,:,:,:,1,:]-tail[:,:,:,:,0,:]
    d=np.array([x['weights'] for x in data['data']]);p=np.array(data['population_weights'])
    # [group,profile,data,repeat,pattern], weight within own data, then panel mean.
    D=(delta*d[None,None,:,None,:]).sum(axis=-1).mean(axis=(2,3))
    P=(delta*p).sum(axis=-1).mean(axis=(2,3))
    return D,P
def run():
    verify();cfg=read(BASE/'CONFIG.json');data=read(BASE/'DATA.json');train=read(BASE/'TRAIN/RESULTS.json');ref=read(BASE/'REFERENCE/RESULTS.json');evaluation=read(BASE/'EVAL/RESULTS.json');cand=read(BASE/'CANDIDATES.json')
    for record,module in [(train,'run'),(ref,'reference'),(evaluation,'eval')]:
        assert record['freeze_sha256']==sha(BASE/'FREEZE.json')
        assert record['source_sha256']==sha(ROOT/f'script/scbmavg_{module}.py')
    assert train['kernel_sha256']==sha(ROOT/'script/scbmavg_kernel.py')
    assert ref['derived_freeze_sha256']==evaluation['derived_freeze_sha256']==sha(BASE/'DERIVED_FREEZE.json')
    assert [x['group'] for x in evaluation['groups']]==list(range(32))
    assert [(x['data_id'],x['repeat']) for x in train['records']]==[(d,r) for d in range(4) for r in range(16)]
    assert [x['index'] for x in ref['records']]==list(range(138))
    rho=np.array([[x['outputs'][arm]['TAIL']['rho'] for arm in ['L1','L4']] for x in train['records']]).reshape(4,16,2,6)
    kl=np.array([x['reference_joint_KL'] for x in ref['records'][2:130]]).reshape(4,16,2)
    risk,dispersion,pair_distance=train_vectors(rho,kl);tf=cfg['training_family']
    primary_train=[summary(tf['risk_name'],risk,.025,2,.0005),summary(tf['dispersion_name'],dispersion,.025,2,.05,'lower')]
    truth=np.array(data['truth_rho']);parameters=[]
    for d in range(4):
        for a,arm in enumerate(['L1','L4']):
            points=rho[d,:,a];mean=points.mean(axis=0);S=float(((points-mean)**2).sum()/15);bias=float(((mean-truth)**2).sum()-S/16);mse=float(((points-truth)**2).sum(axis=1).mean());assert abs(mse-S-bias)<1e-13
            parameters.append(dict(data_id=d,arm=arm,mean_rho=mean.tolist(),sample_trace_covariance_unbiased=S,debiased_conditional_mean_parameter_error=bias,sample_parameter_MSE=mse,identity_error=mse-S-bias,mean_KL=float(kl[d,:,a].mean()),mean_matrix_KL=ref['records'][130+d*2+a]['reference_joint_KL'],disjoint_pair_half_squared_distances=pair_distance[d,:,a].tolist(),scope='Parameter identity only; negative debiased mean-error estimates retained. No KL causal decomposition.'))
    means=np.array([x['means'] for x in evaluation['groups']]);D,P=eval_vectors(means,data)
    primary_eval=[summary('raw_D_L4_minus_L1',D[:,0],.025,2,.0002),summary('raw_P_L4_minus_L1',P[:,0],.025,2,.0002)]
    # Descriptive scoring of the same candidate sets and mean matrices; no extra intervals.
    score_descriptive=[];truthp=np.array(data['population_weights'])
    for d in range(4):
        w=np.array(data['data'][d]['weights'])
        for a,arm in enumerate(['L1','L4']):
            indices=np.array([2+(d*16+r)*2+a for r in range(16)])
            for profile in range(2):
                for label,weight in [('D',w),('P',truthp)]:
                    z=(means[:,profile,indices,:]*weight).sum(axis=-1).mean(axis=1)
                    mean_score=(means[:,profile,130+d*2+a,:]*weight).sum(axis=-1)
                    score_descriptive.append(dict(data_id=d,arm=arm,profile=['raw','FACT'][profile],weights=label,mean_score=float(z.mean()),MC_SE=float(z.std(ddof=1)/np.sqrt(32)),finite_panel_Jensen_mean=float((z-mean_score).mean()),scope='Conditional on realized trained matrices; descriptive, no additional simultaneous interval or training-population inference.'))
    initial_grad=[]
    for d in range(4):
        banks=[]
        for rec in [x for x in train['records'] if x['data_id']==d]:
            assert sha(ROOT/rec['output_file'])==rec['output_sha256']
            with np.load(ROOT/rec['output_file'],allow_pickle=False) as f:
                b=f['bank_gradient_L4'][0].astype(float);one=f['bank_gradient_L1'][0,0].astype(float)
                assert np.array_equal(one,b[0]);banks.append(b)
        b=np.array(banks);single=b.reshape(64,6);average=b.mean(axis=1);trace_one=float(single.var(axis=0,ddof=1).sum());trace_avg=float(average.var(axis=0,ddof=1).sum())
        initial_grad.append(dict(data_id=d,single_trace_variance=trace_one,average_trace_variance=trace_avg,ratio=trace_avg/trace_one,bank0_identical_at_initial_state=True,scope='At I and this dataset first batch only; empirical correlated ratio, no confidence interval or gate. No later-trajectory 1/4 claim.'))
    paths=[BASE/x for x in ['TRAIN/RESULTS.json','REFERENCE/RESULTS.json','EVAL/RESULTS.json','CANDIDATES.json','CONFIG.json','DATA.json']]
    inputs={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    inputs.update({x['output_file']:x['output_sha256'] for x in train['records']})
    result=dict(source_sha256=sha(__file__),freeze_sha256=sha(BASE/'FREEZE.json'),derived_freeze_sha256=sha(BASE/'DERIVED_FREEZE.json'),created_utc=datetime.now(timezone.utc).isoformat(),inputs=inputs,training_primary=primary_train,evaluation_primary=primary_eval,parameter_descriptive=parameters,score_descriptive=score_descriptive,initial_gradient_descriptive=initial_grad,KL_by_data_repeat_arm=kl.tolist(),rho_by_data_repeat_arm=rho.tolist(),mean_KL_by_arm=kl.mean(axis=(0,1)).tolist(),training_mean_risk_difference=float(risk.mean()),mean_disjoint_conditional_dispersion_by_arm=pair_distance.mean(axis=(0,1)).tolist(),evaluation_FACT_descriptive=dict(D_mean=float(D[:,1].mean()),D_SE=float(D[:,1].std(ddof=1)/np.sqrt(32)),P_mean=float(P[:,1].mean()),P_SE=float(P[:,1].std(ddof=1)/np.sqrt(32))),reference_qualification=ref['qualification_passed'],training_numerical_qualification=train['numerical_qualification'],evaluation_numerical_qualification=evaluation['numerical_qualification'],new_RNG=0,new_model_integrals=0,new_training=0,scope='Training q2 and conditional evaluation q2 are separate families, alpha .025 each. Fixed four old datasets. No original precision gate is retrospectively altered.')
    save(BASE/'RESULTS.json',result)
    print(json.dumps(dict(training=primary_train,evaluation=primary_eval,KL=result['mean_KL_by_arm'],dispersion=result['mean_disjoint_conditional_dispersion_by_arm']),ensure_ascii=False))
def selftest():
    assert not (BASE/'FREEZE.json').exists()
    rho=np.empty((4,16,2,6));kl=np.empty((4,16,2))
    for d in range(4):
        for r in range(16):
            rho[d,r,0]=r*np.arange(1,7)/100;rho[d,r,1]=rho[d,r,0]/2
            kl[d,r]=[d+r/100,d+r/100-.02]
    risk,disp,pairs=train_vectors(rho,kl);assert np.allclose(risk,-.02);assert np.allclose(disp,-.0034125)
    data=dict(data=[dict(weights=np.eye(16)[d].tolist()) for d in range(4)],population_weights=(np.arange(1,17)/136).tolist())
    scores=np.empty((3,2,138,16))
    for g in range(3):
        for h in range(2):scores[g,h]=np.arange(138)[:,None]*np.arange(1,17)[None,:]+g+h
    D,P=eval_vectors(scores,data);assert np.allclose(D,2.5);assert np.allclose(P,np.arange(1,17)@np.array(data['population_weights']))
    save(BASE/'STATISTICS_SELFTEST.json',dict(passed=True,source_sha256=sha(__file__),new_RNG=0,new_model_integrals=0,checked='Risk column and independent disjoint-pair units; own-data asymmetric pattern weights and candidate-index mapping.'))
    print(json.dumps(dict(passed=True,new_RNG=0)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['check','run']);a=p.parse_args();selftest() if a.mode=='check' else run()
