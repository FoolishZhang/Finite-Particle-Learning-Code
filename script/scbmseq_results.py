"""Frozen outer-dataset risk and score contrasts; no additional scientific draws."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json
import numpy as np
import scbmavg_results as helper
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/scbm_sequential_generalization'
HELPER_SHA='d8dca7eb961fd6b69e5131df283adc265abcc026b86bbd9cf1a44265eeab6a3a'
ARMS=['IN4','OUT4','SEQ4','FACT'];CONTRASTS=[(1,0),(2,1)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,x):
    with p.open('x',encoding='utf8',newline='\n') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def verify():
    assert sha(helper.__file__)==HELPER_SHA
    for n in ['FREEZE.json','DATA_FREEZE.json','DERIVED_FREEZE.json']:
        v=read(BASE/n);assert v['run_authorized']
        if n!='FREEZE.json':assert v['root_freeze_sha256']==sha(BASE/'FREEZE.json')
        if n=='DERIVED_FREEZE.json':assert v['data_freeze_sha256']==sha(BASE/'DATA_FREEZE.json')
        for p,h in v['inputs'].items():assert sha(ROOT/p)==h,p
def check_config(c):
    assert c['data_ids']==list(range(16)) and c['repeats']==2 and c['arms']==ARMS
    t=c['training_family'];e=c['evaluation']
    assert (t['alpha'],t['size'],t['df'])==(.025,5,15) and t['SE_thresholds']==[.0005,.0005,.0002,.0002,.0002]
    assert t['primary_order']==['OUT4_minus_IN4','SEQ4_minus_OUT4','OUT4_minus_I','SEQ4_minus_I','FACT_minus_I']
    assert (e['alpha'],e['family_size'],e['df'],e['SE_threshold'],e['groups'],e['banks_per_group'],e['local_candidates'])==(.025,2,15,.0005,8,512,10)
    assert e['primary_order']==['raw_P_OUT4_minus_IN4','raw_P_SEQ4_minus_OUT4']
def risk_vectors(kl,identity):
    x=np.asarray(kl,float);assert x.shape==(16,2,4);mean=x.mean(axis=1)
    return np.stack([mean[:,1]-mean[:,0],mean[:,2]-mean[:,1],mean[:,1]-identity,mean[:,2]-identity,mean[:,3]-identity],axis=1)
def eval_vectors(means,pweights,dweights):
    m=np.asarray(means,float);assert m.shape==(16,8,2,10,16)
    P=np.einsum('dghcy,y->dghc',m,np.asarray(pweights));D=np.einsum('dghcy,dy->dghc',m,np.asarray(dweights))
    # Preserve data and group until the within-data averages have been formed.
    Ps=P[:,:,:,2:].reshape(16,8,2,2,4).mean(axis=3);Ds=D[:,:,:,2:].reshape(16,8,2,2,4).mean(axis=3)
    return Ps,Ds,P[:,:,:,:2],D[:,:,:,:2]
def parameter_vectors(rho,truth):
    e=np.asarray(rho)-np.asarray(truth);assert e.shape==(16,2,4,6)
    half=.5*((e[:,0]-e[:,1])**2).sum(axis=-1);cross=(e[:,0]*e[:,1]).sum(axis=-1);mse=(e*e).sum(axis=-1).mean(axis=1)
    assert np.max(abs(half+cross-mse))<1e-12
    return half,cross,mse
def run():
    verify();cfg=read(BASE/'CONFIG.json');check_config(cfg);data=read(BASE/'DATA.json');train=read(BASE/'TRAIN/RESULTS.json');ref=read(BASE/'REFERENCE/RESULTS.json');ev=read(BASE/'EVAL/RESULTS.json');cand=read(BASE/'CANDIDATES.json')
    for x,name in [(train,'run'),(ref,'reference'),(ev,'eval')]:assert x['source_sha256']==sha(ROOT/f'script/scbmseq_{name}.py') and x['freeze_sha256']==sha(BASE/'FREEZE.json')
    assert train['kernel_sha256']==sha(ROOT/'script/scbmseq_kernel.py')
    for x in [ref,ev]:assert x['derived_freeze_sha256']==sha(BASE/'DERIVED_FREEZE.json') and x['candidates_sha256']==sha(BASE/'CANDIDATES.json')
    assert ref['training_results_sha256']==sha(BASE/'TRAIN/RESULTS.json')
    assert [(x['data_id'],x['repeat']) for x in train['records']]==[(d,r) for d in range(16) for r in range(2)]
    assert [(x['data_id'],x['group']) for x in ev['groups']]==[(d,g) for d in range(16) for g in range(8)]
    assert [x['index'] for x in ref['records']]==list(range(130))
    for row in ev['groups']:assert row['candidate_indices']==[0,1]+list(range(2+8*row['data_id'],10+8*row['data_id']))
    rho=np.array([[x['outputs'][a]['TAIL']['rho'] for a in ARMS] for x in train['records']]).reshape(16,2,4,6)
    members=np.array([[x['members'][f'M{i}']['TAIL']['rho'] for i in range(4)] for x in train['records']]).reshape(16,2,4,6)
    assert np.max(abs(members.mean(axis=2)-rho[:,:,1]))<2e-15
    kl=np.array([x['reference_joint_KL'] for x in ref['records'][2:]]).reshape(16,2,4);identity=ref['records'][1]['reference_joint_KL']
    risks=risk_vectors(kl,identity);tf=cfg['training_family']
    training=[helper.summary('TAIL_KL_'+name,risks[:,j],.025,5,tf['SE_thresholds'][j]) for j,name in enumerate(tf['primary_order'])]
    means=np.array([x['means'] for x in ev['groups']]).reshape(16,8,2,10,16);Ps,Ds,Pbase,Dbase=eval_vectors(means,data['population_weights'],[x['weights'] for x in data['data']]);Pm=Ps.mean(axis=1);Dm=Ds.mean(axis=1)
    score_group=np.stack([Ps[:,:,0,a]-Ps[:,:,0,b] for a,b in CONTRASTS],axis=-1);scores=score_group.mean(axis=1)
    evaluation=[helper.summary(name,scores[:,j],.025,2,.0005) for j,name in enumerate(cfg['evaluation']['primary_order'])]
    entropy=float(-np.dot(data['population_weights'],np.log(data['population_weights']))+256*np.log(2));gap=Pm[:,0,:]-entropy-kl.mean(axis=1)
    gap_rows=[]
    for j,name in enumerate(tf['primary_order'][:2]):
        values=scores[:,j]-risks[:,j];a,b=CONTRASTS[j]
        assert np.max(abs(values-(gap[:,a]-gap[:,b])))<1e-12
        gap_rows.append(dict(name='J_P_'+name,values=values.tolist(),mean=float(values.mean()),lower=evaluation[j]['lower']-training[j]['upper'],upper=evaluation[j]['upper']-training[j]['lower'],source_primary_indices={'evaluation':j,'training':j},scope='Derived rectangular interval from original q5 and q2 simultaneous development intervals; no new test, no independence of F/R assumed. Reference integral uncertainty excluded.'))
    half,cross,mse=parameter_vectors(rho,data['truth_rho']);desc=[]
    for d in range(16):
        for a,arm in enumerate(ARMS):desc.append(dict(data_id=d,arm=arm,mean_KL=float(kl[d,:,a].mean()),mean_rho=rho[d,:,a].mean(axis=0).tolist(),half_squared_distance=float(half[d,a]),cross_product_error=float(cross[d,a]),parameter_MSE=float(mse[d,a]),raw_P=float(Pm[d,0,a]),raw_D=float(Dm[d,0,a]),FACT_P=float(Pm[d,1,a]),FACT_D=float(Dm[d,1,a]),raw_J_P=float(gap[d,a]),scope='Descriptive only; two repeat parameter identity is not a KL causal decomposition.'))
    files=[BASE/n for n in ['TRAIN/RESULTS.json','REFERENCE/RESULTS.json','EVAL/RESULTS.json','CANDIDATES.json','CONFIG.json','DATA.json']]
    result=dict(schema='scbmseq-results-v1',source_sha256=sha(__file__),helper_sha256=HELPER_SHA,freeze_sha256=sha(BASE/'FREEZE.json'),data_freeze_sha256=sha(BASE/'DATA_FREEZE.json'),derived_freeze_sha256=sha(BASE/'DERIVED_FREEZE.json'),created_utc=datetime.now(timezone.utc).isoformat(),inputs={p.relative_to(ROOT).as_posix():sha(p) for p in files},training_primary=training,evaluation_primary=evaluation,gap_derived=gap_rows,rho_by_data_repeat_arm=rho.tolist(),member_rho_by_data_repeat_member=members.tolist(),KL_by_data_repeat_arm=kl.tolist(),mean_KL_by_arm=kl.mean(axis=(0,1)).tolist(),identity_reference_risk=float(identity),entropy_reference=entropy,risk_outer_values=risks.tolist(),raw_P_outer_values=scores.tolist(),raw_P_group_values=score_group.tolist(),within_data_MC_SE=(score_group.std(axis=1,ddof=1)/np.sqrt(8)).tolist(),population_score_by_data_profile_arm=Pm.tolist(),data_score_by_data_profile_arm=Dm.tolist(),population_baseline_score_by_data_profile_model=Pbase.mean(axis=1).tolist(),data_baseline_score_by_data_profile_model=Dbase.mean(axis=1).tolist(),raw_J_P_by_data_arm=gap.tolist(),parameter_descriptive=desc,parameter_identity_max_error=float(np.max(abs(half+cross-mse))),reference_qualification=ref['qualification_passed'],training_numerical_qualification=train['numerical_qualification'],evaluation_numerical_qualification=ev['numerical_qualification'],processing_accounting=dict(new_RNG=0,new_model_integrals=0,new_training=0),scope='16 iid outer units include new data/training randomness; raw P units additionally include independent per-data evaluation randomness. Same synthetic law only; development t approximation, qualified numerical references not exact integration certificates.')
    save(BASE/'RESULTS.json',result)
    print(json.dumps({k:result[k] for k in ['training_primary','evaluation_primary','gap_derived','mean_KL_by_arm']},ensure_ascii=False))
def check():
    assert not (BASE/'FREEZE.json').exists() and sha(helper.__file__)==HELPER_SHA;check_config(read(BASE/'CONFIG.json'))
    kl=np.arange(128,dtype=float).reshape(16,2,4)/1000;rv=risk_vectors(kl,.01)
    for d in range(16):
        expected=[sum(kl[d,r,1]-kl[d,r,0] for r in range(2))/2,sum(kl[d,r,2]-kl[d,r,1] for r in range(2))/2]+[sum(kl[d,r,a]-.01 for r in range(2))/2 for a in [1,2,3]]
        assert np.allclose(rv[d],expected,rtol=0,atol=1e-16)
    means=np.arange(16*8*2*10*16,dtype=float).reshape(16,8,2,10,16)/100000;p=np.arange(1,17)/136;dw=np.array([np.roll(p,d) for d in range(16)])
    P,D,PB,DB=eval_vectors(means,p,dw)
    for d in range(16):
        for g in range(8):
            for h in range(2):
                for a in range(4):
                    assert abs(P[d,g,h,a]-sum(means[d,g,h,2+4*r+a,y]*p[y] for r in range(2) for y in range(16))/2)<2e-15
                    assert abs(D[d,g,h,a]-sum(means[d,g,h,2+4*r+a,y]*dw[d,y] for r in range(2) for y in range(16))/2)<2e-15
    rho=np.ones((16,2,4,6));rho[:,1]=-1;half,cross,mse=parameter_vectors(rho,np.zeros(6));assert np.all(cross==-6) and np.all(half+cross==mse)
    for q in [2,5]:assert helper.summary('fixture',np.arange(16)*.01-.1,.025,q,.5)['df']==15
    save(BASE/'STATISTICS_SELFTEST.json',dict(passed=True,source_sha256=sha(__file__),helper_sha256=HELPER_SHA,checks=['nonexchangeable data/group/rep/arm/pattern mapping','two-repeat risk averaging before16outer t','q5/q2 df15','negative parameter cross-product retained'],new_RNG=0,new_model_integrals=0,new_training=0))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['check','run']);a=p.parse_args();check() if a.mode=='check' else run()
