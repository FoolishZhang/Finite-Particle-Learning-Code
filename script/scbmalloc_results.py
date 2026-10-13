"""Prespecified matched-budget contrasts and independent mean-error units."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json
import numpy as np
import scbmavg_results as helper
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/scbm_budget_allocation'
HELPER_SHA='d8dca7eb961fd6b69e5131df283adc265abcc026b86bbd9cf1a44265eeab6a3a'
ARMS=['IN4','OUT4']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def verify():
    assert sha(helper.__file__)==HELPER_SHA
    f=read(BASE/'FREEZE.json');d=read(BASE/'DERIVED_FREEZE.json');assert f['run_authorized'] and d['run_authorized'] and d['root_freeze_sha256']==sha(BASE/'FREEZE.json')
    for x in [f,d]:
        for p,h in x['inputs'].items():assert sha(ROOT/p)==h,p
def parameter_units(rho,truth):
    e=np.asarray(rho)-np.asarray(truth)
    half=.5*((e[:,::2]-e[:,1::2])**2).sum(axis=3)
    cross=(e[:,::2]*e[:,1::2]).sum(axis=3)
    mse=(e*e).sum(axis=3).mean(axis=1)
    assert np.max(abs((half+cross).mean(axis=1)-mse))<1e-13
    return half,cross,mse
def check_config(cfg):
    tf=cfg['training_family'];ef=cfg['evaluation']
    assert cfg['strategies']==ARMS and cfg['data_ids']==[0,1,2,3] and cfg['repeats']==16
    assert (tf['alpha'],tf['size'],tf['risk_df'],tf['dispersion_df'],tf['mean_error_df'])==(.025,3,15,7,7)
    assert (tf['risk_SE_threshold'],tf['dispersion_SE_threshold'],tf['mean_error_SE_threshold'])==(.0005,.05,.05)
    assert (ef['alpha'],ef['family_size'],ef['groups'],ef['df'],ef['SE_threshold'])==(.025,2,32,31,.0002)
def run():
    verify();cfg=read(BASE/'CONFIG.json');data=read(BASE/'DATA.json');train=read(BASE/'TRAIN/RESULTS.json');ref=read(BASE/'REFERENCE/RESULTS.json');ev=read(BASE/'EVAL/RESULTS.json');cand=read(BASE/'CANDIDATES.json')
    check_config(cfg)
    for x,name in [(train,'run'),(ref,'reference'),(ev,'eval')]:assert x['freeze_sha256']==sha(BASE/'FREEZE.json') and x['source_sha256']==sha(ROOT/f'script/scbmalloc_{name}.py')
    assert train['kernel_sha256']==sha(ROOT/'script/scbmalloc_kernel.py')
    assert ref['derived_freeze_sha256']==ev['derived_freeze_sha256']==sha(BASE/'DERIVED_FREEZE.json')
    assert ref['candidates_sha256']==ev['candidates_sha256']==sha(BASE/'CANDIDATES.json')
    assert ref['training_results_sha256']==sha(BASE/'TRAIN/RESULTS.json')
    assert [(x['data_id'],x['repeat']) for x in train['records']]==[(d,r) for d in range(4) for r in range(16)]
    assert [x['index'] for x in ref['records']]==list(range(138)) and [x['group'] for x in ev['groups']]==list(range(32))
    rho=np.array([[x['outputs'][a]['TAIL']['rho'] for a in ARMS] for x in train['records']]).reshape(4,16,2,6)
    members=np.array([[x['members'][f'M{i}']['TAIL']['rho'] for i in range(4)] for x in train['records']]).reshape(4,16,4,6)
    assert np.max(abs(members.mean(axis=2)-rho[:,:,1]))<2e-15
    kl=np.array([x['reference_joint_KL'] for x in ref['records'][2:130]]).reshape(4,16,2)
    risk,dispersion,half0=helper.train_vectors(rho,kl);half,cross,mse=parameter_units(rho,data['truth_rho']);assert np.max(abs(half-half0))<1e-13
    mean_error=(cross[:,:,1]-cross[:,:,0]).mean(axis=0);tf=cfg['training_family']
    training=[helper.summary(tf['risk_name'],risk,.025,3,.0005),helper.summary(tf['dispersion_name'],dispersion,.025,3,.05,'lower'),helper.summary(tf['mean_error_name'],mean_error,.025,3,.05,'lower')]
    assert abs((dispersion+mean_error).mean()-(mse[:,1]-mse[:,0]).mean())<1e-13
    parameter=[];member_desc=[];truth=np.array(data['truth_rho'])
    for d in range(4):
        for a,arm in enumerate(ARMS):
            points=rho[d,:,a];m=points.mean(axis=0);S=float(((points-m)**2).sum()/15);B=float(((m-truth)**2).sum()-S/16)
            parameter.append(dict(data_id=d,arm=arm,mean_rho=m.tolist(),sample_trace_covariance_unbiased=S,debiased_conditional_mean_parameter_error=B,sample_parameter_MSE=float(mse[d,a]),main_disjoint_dispersion_mean=float(half[d,:,a].mean()),main_cross_product_mean_error=float(cross[d,:,a].mean()),disjoint_half_distances=half[d,:,a].tolist(),disjoint_cross_products=cross[d,:,a].tolist(),mean_KL=float(kl[d,:,a].mean()),mean_matrix_KL=ref['records'][130+2*d+a]['reference_joint_KL'],scope='Full-R parameter identity is descriptive; primary parameter inference uses independent disjoint column pairs. Negative estimates retained.'))
        m=members[d].reshape(64,6);grand=m.mean(axis=0);S=float(((m-grand)**2).sum()/63);outS=float(((rho[d,:,1]-grand)**2).sum()/15);within=float(((members[d]-rho[d,:,1,None,:])**2).sum())
        assert np.max(abs(grand-rho[d,:,1].mean(axis=0)))<2e-15
        assert abs(63*S-within-4*15*outS)<2e-12
        member_desc.append(dict(data_id=d,grand_member_mean=grand.tolist(),member_sample_trace_covariance=S,OUT4_sample_trace_covariance=outS,empirical_ratio=outS/S,within_quartet_sum_squares=within,variance_partition_error=63*S-within-60*outS,mean_identity_error=float(np.max(abs(grand-rho[d,:,1].mean(axis=0)))),scope='64 independent member paths conditional on D; empirical variance ratio is descriptive, not forced to 1/4. Member PMFs and risks not computed.'))
    means=np.array([x['means'] for x in ev['groups']]);D,P=helper.eval_vectors(means,data)
    evaluation=[helper.summary('raw_D_OUT4_minus_IN4',D[:,0],.025,2,.0002),helper.summary('raw_P_OUT4_minus_IN4',P[:,0],.025,2,.0002)]
    score=[]
    for d in range(4):
        for a,arm in enumerate(ARMS):
            ids=[2+(d*16+r)*2+a for r in range(16)]
            for h,profile in enumerate(['raw','FACT']):
                for kind,w in [('D',data['data'][d]['weights']),('P',data['population_weights'])]:
                    x=(means[:,h,ids,:]*w).sum(axis=-1).mean(axis=1);m=(means[:,h,130+2*d+a,:]*w).sum(axis=-1)
                    score.append(dict(data_id=d,arm=arm,profile=profile,weights=kind,mean_score=float(x.mean()),MC_SE=float(x.std(ddof=1)/np.sqrt(32)),finite_panel_Jensen_mean=float((x-m).mean()),scope='Conditional descriptive score; no extra intervals or training-population score inference.'))
    files=[BASE/n for n in ['TRAIN/RESULTS.json','REFERENCE/RESULTS.json','EVAL/RESULTS.json','CANDIDATES.json','CONFIG.json','DATA.json']]
    result=dict(schema='scbmalloc-results-v1',source_sha256=sha(__file__),helper_sha256=HELPER_SHA,freeze_sha256=sha(BASE/'FREEZE.json'),derived_freeze_sha256=sha(BASE/'DERIVED_FREEZE.json'),created_utc=datetime.now(timezone.utc).isoformat(),inputs={p.relative_to(ROOT).as_posix():sha(p) for p in files},training_primary=training,evaluation_primary=evaluation,parameter_descriptive=parameter,member_descriptive=member_desc,score_descriptive=score,rho_by_data_repeat_arm=rho.tolist(),member_rho_by_data_repeat_member=members.tolist(),KL_by_data_repeat_arm=kl.tolist(),mean_KL_by_arm=kl.mean(axis=(0,1)).tolist(),mean_disjoint_conditional_dispersion_by_arm=half.mean(axis=(0,1)).tolist(),mean_cross_product_error_by_arm=cross.mean(axis=(0,1)).tolist(),mean_parameter_MSE_by_arm=mse.mean(axis=0).tolist(),parameter_primary_identity_error=float((dispersion+mean_error).mean()-(mse[:,1]-mse[:,0]).mean()),evaluation_FACT_descriptive=dict(D_mean=float(D[:,1].mean()),D_SE=float(D[:,1].std(ddof=1)/np.sqrt(32)),P_mean=float(P[:,1].mean()),P_SE=float(P[:,1].std(ddof=1)/np.sqrt(32))),reference_qualification=ref['qualification_passed'],training_numerical_qualification=train['numerical_qualification'],evaluation_numerical_qualification=ev['numerical_qualification'],processing_accounting=dict(new_RNG=0,new_model_integrals=0,new_training=0),scope='Training q3 alpha .025 on fixed data, evaluation q2 alpha .025 conditional on realized outputs; fresh quartet units. Matched prior/score calls, not equal optimizer calls or guaranteed equal wall time.')
    save(BASE/'RESULTS.json',result)
    print(json.dumps({k:result[k] for k in ['training_primary','evaluation_primary','mean_KL_by_arm','mean_disjoint_conditional_dispersion_by_arm','mean_cross_product_error_by_arm']},ensure_ascii=False))
def selftest():
    assert not (BASE/'FREEZE.json').exists() and sha(helper.__file__)==HELPER_SHA
    check_config(read(BASE/'CONFIG.json'))
    rho=np.empty((4,16,2,6));truth=np.arange(6)/20
    for d in range(4):
        for r in range(16):rho[d,r,0]=(r+d+1)*np.arange(1,7)/100;rho[d,r,1]=rho[d,r,0]/2
    h,c,m=parameter_units(rho,truth)
    for d in range(4):
        for j in range(8):
            for a in range(2):
                x=rho[d,2*j,a]-truth;y=rho[d,2*j+1,a]-truth
                assert abs(c[d,j,a]-sum(float(x[k])*float(y[k]) for k in range(6)))<1e-14
                assert abs(h[d,j,a]+c[d,j,a]-.5*(x@x+y@y))<1e-14
    row=helper.summary('test',np.arange(8)-5,.025,3,.05,'lower');assert row['family_size']==3 and row['df']==7
    # A truly negative cross-product must survive unchanged.
    test=np.empty((4,16,2,6));test[:,::2]=1.;test[:,1::2]=-1.
    hh,cc,mm=parameter_units(test,np.zeros(6));assert np.array_equal(cc,np.full((4,8,2),-6.))
    save(BASE/'STATISTICS_SELFTEST.json',dict(passed=True,source_sha256=sha(__file__),helper_sha256=HELPER_SHA,new_RNG=0,new_model_integrals=0,checks=['independent cross-product component','pair MSE identity','q3 df7 critical value','negative cross-products retained']))
    print(json.dumps(dict(passed=True,new_RNG=0)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['check','run']);a=p.parse_args();selftest() if a.mode=='check' else run()
