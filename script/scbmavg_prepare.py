"""Freeze-friendly configuration and old-data inventory; draws no new data."""
from pathlib import Path
import hashlib,json,itertools
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/scbm_bank_averaging'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main():
    old=ROOT/'artifacts/scbm_multicorrelation/TRAIN/RESULTS.json';train=read(old);truth=ROOT/'artifacts/scbm_invariance/REFERENCE/MODEL.json'
    p0=next(x['main_pmf'] for x in read(truth)['records'] if x['t']==0)
    inputs={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [old,truth,ROOT/'artifacts/scbm_multicorrelation/REFERENCE/RESULTS.json',ROOT/'artifacts/paper_v7_20261003/MANIFEST.json',ROOT/'artifacts/paper_v7_20261003/VERSION_LOCK.json',ROOT/'artifacts/scbm_six_frozen_targets/MANIFEST.json',ROOT/'artifacts/scbm_six_frozen_targets/VERSION_LOCK.json']}
    records=[]
    for d in range(4):
        row=train['records'][d];p=ROOT/row['input_file'];assert sha(p)==row['input_sha256'];inputs[row['input_file']]=sha(p)
        with np.load(p,allow_pickle=False) as f:y=f['active_labels'];yn=f['nuisance_labels']
        assert y.shape==(4096,4) and yn.shape==(4096,256);assert np.isin(y,[0,1]).all() and np.isin(yn,[0,1]).all()
        counts=np.bincount(y.astype(np.int64)@np.array([8,4,2,1]),minlength=16)
        records.append(dict(data_id=d,path=row['input_file'],sha256=sha(p),active_labels_sha256=hashlib.sha256(y.tobytes()).hexdigest(),nuisance_labels_sha256=hashlib.sha256(yn.tobytes()).hexdigest(),counts=counts.tolist(),weights=(counts/4096).tolist()))
    config=dict(schema='scbmavg-config-v1',data_ids=[0,1,2,3],repeats=16,arms=['L1','L4'],outer_banks=[1,4],K=100,nuisance=256,observations=4096,batch_size=64,steps=512,tail_postupdates=64,initial_rho=[0.]*6,scale=1.,lr_rho=.02,betas=[.9,.999],eps=1e-8,weight_decay=0.,eigen_floor=.1,repair_target=.1001,shuffle=False,gradient_clip=False,learning_rate_schedule=False,early_stop=False,workers=8,torch_threads=1,seed_base=202610034000,roles=dict(active=0,noise=1),seed_format='[seed_base,data_id,repeat,bank_index,role]',draw_shape_order='float64[K,batch,coordinate], C-order; cast to float32 once per bank',numerical_preupdate_steps=[0,31,127,511],numerical_loss_tolerance=1e-4,numerical_gradient_tolerance=2e-5,
        training_family=dict(alpha=.025,size=2,risk_SE_threshold=.0005,dispersion_SE_threshold=.05,risk_df=15,dispersion_df=7,risk_name='TAIL_KL_L4_minus_L1',dispersion_name='TAIL_conditional_dispersion_L4_minus_L1',risk_units='16 iid columns, each is average over 4 fixed datasets',dispersion_units='8 iid disjoint column-pairs, each averages half squared parameter distance over 4 fixed datasets',two_sided=True,risk_direction_open=True,dispersion_expected_sign='negative'),
        evaluation=dict(groups=32,banks_per_group=1024,K=100,nuisance=256,chunk=64,workers=2,seed_base=202610034100,roles=dict(active=0,noise=1),seed_format='[seed_base,group,role]',profiles=['raw','FACT'],candidates=138,saved_prefix_banks=2,alpha=.025,family_size=2,df=31,SE_threshold=.0002,primary_order=['raw_D_L4_minus_L1','raw_P_L4_minus_L1'],two_sided=True,direction_open=True,canonical_nuisance_label=0),
        reference=dict(new_candidates=136,old_candidates=2,orders=[24,32,40],main_order=32,maximum_PMF_change=5e-8,normalization_tolerance=2e-12,workers=4,include_LAST=False,include_dataset_arm_means=True),
        audit=dict(training_API_data_ids=[0,1,2,3],training_API_repeats=[0],training_API_steps=[0,31,127,511],evaluation_prefix_banks=2,reference_tail_repeats=[0],reference_mean_all=True,reference_order=36,reference_root='symmetric',reference_max_difference=5e-8),
        new_observations=0,new_training_programs=128,new_training_prior_vectors=838860800,new_training_prior_Gaussians=218103808000,new_training_API_calls=163840,new_eval_prior_vectors=3276800,new_eval_prior_Gaussians=851968000)
    save(OUT/'CONFIG.json',config)
    save(OUT/'DATA.json',dict(data=records,inputs=inputs,patterns=list(map(list,itertools.product([0,1],repeat=4))),population_weights=p0,mu=[-.6,-.2,.2,.6],truth_rho=[.25,-.15,.1,.1,-.15,.25],pair_order=list(map(list,itertools.combinations(range(4),2))),scope='First four old data IDs fixed by index; no new observation. Actual old nuisance labels used in training; canonical zero only in fresh-prior evaluation expectation.'))
    save(OUT/'BASELINE.json',dict(base_commit='9d48bbda8aa0196341b89b1cc2306957d0e5299f',handoff_sha256=sha(ROOT/'HANDOFF.md'),handoff_new_feedback=False,handoff_target='cf4f8f51f14083634bc55375d0fc372fad581d20',paper_v7_unchanged=True,old_science_unchanged=True,source_sha256=sha(__file__),inputs=inputs,user_authorization='Continue the previous research direction; normal commit and push after checks remain authorized.',external_sandbox_obtained=False))
    (OUT/'HANDOFF_INPUT.md').write_bytes((ROOT/'HANDOFF.md').read_bytes())
    print(json.dumps(dict(data_sets=4,old_observations=16384,new_observations=0,new_RNG=0,new_integrals=0,planned_programs=128)))
if __name__=='__main__':main()
