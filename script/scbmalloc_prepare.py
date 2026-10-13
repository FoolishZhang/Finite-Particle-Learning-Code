"""Prepare matched input/API budget allocation on frozen old datasets."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/scbm_budget_allocation'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main():
    prior=ROOT/'artifacts/scbm_bank_averaging/DATA.json';data=read(prior)
    for x in data['data']:
        p=ROOT/x['path'];assert sha(p)==x['sha256']
        with np.load(p,allow_pickle=False) as z:y=z['active_labels'];yn=z['nuisance_labels']
        assert y.shape==(4096,4) and yn.shape==(4096,256)
        assert np.bincount(y.astype(np.int64)@np.array([8,4,2,1]),minlength=16).tolist()==x['counts']
    data['inputs'][prior.relative_to(ROOT).as_posix()]=sha(prior)
    for name in ['MANIFEST.json','VERSION_LOCK.json']:
        p=prior.parent/name;data['inputs'][p.relative_to(ROOT).as_posix()]=sha(p)
    data['scope']='Four fixed old datasets; fresh prior quartets. Exact old labels including nuisance used in training. No new observations.'
    cfg=dict(schema='scbmalloc-config-v1',data_ids=[0,1,2,3],repeats=16,strategies=['IN4','OUT4'],arms=['IN4','OUT4'],programs=['IN4','M0','M1','M2','M3'],members=['M0','M1','M2','M3'],outer_banks=4,K=100,nuisance=256,observations=4096,batch_size=64,steps=512,tail_postupdates=64,initial_rho=[0.]*6,scale=1.,lr_rho=.02,betas=[.9,.999],eps=1e-8,weight_decay=0.,eigen_floor=.1,repair_target=.1001,shuffle=False,gradient_clip=False,learning_rate_schedule=False,early_stop=False,workers=8,torch_threads=1,seed_base=202610035000,roles=dict(active=0,noise=1),seed_format='[seed_base,data_id,repeat,bank_index,role]',draw_shape_order='float64[K,batch,coordinate], C-order, cast once per bank',numerical_preupdate_steps=[0,31,127,511],numerical_loss_tolerance=1e-4,numerical_gradient_tolerance=2e-5,
        training_family=dict(alpha=.025,size=3,risk_SE_threshold=.0005,dispersion_SE_threshold=.05,mean_error_SE_threshold=.05,risk_df=15,dispersion_df=7,mean_error_df=7,risk_name='TAIL_KL_OUT4_minus_IN4',dispersion_name='TAIL_conditional_dispersion_OUT4_minus_IN4',mean_error_name='TAIL_conditional_mean_error_OUT4_minus_IN4',risk_units='16 iid prior-quartet columns, each averages four fixed datasets',parameter_units='8 iid disjoint column pairs, average over four fixed datasets',two_sided=True,risk_direction_open=True,dispersion_expected_sign='negative',mean_error_expected_sign='negative'),
        evaluation=dict(groups=32,banks_per_group=1024,K=100,nuisance=256,chunk=64,workers=2,seed_base=202610035100,roles=dict(active=0,noise=1),seed_format='[seed_base,group,role]',profiles=['raw','FACT'],candidates=138,saved_prefix_banks=2,alpha=.025,family_size=2,df=31,SE_threshold=.0002,primary_order=['raw_D_OUT4_minus_IN4','raw_P_OUT4_minus_IN4'],two_sided=True,direction_open=True,canonical_nuisance_label=0),
        reference=dict(new_candidates=136,old_candidates=2,orders=[24,32,40],main_order=32,maximum_PMF_change=5e-8,normalization_tolerance=2e-12,workers=4,include_LAST=False,include_dataset_arm_means=True),
        audit=dict(training_API_data_ids=[0,1,2,3],training_API_repeats=[0],training_API_steps=[0,31,127,511],evaluation_prefix_banks=2,reference_tail_repeats=[0],reference_mean_all=True,reference_order=36,reference_root='symmetric',reference_max_difference=5e-8),
        new_observations=0,new_training_programs=320,new_training_prior_vectors=838860800,new_training_prior_Gaussians=218103808000,new_training_API_calls=262144,new_training_optimizer_calls=163840,new_eval_prior_vectors=3276800,new_eval_prior_Gaussians=851968000)
    save(OUT/'CONFIG.json',cfg);save(OUT/'DATA.json',data)
    save(OUT/'BASELINE.json',dict(base_commit='1094621f1074dbd5a3608e2068e8147a9566fa9e',handoff_sha256=sha(ROOT/'HANDOFF.md'),handoff_new_feedback=False,handoff_target='cf4f8f51f14083634bc55375d0fc372fad581d20',old_paper_and_science_unchanged=True,source_sha256=sha(__file__),inputs=data['inputs'],user_authorization='Continue the existing research direction; normal commit/push after checks.',external_sandbox_obtained=False))
    (OUT/'HANDOFF_INPUT.md').write_bytes((ROOT/'HANDOFF.md').read_bytes())
    print(json.dumps(dict(data_sets=4,new_observations=0,new_RNG=0,new_integrals=0,programs_planned=320)))
if __name__=='__main__':main()
