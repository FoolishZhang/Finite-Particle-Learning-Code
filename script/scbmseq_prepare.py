"""Prepare a fixed design, then generate new data only after the root freeze."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,subprocess
import numpy as np
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/scbm_sequential_generalization'
DOC=ROOT/'docs/08_research_研究工作/新数据与连续更新对照_20261004'
NAV=['README.md','REVIEW_BRIEF.md','AGENTS.md','ARGUMENT_FRAMEWORK.md','RESEARCH_ROADMAP.md','.gitattributes','.gitignore','docs/00_project_项目管理/START_HERE_当前工作入口.md','docs/00_project_项目管理/CURRENT_WORK_STATE.json']
ARMS=['IN4','OUT4','SEQ4','FACT']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8',newline='\n') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def configure():
    assert not (BASE/'FREEZE.json').exists()
    old=read(ROOT/'artifacts/scbm_budget_allocation/CONFIG.json');olddata=read(ROOT/'artifacts/scbm_budget_allocation/DATA.json')
    cfg={k:old[k] for k in ['outer_banks','K','nuisance','observations','batch_size','steps','tail_postupdates','initial_rho','scale','lr_rho','betas','eps','weight_decay','eigen_floor','repair_target','shuffle','gradient_clip','learning_rate_schedule','early_stop','workers','torch_threads','roles','draw_shape_order','numerical_preupdate_steps','numerical_loss_tolerance','numerical_gradient_tolerance']}
    cfg.update(schema='scbmseq-config-v1',data_ids=list(range(16)),repeats=2,strategies=ARMS,arms=ARMS,programs=['IN4','M0','M1','M2','M3','SEQ4','FACT'],members=['M0','M1','M2','M3'],seed_base=202610040100,seed_format='[seed_base,data_id,repeat,bank_index,role]',seq_bank_order=[0,1,2,3],seq_lr_rho=.02,seq_tail='last64_macrostep_end_states',data_generation=dict(seed_base=202610040000,roles={'active_normal':0,'active_uniform':1,'nuisance_uniform':2},seed_format='[seed_base,data_id,role]',normal_shape=[4096,4],active_uniform_shape=[4096,4],nuisance_uniform_shape=[4096,256],dtype='float64',labels_dtype='int8',law='active u<sigmoid(z@chol(C_truth).T+mu), independent nuisance u<.5'),
      training_family=dict(alpha=.025,size=5,df=15,primary_order=['OUT4_minus_IN4','SEQ4_minus_OUT4','OUT4_minus_I','SEQ4_minus_I','FACT_minus_I'],SE_thresholds=[.0005,.0005,.0002,.0002,.0002],units='16 independent new datasets with independent training priors; each averages2 quartets',two_sided=True,direction_open=True),
      evaluation=dict(data_ids=list(range(16)),groups=8,banks_per_group=512,K=100,nuisance=256,chunk=64,workers=4,seed_base=202610040200,roles={'active':0,'noise':1},seed_format='[seed_base,data_id,group,role]',profiles=['raw','FACT'],candidates=130,local_candidates=10,saved_prefix_banks=2,alpha=.025,family_size=2,df=15,SE_threshold=.0005,primary_order=['raw_P_OUT4_minus_IN4','raw_P_SEQ4_minus_OUT4'],two_sided=True,direction_open=True,canonical_nuisance_label=0,native_data_ids=[0,15],native_groups=[0]),
      reference=dict(new_candidates=128,old_candidates=2,orders=[24,32,40],main_order=32,maximum_PMF_change=5e-8,normalization_tolerance=2e-12,workers=4,include_LAST=False,include_dataset_arm_means=False),
      audit=dict(training_API_data_ids=[0,5,10,15],training_API_repeats=[0],training_API_steps=[0,31,127,511],evaluation_prefix_banks=2,reference_data_ids=[0,4,8,12],reference_tail_repeats=[0],reference_arms=ARMS,reference_order=36,reference_root='symmetric',reference_max_difference=5e-8),
      new_observations=65536,new_training_programs=224,new_training_prior_vectors=419430400,new_training_prior_Gaussians=109051904000,new_training_API_calls=212992,new_training_optimizer_calls=163840,new_eval_prior_vectors=6553600,new_eval_prior_Gaussians=1703936000)
    spec={k:olddata[k] for k in ['patterns','population_weights','mu','truth_rho','pair_order']}
    C=np.eye(4)
    for v,(i,j) in zip(spec['truth_rho'],spec['pair_order']):C[i,j]=C[j,i]=v
    spec['truth_covariance']=C.tolist();spec['identity_risk']=read(ROOT/'artifacts/scbm_multicorrelation/REFERENCE/RESULTS.json')['identity_reference']['reference_joint_KL']
    names=['artifacts/scbm_budget_allocation/DATA.json','artifacts/scbm_invariance/REFERENCE/MODEL.json','artifacts/scbm_multicorrelation/REFERENCE/RESULTS.json','artifacts/scbm_budget_allocation/MANIFEST.json','artifacts/scbm_budget_allocation/VERSION_LOCK.json','artifacts/paper_v7_20261003/MANIFEST.json','artifacts/paper_v7_20261003/VERSION_LOCK.json']
    spec['inputs']={n:sha(ROOT/n) for n in names};spec['scope']='Same prespecified synthetic law, entirely new data realizations; not new domains or natural datasets.'
    save(BASE/'CONFIG.json',cfg);save(BASE/'SPEC.json',spec)
    baseline='4e9b739dff5cc8938ed72a58b5a21a52f4645334';assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==baseline
    save(BASE/'BASELINE.json',dict(base_commit=baseline,handoff_sha256=sha(ROOT/'HANDOFF.md'),handoff_target=baseline,handoff_new_feedback=True,user_instruction='Consider feedback and autonomously continue the next research stage.',external_sandbox_obtained=False,old_science_and_v7_unchanged=True,inputs=spec['inputs']))
    (BASE/'HANDOFF_INPUT.md').write_bytes((ROOT/'HANDOFF.md').read_bytes());nav={}
    for n in NAV:
        raw=subprocess.check_output(['git','show',f'{baseline}:{n}'],cwd=ROOT);p=BASE/'navigation_before'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);nav[n]=sha(p)
    save(BASE/'NAVIGATION_BASELINE.json',dict(base_commit=baseline,files=nav));(BASE/'PREVIOUS_ARGUMENT_FRAMEWORK.md').write_bytes((BASE/'navigation_before/ARGUMENT_FRAMEWORK.md').read_bytes())
    print(json.dumps(dict(configured=True,new_RNG=0,new_observations=0)))
def labels(z,u,un,spec):
    return (u<expit(z@np.linalg.cholesky(spec['truth_covariance']).T+spec['mu'])).astype(np.int8),(un<.5).astype(np.int8)
def check():
    assert not (BASE/'FREEZE.json').exists();spec=read(BASE/'SPEC.json');z=np.sin(np.arange(28)).reshape(7,4);u=(np.arange(28)%11).reshape(7,4)/11;un=(np.arange(35)%13).reshape(7,5)/13
    y,yn=labels(z,u,un,spec);assert y.shape==(7,4) and yn.shape==(7,5) and y.dtype==np.int8
    L=np.linalg.cholesky(spec['truth_covariance'])
    direct=np.array([[int(u[b,i]<1/(1+np.exp(-sum(z[b,k]*L[i,k] for k in range(4))-spec['mu'][i]))) for i in range(4)] for b in range(7)],dtype=np.int8)
    assert np.array_equal(y,direct) and np.array_equal(yn,un<.5)
    save(BASE/'DATA_SELFTEST.json',dict(passed=True,source_sha256=sha(__file__),spec_sha256=sha(BASE/'SPEC.json'),config_sha256=sha(BASE/'CONFIG.json'),deterministic_label_formula_checked=True,new_RNG=0,new_model_integrals=0,new_training=0))
def data():
    freeze=read(BASE/'FREEZE.json');assert freeze['run_authorized']
    for p,h in freeze['inputs'].items():assert sha(ROOT/p)==h,p
    assert freeze['inputs']['script/scbmseq_prepare.py']==sha(__file__)
    cfg=read(BASE/'CONFIG.json');spec=read(BASE/'SPEC.json');dc=cfg['data_generation'];rows=[]
    assert not (BASE/'DATA.json').exists() and not list((BASE/'DATA').glob('*.npz'))
    save(BASE/'DATA_EXECUTION.json',dict(source_sha256=sha(__file__),root_freeze_sha256=sha(BASE/'FREEZE.json'),started_utc=datetime.now(timezone.utc).isoformat(),automatic_retry=False))
    for d in cfg['data_ids']:
        rng={k:np.random.Generator(np.random.PCG64(np.random.SeedSequence([dc['seed_base'],d,r]))) for k,r in dc['roles'].items()};initial={k:v.bit_generator.state for k,v in rng.items()}
        z=rng['active_normal'].standard_normal((4096,4));u=rng['active_uniform'].random((4096,4));un=rng['nuisance_uniform'].random((4096,256));y,yn=labels(z,u,un,spec)
        arr=dict(active_normal=z,active_uniform=u,nuisance_uniform=un,active_labels=y,nuisance_labels=yn);p=BASE/'DATA'/f'input_d{d:02}.npz';p.parent.mkdir(exist_ok=True)
        with p.open('xb') as f:np.savez_compressed(f,**arr)
        counts=np.bincount(y.astype(np.int64)@np.array([8,4,2,1]),minlength=16).tolist()
        row=dict(data_id=d,path=p.relative_to(ROOT).as_posix(),sha256=sha(p),counts=counts,weights=[v/4096 for v in counts],array_sha256={k:ah(v) for k,v in arr.items()},active_labels_sha256=ah(y),nuisance_labels_sha256=ah(yn),role_seed_sequences={k:[dc['seed_base'],d,r] for k,r in dc['roles'].items()},RNG_initial_states=initial,RNG_final_states={k:v.bit_generator.state for k,v in rng.items()})
        save(BASE/'DATA'/f'data_d{d:02}.json',row);rows.append(row)
    result={k:v for k,v in spec.items() if k!='inputs'};result.update(data=rows,inputs={**spec['inputs'],'artifacts/scbm_sequential_generalization/SPEC.json':sha(BASE/'SPEC.json'),'artifacts/scbm_sequential_generalization/CONFIG.json':sha(BASE/'CONFIG.json'),**{r['path']:r['sha256'] for r in rows}},source_sha256=sha(__file__),root_freeze_sha256=sha(BASE/'FREEZE.json'),accounting=dict(new_observations=65536,data_Gaussian=262144,data_Uniform=17039360),scope='16 new iid synthetic datasets; two training-prior quartets per dataset. Same fixed law, not cross-domain evidence.')
    save(BASE/'DATA.json',result)
    save(BASE/'DATA_FREEZE.json',dict(run_authorized=True,root_freeze_sha256=sha(BASE/'FREEZE.json'),inputs={'artifacts/scbm_sequential_generalization/DATA.json':sha(BASE/'DATA.json'),**{r['path']:r['sha256'] for r in rows}},scope='Mechanically generated data; no data-dependent design selection.'))
    print(json.dumps(dict(complete=True,new_observations=65536,data_sha256=sha(BASE/'DATA.json'))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['configure','check','data']);a=p.parse_args();{'configure':configure,'check':check,'data':data}[a.mode]()
