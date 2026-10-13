"""Frozen-panel MC statistics with prespecified discovery/validation splitting."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json
import numpy as np
from scipy.stats import t as student_t
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/scbm_frozen_evaluation'
DOC=ROOT/'docs/08_research_研究工作/冻结表示的条件与总体目标_20261003'
METRICS=['centered_score','price_t','path_t']
POLICY=dict(replicates=32,banks_per_replicate=512,old_models=16,zero_index=10,contrast_index=14,
    primary_alpha=.05,primary_family_size=4,SE_threshold=5e-5,
    selection_replicates=list(range(16)),validation_replicates=list(range(16,32)),
    selection_candidate_indices=[21]+list(range(21)),tie_rule='First candidate, actual LAST first then ascending grid',
    diagnostic_alpha=.05,diagnostic_family_size=7480,diagnostic_metrics=['relative_score','price_t'])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf8'))
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8',newline='\n') as f:
        json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def interval(values,family=4):
    v=np.asarray(values,dtype=np.float64);n=len(v);mean=float(v.mean());se=float(v.std(ddof=1)/np.sqrt(n))
    critical=float(student_t.ppf(1-.05/(2*family),n-1))
    return dict(replicates=n,df=n-1,mean=mean,SE=se,critical_t=critical,
                lower=mean-critical*se,upper=mean+critical*se)
def freeze():
    files=[Path(__file__),OUT/'MODEL_SELECTION.json',OUT/'EVAL/FREEZE.json',
           OUT/'EVAL/RESULTS.json',OUT/'REFERENCE/RESULTS.json',DOC/'EXPERIMENT_DESIGN.md']
    save(OUT/'STATISTICS/FREEZE.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__),policy=POLICY,inputs={rel(p):sha(p) for p in files},
        prespecified_source_bound_by_evaluation_freeze=True,postprocessing_authorized=True))
def run():
    f=read(OUT/'STATISTICS/FREEZE.json')
    assert f['source_sha256']==sha(__file__) and f['policy']==POLICY
    for p,h in f['inputs'].items():assert sha(ROOT/p)==h,p
    production=read(OUT/'EVAL/RESULTS.json');selection=read(OUT/'MODEL_SELECTION.json')
    records=sorted(production['records'],key=lambda r:r['replicate'])
    assert [r['replicate'] for r in records]==list(range(32))
    arrays={k:np.array([r['means'][k] for r in records],dtype=np.float64) for k in METRICS}
    for k,a in arrays.items():
        assert a.shape==(32,16,5,22,2) and np.isfinite(a).all()
        assert np.array_equal(a[:,:,2,:21,:],a[:,:,4,:21,:]),k
    a=arrays['centered_score']
    delta=a[:,:,:,14,0]-a[:,:,:,10,0]
    c1,half,fact=delta[:,:,0],delta[:,:,1],delta[:,:,2]
    discovery=a[:16,:,3,:,1].mean(axis=0)
    candidate_indices=np.array(POLICY['selection_candidate_indices'])
    chosen=candidate_indices[np.argmin(discovery[:,candidate_indices],axis=1)]
    chosen_t=[(selection['models'][m]['LAST']['t'] if j==21 else selection['grid_t'][j]) for m,j in enumerate(chosen)]
    holdout=a[16:,:,3,21,1]-a[16:,np.arange(16),3,chosen,1]
    rows=[]
    definitions=[
        ('P1_population_wrong_preference',c1.mean(axis=1),'negative','Fixed population-weight target at t14 versus truth, learned TAIL scales'),
        ('P2_nuisance_excess_tilt',(c1-fact).mean(axis=1),'negative','Difference of the same t contrast versus same-K factorization'),
        ('P3_half_scale_upward_shift',(half-c1).mean(axis=1),'positive','An upward movement, not automatically smaller absolute bias'),
        ('P4_LAST_conditional_selected_improvement',holdout.mean(axis=1),'positive','Validation only; conditional on the first16-replicate selection')]
    for name,values,direction,scope in definitions:
        row=dict(name=name,replicate_values=values.tolist(),expected=direction,scope=scope,**interval(values))
        row['effect_passed']=bool(row['upper']<0 if direction=='negative' else row['lower']>0)
        row['precision_passed']=row['SE']<=POLICY['SE_threshold']
        row['SE_threshold']=POLICY['SE_threshold'];rows.append(row)
    selected=dict(candidate_order=POLICY['selection_candidate_indices'],tie_rule=POLICY['tie_rule'],
        selection_replicates=POLICY['selection_replicates'],validation_replicates=POLICY['validation_replicates'],
        selected_point_indices=chosen.tolist(),selected_t=chosen_t,discovery_means=discovery.tolist(),
        validation_per_model_gains=holdout.tolist(),
        scope='A frozen LAST-scale one-dimensional candidate comparator; not a continuous optimum, retraining, or a risk decomposition.')
    bands={}
    for name,values in [('relative_score',a-a[:,:,:,10:11,:]),('price_t',arrays['price_t'])]:
        panel=np.concatenate((values,values.mean(axis=1,keepdims=True)),axis=1)
        mean=panel.mean(axis=0);se=panel.std(axis=0,ddof=1)/np.sqrt(32)
        critical=float(student_t.ppf(1-.05/(2*7480),31))
        bands[name]=dict(mean=mean.tolist(),SE=se.tolist(),lower=(mean-critical*se).tolist(),
                         upper=(mean+critical*se).tolist(),critical_t=critical)
    assert 2*np.prod(np.array(bands['relative_score']['mean']).shape)==7480
    # Exact algebraic contrasts, descriptive unless one of the four primary rows.
    relscore=a-a[:,:,:,10:11,:]
    extras=dict(
        tail_population_learned_contrast=c1.mean(axis=1).tolist(),
        tail_population_half_contrast=half.mean(axis=1).tolist(),
        tail_population_fact_contrast=fact.mean(axis=1).tolist(),
        tail_population_half_excess=(half-fact).mean(axis=1).tolist(),
        tail_D_minus_P_relative=(relscore[:,:,:3,:,1]-relscore[:,:,:3,:,0]).mean(axis=1).tolist(),
        LAST_validation_population_gains=(a[16:,:,3,21,0]-a[16:,np.arange(16),3,chosen,0]).tolist(),
        LAST_validation_factorized_D_gains=(a[16:,:,4,21,1]-a[16:,np.arange(16),4,chosen,1]).tolist(),
        price_minus_path_panel=(arrays['price_t']-arrays['path_t']).mean(axis=1).tolist())
    output=dict(source_sha256=sha(__file__),freeze_sha256=sha(OUT/'STATISTICS/FREEZE.json'),
        policy=POLICY,selection_sha256=sha(OUT/'MODEL_SELECTION.json'),
        evaluation_results_sha256=sha(OUT/'EVAL/RESULTS.json'),reference_results_sha256=sha(OUT/'REFERENCE/RESULTS.json'),
        primary_family=rows,primary_effects_passed=sum(r['effect_passed'] for r in rows),
        primary_precision_passed=sum(r['precision_passed'] for r in rows),selected=selected,
        diagnostic_bands=bands,descriptive=extras,
        conditional_on_frozen_old_panel=True,old_training_replicates_not_new_MC_units=True,
        statistical_families_separate=True,development_t_approximation_not_exact_coverage=True,
        no_continuous_optimum_certificate=True,new_random_draws=0,new_integrals=0,new_fits=0)
    save(OUT/'STATISTICS/RESULTS.json',output)
    print(json.dumps(dict(primary_family=rows,selected_points=chosen.tolist(),
        primary_effects_passed=output['primary_effects_passed'],primary_precision_passed=output['primary_precision_passed'])),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['schema','freeze','run']);a=p.parse_args()
    if a.mode=='schema':print(json.dumps(dict(policy=POLICY,source_sha256=sha(__file__))))
    else:globals()[a.mode]()
