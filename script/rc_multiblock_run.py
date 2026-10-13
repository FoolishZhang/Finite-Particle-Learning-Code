"""M: frozen unknown-block learning with exact all-coordinate LOO evaluation."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,math,platform,sys,time,traceback
import numpy as np
import scipy
from scipy.stats import t
import rc_multiblock_core as core
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/rc_quant/M'
DOC=ROOT/'docs/08_research_研究工作/RC定量界与多块查询_20261001'
BASE='bb94a7c257c571aeff277bbe7730d0dfb6185fec'
HANDOFF='57973169b1de76bf8d5981591b4cb64bd578bbc140312cc835d47f312b639e0c'
CONFIG=dict(dimensions=[12,24,48,96],sizes=[2048,8192,32768],reps=16,seed_base=6126100100,
            known_epsilon_min=.7,known_a=.95,teacher_abs_amplitudes=[.75,.9],methods=['MOM','MLE','HIST','PRODUCT'],
            numerical_tolerance=1e-10,primary_query_floor=[87,56320],primary_t_family=4)
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return Path(p).relative_to(ROOT).as_posix()
def read(p):return json.loads(Path(p).read_text('utf-8'))
def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v) for v in x]
    return x
def write(p,x,replace=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w' if replace else 'x',encoding='utf-8',newline='\n') as f:json.dump(serial(x),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def freeze():
    assert not OUT.exists();OUT.mkdir(parents=True)
    assert sha(ROOT/'HANDOFF.md')==HANDOFF
    (OUT/'HANDOFF_INPUT.md').write_bytes((ROOT/'HANDOFF.md').read_bytes())
    write(OUT/'CONFIG.json',CONFIG)
    write(OUT/'ENVIRONMENT.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),threads=2))
    paths=[Path(__file__).resolve(),ROOT/'script/rc_multiblock_core.py',DOC/'PLAN_M.md',DOC/'M_NUMERICS.md',
           DOC/'MULTIBLOCK_THEORY.md',DOC/'QUERY_DUAL.md',OUT/'HANDOFF_INPUT.md',OUT/'CONFIG.json',OUT/'ENVIRONMENT.json']
    write(OUT/'FREEZE.json',dict(frozen_utc=now(),base_commit=BASE,files={rel(p):sha(p) for p in paths},before_scientific_RNG=True))
    write(OUT/'STATE.json',dict(status='frozen',active_jobs=[]))
    print('M frozen',flush=True)
def checkfreeze():
    for name,h in read(OUT/'FREEZE.json')['files'].items():assert sha(ROOT/name)==h,name
def make_teacher(di,rep,D):
    seed=CONFIG['seed_base']+1000000*di+1000*rep
    permutation=np.random.default_rng(seed).permutation(D)
    blocks=[list(core.canonical_block(permutation[i:i+6])) for i in range(0,D,6)]
    signs=np.random.default_rng(seed+1).choice([-1,1],size=D//6)
    eps=[float(signs[b])*[.75,.9][b%2] for b in range(D//6)]
    return blocks,eps,seed
def check_risk(r):
    a=r['normalization_audit'];assert a['finite'] and a['minimum_model_factor_probability']>0
    assert max(a['model_factor_max_mass_error'],a['teacher_marginal_max_mass_error'])<=1e-10
    assert r['joint_kl']>=-1e-10 and min(r['per_target_query_kl'])>=-1e-10
def collect():
    records=[read(p) for p in sorted((OUT/'repetitions').glob('*.json'))];assert len(records)==64
    units=[u for rec in records for u in rec['units']];assert len(units)==192
    summary=[]
    for D in CONFIG['dimensions']:
        for n in CONFIG['sizes']:
            cell=[u for u in units if u['D']==D and u['n']==n];assert len(cell)==16
            for method in CONFIG['methods']+['UNIFORM']:
                vals=[u['risk'][method] for u in cell]
                summary.append(dict(D=D,n1=n,n2=n,method=method,repetitions=16,
                    valid_discoveries=sum(u['discovery_valid'] for u in cell),exact_structure_recoveries=sum(u['exact_recovery'] for u in cell),
                    mean_query_kl=float(np.mean([r['query_kl'] for r in vals])),mean_joint_kl=float(np.mean([r['joint_kl'] for r in vals])),
                    mean_discovery_seconds=float(np.mean([u['discovery_seconds'] for u in cell])),
                    mean_four_model_fit_seconds=float(np.mean([u['four_model_fit_seconds'] for u in cell])),
                    mean_teacher_risk_eval_seconds=float(np.mean([u['evaluation_seconds'][method] for u in cell]))))
    primary=[];critical=float(t.ppf(1-.05/(2*4),15));floor=87/56320
    for D in CONFIG['dimensions']:
        cell=[u for u in units if u['D']==D and u['n']==32768];vals=[u['risk']['MOM']['query_kl'] for u in cell]
        mean=float(np.mean(vals));se=float(np.std(vals,ddof=1)/4);ci=[mean-critical*se,mean+critical*se]
        recovered=sum(u['exact_recovery'] for u in cell)
        primary.append(dict(D=D,n1=32768,n2=32768,method='MOM',values=vals,mean=mean,se=se,interval=ci,
            analytic_RC_floor=[87,56320],observed_exact_recoveries=recovered,
            passed=bool(recovered==16 and ci[1]<floor),family=4,t_is_development_approximation=True))
    valid=sum(u['discovery_valid'] for u in units)
    counts=dict(teacher_realizations=64,training_vector_draws=64*2*32768,draws_not_unique_bit_patterns=True,
        training_scalar_bits=16*2*32768*sum(CONFIG['dimensions']),test_or_validation_draws=0,
        discovery_runs=192,method_output_assignments=768,valid_discoveries=valid,global_uniform_fallback_outputs=3*(192-valid),
        product_joint_fits=192,learned_block_joint_fits=3*valid,total_joint_fit_endpoints=192+3*valid,
        estimated_six_bit_factors=3*sum(u['D']//6 for u in units if u['discovery_valid']),
        estimated_product_bernoulli_factors=sum(u['D'] for u in units),new_neural_networks=0,iterative_optimizer_calls=0,
        mathematical_template_learner_was_actually_run=True,parameters_and_prefixes_reused_not_independent=True)
    write(OUT/'SUMMARY.json',summary);write(OUT/'PRIMARY.json',primary);write(OUT/'COUNTS.json',counts)
    return counts,primary
def run():
    checkfreeze();assert read(OUT/'STATE.json')['status']=='frozen'
    write(OUT/'START.json',dict(first_scientific_RNG_utc=now(),source_sha256=sha(__file__),freeze_sha256=sha(OUT/'FREEZE.json')))
    (OUT/'local').mkdir();(OUT/'.gitignore').write_text('/local/\n',encoding='utf-8')
    for di,D in enumerate(CONFIG['dimensions']):
        for rep in range(CONFIG['reps']):
            write(OUT/'STATE.json',dict(status='running',D=D,rep=rep,pid=os.getpid(),time_utc=now()),True)
            start=time.perf_counter();blocks,eps,seed=make_teacher(di,rep,D);structure_time=time.perf_counter()-start
            start=time.perf_counter();x1=core.sample_teacher(np.random.default_rng(seed+2),32768,blocks,eps,D);x2=core.sample_teacher(np.random.default_rng(seed+3),32768,blocks,eps,D);sample_time=time.perf_counter()-start
            path=OUT/'local'/f'd{D}_r{rep:02}.npz';np.savez_compressed(path,X1=x1,X2=x2)
            rec=dict(D=D,rep=rep,blocks=blocks,epsilon=eps,seeds=dict(layout=seed,signs=seed+1,X1=seed+2,X2=seed+3),
                     raw_arrays=rel(path),raw_sha256=sha(path),teacher_structure_seconds=structure_time,
                     training_generation_seconds=sample_time,units=[])
            for n in CONFIG['sizes']:
                start=time.perf_counter();discovery=core.discover(x1[:n]);disc_s=time.perf_counter()-start
                start=time.perf_counter();models=core.fit_models(x1[:n],x2[:n],discovery);fit_s=time.perf_counter()-start
                trained=OUT/'trained'/f'd{D}_r{rep:02}_n{n}.json'
                write(trained,dict(D=D,rep=rep,n=n,discovery=discovery,models=models,discovery_seconds=disc_s,
                    four_model_fit_seconds=fit_s,before_teacher_risk_evaluation=True,training_only_inputs=True))
                # All methods have been fixed and saved before this teacher-only evaluation.
                risks={};eval_seconds={}
                for method,model in list(models.items())+[('UNIFORM',core.uniform_model(D,'UNIFORM',False))]:
                    start=time.perf_counter();risk=core.eval_risk(model,blocks,eps,D);elapsed=time.perf_counter()-start
                    check_risk(risk);risks[method]=risk;eval_seconds[method]=elapsed
                exact=bool(discovery['valid'] and sorted(map(tuple,discovery['recovered_blocks']))==sorted(map(tuple,blocks)))
                rec['units'].append(dict(D=D,rep=rep,n=n,trained_file=rel(trained),trained_sha256=sha(trained),
                    discovery_valid=discovery['valid'],exact_recovery=exact,selected_triples=discovery['selected_count'],
                    discovery_seconds=disc_s,four_model_fit_seconds=fit_s,risk=risks,evaluation_seconds=eval_seconds))
            write(OUT/'repetitions'/f'd{D}_r{rep:02}.json',rec)
            print('M',D,rep,'valid',[u['discovery_valid'] for u in rec['units']],flush=True)
    counts,primary=collect()
    files=[p for part in ['repetitions','trained'] for p in (OUT/part).glob('*.json')]
    files += [OUT/name for name in ['START.json','SUMMARY.json','PRIMARY.json','COUNTS.json']]
    write(OUT/'RESULTS_FREEZE.json',dict(completed_utc=now(),files={rel(p):sha(p) for p in files},
        local_arrays={rel(p):sha(p) for p in (OUT/'local').glob('*.npz')},source_freeze_sha256=sha(OUT/'FREEZE.json')))
    write(OUT/'STATE.json',dict(status='complete',time_utc=now(),active_jobs=[]),True)
    print(json.dumps(dict(complete=True,primary_passed=sum(r['passed'] for r in primary),**counts)),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','run']);args=p.parse_args()
    if args.phase=='freeze':freeze()
    else:
        try:run()
        except Exception as e:
            if (OUT/'FREEZE.json').exists():
                write(OUT/f'FAILURE_{time.time_ns()}.json',dict(error=repr(e),traceback=traceback.format_exc(),time_utc=now()))
                write(OUT/'STATE.json',dict(status='stopped_failure',time_utc=now(),active_jobs=[]),True)
            raise
