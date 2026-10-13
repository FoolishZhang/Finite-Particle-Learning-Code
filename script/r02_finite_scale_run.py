"""Frozen R02 constant execution and finite-method budgets; never scores D_n."""
from pathlib import Path
from fractions import Fraction as F
from datetime import datetime, timezone
import argparse
import hashlib
import json
import platform
import sys
import traceback
from flint import arb, ctx
import r02_finite_scale_constants as c
import r02_finite_scale_audit as audit

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/r02_finite_scale_v1_20261010'
DOC='docs/08_research_研究工作/R02FiniteScale_v1_20261010/'
BASE='513f15a8c6fc3d2f74db4c770a2f377dd290c922'
FEEDBACK='86b11895ba9fef8311cafe03a302b760ce5ac073c1774b0760cd40388c2c0611'
PAPER='artifacts/submission_candidate_v1_9_20261010/DELIVERY_MANIFEST.json'
PAPER_SHA='321ef3b7f291b5807c06c06b0bdfaa84759ccd47e8e0c0bc965f8a75196bc180'
OLD='artifacts/asymptotic_bridge_v1_20261008/AUTHOR_READOUT.json'
PARAMS=dict(rho='9/20',K=100,step='1/16',N=160,strip='3/4',bits=192,hard_seconds=600)
SOURCES=[PAPER,OLD,'artifacts/asymptotic_bridge_v1_20261008/REFERENCE.json',
 'docs/07_papers_论文工作/ParticleJointSupervision/submission_candidate_v1_9_20261010/APPENDIX_THEORY_EN.md',
 'docs/08_research_研究工作/AsymptoticBridge_v1_20261008/MATH.md',
 'script/r02_finite_scale_run.py','script/r02_finite_scale_constants.py',
 'script/r02_finite_scale_audit.py','script/r02_finite_scale_fixture.py','script/r02_finite_scale_fraction.py',
 'artifacts/r02_finite_scale_v1_20261010/IMPLEMENTATION_CHECK.json',
 'artifacts/r02_finite_scale_v1_20261010/FIXTURE_CHECK.json']+[DOC+x for x in
 ('CONTRACT.md','METHOD_BUDGET.md','AVERAGED_KERNEL.md','SECOND_ORDER.md','CONSTANT_BOUNDS.md','AUDIT_METHOD.md')]

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def now():return datetime.now(timezone.utc).isoformat()
def write_new(path,data):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('x',encoding='utf8',newline='\n') as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def lo(r):return c.endpoint_fraction(r['lower_exact_dyadic'])
def hi(r):return c.endpoint_fraction(r['upper_exact_dyadic'])
def overlap(a,b):return lo(a)<=hi(b) and lo(b)<=hi(a)
def qfloat(x):
    require(isinstance(x,(int,float)) and not isinstance(x,bool),'Saved float required')
    return F.from_float(x) if isinstance(x,float) else F(x)

def freeze():
    require(sha(ROOT/'HANDOFF.md')==FEEDBACK,'Current feedback differs')
    require(sha(ROOT/PAPER)==PAPER_SHA,'Paper manifest identity')
    require(not (OUT/'FREEZE.json').exists(),'Freeze once')
    with (OUT/'HANDOFF_INPUT.md').open('xb') as f:f.write((ROOT/'HANDOFF.md').read_bytes())
    inputs=dict(schema='r02_finite_scale_inputs_v1',base_commit=BASE,feedback_sha256=FEEDBACK,
        parameters=PARAMS,independent_audit=dict(rho='9/20',K=100,hard_seconds=120),
        n_values=[64,256,1024,4096],primary_n_values=[256,1024,4096],
        constant_width_gate_exact='1e-8',audit_distance_gate_exact='1e-9',
        new_Dn_evaluation_authorized=False,new_banks_authorized=False,
        comparison_is_posthoc_descriptive=True,prior_readout_seen_before_design=True)
    write_new(OUT/'INPUTS.json',inputs)
    paths=SOURCES+[(OUT/x).relative_to(ROOT).as_posix() for x in ('INPUTS.json','HANDOFF_INPUT.md')]
    write_new(OUT/'FREEZE.json',dict(schema='r02_finite_scale_freeze_v1',frozen_utc=now(),
        base_commit=BASE,bindings={x:sha(ROOT/x) for x in paths},
        target_constant_nodes_before_freeze=0,independent_audit_nodes_before_freeze=0,
        exploratory_algebra_done=True,manufactured_fixtures_done=True,
        numerical_target_or_grid_adaptation_after_freeze=False))

def verify_freeze():
    lock=read(OUT/'FREEZE.json')
    for path,digest in lock['bindings'].items():
        require(sha(ROOT/path)==digest,'Frozen source drift: '+path)
    inp=read(OUT/'INPUTS.json')
    require(inp['parameters']==PARAMS and inp['n_values']==[64,256,1024,4096],'Input contract')
    return inp

def run(kind):
    inp=verify_freeze();marker=OUT/(kind+'_START.json')
    write_new(marker,dict(schema='r02_run_start_v1',kind=kind,started_utc=now(),
        freeze_sha256=sha(OUT/'FREEZE.json'),input_sha256=sha(OUT/'INPUTS.json'),
        executable=sys.executable,python=platform.python_version(),attempt=1,automatic_retry=False))
    try:
        result=c.compute_constants(**inp['parameters']) if kind=='CONSTANTS' else audit.compute_audit(**inp['independent_audit'])
        result['run_provenance']=dict(marker_sha256=sha(marker),freeze_sha256=sha(OUT/'FREEZE.json'),
            input_sha256=sha(OUT/'INPUTS.json'),finished_utc=now())
        write_new(OUT/(kind+'.json'),result)
        require(result['status']=='complete',kind+' incomplete; original result retained')
        if kind=='CONSTANTS':require(all(result['composition']['checks'].values()),'Constant qualification failed; no retry')
        print(json.dumps(dict(kind=kind,status=result['status'],counts=result.get('counts',result.get('work_counts')),seconds=result.get('elapsed_seconds_including_serialization_check',result.get('elapsed_seconds'))),indent=2))
    except BaseException:
        write_new(OUT/(kind+'_FAILURE.json'),dict(failed_utc=now(),traceback=traceback.format_exc(),automatic_retry=False))
        raise

def validate_results(inp,raw,independent):
    for kind,result in (('CONSTANTS',raw),('AUDIT',independent)):
        require(result['status']=='complete',kind+' status')
        p=result['run_provenance']
        require(p['marker_sha256']==sha(OUT/(kind+'_START.json')) and p['freeze_sha256']==sha(OUT/'FREEZE.json') and p['input_sha256']==sha(OUT/'INPUTS.json'),kind+' run binding')
    config=raw['configuration']
    require(config['rho_exact']=='9/20' and config['K']==100 and config['N']==160 and config['step_exact']=='1/16' and config['precision_bits']==192 and config['strip_exact']=='3/4','Primary grid')
    require(raw['hard_seconds']==600 and raw['elapsed_seconds_including_serialization_check']<=600,'Primary budget')
    require(raw['counts']['univariate_nodes']==321 and raw['counts']['nested_nodes']==103041 and raw['counts']['finite_scalar_sums_completed']==1296,'Primary actual work')
    config2=independent['configuration']
    require(config2['rho_exact']=={'numerator':'9','denominator':'20'} and config2['K']==100 and config2['GH_orders']==[32,48] and not config2['quadrature_weight_renormalization'],'Independent parameters')
    require(config2['hard_seconds']==120 and independent['elapsed_seconds']<=120,'Independent budget')
    require(independent['work_counts']['active_Gaussian_vector_nodes']==143360 and independent['work_counts']['adaptive_quad_calls_completed']==2,'Independent actual work')
    require(all(raw['counts'][k]==0 for k in ('new_D_n_scores','new_MC_draws','new_NPZ_reads','optimizer_steps','new_teacher_PMFs')),'Primary science scope')
    require(all(independent['work_counts'][k]==0 for k in ('new_project_Dn_evaluations','new_random_samples','new_NPZ_reads','new_optimizations')),'Independent science scope')

def analyze():
    inp=verify_freeze();raw=read(OUT/'CONSTANTS.json');ind=read(OUT/'AUDIT.json')
    validate_results(inp,raw,ind)
    rebuilt=c.compose_saved(raw)
    require(rebuilt==raw['composition'] and all(rebuilt['checks'].values()),'Saved constant reconstruction')
    ctx.prec=192
    saved=rebuilt['constants'];k={name:c.from_record(value) for name,value in saved.items()}
    for name in ('v','m_H','V_H','V_A','J_K','B_K'):require(lo(saved[name])>0,'Positive constant '+name)
    widthgate=F(inp['constant_width_gate_exact'])
    require(all(hi(value)-lo(value)<widthgate for value in saved.values()),'Constant widths')
    cross=[];distancegate=F(inp['audit_distance_gate_exact'])
    for name in ind['targets_by_order']['32']:
        a,b=map(qfloat,(ind['targets_by_order']['32'][name],ind['targets_by_order']['48'][name]))
        lower,upper=lo(saved[name]),hi(saved[name])
        dist=[max(F(0),lower-x,x-upper) for x in (a,b)]
        passed=max(dist)<distancegate and abs(a-b)<distancegate
        cross.append(dict(name=name,GH32=ind['targets_by_order']['32'][name],GH48=ind['targets_by_order']['48'][name],
            distance32_exact=str(dist[0]),distance48_exact=str(dist[1]),GH_difference_exact=str(abs(a-b)),passed=passed))
    require(all(row['passed'] for row in cross),'Independent expression gate failed')
    v,mH,J,B,cbar,skew=(k[x] for x in ('v','m_H','J_K','B_K','cbar','kappa3_L'))
    coefficient=k['second_order_coefficient']
    nuisance_component=B/v*(mH*skew/(3*v))
    active_component=-B/v*cbar
    require(overlap(c.record(nuisance_component+active_component),saved['second_order_coefficient']),'Signed component sum')
    Lphi=1/(2*arb.pi()*arb(1).exp()).sqrt()
    marked_numerator=100*Lphi*(k['V_H']*k['V_A']).sqrt()/v
    nfloor=(100*Lphi*(k['V_H']*k['V_A']).sqrt()/(mH*J*v.sqrt()))**2
    I=J/(100*99);tau=c.aq(F(1,2**98))/(2*arb.pi().sqrt()*I)
    old=read(ROOT/OLD);prior={r['n']:r for r in old['per_n']}
    rows=[]
    for n in inp['n_values']:
        s=(n*v).sqrt();w=J/s;signal=mH*w
        leading=-signal;next_term=coefficient/n;truncated=leading+next_term
        Rmin2=4*(100*arb(2)**c.aq('3/2')*(1+w)/signal).log()
        bmin2=c.aq('57/10')*Rmin2/(n*v)
        eta_floor=1-(-bmin2).exp()-tau
        budget=marked_numerator/n
        numeric=prior[n]['D12']
        observed=c.aq(qfloat(numeric['mean']))
        rows.append(dict(n=n,primary=(n in inp['primary_n_values']),
            s_n=c.record(s),w_n=c.record(w),leading=c.record(leading),correction=c.record(next_term),truncated_second_order=c.record(truncated),
            truncated_relative_to_leading=c.record(-truncated/signal),
            condition_event_Rmin_squared=c.record(Rmin2),gaussian_shift_bmin_squared=c.record(bmin2),
            conditional_eta_recipe_lower=c.record(eta_floor),conditional_eta_over_mH=c.record(eta_floor/mH),
            conditional_recipe_cannot_certify_leading=lo(c.record(eta_floor))>hi(saved['m_H']),
            absolute_mark_recipe_lower=c.record(budget),absolute_mark_budget_over_signal=c.record(budget/signal),
            absolute_mark_recipe_cannot_certify_leading=lo(c.record(budget))>hi(c.record(signal)),
            archived_D_mean=numeric['mean'],archived_D_conditional_interval=numeric['interval'],
            old_mean_minus_leading=c.record(observed-leading),old_mean_minus_truncated=c.record(observed-truncated),
            truncated_sign_is_not_true_D_certificate=True,old_event_not_upgraded=True))
    return dict(schema='r02_finite_scale_results_v1',freeze_sha256=sha(OUT/'FREEZE.json'),
        constants_sha256=sha(OUT/'CONSTANTS.json'),audit_sha256=sha(OUT/'AUDIT.json'),
        analytic_second_order='D=-m_H J_K/sqrt(nv)+coefficient/n+O_K(n^-3/2); remainder constant and starting n not explicit',
        constants=saved,nuisance_skew_component=c.record(nuisance_component),active_covariance_component=c.record(active_component),
        coefficient_sum_verified=True,independent_expression_checks=cross,
        conditional_tau_K=c.record(tau),absolute_mark_recipe_n_floor=c.record(nfloor),rows=rows,
        qualifications=dict(strict_constant_intervals=True,new_finite_n_D_values=False,finite_n_remainder_bound_obtained=False,
            fixed_n_leading_sign_certified=False,method_budget_lower_is_not_actual_error_lower=True,
            polynomial_growing_K_extension=False,old_MC_readout_is_conditional_and_numeric=True,
            comparison_is_posthoc_not_independent_prediction=True,coefficient_fitted_to_old_D=False),
        new_integrals_during_check=0,new_D_n_evaluations=0,new_MC_draws=0,new_NPZ_reads=0)

def main():
    require(sys.flags.optimize==0,'Assertions must be enabled')
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    for x in ('freeze','run-constants','run-audit','compose','check'):group.add_argument('--'+x,action='store_true')
    args=parser.parse_args()
    if args.freeze:freeze()
    elif args.run_constants:run('CONSTANTS')
    elif args.run_audit:run('AUDIT')
    else:
        result=analyze()
        if args.compose:write_new(OUT/'RESULTS.json',result)
        else:require(result==read(OUT/'RESULTS.json'),'Saved results differ')
        print(json.dumps(dict(status='passed',constants={k:v['arb_display_only'] for k,v in result['constants'].items()},
            n_floor=result['absolute_mark_recipe_n_floor']['arb_display_only'],
            rows=[{k:(v.get('arb_display_only') if isinstance(v,dict) and 'arb_display_only' in v else v) for k,v in row.items() if k in ('n','truncated_second_order','conditional_eta_over_mH','absolute_mark_budget_over_signal','old_mean_minus_truncated')} for row in result['rows']],
            finite_n_remainder_certified=False,new_integrals_during_check=0),indent=2))

if __name__=='__main__':main()
