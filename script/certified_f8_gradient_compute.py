"""192-bit paired-count finite F8 gradients; no Hessian/optimization/RNG.

Finite sums only; separate analytic inner/outer/teacher bounds are required.
No scientific invocation is permitted without the author's fixed INPUTS/FREEZE.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
from fractions import Fraction
from pathlib import Path
import time
import certified_teacher_compute as common
import certified_f8_gradient_symmetry as symmetry
from flint import arb,arb_mat,ctx

ROOT=common.ROOT;OUT=ROOT/'artifacts/certified_f8_gradient_v1_20261008'
INPUTS=OUT/'INPUTS.json';FREEZE=OUT/'FREEZE.json'
NOMINAL='artifacts/local_witness_feasibility_v1_20261008/NOMINAL_SUMMARY.json'
BITS=192


def integer(value,name):
    if type(value) is not int or value<1:raise ValueError(name+' must be a positive actual integer')
    return value


def precompute_inner(a,h,L):
    """Exact input -> cached z/w/sigma; caller keeps the chosen Arb context."""
    a=tuple(common.exact(x) for x in a);h,L=common.exact(h),common.exact(L)
    if not a or any(x<0 for x in a) or h<=0 or L<=0 or (L/h).denominator!=1:
        raise ValueError('nonnegative a and positive exact integer-width grid required')
    N=(L/h).numerator;aa=list(map(common.aq,a));step=common.aq(h);root=(2*arb.pi()).sqrt()
    z=[step*i for i in range(-N,N+1)]
    w=[step*(-x*x/2).exp()/root for x in z]
    probabilities=[[(1/(1+(aj*x).exp()),1/(1+(-aj*x).exp())) for aj in aa] for x in z]
    return dict(a_exact=[str(x) for x in a],h_exact=str(h),L_exact=str(L),N=N,z=z,w=w,
        probabilities=probabilities,normal_mass=sum(w,arb(0)),pair_count=len(a),normal_nodes=len(z),precision_bits=int(ctx.prec))


def precompute_count(cache,c):
    c=tuple(c);p=cache['pair_count']
    if len(c)!=p or any(type(x) is not int or x not in (0,1,2) for x in c):raise ValueError('ternary pair count vector required')
    sqrt2=arb(2).sqrt();A=[];weighted_scores=[]
    for z,w,probabilities in zip(cache['z'],cache['w'],cache['probabilities']):
        likelihood=arb(1);scores=[]
        for count,(no,yes) in zip(c,probabilities):
            likelihood*=yes**count*no**(2-count)
            scores.append(w*z*(count-2*yes)/sqrt2)
        if not likelihood>0:raise ArithmeticError('conditional likelihood not strictly positive')
        A.append(likelihood);weighted_scores.append(scores)
    return A,weighted_scores


def saturation_enclosures(threshold):
    T=integer(threshold,'saturation threshold')
    if T!=128:raise ValueError('fixed safe saturation threshold is128')
    E=arb(2)**-184;S=arb(2)**-177
    if not (-arb(128)).exp()<E or not 128*(-arb(128)).exp()<S:
        raise ArithmeticError('fixed dyadic saturation bounds were not proved')
    return arb(E/2,E/2),arb(S/2,S/2)


def exponential_factors(t,*,threshold=None,enclosures=None):
    """Rigorous interval replacement, never numerical clipping to zero."""
    if threshold is not None and t.lower()>=arb(threshold):
        e,te=enclosures if enclosures is not None else saturation_enclosures(threshold)
        return e,te,True
    e=(-t).exp()
    return e,t*e,False


class EngineBudgetStop(RuntimeError):
    def __init__(self,partial):
        super().__init__('fixed total wall budget exhausted; no automatic retry')
        self.partial=partial


def finite_representative(cache,c,hv,vmin,vmax,*,K=8,backend='loop',saturation_threshold=None,deadline=None):
    """Conditional four-isometric-score finite sum; caller sets Arb precision."""
    K=integer(K,'K');hv,lo,hi=map(common.exact,(hv,vmin,vmax))
    if int(ctx.prec)!=cache['precision_bits']:raise ValueError('cached and active Arb precision must agree')
    if hv<=0 or hi<=lo or (lo/hv).denominator!=1 or (hi/hv).denominator!=1:raise ValueError('exact integral outer-index grid required')
    if backend not in ('loop','arb_mat'):raise ValueError('unknown contraction backend')
    enclosures=saturation_enclosures(saturation_threshold) if saturation_threshold is not None else None
    tick=time.perf_counter();A,WG=precompute_count(cache,c);p=cache['pair_count'];matrix=arb_mat(WG) if backend=='arb_mat' else None
    total=[arb(0) for _ in range(p)];completed=0;factors=0;saturated=0;psimin=psimax=None
    last_v=None;current_inner_completed=0;nodes=(hi-lo)/hv+1
    def output(complete):
        values=[common.record(-K*common.aq(hv)*x) for x in total]
        return dict(status='completed' if complete else 'budget_stopped',pair_counts=list(c),K=K,precision_bits=int(ctx.prec),
            conditional_gradient_finite=values,gradient_coordinates='orthonormal pair directions (e_j+e_(m-1-j))/sqrt2, m=2*pair_count',
            inner_grid=dict(h_exact=cache['h_exact'],L_exact=cache['L_exact'],nodes=cache['normal_nodes'],full_weights=True),
            outer_grid=dict(h_exact=str(hv),vmin_exact=str(lo),vmax_exact=str(hi),nodes_prescribed=int(nodes),nodes_completed=completed,full_weights=True),
            raw_normal_grid_mass=common.record(cache['normal_mass']),normal_mass_strictly_below_one=bool(cache['normal_mass']<1),
            psi_completed_grid_lower_endpoint=None if psimin is None else common.record(psimin),
            psi_completed_grid_upper_endpoint=None if psimax is None else common.record(psimax),
            psi_upper_bounded_by_raw_normal_mass=None if psimax is None else bool(psimax<=cache['normal_mass'].upper()),
            coefficients_or_Gaussian_mass_renormalized=False,likelihood_scale_exact='1',physical_u='exp(v)',
            saturation_threshold=saturation_threshold,saturation_is_interval_outer_enclosure=saturation_threshold is not None,
            saturation_caps=None if saturation_threshold is None else dict(exp_upper_exact='2^-184',t_exp_upper_exact='2^-177',
                mathematical_lower_exact='0',threshold_exact='128',bounds_verified_with_Arb=True,
                Arb_ball_outward_rounding_retained=True,numeric_nonnegative_lower_endpoint_required=False),
            backend=backend,last_completed_v_index=last_v,current_partial_inner_factors=current_inner_completed,
            resources=dict(laplace_factors_evaluated=factors,saturated_factor_enclosures=saturated,
                ordinary_exponential_calls=factors-saturated,inner_nodes=cache['normal_nodes'],outer_nodes_completed=completed,
                new_kernel_calls=0,new_random_draws=0,new_optimizer_updates=0),wall_seconds=time.perf_counter()-tick,
            finite_sum_arithmetic_enclosed=True,analytic_discretization_tail_teacher_errors_added=False,
            true_F8_gradient_certified=False,true_F8_stationary_existence_certified=False)
    for index in range((lo/hv).numerator,(hi/hv).numerator+1):
        if deadline is not None and time.perf_counter()>=deadline:raise EngineBudgetStop(output(False))
        u=(common.aq(hv)*index).exp();psi=arb(0);coeffs=[];current_inner_completed=0
        for j,(likelihood,w) in enumerate(zip(A,cache['w'])):
            if deadline is not None and j%32==0 and time.perf_counter()>=deadline:raise EngineBudgetStop(output(False))
            e,te,used=exponential_factors(u*likelihood,threshold=saturation_threshold,enclosures=enclosures)
            psi+=w*e;coeffs.append(te);factors+=1;saturated+=used;current_inner_completed+=1
        if backend=='loop':D=[sum((coef*score[k] for coef,score in zip(coeffs,WG)),arb(0)) for k in range(p)]
        else:
            contraction=arb_mat([coeffs])*matrix;D=[contraction[0,k] for k in range(p)]
        power=psi**(K-1)
        for k in range(p):total[k]+=power*D[k]
        psimin=psi.lower() if psimin is None else min(psimin,psi.lower())
        psimax=psi.upper() if psimax is None else max(psimax,psi.upper())
        completed+=1;last_v=index;current_inner_completed=0
    return output(True)


def prepared_inputs(inner_h,backend,threshold,hard_seconds):
    """Call only after root chooses the fixed science configuration."""
    inner_h=common.exact(inner_h);integer(hard_seconds,'hard seconds')
    if inner_h!=Fraction(1,40):raise ValueError('root fixed inner step 1/40')
    if backend not in ('loop','arb_mat'):raise ValueError('backend required')
    if threshold!=128:raise ValueError('root fixed threshold128 with exact dyadic upper caps')
    nominal=common.read(ROOT/NOMINAL);a=symmetry.exact_pair_parameters(nominal['center']['theta'])
    mapping=symmetry.symmetry_map(8)
    return dict(schema='certified_f8_gradient_inputs_v1',a_pairs_exact=[str(x) for x in a],K=8,precision_bits=BITS,
        inner_grid=dict(h_exact='1/40',L_exact='11',nodes=881,full_weights=True),
        outer_grid=dict(h_exact='1/8',vmin_exact='-32',vmax_exact='64',nodes=769,full_weights=True),
        mapping=mapping,backend=backend,saturation_threshold=threshold,hard_total_wall_seconds=hard_seconds,
        saturation_caps=dict(exp_upper_exact='2^-184',t_exp_upper_exact='2^-177',lower_exact='0',
            proof='exp(-128)<2^-184; for t>=128, t exp(-t)<=128 exp(-128)<2^-177',
            Arb_ball_outward_rounding_retained=True,numeric_nonnegative_lower_endpoint_required=False),
        input_source_sha256={NOMINAL:common.sha(ROOT/NOMINAL),'script/certified_f8_gradient_symmetry.py':common.sha(ROOT/'script/certified_f8_gradient_symmetry.py'),
            'script/certified_teacher_compute.py':common.sha(ROOT/'script/certified_teacher_compute.py')},
        derivative_formula='g_c=-8 sum_v hv Psi^7 sum_z hz phi(z) u A_c exp(-u A_c) z(c-2sigma)/sqrt2',
        true_F8_gradient_certified=False,true_F8_stationary_existence_certified=False,
        new_random_draws=0,new_optimizer_updates=0,automatic_retry_or_extension=False)


def verify_freeze():
    inputs=common.read(INPUTS)
    if inputs!=prepared_inputs(inputs['inner_grid']['h_exact'],inputs['backend'],inputs['saturation_threshold'],inputs['hard_total_wall_seconds']):raise ValueError('fixed inputs drift')
    sources=common.read(FREEZE)['sources']
    required=('script/certified_f8_gradient_compute.py','script/certified_f8_gradient_compute_test.py',INPUTS.relative_to(ROOT).as_posix())
    if any(path not in sources for path in required):raise ValueError('FREEZE omits compute/test/INPUTS')
    for path,digest in sources.items():
        candidate=(ROOT/path).resolve()
        if not candidate.is_relative_to(ROOT) or any(s.lower() in ('.npz','.npy') for s in candidate.suffixes) or common.sha(candidate)!=digest:raise ValueError('freeze source drift: '+path)
    return inputs


def run():
    inputs=verify_freeze();start=OUT/'RUN_START.json'
    if start.exists():raise FileExistsError('RUN_START exists; no overwrite or automatic retry')
    tick=time.perf_counter();deadline=tick+inputs['hard_total_wall_seconds']
    common.write_new(start,dict(created_utc=common.now(),inputs_sha256=common.sha(INPUTS),freeze_sha256=common.sha(FREEZE)))
    previous=ctx.prec;ctx.prec=BITS
    try:
        cache=precompute_inner(inputs['a_pairs_exact'],'1/40','11');entries=[]
        for index,c in enumerate(inputs['mapping']['representatives']):
            try:
                row=finite_representative(cache,c,'1/8','-32','64',K=8,backend=inputs['backend'],saturation_threshold=inputs['saturation_threshold'],deadline=deadline)
            except EngineBudgetStop as error:
                row=error.partial;row.update(representative_index=index,inputs_sha256=common.sha(INPUTS),freeze_sha256=common.sha(FREEZE))
                path=OUT/f'partial_rep_{index:02d}.json';common.write_new(path,row)
                stop=dict(status='budget_stopped',completed_representatives=entries,partial_rep=path.relative_to(ROOT).as_posix(),
                    wall_seconds=time.perf_counter()-tick,automatic_retry=False)
                common.write_new(OUT/'RUN_STOP.json',stop);return stop
            row.update(representative_index=index,inputs_sha256=common.sha(INPUTS),freeze_sha256=common.sha(FREEZE))
            path=OUT/f'rep_{index:02d}.json';common.write_new(path,row)
            entries.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=common.sha(path),bytes=path.stat().st_size))
            if time.perf_counter()>=deadline:
                stop=dict(status='budget_stopped',completed_representatives=entries,wall_seconds=time.perf_counter()-tick,
                    automatic_retry=False,all_rep_points_saved=len(entries)==41,final_summary_written=False)
                common.write_new(OUT/'RUN_STOP.json',stop);return stop
        summary=dict(schema='certified_f8_gradient_finite_summary_v1',representative_files=entries,representative_count=41,
            isometric_gradient_dimension=4,inputs_sha256=common.sha(INPUTS),freeze_sha256=common.sha(FREEZE),run_start_sha256=common.sha(start),
            wall_seconds=time.perf_counter()-tick,true_F8_gradient_certified=False,analytic_errors_added=False,
            true_F8_stationary_existence_certified=False,new_random_draws=0,new_optimizer_updates=0)
        common.write_new(OUT/'FINITE_GRADIENTS.json',summary)
        return dict(status='completed',representatives=41,wall_seconds=summary['wall_seconds'])
    finally:ctx.prec=previous


def check():
    inputs=verify_freeze();summary=common.read(OUT/'FINITE_GRADIENTS.json')
    if summary['inputs_sha256']!=common.sha(INPUTS) or summary['freeze_sha256']!=common.sha(FREEZE) or summary['run_start_sha256']!=common.sha(OUT/'RUN_START.json'):
        raise ValueError('summary source lock drift')
    if summary['representative_count']!=41 or len(summary['representative_files'])!=41 or summary['isometric_gradient_dimension']!=4:
        raise ValueError('fixed41rep/four direction coverage drift')
    previous=ctx.prec;ctx.prec=BITS
    try:
        saturation_enclosures(128)
        for index,entry in enumerate(summary['representative_files']):
            path=ROOT/entry['path']
            if path!=OUT/f'rep_{index:02d}.json' or common.sha(path)!=entry['sha256'] or path.stat().st_size!=entry['bytes']:raise ValueError('rep file hash/size/path drift')
            row=common.read(path)
            if row['status']!='completed' or row['representative_index']!=index or row['pair_counts']!=inputs['mapping']['representatives'][index]:raise ValueError('rep order/completeness drift')
            if row['inputs_sha256']!=common.sha(INPUTS) or row['freeze_sha256']!=common.sha(FREEZE):raise ValueError('rep locks drift')
            expected_outer={k:v for k,v in inputs['outer_grid'].items() if k!='nodes'}
            expected_outer.update(nodes_prescribed=769,nodes_completed=769)
            if row['inner_grid']!=inputs['inner_grid'] or row['outer_grid']!=expected_outer:raise ValueError('fixed grids drift')
            if row['backend']!=inputs['backend'] or row['saturation_threshold']!=128 or row['likelihood_scale_exact']!='1' or row['coefficients_or_Gaussian_mass_renormalized'] is not False:
                raise ValueError('backend/scale/saturation policy drift')
            if row['precision_bits']!=192 or row['saturation_caps']['exp_upper_exact']!='2^-184' or row['saturation_caps']['t_exp_upper_exact']!='2^-177':
                raise ValueError('precision/fixed dyadic caps drift')
            counts=row['resources']
            if counts['laplace_factors_evaluated']!=881*769 or counts['saturated_factor_enclosures']+counts['ordinary_exponential_calls']!=881*769:raise ValueError('factor counts drift')
            if len(row['conditional_gradient_finite'])!=4:raise ValueError('four gradient components required')
            for ball in row['conditional_gradient_finite']:
                value=common.from_record(ball)
                if not value.is_finite():raise ValueError('nonfinite gradient endpoints')
            for field in ('new_kernel_calls','new_random_draws','new_optimizer_updates'):
                if counts[field]!=0:raise ValueError('unexpected research extension')
            if row['true_F8_gradient_certified'] is not False or row['true_F8_stationary_existence_certified'] is not False:
                raise ValueError('finite gradients cannot certify complete F8')
    finally:ctx.prec=previous
    return dict(status='passed',new_scientific_grid_calls=0,new_integrals=0,new_kernel_calls=0,new_random_draws=0,
        validation_scope='source locks/public checkpoint endpoints and fixed counts; no independent finite-grid replay')


def main():
    parser=argparse.ArgumentParser();modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--run',action='store_true');modes.add_argument('--prepare-inputs',action='store_true')
    modes.add_argument('--check',action='store_true')
    parser.add_argument('--backend',choices=('loop','arb_mat'),default='loop');parser.add_argument('--threshold',type=int)
    parser.add_argument('--hard-seconds',type=int,default=1800);args=parser.parse_args()
    if args.prepare_inputs:
        OUT.mkdir(parents=True,exist_ok=True);common.write_new(INPUTS,prepared_inputs('1/40',args.backend,args.threshold,args.hard_seconds));output=dict(status='prepared',new_scientific_grid_calls=0)
    elif args.check:output=check()
    else:output=run()
    print(common.json.dumps(output,indent=2))


if __name__=='__main__':main()
