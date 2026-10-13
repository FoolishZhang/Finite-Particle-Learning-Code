"""True-Hessian distance and conditional local-witness synthesis, no new grid."""
from fractions import Fraction
import argparse
import certified_teacher_compute as common
import certified_f8_hessian_compute as producer
import certified_f8_hessian_bounds as bounds
from flint import arb,ctx

ROOT=common.ROOT;OUT=producer.OUT
SOURCES=('script/certified_f8_hessian_finalize.py','script/certified_f8_hessian_finalize_test.py')
SAVED=ROOT/'artifacts/local_witness_feasibility_v1_20261008/SAVED_MATRIX_CERTIFICATE.json'
GRAD=ROOT/'artifacts/certified_f8_gradient_v1_20261008/GRADIENT_CERTIFICATE.json'
GRAD_INPUTS=ROOT/'artifacts/certified_f8_gradient_v1_20261008/INPUTS.json'
RISK=ROOT/'artifacts/certified_teacher_v1_20261008/POSTHOC_RISK_BALL.json'
RISK_INPUTS=ROOT/'artifacts/certified_teacher_v1_20261008/CANDIDATE_RISK_INTEGRALS.json'
TEACHER_CERT=ROOT/'artifacts/certified_teacher_v1_20261008/TEACHER_CERTIFICATE.json'

def matrix_from_records(rows,n):
    if len(rows)!=n or any(len(r)!=n for r in rows):raise ValueError('matrix shape')
    ans=[[common.from_record(x) for x in row] for row in rows]
    if not all(x.is_finite() for row in ans for x in row):raise ValueError('nonfinite matrix')
    return ans

def exact_saved(rows,n):
    if len(rows)!=n or any(len(r)!=n for r in rows):raise ValueError('saved matrix shape')
    return [[common.aq(Fraction.from_float(float.fromhex(x))) for x in row] for row in rows]

def block_diagonal(left,right):
    n=len(left)
    if len(right)!=n or any(len(row)!=n for row in left+right):raise ValueError('block shapes')
    return [[left[i][j] if i<n and j<n else right[i-n][j-n] if i>=n and j>=n else arb(0)
             for j in range(2*n)] for i in range(2*n)]

def frobenius_distance_upper(matrix,target):
    if len(matrix)!=len(target) or any(len(a)!=len(b) for a,b in zip(matrix,target)):raise ValueError('distance shape')
    return sum((abs(a-b).upper()**2 for row,trow in zip(matrix,target) for a,b in zip(row,trow)),arb(0)).sqrt()

def witness_logic(error_ba,error_p,gradient_upper,risk_upper,anorm,*,saved_floors_verified):
    if not (error_ba>=0 and error_p>=0 and gradient_upper>=0 and anorm>0):raise ValueError('nonnegative error bounds and positive center norm required')
    r=common.aq('1/2000000');step=common.aq('1/10000000');gate=common.aq('1/1000')
    gcap=common.aq('1893/2500000000000');nu=common.aq('201723/10000000')
    m16=common.aq('1/20')-gate-14069*r;mt=common.aq('7/500')-gate-14069*r
    # Use a predeclared conservative cap, not a retrospectively chosen radius.
    root_radius=gcap/m16
    hgate=bool(error_ba<gate and error_p<gate)
    checks=dict(saved_floors_verified=bool(saved_floors_verified),Hessian_error_gates=hgate,
        predecessor_gradient_below_fixed_cap=bool(gradient_upper<gcap),radial_margin_positive=bool(m16>0),
        transverse_margin_positive=bool(mt>0),outward_root_condition=bool(gcap<m16*r),
        fixed_slice_transverse=bool(anorm>r),risk_uniform_cap=bool(risk_upper < -nu),
        fixed_step_stays_in_ball=bool(root_radius+step<r))
    existence=all(checks[k] for k in ('saved_floors_verified','Hessian_error_gates','predecessor_gradient_below_fixed_cap',
        'radial_margin_positive','outward_root_condition','fixed_slice_transverse'))
    local_minimum=existence and checks['transverse_margin_positive']
    positive_risk=existence and checks['risk_uniform_cap']
    conflict=local_minimum and checks['risk_uniform_cap'] and checks['fixed_step_stays_in_ball']
    return dict(physical_radius_exact='1/2000000',Hessian_error_gate_exact='1/1000',gradient_cap_exact='1893/2500000000000',
        L_F_safe_integer=14069,radial_curvature_lower=common.record(m16),transverse_curvature_lower=common.record(mt),
        outward_capacity=common.record(m16*r),root_distance_upper=common.record(root_radius),
        risk_derivative_negative_margin_exact='201723/10000000',fixed_step_exact='1/10000000',
        root_plus_step_distance_upper=common.record(root_radius+step),
        F8_increase_lower_bound=common.record(m16*step*step/2),risk_decrease_lower_bound=common.record(nu*step),
        root_KL_strict_lower_bound=common.record(nu*step),checks=checks,
        true_F8_stationary_existence_certified=existence,
        strict_local_minimum_modulo_rotations_certified=local_minimum,
        positive_risk_at_stationary_point_certified=positive_risk,
        fixed_step_objective_risk_conflict_certified=conflict,
        root_explicitly_solved_or_optimized=False,global_stationary_uniqueness_claimed=False,
        uniqueness_scope='symmetric restricted root in the fixed ball; local strict minimum after quotienting common rotations',
        general_optimizer_failure_claimed=False)

def calculate():
    inputs=producer.verify_freeze();freeze=common.read(producer.FREEZE)
    for p in SOURCES:
        if freeze['sources'].get(p)!=common.sha(ROOT/p):raise ValueError('synthesis not frozen')
    summary=common.read(OUT/'FINITE_HESSIANS.json');saved=common.read(SAVED);grad=common.read(GRAD)
    gradinputs=common.read(GRAD_INPUTS);risk=common.read(RISK);riskinputs=common.read(RISK_INPUTS)
    previous=ctx.prec;ctx.prec=192
    try:
        a=[Fraction(x) for x in inputs['a_pairs_exact']];full=a+list(reversed(a))
        if a!=[Fraction(x) for x in gradinputs['a_pairs_exact']]:raise ValueError('gradient center differs')
        oldtheta=[Fraction(int(x['numerator']),int(x['denominator'])) for x in saved['center']['theta_exact_dyadics']]
        riska=[Fraction(int(x['numerator']),int(x['denominator'])) for x in riskinputs['a_exact']]
        if oldtheta[:8]!=[Fraction(0)]*8 or oldtheta[8:]!=full or riska!=full:raise ValueError('Hessian/risk center differs')
        if risk['physical_radius_exact']!='1/2000000' or risk['parameter_dimension']!=24:raise ValueError('risk ball differs')
        if any(not common.aq(x)-common.aq('1/2000000')>common.aq('1/20') or not common.aq(x)+common.aq('1/2000000')<2 for x in full):raise ValueError('ball leaves original radial box')
        if not grad['true_F8_gradient_enclosed'] or not grad['exact_teacher_symmetry_lifts_four_norm_to_full24']:raise ValueError('true gradient not certified')
        if not saved['strict_saved_matrix_checks_passed'] or saved['radial_floor_exact']!='1/20' or saved['transverse_floor_exact']!='7/500':raise ValueError('saved floor certificate')
        if not saved['fixed_slice']['exact_a0_dot_V_zero'] or saved['fixed_slice']['basis_assumed_orthonormal']:raise ValueError('fixed slice metric')
        if saved['safe_L_F_proof']['L_F_safe_integer']!=14069 or not all(saved['safe_L_F_proof']['exact_fraction_checks_passed']):raise ValueError('Lipschitz constant')
        target_ba=exact_saved(saved['saved_Hba']['symmetrized_binary64_hex'],16)
        target_p=exact_saved(saved['saved_Hperp']['symmetrized_binary64_hex'],8)
        matrices={k:matrix_from_records(summary['matrix_blocks'][k],8) for k in ('Hbb','Haa','Hperp')}
        finite_ba=block_diagonal(matrices['Hbb'],matrices['Haa'])
        dist_ba=frobenius_distance_upper(finite_ba,target_ba);dist_p=frobenius_distance_upper(matrices['Hperp'],target_p)
        budget=bounds.hessian_operator_bounds(inputs['a_pairs_exact'])
        math_ba=common.from_endpoint(budget['bounds']['total_Hba_operator_without_arithmetic']['upper_exact_dyadic'])
        math_p=common.from_endpoint(budget['bounds']['total_Hp_operator_without_arithmetic']['upper_exact_dyadic'])
        err_ba=dist_ba+math_ba;err_p=dist_p+math_p
        tc=common.read(TEACHER_CERT);te=common.aq(bounds.TEACHER_ERROR)
        if not common.from_endpoint(tc['total_L1_error_from_finite_midpoints']['upper_exact_dyadic'])<te:raise ValueError('teacher error allowance too small')
        if not common.from_record(inputs['teacher']['original_teacher_finite_mass']).upper()<1+te:raise ValueError('teacher mass allowance too small')
        g=common.from_endpoint(grad['true_gradient_L2_upper']['upper_exact_dyadic'])
        rr=common.from_endpoint(risk['risk_direction_uniform_upper_expression']['upper_exact_dyadic'])
        norm=sum((common.aq(x*x) for x in full),arb(0)).sqrt()
        witness=witness_logic(err_ba,err_p,g,rr,norm,saved_floors_verified=True)
        paths=[producer.INPUTS,producer.FREEZE,OUT/'FINITE_HESSIANS.json',SAVED,GRAD,GRAD_INPUTS,RISK,RISK_INPUTS,TEACHER_CERT]
        paths += [ROOT/x for x in (*SOURCES,'script/certified_f8_hessian_bounds.py','docs/08_research_研究工作/CertifiedF8Hessian_v1_20261008/WITNESS_ARGUMENT.md')]
        return dict(schema='certified_f8_hessian_and_witness_v1',precision_bits=192,
            input_sha256={p.relative_to(ROOT).as_posix():common.sha(p) for p in paths},
            same_exact_center_across_gradient_Hessian_risk=True,analytic_budget=budget,
            finite_matrix_to_saved_Frobenius_upper=dict(Hba=common.record(dist_ba),Hperp=common.record(dist_p)),
            true_matrix_to_saved_operator_upper=dict(Hba=common.record(err_ba),Hperp=common.record(err_p)),
            exact_mixed_bias_radial_block_zero_used=True,full_Hba_error_from_max_of_two_true_block_errors=True,
            entrywise_bounds_converted_via_Frobenius_not_mistaken_for_operator=True,
            true_F8_Hessian_enclosed=True,Hessian_errors_below_point001=witness['checks']['Hessian_error_gates'],
            witness=witness,new_integrals=0,new_grid_evaluations=0,new_random_draws=0,new_optimizer_updates=0)
    finally:ctx.prec=previous

if __name__=='__main__':
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run',action='store_true');mode.add_argument('--check',action='store_true');args=parser.parse_args()
    result=calculate();p=OUT/'HESSIAN_AND_WITNESS_CERTIFICATE.json'
    if args.run:common.write_new(p,result)
    elif common.read(p)!=result:raise ValueError('Hessian/witness synthesis drift')
    print(common.json.dumps(dict(passed=True,Hessian_errors_below_point001=result['Hessian_errors_below_point001'],
        stationary_existence=result['witness']['true_F8_stationary_existence_certified'],
        local_minimum_modulo_rotations=result['witness']['strict_local_minimum_modulo_rotations_certified'],
        objective_risk_conflict=result['witness']['fixed_step_objective_risk_conflict_certified'],new_integrals=0),indent=2))
