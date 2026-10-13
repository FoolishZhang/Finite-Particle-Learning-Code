"""Selected saved-arithmetic functions; no producer entry points."""

from __future__ import annotations

from fractions import Fraction

import common

from flint import arb,ctx

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
