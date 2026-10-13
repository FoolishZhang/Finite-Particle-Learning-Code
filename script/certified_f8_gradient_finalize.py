"""Compose finite 41-class gradient balls with the rigorous vector error."""
import argparse
from fractions import Fraction
import certified_teacher_compute as common
import certified_f8_gradient_compute as producer
import certified_f8_gradient_symmetry as symmetry
import certified_f8_gradient_bounds as bounds
from flint import arb,ctx

OUT=producer.OUT
TEACHER_DIR=common.ROOT/'artifacts/certified_teacher_v1_20261008'
SOURCES=('script/certified_f8_gradient_finalize.py','script/certified_f8_gradient_finalize_test.py')

def weighted_gradient(weights,gradients):
    if len(weights)!=len(gradients) or not gradients:raise ValueError('matching weights/gradients required')
    d=len(gradients[0])
    if not d or any(len(g)!=d for g in gradients):raise ValueError('gradient dimension')
    return [sum((w*g[j] for w,g in zip(weights,gradients)),arb(0)) for j in range(d)]

def certify_vector(vector,error):
    if not vector or not error>=0:raise ValueError('finite vector and nonnegative error')
    finite_norm=sum((abs(x).upper()**2 for x in vector),arb(0)).sqrt()
    total_norm=finite_norm+error
    enclosure=(-error.upper()).union(error.upper())
    return dict(finite_vector=[common.record(x) for x in vector],finite_box_norm_upper=common.record(finite_norm),
        arithmetic_box_L2_radius_upper=common.record(sum(((x.upper()-x.lower())**2/4 for x in vector),arb(0)).sqrt()),
        true_component_enclosures=[common.record(x+enclosure) for x in vector],
        mathematical_vector_error_upper=common.record(error),true_gradient_L2_upper=common.record(total_norm),
        gradient_L2_strictly_below_1e8_inverse=bool(total_norm<common.aq('1/100000000')),
        norm_uses_triangle_vector_error_once=True)

def calculate():
    inputs=producer.verify_freeze();frozen=common.read(producer.FREEZE)
    for name in SOURCES:
        if frozen['sources'].get(name)!=common.sha(common.ROOT/name):raise ValueError('finalize source not frozen')
    summary=common.read(OUT/'FINITE_GRADIENTS.json');teacher=common.read(TEACHER_DIR/'FINITE_GRID_SUM.json')
    teacher_cert=common.read(TEACHER_DIR/'TEACHER_CERTIFICATE.json')
    previous=ctx.prec;ctx.prec=192
    try:
        q=[common.from_record(x) for x in teacher['pmf_finite_sum']]
        te=common.aq(bounds.TEACHER_ERROR);tm=1+te
        if not common.from_endpoint(teacher_cert['total_L1_error_from_finite_midpoints']['upper_exact_dyadic'])<te:
            raise ValueError('teacher rounded error is too small')
        if not sum(q,arb(0)).upper()<tm:raise ValueError('finite teacher mass not within declared upper bound')
        grouped=symmetry.aggregate_weights(q)['representative_weights']
        gradient=[]
        for count,entry in zip(inputs['mapping']['representatives'],summary['representative_files']):
            p=common.ROOT/entry['path']
            if common.sha(p)!=entry['sha256']:raise ValueError('representative hash drift')
            row=common.read(p)
            if row['pair_counts']!=count or row['status']!='completed':raise ValueError('representative coverage')
            gradient.append([common.from_record(x) for x in row['conditional_gradient_finite']])
        if len(gradient)!=41 or len(grouped)!=41:raise ValueError('fixed representative coverage')
        finite=weighted_gradient(grouped,gradient)
        budget=bounds.gradient_l2_bounds(inputs['a_pairs_exact'])
        error=common.from_endpoint(budget['bounds']['total_without_compute_arithmetic_L2']['upper_exact_dyadic'])
        result=certify_vector(finite,error)
        paths=[producer.INPUTS,producer.FREEZE,OUT/'FINITE_GRADIENTS.json',TEACHER_DIR/'FINITE_GRID_SUM.json',TEACHER_DIR/'TEACHER_CERTIFICATE.json']
        paths += [common.ROOT/n for n in (*SOURCES,'script/certified_f8_gradient_bounds.py','script/certified_f8_gradient_symmetry.py')]
        result.update(schema='certified_f8_true_gradient_v1',precision_bits=192,
            input_sha256={p.relative_to(common.ROOT).as_posix():common.sha(p) for p in paths},
            analytic_budget=budget,representative_weights=[common.record(x) for x in grouped],
            finite_teacher_mass_enclosure=common.record(sum(q,arb(0))),teacher_error_counted_once=True,
            weights_or_Gaussian_mass_renormalized=False,teacher_within_pair_exchangeability_assumed=False,
            exact_teacher_symmetry_lifts_four_norm_to_full24=True,
            true_F8_gradient_enclosed=True,true_F8_stationary_existence_certified=False,
            true_F8_Hessian_certified=False,true_F8_local_minimum_certified=False,
            no_new_optimization=True,new_integrals_during_composition=0,new_random_draws=0)
        return result
    finally:ctx.prec=previous

if __name__=='__main__':
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run',action='store_true');mode.add_argument('--check',action='store_true');args=parser.parse_args()
    result=calculate();p=OUT/'GRADIENT_CERTIFICATE.json'
    if args.run:common.write_new(p,result)
    elif common.read(p)!=result:raise ValueError('gradient certificate arithmetic drift')
    print(common.json.dumps(dict(passed=True,gradient_target_met=result['gradient_L2_strictly_below_1e8_inverse'],new_integrals=0,new_grid_evaluations=0),indent=2))
