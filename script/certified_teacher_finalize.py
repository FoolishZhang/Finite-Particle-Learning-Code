"""Compose rigorous finite-sum arithmetic and analytic teacher quadrature error."""
import argparse
from fractions import Fraction
import certified_teacher_compute as common
import certified_teacher_bounds as bounds
from flint import arb,ctx

OUT=common.OUT
SOURCE='script/certified_teacher_finalize.py'

def compose(values,exact_old,err):
    if len(values)!=len(exact_old) or not values or not err>=0:raise ValueError('matching nonempty PMF and nonnegative error required')
    uncertainty=(-err).union(err)
    arithmetic_radius_sum=sum((x.upper()-x.lower() for x in values),arb(0))/2
    total_error=arithmetic_radius_sum+err
    distance=sum((abs(x-y) for x,y in zip(values,exact_old)),arb(0))+err
    true_values=[common.record(x+uncertainty) for x in values]
    total_mass=sum(values,arb(0))+uncertainty
    positive=all(common.from_record(x)>0 for x in true_values)
    if not total_mass.contains(arb(1)) or not positive:raise ValueError('PMF enclosure sanity failed')
    return arithmetic_radius_sum,total_error,distance,true_values,total_mass,positive

def compute():
    common.verify_freeze();freeze=common.read(common.FREEZE)
    if freeze['sources'].get(SOURCE)!=common.sha(common.ROOT/SOURCE):raise ValueError('finalizer source not frozen')
    finite=common.read(OUT/'FINITE_GRID_SUM.json');old=common.read(common.ROOT/common.OLD_SOURCE)['teacher_pmf']
    previous=ctx.prec;ctx.prec=192
    try:
        bound=bounds.teacher_l1_bounds()
        err=common.from_endpoint(bound['bounds']['total_quadrature_L1']['upper_exact_dyadic'])
        values=[common.from_record(x) for x in finite['pmf_finite_sum']]
        exact_old=[common.aq(Fraction.from_float(x)) for x in old]
        arithmetic_radius_sum,total_error,distance,true_values,total_mass,positive=compose(values,exact_old,err)
        teacher_target=bounds.dyadic_lt_q(common.endpoint(total_error.upper()),'1/1000000000000')
        old_target=bounds.dyadic_lt_q(common.endpoint(distance.upper()),'1/10000000000')
        return dict(schema='certified_teacher_true_PMF_v1',
            scope='exact rational-angle mathematical teacher, including finite-sum arithmetic and analytic quadrature',
            source_sha256={name:common.sha(common.ROOT/name) for name in (SOURCE,'script/certified_teacher_bounds.py',common.OLD_SOURCE)},
            freeze_sha256=common.sha(common.FREEZE),finite_grid_sum_sha256=common.sha(OUT/'FINITE_GRID_SUM.json'),
            bounds=bound,true_PMF_enclosures=true_values,
            finite_sum_arithmetic_L1_radius=common.record(arithmetic_radius_sum),
            total_L1_error_from_finite_midpoints=common.record(total_error),
            saved_GH60_L1_distance_upper=common.record(distance),
            saved_GH60_comparison_includes_parameter_and_arithmetic_approximation=True,
            exact_teacher_mass_enclosure=common.record(total_mass),all_true_probabilities_positive=positive,
            total_L1_less_than_1e12_inverse=teacher_target,saved_GH60_L1_less_than_1e10_inverse=old_target,
            F8_teacher_error_transfer={
                'gradient_L2':common.record(arb(8).sqrt()*9*distance),
                'Hba_operator':common.record((arb(16)+arb(1)/4)*9*distance),
                'Hperp_operator':common.record((arb(8)+arb(1)/4)*distance)},
            transfer_scope='teacher-error component only, not total F8 integral error',
            strict_F8_stationary_existence_certified=False,new_grid_evaluations=0,new_random_draws=0)
    finally:ctx.prec=previous

if __name__=='__main__':
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run',action='store_true');mode.add_argument('--check',action='store_true');args=parser.parse_args()
    result=compute();target=OUT/'TEACHER_CERTIFICATE.json'
    if args.run:common.write_new(target,result)
    elif common.read(target)!=result:raise ValueError('teacher certificate arithmetic drift')
    print(common.json.dumps(dict(passed=True,teacher_target_met=result['total_L1_less_than_1e12_inverse'],
        saved_GH60_target_met=result['saved_GH60_L1_less_than_1e10_inverse'],new_grid_evaluations=0,new_integrals=0),indent=2))
