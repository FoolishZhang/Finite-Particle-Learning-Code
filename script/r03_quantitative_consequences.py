"""Exact rational consequences of saved R03 certificates; no integration."""
import argparse
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE='artifacts/r03_teacher_family_v1_20261011/'
SOURCES=[BASE+'TEACHER_CERTIFICATE.json',BASE+'h001/CERTIFICATE.json',
         BASE+'SPECTRAL_REFINEMENT.json','artifacts/angular_witness_v1_20261010/DIRECTION.json']
METHOD='docs/08_research_研究工作/R03TeacherFamily_v1_20261011/QUANTITATIVE_CONSEQUENCES.md'


def ep(x):return F(int(x['mantissa']))*F(2)**x['exponent_of_two']
def interval(x):return ep(x['lower_exact_dyadic']),ep(x['upper_exact_dyadic'])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def outward(lo,hi):
    assert lo<=hi
    display=[]
    with localcontext() as c:
        c.prec=50
        for value,rounding in [(lo,ROUND_FLOOR),(hi,ROUND_CEILING)]:
            d=Decimal(value.numerator)/Decimal(value.denominator)
            display.append(str(d.quantize(Decimal('1e-12'),rounding=rounding)))
    assert F(display[0])<=lo<=hi<=F(display[1])
    return dict(lower_exact=str(lo),upper_exact=str(hi),outer_decimal=display)


def compute_from_saved(teacher,cert,refined,direction):
    h=F(1,1000); L2=F('0.442323');half=L2/2
    assert refined['claimed_lambda_interval_exact']==['-1/1000','1/1000']
    selected=[r for r in refined['rows'] if F(r['h_exact'])==h][0]
    assert selected['core_passed'] and cert['contraction']['passed'] and cert['risk']['passed']
    qlo,qhi=interval(teacher['prime_true_L1'])
    qratio=(qlo-half*h,qhi+half*h)
    dlo,dhi=interval(cert['reversal_derivative']['a1_minus_a8_prime'])
    assert 0<qratio[0] and 0<dlo<dhi
    a0=list(map(F,cert['a0_exact']));r=list(map(F,cert['radii_exact']))
    amin=min(x-y for x,y in zip(a0,r));assert amin>0
    u=direction['u_integer'];S=direction['S_integer']
    assert sum(u)==0 and sum(x*x for x in u)==S and len(set(u))>1
    assert selected['blocks']['Hperp']['target_mu_exact']=='1/200'
    finite_curvature_lower=F(1,200)*amin*amin
    risklo,riskhi=interval(cert['risk']['true_curvature']);assert riskhi<0
    pairs=[]
    for pair in teacher['first_row_spin_pairs']:
        lo,hi=interval(pair['derivative_true']);assert lo-L2*h>0
        pairs.append(dict(rows_one_based=pair['rows_one_based'],
            derivative_for_all_lambda=outward(lo-L2*h,hi+L2*h),
            change_at_positive_endpoint=outward(h*lo-half*h*h,h*hi+half*h*h),
            change_at_negative_endpoint=outward(-h*hi-half*h*h,-h*lo+half*h*h)))
    return dict(schema='r03_quantitative_consequences_v1',passed=True,
        lambda_interval_exact=['-1/1000','1/1000'],
        teacher_L1_change_divided_by_abs_lambda_for_all_nonzero_lambda=outward(*qratio),
        teacher_L1_change_at_either_endpoint=outward(qratio[0]*h,qratio[1]*h),
        root_a1_minus_a8_prime=outward(dlo,dhi),
        root_a1_minus_a8_divided_by_lambda_for_all_nonzero_lambda=outward(dlo,dhi),
        root_endpoint_absolute_reversal_gap=outward(h*dlo,h*dhi),
        original_root_reversal_identity_is_analytic=True,
        teacher_first_row_pair_moments=pairs,
        candidate_min_rowlength_exact=str(amin),
        F8_fixed_angle_curvature_strict_lower_exact=str(finite_curvature_lower),
        F8_fixed_angle_curvature_strict_lower_display=str(float(finite_curvature_lower)),
        F8_fixed_angle_curvature_greater_than_0_0011=finite_curvature_lower>F('.0011'),
        true_risk_fixed_angle_curvature=outward(risklo,riskhi),
        all_nonzero_lambda_break_original_reversal=True,
        all_teachers_have_same_fair_observed_marginals=True,
        candidate_angle_path_preserves_its_logit_and_observed_marginals=True,
        sufficiently_small_nonzero_two_sided_steps_have_rank_two=True,
        explicit_step_size_or_gain_certified=False,
        fixed_slice_mu_is_not_claimed_on_varying_orthogonal_complement=True,
        new_target_integrals=0,new_optimizer_updates=0,new_random_draws=0,
        arithmetic='standard-library Fraction; display endpoints separately audited')


def produce():
    data=[json.loads((ROOT/s).read_text(encoding='utf-8')) for s in SOURCES]
    result=compute_from_saved(*data)
    result['source_bindings']={s:sha(ROOT/s) for s in SOURCES+[METHOD,Path(__file__).resolve().relative_to(ROOT).as_posix()]}
    return result


def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--write',action='store_true');g.add_argument('--check',action='store_true')
    args=p.parse_args();result=produce();path=ROOT/BASE/'QUANTITATIVE_CONSEQUENCES.json'
    if args.write:
        with path.open('x',encoding='utf-8',newline='\n') as f:
            json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    else:assert json.loads(path.read_text(encoding='utf-8'))==result
    print(json.dumps({k:result[k] for k in ['passed','teacher_L1_change_at_either_endpoint',
        'root_endpoint_absolute_reversal_gap','F8_fixed_angle_curvature_greater_than_0_0011',
        'true_risk_fixed_angle_curvature','new_target_integrals']},indent=2))


if __name__=='__main__':main()
