"""Post-execution deterministic risk-ball corollary; no new quadrature."""
import argparse
from fractions import Fraction
from pathlib import Path
import certified_teacher_compute as common
from flint import arb,ctx

SOURCE='script/certified_teacher_corollary.py'
OUT=common.OUT
NOMINAL=common.ROOT/'artifacts/local_witness_feasibility_v1_20261008/NOMINAL_SUMMARY.json'
PROOF=common.ROOT/'docs/08_research_研究工作/LocalWitnessFeasibility_v1_20261008/GEOMETRY.md'
RISK=OUT/'TEACHER_RISK_CERTIFICATE.json'

def calculate():
    common.verify_freeze()
    candidate=common.read(NOMINAL);risk=common.read(RISK)
    previous=ctx.prec;ctx.prec=192
    try:
        a=[common.aq(Fraction.from_float(x)) for x in candidate['center']['theta'][8:]]
        r=common.aq('1/2000000');norm=sum((x*x for x in a),arb(0)).sqrt()
        qmin=(-3*r/2-(r*r+(norm+r)**2)/8).exp()/256
        Mr=24/qmin
        rho=common.from_record(risk['true_center_risk_direction_enclosure'])
        uniform_upper=rho.upper()+Mr*r
        assert qmin>0 and Mr>0
        return dict(schema='certified_teacher_posthoc_risk_ball_v1',
            provenance='post-execution deterministic consequence of the fixed radius and certified center derivative; not a newly tuned grid or statistical test',
            input_sha256={p.relative_to(common.ROOT).as_posix():common.sha(p) for p in (NOMINAL,PROOF,RISK,Path(__file__).resolve())},
            precision_bits=192,physical_radius_exact='1/2000000',parameter_dimension=24,
            candidate_probability_uniform_lower_bound=common.record(qmin),
            risk_Hessian_operator_uniform_upper_expression=common.record(Mr),
            center_risk_direction_upper_endpoint=common.record(rho.upper()),
            risk_direction_uniform_upper_expression=common.record(uniform_upper),
            risk_direction_uniform_upper_below_minus_point020=bool(uniform_upper < -common.aq('1/50')),
            risk_direction_strictly_negative_throughout_ball=bool(uniform_upper<0),
            proof='For every theta in the full physical ball, DR(theta)[d] <= upper(DR(center)[d]) + sup||H_R|| r; Qmin follows GEOMETRY G13-G15. Fixed d is a mathematical unit vector.',
            positive_KL_throughout_ball_follows_from_negative_direction=bool(uniform_upper<0),
            F8_stationary_point_in_ball_proved=False,true_F8_local_minimum_certified=False,
            new_integrals=0,new_grid_evaluations=0,new_random_draws=0)
    finally:ctx.prec=previous

if __name__=='__main__':
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run',action='store_true');mode.add_argument('--check',action='store_true');a=parser.parse_args()
    result=calculate();p=OUT/'POSTHOC_RISK_BALL.json'
    if a.run:common.write_new(p,result)
    elif common.read(p)!=result:raise ValueError('risk-ball scalar reconstruction mismatch')
    print(common.json.dumps(dict(passed=True,uniform_risk_direction_less_than_minus_point020=result['risk_direction_uniform_upper_below_minus_point020'],new_integrals=0,new_grid_evaluations=0),indent=2))
