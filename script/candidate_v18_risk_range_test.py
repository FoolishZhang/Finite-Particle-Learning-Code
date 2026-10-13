"""Manufactured Bernoulli probability boxes and Taylor transfers, no real data."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import unittest
import candidate_v18_risk_range as risk
from flint import arb,ctx

ROOT=Path(__file__).resolve().parents[1]
RESULTS=[]


class ManufacturedTests(unittest.TestCase):
    def test_Bernoulli_Fisher_gradient_and_root_range(self):
        out=risk.risk_range(("3/4","1/4"),("1/2","1/2"),"1/1000","1/4")
        previous=ctx.prec;ctx.prec=192
        try:
            center=arb(3)/4*(arb(3)/2).log()+arb(1)/4*(arb(1)/2).log()
            self.assertTrue(risk.from_record(out["center_KL_enclosure"]).overlaps(center))
            self.assertEqual(out["chi_square_upper_exact_dyadic"],
                             {"mantissa":"1","exponent_of_two":-2})
            self.assertEqual(out["full_physical_gradient_norm_upper_enclosure"]["upper_exact_dyadic"],
                             {"mantissa":"1","exponent_of_two":-2})
            root=risk.from_record(out["root_KL_enclosure"])
            for t in ("-1/1000","-1/2000","0","1/2000","1/1000"):
                theta=risk.aq(t)
                # KL(q||Bernoulli(sigmoid(theta))), exact analytic expression.
                value=(arb(3)/4*(arb(3)/4).log()+arb(1)/4*(arb(1)/4).log()
                       +(1+theta.exp()).log()-arb(3)/4*theta)
                self.assertTrue(root.contains(value))
            RESULTS.append(dict(test="Bernoulli_Fisher_gradient_and_root_range",status="passed",
                exact_bias_gradient="-1/4",complete_and_observed_Fisher_at_center="1/4",
                shifted_values_checked=5,result=out,real_probability_vectors_used=False))
        finally:ctx.prec=previous

    def test_equal_distributions_zero_chi_square(self):
        out=risk.risk_range(("1/2","1/2"),("1/2","1/2"),"1/1000","1/4")
        self.assertEqual(out["chi_square_upper_exact_dyadic"],{"mantissa":"0","exponent_of_two":0})
        self.assertEqual(out["full_physical_gradient_norm_upper_enclosure"]["upper_exact_dyadic"],
                         {"mantissa":"0","exponent_of_two":0})
        RESULTS.append(dict(test="equal_distributions_zero_chi_square",status="passed",result=out))

    def test_uncertain_normalized_probability_corners(self):
        previous=ctx.prec;ctx.prec=192
        try:
            eps=arb(1)/10**7
            q=[arb(3)/4+arb(0,eps.upper()),arb(1)/4+arb(0,eps.upper())]
            Q=[arb(1)/2+arb(0,eps.upper()),arb(1)/2+arb(0,eps.upper())]
            out=risk.risk_range(q,Q,"0","0")
            KL=risk.from_record(out["center_KL_enclosure"])
            chi=risk.from_record(out["chi_square_enclosure"])
            for sq in (-1,1):
                for sQ in (-1,1):
                    qq=[arb(3)/4+sq*eps,arb(1)/4-sq*eps]
                    QQ=[arb(1)/2+sQ*eps,arb(1)/2-sQ*eps]
                    value=sum((qi*(qi/Qi).log() for qi,Qi in zip(qq,QQ)),arb(0))
                    cs=sum(((qi-Qi)**2/Qi for qi,Qi in zip(qq,QQ)),arb(0))
                    self.assertTrue(KL.contains(value));self.assertTrue(chi.contains(cs))
            self.assertFalse(out["probability_balls_renormalized"])
            RESULTS.append(dict(test="uncertain_normalized_probability_corners",status="passed",
                normalized_corners_checked=4,result=out))
        finally:ctx.prec=previous

    def test_validation_and_precision_restoration(self):
        good=(("3/4","1/4"),("1/2","1/2"),"1/1000","1/4")
        before=ctx.prec;count=0
        for q,Q,r,M in ((("1","0"),good[1],good[2],good[3]),
                         (("1/4","1/4"),good[1],good[2],good[3]),
                         (good[0],("1/2",),good[2],good[3]),
                         (good[0],good[1],"-1",good[3]),
                         (good[0],good[1],good[2],"-1"),
                         (good[0],good[1],.001,good[3])):
            with self.assertRaises(ValueError):risk.risk_range(q,Q,r,M)
            count+=1
        for kw in (dict(precision_bits=192.0),dict(precision_bits=True),dict(exact_unit_mass=False)):
            with self.assertRaises(ValueError):risk.risk_range(*good,**kw)
            count+=1
        self.assertEqual(ctx.prec,before)
        RESULTS.append(dict(test="validation_and_precision_restoration",status="passed",rejected_inputs=count))

    def test_exact_endpoint_roundtrip(self):
        previous=ctx.prec;ctx.prec=192
        try:
            cases=[]
            for value in (arb(1)/3,arb(0),arb("0.000000001804339"),arb("8784.37")):
                saved=risk.record(value);back=risk.from_record(saved)
                self.assertTrue(back.contains(value))
                cases.append(saved)
            RESULTS.append(dict(test="exact_endpoint_roundtrip",status="passed",cases=cases))
        finally:ctx.prec=previous


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--write-receipt",action="store_true")
    args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ManufacturedTests))
    if not result.wasSuccessful():raise SystemExit(1)
    if args.write_receipt:
        sources=[ROOT/name for name in risk.CODE_DOC_PATHS]
        out=dict(schema="candidate_v18_risk_range_manufactured_v1",
            created_utc=datetime.now(timezone.utc).isoformat(),status="passed",tests_run=result.testsRun,
            precision_bits=192,results=RESULTS,
            source_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            real_probability_vectors_used=False,new_grid_evaluations=0,new_integrals=0,
            new_PMF_evaluations=0,new_random_draws=0,new_optimizer_updates=0,scientific_NPZ_reads=0,
            unclosed=["frozen real saved-probability synthesis","true center/root KL intervals"])
        risk.write_new(risk.MANUFACTURED,out)
        print(risk.MANUFACTURED.relative_to(ROOT).as_posix())


if __name__=="__main__":main()
