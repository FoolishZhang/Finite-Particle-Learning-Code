"""Wrapper manufacture only: synthetic center planning and m2/K1 Arb calls."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import local_witness_feasibility_tail_run as wrapper


def synthetic_inputs():
    theta=[0.]*8+[.125]*8
    records=[]
    for x in theta:
        numerator,denominator=x.as_integer_ratio()
        records.append(dict(value=x,hex=x.hex(),numerator=str(numerator),denominator=str(denominator)))
    return dict(schema="local_witness_feasibility_inputs_v1",dimension=8,K_values=[8,"inf"],
        rules=wrapper.FIXED_RULES,likelihood_scale=1.,physical_radius="1/2000000",physical_radius_float=5e-7,
        tangent_T=4,teacher_truncation_L=10,center_theta=theta,center_theta_dyadic=records)


class ManufacturedTests(unittest.TestCase):
    def test_synthetic_exact_box_and_input_preservation(self):
        inputs=synthetic_inputs();original=wrapper.json.dumps(inputs,sort_keys=True)
        with patch.object(wrapper.bounds,"bound_box",side_effect=AssertionError("plan must not evaluate")):
            plan=wrapper.argument_plan(inputs)
        self.assertEqual(wrapper.json.dumps(inputs,sort_keys=True),original)
        self.assertEqual(plan["parameter_box_exact"]["center_a_l1"],"1")
        self.assertEqual(Fraction(plan["parameter_box_exact"]["b_l1"]),Fraction(3,2000000))
        self.assertEqual(Fraction(plan["parameter_box_exact"]["a_l1"]),1+Fraction(3,2000000))
        self.assertEqual(plan["arguments_by_rule"]["D1"]["vmax"],"64")

    def test_drift_and_nonfinite_rejected(self):
        for key,value in (("likelihood_scale",2),("tangent_T",5),("physical_radius","1/1000000")):
            inputs=synthetic_inputs();inputs[key]=value
            with self.assertRaises(ValueError):wrapper.argument_plan(inputs)
        with self.assertRaises(ValueError):wrapper.exact_binary64(float("inf"))
        with self.assertRaises(ValueError):wrapper.exact_binary64(True)

    def test_small_manufactured_Arb_record_roundtrip(self):
        arguments=dict(m=2,K=1,b_l1="0",a_l1="0",normalL="7",teacherL="7",vmin="-12",vmax="6",T="4",precision_bits=192,targets=wrapper.TARGETS.copy())
        plan=dict(parameter_box_exact={"manufactured":True},targets_exact=wrapper.TARGETS.copy(),arguments_by_rule={"C1":arguments,"D1":arguments.copy()})
        result=wrapper.evaluate_plan(plan)
        self.assertEqual(result["resources"]["analytic_bound_box_calls"],2)
        self.assertEqual(len(result["total_truncation_comparisons"]),6)
        self.assertEqual(result["resources"]["new_kernel_calls"],0)
        self.assertFalse(result["finite_interval_quadrature_certified"])
        self.assertEqual(wrapper.json.loads(wrapper.json.dumps(result)),result)
        for name,row in result["rules"].items():
            self.assertEqual(row["bounds_readout"]["m"],2)
            self.assertEqual(row["bounds_readout"]["K"],1)
            self.assertEqual(row["bounds_readout"]["precision_bits"],192)
            self.assertEqual(row["bounds_readout"]["likelihood_scale_exact"],"1")

    def test_existing_output_refused_before_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"existing.json";path.write_text("{}",encoding="utf-8")
            with patch.object(wrapper,"OUTPUT",path),patch.object(wrapper,"verify_freeze",side_effect=AssertionError("no reads")):
                with self.assertRaises(FileExistsError):wrapper.run()


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--write-receipt",action="store_true");args=parser.parse_args()
    outcome=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ManufacturedTests))
    if not outcome.wasSuccessful():raise SystemExit(1)
    if args.write_receipt:
        sources=[Path(wrapper.__file__).resolve(),Path(__file__).resolve(),Path(wrapper.bounds.__file__).resolve()]
        receipt=dict(schema="local_witness_tail_wrapper_manufactured_check_v1",created_utc=datetime.now(timezone.utc).isoformat(),
            status="passed",tests_run=outcome.testsRun,source_sha256={p.relative_to(wrapper.ROOT).as_posix():wrapper.sha(p) for p in sources},
            manufactured_bound_box_calls=2,manufactured_bound_m=2,manufactured_bound_K=1,
            synthetic_center_planning_only=True,real_center_bound_box_calls=0,real_INPUTS_reads=0,
            new_kernel_calls=0,new_integrals=0,new_random_draws=0,scientific_array_reads=0,
            finite_interval_quadrature_certified=False,strict_stationary_existence_certified=False)
        wrapper.write_new(wrapper.OUT/"TAIL_WRAPPER_MANUFACTURED_CHECK.json",receipt)


if __name__=="__main__":main()
