"""Manufactured m=2/K=1 tests only; never evaluate the scientific P center."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import local_witness_feasibility_compute as compute


class ManufacturedTests(unittest.TestCase):
    def test_center_float64_pair_and_source_preservation(self):
        old=np.arange(16,dtype=np.float64)/13
        snapshot=old.copy();center=compute.symmetric_center(old)
        self.assertTrue(np.array_equal(old,snapshot))
        self.assertTrue(np.array_equal(center[:8],np.zeros(8)))
        self.assertTrue(np.array_equal(center[8:],center[8:][::-1]))
        for j in range(4):self.assertEqual(center[8+j],np.float64((old[8+j]+old[15-j])/2))
        for row in compute.dyadic_records(center):
            self.assertEqual(float.fromhex(row["hex"]),row["value"])
            self.assertEqual(float(int(row["numerator"])/int(row["denominator"])),row["value"])

    def test_manufactured_K1_shapes_snapshot_and_raw_input(self):
        b=np.zeros(2);a=np.array([np.sin(.2),np.sin(-.3)]);q=np.array([.2,.3,.1,.4])
        copies=[x.copy() for x in (b,a,q)]
        result=compute.kernel.rankone_stationary(b,a,{"fixture":q},K_values=(1,),
            normal_rule="legendre",normal_order=8,normal_limit=4.,logu_min=-2.,logu_max=2.,logu_step=.5,
            likelihood_scale=1.,compute_hessian=True,return_coefficients=False)
        self.assertEqual(result["gradient"].shape,(1,1,4));self.assertEqual(result["H_ba"].shape,(1,1,4,4))
        self.assertEqual(result["Hperp"].shape,(1,1,2,2));self.assertIsNone(result["d_coefficients"])
        self.assertEqual(result["config"]["likelihood_scale_source"],"supplied_positive_scale")
        self.assertTrue(np.array_equal(result["log_likelihood_scale"],np.zeros(4)))
        for x,snapshot in zip((b,a,q),copies):self.assertTrue(np.array_equal(x,snapshot))
        result["gradient"][0,0,0]=123
        self.assertNotEqual(result["bias_gradient"][0,0,0],123)
        result["targets"][0,0]=999;self.assertEqual(q[0],.2)

    def test_fixed_slice_and_plus_minus_retention(self):
        a=np.array([.5,.6,.7,.8,.8,.7,.6,.5]);V,Q=compute.fixed_slice(a);plus,minus=compute.reversal_basis(8)
        np.testing.assert_allclose(a@V,0,atol=1e-15)
        np.testing.assert_allclose(Q.T@Q,np.eye(7),atol=1e-15)
        np.testing.assert_allclose(plus.T@minus,0,atol=1e-15)
        self.assertEqual(plus.shape,(8,4));self.assertEqual(minus.shape,(8,4))
        self.assertFalse(np.allclose(V.T@V,np.eye(7)))

    def test_saved_rebuild_readout_without_kernel(self):
        b=np.zeros(2);a=np.array([.2,.2]);q=np.array([.2,.3,.3,.2]);theta=np.r_[b,a]
        result=compute.kernel.rankone_stationary(b,a,{"fixture":q},K_values=(1,),
            normal_rule="legendre",normal_order=8,normal_limit=4.,logu_min=-2.,logu_max=2.,logu_step=.5,
            likelihood_scale=1.,compute_hessian=True,return_coefficients=False)
        compute.previous.project_result(result);row=compute.readout(result,theta)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"fixture.npz";np.savez_compressed(path,**compute.snapshot_arrays(result,theta))
            with patch.object(compute.kernel,"rankone_stationary",side_effect=AssertionError("no replay kernel")):
                rebuilt=compute.rebuild_saved(path,row)
                self.assertEqual(compute.readout(rebuilt,theta),row)

    def test_existing_start_refused_before_kernel(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);(folder/"RUN_START.json").write_text("{}",encoding="utf-8")
            with patch.object(compute,"OUT",folder),patch.object(compute,"verify_freeze",return_value=({},{})),patch.object(compute.kernel,"rankone_stationary",side_effect=AssertionError("unexpected kernel")):
                with self.assertRaises(FileExistsError):compute.run()

    def test_nominal_direction_saved_dyadic_definition(self):
        ga=np.array([.1,.2,.3,.4,.5,.6,.7,.8]);g=np.r_[np.zeros(8),ga]
        result=dict(a=np.ones(8),K_values=(8,"inf"),gradient=np.stack([np.zeros(16),g])[None])
        row=compute.risk_direction(result)
        for j in range(4):self.assertEqual(row["h"][j],np.float64((ga[j]+ga[7-j])/2))
        self.assertEqual(row["h"],row["h"][::-1])
        for value,dyadic in zip(row["h"],row["h_dyadic"]):self.assertEqual(float.fromhex(dyadic["hex"]),value)
        self.assertLess(row["dot_nominal"],0)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--record",action="store_true");args=parser.parse_args()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ManufacturedTests)
    outcome=unittest.TextTestRunner(verbosity=2).run(suite)
    if args.record:
        record=dict(schema="local_witness_compute_manufactured_check_v1",tests_run=outcome.testsRun,
            passed=outcome.wasSuccessful(),failures=len(outcome.failures),errors=len(outcome.errors),
            manufactured_kernel_calls=2,manufactured_dimension=2,manufactured_K=1,
            real_P_kernel_calls=0,scientific_NPZ_reads=0,new_optimizer_updates=0,new_random_draws=0,
            check_reconstruction_kernel_calls=0,
            sources={compute.rel(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(compute.__file__),Path(__file__))})
        compute.write_new(compute.OUT/"COMPUTE_MANUFACTURED_CHECK.json",record)
    if not outcome.wasSuccessful():raise SystemExit(1)


if __name__=="__main__":main()
