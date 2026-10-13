"""Deterministic manufactured floating arithmetic; no scientific input/kernel."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import local_witness_feasibility_saved_check as check


def fixture():
    q=np.array([[.25,.25,.25,.25]]);a=np.array([.2,.3]);ii=np.arange(2)
    M=np.zeros((1,4,2,2));M[0,:,0,0]=[1e16,.3,.6,.9];M[0,:,1,1]=[.1,.2,.3,.4]
    M[0,:,0,1]=[.01,.02,.03,.04];M[0,:,1,0]=M[0,:,0,1]
    sigma=np.array([[[1e16,.5],[.6,.6],[.9,.7],[1.2,.8]]])
    combine=lambda x:np.einsum('tp,kp...->tk...',q,x,optimize=True)
    LP=-M.copy();LP[...,ii,ii]+=sigma
    LE=M*a[None,:,None]*a[None,None,:];LE[...,ii,ii]=0;LA=-LE.copy();LA[...,ii,ii]=LE.sum(axis=-1)
    AM=combine(M);P=-AM.copy();P[...,ii,ii]+=combine(sigma)
    E=AM*a[None,None,:,None]*a[None,None,None,:];E[...,ii,ii]=0;A=-E.copy();A[...,ii,ii]=E.sum(axis=-1)
    gradient=np.zeros((1,1,4));ward=P*a[None,None,:,None]*a[None,None,None,:]
    labelsH=np.zeros((1,4,4,4))
    return dict(targets=q,a=a,label_M=M,label_sigma_prime_average=sigma,label_Hperp=LP,label_Hangle=LA,
        Hperp=P,Hangle=A,edge_weights=E,label_loss=np.zeros((1,4)),loss=np.zeros((1,1)),
        label_gradient=np.zeros((1,4,4)),gradient=gradient,label_H_ba=labelsH,H_ba=combine(labelsH),
        label_B=labelsH.copy(),label_same_particle_term=labelsH.copy(),label_cross_particle_term=labelsH.copy(),
        directWard=ward,ward_residual=ward-A,bias_gradient=gradient[...,:2],radial_gradient=gradient[...,2:])


class ManufacturedTests(unittest.TestCase):
    def test_kernel_operation_order_and_preservation(self):
        result=fixture();snapshots={k:v.copy() for k,v in result.items()}
        with patch.object(check.frozen.kernel,'rankone_stationary',side_effect=AssertionError('no kernel')):
            diagnosis=check.aggregate_diagnosis(result)
        self.assertGreater(diagnosis['Hperp']['direct_label_aggregation_Linf'],0)
        self.assertTrue(diagnosis['Hperp']['kernel_operation_order_bitwise_equal'])
        self.assertTrue(diagnosis['Hangle']['kernel_operation_order_bitwise_equal'])
        for key,value in snapshots.items():self.assertTrue(np.array_equal(value,result[key]))

    def test_changed_matrix_rejected_without_tolerance(self):
        result=fixture();result['Hperp'][0,0,0,0]=np.nextafter(result['Hperp'][0,0,0,0],np.inf)
        with self.assertRaises(ValueError):check.aggregate_diagnosis(result)

    def test_changed_label_component_rejected(self):
        result=fixture();result['label_Hperp'][0,1,0,0]+=.001
        with self.assertRaises(ValueError):check.aggregate_diagnosis(result)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--write-receipt',action='store_true');args=parser.parse_args()
    outcome=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ManufacturedTests))
    if not outcome.wasSuccessful():raise SystemExit(1)
    if args.write_receipt:
        check.frozen.write_new(check.RECEIPT,dict(schema='local_witness_saved_check_manufactured_v1',created_utc=datetime.now(timezone.utc).isoformat(),
            status='passed',tests_run=outcome.testsRun,source_sha256={name:check.frozen.sha(check.frozen.ROOT/name) for name in check.SOURCE_NAMES},
            new_kernel_calls=0,new_integrals=0,new_random_draws=0,scientific_array_reads=0))


if __name__=='__main__':main()
