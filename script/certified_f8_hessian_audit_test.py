"""Nonexchangeable m4 manufacture, validation, caps and partial budget only."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time
import unittest
import certified_f8_hessian_audit as audit
import certified_f8_hessian_compute as producer
from flint import arb,ctx

ROOT=Path(__file__).resolve().parents[1]
RECEIPT=ROOT/"artifacts/certified_f8_hessian_v1_20261008/AUDIT_MANUFACTURED_CHECK.json"
RESULTS=[]


def manufactured_weights():
    return [Fraction((label+1)**2,1496) for label in range(16)]


def explicit_small_moments(c,q):
    # Separate literal enumeration, no audit/producer folding or W/M/S helper.
    selected=(0,15) if c==(0,0) else (3,5,10,12)
    labels=[(tuple(int(v) for v in f"{label:04b}"),q[label]) for label in selected]
    if c==(0,0):labels=[((0,0,0,0),q[0]+q[15])]
    W=sum(v for _,v in labels)
    M=[sum(v*y[i] for y,v in labels) for i in range(4)]
    S=[[sum(v*y[i]*y[j] for y,v in labels) for j in range(4)] for i in range(4)]
    return W,M,S


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.previous=ctx.prec;ctx.prec=192

    def tearDown(self):
        ctx.prec=self.previous

    def test_small_nonexchangeable_classes_overlap_producer(self):
        q=manufactured_weights();a=("1/3","2/5")
        cache=producer.precompute_inner(a,"1/4","2");results=[]
        for c in ((0,0),(1,1)):
            direct=audit.direct_weighted_class(a,c,q,"1/4","2","1/4","-2","2",K=8)
            W,M,S=explicit_small_moments(c,q)
            source=producer.finite_representative(cache,c,W,M,S,"1/4","-2","2",K=8)
            comparisons=audit.compare(direct,source)
            self.assertEqual(direct["resources"]["laplace_factors_evaluated"],289)
            self.assertEqual(direct["resources"]["matrix_entries_returned"],48)
            self.assertEqual(direct["folded_label_count"],1 if c==(0,0) else 4)
            results.append(dict(pair_counts=list(c),direct=direct,producer=source,comparisons=comparisons))
        RESULTS.append(dict(test="small_nonexchangeable_classes_overlap_producer",status="passed",
            m=4,weights_exact=[str(v) for v in q],matrix_entries_overlap=96,
            independent_laplace_factors=578,results=results,
            scientific_grid_evaluations=0))

    def test_count_only_folding_retains_self_labels(self):
        q=manufactured_weights()
        low,selected=audit.fold_class(q,(0,0))
        self.assertEqual(selected,[0,15])
        self.assertEqual(len(low),1);self.assertEqual(low[0]["original_labels"],[0,15])
        self.assertTrue((low[0]["q"]-audit.ball(q[0]+q[15])).contains(0))
        mid,selected=audit.fold_class(q,(1,1))
        self.assertEqual(selected,[3,5,10,12])
        self.assertEqual(len(mid),4)
        self.assertEqual([e["original_labels"] for e in mid],[[3],[5],[10],[12]])
        self.assertEqual(len({q[label] for label in selected}),4)
        # This only enumerates labels/weights, not an actual m8 grid.
        q8=[Fraction(label+1,32896) for label in range(256)]
        low8,s0=audit.fold_class(q8,(0,0,0,0))
        mid8,s1=audit.fold_class(q8,(1,1,1,1))
        self.assertEqual((len(s0),len(low8)),(2,1))
        self.assertEqual((len(s1),len(mid8)),(16,16))
        RESULTS.append(dict(test="count_only_folding_retains_self_labels",status="passed",
            m4_nonexchangeable_self_class_weights=[str(q[label]) for label in selected],
            m8_original_and_folded_counts=[[2,1],[16,16]],new_Laplace_factors=0))

    def test_independent_saturation_caps(self):
        caps=audit.saturation_caps()
        cases=[]
        for tq in ("127","128","129","1000"):
            t=audit.common.aq(tq)
            e,te,used=audit.exponential_pair(t,128,caps)
            actual=(-t).exp();actualte=t*actual
            self.assertTrue(e.overlaps(actual));self.assertTrue(te.overlaps(actualte))
            self.assertEqual(used,int(tq)>=128)
            cases.append(dict(t_exact=tq,used=used,e=audit.common.record(e),
                              te=audit.common.record(te),overlap=True))
        RESULTS.append(dict(test="independent_saturation_caps",status="passed",cases=cases,
                            outer_rounding_retained=True))

    def test_invalid_grids_precision_and_partial_budget(self):
        args=[("1/3","2/5"),(1,1),manufactured_weights(),"1/4","2","1/4","-2","2"]
        rejected=0
        for position,value in ((0,(.333,"2/5")),(1,(1.0,1)),(1,(True,1)),
                               (3,.25),(4,"1/3"),(5,0),(6,"1/3")):
            bad=args.copy();bad[position]=value
            with self.assertRaises(ValueError):audit.direct_weighted_class(*bad)
            rejected+=1
        for kwargs in (dict(K=8.0),dict(precision_bits=192.0),dict(precision_bits=128),
                       dict(saturation_threshold=127)):
            with self.assertRaises(ValueError):audit.direct_weighted_class(*args,**kwargs)
            rejected+=1
        with self.assertRaises(audit.AuditBudgetStop) as stop:
            audit.direct_weighted_class(*args,deadline=time.perf_counter()-1)
        partial=stop.exception.partial
        self.assertEqual(partial["status"],"budget_stopped")
        self.assertEqual(partial["resources"]["laplace_factors_evaluated"],0)
        self.assertEqual(partial["outer_grid"]["nodes_completed"],0)
        self.assertIn("d_coefficients",partial["partial_state"])
        self.assertIn("samebb",partial["partial_state"])
        with self.assertRaises(ValueError):
            audit.fold_class(manufactured_weights(),(2,2))
        RESULTS.append(dict(test="invalid_grids_precision_and_partial_budget",status="passed",
                            rejected_inputs=rejected+1,partial_zero_factors=True))

    def test_overlap_checker_rejects_injected_failure(self):
        row=dict(status="completed",pair_counts=[0],matrix_blocks={
            name:[[audit.common.record(arb(1))]] for name in audit.BLOCKS})
        saved=dict(status="completed",pair_counts=[0],matrix_blocks={
            name:[[audit.common.record(arb(2))]] for name in audit.BLOCKS})
        with self.assertRaises(ValueError):audit.compare(row,saved)
        RESULTS.append(dict(test="overlap_checker_rejects_injected_failure",status="passed",
                            injected_matrix_nonoverlap_rejected=True,new_Laplace_factors=0))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--write-receipt",action="store_true")
    args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AuditTests))
    if not result.wasSuccessful():raise SystemExit(1)
    if args.write_receipt:
        sources=[ROOT/name for name in (*audit.SOURCES,"script/certified_teacher_compute.py",
            "script/certified_f8_hessian_compute.py","script/certified_f8_hessian_moments.py",
            "script/certified_f8_gradient_compute.py","script/certified_f8_gradient_symmetry.py")]
        receipt=dict(schema="certified_f8_hessian_audit_manufactured_v1",
            created_utc=datetime.now(timezone.utc).isoformat(),status="passed",tests_run=result.testsRun,
            precision_bits=192,results=RESULTS,
            source_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            real_Hessian_evaluations=0,scientific_grid_evaluations=0,scientific_NPZ_reads=0,
            new_random_draws=0,new_optimizer_updates=0,acb_integral_calls=0,
            scientific_audit_frozen=False,unclosed=["two fixed full-grid scientific replays",
                "384 scientific matrix overlaps","no complete independent replay of all41"])
        RECEIPT.parent.mkdir(parents=True,exist_ok=True)
        with RECEIPT.open("x",encoding="utf-8",newline="\n") as handle:
            json.dump(receipt,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write("\n")
        print(RECEIPT.relative_to(ROOT).as_posix())


if __name__=="__main__":main()
