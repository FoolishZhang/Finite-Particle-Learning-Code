"""Manufactured deterministic tests; no study banks, NPZ, or integration."""
from __future__ import annotations

import argparse
import itertools
import json
import math

import numpy as np

import asymptotic_bridge_statistics as s

RECEIPT = s.ART / "STATISTICS_MANUFACTURED_CHECK.json"


def manufactured_reference():
    v, j = .27, 2.5
    return dict(v=v, J100=j, gapV=.05, m_H45=.02,
                w_by_n={str(n):j/math.sqrt(n*v) for n in s.N_VALUES})


def manufactured_scores(reference):
    t = np.arange(32, dtype=float)
    result = np.empty((len(t),4,5))
    for ni in range(4):
        g = .4 + .1*np.sin(.2*t+.1*ni)
        c = .005*np.cos(.4*t+.2*ni)
        result[:,ni,0] = .02+.008*np.sin(.3*t)*(1+.1*ni)
        result[:,ni,1] = -reference["m_H45"]*g-c
        result[:,ni,2] = g
        result[:,ni,3] = c
        result[:,ni,4] = result[:,ni,0]+.01*np.cos(.2*t)
    return result


def run_tests():
    passed = []

    def close(a, b, rtol=3e-12, atol=2e-15):
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)

    def rejects(fn):
        try:
            fn()
        except (ValueError, KeyError):
            return
        raise AssertionError("Expected invalid input rejection")

    reference = manufactured_reference()
    scores = manufactured_scores(reference)
    previous = scores.copy()
    result = s.analyse(scores, reference)
    assert np.array_equal(previous, scores)
    assert result["family_target_count"] == s.TARGET_COUNT == 26
    assert len(result["target_order"]) == len(set(result["target_order"])) == 26
    assert result["target_order"][:5] == ["n64:"+metric for metric in s.METRIC_NAMES]
    assert result["target_order"][20:] == [f"{kind}:n{s.N_VALUES[i+1]}_minus_n{s.N_VALUES[i]}" for kind in ("A","B") for i in range(3)]
    assert not result["formal_bank_count_matches"]
    assert result["config"]["formal_look_count"] == 1 and result["config"]["delta_per_two_sided_target"] == .01/26
    passed.append("pure_input_preservation_fixed_26_target_order_and_single_look")

    for ni, row in enumerate(result["per_n"]):
        for mi, metric in enumerate(s.METRIC_NAMES):
            raw = row["raw"][metric]
            close(raw["mean"], scores[:,ni,mi].mean())
            close(raw["unbiased_variance"], scores[:,ni,mi].var(ddof=1))
            logterm = math.log(4/(.01/26))
            half = math.sqrt(2*raw["unbiased_variance"]*logterm/len(scores)) + 7*raw["support_width"]*logterm/(3*(len(scores)-1))
            close(raw["statistical_halfwidth"], half)
            close(raw["halfwidth"], half+raw["numerical_allowance"]+raw["tail_allowance"])
            close(raw["interval"], [raw["mean"]-raw["halfwidth"], raw["mean"]+raw["halfwidth"]])
            assert raw["delta_per_side"] == .01/52
            assert not raw["numerical_error_certified"]
            assert raw["tail_allowance"] == (s.direct_gap_constants()["tail_allowance"] if mi == 4 else 0)
    close(result["covariance20"], np.cov(scores.reshape(len(scores),20),rowvar=False,ddof=1))
    passed.append("unbiased_variance_two_sided_log_all_20_raw_support_and_allowances")

    for ni, row in enumerate(result["per_n"]):
        w, h, gapv = row["w"], reference["m_H45"], reference["gapV"]
        formulas = dict(A=("path_gap",1/(w*gapv),0), B=("D12",-1/(w*h),0),
                        omega_over_w=("overlap",1/w,0), R=("D12",1/w,h),
                        O=("overlap",-h/w,h), J=("coupling",-1/w,0))
        for name,(metric,scale,shift) in formulas.items():
            derived, raw = row["derived"][name], row["raw"][metric]
            close(derived["mean"], scale*raw["mean"]+shift)
            close(derived["unbiased_variance"], scale*scale*raw["unbiased_variance"])
            close(derived["interval"], sorted(scale*v+shift for v in raw["interval"]))
            assert derived["extra_alpha"] == 0 and derived["reuses_target"] == f"n{row['n']}:{metric}"
        residual = np.column_stack((scores[:,ni,1]/w+h, h*(1-scores[:,ni,2]/w), -scores[:,ni,3]/w))
        close(row["residual_covariance"], np.cov(residual,rowvar=False,ddof=1))
        close(row["derived"]["R"]["mean"], row["derived"]["O"]["mean"]+row["derived"]["J"]["mean"])
        close(row["derived"]["B"]["mean"]-1, -row["derived"]["R"]["mean"]/h)
        cov = np.asarray(row["residual_covariance"])
        close(cov[0,0], cov[1,1]+cov[2,2]+2*cov[1,2])
        discrepancy = scores[:,ni,4]-scores[:,ni,0]
        close(row["direct_minus_path_diagnostic"]["paired_SE"], discrepancy.std(ddof=1)/math.sqrt(len(scores)))
        assert row["direct_minus_path_diagnostic"]["enforced_pass_threshold"] is None
    passed.append("affine_confidence_event_reuse_negative_scale_and_residual_covariance_identity")

    matrices = [scores.reshape(len(scores),20)]
    differences = []
    support = s.support_bounds(reference)
    for kind, output, mi in (("A",result["paired_gap"],0), ("B",result["paired_gradient"],1)):
        for ni, pair in enumerate(output):
            sp, sn = pair["previous_scale"], pair["next_scale"]
            values = sn*scores[:,ni+1,mi]-sp*scores[:,ni,mi]
            differences.append(values)
            summary = pair["summary"]
            close(summary["mean"], values.mean())
            close(summary["unbiased_variance"], values.var(ddof=1))
            close(pair["covariance_reconstructed_variance"], summary["unbiased_variance"])
            assert abs(pair["independent_variance_not_used"]-summary["unbiased_variance"]) > 1e-5
            original = support[s.METRIC_NAMES[mi]]
            lp,hp = sorted(sp*v for v in original)
            ln,hn = sorted(sn*v for v in original)
            close(summary["support"], [ln-hp,hn-lp])
            close(summary["numerical_allowance"], (abs(sp)+abs(sn))*1e-9)
    matrix = np.column_stack([matrices[0]]+differences)
    close(result["covariance26"], np.cov(matrix,rowvar=False,ddof=1))
    passed.append("six_next_minus_previous_bankwise_contrasts_supports_and_covariance_not_independence")

    constant = np.zeros((8,4,5))
    constant[:,:,0] = .1
    constant[:,:,2] = .2
    constant[:,:,3] = -.2*reference["m_H45"]
    constant[:,:,4] = .1
    degenerate = s.analyse(constant,reference)
    for row in degenerate["per_n"]:
        for raw in row["raw"].values():
            close(raw["unbiased_variance"],0.)
            assert raw["statistical_halfwidth"] > 0
    narrow = s.bounded_summary([2.,2.,2.],2.,2.,numerical_allowance=0.)
    assert narrow["statistical_halfwidth"] == 0
    passed.append("zero_observed_variance_retains_positive_global_range_penalty")

    cube = [sum(v[i]*v[j] for i,j in ((0,1),(0,2),(1,2))) for v in itertools.product((-1.,1.),repeat=3)]
    assert min(cube) == -1 and max(cube) == 3
    mixed = [a*b-a*c-b*c for a,b,c in itertools.product((0.,1.),repeat=3)]
    assert min(mixed) == -1 and max(mixed) == 1
    close(support["path_gap"],[-.66825,1.3365])
    close(support["D12"],[-.495,.495])
    close(support["coupling"],[-.5148,.4851])
    direct = s.direct_gap_constants()
    close(direct["support_width"],6*max(1-math.sqrt(.1),math.sqrt(2.8)-1)*8)
    close(direct["b_K"],(8/3)*math.log(8*100))
    close(direct["tail_allowance"],math.sqrt(3)*direct["A"]*math.sqrt(direct["b_K"]*2*100*3*math.exp(-32)/(math.sqrt(2*math.pi)*8)))
    passed.append("multiaffine_cube_and_label_specific_supports_old_K_general_tail")

    rejects(lambda:s.analyse(scores[:1],reference))
    rejects(lambda:s.analyse(scores.astype(complex),reference))
    rejects(lambda:s.analyse(scores[:,:,:4],reference))
    invalid = scores.copy(); invalid[0,0,0] = 10
    rejects(lambda:s.analyse(invalid,reference))
    invalid = scores.copy(); invalid[0,0,0] = math.nan
    rejects(lambda:s.analyse(invalid,reference))
    invalid = scores.copy(); invalid[0,0,3] += .01
    rejects(lambda:s.analyse(invalid,reference))
    rejects(lambda:s.analyse(scores,dict(reference,m_H45=0)))
    rejects(lambda:s.analyse(scores,dict(reference,m_H45=.1)))
    rejects(lambda:s.analyse(scores,dict(reference,w_by_n={str(n):1. for n in s.N_VALUES})))
    rejects(lambda:s.bounded_summary([0,1],0,1,delta=0))
    rejects(lambda:s.bounded_summary([0,1],0,1,numerical_allowance=-1))
    passed.append("explicit_shape_real_support_identity_and_reference_domain_rejections")

    return dict(schema="asymptotic_bridge_statistics_manufactured_check_v1", passed=True,
                test_groups=passed, test_group_count=len(passed), manufactured_banks=len(scores),
                manufactured_only=True, source_sha256=s.sha(s.__file__), test_sha256=s.sha(__file__),
                new_random_samples=0, new_reference_integrals=0, scientific_array_reads=0,
                illustrative_family_target_count=result["family_target_count"],
                max_paired_covariance_reconstruction_error=max(p["covariance_reconstruction_absolute_error"] for p in result["paired_gap"]+result["paired_gradient"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run",action="store_true")
    mode.add_argument("--check",action="store_true")
    args = parser.parse_args()
    result = run_tests()
    if args.run:
        s.save_new(RECEIPT,result)
    elif result != s.read(RECEIPT):
        raise ValueError("Manufactured check receipt changed")
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
