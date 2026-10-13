"""Manufactured arithmetic checks; no scientific bank or scientific RNG read."""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import numpy as np
import torch
import asymptotic_bridge_audit as audit
import asymptotic_bridge_core as producer
import asymptotic_bridge_statistics as statistics_producer


def manufactured() -> dict:
    results = {}
    m_h = .017
    for k in (2, 5, 100):
        z = 2.8*np.sin(np.arange(3*k*3, dtype=float).reshape(3, k, 3)*.173)
        s = np.empty((3, 4, k))
        for arm, n in enumerate(audit.NS):
            s[:, arm] = -n*.8+(arm+1)*4*np.cos(np.arange(3*k).reshape(3, k)*.37)
        u = np.array([0., .45, .9])
        reconstructed, mask, details = audit.explicit_score(z, s, u, m_h)
        original, original_mask = producer.score_block(z, s, u, m_h)
        difference = audit.assert_close(reconstructed, original, f"score_K{k}", 2e-11)
        assert np.array_equal(mask, original_mask)
        assert details["D_identity_maxabs"] < 2e-15
        assert details["maximum_weight_sum_error"] < 2e-15
        assert details["minimum_gamma"] >= 0 and details["maximum_gamma_sum"] <= 1+2e-15
        results[f"K{k}_scores"] = dict(maxabs=difference, **details)
        for rho in (0., .45, .9):
            root = audit.independent_root(rho)
            C = np.full((3, 3), rho)
            np.fill_diagonal(C, 1.)
            audit.assert_close(root@root.T, C, "root_covariance", 2e-15)
            audit.assert_close(audit.transform(z, rho), z@root, "projection_transform", 2e-15)
        # Independent autograd mixed-logit Hessian, without differentiating any
        # producer routine. Summing same-particle (0,1) entries is physical Price.
        x = torch.tensor(audit.transform(z[:1], .45)[0], dtype=torch.float64, requires_grad=True)
        nuisance = torch.tensor(s[0, 1]-s[0, 1].max(), dtype=torch.float64)
        maximum_mixed_error = 0.
        labels = audit.bits() if k < 100 else audit.bits()[[5]]
        for label in labels:
            y = torch.tensor(label, dtype=torch.float64)
            def loss(v):
                logp = y*torch.nn.functional.logsigmoid(v)+(1-y)*torch.nn.functional.logsigmoid(-v)
                return -torch.logsumexp(nuisance+logp.sum(dim=1), dim=0)
            hessian = torch.autograd.functional.hessian(loss, x, vectorize=True)
            mixed = float(sum(hessian[p, 0, p, 1] for p in range(k)))
            _, gamma, residual, _ = audit.label_kernel(x.detach().numpy()[None], nuisance.numpy()[None], label)
            price = -float(np.sum(gamma*residual[:, :, 0]*residual[:, :, 1]))
            maximum_mixed_error = max(maximum_mixed_error, abs(mixed-price))
            assert abs(mixed-price) < 2e-14
        results[f"K{k}_autograd_Price"] = dict(labels=len(labels), maxabs=maximum_mixed_error)
    # A dominant particle's gamma must retain its tiny, nonzero complement.
    extreme_x = np.zeros((1, 3, 3))
    _, gamma, _, _ = audit.label_kernel(extreme_x, np.array([[0., -40., -80.]]), np.ones(3))
    assert gamma[0, 0] > 0
    assert abs(gamma[0, 0]/math.exp(-40)-1) < 1e-14
    results["dominant_complement"] = float(gamma[0, 0])
    # Evaluate every possible sign-specific residual cube vertex. Gamma<=1
    # yields conservative path/D/centered support via these extrema.
    all_marks, all_h = [], []
    for y in audit.bits():
        endpoints = [(-1., 0.) if b == 0 else (0., 1.) for b in y]
        for r in itertools.product(*endpoints):
            all_marks.append(r[0]*r[1]+r[0]*r[2]+r[1]*r[2])
            all_h.append(r[0]*r[1])
    assert min(all_marks) == -1 and max(all_marks) == 3
    assert min(all_h) == -1 and max(all_h) == 1
    results["residual_vertex_extrema"] = dict(pair=[min(all_h), max(all_h)], sum3=[min(all_marks), max(all_marks)])
    label_extrema = []
    for y in audit.bits():
        endpoints = [(-1., 0.) if b == 0 else (0., 1.) for b in y]
        marks = [r[0]*r[1]+r[0]*r[2]+r[1]*r[2] for r in itertools.product(*endpoints)]
        pairs = [r[0]*r[1] for r in itertools.product(*endpoints)]
        label_extrema.append((min(marks),max(marks),min(pairs),max(pairs)))
    average_extrema = np.mean(label_extrema,axis=0)
    audit.assert_close(average_extrema,[-.75,1.5,-.5,.5],"label-averaged support vertices")
    bounds = audit.supports(m_h)
    finite_k = 1-1/100
    audit.assert_close(bounds["path_gap"], .9*finite_k*average_extrema[:2], "path support")
    audit.assert_close(bounds["D12"], finite_k*average_extrema[2:], "D support")
    centered_extrema = []
    for y in audit.bits():
        endpoints = [(-1.,0.) if b == 0 else (0.,1.) for b in y]
        centered = [g*(r[0]*r[1]-m_h) for r in itertools.product(*endpoints) for g in (0.,finite_k)]
        centered_extrema.append((min(centered),max(centered)))
    audit.assert_close(bounds["coupling"],np.mean(centered_extrema,axis=0),"centered support")
    norm = float(np.max(np.abs(np.linalg.eigvalsh(audit.independent_root(.9)-np.eye(3)))))
    audit.assert_close(audit.direct_constants()["M"],3*norm*8,"direct Lipschitz support")
    results["finite_K_label_averaged_support"] = dict(average_vertex_extrema=average_extrema.tolist(),gamma_upper=finite_k,
                                                       centered_bounds=list(bounds["coupling"]),direct_M=audit.direct_constants()["M"])
    z = np.zeros((2, 5, 3)); z[1, 0, 0] = 8.00001
    s = np.zeros((2, 4, 5))
    clipped, mask, _ = audit.explicit_score(z, s, np.array([.2, .4]), m_h)
    assert np.array_equal(mask, [False, True])
    assert np.all(clipped[1, :, 4] == 0)
    assert np.any(clipped[1, :, :4] != 0)
    results["clip_only_direct_gap"] = True
    x = np.array([-.9, -.2, .4, .8, .1, -.3])
    stats = audit.eb_summary(x, (-1., 1.), .01/26, 2e-11)
    audit.assert_close(stats["mean"], np.mean(x), "mean")
    audit.assert_close(stats["unbiased_variance"], np.var(x, ddof=1), "variance")
    expected_radius = math.sqrt(2*np.var(x, ddof=1)*math.log(10400)/len(x))+7*2*math.log(10400)/(3*(len(x)-1))
    audit.assert_close(stats["statistical_halfwidth"], expected_radius, "EB formula")
    affine = audit.affine(stats, -3., .7)
    audit.assert_close(affine["interval"], sorted(.7-3*np.asarray(stats["interval"])), "negative affine endpoints")
    audit.assert_close(affine["unbiased_variance"], 9*np.var(x, ddof=1), "affine variance")
    columns = np.column_stack([x, x*.8+np.sin(np.arange(len(x)))])
    cov = audit.covariance(columns)
    audit.assert_close(cov, np.cov(columns, rowvar=False, ddof=1), "joint covariance")
    paired = columns[:, 1]-columns[:, 0]
    _, var_paired = audit.scalar_moments(paired)
    audit.assert_close(var_paired, cov[0, 0]+cov[1, 1]-2*cov[0, 1], "paired covariance cancellation")
    assert abs(var_paired-(cov[0, 0]+cov[1, 1])) > .01
    results["statistics"] = dict(EB=True, affine=True, paired_variance=var_paired, covariance_maxabs=float(np.max(np.abs(cov-np.cov(columns,rowvar=False,ddof=1)))))
    # End-to-end manufactured statistics serialization check. Independently
    # reconstruct every interval, affine transform, and joint covariance.
    reference = dict(v=.27,J100=1.17,gapV=.037,m_H45=m_h,
                     w_by_n={str(n):1.17/math.sqrt(n*.27) for n in audit.NS})
    panel = np.empty((32,4,5))
    t = np.arange(32,dtype=float)
    for arm in range(4):
        panel[:,arm,0] = .11*np.sin(t*.3+arm*.2)
        panel[:,arm,2] = .45+.24*np.cos(t*.21+arm*.1)
        panel[:,arm,3] = .021*np.sin(t*.35+arm*.3)
        panel[:,arm,1] = -m_h*panel[:,arm,2]-panel[:,arm,3]
        panel[:,arm,4] = .6*np.sin(t*.3+arm*.4)
    saved = statistics_producer.analyse(panel,reference)
    results["full_statistics_schema"] = audit.check_statistics(panel,reference,saved,expected_banks=32)
    damaged = json.loads(json.dumps(saved))
    damaged["paired_gap"][0]["summary"]["unbiased_variance"] += .001
    try:
        audit.check_statistics(panel,reference,damaged,expected_banks=32)
    except AssertionError:
        results["damaged_paired_variance_rejected"] = True
    else:
        raise AssertionError("modified statistic was not rejected")
    damaged = json.loads(json.dumps(saved))
    damaged["covariance26"][0][25] += .001
    try:
        audit.check_statistics(panel,reference,damaged,expected_banks=32)
    except AssertionError:
        results["damaged_covariance_rejected"] = True
    else:
        raise AssertionError("modified covariance was not rejected")
    assert len(set(audit.REPLAY_BLOCKS)) == 16 and audit.REPLAY_BLOCKS[0] == 0 and audit.REPLAY_BLOCKS[-1] == 2047
    results["predetermined_replay_indices"] = list(audit.REPLAY_BLOCKS)
    return dict(schema_version="1.0", passed=True, checks=results,
                scientific_npz_read=0, scientific_rng_calls=0, new_independent_banks=0,
                manufactured_inputs="fixed sin/cos/vertices; no random fixture", new_reference_integrals=0,
                scope="native arithmetic implementation checks, not interval arithmetic or external scientific review",
                sources={p.name:audit.digest(p) for p in (Path(audit.__file__), Path(__file__), Path(producer.__file__),Path(statistics_producer.__file__))})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=audit.DEFAULT_ART/"AUDIT_MANUFACTURED_CHECK.json")
    args = parser.parse_args()
    receipt = manufactured()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({"passed":True, "output":str(args.output), "check_groups":len(receipt["checks"])}))
