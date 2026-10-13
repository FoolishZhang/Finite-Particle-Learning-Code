"""G: fixed d3 Gaussian-logistic bridge, no optimizer or observations."""
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "2"
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback
import numpy as np
import scipy
from scipy.special import roots_hermitenorm, roots_legendre, ndtr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"artifacts/rc_quant/G"
DOC = ROOT/"docs/08_research_研究工作/RC定量界与多块查询_20261001"
CFG = dict(d=3, rho=-.45, c=.9, scales=[.8, 1.4, 4.],
           GH_orders=[64, 128, 256, 512], GL_order=512, GL_half_width=9.,
           pair_moment_gate=1e-10, identity_gate=1e-12,
           primary_moments="GH512", new_fits=0, random_observations=0,
           adaptive_orders=False, interval_certificate=False)
PAIRS = list(itertools.combinations(range(3), 2))


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, obj, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if replace else "x", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def verify(mapping):
    for name, digest in mapping.items():
        assert sha(ROOT/name) == digest, name


def freeze():
    assert not OUT.exists()
    OUT.mkdir(parents=True)
    snapshot = OUT/"source_inputs/HANDOFF.md"
    snapshot.parent.mkdir()
    snapshot.write_bytes((ROOT/"HANDOFF.md").read_bytes())
    write(OUT/"CONFIG.json", CFG)
    paths = [Path(__file__).resolve(), DOC/"PLAN_G.md", DOC/"GAUSSIAN_BRIDGE.md",
             OUT/"CONFIG.json", snapshot]
    write(OUT/"FREEZE.json", dict(frozen_utc=now(), before_selected_Gaussian_evaluation=True,
        files={p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
        python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
        platform=platform.platform(), threads=2))
    write(OUT/"STATE.json", dict(status="frozen", active_jobs=[]))
    print("G frozen", sha(OUT/"FREEZE.json"), flush=True)


def pair_integrals(nodes, weights):
    rho = CFG["rho"]
    independent = nodes[None, :]
    conditioned = rho*nodes[:, None]+math.sqrt(1-rho*rho)*independent
    values = []
    for i, j in PAIRS:
        left = np.tanh(CFG["scales"][i]*nodes/2)
        right = np.tanh(CFG["scales"][j]*conditioned/2)
        values.append(float((weights*left) @ right @ weights))
    return values


def moments():
    rows = []
    for order in CFG["GH_orders"]:
        began = time.perf_counter()
        nodes, weights = roots_hermitenorm(order)
        values = pair_integrals(nodes, weights/math.sqrt(2*math.pi))
        rows.append(dict(method="GH", order=order, moments=values,
                         elapsed_seconds=time.perf_counter()-began))
    began = time.perf_counter()
    x, weights = roots_legendre(CFG["GL_order"])
    radius = CFG["GL_half_width"]
    nodes = radius*x
    normal_weights = radius*weights*np.exp(-nodes*nodes/2)/math.sqrt(2*math.pi)
    values = pair_integrals(nodes, normal_weights)
    rows.append(dict(method="GL_truncated", order=CFG["GL_order"], half_width=radius,
        moments=values, elapsed_seconds=time.perf_counter()-began,
        analytic_domain_tail_upper=4*float(ndtr(-radius)),
        quadrature_remainder_certified=False))
    first = np.asarray(rows[2]["moments"])
    final = np.asarray(rows[3]["moments"])
    separate = np.asarray(rows[4]["moments"])
    errors = dict(GH512_minus_GH256=float(np.max(np.abs(final-first))),
                  GH512_minus_GL512=float(np.max(np.abs(final-separate))))
    qualified = max(errors.values()) <= CFG["pair_moment_gate"]
    return dict(pair_order=[list(p) for p in PAIRS], rows=rows,
                errors=errors, qualified=qualified,
                convergence_check_not_interval_certificate=True)


def kappa(x):
    return ((1+x)*math.log1p(x)+(1-x)*math.log1p(-x))/2


def probabilities(moment_values):
    bits = ((np.arange(8)[:, None] >> np.arange(3)) & 1)
    spin = 2*bits-1
    p = np.ones(8)
    for (i, j), value in zip(PAIRS, moment_values):
        p += value*spin[:, i]*spin[:, j]
    p /= 8
    assert np.min(p) > 0 and abs(p.sum()-1) <= CFG["identity_gate"]
    matrix = np.eye(3)
    for (i, j), value in zip(PAIRS, moment_values):
        matrix[i, j] = matrix[j, i] = value
    return bits, spin, p, matrix


def all_query_risk(p, q):
    risks = []
    for target in range(3):
        flip = np.arange(8) ^ (1 << target)
        pc = p/(p+p[flip])
        qc = q/(q+q[flip])
        risks.append(float(p @ np.log(pc/qc)))
    return risks


def projections(moment_values):
    bits, spin, p, m = probabilities(moment_values)
    assert m[1, 2] < m[0, 2] < m[0, 1] < 0
    rows, maximum_identity_error = [], 0.
    for a, b in PAIRS:
        j = next(i for i in range(3) if i not in (a, b))
        swapped = bits.copy()
        swapped[:, a], swapped[:, b] = bits[:, b], bits[:, a]
        permutation = swapped @ (1 << np.arange(3))
        q = (p+p[permutation])/2
        assert np.min(q) > 0 and abs(q.sum()-1) <= CFG["identity_gate"]
        delta = float((m[a, j]-m[b, j])/(1-m[a, b]))
        gamma = (1-m[a, b])*kappa(delta)/6
        joint = float(p @ np.log(p/q))
        per_target = all_query_risk(p, q)
        logq = np.log(q)
        A = float(np.mean(logq*spin[:, a]*spin[:, b]))
        B = float(np.mean(logq*spin[:, a]*spin[:, j]))
        energy = A*spin[:, a]*spin[:, b]+B*(spin[:, a]*spin[:, j]+spin[:, b]*spin[:, j])
        reconstruction = np.exp(energy-np.max(energy))
        reconstruction /= reconstruction.sum()
        count = bits.sum(1)
        differences = p-q
        moment_error = max(float(np.max(np.abs(differences @ spin))),
            float(np.max(np.abs(np.bincount(count, weights=differences, minlength=4)))),
            abs(float(differences @ (bits[:, a]*bits[:, b]))))
        identity_error = max(moment_error, float(np.max(np.abs(q-reconstruction))),
                             abs(per_target[j]/3-gamma))
        maximum_identity_error = max(maximum_identity_error, identity_error)
        assert gamma > 0 and identity_error <= CFG["identity_gate"]
        rows.append(dict(cherry=[a, b], outside_target=j, delta=delta,
            Gamma_lower=gamma, exact_I_projection_q=q.tolist(),
            joint_KL_at_I_projection=joint,
            mean_LOO_KL_at_I_projection=float(np.mean(per_target)),
            per_target_LOO_KL=per_target,
            outside_target_contribution_to_mean=per_target[j]/3,
            spin_pair_coefficients=dict(cherry=A, equal_cross=B),
            moment_matching_max_error=moment_error,
            identity_max_error=identity_error,
            actual_queries_are_not_query_optimized=True))
    return dict(p=p.tolist(), moments=moment_values,
        all_tree_query_lower_min=min(r["Gamma_lower"] for r in rows),
        projections=rows, maximum_identity_error=maximum_identity_error,
        strict_positivity_is_analytic_not_a_numerical_interval_certificate=True)


def run():
    verify(read(OUT/"FREEZE.json")["files"])
    assert read(OUT/"STATE.json")["status"] == "frozen"
    write(OUT/"START.json", dict(first_selected_Gaussian_evaluation_utc=now(),
                                freeze_sha256=sha(OUT/"FREEZE.json")))
    numerical = moments()
    write(OUT/"MOMENTS.json", numerical)
    outputs = [OUT/"START.json", OUT/"MOMENTS.json"]
    if numerical["qualified"]:
        result = projections(numerical["rows"][3]["moments"])
        write(OUT/"RESULTS.json", result)
        outputs.append(OUT/"RESULTS.json")
        status = "complete"
    else:
        status = "stopped_numerical_gate"
    decision = dict(qualified=numerical["qualified"], status=status,
        moment_errors=numerical["errors"], numerical_gate=CFG["pair_moment_gate"],
        interval_certificate=False, new_fits=0, new_observations=0,
        additional_orders=0, completed_utc=now())
    write(OUT/"DECISION.json", decision)
    outputs.append(OUT/"DECISION.json")
    write(OUT/"RESULTS_FREEZE.json", dict(source_freeze_sha256=sha(OUT/"FREEZE.json"),
        files={p.relative_to(ROOT).as_posix(): sha(p) for p in outputs}))
    write(OUT/"STATE.json", dict(status=status, active_jobs=[]), replace=True)
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


def check():
    verify(read(OUT/"FREEZE.json")["files"])
    seal = read(OUT/"RESULTS_FREEZE.json")
    assert seal["source_freeze_sha256"] == sha(OUT/"FREEZE.json")
    verify(seal["files"])
    numerical = read(OUT/"MOMENTS.json")
    gh256, gh512, gl512 = [np.asarray(numerical["rows"][i]["moments"]) for i in (2, 3, 4)]
    errors = dict(GH512_minus_GH256=float(np.max(abs(gh512-gh256))),
                  GH512_minus_GL512=float(np.max(abs(gh512-gl512))))
    assert errors == numerical["errors"]
    assert numerical["qualified"] == (max(errors.values()) <= CFG["pair_moment_gate"])
    if numerical["qualified"]:
        actual = projections(gh512.tolist())
        assert actual == read(OUT/"RESULTS.json")
    decision = read(OUT/"DECISION.json")
    assert decision["qualified"] == numerical["qualified"]
    print(json.dumps(dict(passed=True, qualified=decision["qualified"],
                         quadrature_not_rerun=True, new_fits=0, new_observations=0)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["freeze", "run", "check"])
    args = parser.parse_args()
    try:
        {"freeze": freeze, "run": run, "check": check}[args.phase]()
    except Exception as error:
        if args.phase == "run" and (OUT/"FREEZE.json").exists() and not (OUT/"RESULTS_FREEZE.json").exists():
            write(OUT/f"FAILURE_{time.time_ns()}.json",
                  dict(error=repr(error), traceback=traceback.format_exc(), utc=now()))
            write(OUT/"STATE.json", dict(status="stopped_failure", active_jobs=[]), replace=True)
        raise
