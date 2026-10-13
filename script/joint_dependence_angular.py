"""Fixed-design weak-loading angular study; no training and no random draws."""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import platform
import time
import traceback
import numpy as np
import scipy
from scipy.special import expit, roots_hermitenorm, logsumexp
import joint_dependence_kernel as core

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/joint_dependence_v1_20261007"
DOC = ROOT / "docs/08_research_研究工作/JointDependence_v1_20261007"
PHI = np.arange(-7, 8, 2, dtype=float) / 10
CONFIG = dict(epsilons=[.125, .25, .5, .8], t_values=[0., .05],
              orders=[8, 12, 16, 20], pmf_orders=[40, 60, 80],
              rotated_order=72, rotation=float(np.pi/7), chunk_banks=512,
              mass_tolerance=2e-12, pmf_tolerance=2e-7,
              risk_tolerance=2e-6, angular_tolerance=1e-7,
              finite_difference_step=1e-5, finite_difference_tolerance=1e-7)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def sources():
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in
            [Path(__file__), ROOT/"script/joint_dependence_kernel.py",
             DOC/"DESIGN.md", DOC/"PROOF.md", OUT/"HANDOFF_INPUT.md"]}


def geometry(epsilon, t=0.):
    angles = (1-t)*PHI
    W = epsilon*np.stack((np.cos(angles), np.sin(angles)), axis=1)
    dW = epsilon*np.stack((-np.sin(angles), np.cos(angles)), axis=1)*(-PHI[:, None])
    return W, dW


def quadrature(epsilon, t, order, teacher, keep=False):
    """Tensor Gaussian quadrature in the four coordinates of a two-particle bank."""
    x, w = roots_hermitenorm(order)
    w = w/np.sqrt(2*np.pi)
    indices = np.array(list(itertools.product(range(order), repeat=4)), dtype=np.int16)
    weights = np.prod(w[indices], axis=1)
    W, dW = geometry(epsilon, t)
    Y = core.patterns(8)
    values = np.empty((len(indices), 2))
    for start in range(0, len(indices), CONFIG["chunk_banks"]):
        end = min(start+CONFIG["chunk_banks"], len(indices))
        z = x[indices[start:end]].reshape(-1, 2, 2)
        logits = z @ W.T
        eta_dot = z @ dW.T
        loglike = np.einsum("bkm,ym->byk", logits, Y, optimize=True)
        loglike -= np.logaddexp(0, logits).sum(axis=2)[:, None, :]
        lse = logsumexp(loglike, axis=2)
        pi = np.exp(loglike-lse[:, :, None])
        dotlog = np.einsum("bkm,ym->byk", eta_dot, Y, optimize=True)
        dotlog -= (expit(logits)*eta_dot).sum(axis=2)[:, None, :]
        values[start:end, 0] = -(lse-np.log(2)) @ teacher
        values[start:end, 1] = -np.einsum("byk,byk,y->b", pi, dotlog, teacher, optimize=True)
    result = dict(order=order, banks=order**4, particle_vectors=2*order**4,
                  loss=float(weights @ values[:, 0]),
                  path_derivative=float(weights @ values[:, 1]),
                  quadrature_mass=float(weights.sum()))
    if keep:
        name = f"angular_e{str(epsilon).replace('.', 'p')}_t{str(t).replace('.', 'p')}_q{order}.npz"
        path = OUT/name
        assert not path.exists(), path
        np.savez_compressed(path, values=values, weights=weights,
                            indices=indices, nodes_1d=x, weights_1d=w,
                            teacher_pmf=teacher, W=W, path_dW=dW)
        result.update(raw=path.relative_to(ROOT).as_posix(), raw_sha256=sha(path))
    return result


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT/"ANGULAR_FREEZE.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
         config=CONFIG, sources=sources(), scientific_execution_started=False,
         method="deterministic_Gaussian_quadrature_not_interval_certification",
         python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__))
    print("Angular design/source freeze written", flush=True)


def run():
    f = json.loads((OUT/"ANGULAR_FREEZE.json").read_text("utf-8"))
    assert f["sources"] == sources() and f["config"] == CONFIG
    assert not (OUT/"ANGULAR_RESULTS.json").exists()
    started = time.perf_counter()
    pairs = np.triu_indices(8, 1)
    delta = PHI[:, None]-PHI[None, :]
    c0 = np.cos(delta)[pairs]
    cdot = (delta*np.sin(delta))[pairs]
    D = float(c0 @ cdot)
    prediction = -D/32
    rows = []
    for epsilon in CONFIG["epsilons"]:
        W0, _ = geometry(epsilon)
        teacher = core.model_pmf(np.zeros(8), W0, order=80)
        teacher_checks = [core.model_pmf(np.zeros(8), W0, order=o) for o in (40,60)]
        teacher_checks.append(core.model_pmf(np.zeros(8), W0, order=72, angle=np.pi/7))
        row = dict(epsilon=epsilon, teacher_mass_error=float(abs(teacher.sum()-1)),
                   teacher_pmf_discrepancy=float(max(np.max(abs(q-teacher)) for q in teacher_checks)),
                   teacher_pmf=teacher.tolist(), evaluations=[])
        for t in CONFIG["t_values"]:
            W, _ = geometry(epsilon, t)
            pmfs = [core.model_pmf(np.zeros(8), W, order=o) for o in (40,60,80)]
            pmfs.append(core.model_pmf(np.zeros(8), W, order=72, angle=np.pi/7))
            risks = [float(teacher @ (np.log(teacher)-np.log(q))) for q in pmfs]
            exact = dict(t=t, pmf_mass_error=float(max(abs(q.sum()-1) for q in pmfs)),
                         pmf_discrepancy=float(max(np.max(abs(q-pmfs[2])) for q in pmfs)),
                         risk=risks[2], risk_discrepancy=max(abs(x-risks[2]) for x in risks),
                         risk_all_orders=risks)
            exact["finite_K2"] = [quadrature(epsilon,t,o,teacher,keep=True) for o in CONFIG["orders"]]
            row["evaluations"].append(exact)
            print(json.dumps(dict(epsilon=epsilon,t=t,loss_K2=exact["finite_K2"][-1]["loss"],
                             derivative_K2=exact["finite_K2"][-1]["path_derivative"],risk=risks[2])),flush=True)
        left, right = row["evaluations"]
        differences = [b["loss"]-a["loss"] for a,b in zip(left["finite_K2"],right["finite_K2"])]
        derivs = [x["path_derivative"] for x in left["finite_K2"]]
        row.update(K2_difference= differences[-1], K2_difference_all_orders=differences,
                   K2_difference_discrepancy=max(abs(x-differences[-1]) for x in differences[-2:]),
                   K2_derivative=derivs[-1], K2_derivative_discrepancy=max(abs(x-derivs[-1]) for x in derivs[-2:]),
                   derivative_over_epsilon4=derivs[-1]/epsilon**4, predicted_limit=prediction)
        # Numerical qualifications concern only the high-order comparison; all
        # lower-order failures/deviations remain in the public result.
        row["qualified"] = bool(row["teacher_mass_error"]<=CONFIG["mass_tolerance"] and
            row["teacher_pmf_discrepancy"]<=CONFIG["pmf_tolerance"] and
            all(x["pmf_mass_error"]<=CONFIG["mass_tolerance"] and x["pmf_discrepancy"]<=CONFIG["pmf_tolerance"]
                and x["risk_discrepancy"]<=CONFIG["risk_tolerance"] for x in row["evaluations"]) and
            row["K2_difference_discrepancy"]<=CONFIG["angular_tolerance"] and
            row["K2_derivative_discrepancy"]<=CONFIG["angular_tolerance"])
        rows.append(row)
    # Independent fixed-input differentiation check of this quadrature driver.
    p = core.model_pmf(np.zeros(8), geometry(.5)[0], order=80)
    h = CONFIG["finite_difference_step"]
    center = quadrature(.5,0,8,p)
    plus = quadrature(.5,h,8,p)
    minus = quadrature(.5,-h,8,p)
    fd = (plus["loss"]-minus["loss"])/(2*h)
    fd_error = abs(fd-center["path_derivative"])
    assert fd_error < CONFIG["finite_difference_tolerance"], fd_error
    assert sources() == f["sources"]
    totals = dict(quadrature_banks=sum(z["banks"] for r in rows for e in r["evaluations"] for z in e["finite_K2"])+3*8**4,
                  random_draws=0, full_observation_patterns=256,
                  PMF_node_evaluations=4*3*(40**2+60**2+80**2+72**2)+80**2,
                  PMF_function_calls=49,
                  monte_carlo_confidence_claim=False)
    totals["two_particle_vectors_at_quadrature_nodes"] = 2*totals["quadrature_banks"]
    totals["conditional_pattern_particle_scores"] = 256*totals["two_particle_vectors_at_quadrature_nodes"]
    save(OUT/"ANGULAR_RESULTS.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
         freeze_sha256=sha(OUT/"ANGULAR_FREEZE.json"), config=CONFIG, rows=rows,
         geometry=dict(phi=PHI.tolist(),sum_phi=float(PHI.sum()),
                       minimum_unscaled_offdiagonal=float(c0.min()),complete_dependency_graph=True,
                       S_directional_sum=D, K2_derivative_limit=prediction,
                       s_fixed=True,b_fixed=True,joint_distribution_preserved=False),
         finite_difference=dict(analytic=center["path_derivative"],numeric=fd,absolute_error=fd_error),
         all_qualified=all(r["qualified"] for r in rows), counts=totals,
         seconds=time.perf_counter()-started, numeric_qualification_is_rigorous_interval_certificate=False,
         external_scientific_review_completed=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("freeze","run"))
    args = parser.parse_args()
    try:
        freeze() if args.mode == "freeze" else run()
    except Exception:
        if args.mode == "run":
            name = OUT/("ANGULAR_ERROR_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+".json")
            save(name,dict(traceback=traceback.format_exc(),sources=sources()))
        raise
