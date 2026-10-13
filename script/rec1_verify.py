"""Manufactured REC1 qualification and prespecified independent GH44 checks.

The audit quadrature uses numpy Hermite nodes and a symmetric covariance root.
The producer's quadrature/score/summary implementations are not used in audit.
"""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
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
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/rec_v1_20261004/REC1"
Y = np.array(list(itertools.product((0, 1), repeat=4)))


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def independent_pmf(mu, covariance, order):
    lam, U = np.linalg.eigh(covariance)
    assert min(lam) > 0.
    root = (U * np.sqrt(lam)) @ U.T
    x, w = np.polynomial.hermite.hermgauss(order)
    x *= math.sqrt(2.)
    w /= math.sqrt(math.pi)
    total = np.zeros(16)
    for begin in range(0, order ** 4, 65536):
        digits = np.array(np.unravel_index(np.arange(begin, min(begin + 65536, order ** 4)), (order,) * 4)).T
        logits = x[digits] @ root.T + mu
        weight = np.prod(w[digits], axis=1)
        plus, minus = expit(logits), expit(-logits)
        for j, pattern in enumerate(Y):
            total[j] += np.dot(weight, np.prod(np.where(pattern, plus, minus), axis=1))
    return dict(order=order, pmf=(total / total.sum()).tolist(), mass_error=abs(float(total.sum()) - 1.),
                root="symmetric_eigendecomposition", nodes=order ** 4)


def qualify():
    import rec1_diagnostics as producer
    import jointscale_kernel as legacy
    import scipy
    import torch
    tick = time.perf_counter()
    result = dict(passed=False, producer_sha256=sha(producer.__file__), verifier_sha256=sha(__file__),
                  new_scientific_data=0, new_scientific_RNG=0, new_panel_models=0,
                  runtime=dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                               numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__),
                  legacy_kernel_provenance=legacy.provenance(),
                  scope="Manufactured deterministic tensors only; no actual saved output or observation used")
    try:
        state = producer.model([-.35, -.1, .12, .4], [.4, .6, .7, .9], [.11, -.07, .05, .09, -.06, .13], .25)
        active = np.sin(np.arange(3 * 7 * 4).reshape(3, 7, 4) * .37)
        nuisance = np.cos(np.arange(3 * 7 * 5).reshape(3, 7, 5) * .29)
        actual = producer.scores(state, active, nuisance)
        comparisons = []
        for yi, y in enumerate(Y):
            labels = np.broadcast_to(y, (3, 4)).copy()
            nuisance_labels = np.ones((3, 5), dtype=int)
            for mode, method in enumerate(("FREE", "ZERO")):
                for path, tolerance in (("math64", 3e-13), ("official32", 2e-5)):
                    m = legacy.ScoreProgram(method, state["mu"], state["s"], K=7, n=5, tau0=state["tau"], path=path)
                    restored = {key: state[key] for key in ("mu", "s", "rho", "tau")}
                    if method == "ZERO":
                        restored["tau"] = 0.
                    m.load_physical_state(restored)
                    batch = legacy.prepare_batch(active.transpose(1, 0, 2), nuisance.transpose(1, 0, 2), labels, nuisance_labels, path=path)
                    expected = float(m(batch).detach()) - 5. * math.log(2.)
                    error = abs(expected - actual[:, mode, yi].mean())
                    comparisons.append(dict(pattern=int(yi), mode=method, implementation=path, error=error, tolerance=tolerance))
                    assert error <= tolerance, comparisons[-1]
        zero = dict(state, tau=0.)
        zero_scores = producer.scores(zero, active, nuisance)
        zero_error = float(np.max(abs(zero_scores[:, 0] - zero_scores[:, 1])))
        assert zero_error < 1e-13
        empirical = np.arange(1., 17.)
        empirical /= empirical.sum()
        derivatives = producer.quadrature(state, 24, True)
        expected_pmf = independent_pmf(state["mu"], np.array(state["covariance"]), 30)
        pmf_error = float(np.max(abs(np.asarray(derivatives["pmf"]) - expected_pmf["pmf"])))
        assert pmf_error < 2e-10, pmf_error
        gradients = []
        for name, key, size in (("mu", "dmu", 4), ("rho", "drho", 6)):
            analytical = -np.asarray(derivatives[key]).T @ (empirical / derivatives["pmf"])
            for index in range(size):
                finite_values = []
                for sign in (-1., 1.):
                    mu, rho = np.array(state["mu"]), np.array(state["rho"])
                    (mu if name == "mu" else rho)[index] += sign * 1e-4
                    trial = producer.model(mu, state["s"], rho, state["tau"])
                    q = independent_pmf(trial["mu"], np.asarray(trial["covariance"]), 24)["pmf"]
                    finite_values.append(-empirical @ np.log(q))
                numeric = (finite_values[1] - finite_values[0]) / 2e-4
                error = abs(numeric - analytical[index])
                gradients.append(dict(coordinate=name, index=index, analytic=float(analytical[index]), independent_symmetric_root_finite_difference=float(numeric), error=float(error)))
                assert error <= 5e-8, gradients[-1]
        states, construction = producer.candidates(state, empirical)
        for kind in ("MU", "RHO"):
            assert states[kind]["s"] == state["s"] and states[kind]["tau"] == state["tau"]
            assert states[kind]["rho"] == state["rho"] if kind == "MU" else states[kind]["mu"] == state["mu"]
            assert max(abs(np.asarray(construction["directions"][kind]["normalized_proposal_delta"]))) <= .05 + 1e-15
        # Force the same candidate-construction repair path using a deterministic manufactured derivative.
        original = producer.quadrature
        try:
            def forced_quad(*args, **kwargs):
                return dict(pmf=(np.ones(16) / 16).tolist(), dmu=(np.ones((16, 4)) * .01).tolist(), drho=(np.ones((16, 6)) * .01).tolist())
            producer.quadrature = forced_quad
            edge = producer.model([1.99] * 4, [1.] * 4, [.8999] * 6, .25)
            repaired, repair = producer.candidates(edge, np.ones(16) / 16)
            assert repaired["MU"]["mu"] == [2.] * 4
            assert repair["directions"]["RHO"]["radial_alpha"] < 1.
            assert abs(repaired["RHO"]["eigmin"] - .1001) < 1e-12
        finally:
            producer.quadrature = original
        result.update(passed=True, scores=comparisons, zero_nuisance_max_error=zero_error,
                      independent_quadrature_max_PMF_error=pmf_error, gradients=gradients,
                      candidate_same_coordinates_passed=True, forced_box_and_spectral_repair_passed=True,
                      seconds=time.perf_counter() - tick)
        save(OUT / "QUALIFICATION.json", result)
        print(json.dumps({key: result[key] for key in ("passed", "seconds", "zero_nuisance_max_error", "independent_quadrature_max_PMF_error")}))
    except BaseException as error:
        result.update(error=str(error), traceback=traceback.format_exc(), seconds=time.perf_counter() - tick)
        attempts = len(list(OUT.glob("QUALIFICATION_FAILURE_*.json")))
        save(OUT / f"QUALIFICATION_FAILURE_{attempts:02}.json", result)
        raise


def audit():
    tick = time.perf_counter()
    freeze = read(OUT / "FREEZE.json")
    assert freeze["sources"]["script/rec1_verify.py"] == sha(__file__)
    results = read(OUT / "RESULTS.json")
    save(OUT / "INDEPENDENT_GH_EXECUTION.json", dict(started_utc=datetime.now(timezone.utc).isoformat(), freeze_sha256=sha(OUT / "FREEZE.json"), automatic_retry=False))
    try:
        P = np.asarray(results["teacher_P"])
        rows = []
        for stream in (0, 4, 8, 12):
            record = read(OUT / f"models_{stream:02}.json")
            for arm in ("NEAR_FREE", "UNIT_FREE"):
                selected = next(row for row in record["records"] if row["arm"] == arm and row["state"] == "RHO")
                m = selected["model"]
                independent = independent_pmf(m["mu"], np.asarray(m["covariance"]), 44)
                q = np.asarray(independent["pmf"])
                R = float(np.dot(P, np.log(P / q)))
                pmf_error = float(np.max(abs(q - selected["main_pmf"])))
                risk_error = abs(R - selected["risk"])
                passed = pmf_error <= 2e-7 and risk_error <= 2e-6 and independent["mass_error"] <= 2e-12
                rows.append(dict(stream=stream, arm=arm, state="RHO", independent=independent,
                                 max_PMF_error=pmf_error, KL_error=risk_error, passed=passed))
                print(json.dumps(dict(audit_stream=stream, arm=arm, passed=passed)), flush=True)
        result = dict(passed=all(row["passed"] for row in rows), records=rows,
                      numerical_scope="8 prespecified RHO candidates; independent numpy Hermite nodes and symmetric root; no rigorous remainder bound",
                      new_scientific_fits=0, new_data=0, nodes=8 * 44 ** 4, seconds=time.perf_counter() - tick,
                      source_sha256=sha(__file__), result_sha256=sha(OUT / "RESULTS.json"))
        save(OUT / "INDEPENDENT_GH.json", result)
        print(json.dumps(dict(passed=result["passed"], seconds=result["seconds"])), flush=True)
    except BaseException as error:
        save(OUT / "INDEPENDENT_GH_ERROR.json", dict(error=str(error), traceback=traceback.format_exc(), automatic_retry=False))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("qualify", "audit"))
    globals()[parser.parse_args().command]()
