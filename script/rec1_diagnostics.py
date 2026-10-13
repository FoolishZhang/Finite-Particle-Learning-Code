"""REC1 frozen saved-panel diagnostics; never writes an older scientific package.

Scientific work requires an explicit freeze and an exclusive execution receipt.
The two directions use only the original active labels and GH24 observed NLL.
All finite-particle evaluation banks are independent of direction construction.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
from scipy.special import expit, logsumexp, roots_hermitenorm
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/rec_v1_20261004/REC1"
DOC = ROOT / "docs/08_research_研究工作/REC_v1_20261004"
OLD = ROOT / "artifacts/joint_scale_learning"
TRUTH = ROOT / "artifacts/active_scale_boundary/REFERENCE/RESULTS.json"
ARMS = ("NEAR_FREE", "UNIT_FREE")
STATES = ("BASE", "MU", "RHO")
MODES = ("retained", "eliminated")
TARGETS = ("F_D", "F_P")
PATTERNS = np.array(list(itertools.product((0, 1), repeat=4)), dtype=np.int8)
PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
CONFIG = dict(streams=16, banks_per_stream=2048, K=100, nuisance=256,
              seed_base=202610046110, rng="PCG64DXSM", active_role=0, nuisance_role=1,
              bank_chunk=32, workers=4, BLAS_threads=1, quadrature_chunk=65536,
              direction_order=24, direction_maxabs_step=.05,
              mean_bounds=[-2., 2.], scale_bounds=[.001, 2.], eigen_floor=.1,
              repair_target=.1001, gradient_zero_threshold=1e-14,
              reference_orders=[32, 40, 48], main_order=40,
              max_PMF_change=2e-7, max_KL_change=2e-6, max_mass_error=2e-12,
              audit_streams=[0, 4, 8, 12], audit_state="RHO", audit_order=44,
              alpha=.05, family_size=16, MC_SE_limit=2e-5, MC_df=2047,
              risk_resolution=2e-6, automatic_retry=False)


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def risk(P, Q):
    P, Q = np.asarray(P), np.asarray(Q)
    return float(np.dot(P, np.log(P / Q)))


def correlation(rho):
    result = np.eye(4)
    for value, (i, j) in zip(rho, PAIRS):
        result[i, j] = result[j, i] = value
    return result


def model(mu, s, rho, tau):
    mu, s, rho = np.asarray(mu), np.asarray(s), np.asarray(rho)
    C = correlation(rho)
    assert mu.shape == s.shape == (4,) and rho.shape == (6,)
    assert np.isfinite(mu).all() and np.isfinite(s).all() and np.isfinite(rho).all()
    assert np.all(mu >= -2.) and np.all(mu <= 2.)
    assert np.all(s >= .001) and np.all(s <= 2.)
    eigmin = float(np.linalg.eigvalsh(C).min())
    assert eigmin >= .1 - 1e-12 and 0. <= tau <= 2.
    return dict(mu=mu.tolist(), s=s.tolist(), rho=rho.tolist(), tau=float(tau),
                covariance=(s[:, None] * C * s[None, :]).tolist(), eigmin=eigmin)


def quadrature(state, order, derivatives=False):
    """Cholesky GH; Price identities give mu and marginal-preserving rho derivatives."""
    nodes, weights = roots_hermitenorm(order)
    weights /= np.sqrt(2. * np.pi)
    lower = np.linalg.cholesky(state["covariance"])
    total = np.zeros(16, dtype=np.longdouble)
    dmu = np.zeros((16, 4), dtype=np.longdouble)
    drho = np.zeros((16, 6), dtype=np.longdouble)
    s = np.asarray(state["s"])
    for begin in range(0, order ** 4, CONFIG["quadrature_chunk"]):
        ids = np.array(np.unravel_index(np.arange(begin, min(begin + CONFIG["quadrature_chunk"], order ** 4)), (order,) * 4)).T
        z = nodes[ids] @ lower.T + state["mu"]
        w = np.prod(weights[ids], axis=1)
        plus, minus = expit(z), expit(-z)
        for yi, y in enumerate(PATTERNS):
            integrand = w * np.prod(np.where(y, plus, minus), axis=1)
            total[yi] += np.sum(integrand, dtype=np.longdouble)
            if derivatives:
                residual = y - plus
                dmu[yi] += np.sum(integrand[:, None] * residual, axis=0, dtype=np.longdouble)
                for pi, (i, j) in enumerate(PAIRS):
                    drho[yi, pi] += s[i] * s[j] * np.sum(integrand * residual[:, i] * residual[:, j], dtype=np.longdouble)
    raw = np.asarray(total, dtype=float)
    mass = float(raw.sum())
    result = dict(order=order, root="cholesky", nodes=order ** 4,
                  raw_pmf=raw.tolist(), raw_mass=mass, mass_error=abs(mass - 1.),
                  pmf=(raw / mass).tolist())
    if derivatives:
        # The integral sums to one analytically; normalization derivative is retained.
        for key, value in (("dmu", dmu), ("drho", drho)):
            deriv = np.asarray(value, dtype=float)
            result[key] = ((deriv * mass - raw[:, None] * deriv.sum(axis=0)) / mass ** 2).tolist()
    return result


def candidates(base, empirical):
    construction = quadrature(base, CONFIG["direction_order"], True)
    q = np.asarray(construction["pmf"])
    gradients = {name: -np.asarray(construction[key]).T @ (empirical / q)
                 for name, key in (("MU", "dmu"), ("RHO", "drho"))}
    result = {"BASE": base}
    metadata = {}
    for name in ("MU", "RHO"):
        g = gradients[name]
        norm = float(np.max(np.abs(g)))
        delta = np.zeros_like(g) if norm <= CONFIG["gradient_zero_threshold"] else -.05 * g / norm
        mu, rho = np.array(base["mu"]), np.array(base["rho"])
        alpha = 1.
        if name == "MU":
            proposed = mu + delta
            mu = np.clip(proposed, -2., 2.)
        else:
            proposed = rho + delta
            emin = float(np.linalg.eigvalsh(correlation(proposed)).min())
            if emin < .1:
                alpha = (1. - .1001) / (1. - emin)
            rho = proposed * alpha
        result[name] = model(mu, base["s"], rho, base["tau"])
        metadata[name] = dict(gradient=g.tolist(), maxabs_gradient=norm,
                              normalized_proposal_delta=delta.tolist(), proposal=proposed.tolist(),
                              radial_alpha=alpha, teacher_used=False, evaluation_bank_used=False,
                              actual_delta=(mu - base["mu"] if name == "MU" else rho - base["rho"]).tolist())
    return result, dict(order=CONFIG["direction_order"], empirical_observed_NLL=float(-empirical @ np.log(q)),
                        derivative_formula="Price Gaussian identity; mu: E[f(y-p)]; rho_ij: s_i*s_j E[f(y_i-p_i)(y_j-p_j)]",
                        construction=construction, directions=metadata)


def scores(state, active_eps, nuisance_eps, log_nuisance_relative=None):
    """Per-bank/pattern reduced F, with n log 2 subtracted in both modes."""
    z = active_eps @ np.linalg.cholesky(state["covariance"]).T + state["mu"]
    log_active = z @ PATTERNS.T - np.logaddexp(0., z).sum(axis=2)[:, :, None]
    K, n = active_eps.shape[1], nuisance_eps.shape[2]
    if log_nuisance_relative is None:
        log_nuisance_relative = (math.log(2.) - np.logaddexp(0., -state["tau"] * nuisance_eps)).sum(axis=2)
    retained = math.log(K) - logsumexp(log_active + log_nuisance_relative[:, :, None], axis=1)
    eliminated = math.log(K) - logsumexp(log_active, axis=1)
    assert retained.shape == eliminated.shape == (len(active_eps), 16)
    assert n > 0 and np.isfinite(retained).all() and np.isfinite(eliminated).all()
    return np.stack((retained, eliminated), axis=1)


def teacher():
    row = next(r for r in read(TRUTH)["records"] if r["name"] == "mixed")
    P = np.asarray(row["teacher_pmf"])
    p = P @ PATTERNS
    product = np.prod(np.where(PATTERNS, p, 1. - p), axis=1)
    return P, p, product


def sources_and_inputs():
    sources = [Path(__file__), ROOT / "script/rec1_verify.py", ROOT / "script/rec1_independent_audit.py",
               DOC / "CONTRACT_REC1.md", DOC / "REC1_IMPLEMENTATION.md"]
    inputs = [TRUTH, OLD / "FREEZE.json", OLD / "MANIFEST.json"]
    for stream in range(16):
        record_path = OLD / "TRAIN" / f"stream_{stream:02}.json"
        record = read(record_path)
        inputs += [record_path, ROOT / record["input_file"], ROOT / record["output_file"], OLD / "REFERENCE" / f"stream_{stream:02}.json"]
    return {rel(p): sha(p) for p in sources}, {rel(p): sha(p) for p in inputs}


def validate_freeze():
    freeze = read(OUT / "FREEZE.json")
    sources, _ = sources_and_inputs()
    assert freeze["authorized"] is True and freeze["config"] == CONFIG
    assert sources == freeze["sources"], "Frozen source or contract changed"
    for path, digest in freeze["inputs"].items():
        assert sha(ROOT / path) == digest, path
    assert sha(OUT / "QUALIFICATION.json") == freeze["qualification_sha256"]
    assert read(OUT / "QUALIFICATION.json")["passed"]
    return freeze


def freeze():
    sources, inputs = sources_and_inputs()
    qualification = read(OUT / "QUALIFICATION.json")
    assert qualification["passed"]
    assert qualification["producer_sha256"] == sha(__file__)
    assert qualification["verifier_sha256"] == sha(ROOT / "script/rec1_verify.py")
    result = dict(authorized=True, frozen_utc=datetime.now(timezone.utc).isoformat(),
                  git_HEAD=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  config=CONFIG, sources=sources, inputs=inputs,
                  qualification_sha256=sha(OUT / "QUALIFICATION.json"),
                  authorization="Current user explicitly requested continuation of REC/USE; parent approved this fixed REC1 contract before scientific work")
    save(OUT / "FREEZE.json", result)
    print(json.dumps(dict(frozen=True, sha256=sha(OUT / "FREEZE.json"))))


def stream_run(stream):
    tick = time.perf_counter()
    try:
        P, p, true_product = teacher()
        old = read(OLD / "TRAIN" / f"stream_{stream:02}.json")
        assert sha(ROOT / old["input_file"]) == old["input_sha256"]
        with np.load(ROOT / old["input_file"], allow_pickle=False) as data:
            labels = data["active_labels"].copy()
            assert labels.shape == (4096, 4) and data["nuisance_labels"].shape == (4096, 256)
        counts = labels.sum(axis=0)
        assert counts.tolist() == old["data_counts"]
        empirical = np.bincount(labels @ np.array([8, 4, 2, 1]), minlength=16) / 4096.
        fitted_p = (counts + .5) / 4097.
        fitted_q = np.prod(np.where(PATTERNS, fitted_p, 1. - fitted_p), axis=1)
        baseline = dict(counts=counts.tolist(), p=fitted_p.tolist(), pmf=fitted_q.tolist(),
                        risk=risk(P, fitted_q), empirical_NLL=float(-empirical @ np.log(fitted_q)),
                        population_NLL=float(-P @ np.log(fitted_q)), new_closed_form_fit=True)
        refs = {row["arm"]: row for row in read(OLD / "REFERENCE" / f"stream_{stream:02}.json")["rows"]}
        states, rows = {}, []
        for arm in ARMS:
            original = old["outputs"][arm]["TAIL"]
            base = model(original["mu"], original["s"], original["rho"], original["tau"])
            assert np.max(abs(np.asarray(base["covariance"]) - original["covariance"])) < 1e-14
            states[arm], construction = candidates(base, empirical)
            for state_name in STATES:
                state = states[arm][state_name]
                if state_name == "BASE":
                    variants = refs[arm]["variants"]
                    assert refs[arm]["qualification_passed"]
                else:
                    variants = [quadrature(state, order) for order in CONFIG["reference_orders"]]
                main = np.asarray(next(v["pmf"] for v in variants if v["order"] == 40))
                R = risk(P, main)
                qualifications = []
                for variant in variants:
                    q = np.asarray(variant["pmf"])
                    pmf_change, risk_change = float(np.max(abs(q - main))), abs(risk(P, q) - R)
                    ok = pmf_change <= 2e-7 and risk_change <= 2e-6 and variant["mass_error"] <= 2e-12 and bool(np.all(q > 0.))
                    qualifications.append(dict(order=variant["order"], max_PMF_change=pmf_change,
                                               KL_change=risk_change, mass_error=variant["mass_error"], passed=ok))
                marginals = main @ PATTERNS
                own_product = np.prod(np.where(PATTERNS, marginals, 1. - marginals), axis=1)
                rows.append(dict(stream=stream, arm=arm, state=state_name, model=state,
                                 main_pmf=main.tolist(), risk=R, variants=variants,
                                 reference_reused=state_name == "BASE", qualifications=qualifications,
                                 qualification_passed=all(q["passed"] for q in qualifications),
                                 empirical_exact_NLL=float(-empirical @ np.log(main)),
                                 population_exact_NLL=float(-P @ np.log(main)),
                                 own_product_risk=risk(P, own_product),
                                 risk_minus_same_data_independent=R - baseline["risk"],
                                 risk_minus_true_product=R - risk(P, true_product),
                                 construction=construction if state_name == "BASE" else None))
        # Direction/model bytes are committed locally before evaluation RNG begins.
        save(OUT / f"models_{stream:02}.json", dict(stream=stream, baseline=baseline, empirical=empirical.tolist(), records=rows))
        ra = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([CONFIG["seed_base"], stream, 0])))
        rn = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([CONFIG["seed_base"], stream, 1])))
        draws = np.empty((2048, 2, 3, 2, 2), dtype=np.float64)
        hashes = []
        for start in range(0, 2048, 32):
            active = ra.standard_normal((32, 100, 4))
            nuisance = rn.standard_normal((32, 100, 256))
            hashes.append(dict(start=start, active=hashlib.sha256(active.tobytes()).hexdigest(),
                               nuisance=hashlib.sha256(nuisance.tobytes()).hexdigest()))
            for ai, arm in enumerate(ARMS):
                log_nuisance = (math.log(2.) - np.logaddexp(0., -states[arm]["BASE"]["tau"] * nuisance)).sum(axis=2)
                for si, state_name in enumerate(STATES):
                    by_pattern = scores(states[arm][state_name], active, nuisance, log_nuisance)
                    draws[start:start + 32, ai, si] = by_pattern @ np.stack((empirical, P), axis=1)
        filename = OUT / f"scores_{stream:02}.npz"
        assert not filename.exists()
        np.savez_compressed(filename, reduced_scores=draws)
        score_rows = []
        for ai, arm in enumerate(ARMS):
            for si, state_name in enumerate(STATES):
                for mi, mode in enumerate(MODES):
                    for ti, target in enumerate(TARGETS):
                        values = draws[:, ai, si, mi, ti]
                        delta = values - draws[:, ai, 0, mi, ti]
                        score_rows.append(dict(arm=arm, state=state_name, mode=mode, target=target,
                                               reduced_mean=float(values.mean()), actual_mean=float(values.mean() + 256. * math.log(2.)),
                                               MC_SE=float(values.std(ddof=1) / math.sqrt(2048)),
                                               delta_mean=float(delta.mean()), delta_bank_variance=float(delta.var(ddof=1))))
        result = dict(stream=stream, model_file=rel(OUT / f"models_{stream:02}.json"),
                      model_sha256=sha(OUT / f"models_{stream:02}.json"), score_file=rel(filename), score_sha256=sha(filename),
                      score_rows=score_rows, bank_hashes=hashes, seconds=time.perf_counter() - tick,
                      numerical_qualification_passed=all(row["qualification_passed"] for row in rows))
        save(OUT / f"stream_{stream:02}.json", result)
        return result
    except BaseException as error:
        save(OUT / f"ERROR_stream_{stream:02}.json", dict(stream=stream, error=str(error), traceback=traceback.format_exc(), automatic_retry=False))
        raise


def summarize(records):
    P, p, true_product = teacher()
    primary = []
    tcrit = float(student_t.ppf(1. - .05 / (2. * 16), 2047))
    for arm, state_name, mode, target in itertools.product(ARMS, STATES[1:], MODES, TARGETS):
        chosen = [next(row for row in record["score_rows"] if (row["arm"], row["state"], row["mode"], row["target"]) == (arm, state_name, mode, target)) for record in records]
        mean = float(np.mean([row["delta_mean"] for row in chosen]))
        SE = float(math.sqrt(sum(row["delta_bank_variance"] / 2048 for row in chosen)) / 16)
        interval = [mean - tcrit * SE, mean + tcrit * SE]
        primary.append(dict(arm=arm, candidate=state_name, mode=mode, target=target,
                            stream_paired_means=[row["delta_mean"] for row in chosen], delta=mean, MC_SE=SE,
                            interval=interval, direction="decrease" if interval[1] < 0. else "increase" if interval[0] > 0. else "unresolved",
                            precision_passed=SE <= 2e-5))
    all_models = [row for record in records for row in read(ROOT / record["model_file"])["records"]]
    risk_rows = []
    for arm, state_name in itertools.product(ARMS, STATES):
        group = [row for row in all_models if row["arm"] == arm and row["state"] == state_name]
        delta = [row["risk"] - next(base["risk"] for base in all_models if base["stream"] == row["stream"] and base["arm"] == arm and base["state"] == "BASE") for row in group]
        risk_rows.append(dict(arm=arm, state=state_name, risks=[row["risk"] for row in group],
                              mean_risk=float(np.mean([row["risk"] for row in group])),
                              paired_delta=delta, mean_delta=float(np.mean(delta)), delta_range=[min(delta), max(delta)],
                              risk_decrease_resolved=float(np.mean(delta)) < -2e-6,
                              versus_same_data_independent=[row["risk_minus_same_data_independent"] for row in group],
                              mean_versus_same_data_independent=float(np.mean([row["risk_minus_same_data_independent"] for row in group])),
                              all_quadrature_passed=all(row["qualification_passed"] for row in group)))
    baselines = [read(ROOT / record["model_file"])["baseline"] for record in records]
    return dict(config=CONFIG, freeze_sha256=sha(OUT / "FREEZE.json"), records=records,
                teacher_P=P.tolist(), teacher_p=p.tolist(), true_product_risk=risk(P, true_product),
                baseline_risks=[b["risk"] for b in baselines], baseline_mean_risk=float(np.mean([b["risk"] for b in baselines])),
                primary_MC_family=primary, descriptive_risk=risk_rows, MC_tcritical=tcrit,
                numerical_qualification_passed=all(r["numerical_qualification_passed"] for r in records),
                statistical_scope="Conditional fixed saved panel; family intervals cover only Monte Carlo uncertainty, development Student t approximation, no new training-population inference",
                accounting=dict(new_observations=0, new_closed_form_baseline_fits=16, new_optimizations=0,
                                frozen_base_models=32, finite_candidates=64, construction_GH_nodes=32 * 24 ** 4,
                                candidate_GH_PMF=192, candidate_GH_nodes=64 * sum(o ** 4 for o in (32, 40, 48)),
                                reused_base_PMF=96, prior_vectors=16 * 2048 * 100,
                                prior_Gaussian=16 * 2048 * 100 * 260,
                                evaluated_nuisance_logits=16 * 2048 * 2 * 100 * 256,
                                evaluated_active_logits=16 * 2048 * 2 * 3 * 100 * 4,
                                pattern_bank_state_mode_scores=16 * 2048 * 2 * 3 * 2 * 16,
                                independent_audit_PMF=8, independent_audit_GH_nodes=8 * 44 ** 4),
                limitations=["Risk and teacher finite target are controlled diagnostic oracle evaluations, not deployable direction information.",
                             "GH comparisons are numerical qualification, not rigorous remainder bounds.",
                             "MU changes mean coordinates; RHO changes dependence while exactly preserving univariate marginals.",
                             "FD-FP changes active label weights only; no complete attribution to data noise.",
                             "Retained-minus-eliminated target is a same-state calculation difference, not a causal decomposition of training."])


def run():
    validate_freeze()
    save(OUT / "EXECUTION.json", dict(started_utc=datetime.now(timezone.utc).isoformat(), freeze_sha256=sha(OUT / "FREEZE.json"), automatic_retry=False))
    tick = time.perf_counter()
    try:
        records = []
        with ProcessPoolExecutor(max_workers=4) as pool:
            for future in as_completed([pool.submit(stream_run, stream) for stream in range(16)]):
                row = future.result()
                records.append(row)
                print(json.dumps(dict(stream=row["stream"], seconds=row["seconds"], numerical_passed=row["numerical_qualification_passed"])), flush=True)
        records.sort(key=lambda row: row["stream"])
        validate_freeze()
        result = summarize(records)
        result["seconds"] = time.perf_counter() - tick
        save(OUT / "RESULTS.json", result)
        print(json.dumps(dict(complete=True, seconds=result["seconds"], numerical_passed=result["numerical_qualification_passed"])), flush=True)
    except BaseException as error:
        save(OUT / "ERROR.json", dict(error=str(error), traceback=traceback.format_exc(), automatic_retry=False))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run"))
    args = parser.parse_args()
    globals()[args.command]()
