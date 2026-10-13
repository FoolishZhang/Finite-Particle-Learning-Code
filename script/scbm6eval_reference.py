"""Frozen PMF reuse, one new PMF, and a finite matrix-potential reference.

Only run() evaluates Gaussian model integrals. It requires the root package
freeze and performs exactly the prespecified orders; failures are retained.
"""
from pathlib import Path
from datetime import datetime, timezone
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import hashlib
import json
import math
import time
import traceback
import numpy as np
from scipy.special import roots_hermitenorm, roots_legendre
import scbminv_reference as pmf_helper

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/scbm_six_frozen_targets"
OUT = BASE / "REFERENCE"
CONFIG = BASE / "CONFIG.json"
CANDIDATES = BASE / "CANDIDATES.json"
PLAN = ROOT / "docs/08_research_研究工作/六相关冻结输出与有限目标_20261003/PLAN.md"
OLD_TRUTH = ROOT / "artifacts/scbm_invariance/REFERENCE/MODEL.json"
OLD_REFERENCE = ROOT / "artifacts/scbm_multicorrelation/REFERENCE/RESULTS.json"
OLD_TRAIN = ROOT / "artifacts/scbm_multicorrelation/TRAIN/RESULTS.json"
PAPER = ROOT / "artifacts/paper_v7_20261003"
HELPER_SHA = "79c911d1601ae9740b0e045a5d879394cfd4a82aaa9e48f72070fc46b5098a41"
PAPER_MANIFEST_SHA = "801306f7797b3cb4231cf81b5575b2b6c70db3a5e5f4b87d5cd939eb58e2621d"
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
MU = np.array([-.6, -.2, .2, .6])
RULES = [(16, 24), (24, 32), (32, 40)]
EXPECTED_REFERENCE = {
    "pmf_orders": [24, 32, 40], "pmf_main_order": 32,
    "pmf_maximum_change": 5e-8, "normalization_tolerance": 2e-12,
    "V_orders": [list(r) for r in RULES], "V_main_orders": [24, 32],
    "V_maximum_change": 1e-8,
    "V_center": "fixed old truth PMF spin marginal means and true spin covariances; candidate pair moment by GH",
}
PROGRESS = {
    "completed_new_PMF_integrations": 0, "completed_four_dimensional_nodes": 0,
    "completed_V_pair_rule_integrals": 0, "completed_inner_pair_expectations": 0,
    "completed_bivariate_nodes": 0, "completed_V_candidates": 0, "in_flight": None,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def save(path, value, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def covariance(rho):
    assert len(rho) == 6 and np.isfinite(rho).all()
    result = np.eye(4)
    for value, (i, j) in zip(rho, PAIRS):
        result[i, j] = result[j, i] = value
    return result


def key_of(rho):
    return "|".join(float(x).hex() for x in rho)


def patterns():
    return [[(number >> bit) & 1 for bit in (3, 2, 1, 0)] for number in range(16)]


def planned_accounting():
    return {
        "candidates": 35, "reused_PMFs": 34, "new_PMF_candidates": 1,
        "new_PMF_integrations": 3, "new_PMF_scalar_pattern_integrals": 48,
        "new_four_dimensional_Gaussian_nodes": sum(o**4 for o in (24, 32, 40)),
        "V_candidate_rule_evaluations": 35 * 3,
        "V_pair_rule_integrals": 35 * 6 * 3,
        "inner_pair_GH_expectations": 35 * 6 * sum(gl for gl, gh in RULES),
        "new_bivariate_Gaussian_nodes": 35 * 6 * sum(gl * gh**2 for gl, gh in RULES),
        "new_one_dimensional_marginal_integrals": 0,
        "new_truth_PMF_integrals": 0, "new_random_draws": 0,
        "new_observations": 0, "new_fits": 0, "additional_rules_or_retries": 0,
    }


def spin_center(pmf):
    labels = np.asarray(patterns(), dtype=int)
    spins = 2 * labels - 1
    b = [math.fsum(float(p) * int(y[i]) for p, y in zip(pmf, spins)) for i in range(4)]
    q = [math.fsum(float(p) * int(y[i] * y[j]) for p, y in zip(pmf, spins)) for i, j in PAIRS]
    kappa = [value - b[i] * b[j] for value, (i, j) in zip(q, PAIRS)]
    return {
        "raw_PMF_mass": math.fsum(pmf), "spin_means": b,
        "spin_pair_moments": q, "true_spin_covariances": kappa,
        "renormalized": False, "summation": "math.fsum over old qualified truth32 PMF",
    }


def validate_candidates():
    cfg, selection = read(CONFIG), read(CANDIDATES)
    assert cfg["reference"] == EXPECTED_REFERENCE
    assert selection["schema"] == "scbm6eval-candidates-v1"
    assert selection["mu"] == selection["mean"] == MU.tolist()
    assert selection["pair_order"] == [list(p) for p in PAIRS] and selection["patterns"] == patterns()
    for name, digest in selection["inputs"].items():
        assert sha(ROOT / name) == digest, name
    old = read(OLD_TRUTH)
    reference = read(OLD_REFERENCE)
    train = read(OLD_TRAIN)
    assert old["qualification_passed"] and reference["qualification_passed"]
    assert old["patterns"] == patterns() and old["config"]["mean"] == MU.tolist()
    assert [r["stream"] for r in train["records"]] == list(range(16))
    table = {r["key"]: r for r in reference["records"]}
    rows = selection["candidates"]
    names = ["truth", "high"] + [f"FIXED_TAIL_{s:02}" for s in range(16)] + [
        f"FACT_TAIL_{s:02}" for s in range(16)] + ["MEAN_FIXED"]
    assert len(rows) == 35 and [r["index"] for r in rows] == list(range(35))
    assert [r["name"] for r in rows] == names
    for index, row in enumerate(rows):
        C = covariance(row["rho"])
        assert np.array_equal(C, np.asarray(row["covariance"]))
        eigmin = float(np.linalg.eigvalsh(C).min())
        assert eigmin >= .09999 and abs(eigmin - row["minimum_eigenvalue"]) < 2e-13
        if index < 2:
            t = [0., .75][index]
            origin = next(r for r in old["records"] if r["t"] == t)
            variant = next(v for v in origin["variants"] if v["order"] == 32 and v["rotation"] == "cholesky")
            assert row["origin"] == {"path": rel(OLD_TRUTH), "t": t}
            assert row["reused_pmf"] == origin["main_pmf"]
            assert np.array_equal(C, np.asarray(variant["covariance"]))
        elif index < 34:
            method, stream = ("FIXED", index - 2) if index < 18 else ("FACT", index - 18)
            assert row["stream"] == stream and row["rho"] == train["records"][stream]["outputs"][method]["TAIL"]["rho"]
            origin = row["origin"]
            assert origin["path"] == rel(OLD_REFERENCE) and origin["method"] == method and origin["kind"] == "TAIL"
            assert origin["key"] == key_of(row["rho"])
            assert table[origin["key"]]["passed"] and row["reused_pmf"] == table[origin["key"]]["main_pmf"]
        else:
            assert row["reused_pmf"] is None
            expected = [math.fsum(r["outputs"]["FIXED"]["TAIL"]["rho"][j] for r in train["records"]) / 16
                        for j in range(6)]
            assert max(abs(x - y) for x, y in zip(row["rho"], expected)) <= 2e-15
    assert selection["population_weights"] == rows[0]["reused_pmf"]
    assert [r["stream"] for r in selection["data_weights"]] == list(range(16))
    for row in selection["data_weights"]:
        assert len(row["counts"]) == 16 and sum(row["counts"]) == 4096
        assert row["weights"] == [count / 4096 for count in row["counts"]]
    return cfg, selection


def validate_freeze():
    frozen = read(BASE / "FREEZE.json")
    assert frozen["run_authorized"] is True
    required = [Path(__file__), CONFIG, CANDIDATES, PLAN, Path(pmf_helper.__file__),
                OLD_TRUTH, OLD_REFERENCE, OLD_TRAIN, PAPER / "MANIFEST.json", PAPER / "VERSION_LOCK.json"]
    assert all(rel(p) in frozen["inputs"] for p in required)
    for name, digest in frozen["inputs"].items():
        assert sha(ROOT / name) == digest, name
    assert sha(pmf_helper.__file__) == HELPER_SHA
    assert sha(PAPER / "MANIFEST.json") == PAPER_MANIFEST_SHA
    assert read(PAPER / "VERSION_LOCK.json")["manifest_sha256"] == PAPER_MANIFEST_SHA
    for name, digest in read(PAPER / "MANIFEST.json")["files"].items():
        assert sha(ROOT / name) == digest, name
    cfg, selection = validate_candidates()
    return frozen, cfg, selection


def pmf_qualifications(variants):
    main = next(v["pmf"] for v in variants if v["order"] == 32)
    rows = []
    for variant in variants:
        p = variant["pmf"]
        delta = max(abs(x - y) for x, y in zip(p, main))
        norm = abs(math.fsum(p) - 1.)
        positive = len(p) == 16 and all(math.isfinite(x) and x > 0 for x in p)
        rows.append({"order": variant["order"], "maximum_change": delta,
                     "normalization_error": norm, "positive": positive,
                     "passed": bool(positive and delta <= 5e-8 and norm <= 2e-12)})
    return main, rows


def outer_integral(correlation, weights, kappa_values):
    return .5 * float(correlation) * math.fsum(float(w) * float(k) for w, k in zip(weights, kappa_values))


def potential_value(rho, kappa_star, pair_integrals):
    return -.25 * math.fsum(float(k) * float(c) + float(v)
                            for k, c, v in zip(kappa_star, rho, pair_integrals))


def integrate_pair(correlation, pair, gl_order, gh_order, center):
    # No zero-correlation shortcut: all prescribed nodes are actually evaluated.
    legendre_x, legendre_w = roots_legendre(gl_order)
    gaussian_x, gaussian_w = roots_hermitenorm(gh_order)
    gaussian_w = gaussian_w / math.sqrt(2 * math.pi)
    u = .5 * float(correlation) * (legendre_x + 1.)
    assert np.all(abs(u) < 1.)
    i, j = pair
    z1 = gaussian_x[None, :, None]
    z2 = gaussian_x[None, None, :]
    x1 = MU[i] + z1
    x2 = MU[j] + u[:, None, None] * z1 + np.sqrt(1-u**2)[:, None, None] * z2
    product = np.tanh(x1 / 2) * np.tanh(x2 / 2)
    weight = gaussian_w[None, :, None] * gaussian_w[None, None, :]
    moments = np.sum(product * weight, axis=(1, 2), dtype=np.longdouble).astype(float)
    kappa = moments - center["spin_means"][i] * center["spin_means"][j]
    integral = outer_integral(correlation, legendre_w, kappa)
    return {
        "pair": list(pair), "correlation": float(correlation), "GL_order": gl_order, "GH_order": gh_order,
        "outer_correlations": u.tolist(), "outer_weights_on_minus1_plus1": legendre_w.tolist(),
        "inner_spin_pair_moments": moments.tolist(), "inner_centered_covariances": kappa.tolist(),
        "integral_0_to_c": integral, "inner_pair_expectations": gl_order,
        "bivariate_Gaussian_nodes": gl_order * gh_order**2,
        "root": "bivariate Cholesky", "evaluated_at_zero_without_shortcut": True,
    }


def exact_reference_arithmetic(pmf, selection):
    truth = selection["population_weights"]
    assert all(x > 0 and math.isfinite(x) for x in pmf)
    logs = [math.log(x) for x in pmf]
    return {
        "reference_joint_KL": math.fsum(p * math.log(p/q) for p, q in zip(truth, pmf)),
        "reference_cross_entropy_P": -math.fsum(p * l for p, l in zip(truth, logs)),
        "reference_cross_entropy_D": [-math.fsum(p * l for p, l in zip(row["weights"], logs))
                                      for row in selection["data_weights"]],
    }


def run():
    frozen, cfg, selection = validate_freeze()
    assert not (OUT / "EXECUTION.json").exists() and not (OUT / "RESULTS.json").exists()
    test = read(OUT / "SELFTEST.json")
    assert test["passed"] and test["source_sha256"] == sha(__file__) and test["config_sha256"] == sha(CONFIG)
    save(OUT / "EXECUTION.json", {
        "started_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": sha(__file__),
        "root_freeze_sha256": sha(BASE / "FREEZE.json"), "automatic_retry": False,
    })
    started = time.perf_counter()
    try:
        center = spin_center(selection["population_weights"])
        center["source_path"] = rel(OLD_TRUTH)
        center["source_sha256"] = sha(OLD_TRUTH)
        variants = []
        for order in EXPECTED_REFERENCE["pmf_orders"]:
            PROGRESS["in_flight"] = {"phase": "new_mean_PMF", "candidate": 34, "order": order,
                                      "partial_nodes_unknown_if_interrupted": True}
            save(OUT / "PROGRESS.json", PROGRESS, False)
            variants.append(pmf_helper.pmf(np.asarray(selection["candidates"][34]["covariance"]), order, "cholesky"))
            PROGRESS["completed_new_PMF_integrations"] += 1
            PROGRESS["completed_four_dimensional_nodes"] += order**4
            PROGRESS["in_flight"] = None
            save(OUT / "PROGRESS.json", PROGRESS, False)
        mean_pmf, mean_quals = pmf_qualifications(variants)
        save(OUT / "MEAN_FIXED_PMF.json", {"index": 34, "name": "MEAN_FIXED", "variants": variants,
            "main_pmf": mean_pmf, "qualifications": mean_quals, "passed": all(q["passed"] for q in mean_quals)})
        records = []
        for candidate in selection["candidates"]:
            index = candidate["index"]
            V_variants = []
            for gl, gh in RULES:
                pairs = []
                for j, pair in enumerate(PAIRS):
                    PROGRESS["in_flight"] = {"phase": "V_pair_integral", "candidate": index,
                        "pair": list(pair), "GL_order": gl, "GH_order": gh,
                        "partial_nodes_unknown_if_interrupted": True}
                    save(OUT / "PROGRESS.json", PROGRESS, False)
                    pairs.append(integrate_pair(candidate["rho"][j], pair, gl, gh, center))
                    PROGRESS["completed_V_pair_rule_integrals"] += 1
                    PROGRESS["completed_inner_pair_expectations"] += gl
                    PROGRESS["completed_bivariate_nodes"] += gl * gh**2
                    PROGRESS["in_flight"] = None
                V = potential_value(candidate["rho"], center["true_spin_covariances"],
                                    [p["integral_0_to_c"] for p in pairs])
                V_variants.append({"GL_order": gl, "GH_order": gh, "V": V, "pairs": pairs})
                save(OUT / "PROGRESS.json", PROGRESS, False)
            main = next(v["V"] for v in V_variants if (v["GL_order"], v["GH_order"]) == (24, 32))
            V_quals = [{"GL_order": v["GL_order"], "GH_order": v["GH_order"],
                        "maximum_change": abs(v["V"]-main), "passed": bool(abs(v["V"]-main) <= 1e-8)}
                       for v in V_variants]
            pmf = mean_pmf if index == 34 else candidate["reused_pmf"]
            pmf_quals = mean_quals if index == 34 else [{
                "reused": True, "source_path": candidate["origin"]["path"],
                "source_sha256": sha(ROOT / candidate["origin"]["path"]),
                "normalization_error": abs(math.fsum(pmf)-1.),
                "positive": all(p > 0 for p in pmf),
                "passed": bool(all(p > 0 for p in pmf) and abs(math.fsum(pmf)-1.) <= 2e-12)}]
            row = {
                "index": index, "name": candidate["name"], "rho": candidate["rho"],
                "covariance": candidate["covariance"], "origin": candidate["origin"],
                "main_pmf": pmf, "PMF_reused": index != 34,
                **exact_reference_arithmetic(pmf, selection),
                "V_main": main, "V_variants": V_variants,
                "qualifications": {"PMF": pmf_quals, "V": V_quals},
                "passed": all(q["passed"] for q in pmf_quals + V_quals),
            }
            records.append(row)
            save(OUT / f"candidate_{index:02}.json", row)
            PROGRESS["completed_V_candidates"] += 1
            save(OUT / "PROGRESS.json", PROGRESS, False)
        account = planned_accounting()
        assert PROGRESS["completed_new_PMF_integrations"] == account["new_PMF_integrations"] == 3
        assert PROGRESS["completed_four_dimensional_nodes"] == account["new_four_dimensional_Gaussian_nodes"] == 3940352
        assert PROGRESS["completed_V_pair_rule_integrals"] == account["V_pair_rule_integrals"] == 630
        assert PROGRESS["completed_inner_pair_expectations"] == account["inner_pair_GH_expectations"] == 15120
        assert PROGRESS["completed_bivariate_nodes"] == account["new_bivariate_Gaussian_nodes"] == 17848320
        assert PROGRESS["completed_V_candidates"] == 35
        validate_freeze()
        result = {
            "schema": "scbm6eval-reference-results-v1", "source_sha256": sha(__file__),
            "root_freeze_sha256": sha(BASE / "FREEZE.json"), "freeze_sha256": sha(BASE / "FREEZE.json"),
            "config_sha256": sha(CONFIG),
            "candidates_sha256": sha(CANDIDATES), "helper_sha256": HELPER_SHA,
            "config": EXPECTED_REFERENCE, "patterns": patterns(), "pair_order": [list(p) for p in PAIRS],
            "V_center": center, "records": records, "qualification_passed": all(r["passed"] for r in records),
            "PMF_qualification_passed": all(q["passed"] for r in records for q in r["qualifications"]["PMF"]),
            "V_qualification_passed": all(q["passed"] for r in records for q in r["qualifications"]["V"]),
            "maximum_V_change": max(q["maximum_change"] for r in records for q in r["qualifications"]["V"]),
            "maximum_new_PMF_change": max(q["maximum_change"] for q in mean_quals),
            "accounting": account, "seconds": time.perf_counter()-started,
            "scope": "Frozen numerical PMF and unscaled limiting-potential references. V is not finite n/K expected score or a common normalization with score differences. Reference-order agreement is a numerical qualification, not a rigorous remainder certificate. No J_K calculation, added fitting, new test, or automatic refinement.",
        }
        save(OUT / "RESULTS.json", result)
        print(json.dumps({k: result[k] for k in ("qualification_passed", "maximum_V_change",
                                                "maximum_new_PMF_change", "accounting", "seconds")}), flush=True)
    except BaseException:
        if not (OUT / "FAILURE.json").exists():
            save(OUT / "FAILURE.json", {"utc": datetime.now(timezone.utc).isoformat(),
                "traceback": traceback.format_exc(), "progress": PROGRESS, "automatic_retry": False})
        raise


def check():
    # Structural, old-input, and hand-polynomial arithmetic only. No GH/GL nodes.
    cfg, selection = validate_candidates()
    assert sha(pmf_helper.__file__) == HELPER_SHA
    assert planned_accounting()["new_bivariate_Gaussian_nodes"] == 17848320
    x = [-1., 0., 1.]
    w = [1/3, 4/3, 1/3]
    for c in [-.8, 0., .8]:
        u = [.5*c*(z+1) for z in x]
        assert abs(outer_integral(c, w, u)-c*c/2) < 2e-16
    r = [.25, -.15, .1, .1, -.15, .25]
    assert abs(potential_value(r, [0.]*6, [v*v/2 for v in r]) + sum(v*v for v in r)/8) < 1e-16
    toy = [math.prod(p if y else 1-p for p, y in zip([.2, .4, .6, .8], label)) for label in patterns()]
    center = spin_center(toy)
    assert max(abs(x) for x in center["true_spin_covariances"]) < 2e-16
    assert max(abs(x-y) for x,y in zip(center["spin_means"],[-.6,-.2,.2,.6])) < 2e-16
    main, quals = pmf_qualifications([{"order": o, "pmf": toy} for o in [24, 32, 40]])
    assert main == toy and all(q["passed"] for q in quals)
    result = {
        "passed": True, "source_sha256": sha(__file__), "config_sha256": sha(CONFIG),
        "candidates_sha256": sha(CANDIDATES), "old_PMF_mappings_checked": 34,
        "candidate_covariances_checked": 35, "data_count_vectors_checked": 16,
        "hand_polynomial_integration_arithmetic": True, "positive_probability_arithmetic": True,
        "new_GH_or_GL_nodes_created": 0, "new_model_integrals": 0, "new_random_draws": 0,
        "planned_accounting": planned_accounting(),
    }
    save(OUT / "SELFTEST.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["schema", "check", "run"])
    mode = parser.parse_args().mode
    if mode == "schema":
        print(json.dumps({"source_sha256": sha(__file__), "reference": EXPECTED_REFERENCE,
                          "planned_accounting": planned_accounting()}))
    else:
        globals()[mode]()
