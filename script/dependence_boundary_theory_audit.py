"""Independent frozen Gauss-Hermite audit of the population boundary formula.

Does not import the experiment implementation for mathematical integration.
The separate source checks import it only to compare its outputs to this audit.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import itertools
import json
import math

import numpy as np
from scipy.special import expit, roots_hermitenorm, logsumexp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/dependence_boundary_v1/THEORY_AUDIT"
THEORY = ROOT / "docs/08_research_研究工作/边际依赖接口_20260929/THEORY.md"
SYNTHETIC = ROOT / "script/dependence_boundary_synthetic.py"
CONTRACT = {
    "case": "A_noncentral_correlated",
    "states_lexicographic": [[0, 0], [0, 1], [1, 0], [1, 1]],
    "true_probabilities": [.14, .06, .16, .64],
    "true_marginals": [.8, .7],
    "covariance_direction": [[1., .9], [.9, 1.]],
    "t": [1e-4, 2e-4],
    "iid_K1_orders": [16, 24],
    "iid_K2_orders": [16, 24],
    "antithetic_K2_orders": [48, 80],
    "marginal_orders": [48, 80],
    "expected_derivative": {"iid_K1": .185, "iid_K2": .0565,
                            "antithetic_K2": -.072},
    "expected_marginal_KL_quadratic_coefficient": .0114,
    "derivative_absolute_tolerance": 1e-6,
    "quadrature_objective_absolute_tolerance": 1e-12,
    "marginal_coefficient_richardson_absolute_tolerance": 1e-6,
    "source_risk_t": [0., 1e-4, .1, 1., 16.],
    "source_risk_rho": [0., .05, .108, .25],
    "source_risk_GH_orders": [256, 400],
    "source_risk_absolute_tolerance": 1e-8,
    "source_components_t": [0., 1e-4, .1, 1.],
    "source_fixed_draws": [[-.8, .1, 1.2, -.4], [.5, -.5, 1.3, -1.3]],
    "source_components_absolute_tolerance": 1e-12,
    "source_components_derivative_absolute_tolerance": 1e-6,
    "failure_policy": "Preserve every planned output and failed gate; no added orders or parameter search.",
    "natural_data_used": False,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n",
                          encoding="utf-8", newline="\n")


def freeze():
    assert not (OUT / "FREEZE.json").exists(), "No overwrite of an existing freeze"
    write(OUT / "CONTRACT.json", CONTRACT)
    files = [Path(__file__), THEORY, SYNTHETIC, OUT / "CONTRACT.json"]
    write(OUT / "FREEZE.json", {
        "time": stamp(), "phase": "before_numerical_evaluation",
        "files": {p.relative_to(ROOT).as_posix(): sha(p) for p in files},
        "method": "Cartesian probabilists' Gauss-Hermite; exact sum over all true states",
        "source_review": "Independent equations, plus direct comparison to frozen source outputs",
    })
    print("frozen independent theory audit", flush=True)


def verify_freeze():
    data = json.loads((OUT / "FREEZE.json").read_text(encoding="utf-8"))
    for path, digest in data["files"].items():
        assert sha(ROOT / path) == digest, f"Frozen file changed: {path}"


def normal_tensor(order, dimension):
    nodes, weights = roots_hermitenorm(order)
    weights = weights / math.sqrt(2 * math.pi)
    grid = np.indices((order,) * dimension).reshape(dimension, -1).T
    return nodes[grid], np.prod(weights[grid], axis=1)


def population_objective(t, k, paired, order):
    # Separate integration dimensions for each iid latent vector, two per draw.
    dimension = 2 if paired else 2 * k
    z, w = normal_tensor(order, dimension)
    s = np.asarray(CONTRACT["covariance_direction"])
    chol = np.linalg.cholesky(s)
    if paired:
        one = z @ chol.T
        draws = np.stack([one, -one], axis=1)
    else:
        draws = z.reshape(-1, k, 2) @ chol.T
    p = np.asarray(CONTRACT["true_marginals"])
    mu = np.log(p) - np.log1p(-p)
    eta = mu + math.sqrt(t) * draws
    result = 0.
    state_risks = []
    for c, prob in zip(CONTRACT["states_lexicographic"],
                       CONTRACT["true_probabilities"]):
        sign = 2 * np.asarray(c) - 1
        # State likelihood inside the sample mean; no log of exact integral.
        loglik = -np.logaddexp(0., -eta * sign).sum(axis=2)
        nll = math.log(k) - logsumexp(loglik, axis=1)
        integrated = float(np.dot(w, nll))
        state_risks.append(integrated)
        result += prob * integrated
    return result, state_risks, float(w.sum())


def bernoulli_kl_from_small_delta(p, delta):
    # Analytic cancellation of the linear term avoids subtractive KL error.
    # The arguments here are O(1e-5); discarded series terms are negligible.
    return sum(((-1.) ** n / p ** (n - 1) + 1. / (1. - p) ** (n - 1))
               * delta ** n / n for n in range(2, 13))


def marginal_audit():
    p = np.asarray(CONTRACT["true_marginals"])
    mu = np.log(p / (1 - p))
    rows, extrapolated = [], []
    for order in CONTRACT["marginal_orders"]:
        z, w = roots_hermitenorm(order)
        w /= math.sqrt(2 * math.pi)
        ratios = []
        for t in CONTRACT["t"]:
            probs = w @ expit(mu[None, :] + math.sqrt(t) * z[:, None])
            delta = probs - p
            terms = [bernoulli_kl_from_small_delta(float(pp), float(dd))
                     for pp, dd in zip(p, delta)]
            ratio = float(sum(terms) / t ** 2)
            ratios.append(ratio)
            rows.append({"order": order, "t": t,
                         "probabilities": probs.tolist(),
                         "marginal_KL_terms": terms, "KL_over_t_squared": ratio})
        extrapolated.append({"order": order, "richardson_coefficient":
                             2 * ratios[0] - ratios[1],
                             "small_t_error": abs(ratios[0] - .0114),
                             "large_t_error": abs(ratios[1] - .0114),
                             "closer_at_smaller_t":
                             bool(abs(ratios[0] - .0114) < abs(ratios[1] - .0114))})
    return rows, extrapolated


def exact_symmetric_joint(t, order):
    z, w = roots_hermitenorm(order)
    w /= math.sqrt(2 * math.pi)
    pp = expit(math.sqrt(t) * z)
    return np.asarray([w @ ((1 - pp) ** 2), w @ ((1 - pp) * pp),
                       w @ (pp * (1 - pp)), w @ (pp ** 2)])


def explicit_conditional_risk(rho, model):
    # True state order 00,01,10,11. Query concept 1 and predict concept 2.
    truth = np.asarray([(1 + rho) / 4, (1 - rho) / 4,
                        (1 - rho) / 4, (1 + rho) / 4])
    marginal = np.asarray([model[0] + model[1], model[2] + model[3]])
    cond = np.asarray([model[i] / marginal[i // 2] for i in range(4)])
    return float(-truth @ np.log(cond))


def fixed_stream_objective(t, rho, z):
    # Direct four-state mixture evaluation; no A + rho B decomposition.
    p = expit(math.sqrt(t) * z)
    state_probs = np.stack([((1 - p) ** 2).mean(axis=1),
                            ((1 - p) * p).mean(axis=1),
                            (p * (1 - p)).mean(axis=1),
                            (p ** 2).mean(axis=1)], axis=1)
    truth = np.asarray([(1 + rho) / 4, (1 - rho) / 4,
                        (1 - rho) / 4, (1 + rho) / 4])
    return float(np.mean(-np.log(state_probs) @ truth))


def source_audit():
    spec = importlib.util.spec_from_file_location("frozen_synthetic_audited", SYNTHETIC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows, derivative_rows, risk_rows = [], [], []
    z = np.asarray(CONTRACT["source_fixed_draws"])
    for rho in CONTRACT["source_risk_rho"]:
        for t in CONTRACT["source_components_t"]:
            source = mod.components(t, z)
            direct = fixed_stream_objective(t, rho, z)
            rows.append({"rho": rho, "t": t, "direct_objective": direct,
                         "source_objective": float(source[0] + rho * source[1]),
                         "absolute_error": abs(float(source[0] + rho * source[1]) - direct)})
        # Independent right finite difference of four-state objective, not source gradient.
        h = 1e-4
        f0 = fixed_stream_objective(0., rho, z)
        fd = (4 * (fixed_stream_objective(h, rho, z) - f0)
              - (fixed_stream_objective(2 * h, rho, z) - f0)) / (2 * h)
        src = mod.components(0., z)
        derivative_rows.append({"rho": rho, "richardson_direct": fd,
                                "source_zero_derivative": float(src[2] + rho * src[3]),
                                "absolute_error": abs(fd - float(src[2] + rho * src[3]))})
        for t in CONTRACT["source_risk_t"]:
            src = mod.risks(rho, t)
            for order in CONTRACT["source_risk_GH_orders"]:
                joint = exact_symmetric_joint(t, order)
                conditional = explicit_conditional_risk(rho, joint)
                truth = np.asarray([(1 + rho) / 4, (1 - rho) / 4,
                                    (1 - rho) / 4, (1 + rho) / 4])
                joint_nll = float(-truth @ np.log(joint))
                risk_rows.append({"rho": rho, "t": t, "order": order,
                                  "independent_joint": joint.tolist(),
                                  "direct_conditional_NLL": conditional,
                                  "source_conditional_NLL": src["conditional_NLL"],
                                  "conditional_absolute_error": abs(conditional - src["conditional_NLL"]),
                                  "joint_absolute_error": abs(joint_nll - src["joint_NLL"]),
                                  "source_quadrature_delta": src["quadrature_delta"]})
    return {"frozen_source_sha256": sha(SYNTHETIC), "components": rows,
            "zero_derivatives": derivative_rows, "risks": risk_rows,
            "review": [
                "components(t=0): exact label averaging cancels the square-root term. da=mean(z)^2/4 and db=(mean(z)^2-mean(z^2))/4 are correct for the shared rank-one latent family.",
                "The conditional_NLL formula uses true same/different probabilities (1+rho)/2 and (1-rho)/2, and model probabilities derived from E[tanh(sqrt(t) Z/2)^2]; it matches exact binary event conditioning.",
                "The symmetric rank-one family preserves .5 marginals for all t and does not test the noncentral marginal incompatibility. It is appropriate for isolating the finite-K mechanism.",
                "A local boundary derivative is not a certificate that the grid/bounded optimizer found a global minimum. The source explicitly disclaims such a certificate.",
                "New QMC streams measure integration sensitivity; they are not independent natural-data model replications or a frequentist coverage guarantee.",
                "The independent risk formula check covers only the stipulated rank-one symmetric family, not arbitrary multi-concept Gaussian feedback.",
            ]}


def run():
    verify_freeze()
    assert not (OUT / "RESULTS.json").exists(), "Do not overwrite prior audit results"
    rows, derivative_rows = [], []
    p = np.asarray(CONTRACT["true_marginals"])
    f0 = float(-np.sum(p * np.log(p) + (1 - p) * np.log1p(-p)))
    for label, k, paired in [("iid_K1", 1, False), ("iid_K2", 2, False),
                             ("antithetic_K2", 2, True)]:
        for order in CONTRACT[label + "_orders"]:
            values = []
            for t in CONTRACT["t"]:
                value, states, weight_sum = population_objective(t, k, paired, order)
                values.append(value)
                rows.append({"sampler": label, "K": k, "order": order,
                             "integration_dimension": 2 if paired else 2 * k,
                             "t": t, "population_NLL": value,
                             "state_integrated_NLL": states,
                             "weight_sum": weight_sum, "first_difference": (value - f0) / t})
            h = CONTRACT["t"][0]
            richardson = (4 * (values[0] - f0) - (values[1] - f0)) / (2 * h)
            expected = CONTRACT["expected_derivative"][label]
            derivative_rows.append({"sampler": label, "order": order,
                                    "richardson_derivative": richardson,
                                    "expected": expected,
                                    "absolute_error": abs(richardson - expected)})
    marginal_rows, coefficients = marginal_audit()
    source = source_audit()
    quadrature_deltas = []
    for label in CONTRACT["expected_derivative"]:
        for t in CONTRACT["t"]:
            match = [r for r in rows if r["sampler"] == label and r["t"] == t]
            quadrature_deltas.append({"sampler": label, "t": t,
                                      "absolute_objective_difference": abs(match[0]["population_NLL"] - match[1]["population_NLL"])})
    checks = {
        "all_population_derivatives": all(r["absolute_error"] <= 1e-6 for r in derivative_rows),
        "all_GH_order_objective_agreement": all(r["absolute_objective_difference"] <= 1e-12 for r in quadrature_deltas),
        "marginal_KL_coefficient": all(abs(r["richardson_coefficient"] - .0114) <= 1e-6 for r in coefficients),
        "marginal_KL_two_level_convergence": all(r["closer_at_smaller_t"] for r in coefficients),
        "source_components": all(r["absolute_error"] <= 1e-12 for r in source["components"]),
        "source_zero_derivative": all(r["absolute_error"] <= 1e-6 for r in source["zero_derivatives"]),
        "source_conditional_risk": all(max(r["conditional_absolute_error"], r["joint_absolute_error"]) <= 1e-8 for r in source["risks"]),
    }
    result = {"time": stamp(), "baseline_population_NLL": f0,
              "finite_K_integrals": rows, "derivatives": derivative_rows,
              "quadrature_deltas": quadrature_deltas, "marginal_KL": marginal_rows,
              "marginal_coefficients": coefficients, "checks": checks,
              "pass": all(checks.values()), "natural_data_used": False,
              "no_parameter_or_order_selection": True}
    write(OUT / "RESULTS.json", result)
    write(OUT / "SOURCE_REVIEW.json", source)
    write(OUT / "COMPLETE.json", {"time": stamp(), "pass": all(checks.values()),
                                   "files": {p.name: sha(p) for p in
                                             [OUT / "RESULTS.json", OUT / "SOURCE_REVIEW.json"]}})
    print(json.dumps({"checks": checks, "derivatives": derivative_rows,
                      "marginal_coefficients": coefficients}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["freeze", "run"])
    args = parser.parse_args()
    globals()[args.phase]()
