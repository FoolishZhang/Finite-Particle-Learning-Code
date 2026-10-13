"""Bounded deterministic calibration; no data/bank/NPZ/optimizer access.

--run produces a new scalar receipt once. --check only replays algebra and
eigenvalue identities from saved quadrature scalars; it does not reintegrate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.special import log_ndtr, roots_hermitenorm, roots_legendre

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "artifacts/submission_candidate_v1_7_1_20261008"
CONTRACT = DEST / "CALIBRATION_CONTRACT.json"
RESULT = DEST / "CALIBRATION_RESULTS.json"
MANIFEST = DEST / "CALIBRATION_MANIFEST.json"
GAP = "artifacts/nuisance_contrast_v1_20261006/POPULATION.json"
CURV = "artifacts/local_geometry_v1_20261006/STATISTICS.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sources(contract):
    for rel, expected in contract["source_hashes"].items():
        assert sha(ROOT / rel) == expected, f"changed source: {rel}"
    gap = next(r for r in read(ROOT / GAP)["results"] if r["contrast"] == "g256")
    curv = next(r for r in read(ROOT / CURV)["results"] if r["contrast"] == "h256")
    return {"gap": {k: gap[k] for k in ("mean", "interval", "numerical_allowance")},
            "curvature": {k: curv[k] for k in ("mean", "interval", "numerical_allowance")},
            "qualification": "Separate 99% families; together at least 98% under original iid and numerical conditions. No new statistical family."}


def normal_rule(rule):
    if rule["normal"] == "Gauss-Hermite":
        x, w = roots_hermitenorm(rule["normal_order"])
        return x, w / np.sqrt(2 * np.pi)
    x, w = roots_legendre(rule["normal_order"])
    lo, hi = rule["normal_interval"]
    x = (hi + lo) / 2 + (hi - lo) / 2 * x
    w = w * (hi - lo) / 2 * np.exp(-x*x/2) / np.sqrt(2*np.pi)
    return x, w  # Do not renormalize truncated Gaussian mass.


def coefficient_rule(rule, rho):
    x, w = normal_rule(rule)
    ell = -np.logaddexp(0, -x)
    mean = float(w @ ell)
    second = float(w @ (ell*ell))
    sech = float(w @ (1/np.cosh(x/2)**2))
    t, tw = roots_legendre(rule["correlation_order"])
    corr = (t+1)*rho/2
    cw = tw*rho/2
    values = []
    for u in corr:
        if rule["normal"] == "Gauss-Hermite":
            a = x[:, None]
            b = u*a + np.sqrt(1-u*u)*x[None, :]
        else:
            # Independent symmetric square root, not the primary Cholesky factor.
            a = np.sqrt((1+u)/2)*x[:, None] + np.sqrt((1-u)/2)*x[None, :]
            b = np.sqrt((1+u)/2)*x[:, None] - np.sqrt((1-u)/2)*x[None, :]
        values.append(float(w @ (np.tanh(a/2)*np.tanh(b/2)) @ w))
    integral = float(cw @ np.asarray(values))
    return {"id": rule["id"], "normal_mass": float(w.sum()), "logsigmoid_mean": mean,
            "logsigmoid_second_moment": second, "v": second-mean*mean,
            "mean_sech_squared": sech, "kappa_prime_zero": sech*sech/4,
            "correlation_nodes": corr.tolist(), "correlation_weights": cw.tolist(),
            "kappa_at_nodes": values, "integral_kappa_0_to_rho": integral,
            "V_Cb": -3*integral/4,
            "work": {"scalar_integrals": 3 + len(corr) + 1,
                     "normal_moment_scalar_node_evaluations": 3*len(x),
                     "bivariate_scalar_node_evaluations": len(corr)*len(x)**2,
                     "outer_correlation_nodes": len(corr)}}


def j_pair(rule, K):
    x, w = roots_legendre(rule["order"])
    lo, hi = rule["interval"]
    x = (hi+lo)/2 + (hi-lo)/2*x
    y = np.exp(-x*x - np.log(2*np.pi) + (K-2)*log_ndtr(x))
    val = float(K*(K-1)*(hi-lo)/2*(w@y))
    return {"order":rule["order"], "J":val, "I":val/(K*(K-1)),
            "work":{"scalar_integrals":1,"scalar_node_evaluations":len(x)}}


def independent_1d(contract, K):
    rule = contract["independent_1d_check"]
    norm = lambda x: np.exp(-x*x/2)/np.sqrt(2*np.pi)
    ell = lambda x: -np.logaddexp(0, -x)
    functions = {
        "logsigmoid_mean": lambda x: norm(x)*ell(x),
        "logsigmoid_second_moment": lambda x: norm(x)*ell(x)**2,
        "mean_sech_squared": lambda x: norm(x)*4*np.exp(-abs(x))/(1+np.exp(-abs(x)))**2,
        "J_maximum_density": lambda x: x*K*norm(x)*np.exp((K-1)*log_ndtr(x)),
    }
    result = {}
    for name, fn in functions.items():
        output = quad(fn, -np.inf, np.inf, epsabs=rule["epsabs"],
                      epsrel=rule["epsrel"], limit=rule["limit"], full_output=1)
        assert len(output) == 3, (name, output[3:])
        value, error, info = output
        result[name] = {"value":float(value), "nonrigorous_error_estimate":float(error),
                        "integrand_evaluations":int(info["neval"])}
    return result


def weak_root(angles):
    angles = np.asarray(angles)
    m = len(angles)
    cos = np.cos(angles[:,None]-angles[None,:])
    np.fill_diagonal(cos, 0)
    L = np.diag(cos.sum(axis=1))-cos
    # Explicit orthonormal Helmert basis for common-rotation complement.
    B = np.zeros((m,m-1))
    for j in range(1,m):
        B[:j,j-1] = 1/np.sqrt(j*(j+1))
        B[j,j-1] = -j/np.sqrt(j*(j+1))
    eigenvalues, vectors = np.linalg.eigh(B.T@L@B)
    lam = float(eigenvalues[0])
    root = 2*m/(m-lam)
    d = B@vectors[:,0]
    a = (root-1)*(root-2)/(32*root**2)
    b = (root-1)/(16*root)
    residual = float(np.linalg.norm(b*L@d - 2*a*m*d))
    return {"angles":angles.tolist(), "row_lengths": [1.0]*m, "L_cos":L.tolist(),
            "projected_eigenvalues":eigenvalues.tolist(), "lambda_min":lam,
            "formal_K_root":root, "mode":d.tolist(), "root_residual":residual,
            "rotation_residual":float(np.linalg.norm(L@np.ones(m))),
            "formula":"H_leading/epsilon^4=b_K L_cos-2a_K m I on 1-perp; K_root=2m/(m-lambda_min)",
            "square_completion":"b_K/(2a_K)=K/(K-2), K>2; candidate-angle-independent constant absorbed"}


def derived(primary, J, weak, old):
    w = J/np.sqrt(256*primary["v"])
    gap = -w*primary["V_Cb"]
    formal = -w*primary["kappa_prime_zero"]/4
    return {"v":primary["v"], "J100":J, "w256100":float(w), "V_Cb":primary["V_Cb"],
            "leading_gap":float(gap), "kappa_prime_zero":primary["kappa_prime_zero"],
            "formal_curvature":float(formal), "finite_gap_mean":old["gap"]["mean"],
            "finite_gap_interval":old["gap"]["interval"],
            "finite_curvature_mean":old["curvature"]["mean"],
            "finite_curvature_interval":old["curvature"]["interval"],
            "finite_gap_over_leading":float(old["gap"]["mean"]/gap),
            "leading_over_finite_gap":float(gap/old["gap"]["mean"]),
            "gap_relative_discrepancy_to_finite_mean":float(gap/old["gap"]["mean"]-1),
            "gap_relative_discrepancy_to_leading":float(1-old["gap"]["mean"]/gap),
            "formal_over_finite_curvature_magnitude":float(formal/old["curvature"]["mean"]),
            "formal_weak_K_root":weak["formal_K_root"],
            "leading_gap_above_old_upper":bool(gap>old["gap"]["interval"][1]),
            "formal_curvature_below_old_lower":bool(formal<old["curvature"]["interval"][0])}


def sensitivity(rows, js, independent, weak, contract):
    primary = rows[2]
    absolute = {key:max(abs(r[key]-primary[key]) for r in rows) for key in
                ("v","V_Cb","kappa_prime_zero")}
    vcheck = independent["logsigmoid_second_moment"]["value"]-independent["logsigmoid_mean"]["value"]**2
    kpcheck = independent["mean_sech_squared"]["value"]**2/4
    absolute["v"] = max(absolute["v"],abs(vcheck-primary["v"]))
    absolute["kappa_prime_zero"] = max(absolute["kappa_prime_zero"],abs(kpcheck-primary["kappa_prime_zero"]))
    absolute["J"] = max(*(abs(r["J"]-js[-1]["J"]) for r in js),
                        abs(independent["J_maximum_density"]["value"]-js[-1]["J"]))
    gates = contract["numerical_sensitivity_gates"]
    passed = all(absolute[key] <= gates[gate] for key,gate in
                 (("v","v_absolute"),("V_Cb","V_absolute"),("kappa_prime_zero","kappa_prime_absolute"),("J","J_absolute")))
    passed = passed and weak["root_residual"] <= gates["weak_root_eigen_residual"]
    return {"max_absolute_rule_differences":absolute, "passed":bool(passed),
            "interpretation":"Sensitivity gates and adaptive error estimates are numerical diagnostics, not rigorous integration bounds."}


def run():
    assert not RESULT.exists() and not MANIFEST.exists(), "Refusing to overwrite existing calibration output"
    contract = read(CONTRACT)
    old = sources(contract)
    rows = [coefficient_rule(rule, contract["scope"]["Cb_offdiagonal"]) for rule in contract["rules"]]
    js = [j_pair(rule,100) for rule in contract["J_pair_rules"]]
    independent = independent_1d(contract,100)
    weak = weak_root(contract["scope"]["weak_teacher_angles"])
    checks = sensitivity(rows,js,independent,weak,contract)
    assert checks["passed"], checks
    budget = {"scalar_integrals":sum(r["work"]["scalar_integrals"] for r in rows+js)+len(independent),
              "fixed_scalar_integrand_node_evaluations":sum(sum(v for k,v in r["work"].items() if k != "scalar_integrals") for r in rows+js),
              "adaptive_scalar_integrand_evaluations":sum(r["integrand_evaluations"] for r in independent.values()),
              "new_random_samples":0,"new_training_steps":0,"NPZ_reads":0,"new_banks":0,"K_scan":False,
              "counting":"Each 2D Gaussian expectation at a correlation node counts as an integral; outer weighted correlation sums count separately. Scalar moment integrands share nodes. Outer nodes are counted in fixed total."}
    result = {"schema_version":1, "created_at":datetime.now(timezone.utc).isoformat(),
              "contract_sha256":sha(CONTRACT),"script_sha256":sha(__file__),
              "environment":{"python":sys.version,"numpy":np.__version__,"scipy":scipy.__version__},
              "sources":contract["source_hashes"],"old_public_statistics":old,
              "coefficient_rules":rows,"J_pair_rules":js,"independent_1d":independent,
              "weak_loading":weak,"derived":derived(rows[2],js[-1]["J"],weak,old),
              "sensitivity":checks,"actual_work":budget,
              "units":"Natural-log gap: nats per complete observation. h: nats per squared free correlation coordinate. v: squared natural-log units. Weak root: dimensionless K; loading-angle Hessian uses radians and common epsilon^4.",
              "limitations":contract["qualification"]}
    RESULT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    MANIFEST.write_text(json.dumps({"contract":sha(CONTRACT),"script":sha(__file__),"results":sha(RESULT)},indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"passed":True,"derived":result["derived"],"actual_work":budget,"sensitivity":checks},indent=2))


def check():
    contract = read(CONTRACT)
    manifest = read(MANIFEST)
    for key,path in (("contract",CONTRACT),("script",Path(__file__)),("results",RESULT)):
        assert sha(path)==manifest[key], f"changed {key}"
    saved = read(RESULT)
    old = sources(contract)
    assert old == saved["old_public_statistics"]
    for row in saved["coefficient_rules"]:
        assert np.isclose(row["v"],row["logsigmoid_second_moment"]-row["logsigmoid_mean"]**2,rtol=0,atol=1e-15)
        assert np.isclose(row["kappa_prime_zero"],row["mean_sech_squared"]**2/4,rtol=0,atol=1e-15)
        integral = np.asarray(row["correlation_weights"])@np.asarray(row["kappa_at_nodes"])
        assert np.isclose(row["V_Cb"],-3*integral/4,rtol=0,atol=1e-15)
    weak = weak_root(contract["scope"]["weak_teacher_angles"])
    for key in ("formal_K_root","lambda_min","root_residual"):
        assert np.isclose(weak[key],saved["weak_loading"][key],rtol=0,atol=1e-13)
    actual = derived(saved["coefficient_rules"][2],saved["J_pair_rules"][-1]["J"],weak,old)
    for key,val in actual.items():
        assert np.allclose(val,saved["derived"][key],rtol=0,atol=1e-13), key
    replay_sensitivity = sensitivity(saved["coefficient_rules"],saved["J_pair_rules"],saved["independent_1d"],weak,contract)
    assert replay_sensitivity == saved["sensitivity"] and replay_sensitivity["passed"]
    print(json.dumps({"passed":True,"mode":"saved scalar algebra and source replay","new_integrals":0,"NPZ_reads":0,"new_random_samples":0,"new_training_steps":0,"run_integrals":saved["actual_work"]["scalar_integrals"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run",action="store_true")
    mode.add_argument("--check",action="store_true")
    args = parser.parse_args()
    run() if args.run else check()
