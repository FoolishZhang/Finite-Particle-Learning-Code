"""Independent reconstruction of GROWTH summaries from saved score arrays.

No production or summary function is imported. Gaussian inputs and scores are
not regenerated here. Three new deterministic J_K integrals use a different
identity from the production E-max tail-integral formula.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import time
import traceback

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.integrate import quad
from scipy.special import stdtrit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/regularization_learning/GROWTH"
REF = ROOT / "artifacts/regularization_learning/NUMERIC/GROWTH_REFERENCE.json"
PROTOCOL = OUT / "STATS_AUDIT_PROTOCOL.json"
DEST = OUT / "STATS_AUDIT.json"
TOL = 1e-11
JTOL = 1e-10


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text("utf-8"))


def save(p, x):
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                       encoding="utf-8", newline="\n")


def mean(values):
    return math.fsum(values) / len(values)


def stats(values, critical):
    mu = mean(values)
    se = math.sqrt(math.fsum((x - mu) ** 2 for x in values) / (len(values) * (len(values) - 1)))
    return dict(mean=mu, SE=se, lower=mu - critical * se, upper=mu + critical * se)


def first_min_index(values):
    return min(range(len(values)), key=lambda i: values[i])


def key(x):
    return (x["theta"], x["n"], x["method"])


def freeze():
    assert not PROTOCOL.exists() and not DEST.exists(), "Preserve existing audit files"
    results = read(OUT / "RESULTS.json")
    files = [ROOT / "script/reglearn_growth_summary.py", OUT / "SUMMARY.json",
             OUT / "SUMMARY_PROTOCOL.json", OUT / "J_REFERENCE.json", OUT / "RESULTS.json", REF]
    save(PROTOCOL, dict(
        schema="reglearn-growth-stats-audit-v1", source_sha256=sha(__file__),
        input_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in files},
        npz_sha256={r["file"]: r["file_sha256"] for r in results["records"]},
        statistics_and_point_absolute_tolerance=TOL, J_absolute_tolerance=JTOL,
        exact_checks="candidate argmin indices, all 360 qualification decisions and counts, risk-table lookups",
        recomputation="math.fsum from all NPZ mode scores, teacher weights, raw/marg/quadratic/informed stream curves, means/SE/t intervals, all fixed comparisons",
        tie_rule="smallest candidate index on exact tie",
        confidence="Student-t pointwise development intervals over 16 independent streams; no simultaneous or selected-minimum guarantee",
        t_critical_method="scipy.special.stdtrit(df=15, p=.975)",
        J_identity="K(K-1) integral_-12^12 phi(z)^2 Phi(z)^(K-2) dz; Phi via math.erfc",
        J_quadrature=dict(epsabs=1e-12, epsrel=1e-12, limit=200),
        J_integrals_planned=3,
        J_scope="alternative-identity numerical qualification plus elementary tail upper bound; not rigorous total quadrature certification",
        raw_integral_reference_audit="not repeated here; GROWTH_REFERENCE treated as a fixed table and independently reviewed elsewhere",
        preserve_failures=True, automatic_reruns=False,
        new_gaussian_samples=0, new_observed_samples=0, new_training_fits=0))
    print(json.dumps({"frozen": True, "protocol_sha256": sha(PROTOCOL)}), flush=True)


def run():
    assert not DEST.exists(), "Existing audit results must be preserved"
    protocol = read(PROTOCOL)
    assert protocol["source_sha256"] == sha(__file__)
    for p, digest in protocol["input_sha256"].items():
        assert sha(ROOT / p) == digest, ("input changed", p)
    started = time.perf_counter()
    comparisons, exact_checks, J_rows, panel_rows, gate_rows, file_rows = [], [], [], [], [], []
    new_integrals = 0
    mode_values_aggregated = 0
    error = None

    def compare(name, actual, expected, tolerance=TOL):
        aa, ee = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        assert aa.shape == ee.shape, (name, aa.shape, ee.shape)
        diff = np.abs(aa - ee)
        row = dict(name=name, count=int(diff.size), tolerance=tolerance,
                   max_absolute_error=float(diff.max()), failures=int(np.count_nonzero(diff > tolerance)))
        comparisons.append(row)

    def exact(name, actual, expected):
        exact_checks.append(dict(name=name, passed=bool(actual == expected)))

    try:
        summary, results, reference = read(OUT / "SUMMARY.json"), read(OUT / "RESULTS.json"), read(REF)
        J_reference = read(OUT / "J_REFERENCE.json")
        rho, ns, signs = results["rho"], results["ns"], results["signs"]
        assert len(results["records"]) == 16 and ns == [14, 62, 254]
        assert len(rho) == 21 and rho[10] == 0 and rho[-1] == .75
        assert summary["source_sha256"] == sha(ROOT / "script/reglearn_growth_summary.py")
        assert summary["production_sha256"] == sha(OUT / "RESULTS.json")
        assert summary["reference_sha256"] == sha(REF)
        assert summary["J_reference_sha256"] == sha(OUT / "J_REFERENCE.json")
        assert summary["protocol_sha256"] == sha(OUT / "SUMMARY_PROTOCOL.json")
        assert summary["independent_streams"] == 16
        compare("reference_candidate_axis", rho, reference["rho_grid21"])
        for ni, n in enumerate(ns):
            k = n + 2
            def integrand(x):
                phi_cdf = .5 * math.erfc(-x / math.sqrt(2))
                return k * (k - 1) * math.exp(-x*x) * phi_cdf**(k - 2) / (2 * math.pi)
            q = quad(integrand, -12., 12., epsabs=1e-12, epsrel=1e-12, limit=200, full_output=True)
            new_integrals += 1
            value, estimated_error, info = q[:3]
            message = None if len(q) == 3 else str(q[3])
            tail_upper = k * (k - 1) * math.exp(-144.) / (24 * math.pi)
            compare("J_K_%d" % k, value, J_reference["rows"][ni]["main"], JTOL)
            exact("J_K_%d_integrator_completion" % k, message, None)
            J_rows.append(dict(n=n, K=k, value=value, estimated_quadrature_error=estimated_error,
                               evaluations=int(info["neval"]), message=message,
                               omitted_two_tail_upper_bound=tail_upper))

        modes = {"raw": [], "marg": []}
        for rec in results["records"]:
            filename = ROOT / rec["file"]
            digest = sha(filename)
            exact("npz_sha_stream_%02d" % rec["stream"], digest, protocol["npz_sha256"][rec["file"]])
            file_rows.append(dict(file=rec["file"], sha256=digest))
            with np.load(filename, allow_pickle=False) as data:
                for name in ["raw", "marg"]:
                    a = data[name]
                    assert a.shape == (1024, 3, 21, 4) and np.isfinite(a).all()
                    out = [[[mean(a[:, ni, ri, si]) for si in range(4)]
                            for ri in range(21)] for ni in range(3)]
                    modes[name].append(out)
                    mode_values_aggregated += int(a.size)
        critical = float(stdtrit(15, .975))
        curves = {key(x): x for x in summary["curves"]}
        selections = {key(x): x for x in summary["selections"]}
        fixed = {key(x): x for x in summary["fixed_comparisons"]}
        assert len(curves) == len(selections) == len(fixed) == 36
        target_gates = {(g["theta"], g["n"], g["method"], g["left"], g["right"]): g
                        for g in summary["qualification"]["all"]}
        assert len(target_gates) == 360
        for teacher in reference["teachers"]:
            theta, moment = teacher["theta"], teacher["r_theta"]
            weights = [(1 + moment*s1*s2)/4 for s1,s2 in signs]
            risk = teacher["risk_grid21"]
            true_i = min(range(21), key=lambda i: abs(rho[i] - theta))
            assert abs(rho[true_i] - theta) < TOL
            for ni, n in enumerate(ns):
                w = J_rows[ni]["value"] / math.sqrt(n * reference["var_logsigmoid"])
                by = {}
                for name in ["raw", "marg"]:
                    by[name] = [[math.fsum(weights[s]*modes[name][b][ni][r][s] for s in range(4))
                                 for r in range(21)] for b in range(16)]
                by["quadratic"] = [[by["raw"][b][r] + w*reference["gamma0"]*rho[r]**2
                                    for r in range(21)] for b in range(16)]
                by["informed"] = [[by["raw"][b][r] + 2*reference["A"]*w*reference["H_grid21"][r]
                                   for r in range(21)] for b in range(16)]
                for name, x in by.items():
                    pkey = (theta, n, name)
                    expected_curve, expected_selection, expected_fixed = curves[pkey], selections[pkey], fixed[pkey]
                    per_point = [stats([x[b][r] for b in range(16)], critical) for r in range(21)]
                    curve = {field: [row[field] for row in per_point] for field in ["mean", "SE", "lower", "upper"]}
                    curve["normalized_mean"] = [value/w for value in curve["mean"]]
                    for field, values in curve.items():
                        compare(str(pkey) + " curve " + field, values, expected_curve[field])
                    compare(str(pkey) + " w", w, expected_curve["w_n"])
                    compare(str(pkey) + " selected-w", w, expected_selection["w_n"])
                    exact(str(pkey) + " curve risk mapping", risk, expected_curve["reference_joint_KL"])
                    idx = first_min_index(curve["mean"])
                    each = [first_min_index(x[b]) for b in range(16)]
                    actual_selection = dict(rho=rho[idx], reference_joint_KL=risk[idx],
                        mean_score_contrast=curve["mean"][idx],
                        mean_score_vs_truth=curve["mean"][idx] - curve["mean"][true_i],
                        stream_selected_rhos=[rho[i] for i in each],
                        mean_stream_selected_reference_KL=mean([risk[i] for i in each]))
                    exact(str(pkey) + " mean argmin candidate", rho[idx], expected_selection["rho"])
                    exact(str(pkey) + " stream argmin candidates", actual_selection["stream_selected_rhos"], expected_selection["stream_selected_rhos"])
                    exact(str(pkey) + " selected risk exact lookup", risk[idx], expected_selection["reference_joint_KL"])
                    for field in ["mean_score_contrast", "mean_score_vs_truth", "mean_stream_selected_reference_KL"]:
                        compare(str(pkey) + " selection " + field, actual_selection[field], expected_selection[field])
                    delta = [x[b][-1] - x[b][true_i] for b in range(16)]
                    actual_fixed = stats(delta, critical)
                    actual_fixed["normalized_mean"] = actual_fixed["mean"] / w
                    for field, value in actual_fixed.items():
                        compare(str(pkey) + " R_minus_truth " + field, value, expected_fixed[field])
                    panel_rows.append(dict(theta=theta, n=n, K=n+2, method=name, weights=weights,
                        w_n=w, curve=curve, selection=actual_selection, fixed_comparison=actual_fixed))
                    if name in ["raw", "marg"]:
                        for r in range(20):
                            gs = stats([x[b][r+1] - x[b][r] for b in range(16)], critical)
                            gkey = (theta, n, name, rho[r], rho[r+1])
                            expected_gate = target_gates[gkey]
                            passed = gs["SE"] <= 5e-4
                            compare(str(gkey) + " adjacent SE", gs["SE"], expected_gate["SE"])
                            exact(str(gkey) + " adjacent qualification", passed, expected_gate["pass_"])
                            gate_rows.append(dict(theta=theta, n=n, method=name, left=rho[r], right=rho[r+1], SE=gs["SE"], pass_=passed))
        exact("all panel count", len(panel_rows), 36)
        exact("all gate count", len(gate_rows), summary["qualification"]["checks"])
        exact("all gate pass count", sum(g["pass_"] for g in gate_rows), summary["qualification"]["passed"])
        failkeys = [tuple(g[k] for k in ["theta", "n", "method", "left", "right"]) for g in gate_rows if not g["pass_"]]
        expected_failkeys = [tuple(g[k] for k in ["theta", "n", "method", "left", "right"]) for g in summary["qualification"]["failed"]]
        exact("all failed gates retained", failkeys, expected_failkeys)
    except Exception:
        error = traceback.format_exc()
    failures = sum(x["failures"] for x in comparisons)
    exact_failures = sum(not x["passed"] for x in exact_checks)
    output = dict(status="passed" if error is None and failures == 0 and exact_failures == 0 else "failed",
        code_sha256=sha(__file__), protocol_sha256=sha(PROTOCOL), input_sha256=protocol["input_sha256"],
        elapsed_seconds=time.perf_counter() - started, t_critical_df15_975=float(stdtrit(15, .975)),
        numerical_comparisons=comparisons, exact_checks=exact_checks,
        numerical_failures=failures, exact_failures=exact_failures,
        npz_inputs=file_rows, alternative_J_integrals=J_rows,
        panels=panel_rows, adjacent_gates=gate_rows,
        ledger=dict(new_Gaussian_samples=0, new_observations=0, new_fits=0,
                    new_independent_J_integrals=new_integrals,
                    stored_mode_score_values_reaggregated=mode_values_aggregated,
                    independent_prior_streams_reused=16,
                    panels_reconstructed=len(panel_rows), adjacent_gates_reconstructed=len(gate_rows)),
        software_error=error,
        scope="Summary arithmetic from fixed NPZ/reference table plus 3 alternative-identity J_K integrals; numerical qualification, not rigorous integration remainder or exact statistical coverage")
    save(DEST, output)
    print(json.dumps({"status": output["status"], "numeric_failures": failures,
                      "exact_failures": exact_failures, "ledger": output["ledger"],
                      "output_sha256": sha(DEST)}), flush=True)
    if output["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["freeze", "run"])
    a = p.parse_args()
    freeze() if a.mode == "freeze" else run()
