"""Independent complete active-statistic audit and bounded noise-prefix replay.

No producer computation functions are imported. RNG calls replay frozen inputs;
they are not new independent scientific samples. Full billion-scalar nuisance
replay is explicitly outside scope.
"""
from __future__ import annotations
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
import time
import traceback
import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
MC = ROOT/"artifacts/nuisance_dimensions/MC"
OUT = ROOT/"artifacts/nuisance_dimensions/MC_AUDIT"
SCOPE = {
    "streams": 8, "banks_per_stream": 8192, "K": 4,
    "ns": [0, 14, 62, 254, 1022, 4094], "rhos": [.3, .75],
    "all_saved_active_statistics_rebuilt": True,
    "active_and_uniform_RNG_replay": "all stored banks",
    "nuisance_RNG_replay_banks_per_stream": 64,
    "stored_raw_noise_banks_per_stream_checked": 2,
    "nuisance_summaries_rebuilt_banks_per_stream": 64,
    "all_nuisance_Gaussian_scalars_replayed": False,
    "statistics_absolute_tolerance": 1e-10,
    "summary_absolute_tolerance": 1e-12,
    "prefix_sum_tolerance": 1e-9,
    "new_independent_scientific_samples": 0,
    "new_integrals": 0, "new_fits": 0,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def serial(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x).__name__)


def save(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, default=serial, allow_nan=False)
        handle.write("\n")


def utc():
    return datetime.now(timezone.utc).isoformat()


class Audit:
    def __init__(self):
        self.checks = 0
        self.maximum = {}

    def test(self, condition, message):
        self.checks += 1
        if not condition:
            raise AssertionError(message)

    def close(self, x, y, category, tolerance=None):
        tolerance = SCOPE["summary_absolute_tolerance"] if tolerance is None else tolerance
        x, y = np.asarray(x), np.asarray(y)
        self.test(x.shape == y.shape, category+" shape")
        self.test(np.isfinite(x).all() and np.isfinite(y).all(), category+" finite")
        delta = float(np.max(np.abs(x-y))) if x.size else 0.
        self.maximum[category] = max(self.maximum.get(category, 0.), delta)
        self.test(delta <= tolerance, f"{category}: {delta} > {tolerance}")
        return delta


def verify(files, audit):
    for name, digest in files.items():
        audit.test(sha(ROOT/name) == digest, "hash changed: "+name)


def probabilities(logits):
    values = np.exp(logits-np.max(logits, axis=1, keepdims=True))
    return values/np.sum(values, axis=1, keepdims=True)


def independent_statistics(active, nuisance, rho, a=1.):
    """Full spin-label enumeration; independent softmax and residual forms."""
    banks, K, _ = active.shape
    rho = np.asarray(rho, dtype=float)
    if rho.ndim == 0:
        rho = np.full(banks, rho)
    rr = rho[:, None]
    z1, z2 = active[:, :, 0], active[:, :, 1]
    root = np.sqrt(1-rr*rr)
    eta1, eta2 = a*z1, a*(rr*z1+root*z2)
    velocity = a*(z1-rr/root*z2)
    noise = nuisance-nuisance.max(axis=1, keepdims=True)
    no_active = probabilities(noise)
    price, path = np.zeros(banks), np.zeros(banks)
    for y1, y2 in itertools.product((-1., 1.), repeat=2):
        loglike = -np.logaddexp(0., -y1*eta1)-np.logaddexp(0., -y2*eta2)
        pi = probabilities(noise+loglike)
        residual1, residual2 = y1*expit(-y1*eta1), y2*expit(-y2*eta2)
        diagonal = np.empty_like(pi)
        for k in range(K):
            # Sum off-diagonal softmax products instead of forming 1-pi_k.
            diagonal[:, k] = pi[:, k]*np.sum(pi[:, [j for j in range(K) if j != k]], axis=1)
        price -= .25*a*a*np.sum(diagonal*residual1*residual2, axis=1)
        path += .25*np.sum((no_active-pi)*residual2*velocity, axis=1)
    return price, path


def independent_interval(values, bound, alpha, family):
    values = np.asarray(values, dtype=float)
    M = len(values)
    mean = math.fsum(float(x) for x in values)/M
    variance = math.fsum((float(x)-mean)**2 for x in values)/(M-1)
    logarithm = math.log(4*family/alpha)
    radius = math.sqrt(2*variance*logarithm/M)+7*(2*bound)*logarithm/(3*(M-1))
    return {"mean": mean, "variance": variance, "samples": M, "absolute_bound": bound,
            "halfwidth": radius, "low": mean-radius, "high": mean+radius,
            "family": family, "alpha": alpha}


def execute():
    audit, started = Audit(), time.perf_counter()
    audit.test(read(MC/"STATE.json")["status"] == "complete", "MC must be complete")
    audit.test(not (OUT/"START.json").exists(), "existing attempt retained; no silent restart")
    freeze, seal = read(MC/"FREEZE.json"), read(MC/"RESULTS_FREEZE.json")
    verify(freeze["files"], audit)
    verify(seal["files"], audit)
    audit.test(seal["source_freeze_sha256"] == sha(MC/"FREEZE.json"), "source/result chain")
    cfg = freeze["config"]
    for key in ("streams", "banks_per_stream", "K", "ns", "rhos"):
        audit.test(cfg[key] == SCOPE[key], "frozen scope "+key)
    audit.test(cfg["a"] == 1. and cfg["theta"] == 0. and cfg["R"] == .75, "fixed physical model")
    audit.test(cfg["roles"] == {"active": 0, "uniform_path": 1, "nuisance": 2}, "role contract")
    audit.test(cfg["chunk"] == 64 and cfg["raw_noise_prefix_banks"] == 2, "prefix contract")
    audit.test(cfg["family"] == 18 and cfg["alpha"] == .05, "statistical family")
    audit.test(cfg["price_abs_bound"] == .25 and cfg["gap_abs_bound"] == .1875, "range contract")
    public = {**freeze["files"], **{p: h for p, h in seal["files"].items() if not p.endswith(".npz")},
              rel(MC/"RESULTS_FREEZE.json"): sha(MC/"RESULTS_FREEZE.json"),
              rel(Path(__file__).resolve()): sha(Path(__file__))}
    arrays = {p: h for p, h in seal["files"].items() if p.endswith(".npz")}
    save(OUT/"START.json", {"created": utc(), "source_sha256": sha(Path(__file__)),
         "scope": SCOPE, "public_files": public, "workspace_arrays": arrays,
         "MC_source_freeze_sha256": sha(MC/"FREEZE.json"),
         "MC_results_freeze_sha256": sha(MC/"RESULTS_FREEZE.json"),
         "before_audit_RNG_replay": True})
    try:
        all_price, all_path, all_gap, summaries = [], [], [], []
        B, K, maxn = cfg["banks_per_stream"], cfg["K"], max(cfg["ns"])
        replay_normals = replay_uniforms = 0
        for stream in range(cfg["streams"]):
            folder = MC/f"s{stream}"
            summary = read(folder/"SUMMARY.json")
            verify(summary["files"], audit)
            with np.load(folder/"INPUT.npz") as data:
                active, uniform, noise_sums, raw_prefix = [data[name].copy() for name in
                    ("active", "uniform", "nuisance_sums", "noise_prefix")]
            with np.load(folder/"STATISTICS.npz") as data:
                stored_price, stored_path, stored_gap = [data[name].copy() for name in
                    ("price", "path_gradient", "gap")]
            audit.test(active.shape == (B, K, 2) and uniform.shape == (B,), "active input shape")
            audit.test(noise_sums.shape == (B, 6, K) and raw_prefix.shape == (2, K, maxn), "noise input shape")
            audit.test(np.isfinite(active).all() and np.isfinite(noise_sums).all(), "finite input")
            audit.test(np.all((uniform >= 0)&(uniform < 1)), "uniform domain")
            audit.close(noise_sums[:, 0], np.zeros((B, K)), "n0_noise", 0)
            audit.test(np.all(np.diff(noise_sums, axis=1) <= 0), "all stored negative prefix increments")
            make_rng = lambda role: np.random.Generator(np.random.PCG64(np.random.SeedSequence([cfg["seed"], stream, role])))
            audit.close(active, make_rng(0).standard_normal((B, K, 2)), "RNG_active_all", 0)
            audit.close(uniform, make_rng(1).random(B), "RNG_uniform_all", 0)
            first_chunk = make_rng(2).standard_normal((64, K, maxn))
            audit.close(raw_prefix, first_chunk[:2], "RNG_saved_noise_first2", 0)
            replay_prefix = np.cumsum(-np.logaddexp(0., -first_chunk), axis=2)
            for ni, n in enumerate(cfg["ns"]):
                rebuilt = np.zeros((64, K)) if n == 0 else replay_prefix[:, :, n-1]
                audit.close(rebuilt, noise_sums[:64, ni], "RNG_noise_first64_sums", SCOPE["prefix_sum_tolerance"])
            replay_normals += B*K*2+64*K*maxn
            replay_uniforms += B
            price = np.empty((B, 6, 2)); path = np.empty_like(price); gap = np.empty((B, 6))
            for ni, n in enumerate(cfg["ns"]):
                for ri, rho in enumerate(cfg["rhos"]):
                    price[:, ni, ri], path[:, ni, ri] = independent_statistics(active, noise_sums[:, ni], rho, cfg["a"])
                gap[:, ni] = cfg["R"]*independent_statistics(active, noise_sums[:, ni], cfg["R"]*uniform, cfg["a"])[0]
            differences = {
                "Price": audit.close(price, stored_price, "all_active_Price", SCOPE["statistics_absolute_tolerance"]),
                "Path": audit.close(path, stored_path, "all_active_Path", SCOPE["statistics_absolute_tolerance"]),
                "H": audit.close(gap, stored_gap, "all_active_H", SCOPE["statistics_absolute_tolerance"]),
            }
            tol = cfg["numerical_bound_tolerance"]
            audit.test(np.isfinite(stored_path).all(), "stored auxiliary finite")
            for value, bound, name in ((price, .25, "Price rebuilt"), (stored_price, .25, "Price stored"),
                                       (gap, .1875, "H rebuilt"), (stored_gap, .1875, "H stored")):
                audit.test(np.max(np.abs(value)) <= bound+tol, "range "+name)
            for key, value in (("price_mean", price.mean(axis=0)), ("path_mean", path.mean(axis=0)),
                               ("gap_mean", gap.mean(axis=0)), ("max_abs_price", float(np.max(abs(price)))),
                               ("max_abs_gap", float(np.max(abs(gap))))):
                audit.close(value, summary[key], "stream_summary_"+key)
            all_price.append(price); all_path.append(path); all_gap.append(gap)
            record = {"stream": stream, "passed": True, "all_active_banks": B, "differences": differences,
                      "noise_raw_banks_compared": 2, "noise_summary_banks_replayed": 64,
                      "all_nuisance_Gaussians_replayed": False}
            summaries.append(record)
            save(OUT/f"s{stream}.json", record)
            print(f"MC audit stream {stream} passed", flush=True)
        price, path, gap = np.concatenate(all_price), np.concatenate(all_path), np.concatenate(all_gap)
        results = read(MC/"RESULTS.json")
        original = {(r["kind"], r["n"], r["rho"]): r for r in results["records"]}
        expected_keys = {(kind, n, rho) for n in cfg["ns"] for kind, rho in
                         (("gradient", .3), ("gradient", .75), ("gap_R_minus_0", .75))}
        audit.test(len(results["records"]) == len(original) == 18 and set(original) == expected_keys, "all 18 quantities")
        gates, interval_rows, certificate_rows = [], [], []
        for ni, n in enumerate(cfg["ns"]):
            scale = math.sqrt(max(1, n))
            for kind, rho, values, bound in (("gradient", .3, price[:, ni, 0], .25),
                                            ("gradient", .75, price[:, ni, 1], .25),
                                            ("gap_R_minus_0", .75, gap[:, ni], .1875)):
                current = independent_interval(values, bound, .05, 18)
                saved = original[(kind, n, rho)]
                for key in ("mean", "variance", "samples", "absolute_bound", "halfwidth", "low", "high", "alpha", "family"):
                    audit.close(current[key], saved[key], "interval_"+key)
                for key in ("mean", "low", "high"):
                    audit.close(scale*current[key], saved["scaled_"+key], "scaled_interval_"+key)
                audit.test(saved["t_approximation"] is False, "not a t interval")
                expected_primary = n in (0, maxn)
                audit.test(("main_gate" in saved) == expected_primary, "original main membership")
                if expected_primary:
                    passed = current["low"] > 0 if n == 0 else current["high"] < 0
                    audit.test(passed == saved["main_gate"], "original main decision")
                    gates.append(bool(passed))
                interval_rows.append({"kind": kind, "n": n, "rho": rho, **current})
                if kind == "gap_R_minus_0":
                    eta = max(0., -current["high"])
                    risk = (2/9)*eta*eta
                    cert = next(x for x in results["finite_n_risk_certificates"] if x["n"] == n)
                    audit.close([eta, risk], [cert["eta"], cert["exact_min_joint_KL_lower"]], "finite_n_certificate")
                    audit.test(cert["sampling_event_only"] and cert["not_the_asymptotic_endpoint_bound"]
                               and cert["numerical_rounding_not_interval_certified"], "certificate scope")
                    certificate_rows.append({"n": n, "eta": eta, "joint_KL_lower_exact_min": risk})
            for ri, rho in enumerate(cfg["rhos"]):
                expected = next(x for x in results["estimator_diagnostics"] if x["n"] == n and x["rho"] == rho)
                difference = path[:, ni, ri]-price[:, ni, ri]
                values = [price[:, ni, ri].mean(), path[:, ni, ri].mean(), difference.mean(), difference.std(ddof=1)/math.sqrt(len(difference))]
                saved = [expected[k] for k in ("Price_mean", "path_mean", "path_minus_Price_mean", "path_minus_Price_SE")]
                audit.close(values, saved, "auxiliary_path_diagnostics")
                audit.test(expected["descriptive_only_no_new_gate"], "auxiliary no extra gate")
        audit.test(gates == results["main_gates"] and len(gates) == results["main_total"] == 6, "six primary outcomes")
        audit.test(sum(gates) == results["main_passed"], "primary pass count")
        M = 65536
        ledger = {"independent_banks": M, "seed_groups": 8, "role_rng_streams": 24,
                  "prior_vectors": M*4, "Gaussian_scalars": M*4*4096,
                  "uniform_path_parameters": M, "nested_n_bank_evaluations": M*6,
                  "new_labels": 0, "new_fits": 0, "new_NN": 0}
        audit.test(results["ledger"] == ledger, "full production ledger")
        old = read(ROOT/"artifacts/weak_emission/NUMERIC/GRID.json")["rows"]
        old = {x["rho"]: x for x in old if x["a"] == 1 and x["variant"] == "primary"}
        ki = old[0.]["K"].index(4)
        old_gradient = {str(r): .5*(old[r]["path_gradient"][ki]-old[-r]["path_gradient"][ki]) for r in (.3, .75)}
        old_gap = .5*(old[.75]["score_minus_rho0"][ki]+old[-.75]["score_minus_rho0"][ki])
        for key, value in old_gradient.items():
            audit.close(value, results["old_n0_references"]["gradient"][key], "old_null_reference")
        audit.close(old_gap, results["old_n0_references"]["gap"], "old_null_reference")
        audit.test(results["old_n0_references"]["known_during_design_not_new_discovery"], "development reference scope")
        save(OUT/"STATISTICS_RECHECK.json", {"intervals": interval_rows, "certificates": certificate_rows,
             "main_gates": gates, "source_data_rebuilt_independently": True})
        receipt = {"created": utc(), "passed": True, "audit_source_sha256": sha(Path(__file__)),
                   "source_freeze_sha256": sha(MC/"FREEZE.json"), "results_freeze_sha256": sha(MC/"RESULTS_FREEZE.json"),
                   "files": public, "workspace_arrays": arrays, "scope": SCOPE,
                   "checks": audit.checks, "maximum_differences": audit.maximum,
                   "primary_bank_values_rebuilt": M*18, "auxiliary_bank_values_rebuilt": M*12,
                   "EB_intervals_recomputed": 18, "main_gates_recomputed": 6,
                   "finite_n_certificates_recomputed": 6, "main_passed": sum(gates),
                   "RNG_replayed_normals": replay_normals, "RNG_replayed_uniforms": replay_uniforms,
                   "RNG_active_normals": M*4*2, "RNG_nuisance_firstchunk_normals": 8*64*4*4094,
                   "raw_nuisance_normals_compared": 8*2*4*4094,
                   "nuisance_summary_banks_independently_replayed": 8*64,
                   "nuisance_summary_banks_used_without_raw_replay": M-8*64,
                   "no_new_independent_scientific_samples": True, "new_integrals": 0, "new_fits": 0,
                   "floating_point_and_PRNG_not_interval_certified": True,
                   "elapsed_seconds": time.perf_counter()-started,
                   "audit_artifacts": {rel(p): sha(p) for p in OUT.glob("*.json")}}
        save(OUT/"RECEIPT.json", receipt)
        print(json.dumps({"passed": True, "receipt_sha256": sha(OUT/"RECEIPT.json"), "checks": audit.checks}))
    except Exception as exc:
        save(OUT/"FAILURE.json", {"created": utc(), "passed": False, "error": repr(exc),
             "traceback": traceback.format_exc(), "checks_completed": audit.checks,
             "maximum_differences": audit.maximum, "audit_source_sha256": sha(Path(__file__))})
        raise


def check(workspace=False):
    receipt = read(OUT/"RECEIPT.json")
    audit = Audit()
    audit.test(receipt["passed"], "audit not passed")
    verify(receipt["files"], audit)
    verify(receipt["audit_artifacts"], audit)
    audit.test(receipt["audit_source_sha256"] == sha(Path(__file__)), "audit source")
    if workspace:
        verify(receipt["workspace_arrays"], audit)
    print(json.dumps({"passed": True, "workspace": workspace, "read_only": True,
                      "RNG_or_statistics_reexecuted": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("write", "check"))
    parser.add_argument("--workspace", action="store_true")
    args = parser.parse_args()
    if args.action == "write":
        execute()
    else:
        check(args.workspace)
