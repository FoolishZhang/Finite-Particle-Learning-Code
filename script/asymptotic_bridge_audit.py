"""Independent saved-input arithmetic audit for AsymptoticBridge-v1.

The numerical implementation uses explicit 0/1 labels, residual y-sigmoid(x),
and particle-exclusion sums. It does not import the producer or statistics code.
Scientific RNG replay is an explicitly requested audit of 16 saved blocks, not
an additional independent bank. Importing this module performs no computation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from datetime import datetime, timezone
for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_var] = "1"
import numpy as np

NS = (64, 256, 1024, 4096)
METRICS = ("path_gap", "D12", "overlap", "coupling", "direct_gap_truncated")
SEED = 202610080711
BLOCK = 64
BANKS = 131072
PARTICLES = 100
REPLAY_BLOCKS = tuple(i * 2047 // 15 for i in range(16))
REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_ART = REPOSITORY / "artifacts/asymptotic_bridge_v1_20261008"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        h = hashlib.file_digest(stream, "sha256")
    return h.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def independent_root(rho: float) -> np.ndarray:
    """Spectral projectors onto the common and orthogonal subspaces."""
    if not -.5 < rho < 1:
        raise ValueError("equicorrelation must lie in (-.5,1)")
    p = np.ones((3, 3), dtype=np.float64) / 3
    return math.sqrt(1-rho) * (np.eye(3)-p) + math.sqrt(1+2*rho) * p


def transform(z: np.ndarray, rho: float | np.ndarray) -> np.ndarray:
    rr = np.broadcast_to(np.asarray(rho, dtype=np.float64), (len(z),))
    if not np.all((rr > -.5) & (rr < 1)):
        raise ValueError("rho out of range")
    common = np.mean(z, axis=-1, keepdims=True)
    return (np.sqrt(1-rr)[:, None, None]*(z-common)
            + np.sqrt(1+2*rr)[:, None, None]*common)


def bits() -> np.ndarray:
    return ((np.arange(8)[:, None] >> np.arange(2, -1, -1)) & 1).astype(float)


def label_kernel(x: np.ndarray, s: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # The bit form differs from the signed-label producer implementation.
    # Evaluate both log-probabilities directly to avoid extreme-logit cancellation.
    logp = np.where(y == 1, -np.logaddexp(0., -x), -np.logaddexp(0., x))
    a = s + np.sum(logp, axis=2)
    amax = np.max(a, axis=1, keepdims=True)
    q = np.exp(a-amax)
    normalizer = np.sum(q, axis=1, keepdims=True)
    pi = q / normalizer
    # Sum each particle's competitors, without subtracting from the total.
    left = np.zeros_like(q)
    right = np.zeros_like(q)
    left[:, 1:] = np.cumsum(q[:, :-1], axis=1)
    right[:, :-1] = np.cumsum(q[:, :0:-1], axis=1)[:, ::-1]
    gamma = pi * ((left+right)/normalizer)
    sigmoid = np.exp(-np.logaddexp(0., -x))
    residual = np.where(y == 1, np.exp(-np.logaddexp(0., x)), -sigmoid)
    log_sum = amax[:, 0] + np.log(normalizer[:, 0])
    return pi, gamma, residual, log_sum


def explicit_score(z: np.ndarray, s_prefix: np.ndarray, u: np.ndarray, m_h: float) -> tuple[np.ndarray, np.ndarray, dict]:
    z, s, u = (np.asarray(t, dtype=np.float64) for t in (z, s_prefix, u))
    if z.ndim != 3 or z.shape[-1] != 3 or s.shape != (len(z), 4, z.shape[1]) or u.shape != (len(z),):
        raise ValueError("invalid primitive shapes")
    if not all(np.isfinite(t).all() for t in (z, s, u)) or not np.all((u >= 0) & (u <= .9)):
        raise ValueError("nonfinite or invalid input")
    xmid, xpath, xend = transform(z, .45), transform(z, u), transform(z, .9)
    scores = np.zeros((len(z), 4, 5))
    clipped = np.max(np.abs(z), axis=(1, 2)) > 8
    maximum_weight_error = 0.
    minimum_gamma = 1.
    maximum_gamma_sum = 0.
    for arm in range(4):
        centered = s[:, arm] - np.max(s[:, arm], axis=1, keepdims=True)
        for label in bits():
            pi, gamma, r, _ = label_kernel(xmid, centered, label)
            h = r[:, :, 0] * r[:, :, 1]
            scores[:, arm, 1] -= np.sum(gamma*h, axis=1)/8
            scores[:, arm, 2] += np.sum(gamma, axis=1)/8
            scores[:, arm, 3] += np.sum(gamma*(h-m_h), axis=1)/8
            maximum_weight_error = max(maximum_weight_error, float(np.max(np.abs(np.sum(pi, axis=1)-1))))
            minimum_gamma = min(minimum_gamma, float(gamma.min()))
            maximum_gamma_sum = max(maximum_gamma_sum, float(np.max(np.sum(gamma, axis=1))))
            _, gp, rp, _ = label_kernel(xpath, centered, label)
            mark = rp[:, :, 0]*rp[:, :, 1]+rp[:, :, 0]*rp[:, :, 2]+rp[:, :, 1]*rp[:, :, 2]
            scores[:, arm, 0] += .9*np.sum(gp*mark, axis=1)/8
            _, _, _, li = label_kernel(z, centered, label)
            _, _, _, lb = label_kernel(xend, centered, label)
            scores[:, arm, 4] += (lb-li)/8
        scores[clipped, arm, 4] = 0.
    diagnostics = dict(maximum_weight_sum_error=maximum_weight_error,
                       minimum_gamma=minimum_gamma, maximum_gamma_sum=maximum_gamma_sum,
                       D_identity_maxabs=float(np.max(np.abs(scores[:, :, 1]+m_h*scores[:, :, 2]+scores[:, :, 3]))))
    return scores, clipped, diagnostics


def replay_block(block_id: int, banks: int = BLOCK, particles: int = PARTICLES,
                 seed: int = SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent recipe: same RNG shapes, alternate log-sigmoid and extended sum."""
    generators = [np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([seed, block_id, role])))
                  for role in range(3)]
    z = generators[0].standard_normal((banks, particles, 3))
    gaussian = generators[1].standard_normal((banks, particles, 4096))
    # This is arithmetic replay of the originally generated Gaussian stream.
    np.negative(gaussian, out=gaussian)
    np.logaddexp(0., gaussian, out=gaussian)
    np.negative(gaussian, out=gaussian)
    s = np.empty((banks, 4, particles))
    running = np.zeros((banks, particles), dtype=np.longdouble)
    start = 0
    for arm, stop in enumerate(NS):
        running += np.sum(gaussian[:, :, start:stop], axis=2, dtype=np.longdouble)
        s[:, arm] = running
        start = stop
    u = .9*generators[2].random(banks)
    return z, s, u


def scalar_moments(x: np.ndarray) -> tuple[float, float]:
    values = np.asarray(x, dtype=float).ravel()
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("at least two finite observations required")
    mean = math.fsum(values)/len(values)
    variance = math.fsum((float(t)-mean)**2 for t in values)/(len(values)-1)
    return mean, variance


def covariance(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or len(x) < 2:
        raise ValueError("covariance expects a (B,p) array")
    means = np.array([math.fsum(x[:, j])/len(x) for j in range(x.shape[1])])
    centered = x-means
    result = np.empty((x.shape[1], x.shape[1]))
    for i in range(x.shape[1]):
        for j in range(i, x.shape[1]):
            result[i, j] = result[j, i] = math.fsum(centered[:, i]*centered[:, j])/(len(x)-1)
    return result


def eb_summary(x: np.ndarray, support: tuple[float, float], delta: float,
               numerical: float = 0., tail: float = 0.) -> dict:
    mean, variance = scalar_moments(x)
    count, width = len(x), support[1]-support[0]
    if width < 0 or not 0 < delta < 1 or numerical < 0 or tail < 0:
        raise ValueError("invalid interval configuration")
    if np.any(np.asarray(x) < support[0]) or np.any(np.asarray(x) > support[1]):
        raise ValueError("score outside fixed support")
    logterm = math.log(4/delta)
    stat = math.sqrt(2*variance*logterm/count)+7*width*logterm/(3*(count-1))
    halfwidth = stat+numerical+tail
    return dict(banks=count, mean=mean, unbiased_variance=variance, SE=math.sqrt(variance/count),
                support=list(support), support_width=width, statistical_halfwidth=stat,
                numerical_allowance=numerical, tail_allowance=tail, halfwidth=halfwidth,
                sample_min=float(np.min(x)), sample_max=float(np.max(x)),
                interval=[mean-halfwidth, mean+halfwidth], delta=delta, delta_per_side=delta/2)


def affine(summary: dict, scale: float, offset: float = 0.) -> dict:
    result = dict(summary)
    for key in ("mean",):
        result[key] = scale*summary[key]+offset
    result["unbiased_variance"] = scale**2*summary["unbiased_variance"]
    for key in ("SE", "support_width", "statistical_halfwidth", "numerical_allowance", "tail_allowance", "halfwidth"):
        result[key] = abs(scale)*summary[key]
    for key in ("support", "interval"):
        result[key] = sorted(scale*float(t)+offset for t in summary[key])
    result["sample_min"], result["sample_max"] = sorted(scale*summary[key]+offset for key in ("sample_min", "sample_max"))
    return result


def assert_close(actual, expected, name: str, tolerance: float = 1e-12) -> float:
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError(f"{name}: invalid shape/nonfinite")
    difference = float(np.max(np.abs(a-b))) if a.size else 0.
    if np.any(np.abs(a-b) > tolerance*np.maximum(1., np.maximum(np.abs(a), np.abs(b)))):
        raise AssertionError(f"{name}: difference {difference}")
    return difference


def check_summary(saved: dict, rebuilt: dict, name: str) -> None:
    for key in rebuilt:
        assert_close(saved[key], rebuilt[key], f"{name}.{key}")


def direct_constants() -> dict:
    a = max(1-math.sqrt(.1), math.sqrt(2.8)-1)
    maximum = 24*a
    b_k = (8/3)*math.log(PARTICLES*8)
    union = 600*math.exp(-32)/(8*math.sqrt(2*math.pi))
    tail = math.sqrt(3)*a*math.sqrt(b_k*union)
    return dict(A=a, M=maximum, b_K=b_k, Gaussian_union_probability_bound=union,
                tail_allowance=tail, support=[-maximum, maximum], support_width=2*maximum, T=8.)


def supports(m_h: float) -> dict:
    s = 1-1/PARTICLES
    return dict(path_gap=(-.675*s, 1.35*s), D12=(-s/2, s/2), overlap=(0., s),
                coupling=(-(1/2+m_h)*s, (1-m_h)*s/2),
                direct_gap_truncated=tuple(direct_constants()["support"]))


def check_reference(artifact: Path) -> tuple[dict, dict]:
    contract = load_json(artifact/"REFERENCE_CONTRACT.json")
    ref = load_json(artifact/"REFERENCE.json")
    if ref["contract_sha256"] != digest(artifact/"REFERENCE_CONTRACT.json") or ref["source_hashes"] != contract["source_hashes"]:
        raise AssertionError("reference contract/input hash mismatch")
    for rel, expected in contract["source_hashes"].items():
        if digest(REPOSITORY/rel) != expected:
            raise AssertionError("changed reference input: "+rel)
    if ref["script_sha256"] != digest(REPOSITORY/"script/asymptotic_bridge_statistics.py"):
        raise AssertionError("reference script mismatch")
    old = load_json(REPOSITORY/contract["old_coefficients"]["result"])
    manifest = load_json(REPOSITORY/contract["old_coefficients"]["manifest"])
    if manifest["results"] != digest(REPOSITORY/contract["old_coefficients"]["result"]):
        raise AssertionError("old reference result manifest mismatch")
    if manifest["script"] != digest(REPOSITORY/"script/candidate_v171_calibration.py"):
        raise AssertionError("old reference script manifest mismatch")
    for name, expected in (("v", old["derived"]["v"]), ("J100", old["derived"]["J100"]), ("gapV", -old["derived"]["V_Cb"])):
        if ref[name] != expected or not math.isfinite(ref[name]) or ref[name] <= 0:
            raise AssertionError("reference coefficient changed/invalid: "+name)
    h = ref["m_H45"]
    if not 0 < h <= .028125:
        raise AssertionError("invalid mean mark")
    rows = ref["quadrature_records"]
    if len(rows) != 4 or len(contract["rules"]) != 4:
        raise AssertionError("four reference rules required")
    for row, rule in zip(rows, contract["rules"]):
        if row["rule"] != rule or row["id"] != rule["id"] or row["rho"] != .45:
            raise AssertionError("reference rule ledger mismatch")
        if len(row["normal_nodes"]) != rule["order"] or len(row["normal_weights"]) != rule["order"]:
            raise AssertionError("reference rule shape")
        if not np.isfinite(row["normal_nodes"]).all() or not np.isfinite(row["normal_weights"]).all() or min(row["normal_weights"]) <= 0:
            raise AssertionError("reference rule finite/positive weights")
        if row["m_H45"] != row["kappa45"]/4 or row["normal_mass"] != math.fsum(row["normal_weights"]):
            raise AssertionError("reference scalar algebra")
        if row["bivariate_node_evaluations"] != rule["order"]**2 or row["integral_count"] != 1:
            raise AssertionError("reference integral ledger")
    primary = next(row for row in rows if row["id"] == contract["primary_rule"])
    difference = max(abs(row["m_H45"]-h) for row in rows)
    masserror = max(abs(row["normal_mass"]-1) for row in rows)
    if h != primary["m_H45"] or ref["kappa45"] != primary["kappa45"]:
        raise AssertionError("primary reference mismatch")
    qualification = contract["qualification"]
    if difference > qualification["max_absolute_m_H45_rule_difference"] or masserror > qualification["maximum_normal_mass_error"]:
        raise AssertionError("reference numerical qualification failed")
    assert_close(ref["qualification"]["max_absolute_m_H45_rule_difference"], difference, "qualification difference")
    assert_close(ref["qualification"]["maximum_normal_mass_error"], masserror, "qualification mass")
    if not ref["qualification"]["passed"] or ref["qualification"]["rigorous_integration_enclosure"]:
        raise AssertionError("reference qualification type")
    for n in NS:
        assert_close(ref["w_by_n"][str(n)], ref["J100"]/math.sqrt(n*ref["v"]), "selection coefficient", 2e-15)
    for name, value in direct_constants().items():
        assert_close(ref["direct_gap_T8"][name], value, "reference direct tail."+name, 2e-15)
    return ref, dict(passed=True, reference_sha256=digest(artifact/"REFERENCE.json"),
                     input_source_hashes_checked=len(contract["source_hashes"]),
                     quadrature_receipts_checked=4, quadrature_integrals_recomputed=0,
                     native_rule_difference=difference, native_mass_error=masserror,
                     exact_coefficient_enclosure=False)


def check_statistics(scores: np.ndarray, reference: dict, saved: dict, *, expected_banks: int = BANKS) -> dict:
    x = np.asarray(scores, dtype=float)
    count = expected_banks
    if x.shape != (count, 4, 5) or not np.isfinite(x).all():
        raise AssertionError("statistics input panel shape/nonfinite")
    if saved["banks"] != count or saved["family_target_count"] != 26 or saved["covariance_ddof"] != 1:
        raise AssertionError("formal statistics metadata")
    if saved["n"] != list(NS) or saved["K"] != PARTICLES or saved["metric_order"] != list(METRICS) or saved["formal_bank_count_matches"] != (count == BANKS):
        raise AssertionError("statistics panel identity")
    config = saved["config"]
    for key, expected in (("alpha", .01), ("target_count", 26), ("formal_look_count", 1),
                          ("delta_per_two_sided_target", .01/26), ("delta_per_side", .01/52),
                          ("numerical_allowance_raw", 1e-9), ("empirical_variance_ddof", 1)):
        if config[key] != expected:
            raise AssertionError("statistics config."+key)
    if config["radius_logarithm"] != "log(4/delta)" or config["exact_coefficient_enclosures"]:
        raise AssertionError("statistics radius/coefficient scope")
    h, gap_v = reference["m_H45"], reference["gapV"]
    ws = [reference["w_by_n"][str(n)] for n in NS]
    ranges, raw_summaries, target_order, columns = supports(h), [], [], []
    for arm, (n, w) in enumerate(zip(NS, ws)):
        row = saved["per_n"][arm]
        if row["n"] != n:
            raise AssertionError("per_n order")
        assert_close(row["w"], w, "w", 2e-15)
        raw = {}
        for mi, metric in enumerate(METRICS):
            values, support = x[:, arm, mi], ranges[metric]
            if np.any(values < support[0]) or np.any(values > support[1]):
                raise AssertionError("saved support violation: "+metric)
            target = f"n{n}:{metric}"
            raw[metric] = eb_summary(values, support, .01/26, 1e-9, direct_constants()["tail_allowance"] if mi == 4 else 0.)
            check_summary(row["raw"][metric], raw[metric], target)
            if row["raw"][metric]["target_id"] != target:
                raise AssertionError("target ordering")
            if row["raw"][metric]["numerical_error_certified"] or row["raw"][metric]["intervals_clipped_to_support"]:
                raise AssertionError("native numerical/interval qualification")
            target_order.append(target); columns.append(values)
        raw_summaries.append(raw)
        derived_recipes = dict(A=("path_gap", 1/(w*gap_v), 0.), B=("D12", -1/(w*h), 0.),
                               omega_over_w=("overlap", 1/w, 0.), R=("D12", 1/w, h),
                               O=("overlap", -h/w, h), J=("coupling", -1/w, 0.))
        for name, (source, scale, shift) in derived_recipes.items():
            actual = row["derived"][name]
            check_summary(actual, affine(raw[source], scale, shift), f"n{n}.{name}")
            if actual["reuses_target"] != f"n{n}:{source}" or actual["extra_alpha"] != 0:
                raise AssertionError("derived event reuse")
            if actual["coefficient_target"] != "frozen numeric reference":
                raise AssertionError("derived coefficient qualification")
            assert_close(actual["affine_scale"], scale, "derived scale")
            assert_close(actual["affine_shift"], shift, "derived shift")
        d, overlap, coupling = x[:, arm, 1], x[:, arm, 2], x[:, arm, 3]
        identity = float(np.max(np.abs(d+h*overlap+coupling)))
        if identity > 1e-9:
            raise AssertionError("D bank identity")
        assert_close(row["bank_identity_max_absolute_residual"], identity, "bank identity")
        residuals = np.column_stack([d/w+h, h*(1-overlap/w), -coupling/w])
        assert_close(row["residual_covariance"], covariance(residuals), "residual covariance")
        derived = row["derived"]
        assert_close(row["residual_mean_identity_error"], derived["R"]["mean"]-derived["O"]["mean"]-derived["J"]["mean"], "residual mean identity")
        if row["residual_covariance_order"] != ["R", "O", "J"]:
            raise AssertionError("residual order")
        discrepancy_mean, discrepancy_var = scalar_moments(x[:, arm, 4]-x[:, arm, 0])
        diagnostic = row["direct_minus_path_diagnostic"]
        assert_close([diagnostic["mean"], diagnostic["unbiased_variance"], diagnostic["paired_SE"]],
                     [discrepancy_mean, discrepancy_var, math.sqrt(discrepancy_var/count)], "direct-path diagnostic")
        if not diagnostic["diagnostic_only"] or diagnostic["additional_confidence_claim"] or diagnostic["enforced_pass_threshold"] is not None:
            raise AssertionError("diagnostic cannot create a new success gate")
    for group, mi, denominator, sign in (("paired_gap", 0, gap_v, 1.), ("paired_gradient", 1, h, -1.)):
        kind = "A" if mi == 0 else "B"
        if len(saved[group]) != 3:
            raise AssertionError("three neighbor contrasts required")
        for arm in range(3):
            row = saved[group][arm]
            scales = [sign/(ws[arm]*denominator), sign/(ws[arm+1]*denominator)]
            prev = sorted(scales[0]*v for v in ranges[METRICS[mi]])
            nxt = sorted(scales[1]*v for v in ranges[METRICS[mi]])
            values = scales[1]*x[:, arm+1, mi]-scales[0]*x[:, arm, mi]
            support = (nxt[0]-prev[1], nxt[1]-prev[0])
            summary = eb_summary(values, support, .01/26, sum(map(abs,scales))*1e-9)
            check_summary(row["summary"], summary, group+str(arm))
            target = f"{kind}:n{NS[arm+1]}_minus_n{NS[arm]}"
            if row["summary"]["target_id"] != target or row["direction"] != "next_minus_previous" or row["n_previous"] != NS[arm] or row["n_next"] != NS[arm+1]:
                raise AssertionError("neighbor pair order")
            within = covariance(np.column_stack([x[:, arm+1, mi], x[:, arm, mi]]))[0, 1]
            independent_variance = scales[1]**2*raw_summaries[arm+1][METRICS[mi]]["unbiased_variance"]+scales[0]**2*raw_summaries[arm][METRICS[mi]]["unbiased_variance"]
            covariance_variance = independent_variance-2*scales[0]*scales[1]*within
            assert_close([row["previous_scale"], row["next_scale"], row["raw_same_bank_covariance"], row["independent_variance_not_used"], row["covariance_reconstructed_variance"]],
                         scales+[within, independent_variance, covariance_variance], "paired covariance reconstruction")
            assert_close(summary["unbiased_variance"], covariance_variance, "paired variance identity")
            assert_close(row["covariance_reconstruction_absolute_error"], abs(summary["unbiased_variance"]-covariance_variance), "paired variance discrepancy")
            target_order.append(target); columns.append(values)
    matrix = np.column_stack(columns)
    cov = covariance(matrix)
    if saved["target_order"] != target_order or saved["covariance26_order"] != target_order or saved["covariance20_order"] != target_order[:20]:
        raise AssertionError("target/covariance ordering")
    assert_close(saved["covariance26"], cov, "covariance26")
    assert_close(saved["covariance20"], cov[:20, :20], "covariance20")
    assert_close(saved["target_means"], [math.fsum(v)/count for v in columns], "target means")
    for key in ("v", "J100", "gapV", "m_H45"):
        assert_close(saved["reference_coefficients"][key], reference[key], "statistics reference")
    for n in NS:
        assert_close(saved["reference_coefficients"]["w_by_n"][str(n)],reference["w_by_n"][str(n)],"statistics w reference")
    for key, expected in direct_constants().items():
        assert_close(saved["direct_gap_T8"][key], expected, "statistics direct tail")
    return dict(passed=True, raw_target_count=20, paired_target_count=6, family_target_count=26,
                covariance20_checked=True, covariance26_checked=True, residual_covariance_matrices_checked=4,
                derived_event_count=24, EB_logarithm="log(4/delta)", independent_bank_unit_preserved=True,
                coefficient_target="frozen numeric reference, no exact coefficient enclosure")


def check_freeze(artifact: Path) -> dict:
    frozen = load_json(artifact/"FREEZE.json")
    if not frozen.get("sources"):
        raise AssertionError("nonempty source freeze required")
    for relative, expected in frozen["sources"].items():
        path = (REPOSITORY/relative).resolve()
        if not path.is_relative_to(REPOSITORY.resolve()) or digest(path) != expected:
            raise AssertionError("frozen source mismatch: "+relative)
    for required in ("script/asymptotic_bridge_audit.py", "script/asymptotic_bridge_audit_test.py",
                     "script/asymptotic_bridge_core.py", "script/asymptotic_bridge_statistics.py",
                     (artifact/"REFERENCE.json").relative_to(REPOSITORY).as_posix()):
        if required not in frozen["sources"]:
            raise AssertionError("missing audit freeze source: "+required)
    expected = dict(banks=BANKS, particles=PARTICLES, n_values=list(NS), block=BLOCK, block_count=2048,
                    workers=6, seed_family=SEED, roles={"active":0,"nuisance":1,"uniform_rho":2},
                    rho_fixed=.45, rho_end=.9, metric_names=list(METRICS),
                    label_encoding="spins_minus1_plus1_lexicographic", dtype="float64",
                    nuisance_prefix_method="inplace_log_expit_segment_sum_then_add",
                    gamma_method="dominant_complement_from_other_weights", cpu_threads_per_worker=1)
    for key, value in expected.items():
        if frozen["config"].get(key) != value:
            raise AssertionError("freeze config mismatch: "+key)
    return frozen


def binding_files(artifact: Path) -> dict:
    files = [artifact/name for name in ("FREEZE.json", "REFERENCE_CONTRACT.json", "REFERENCE.json", "INDEX.json",
                                      "RUN_START.json", "RUN_COMPLETE.json", "STATISTICS.json", "BANK_SCORES.npy")]
    return {path.relative_to(REPOSITORY).as_posix():digest(path) for path in files}


def audit(artifact: Path, output: Path) -> dict:
    artifact, output = artifact.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("refusing to replace an audit receipt")
    started = time.monotonic()
    frozen = check_freeze(artifact)
    reference, ref_check = check_reference(artifact)
    index, complete, run_start = (load_json(artifact/name) for name in ("INDEX.json", "RUN_COMPLETE.json", "RUN_START.json"))
    freeze_hash, ref_hash = digest(artifact/"FREEZE.json"), digest(artifact/"REFERENCE.json")
    for document in (index, complete, run_start):
        if document["freeze_sha256"] != freeze_hash or document["reference_sha256"] != ref_hash:
            raise AssertionError("run/index provenance mismatch")
    if complete["index_sha256"] != digest(artifact/"INDEX.json") or run_start["config"] != frozen["config"]:
        raise AssertionError("completion/index/start binding")
    if index["metric_names"] != list(METRICS) or index["n_values"] != list(NS) or not index["scientific"]:
        raise AssertionError("index scientific schema mismatch")
    resources = dict(independent_banks=BANKS, block_count=2048, particles_per_bank=PARTICLES,
                     active_Gaussian_scalars=BANKS*PARTICLES*3, nuisance_Gaussian_scalars=BANKS*PARTICLES*4096,
                     uniform_rho_scalars=BANKS, prior_vectors=BANKS*PARTICLES, new_observations=0,
                     training_updates=0, scientific_banks=BANKS, score_entries=BANKS*20,
                     label_loop_iterations_per_bank=4*8*3, active_likelihood_evaluations_per_bank=4*8*4)
    if index["resources"] != resources or complete["resources"] != resources:
        raise AssertionError("input/resource ledger mismatch")
    aggregate_path = artifact/index["scores"]["path"]
    if aggregate_path != artifact/"BANK_SCORES.npy" or digest(aggregate_path) != index["scores"]["sha256"] or aggregate_path.stat().st_size != index["scores"]["bytes"]:
        raise AssertionError("aggregate binding")
    aggregate = np.load(aggregate_path, allow_pickle=False)
    if aggregate.shape != (BANKS,4,5) or aggregate.dtype != np.float64 or not np.isfinite(aggregate).all():
        raise AssertionError("aggregate shape/dtype/nonfinite")
    records = index["records"]
    if len(records) != 2048 or len(list((artifact/"raw").glob("block_*.npz"))) != 2048:
        raise AssertionError("complete 2048-block panel required")
    block_checks, replay_checks, raw_hashes = [], [], {}
    max_score_error = max_identity = max_sumweights = 0.
    minimum_gamma, maximum_gamma_sum = 1., 0.
    clipped_banks = 0
    maximum_replay_s_error = maximum_replay_score_error = 0.
    for block_id, record in enumerate(records):
        if record["block_id"] != block_id or record["banks"] != BLOCK or record["path"] != f"raw/block_{block_id:05d}.npz":
            raise AssertionError("raw index ordering")
        path = artifact/record["path"]
        if digest(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise AssertionError("raw hash/size mismatch: "+path.name)
        if load_json(path.with_suffix(".json")) != record:
            raise AssertionError("per-block receipt mismatch")
        raw_hashes[path.relative_to(REPOSITORY).as_posix()] = record["sha256"]
        raw_hashes[path.with_suffix(".json").relative_to(REPOSITORY).as_posix()] = digest(path.with_suffix(".json"))
        with np.load(path, allow_pickle=False) as stored:
            if set(stored.files) != {"z","s_prefix","u","scores","clipmask","metadata"}:
                raise AssertionError("raw fields")
            z,s,u,score,mask = (stored[name] for name in ("z","s_prefix","u","scores","clipmask"))
            for name, values, shape in (("z",z,(64,100,3)),("s",s,(64,4,100)),("u",u,(64,)),("score",score,(64,4,5)),("mask",mask,(64,))):
                if values.shape != shape or values.dtype != (np.dtype(bool) if name == "mask" else np.dtype("float64")) or not np.isfinite(values).all():
                    raise AssertionError("primitive schema: "+name)
            metadata = json.loads(str(stored["metadata"].item()))
            expected_metadata = dict(schema_version="1.0",block_id=block_id,bank_first=block_id*BLOCK,banks=BLOCK,
                                     particles=PARTICLES,n_values=list(NS),metric_names=list(METRICS),
                                     label_encoding="spins_minus1_plus1_lexicographic",freeze_sha256=freeze_hash,
                                     reference_sha256=ref_hash,freeze_reference="FREEZE.json",coefficient_reference="REFERENCE.json",scientific=True,
                                     role_seed_sequences={str(role):[SEED,block_id,role] for role in range(3)},
                                     nuisance_method="inplace_log_expit_segment_sum_then_add",
                                     gamma_method="dominant_complement_from_other_weights")
            for key, value in expected_metadata.items():
                if metadata.get(key) != value:
                    raise AssertionError("primitive metadata: "+key)
            if record["metadata"] != metadata:
                raise AssertionError("raw receipt metadata mismatch")
            if not np.array_equal(score, aggregate[block_id*BLOCK:(block_id+1)*BLOCK]):
                raise AssertionError("raw/aggregate score assembly")
            recomputed, independent_mask, diagnostic = explicit_score(z,s,u,reference["m_H45"])
            error = float(np.max(np.abs(recomputed-score)))
            if error > 2e-11:
                raise AssertionError(f"independent score gate failed block {block_id}: {error}")
            if not np.array_equal(mask, independent_mask):
                raise AssertionError("clipmask mismatch")
            for mi, metric in enumerate(METRICS):
                lower, upper = supports(reference["m_H45"])[metric]
                if np.any(recomputed[:,:,mi] < lower-2e-11) or np.any(recomputed[:,:,mi] > upper+2e-11):
                    raise AssertionError("independent support: "+metric)
            if diagnostic["D_identity_maxabs"] > 2e-11 or diagnostic["maximum_weight_sum_error"] > 2e-14 or diagnostic["minimum_gamma"] < 0 or diagnostic["maximum_gamma_sum"] > 1-1/PARTICLES+2e-14:
                raise AssertionError("weight/Price identity gate")
            max_score_error = max(max_score_error,error)
            max_identity = max(max_identity,diagnostic["D_identity_maxabs"])
            max_sumweights = max(max_sumweights,diagnostic["maximum_weight_sum_error"])
            minimum_gamma = min(minimum_gamma,diagnostic["minimum_gamma"])
            maximum_gamma_sum = max(maximum_gamma_sum,diagnostic["maximum_gamma_sum"])
            clipped_banks += int(mask.sum())
            block_checks.append(dict(block_id=block_id,score_maxabs=error,**diagnostic))
            if block_id in REPLAY_BLOCKS:
                rz, rs, ru = replay_block(block_id)
                if not np.array_equal(z, rz) or not np.array_equal(u, ru):
                    raise AssertionError("pure RNG replay is not bitwise identical")
                s_error = float(np.max(np.abs(s-rs)))
                if s_error > 2e-10:
                    raise AssertionError(f"nuisance prefix replay failed: {s_error}")
                replayed_scores, replayed_mask, _ = explicit_score(rz,rs,ru,reference["m_H45"])
                replay_score_error = float(np.max(np.abs(replayed_scores-score)))
                if replay_score_error > 2e-11 or not np.array_equal(replayed_mask,mask):
                    raise AssertionError("replayed-input score gate failed")
                maximum_replay_s_error = max(maximum_replay_s_error,s_error)
                maximum_replay_score_error = max(maximum_replay_score_error,replay_score_error)
                replay_checks.append(dict(block_id=block_id,z_bitwise_identical=True,u_bitwise_identical=True,
                                          s_prefix_maxabs=s_error,score_maxabs=replay_score_error))
        if block_id % 64 == 63:
            print(json.dumps(dict(stage="independent_audit",checked_blocks=block_id+1,total_blocks=2048,
                                  elapsed_seconds=time.monotonic()-started)),flush=True)
    if [row["block_id"] for row in replay_checks] != list(REPLAY_BLOCKS):
        raise AssertionError("predetermined replay index coverage")
    saved_stats = load_json(artifact/"STATISTICS.json")
    if saved_stats["raw_sha256"] != digest(aggregate_path) or saved_stats["reference_sha256"] != ref_hash or saved_stats["statistics_script_sha256"] != digest(REPOSITORY/"script/asymptotic_bridge_statistics.py"):
        raise AssertionError("statistics input/source binding")
    stats_check = check_statistics(aggregate,reference,saved_stats)
    root_errors = []
    for rho in (0.,.45,.9):
        root = independent_root(rho)
        target = np.full((3,3),rho); np.fill_diagonal(target,1.)
        root_errors.append(assert_close(root@root.T,target,"root covariance",2e-15))
    receipt = dict(schema_version="asymptotic_bridge_independent_audit_v1",passed=True,
                   created_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,
                   source_hashes=dict(frozen["sources"]),input_hashes=binding_files(artifact),raw_input_hashes=raw_hashes,
                   independent_implementation="explicit bits; y-sigmoid residual; exclusion complement sums; spectral root projectors",
                   scope="saved native array arithmetic and directed source-generation replay; not strict floating-point enclosure or external scientific review",
                   score_audit=dict(banks=BANKS,blocks=2048,score_entries=BANKS*20,score_block_calls=2048,
                                    score_maxabs=max_score_error,score_tolerance=2e-11,D_identity_maxabs=max_identity,
                                    weight_sum_maxabs=max_sumweights,minimum_gamma=minimum_gamma,maximum_gamma_sum=maximum_gamma_sum,
                                    clipped_bank_count=clipped_banks,clipping_applies_only_to_direct_gap=True,block_checks=block_checks),
                   rng_recipe_replay=dict(block_indices=list(REPLAY_BLOCKS),block_checks=replay_checks,
                                          replayed_existing_banks=1024,new_independent_scientific_banks=0,
                                          RNG_array_calls=48,Gaussian_array_calls=32,uniform_array_calls=16,
                                          replayed_active_Gaussian_scalars=1024*100*3,
                                          replayed_nuisance_Gaussian_scalars=1024*100*4096,
                                          replayed_uniform_scalars=1024,s_prefix_maxabs=maximum_replay_s_error,
                                          score_maxabs=maximum_replay_score_error,s_prefix_tolerance=2e-10,score_tolerance=2e-11,
                                          exact_rng_arrays="Z and U bitwise",nuisance_arithmetic="-logaddexp(0,-G), np.longdouble segment sums",
                                          segment_sum_mantissa_bits=int(np.finfo(np.longdouble).nmant)),
                   root_covariance_maxabs=max(root_errors),reference=ref_check,statistics=stats_check,
                   actual_work=dict(scientific_npz_read=2048,independent_stored_score_calls=2048,
                                    replayed_input_score_calls=16,new_independent_scientific_banks=0,
                                    scientific_rng_array_replay_calls=48,new_reference_integrals=0,new_training_steps=0,
                                    old_scientific_npz_read=0,new_observations=0))
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf-8",newline="\n") as handle:
        json.dump(receipt,handle,ensure_ascii=False,indent=2,allow_nan=False); handle.write("\n")
    return receipt


def check_saved_audit(artifact: Path, output: Path) -> dict:
    artifact, output = artifact.resolve(), output.resolve()
    check_freeze(artifact)
    receipt = load_json(output)
    if not receipt["passed"] or receipt["schema_version"] != "asymptotic_bridge_independent_audit_v1":
        raise AssertionError("invalid audit receipt")
    frozen = load_json(artifact/"FREEZE.json")
    if receipt["source_hashes"] != frozen["sources"] or receipt["input_hashes"] != binding_files(artifact):
        raise AssertionError("changed audit sources/inputs")
    for rel, expected in receipt["raw_input_hashes"].items():
        if digest(REPOSITORY/rel) != expected:
            raise AssertionError("changed raw audit input: "+rel)
    if len(receipt["raw_input_hashes"]) != 4096 or receipt["rng_recipe_replay"]["block_indices"] != list(REPLAY_BLOCKS):
        raise AssertionError("audit coverage ledger")
    reference, _ = check_reference(artifact)
    statistics = check_statistics(np.load(artifact/"BANK_SCORES.npy",allow_pickle=False),reference,load_json(artifact/"STATISTICS.json"))
    if statistics != receipt["statistics"]:
        raise AssertionError("saved statistics audit mismatch")
    return dict(passed=True,scope="saved audit/input hashes and independent saved-score statistics replay",
                RNG_calls=0,score_kernel_calls=0,new_reference_integrals=0,new_independent_scientific_banks=0,
                original_replayed_rng_array_calls=48,original_scientific_npz_score_checks=2048)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true", help="audit saved scientific inputs; includes 16 RNG recipe replays")
    mode.add_argument("--check", action="store_true", help="verify saved receipt hashes and replay statistics; no RNG/score-kernel calls")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ART)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.artifact_root = args.artifact_root.resolve()
    output = args.output or args.artifact_root / "AUDIT.json"
    receipt = audit(args.artifact_root, output) if args.run else check_saved_audit(args.artifact_root,output)
    print(json.dumps({"passed": receipt["passed"], "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
