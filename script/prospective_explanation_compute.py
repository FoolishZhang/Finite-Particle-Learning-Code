"""Frozen, post-hoc explanation of the twelve saved development units.

This producer performs saved-array reweighting, finite-alphabet risk algebra,
and independent Bernoulli closed-form fits only. It never calls a likelihood
kernel, generates data, fits a new joint model, changes an alarm, or selects a
population eigenvector. Independent GL/GH80 verification is a separate program.
The CLI refuses production work until the coordinating author writes FREEZE.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import prospective_diagnostic_dev as legacy

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "artifacts/prospective_diagnostic_dev_v1_20261007"
OUT = ROOT / "artifacts/prospective_explanation_v1_20261007"
DOC = ROOT / "docs/08_research_研究工作/ProspectiveExplanation_v1_20261007"
BASE = "8478793de147fa4d2162a63e5ad1fb2cfc3c9380"
HANDOFF_SHA = "f6a58dfad46e02ddf424d69600cc7066aca52a795abd38670252cb656cca0089"
OLD_MANIFEST_SHA = "2b537dd6e86f9547a1e540dd429b14a86a46c04d327ddb97c5b58fb783a9abd8"
PAPER_MANIFEST = ROOT / "artifacts/submission_candidate_v1_6_1_20261007/DELIVERY_MANIFEST.json"
PAPER_MANIFEST_SHA = "c47d8f3f732458ca96830f729a4c588685f5c56f2c634af062b8659e41da410d"
LOCKS = ("INPUTS_LOCK", "DIAGNOSIS_LOCK", "ENDPOINTS_LOCK", "EVALUATION_LOCK")


def config_dump():
    return dict(schema="prospective_explanation_config_v1", baseline=BASE,
                original_units=[dict(id=f"D{d}_N{N}_r{rep}", teacher=d - 1, N=N, rep=rep)
                                for d in (1, 2, 3) for N in (1024, 4096) for rep in (0, 1)],
                conditional_rules=["C", "D"], teacher_rule_order=[[40, 0.], [60, 0.], [80, 0.], [72, math.pi / 7]],
                main_conditional_index=1, main_teacher_index=1, original_direction_unchanged=True,
                numerical_sign_floor=1e-8, numerical_disagreement_multiplier=10.,
                sign_requires_all_variants_beyond_tau=True, radial_direction_gradient_floor=1e-10,
                identity_absolute_tolerance=1e-12, baseline_mass_tolerance=2e-12,
                baseline_risk_cross_rule_tolerance=2e-6,
                population_direction_reselected=False, old_alarm_resolution_floor_unchanged=5e-4,
                independent_baseline="same-data Bernoulli MLE, no smoothing, no optimizer",
                endpoint_rule_pairing="same rule index for teacher and saved endpoint PMF",
                leave_largest_alarm="remove first maximal saved delta_R among original FULL alarms, descriptive only",
                producer_new_quadrature_calls=0, producer_new_optimizer_updates=0, new_random_draws=0,
                post_hoc_not_independent_validation=True)


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def rel(path): return Path(path).resolve().relative_to(ROOT).as_posix()
def now(): return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition: raise AssertionError(message)


def snapshot(x):
    if isinstance(x, np.ndarray): return x.copy()
    if isinstance(x, dict): return {k: snapshot(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return type(x)(snapshot(v) for v in x)
    return x


def public(x):
    if isinstance(x, np.ndarray): return public(x.tolist())
    if isinstance(x, np.generic): return public(x.item())
    if isinstance(x, float) and not math.isfinite(x): return {"__nonfinite_float__": repr(x)}
    if isinstance(x, dict): return {str(k): public(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [public(v) for v in x]
    return x


def save(path, value):
    path = Path(path)
    require(path.resolve().is_relative_to(OUT), "new output outside explanation package")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(public(value), f, ensure_ascii=False, indent=2, allow_nan=False); f.write("\n")


def binding(path): return dict(path=rel(path), sha256=sha(path), bytes=Path(path).stat().st_size)


def patterns(m):
    return ((np.arange(1 << m, dtype=np.int64)[:, None] >> np.arange(m - 1, -1, -1)) & 1).astype(np.int8)


def pmf_array(x, name="PMF"):
    x = np.asarray(x)
    if np.iscomplexobj(x): raise ValueError(f"{name} must be real")
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 4 or len(x) & (len(x) - 1) or not np.isfinite(x).all() or np.any(x < 0) or x.sum() <= 0:
        raise ValueError(f"{name} must be a finite, nonnegative, positive-mass binary pattern vector")
    return x.copy(), len(x).bit_length() - 1


def raw_marginals(p):
    p, m = pmf_array(p); Y = patterns(m)
    # Both cells are summed from the original raw measure. In particular the
    # zero cell is not replaced by 1 minus the one cell.
    return np.column_stack((p @ (1 - Y), p @ Y))


def relative_entropy(P, Q):
    P, m = pmf_array(P, "P"); Q, mq = pmf_array(Q, "Q")
    if m != mq: raise ValueError("PMF dimensions differ")
    active = P > 0
    if np.any(Q[active] == 0): return math.inf
    return float(P[active] @ (np.log(P[active]) - np.log(Q[active])))


def _two_cell_KL(p, q):
    active = p > 0
    if np.any(q[active] == 0): return math.inf
    return float(p[active] @ (np.log(p[active]) - np.log(q[active])))


def risk_decomposition(P, Q):
    """Raw KL = raw TC + sum raw marginal KL - dependence score.

    Zero-support conflicts are retained. Undefined infinity-minus-infinity is
    never converted to a zero residual or a finite qualified decomposition.
    """
    P, m = pmf_array(P, "teacher PMF"); Q, mq = pmf_array(Q, "model PMF")
    if m != mq: raise ValueError("PMF dimensions differ")
    Y = patterns(m); active = P > 0
    pj, qj = raw_marginals(P), raw_marginals(Q)
    teacher_log_product = np.log(pj[np.arange(m)[None, :], Y[active]]).sum(axis=1)
    TC = float(P[active] @ (np.log(P[active]) - teacher_log_product))
    marginal = np.array([_two_cell_KL(pj[j], qj[j]) for j in range(m)])
    marginal_sum = float(marginal.sum()); KL = relative_entropy(P, Q)
    model_marginal_support = qj[np.arange(m)[None, :], Y[active]]
    if np.any(model_marginal_support == 0):
        J = None; residual = None; reconstruction = None
        identity_status = "undefined_support_conflict_infinity_minus_infinity"
    else:
        model_log_product = np.log(model_marginal_support).sum(axis=1)
        if np.any(Q[active] == 0):
            J = -math.inf; reconstruction = math.inf; residual = None
            identity_status = "extended_positive_infinity_agreement"
        else:
            J = float(P[active] @ (np.log(Q[active]) - model_log_product))
            reconstruction = TC + marginal_sum - J
            residual = float(KL - reconstruction)
            identity_status = "finite"
    qualified = identity_status == "finite" and abs(residual) <= 1e-12
    return snapshot(dict(raw_KL=KL, teacher_TC=TC, marginal_KL_by_bit=marginal, marginal_KL_sum=marginal_sum,
                         dependence_score=J, teacher_marginals=pj, model_marginals=qj,
                         teacher_mass=float(P.sum()), model_mass=float(Q.sum()), reconstructed_raw_KL=reconstruction,
                         identity_residual=residual, identity_status=identity_status, identity_qualified=qualified,
                         identity_absolute_tolerance=1e-12, masses_renormalized=False, rounding_negatives_clipped=False,
                         dependence_score_is_dependency_KL=False, coordinate_causal_attribution=False))


def gain_decomposition(P, low, high, saved_gain=None):
    L, H = risk_decomposition(P, low), risk_decomposition(P, high)
    P, _ = pmf_array(P); active = P > 0
    if np.any(np.asarray(low)[active] == 0) or np.any(np.asarray(high)[active] == 0):
        direct = marginal_gain = dependence_gain = residual = None
        status = "undefined_endpoint_support_conflict"
    else:
        direct = float(P[active] @ (np.log(np.asarray(high)[active]) - np.log(np.asarray(low)[active])))
        marginal_gain = L["marginal_KL_sum"] - H["marginal_KL_sum"]
        dependence_gain = H["dependence_score"] - L["dependence_score"]
        residual = direct - (marginal_gain + dependence_gain)
        status = "finite"
    return dict(saved_gain=saved_gain, direct_gain=direct, marginal_gain=marginal_gain,
                dependence_score_gain=dependence_gain, identity_residual=residual, identity_status=status,
                identity_qualified=status == "finite" and abs(residual) <= 1e-12,
                saved_gain_residual=None if direct is None or saved_gain is None else direct - saved_gain,
                dependence_component_is_dependency_KL=False, radial_angle_causal_decomposition=False)


def independent_bernoulli_mle(q):
    q, m = pmf_array(q, "empirical frequency")
    if not math.isclose(float(q.sum()), 1., abs_tol=1e-12, rel_tol=1e-10):
        raise ValueError("empirical frequency must have mass one; no normalization")
    Y = patterns(m); p = q @ Y
    if np.any(p < 0) or np.any(p > 1): raise ValueError("empirical marginal outside [0,1]")
    Q = np.prod(np.where(Y, p, 1 - p), axis=1)
    positive = q > 0
    empirical_NLL = math.inf if np.any(Q[positive] == 0) else -float(q[positive] @ np.log(Q[positive]))
    return snapshot(dict(one_probabilities=p, pmf=Q, raw_marginals=raw_marginals(Q),
                         model_mass=float(Q.sum()), empirical_NLL=empirical_NLL,
                         boundary=bool(np.any((p == 0) | (p == 1))), boundary_coordinates=np.flatnonzero((p == 0) | (p == 1)),
                         analytic_fit=True, optimizer_updates=0, no_smoothing=True, target_renormalized=False))


def recode_weights(p, signs):
    p, m = pmf_array(p); signs = np.asarray(signs)
    if signs.shape != (m,) or not np.all((signs == -1) | (signs == 1)): raise ValueError("invalid saved signs")
    Y = patterns(m); recoded = np.where(signs > 0, Y, 1 - Y)
    index = recoded @ (2 ** np.arange(m - 1, -1, -1, dtype=np.int64))
    result = np.empty_like(p); result[index] = p
    return result


def numerical_readout(values, main_index=(1, 1)):
    if values is None:
        return dict(defined=False, main_value=None, max_difference=None, tau_num=None, sign="unresolved",
                    all_variants_beyond_band=False, reason="direction_undefined")
    values = np.asarray(values, dtype=float)
    if values.shape != (2, 4) or not np.isfinite(values).all():
        return dict(defined=False, main_value=None, max_difference=None, tau_num=None, sign="unresolved",
                    all_variants_beyond_band=False, reason="nonfinite_or_missing_combination")
    main = float(values[main_index]); difference = float(np.max(abs(values - main)))
    tau = max(1e-8, 10. * difference)
    sign = "positive" if np.all(values > tau) else "negative" if np.all(values < -tau) else "unresolved"
    return dict(defined=True, main_value=main, minimum=float(values.min()), maximum=float(values.max()),
                max_difference=difference, tau_num=tau, sign=sign, all_variants_beyond_band=sign != "unresolved",
                floor=1e-8, discrepancy_multiplier=10., original_D3_gate_changed=False,
                statistical_or_rigorous_error_bound=False)


def vector_norms(g):
    g = np.asarray(g, dtype=float)
    if g.ndim != 1 or len(g) % 2 or not np.isfinite(g).all(): raise ValueError("invalid gradient vector")
    m = len(g) // 2
    norm = lambda v: dict(L2=math.hypot(*(float(x) for x in v)), Linf=float(np.max(abs(v))))
    return dict(total=norm(g), bias=norm(g[:m]), row_amplitude=norm(g[m:]))


def derive_unit(unit, q, anchor, diagnosis, risk, teacher_pmfs, teacher_rules):
    """Pure saved-object derivation; no kernel or optimizer is called."""
    q, m = pmf_array(q, "empirical frequency")
    theta = np.asarray(anchor["theta_recoded"], dtype=float)
    signs = np.asarray(anchor["signs"])
    require(theta.shape == (2 * m,) and np.isfinite(theta).all(), "anchor dimensions")
    require(np.array_equal(anchor["q_original"], q), "anchor empirical target unchanged")
    qr = recode_weights(q, signs)
    require(np.array_equal(qr, anchor["q_recoded"]), "saved recoding identity")
    teachers = np.asarray(teacher_pmfs, dtype=float)
    require(teachers.shape == (4, len(q)), "four saved teacher PMF variants")
    require(public(teacher_rules) == public(risk["rules"]), "teacher/endpoint rule ordering")
    require(risk["main_index"] == 1, "original GH60 main rule")
    teacher_recoded = np.array([recode_weights(p, signs) for p in teachers])
    saved_direction = diagnosis["numerics"]["direction"]
    d = None if saved_direction is None else np.asarray(saved_direction).copy()
    if d is not None:
        require(d.shape == (m,) and np.isfinite(d).all(), "saved direction shape")
        require(d.tobytes() == np.asarray(saved_direction).tobytes(), "fixed direction bytes")
    H = np.empty((2, 4, m, m)); G = np.empty((2, 4, 2 * m))
    empirical = {name: np.empty((2, 2 * m)) for name in ("8", "64", "inf")}
    empirical_curvature = None if d is None else np.empty(2)
    aggregation = {}
    for ri, rule in enumerate(("C", "D")):
        raw = diagnosis["reference_only"]["raw"][rule]
        finite = diagnosis["finite_extra"]["raw"][rule]
        require(raw["K_values"] == ["inf"] or raw["K_values"] == ("inf",), "reference block K")
        require(list(finite["K_values"]) == [8, 64], "finite block K order")
        require(np.array_equal(raw["patterns"], patterns(m)), "reference label ordering")
        label_H = np.asarray(raw["label_Hangle"])[0]
        label_G = np.asarray(raw["label_gradient"])[0]
        require(label_H.shape == (len(q), m, m) and label_G.shape == (len(q), 2 * m), "label derivative shapes")
        require(np.isfinite(label_H).all() and np.isfinite(label_G).all(), "saved label derivatives finite")
        H[ri] = np.einsum("tp,pij->tij", teacher_recoded, label_H, optimize=True)
        G[ri] = teacher_recoded @ label_G
        empirical["inf"][ri] = np.asarray(raw["gradient"])[0, 0].copy()
        empirical["8"][ri] = np.asarray(finite["gradient"])[0, 0].copy()
        empirical["64"][ri] = np.asarray(finite["gradient"])[0, 1].copy()
        aggregation[rule] = dict(Hangle_max_error=float(np.max(abs(np.einsum("p,pij->ij", qr, label_H) - raw["Hangle"][0, 0]))),
                                 gradient_max_error=float(np.max(abs(qr @ label_G - empirical["inf"][ri]))))
        if d is not None: empirical_curvature[ri] = float(d @ raw["Hangle"][0, 0] @ d)
    curvature = None if d is None else np.einsum("i,rtij,j->rt", d, H, d)
    norms = {K: {rule: vector_norms(values[i]) for i, rule in enumerate(("C", "D"))} for K, values in empirical.items()}
    g64 = empirical["64"][1].copy(); gnorm = norms["64"]["D"]["total"]["L2"]
    if gnorm <= 1e-10:
        direction_ba = derivatives = None
        channel = dict(defined=False, empirical_F64_D_L2=gnorm, direction_recoded=None,
                       directional_derivatives=None, readout=numerical_readout(None),
                       reason="empirical_F64_gradient_norm_at_or_below_1e-10", maximum_forward_box_step=None)
    else:
        direction_ba = -g64 / gnorm
        derivatives = np.einsum("rtj,j->rt", G, direction_ba)
        lower, upper = np.r_[np.full(m, -2.), np.full(m, .05)], np.full(2 * m, 2.)
        ratios = [(upper[j] - theta[j]) / direction_ba[j] if direction_ba[j] > 0 else
                  (lower[j] - theta[j]) / direction_ba[j] for j in range(2 * m) if direction_ba[j] != 0]
        step = float(min(ratios))
        channel = dict(defined=True, empirical_F64_D_L2=gnorm, direction_recoded=direction_ba,
                       directional_derivatives=derivatives, readout=numerical_readout(derivatives),
                       maximum_forward_box_step=step, positive_feasible_local_step=step > 0,
                       finite_step_executed=False, interpretation="bias and row-amplitude first-order local channel, not 600-step causal contribution")
    signed_map = np.r_[signs, signs]
    channel["direction_signed_original"] = None if direction_ba is None else signed_map * direction_ba
    account = {}
    for arm in ("LOW", "HIGH", "REF"):
        models = np.asarray(risk["pmfs"][arm])
        require(models.shape == teachers.shape, "endpoint PMF rule shape")
        account[arm] = [risk_decomposition(P, Q) for P, Q in zip(teachers, models)]
    gains = [gain_decomposition(teachers[i], risk["pmfs"]["LOW"][i], risk["pmfs"]["HIGH"][i], float(risk["paired_gain"][i])) for i in range(4)]
    baseline = independent_bernoulli_mle(q)
    baseline["raw_KL"] = np.array([relative_entropy(P, baseline["pmf"]) for P in teachers])
    baseline["teacher_rules"] = teacher_rules
    baseline["teacher_masses"] = teachers.sum(axis=1)
    allfinite = bool(np.isfinite(baseline["raw_KL"]).all())
    discrepancy = float(np.max(abs(baseline["raw_KL"] - baseline["raw_KL"][1]))) if allfinite else None
    mass_error = max(abs(baseline["model_mass"] - 1.), float(np.max(abs(teachers.sum(axis=1) - 1.))))
    baseline.update(risk_status="finite" if allfinite else "positive_infinity_from_MLE_support",
                    risk_cross_rule_difference=discrepancy, mass_error=mass_error,
                    numerical_qualified=allfinite and mass_error <= 2e-12 and discrepancy <= 2e-6,
                    infinite_risk_smoothed_or_replaced=False)
    return snapshot(dict(schema="prospective_explanation_unit_v1", unit=unit, theta_recoded=theta,
                         theta_signed=np.r_[signs * theta[:m], signs * theta[m:]], signs=signs,
                         empirical_frequency=q, recoded_empirical_frequency=qr, teacher_rules=teacher_rules,
                         recoded_teacher_pmfs=teacher_recoded, teacher_masses=teachers.sum(axis=1),
                         fixed_empirical_direction=d, original_direction_bytes_preserved=d is not None,
                         original_A=bool(diagnosis["full_raise"]), original_B=bool(diagnosis["ref_raise"]),
                         original_full_level=diagnosis["full_level"], original_full_reasons=diagnosis["full_reasons"],
                         original_ref_reasons=diagnosis["ref_reasons"], original_D0_reason_categories=diagnosis["D0_reason_categories"],
                         original_diagnostic_thresholds=diagnosis["thresholds"], original_reference_sign=diagnosis["signs"]["reference"],
                         population_Hangle=H, population_gradient=G, population_gradient_signed_original=G * signed_map,
                         population_direction_curvature=curvature, population_curvature_readout=numerical_readout(curvature),
                         empirical_gradients=empirical, empirical_gradient_norms=norms,
                         empirical_fixed_direction_curvature=empirical_curvature, empirical_reaggregation_errors=aggregation,
                         gradient_channel=channel, anchor_boundary=anchor["box"],
                         risk_decomposition=account, gain_decomposition=gains, independent_baseline=baseline,
                         saved_paired_gain=np.asarray(risk["paired_gain"]).copy(), saved_KL=risk["KL"],
                         original_risk_qualified=bool(risk["LOW_HIGH_risk_qualified"]),
                         original_risk_qualified_by_arm=risk["qualified_by_arm"],
                         coordinate_system="positive-amplitude recoding; signed-original gradient map diag(s,s) is orthogonal",
                         no_new_kernel_or_training=True, population_direction_reselected=False, original_alarms_changed=False,
                         post_hoc_explanation_not_independent_validation=True))


class SavedReader:
    """Hash-bound selective restoration; unused priors/training tapes stay packed."""
    def __init__(self, locked_files):
        self.locked_files = locked_files; self.hashes = {}; self.array_reads = {}

    def selected(self, stem, selections):
        metadata_path = Path(stem).with_suffix(".json")
        require(rel(metadata_path) in self.locked_files, "unlocked input metadata")
        metadata = read(metadata_path); raw = ROOT / metadata["raw"]["path"]
        require(raw.resolve().is_relative_to(OLD), "input array outside sealed dev package")
        require(rel(raw) in self.locked_files, "unlocked input array")
        for path in (metadata_path, raw):
            expected = self.locked_files[rel(path)]
            require(sha(path) == expected["sha256"], "old input hash changed: " + rel(path))
            self.hashes[rel(path)] = expected["sha256"]
        require(metadata["raw"]["sha256"] == self.hashes[rel(raw)], "metadata/lock array hash")
        result = {}
        with np.load(raw, allow_pickle=False) as archive:
            def decode(value):
                if isinstance(value, dict) and "__ndarray__" in value:
                    a = archive[value["__ndarray__"]].copy()
                    require(list(a.shape) == value["shape"] and str(a.dtype) == value["dtype"], "array schema")
                    self.array_reads.setdefault(rel(raw), set()).add(value["__ndarray__"])
                    return a
                if isinstance(value, dict) and "__nonfinite_float__" in value: return float(value["__nonfinite_float__"])
                if isinstance(value, dict): return {k: decode(v) for k, v in value.items()}
                if isinstance(value, list): return [decode(v) for v in value]
                return value
            for selection in selections:
                path = selection.split("."); value = metadata["tree"]
                for part in path: value = value[part]
                cursor = result
                for part in path[:-1]: cursor = cursor.setdefault(part, {})
                cursor[path[-1]] = decode(value)
        return result


def _bind_sources(mapping):
    for name, item in mapping.items():
        path = (ROOT / name).resolve(); require(path.is_relative_to(ROOT), "source outside repository")
        digest = item if isinstance(item, str) else item["sha256"]
        require(sha(path) == digest, "source hash mismatch: " + name)
        if isinstance(item, dict) and "bytes" in item: require(path.stat().st_size == item["bytes"], "source size mismatch")


def verify_freeze():
    frozen = read(OUT / "FREEZE.json")
    require(frozen["schema"] == "prospective_explanation_freeze_v1", "new freeze schema")
    require(frozen["baseline"] == BASE and frozen["handoff_sha256"] == HANDOFF_SHA, "new freeze provenance")
    require(frozen["production_not_started"] is True and frozen["config"] == config_dump(), "new freeze/config")
    _bind_sources(frozen["sources"])
    required = [Path(__file__).resolve(), Path(__file__).with_name("prospective_explanation_compute_test.py"),
                DOC / "CONTRACT.md", OUT / "HANDOFF_INPUT.md", Path(legacy.__file__),
                OLD / "DELIVERY_MANIFEST.json", OLD / "CONFIG.json", OLD / "SUMMARY.json", PAPER_MANIFEST,
                *[OLD / (name + ".json") for name in LOCKS]]
    require(all(rel(p) in frozen["sources"] for p in required), "required source not frozen")
    require(sha(OUT / "HANDOFF_INPUT.md") == HANDOFF_SHA, "archived feedback")
    require(sha(OLD / "DELIVERY_MANIFEST.json") == OLD_MANIFEST_SHA, "old development delivery seal")
    require(sha(PAPER_MANIFEST) == PAPER_MANIFEST_SHA, "old manuscript seal")
    config = read(OLD / "CONFIG.json")
    require(config["units"] == config_dump()["original_units"], "all original twelve units")
    files = {}; hashes = {}
    old_freeze_sha = sha(OLD / "FREEZE.json")
    for name in LOCKS:
        path = OLD / (name + ".json"); lock = read(path)
        require(lock["freeze_sha256"] == old_freeze_sha, "old global lock freeze")
        hashes[rel(path)] = sha(path)
        for name, entry in lock["files"].items():
            require(entry["path"] == name and (ROOT / name).resolve().is_relative_to(OLD), "old input boundary")
            if name in files: require(files[name] == entry, "conflicting old lock entries")
            files[name] = entry
    # Verify every old lock binding without interpreting old training arrays.
    _bind_sources(files)
    hashes.update({name: entry["sha256"] for name, entry in files.items()})
    return frozen, config, files, hashes


def scalar_row(unit):
    main = (1, 1)
    risk = unit["risk_decomposition"]
    finite_identities = [r for rows in risk.values() for r in rows if r["identity_residual"] is not None]
    channel = unit["gradient_channel"]
    return public(dict(unit=unit["unit"]["id"], D=unit["unit"]["teacher"] + 1, N=unit["unit"]["N"], rep=unit["unit"]["rep"],
                       A=unit["original_A"], B=unit["original_B"], full_level=unit["original_full_level"],
                       full_reasons=unit["original_full_reasons"], reference_reasons=unit["original_ref_reasons"],
                       original_reference_sign=unit["original_reference_sign"],
                       empirical_curvature=unit["empirical_fixed_direction_curvature"],
                       population_curvature_variants=unit["population_direction_curvature"],
                       population_curvature=unit["population_curvature_readout"],
                       empirical_gradient_norms=unit["empirical_gradient_norms"],
                       gradient_channel_defined=channel["defined"], gradient_channel=channel["readout"],
                       gradient_channel_variants=channel["directional_derivatives"],
                       anchor_box_margin=unit["anchor_boundary"]["minimum_margin"],
                       original_risk_qualified=unit["original_risk_qualified"], saved_delta_R=float(unit["saved_paired_gain"][1]),
                       main_risk_accounts={arm: {k: rows[1][k] for k in ("raw_KL", "teacher_TC", "marginal_KL_sum", "dependence_score", "identity_residual", "identity_status", "identity_qualified")}
                                           for arm, rows in risk.items()},
                       main_gain_account=unit["gain_decomposition"][1],
                       maximum_finite_risk_identity_residual=max((abs(r["identity_residual"]) for r in finite_identities), default=None),
                       all_joint_risk_identities_qualified=all(r["identity_qualified"] for rows in risk.values() for r in rows),
                       independent_MLE_risk=unit["independent_baseline"]["raw_KL"],
                       independent_MLE_boundary=unit["independent_baseline"]["boundary"],
                       independent_MLE_risk_status=unit["independent_baseline"]["risk_status"],
                       independent_MLE_numerical_qualified=unit["independent_baseline"]["numerical_qualified"]))


def descriptive_groups(rows):
    def group(items):
        alarms = [r for r in items if r["A"]]
        return dict(units=len(items), full_alarms=len(alarms), reference_alarms=sum(r["B"] for r in items),
                    D0=sum(r["full_level"] == "D0" for r in items),
                    delta_R_positive=sum(r["saved_delta_R"] > 0 for r in items),
                    delta_R_above_0p001=sum(r["saved_delta_R"] > .001 for r in items),
                    mean_saved_delta_R=math.fsum(r["saved_delta_R"] for r in items) / len(items) if items else None,
                    full_alarm_mean_saved_delta_R=math.fsum(r["saved_delta_R"] for r in alarms) / len(alarms) if alarms else None)
    alarms = [r for r in rows if r["A"]]
    removed = max(alarms, key=lambda r: r["saved_delta_R"]) if alarms else None
    remainder = [r for r in alarms if r is not removed]
    return dict(all_units=group(rows), by_N={str(N): group([r for r in rows if r["N"] == N]) for N in sorted({r["N"] for r in rows})},
                leave_largest_alarm=dict(removed_unit=None if removed is None else removed["unit"],
                    removed_saved_delta_R=None if removed is None else removed["saved_delta_R"], original_alarm_count=len(alarms),
                    remaining_alarm_count=len(remainder), remaining_mean_saved_delta_R=math.fsum(r["saved_delta_R"] for r in remainder) / len(remainder) if remainder else None,
                    tie_rule="first in original unit order", post_hoc_descriptive_only=True, all_original_units_retained=True,
                    selection_rule_or_inference_changed=False))


def teacher_distance(teachers):
    values = np.array([relative_entropy(P, Q) for P, Q in zip(teachers[1]["pmfs"], teachers[0]["pmfs"])])
    return dict(direction="raw KL(P_D2 || P_D1)", values=values, main_index=1, main_value=float(values[1]),
                maximum_cross_rule_difference=float(np.max(abs(values - values[1]))),
                rules=teachers[0]["rules"], D1_is_a_legal_one_factor_model=True,
                numerical_value_is_not_a_certified_upper_bound=True,
                pinsker_numeric_reference={str(N): np.sqrt(N * values / 2) if np.all(values >= 0) else None for N in (1024, 4096)},
                new_teacher_PMF_calls=0, strict_numerical_certificate=False)


def compute_all():
    frozen, old_config, locked_files, hashes = verify_freeze()
    reader = SavedReader(locked_files)
    teachers = [reader.selected(OLD / f"teacher_D{i}", ["pmfs", "rules"]) for i in (1, 2, 3)]
    old_summary = read(OLD / "SUMMARY.json"); old_rows = {r["unit"]: r for r in old_summary["rows"]}
    derived = []
    diag_fields = ["full_level", "full_raise", "ref_raise", "full_reasons", "ref_reasons", "D0_reason_categories", "thresholds", "signs", "numerics.direction"]
    for rule in ("C", "D"):
        diag_fields += [f"reference_only.raw.{rule}.{key}" for key in ("label_Hangle", "label_gradient", "Hangle", "gradient", "patterns", "K_values")]
        diag_fields += [f"finite_extra.raw.{rule}.{key}" for key in ("gradient", "K_values")]
    for unit in old_config["units"]:
        folder = OLD / "units" / unit["id"]
        data = reader.selected(folder / "inputs", ["q", "unit"])
        anchor = reader.selected(folder / "anchor", ["theta_recoded", "signs", "q_original", "q_recoded", "box"])
        diag = reader.selected(folder / "diagnosis", diag_fields)
        risk = reader.selected(folder / "risk", ["pmfs", "KL", "paired_gain", "main_index", "rules", "qualified_by_arm", "LOW_HIGH_risk_qualified"])
        require(data["unit"] == unit, "source unit identity")
        target = teachers[unit["teacher"]]
        row = derive_unit(unit, data["q"], anchor, diag, risk, target["pmfs"], target["rules"])
        old = old_rows[unit["id"]]
        require(row["original_A"] == old["A"] and row["original_B"] == old["B"] and row["original_full_level"] == old["full_level"], "old alarms unchanged")
        require(float(row["saved_paired_gain"][1]) == old["delta_R"], "published old paired gain unchanged")
        derived.append(row)
    rows = [scalar_row(row) for row in derived]
    dist = teacher_distance(teachers)
    counts = dict(units=len(derived), saved_label_Hangle_reweightings=len(derived) * 8,
                  saved_label_gradient_reweightings=len(derived) * 8, empirical_gradient_vectors_read=len(derived) * 6,
                  endpoint_risk_bookkeeping_cells=len(derived) * 3 * 4, analytic_independent_fits=len(derived),
                  independent_baseline_teacher_scores=len(derived) * 4,
                  producer_new_quadrature_calls=0, new_kernel_calls=0, new_optimizer_updates=0,
                  new_training_units=0, new_random_draws=0, formal_units=0,
                  source_array_archives_selectively_decoded=len(reader.array_reads),
                  source_array_leaves_decoded=sum(len(v) for v in reader.array_reads.values()))
    summary = dict(schema="prospective_explanation_summary_v1", rows=rows, grouping=descriptive_groups(rows),
                   teacher_D2_to_D1=dist, resources=counts,
                   all_joint_risk_identities_qualified=all(r["all_joint_risk_identities_qualified"] for r in rows),
                   original_units_all_retained=True, original_directions_or_alarms_changed=False,
                   scientific_closure_judgment_owner="primary author", confirmatory_inference=False,
                   radial_angle_or_trajectory_causal_attribution=False, manuscript_automatically_updated=False)
    inputs = dict(hashes); inputs.update(reader.hashes)
    for p in (OLD / "DELIVERY_MANIFEST.json", OLD / "SUMMARY.json", OLD / "CONFIG.json", OLD / "FREEZE.json", OUT / "FREEZE.json"):
        inputs[rel(p)] = sha(p)
    return dict(units=derived, summary=summary, input_hashes=inputs,
                source_array_read_scope={name: sorted(keys) for name, keys in reader.array_reads.items()})


def same_tree(a, b, context="result"):
    if isinstance(a, np.ndarray):
        require(isinstance(b, np.ndarray) and a.shape == b.shape and a.dtype == b.dtype, context + " array schema")
        require(np.array_equal(a, b, equal_nan=True) if a.dtype.kind in "fc" else np.array_equal(a, b), context + " array contents")
    elif isinstance(a, dict):
        require(isinstance(b, dict) and a.keys() == b.keys(), context + " dict keys")
        for key in a: same_tree(a[key], b[key], context + "." + key)
    elif isinstance(a, (list, tuple)):
        require(isinstance(b, (list, tuple)) and len(a) == len(b), context + " sequence")
        for i, (x, y) in enumerate(zip(a, b)): same_tree(x, y, context + f"[{i}]")
    else: require(a == b, context + " scalar")


def readable(summary):
    lines = ["# Saved-object explanation: machine-derived readout", "", "Post-hoc description of all twelve original units; no new training or alarm/direction selection.", "",
             "|unit|A/B|original level|empirical fixed-d curvature (D)|population fixed-d curvature main|new population sign|radial/bias derivative main|old delta R|marginal gain|dependence-score gain|independent MLE risk main|",
             "|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    def fmt(x):
        if x is None: return "undefined"
        if isinstance(x, dict) and "__nonfinite_float__" in x: return x["__nonfinite_float__"]
        if isinstance(x, (float, np.floating)): return f"{x:.10g}"
        return str(x)
    for row in summary["rows"]:
        curv = row["population_curvature"]; channel = row["gradient_channel"]; gain = row["main_gain_account"]
        values = [row["unit"], f"{int(row['A'])}/{int(row['B'])}", row["full_level"],
                  None if row["empirical_curvature"] is None else row["empirical_curvature"][1],
                  curv["main_value"], curv["sign"], channel["main_value"], row["saved_delta_R"],
                  gain["marginal_gain"], gain["dependence_score_gain"], row["independent_MLE_risk"][1]]
        lines.append("|" + "|".join(fmt(x) for x in values) + "|")
    lines += ["", "The two gain columns are marginal-error improvement and joint dependence-score improvement. They are not coordinate-causal contributions, and the second is not a dependency KL.",
              "", "Population signs require all eight fixed C/D-by-teacher-rule variants beyond the same new numerical band. The original diagnosis and its 5e-4 floor are unchanged.", ""]
    return "\n".join(lines)


def main(mode):
    if mode == "run":
        require(not (OUT / "RESULTS.json").exists() and not (OUT / "SUMMARY.json").exists(), "no overwrite or implicit rerun")
    elif mode == "check": require((OUT / "RESULTS.json").is_file(), "saved producer result required")
    tick = time.perf_counter()
    produced = compute_all()
    source = sha(Path(__file__))
    if mode == "check":
        result = read(OUT / "RESULTS.json")
        require(result["source_sha256"] == source and result["freeze_sha256"] == sha(OUT / "FREEZE.json"), "producer source/freeze")
        same_tree(produced["input_hashes"], result["input_hashes"])
        same_tree(produced["source_array_read_scope"], result["source_array_read_scope"])
        for unit in produced["units"]:
            saved = legacy.restore(OUT / "units" / unit["unit"]["id"] / "explanation")
            same_tree(unit, saved, unit["unit"]["id"])
        summary = read(OUT / "SUMMARY.json")
        same_tree(public(produced["summary"]), summary["derived"])
        _bind_sources(result["artifacts"])
        print(json.dumps(dict(passed=True, mode="check", units=len(produced["units"]), new_quadrature_calls=0,
                              new_training=0, saved_array_derivation_repeated=True, seconds=time.perf_counter() - tick)))
        return
    artifacts = []; saved_units = []
    for unit in produced["units"]:
        stem = OUT / "units" / unit["unit"]["id"] / "explanation"
        require(stem.resolve().is_relative_to(OUT), "unit output boundary")
        legacy.persist(stem, unit)
        paths = [stem.with_suffix(ext) for ext in (".json", ".npz")]
        artifacts.extend(paths)
        saved_units.append(dict(unit=unit["unit"]["id"], json=binding(paths[0]), npz=binding(paths[1])))
    save(OUT / "SUMMARY.json", dict(created_utc=now(), source_sha256=source, freeze_sha256=sha(OUT / "FREEZE.json"),
         input_hashes=produced["input_hashes"], derived=produced["summary"]))
    table = OUT / "READABLE_UNIT_TABLE.md"
    with table.open("x", encoding="utf-8", newline="\n") as f: f.write(readable(produced["summary"]))
    artifacts.extend([OUT / "SUMMARY.json", table])
    save(OUT / "RESULTS.json", dict(schema="prospective_explanation_results_v1", status="completed_saved_object_derivation",
         created_utc=now(), source_sha256=source, freeze_sha256=sha(OUT / "FREEZE.json"), baseline=BASE, handoff_sha256=HANDOFF_SHA,
         input_hashes=produced["input_hashes"], source_array_read_scope=produced["source_array_read_scope"],
         unit_artifacts=saved_units, artifacts={rel(p): binding(p) for p in artifacts},
         resources=produced["summary"]["resources"], seconds=time.perf_counter() - tick,
         post_hoc_not_independent_validation=True, producer_new_quadrature_calls=0,
         independent_numerical_verification_separate=True, scientific_interpretation_owner="primary author"))
    print(json.dumps(dict(passed=True, mode="run", units=len(produced["units"]), new_quadrature_calls=0, new_training=0)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", action="store_true"); group.add_argument("--self-test", action="store_true")
    group.add_argument("--run", action="store_true"); group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.config: print(json.dumps(config_dump(), indent=2))
    elif args.self_test:
        import prospective_explanation_compute_test as tests
        raise SystemExit(tests.run_tests())
    else:
        try: main("run" if args.run else "check")
        except BaseException as error:
            if args.run and (OUT / "FREEZE.json").is_file() and not (OUT / "FAILURE.json").exists():
                save(OUT / "FAILURE.json", dict(created_utc=now(), status="failed_no_automatic_retry", reason=str(error),
                     traceback=traceback.format_exc(), source_sha256=sha(Path(__file__)), new_quadrature_calls=0, new_training=0))
            raise
