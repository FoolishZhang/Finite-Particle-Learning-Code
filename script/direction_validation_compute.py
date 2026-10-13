"""Fixed-direction feasibility on saved development objects, not validation data.

Production generates exactly four candidate models by four fixed GH rules per
unit. All remaining operations are finite-alphabet algebra. ``--check`` restores
these PMFs and redoes the algebra without calling a quadrature kernel. Raw PMFs
are never corrected: population moments explicitly use P/sum(P), and retain
raw means and the normalization correction. Numeric rule agreement is not a
rigorous integration-error bound; oracle planning is not a confidence interval,
power estimate, necessary sample-size bound, or a deployable budget policy.
"""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
import joint_dependence_kernel as joint
import prospective_diagnostic_dev as legacy
import prospective_explanation_compute as explanation
import direction_validation_statistics as statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/direction_validation_v1_20261007"
DOC = ROOT / "docs/08_research_研究工作/DirectionValidation_v1_20261007"
OLD = ROOT / "artifacts/prospective_diagnostic_dev_v1_20261007"
EXPLANATION = ROOT / "artifacts/prospective_explanation_v1_20261007"
BASE = "df31ecd3d93c50f72859d364f1e5fd22faf0eed4"
HANDOFF_SHA = "11c9d0d8518c8f7fff6b1abd74bde35abe4e00a2ff49d75228d1126c9c1f8f89"
DEV_MANIFEST_SHA = "2b537dd6e86f9547a1e540dd429b14a86a46c04d327ddb97c5b58fb783a9abd8"
EXPLANATION_MANIFEST_SHA = "8a3ca2e301540e1e89f41dc7d69cc0feafc84125899ffb0c69f07fdb203feabd"
T_VALUES = (.02, .1, .25)
PMF_RULES = ((40, 0.), (60, 0.), (80, 0.), (72, math.pi / 7))
N_GRID = (1024, 4096, 16384, 65536, 262144, 1048576)
DELTA = .05 / (48 * 6)
GATES = dict(PMF_mass=2e-12, PMF_maxdiff=2e-8, z_maxdiff=1e-9,
             paired_mean_maxdiff=1e-10, label_curvature_maxdiff=2e-7,
             source_curvature_identity=1e-12)


def config_dump():
    return dict(schema="direction_validation_config_v1", baseline=BASE,
        original_units=legacy.units(), t_values=list(T_VALUES), PMF_rules=[list(r) for r in PMF_RULES],
        main_PMF_index=1, main_teacher_index=1, main_curvature_index=1,
        conditional_rule_order=["C", "D"], PMF_model_order=[0., *T_VALUES],
        candidate_PMF_calls=192, candidate_models=48, gates=GATES.copy(),
        cross_rule_difference="maximum pairwise; relative-to-main also retained",
        paired_mean_gate="maximum over candidate rule by teacher rule raw means",
        statistical_alpha=.05, score_families=48, looks=6, per_score_per_look_delta=DELTA,
        N_grid=list(N_GRID), maximum_planning_n=2**30,
        curvature_margins={"zero": 0., "useful": 1e-4},
        finite_gain_margins="0, t^2*1e-4/2, 1e-5, 1e-4; same score bound reused",
        enumerated_score_transfer="e_minus=max_y(Xc-Xt_norm), not clipped; curvature threshold e_minus+2*eta/t^2",
        transfer_new_families=0, transfer_is_exact_score_table_algebra_not_quadrature_certificate=True,
        population_variance_EB_is_oracle_proxy=True, normalized_moments_explicit=True,
        raw_teacher_probabilities_preserved=True, old_A_B_and_d_unchanged=True,
        new_observations=0, new_random_draws=0, new_training=0,
        new_teacher_PMF_calls=0, new_curvature_integrals=0,
        independent_validation=False, automatically_escalate_quadrature=False)


read, rel, sha = explanation.read, explanation.rel, explanation.sha
require, snapshot, public = explanation.require, explanation.snapshot, explanation.public
binding, same_tree = explanation.binding, explanation.same_tree


def now(): return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    require(path.resolve().is_relative_to(OUT), "output outside new package")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(public(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def _real_vector(x, name):
    x = np.asarray(x)
    if np.iscomplexobj(x): raise ValueError(name + " must be real")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError(name + " must be a finite vector of length >= 2")
    return x.copy()


def recoded_indices(signs):
    signs = np.asarray(signs)
    if signs.ndim != 1 or len(signs) < 2 or not np.isin(signs, (-1, 1)).all():
        raise ValueError("signs must be a vector of +/-1")
    bits = joint.patterns(len(signs))
    recoded = np.where(signs[None, :] < 0, 1 - bits, bits)
    return np.asarray(recoded @ (1 << np.arange(len(signs) - 1, -1, -1)), dtype=np.int64)


def path_models(b, signed_a, d, t_values=T_VALUES):
    b, a, d = (_real_vector(x, n) for x, n in ((b, "b"), (signed_a, "signed_a"), (d, "d")))
    if b.shape != a.shape or b.shape != d.shape: raise ValueError("path vector dimensions differ")
    t = np.asarray(t_values, dtype=float)
    if t.ndim != 1 or not len(t) or not np.isfinite(t).all() or np.any(t <= 0):
        raise ValueError("nonzero fixed amplitudes must be finite and positive")
    angles = np.r_[0., t][:, None] * d[None, :]
    W = a[None, :, None] * np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    return b, a, d, W


def generate_candidate_pmfs(b, signed_a, d, *, t_values=T_VALUES, rules=PMF_RULES, pmf_fn=None):
    """Only this function calls model_pmf; each Q0 is computed once per rule.

    ``pmfs`` axes are model(0 then fixed t), candidate rule, original label.
    Numerical failures are saved, not retried or silently repaired.
    """
    b, a, d, W = path_models(b, signed_a, d, t_values)
    rules = [(int(order), float(angle)) for order, angle in rules]
    if not rules or any(o < 1 or not math.isfinite(a0) for o, a0 in rules):
        raise ValueError("invalid fixed PMF rule")
    fn = joint.model_pmf if pmf_fn is None else pmf_fn
    pmfs = np.full((len(W), len(rules), 1 << len(b)), np.nan)
    calls = []
    for mi, model in enumerate(W):
        for ri, (order, angle) in enumerate(rules):
            tick = time.perf_counter(); error = None
            try:
                result = np.asarray(fn(b.copy(), model.copy(), order=order, angle=angle), dtype=float)
                if result.shape != (1 << len(b),): raise ValueError("PMF output shape")
                pmfs[mi, ri] = result
                valid = bool(np.isfinite(result).all() and np.all(result > 0))
                status = "finite_positive" if valid else "nonpositive_or_nonfinite_PMF"
            except (FloatingPointError, ValueError, OverflowError) as exc:
                status, error = "PMF_call_failed", str(exc)
            calls.append(dict(model_index=mi, rule_index=ri, order=order, angle=angle,
                t=float(np.r_[0., t_values][mi]), quadrature_nodes=order**2,
                node_pattern_products=order**2 * (1 << len(b)),
                seconds=time.perf_counter() - tick, status=status, error=error))
    return snapshot(dict(schema="direction_validation_candidate_pmfs_v1", b=b, signed_a=a,
        fixed_d=d, t_values=list(t_values), rules=[list(r) for r in rules], model_W=W, pmfs=pmfs,
        calls=calls, resources=dict(PMF_calls=len(calls),
            quadrature_nodes=sum(c["quadrature_nodes"] for c in calls),
            node_pattern_products=sum(c["node_pattern_products"] for c in calls),
            seconds=math.fsum(c["seconds"] for c in calls), new_random_draws=0, new_training=0),
        raw_mass_preserved=True, baseline_repeated_per_t=False))


def _difference_summary(values):
    v = np.asarray(values, dtype=float)
    if not np.isfinite(v).all():
        return dict(max_pairwise=None, max_from_main=None, finite=False)
    # First axis enumerates rules; arbitrary trailing dimensions are retained.
    return dict(max_pairwise=float(np.max(np.max(v, axis=0) - np.min(v, axis=0))),
                max_from_main=float(np.max(abs(v - v[1]))), finite=True)


def _ratio(numerator, denominator):
    if numerator is None or denominator is None or not math.isfinite(numerator) or not math.isfinite(denominator):
        return dict(value=None, status="undefined_nonfinite_or_missing")
    if denominator == 0: return dict(value=None, status="undefined_zero_denominator")
    value = float(numerator / denominator)
    return dict(value=value, status="defined") if math.isfinite(value) else dict(value=None, status="undefined_ratio_overflow")


def _record(score, P, targets):
    if not np.isfinite(score).all():
        return dict(status="undefined_nonfinite_score", moments=None, standardized=None, planning=None)
    moments = statistics.weighted_moments(score, P)
    standardized = {"signal_to_SD": _ratio(moments["mean"], moments["std"]),
        "excess_signal_to_SD": {name: _ratio(moments["mean"] - margin, moments["std"])
                                for name, margin in targets.items()}}
    return dict(status="defined", moments=moments, standardized=standardized,
        planning=statistics.planning_table(moments["mean"], moments["variance"], moments["range"],
                                          targets, Ngrid=N_GRID, delta=DELTA))


def score_records(scores, teachers, targets):
    records = [[_record(score, P, targets) for P in teachers] for score in scores]
    return dict(targets=targets, variant_axes=["conditional_or_candidate_rule", "teacher_rule"],
                main_index=[1, 1], variants=records, main=records[1][1])


def _minimum(record, margin, method):
    if record["status"] != "defined": return dict(n=None, status="undefined_nonfinite_score")
    m = record["moments"]
    return statistics.minimum_n(m["mean"], margin, m["variance"], m["range"], method, DELTA,
                                max_n=2**30)


def bridge_readout(Xc, Xt, P, curvature_record, finite_record, t):
    """Small-step comparison; no claim about a remainder beyond saved t values."""
    if not np.isfinite(Xt).all(): return dict(status="undefined_nonfinite_score", t=t)
    c = statistics.weighted_moments(Xc, P); z = statistics.weighted_moments(Xt, P)
    pb = c["pbar"]
    covariance = math.fsum(float(w * (x - c["mean"]) * (y - z["mean"])) for w, x, y in zip(pb, Xc, Xt))
    corr = _ratio(covariance, c["std"] * z["std"])
    snrc, snrz = _ratio(c["mean"], c["std"]), _ratio(z["mean"], z["std"])
    ratios = {}
    for method in ("hoeffding", "eb_proxy"):
        cn = _minimum(curvature_record, 1e-4, method)
        zn = _minimum(finite_record, t*t*1e-4/2, method)
        ratios[method] = dict(curvature=cn, finite_gain=zn,
                             finite_over_curvature=_ratio(zn["n"], cn["n"]))
    return dict(status="defined", t=t, maxabs_label_residual=float(np.max(abs(Xt - Xc))),
        weighted_RMS_residual=math.sqrt(math.fsum(float(w * v*v) for w, v in zip(pb, Xt-Xc))),
        range_ratio_Xt_over_Xc=_ratio(z["range"], c["range"]),
        SD_ratio_Xt_over_Xc=_ratio(z["std"], c["std"]),
        range_to_SD_curvature=_ratio(c["range"], c["std"]),
        range_to_SD_finite_normalized=_ratio(z["range"], z["std"]),
        correlation=corr, curvature_SNR=snrc, finite_normalized_SNR=snrz,
        SNR_ratio_Xt_over_Xc=_ratio(snrz["value"], snrc["value"]),
        matched_margin_required_n=ratios, no_Taylor_remainder_certificate=True)


def transfer_readout(curvature_scores, finite_scores, P, curvature_record, finite_record, t):
    """Reuse one curvature event through an enumerated finite score inequality.

    The inequality is exact for the supplied real score tables. Float residuals
    are reported, and neither these residuals nor inter-rule agreement certify
    that these tables equal exact Gaussian likelihood scores.
    """
    if not np.isfinite(finite_scores).all():
        return dict(status="undefined_nonfinite_score", t=t)
    scaled = 2*finite_scores/(t*t)
    residual = curvature_scores[:, None, :] - scaled[None, :, :]
    e_minus = np.max(residual, axis=2)  # retain negative maxima
    slack = scaled[None, :, :] - curvature_scores[:, None, :] + e_minus[:, :, None]
    finite_scale = .5*t*t
    scaled_slack = finite_scores[None, :, :] - finite_scale*(curvature_scores[:, None, :]-e_minus[:, :, None])
    pbar = np.asarray(P)/math.fsum(map(float, P))
    curv_mean = math.fsum(float(w*x) for w,x in zip(pbar,curvature_scores[1]))
    finite_mean = math.fsum(float(w*x) for w,x in zip(pbar,finite_scores[1]))
    threshold = dict(zero=0., matched_curvature=t*t*1e-4/2, gain_1e_5=1e-5, gain_1e_4=1e-4)
    targets = {key:float(e_minus[1,1]+2*eta/(t*t)) for key,eta in threshold.items()}
    m = curvature_record["moments"]
    comparisons = {}
    for key,eta in threshold.items():
        methods = {}
        for method in ("hoeffding","eb_proxy"):
            transferred = _minimum(curvature_record,targets[key],method)
            direct = _minimum(finite_record,eta,method)
            methods[method] = dict(transferred=transferred,direct=direct,
                transferred_over_direct_n=_ratio(transferred["n"],direct["n"]))
        comparisons[key] = dict(eta=eta,curvature_threshold=targets[key],
            oracle_curvature_gap=m["mean"]-targets[key],
            transfer_gives_no_support_even_with_unlimited_data=bool(m["mean"]<=targets[key]),
            no_support_does_not_mean_candidate_has_no_gain=True,methods=methods)
    raw_mass=math.fsum(map(float,P))
    raw_curv=math.fsum(float(w*x) for w,x in zip(P,curvature_scores[1]))
    raw_gain=math.fsum(float(w*x) for w,x in zip(P,finite_scores[1]))
    lower=finite_scale*(curv_mean-float(e_minus[1,1]))
    return dict(status="defined",t=t,e_minus=e_minus,axes=["curvature_rule","candidate_rule"],
        main_index=[1,1],main_e_minus=float(e_minus[1,1]),negative_e_minus_preserved=True,
        minimum_pointwise_slack=np.min(slack,axis=2),
        minimum_scaled_pointwise_slack=np.min(scaled_slack,axis=2),
        scaling_identity_maxabs_residual=float(np.max(abs(scaled_slack-finite_scale*slack))),
        normalized_mean_gain=finite_mean,normalized_mean_gain_lower_from_curvature=lower,
        normalized_mean_slack=finite_mean-lower,
        raw_mean_gain=raw_gain,raw_mass=raw_mass,
        raw_mean_gain_lower_from_curvature=finite_scale*(raw_curv-raw_mass*float(e_minus[1,1])),
        thresholds=comparisons,
        curvature_event_planning=statistics.planning_table(m["mean"],m["variance"],m["range"],targets,Ngrid=N_GRID,delta=DELTA),
        same_curvature_event_reused_for_all_candidates_and_margins=True,new_score_families=0,
        exact_real_score_table_inequality=True,strict_Gaussian_likelihood_certificate=False,
        strict_float_certificate=False,Taylor_remainder_certified=False)


def derive_unit(source, candidates):
    """Pure finite-alphabet derivation; no I/O, quadrature, fitting or RNG."""
    q, m = explanation.pmf_array(source["q"], "saved empirical frequency")
    theta = _real_vector(source["theta_signed"], "saved theta_signed")
    d = _real_vector(source["fixed_d"], "saved d")
    signs = np.asarray(source["signs"])
    require(theta.shape == (2*m,) and d.shape == (m,), "source parameter dimensions")
    permutation = recoded_indices(signs)
    require(len(signs) == m, "source signs dimension")
    P = np.asarray(source["teacher_pmfs"], dtype=float)
    require(P.shape == (4, len(q)), "four original teacher variants")
    for row in P: explanation.pmf_array(row, "raw teacher")
    H = np.asarray(source["label_Hangle"], dtype=float)
    require(H.shape == (2, len(q), m, m) and np.isfinite(H).all(), "two saved label Hessian rules")
    b, a, d_copy, W = path_models(theta[:m], theta[m:], d, candidates["t_values"])
    for name, expect in (("b", b), ("signed_a", a), ("fixed_d", d_copy), ("model_W", W)):
        require(np.array_equal(candidates[name], expect), "candidate path changed: " + name)
    require(np.asarray(candidates["fixed_d"]).tobytes() == np.asarray(source["fixed_d"]).tobytes(), "original direction bytes")
    Q = np.asarray(candidates["pmfs"], dtype=float)
    t_values = candidates["t_values"]
    require(Q.shape == (1 + len(t_values), 4, len(q)), "candidate PMF axes")
    recoded_s = np.einsum("i,ryij,j->ry", d, H, d)
    s = recoded_s[:, permutation].copy()
    curv_raw = s @ P.T
    emp_raw = s @ q
    old_curv = np.asarray(source["old_population_curvature"])
    old_emp = np.asarray(source["old_empirical_curvature"])
    require(old_curv.shape == (2, 4) and old_emp.shape == (2,), "old fixed direction curvature shape")
    source_errors = dict(population=float(np.max(abs(curv_raw-old_curv))),
                         empirical=float(np.max(abs(emp_raw-old_emp))))
    require(max(source_errors.values()) <= GATES["source_curvature_identity"], "old curvature mean mismatch")
    with np.errstate(divide="ignore", invalid="ignore"):
        logQ = np.log(Q)
        z = logQ[1:] - logQ[0][None, :, :]
    Xnorm = 2 * z / np.square(np.asarray(t_values))[:, None, None]
    curvature_records = score_records(-s, P, {"zero": 0., "useful": 1e-4})
    finite_records = []
    bridges = []
    transfers = []
    qualifications = []
    masses = Q.sum(axis=-1)
    teacher_masses = np.sum(P, axis=1)
    teacher_PMF_difference = _difference_summary(P)
    pmf_discrepancies = [_difference_summary(row) for row in Q]
    curvature_discrepancy = float(np.max(abs(s[0]-s[1])))
    base_mass_ok = bool(np.isfinite(masses[0]).all() and np.max(abs(masses[0]-1)) <= GATES["PMF_mass"])
    teacher_mass_ok = bool(np.max(abs(teacher_masses-1)) <= GATES["PMF_mass"])
    teacher_PMF_ok = teacher_PMF_difference["max_pairwise"] <= GATES["PMF_maxdiff"]
    base_pmf_ok = pmf_discrepancies[0]["finite"] and pmf_discrepancies[0]["max_pairwise"] <= GATES["PMF_maxdiff"]
    curvature_qualified = teacher_mass_ok and teacher_PMF_ok and curvature_discrepancy <= GATES["label_curvature_maxdiff"]
    for ti, t in enumerate(t_values):
        targets = dict(zero=0., matched_curvature=t*t*1e-4/2, gain_1e_5=1e-5, gain_1e_4=1e-4)
        records = score_records(z[ti], P, targets)
        finite_records.append(records)
        z_discrepancy = _difference_summary(z[ti])
        paired_means = z[ti] @ P.T
        all_finite = bool(np.isfinite(paired_means).all())
        mean_diff = float(np.max(paired_means)-np.min(paired_means)) if all_finite else None
        gates = dict(raw_teacher_mass=teacher_mass_ok, raw_teacher_PMF=teacher_PMF_ok, base_mass=base_mass_ok, base_PMF=base_pmf_ok,
            perturbed_mass=bool(np.isfinite(masses[ti+1]).all() and np.max(abs(masses[ti+1]-1)) <= GATES["PMF_mass"]),
            perturbed_PMF=pmf_discrepancies[ti+1]["finite"] and pmf_discrepancies[ti+1]["max_pairwise"] <= GATES["PMF_maxdiff"],
            z=z_discrepancy["finite"] and z_discrepancy["max_pairwise"] <= GATES["z_maxdiff"],
            paired_mean=all_finite and mean_diff <= GATES["paired_mean_maxdiff"],
            strictly_positive_candidate_PMFS=bool(np.isfinite(Q[[0, ti+1]]).all() and np.all(Q[[0, ti+1]] > 0)))
        qualifications.append(dict(t=t, qualified=all(gates.values()), gates=gates,
            z_rule_difference=z_discrepancy, paired_raw_means=paired_means,
            paired_mean_max_pairwise=mean_diff,
            paired_mean_max_from_main=float(np.max(abs(paired_means-paired_means[1, 1]))) if all_finite else None,
            candidate_only_mean_maxdiff=float(np.max(np.ptp(paired_means, axis=0))) if all_finite else None,
            teacher_only_mean_maxdiff=float(np.max(np.ptp(paired_means, axis=1))) if all_finite else None,
            rigorous_error_bound=False))
        bridges.append(bridge_readout(-s[1], Xnorm[ti, 1], P[1], curvature_records["main"], records["main"], t))
        transfers.append(transfer_readout(-s,z[ti],P[1],curvature_records["main"],records["main"],t))
    return snapshot(dict(schema="direction_validation_unit_v1", source=source,
        original_direction_bytes_preserved=True, recoded_index_of_original_label=permutation,
        label_curvature_recoded=recoded_s, label_curvature_original=s, Xc=-s[1], z=z, Xt_norm=Xnorm,
        Xt_norm_minus_Xc=Xnorm+s[1][None,None,:],
        teacher_raw_pmfs=P, teacher_masses=teacher_masses,
        teacher_PMF_difference=teacher_PMF_difference,
        teacher_pbar=np.array([row / math.fsum(map(float, row)) for row in P]),
        teacher_normalization_maxabs=np.array([np.max(abs(row/math.fsum(map(float,row))-row)) for row in P]),
        raw_population_curvature=curv_raw, raw_empirical_curvature=emp_raw,
        source_mean_identity_errors=source_errors, candidate_raw_masses=masses,
        candidate_PMF_differences=pmf_discrepancies,
        curvature_qualification=dict(qualified=curvature_qualified, label_C_D_maxdiff=curvature_discrepancy,
            teacher_mass_gate=teacher_mass_ok, teacher_PMF_gate=teacher_PMF_ok, rigorous_error_bound=False),
        curvature=curvature_records, finite_gain=finite_records, finite_qualification=qualifications,
        bridge=bridges, transfer=transfers, all_numerically_qualified=curvature_qualified and all(x["qualified"] for x in qualifications),
        old_alarm_policy_unchanged=True, new_independent_validation=False, teacher_use="oracle feasibility only",
        normalization_policy="raw source preserved; moments explicitly use P / sum(P)",
        source_coordinate="original labels and signed amplitudes; saved d unchanged"))


def scalar_record(record):
    value = snapshot(record)
    if value["moments"] is not None: value["moments"].pop("pbar", None)
    return value


def scalar_row(unit):
    s = unit["source"]
    return public(dict(unit=s["unit"]["id"], D=s["unit"]["teacher"]+1, N=s["unit"]["N"], rep=s["unit"]["rep"],
        A=s["A"], B=s["B"], full_level=s["full_level"], saved_delta_R=s["saved_delta_R"],
        old_population_curvature=float(s["old_population_curvature"][1, 1]),
        old_empirical_curvature=float(s["old_empirical_curvature"][1]),
        curvature=scalar_record(unit["curvature"]["main"]),
        finite_gain=[scalar_record(row["main"]) for row in unit["finite_gain"]],
        bridge=unit["bridge"], transfer=unit["transfer"], curvature_qualification=unit["curvature_qualification"],
        finite_qualification=unit["finite_qualification"], all_numerically_qualified=unit["all_numerically_qualified"],
        teacher_masses=unit["teacher_masses"], source_mean_identity_errors=unit["source_mean_identity_errors"]))


def oracle_retrospective(rows):
    def group(rs):
        if not rs: return dict(units=0)
        policies = {}
        for name in ("A", "B", "A_star", "B_star"):
            masks = [bool(r[name[0]]) and (not name.endswith("star") or r["old_population_curvature"] < 0) for r in rs]
            policies[name] = dict(selected=sum(masks),
                mean_risk_difference_from_always_HIGH=math.fsum((1-a)*r["saved_delta_R"] for a, r in zip(masks, rs))/len(rs),
                captured_gain_above_0p001=sum(a and r["saved_delta_R"] > .001 for a,r in zip(masks,rs)),
                total_gain_above_0p001=sum(r["saved_delta_R"] > .001 for r in rs),
                selected_HIGH_disadvantages=sum(a and r["saved_delta_R"] < 0 for a,r in zip(masks,rs)))
        gaps = [r["old_population_curvature"]-r["old_empirical_curvature"] for r in rs]
        return dict(units=len(rs), policies=policies, hP_minus_hemp_mean=math.fsum(gaps)/len(gaps),
            hP_minus_hemp_min=min(gaps), hP_minus_hemp_max=max(gaps), hP_above_hemp_count=sum(x>0 for x in gaps))
    return dict(all_units=group(rows), by_N={str(n):group([r for r in rows if r["N"]==n]) for n in sorted({r["N"] for r in rows})},
        raw_main_population_curvature_used=True, retrospective_oracle_only=True,
        new_policy_executed=False, not_an_upper_bound_over_all_validation_policies=True,
        bias_causal_components_identified=False)


def required_freeze_sources():
    return [Path(__file__), Path(__file__).with_name("direction_validation_compute_test.py"),
        Path(statistics.__file__), Path(statistics.__file__).with_name("direction_validation_statistics_test.py"),
        Path(joint.__file__), Path(legacy.__file__), Path(explanation.__file__), DOC/"CONTRACT.md", OUT/"HANDOFF_INPUT.md",
        OLD/"DELIVERY_MANIFEST.json", OLD/"CONFIG.json", OLD/"SUMMARY.json", OLD/"FREEZE.json",
        EXPLANATION/"DELIVERY_MANIFEST.json", EXPLANATION/"RESULTS.json", EXPLANATION/"SUMMARY.json",
        *[OLD/(name+".json") for name in explanation.LOCKS]]


def verify_freeze():
    freeze = read(OUT/"FREEZE.json")
    require(freeze["schema"] == "direction_validation_freeze_v1", "freeze schema")
    require(freeze["baseline"] == BASE and freeze["handoff_sha256"] == HANDOFF_SHA, "freeze provenance")
    require(freeze["production_not_started"] is True and freeze["config"] == config_dump(), "configuration freeze")
    explanation._bind_sources(freeze["sources"])
    require(all(rel(p) in freeze["sources"] for p in required_freeze_sources()), "missing required frozen source")
    require(sha(OUT/"HANDOFF_INPUT.md") == HANDOFF_SHA, "archived HANDOFF hash")
    require(sha(OLD/"DELIVERY_MANIFEST.json") == DEV_MANIFEST_SHA, "old dev seal")
    require(sha(EXPLANATION/"DELIVERY_MANIFEST.json") == EXPLANATION_MANIFEST_SHA, "old explanation seal")
    old_config = read(OLD/"CONFIG.json")
    require(old_config["units"] == config_dump()["original_units"], "unchanged twelve original units")
    locked = {}
    for name in explanation.LOCKS:
        lock = read(OLD/(name+".json"))
        require(lock["freeze_sha256"] == sha(OLD/"FREEZE.json"), "old global lock freeze")
        for path, item in lock["files"].items():
            require((ROOT/path).resolve().is_relative_to(OLD) and path == item["path"], "old input boundary")
            if path in locked: require(item == locked[path], "conflicting old lock")
            locked[path] = item
    # Input hashing does not decompress or numerically replay old priors/tapes.
    explanation._bind_sources(locked)
    exp_result = read(EXPLANATION/"RESULTS.json")
    explanation._bind_sources(exp_result["artifacts"])
    if "inputs" in freeze: explanation._bind_sources(freeze["inputs"])
    hashes = {p:v["sha256"] for p,v in locked.items()}
    hashes.update({p:(v if isinstance(v,str) else v["sha256"]) for p,v in freeze["sources"].items()})
    hashes.update({p:v["sha256"] for p,v in exp_result["artifacts"].items()})
    if "inputs" in freeze:
        hashes.update({p:(v if isinstance(v,str) else v["sha256"]) for p,v in freeze["inputs"].items()})
    return freeze, locked, hashes


def load_sources(locked):
    reader = explanation.SavedReader(locked)
    teachers = [reader.selected(OLD/f"teacher_D{i}", ["pmfs", "rules"]) for i in (1,2,3)]
    old_rows = {r["unit"]:r for r in read(OLD/"SUMMARY.json")["rows"]}
    sources = []; explanation_scope = {}
    for unit in config_dump()["original_units"]:
        folder = OLD/"units"/unit["id"]
        anchor = reader.selected(folder/"anchor", ["theta_signed", "theta_recoded", "signs", "q_original"])
        diag = reader.selected(folder/"diagnosis", ["numerics.direction", "full_raise", "ref_raise", "full_level",
            "reference_only.raw.C.label_Hangle", "reference_only.raw.D.label_Hangle",
            "reference_only.raw.C.patterns", "reference_only.raw.D.patterns"])
        explanation_stem = EXPLANATION/"units"/unit["id"]/"explanation"
        old = legacy.restore(explanation_stem)
        def array_keys(value):
            if isinstance(value,dict):
                if "__ndarray__" in value:return [value["__ndarray__"]]
                return [k for child in value.values() for k in array_keys(child)]
            if isinstance(value,list):return [k for child in value for k in array_keys(child)]
            return []
        explanation_scope[rel(explanation_stem.with_suffix(".npz"))] = sorted(set(array_keys(read(explanation_stem.with_suffix(".json"))["tree"])))
        teacher = teachers[unit["teacher"]]
        require(teacher["rules"] == [list(r) for r in PMF_RULES], "saved teacher rule ordering")
        d = diag["numerics"]["direction"]
        require(d is not None and np.isfinite(d).all(), "old direction undefined; do not replace")
        require(d.tobytes() == old["fixed_empirical_direction"].tobytes(), "direction archive bytes")
        for key in ("theta_signed", "theta_recoded", "signs"):
            require(np.array_equal(anchor[key], old[key]), "original anchor unchanged: "+key)
        require(np.array_equal(anchor["q_original"], old["empirical_frequency"]), "original empirical frequency")
        for rule in ("C", "D"):
            require(np.array_equal(diag["reference_only"]["raw"][rule]["patterns"], joint.patterns(len(d))), "original label order")
        A, B = bool(diag["full_raise"]), bool(diag["ref_raise"])
        require(A == old["original_A"] == old_rows[unit["id"]]["A"] and B == old["original_B"] == old_rows[unit["id"]]["B"], "old alarms")
        gain = float(old["saved_paired_gain"][1])
        require(gain == old_rows[unit["id"]]["delta_R"], "old endpoint gain")
        sources.append(snapshot(dict(unit=unit, theta_signed=anchor["theta_signed"], theta_recoded=anchor["theta_recoded"],
            signs=anchor["signs"], q=anchor["q_original"], teacher_pmfs=teacher["pmfs"], teacher_rules=teacher["rules"],
            fixed_d=d, label_Hangle=np.stack([diag["reference_only"]["raw"][r]["label_Hangle"][0] for r in ("C","D")]),
            old_population_curvature=old["population_direction_curvature"], old_empirical_curvature=old["empirical_fixed_direction_curvature"],
            A=A, B=B, full_level=diag["full_level"], saved_delta_R=gain,
            original_numeric_risk_qualification=old["original_risk_qualified"])))
    scope = {p:sorted(v) for p,v in reader.array_reads.items()}
    scope.update(explanation_scope)
    return sources, scope


def summarize(units, candidates):
    rows = [scalar_row(u) for u in units]
    resource = dict(units=len(units), candidate_models=4*len(units), score_functions=4*len(units),
        PMF_calls=sum(c["resources"]["PMF_calls"] for c in candidates),
        quadrature_nodes=sum(c["resources"]["quadrature_nodes"] for c in candidates),
        node_pattern_products=sum(c["resources"]["node_pattern_products"] for c in candidates),
        PMF_seconds=math.fsum(c["resources"]["seconds"] for c in candidates),
        new_teacher_PMF_calls=0, new_curvature_integrals=0, new_observations=0, new_random_draws=0, new_training=0)
    return dict(schema="direction_validation_summary_v1", rows=rows, oracle_retrospective=oracle_retrospective(rows),
        resources=resource, numerical_qualified_units=sum(r["all_numerically_qualified"] for r in rows),
        numerical_qualified_curvature_scores=sum(r["curvature_qualification"]["qualified"] for r in rows),
        numerical_qualified_finite_scores=sum(v["qualified"] for r in rows for v in r["finite_qualification"]),
        all_original_units_retained=True, independent_validation_executed=False,
        oracle_planning_is_not_confidence_or_power=True, automatic_next_stage=False)


def readable(summary):
    lines = ["# Fixed-direction oracle feasibility", "", "All original objects are retained. No validation sample was drawn. Planning uses known-teacher moments; it is not a confidence interval, power calculation, or necessary lower bound.", "",
        "|unit|A/B|old population curvature|Xc mean|Xc SD|range|numeric qualified|", "|---|---|---:|---:|---:|---:|---|"]
    def fmt(x): return "undefined" if x is None else (f"{x:.10g}" if isinstance(x,float) else str(x))
    for row in summary["rows"]:
        m = row["curvature"]["moments"]
        lines.append("|"+"|".join(map(fmt,[row["unit"],f"{int(row['A'])}/{int(row['B'])}",row["old_population_curvature"],m["mean"],m["std"],m["range"],row["curvature_qualification"]["qualified"]]))+"|")
    for ti,t in enumerate(T_VALUES):
        lines += ["", f"## Fixed finite amplitude t={t}", "", "|unit|raw mean z|normalized mean z|SD(z)|range(z)|numeric qualified|label bridge max residual|EB oracle n, zero / matched / 1e-5 / 1e-4|", "|---|---:|---:|---:|---:|---|---:|---|"]
        for row in summary["rows"]:
            r = row["finite_gain"][ti]; m = r["moments"]
            if m is None: values=[row["unit"],None,None,None,None,False,None,"undefined"]
            else:
                margins=[0.,t*t*1e-4/2,1e-5,1e-4]
                ns=[_minimum(r,margin,"eb_proxy")["n"] for margin in margins]
                values=[row["unit"],m["raw_mean"],m["mean"],m["std"],m["range"],row["finite_qualification"][ti]["qualified"],row["bridge"][ti].get("maxabs_label_residual")," / ".join("none <= 2^30" if n is None else str(n) for n in ns)]
            lines.append("|"+"|".join(map(fmt,values))+"|")
    lines += ["", "Numeric failure does not remove a row or trigger a higher order. Raw means use the saved teacher mass; statistical moments explicitly use its normalized weights. Complete six-look planning, all rule-by-teacher sensitivities, and null reasons are in SUMMARY.json and each unit analysis.", ""]
    return "\n".join(lines)


def main(mode):
    tick = time.perf_counter()
    if mode == "run":
        require(not (OUT/"RESULTS.json").exists() and not (OUT/"units").exists() and not (OUT/"FAILURE.json").exists(), "no overwrite or implicit rerun")
    else: require((OUT/"RESULTS.json").is_file(), "saved result required")
    frozen, locked, input_hashes = verify_freeze()
    sources, scope = load_sources(locked)
    if mode == "check":
        saved_result = read(OUT/"RESULTS.json")
        require(saved_result["source_sha256"] == sha(__file__) and saved_result["freeze_sha256"] == sha(OUT/"FREEZE.json"), "producer source/freeze")
        same_tree(input_hashes, saved_result["input_hashes"])
        same_tree(scope, saved_result["source_array_read_scope"])
        explanation._bind_sources(saved_result["artifacts"])
    unit_results=[]; candidates=[]; artifacts=[]; unit_artifacts=[]
    for source in sources:
        folder=OUT/"units"/source["unit"]["id"]
        candidate_stem, analysis_stem = folder/"candidate_pmfs", folder/"analysis"
        if mode == "run":
            theta=source["theta_signed"]; m=len(source["fixed_d"])
            cand=generate_candidate_pmfs(theta[:m],theta[m:],source["fixed_d"])
            legacy.persist(candidate_stem,cand)  # preserve PMFs even if later algebra fails
        else: cand=legacy.restore(candidate_stem)
        require(cand["rules"] == [list(r) for r in PMF_RULES] and cand["t_values"] == list(T_VALUES), "fixed candidate rules/amplitudes")
        require(cand["resources"]["PMF_calls"] == 16 and len(cand["calls"]) == 16, "sixteen candidate calls per unit")
        result=derive_unit(source,cand)
        if mode == "run": legacy.persist(analysis_stem,result)
        else: same_tree(result,legacy.restore(analysis_stem),source["unit"]["id"])
        paths=[s.with_suffix(ext) for s in (candidate_stem,analysis_stem) for ext in (".json",".npz")]
        artifacts.extend(paths)
        unit_artifacts.append(dict(unit=source["unit"]["id"],candidate_json=binding(paths[0]),candidate_npz=binding(paths[1]),analysis_json=binding(paths[2]),analysis_npz=binding(paths[3])))
        unit_results.append(result); candidates.append(cand)
        if mode == "run": print(json.dumps(dict(unit=source["unit"]["id"],PMF_calls=16,numerically_qualified=result["all_numerically_qualified"])),flush=True)
    summary=summarize(unit_results,candidates)
    require(summary["resources"]["PMF_calls"] == 192, "fixed total PMF call count")
    if mode == "check":
        same_tree(public(summary),read(OUT/"SUMMARY.json")["derived"])
        same_tree(unit_artifacts,saved_result["unit_artifacts"])
        require((OUT/"READABLE_UNIT_TABLE.md").read_text(encoding="utf-8") == readable(summary), "readable table derivation")
        print(json.dumps(dict(passed=True,mode=mode,units=12,new_quadrature_calls=0,new_random_draws=0,
            saved_PMF_and_scalar_derivation_repeated=True,numerically_qualified_units=summary["numerical_qualified_units"],seconds=time.perf_counter()-tick)))
        return
    save(OUT/"SUMMARY.json",dict(created_utc=now(),source_sha256=sha(__file__),freeze_sha256=sha(OUT/"FREEZE.json"),input_hashes=input_hashes,derived=summary))
    with (OUT/"READABLE_UNIT_TABLE.md").open("x",encoding="utf-8",newline="\n") as f: f.write(readable(summary))
    artifacts.extend([OUT/"SUMMARY.json",OUT/"READABLE_UNIT_TABLE.md"])
    save(OUT/"RESULTS.json",dict(schema="direction_validation_results_v1",status="completed_fixed_object_feasibility",
        created_utc=now(),baseline=BASE,handoff_sha256=HANDOFF_SHA,source_sha256=sha(__file__),freeze_sha256=sha(OUT/"FREEZE.json"),
        input_hashes=input_hashes,source_array_read_scope=scope,unit_artifacts=unit_artifacts,
        artifacts={rel(p):binding(p) for p in artifacts},resources=summary["resources"],seconds=time.perf_counter()-tick,
        no_new_validation_data=True,original_A_B_d_unchanged=True,numerical_failure_is_retained_not_engineering_failure=True))
    print(json.dumps(dict(passed=True,mode=mode,units=12,PMF_calls=192,numerically_qualified_units=summary["numerical_qualified_units"])))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    for name in ("config","self-test","run","check"): group.add_argument("--"+name,action="store_true")
    args=parser.parse_args()
    if args.config: print(json.dumps(config_dump(),indent=2))
    elif args.self_test:
        import direction_validation_compute_test as tests
        raise SystemExit(tests.run_tests())
    else:
        try: main("run" if args.run else "check")
        except BaseException as exc:
            if args.run and (OUT/"FREEZE.json").is_file() and not (OUT/"FAILURE.json").exists():
                save(OUT/"FAILURE.json",dict(created_utc=now(),status="failed_no_automatic_retry",reason=str(exc),
                    traceback=traceback.format_exc(),source_sha256=sha(__file__),saved_PMFS_retained=True,
                    new_training=0,new_random_draws=0))
            raise
