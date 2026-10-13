"""Bounded R03 cross-checks, with no target execution on import.

The root caller must freeze sources before produce_audit(). This module reads
saved JSON evidence but never writes files. Only --self-test is a CLI action.
The GH checks and saved GL comparisons are numerical audits, not a second
strict certificate. The point interval calculation reuses the frozen new
candidate evaluator; independent saved GL values are not recomputed.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import scipy
from scipy.special import expit, roots_hermitenorm

import r03_candidate_interval as candidate


ROOT = Path(__file__).resolve().parents[1]
OUT = "artifacts/r03_teacher_family_v1_20261011/"
SOURCE_FILES = [OUT + "BOX_INPUTS.json", OUT + "DEVELOPMENT.json",
                OUT + "TEACHER_CERTIFICATE.json",
                "artifacts/certified_teacher_v1_20261008/TEACHER_CERTIFICATE.json"]
POINT_CONFIG = dict(K=8, precision_bits=192, hz="1/32", L="10", dz="1/6",
                    hv="1/4", vmin="-28", vmax="56", dv="1", T="4",
                    saturation_threshold=128, label_mode="complement")
LABELS = (0, 85, 127)
ORDERS = (80, 100)
ANGLES = ("-7/10", "-1/2", "-3/10", "-1/10", "1/10", "3/10", "1/2", "7/10")


def prepared_inputs():
    """Fixed audit specification only; no file reads or target evaluation."""
    return dict(schema="r03_independent_audit_inputs_v1", source_files=list(SOURCE_FILES),
                labels=list(LABELS), teacher_GH_orders=list(ORDERS),
                teacher_L1_gate_exact="1e-11", candidate_entry_gate_exact="1e-8",
                candidate_config=dict(POINT_CONFIG), lambda_centre_exact="0",
                old_hessian_rep00_comparison=False)


def _endpoint(value):
    if type(value["exponent_of_two"]) is not int:
        raise ValueError("integer dyadic exponent required")
    return Fraction(int(value["mantissa"])) * Fraction(2) ** value["exponent_of_two"]


def _interval(value):
    lo, hi = _endpoint(value["lower_exact_dyadic"]), _endpoint(value["upper_exact_dyadic"])
    if lo > hi:
        raise ValueError("reversed saved interval")
    return lo, hi


def _encode_endpoint(q):
    q = Fraction(q)
    denominator = q.denominator
    if denominator & (denominator - 1):
        raise ValueError("dyadic endpoint required")
    return dict(mantissa=str(q.numerator), exponent_of_two=-(denominator.bit_length() - 1))


def _encode_interval(lo, hi):
    if lo > hi:
        raise ValueError("reversed endpoints")
    return dict(arb_display_only=f"[{float(lo):.17g}, {float(hi):.17g}]",
                lower_exact_dyadic=_encode_endpoint(lo), upper_exact_dyadic=_encode_endpoint(hi))


def _shape(value):
    if not isinstance(value, list):
        return ()
    if not value:
        return (0,)
    child = _shape(value[0])
    if any(_shape(item) != child for item in value):
        raise ValueError("ragged saved array")
    return (len(value),) + child


def _require_shape(value, shape, name):
    if _shape(value) != tuple(shape):
        raise ValueError(f"{name}: expected shape {shape}, got {_shape(value)}")


def _flatten(value):
    if isinstance(value, list):
        return [leaf for item in value for leaf in _flatten(item)]
    return [value]


def _expanded(record, radius):
    lo, hi = _interval(record)
    return _encode_interval(lo - radius, hi + radius)


def _compare(values, intervals, gate, *, use_L1_gate):
    """Compare stored binary64 numbers with dyadic intervals exactly.

    gap is distance to the set. worst_abs bounds distance to every member of
    the interval and is used for the declared gate, making a wide interval
    unable to pass merely by containing an inaccurate ordinary value.
    """
    if _shape(values) != _shape(intervals):
        raise ValueError("comparison shape mismatch")
    raw, boxes = _flatten(values), _flatten(intervals)
    gaps, worst = [], []
    for value, box in zip(raw, boxes):
        if isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError("finite numerical audit value required")
        x = Fraction.from_float(float(value))
        lo, hi = _interval(box)
        gaps.append(max(Fraction(0), lo - x, x - hi))
        worst.append(max(abs(x - lo), abs(x - hi)))
    maximum, total = max(worst, default=Fraction(0)), sum(worst, Fraction(0))
    threshold = Fraction(gate)
    return dict(shape=list(_shape(values)), component_count=len(raw),
                distance_to_interval_exact=[str(x) for x in gaps],
                worst_absolute_difference_exact=[str(x) for x in worst],
                max_worst_absolute_difference_exact=str(maximum),
                max_worst_absolute_difference_display=float(maximum),
                L1_worst_absolute_difference_exact=str(total),
                L1_worst_absolute_difference_display=float(total),
                gate_exact=str(threshold), gate_uses="L1 worst absolute" if use_L1_gate else "maximum entry worst absolute",
                passed=(total if use_L1_gate else maximum) < threshold)


def _read_sources():
    documents, sources = [], []
    for name in SOURCE_FILES:
        raw = (ROOT / name).read_bytes()
        documents.append(json.loads(raw))
        sources.append(dict(path=name, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)))
    return documents, sources


def _validate_saved(box, development, teacher, old_teacher):
    if box.get("schema") != "r03_box_inputs_v1" or box.get("candidate_config") != POINT_CONFIG:
        raise ValueError("frozen coarse candidate configuration changed")
    if development.get("schema") != "r03_development_v1":
        raise ValueError("unexpected development schema")
    if teacher.get("schema") != "r03_teacher_derivative_certificate_v1":
        raise ValueError("unexpected new teacher certificate schema")
    spec = teacher["family"]
    if (spec["lambda_centre_exact"] != "0" or spec["effective_angles_exact"] != list(ANGLES)
            or Fraction(spec["config"]["amplitude"]) != Fraction(4, 5)
            or spec["bias_exact"] != ["0"] * 8 or teacher.get("renormalized") is not False):
        raise ValueError("new teacher does not bind the original exact zero-bias family centre")
    if old_teacher.get("schema") != "certified_teacher_true_PMF_v1":
        raise ValueError("unexpected old teacher schema")
    for key in ("q_true_component_intervals", "prime_true_component_intervals"):
        _require_shape(teacher[key], (256,), key)
        for value in teacher[key]:
            _interval(value)
    _require_shape(old_teacher["true_PMF_enclosures"], (256,), "old teacher PMF")
    point = development["points"][0]
    if point["index"] != 0 or point["lambda_exact"] != "0":
        raise ValueError("development point0 identity drift")
    saved = point["evaluations"]["D"]["result"]
    if (saved["K_values"] != [8, "inf"] or saved["config"]["coordinate_order"] != "b_then_a"
            or saved["config"]["normal_rule"] != "legendre"
            or saved["config"]["normal_order"] != 384
            or saved["config"]["normal_limit"] != 11.0
            or saved["config"]["logu_min"] != -32.0
            or saved["config"]["logu_max"] != 64.0
            or saved["config"]["logu_step"] != 0.125):
        raise ValueError("saved D numerical rule drift")
    _require_shape(saved["patterns"], (256, 8), "saved patterns")
    for label, bits in enumerate(saved["patterns"]):
        if bits != [(label >> (7 - i)) & 1 for i in range(8)]:
            raise ValueError("saved label order mismatch")
    for key, shape in (("label_radial_gradient", (2, 256, 8)),
                       ("label_bias_gradient", (2, 256, 8)),
                       ("label_H_ba", (2, 256, 16, 16)),
                       ("label_Hperp", (2, 256, 8, 8))):
        _require_shape(saved[key], shape, key)
    a0 = box["boxes"]["h01"]["a0_exact"]
    if any(other["a0_exact"] != a0 for other in box["boxes"].values()):
        raise ValueError("boxes disagree on exact original centre")
    old_a0 = [Fraction(int(x["numerator"]), int(x["denominator"])) for x in development["initial_a_exact"]]
    if list(map(Fraction, a0)) != old_a0:
        raise ValueError("development original centre mismatch")
    if (saved["b"] != [0.0] * 8 or len(saved["a"]) != 8
            or any(Fraction.from_float(float(value)) != Fraction(exact) for value, exact in zip(saved["a"], a0))):
        raise ValueError("saved D was not evaluated at exact old dyadic a0")
    return a0, saved


def _old_q_check(teacher, old_teacher):
    rows = []
    for label, (new, old) in enumerate(zip(teacher["q_true_component_intervals"], old_teacher["true_PMF_enclosures"])):
        a, b = _interval(new)
        c, d = _interval(old)
        gap = max(Fraction(0), c - b, a - d)
        rows.append(dict(label_index=label, overlap=gap == 0, gap_exact=str(gap),
                         new_minus_old_interval=_encode_interval(a - d, b - c)))
    return dict(scope="saved exact dyadic interval comparison only", rows=rows,
                all_256_overlap=all(row["overlap"] for row in rows), new_target_nodes=0)


def _teacher_GH(order, deadline):
    """Independent ordinary quadrature: full likelihood times first-row score."""
    nodes, weights = roots_hermitenorm(order)
    weights = weights / math.sqrt(2 * math.pi)
    x, y = np.meshgrid(nodes, nodes, indexing="ij")
    z = np.column_stack((x.ravel(), y.ravel()))
    gaussian_weights = np.multiply.outer(weights, weights).ravel()
    phi = np.array([float(Fraction(s)) for s in ANGLES])
    W = .8 * np.column_stack((np.cos(phi), np.sin(phi)))
    p = expit(z @ W.T)
    tangent = .8 * (-math.sin(phi[0]) * z[:, 0] + math.cos(phi[0]) * z[:, 1])
    q, prime = [], []
    for label in range(256):
        if time.perf_counter() >= deadline:
            return dict(order=order, status="budget_stopped", completed_labels=len(q),
                        q=q, prime=prime, quadrature_nodes=order * order,
                        label_probabilities_evaluated=len(q), gaussian_mass=float(gaussian_weights.sum()),
                        renormalized=False)
        bits = np.array([(label >> (7 - i)) & 1 for i in range(8)], dtype=bool)
        # No first-factor derivative products, prefix tree, teacher producer,
        # class compression, or complement averaging is reused here.
        likelihood = np.prod(np.where(bits[None, :], p, 1 - p), axis=1)
        score = (int(bits[0]) - p[:, 0]) * tangent
        q.append(float(np.dot(gaussian_weights, likelihood)))
        prime.append(float(np.dot(gaussian_weights, likelihood * score)))
    return dict(order=order, status="completed", completed_labels=256, q=q, prime=prime,
                quadrature_nodes=order * order, label_probabilities_evaluated=256,
                gaussian_mass=float(gaussian_weights.sum()), q_mass=float(sum(q)),
                prime_mass=float(sum(prime)), renormalized=False,
                exact_family_inputs=dict(angles=list(ANGLES), amplitude="4/5", lambda_centre="0"),
                node_rule="scipy.special.roots_hermitenorm; standard Gaussian weights divided by sqrt(2*pi)",
                arithmetic="ordinary binary64; no certified GH truncation or rounding bound")


def _rule_difference(first, second):
    result = {}
    gate = Fraction("1e-11")
    for key in ("q", "prime"):
        differences = [abs(Fraction.from_float(a) - Fraction.from_float(b)) for a, b in zip(first[key], second[key])]
        total = sum(differences, Fraction(0))
        result[key] = dict(component_absolute_differences=[float(x) for x in differences],
                           L1_exact=str(total), L1_display=float(total), gate_exact=str(gate), passed=total < gate)
    return result


def _candidate_comparisons(block, budgets, saved):
    caps = {"g_a": _interval(budgets["bounds"]["uniform_gradient_L2"])[1],
            "Hbb": _interval(budgets["bounds"]["uniform_Hba_operator"])[1],
            "Haa": _interval(budgets["bounds"]["uniform_Hba_operator"])[1],
            "Hperp": _interval(budgets["bounds"]["uniform_Hperp_operator"])[1]}
    rows = []
    for row in block["conditional_rows"]:
        label = row["label_index"]
        if label not in LABELS or row["status"] != "completed":
            raise ValueError("unexpected completed point label")
        H = saved["label_H_ba"][0][label]
        reference = dict(g_a=saved["label_radial_gradient"][0][label],
                         Hbb=[line[:8] for line in H[:8]],
                         Haa=[line[8:] for line in H[8:]],
                         Hperp=saved["label_Hperp"][0][label])
        raw = dict(g_a=row["conditional_gradient_finite"], **row["matrix_blocks"])
        true, comparisons = {}, {}
        for name, values in raw.items():
            expected = (8,) if name == "g_a" else (8, 8)
            _require_shape(values, expected, "new point " + name)
            true[name] = ([_expanded(x, caps[name]) for x in values] if name == "g_a"
                          else [[_expanded(x, caps[name]) for x in line] for line in values])
            comparisons[name] = _compare(reference[name], true[name], "1e-8", use_L1_gate=False)
        rows.append(dict(label_index=label, point_exact_a0=block["a_input_audit"],
                         true_conditional_intervals=true, saved_GL_values=reference,
                         saved_GL_bias_gradient_not_compared=saved["label_bias_gradient"][0][label],
                         saved_GL_full_Hba_including_mixed_not_fully_compared=H,
                         comparisons=comparisons,
                         passed=all(x["passed"] for x in comparisons.values())))
    return dict(rows=rows, compared_components_per_label=8 + 3 * 64,
                analytic_uniform_caps_exact={name: str(value) for name, value in caps.items()},
                all_requested_labels_compared=[x["label_index"] for x in rows] == list(LABELS),
                passed=len(rows) == 3 and all(row["passed"] for row in rows),
                scope="g_a and Hbb/Haa/Hperp only; single-label g_b and mixed Hba are not produced by the new point API")


def produce_audit(inputs=None, wall_seconds=120):
    """Run only after a caller has frozen this module, sources, and inputs."""
    if inputs is None:
        inputs = prepared_inputs()
    if inputs != prepared_inputs():
        raise ValueError("audit specification drift")
    if isinstance(wall_seconds, bool) or not math.isfinite(float(wall_seconds)) or wall_seconds <= 0:
        raise ValueError("positive finite wall_seconds required")
    tick, deadline = time.perf_counter(), time.perf_counter() + wall_seconds
    result = dict(schema="r03_independent_audit_v1", inputs=inputs,
                  status="running", passed=False, wall_seconds_budget=wall_seconds,
                  teacher_runs=[], candidate_point=None, ordinary_checks_are_not_second_strict_certificate=True,
                  old_GL_kernel_reexecuted=False, old_NPZ_reads=0, automatic_retry=False,
                  runtime=dict(numpy=np.__version__, scipy=scipy.__version__,
                               python_flint=candidate.flint.__version__),
                  planned_counts=dict(teacher_GH_nodes=16400, point_labels=list(LABELS),
                                      point_laplace_factors=648051))
    try:
        documents, sources = _read_sources()
        result["sources"] = sources
        box, development, teacher, old_teacher = documents
        a0, saved = _validate_saved(box, development, teacher, old_teacher)
        result["verified_saved_schema"] = dict(K_values=saved["K_values"],
                                               radial_shape=[2, 256, 8], Hba_shape=[2, 256, 16, 16],
                                               Hperp_shape=[2, 256, 8, 8], exact_old_a0=a0)
        result["old_teacher_interval_check"] = _old_q_check(teacher, old_teacher)
        for order in ORDERS:
            if time.perf_counter() >= deadline:
                result["status"] = "budget_stopped_before_teacher_rule"
                return result
            run = _teacher_GH(order, deadline)
            result["teacher_runs"].append(run)
            if run["status"] != "completed":
                result["status"] = "budget_stopped_in_teacher_rule"
                return result
            run["strict_reference_comparisons"] = {
                "q": _compare(run["q"], teacher["q_true_component_intervals"], "1e-11", use_L1_gate=True),
                "prime": _compare(run["prime"], teacher["prime_true_component_intervals"], "1e-11", use_L1_gate=True)}
        result["teacher_two_rule_difference"] = _rule_difference(*result["teacher_runs"])
        if time.perf_counter() >= deadline:
            result["status"] = "budget_stopped_before_candidate_point"
            return result
        budgets = candidate.analytic_error_bounds(a0, POINT_CONFIG,
                                                  teacher_error_L1_upper="0", teacher_mass_upper="1")
        cache = candidate.precompute(a0, POINT_CONFIG)
        result["candidate_point_precompute"] = dict(
            signature=cache["signature"], normal_nodes=cache["normal_nodes"],
            logistic_exponential_calls=cache["precompute_logistic_exponentials"],
            gaussian_exponential_calls=cache["precompute_gaussian_exponentials"])
        result["candidate_point_analytic_bounds"] = budgets
        try:
            point = candidate.finite_label_block(cache, LABELS, deadline=deadline)
        except candidate.CandidateBudgetStop as exc:
            result["candidate_point"] = exc.partial
            result["candidate_comparisons"] = _candidate_comparisons(exc.partial, budgets, saved)
            result["status"] = "budget_stopped_in_candidate_point"
            return result
        result["candidate_point"] = point
        if point["counts"]["laplace_factors_evaluated"] != 648051:
            raise ValueError("point factor count drift")
        result["candidate_comparisons"] = _candidate_comparisons(point, budgets, saved)
        teacher_pass = all(comp["passed"] for run in result["teacher_runs"] for comp in run["strict_reference_comparisons"].values())
        teacher_pass &= all(comp["passed"] for comp in result["teacher_two_rule_difference"].values())
        result["passed"] = bool(teacher_pass and result["old_teacher_interval_check"]["all_256_overlap"]
                                and result["candidate_comparisons"]["passed"])
        result["status"] = "completed_passed" if result["passed"] else "completed_with_failed_gates"
        return result
    except Exception as exc:
        result["status"] = "failed_with_partial_evidence"
        result["error"] = dict(type=type(exc).__name__, message=str(exc))
        return result
    finally:
        result["elapsed_seconds"] = time.perf_counter() - tick
        result["actual_counts"] = dict(
            teacher_GH_nodes=sum(run["quadrature_nodes"] for run in result["teacher_runs"]),
            teacher_rule_label_sums=sum(2 * run["completed_labels"] for run in result["teacher_runs"]),
            point_laplace_factors=(result["candidate_point"] or {}).get("counts", {}).get("laplace_factors_evaluated", 0),
            optimizer_updates=0, random_draws=0, scientific_NPZ_reads=0)


def self_test():
    """Codec/shape/gate tests only; zero Gaussian or candidate target nodes."""
    checks = {}
    box = _encode_interval(Fraction(-3, 8), Fraction(5, 16))
    checks["dyadic_codec_roundtrip"] = _interval(box) == (Fraction(-3, 8), Fraction(5, 16))
    expanded = _expanded(box, Fraction(1, 32))
    checks["dyadic_expansion"] = _interval(expanded) == (Fraction(-13, 32), Fraction(11, 32))
    small = _encode_interval(Fraction(1, 4), Fraction(1, 4))
    cmp = _compare([.25], [small], "1e-8", use_L1_gate=True)
    checks["exact_float_comparison"] = cmp["passed"] and cmp["L1_worst_absolute_difference_exact"] == "0"
    wide = _compare([0.0], [box], "1/100", use_L1_gate=False)
    checks["wide_interval_does_not_false_pass_by_overlap"] = not wide["passed"] and wide["distance_to_interval_exact"] == ["0"]
    checks["rectangular_shape"] = _shape([[1, 2], [3, 4]]) == (2, 2)
    try:
        _shape([[1], [2, 3]])
    except ValueError:
        checks["ragged_shape_rejected"] = True
    else:
        checks["ragged_shape_rejected"] = False
    checks["fixed_count_arithmetic"] = 3 * 641 * 337 == 648051 and 80 ** 2 + 100 ** 2 == 16400
    if not all(checks.values()):
        raise AssertionError(json.dumps(checks))
    return dict(schema="r03_independent_audit_manufactured_v1", passed=True, checks=checks,
                new_teacher_nodes=0, new_candidate_nodes=0, scientific_file_reads=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(self_test(), ensure_ascii=False, indent=2, allow_nan=False))
