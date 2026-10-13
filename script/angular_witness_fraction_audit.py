"""Independent standard-library/Fraction audit of saved AngularWitness data.

No project modules, Arb, quadrature, analytic-bound recomputation or scientific
arrays are used. The saved analytic error upper bounds remain mathematical
inputs; this program independently composes their exact rational intervals.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "artifacts/angular_witness_v1_20261010"
OUTPUT = PACKAGE / "FRACTION_AUDIT.json"
SOURCE_PATHS = {
    "center": "artifacts/certified_teacher_v1_20261008/CANDIDATE_RISK_INTEGRALS.json",
    "teacher_finite": "artifacts/certified_teacher_v1_20261008/FINITE_GRID_SUM.json",
    "teacher_certificate": "artifacts/certified_teacher_v1_20261008/TEACHER_CERTIFICATE.json",
    "Q_certificate": "artifacts/certified_teacher_v1_20261008/TEACHER_RISK_CERTIFICATE.json",
    "root_certificate": "artifacts/certified_f8_hessian_v1_20261008/HESSIAN_AND_WITNESS_CERTIFICATE.json",
}
EXPECTED_U = [-8440215, -6758298, -4585241, -1673686,
              1673686, 4585241, 6758298, 8440215]
EXPECTED_S = 281474961915412


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact(value):
    require(not isinstance(value, (bool, float)), "explicit exact rational required")
    return Fraction(value)


def endpoint(data):
    mantissa, exponent = exact(data["mantissa"]), exact(data["exponent_of_two"])
    require(mantissa.denominator == exponent.denominator == 1, "integer dyadic fields required")
    m, e = mantissa.numerator, exponent.numerator
    return Fraction(m << e) if e >= 0 else Fraction(m, 1 << -e)


def interval(record):
    lower = endpoint(record["lower_exact_dyadic"])
    upper = endpoint(record["upper_exact_dyadic"])
    require(lower <= upper, "inverted exact saved endpoints")
    return lower, upper


def add(left, right):
    return left[0] + right[0], left[1] + right[1]


def multiply(left, right):
    products = [x * y for x in left for y in right]
    return min(products), max(products)


def divide_positive(numerator, denominator):
    require(denominator[0] > 0, "denominator must be strictly positive")
    return multiply(numerator, (1 / denominator[1], 1 / denominator[0]))


def enlarge(value, radius):
    radius = exact(radius)
    require(radius >= 0, "nonnegative exact error radius required")
    return value[0] - radius, value[1] + radius


def overlap(left, right):
    return left[0] <= right[1] and right[0] <= left[1]


def contained(inner, outer):
    return outer[0] <= inner[0] and inner[1] <= outer[1]


def fraction_record(value):
    # Different positive Q endpoints create large rational denominators when
    # 256 quotients are summed. Hex is exact and avoids Python's decimal digit
    # conversion limit without modifying that process-wide security setting.
    return {"numerator_hex": hex(value.numerator), "denominator_hex": hex(value.denominator)}


def interval_record(value):
    return {"lower_exact_fraction": fraction_record(value[0]),
            "upper_exact_fraction": fraction_record(value[1])}


def safe_source(name):
    path = (ROOT / name).resolve()
    require(path.is_relative_to(ROOT), "source path escapes repository")
    require(path.suffix.lower() not in (".npz", ".npy"), "scientific arrays prohibited")
    return path


def check_bindings(bindings):
    for name, digest in bindings.items():
        require(sha(safe_source(name)) == digest, "source binding mismatch: " + name)


def self_test():
    F = Fraction
    require(multiply((F(-2), F(3)), (F(-4), F(5))) == (F(-12), F(15)), "mixed-sign multiply")
    require(divide_positive((F(-6), F(4)), (F(2), F(3))) == (F(-3), F(2)), "mixed-sign divide")
    require(divide_positive((F(-6), F(-4)), (F(2), F(3))) == (F(-3), F(-4, 3)), "negative divide")
    require(enlarge((F(1, 3), F(2, 3)), F(1, 6)) == (F(1, 6), F(5, 6)), "exact enlargement")
    require(add((F(-1, 3), F(2, 5)), (F(1, 7), F(3, 8))) == (F(-4, 21), F(31, 40)), "exact sum")
    require(endpoint({"mantissa": "-7", "exponent_of_two": -3}) == F(-7, 8), "dyadic negative")
    require(endpoint({"mantissa": "3", "exponent_of_two": 5}) == F(96), "dyadic positive exponent")
    for rational in (F(-1, 3), F(7, 8), F(0)):
        encoded = fraction_record(rational)
        require(F(int(encoded["numerator_hex"], 16), int(encoded["denominator_hex"], 16)) == rational,
                "hex rational round trip")
    require(overlap((F(0), F(1)), (F(1), F(2))), "closed endpoint intersection")
    require(contained((F(0), F(1)), (F(-1), F(2))), "interval containment")
    try:
        divide_positive((F(1), F(2)), (F(0), F(1)))
    except ValueError:
        pass
    else:
        raise ValueError("zero denominator accepted")
    return {"passed": True, "arithmetic": "stdlib fractions.Fraction only", "new_integrals": 0}


def reconstruct():
    self_test()
    certificate = read(PACKAGE / "CERTIFICATE.json")
    inputs = read(PACKAGE / "INPUTS.json")
    selection = read(PACKAGE / "SELECTION_FREEZE.json")
    production = read(PACKAGE / "PRODUCTION_FREEZE.json")
    pair = read(PACKAGE / "PAIR_FINITE.json")
    audit = read(PACKAGE / "AUDIT_FINITE.json")
    require(certificate["schema"] == "angular_same_root_certificate_v1" and certificate["passed"],
            "completed passing primary certificate required")
    require(inputs["u_integer"] == EXPECTED_U and inputs["S_integer"] == EXPECTED_S
            == sum(x * x for x in EXPECTED_U) and sum(EXPECTED_U) == 0, "fixed direction")
    require(inputs["bits"] == 192 and inputs["step_exact"] == "1/8" and inputs["N"] == 88,
            "fixed grid identity")
    require(inputs["direction_sha256"] == sha(PACKAGE / "DIRECTION.json"), "direction file identity")
    direction = read(PACKAGE / "DIRECTION.json")
    require(direction["u_integer"] == EXPECTED_U and direction["S_integer"] == EXPECTED_S,
            "direction file values")
    require(inputs["selection_freeze_sha256"] == sha(PACKAGE / "SELECTION_FREEZE.json"), "selection identity")
    check_bindings({name: digest for name, digest in selection["bindings"].items() if name != "HANDOFF.md"})
    require(sha(PACKAGE / "HANDOFF_INPUT.md") == selection["bindings"]["HANDOFF.md"], "archived user feedback")
    check_bindings(production["bindings"])
    binding_fields = {"input_sha256": "INPUTS.json", "freeze_sha256": "PRODUCTION_FREEZE.json",
                      "pair_finite_sha256": "PAIR_FINITE.json", "audit_finite_sha256": "AUDIT_FINITE.json"}
    for field, name in binding_fields.items():
        require(certificate[field] == sha(PACKAGE / name), "certificate binding: " + field)
    for kind, finite in (("PAIR", pair), ("AUDIT", audit)):
        require(finite["status"] == "completed" and finite["nodes_completed"] == finite["nodes_prescribed"] == 177,
                kind + " incomplete finite sum")
        require(finite["m"] == 8 and finite["label_count"] == 256 and finite["precision_bits"] == 192
                and finite["step_exact"] == "1/8" and finite["N"] == 88
                and finite["endpoint_weights"] == "full", kind + " finite configuration")
        require(finite["input_sha256"] == sha(PACKAGE / "INPUTS.json")
                and finite["production_freeze_sha256"] == sha(PACKAGE / "PRODUCTION_FREEZE.json")
                and finite["run_start_sha256"] == sha(PACKAGE / (kind + "_START.json")), kind + " finite binding")
        require(finite["hard_budget_seconds"] == 600 and 0 <= finite["wall_seconds"] <= 600, kind + " budget")
    source = {}
    for key, name in SOURCE_PATHS.items():
        require(inputs["source_paths"][key] == name and name in selection["bindings"], "fixed source mapping")
        source[key] = read(safe_source(name))
    teacher_cap = exact(inputs["teacher_L1_cap_exact"])
    require(teacher_cap == Fraction("4.894e-21")
            and interval(source["teacher_certificate"]["total_L1_error_from_finite_midpoints"])[1] < teacher_cap,
            "saved teacher error cap")
    require(interval(source["root_certificate"]["witness"]["root_distance_upper"])[1] < Fraction("1.805e-8"),
            "saved root distance cap")
    transfer = exact(inputs["transfer_cap_exact"])
    require(transfer == Fraction("7.52685e-6") == Fraction(417) * Fraction("1.805e-8"), "root transfer")
    allowance = 8 * teacher_cap
    require(exact(certificate["teacher_curvature_allowance_exact"]) == allowance
            and exact(certificate["transfer_cap_exact"]) == transfer
            and exact(certificate["teacher_L1_cap_exact"]) == teacher_cap, "saved composition constants")
    require(certificate["teacher_midpoints_renormalized"] is False, "teacher must not be renormalized")
    pb, ab = certificate["pair_bounds"], certificate["audit_bounds"]
    for bounds in (pb, ab):
        require(bounds["m"] == 8 and bounds["N"] == 88 and bounds["step_exact"] == "1/8"
                and bounds["strip_exact"] == "1" and bounds["precision_bits"] == 192, "saved analytic-bound configuration")
    pe = interval(pb["errors"]["N_total"])
    ae = interval(ab["errors"]["N_path_total"])
    qe = interval(ab["errors"]["Q_total"])
    require(min(pe[0], ae[0], qe[0]) >= 0, "nonnegative saved analytic errors")
    teacher = source["teacher_finite"]["pmf_finite_sum"]
    Qsaved = source["Q_certificate"]["candidate_Q_true_enclosures"]
    old_Qfinite = source["center"]["finite_sum"]["Q_finite_sum"]
    require(all(len(items) == 256 for items in (teacher, Qsaved, old_Qfinite, pair["N_finite_sum"],
                audit["N_path_finite_sum"], audit["Q_finite_sum"], certificate["rows"])), "all 256 labels required")
    total = audit_total = mass_pair = mass_audit = (Fraction(0), Fraction(0))
    rows = []
    for label in range(256):
        saved_row = certificate["rows"][label]
        require(saved_row["label"] == saved_row["Q_source_label"] == label, "saved label order")
        qhat_bounds = interval(teacher[label])
        qhat = (qhat_bounds[0] + qhat_bounds[1]) / 2
        require(qhat >= 0 and exact(saved_row["teacher_midpoint_exact"]) == qhat, "exact teacher midpoint")
        denominator = interval(Qsaved[label])
        require(denominator[0] > 0, "old true-Q positivity")
        raw_pair = interval(pair["N_finite_sum"][label])
        raw_audit = interval(audit["N_path_finite_sum"][label])
        raw_Qaudit = interval(audit["Q_finite_sum"][label])
        numerator = enlarge(raw_pair, pe[1])
        numerator_audit = enlarge(raw_audit, ae[1])
        Qaudit = enlarge(raw_Qaudit, qe[1])
        ratio = divide_positive(numerator, denominator)
        ratio_audit = divide_positive(numerator_audit, denominator)
        values = {"N_true": numerator, "N_path_true": numerator_audit, "audit_Q_true": Qaudit,
                  "ratio": ratio, "audit_ratio": ratio_audit}
        inclusion = {name: contained(value, interval(saved_row[name])) for name, value in values.items()}
        require(all(inclusion.values()), "exact Fraction interval escapes saved Arb enclosure at label " + str(label))
        intersections = {
            "numerator_true_interval_overlap": overlap(numerator, numerator_audit),
            "denominator_true_interval_overlap": overlap(Qaudit, denominator),
            "denominator_same_finite_sum_overlap": overlap(raw_Qaudit, interval(old_Qfinite[label])),
        }
        require(all(intersections.values()), "independent interval disagreement at label " + str(label))
        require(all(saved_row[name] is True for name in intersections), "saved overlap flag disagrees")
        total = add(total, multiply((qhat, qhat), ratio))
        audit_total = add(audit_total, multiply((qhat, qhat), ratio_audit))
        mass_pair = add(mass_pair, raw_pair)
        mass_audit = add(mass_audit, raw_audit)
        rows.append({"label": label, "teacher_midpoint_exact": str(qhat),
                     "fraction_intervals": {name: interval_record(value) for name, value in values.items()},
                     "contained_in_primary_Arb_intervals": inclusion, **intersections})
    center, audit_center = enlarge(total, allowance), enlarge(audit_total, allowance)
    root, audit_root = enlarge(center, transfer), enlarge(audit_center, transfer)
    summary = {"center": center, "audit_center": audit_center, "root": root, "audit_root": audit_root}
    inclusion = {name: contained(value, interval(certificate["summary"][name])) for name, value in summary.items()}
    negative = {name: value[1] < 0 for name, value in summary.items()}
    gate = exact(inputs["center_width_gate_exact"])
    checks = {
        "all_256_exact_teacher_midpoints_match": True,
        "all_1280_row_intervals_contained_in_saved_Arb_intervals": True,
        "all_256_numerator_true_intervals_overlap": True,
        "all_256_Q_true_intervals_overlap": True,
        "all_256_Q_finite_intervals_overlap": True,
        "finite_pair_numerator_mass_contains_zero": mass_pair[0] <= 0 <= mass_pair[1],
        "finite_path_numerator_mass_contains_zero": mass_audit[0] <= 0 <= mass_audit[1],
        "all_four_summary_intervals_contained_in_saved_Arb_intervals": all(inclusion.values()),
        "all_four_summary_upper_endpoints_strictly_negative": all(negative.values()),
        "center_compositions_overlap": overlap(center, audit_center),
        "root_compositions_overlap": overlap(root, audit_root),
        "primary_center_width_below_gate": center[1] - center[0] < gate,
        "audit_center_width_below_gate": audit_center[1] - audit_center[0] < gate,
    }
    require(all(checks.values()), "independent exact Fraction composition failed: " + str(checks))
    names = ["CERTIFICATE.json", "INPUTS.json", "DIRECTION.json", "SELECTION_FREEZE.json",
             "PRODUCTION_FREEZE.json", "PAIR_FINITE.json", "AUDIT_FINITE.json", "PAIR_START.json", "AUDIT_START.json"]
    binding_paths = [str((PACKAGE / name).relative_to(ROOT)).replace("\\", "/") for name in names]
    binding_paths += list(SOURCE_PATHS.values()) + [str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/")]
    return {
        "schema": "angular_witness_fraction_audit_v1", "passed": True,
        "arithmetic": "Python standard-library fractions.Fraction; exact rational endpoint arithmetic",
        "source_bindings_sha256": {name: sha(safe_source(name)) for name in binding_paths},
        "direction_u_integer": EXPECTED_U, "direction_S_integer": EXPECTED_S,
        "label_count": 256, "rows": rows,
        "used_analytic_error_upper_bounds": {"pair_N": fraction_record(pe[1]),
            "audit_N": fraction_record(ae[1]), "audit_Q": fraction_record(qe[1])},
        "teacher_L1_cap_exact": str(teacher_cap), "teacher_curvature_allowance_exact": str(allowance),
        "transfer_cap_exact": str(transfer), "teacher_midpoints_renormalized": False,
        "summary_exact_fraction_intervals": {name: interval_record(value) for name, value in summary.items()},
        "summary_contained_in_primary_Arb_intervals": inclusion,
        "summary_upper_endpoints_strictly_negative": negative,
        "summary_endpoint_denominator_max_bits": max(x.denominator.bit_length() for value in summary.values() for x in value),
        "finite_numerator_mass_intervals": {"pair": interval_record(mass_pair), "path": interval_record(mass_audit)},
        "checks": checks,
        "scope": {"analytic_error_proof_repeated": False, "saved_error_upper_bounds_used": True,
            "new_integrals": 0, "new_target_grid_nodes": 0, "scientific_NPZ_reads": 0,
            "project_modules_imported": 0, "Arb_or_other_nonstandard_packages_imported": 0,
            "floating_point_endpoint_arithmetic": False, "external_scientific_review": False},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("write", "check", "self-test"):
        modes.add_argument("--" + name, action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
    else:
        result = reconstruct()
        if args.write:
            with OUTPUT.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write("\n")
        else:
            require(read(OUTPUT) == result, "saved Fraction audit differs from exact reconstruction")
        result = {"passed": result["passed"], "label_count": result["label_count"],
                  "checks": result["checks"],
                  "summary_endpoint_denominator_max_bits": result["summary_endpoint_denominator_max_bits"],
                  "new_integrals": 0}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
