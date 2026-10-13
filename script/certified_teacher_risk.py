"""Fixed 177-node candidate Q/B_h Arb sums and teacher-risk composition.

No NPZ, scientific kernel, optimizer or RNG. --run must follow the shared
freeze; finite candidate sums are retained even when teacher output is pending.
--combine and --check use saved JSON and scalar Arb algebra only, no new sums.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/".cache/finite_selection_tools"))
from flint import arb, ctx
import flint

OUT = ROOT/"artifacts/certified_teacher_v1_20261008"
FREEZE = OUT/"FREEZE.json"
NOMINAL = ROOT/"artifacts/local_witness_feasibility_v1_20261008/NOMINAL_SUMMARY.json"
CANDIDATE = OUT/"CANDIDATE_RISK_INTEGRALS.json"
RUN_START = OUT/"CANDIDATE_RISK_RUN_START.json"
COMPOSED = OUT/"TEACHER_RISK_CERTIFICATE.json"
TEACHER = OUT/"FINITE_GRID_SUM.json"
BITS, STEP, N, STRIP, HARD_SECONDS = 192, Fraction(1, 8), 88, Fraction(1), 60
SOURCE_NAMES = ("script/certified_teacher_risk.py", "script/certified_teacher_risk_test.py",
               "artifacts/local_witness_feasibility_v1_20261008/NOMINAL_SUMMARY.json",
               "script/certified_teacher_bounds.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def exact(value):
    if isinstance(value, (bool, float)):
        raise ValueError("explicit exact rational required")
    return Fraction(value)


def aq(value):
    value = exact(value)
    return arb(value.numerator)/value.denominator


def ep(value):
    mantissa, exponent = value.man_exp()
    return {"mantissa": str(mantissa), "exponent_of_two": int(exponent)}


def record(value):
    if not value.is_finite():
        raise ArithmeticError("nonfinite Arb result")
    return {"arb_display_only": str(value), "lower_exact_dyadic": ep(value.lower()),
            "upper_exact_dyadic": ep(value.upper())}


def endpoint_fraction(data):
    m, e = int(data["mantissa"]), int(data["exponent_of_two"])
    return Fraction(m*(2**max(e, 0)), 2**max(-e, 0))


def from_record(data):
    lo, hi = endpoint_fraction(data["lower_exact_dyadic"]), endpoint_fraction(data["upper_exact_dyadic"])
    if lo > hi:
        raise ValueError("inverted saved Arb endpoints")
    return aq(lo).union(aq(hi))


def enlarge(value, absolute_error):
    if not absolute_error >= 0:
        raise ValueError("nonnegative absolute error required")
    return value+arb(0, absolute_error.upper())


def dyadic(value):
    value = exact(value)
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def rational_vector(records):
    return [Fraction(int(x["numerator"]), int(x["denominator"])) for x in records]


def strip_bounds(a, h, *, step=STEP, N=N, strip=STRIP):
    """Uniform absolute scalar errors, not a numerical integration call."""
    a, h = list(map(exact, a)), list(map(exact, h))
    step, strip = exact(step), exact(strip)
    if len(a) != len(h) or not a or step <= 0 or strip <= 0 or N < 1:
        raise ValueError("invalid candidate strip/grid parameters")
    limit = step*N
    if limit < 1:
        raise ValueError("Gaussian x*phi tail bound requires grid limit at least 1")
    if any(not aq(abs(x)*strip) < arb.pi() for x in a):
        raise ValueError("candidate analytic strip must avoid sigmoid poles: |a_j|*strip < pi")
    cosines = [(aq(x)*aq(strip)/2).cos() for x in a]
    if any(not c > 0 for c in cosines):
        raise ValueError("candidate sigmoid strip denominator not uniformly positive")
    C = arb(1)
    for cosine in cosines:
        C /= cosine
    h_norm = aq(sum((x*x for x in h), Fraction(0))).sqrt()
    if not h_norm > 0:
        raise ValueError("fixed nonzero h required")
    h_l1 = aq(sum(map(abs, h), Fraction(0)))
    Hsec = sum((aq(abs(x))/c for x, c in zip(h, cosines)), arb(0))
    gaussian_shift = (aq(strip)*aq(strip)/2).exp()
    Mq = gaussian_shift*C
    Mb = Mq*Hsec*(1+aq(strip)*aq(strip)).sqrt()
    denominator = (2*arb.pi()*aq(strip)/aq(step)).exp()-1
    phiL = (-aq(limit)*aq(limit)/2).exp()/(2*arb.pi()).sqrt()
    q_alias, b_alias = 2*Mq/denominator, 2*Mb/denominator
    q_tail, b_tail = 2*phiL/aq(limit), 2*h_l1*phiL
    qmin = (-(aq(sum((x*x for x in a), Fraction(0))))/8).exp()/(arb(2)**len(a))
    return {"step_exact": str(step), "N": N, "node_count": 2*N+1,
        "limit_exact": str(limit), "strip_exact": str(strip), "endpoint_weights": "full",
        "label_uniform": True, "h_normalized_in_finite_sum": False,
        "constants": {"C_A": record(C), "H_sec": record(Hsec), "h_norm": record(h_norm),
            "h_l1": record(h_l1), "boundary_Q_single_side": record(Mq),
            "boundary_B_h_single_side": record(Mb), "Qmin_center_Jensen": record(qmin)},
        "errors": {"Q_alias": record(q_alias), "Q_grid_tail": record(q_tail),
            "Q_total": record(q_alias+q_tail), "B_h_alias": record(b_alias),
            "B_h_grid_tail": record(b_tail), "B_h_total": record(b_alias+b_tail),
            "B_unit_total": record((b_alias+b_tail)/h_norm)},
        "finite_sum_arithmetic_included_separately_in_saved_balls": True}


def finite_candidate(a, h, *, step=STEP, N=N, hard_seconds=HARD_SECONDS):
    """Fixed full-weight 1D finite sum for every label, in lexicographic order."""
    a, h = list(map(exact, a)), list(map(exact, h))
    step = exact(step)
    if len(a) != len(h) or not a or step <= 0 or N < 1 or hard_seconds <= 0:
        raise ValueError("invalid finite candidate configuration")
    m = len(a)
    label_h = [Fraction(0)]
    for hj in h:
        label_h = [item for old in label_h for item in (old, old+hj)]
    label_h_arb = list(map(aq, label_h))
    aa, hh, dx = list(map(aq, a)), list(map(aq, h)), aq(step)
    Q, B = [arb(0) for _ in range(2**m)], [arb(0) for _ in range(2**m)]
    normalizer = (2*arb.pi()).sqrt()
    tick, completed = time.perf_counter(), 0
    for node in range(-N, N+1):
        if time.perf_counter()-tick > hard_seconds:
            break
        x = dx*node
        probabilities = [1/(1+(-aj*x).exp()) for aj in aa]
        h_p = sum((hj*pj for hj, pj in zip(hh, probabilities)), arb(0))
        products = [arb(1)]
        for p in probabilities:
            products = [item for old in products for item in (old*(1-p), old*p)]
        weight = dx*(-x*x/2).exp()/normalizer
        for label, product in enumerate(products):
            weighted = weight*product
            Q[label] += weighted
            B[label] += weighted*x*(label_h_arb[label]-h_p)
        completed += 1
    seconds = time.perf_counter()-tick
    complete = completed == 2*N+1 and seconds <= hard_seconds
    return {"status": "completed" if complete else "budget_stopped",
        "m": m, "label_count": 2**m, "label_order": "integer labels 0..2^m-1; observation 0 is MSB",
        "step_exact": str(step), "N": N, "endpoint_weights": "full",
        "nodes_prescribed": 2*N+1, "nodes_completed": completed,
        "Q_finite_sum": [record(x) for x in Q], "B_h_finite_sum": [record(x) for x in B],
        "wall_seconds": seconds, "hard_budget_seconds": hard_seconds,
        "real_sigmoid_evaluations": completed*m, "real_Gaussian_density_evaluations": completed,
        "label_probability_accumulations": completed*(2**m),
        "label_directional_accumulations": completed*(2**m),
        "finite_scalar_integrals_completed": 2*(2**m) if complete else 0,
        "new_kernel_calls": 0, "new_random_draws": 0, "new_optimizer_updates": 0,
        "scientific_NPZ_reads": 0, "finite_sum_only": True}


def candidate_ratios(candidate):
    if candidate["finite_sum"]["status"] != "completed":
        raise ValueError("unfinished candidate sums cannot be promoted to true integral enclosures")
    finite = candidate["finite_sum"]
    a, h = rational_vector(candidate["a_exact"]), rational_vector(candidate["h_exact"])
    if len(a) != 8 or len(h) != 8 or len(finite["Q_finite_sum"]) != 256 or len(finite["B_h_finite_sum"]) != 256:
        raise ValueError("fixed full256 candidate shape required")
    budget = strip_bounds(a, h)
    if candidate["analytic_bounds"] != budget:
        raise ValueError("saved candidate strip-bound drift")
    eq = from_record(budget["errors"]["Q_total"])
    eb = from_record(budget["errors"]["B_h_total"])
    hn = from_record(budget["constants"]["h_norm"])
    ratios, Qtrue, Btrue = [], [], []
    for q, b in zip(finite["Q_finite_sum"], finite["B_h_finite_sum"]):
        qi, bi = enlarge(from_record(q), eq), enlarge(from_record(b), eb)
        if not qi > 0:
            raise ValueError("candidate true-Q enclosure is not strictly positive")
        Qtrue.append(qi); Btrue.append(bi)
        ratios.append(bi/(qi*hn))
    return ratios, Qtrue, Btrue, budget


def compose(candidate, teacher, teacher_bound):
    ratios, Qtrue, Btrue, budget = candidate_ratios(candidate)
    probabilities = teacher["pmf_finite_sum"]
    if len(probabilities) != 256:
        raise ValueError("teacher finite PMF must retain all256 labels")
    finite_rho = sum((from_record(q)*r for q, r in zip(probabilities, ratios)), arb(0))
    delta = from_record(teacher_bound["bounds"]["total_quadrature_L1"])
    max_ratio_upper = max(abs(r).upper() for r in ratios)
    teacher_allowance = delta*max_ratio_upper
    true_rho = enlarge(finite_rho, teacher_allowance)
    return {"schema": "certified_teacher_risk_composition_v1", "precision_bits": BITS,
        "direction": "d=(0,-h/sqrt(h.T h),0), fixed saved dyadic h",
        "B_h_definition": "E[A_y(Z) Z sum_j h_j(y_j-sigma(a_j Z))]; derivative of Q along +h",
        "risk_directional_derivative_sign": "DR[d]=sum_y q_true(y) B_h(y)/(Q_y sqrt(h.T h))",
        "label_count": 256, "candidate_Q_true_enclosures": [record(q) for q in Qtrue],
        "candidate_B_h_true_enclosures": [record(b) for b in Btrue],
        "candidate_ratio_unit_enclosures": [record(r) for r in ratios],
        "teacher_finite_weighted_risk_direction": record(finite_rho),
        "teacher_total_quadrature_L1_bound": record(delta),
        "candidate_ratio_absolute_max_upper": record(max_ratio_upper),
        "teacher_L1_transfer_allowance": record(teacher_allowance),
        "true_center_risk_direction_enclosure": record(true_rho),
        "true_center_risk_direction_upper_below_minus_point020": bool(true_rho < -aq(Fraction(1, 50))),
        "center_risk_direction_certified_negative": bool(true_rho < 0),
        "teacher_finite_sum_arithmetic_enclosures_retained": True,
        "teacher_or_candidate_mass_renormalized": False,
        "teacher_total_L1_not_repeated_per_label": True,
        "true_F8_stationary_existence_certified": False,
        "true_F8_local_minimum_certified": False,
        "new_kernel_calls": 0, "new_integrals_during_composition": 0,
        "new_random_draws": 0, "scientific_NPZ_reads": 0}


def verify_freeze():
    if not FREEZE.is_file():
        raise RuntimeError("shared freeze required before real candidate sums")
    frozen = read(FREEZE)
    sources = frozen.get("sources")
    if not isinstance(sources, dict) or any(name not in sources for name in SOURCE_NAMES):
        raise ValueError("freeze must bind risk/test, teacher bounds and old nominal source")
    for name, digest in sources.items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or path.suffix.lower() in (".npz", ".npy"):
            raise ValueError("risk source lock cannot access arrays or escape repository")
        if sha(path) != digest:
            raise ValueError("frozen source drift: "+name)
    return frozen


def fixed_input():
    nominal = read(NOMINAL)
    theta = nominal["center"]["theta"]
    if len(theta) != 16 or any(float(x) != 0. for x in theta[:8]):
        raise ValueError("fixed zero-bias center required")
    a = [Fraction.from_float(float(x)) for x in theta[8:]]
    if a != a[::-1] or any(x <= 0 for x in a):
        raise ValueError("positive exact reversal-symmetric candidate required")
    hh = nominal["risk_direction"]["h_dyadic"]
    h = rational_vector(hh)
    if len(h) != 8 or h != h[::-1] or not any(h):
        raise ValueError("fixed nonzero paired h required")
    for value, item in zip(h, hh):
        if value != Fraction.from_float(float.fromhex(item["hex"])) or value != Fraction.from_float(float(item["value"])):
            raise ValueError("saved h dyadic/hex/value mismatch")
    return a, h


def write_new(path, data):
    OUT.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def now():
    return datetime.now(timezone.utc).isoformat()


def run():
    if CANDIDATE.exists() or RUN_START.exists():
        raise FileExistsError("candidate risk run already started/saved; no overwrite or retry")
    verify_freeze()
    a, h = fixed_input()
    previous = ctx.prec
    ctx.prec = BITS
    try:
        write_new(RUN_START, {"created_utc": now(), "freeze_sha256": sha(FREEZE),
            "nominal_source_sha256": sha(NOMINAL), "risk_source_sha256": sha(__file__),
            "nodes_prescribed": 177, "hard_budget_seconds": HARD_SECONDS,
            "automatic_retry_or_node_extension": False})
        finite = finite_candidate(a, h)
        payload = {"schema": "certified_candidate_risk_integrals_v1", "created_utc": now(),
            "precision_bits": BITS, "python_flint_version": flint.__version__,
            "freeze_sha256": sha(FREEZE), "nominal_source_sha256": sha(NOMINAL),
            "risk_source_sha256": sha(__file__), "run_start_sha256": sha(RUN_START),
            "a_exact": [dyadic(x) for x in a],
            "h_exact": [dyadic(x) for x in h], "finite_sum": finite,
            "analytic_bounds": strip_bounds(a, h),
            "true_risk_direction_composed": False, "true_F8_stationary_existence_certified": False}
        write_new(CANDIDATE, payload)
        return {"status": finite["status"], "candidate_saved": True,
            "teacher_composition_pending": True, "candidate_nodes_completed": finite["nodes_completed"],
            "candidate_scalar_integrals_completed": finite["finite_scalar_integrals_completed"],
            "wall_seconds": finite["wall_seconds"], "new_kernel_calls": 0,
            "new_random_draws": 0, "scientific_NPZ_reads": 0}
    finally:
        ctx.prec = previous


def compose_files():
    candidate = check_candidate()
    if candidate["freeze_sha256"] != sha(FREEZE) or candidate["nominal_source_sha256"] != sha(NOMINAL) or candidate["risk_source_sha256"] != sha(__file__) or candidate["run_start_sha256"] != sha(RUN_START):
        raise ValueError("candidate source/freeze drift")
    a, h = fixed_input()
    if rational_vector(candidate["a_exact"]) != a or rational_vector(candidate["h_exact"]) != h:
        raise ValueError("fixed candidate/direction drift")
    if not TEACHER.is_file():
        raise FileNotFoundError("teacher finite sum pending; saved candidate retained")
    teacher = read(TEACHER)
    if teacher.get("schema") != "certified_teacher_finite_grid_sum_v1" or teacher.get("freeze_sha256") != sha(FREEZE) or teacher.get("inputs_sha256") != sha(OUT/"INPUTS.json"):
        raise ValueError("teacher freeze binding drift")
    import certified_teacher_bounds as bounds
    previous = ctx.prec
    ctx.prec = BITS
    try:
        tbound = bounds.teacher_l1_bounds()
        result = compose(candidate, teacher, tbound)
        result["input_bindings"] = {"freeze_sha256": sha(FREEZE), "candidate_sha256": sha(CANDIDATE),
            "teacher_finite_grid_sha256": sha(TEACHER), "teacher_bounds_source_sha256": sha(bounds.__file__),
            "risk_source_sha256": sha(__file__), "nominal_source_sha256": sha(NOMINAL)}
        result["teacher_bound_readout"] = tbound
        return result
    finally:
        ctx.prec = previous


def check_candidate():
    verify_freeze()
    candidate = read(CANDIDATE)
    if candidate["freeze_sha256"] != sha(FREEZE) or candidate["nominal_source_sha256"] != sha(NOMINAL) or candidate["risk_source_sha256"] != sha(__file__) or candidate["run_start_sha256"] != sha(RUN_START):
        raise ValueError("candidate source/freeze drift")
    a, h = fixed_input()
    if rational_vector(candidate["a_exact"]) != a or rational_vector(candidate["h_exact"]) != h:
        raise ValueError("fixed candidate/direction drift")
    finite = candidate["finite_sum"]
    if finite["status"] != "completed" or finite["nodes_completed"] != 177 or finite["N"] != N or finite["step_exact"] != str(STEP) or finite["endpoint_weights"] != "full" or finite["hard_budget_seconds"] != HARD_SECONDS or finite["wall_seconds"] > HARD_SECONDS:
        raise ValueError("fixed candidate grid incomplete or changed")
    if finite["m"] != 8 or finite["label_count"] != 256 or finite["finite_scalar_integrals_completed"] != 512 or finite["real_sigmoid_evaluations"] != 177*8 or finite["real_Gaussian_density_evaluations"] != 177 or finite["label_probability_accumulations"] != 177*256 or finite["label_directional_accumulations"] != 177*256:
        raise ValueError("fixed candidate operation count drift")
    if any(finite[name] != 0 for name in ("new_kernel_calls", "new_random_draws", "new_optimizer_updates", "scientific_NPZ_reads")):
        raise ValueError("unexpected candidate research extension")
    previous = ctx.prec
    ctx.prec = BITS
    try:
        candidate_ratios(candidate)
    finally:
        ctx.prec = previous
    return candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--combine", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.run:
        answer = run()
        code = 0 if answer["status"] == "completed" else 1
    elif args.combine:
        if COMPOSED.exists():
            raise FileExistsError("risk composition already saved; no overwrite")
        result = compose_files()
        write_new(COMPOSED, result)
        answer = {"status": "completed", "true_center_risk_direction_upper_below_minus_point020": result["true_center_risk_direction_upper_below_minus_point020"], "new_integrals": 0}
        code = 0 if result["true_center_risk_direction_upper_below_minus_point020"] else 1
    else:
        check_candidate()
        if COMPOSED.exists():
            result = compose_files()
            if result != read(COMPOSED):
                raise ValueError("saved risk composition differs from JSON/Arb recomputation")
            answer = {"status": "passed", "candidate_checked": True, "teacher_composition_checked": True,
                "true_center_risk_direction_upper_below_minus_point020": result["true_center_risk_direction_upper_below_minus_point020"],
                "check_scope": "saved source/hash/JSON/Arb arithmetic; scientific target reported separately", "new_integrals": 0}
            code = 0
        else:
            answer, code = {"status": "candidate_checked_teacher_pending", "new_integrals": 0}, 0
    print(json.dumps(answer, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
