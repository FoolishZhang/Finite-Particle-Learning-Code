"""Small manufactured tests only: never loads project centers or scientific data."""
import ast
from fractions import Fraction as F
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import flint
from flint import arb, ctx

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "script/angular_witness_integrals.py"
spec = importlib.util.spec_from_file_location("pair_under_test", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def check(condition, name):
    if not condition:
        raise AssertionError(name)
    return name


def independent_sv(a, u, h, n):
    """Toy-grid independent expansion (S*V-U^2)/sum(u^2), direct bit products."""
    out = [arb(0) for _ in range(256)]
    aa = [module.aq(x) for x in a]
    norm = sum(x*x for x in u)
    for k in range(-n, n + 1):
        z = module.aq(h * k)
        pp = [(-x*z).exp() for x in aa]
        pp = [1/(1+p) for p in pp]
        w = module.aq(h)*(-z*z/2).exp()/(2*arb.pi()).sqrt()
        for label in range(256):
            yy = [(label >> (7-i)) & 1 for i in range(8)]
            rr = [y-p for y, p in zip(yy, pp)]
            product = arb(1)
            for y, p in zip(yy, pp):
                product *= p if y else 1-p
            s = sum((x*r for x, r in zip(aa, rr)), arb(0))
            v = sum((x*(ui*ui)*r for x, ui, r in zip(aa, u, rr)), arb(0))
            unit_u = sum((x*ui*r for x, ui, r in zip(aa, u, rr)), arb(0))
            out[label] += w*product*(s*v-unit_u*unit_u)/norm
    return out


checks = []
start_prec = ctx.prec
a = [F(1, 5), F(-1, 4), F(1, 3), F(-1, 2), F(2, 3), F(0), F(3, 4), F(-1)]
u = [3, -2, 1, -1, 0, 2, -3, 0]
toy = module.finite_pair_sum(a, u, step="3/7", N=2)
checks.append(check(ctx.prec == start_prec, "precision_restored_after_finite_sum"))
checks.append(check(toy["status"] == "completed" and toy["nodes_completed"] == 5,
                    "five_toy_nodes_completed"))
checks.append(check(toy["real_sigmoid_evaluations"] == 40
                    and toy["pair_terms_evaluated_including_zero_coefficients"] == 5*256*28,
                    "operation_counts"))
ctx.prec = 192
try:
    sv = independent_sv(a, u, F(3, 7), 2)
    values = [module.from_record(x) for x in toy["N_finite_sum"]]
    checks.append(check(all(x.overlaps(y) for x, y in zip(values, sv)),
                        "all_256_pairs_overlap_independent_sv_expansion"))
    checks.append(check(all(abs(x-y) < module.aq("1e-48") for x, y in zip(values, sv)),
                        "all_256_independent_difference_below_1e_minus48"))
    checks.append(check(all(values[y].overlaps(values[255-y]) for y in range(256)),
                        "zero_bias_complement_symmetry"))
    checks.append(check(sum(values, arb(0)).contains(0), "all_label_pair_sum_zero"))
    r = module.record(module.aq(F(7, 3)))
    checks.append(check(module.from_record(r).contains(module.aq(F(7, 3))), "dyadic_codec_round_trip"))
    wider = module.enlarge(module.aq(F(7, 3)), module.aq(F(1, 10)))
    checks.append(check(wider.contains(module.aq(F(7, 3)-F(1, 10)))
                        and wider.contains(module.aq(F(7, 3)+F(1, 10))), "symmetric_error_enlargement"))
finally:
    ctx.prec = start_prec

scaled = module.finite_pair_sum(a, [7*x for x in u], step="3/7", N=2)
reverse = module.finite_pair_sum(a[::-1], u[::-1], step="3/7", N=2)
ctx.prec = 192
try:
    checks.append(check(all(module.from_record(x).overlaps(module.from_record(y))
                           for x, y in zip(toy["N_finite_sum"], scaled["N_finite_sum"])),
                        "integer_direction_rescaling_invariance"))
    checks.append(check(all(module.from_record(toy["N_finite_sum"][y]).overlaps(
                              module.from_record(reverse["N_finite_sum"][int(f'{y:08b}'[::-1], 2)]))
                           for y in range(256)), "msb_order_under_index_reversal"))
finally:
    ctx.prec = start_prec
zero = module.finite_pair_sum([0]*8, u, step="1/2", N=1)
checks.append(check(all(module.endpoint_fraction(v["lower_exact_dyadic"]) == 0
                        and module.endpoint_fraction(v["upper_exact_dyadic"]) == 0
                        for v in zero["N_finite_sum"]), "zero_loading_exact_zero"))
stopped = module.finite_pair_sum(a, u, step="1/2", N=1, hard_seconds=1e-300)
checks.append(check(stopped["status"] == "budget_stopped"
                    and stopped["nodes_completed"] == 0
                    and stopped["finite_scalar_integrals_completed"] == 0,
                    "budget_stop_retains_zero_prefix_without_retry"))

invalid = [
    ("zero_direction", lambda: module.finite_pair_sum(a, [0]*8, N=1)),
    ("common_rotation", lambda: module.finite_pair_sum(a, [1]*8, N=1)),
    ("noninteger_direction", lambda: module.finite_pair_sum(a, [F(x, 2) for x in u], N=1)),
    ("float_loading", lambda: module.finite_pair_sum([0.0]+a[1:], u, N=1)),
    ("float_step", lambda: module.finite_pair_sum(a, u, step=0.125, N=1)),
    ("loading_envelope", lambda: module.finite_pair_sum([2]+a[1:], u, N=1)),
    ("wrong_shape", lambda: module.finite_pair_sum(a[:-1], u, N=1)),
    ("boolean_N", lambda: module.finite_pair_sum(a, u, N=True)),
    ("invalid_bits", lambda: module.finite_pair_sum(a, u, N=1, bits=31)),
    ("nonpositive_step", lambda: module.pair_bounds(a, u, step="0")),
    ("strip_outside_proved_range", lambda: module.pair_bounds(a, u, strip="2")),
    ("nonfinite_clock_budget", lambda: module.finite_pair_sum(a, u, N=1, hard_seconds=float("inf"))),
    ("float_endpoint", lambda: module.endpoint_fraction({"mantissa":1.25,"exponent_of_two":0})),
    ("inverted_record", lambda: module.from_record({"lower_exact_dyadic":{"mantissa":"2","exponent_of_two":0},"upper_exact_dyadic":{"mantissa":"1","exponent_of_two":0}})),
]
for name, call in invalid:
    try:
        call()
    except ValueError:
        checks.append("rejected_" + name)
    else:
        raise AssertionError("invalid input accepted: " + name)
checks.append(check(ctx.prec == start_prec, "precision_restored_after_all_calls"))
bounds = module.pair_bounds(a, u)
checks.append(check(ctx.prec == start_prec, "precision_restored_after_analytic_bounds"))
ctx.prec = 192
try:
    checks.append(check(module.from_record(bounds["errors"]["N_alias"]) > 0
                        and module.from_record(bounds["errors"]["N_grid_tail"]) > 0,
                        "positive_analytic_error_enclosures"))
    checks.append(check(module.from_record(bounds["errors"]["N_total"]) < module.aq("3e-20"),
                        "default_analytic_error_below_3e_minus20"))
finally:
    ctx.prec = start_prec
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
checks.append(check(set(imports) <= {"__future__", "fractions", "math", "time", "flint"},
                    "no_project_or_array_imports"))
answer = {
    "status": "passed", "scope": "manufactured eight-variable inputs only, never project center",
    "python": sys.executable, "python_flint_version": flint.__version__,
    "flint_path": flint.__file__, "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "checks": checks, "check_count": len(checks), "pair_function_toy_completed_nodes_total": 18,
    "independent_sv_toy_nodes": 5, "all_manufactured_node_visits": 23,
    "project_nodes_evaluated": 0, "scientific_array_reads": 0,
    "new_random_draws": 0, "optimizer_calls": 0, "F8_calls": 0,
    "analytic_errors_default": bounds["errors"],
}
print(json.dumps(answer, ensure_ascii=False, indent=2))
