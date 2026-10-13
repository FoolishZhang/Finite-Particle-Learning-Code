"""Exact-moment entropy certificates; no optimizer or scientific RNG.

Shared J/Q pure APIs plus a frozen J runner. --self-test uses a two-variable
uniform algebra fixture. No scientific certificate generation before --freeze.
"""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "2"
import argparse
from datetime import datetime, timezone
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback

LEVELS = (100, 160)
LOG_TERMS = 96
ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/"artifacts/v3_select/T"
OUT = ROOT/"artifacts/rc_quant/J"
DOC = ROOT/"docs/08_research_研究工作/RC定量界与多块查询_20261001"


def ceil_div(a, b):
    assert b > 0
    return -((-a)//b)


def encode(x):
    x = F(x)
    return [x.numerator, x.denominator]


def decode(x):
    assert len(x) == 2 and all(isinstance(v, int) for v in x)
    assert x[1] > 0
    return F(*x)


def canonical_nodes(d, nodes):
    nodes = tuple(sorted((tuple(sorted(n)) for n in nodes), key=lambda n: (len(n), n)))
    assert len(set(nodes)) == len(nodes) and tuple(range(d)) in nodes
    for a in nodes:
        assert len(a) >= 2 and len(set(a)) == len(a) and min(a) >= 0 and max(a) < d
        for b in nodes:
            sa, sb = set(a), set(b)
            assert not sa & sb or sa <= sb or sb <= sa
    return nodes


def integer_features(d, nodes):
    """Rows are normalization, then the producer's spin/count feature order."""
    nodes = canonical_nodes(d, nodes)
    states = [tuple((x >> i) & 1 for i in range(d)) for x in range(1 << d)]
    rows = [[1]*(1 << d)]
    rows.extend([[2*c[i]-1 for c in states] for i in range(d)])
    for node in nodes:
        for k in range(2, len(node)+1):
            rows.append([int(sum(c[i] for i in node) == k) for c in states])
    return rows


def exact_teacher(d=6):
    assert d % 2 == 0
    numerators = []
    for x in range(1 << d):
        c = [(x >> i) & 1 for i in range(d)]
        value = math.prod(c[i]-c[i+1] for i in range(0, d, 2))
        numerators.append(10+9*value)
    denominator = 10*(1 << d)
    assert min(numerators) > 0 and sum(numerators) == denominator
    return [F(n, denominator) for n in numerators]


def matvec(rows, vector):
    return [sum((a*b for a, b in zip(row, vector)), F(0)) for row in rows]


@lru_cache(maxsize=64)
def _exact_basis(rows):
    import sympy as sp
    matrix = sp.Matrix(rows)
    independent_rows = tuple(matrix.T.rref()[1])
    reduced = matrix[list(independent_rows), :]
    pivot_states = tuple(reduced.rref()[1])
    square = reduced[:, list(pivot_states)]
    assert square.rows == square.cols == len(independent_rows)
    return independent_rows, pivot_states, square.inv().as_immutable()


def dual_feasible_projection(rows, target, proposal, pbits=100):
    """Positive rational r with A r=b; normalization must be a row if needed.

    Inputs A and b are exact integers/rationals. Floating proposal and RREF
    choices only construct a candidate. Every original constraint is checked.
    """
    import sympy as sp
    assert pbits in LEVELS and len(rows) == len(target) and len(rows) > 0
    assert all(len(row) == len(proposal) for row in rows)
    assert all(isinstance(v, int) for row in rows for v in row)
    target = [F(v) for v in target]
    scale = 1 << pbits
    q = [F(round(F.from_float(float(v))*scale), scale) for v in proposal]
    assert min(q) > 0
    independent_rows, pivot_states, inverse = _exact_basis(tuple(tuple(row) for row in rows))
    residual = [a-b for a, b in zip(target, matvec(rows, q))]
    rhs = sp.Matrix([sp.Rational(residual[i].numerator, residual[i].denominator)
                     for i in independent_rows])
    delta = inverse*rhs
    r = q.copy()
    for index, value in zip(pivot_states, delta):
        r[index] += F(int(value.p), int(value.q))
    assert min(r) > 0
    assert matvec(rows, r) == target
    return r, dict(independent_rows=list(independent_rows), pivot_states=list(pivot_states),
                   rank=len(independent_rows),
                   maximum_correction=encode(max(abs(a-b) for a, b in zip(r, q))),
                   minimum_probability=encode(min(r)))


def exact_moment_correction(rows, p, q_float, bits):
    r, metadata = dual_feasible_projection(rows, matvec(rows, p), q_float, bits)
    assert sum(r) == 1
    return r, metadata


def _log_unit_interval(y, bits):
    """Dyadic outward interval for log(y), 1<=y<=2, using atanh."""
    y = F(y)
    assert 1 <= y <= 2
    scale = 1 << bits
    z = (y-1)/(y+1)
    a, b = z.numerator, z.denominator
    lo, hi = (a*scale)//b, ceil_div(a*scale, b)
    total_lo = total_hi = 0
    for j in range(LOG_TERMS):
        odd = 2*j+1
        total_lo += lo//odd
        total_hi += ceil_div(hi, odd)
        lo = lo*a*a//(b*b)
        hi = ceil_div(hi*a*a, b*b)
    # At this point lo/scale <= z**(2N+1) <= hi/scale.
    tail = ceil_div(2*hi*b*b, (2*LOG_TERMS+1)*(b*b-a*a))
    return 2*total_lo, 2*total_hi+tail


@lru_cache(None)
def _ln2_interval(bits):
    return _log_unit_interval(F(2), bits)


@lru_cache(maxsize=32768)
def log_interval(x, bits):
    """Return exact Fraction endpoints containing log of positive rational x."""
    x = F(x)
    assert x > 0
    k = x.numerator.bit_length()-x.denominator.bit_length()
    power = F(1 << k) if k >= 0 else F(1, 1 << -k)
    y = x/power
    while y < 1:
        y *= 2
        k -= 1
    while y >= 2:
        y /= 2
        k += 1
    assert bits in LEVELS
    lo, hi = _log_unit_interval(y, bits)
    l2, u2 = _ln2_interval(bits)
    if k >= 0:
        lo, hi = lo+k*l2, hi+k*u2
    else:
        lo, hi = lo+k*u2, hi+k*l2
    scale = 1 << bits
    return F(lo, scale), F(hi, scale)


def rational_log_bounds(x, pbits=100):
    """Public shared J/Q API; exact Fraction endpoints, no floating log."""
    return log_interval(F(x), pbits)


@lru_cache(maxsize=4096)
def entropy_interval(probabilities, bits):
    assert min(probabilities) > 0 and sum(probabilities) == 1
    lower = upper = F(0)
    for r in probabilities:
        lo, hi = log_interval(r, bits)
        lower -= r*hi
        upper -= r*lo
    assert lower <= upper
    return lower, upper


def rational_primal(rows, d, weight_numerators, bits):
    """Weights use binary C_i unaries and the same binary count indicators."""
    assert len(weight_numerators) == len(rows)-1
    assert all(isinstance(w, int) and w > 0 for w in weight_numerators)
    scale = 1 << bits
    weights = []
    for x in range(len(rows[0])):
        value = 1
        for j, numerator in enumerate(weight_numerators):
            feature = (rows[j+1][x]+1)//2 if j < d else rows[j+1][x]
            assert feature in (0, 1)
            # All states have the common denominator scale**parameter_count.
            value *= numerator if feature else scale
        weights.append(value)
    divisor = math.gcd(*weights)
    weights = [v//divisor for v in weights]
    total = sum(weights)
    return tuple(F(v, total) for v in weights)


def primal_KL_interval(p, q, bits):
    lower = upper = F(0)
    for a, b in zip(p, q):
        lo, hi = log_interval(a/b, bits)
        lower += a*lo
        upper += a*hi
    return lower, upper


def verify_tree_certificate(certificate):
    bits = certificate["arithmetic"]["bits"]
    assert bits in LEVELS
    assert certificate["arithmetic"] == dict(bits=bits, log_terms=LOG_TERMS)
    d = certificate["d"]
    rows = integer_features(d, certificate["nodes"])
    p = tuple(decode(v) for v in certificate["target"])
    assert min(p) > 0 and sum(p) == 1
    denominator = certificate["r_denominator"]
    numerators = certificate["r_numerators"]
    assert isinstance(denominator, int) and denominator > 0
    assert len(numerators) == len(p) == 1 << d
    assert all(isinstance(x, int) and x > 0 for x in numerators)
    assert sum(numerators) == denominator
    # Full original rows, including dependent rows, are checked exactly.
    for row, moment in zip(rows, matvec(rows, p)):
        assert F(sum(a*b for a, b in zip(row, numerators)), denominator) == moment
    r = tuple(F(n, denominator) for n in numerators)
    hr = entropy_interval(r, bits)
    hp = entropy_interval(p, bits)
    dual = (hr[0]-hp[1], hr[1]-hp[0])
    q = rational_primal(rows, d, certificate["primal_weight_numerators"], bits)
    primal = primal_KL_interval(p, q, bits)
    assert [encode(v) for v in dual] == certificate["dual_entropy_difference_interval"]
    assert [encode(v) for v in primal] == certificate["primal_KL_interval"]
    assert dual[0] <= primal[1]
    return dict(exact_moments=True, positive=True, probability_mass_exact=True,
                lower_bound=dual[0], upper_bound=primal[1],
                interval_width=primal[1]-dual[0])


def certify_tree(tree_id, d, nodes, q_float, theta_float, bits=100, target=None):
    started = time.perf_counter()
    rows = integer_features(d, nodes)
    assert bits in LEVELS
    p = tuple(exact_teacher(d) if target is None else [F(x) for x in target])
    assert min(p) > 0 and sum(p) == 1 and len(p) == 1 << d
    r, basis = exact_moment_correction(rows, p, q_float, bits)
    denominator = math.lcm(*(x.denominator for x in r))
    numerators = [int(x*denominator) for x in r]
    assert len(theta_float) == len(rows)-1
    scale = 1 << bits
    # exp is an untrusted proposal only; every positive rational weight defines
    # a member of the unbounded family regardless of proposal accuracy.
    weight_numerators = [round(F.from_float(math.exp(float(v)*(2 if j < d else 1)))*scale)
                         for j, v in enumerate(theta_float)]
    assert min(weight_numerators) > 0
    q_primal = rational_primal(rows, d, weight_numerators, bits)
    hr, hp = entropy_interval(tuple(r), bits), entropy_interval(p, bits)
    dual = hr[0]-hp[1], hr[1]-hp[0]
    primal = primal_KL_interval(p, q_primal, bits)
    certificate = dict(tree_id=int(tree_id), d=d,
        nodes=[list(n) for n in canonical_nodes(d, nodes)],
        target=[encode(x) for x in p], r_denominator=denominator,
        r_numerators=numerators, primal_weight_numerators=weight_numerators,
        primal_parameterization="binary C unary; binary count indicators; weights/2**bits",
        arithmetic=dict(bits=bits, log_terms=LOG_TERMS),
        correction_basis=basis,
        dual_entropy_difference_interval=[encode(x) for x in dual],
        primal_KL_interval=[encode(x) for x in primal],
        source_q_is_initializer_only=True, constraints_are_unbounded=True,
        elapsed_generation_seconds=time.perf_counter()-started)
    verify_tree_certificate(certificate)
    return certificate


def aggregate_certificates(certificates):
    assert len({c["tree_id"] for c in certificates}) == len(certificates)
    assert len(certificates) == 945
    target = [encode(x) for x in exact_teacher(6)]
    assert all(c["d"] == 6 and c["target"] == target for c in certificates)
    assert {canonical_nodes(6, c["nodes"]) for c in certificates} == set(labelled_binary_trees(tuple(range(6))))
    checks = [verify_tree_certificate(c) for c in certificates]
    lower = min(c["lower_bound"] for c in checks)
    upper = min(c["upper_bound"] for c in checks)
    assert lower <= upper
    return dict(lower_bound=encode(lower), upper_bound=encode(upper),
                width=encode(upper-lower),
                width_le_1e_minus_8=bool(upper-lower <= F(1, 10**8)),
                exact_feasible_duals=len(checks), all_945_binary_trees_exactly_covered=True,
                target_is_exact_epsilon_9_over_10=True)


@lru_cache(None)
def labelled_binary_trees(leaves):
    if len(leaves) == 1:
        return ((),)
    first, rest = leaves[0], leaves[1:]
    answers = set()
    for code in range(1 << len(rest)):
        a = (first,)+tuple(x for i, x in enumerate(rest) if (code >> i) & 1)
        if len(a) == len(leaves):
            continue
        b = tuple(x for x in leaves if x not in a)
        for ta in labelled_binary_trees(a):
            for tb in labelled_binary_trees(b):
                answers.add(tuple(sorted((leaves,)+ta+tb, key=lambda n: (len(n), n))))
    return tuple(sorted(answers))


def self_test():
    # Algebra checks, not the d6 scientific teacher or a new optimization.
    widths = []
    for bits in LEVELS:
        for x in (F(1), F(2), F(1, 2), F(17, 13), F(1, 100)):
            lo, hi = log_interval(x, bits)
            assert lo <= hi and hi-lo < F(1, 10**25)
        assert log_interval(F(1), bits) == (F(0), F(0))
        a, b = log_interval(F(2), bits), log_interval(F(1, 2), bits)
        assert a[0]+b[1] == 0 and a[1]+b[0] == 0
        # log 2 = 2 atanh(1/3): the first term and elementary geometric
        # upper bound bracket our interval with exact rational inequalities.
        assert F(2, 3) < a[0] < a[1] < F(3, 4)
        # Independent exact-Fraction series evaluation, without the integer
        # recurrence, checks the tight outward endpoints on fixed rationals.
        for y in (F(2), F(17, 13), F(3, 2)):
            z = (y-1)/(y+1)
            partial = 2*sum((z**(2*j+1)/F(2*j+1) for j in range(200)), F(0))
            tail = 2*z**401/(401*(1-z*z))
            lo, hi = rational_log_bounds(y, bits)
            assert lo <= partial and partial+tail <= hi
        r, metadata = dual_feasible_projection([[1, 1], [1, 0], [2, 2]],
                                               [F(2), F(1, 3), F(4)],
                                               [.34, 1.66], bits)
        assert r == [F(1, 3), F(5, 3)] and metadata["rank"] == 2
        certificate = certify_tree(0, 2, [[0, 1]], [.25]*4, [0.]*3,
                                  bits=bits, target=[F(1, 4)]*4)
        result = verify_tree_certificate(certificate)
        assert result["lower_bound"] <= 0 <= result["upper_bound"]
        assert result["interval_width"] < F(1, 10**25)
        widths.append(encode(result["interval_width"]))
    return dict(passed=True, algebra_fixture_d=2, optimizer_calls=0,
                scientific_certificates=0, exact_row_count=4,
                certificate_widths=widths, predefined_precision_levels=list(LEVELS),
                rigorous_integer_series=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if replace else "x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def check_hashes(mapping):
    for name, digest in mapping.items():
        assert sha(ROOT/name) == digest, name


def old_sources_check():
    check_hashes(read_json(OLD/"FREEZE.json")["inputs"])
    check_hashes(read_json(OLD/"RESULTS_FREEZE.json")["files"])


def freeze():
    import sympy
    assert not OUT.exists(), "J directory already exists"
    old_sources_check()
    OUT.mkdir(parents=True)
    snapshot = OUT/"source_inputs/HANDOFF.md"
    snapshot.parent.mkdir()
    snapshot.write_bytes((ROOT/"HANDOFF.md").read_bytes())
    config = dict(d=6, epsilon=[9, 10], trees=945, precision_levels=list(LEVELS),
        log_terms=LOG_TERMS, global_width_gate=[1, 10**8], new_fits=0, new_observations=0,
        level_160_trigger="all 100-bit certificates valid but global width exceeds 1e-8",
        any_exception_stops=True, no_hidden_refits=True)
    write_json(OUT/"CONFIG.json", config)
    paths = [Path(__file__).resolve(), DOC/"PLAN_J.md", DOC/"CERTIFICATE_DESIGN.md",
             OUT/"CONFIG.json", snapshot]+[OLD/name for name in
             ["FREEZE.json", "RESULTS_FREEZE.json", "FITS.json", "PMFS.json",
              "TEACHERS.json", "TREE_REGISTRY.json"]]
    write_json(OUT/"FREEZE.json", dict(frozen_utc=utc_now(),
        before_scientific_certificate_generation=True,
        files={p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
        python=sys.version, sympy=sympy.__version__, platform=platform.platform(),
        external_review_files_accessed=False))
    write_json(OUT/"STATE.json", dict(status="frozen", utc=utc_now(), active_jobs=[]))
    print("J frozen", sha(OUT/"FREEZE.json"), flush=True)


def check_freeze():
    check_hashes(read_json(OUT/"FREEZE.json")["files"])
    old_sources_check()


def run():
    check_freeze()
    assert read_json(OUT/"STATE.json")["status"] == "frozen", "no silent certificate restart"
    write_json(OUT/"START.json", dict(utc=utc_now(), freeze_sha256=sha(OUT/"FREEZE.json")))
    records = read_json(OLD/"FITS.json")
    proposals = read_json(OLD/"PMFS.json")
    registry = read_json(OLD/"TREE_REGISTRY.json")
    assert len(records) == len(proposals) == len(registry) == 945
    outputs = [OUT/"START.json"]
    levels = []
    for bits in LEVELS:
        began = time.perf_counter()
        certificates = []
        write_json(OUT/"STATE.json", dict(status="generating", bits=bits, utc=utc_now(),
                                        pid=os.getpid()), replace=True)
        for index, (record, proposal, registered) in enumerate(zip(records, proposals, registry)):
            assert record["tree_id"] == registered["tree_id"] == index
            assert record["nodes"] == registered["nodes"]
            certificate = certify_tree(index, 6, record["nodes"], proposal,
                                       record["params"]["theta"], bits)
            path = OUT/f"certificates_b{bits}"/f"tree_{index:04}.json"
            write_json(path, certificate)
            outputs.append(path)
            certificates.append(certificate)
            if (index+1) % 50 == 0 or index+1 == 945:
                print("J", bits, "bits", index+1, "/945", flush=True)
        summary = aggregate_certificates(certificates)
        summary.update(bits=bits, elapsed_seconds=time.perf_counter()-began,
                       optimizer_calls=0, scientific_observations=0)
        path = OUT/f"SUMMARY_b{bits}.json"
        write_json(path, summary)
        outputs.append(path)
        levels.append(summary)
        if summary["width_le_1e_minus_8"]:
            break
    decision = dict(qualified=levels[-1]["width_le_1e_minus_8"], levels=levels,
        final_lower_bound=levels[-1]["lower_bound"], final_upper_bound=levels[-1]["upper_bound"],
        all_945_unbounded_joint=True, optimizer_calls=0, new_observations=0,
        lower_is_not_a_query_certificate=True, completed_utc=utc_now())
    write_json(OUT/"DECISION.json", decision)
    outputs.append(OUT/"DECISION.json")
    write_json(OUT/"RESULTS_FREEZE.json", dict(utc=utc_now(),
        source_freeze_sha256=sha(OUT/"FREEZE.json"),
        files={p.relative_to(ROOT).as_posix(): sha(p) for p in outputs}))
    write_json(OUT/"STATE.json", dict(status="complete", qualified=decision["qualified"],
                                   utc=utc_now(), active_jobs=[]), replace=True)
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


def check():
    check_freeze()
    assert read_json(OUT/"STATE.json")["status"] == "complete"
    seal = read_json(OUT/"RESULTS_FREEZE.json")
    assert seal["source_freeze_sha256"] == sha(OUT/"FREEZE.json")
    check_hashes(seal["files"])
    decision = read_json(OUT/"DECISION.json")
    checked = 0
    for summary in decision["levels"]:
        bits = summary["bits"]
        certificates = [read_json(OUT/f"certificates_b{bits}"/f"tree_{i:04}.json") for i in range(945)]
        actual = aggregate_certificates(certificates)
        for name, value in actual.items():
            assert summary[name] == value, name
        checked += len(certificates)
    final = decision["levels"][-1]
    assert decision["qualified"] == final["width_le_1e_minus_8"]
    assert decision["final_lower_bound"] == final["lower_bound"]
    assert decision["final_upper_bound"] == final["upper_bound"]
    print(json.dumps(dict(passed=True, exact_certificates_checked=checked,
                         qualified=decision["qualified"], new_fits=0, new_observations=0)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("self-test", "freeze", "run", "check"):
        modes.add_argument("--"+name, action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
    elif args.freeze:
        freeze()
    elif args.check:
        check()
    else:
        try:
            run()
        except Exception as error:
            if (OUT/"FREEZE.json").exists() and not (OUT/"RESULTS_FREEZE.json").exists():
                write_json(OUT/f"FAILURE_{time.time_ns()}.json",
                           dict(utc=utc_now(), error=repr(error), traceback=traceback.format_exc(),
                                original_state=read_json(OUT/"STATE.json"),
                                optimizer_calls=0, new_observations=0))
                write_json(OUT/"STATE.json", dict(status="stopped_failure", utc=utc_now(),
                                                active_jobs=[]), replace=True)
            raise


if __name__ == "__main__":
    main()
