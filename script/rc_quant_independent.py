"""Independent exact J/Q audit: no producer imports, optimizers, or RNG."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"artifacts/rc_quant/INDEPENDENT"
DOC = ROOT/"docs/08_research_研究工作/RC定量界与多块查询_20261001/INDEPENDENT_CERTIFICATE_REVIEW.md"
LEVELS = ((64, 96), (224, 256))
N = 6
STATES = [tuple((s >> j) & 1 for j in range(N)) for s in range(1 << N)]
PN = [10+9*(c[0]-c[1])*(c[2]-c[3])*(c[4]-c[5]) for c in STATES]
P = tuple(F(n, 640) for n in PN)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def enc(x):
    x = F(x)
    return [x.numerator, x.denominator]


def dec(x):
    assert len(x) == 2 and all(isinstance(v, int) for v in x) and x[1] > 0
    return F(*x)


def enc_interval(value):
    return [enc(x) for x in value]


def ceil_div(a, b):
    assert b > 0
    return -((-a)//b)


def unit_log_bounds(y, bits, terms):
    """ln(y), 1<=y<=3/2, using S_even <= ln <= S_previous_odd.

    u and all powers are enclosed by outward-rounded dyadic arithmetic.
    The exact alternating-series brackets remove any asymptotic tail claim.
    """
    y = F(y)
    assert 1 <= y <= F(3, 2) and terms >= 2 and terms % 2 == 0
    scale = 1 << bits
    u = y-1
    ulo = (u.numerator*scale)//u.denominator
    uhi = ceil_div(u.numerator*scale, u.denominator)
    plo, phi = ulo, uhi
    lo = hi = 0
    odd_upper = None
    for k in range(1, terms+1):
        tlo, thi = plo//k, ceil_div(phi, k)
        if k % 2:
            lo += tlo
            hi += thi
        else:
            lo -= thi
            hi -= tlo
        if k == terms-1:
            odd_upper = hi
        plo, phi = (plo*ulo)//scale, ceil_div(phi*uhi, scale)
    assert odd_upper is not None and lo <= odd_upper
    return lo, odd_upper


@lru_cache(maxsize=400000)
def logarithm(x, bits, terms):
    """Exact rational interval for ln(x), using independent base 3/2 reduction."""
    x = F(x)
    assert x > 0
    y, k = x, 0
    while y < 1:
        y *= F(3, 2)
        k -= 1
    while y >= F(3, 2):
        y *= F(2, 3)
        k += 1
    lo, hi = unit_log_bounds(y, bits, terms)
    blo, bhi = unit_log_bounds(F(3, 2), bits, terms)
    if k >= 0:
        lo, hi = lo+k*blo, hi+k*bhi
    else:
        lo, hi = lo+k*bhi, hi+k*blo
    scale = 1 << bits
    return F(lo, scale), F(hi, scale)


def log_combination(terms, level):
    combined = {}
    for coefficient, value in terms:
        coefficient, value = F(coefficient), F(value)
        if coefficient:
            assert value > 0
            combined[value] = combined.get(value, F(0))+coefficient
    lo = hi = F(0)
    for value, coefficient in combined.items():
        if not coefficient:
            continue
        a, b = logarithm(value, *level)
        if coefficient > 0:
            lo += coefficient*a
            hi += coefficient*b
        else:
            lo += coefficient*b
            hi += coefficient*a
    assert lo <= hi
    return lo, hi


def tree_key(tree):
    if isinstance(tree, int):
        return (0, tree)
    return (1, tree_key(tree[0]), tree_key(tree[1]))


def join(left, right):
    return (left, right) if tree_key(left) < tree_key(right) else (right, left)


def insert_leaf(tree, new):
    # Insert at the virtual root edge and recursively at every child edge.
    answers = {join(tree, new)}
    if not isinstance(tree, int):
        left, right = tree
        answers.update(join(t, right) for t in insert_leaf(left, new))
        answers.update(join(left, t) for t in insert_leaf(right, new))
    return answers


def hierarchy(tree):
    if isinstance(tree, int):
        return {tree}, []
    a, na = hierarchy(tree[0])
    b, nb = hierarchy(tree[1])
    union = a | b
    return union, na+nb+[tuple(sorted(union))]


def tree_inventory():
    trees = {0}
    for new in range(1, 6):
        trees = {candidate for tree in trees for candidate in insert_leaf(tree, new)}
    result = {tuple(sorted(hierarchy(t)[1], key=lambda z: (len(z), z))) for t in trees}
    assert len(result) == 945
    return result


def features(nodes):
    nodes = tuple(tuple(node) for node in nodes)
    assert nodes == tuple(sorted(nodes, key=lambda z: (len(z), z)))
    rows = []
    for c in STATES:
        row = list(c)
        for node in nodes:
            count = sum(c[i] for i in node)
            row.extend(int(count == k) for k in range(2, len(node)+1))
        rows.append(tuple(row))
    return rows


def intersects(a, b):
    return max(a[0], b[0]) <= min(a[1], b[1])


def verify_joint(c):
    assert c["d"] == 6 and tuple(dec(v) for v in c["target"]) == P
    den, nums = c["r_denominator"], c["r_numerators"]
    assert isinstance(den, int) and den > 0
    assert len(nums) == 64 and all(isinstance(v, int) and v > 0 for v in nums)
    assert sum(nums) == den
    # Check all unary/count moments, including the omitted count-0/count-1 levels.
    rows = [[c0[j] for c0 in STATES] for j in range(6)]
    for node in c["nodes"]:
        for k in range(len(node)+1):
            rows.append([int(sum(c0[j] for j in node) == k) for c0 in STATES])
    for row in rows:
        assert 640*sum(a*n for a, n in zip(row, nums)) == den*sum(a*n for a, n in zip(row, PN))
    return len(rows)+1


def joint_intervals(c, level):
    den, nums = c["r_denominator"], c["r_numerators"]
    dual_terms = [(p, p) for p in P]
    dual_terms += [(-F(n, den), F(n, den)) for n in nums]
    dual = log_combination(dual_terms, level)
    design = features(c["nodes"])
    bits = c["arithmetic"]["bits"]
    assert bits == 100
    weights = c["primal_weight_numerators"]
    assert len(weights) == len(design[0]) and all(isinstance(w, int) and w > 0 for w in weights)
    scale = 1 << bits
    unnormalized = []
    for row in design:
        value = 1
        for active, weight in zip(row, weights):
            assert active in (0, 1)
            value *= weight if active else scale
        unnormalized.append(value)
    total = sum(unnormalized)
    primal = log_combination([(p, p/F(q, total)) for p, q in zip(P, unnormalized)], level)
    assert intersects(dual, tuple(dec(v) for v in c["dual_entropy_difference_interval"]))
    assert intersects(primal, tuple(dec(v) for v in c["primal_KL_interval"]))
    assert dual[0] <= primal[1]
    return {"dual": dual, "primal": primal}


def events(nodes):
    design = features(nodes)
    output = []
    for target in range(6):
        path = sorted((set(node) for node in nodes if target in node), key=len)
        current = {target}
        sibling = []
        for parent in path:
            sibling.append(tuple(sorted(parent-current)))
            current = parent
        assert current == set(range(6))
        assert sorted(i for block in sibling for i in block) == [i for i in range(6) if i != target]
        for s0 in range(64):
            if (s0 >> target) & 1:
                continue
            s1 = s0 | (1 << target)
            row = tuple(b-a for a, b in zip(design[s0], design[s1]))
            key = (target,)+tuple(sum(STATES[s0][i] for i in block) for block in sibling)
            output.append((row, PN[s0], PN[s1], key))
    assert len(output) == 192 and sum(a+b for _, a, b, _ in output) == 3840
    return output


def verify_query(c):
    assert c["d"] == 6 and c["bits"] == 160 and c["event_denominator"] == 3840
    ev = events(c["nodes"])
    checked = 0
    for name in ("dual_eventmass", "summary_eventmass"):
        record = c[name]
        den, nums = record["denominator"], record["numerators"]
        assert isinstance(den, int) and den > 0 and len(nums) == 192
        assert all(isinstance(v, int) and 0 < v < (a+b)*den
                   for v, (_, a, b, _) in zip(nums, ev))
        for column in range(len(ev[0][0])):
            assert sum(event[0][column]*v for event, v in zip(ev, nums)) == den*sum(row[column]*b for row, _, b, _ in ev)
            checked += 1
    totals = {}
    for _, a, b, key in ev:
        n, m = totals.get(key, (0, 0))
        totals[key] = (n+b, m+a+b)
    record = c["summary_eventmass"]
    for v, (_, a, b, key) in zip(record["numerators"], ev):
        numerator, denominator = totals[key]
        assert v*denominator == record["denominator"]*(a+b)*numerator
    return checked


def event_dual(ev, record, level):
    den, nums = record["denominator"], record["numerators"]
    terms = []
    for value, (_, a, b, _) in zip(nums, ev):
        v = F(value, den)
        other = F(a+b)-v
        terms.extend(((a, a), (b, b), (-v, v), (-other, other)))
    interval = log_combination(terms, level)
    return interval[0]/3840, interval[1]/3840


def query_intervals(c, level):
    ev = events(c["nodes"])
    dual = event_dual(ev, c["dual_eventmass"], level)
    summary = event_dual(ev, c["summary_eventmass"], level)
    weights = c["primal_weight_numerators"]
    assert len(weights) == len(ev[0][0]) and all(isinstance(w, int) and w > 0 for w in weights)
    scale = 1 << c["bits"]
    terms = []
    for row, a, b, _ in ev:
        numerator = denominator = 1
        for delta, weight in zip(row, weights):
            assert delta in (-1, 0, 1)
            if delta == 1:
                numerator *= weight
                denominator *= scale
            elif delta == -1:
                numerator *= scale
                denominator *= weight
        prob = F(numerator, numerator+denominator)
        terms.extend(((a, F(a, a+b)/(1-prob)), (b, F(b, a+b)/prob)))
    raw = log_combination(terms, level)
    primal = raw[0]/3840, raw[1]/3840
    for name, interval in (("dual_risk_interval", dual), ("primal_risk_interval", primal), ("summary_CMI_interval", summary)):
        assert intersects(interval, tuple(dec(v) for v in c[name])), (c["tree_id"], name, level)
    assert dual[0] <= primal[1] and summary[0] <= primal[1]
    return {"dual": dual, "primal": primal, "summary": summary}


def encode_result(row):
    return {name: enc_interval(value) for name, value in row.items()}


def verify_inputs():
    hashes = {}
    for package in ("J", "Q"):
        folder = ROOT/"artifacts/rc_quant"/package
        for name in ("FREEZE.json", "RESULTS_FREEZE.json"):
            path = folder/name
            hashes[path.relative_to(ROOT).as_posix()] = sha(path)
            receipt = read(path)
            for relative, digest in receipt["files"].items():
                assert sha(ROOT/relative) == digest, relative
        assert read(folder/"STATE.json")["status"] == "complete"
    return hashes


def audit_class(package, certificates, producer, evaluator, verifier):
    assert len(certificates) == 945
    assert {c["tree_id"] for c in certificates} == set(range(945))
    actual = {tuple(tuple(n) for n in c["nodes"]) for c in certificates}
    assert len(actual) == 945 and actual == tree_inventory()
    exact_constraints = sum(verifier(c) for c in certificates)
    coarse = []
    for i, c in enumerate(certificates):
        coarse.append(evaluator(c, LEVELS[0]))
        if (i+1) % 100 == 0:
            print(package, "coarse", i+1, flush=True)
    upper = min(row["primal"][1] for row in coarse)
    potential = [i for i, row in enumerate(coarse) if row["dual"][0] <= upper]
    refined = {}
    for index in potential:
        refined[index] = evaluator(certificates[index], LEVELS[1])
    final = [refined.get(i, row) for i, row in enumerate(coarse)]
    lower = min(row["dual"][0] for row in final)
    upper = min(row["primal"][1] for row in final)
    assert 0 < lower <= upper and upper-lower < F(1, 10**8)
    assert intersects((lower, upper), producer)
    summary = {
        "package": package, "trees": 945, "exact_equations_checked": exact_constraints,
        "independent_tree_coverage": True,
        "levels": [list(level) for level in LEVELS],
        "refinement_rule": "coarse dual lower <= minimum coarse primal upper",
        "refined_tree_ids": [certificates[i]["tree_id"] for i in potential],
        "coarse_global_upper": enc(min(row["primal"][1] for row in coarse)),
        "global_interval": enc_interval((lower, upper)),
        "width": enc(upper-lower), "display_interval": [float(lower), float(upper)],
        "display_width": float(upper-lower),
        "producer_global_interval": enc_interval(producer),
        "intersection": enc_interval((max(lower, producer[0]), min(upper, producer[1]))),
        "lowest_dual_tree": certificates[min(range(945), key=lambda i: final[i]["dual"][0])]["tree_id"],
        "best_primal_tree": certificates[min(range(945), key=lambda i: final[i]["primal"][1])]["tree_id"],
        "all_candidate_intervals_intersect_producer": True,
        "new_fits": 0, "new_RNG": 0,
    }
    if package == "Q":
        summary_min = min(row["summary"][0] for row in final), min(row["summary"][1] for row in final)
        summary["summary_CMI_global_interval"] = enc_interval(summary_min)
        summary["summary_CMI_display_interval"] = [float(v) for v in summary_min]
    rows = [{"tree_id": c["tree_id"], "coarse": encode_result(coarse[i]),
             "refined": encode_result(refined[i]) if i in refined else None}
            for i, c in enumerate(certificates)]
    return summary, rows


def run():
    assert not OUT.exists(), "independent audit already exists; no silent replacement"
    inputs = verify_inputs()
    OUT.mkdir(parents=True)
    pre = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha(__file__), "predeclared_doc_sha256": sha(DOC),
        "input_receipts": inputs, "levels": [list(v) for v in LEVELS],
        "algorithm": "base_3_over_2_alternating_log_dyadic_outward",
        "rule": "all coarse; refine exactly candidates with lower<=coarse global upper",
        "science": {"new_fits": 0, "new_samples": 0, "new_RNG": 0},
    }
    write(OUT/"PREDECLARATION.json", pre)
    started = time.perf_counter()
    joint = [read(p) for p in sorted((ROOT/"artifacts/rc_quant/J/certificates_b100").glob("*.json"))]
    query = [read(p) for p in sorted((ROOT/"artifacts/rc_quant/Q/certificates").glob("*_p160.json"))]
    js = read(ROOT/"artifacts/rc_quant/J/SUMMARY_b100.json")
    qs = read(ROOT/"artifacts/rc_quant/Q/SUMMARY.json")["precision_levels"]["160"]
    summaries = []
    for package, certs, producer, evaluator, verifier in (
        ("J", joint, (dec(js["lower_bound"]), dec(js["upper_bound"])), joint_intervals, verify_joint),
        ("Q", query, (dec(qs["global_lower"]), dec(qs["global_upper"])), query_intervals, verify_query),
    ):
        summary, rows = audit_class(package, certs, producer, evaluator, verifier)
        write(OUT/f"{package}_ROWS.json", rows)
        write(OUT/f"{package}_SUMMARY.json", summary)
        summaries.append(summary)
    result = {"passed": True, "packages": summaries,
              "elapsed_seconds": time.perf_counter()-started,
              "producer_modules_imported": False, "new_fits": 0, "new_samples": 0,
              "new_RNG": 0, "exact_feasible_J_duals": 945,
              "exact_feasible_Q_duals": 945, "exact_feasible_Q_summary_duals": 945}
    write(OUT/"AUDIT.json", result)
    paths = list(OUT.glob("*.json"))
    write(OUT/"MANIFEST.json", {"files": {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
                              "source_sha256": sha(__file__), "input_receipts": inputs})
    print(json.dumps({"passed": True, "elapsed_seconds": result["elapsed_seconds"],
                      "summaries": summaries}, ensure_ascii=False, indent=2), flush=True)


def check():
    manifest = read(OUT/"MANIFEST.json")
    assert sha(__file__) == manifest["source_sha256"]
    assert verify_inputs() == manifest["input_receipts"]
    for name, digest in manifest["files"].items():
        assert sha(ROOT/name) == digest
    assert read(OUT/"AUDIT.json")["passed"]
    print(json.dumps({"passed": True, "mode": "read_only_source_input_output_hash_check",
                      "arithmetic_not_reexecuted": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "check"))
    mode = parser.parse_args().mode
    if mode == "check":
        check()
    else:
        try:
            run()
        except Exception as exc:
            if OUT.exists():
                write(OUT/f"FAILURE_{time.time_ns()}.json",
                      {"error": repr(exc), "traceback": traceback.format_exc()})
            raise
