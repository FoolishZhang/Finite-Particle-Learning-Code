"""Independent algebra/statistics replay of A; no core imports or fits."""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "2"
import argparse
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path
import traceback

import numpy as np
from scipy.special import logsumexp
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/v3_select/A"
METHODS = ["product", "CAR", "RC_MI", "RC_residual", "RC_gain",
           "RC_agreement", "ChowLiu", "Ising", "histogram"]
TEACHERS = ["indep", "count", "pair", "pair_count", "blindspot"]
BITS = ((np.arange(256)[:, None] >> np.arange(8)) & 1)
SPIN = 2. * BITS - 1
ERRORS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def same(name, actual, expected, tolerance=2e-10):
    a, b = np.asarray(actual), np.asarray(expected)
    assert a.shape == b.shape, (name, a.shape, b.shape)
    assert np.isfinite(a).all() and np.isfinite(b).all(), name
    err = float(np.max(np.abs(a-b))) if a.size else 0.
    ERRORS[name] = max(ERRORS.get(name, 0.), err)
    assert err <= tolerance, (name, err, tolerance)


def truth(name):
    n = BITS.sum(1)
    h = np.array([-.6, .4, -.3, .7, -.5, .2, .6, -.4])
    pair = SPIN @ h
    for (i, j), value in zip([(0, 5), (1, 6), (2, 4), (3, 7)], [.35, .5, .65, .8]):
        pair += value * SPIN[:, i] * SPIN[:, j]
    if name == "blindspot":
        q = (1 + .9*np.prod(BITS[:, ::2]-BITS[:, 1::2], axis=1))/256
    else:
        score = {"indep": BITS @ np.array([-.8, -.4, 0., .4, .8, .2, -.2, .6]),
                 "count": -.35*(n-4)**2,
                 "pair": pair, "pair_count": pair-.35*(n-4)**2}[name]
        q = np.exp(score-logsumexp(score))
    assert np.min(q) > 0
    same("teacher mass", q.sum(), 1., 2e-15)
    return q


def target_for_id(identifier, counts):
    part = identifier.split("/")
    mask = 0 if part[2] == "native" else 21
    assert part[2] in ("native", "flip21")
    order = np.arange(256) ^ mask
    if part[0] == "population":
        return truth(part[1])[order]
    assert part[0] == "finite" and len(part) == 6
    n = int(part[4])
    count = np.asarray(counts[f"{part[1]}/{int(part[3])}/train{n}"])
    return (count[order]+1/256)/(n+1)


def pair_stats(p):
    marginal = np.array([[p[BITS[:, i] == b].sum() for b in (0, 1)] for i in range(8)])
    pairs = {}
    MI = np.zeros((8, 8))
    for i, j in itertools.combinations(range(8), 2):
        joint = np.array([[p[(BITS[:, i] == a) & (BITS[:, j] == b)].sum()
                           for b in (0, 1)] for a in (0, 1)])
        pairs[(i, j)] = joint
        MI[i, j] = MI[j, i] = float(np.sum(joint*np.log(joint/(marginal[i, :, None]*marginal[j, None, :]))))
    return marginal, pairs, MI


def mst(MI):
    # Prim, independently of the producer's Kruskal implementation.
    # Exact computed-weight ties use the same lexicographic edge refinement.
    in_tree = {0}
    selected = []
    while len(in_tree) < 8:
        options = [(i, j) for i in range(8) for j in range(i+1, 8)
                   if (i in in_tree) != (j in in_tree)]
        edge = min(options, key=lambda e: (-MI[e], e))
        selected.append(edge)
        in_tree.update(edge)
    return sorted(selected)


def canonical(nodes):
    return sorted([tuple(sorted(n)) for n in nodes], key=lambda n: (len(n), n))


def rebuild(record, p):
    params = record["params"]
    family = params["family"]
    assert params["d"] == 8
    gradient = None
    if family == "histogram":
        q = np.asarray(params["q"])
        same("histogram equals target", q, p, 2e-15)
        gap, count = 0., 255
    elif family == "chow_liu":
        marginal, pairs, MI = pair_stats(p)
        same("tree node marginals", params["node_marginals"], marginal, 3e-15)
        edges = [tuple(e) for e in params["edges"]]
        assert len(set(edges)) == 7 and all(0 <= i < j < 8 for i, j in edges)
        connected = {0}
        for _ in range(8):
            for i, j in edges:
                if i in connected or j in connected:
                    connected.update((i, j))
        assert len(connected) == 8
        # Alternative summation of MI may disturb roundoff ties. Validate
        # exact stored-tree weight against an independent maximum tree.
        chosen_weight = math.fsum(MI[e] for e in edges)
        optimum_weight = math.fsum(MI[e] for e in mst(MI))
        same("tree maximum weight", chosen_weight, optimum_weight, 2e-13)
        assert len(edges) == 7
        degrees = np.zeros(8, dtype=int)
        logq = np.zeros(256)
        for edge, saved in zip(edges, params["edge_marginals"]):
            i, j = edge
            same("tree pair marginals", saved, pairs[edge], 3e-15)
            degrees[i] += 1
            degrees[j] += 1
            logq += np.log(np.asarray(saved)[BITS[:, i], BITS[:, j]])
        for i in range(8):
            logq -= (degrees[i]-1)*np.log(marginal[i, BITS[:, i]])
        q = np.exp(logq)
        same("tree normalization", q.sum(), 1., 2e-13)
        q /= q.sum()
        theoretical = -np.sum(marginal*np.log(marginal))-optimum_weight
        gap = abs(float(-p @ np.log(q))-theoretical)
        count = 15
    else:
        theta = np.asarray(params["theta"], dtype=float)
        columns = [SPIN]
        bounds = [8.]*8
        if family in ("rc", "cardinality"):
            nodes = canonical(params["nodes"])
            assert len(set(nodes)) == len(nodes) and tuple(range(8)) in nodes
            for node in nodes:
                assert len(node) >= 2 and len(set(node)) == len(node)
                assert min(node) >= 0 and max(node) < 8
                for other in nodes:
                    a, b = set(node), set(other)
                    assert not a & b or a <= b or b <= a
                n = BITS[:, list(node)].sum(1)
                columns.extend([(n == k).astype(float) for k in range(2, len(node)+1)])
                bounds.extend([64.]*(len(node)-1))
            assert [list(n) for n in nodes] == params["nodes"]
            offset = 8
            same("RC unary copy", params["spin_unary"], theta[:8], 0.)
            for node, values in zip(nodes, params["node_potentials"]):
                same("RC potential copy", values, [0., 0.]+theta[offset:offset+len(node)-1].tolist(), 0.)
                offset += len(node)-1
        elif family == "ising":
            columns.extend([SPIN[:, i]*SPIN[:, j] for i, j in itertools.combinations(range(8), 2)])
            bounds += [8.]*28
        else:
            assert family == "product"
            expected = np.clip(.5*np.log((p @ BITS)/(p @ (1-BITS))), -8, 8)
            same("product closed optimum", theta, expected, 2e-13)
        F = np.column_stack(columns)
        bound = np.asarray(bounds)
        assert len(theta) == len(bound)
        assert np.all(np.abs(theta) <= bound)
        logq = F @ theta
        logq -= logsumexp(logq)
        q = np.exp(logq)
        gradient = q @ F-p @ F
        # Support-function expression, independent of producer's branch form.
        gap = float(gradient @ theta+np.abs(gradient) @ bound)
        if "gradient" in record["receipt"]:
            same("saved gradient", record["receipt"]["gradient"], gradient, 2e-13)
        if "feature_sha256" in params:
            assert hashlib.sha256(np.ascontiguousarray(F).tobytes()).hexdigest() == params["feature_sha256"]
            same("saved lower box", params["lower_bounds"], -bound, 0.)
            same("saved upper box", params["upper_bounds"], bound, 0.)
        count = len(theta)
    assert np.isfinite(q).all() and np.min(q) > 0
    same("PMF mass", q.sum(), 1., 2e-12)
    objective = -float(p @ np.log(q))
    same("training objective", objective, record["objective"], 3e-11)
    same("support gap", gap, record["gap"], 3e-10)
    assert count == record["parameter_count"]
    assert record["qualified"] == (record["gap"] <= 1e-8)
    # The comparison tolerance is not an enlarged scientific gate.
    # If the two finite-arithmetic evaluations cross the gate, stop explicitly.
    assert record["qualified"] == (gap <= 1e-8), ("independent gap qualification", record["id"], gap)
    return q


@lru_cache(None)
def all_matchings(items=tuple(range(8))):
    if not items:
        return ((),)
    result = []
    for j in items[1:]:
        remain = tuple(i for i in items if i not in (items[0], j))
        result.extend((((items[0], j),)+rest) for rest in all_matchings(remain))
    return tuple(sorted(result))


def weights(p, q, kind):
    marginal, pairs, MI = pair_stats(p)
    if kind == "mi":
        return MI
    qpairs = pair_stats(q)[1]
    result = np.zeros((8, 8))
    for edge, joint in pairs.items():
        equal = float(np.trace(joint))
        eq_q = float(np.trace(qpairs[edge]))
        if kind == "residual":
            value = abs(2*(equal-eq_q))
        else:
            value = equal*np.log(equal/eq_q)+(1-equal)*np.log((1-equal)/(1-eq_q))
        result[edge] = result[edge[::-1]] = value
    return result


def agreement_nodes(p):
    pairs = pair_stats(p)[1]
    matrix = np.eye(8)
    for edge, joint in pairs.items():
        matrix[edge] = matrix[edge[::-1]] = np.trace(joint)
    clusters = [(i,) for i in range(8)]
    nodes = []
    while len(clusters) > 1:
        choices = []
        for a, b in itertools.combinations(sorted(clusters), 2):
            value = math.fsum(float(matrix[i, j]) for i in a for j in b)/(len(a)*len(b))
            choices.append((a, b, value))
        best = max(row[2] for row in choices)
        a, b, _ = next(row for row in choices if row[2] >= best-1e-12)
        clusters.remove(a)
        clusters.remove(b)
        merged = tuple(sorted(a+b))
        nodes.append(merged)
        clusters.append(merged)
    return canonical(nodes)


def check_structures(p, traces, q_by_id, fit_by_id):
    qcar = q_by_id[traces["CAR"]["model_id"]]
    assert canonical(fit_by_id[traces["CAR"]["model_id"]]["params"]["nodes"]) == [tuple(range(8))]
    for method, kind in [("RC_MI", "mi"), ("RC_residual", "residual"), ("RC_gain", "gain")]:
        expected = weights(p, qcar, kind)
        same("structure weight", traces[method]["weights"], expected, 2e-13)
        candidates = all_matchings()
        values = [math.fsum(float(expected[e]) for e in edges) for edges in candidates]
        best = max(values)
        selected = next(edges for edges, value in zip(candidates, values) if value >= best-1e-12)
        assert canonical(traces[method]["nodes"]) == canonical([tuple(range(8))]+list(selected)), method
        assert canonical(fit_by_id[traces[method]["model_id"]]["params"]["nodes"]) == canonical(traces[method]["nodes"])
    assert canonical(traces["RC_agreement"]["nodes"]) == agreement_nodes(p)
    assert canonical(fit_by_id[traces["RC_agreement"]["model_id"]]["params"]["nodes"]) == agreement_nodes(p)
    car_qualified = fit_by_id[traces["CAR"]["model_id"]]["qualified"]
    for method in METHODS:
        model_qualified = fit_by_id[traces[method]["model_id"]]["qualified"]
        expected = model_qualified and (car_qualified if method in ("RC_residual", "RC_gain") else True)
        assert traces[method]["pipeline_qualified"] == expected


@lru_cache(None)
def channels(k):
    # Explicit index pairs for all complete evidence configurations.
    result = []
    for observed in itertools.combinations(range(8), k):
        for target in range(8):
            if target in observed:
                continue
            code = BITS[:, observed] @ (1 << np.arange(k))
            for e in range(1 << k):
                result.append((np.flatnonzero((code == e) & (BITS[:, target] == 0)),
                               np.flatnonzero((code == e) & (BITS[:, target] == 1))))
    return result


def score_table(q, k):
    if k == 7:
        # Each observed pattern has exactly two states differing in target.
        result = np.zeros(256)
        for target in range(8):
            result += np.log(q+q[np.arange(256) ^ (1 << target)])-np.log(q)
        return result/8
    result = np.zeros(256)
    for zero, one in channels(k):
        mass0, mass1 = float(q[zero].sum()), float(q[one].sum())
        total = mass0+mass1
        result[zero] -= math.log(mass0/total)
        result[one] -= math.log(mass1/total)
    return result/(math.comb(8, k)*(8-k))


def risks(p, q, tables, true_tables):
    return {"joint_KL": float(p @ np.log(p/q)),
            "query7": float(p @ (tables[7]-true_tables[7])),
            "query3": float(p @ (tables[3]-true_tables[3]))}


def audit(workspace=False):
    ERRORS.clear()
    freeze = read(OUT/"FREEZE.json")
    for name, digest in freeze["inputs"].items():
        assert sha(ROOT/name) == digest, name
    fits = read(OUT/"FITS.json")
    counts = read(OUT/"COUNTS.json")
    results = read(OUT/"RESULTS.json")
    assert results["freeze_sha256"] == sha(OUT/"FREEZE.json")
    expected_count_keys = set()
    for teacher in TEACHERS:
        for rep in range(16):
            for tag, n in [("train512", 512), ("train8192", 8192), ("val128", 128), ("val512", 512)]:
                key = f"{teacher}/{rep}/{tag}"
                expected_count_keys.add(key)
                h = np.asarray(counts[key])
                assert h.shape == (256,) and np.all(h >= 0) and np.all(h == h.astype(np.int64))
                assert int(h.sum()) == n
            assert np.all(np.asarray(counts[f"{teacher}/{rep}/train8192"]) >= counts[f"{teacher}/{rep}/train512"])
            assert np.all(np.asarray(counts[f"{teacher}/{rep}/val512"]) >= counts[f"{teacher}/{rep}/val128"])
    assert set(counts) == expected_count_keys
    fit_by_id = {row["id"]: row for row in fits}
    assert len(fit_by_id) == len(fits)
    q_by_id, tables_by_id = {}, {}
    for number, record in enumerate(fits):
        p = target_for_id(record["id"], counts)
        q = rebuild(record, p)
        q_by_id[record["id"]] = q
        tables_by_id[record["id"]] = {k: score_table(q, k) for k in (3, 7)}
        if number % 200 == 0:
            print("audit reconstructed", number, "/", len(fits), flush=True)
    assert len(results["population"]) == 10 and len(results["units"]) == 320
    used_ids = set()
    scientific_risk_rows = 0
    choices = 0
    for unit in results["population"]+results["units"]:
        teacher, encoding = unit["teacher"], unit["encoding"]
        order = np.arange(256) ^ (0 if encoding == "native" else 21)
        p = truth(teacher)[order]
        true_tables = {k: score_table(p, k) for k in (3, 7)}
        trace = unit["structure"]
        first_id = trace["CAR"]["model_id"]
        train_p = target_for_id(first_id, counts)
        check_structures(train_p, trace, q_by_id, fit_by_id)
        for method in METHODS:
            identifier = trace[method]["model_id"]
            used_ids.add(identifier)
            row = unit["candidates"][method]
            assert row["model_id"] == identifier
            assert row["qualified"] == trace[method]["pipeline_qualified"]
            risk = risks(p, q_by_id[identifier], tables_by_id[identifier], true_tables)
            for key, value in risk.items():
                same("teacher risk "+key, row[key], value, 3e-11)
            scientific_risk_rows += 1
        if "rep" not in unit:
            continue
        assert unit["unique_fits"] == len({trace[m]["model_id"] for m in METHODS})
        unique = {trace[m]["model_id"] for m in METHODS}
        same("bank fit time sum", unit["bank_fit_seconds"], math.fsum(fit_by_id[i]["fit_seconds"] for i in unique), 1e-8)
        for selection in unit["selections"]:
            nv = selection["nval"]
            val = np.asarray(counts[f"{teacher}/{unit['rep']}/val{nv}"])[order]/nv
            scores = {}
            for method in METHODS:
                identifier = trace[method]["model_id"]
                scores[method] = {"joint": -float(val @ np.log(q_by_id[identifier])),
                                  "query": float(val @ tables_by_id[identifier][7])}
                for key in ("joint", "query"):
                    same("validation score", selection["scores"][method][key], scores[method][key], 3e-11)
            for policy, field in [("joint_holdout", "joint"), ("query_holdout", "query")]:
                minimum = min(scores[m][field] for m in METHODS)
                tied = [m for m in METHODS if scores[m][field] <= minimum+1e-12]
                chosen = min(tied, key=lambda m: (fit_by_id[trace[m]["model_id"]]["parameter_count"], METHODS.index(m)))
                assert selection[policy]["selected"] == chosen
                assert selection[policy]["qualified"] == all(trace[m]["pipeline_qualified"] for m in METHODS)
            residual = np.max(np.abs((SPIN.T @ (val[:, None]*SPIN)
                                     -SPIN.T @ (q_by_id[trace["CAR"]["model_id"]][:, None]*SPIN))[np.triu_indices(8, 1)]))
            threshold = math.sqrt(2*math.log(2*28*8/.05)/nv)
            alarm = bool(residual > threshold)
            expected = "RC_gain" if alarm else "CAR"
            saved = selection["moment_alarm"]
            same("moment residual", saved["max_pair_discrepancy"], residual, 3e-13)
            same("moment threshold", saved["threshold"], threshold, 1e-15)
            assert saved["alarm"] == alarm and saved["selected"] == expected
            assert saved["qualified"] == (trace["CAR"]["pipeline_qualified"] and trace[expected]["pipeline_qualified"])
            for policy in ("joint_holdout", "query_holdout", "moment_alarm"):
                saved_risk = selection[policy]["risk"]
                chosen = selection[policy]["selected"]
                assert saved_risk == unit["candidates"][chosen]
                choices += 1
    assert used_ids == set(fit_by_id)
    assert results["unique_model_fits"] == len(fits)
    assert results["candidate_assignments"] == 2970
    assert scientific_risk_rows == 2970 and choices == 1920
    assert results["qualified_model_fits"] == sum(r["qualified"] for r in fits)
    assert results["failed_model_fits"] == [r["id"] for r in fits if not r["qualified"]]
    assert results["training_observations"] == 655360 and results["validation_observations"] == 40960
    assert len(results["contrasts"]) == 40
    critical = float(student_t.ppf(1-.05/80, 15))
    for contrast in results["contrasts"]:
        rows = [u for u in results["units"] if (u["teacher"], u["encoding"], u["ntrain"])
                == (contrast["teacher"], contrast["encoding"], 8192)]
        assert len(rows) == 16
        values, qualified = [], True
        for unit in rows:
            candidates = unit["candidates"]
            selection = next(s for s in unit["selections"] if s["nval"] == 512)
            name = contrast["contrast"]
            if name == "MI_minus_residual":
                a, b = candidates["RC_MI"], candidates["RC_residual"]
            elif name == "gain_minus_residual":
                a, b = candidates["RC_gain"], candidates["RC_residual"]
            elif name == "joint_minus_query_holdout":
                a, b = selection["joint_holdout"], selection["query_holdout"]
            else:
                assert name == "alarm_minus_query_holdout"
                a, b = selection["moment_alarm"], selection["query_holdout"]
            values.append(a.get("risk", a)["query7"]-b.get("risk", b)["query7"])
            qualified = qualified and a["qualified"] and b["qualified"]
        mean = math.fsum(values)/16
        se = math.sqrt(math.fsum((v-mean)**2 for v in values)/(15*16))
        interval = [mean-critical*se, mean+critical*se]
        same("contrast values", contrast["values"], values, 0.)
        same("contrast mean", contrast["mean"], mean, 2e-15)
        same("contrast SE", contrast["se"], se, 2e-15)
        same("contrast CI", contrast["interval"], interval, 3e-15)
        assert contrast["qualified"] == qualified
        assert contrast["practical_positive"] == bool(qualified and interval[0] > .001)
    local_pmfs = 0
    if workspace:
        assert sha(OUT/"PMFS.npz") == results["array_sha256"]
        with np.load(OUT/"PMFS.npz", allow_pickle=False) as arrays:
            expected = {i for i in fit_by_id if i.startswith("finite/")}
            assert set(arrays.files) == expected
            for identifier in arrays.files:
                same("local PMF reconstruction", arrays[identifier], q_by_id[identifier], 3e-13)
                local_pmfs += 1
    sources = {p.relative_to(ROOT).as_posix(): sha(p) for p in
               [Path(__file__), OUT/"FREEZE.json", OUT/"FITS.json", OUT/"COUNTS.json", OUT/"RESULTS.json"]}
    return dict(passed=True, audited_utc=datetime.now(timezone.utc).isoformat(),
                source_sha256=sources, workspace=workspace, local_pmfs_checked=local_pmfs,
                unique_model_fits_reconstructed=len(fits), candidate_risk_rows=scientific_risk_rows,
                holdout_and_alarm_choices=choices, t_intervals=40, family=40,
                public_integer_histograms=len(counts), new_fits=0, new_observations=0,
                independent_of_producer_core=True, max_errors=ERRORS,
                floating_gap_comparison_tolerance=3e-10,
                scope="algebra/statistics replay; not interval arithmetic, not independent scientific data")


def main():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    parser.add_argument("--workspace", action="store_true")
    args = parser.parse_args()
    output = OUT/"INDEPENDENT_AUDIT.json"
    if args.check:
        receipt = read(output)
        assert receipt["passed"]
        for name, digest in receipt["source_sha256"].items():
            assert sha(ROOT/name) == digest, name
        freeze = read(OUT/"FREEZE.json")
        for name, digest in freeze["inputs"].items():
            assert sha(ROOT/name) == digest, name
        if args.workspace:
            assert receipt["workspace"], "original audit did not replay local arrays"
            assert sha(OUT/"PMFS.npz") == read(OUT/"RESULTS.json")["array_sha256"]
        print("A independent audit receipt and frozen inputs valid", flush=True)
        return
    assert not output.exists(), "audit is append-only"
    try:
        result = audit(args.workspace)
    except Exception as error:
        result = dict(passed=False, error=repr(error), traceback=traceback.format_exc(),
                      audited_utc=datetime.now(timezone.utc).isoformat(),
                      max_errors=ERRORS, new_fits=0, new_observations=0)
        with output.open("x", encoding="utf-8", newline="\n") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
        raise
    with output.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
