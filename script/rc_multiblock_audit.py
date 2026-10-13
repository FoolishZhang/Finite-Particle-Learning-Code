"""Independent M replay: no producer imports, fits, or random numbers."""
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
import time
import traceback
import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"artifacts/rc_quant/M"
ERRORS = {}
DIMENSIONS = (12, 24, 48, 96)
SIZES = (2048, 8192, 32768)
METHODS = ("MOM", "MLE", "HIST", "PRODUCT")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same(name, a, b, tolerance=3e-11):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (name, a.shape, b.shape)
    assert np.isfinite(a).all() and np.isfinite(b).all(), name
    error = float(np.max(np.abs(a-b))) if a.size else 0.
    ERRORS[name] = max(ERRORS.get(name, 0.), error)
    assert error <= tolerance, (name, error, tolerance)


def hashes(mapping):
    for name, digest in mapping.items():
        assert sha(ROOT/name) == digest, name


@lru_cache(None)
def states(d):
    return ((np.arange(1 << d)[:, None] >> np.arange(d)) & 1)


def hvector(x):
    return np.prod(x[:, 0::2].astype(np.int64)-x[:, 1::2].astype(np.int64), axis=1)


def independent_discovery(x):
    """AND/intersection moment identity, independently of XOR production."""
    n, d = x.shape
    packed = np.packbits(x.T, axis=1, bitorder="big")
    columns = [int.from_bytes(row.tobytes(), "big") for row in packed]
    ones = [value.bit_count() for value in columns]
    intersections = [[0]*d for _ in range(d)]
    pair_counts = [[0]*d for _ in range(d)]
    for i in range(d):
        assert ones[i] == int(x[:, i].sum())
        for j in range(i+1, d):
            intersections[i][j] = columns[i] & columns[j]
            pair_counts[i][j] = intersections[i][j].bit_count()
    triples, nums = [], []
    for i in range(d-2):
        for j in range(i+1, d-1):
            pair = intersections[i][j]
            nij = pair_counts[i][j]
            base = 2*(ones[i]+ones[j])-4*nij-n
            for k in range(j+1, d):
                count111 = (pair & columns[k]).bit_count()
                value = 8*count111-4*(pair_counts[i][k]+pair_counts[j][k])+2*ones[k]+base
                if 160*abs(value) > 7*n:
                    triples.append((i, j, k))
                    nums.append(value)
    parent = list(range(d))
    adjacency = [set() for _ in range(d)]
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for triple in triples:
        for i, j in itertools.combinations(triple, 2):
            parent[find(i)] = find(j)
            adjacency[i].add(j)
            adjacency[j].add(i)
    groups = {}
    for i in range(d):
        groups.setdefault(find(i), []).append(i)
    components = sorted(groups.values())
    errors, recovered = [], []
    if len(components) != d//6:
        errors.append("component_count")
    if any(len(c) != 6 for c in components):
        errors.append("component_size")
    if not errors:
        expected = set()
        for comp in components:
            missing = sorted((i, j) for i, j in itertools.combinations(comp, 2) if j not in adjacency[i])
            incidences = [sum(i in edge for edge in missing) for i in comp]
            if len(missing) != 3 or incidences != [1]*6:
                errors.append("missing_edges_not_matching")
                continue
            block = tuple(i for pair in missing for i in pair)
            recovered.append(block)
            expected.update(tuple(sorted(t)) for t in itertools.product(*missing))
        if set(triples) != expected:
            errors.append("not_exact_transversal_hypergraph")
    return dict(selected_triples=[list(t) for t in triples], selected_numerators=nums,
                selected_count=len(triples), components=components,
                validation_errors=sorted(set(errors)), valid=not errors,
                recovered_blocks=[list(b) for b in sorted(recovered)] if not errors else [],
                examined_triples=math.comb(d, 3))


def audit_model_parameters(models, discovery, x1, x2):
    n, d = x1.shape
    assert set(models) == set(METHODS)
    parameter_counts = {}
    for method, model in models.items():
        assert model["D"] == d and model["n1"] == model["n2"] == n
        assert model["method"] == method
        assert model["discovery_valid"] == discovery["valid"]
        assert sorted(i for f in model["factors"] for i in f["coords"]) == list(range(d))
        for factor in model["factors"]:
            q = np.asarray(factor["pmf"])
            assert len(q) == 1 << len(factor["coords"]) and q.min() > 0
            same("factor normalization", q.sum(), 1., 1e-12)
        if method == "PRODUCT":
            assert not model["fallback"] and model["fitting_sample"] == "X1_and_X2_pooled"
            count = x1.sum(0, dtype=np.int64)+x2.sum(0, dtype=np.int64)
            for i, factor in enumerate(model["factors"]):
                assert factor["coords"] == [i] and factor["successes"] == int(count[i])
                p1 = (int(count[i])+.5)/(2*n+1)
                same("product closed parameters", factor["pmf"], [1-p1, p1], 2e-15)
            parameter_counts[method] = d
        elif not discovery["valid"]:
            assert model["fallback"] and model["fitting_sample"] == "invalid_discovery_global_uniform"
            assert len(model["factors"]) == d
            for i, factor in enumerate(model["factors"]):
                assert factor["coords"] == [i] and factor["pmf"] == [.5, .5]
            parameter_counts[method] = 0
        else:
            assert not model["fallback"] and model["fitting_sample"] == "X2_only"
            assert [f["coords"] for f in model["factors"]] == discovery["recovered_blocks"]
            for factor in model["factors"]:
                local = x2[:, factor["coords"]]
                code = sum(local[:, j].astype(np.int64)*(1 << j) for j in range(6))
                counts = np.bincount(code, minlength=64)
                hs = hvector(states(6))
                plus, minus = int(counts[hs == 1].sum()), int(counts[hs == -1].sum())
                if method == "HIST":
                    assert factor["counts"] == counts.tolist()
                    same("histogram closed parameters", factor["pmf"], (counts+1/64)/(n+1), 2e-15)
                else:
                    active = plus+minus
                    raw = 8*(plus-minus)/n if method == "MOM" else (plus-minus)/active if active else 0.
                    amplitude = min(.95, max(-.95, raw))
                    assert factor["nplus"] == plus and factor["nminus"] == minus and factor["nactive"] == active
                    same("template raw amplitude", factor["raw_epsilon"], raw, 0.)
                    same("template clipped amplitude", factor["epsilon"], amplitude, 0.)
                    same("template factor probabilities", factor["pmf"], (1+amplitude*hs)/64, 2e-15)
            parameter_counts[method] = d//6*(63 if method == "HIST" else 1)
    return parameter_counts


class TeacherEvaluation:
    """Marginalize explicit 64-state true blocks, not missing-bit substitution."""
    def __init__(self, d, blocks, eps):
        self.d, self.blocks, self.eps = d, blocks, eps
        self.block_pmfs = [(1+e*hvector(states(6)))/64 for e in eps]
        self.true_entropy = math.fsum(-float(p @ np.log(p)) for p in self.block_pmfs)
        self.true_conditionals = np.zeros(d)
        self.marginal_cache = {}
        for block, p in zip(blocks, self.block_pmfs):
            same("true block normalization", p.sum(), 1., 2e-15)
            for j, target in enumerate(block):
                conditional = p/(p+p[np.arange(64) ^ (1 << j)])
                self.true_conditionals[target] = -float(p @ np.log(conditional))
        if d == 12:
            bits = states(12)
            self.dense_p = np.ones(4096)
            for block, p in zip(blocks, self.block_pmfs):
                code = bits[:, block] @ (1 << np.arange(6))
                self.dense_p *= p[code]
            same("dense true mass", self.dense_p.sum(), 1., 3e-14)

    def marginal(self, coords):
        key = tuple(coords)
        if key in self.marginal_cache:
            return self.marginal_cache[key]
        bits = states(len(coords))
        p = np.ones(len(bits))
        for block, block_p in zip(self.blocks, self.block_pmfs):
            overlap = [v for v in coords if v in block]
            if not overlap:
                continue
            positions = [block.index(v) for v in overlap]
            true_codes = states(6)[:, positions] @ (1 << np.arange(len(overlap)))
            local_marginal = np.bincount(true_codes, weights=block_p, minlength=1 << len(overlap))
            codes = bits[:, [coords.index(v) for v in overlap]] @ (1 << np.arange(len(overlap)))
            p *= local_marginal[codes]
        same("enumerated true marginal mass", p.sum(), 1., 3e-14)
        self.marginal_cache[key] = p
        return p

    def evaluate(self, model):
        query_ce = np.zeros(self.d)
        factor_ce, total_ce = [], 0.
        dense_q = np.ones(4096) if self.d == 12 else None
        for factor in model["factors"]:
            coords = factor["coords"]
            q = np.asarray(factor["pmf"])
            p = self.marginal(coords)
            ce = -float(p @ np.log(q))
            total_ce += ce
            factor_ce.append(ce)
            # Gather two-state conditional tables by explicit evidence codes.
            bits = states(len(coords))
            for position, target in enumerate(coords):
                others = [j for j in range(len(coords)) if j != position]
                code = bits[:, others] @ (1 << np.arange(len(others))) if others else np.zeros(len(q), dtype=int)
                joint_code = 2*code+bits[:, position]
                grouped = np.bincount(joint_code, weights=q, minlength=2*(1 << len(others))).reshape(-1, 2)
                conditional = grouped/grouped.sum(1)[:, None]
                query_ce[target] = -float(p @ np.log(conditional.ravel()[joint_code]))
            if self.d == 12:
                code = states(12)[:, coords] @ (1 << np.arange(len(coords)))
                dense_q *= q[code]
        per_target = query_ce-self.true_conditionals
        risk = dict(joint_kl=total_ce-self.true_entropy, joint_cross_entropy=total_ce,
                    teacher_entropy=self.true_entropy, query_kl=float(per_target.mean()),
                    query_cross_entropy=float(query_ce.mean()),
                    teacher_query_entropy=float(self.true_conditionals.mean()),
                    per_target_query_kl=per_target, per_target_query_cross_entropy=query_ce,
                    per_target_teacher_query_entropy=self.true_conditionals,
                    factor_cross_entropies=factor_ce)
        if self.d == 12:
            same("dense model mass", dense_q.sum(), 1., 3e-14)
            direct_joint = float(self.dense_p @ np.log(self.dense_p/dense_q))
            direct_query = []
            for target in range(12):
                flip = np.arange(4096) ^ (1 << target)
                pc = self.dense_p/(self.dense_p+self.dense_p[flip])
                qc = dense_q/(dense_q+dense_q[flip])
                direct_query.append(float(self.dense_p @ np.log(pc/qc)))
            same("dense vs factor joint risk", direct_joint, risk["joint_kl"], 2e-12)
            same("dense vs factor query risk", direct_query, risk["per_target_query_kl"], 2e-12)
        assert risk["joint_kl"] >= -1e-10 and min(per_target) >= -1e-10
        return risk


def audit(workspace):
    assert workspace, "--write requires --workspace for full discovery replay"
    began = time.perf_counter()
    ERRORS.clear()
    freeze = read(OUT/"FREEZE.json")
    seal = read(OUT/"RESULTS_FREEZE.json")
    assert read(OUT/"STATE.json")["status"] == "complete"
    assert seal["source_freeze_sha256"] == sha(OUT/"FREEZE.json")
    hashes(freeze["files"])
    hashes(seal["files"])
    hashes(seal["local_arrays"])
    records = [read(p) for p in sorted((OUT/"repetitions").glob("*.json"))]
    assert len(records) == 64
    seen_cells, seen_seeds = set(), set()
    units, risk_rows = [], 0
    scans = examined = valid = exact = scalar_bits = 0
    six_factors = product_factors = 0
    param_counts = {m: 0 for m in METHODS}
    wrong_valid = 0
    actual_arrays = {}
    for rec in records:
        d, rep = rec["D"], rec["rep"]
        assert (d, rep) not in seen_cells
        seen_cells.add((d, rep))
        base = 6126100100+1000000*DIMENSIONS.index(d)+1000*rep
        expected_seeds = dict(layout=base, signs=base+1, X1=base+2, X2=base+3)
        assert rec["seeds"] == expected_seeds
        assert not seen_seeds.intersection(expected_seeds.values())
        seen_seeds.update(expected_seeds.values())
        blocks, eps = rec["blocks"], rec["epsilon"]
        assert sorted(v for b in blocks for v in b) == list(range(d))
        for index, (block, e) in enumerate(zip(blocks, eps)):
            pairs = [tuple(block[i:i+2]) for i in (0, 2, 4)]
            assert pairs == sorted(pairs) and all(i < j for i, j in pairs)
            assert abs(e) == (.75, .9)[index % 2]
        raw_path = ROOT/rec["raw_arrays"]
        assert sha(raw_path) == rec["raw_sha256"] == seal["local_arrays"][rec["raw_arrays"]]
        actual_arrays[rec["raw_arrays"]] = rec["raw_sha256"]
        with np.load(raw_path, allow_pickle=False) as archive:
            assert set(archive.files) == {"X1", "X2"}
            x1, x2 = archive["X1"], archive["X2"]
            assert x1.shape == x2.shape == (32768, d)
            assert np.isin(x1, (0, 1)).all() and np.isin(x2, (0, 1)).all()
            scalar_bits += x1.size+x2.size
            evaluation = TeacherEvaluation(d, blocks, eps)
            assert len(rec["units"]) == 3
            for unit in rec["units"]:
                n = unit["n"]
                assert n in SIZES and unit["D"] == d and unit["rep"] == rep
                assert sha(ROOT/unit["trained_file"]) == unit["trained_sha256"]
                trained = read(ROOT/unit["trained_file"])
                assert (trained["D"], trained["rep"], trained["n"]) == (d, rep, n)
                assert trained["before_teacher_risk_evaluation"] and trained["training_only_inputs"]
                discovery = independent_discovery(x1[:n])
                scans += 1
                examined += discovery.pop("examined_triples")
                for key, value in discovery.items():
                    assert trained["discovery"][key] == value, (d, rep, n, key)
                assert not trained["discovery"]["teacher_access"]
                recovered = discovery["valid"] and sorted(discovery["recovered_blocks"]) == sorted(blocks)
                assert unit["discovery_valid"] == discovery["valid"] and unit["exact_recovery"] == recovered
                assert unit["selected_triples"] == discovery["selected_count"]
                valid += discovery["valid"]
                exact += recovered
                wrong_valid += int(discovery["valid"] and not recovered)
                params = audit_model_parameters(trained["models"], discovery, x1[:n], x2[:n])
                for key, value in params.items():
                    param_counts[key] += value
                product_factors += d
                six_factors += 3*d//6 if discovery["valid"] else 0
                models = list(trained["models"].items())
                models.append(("UNIFORM", dict(factors=[dict(coords=[i], pmf=[.5, .5]) for i in range(d)])))
                independent_risks = {}
                for method, model in models:
                    actual = evaluation.evaluate(model)
                    saved = unit["risk"][method]
                    for key, value in actual.items():
                        if key == "factor_cross_entropies":
                            value_saved = [f["cross_entropy"] for f in saved[key]]
                            assert [f["coords"] for f in saved[key]] == [f["coords"] for f in model["factors"]]
                        else:
                            value_saved = saved[key]
                        same("saved risk "+key, value_saved, value, 3e-11)
                    assert saved["negative_roundoff_clipped"] is False
                    numerical = saved["normalization_audit"]
                    assert numerical["finite"]
                    assert 0 <= numerical["model_factor_max_mass_error"] <= 1e-10
                    assert 0 <= numerical["teacher_marginal_max_mass_error"] <= 1e-10
                    same("minimum model probability", numerical["minimum_model_factor_probability"],
                         min(min(f["pmf"]) for f in model["factors"]), 2e-15)
                    independent_risks[method] = actual
                    risk_rows += 1
                units.append(dict(D=d, rep=rep, n=n, valid=discovery["valid"], exact=recovered,
                                  risk=independent_risks, saved=unit))
        print("M independent replay", d, rep, "scans", scans, flush=True)
    assert seen_cells == set(itertools.product(DIMENSIONS, range(16)))
    assert actual_arrays == seal["local_arrays"]
    assert scans == 192 and risk_rows == 960
    assert examined == 16*3*sum(math.comb(d, 3) for d in DIMENSIONS)
    counts = read(OUT/"COUNTS.json")
    expected_counts = dict(teacher_realizations=64, training_vector_draws=64*2*32768,
        training_scalar_bits=scalar_bits, test_or_validation_draws=0, discovery_runs=scans,
        method_output_assignments=768, valid_discoveries=valid,
        global_uniform_fallback_outputs=3*(192-valid), product_joint_fits=192,
        learned_block_joint_fits=3*valid, total_joint_fit_endpoints=192+3*valid,
        estimated_six_bit_factors=six_factors, estimated_product_bernoulli_factors=product_factors,
        new_neural_networks=0, iterative_optimizer_calls=0)
    for key, value in expected_counts.items():
        assert counts[key] == value, key
    summary = read(OUT/"SUMMARY.json")
    assert len(summary) == 60
    for row in summary:
        selected = [u for u in units if (u["D"], u["n"]) == (row["D"], row["n1"])]
        assert len(selected) == row["repetitions"] == 16 and row["n1"] == row["n2"]
        assert row["valid_discoveries"] == sum(u["valid"] for u in selected)
        assert row["exact_structure_recoveries"] == sum(u["exact"] for u in selected)
        for key in ("query_kl", "joint_kl"):
            mean = math.fsum(u["risk"][row["method"]][key] for u in selected)/16
            same("summary "+key, row["mean_"+key], mean, 3e-11)
    primary = read(OUT/"PRIMARY.json")
    assert len(primary) == 4
    critical = float(student_t.ppf(1-.05/8, 15))
    primary_passed = 0
    for row in primary:
        selected = [u for u in units if (u["D"], u["n"]) == (row["D"], 32768)]
        values = [u["risk"]["MOM"]["query_kl"] for u in selected]
        mean = math.fsum(values)/16
        se = math.sqrt(math.fsum((v-mean)**2 for v in values)/(16*15))
        interval = [mean-critical*se, mean+critical*se]
        recovered = sum(u["exact"] for u in selected)
        assert row["analytic_RC_floor"] == [87, 56320] and row["family"] == 4
        assert row["method"] == "MOM" and row["n1"] == row["n2"] == 32768
        same("primary values", row["values"], values, 3e-12)
        same("primary mean", row["mean"], mean, 3e-12)
        same("primary SE", row["se"], se, 3e-12)
        same("primary interval", row["interval"], interval, 3e-12)
        assert row["observed_exact_recoveries"] == recovered
        passed = recovered == 16 and interval[1] < 87/56320
        assert row["passed"] == passed
        primary_passed += passed
    paths = [Path(__file__).resolve(), OUT/"FREEZE.json", OUT/"RESULTS_FREEZE.json"]
    return dict(passed=True, audit_utc=datetime.now(timezone.utc).isoformat(),
        source_hashes={p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
        workspace=True, original_arrays=actual_arrays,
        raw_datasets=64, input_vectors=64*2*32768, scalar_bits=scalar_bits,
        full_triple_scans=scans, examined_triple_prefixes=examined,
        valid_discoveries=valid, exact_recoveries=exact, wrong_valid_discoveries=wrong_valid,
        maximum_N_exact_recoveries=sum(u["exact"] for u in units if u["n"] == 32768),
        model_output_assignments_replayed=768, risk_records_recomputed=960,
        dense_D12_risk_records=16*3*5, primary_intervals=4, primary_passed=primary_passed,
        estimated_scalar_parameter_ledger=param_counts, max_errors=ERRORS,
        producer_imports=False, new_fits=0, new_observations=0,
        elapsed_seconds=time.perf_counter()-began,
        scope="full stored-array algebra replay; float64 risk arithmetic; t development intervals, not exact coverage")


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--workspace", action="store_true")
    args = parser.parse_args()
    path = OUT/"INDEPENDENT_AUDIT.json"
    if args.check:
        receipt = read(path)
        assert receipt["passed"]
        hashes(receipt["source_hashes"])
        hashes(read(OUT/"FREEZE.json")["files"])
        hashes(read(OUT/"RESULTS_FREEZE.json")["files"])
        if args.workspace:
            hashes(receipt["original_arrays"])
        print("M independent audit receipt/source/public evidence valid", flush=True)
        return
    assert not path.exists(), "append-only audit"
    try:
        receipt = audit(args.workspace)
    except Exception as error:
        receipt = dict(passed=False, error=repr(error), traceback=traceback.format_exc(),
                       max_errors=ERRORS, new_fits=0, new_observations=0)
        with path.open("x", encoding="utf-8", newline="\n") as f:
            json.dump(receipt, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
        raise
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
