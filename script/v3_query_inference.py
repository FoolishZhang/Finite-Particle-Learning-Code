"""C: same-model conditional queries for a fixed count-state autoregression.

Classical forward/backward DP and likelihood-weighted ancestral sampling.
No model training, hidden teacher count, probability floor, or external core.
"""
from __future__ import annotations

import os
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.special import logsumexp
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/v3_query"
OUT = BASE / "C"
SEAL = BASE / "C_FREEZE.json"
PLAN = ROOT / "docs/08_research_研究工作/v3表示与条件查询_20260930/PLAN_C.md"
CFG = dict(dimensions=[8, 32, 64], evidence_fractions=[[1, 4], [3, 4]],
           layouts=["prefix", "mixed"], streams=16, queries_per_stream=32,
           particles=[16, 64, 256], seed_base=3026093000,
           seed_cell_stride=100000, seed_stream_stride=1000,
           seed_query_stride=10, seed_offsets=dict(mask=0, teacher=1, IS=2),
           alpha=.05, interval_family=36, probability_gate=2e-11,
           log_evidence_gate=2e-10, arithmetic_gate=2e-10,
           model="theta_i(k)=0.8 sin(0.7(i+1))+0.9 tanh((2k-i)/sqrt(i+1))+1.25 (-1)^k",
           zero_based_i=True, no_probability_floor=True, fitted_models=0,
           teacher_iid_draws=6144, independent_IS_banks=6144,
           unique_particle_trajectories=1572864, executed_prefix_trajectories=2064384,
           main_metric="scalar conditional excess Brier = (p_IS-p_exact)^2")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write(path, value, replace=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if replace else "x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=serial, allow_nan=False)
        stream.write("\n")


def csvwrite(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def hashes(paths):
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def verify_hashes(mapping):
    for name, expected in mapping.items():
        assert sha(ROOT/name) == expected, "frozen file changed: "+name


def cells():
    out = []
    for d in CFG["dimensions"]:
        for numerator, denominator in CFG["evidence_fractions"]:
            for layout in CFG["layouts"]:
                out.append(dict(cell=len(out), d=d, observed=d*numerator//denominator,
                                layout=layout))
    return out


def seeds(cell, stream, query):
    base = (CFG["seed_base"]+CFG["seed_cell_stride"]*cell+
            CFG["seed_stream_stride"]*stream+CFG["seed_query_stride"]*query)
    return {name: base+offset for name, offset in CFG["seed_offsets"].items()}


def model_table(d):
    lp, lq = np.full((d, d+1), -np.inf), np.full((d, d+1), -np.inf)
    for i in range(d):
        count = np.arange(i+1)
        theta = (.8*np.sin(.7*(i+1))+.9*np.tanh((2*count-i)/math.sqrt(i+1))+
                 1.25*np.where(count % 2 == 0, 1., -1.))
        lp[i, :i+1] = -np.logaddexp(0., -theta)
        lq[i, :i+1] = -np.logaddexp(0., theta)
    return dict(d=d, lp=lp, lq=lq, probability=np.exp(lp))


def validate_query(evidence, target):
    evidence = np.asarray(evidence, dtype=np.int8)
    assert evidence.ndim == 1 and np.isin(evidence, [-1, 0, 1]).all()
    assert 0 <= target < len(evidence) and evidence[target] == -1
    return evidence


def exact_query(table, evidence, target):
    """Log-domain count-state forward/backward; terminal count is unrestricted."""
    evidence = validate_query(evidence, target)
    d, lp, lq = table["d"], table["lp"], table["lq"]
    assert len(evidence) == d
    forward = np.full((d+1, d+1), -np.inf)
    backward = np.full((d+1, d+1), -np.inf)
    forward[0, 0] = 0.
    backward[d, :d+1] = 0.  # Sum over every final count; no hidden N is supplied.
    terms = 0
    for i in range(d):
        row = forward[i, :i+1]
        if evidence[i] == 0:
            forward[i+1, :i+1] = row+lq[i, :i+1]
        elif evidence[i] == 1:
            forward[i+1, 1:i+2] = row+lp[i, :i+1]
        else:
            forward[i+1, :i+1] = row+lq[i, :i+1]
            forward[i+1, 1:i+2] = np.logaddexp(forward[i+1, 1:i+2], row+lp[i, :i+1])
        terms += (i+1)*(2 if evidence[i] == -1 else 1)
    for i in range(d-1, -1, -1):
        zero = lq[i, :i+1]+backward[i+1, :i+1]
        one = lp[i, :i+1]+backward[i+1, 1:i+2]
        backward[i, :i+1] = zero if evidence[i] == 0 else one if evidence[i] == 1 else np.logaddexp(zero, one)
    logz = float(backward[0, 0])
    other_logz = float(logsumexp(forward[d, :d+1]))
    j = target
    logone = float(logsumexp(forward[j, :j+1]+lp[j, :j+1]+backward[j+1, 1:j+2]))
    logzero = float(logsumexp(forward[j, :j+1]+lq[j, :j+1]+backward[j+1, :j+1]))
    p = math.exp(logone-logz)
    p0 = math.exp(logzero-logz)
    assert math.isfinite(logz) and 0 <= p <= 1
    assert abs(p+p0-1) <= CFG["probability_gate"]
    assert abs(logz-other_logz) <= CFG["log_evidence_gate"]
    return dict(probability=p, log_evidence=logz,
                forward_backward_log_error=abs(logz-other_logz),
                probability_mass_error=abs(p+p0-1),
                transition_terms=terms+d*(d+1)+2*(j+1))


def prefix_direct(table, evidence, target):
    """First unobserved concept after a fully observed prefix: one AR transition."""
    evidence = validate_query(evidence, target)
    assert np.all(evidence[:target] >= 0) and np.all(evidence[target:] == -1)
    return float(table["probability"][target, int(evidence[:target].sum())])


def enumerate_query(table, evidence, target, max_free=16):
    """Independent full-completion summation, not count-state message passing."""
    evidence = validate_query(evidence, target)
    free = np.flatnonzero(evidence == -1)
    assert len(free) <= max_free
    codes = np.arange(1 << len(free), dtype=np.uint64)
    bit_position = {int(i): j for j, i in enumerate(free)}
    count = np.zeros(len(codes), dtype=int)
    logprob = np.zeros(len(codes))
    target_bits = ((codes >> np.uint64(bit_position[target])) & np.uint64(1)).astype(bool)
    for i in range(table["d"]):
        bit = (np.full(len(codes), int(evidence[i]), dtype=int) if evidence[i] >= 0 else
               ((codes >> np.uint64(bit_position[i])) & np.uint64(1)).astype(int))
        logprob += np.where(bit == 1, table["lp"][i, count], table["lq"][i, count])
        count += bit
    logz = float(logsumexp(logprob))
    return math.exp(float(logsumexp(logprob[target_bits]))-logz), logz, len(codes)


def draw_teacher(table, rng):
    """Simulation service. Its unobserved values/count are never inference inputs."""
    count = 0
    values = np.zeros(table["d"], dtype=np.int8)
    for i in range(table["d"]):
        values[i] = int(rng.random() < table["probability"][i, count])
        count += int(values[i])
    return values


def particle_paths(table, evidence, target, uniforms):
    """Havasi-style proposal: clamp evidence; weight by its AR likelihood factors."""
    evidence = validate_query(evidence, target)
    M = uniforms.shape[1]
    assert uniforms.shape == (int(np.sum(evidence == -1)), M)
    assert np.all((uniforms >= 0) & (uniforms < 1))
    count = np.zeros(M, dtype=int)
    logweight = np.zeros(M)
    paths = np.zeros(M, dtype=np.uint64)
    row = 0
    for i in range(table["d"]):
        if evidence[i] >= 0:
            bit = np.full(M, int(evidence[i]), dtype=np.int8)
            logweight += table["lp"][i, count] if evidence[i] else table["lq"][i, count]
        else:
            bit = (uniforms[row] < table["probability"][i, count]).astype(np.int8)
            row += 1
        paths |= bit.astype(np.uint64) << np.uint64(i)
        count += bit
    bits = ((paths >> np.uint64(target)) & np.uint64(1)).astype(np.int8)
    return paths, logweight, bits


def normalized_estimate(logweight, bits):
    shift = float(np.max(logweight))
    w = np.exp(logweight-shift)
    total = float(w.sum())
    p = float((w*bits).sum()/total)
    ess = total*total/float(w @ w)
    assert 0 <= p <= 1 and 1-1e-10 <= ess <= len(w)+1e-10
    return dict(probability=p, ESS=ess, max_normalized_weight=float(w.max()/total),
                log_evidence_estimate=shift+math.log(total)-math.log(len(w)))


def arithmetic_smoke():
    """Deterministic QA only, no RNG or study observations."""
    largest_p, largest_log, largest_weight = 0., 0., 0.
    for d in (3, 5, 8):
        table = model_table(d)
        fixtures = [np.full(d, -1, dtype=np.int8),
                    np.array([0 if i % 3 == 0 else 1 if i % 3 == 1 else -1 for i in range(d)], dtype=np.int8)]
        for evidence in fixtures:
            target = int(np.flatnonzero(evidence == -1)[0])
            exact = exact_query(table, evidence, target)
            enumerated, logz, _ = enumerate_query(table, evidence, target)
            largest_p = max(largest_p, abs(exact["probability"]-enumerated))
            largest_log = max(largest_log, abs(exact["log_evidence"]-logz))
            uniforms = np.tile(np.array([.07, .31, .68, .94]), (np.sum(evidence == -1), 1))
            paths, logw, bits = particle_paths(table, evidence, target, uniforms)
            for n, path in enumerate(paths):
                count, full, proposal = 0, 0., 0.
                for i in range(d):
                    bit = (int(path) >> i) & 1
                    term = float(table["lp" if bit else "lq"][i, count])
                    full += term
                    if evidence[i] == -1:
                        proposal += term
                    else:
                        assert bit == evidence[i]
                    count += bit
                largest_weight = max(largest_weight, abs(full-proposal-logw[n]))
                assert bits[n] == ((int(path) >> target) & 1)
        evidence = np.full(d, -1, dtype=np.int8)
        evidence[:d//2] = np.arange(d//2) % 2
        largest_p = max(largest_p, abs(prefix_direct(table, evidence, d//2)-exact_query(table, evidence, d//2)["probability"]))
    # Deterministic high-bit / d64 numerical coverage without new study draws.
    table = model_table(64)
    evidence = np.full(64, -1, dtype=np.int8)
    paths, logw, bits = particle_paths(table, evidence, 63, np.zeros((64, 4)))
    assert np.all(paths == np.uint64(2**64-1)) and np.all(bits == 1) and np.all(logw == 0)
    evidence[:48] = np.arange(48) % 2
    largest_p = max(largest_p, abs(prefix_direct(table, evidence, 48)-exact_query(table, evidence, 48)["probability"]))
    allocated = [seed for cell in cells() for rep in range(16) for q in range(32)
                 for seed in seeds(cell["cell"], rep, q).values()]
    assert len(allocated) == len(set(allocated)) == 18432
    assert max(largest_p, largest_log, largest_weight) < 2e-12
    return dict(passed=True, maximum_probability_error=largest_p,
                maximum_log_evidence_error=largest_log, maximum_IS_ratio_error=largest_weight,
                scientific_random_draws=0, model_fits=0)


def freeze():
    assert not SEAL.exists() and not OUT.exists(), "C output already exists"
    OUT.mkdir(parents=True)
    (OUT/"local").mkdir()
    (OUT/".gitignore").write_text("/local/\n", encoding="utf-8", newline="\n")
    (OUT/"source_inputs").mkdir()
    (OUT/"source_inputs/HANDOFF.md").write_bytes((ROOT/"HANDOFF.md").read_bytes())
    write(OUT/"CONFIG.json", CFG)
    write(OUT/"ENVIRONMENT.json", dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
          platform=platform.platform(), cpu_count=os.cpu_count(), threads=2,
          timing_is_descriptive=True, isolated_CPU_execution_not_guaranteed=True))
    paths = [Path(__file__), PLAN, OUT/"CONFIG.json", OUT/"ENVIRONMENT.json", OUT/"source_inputs/HANDOFF.md"]
    write(SEAL, dict(time_utc=now(), files=hashes(paths), before_scientific_RNG=True,
                    first_scientific_start_utc=None, source_sha256=sha(__file__)))
    write(OUT/"STATE.json", dict(status="frozen", time_utc=now(), first_scientific_start_utc=None))
    print(json.dumps(dict(frozen=True, source_sha256=sha(__file__), freeze_sha256=sha(SEAL))), flush=True)


def checkfreeze():
    assert read(OUT/"CONFIG.json") == CFG
    verify_hashes(read(SEAL)["files"])


def warmup(table, cell):
    d, k = cell["d"], cell["observed"]
    indices = np.arange(k) if cell["layout"] == "prefix" else np.arange(k)*d//k
    evidence = np.full(d, -1, dtype=np.int8)
    evidence[indices] = np.arange(k) % 2
    target = int(np.flatnonzero(evidence == -1)[0])
    uniforms = np.full((d-k, 256), .5)
    start = time.perf_counter()
    exact_query(table, evidence, target)
    for K in CFG["particles"]:
        _, logw, bits = particle_paths(table, evidence, target, uniforms[:, :K])
        normalized_estimate(logw, bits)
    if cell["layout"] == "prefix":
        prefix_direct(table, evidence, target)
    return time.perf_counter()-start


def stream_run(cell, rep, table):
    d, k, ci = cell["d"], cell["observed"], cell["cell"]
    qn = CFG["queries_per_stream"]
    data = dict(evidence=np.full((qn, d), -1, dtype=np.int8), target=np.zeros(qn, dtype=np.int16),
                exact=np.zeros(qn), log_evidence=np.zeros(qn), p_IS=np.zeros((qn, 3)),
                excess_Brier=np.zeros((qn, 3)), ESS=np.zeros((qn, 3)), max_weight=np.zeros((qn, 3)),
                log_evidence_IS=np.zeros((qn, 3)), dp_seconds=np.zeros(qn),
                is_seconds=np.zeros((qn, 3)), uniform_seconds=np.zeros(qn),
                teacher_seconds=np.zeros(qn), query_design_seconds=np.zeros(qn),
                direct_prefix=np.zeros(qn), direct_seconds=np.zeros(qn),
                logweights=np.zeros((qn, 256)), paths=np.zeros((qn, 256), dtype=np.uint64),
                target_samples=np.zeros((qn, 256), dtype=np.int8),
                enum_checked=np.zeros(qn, dtype=bool), enum_prob=np.zeros(qn),
                enum_log_evidence=np.zeros(qn), enum_seconds=np.zeros(qn),
                dp_terms=np.zeros(qn, dtype=np.int64), query_seeds=np.zeros((qn, 3), dtype=np.int64))
    queries, errors, completed = [], dict(DP_enum_probability=0., DP_enum_log=0., prefix_direct=0., nested_logweight=0.), 0
    active = None
    try:
        for q in range(qn):
            sd = seeds(ci, rep, q)
            active = dict(cell=cell, stream=rep, query=q, seeds=sd)
            start = time.perf_counter()
            if cell["layout"] == "prefix":
                observed, target = np.arange(k), k
            else:
                mask_rng = np.random.Generator(np.random.PCG64(sd["mask"]))
                observed = np.sort(mask_rng.choice(d, size=k, replace=False))
                unobserved = np.setdiff1d(np.arange(d), observed)
                target = int(mask_rng.choice(unobserved))
            data["query_design_seconds"][q] = time.perf_counter()-start
            start = time.perf_counter()
            teacher_rng = np.random.Generator(np.random.PCG64(sd["teacher"]))
            values = draw_teacher(table, teacher_rng)
            evidence = np.full(d, -1, dtype=np.int8)
            evidence[observed] = values[observed]
            del values  # No teacher hidden target or final count reaches either estimator.
            data["teacher_seconds"][q] = time.perf_counter()-start
            active.update(evidence=evidence.tolist(), target=target)
            data["evidence"][q], data["target"][q] = evidence, target
            data["query_seeds"][q] = [sd["mask"], sd["teacher"], sd["IS"]]
            start = time.perf_counter()
            dp = exact_query(table, evidence, target)
            data["dp_seconds"][q] = time.perf_counter()-start
            data["exact"][q], data["log_evidence"][q] = dp["probability"], dp["log_evidence"]
            data["dp_terms"][q] = dp["transition_terms"]
            if cell["layout"] == "prefix":
                start = time.perf_counter()
                direct = prefix_direct(table, evidence, target)
                data["direct_seconds"][q] = time.perf_counter()-start
                data["direct_prefix"][q] = direct
                errors["prefix_direct"] = max(errors["prefix_direct"], abs(direct-dp["probability"]))
                assert errors["prefix_direct"] <= CFG["probability_gate"]
            do_enum = d == 8 or (k == 3*d//4 and rep == 0 and q == 0)
            if do_enum:
                start = time.perf_counter()
                enum, enumlog, _ = enumerate_query(table, evidence, target)
                data["enum_seconds"][q] = time.perf_counter()-start
                data["enum_checked"][q] = True
                data["enum_prob"][q], data["enum_log_evidence"][q] = enum, enumlog
                errors["DP_enum_probability"] = max(errors["DP_enum_probability"], abs(enum-dp["probability"]))
                errors["DP_enum_log"] = max(errors["DP_enum_log"], abs(enumlog-dp["log_evidence"]))
                assert errors["DP_enum_probability"] <= CFG["probability_gate"]
                assert errors["DP_enum_log"] <= CFG["log_evidence_gate"]
            start = time.perf_counter()
            is_rng = np.random.Generator(np.random.PCG64(sd["IS"]))
            uniforms = is_rng.random((d-k, 256))
            data["uniform_seconds"][q] = time.perf_counter()-start
            prefixes = []
            for ki, K in enumerate(CFG["particles"]):
                start = time.perf_counter()
                paths, logw, bits = particle_paths(table, evidence, target, uniforms[:, :K])
                estimate = normalized_estimate(logw, bits)
                data["is_seconds"][q, ki] = time.perf_counter()-start
                data["p_IS"][q, ki] = estimate["probability"]
                data["excess_Brier"][q, ki] = (estimate["probability"]-dp["probability"])**2
                data["ESS"][q, ki] = estimate["ESS"]
                data["max_weight"][q, ki] = estimate["max_normalized_weight"]
                data["log_evidence_IS"][q, ki] = estimate["log_evidence_estimate"]
                prefixes.append((paths.copy(), logw.copy()))
                if K == 256:
                    data["paths"][q], data["logweights"][q], data["target_samples"][q] = paths, logw, bits
            for paths, logw in prefixes:
                assert np.array_equal(paths, data["paths"][q, :len(paths)])
                err = float(np.max(abs(logw-data["logweights"][q, :len(paths)])))
                errors["nested_logweight"] = max(errors["nested_logweight"], err)
                assert err == 0.
            if cell["layout"] == "prefix":
                assert np.ptp(data["logweights"][q]) == 0.
                assert abs(data["logweights"][q, 0]-dp["log_evidence"]) <= CFG["log_evidence_gate"]
            queries.append(dict(query=q, seeds=sd, evidence=evidence.tolist(), target=target,
                observed_indices=observed.tolist(), evidence_ones=int(evidence[observed].sum()),
                evidence_after_target=int(np.sum(observed > target)),
                mask_is_prefix=bool(np.array_equal(observed, np.arange(k))),
                exact_probability=dp["probability"], exact_log_evidence=dp["log_evidence"],
                enumerated=bool(do_enum), teacher_hidden_count_available_to_inference=False,
                unique_teacher_draws=1, unique_proposal_trajectories=256,
                executed_prefix_trajectories=336, unique_proposal_uniforms=(d-k)*256))
            completed += 1
    except Exception:
        failure = OUT/"failures"/f"cell{ci:02}_s{rep:02}.npz"
        failure.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(failure, **data)
        write(failure.with_suffix(".json"), dict(completed_queries=completed, active_query=active,
              completed_query_log=queries, partial_array_sha256=sha(failure), errors=errors))
        raise
    target_path = OUT/"local"/f"cell{ci:02}_s{rep:02}.npz"
    assert not target_path.exists()
    np.savez_compressed(target_path, **data)
    rows = []
    for ki, K in enumerate(CFG["particles"]):
        rows.append(dict(**cell, stream=rep, K=K, queries=qn,
            mean_excess_Brier=float(data["excess_Brier"][:, ki].mean()),
            mean_ESS=float(data["ESS"][:, ki].mean()),
            mean_ESS_fraction=float((data["ESS"][:, ki]/K).mean()),
            mean_max_weight=float(data["max_weight"][:, ki].mean()),
            mean_dp_seconds=float(data["dp_seconds"].mean()),
            mean_is_path_seconds=float(data["is_seconds"][:, ki].mean()),
            mean_uniform_generation_seconds=float(data["uniform_seconds"].mean()),
            mean_teacher_seconds=float(data["teacher_seconds"].mean()),
            mean_prefix_direct_seconds=(float(data["direct_seconds"].mean()) if cell["layout"] == "prefix" else None),
            mean_dp_transition_terms=float(data["dp_terms"].mean()),
            IS_transition_lookups=K*d, IS_weight_factors=K*k, IS_uniforms_consumed=K*(d-k),
            mean_future_evidence=float(np.mean([r["evidence_after_target"] for r in queries]))))
    write(OUT/"streams"/f"cell{ci:02}_s{rep:02}.json", dict(**cell, stream=rep,
          queries=queries, stream_scores=rows, numerical_pass=True, numerical_errors=errors,
          local_array=target_path.relative_to(ROOT).as_posix(), local_sha256=sha(target_path),
          source_sha256=sha(__file__), freeze_sha256=sha(SEAL)))
    print(json.dumps(dict(cell=ci, d=d, observed=k, layout=cell["layout"], stream=rep,
                         queries=qn, mean_error_K256=rows[-1]["mean_excess_Brier"])), flush=True)


def collect():
    scores = []
    for cell in cells():
        for rep in range(16):
            rec = read(OUT/"streams"/f"cell{cell['cell']:02}_s{rep:02}.json")
            assert rec["numerical_pass"]
            scores.extend(rec["stream_scores"])
    assert len(scores) == 576
    csvwrite(OUT/"STREAM_SCORES.csv", scores)
    summary = []
    critical = float(student_t.ppf(1-.05/(2*36), 15))
    for cell in cells():
        for K in CFG["particles"]:
            subset = [r for r in scores if r["cell"] == cell["cell"] and r["K"] == K]
            values = np.array([r["mean_excess_Brier"] for r in subset])
            mean, se = float(values.mean()), float(values.std(ddof=1)/4)
            summary.append(dict(**cell, K=K, streams=16, queries=512, mean_excess_Brier=mean,
                SE=se, lower=mean-critical*se, upper=mean+critical*se,
                confidence_family=36, t_approximation=True,
                mean_dp_seconds=float(np.mean([r["mean_dp_seconds"] for r in subset])),
                mean_is_path_seconds=float(np.mean([r["mean_is_path_seconds"] for r in subset])),
                mean_uniform_generation_seconds=float(np.mean([r["mean_uniform_generation_seconds"] for r in subset])),
                mean_prefix_direct_seconds=(float(np.mean([r["mean_prefix_direct_seconds"] for r in subset])) if cell["layout"] == "prefix" else None),
                mean_ESS_fraction=float(np.mean([r["mean_ESS_fraction"] for r in subset])),
                mean_max_weight=float(np.mean([r["mean_max_weight"] for r in subset]))))
    csvwrite(OUT/"SUMMARY.csv", summary)
    write(OUT/"COUNTS.json", dict(cells=12, streams=192, queries=6144,
          teacher_iid_draws=6144, independent_IS_banks=6144,
          unique_proposal_trajectories=1572864, executed_prefix_trajectories=2064384,
          reused_prefix_trajectory_evaluations=491520,
          full_completion_reference_checks=2052, fitted_models=0,
          no_equal_cost_claim=True, no_hidden_teacher_N=True,
          prefix_direct_available=True, timing_is_descriptive_not_isolated_benchmark=True))


def independent_forward(table, evidence, target):
    """Audit DP: probability-domain forward on (prefix count,target bit), scaled."""
    d = len(evidence)
    state = np.zeros((d+1, 2)); state[0, 0] = 1.
    logz = 0.
    for i in range(d):
        nxt = np.zeros_like(state)
        for count in range(i+1):
            theta = .8*math.sin(.7*(i+1))+.9*math.tanh((2*count-i)/math.sqrt(i+1))+1.25*(1 if count % 2 == 0 else -1)
            p = 1/(1+math.exp(-theta))
            for bit, prob in ((0, 1-p), (1, p)):
                if evidence[i] >= 0 and evidence[i] != bit:
                    continue
                if i == target:
                    nxt[count+bit, bit] += float(state[count].sum())*prob
                else:
                    nxt[count+bit] += state[count]*prob
        norm = float(nxt.sum())
        assert norm > 0
        state, logz = nxt/norm, logz+math.log(norm)
    return float(state[:, 1].sum()), logz


def audit(workspace=False):
    """Stored-array arithmetic replay; no resampling or model fitting."""
    checkfreeze()
    max_p = max_log = max_weights = max_readout = 0.
    queried = enums = 0
    rows = []
    for cell in cells():
        table = model_table(cell["d"])
        for rep in range(16):
            rec = read(OUT/"streams"/f"cell{cell['cell']:02}_s{rep:02}.json")
            assert rec["source_sha256"] == sha(__file__) and rec["freeze_sha256"] == sha(SEAL)
            assert len(rec["queries"]) == 32 and rec["numerical_pass"]
            if not workspace:
                rows.extend(rec["stream_scores"])
                continue
            path = ROOT/rec["local_array"]
            assert sha(path) == rec["local_sha256"]
            with np.load(path) as saved:
                risks = np.zeros((32, 3))
                for q, query in enumerate(rec["queries"]):
                    evidence, j = np.array(query["evidence"], dtype=np.int8), query["target"]
                    assert np.array_equal(evidence, saved["evidence"][q]) and j == saved["target"][q]
                    assert query["seeds"] == seeds(cell["cell"], rep, q)
                    assert np.sum(evidence >= 0) == cell["observed"] and evidence[j] == -1
                    exact, logz = independent_forward(table, evidence, j)
                    max_p = max(max_p, abs(exact-saved["exact"][q]))
                    max_log = max(max_log, abs(logz-saved["log_evidence"][q]))
                    words = saved["paths"][q]
                    count = np.zeros(256, dtype=int)
                    joint_log, proposal_log = np.zeros(256), np.zeros(256)
                    for i in range(cell["d"]):
                        bit = ((words >> np.uint64(i)) & np.uint64(1)).astype(int)
                        theta = .8*np.sin(.7*(i+1))+.9*np.tanh((2*count-i)/math.sqrt(i+1))+1.25*np.where(count % 2 == 0, 1., -1.)
                        term = bit*theta-np.log1p(np.exp(theta))
                        joint_log += term
                        if evidence[i] == -1:
                            proposal_log += term
                        else:
                            assert np.all(bit == evidence[i])
                        count += bit
                    logw = joint_log-proposal_log
                    max_weights = max(max_weights, float(np.max(abs(logw-saved["logweights"][q]))))
                    bits = ((words >> np.uint64(j)) & np.uint64(1)).astype(int)
                    assert np.array_equal(bits, saved["target_samples"][q])
                    for ki, K in enumerate(CFG["particles"]):
                        source = saved["logweights"][q, :K]
                        weights = [math.exp(float(v)-float(max(source))) for v in source]
                        den = math.fsum(weights)
                        estimate = math.fsum(w*int(y) for w, y in zip(weights, bits[:K]))/den
                        risks[q, ki] = (estimate-float(saved["exact"][q]))**2
                        ess = den*den/math.fsum(w*w for w in weights)
                        logev = float(max(source))+math.log(den)-math.log(K)
                        max_readout = max(max_readout, abs(estimate-saved["p_IS"][q, ki]),
                            abs(risks[q, ki]-saved["excess_Brier"][q, ki]),
                            abs(ess-saved["ESS"][q, ki]), abs(logev-saved["log_evidence_IS"][q, ki]))
                    if cell["layout"] == "prefix":
                        direct = prefix_direct(table, evidence, j)
                        max_p = max(max_p, abs(direct-saved["direct_prefix"][q]))
                    if saved["enum_checked"][q]:
                        max_p = max(max_p, abs(saved["enum_prob"][q]-saved["exact"][q]))
                        max_log = max(max_log, abs(saved["enum_log_evidence"][q]-saved["log_evidence"][q]))
                        enums += 1
                    queried += 1
                for ki, row in enumerate(rec["stream_scores"]):
                    max_readout = max(max_readout, abs(float(risks[:, ki].mean())-row["mean_excess_Brier"]))
                    rows.append(row)
    if workspace:
        assert queried == 6144 and enums == 2052
        assert max_p <= CFG["probability_gate"] and max_log <= CFG["log_evidence_gate"]
        assert max_weights <= CFG["arithmetic_gate"] and max_readout <= CFG["arithmetic_gate"]
    with (OUT/"STREAM_SCORES.csv").open(encoding="utf-8", newline="") as f:
        public = list(csv.DictReader(f))
    assert len(public) == len(rows) == 576
    for a, b in zip(public, rows):
        for name in ("cell", "stream", "K"):
            assert int(a[name]) == b[name]
        assert abs(float(a["mean_excess_Brier"])-b["mean_excess_Brier"]) < 1e-14
    with (OUT/"SUMMARY.csv").open(encoding="utf-8", newline="") as f:
        summary = list(csv.DictReader(f))
    critical = float(student_t.ppf(1-.05/(2*36), 15))
    assert len(summary) == 36
    for row in summary:
        values = [r["mean_excess_Brier"] for r in rows if r["cell"] == int(row["cell"]) and r["K"] == int(row["K"])]
        mean = math.fsum(values)/16
        se = math.sqrt(math.fsum((v-mean)**2 for v in values)/(15*16))
        for key, val in (("mean_excess_Brier", mean), ("SE", se), ("lower", mean-critical*se), ("upper", mean+critical*se)):
            assert abs(float(row[key])-val) <= 1e-13
    return dict(passed=True, time_utc=now(), workspace=workspace,
        source_sha256=sha(__file__), freeze_sha256=sha(SEAL),
        stored_queries_replayed=queried, complete_enumeration_checks=enums,
        independently_replayed_stream_metrics=576, simultaneous_t_readouts=36,
        maximum_probability_error=max_p, maximum_log_evidence_error=max_log,
        maximum_IS_logweight_error=max_weights, maximum_readout_error=max_readout,
        no_RNG_replay=True, new_scientific_draws=0, new_fits=0,
        scope="alternate scaled forward target-state DP; full/proposal path-ratio weights; scalar weighted readouts; statistics",
        timing_not_reproduced=True)


def run():
    checkfreeze()
    assert read(OUT/"STATE.json")["status"] == "frozen", "no silent scientific restart"
    write(OUT/"SMOKE.json", arithmetic_smoke())
    tables, model_costs = {}, []
    for d in CFG["dimensions"]:
        start = time.perf_counter()
        tables[d] = model_table(d)
        model_costs.append(dict(d=d, seconds=time.perf_counter()-start, transition_probability_states=d*(d+1)//2))
    warmups = [dict(**cell, seconds=warmup(tables[cell["d"]], cell), scientific_draws=0) for cell in cells()]
    write(OUT/"TIMING_SETUP.json", dict(model_tables=model_costs, deterministic_warmups=warmups,
        timers="time.perf_counter; one warmed call per query and method; no timing reruns",
        IS_uniform_generation_is_separate=True, possible_concurrent_CPU_work=True))
    started = now()
    write(OUT/"START.json", dict(first_scientific_start_utc=started, freeze_sha256=sha(SEAL),
                                 source_sha256=sha(__file__), root_authorized_execution=True))
    for cell in cells():
        for rep in range(16):
            write(OUT/"STATE.json", dict(status="running", cell=cell, stream=rep,
                  first_scientific_start_utc=started, time_utc=now(), pid=os.getpid()), replace=True)
            stream_run(cell, rep, tables[cell["d"]])
    collect()
    write(OUT/"AUDIT.json", audit(workspace=True))
    paths = list((OUT/"streams").glob("*.json"))
    paths += [OUT/name for name in ("SMOKE.json", "TIMING_SETUP.json", "START.json", "STREAM_SCORES.csv", "SUMMARY.csv", "COUNTS.json", "AUDIT.json")]
    write(OUT/"RESULTS_FREEZE.json", dict(time_utc=now(), files=hashes(paths), source_freeze_sha256=sha(SEAL)))
    write(OUT/"STATE.json", dict(status="complete", first_scientific_start_utc=started,
          time_utc=now(), active_jobs=[]), replace=True)
    print(json.dumps(dict(complete=True, queries=6144, new_teacher_draws=6144,
                         unique_IS_trajectories=1572864, fitted_models=0)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["freeze", "run", "check", "toyqa"])
    parser.add_argument("--workspace", action="store_true")
    args = parser.parse_args()
    try:
        if args.phase == "toyqa":
            print(json.dumps(arithmetic_smoke()))
        elif args.phase == "freeze":
            freeze()
        elif args.phase == "run":
            run()
        else:
            verify_hashes(read(OUT/"RESULTS_FREEZE.json")["files"])
            assert read(OUT/"STATE.json")["status"] == "complete"
            print(json.dumps(audit(workspace=args.workspace)))
    except Exception as error:
        if args.phase == "run" and SEAL.exists() and not (OUT/"RESULTS_FREEZE.json").exists():
            write(OUT/f"FAILURE_{time.time_ns()}.json", dict(time_utc=now(), error=repr(error),
                  traceback=traceback.format_exc(), state=read(OUT/"STATE.json")))
            state = read(OUT/"STATE.json")
            state.update(status="stopped_failure", time_utc=now(), active_jobs=[])
            write(OUT/"STATE.json", state, replace=True)
        raise


if __name__ == "__main__":
    main()
