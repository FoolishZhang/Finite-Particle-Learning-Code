"""Frozen equicorrelation/control simulation: moments and coarse information."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
for _name in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"]:
    os.environ[_name] = "2"
import numpy as np
import scipy
from scipy.special import expit, roots_hermitenorm, xlogy
from scipy.optimize import brentq
from scipy.stats import beta, binom

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/constraint_boundary_v2/M"
PLAN = ROOT / "docs/08_research_研究工作/全局约束与采样门槛_20260929/PLAN_M.md"
CONFIG = dict(
    d=[8, 24, 128, 512], sigma=4., c=.9,
    families=["NEG", "POS_FULL", "POS_RANK1"], K=[4, 10, 100, 512],
    GH=[160, 240], quadrature_gate=1e-9, root_gate=1e-11,
    streams=8, per_stream=32768, batch=2048, seed_base=20269701,
    simultaneous_events=24, alpha=.05, material=.005,
    primary_family="NEG", primary_d=128, primary_K=100,
    root_xtol=1e-12, root_rtol=1e-14, root_bracket=[0., 16.],
    additional_model_fitting=0, natural_data=False, clinical_data=False,
)

def now():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value, exclusive=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

def csvwrite(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def freeze():
    assert not OUT.exists()
    write(OUT / "CONFIG.json", CONFIG)
    write(OUT / "ENVIRONMENT.json", dict(time_utc=now(), python=sys.version,
        platform=platform.platform(), numpy=np.__version__, scipy=scipy.__version__,
        random_generator="numpy.default_rng / PCG64", BLAS_threads=2))
    paths = [Path(__file__), PLAN, OUT / "CONFIG.json", OUT / "ENVIRONMENT.json"]
    write(OUT / "FREEZE.json", dict(time_utc=now(),
        files={p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
        before_parameter_quadrature_and_simulation=True))

def check_freeze():
    assert read(OUT / "CONFIG.json") == CONFIG
    for name, h in read(OUT / "FREEZE.json")["files"].items():
        assert sha(ROOT / name) == h, name

def gh(order):
    z, w = roots_hermitenorm(order)
    return z, w / np.sqrt(2 * np.pi)

def paircorr(rho, order):
    z, w = gh(order)
    z1 = z[:, None]
    z2 = rho * z1 + math.sqrt(1 - rho * rho) * z[None, :]
    return float(np.sum(w[:, None] * w[None, :] *
        np.tanh(CONFIG["sigma"] * z1 / 2) * np.tanh(CONFIG["sigma"] * z2 / 2)))

def rank1corr(v, order):
    z, w = gh(order)
    return float(w @ np.tanh(math.sqrt(v) * z / 2) ** 2)

def bernkl(p, q):
    return float(xlogy(p, p / q) + xlogy(1 - p, (1 - p) / (1 - q)))

def parameter_rows():
    rows = []
    for d in CONFIG["d"]:
        rho = CONFIG["c"] / (d - 1)
        rs = [paircorr(rho, n) for n in CONFIG["GH"]]
        negative = [paircorr(-rho, n) for n in CONFIG["GH"]]
        r = rs[-1]
        v = brentq(lambda value: rank1corr(value, 240) - r,
            *CONFIG["root_bracket"], xtol=CONFIG["root_xtol"], rtol=CONFIG["root_rtol"])
        match = [rank1corr(v, n) for n in CONFIG["GH"]]
        upper = -.5 * (math.log(1 - CONFIG["c"]) + (d - 1) * math.log1p(rho))
        R = np.full((d, d), -rho)
        np.fill_diagonal(R, 1.)
        sign, logdet = np.linalg.slogdet(R)
        assert sign == 1 and abs(upper + .5 * logdet) < 1e-11
        noise_projection = np.eye(d) - np.ones((d, d)) / d
        B = math.sqrt(1 + rho) * noise_projection + math.sqrt(1 - CONFIG["c"]) * np.ones((d, d)) / d
        factor_error = float(np.max(abs(B @ B.T - R)))
        passed = abs(rs[1] - rs[0]) < CONFIG["quadrature_gate"]
        passed = passed and max(abs(negative[i] + rs[i]) for i in range(2)) < 1e-12
        passed = passed and abs(match[1] - r) < CONFIG["root_gate"] and abs(match[1] - match[0]) < CONFIG["quadrature_gate"]
        passed = passed and factor_error < 1e-11
        rows.append(dict(d=d, latent_rho_positive=rho, spin_r_160=rs[0],
            spin_r_240=r, quadrature_difference=abs(rs[1] - rs[0]),
            negative_spin_r_160=negative[0], negative_spin_r_240=negative[1],
            rank1_variance=v, rank1_match_160=match[0], rank1_match_240=match[1],
            rank1_match_error=abs(match[1] - r),
            rank1_quadrature_difference=abs(match[1] - match[0]),
            pair_MI=bernkl((1 + r) / 2, .5), NEG_TC_latent_upper=upper,
            NEG_generation_factor_error=factor_error, numerical_pass=bool(passed)))
    write(OUT / "PARAMETERS.json", dict(time_utc=now(), rows=rows,
        all_numerical_pass=all(r["numerical_pass"] for r in rows)))
    return rows

def formulas(params):
    moments, bounds = [], []
    max_matrix_error = 0.
    for p in params:
        d, r = p["d"], p["spin_r_240"]
        info = max(0., .5 * r * d * (d - 1) - p["NEG_TC_latent_upper"])
        for K in CONFIG["K"]:
            bounds.append(dict(d=d, K=K, spin_pair_abs=r, pair_MI=p["pair_MI"],
                NEG_trace_lower=4 * r * d * (d - 1), NEG_rank_lower=d - 1,
                NEG_TC_latent_upper=p["NEG_TC_latent_upper"],
                conditional_independent_representation_information_lower=info,
                SIR_forward_KL_lower=max(0., info - math.log(K)),
                POS_RANK1_trace=d * p["rank1_variance"], POS_RANK1_rank=1,
                NEG_Kcrit=1 + 1 / r, POS_Kcrit=1 + 1 / ((d - 1) * r),
                bounds_are_not_training_or_observed_risk_gaps=True))
            for family in CONFIG["families"]:
                signed = -r if family == "NEG" else r
                eig = dict(common=1 + (d - 1) * signed, contrast=1 - signed)
                for sampler in ["iid", "antithetic"]:
                    for mode, lam in eig.items():
                        coefficient = 1 - (1 - 1 / K) * lam if sampler == "iid" else 1 - lam
                        observed = np.full((d, d), signed)
                        np.fill_diagonal(observed, 1.)
                        common = np.ones((d, d)) / d
                        direction = common if mode == "common" else (np.eye(d) - common) / (d - 1)
                        A = (np.eye(d) - (1 - 1 / K) * observed) / 4 if sampler == "iid" else (np.eye(d) - observed) / 4
                        direct = float(np.sum(direction * A) / 2)
                        assert abs(np.trace(direction) - 1) < 1e-12
                        max_matrix_error = max(max_matrix_error, abs(direct - coefficient / 8))
                        moments.append(dict(family=family, d=d, K=K, sampler=sampler,
                            mode=mode, mode_multiplicity=1 if mode == "common" else d - 1,
                            covariance_direction_trace=1., observed_eigenvalue=lam,
                            objective_covariance_derivative=coefficient / 8,
                            observed_pair_corr=signed, pair_MI=p["pair_MI"],
                            negative_direction=bool(coefficient < 0)))
    assert len(moments) == 192 and len(bounds) == 16
    assert max_matrix_error < 1e-12
    csvwrite(OUT / "MOMENTS.csv", moments)
    csvwrite(OUT / "BOUNDS.csv", bounds)
    write(OUT / "FORMULA_AUDIT.json", dict(passed=True,
        matrix_trace_derivative_max_error=max_matrix_error,
        trace_one_directions_checked=192, nonlinear_risk_or_training_gap_claimed=False))
    return moments, bounds

def simulate(family, familyidx, p):
    d, half = p["d"], p["d"] // 2
    streams, histograms = [], []
    for stream in range(CONFIG["streams"]):
        seed = CONFIG["seed_base"] + 1000 * familyidx + 100 * d + stream
        rng = np.random.default_rng(seed)
        joint = np.zeros((half + 1, half + 1), dtype=np.int64)
        total_hist = np.zeros(d + 1, dtype=np.int64)
        sign_hist = np.zeros(4, dtype=np.int64)
        central = 0
        for begin in range(0, CONFIG["per_stream"], CONFIG["batch"]):
            n = min(CONFIG["batch"], CONFIG["per_stream"] - begin)
            if family == "POS_RANK1":
                prob = expit(math.sqrt(p["rank1_variance"]) * rng.normal(size=n))
                first, second = rng.binomial(half, prob), rng.binomial(half, prob)
            else:
                if family == "NEG":
                    eps = rng.normal(size=(n, d))
                    mean = eps.mean(1, keepdims=True)
                    z = math.sqrt(1 + CONFIG["c"] / (d - 1)) * (eps - mean) + math.sqrt(1 - CONFIG["c"]) * mean
                else:
                    common = rng.normal(size=(n, 1))
                    eps = rng.normal(size=(n, d))
                    z = math.sqrt(p["latent_rho_positive"]) * common + math.sqrt(1 - p["latent_rho_positive"]) * eps
                labels = rng.random((n, d)) < expit(CONFIG["sigma"] * z)
                first, second = labels[:, :half].sum(1), labels[:, half:].sum(1)
            total = first + second
            total_hist += np.bincount(total, minlength=d + 1)
            joint += np.bincount(first * (half + 1) + second,
                                 minlength=(half + 1) ** 2).reshape(half + 1, half + 1)
            central += int((np.abs(2 * total - d) <= math.sqrt(d)).sum())
            coins = rng.integers(0, 2, size=(2, n))
            spin_first, spin_second = 2 * first - half, 2 * second - half
            sign_first = np.where(spin_first == 0, coins[0], spin_first > 0).astype(int)
            sign_second = np.where(spin_second == 0, coins[1], spin_second > 0).astype(int)
            sign_hist += np.bincount(2 * sign_first + sign_second, minlength=4)
        spin = 2 * np.arange(d + 1) - d
        n = CONFIG["per_stream"]
        row = dict(family=family, d=d, stream=stream, seed=seed, observations=n,
            central_count=central, equal_sign_count=int(sign_hist[0] + sign_hist[3]),
            first_sign_positive=int(sign_hist[2] + sign_hist[3]),
            second_sign_positive=int(sign_hist[1] + sign_hist[3]),
            sum_spin=int(total_hist @ spin),
            sum_spin_squared=int(total_hist @ (spin * spin)),
            sum_spin_fourth=int(total_hist @ (spin ** 4)))
        assert total_hist.sum() == joint.sum() == sign_hist.sum() == n
        aa, bb = np.nonzero(joint)
        histograms.append(dict(stream=stream, seed=seed,
            total_count_histogram=total_hist.tolist(),
            first_half_histogram=joint.sum(1).tolist(),
            second_half_histogram=joint.sum(0).tolist(),
            half_joint_shape=[half + 1, half + 1],
            half_joint_sparse=[[int(a), int(b), int(joint[a, b])] for a, b in zip(aa, bb)],
            randomized_sign_joint=sign_hist.tolist()))
        streams.append(row)
        print(f"{family} d={d} stream={stream} complete", flush=True)
    path = OUT / "histograms" / f"{family}_d{d}.json"
    write(path, dict(family=family, d=d, zeros_omitted_from_joint_histogram=True, streams=histograms))
    return streams, path

def clopper(count, n):
    tail = CONFIG["alpha"] / (2 * CONFIG["simultaneous_events"])
    low = 0. if count == 0 else float(beta.ppf(tail, count, n - count + 1))
    high = 1. if count == n else float(beta.isf(tail, count + 1, n - count))
    return low, high

def minimum_kl(low, high, baseline):
    point = min(high, max(low, baseline))
    return bernkl(point, baseline)

def intervals(rows, params):
    result, empirical = [], []
    for p in params:
        d = p["d"]
        low_count = math.ceil((d - math.sqrt(d)) / 2)
        high_count = math.floor((d + math.sqrt(d)) / 2)
        baseline = float(binom.cdf(high_count, d, .5) - binom.cdf(low_count - 1, d, .5))
        direct = float(binom.pmf(np.arange(low_count, high_count + 1), d, .5).sum())
        assert abs(baseline - direct) < 1e-12
        for family in CONFIG["families"]:
            rr = [r for r in rows if r["family"] == family and r["d"] == d]
            assert len(rr) == 8
            n = sum(r["observations"] for r in rr)
            assert n == 262144
            for event, field, base, meaning in [
                ("central", "central_count", baseline, "full_joint_total_correlation_lower"),
                ("equal_block_sign", "equal_sign_count", .5, "full_two_block_mutual_information_lower")]:
                count = sum(r[field] for r in rr)
                lo, hi = clopper(count, n)
                result.append(dict(family=family, d=d, event=event, observations=n,
                    successes=count, estimate=count / n, lower=lo, upper=hi,
                    independent_baseline=base, point_Bernoulli_KL=bernkl(count / n, base),
                    simultaneous_information_lower=minimum_kl(lo, hi, base),
                    lower_bound_target=meaning, family_size=24, confidence=.95,
                    central_low_count=low_count if event == "central" else "",
                    central_high_count=high_count if event == "central" else ""))
            second = sum(r["sum_spin_squared"] for r in rr) / n
            fourth = sum(r["sum_spin_fourth"] for r in rr) / n
            target_r = -p["spin_r_240"] if family == "NEG" else p["spin_r_240"]
            target_second = d + d * (d - 1) * target_r
            se = math.sqrt(max(0., fourth - second * second) / (n - 1))
            empirical.append(dict(family=family, d=d, observations=n,
                mean_spin=sum(r["sum_spin"] for r in rr) / n,
                mean_spin_squared=second, analytic_mean_spin_squared=target_second,
                mean_spin_squared_SE=se, diagnostic_z=(second - target_second) / se,
                pair_corr_from_count=(second - d) / (d * (d - 1)),
                analytic_pair_corr=target_r, no_moment_based_cell_selection=True))
    assert len(result) == 24 and len(empirical) == 12
    csvwrite(OUT / "EVENT_INTERVALS.csv", result)
    csvwrite(OUT / "EMPIRICAL_MOMENTS.csv", empirical)
    return result

def audit_tables(rows, params, events):
    all_hist = read(OUT / "HISTOGRAMS.json")["files"]
    bykey = {(r["family"], r["d"], r["stream"]): r for r in rows}
    for name, h in all_hist.items():
        path = ROOT / name
        assert sha(path) == h
        cell = read(path)
        d, family = cell["d"], cell["family"]
        for h in cell["streams"]:
            row = bykey[(family, d, h["stream"])]
            hist = np.asarray(h["total_count_histogram"], dtype=np.int64)
            joint = np.zeros(h["half_joint_shape"], dtype=np.int64)
            for a, b, n in h["half_joint_sparse"]:
                assert joint[a, b] == 0 and n > 0
                joint[a, b] = n
            total = np.zeros(d + 1, dtype=np.int64)
            for a in range(joint.shape[0]):
                for b in range(joint.shape[1]):
                    total[a + b] += joint[a, b]
            assert np.array_equal(total, hist)
            assert np.array_equal(joint.sum(1), h["first_half_histogram"])
            assert np.array_equal(joint.sum(0), h["second_half_histogram"])
            spin = 2 * np.arange(d + 1) - d
            assert int(hist[np.abs(spin) <= math.sqrt(d)].sum()) == row["central_count"]
            signs = h["randomized_sign_joint"]
            assert sum(signs) == 32768 and signs[0] + signs[3] == row["equal_sign_count"]
            assert int(hist @ spin ** 2) == row["sum_spin_squared"]
            assert int(hist @ spin ** 4) == row["sum_spin_fourth"]
    for event in events:
        k, n = event["successes"], event["observations"]
        lo, hi = clopper(k, n)
        assert abs(lo - event["lower"]) < 1e-14 and abs(hi - event["upper"]) < 1e-14
        target = CONFIG["alpha"] / (2 * CONFIG["simultaneous_events"])
        assert abs(float(binom.sf(k - 1, n, lo)) - target) < 1e-10
        assert abs(float(binom.cdf(k, n, hi)) - target) < 1e-10
    assert len(rows) == 96 and len(bykey) == 96
    assert len({r["seed"] for r in rows}) == 96
    return dict(histogram_identities=True, all_96_seeds_unique=True,
                CP_binomial_tail_inversion_pass=True, all_cells_preserved=True,
                quadrature_pass=all(p["numerical_pass"] for p in params))

def run():
    check_freeze()
    assert not (OUT / "PARAMETERS.json").exists(), "No overwrite or silent restart"
    start = time.perf_counter()
    params = parameter_rows()
    if not all(p["numerical_pass"] for p in params):
        write(OUT / "NUMERICAL_FAILURE.json", dict(time_utc=now(), rows=params,
            simulation_started=False, added_orders_or_seeds=False))
        raise RuntimeError("Frozen quadrature or parameter gate failed; simulation not started")
    moments, bounds = formulas(params)
    all_rows, hist_files = [], []
    for familyidx, family in enumerate(CONFIG["families"]):
        for p in params:
            write(OUT / "RUN_STATE.json", dict(status="running", family=family, d=p["d"],
                pid=os.getpid(), time_utc=now()), exclusive=False)
            rows, path = simulate(family, familyidx, p)
            all_rows.extend(rows)
            hist_files.append(path)
    csvwrite(OUT / "STREAM_COUNTS.csv", all_rows)
    write(OUT / "HISTOGRAMS.json", dict(files={p.relative_to(ROOT).as_posix(): sha(p) for p in hist_files}))
    events = intervals(all_rows, params)
    checks = audit_tables(all_rows, params, events)
    r128 = next(p["spin_r_240"] for p in params if p["d"] == 128)
    central = next(e for e in events if e["family"] == "NEG" and e["d"] == 128 and e["event"] == "central")
    block = next(e for e in events if e["family"] == "NEG" and e["d"] == 128 and e["event"] == "equal_block_sign")
    tests = dict(
        NEG_d128_K100_strict_local_trap=1 + r128 < 100 / 99,
        NEG_d128_TC_lower_gt_005=central["simultaneous_information_lower"] > .005,
        NEG_d128_block_MI_lower_gt_005=block["simultaneous_information_lower"] > .005,
        both_POS_pair_matched_d128_K100_have_descent=1 + 127 * r128 > 100 / 99)
    write(OUT / "DECISION.json", dict(time_utc=now(), primary_tests=tests,
        primary_pass=all(tests.values()), NEG_d128_TC_lower=central["simultaneous_information_lower"],
        NEG_d128_block_MI_lower=block["simultaneous_information_lower"],
        NEG_d512_SIR_lower_by_K={str(b["K"]): b["SIR_forward_KL_lower"] for b in bounds if b["d"] == 512},
        originality_certified=False, natural_confirmation=False,
        finite_MC_training_gap_or_semantic_risk_bound_claimed=False))
    write(OUT / "AUDIT.json", dict(time_utc=now(), passed=all(checks.values()), checks=checks,
        cells=12, independent_streams=96, observations=3145728, simultaneous_event_intervals=24,
        elapsed_seconds=time.perf_counter() - start, all_parameters_and_seeds_frozen=True,
        neural_models_fitted=0, raw_labels_saved=False))
    write(OUT / "RUN_STATE.json", dict(status="complete", active_jobs=[], time_utc=now()), exclusive=False)
    paths = sorted(p for p in OUT.rglob("*") if p.is_file())
    write(OUT / "MANIFEST.json", dict(time_utc=now(),
        files={p.relative_to(ROOT).as_posix(): sha(p) for p in paths}))
    check()

def check():
    check_freeze()
    for name, h in read(OUT / "MANIFEST.json")["files"].items():
        assert sha(ROOT / name) == h, name
    assert read(OUT / "AUDIT.json")["passed"]
    assert len(read(OUT / "PARAMETERS.json")["rows"]) == 4
    with (OUT / "EVENT_INTERVALS.csv").open(encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 24
    print(json.dumps(dict(pass_frozen_M=True, decision=read(OUT / "DECISION.json")), ensure_ascii=False), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["freeze", "run", "check"])
    args = parser.parse_args()
    try:
        globals()[args.phase]()
    except Exception:
        if OUT.exists():
            write(OUT / f"FAILURE_{time.time_ns()}.json", dict(time_utc=now(),
                phase=args.phase, traceback=traceback.format_exc()))
        raise
