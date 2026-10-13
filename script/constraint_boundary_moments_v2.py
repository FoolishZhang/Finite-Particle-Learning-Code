"""M2: fixed Hermite/quad parameter repair; unchanged frozen M simulation."""
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
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import eval_hermitenorm, ndtr
from scipy.stats import binom
from script import constraint_boundary_moments as legacy

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "artifacts/constraint_boundary_v2/M"
OUT = ROOT / "artifacts/constraint_boundary_v2/M2"
PLAN = ROOT / "docs/08_research_研究工作/全局约束与采样门槛_20260929/PLAN_M2.md"
# This redirects only the already-frozen count simulator/auditor's output I/O.
# The original script and its CONFIG remain untouched. No legacy run is called.
legacy.OUT = OUT
CONFIG = dict(legacy.CONFIG, parameter_method="odd_normalized_Hermite_quad",
    Hermite_orders=list(range(1, 14, 2)), quad_interval=[-12., 12.],
    quad_epsabs=1e-12, quad_epsrel=1e-12, quad_limit=300,
    r_error_gate=1e-10, partition_gate=1e-11,
    rank1_GH_match_gate=1e-9,
    original_M_observations=0, original_M_numerical_failure_preserved=True)

now, read, sha, write, csvwrite = legacy.now, legacy.read, legacy.sha, legacy.write, legacy.csvwrite

def old_files():
    return [OLD / "FREEZE.json", OLD / "PARAMETERS.json", OLD / "NUMERICAL_FAILURE.json",
            *sorted(OLD.glob("FAILURE_*.json")), Path(legacy.__file__)]

def freeze():
    assert not OUT.exists()
    assert not (OLD / "STREAM_COUNTS.csv").exists()
    assert not list((OLD / "histograms").glob("*"))
    assert read(OLD / "NUMERICAL_FAILURE.json")["simulation_started"] is False
    for name, h in read(OLD / "FREEZE.json")["files"].items():
        assert sha(ROOT / name) == h, name
    write(OUT / "CONFIG.json", CONFIG)
    write(OUT / "ENVIRONMENT.json", dict(time_utc=now(), python=sys.version,
        platform=platform.platform(), numpy=np.__version__, scipy=scipy.__version__,
        simulator_source=str(Path(legacy.__file__).relative_to(ROOT).as_posix()),
        simulator_source_sha256=sha(legacy.__file__), original_simulation_config=legacy.CONFIG))
    write(OUT / "M_RELATIONSHIP.json", dict(time_utc=now(), old_package="M",
        old_observations=0, old_failure="d24 GH160/240 difference > 1e-9",
        old_failure_retained=True, scientific_grid_seeds_events_gates_unchanged=True,
        repaired_algorithm_only=True, original_M_reclassified_as_pass=False))
    paths = [Path(__file__), PLAN, OUT / "CONFIG.json", OUT / "ENVIRONMENT.json",
             OUT / "M_RELATIONSHIP.json", *old_files()]
    write(OUT / "FREEZE.json", dict(time_utc=now(),
        files={p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
        before_new_parameter_computation_and_all_simulation=True))

def check_freeze():
    assert read(OUT / "CONFIG.json") == CONFIG
    for name, h in read(OUT / "FREEZE.json")["files"].items():
        assert sha(ROOT / name) == h, name
    assert not (OLD / "STREAM_COUNTS.csv").exists()

def integrate(fn, left, right):
    value = quad(fn, left, right, epsabs=CONFIG["quad_epsabs"],
                 epsrel=CONFIG["quad_epsrel"], limit=CONFIG["quad_limit"], full_output=1)
    return dict(value=float(value[0]), estimated_error=float(value[1]),
        successful=len(value) == 3, evaluations=int(value[2]["neval"]),
        subdivisions=int(value[2]["last"]), message="" if len(value) == 3 else str(value[3]))

def coefficients():
    tail_probability = float(2 * ndtr(-12.))
    coefficient_tail_bound = math.sqrt(tail_probability)
    rows = []
    for n in CONFIG["Hermite_orders"]:
        normalizer = math.sqrt(math.factorial(n))
        def fn(z):
            return math.tanh(2 * z) * float(eval_hermitenorm(n, z)) / normalizer * math.exp(-z*z/2) / math.sqrt(2*math.pi)
        whole = integrate(fn, -12., 12.)
        left, right = integrate(fn, -12., 0.), integrate(fn, 0., 12.)
        split = left["value"] + right["value"]
        rows.append(dict(n=n, coefficient=whole["value"], whole=whole,
            left=left, right=right, split_value=split,
            partition_difference=abs(whole["value"] - split),
            coefficient_truncation_bound=coefficient_tail_bound,
            propagated_coefficient_error_estimate=whole["estimated_error"] + coefficient_tail_bound,
            passed=bool(whole["successful"] and left["successful"] and right["successful"]
                        and abs(whole["value"] - split) < CONFIG["partition_gate"])))
    result = dict(time_utc=now(), rows=rows, tail_probability=tail_probability,
        coefficient_tail_bound=coefficient_tail_bound,
        squared_coefficient_partial_sum=sum(r["coefficient"] ** 2 for r in rows),
        quad_error_is_estimated_not_rigorous=True, all_pass=all(r["passed"] for r in rows))
    write(OUT / "COEFFICIENTS.json", result)
    return result

def rank1_integral(v, left=-12., right=12.):
    scale = math.sqrt(v) / 2
    return integrate(lambda z: math.tanh(scale * z) ** 2 * math.exp(-z*z/2) / math.sqrt(2*math.pi), left, right)

def parameter_rows():
    coeff = coefficients()
    old = {r["d"]: r for r in read(OLD / "PARAMETERS.json")["rows"]}
    rows = []
    for d in CONFIG["d"]:
        rho = CONFIG["c"] / (d - 1)
        r = sum(a["coefficient"] ** 2 * rho ** a["n"] for a in coeff["rows"])
        r_split = sum(a["split_value"] ** 2 * rho ** a["n"] for a in coeff["rows"])
        estimated = sum((2 * abs(a["coefficient"]) * a["propagated_coefficient_error_estimate"]
                         + a["propagated_coefficient_error_estimate"] ** 2) * rho ** a["n"] for a in coeff["rows"])
        series_tail = rho ** 15
        total_error = estimated + series_tail
        evaluations = []
        def objective(v):
            result = rank1_integral(v)
            evaluations.append(dict(v=float(v), **result))
            return result["value"] - r
        v = brentq(objective, *CONFIG["root_bracket"],
            xtol=CONFIG["root_xtol"], rtol=CONFIG["root_rtol"])
        root = rank1_integral(v)
        root_split = rank1_integral(v, -12., 0.)["value"] + rank1_integral(v, 0., 12.)["value"]
        gh = [legacy.rank1corr(v, order) for order in [160, 240]]
        upper = -.5 * (math.log(1 - CONFIG["c"]) + (d - 1) * math.log1p(rho))
        R = np.full((d, d), -rho)
        np.fill_diagonal(R, 1.)
        sign, logdet = np.linalg.slogdet(R)
        P = np.eye(d) - np.ones((d, d)) / d
        B = math.sqrt(1 + rho) * P + math.sqrt(1 - CONFIG["c"]) * np.ones((d, d)) / d
        factor_error = float(np.max(abs(B @ B.T - R)))
        numerical_pass = (
            coeff["all_pass"] and total_error < CONFIG["r_error_gate"]
            and abs(r - r_split) < CONFIG["partition_gate"]
            and all(a["successful"] for a in evaluations) and root["successful"]
            and abs(root["value"] - r) < CONFIG["root_gate"]
            and abs(root["value"] - root_split) < CONFIG["partition_gate"]
            and max(abs(g - r) for g in gh) < CONFIG["rank1_GH_match_gate"]
            and abs(gh[1] - gh[0]) < CONFIG["rank1_GH_match_gate"]
            and factor_error < 1e-11 and sign == 1 and abs(upper + .5 * logdet) < 1e-11)
        rows.append(dict(d=d, latent_rho_positive=rho, spin_r=r,
            spin_r_source="odd Hermite coefficients from fixed quad algorithm",
            split_coefficient_spin_r=r_split, r_partition_difference=abs(r-r_split),
            r_propagated_quad_and_coefficient_tail_estimate=estimated,
            r_analytic_series_tail_bound=series_tail,
            r_total_estimated_error_plus_analytic_tail=total_error,
            rank1_variance=v, rank1_quad=root, rank1_root_residual=abs(root["value"]-r),
            rank1_quad_partition_difference=abs(root["value"]-root_split),
            rank1_GH160=gh[0], rank1_GH240=gh[1], rank1_solver_evaluations=evaluations,
            old_GH160_diagnostic=old[d]["spin_r_160"],
            old_GH240_diagnostic=old[d]["spin_r_240"],
            old_GH_difference_diagnostic=old[d]["quadrature_difference"],
            old_GH240_minus_new_r=old[d]["spin_r_240"]-r,
            pair_MI=legacy.bernkl((1+r)/2, .5), NEG_TC_latent_upper=upper,
            NEG_generation_factor_error=factor_error,
            numerical_pass=bool(numerical_pass)))
    write(OUT / "PARAMETERS.json", dict(time_utc=now(), rows=rows,
        all_numerical_pass=all(r["numerical_pass"] for r in rows),
        previous_GH_failed_gate_not_reclassified=True))
    return rows

def formulas(params):
    moments, bounds = [], []
    max_error = 0.
    for p in params:
        d, r = p["d"], p["spin_r"]
        info = max(0., .5*r*d*(d-1)-p["NEG_TC_latent_upper"])
        for K in CONFIG["K"]:
            bounds.append(dict(d=d, K=K, spin_pair_abs=r, pair_MI=p["pair_MI"],
                NEG_trace_lower=4*r*d*(d-1), NEG_rank_lower=d-1,
                NEG_TC_latent_upper=p["NEG_TC_latent_upper"],
                conditional_independent_representation_information_lower=info,
                SIR_forward_KL_lower=max(0., info-math.log(K)),
                POS_RANK1_trace=d*p["rank1_variance"], POS_RANK1_rank=1,
                NEG_Kcrit=1+1/r, POS_Kcrit=1+1/((d-1)*r),
                bounds_are_not_training_or_observed_risk_gaps=True))
            for family in CONFIG["families"]:
                signed = -r if family == "NEG" else r
                R = np.full((d,d), signed)
                np.fill_diagonal(R,1.)
                common = np.ones((d,d))/d
                for sampler in ["iid","antithetic"]:
                    A = (np.eye(d)-(1-1/K)*R)/4 if sampler == "iid" else (np.eye(d)-R)/4
                    for mode, lam in [("common",1+(d-1)*signed),("contrast",1-signed)]:
                        direction = common if mode == "common" else (np.eye(d)-common)/(d-1)
                        coefficient = 1-(1-1/K)*lam if sampler == "iid" else 1-lam
                        direct = float(np.sum(direction*A)/2)
                        max_error = max(max_error,abs(direct-coefficient/8))
                        moments.append(dict(family=family,d=d,K=K,sampler=sampler,mode=mode,
                            mode_multiplicity=1 if mode=="common" else d-1,
                            covariance_direction_trace=float(np.trace(direction)),
                            observed_eigenvalue=lam,objective_covariance_derivative=coefficient/8,
                            observed_pair_corr=signed,pair_MI=p["pair_MI"],negative_direction=bool(coefficient<0)))
    assert len(moments)==192 and len(bounds)==16 and max_error<1e-12
    csvwrite(OUT/"MOMENTS.csv",moments)
    csvwrite(OUT/"BOUNDS.csv",bounds)
    write(OUT/"FORMULA_AUDIT.json",dict(passed=True,matrix_trace_derivative_max_error=max_error,
        trace_one_directions_checked=192,nonlinear_risk_or_training_gap_claimed=False))
    return moments,bounds

def intervals(rows,params):
    events,empirical=[],[]
    for p in params:
        d=p["d"]
        low_count=math.ceil((d-math.sqrt(d))/2)
        high_count=math.floor((d+math.sqrt(d))/2)
        baseline=float(binom.cdf(high_count,d,.5)-binom.cdf(low_count-1,d,.5))
        assert abs(baseline-binom.pmf(np.arange(low_count,high_count+1),d,.5).sum())<1e-12
        for family in CONFIG["families"]:
            rr=[r for r in rows if r["family"]==family and r["d"]==d]
            assert len(rr)==8
            n=sum(r["observations"] for r in rr)
            assert n==262144
            for event,field,base,meaning in [
                ("central","central_count",baseline,"full_joint_total_correlation_lower"),
                ("equal_block_sign","equal_sign_count",.5,"full_two_block_mutual_information_lower")]:
                count=sum(r[field] for r in rr)
                lo,hi=legacy.clopper(count,n)
                events.append(dict(family=family,d=d,event=event,observations=n,successes=count,
                    estimate=count/n,lower=lo,upper=hi,independent_baseline=base,
                    point_Bernoulli_KL=legacy.bernkl(count/n,base),
                    simultaneous_information_lower=legacy.minimum_kl(lo,hi,base),
                    lower_bound_target=meaning,family_size=24,confidence=.95,
                    central_low_count=low_count if event=="central" else "",
                    central_high_count=high_count if event=="central" else ""))
            second=sum(r["sum_spin_squared"] for r in rr)/n
            fourth=sum(r["sum_spin_fourth"] for r in rr)/n
            target_r=-p["spin_r"] if family=="NEG" else p["spin_r"]
            target_second=d+d*(d-1)*target_r
            se=math.sqrt(max(0.,fourth-second*second)/(n-1))
            empirical.append(dict(family=family,d=d,observations=n,
                mean_spin=sum(r["sum_spin"] for r in rr)/n,
                mean_spin_squared=second,analytic_mean_spin_squared=target_second,
                mean_spin_squared_SE=se,diagnostic_z=(second-target_second)/se,
                pair_corr_from_count=(second-d)/(d*(d-1)),analytic_pair_corr=target_r,
                analytic_parameter_source="Hermite_quad_M2",no_moment_based_cell_selection=True))
    assert len(events)==24 and len(empirical)==12
    csvwrite(OUT/"EVENT_INTERVALS.csv",events)
    csvwrite(OUT/"EMPIRICAL_MOMENTS.csv",empirical)
    return events

def run():
    check_freeze()
    assert not (OUT/"PARAMETERS.json").exists(),"No overwrite or silent restart"
    started=time.perf_counter()
    params=parameter_rows()
    if not all(p["numerical_pass"] for p in params):
        write(OUT/"NUMERICAL_FAILURE.json",dict(time_utc=now(),rows=params,simulation_started=False,
            added_orders_or_seeds=False,old_M_preserved=True))
        raise RuntimeError("M2 fixed numerical algorithm failed; simulation not started")
    moments,bounds=formulas(params)
    all_rows,hist_files=[],[]
    for familyidx,family in enumerate(CONFIG["families"]):
        for p in params:
            write(OUT/"RUN_STATE.json",dict(status="running",family=family,d=p["d"],
                pid=os.getpid(),time_utc=now()),exclusive=False)
            rows,path=legacy.simulate(family,familyidx,p)
            all_rows.extend(rows)
            hist_files.append(path)
    csvwrite(OUT/"STREAM_COUNTS.csv",all_rows)
    write(OUT/"HISTOGRAMS.json",dict(files={p.relative_to(ROOT).as_posix():sha(p) for p in hist_files}))
    events=intervals(all_rows,params)
    checks=legacy.audit_tables(all_rows,params,events)
    r128=next(p["spin_r"] for p in params if p["d"]==128)
    central=next(e for e in events if e["family"]=="NEG" and e["d"]==128 and e["event"]=="central")
    block=next(e for e in events if e["family"]=="NEG" and e["d"]==128 and e["event"]=="equal_block_sign")
    tests=dict(NEG_d128_K100_strict_local_trap=1+r128<100/99,
        NEG_d128_TC_lower_gt_005=central["simultaneous_information_lower"]>.005,
        NEG_d128_block_MI_lower_gt_005=block["simultaneous_information_lower"]>.005,
        both_POS_pair_matched_d128_K100_have_descent=1+127*r128>100/99)
    write(OUT/"DECISION.json",dict(time_utc=now(),primary_tests=tests,primary_pass=all(tests.values()),
        NEG_d128_TC_lower=central["simultaneous_information_lower"],
        NEG_d128_block_MI_lower=block["simultaneous_information_lower"],
        NEG_d512_SIR_lower_by_K={str(b["K"]):b["SIR_forward_KL_lower"] for b in bounds if b["d"]==512},
        original_M_pass=False,original_M_observations=0,originality_certified=False,
        natural_confirmation=False,finite_MC_training_gap_or_semantic_risk_bound_claimed=False))
    write(OUT/"AUDIT.json",dict(time_utc=now(),passed=all(checks.values()),checks=checks,
        cells=12,independent_streams=96,observations=3145728,simultaneous_event_intervals=24,
        elapsed_seconds=time.perf_counter()-started,all_parameters_and_seeds_frozen=True,
        neural_models_fitted=0,raw_labels_saved=False,old_M_preserved_with_failed_numerical_gate=True,
        reused_unchanged_simulator_sha256=sha(legacy.__file__)))
    write(OUT/"RUN_STATE.json",dict(status="complete",active_jobs=[],time_utc=now()),exclusive=False)
    paths=sorted(p for p in OUT.rglob("*") if p.is_file())
    write(OUT/"MANIFEST.json",dict(time_utc=now(),files={p.relative_to(ROOT).as_posix():sha(p) for p in paths}))
    check()

def check():
    check_freeze()
    for name,h in read(OUT/"MANIFEST.json")["files"].items():
        assert sha(ROOT/name)==h,name
    assert read(OUT/"AUDIT.json")["passed"]
    assert read(OUT/"PARAMETERS.json")["all_numerical_pass"]
    with (OUT/"EVENT_INTERVALS.csv").open(encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream)))==24
    print(json.dumps(dict(pass_frozen_M2=True,decision=read(OUT/"DECISION.json")),ensure_ascii=False),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=["freeze","run","check"])
    args=parser.parse_args()
    try:
        globals()[args.phase]()
    except Exception:
        if args.phase=="run" and OUT.exists():
            write(OUT/f"FAILURE_{time.time_ns()}.json",dict(time_utc=now(),phase=args.phase,
                traceback=traceback.format_exc(),old_M_unchanged=True))
        raise
