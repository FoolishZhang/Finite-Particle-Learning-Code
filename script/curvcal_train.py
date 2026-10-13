"""Fresh observations and nested prior prefixes across four fixed cases.

check/bench use deterministic arithmetic only. Scientific run requires the
root-authored source-bound FREEZE. Imported old helpers are pure calculations
and JSON writing to this package only; no old generator/runner is invoked.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
import multiprocessing as mp
from fractions import Fraction
import json
import os
from pathlib import Path
import time

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from scipy.special import expit

import fincal_train as old
import bankvar_train as bank

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/calibration_precision/TRAIN"
METHODS = ("raw", "marg", "asymquad", "fullfinite", "localquad", "exact")
CASES = [dict(n=n, K=K) for n in (62, 254) for K in (64, 256)]
DEFAULTS = dict(a=1., R=.75, N=4096, streams=16, cases=CASES,
                bank_prefixes=[128, 512], max_B=512, max_K=256, max_n=254,
                grid_points=801, workers=2, teachers=[0., .3, .6],
                data_seed_base=202610022200, prior_seed_base=202610022300,
                bank_block=128, rho_batch=16, chebyshev_degree=192,
                ellipse_q_numerator=28, ellipse_q_denominator=25,
                active_radius_bound=7.)


def case_key(case):
    return f"n{case['n']}_K{case['K']}"


def local_coefficient(model, case):
    row = next(c for c in model["cases"] if c["n"] == case["n"] and c["K"] == case["K"])
    answer = float(row["coefficient"])
    assert np.isfinite(answer)
    return answer


def model_interfaces(reference_path, local_path, config):
    local = json.loads(Path(local_path).read_text("utf-8"))
    assert local["a"] == config["a"] and local["R"] == config["R"]
    result = {}
    first_grid, first_r = None, None
    for case in config["cases"]:
        cc = dict(config, **case)
        ref, rfun, _, gfun = old.load_reference(reference_path, cc)
        master = np.asarray(ref["H_knots"])
        assert len(master) == config["grid_points"] and master[len(master)//2] == 0.
        assert set(ref["gprime_knots"]).issubset(set(master))
        if first_grid is None:
            first_grid, first_r = master.copy(), rfun(master)
        else:
            assert np.array_equal(master, first_grid) and np.array_equal(rfun(master), first_r)
        A, d, w = (float(ref["constants"][k]) for k in ("A", "d", "w"))
        coefficients = dict(asymquad=A*d*w, localquad=local_coefficient(local, case))
        penalties = dict(raw=np.zeros_like(master), marg=np.zeros_like(master),
                         fullfinite=gfun(master)-ref["g_zero"])
        curvature = dict(raw=0., marg=0., fullfinite=ref["g_second_sup"])
        for method, coefficient in coefficients.items():
            penalties[method] = coefficient*master*master
            curvature[method] = old.positive_product_upper(2., abs(coefficient))
        result[case_key(case)] = dict(reference=ref, rfun=rfun, master=master,
                                     coefficients=coefficients, penalties=penalties,
                                     penalty_curvature=curvature)
    return result


def evaluate_fullN(weights, zactive, y1, y2, nodes, config):
    """Full-N binary loss reduction; exact algebra, no nuisance-row changes.

    Average L_minus plus positive-label indicator matrix times (L_plus-L_minus).
    Both sigmoid branches are evaluated directly by the frozen helper.
    """
    N = config["N"]
    assert y2.shape[1] == N and weights.shape[1] == N
    indicator = (y2 > 0).astype(np.float64)[None, :, :]
    shape = (len(zactive), len(y2), len(nodes))
    raw, marg = np.empty(shape), np.empty(shape)
    for start in range(0, len(nodes), config["rho_batch"]):
        rr = nodes[start:start+config["rho_batch"]]
        logp = -np.log(old.second_probabilities_batch(weights, zactive, rr, config["a"]))
        minus, plus = logp[..., 0], logp[..., 1]
        raw[:, :, start:start+len(rr)] = minus.mean(axis=1)[:, None, :]+np.matmul(indicator, plus-minus)/N
        marg[:, :, start:start+len(rr)] = old.marg_bank_losses_batch(
            zactive, y1, y2, [N], rr, config["a"])[:, :, 0]
    return raw, marg


def generate_fresh_input(config, stream):
    """New experiment's explicit data contract; called only after FREEZE checks."""
    N, n = config["N"], config["max_n"]
    drng = np.random.Generator(np.random.PCG64(config["data_seed_base"]+stream))
    noise_uniform = drng.random((N, n))
    yn = np.where(noise_uniform < .5, -1, 1).astype(np.int8)
    active = drng.standard_normal((N, 2))
    uniform = drng.random((N, 2))
    y1 = np.where(uniform[:, 0] < expit(config["a"]*active[:, 0]), 1, -1).astype(np.int8)
    y2 = np.stack([np.where(uniform[:, 1] < expit(config["a"]*(theta*active[:, 0]+
                  np.sqrt(1-theta*theta)*active[:, 1])), 1, -1)
                   for theta in config["teachers"]]).astype(np.int8)
    prng = np.random.Generator(np.random.PCG64(config["prior_seed_base"]+stream))
    z = prng.standard_normal((config["max_B"], config["max_K"], config["max_n"]+2))
    provenance = dict(data_seed=config["data_seed_base"]+stream,
                      prior_seed=config["prior_seed_base"]+stream,
                      noise_uniform_sha256=old.array_sha(noise_uniform),
                      active_gaussian_sha256=old.array_sha(active),
                      active_uniform_sha256=old.array_sha(uniform),
                      prior_sha256=old.array_sha(z), y1_sha256=old.array_sha(y1),
                      y2_sha256=old.array_sha(y2), ynoise_sha256=old.array_sha(yn))
    return z, y1, yn, y2, provenance


def deterministic_check():
    """Checks all four prefix shapes without invoking any scientific RNG."""
    z, y1, yn, y2 = old.deterministic_inputs(9, 3, 256, 254)
    master, nodes = np.linspace(-.75, .75, 801), bank.chebyshev_nodes()
    matrix = bank.interpolation_matrix(nodes, master)
    errors, path_errors, prefixes = {}, {}, {}
    for case in CASES:
        key = case_key(case)
        zz, yy = z[:, :case["K"], :case["n"]+2], yn[:, :case["n"]]
        cfg = dict(DEFAULTS, N=9, **case)
        weights, _ = old.build_weights(zz, y1, yy)
        nr, nm = evaluate_fullN(weights, zz[:, :, :2], y1, y2, nodes, cfg)
        dr, dm = bank.evaluate_nodes(weights, zz[:, :, :2], y1, y2, master, cfg)
        fr, fm = evaluate_fullN(weights, zz[:, :, :2], y1, y2, master, cfg)
        errors[key] = dict(raw=float(np.max(np.abs(nr@matrix.T-dr))),
                           marg=float(np.max(np.abs(nm@matrix.T-dm))),
                           fullN_raw_vs_old_direct=float(np.max(np.abs(fr-dr))),
                           fullN_marg_vs_old_direct=float(np.max(np.abs(fm-dm))))
        assert max(errors[key].values()) < 3e-12
        query = np.array([-.41, .03, .67])
        gr, gm = bank.path_gradients(weights, zz[:, :, :2], y1, y2, query)
        h, maxima = 2e-5, []
        stencil = np.array([1., -8., 8., -1.])/(12*h)
        for ti, rho in enumerate(query):
            r, m = evaluate_fullN(weights, zz[:, :, :2], y1, y2,
                                  rho+np.array([-2., -1., 1., 2.])*h, cfg)
            maxima.extend([float(np.max(np.abs(r[:, ti]@stencil-gr[:, ti]))),
                           float(np.max(np.abs(m[:, ti]@stencil-gm[:, ti])))])
        path_errors[key] = max(maxima)
        assert max(maxima) < 3e-10
        prefixes[key] = dict(prior_shape=list(zz.shape), noise_shape=list(yy.shape),
                             original_prior_shared=bool(np.shares_memory(z, zz)),
                             original_noise_shared=bool(np.shares_memory(yn, yy)))
    assert bank.active_gate(z[:, :, :2])["pass_all"]
    return dict(passed=True, cases=CASES, score_interpolation_errors=errors,
                path_fourth_order_difference_errors=path_errors, prefixes=prefixes,
                analytic_interpolation_error_upper=bank.interpolation_error_upper(DEFAULTS),
                helper_check=bank.deterministic_check(),
                new_Gaussian_draws=0, new_observations=0, new_scientific_fits=0)


def benchmark():
    destination = OUT/"OPTIMIZED_BENCHMARK.json"
    assert not destination.exists(), "refuse deterministic benchmark overwrite"
    check = deterministic_check()
    z, y1, yn, y2 = old.deterministic_inputs(DEFAULTS["N"], 128, 256, 254)
    # A prespecified, distributed 33-node subset for shape/time estimation.
    nodes = bank.chebyshev_nodes()[::6]
    cases = []
    for case in CASES:
        zz, yy = z[:, :case["K"], :case["n"]+2], yn[:, :case["n"]]
        config = dict(DEFAULTS, **case)
        start = time.perf_counter()
        weights, offsets = old.build_weights(zz, y1, yy)
        wsec = time.perf_counter()-start
        start = time.perf_counter()
        raw, marg = evaluate_fullN(weights, zz[:, :, :2], y1, y2, nodes, config)
        scoresec = time.perf_counter()-start
        start = time.perf_counter()
        baseline_raw, baseline_marg = bank.evaluate_nodes(weights, zz[:, :, :2], y1, y2, nodes, config)
        baseline_scoresec = time.perf_counter()-start
        raw_difference = float(np.max(np.abs(raw-baseline_raw)))
        marg_difference = float(np.max(np.abs(marg-baseline_marg)))
        assert max(raw_difference, marg_difference) <= 3e-12
        start = time.perf_counter()
        gradient = bank.path_gradients(weights, zz[:, :, :2], y1, y2, np.array([0., .3, .6]))
        gradsec = time.perf_counter()-start
        per_block = wsec+scoresec*193/len(nodes)+gradsec
        cases.append(dict(**case, bank_block=128, N=DEFAULTS["N"], nodes=len(nodes),
                          weight_seconds=wsec, score_seconds=scoresec, gradient_seconds=gradsec,
                          same_input_old_score_seconds=baseline_scoresec,
                          raw_max_difference_from_old=raw_difference,
                          marg_max_difference_from_old=marg_difference,
                          projected_case_B512_seconds=4*per_block,
                          weights_bytes=weights.nbytes,
                          raw_checksum=float(raw.sum()), marg_checksum=float(marg.sum()),
                          gradient_checksum=float(sum(g.sum() for g in gradient))))
        del weights
    per_flow = sum(c["projected_case_B512_seconds"] for c in cases)
    result = dict(source_sha256=old.sha(__file__), helper_sha256=old.sha(old.__file__),
                  bank_helper_sha256=old.sha(bank.__file__), check=check, cases=cases,
                  baseline_source_sha256=old.sha(OUT/"BASELINE_IMPLEMENTATION.py"),
                  baseline_benchmark_sha256=old.sha(OUT/"BENCHMARK.json"),
                  paired_scoring_faster=bool(sum(c["score_seconds"] for c in cases)<sum(c["same_input_old_score_seconds"] for c in cases)),
                  projected_four_case_stream_seconds=per_flow,
                  projected_16_stream_seconds=16*per_flow,
                  peak_weights_bytes=max(c["weights_bytes"] for c in cases),
                  new_Gaussian_draws=0, new_observations=0, new_scientific_fits=0,
                  timing_scope="Bounded deterministic arrays; costs exclude serialization and system contention, not a scientific result or runtime guarantee.")
    old.save(destination, result)
    print(json.dumps(result), flush=True)


def verified_context(expected_freeze_sha=None):
    frozen_path = OUT/"FREEZE.json"
    if expected_freeze_sha is not None:
        assert old.sha(frozen_path) == expected_freeze_sha
    freeze = json.loads(frozen_path.read_text("utf-8"))
    assert freeze["run_authorized"] is True
    assert freeze["source_sha256"] == old.sha(__file__)
    assert freeze["helper_sha256"] == old.sha(old.__file__)
    assert freeze["bank_helper_sha256"] == old.sha(bank.__file__)
    bank.verify_file(freeze["plan_path"], freeze["plan_sha256"])
    config = freeze["config"]
    for key, value in DEFAULTS.items():
        assert config[key] == value, (key, config[key], value)
    refpath = bank.verify_file(freeze["reference_path"], freeze["reference_sha256"])
    localpath = bank.verify_file(freeze["local_model_path"], freeze["local_model_sha256"])
    models = model_interfaces(refpath, localpath, config)
    first = models[case_key(config["cases"][0])]
    master, rfun = first["master"], first["rfun"]
    h = old.mesh_width_upper(master)
    nodes = bank.chebyshev_nodes(config["R"], config["chebyshev_degree"])
    matrix = bank.interpolation_matrix(nodes, master)
    eps = bank.interpolation_error_upper(config)
    assert os.environ["OPENBLAS_NUM_THREADS"] == os.environ["OMP_NUM_THREADS"] == "1"
    return dict(frozen_path=frozen_path, config=config, refpath=refpath, localpath=localpath,
                models=models, master=master, rfun=rfun, h=h, nodes=nodes, matrix=matrix, eps=eps)

_WORKER_CONTEXT = None
_SMOKE_BARRIER = None


def initialize_worker(freeze_sha):
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = verified_context(freeze_sha)


def run_flow(stream):
    assert _WORKER_CONTEXT is not None
    context = _WORKER_CONTEXT
    config, models = context["config"], context["models"]
    master, rfun = context["master"], context["rfun"]
    h, nodes, matrix, eps = (context[k] for k in ("h", "nodes", "matrix", "eps"))
    start_flow = time.perf_counter()
    assert not list(OUT.glob(f"stream_{stream:02}*.npz")), "own flow exists; no retry"
    assert not (OUT/f"data_{stream:02}.npz").exists(), "own data exists; no retry"
    z, y1, yn, y2, provenance = generate_fresh_input(config, stream)
    # One gate for the entire maximum prior flow. A failure keeps the input
    # and sends every n/K case in this flow to direct801.
    gate = bank.active_gate(z[:, :, :2], config["active_radius_bound"])
    use_nodes = nodes if gate["pass_all"] else master
    interp = eps if gate["pass_all"] else 0.
    mle, boundary, pairs = bank.mle_estimates(y1, y2, rfun, config["R"])
    datapath = OUT/f"data_{stream:02}.npz"
    np.savez_compressed(datapath, y1=y1, ynoise=yn, y2=y2,
                        prior_active=z[:, :, :2], gradient_rho=mle,
                        gradient_MLE_boundary=boundary, teachers=np.array(config["teachers"]))
    case_records = []
    for case in config["cases"]:
        started = time.perf_counter()
        key, cfg = case_key(case), dict(config, **case)
        zz, yy = z[:, :case["K"], :case["n"]+2], yn[:, :case["n"]]
        model = models[key]
        shape = (config["max_B"], len(config["teachers"]), len(master))
        raw, marg = np.empty(shape), np.empty(shape)
        ns = shape[:2]+(len(use_nodes),)
        raw_nodes, marg_nodes = np.empty(ns), np.empty(ns)
        raw_grad, marg_grad = np.empty(shape[:2]), np.empty(shape[:2])
        offsets = np.empty((config["max_B"], config["N"]))
        blocks = []
        for begin in range(0, config["max_B"], config["bank_block"]):
            tb = time.perf_counter()
            end = min(config["max_B"], begin+config["bank_block"])
            zb = zz[begin:end]
            weights, offset = old.build_weights(zb, y1, yy, config["a"])
            nr, nm = evaluate_fullN(weights, zb[:, :, :2], y1, y2, use_nodes, cfg)
            raw_nodes[begin:end], marg_nodes[begin:end] = nr, nm
            if gate["pass_all"]:
                raw[begin:end], marg[begin:end] = nr@matrix.T, nm@matrix.T
            else:
                raw[begin:end], marg[begin:end] = nr, nm
            raw_grad[begin:end], marg_grad[begin:end] = bank.path_gradients(
                weights, zb[:, :, :2], y1, y2, mle, config["a"])
            offsets[begin:end] = offset
            del weights
            blocks.append(time.perf_counter()-tb)
        zero = len(master)//2
        raw -= raw[:, :, zero:zero+1].copy()
        marg -= marg[:, :, zero:zero+1].copy()
        assert np.isfinite(raw).all() and np.isfinite(marg).all()
        rows, curvatures = [], {}
        for B in config["bank_prefixes"]:
            baseD = old.curvature_bound(zz[:B, :, :2], config["a"], config["R"])
            curvatures[str(B)] = baseD
            for ti, theta in enumerate(config["teachers"]):
                teacher_r = float(model["rfun"](theta))
                for method in METHODS:
                    row = dict(stream=stream, case=key, **case, theta=theta,
                               N=config["N"], B=B, method=method,
                               empirical_pair=float(pairs[ti]), teacher_r=teacher_r)
                    if method == "exact":
                        estimate = float(mle[ti])
                        row.update(reused_same_data_MLE=True, MLE_at_boundary=bool(boundary[ti]))
                    else:
                        curve = marg[:B, ti] if method == "marg" else raw[:B, ti]
                        scores = curve.mean(axis=0)+model["penalties"][method]
                        chosen = int(np.argmin(scores))
                        estimate = float(master[chosen])
                        D = float(old.up_add(baseD, model["penalty_curvature"][method]))
                        mb = old.mesh_regret_upper(D, h)
                        total = old.outward_float(Fraction.from_float(mb)+2*Fraction.from_float(interp))
                        row.update(grid_index=chosen, selected_score=float(scores[chosen]),
                            curvature_D=D, mesh_step_upper=h,
                            mesh_regret_bound_real_arithmetic=mb,
                            interpolation_uncentered_sup_error_upper=interp,
                            interpolation_minimizer_regret_upper=2*interp,
                            continuous_objective_regret_bound_real_arithmetic=total,
                            bank_score_SE=float(curve[:, chosen].std(ddof=1)/np.sqrt(B)),
                            certificate_scope="Real-arithmetic finite-bank score plus stored penalty: D h²/8 + 2 interpolation epsilon. Excludes floating-point zeta, CAL error and MC error.")
                    row.update(rho=estimate, observed_r=float(model["rfun"](estimate)),
                               reference_joint_KL=old.observed_kl(teacher_r, float(model["rfun"](estimate))))
                    rows.append(row)
        dest = OUT/f"stream_{stream:02}_{key}.npz"
        np.savez_compressed(dest, raw=raw, marg=marg, rho=master,
            raw_node_scores=raw_nodes, marg_node_scores=marg_nodes, nodes=use_nodes,
            offset=offsets, raw_path_gradient=raw_grad, marg_path_gradient=marg_grad,
            gradient_rho=mle, gradient_MLE_boundary=boundary,
            teachers=np.array(config["teachers"]), bank_prefixes=np.array(config["bank_prefixes"]))
        record = dict(stream=stream, case=key, **case,
                      file=dest.relative_to(ROOT).as_posix(), file_sha256=old.sha(dest),
                      prior_prefix_sha256=old.array_sha(zz), noise_prefix_sha256=old.array_sha(yy),
                      base_curvature_D=curvatures, outputs=rows, block_seconds=blocks,
                      elapsed_seconds=time.perf_counter()-started)
        old.save(OUT/f"stream_{stream:02}_{key}.json", record)
        case_records.append(record)
        print(json.dumps(dict(completed_stream=stream, completed_case=key,
                              seconds=record["elapsed_seconds"])), flush=True)
        del raw, marg, raw_nodes, marg_nodes, raw_grad, marg_grad, offsets
    flow = dict(stream=stream, worker_pid=os.getpid(), blas_threads=1, provenance=provenance,
                data_file=datapath.relative_to(ROOT).as_posix(), data_file_sha256=old.sha(datapath),
                active_gate=gate, evaluation_mode="chebyshev193" if gate["pass_all"] else "direct801",
                gradient_rho=mle.tolist(), gradient_MLE_boundary=boundary.tolist(),
                cases=case_records, elapsed_seconds=time.perf_counter()-start_flow)
    old.save(OUT/f"stream_{stream:02}.json", flow)
    return flow


def run():
    context = verified_context()
    frozen_path, config = context["frozen_path"], context["config"]
    refpath, localpath, models = context["refpath"], context["localpath"], context["models"]
    assert not (OUT/"RESULTS.json").exists(), "refuse scientific overwrite"
    assert not list(OUT.glob("stream_*.npz")), "resume needs separate root review"
    assert not list(OUT.glob("data_*.npz")), "resume needs separate root review"
    records_by_stream, start_all = {}, time.perf_counter()
    # At most two submitted flows: no backlog continues after an observed error.
    # On error, the already-active other flow is allowed to finish, never retried.
    with ProcessPoolExecutor(max_workers=config["workers"],
            mp_context=mp.get_context("spawn"), initializer=initialize_worker,
            initargs=(old.sha(frozen_path),)) as pool:
        pending = {pool.submit(run_flow, i):i for i in range(config["workers"])}
        next_stream = config["workers"]
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            completed = []
            for future in sorted(done, key=lambda f: pending[f]):
                stream = pending.pop(future)
                try:
                    flow = future.result()
                except BaseException as error:
                    for remaining in pending:
                        remaining.cancel()
                    old.save(OUT/"PROGRESS.json", dict(status="failed_no_retry",
                        failed_stream=stream, error_type=type(error).__name__, error=str(error),
                        completed_stream_indices=sorted(records_by_stream),
                        active_other_streams=list(pending.values()),
                        elapsed_seconds=time.perf_counter()-start_all))
                    raise
                assert flow["stream"] == stream and stream not in records_by_stream
                records_by_stream[stream] = flow
                completed.append(stream)
            old.save(OUT/"PROGRESS.json", dict(status="running",
                completed_streams=len(records_by_stream), completed_cases=4*len(records_by_stream),
                completed_stream_indices=sorted(records_by_stream),
                elapsed_seconds=time.perf_counter()-start_all))
            for stream in completed:
                print(json.dumps(dict(completed_stream=stream,
                    flow_seconds=records_by_stream[stream]["elapsed_seconds"],
                    evaluation_mode=records_by_stream[stream]["evaluation_mode"])), flush=True)
            for _ in completed:
                if next_stream < config["streams"]:
                    pending[pool.submit(run_flow, next_stream)] = next_stream
                    next_stream += 1
    assert sorted(records_by_stream) == list(range(config["streams"]))
    records = [records_by_stream[i] for i in range(config["streams"])]
    old.save(OUT/"RESULTS.json", dict(schema=1, source_sha256=old.sha(__file__),
        helper_sha256=old.sha(old.__file__), bank_helper_sha256=old.sha(bank.__file__),
        freeze_sha256=old.sha(frozen_path), reference_sha256=old.sha(refpath),
        local_model_sha256=old.sha(localpath), config=config, methods=METHODS, records=records,
        coefficients={key:m["coefficients"] for key,m in models.items()},
        penalty_curvature_D={key:m["penalty_curvature"] for key,m in models.items()},
        reference_interfaces={key:m["reference"] for key,m in models.items()},
        elapsed_seconds=time.perf_counter()-start_all,
        output_count=sum(len(c["outputs"]) for r in records for c in r["cases"]),
        grid_selection_count=1920, repeated_MLE_rows=384, unique_data_teacher_MLE=48,
        maximum_prior_banks=8192, maximum_prior_vectors=2097152, prior_Gaussian_scalars=536870912,
        new_base_observation_rows=65536, teacher_observation_vectors=196608,
        data_Gaussian_scalars=131072, data_uniform_scalars=16777216,
        independent_base_streams=16,
        tie_rule="First exact computed-score minimum on ascending fixed grid; never teacher-risk dependent.",
        qualification="Four cases and both B values share fresh data and maximum prior prefixes within 16 streams. Calibrations frozen before training. Reference risks and MLE are numerical; real-arithmetic mesh/interpolation bound excludes zeta and statistical errors."))
    old.save(OUT/"PROGRESS.json", dict(status="complete", completed_streams=len(records),
        completed_cases=4*len(records), completed_stream_indices=list(range(len(records))),
        elapsed_seconds=time.perf_counter()-start_all))


def smoke_payload(index):
    z, y1, yn, y2 = old.deterministic_inputs(9, 2, 7, 4)
    z = z+index*.03125
    weights, _ = old.build_weights(z, y1, yn)
    cfg = dict(DEFAULTS, N=9, n=4, K=7)
    raw, marg = evaluate_fullN(weights, z[:, :, :2], y1, y2, np.array([-.75, -.2, 0., .3, .75]), cfg)
    return dict(index=index, raw_sha256=old.array_sha(raw), marg_sha256=old.array_sha(marg))


def initialize_smoke(barrier):
    global _SMOKE_BARRIER
    _SMOKE_BARRIER = barrier


def smoke_job(index):
    _SMOKE_BARRIER.wait(timeout=45)
    return dict(**smoke_payload(index), pid=os.getpid(),
                blas_threads=os.environ["OPENBLAS_NUM_THREADS"], omp_threads=os.environ["OMP_NUM_THREADS"])


def parallel_check():
    destination = OUT/"PARALLEL_SELFTEST.json"
    assert not destination.exists(), "refuse parallel selftest overwrite"
    expected = [smoke_payload(i) for i in range(4)]
    context = mp.get_context("spawn")
    barrier = context.Barrier(2)
    with ProcessPoolExecutor(max_workers=2, mp_context=context,
            initializer=initialize_smoke, initargs=(barrier,)) as pool:
        actual = list(pool.map(smoke_job, range(4)))
    assert len({r["pid"] for r in actual}) == 2
    for wanted, got in zip(expected, actual):
        for key, value in wanted.items():
            assert got[key] == value
        assert got["blas_threads"] == got["omp_threads"] == "1"
    result = dict(passed=True, source_sha256=old.sha(__file__), workers=2, start_method="spawn",
                  worker_pids=sorted({r["pid"] for r in actual}), outputs=actual,
                  serial_parallel_exact_array_hash_match=True, new_Gaussian_draws=0,
                  new_observations=0, new_scientific_fits=0,
                  scope="Four deterministic small-array tasks; each worker single BLAS thread. Tests process dispatch/order only, not scientific RNG or full-size runtime.")
    old.save(destination, result)
    print(json.dumps(result), flush=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["check", "bench", "parallel-check", "run"])
    args = parser.parse_args()
    if args.mode == "check":
        print(json.dumps(deterministic_check()), flush=True)
    elif args.mode == "bench":
        benchmark()
    elif args.mode == "parallel-check":
        parallel_check()
    else:
        run()
