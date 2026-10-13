"""Nested fresh-prior experiment on frozen complete observations.

Only ``check`` and ``bench`` are RNG-free. Scientific execution requires a
root-authored FREEZE plus explicit ``run``. Old fincal helpers are read-only
dependencies; this module never invokes their scientific generators or runners.
The shared JSON writer is used only for the new package's output paths.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

import fincal_train as old

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/bank_precision/TRAIN"
METHODS = ("raw", "marg", "asymquad", "fullfinite", "oldTaylor", "localquad", "exact")
DEFAULTS = dict(a=1., R=.75, n=254, K=256, N=4096, streams=16,
                bank_prefixes=[32, 128, 512], max_B=512, grid_points=801,
                teachers=[0., .3, .6], prior_seed_base=202610021800,
                bank_block=128, rho_batch=16, chebyshev_degree=192,
                ellipse_q_numerator=28, ellipse_q_denominator=25,
                active_radius_bound=7.)


def interpolation_error_upper(config):
    """4 M q^-m/(q-1), evaluated over exact rationals and rounded upward."""
    assert config["a"] == 1. and config["R"] == .75
    assert config["active_radius_bound"] == 7.
    assert config["chebyshev_degree"] == 192
    q = Fraction(config["ellipse_q_numerator"], config["ellipse_q_denominator"])
    assert q == Fraction(28, 25)
    return old.outward_float(Fraction(4*21)/(q**192*(q-1)))


def chebyshev_nodes(R=.75, degree=192):
    assert degree % 2 == 0
    # Construct exact sign symmetry, including the exact candidate zero.
    positive = R*np.cos(np.pi*np.arange(degree//2)/degree)
    nodes = np.concatenate((-positive, [0.], positive[::-1]))
    nodes[0], nodes[-1] = -R, R
    assert len(nodes) == degree+1 and np.all(np.diff(nodes) > 0)
    return nodes


def interpolation_matrix(nodes, query):
    """Deterministic Lobatto barycentric matrix, no SciPy random permutation."""
    degree = len(nodes)-1
    weights = np.where(np.arange(degree+1) % 2, -1., 1.)
    weights[[0, -1]] *= .5
    matrix = np.empty((len(query), len(nodes)))
    for i, q in enumerate(query):
        hits = np.flatnonzero(nodes == q)
        if len(hits):
            matrix[i] = 0.
            matrix[i, hits[0]] = 1.
        else:
            terms = weights/(q-nodes)
            matrix[i] = terms/terms.sum()
    assert np.isfinite(matrix).all()
    return matrix


def active_gate(zactive, radius=7.):
    """Outward floating operations; an ambiguous boundary fails to direct801."""
    sq = old.up_add(old.up_mul(np.abs(zactive[..., 0]), np.abs(zactive[..., 0])),
                    old.up_mul(np.abs(zactive[..., 1]), np.abs(zactive[..., 1])))
    maximum = float(sq.max())
    threshold = float(Fraction.from_float(radius)**2)
    return dict(pass_all=bool(maximum <= threshold),
                max_squared_radius_upper=maximum, radius_bound=radius,
                squared_threshold=threshold,
                scope="Stored binary active pairs; each square and addition rounded outward with nextafter. Failure selects direct801 without inspecting objective values.")


def evaluate_nodes(weights, zactive, y1, y2, nodes, config):
    shape = (len(zactive), len(y2), len(nodes))
    raw, marg = np.empty(shape), np.empty(shape)
    for start in range(0, len(nodes), config["rho_batch"]):
        rr = nodes[start:start+config["rho_batch"]]
        raw[:, :, start:start+len(rr)] = old.raw_bank_losses_batch(
            weights, zactive, y2, [config["N"]], rr, config["a"])[:, :, 0]
        marg[:, :, start:start+len(rr)] = old.marg_bank_losses_batch(
            zactive, y1, y2, [config["N"]], rr, config["a"])[:, :, 0]
    return raw, marg


def path_gradients(weights, zactive, y1, y2, rho_per_teacher, a=1.):
    """Actual fixed-bank path gradients, one data-dependent MLE per teacher.

    The nuisance and first-bit weights are rho-independent. Each derivative
    averages -p'/p over the original complete data rows, not Price identities.
    Marginalized derivatives use their four-pattern empirical frequencies.
    """
    raw = np.empty((len(zactive), len(y2)))
    marg = np.empty_like(raw)
    z1, z2 = zactive[:, :, 0], zactive[:, :, 1]
    for ti, rho in enumerate(rho_per_teacher):
        x = a*(rho*z1+np.sqrt(1-rho*rho)*z2)
        dx = a*(z1-rho/np.sqrt(1-rho*rho)*z2)
        probs = np.stack((expit(-x), expit(x)), axis=-1)
        deriv = probs*(1-probs)*dx[:, :, None]*np.array([-1., 1.])[None, None, :]
        p, dp = np.matmul(weights, probs), np.matmul(weights, deriv)
        assert np.all(p > 0) and np.isfinite(dp).all()
        point_gradient = -dp/p
        raw[:, ti] = np.where(y2[ti][None, :] > 0, point_gradient[:, :, 1],
                              point_gradient[:, :, 0]).mean(axis=1)
        pattern = 2*(y1 > 0)+(y2[ti] > 0).astype(np.int8)
        freq = np.bincount(pattern, minlength=4)/len(y1)
        values = np.empty((len(zactive), 4))
        for si, (s1, s2) in enumerate(old.SIGNS):
            first = expit(a*s1*z1)
            index = int(s2 > 0)
            active_p = np.mean(first*probs[:, :, index], axis=1)
            active_dp = np.mean(first*deriv[:, :, index], axis=1)
            values[:, si] = -active_dp/active_p
        marg[:, ti] = values @ freq
    assert np.isfinite(raw).all() and np.isfinite(marg).all()
    return raw, marg


def mle_estimates(y1, y2, rfun, R):
    out, flags, pairs = [], [], []
    lo, hi = float(rfun(-R)), float(rfun(R))
    for second in y2:
        pair = float(np.mean(y1*second))
        target = float(np.clip(pair, lo, hi))
        if target == lo:
            point, flag = -R, True
        elif target == hi:
            point, flag = R, True
        else:
            point = float(brentq(lambda rho: float(rfun(rho))-target,
                                 -R, R, xtol=1e-13, rtol=1e-13))
            flag = False
        out.append(point)
        flags.append(flag)
        pairs.append(pair)
    return np.array(out), np.array(flags), np.array(pairs)


def deterministic_check():
    config = dict(DEFAULTS, N=9, K=7, n=4)
    z, y1, yn, y2 = old.deterministic_inputs(9, 3, 7, 4)
    weights, _ = old.build_weights(z, y1, yn)
    master = np.linspace(-.75, .75, 801)
    nodes = chebyshev_nodes()
    matrix = interpolation_matrix(nodes, master)
    nr, nm = evaluate_nodes(weights, z[:, :, :2], y1, y2, nodes, config)
    direct_raw, direct_marg = evaluate_nodes(weights, z[:, :, :2], y1, y2, master, config)
    ir, im = nr @ matrix.T, nm @ matrix.T
    errors = dict(raw=float(np.max(np.abs(ir-direct_raw))),
                  marg=float(np.max(np.abs(im-direct_marg))))
    assert max(errors.values()) < 2e-12, errors
    at = np.array([-.41, .03, .67])
    gr, gm = path_gradients(weights, z[:, :, :2], y1, y2, at)
    fd_raw, fd_marg = [], []
    step = 2e-5
    for ti, rho in enumerate(at):
        # Symmetric fourth-order derivative gives an independent numerical
        # check of both path expressions without any fresh scientific RNG.
        rr = np.array([rho-2*step, rho-step, rho+step, rho+2*step])
        vr, vm = evaluate_nodes(weights, z[:, :, :2], y1, y2, rr, config)
        stencil = np.array([1., -8., 8., -1.])/(12*step)
        fd_raw.append(float(np.max(np.abs(vr[:, ti] @ stencil-gr[:, ti]))))
        fd_marg.append(float(np.max(np.abs(vm[:, ti] @ stencil-gm[:, ti]))))
    assert max(fd_raw+fd_marg) < 3e-10
    gate = active_gate(z[:, :, :2])
    assert gate["pass_all"]
    boundary = np.array([[[7., 0.]]])
    # Exact-radius entries may fail conservatively, which is safe.
    assert not active_gate(boundary)["pass_all"]
    assert not active_gate(np.array([[[7., 1e-3]]]))["pass_all"]
    assert active_gate(np.array([[[6.99, 0.]]]))["pass_all"]
    eps = interpolation_error_upper(config)
    exact_eps = Fraction(84)/(Fraction(28, 25)**192*Fraction(3, 25))
    assert Fraction.from_float(eps) >= exact_eps
    return dict(pass_all=True, old_helper_checks=old.deterministic_check(),
                interpolation_max_error=errors, analytic_interpolation_error_upper=eps,
                path_gradient_fourth_order_difference_max=max(fd_raw+fd_marg),
                gradient_test_points=at.tolist(), active_gate=gate,
                equality_boundary_conservatively_rejected=True,
                random_draws=0, scientific_fits=0,
                note="Bounded arithmetic inputs; direct801 vs Cheb193 plus true path gradient check. Floating point zeta is not rigorously certified.")


def benchmark():
    destination = OUT/"BENCHMARK.json"
    assert not destination.exists(), "refuse deterministic benchmark overwrite"
    check = deterministic_check()
    config = dict(DEFAULTS)
    started = time.perf_counter()
    z, y1, yn, y2 = old.deterministic_inputs(config["N"], 128, 256, 254)
    construction = time.perf_counter()-started
    start = time.perf_counter()
    weights, offset = old.build_weights(z, y1, yn)
    weight_seconds = time.perf_counter()-start
    queries = chebyshev_nodes()[::6]
    start = time.perf_counter()
    raw, marg = evaluate_nodes(weights, z[:, :, :2], y1, y2, queries, config)
    query_seconds = time.perf_counter()-start
    start = time.perf_counter()
    gradients = path_gradients(weights, z[:, :, :2], y1, y2, np.array([0., .3, .6]))
    gradient_seconds = time.perf_counter()-start
    per_block = weight_seconds+query_seconds*193/len(queries)+gradient_seconds
    result = dict(schema=1, source_sha256=old.sha(__file__), helper_sha256=old.sha(old.__file__),
                  check=check, deterministic_shape=dict(N=4096, B=128, K=256, n=254),
                  input_seconds=construction, weight_seconds=weight_seconds,
                  queries=len(queries), query_seconds=query_seconds,
                  gradient_seconds=gradient_seconds, projected_block_seconds=per_block,
                  projected_stream_B512_seconds=4*per_block,
                  projected_16_stream_seconds=64*per_block,
                  weights_bytes=weights.nbytes, raw_checksum=float(raw.sum()),
                  marg_checksum=float(marg.sum()), gradient_checksum=float(sum(g.sum() for g in gradients)),
                  random_draws=0, scientific_fits=0,
                  note="Runtime estimate only; bounded deterministic arithmetic, no Gaussian or observation draws.")
    old.save(destination, result)
    print(json.dumps(result), flush=True)


def verify_file(relative, expected):
    path = ROOT/relative
    assert old.sha(path) == expected, f"byte mismatch: {relative}"
    return path


def run():
    frozen_path = OUT/"FREEZE.json"
    freeze = json.loads(frozen_path.read_text("utf-8"))
    assert freeze["run_authorized"] is True
    assert freeze["source_sha256"] == old.sha(__file__)
    assert freeze["helper_sha256"] == old.sha(old.__file__)
    verify_file(freeze["plan_path"], freeze["plan_sha256"])
    config = freeze["config"]
    for key, value in DEFAULTS.items():
        assert config[key] == value, (key, config[key], value)
    assert not (OUT/"RESULTS.json").exists(), "refuse scientific overwrite"
    assert not list(OUT.glob("stream_*.npz")), "resume needs separate root review"
    reference_path = verify_file(freeze["reference_path"], freeze["reference_sha256"])
    local_path = verify_file(freeze["local_model_path"], freeze["local_model_sha256"])
    taylor_path = verify_file(freeze["old_taylor_path"], freeze["old_taylor_sha256"])
    local = json.loads(local_path.read_text("utf-8"))
    local_c = float(local["coefficient"])
    taylor_c = float(json.loads(taylor_path.read_text("utf-8"))["coefficient"])
    assert np.isfinite(local_c) and np.isfinite(taylor_c)
    ref, rfun, _, gfun = old.load_reference(reference_path, config)
    master = np.asarray(ref["H_knots"])
    assert len(master) == config["grid_points"]
    assert set(ref["gprime_knots"]).issubset(set(master))
    h = old.mesh_width_upper(master)
    zero_index = len(master)//2
    assert master[zero_index] == 0.
    nodes = chebyshev_nodes(config["R"], config["chebyshev_degree"])
    matrix = interpolation_matrix(nodes, master)
    A, d, scale = (float(ref["constants"][key]) for key in ("A", "d", "w"))
    asym_c = A*d*scale
    coeff = dict(asymquad=asym_c, oldTaylor=taylor_c, localquad=local_c)
    penalties = dict(raw=np.zeros_like(master), marg=np.zeros_like(master),
                     fullfinite=gfun(master)-ref["g_zero"])
    penalty_D = dict(raw=0., marg=0., fullfinite=ref["g_second_sup"])
    for method, coefficient in coeff.items():
        penalties[method] = coefficient*master*master
        penalty_D[method] = old.positive_product_upper(2., abs(coefficient))
    eps = interpolation_error_upper(config)
    records, total_start = [], time.perf_counter()
    for stream in range(config["streams"]):
        started = time.perf_counter()
        data_ref = next(x for x in freeze["data_inputs"] if x["stream"] == stream)
        data_path = verify_file(data_ref["path"], data_ref["sha256"])
        with np.load(data_path) as data:
            y1 = data["y1"].copy()
            y2 = data["y2"].copy()
            yn = data["ynoise"].copy()
        assert y1.shape == (config["N"],)
        assert y2.shape == (len(config["teachers"]), config["N"])
        assert yn.shape == (config["N"], config["n"])
        mle, boundary, pairs = mle_estimates(y1, y2, rfun, config["R"])
        prng = np.random.Generator(np.random.PCG64(config["prior_seed_base"]+stream))
        z = prng.standard_normal((config["max_B"], config["K"], config["n"]+2))
        gate = active_gate(z[:, :, :2], config["active_radius_bound"])
        use_nodes = nodes if gate["pass_all"] else master
        interpolation_eps = eps if gate["pass_all"] else 0.
        shape = (config["max_B"], len(config["teachers"]), len(master))
        raw, marg = np.empty(shape), np.empty(shape)
        node_shape = (config["max_B"], len(config["teachers"]), len(use_nodes))
        raw_nodes, marg_nodes = np.empty(node_shape), np.empty(node_shape)
        raw_grad, marg_grad = np.empty(shape[:2]), np.empty(shape[:2])
        offsets = np.empty((config["max_B"], config["N"]))
        block_seconds = []
        for start in range(0, config["max_B"], config["bank_block"]):
            block_started = time.perf_counter()
            stop = min(config["max_B"], start+config["bank_block"])
            zz = z[start:stop]
            weights, offset = old.build_weights(zz, y1, yn, config["a"])
            nr, nm = evaluate_nodes(weights, zz[:, :, :2], y1, y2, use_nodes, config)
            raw_nodes[start:stop], marg_nodes[start:stop] = nr, nm
            if gate["pass_all"]:
                raw[start:stop], marg[start:stop] = nr @ matrix.T, nm @ matrix.T
            else:
                raw[start:stop], marg[start:stop] = nr, nm
            raw_grad[start:stop], marg_grad[start:stop] = path_gradients(
                weights, zz[:, :, :2], y1, y2, mle, config["a"])
            offsets[start:stop] = offset
            del weights
            block_seconds.append(time.perf_counter()-block_started)
        raw -= raw[:, :, zero_index:zero_index+1].copy()
        marg -= marg[:, :, zero_index:zero_index+1].copy()
        assert np.isfinite(raw).all() and np.isfinite(marg).all()
        rows, curvature = [], {}
        for B in config["bank_prefixes"]:
            baseD = old.curvature_bound(z[:B, :, :2], config["a"], config["R"])
            curvature[str(B)] = baseD
            for ti, theta in enumerate(config["teachers"]):
                teacher_r = float(rfun(theta))
                for method in METHODS:
                    row = dict(stream=stream, theta=theta, N=config["N"], B=B,
                               K=config["K"], method=method, empirical_pair=float(pairs[ti]),
                               teacher_r=teacher_r)
                    if method == "exact":
                        estimate = float(mle[ti])
                        row.update(reused_exact_baseline=True, MLE_at_boundary=bool(boundary[ti]))
                    else:
                        bankcurve = marg[:B, ti] if method == "marg" else raw[:B, ti]
                        scores = bankcurve.mean(axis=0)+penalties[method]
                        chosen = int(np.argmin(scores))
                        estimate = float(master[chosen])
                        D = float(old.up_add(baseD, penalty_D[method]))
                        mesh_bound = old.mesh_regret_upper(D, h)
                        total_bound = old.outward_float(Fraction.from_float(mesh_bound)+
                                                       2*Fraction.from_float(interpolation_eps))
                        row.update(grid_index=chosen, selected_score=float(scores[chosen]),
                                   curvature_D=D, mesh_step_upper=h,
                                   mesh_regret_bound_real_arithmetic=mesh_bound,
                                   interpolation_uncentered_sup_error_upper=interpolation_eps,
                                   interpolation_minimizer_regret_upper=2*interpolation_eps,
                                   continuous_objective_regret_bound_real_arithmetic=total_bound,
                                   bank_score_SE=float(bankcurve[:, chosen].std(ddof=1)/np.sqrt(B)),
                                   certificate_scope="Stored finite-bank objective plus frozen real spline/coefficients; mesh D h²/8 + 2 interpolation epsilon. Excludes all floating-point zeta, calibration estimation and population error.")
                    row.update(rho=estimate, observed_r=float(rfun(estimate)),
                               reference_joint_KL=old.observed_kl(teacher_r, float(rfun(estimate))))
                    rows.append(row)
        dest = OUT/f"stream_{stream:02}.npz"
        np.savez_compressed(dest, raw=raw, marg=marg, rho=master,
                            raw_node_scores=raw_nodes, marg_node_scores=marg_nodes, nodes=use_nodes,
                            prior_active=z[:, :, :2], offset=offsets,
                            raw_path_gradient=raw_grad, marg_path_gradient=marg_grad,
                            gradient_rho=mle, gradient_MLE_boundary=boundary,
                            teachers=np.array(config["teachers"]), bank_prefixes=np.array(config["bank_prefixes"]))
        record = dict(stream=stream, input_ref=data_ref,
                      provenance=dict(prior_seed=config["prior_seed_base"]+stream,
                                      prior_sha256=old.array_sha(z), y1_sha256=old.array_sha(y1),
                                      y2_sha256=old.array_sha(y2), ynoise_sha256=old.array_sha(yn)),
                      file=dest.relative_to(ROOT).as_posix(), file_sha256=old.sha(dest),
                      active_gate=gate, evaluation_mode="chebyshev193" if gate["pass_all"] else "direct801",
                      gradient_rho=mle.tolist(), gradient_MLE_boundary=boundary.tolist(),
                      base_curvature_D=curvature, outputs=rows,
                      block_seconds=block_seconds, elapsed_seconds=time.perf_counter()-started)
        old.save(OUT/f"stream_{stream:02}.json", record)
        records.append(record)
        old.save(OUT/"PROGRESS.json", dict(completed_streams=len(records), elapsed_seconds=time.perf_counter()-total_start))
        print(json.dumps(dict(completed_stream=stream, seconds=record["elapsed_seconds"],
                              evaluation_mode=record["evaluation_mode"])), flush=True)
        del z, raw, marg, raw_nodes, marg_nodes
    old.save(OUT/"RESULTS.json", dict(schema=1, source_sha256=old.sha(__file__),
             helper_sha256=old.sha(old.__file__), freeze_sha256=old.sha(frozen_path),
             reference_sha256=old.sha(reference_path), local_model_sha256=old.sha(local_path),
             old_taylor_sha256=old.sha(taylor_path), config=config, methods=METHODS,
             records=records, coefficients=coeff, penalty_curvature_D=penalty_D,
             reference_interface=ref, elapsed_seconds=time.perf_counter()-total_start,
             output_count=sum(len(x["outputs"]) for x in records),
             prior_bank_count=config["streams"]*config["max_B"],
             new_observation_count=0, independent_data_streams=config["streams"],
             tie_rule="First exact computed-score minimum on ascending master grid; never truth-dependent.",
             qualification="B prefixes and methods reuse observations and nested banks; 16 independent prior/data streams, shared frozen calibration. Interpolation/mesh bound is real-arithmetic only, not a statistical accuracy certificate."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["check", "bench", "run"])
    args = parser.parse_args()
    if args.mode == "check":
        print(json.dumps(deterministic_check()), flush=True)
    elif args.mode == "bench":
        benchmark()
    else:
        run()
