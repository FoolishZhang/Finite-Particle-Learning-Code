"""Actual shared-full-bank empirical training; deterministic bench is RNG-free.

The fast conditional second-bit loss differs from full raw loss by a parameter-
independent constant, retained by ``build_weights`` for deterministic audits.
Scientific runs require the root-authored TRAIN/FREEZE.json and explicit --run.
No pre-run benchmark selects parameters or produces observations from an RNG.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator, PPoly
from scipy.optimize import brentq
from scipy.special import expit, logsumexp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/finite_calibration/TRAIN"
DEFAULTS = dict(a=1.0, R=.75, n=254, K=256, streams=16,
                sample_sizes=[1024, 4096], B=128, grid_points=801,
                teachers=[0.0, .3, .6], data_seed_base=202610021300,
                prior_seed_base=202610021400)
METHODS = ("raw", "marg", "quad", "asymcorr", "finitecal", "exact", "zero")
SIGNS = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.int8)
BATCH_RHO = 16


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)
                    + "\n", encoding="utf-8", newline="\n")


def build_weights(z, y1, ynoise, a=1.0):
    """Return B,N,K normalized weights and rho-independent full-loss offset.

    log sigma(a*y*z)=a*y*z/2-log(2*cosh(a*z/2)).  The observed
    nuisance signs are retained in the full matrix product.  At no stage are
    data rows canonicalized to a common all-positive nuisance observation.
    """
    B, K, D = z.shape
    N = len(y1)
    assert ynoise.shape == (N, D-2)
    w = np.empty((B, N, K), dtype=np.float64)
    offset = np.empty((B, N), dtype=np.float64)
    yfloat = np.asarray(ynoise, dtype=np.float64)
    for b in range(B):
        logw = (a/2) * (yfloat @ z[b, :, 2:].T)
        logw -= np.logaddexp(a*z[b, :, 2:]/2, -a*z[b, :, 2:]/2).sum(axis=1)
        logw -= np.logaddexp(0., -a*y1[:, None]*z[b, None, :, 0])
        norm = logsumexp(logw, axis=1)
        offset[b] = np.log(K)-norm
        np.subtract(logw, norm[:, None], out=logw)
        np.exp(logw, out=w[b])
    return w, offset


def second_probabilities(w, zactive, rho, a=1.0):
    x = a*(rho*zactive[:, :, 0] + np.sqrt(1-rho*rho)*zactive[:, :, 1])
    # Evaluate both branches directly.  Computing 1-p_plus could lose small
    # probabilities to cancellation and change the numerical loss contract.
    probs = np.stack((expit(-x), expit(x)), axis=-1)
    out = np.matmul(w, probs)
    if not (np.isfinite(out).all() and np.all(out > 0) and np.all(out <= 1+2e-14)):
        raise FloatingPointError("nonpositive or invalid conditional probability")
    return out


def second_probabilities_batch(w, zactive, rho, a=1.0):
    rho = np.asarray(rho, dtype=np.float64)
    x = a*(zactive[:, :, 0, None]*rho[None, None, :] +
           zactive[:, :, 1, None]*np.sqrt(1-rho*rho)[None, None, :])
    probs = np.stack((expit(-x), expit(x)), axis=-1)
    out = np.matmul(w, probs.reshape(len(zactive), zactive.shape[1], -1))
    out = out.reshape(len(zactive), w.shape[1], len(rho), 2)
    if not (np.isfinite(out).all() and np.all(out > 0) and np.all(out <= 1+2e-14)):
        raise FloatingPointError("nonpositive or invalid batched conditional probability")
    return out


def raw_bank_losses_batch(w, zactive, y2, sizes, rho, a=1.0):
    logp = -np.log(second_probabilities_batch(w, zactive, rho, a))
    plus, minus = logp[:, :, :, 1], logp[:, :, :, 0]
    out = np.empty((len(w), len(y2), len(sizes), len(rho)))
    for ti, second in enumerate(y2):
        rows = np.where(second[None, :, None] > 0, plus, minus)
        sums = np.cumsum(rows, axis=1)
        for ni, N in enumerate(sizes):
            out[:, ti, ni, :] = sums[:, N-1, :]/N
    return out


def raw_bank_losses(w, zactive, y2, sizes, rho, a=1.0):
    """B,teacher,Nprefix means; each full observation and bank retained."""
    logp = -np.log(second_probabilities(w, zactive, rho, a))
    plus, minus = logp[:, :, 1], logp[:, :, 0]
    out = np.empty((len(w), len(y2), len(sizes)))
    for ti, second in enumerate(y2):
        rows = np.where(second[None, :] > 0, plus, minus)
        sums = np.cumsum(rows, axis=1)
        for ni, N in enumerate(sizes):
            out[:, ti, ni] = sums[:, N-1]/N
    return out


def marg_bank_losses(zactive, y1, y2, sizes, rho, a=1.0):
    """Exact empirical four-pattern reduction after nuisance marginalization."""
    z1 = a*zactive[:, :, 0]
    x2 = a*(rho*zactive[:, :, 0] + np.sqrt(1-rho*rho)*zactive[:, :, 1])
    logs = np.empty((len(zactive), 4))
    for si, (s1, s2) in enumerate(SIGNS):
        act = -np.logaddexp(0., -s1*z1)-np.logaddexp(0., -s2*x2)
        logs[:, si] = np.log(zactive.shape[1])-logsumexp(act, axis=1)
    out = np.empty((len(zactive), len(y2), len(sizes)))
    for ti, second in enumerate(y2):
        pattern = 2*(y1 > 0)+(second > 0).astype(np.int8)
        for ni, N in enumerate(sizes):
            freq = np.bincount(pattern[:N], minlength=4)/N
            out[:, ti, ni] = logs @ freq
    return out


def marg_bank_losses_batch(zactive, y1, y2, sizes, rho, a=1.0):
    rho = np.asarray(rho, dtype=np.float64)
    z1 = a*zactive[:, :, 0, None]
    x2 = a*(zactive[:, :, 0, None]*rho[None, None, :] +
            zactive[:, :, 1, None]*np.sqrt(1-rho*rho)[None, None, :])
    logs = np.empty((len(zactive), 4, len(rho)))
    for si, (s1, s2) in enumerate(SIGNS):
        act = -np.logaddexp(0., -s1*z1)-np.logaddexp(0., -s2*x2)
        logs[:, si, :] = np.log(zactive.shape[1])-logsumexp(act, axis=1)
    out = np.empty((len(zactive), len(y2), len(sizes), len(rho)))
    for ti, second in enumerate(y2):
        pattern = 2*(y1 > 0)+(second > 0).astype(np.int8)
        for ni, N in enumerate(sizes):
            freq = np.bincount(pattern[:N], minlength=4)/N
            out[:, ti, ni, :] = np.einsum("bsq,s->bq", logs, freq)
    return out


def curvature_bound(zactive, a, R):
    """Uniform absolute empirical raw/marg second-derivative upper bound."""
    c = a*np.max(np.abs(zactive[:, :, 0]) +
                 R/np.sqrt(1-R*R)*np.abs(zactive[:, :, 1]), axis=1)
    q = a*np.max(np.abs(zactive[:, :, 1]), axis=1)/(1-R*R)**1.5
    return float(np.mean(q+1.25*c*c))


def deterministic_inputs(N, B, K, n):
    """Bounded arithmetic patterns; not pseudorandom scientific input."""
    j = np.arange(N, dtype=np.int64)
    y1 = np.where(j % 3 == 0, -1, 1).astype(np.int8)
    jj = np.arange(n, dtype=np.int64)
    yn = np.where((j[:, None]*7+jj[None, :]*3) % 11 < 5, -1, 1).astype(np.int8)
    y2 = np.stack([np.where((j*7+t*3) % (11+2*t) < 5+t, -1, 1)
                   for t in range(3)]).astype(np.int8)
    z = np.sin(np.arange(B*K*(n+2), dtype=np.float64)*.017).reshape(B, K, n+2)
    z += .3*np.cos(np.arange(B*K*(n+2), dtype=np.float64)*.031).reshape(z.shape)
    return z, y1, yn, y2


def deterministic_check():
    z, y1, yn, y2 = deterministic_inputs(7, 3, 5, 4)
    w, offset = build_weights(z, y1, yn)
    diffs, margdiffs, single_batch_diffs = [], [], []
    rhos = np.array([-.75, -.13, 0., .31, .75])
    batched = -np.log(second_probabilities_batch(w, z[:, :, :2], rhos))
    mbatched = marg_bank_losses_batch(z[:, :, :2], y1, y2, [3, 7], rhos)
    for ri, rho in enumerate(rhos):
        fast = batched[:, :, ri, :]
        mfast = mbatched[:, :, :, ri]
        single_batch_diffs.append(float(np.max(np.abs(fast+np.log(second_probabilities(w, z[:, :, :2], rho))))))
        single_batch_diffs.append(float(np.max(np.abs(mfast-marg_bank_losses(z[:, :, :2], y1, y2, [3, 7], rho)))))
        for ti in range(3):
            direct_marg = np.empty((3, 7))
            for b in range(3):
                x2 = rho*z[b, :, 0]+np.sqrt(1-rho*rho)*z[b, :, 1]
                for j in range(7):
                    active = (-np.logaddexp(0., -y1[j]*z[b, :, 0])
                              -np.logaddexp(0., -y2[ti, j]*x2))
                    full = active-np.logaddexp(0., -yn[j]*z[b, :, 2:]).sum(axis=1)
                    direct = np.log(5)-logsumexp(full)
                    got = offset[b, j]+fast[b, j, int(y2[ti, j] > 0)]
                    diffs.append(abs(direct-got))
                    direct_marg[b, j] = np.log(5)-logsumexp(active)
            for ni, N in enumerate((3, 7)):
                margdiffs.extend(np.abs(direct_marg[:, :N].mean(axis=1)-mfast[:, ti, ni]))
    result = dict(raw_point_comparisons=len(diffs), raw_max_absolute_error=float(max(diffs)),
                  marg_bank_comparisons=len(margdiffs), marg_max_absolute_error=float(max(margdiffs)),
                  single_vs_batch_max_absolute_error=float(max(single_batch_diffs)),
                  tolerance=2e-13, random_draws=0, optimizations=0,
                  nuisance_signs_retained=True)
    assert max(diffs+margdiffs) < 2e-13, result
    return result


def benchmark_batch():
    destination = OUT/"BENCHMARK_BATCH.json"
    assert not destination.exists(), "refuse benchmark overwrite"
    check = deterministic_check()
    N, B, K, n = 4096, 128, 256, 254
    z, y1, yn, y2 = deterministic_inputs(N, B, K, n)
    start = time.perf_counter()
    w, offset = build_weights(z, y1, yn)
    build_seconds = time.perf_counter()-start
    queries = np.linspace(-.75, .75, 32)
    cases, baseline = [], None
    for batch in (1, 8, 16):
        chunks = []
        start = time.perf_counter()
        for pos in range(0, len(queries), batch):
            rhos = queries[pos:pos+batch]
            raw = raw_bank_losses_batch(w, z[:, :, :2], y2, [1024, 4096], rhos)
            marg = marg_bank_losses_batch(z[:, :, :2], y1, y2, [1024, 4096], rhos)
            chunks.append(np.stack((raw, marg)))
        seconds = time.perf_counter()-start
        scores = np.concatenate(chunks, axis=-1)
        if baseline is None:
            baseline = scores
        difference = float(np.max(np.abs(scores-baseline)))
        assert difference < 2e-13
        cases.append(dict(batch_rho=batch, query_count=len(queries), seconds=seconds,
                          seconds_per_query=seconds/len(queries), max_difference_vs_batch1=difference,
                          projected_801_query_seconds=build_seconds+seconds*801/len(queries),
                          probability_output_bytes=B*N*batch*2*8))
    result = dict(schema=1, source_sha256=sha(__file__), deterministic_check=check,
                  shape=dict(N=N, B=B, K=K, n=n), weight_construction_seconds=build_seconds,
                  cases=cases, weights_bytes=w.nbytes, random_draws=0, fits=0,
                  note="All batches share identical deterministic arrays and candidate queries. No scientific input generation or parameter selection.")
    save(destination, result)
    print(json.dumps(result), flush=True)


def benchmark():
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT/"BENCHMARK.json"
    assert not destination.exists(), "do not overwrite benchmark; archive first with root approval"
    check = deterministic_check()
    N, B, K, n = 4096, 128, 256, 254
    start = time.perf_counter()
    z, y1, yn, y2 = deterministic_inputs(N, B, K, n)
    inputs_seconds = time.perf_counter()-start
    start = time.perf_counter()
    w, offset = build_weights(z, y1, yn)
    build_seconds = time.perf_counter()-start
    queries = np.linspace(-.75, .75, 32)
    timings, checksum = [], 0.
    for rho in queries:
        start = time.perf_counter()
        raw = raw_bank_losses(w, z[:, :, :2], y2, [1024, 4096], float(rho))
        marg = marg_bank_losses(z[:, :, :2], y1, y2, [1024, 4096], float(rho))
        timings.append(time.perf_counter()-start)
        checksum += float(raw.sum()+marg.sum())
    payload = dict(schema=1, source_sha256=sha(__file__), deterministic_check=check,
                   shape=dict(N=N, B=B, K=K, n=n), queries=len(queries),
                   input_construction_seconds=inputs_seconds, weight_construction_seconds=build_seconds,
                   query_seconds=timings, query_mean_seconds=float(np.mean(timings)),
                   projected_801_query_seconds=float(build_seconds+801*np.mean(timings)),
                   projected_16_stream_seconds=float(16*(build_seconds+801*np.mean(timings))),
                   weights_bytes=w.nbytes, prior_bytes=z.nbytes, offsets_bytes=offset.nbytes,
                   minimum_resident_array_bytes=w.nbytes+z.nbytes+offset.nbytes+y1.nbytes+yn.nbytes+y2.nbytes,
                   checksum=checksum, random_draws=0, scientific_data=0, fits=0,
                   note="Deterministic bounded shape benchmark only, not Gaussian samples or scientific curves. Timing is an estimate, not a fixed runtime promise.")
    save(destination, payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def polynomial_sup_abs(poly):
    """Exact polynomial-extremum recipe; evaluation remains floating point."""
    xs = list(poly.x)
    for root in poly.derivative().roots(extrapolate=False):
        if np.isfinite(root) and poly.x[0] <= root <= poly.x[-1]:
            xs.append(float(root))
    return float(np.max(np.abs(poly(np.asarray(xs)))))


def load_reference(path, config):
    """Consume CAL's frozen cubic derivative spline, never refit its g values."""
    data = json.loads(Path(path).read_text("utf-8"))
    for key in ("a", "R"):
        assert data[key] == config[key], (key, data.get(key), config[key])
    case = next(c for c in data["cases"] if c["n"] == config["n"] and c["K"] == config["K"])
    reference = data["reference"]
    rho = np.asarray(reference["rho801"], dtype=np.float64)
    rr = np.asarray(reference["r801"], dtype=np.float64)
    assert rho[0] == -config["R"] and rho[-1] == config["R"]
    assert np.all(np.diff(rho) > 0) and np.all(np.diff(rr) > 0)
    gp = PPoly(np.asarray(case["gprime_spline_c"], dtype=np.float64),
               np.asarray(case["gprime_spline_x"], dtype=np.float64), extrapolate=False)
    g = gp.antiderivative()
    independent_D = polynomial_sup_abs(gp.derivative())
    assert abs(independent_D-case["g_second_sup"]) <= 1e-10*max(1., independent_D)
    rpoly = CubicSpline(rho, rr, extrapolate=False)
    H = rpoly.antiderivative()
    interface = dict(constants={"A": reference["A"], "d": reference["d"], "w": case["w"]},
                     H_second_sup=polynomial_sup_abs(rpoly.derivative()),
                     g_second_sup=independent_D,
                     H_zero=float(H(0.)), g_zero=float(g(0.)),
                     calibration_case={"n": case["n"], "K": case["K"]},
                     H_table_max_error=float(np.max(np.abs((H(rho)-H(0.))-np.asarray(reference["H801"])))))
    return interface, PchipInterpolator(rho, rr, extrapolate=False), H, g


def observed_kl(t, r):
    ans = .5*((1+t)*np.log1p((t-r)/(1+r))+(1-t)*np.log1p((r-t)/(1-r)))
    return float(max(0., ans))


def generate_input(config, stream):
    """Three teacher worlds share base randomness; only streams are independent."""
    N, n, B, K = max(config["sample_sizes"]), config["n"], config["B"], config["K"]
    data_seed = config["data_seed_base"]+stream
    prior_seed = config["prior_seed_base"]+stream
    drng = np.random.Generator(np.random.PCG64(data_seed))
    noise_uniform = drng.random((N, n))
    yn = np.where(noise_uniform < .5, -1, 1).astype(np.int8)
    g = drng.standard_normal((N, 2))
    uniforms = drng.random((N, 2))
    y1 = np.where(uniforms[:, 0] < expit(config["a"]*g[:, 0]), 1, -1).astype(np.int8)
    y2 = np.stack([np.where(uniforms[:, 1] < expit(config["a"]*(theta*g[:, 0]+
                  np.sqrt(1-theta*theta)*g[:, 1])), 1, -1)
                   for theta in config["teachers"]]).astype(np.int8)
    prng = np.random.Generator(np.random.PCG64(prior_seed))
    z = prng.standard_normal((B, K, n+2))
    provenance = dict(data_seed=data_seed, prior_seed=prior_seed,
                      noise_uniform_sha256=array_sha(noise_uniform),
                      active_gaussian_sha256=array_sha(g), active_uniform_sha256=array_sha(uniforms),
                      prior_sha256=array_sha(z), y1_sha256=array_sha(y1),
                      ynoise_sha256=array_sha(yn), y2_sha256=array_sha(y2))
    return z, y1, yn, y2, provenance


def run():
    frozen_path = OUT/"FREEZE.json"
    frozen = json.loads(frozen_path.read_text("utf-8"))
    assert frozen["source_sha256"] == sha(__file__), "training source not frozen"
    assert frozen["run_authorized"] is True
    assert sha(ROOT/frozen["plan_path"]) == frozen["plan_sha256"]
    config = frozen["config"]
    refpath = ROOT/frozen["reference_path"]
    assert sha(refpath) == frozen["reference_sha256"]
    assert not (OUT/"RESULTS.json").exists(), "refuse scientific overwrite"
    assert not list(OUT.glob("stream_*.npz")), "resume must be separately reviewed"
    ref, rfun, Hfun, gfun = load_reference(refpath, config)
    rho = np.linspace(-config["R"], config["R"], config["grid_points"])
    grid_step = float(rho[1]-rho[0])
    sizes = config["sample_sizes"]
    A, d, wscale = (float(ref["constants"][key]) for key in ("A", "d", "w"))
    penalties = dict(raw=np.zeros_like(rho), marg=np.zeros_like(rho),
                     quad=A*d*wscale*rho*rho,
                     asymcorr=2*A*wscale*(Hfun(rho)-ref["H_zero"]),
                     finitecal=gfun(rho)-ref["g_zero"])
    penalty_D = dict(raw=0., marg=0., quad=2*A*d*wscale,
                     asymcorr=2*A*wscale*ref["H_second_sup"], finitecal=ref["g_second_sup"])
    records, begin = [], time.perf_counter()
    for stream in range(config["streams"]):
        started = time.perf_counter()
        z, y1, yn, y2, provenance = generate_input(config, stream)
        weights, offset = build_weights(z, y1, yn, config["a"])
        baseD = curvature_bound(z[:, :, :2], config["a"], config["R"])
        shape = (config["B"], len(config["teachers"]), len(sizes), len(rho))
        raw, marg = np.empty(shape), np.empty(shape)
        for ri in range(0, len(rho), BATCH_RHO):
            candidates = rho[ri:ri+BATCH_RHO]
            raw[:, :, :, ri:ri+len(candidates)] = raw_bank_losses_batch(weights, z[:, :, :2], y2, sizes, candidates, config["a"])
            marg[:, :, :, ri:ri+len(candidates)] = marg_bank_losses_batch(z[:, :, :2], y1, y2, sizes, candidates, config["a"])
        zero_index = config["grid_points"]//2
        assert rho[zero_index] == 0.
        raw -= raw[:, :, :, zero_index:zero_index+1].copy()
        marg -= marg[:, :, :, zero_index:zero_index+1].copy()
        assert np.isfinite(raw).all() and np.isfinite(marg).all()
        rows = []
        for ti, theta in enumerate(config["teachers"]):
            teacher_r = float(rfun(theta))
            for ni, N in enumerate(sizes):
                t = float(np.mean(y1[:N]*y2[ti, :N]))
                observed_target = float(np.clip(t, rfun(-config["R"]), rfun(config["R"])))
                if observed_target == float(rfun(-config["R"])):
                    mle = -config["R"]
                elif observed_target == float(rfun(config["R"])):
                    mle = config["R"]
                else:
                    mle = float(brentq(lambda x: float(rfun(x))-observed_target,
                                       -config["R"], config["R"], xtol=1e-13, rtol=1e-13))
                for method in METHODS:
                    record = dict(stream=stream, theta=theta, N=N, B=config["B"], K=config["K"],
                                  method=method, empirical_pair=t, teacher_r=teacher_r)
                    if method in penalties:
                        bankcurve = marg[:, ti, ni] if method == "marg" else raw[:, ti, ni]
                        scores = bankcurve.mean(axis=0)+penalties[method]
                        chosen = int(np.argmin(scores))
                        estimate = float(rho[chosen])
                        D = baseD+penalty_D[method]
                        record.update(grid_index=chosen, selected_score=float(scores[chosen]),
                                      curvature_D=D, mesh_step=grid_step,
                                      continuous_objective_regret_bound_real_arithmetic=D*grid_step*grid_step/8,
                                      raw_mesh_score_se=float(bankcurve[:, chosen].std(ddof=1)/np.sqrt(config["B"])),
                                      certificate_scope="Same finite-bank implemented spline objective; exact arithmetic interpolation bound, not floating-point or population certificate.")
                    else:
                        estimate = mle if method == "exact" else 0.
                    record.update(rho=estimate, observed_r=float(rfun(estimate)),
                                  reference_joint_KL=observed_kl(teacher_r, float(rfun(estimate))))
                    rows.append(record)
        dest = OUT/f"stream_{stream:02}.npz"
        np.savez_compressed(dest, raw=raw, marg=marg, rho=rho, y1=y1, ynoise=yn, y2=y2,
                            prior_active=z[:, :, :2], offset=offset, teachers=np.array(config["teachers"]),
                            sample_sizes=np.array(sizes))
        record = dict(stream=stream, provenance=provenance, file=dest.relative_to(ROOT).as_posix(),
                      file_sha256=sha(dest), elapsed_seconds=time.perf_counter()-started,
                      base_curvature_D=baseD, outputs=rows)
        save(OUT/f"stream_{stream:02}.json", record)
        records.append(record)
        save(OUT/"PROGRESS.json", dict(completed_streams=len(records), elapsed_seconds=time.perf_counter()-begin))
        print(json.dumps(dict(completed_stream=stream, seconds=record["elapsed_seconds"])), flush=True)
        del weights, raw, marg, z
    save(OUT/"RESULTS.json", dict(schema=1, source_sha256=sha(__file__), freeze_sha256=sha(frozen_path),
         reference_sha256=sha(refpath), config=config, methods=METHODS, records=records,
         penalty_curvature_D=penalty_D, reference_interface=ref, elapsed_seconds=time.perf_counter()-begin,
         output_count=sum(len(r["outputs"]) for r in records),
         tie_rule="First index among exact computed-score minima on ascending rho grid; no truth-based selection.",
         qualification="Reference risks and reference-continuous MLE use the independently frozen monotone r interpolant. Grid certificate bounds only real-arithmetic discretization of the implemented finite-bank/spline objective."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["check", "bench", "bench-batch", "run"])
    args = parser.parse_args()
    if args.mode == "check":
        print(json.dumps(deterministic_check()))
    elif args.mode == "bench":
        benchmark()
    elif args.mode == "bench-batch":
        benchmark_batch()
    else:
        run()
