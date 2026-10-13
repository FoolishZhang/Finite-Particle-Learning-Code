"""Independent, bounded audit of the frozen growing-K score panel.

The production module is never imported. Scores use sigmoid probabilities,
products, arithmetic particle means and then logarithms. Full RNG replay is
separate from the explicitly limited first-64-bank score reconstruction.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import time
import traceback

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/regularization_learning/GROWTH"
DOC = ROOT / "docs/08_research_研究工作/统一正则与真实相关学习_20261002"
PROTOCOL = OUT / "AUDIT_PROTOCOL.json"
DEST = OUT / "AUDIT.json"
ATOL = 1e-10
FIRST_BANKS = 64


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n",
                          encoding="utf-8", newline="\n")


def sigmoid(x):
    # All finite normal inputs here are modest; the runtime checks the result.
    return 1.0 / (1.0 + np.exp(-x))


def independent_probabilities(z, ns, rho, signs):
    """Rebuild every first-block score through probability products."""
    shape = (len(z), len(ns), len(rho), len(signs))
    raw = np.empty(shape, dtype=np.float64)
    marg = np.empty(shape, dtype=np.float64)
    smallest_mean = 1.0
    largest_mean = 0.0
    for ni, n in enumerate(ns):
        k = int(n) + 2
        noise_probability = np.prod(sigmoid(z[:, :k, 2:k]), axis=2)
        assert np.all(noise_probability > 0), "nuisance product underflow"
        for ri, r in enumerate(rho):
            x_second = r * z[:, :k, 0] + np.sqrt(1.0 - r * r) * z[:, :k, 1]
            for si, (s1, s2) in enumerate(signs):
                active_probability = sigmoid(s1 * z[:, :k, 0]) * sigmoid(s2 * x_second)
                raw_mean = np.mean(noise_probability * active_probability, axis=1)
                marg_mean = np.mean(active_probability, axis=1)
                assert np.all(raw_mean > 0) and np.all(marg_mean > 0)
                smallest_mean = min(smallest_mean, float(raw_mean.min()), float(marg_mean.min()))
                largest_mean = max(largest_mean, float(raw_mean.max()), float(marg_mean.max()))
                raw[:, ni, ri, si] = -np.log(raw_mean)
                marg[:, ni, ri, si] = -np.log(marg_mean)
    zero = np.flatnonzero(np.asarray(rho) == 0)
    assert len(zero) == 1
    zero = int(zero[0])
    raw -= raw[:, :, zero:zero + 1, :].copy()
    marg -= marg[:, :, zero:zero + 1, :].copy()
    return raw, marg, smallest_mean, largest_mean


def high_precision_mean(a, axis=0):
    assert axis == 0
    values = np.asarray(a)
    flat = values.reshape(len(values), -1)
    mean = [math.fsum(flat[:, j]) / len(values) for j in range(flat.shape[1])]
    return np.asarray(mean).reshape(values.shape[1:])


def mean_and_stream_se(a):
    values = np.asarray(a, dtype=float)
    mean = high_precision_mean(values)
    residuals = ((values - mean) ** 2).reshape(len(values), -1)
    se = [math.sqrt(math.fsum(residuals[:, j]) / ((len(values) - 1) * len(values)))
          for j in range(residuals.shape[1])]
    return mean, np.asarray(se).reshape(values.shape[1:])


def difference_record(actual, expected, threshold=ATOL):
    a = np.asarray(actual, dtype=np.longdouble)
    e = np.asarray(expected, dtype=np.longdouble)
    assert a.shape == e.shape, (a.shape, e.shape)
    diff = np.abs(a - e)
    index = np.unravel_index(int(np.argmax(diff)), diff.shape)
    return dict(count=int(diff.size), threshold_absolute=threshold,
                max_absolute_error=float(diff[index]),
                max_error_index=[int(x) for x in index],
                failures=int(np.count_nonzero(diff > threshold)),
                passed=bool(np.all(diff <= threshold)))


def freeze():
    assert not PROTOCOL.exists(), "Audit protocol already frozen"
    assert not DEST.exists(), "Audit result already exists"
    production = json.loads((OUT / "FREEZE.json").read_text("utf-8"))
    records = json.loads((OUT / "RESULTS.json").read_text("utf-8"))["records"]
    save(PROTOCOL, dict(
        schema="reglearn-growth-independent-audit-v1",
        source_sha256=sha(__file__),
        inputs={p.relative_to(ROOT).as_posix(): sha(p) for p in [
            ROOT / "script/reglearn_growth.py", DOC / "PLAN.md",
            OUT / "FREEZE.json", OUT / "RESULTS.json"]},
        npz_expected_sha256={r["file"]: r["file_sha256"] for r in records},
        comparison_absolute_threshold=ATOL,
        score_scope="Every stream's first 64 banks; all n/K/rho/four active modes; raw and marg rho=0 contrasts",
        score_formula="Direct sigmoid probability products, arithmetic K means, then negative logarithms; no production import or logsumexp",
        first_banks_per_stream=FIRST_BANKS,
        rng_scope="All sixteen blocks of 64 banks in every stream, using the production's frozen seed/shape/call sequence; compare full and first64 byte hashes",
        aggregation="All stored banks: math.fsum per-stream mean versus JSON; independently recompute 16-stream means, standard errors, and adjacent-rho contrast standard errors",
        aggregate_se_unit="independent stream (16), not bank or reused n/K/rho/mode",
        precision_scope="Unweighted active-mode curves only; theta-weighted gates and conclusions are outside this audit and checked by the parent summary",
        preserve_failures=True,
        retries="No automatic rerun or budget extension; a failure output is preserved",
        production_freeze=production,
        new_scientific_samples=0, new_training_fits=0,
        replay_is_new_independent_data=False))
    print(json.dumps({"audit_protocol_frozen": True, "sha256": sha(PROTOCOL)}), flush=True)


def run():
    assert not DEST.exists(), "Existing audit is preserved; do not overwrite"
    protocol = json.loads(PROTOCOL.read_text("utf-8"))
    assert protocol["source_sha256"] == sha(__file__), "Audit code changed after freeze"
    for path, digest in protocol["inputs"].items():
        assert sha(ROOT / path) == digest, ("frozen audit input changed", path)
    freeze_data = json.loads((OUT / "FREEZE.json").read_text("utf-8"))
    results = json.loads((OUT / "RESULTS.json").read_text("utf-8"))
    started = time.perf_counter()
    audited = []
    checks = []
    records = results["records"]
    ns = np.array(freeze_data["ns"], dtype=int)
    rho = np.array(freeze_data["rho"], dtype=float)
    signs = np.array(freeze_data["signs"], dtype=int)
    streams, banks, block = 16, 1024, 64
    max_k = max_d = 256
    gaussian_replay_count = 0
    scored_banks = 0
    scored_values = 0
    mean_comparisons = 0
    all_raw_means, all_marg_means = [], []
    aggregate = {}
    failure = None
    try:
        assert freeze_data["streams"] == streams and len(records) == streams
        assert freeze_data["banks_per_stream"] == banks and freeze_data["block"] == block
        assert freeze_data["seed_base"] == 202610020700
        assert np.array_equal(ns, [14, 62, 254])
        assert np.array_equal(rho, np.linspace(-.75, .75, 21))
        assert np.array_equal(signs, [[-1, -1], [-1, 1], [1, -1], [1, 1]])
        assert results["source_sha256"] == sha(ROOT / "script/reglearn_growth.py")
        assert results["freeze_sha256"] == sha(OUT / "FREEZE.json")
        assert freeze_data["source_sha256"] == results["source_sha256"]
        assert freeze_data["plan_sha256"] == sha(DOC / "PLAN.md")
        assert results["science_ledger"] == freeze_data
        assert results["rho"] == freeze_data["rho"]
        assert results["ns"] == freeze_data["ns"] and results["signs"] == freeze_data["signs"]
        for stream, saved in enumerate(records):
            assert saved["stream"] == stream and saved["seed"] == freeze_data["seed_base"] + stream
            filename = ROOT / saved["file"]
            file_hash = sha(filename)
            file_pass = file_hash == saved["file_sha256"]
            with np.load(filename, allow_pickle=False) as npz:
                assert set(npz.files) == {"raw", "marg", "rho", "ns", "signs"}
                assert np.array_equal(npz["rho"], rho)
                assert np.array_equal(npz["ns"], ns)
                assert np.array_equal(npz["signs"], signs)
                raw, marg = npz["raw"], npz["marg"]
            assert raw.shape == marg.shape == (banks, 3, 21, 4)
            assert np.isfinite(raw).all() and np.isfinite(marg).all()
            rng = np.random.Generator(np.random.PCG64(saved["seed"]))
            full_hash = hashlib.sha256()
            prefix_hash = None
            first_raw = first_marg = None
            smallest = largest = None
            for offset in range(0, banks, block):
                z = rng.standard_normal((block, max_k, max_d))
                gaussian_replay_count += int(z.size)
                full_hash.update(z.tobytes(order="C"))
                if offset == 0:
                    prefix_hash = hashlib.sha256(z.tobytes(order="C")).hexdigest()
                    first_raw, first_marg, smallest, largest = independent_probabilities(z, ns, rho, signs)
                    scored_banks += block
                    scored_values += int(first_raw.size + first_marg.size)
            raw_diff = difference_record(first_raw, raw[:FIRST_BANKS])
            marg_diff = difference_record(first_marg, marg[:FIRST_BANKS])
            raw_mean = high_precision_mean(raw)
            marg_mean = high_precision_mean(marg)
            raw_json_diff = difference_record(raw_mean, saved["raw_mean"])
            marg_json_diff = difference_record(marg_mean, saved["marg_mean"])
            mean_comparisons += raw_json_diff["count"] + marg_json_diff["count"]
            all_raw_means.append(raw_mean)
            all_marg_means.append(marg_mean)
            checks.extend([file_pass, full_hash.hexdigest() == saved["full_prior_sha256"],
                           prefix_hash == saved["first64_prior_sha256"], raw_diff["passed"],
                           marg_diff["passed"], raw_json_diff["passed"], marg_json_diff["passed"]])
            audited.append(dict(stream=stream, file=saved["file"],
                file_sha256=file_hash, file_sha_passed=file_pass,
                regenerated_full_prior_sha256=full_hash.hexdigest(),
                full_prior_sha_passed=full_hash.hexdigest() == saved["full_prior_sha256"],
                regenerated_first64_prior_sha256=prefix_hash,
                first64_prior_sha_passed=prefix_hash == saved["first64_prior_sha256"],
                raw_first64_contrasts=raw_diff, marg_first64_contrasts=marg_diff,
                raw_all_bank_mean_vs_json=raw_json_diff, marg_all_bank_mean_vs_json=marg_json_diff,
                smallest_reconstructed_probability_mean=smallest,
                largest_reconstructed_probability_mean=largest,
                raw_mean=np.asarray(raw_mean, dtype=float).tolist(),
                marg_mean=np.asarray(marg_mean, dtype=float).tolist()))
            print(json.dumps({"stream": stream, "passed_so_far": all(checks),
                              "elapsed_seconds": round(time.perf_counter() - started, 3)}), flush=True)
        for name, stream_means in [("raw", all_raw_means), ("marg", all_marg_means)]:
            mean, se = mean_and_stream_se(stream_means)
            adjacent = np.diff(np.asarray(stream_means, dtype=np.longdouble), axis=2)
            adjacent_mean, adjacent_se = mean_and_stream_se(adjacent)
            aggregate[name] = dict(mean=mean.tolist(), standard_error_across_16_streams=se.tolist(),
                adjacent_rho_mean=adjacent_mean.tolist(),
                adjacent_rho_standard_error_across_16_streams=adjacent_se.tolist(),
                note="four unweighted active modes; not theta-weighted qualification gates")
    except Exception:
        failure = traceback.format_exc()
    ledger = dict(new_scientific_samples=0, new_observed_samples=0, new_fits=0,
        replayed_gaussian_scalars=gaussian_replay_count,
        replayed_unique_prior_vectors=gaussian_replay_count // max_d,
        replayed_banks=gaussian_replay_count // (max_d * max_k),
        independently_rescored_banks=scored_banks,
        independently_rescored_prior_vectors=scored_banks * max_k,
        independently_rescored_gaussian_prefix_values=scored_banks * max_k * max_d,
        independently_rescored_raw_marg_mode_contrasts=scored_values,
        mean_scalar_comparisons=mean_comparisons,
        remaining_saved_banks_only_aggregate_checked=banks * len(audited) - scored_banks,
        repeated_rng_is_not_new_independent_science=True,
        first64_score_rebuild_is_not_full_score_replay=True)
    record = dict(status="passed" if failure is None and all(checks) else "failed",
        audit_code_sha256=sha(__file__), protocol_sha256=sha(PROTOCOL),
        input_hashes=protocol["inputs"], elapsed_seconds=time.perf_counter() - started,
        numpy_version=np.__version__, longdouble_precision=np.finfo(np.longdouble).precision,
        criterion_absolute=ATOL, checks_passed=sum(checks), checks_total=len(checks),
        stream_records=audited, aggregate_mode_curves=aggregate,
        ledger=ledger, software_error=failure,
        scope="Complete prior RNG byte replay; independent score formula only first 64 banks per stream; all saved-bank mean/SE arithmetic",
        excluded="No empirical-data training, no independent new streams, no global-optimum proof, no Gaussian integral certification, no theta-weighted gate certification")
    save(DEST, record)
    print(json.dumps({"status": record["status"], "checks": [sum(checks), len(checks)],
                      "audit_sha256": sha(DEST), "ledger": ledger}), flush=True)
    if record["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["freeze", "run"])
    args = parser.parse_args()
    freeze() if args.mode == "freeze" else run()
