"""Independent N arithmetic and bounded RNG-prefix replay; no producer import."""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "2"
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
import traceback
import numpy as np
import scipy
from scipy.optimize import brentq
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"artifacts/objective_strengthening/N"
ERRORS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_hashes(mapping):
    for name, digest in mapping.items():
        assert sha(ROOT/name) == digest, name


def equal(name, actual, expected, tolerance=3e-12):
    x, y = np.asarray(actual), np.asarray(expected)
    assert x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all(), name
    error = float(np.max(np.abs(x-y))) if x.size else 0.
    ERRORS[name] = max(ERRORS.get(name, 0.), error)
    assert error <= tolerance, (name, error, tolerance)


def bernoulli_kl(p, q):
    assert 0 <= p <= 1 and 0 < q < 1
    value = 0.
    if p:
        value += p*(math.log(p)-math.log(q))
    if p < 1:
        value += (1-p)*(math.log1p(-p)-math.log1p(-q))
    return value


def tail_inverted_cp(successes, n, alpha, family):
    """Invert binomial tails, independently of producer beta-quantile calls."""
    tail = alpha/(2*family)
    if successes == 0:
        lower = 0.
    else:
        lower = brentq(lambda p: float(binom.sf(successes-1, n, p))-tail,
                       0., 1., xtol=5e-15, rtol=1e-14)
    if successes == n:
        upper = 1.
    else:
        upper = brentq(lambda p: float(binom.cdf(successes, n, p))-tail,
                       0., 1., xtol=5e-15, rtol=1e-14)
    return [lower, upper]


def rectangle_by_corners(p_interval, q_interval):
    corners = list(itertools.product(p_interval, q_interval))
    differences = [p-q for p, q in corners]
    squared = [(p-q)**2 for p, q in corners]
    divergences = [bernoulli_kl(p, q) for p, q in corners]
    intersects = max(p_interval[0], q_interval[0]) <= min(p_interval[1], q_interval[1])
    return dict(difference=[min(differences), max(differences)],
                brier=[0. if intersects else min(squared), max(squared)],
                logloss=[0. if intersects else min(divergences), max(divergences)])


def derive_from_stored_moments(cfg, moments):
    assert [row["order"] for row in moments["gh"]] == cfg["orders"]
    final, previous = moments["gh"][-1]["values"], moments["gh"][-2]["values"]
    keys = [str(float(s)) for s in cfg["scales"]]
    assert set(final) == set(previous) == set(keys)
    order_difference = max(abs(final[s][k]-previous[s][k]) for s in keys for k in ("alpha", "ell"))
    independent_difference = max(abs(final[s][k]-moments["quad"][s][k]["value"])
                                 for s in keys for k in ("alpha", "ell"))
    reported_error = max(moments["quad"][s][k]["reported_error"]
                         for s in keys for k in ("alpha", "ell"))
    equal("stored GH refinement difference", order_difference, moments["last_order_difference"], 0.)
    equal("stored quadrature cross difference", independent_difference, moments["independent_difference"], 0.)
    equal("stored quad reported error", reported_error, moments["quad_reported_error_max"], 0.)
    assert moments["qualified"] == (max(order_difference, independent_difference, reported_error)
                                     <= cfg["quadrature_gate"])
    assert moments["qualified"] and not moments["interval_certified"]
    gh_evals = len(cfg["scales"])*sum(cfg["orders"])*2
    assert moments["GH_scalar_integrand_evaluations"] == gh_evals
    alpha_truth = math.fsum(final[str(float(s))]["alpha"] for s in cfg["truth"])/2
    alpha_floor = final[str(float(cfg["a"]))]["alpha"]
    limits = []
    for sign in cfg["signs"]:
        vt = 1+sign*cfg["c"]*alpha_truth**2
        vf = 1+sign*cfg["c"]*alpha_floor**2
        p, q = math.erf(1/math.sqrt(2*vt)), math.erf(1/math.sqrt(2*vf))
        independent = math.erf(1/math.sqrt(2))
        limits.append(dict(sign=sign, teacher_variance=vt, floor_variance=vf,
            teacher_event=p, floor_event=q, independent_event=independent,
            teacher_minus_floor=p-q, brier_excess=(p-q)**2,
            event_logloss_excess=bernoulli_kl(p, q),
            teacher_minus_independent=p-independent, independent_brier=(p-independent)**2))
    # This frozen diagnostic uses the negative-family spectrum at d=128.
    d, k = 128, 100
    rho = cfg["c"]/(d-1)
    negative_R = np.eye(d)-rho*(np.ones((d, d))-np.eye(d))
    lam = cfg["b"]**2*float(np.linalg.eigvalsh(negative_R)[-1])
    slope = alpha_floor/2
    coarse = 2*math.sqrt(2*lam*d*math.log(k))/(d*slope)
    return dict(limits=limits, coarse_bound_d128_K100=coarse,
        trivial_max_scale_excess=2*(cfg["b"]-cfg["a"]),
        coarse_bound_informative=bool(coarse < 2*(cfg["b"]-cfg["a"])),
        coarse_lambda=lam, ell_prime_floor=slope), gh_evals


def householder_prefix_replay(cfg, row, stored_counts):
    """A different matrix construction of the principal symmetric square root."""
    d, sign, seed = row["d"], row["sign"], row["seed"]
    rng = np.random.Generator(np.random.PCG64(seed))
    assert rng.bit_generator.state == row["state_initial"]
    chunk = min(cfg["chunk"], cfg["N"])
    assert chunk >= 64
    # Consume the entire first Gaussian chunk before any uniform variates.
    z = rng.standard_normal((chunk, d))
    uniforms = rng.random((chunk, d))
    unit = np.full(d, 1/math.sqrt(d))
    reflection_vector = -unit.copy()
    reflection_vector[0] += 1
    reflection_vector /= np.linalg.norm(reflection_vector)
    orthogonal = np.eye(d)-2*np.outer(reflection_vector, reflection_vector)
    eigenvalues = np.full(d, 1-sign*cfg["c"]/(d-1))
    eigenvalues[0] = 1+sign*cfg["c"]
    square_root = (orthogonal*np.sqrt(eigenvalues)[None, :]) @ orthogonal.T
    expected_R = np.eye(d)+sign*cfg["c"]/(d-1)*(np.ones((d, d))-np.eye(d))
    root_error = float(np.max(np.abs(square_root @ square_root.T-expected_R)))
    symmetry_error = float(np.max(np.abs(square_root-square_root.T)))
    assert root_error <= 3e-12 and symmetry_error <= 3e-12
    equal("Householder common eigendirection", orthogonal[:, 0], unit, 3e-14)
    gaussian = z[:64] @ square_root.T
    scale_truth = np.repeat(np.asarray(cfg["truth"]), d//2)
    scale_floor = np.repeat(np.asarray(cfg["floor"]), d//2)
    # tanh expression independently checks expit's label thresholding.
    truth_probability = (1+np.tanh(gaussian*scale_truth/2))/2
    floor_probability = (1+np.tanh(gaussian*scale_floor/2))/2
    threshold = uniforms[:64]
    replay = np.column_stack([(threshold < truth_probability).sum(1),
                             (threshold < floor_probability).sum(1)])
    assert np.array_equal(replay, stored_counts[:64]), (d, sign, "count prefix mismatch")
    return dict(d=d, sign=sign, seed=seed, checked_base_observations=64,
        checked_coupled_law_counts=128, RNG_base_vectors_consumed=chunk,
        gaussian_variates_reexecuted=chunk*d, uniform_variates_reexecuted=chunk*d,
        root_gram_max_error=root_error, root_symmetry_max_error=symmetry_error,
        smallest_threshold_distance=float(min(np.min(abs(threshold-truth_probability)),
                                             np.min(abs(threshold-floor_probability)))),
        exact_saved_count_match=True, full_bit_labels_not_available_for_comparison=True,
        final_RNG_state_not_replayed=True)


def audit(workspace):
    assert workspace, "--write requires --workspace"
    began = time.perf_counter()
    ERRORS.clear()
    cfg = read(OUT/"CONFIG.json")
    freeze, seal = read(OUT/"FREEZE.json"), read(OUT/"RESULTS_FREEZE.json")
    assert read(OUT/"STATE.json")["status"] == "complete"
    assert seal["source_freeze_sha256"] == sha(OUT/"FREEZE.json")
    verify_hashes(freeze["files"])
    verify_hashes(seal["files"])
    verify_hashes(seal["local_arrays"])
    assert cfg["floor"] == [cfg["a"], cfg["a"]]
    assert len(cfg["truth"]) == 2 and len(cfg["signs"]) == 2
    expected_probabilities = len(cfg["dimensions"])*len(cfg["signs"])*2
    assert cfg["family"] == expected_probabilities
    derived, gh_evals = derive_from_stored_moments(cfg, read(OUT/"MOMENTS.json"))
    saved_limits = read(OUT/"LIMITS.json")
    for expected, actual in zip(derived["limits"], saved_limits["limits"]):
        for key, value in expected.items():
            equal("limit "+key, actual[key], value, 2e-12)
    for key in derived:
        if key != "limits":
            if isinstance(derived[key], bool):
                assert saved_limits[key] == derived[key]
            else:
                equal("coarse "+key, saved_limits[key], derived[key], 3e-11)
    rows = read(OUT/"RESULTS.json")
    assert len(rows) == len(cfg["dimensions"])*len(cfg["signs"])
    expected_cells = set(itertools.product(cfg["dimensions"], cfg["signs"]))
    assert {(row["d"], row["sign"]) for row in rows} == expected_cells
    replays, local_arrays, interval_rows = [], {}, []
    passes = 0
    for row in rows:
        d, sign, n = row["d"], row["sign"], row["N"]
        assert n == cfg["N"] and d % 2 == 0
        di, si = cfg["dimensions"].index(d), cfg["signs"].index(sign)
        assert row["seed"] == cfg["seed"]+100*di+si
        assert row == read(OUT/f"d{d}_sign{sign}.json")
        assert row["shared_randomness_between_two_laws"]
        path = ROOT/row["array"]
        assert sha(path) == row["array_sha256"] == seal["local_arrays"][row["array"]]
        local_arrays[row["array"]] = row["array_sha256"]
        with np.load(path, allow_pickle=False) as archive:
            assert archive.files == ["counts"]
            counts = archive["counts"].astype(np.int64)
        assert counts.shape == (n, 2) and counts.min() >= 0 and counts.max() <= d
        valid_counts = [i for i in range(d+1) if abs(2*i-d) <= math.isqrt(d)]
        events = []
        intervals = []
        for law in range(2):
            histogram = np.bincount(counts[:, law], minlength=d+1)
            assert histogram.tolist() == row["count_histograms"][law]
            assert int(histogram.sum()) == n
            successes = int(np.isin(counts[:, law], valid_counts).sum())
            assert successes == sum(int(histogram[i]) for i in valid_counts)
            events.append(successes)
            interval = tail_inverted_cp(successes, n, cfg["alpha"], cfg["family"])
            intervals.append(interval)
            equal("CP binomial-tail inversion", interval, row["probability_intervals"][law])
            interval_rows.append(dict(d=d, sign=sign, law=law, successes=successes,
                                      interval=interval))
        assert events == row["events"]
        equal("empirical event probabilities", [x/n for x in events], row["probabilities"], 0.)
        rectangle = rectangle_by_corners(*intervals)
        for key, values in rectangle.items():
            equal("rectangle "+key, values, row["risk_rectangle"][key])
        signed_upper = max(sign*x for x in rectangle["difference"])
        primary = bool(signed_upper < 0 and rectangle["brier"][0] > cfg["brier_threshold"])
        assert primary == row["primary_pass"]
        passes += primary
        # Integer binomial mass, not the producer's binomial-CDF subtraction.
        independent_probability = float(Fraction(sum(math.comb(d, i) for i in valid_counts), 1 << d))
        equal("finite independent binomial event", independent_probability,
              row["independent_binomial_event"], 3e-14)
        replays.append(householder_prefix_replay(cfg, row, counts))
        print("N independent audit", d, sign, "prefix64/statistics pass", flush=True)
    assert local_arrays == seal["local_arrays"]
    counts_saved = read(OUT/"COUNTS.json")
    cells = len(cfg["dimensions"])*len(cfg["signs"])
    base_vectors = cells*cfg["N"]
    bit_outputs = sum(cfg["dimensions"])*len(cfg["signs"])*cfg["N"]*2
    expected_counts = dict(base_vector_draws=base_vectors, coupled_law_observations=2*base_vectors,
        label_bit_outputs=bit_outputs, new_fits=0, new_neural_models=0,
        full_bit_arrays_saved=False, per_example_count_arrays_saved=True,
        binomial_probability_family=expected_probabilities, primary_pass=passes, primary_total=cells)
    assert counts_saved == expected_counts
    assert len(interval_rows) == cfg["family"]
    sources = [Path(__file__).resolve(), OUT/"FREEZE.json", OUT/"RESULTS_FREEZE.json"]
    return dict(passed=True, audit_utc=datetime.now(timezone.utc).isoformat(),
        source_hashes={p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
        local_arrays=local_arrays, workspace=True, counts_rebuilt_from_config=expected_counts,
        probability_intervals_recomputed=interval_rows, primary_passed=passes,
        primary_total=cells, replay=replays, max_errors=ERRORS,
        checked_base_observations=sum(r["checked_base_observations"] for r in replays),
        checked_coupled_law_counts=sum(r["checked_coupled_law_counts"] for r in replays),
        RNG_base_vectors_consumed=sum(r["RNG_base_vectors_consumed"] for r in replays),
        gaussian_variates_reexecuted=sum(r["gaussian_variates_reexecuted"] for r in replays),
        uniform_variates_reexecuted=sum(r["uniform_variates_reexecuted"] for r in replays),
        integrals_reexecuted=0, stored_GH_scalar_evaluations=gh_evals,
        new_independent_scientific_draws=0, new_fits=0, producer_calculation_functions_imported=False,
        numpy=np.__version__, scipy=scipy.__version__,
        coarse_bound_scope="d128/K100 negative-family sharpened spectral bound; not the positive-family bound",
        numerical_limits_scope="derived from saved final GH moments; no new quadrature or strict remainder certificate",
        replay_scope="all stored count arrays/histograms; first64 eachcell from entire first1024 RNG chunk; final RNG states not replayed",
        elapsed_seconds=time.perf_counter()-began)


def main():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    parser.add_argument("--workspace", action="store_true")
    args = parser.parse_args()
    path = OUT/"INDEPENDENT_AUDIT.json"
    if args.check:
        receipt = read(path)
        assert receipt["passed"]
        verify_hashes(receipt["source_hashes"])
        verify_hashes(read(OUT/"FREEZE.json")["files"])
        verify_hashes(read(OUT/"RESULTS_FREEZE.json")["files"])
        if args.workspace:
            verify_hashes(receipt["local_arrays"])
        print("N independent audit receipt and hashes valid; no arithmetic/RNG replay", flush=True)
        return
    assert not path.exists(), "audit receipt is append-only"
    try:
        result = audit(args.workspace)
    except Exception as error:
        result = dict(passed=False, error=repr(error), traceback=traceback.format_exc(),
                      max_errors=ERRORS, new_independent_scientific_draws=0, new_fits=0)
        with path.open("x", encoding="utf-8", newline="\n") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
        raise
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("replay", "local_arrays", "probability_intervals_recomputed")},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
