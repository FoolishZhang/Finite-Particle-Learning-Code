"""Independent, read-only replay of the saved unknown-group learning study.

No core/runner/statistics imports, reference integration, or scientific RNG.
--write creates one new AUDIT.json; --check never writes.  --workspace also
links public sufficient statistics to local integer data and old B panels.
"""
from __future__ import annotations

import os
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import argparse
import csv
import hashlib
import itertools
import json
import math
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import stdtr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/structure_learning_v2/C"
BROOT = ROOT / "artifacts/structure_learning_v2/B"
PROFILES = {"null": (4., 4.), "contrast": (6., 2.)}
METHODS = ("ORACLE_J", "ORACLE_P", "GLOBAL_P", "ROW_P", "GLOBAL_G10", "GLOBAL_G100")
NS = (256, 4096, 65536)
TOL = 2e-10
TIE = 1e-12


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csvrows(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_bool(value):
    assert value in (True, False, "True", "False"), value
    return value is True or value == "True"


def integer(value):
    answer = int(value)
    assert float(answer) == float(value), value
    return answer


class Checks:
    def __init__(self):
        self.maxima = {}
        self.hash_count = 0

    def near(self, name, actual, expected, tol=TOL):
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        assert a.shape == b.shape, (name, a.shape, b.shape)
        assert np.isfinite(a).all() and np.isfinite(b).all(), name
        error = float(np.max(abs(a-b))) if a.size else 0.
        self.maxima[name] = max(error, self.maxima.get(name, 0.))
        assert error <= tol, (name, error, tol)

    def hashes(self, mapping):
        for name, expected in mapping.items():
            assert name != "HANDOFF.md", "live HANDOFF is not an audit input"
            path = ROOT / name
            assert path.is_file(), f"missing frozen input {name}"
            assert sha(path) == expected, f"hash mismatch {name}"
            self.hash_count += 1


@lru_cache(maxsize=None)
def critical():
    return brentq(lambda z: stdtr(15, z)-(1-.05/72), 0., 100.,
                  xtol=1e-13, rtol=9e-16)


def interval(values):
    values = [float(v) for v in values]
    assert len(values) == 16 and all(math.isfinite(v) for v in values)
    avg = math.fsum(values)/16
    se = math.sqrt(math.fsum((v-avg)**2 for v in values)/(16*15))
    radius = critical()*se
    return dict(mean=avg, SE=se, lower=avg-radius, upper=avg+radius,
                precision_pass=se <= .002)


def geom():
    cells = list(itertools.product(range(9), repeat=2))
    a, b = np.array(cells, dtype=int).T
    multiplicity = np.array([math.comb(8, x)*math.comb(8, y) for x, y in cells])
    lg = np.log(multiplicity.astype(float))
    ind = ((a+b)[:, None] == np.arange(17)[None, :]).astype(float)
    same = np.array([[math.comb(x, 2)+math.comb(8-x, 2),
                      math.comb(y, 2)+math.comb(8-y, 2),
                      x*y+(8-x)*(8-y)] for x, y in cells], dtype=float)
    different = np.array([28, 28, 64])[None, :]-same
    return a, b, lg, ind, same, different


def choose(n, k):
    return math.comb(n, k) if 0 <= k <= n else 0


@lru_cache(maxsize=None)
def mapping(r):
    """Candidate-A count = retained true-A successes + imported true-B ones."""
    assert r in range(9)
    result = np.zeros((81, 81))
    denominator = math.comb(8, 8-r)*math.comb(8, r)
    for a in range(9):
        for b in range(9):
            for x in range(9):
                wx = choose(a, x)*choose(8-a, 8-r-x)
                if not wx:
                    continue
                for y in range(9):
                    wy = choose(b, y)*choose(8-b, r-y)
                    if wy:
                        u = x+y
                        v = a+b-u
                        assert 0 <= u <= 8 and 0 <= v <= 8
                        result[9*a+b, 9*u+v] += wx*wy/denominator
    assert np.max(abs(result.sum(axis=1)-1)) < 1e-14
    return result


def deterministic_smoke():
    """Pure enumeration, no simulated teacher or random numbers."""
    errors = []
    r = [.1, .15, .2]
    base = math.log(4)-sum(n*math.log1p(-x*x) for n, x in zip((1, 1, 4), r))/12
    slopes = [.5*math.log((1+x)/(1-x))/6 for x in r]
    for y in itertools.product((-1, 1), repeat=4):
        for group in ((0, 1), (0, 2)):
            sums = [0., 0., 0.]
            direct = 0.
            for i, j in itertools.combinations(range(4), 2):
                kind = 0 if i in group and j in group else (1 if i not in group and j not in group else 2)
                sums[kind] += y[i]*y[j]
                direct -= math.log((1-r[kind]*y[i]*y[j])/4)/6
            errors.append(abs(direct-base-sum(x*z for x, z in zip(sums, slopes))))
    assert max(errors) < 1e-12
    assert np.array_equal(mapping(0), np.eye(81))
    for a in range(9):
        for b in range(9):
            assert mapping(8)[9*a+b, 9*b+a] == 1.
    return dict(enumerated_patterns=16, affine_max_error=max(errors),
                exact_mapping_rows=9*81, scientific_samples=0)


def load_library(check):
    _, _, lg, ind, same, different = geom()
    refs = [read(BROOT / "reference" / f"g{i:03}.json") for i in range(361)]
    variants = []
    mags = []
    for i, r in enumerate(refs):
        assert r.get("qualified", r.get("numerical_pass", False))
        check.near("reference_scales", r["scales"], [i//19/2, i % 19/2], 0.)
        v = np.array(r["variants"], dtype=float)
        assert v.shape == (3, 81) and np.isfinite(v).all() and (v > 0).all()
        check.near("reference_mass", v.sum(1), np.ones(3), 2e-12)
        variants.append(v)
        mags.append(r["independent_Hermite_moments"]["pair_magnitudes_AA_BB_AB"])
    variants, mags = np.array(variants), np.array(mags)
    assert mags.shape == (361, 3) and (abs(mags) < 1).all()
    joint = -np.log(variants[:, 2])+lg
    # Direct pair probabilities rather than the runner's affine loss constructor.
    pair = -(np.log((1-mags)/4) @ same.T + np.log((1+mags)/4) @ different.T)/120
    slope = .5*np.log((1+mags)/(1-mags))/120
    intercept = math.log(4)-np.log(1-mags*mags) @ np.array([28., 28., 64.])/240
    ordered = np.array([19*u+v for u in range(19) for v in range(u+1)])
    assert len(ordered) == 190
    return dict(variants=variants, joint=joint, pair=pair, slope=slope,
                intercept=intercept, ordered=ordered, lg=lg, ind=ind)


def risk_library(lib, profile, check):
    s, t = PROFILES[profile]
    variant = lib["variants"][19*int(2*s)+int(2*t)]
    p = variant[2]
    H = -float(p @ (np.log(p)-lib["lg"]))
    mapped = np.array([p @ mapping(r) for r in range(9)])
    risk = mapped @ lib["joint"].T-H
    pn = p @ lib["ind"]
    qn = lib["variants"][:, 2] @ lib["ind"]
    count = -np.log(qn) @ pn + float(pn @ np.log(pn))
    sensitivity = 0.
    for pv in variant:
        hv = -float(pv @ (np.log(pv)-lib["lg"]))
        mappedv = np.array([pv @ mapping(r) for r in range(9)])
        for vi in range(3):
            value = mappedv @ (-np.log(lib["variants"][:, vi])+lib["lg"]).T-hv
            sensitivity = max(sensitivity, float(abs(value-risk).max()))
    # Bayes lower bound for a uniformly unknown, labelled high-scale mask.
    n = geom()[0]+geom()[1]
    total_multiplicity = np.array([math.comb(16, int(x)) for x in n], dtype=float)
    sym_group = pn[n]*np.exp(lib["lg"])/total_multiplicity
    count_bound = 0. if profile == "null" else float(p @ np.log(p/sym_group))
    bound_variants = []
    for pv in variant:
        pnv = pv @ lib["ind"]
        symv = pnv[n]*np.exp(lib["lg"])/total_multiplicity
        bound_variants.append(float(pv @ np.log(pv/symv)))
    stored = read(OUT / "RISK_LIBRARY.json")[profile]
    for field, value in (("H", H), ("teacher_count_p", p), ("mapped_teacher_p", mapped),
                         ("candidate_joint_KL_by_swap", risk), ("candidate_total_count_KL", count),
                         ("reference_sensitivity", sensitivity), ("count_only_bound", count_bound),
                         ("count_bound_variants", bound_variants)):
        check.near("risk_library_"+field, stored[field], value)
    assert sensitivity <= 1e-7
    assert max(bound_variants)-min(bound_variants) <= 1e-7
    return dict(p=p, H=H, risk=risk, count=count, count_only_bound=count_bound)


def search_geometry():
    masks = np.array([sum(1 << i for i in group)
                      for group in itertools.combinations(range(16), 8)], dtype=np.uint32)
    masks.sort()
    assert len(masks) == 12870
    pairs = list(itertools.combinations(range(16), 2))
    i, j = np.array(pairs).T
    member = ((masks[:, None] >> np.arange(16)) & 1).astype(bool)
    aa = (member[:, i] & member[:, j]).astype(float)
    bb = (~member[:, i] & ~member[:, j]).astype(float)
    return dict(masks=masks, i=i, j=j, aa=aa, bb=bb)


def first_minimum(values, ids=None):
    values = np.asarray(values, dtype=float)
    assert values.ndim == 1 and np.isfinite(values).all()
    low = float(np.min(values))
    at = int(np.nonzero(values-low <= TIE)[0][0])
    return (at if ids is None else int(ids[at])), low, float(values[at]-low)


def exhaustive_pair(M, sg, lib):
    moments = M[sg["i"], sg["j"]]
    aa, bb = sg["aa"] @ moments, sg["bb"] @ moments
    total = math.fsum(float(x) for x in moments)
    features = np.stack([aa, bb, total-aa-bb], axis=1)
    ids = lib["ordered"]
    values = features @ lib["slope"][ids].T+lib["intercept"][ids]
    for col, idx in enumerate(ids):
        if idx//19 == idx % 19:
            values[:, col] = lib["intercept"][idx]+total*lib["slope"][idx, 0]
    at, minimum, gap = first_minimum(values.ravel())
    mi, si = divmod(at, 190)
    return int(sg["masks"][mi]), int(ids[si]), minimum, gap


def codes_for(mask):
    return np.fromiter((9*(x & mask).bit_count()+(x & (65535 ^ mask)).bit_count()
                        for x in range(65536)), dtype=np.int16, count=65536)


def histogram_moments(hist, sg):
    """Recover uncentered moments by exact counts of unequal bit pairs."""
    ids = np.arange(65536, dtype=np.uint32)
    values = np.eye(16)
    N = int(hist.sum())
    for i, j in zip(sg["i"], sg["j"]):
        mismatch = ((ids >> int(i)) ^ (ids >> int(j))) & 1
        different = int(np.dot(hist, mismatch.astype(np.int64)))
        values[i, j] = values[j, i] = 1-2*different/N
    return values


def row_bound(lib, check):
    teachers = lib["variants"][12*19+4]
    lambdas = [.0001, .0003, .001, .003, .01, .03, .1]
    support, weights = [], []
    for a in range(9):
        for b in range(9):
            for ya, yb in itertools.product((-1, 1), repeat=2):
                pa = choose(7, a-(ya == 1))/math.comb(8, a)
                pb = choose(7, b-(yb == 1))/math.comb(8, b)
                support.append(-(ya-yb)*((2*a-8-ya)+(2*b-8-yb)))
                weights.append(teachers[:, 9*a+b]*pa*pb)
    w = np.array(support, dtype=float)
    p = np.array(weights).T
    check.near("ROW_probability_mass", p.sum(1), np.ones(3), 2e-12)
    mgf = np.array([[math.fsum(float(x*y) for x, y in zip(row, np.exp(-lam*w)))
                     for row in p] for lam in lambdas])
    idx = int(np.argmin(mgf[:, 2]))
    primary, safe = float(mgf[idx, 2]), float(max(mgf[idx])+1e-10)
    stored = read(OUT / "ROW_RECOVERY_BOUND.json")
    assert stored["profile"] == "contrast" and stored["lambda_grid"] == lambdas
    assert stored["chosen_lambda"] == lambdas[idx]
    check.near("ROW_mgf", stored["mgf_by_lambda_and_variant"], mgf, 2e-13)
    check.near("ROW_mean", stored["mean_W"], p @ w, 2e-12)
    check.near("ROW_second_moment", stored["second_moment_W"], p @ (w*w), 2e-11)
    assert stored["simultaneous_pairs"] == 64 and stored["independence_between_pairs_not_assumed"]
    assert stored["conditional_on_fixed_numerical_teacher"]
    assert stored["allowance_not_rigorous_Gaussian_integration_certificate"]
    for record, N in zip(stored["bounds"], NS):
        assert record["N"] == N
        for name, m in (("primary_union_failure_upper", primary),
                        ("numerical_allowance_union_failure_upper", safe)):
            value = math.exp(min(0., math.log(64)+N*math.log(m)))
            check.near("ROW_"+name, record[name], value, 2e-9)
    assert len(stored["bounds"]) == 3
    return dict(lambdas=7, support_rows=len(w), probability_variants=3, union_pairs=64,
                budgets=list(NS), chosen_lambda=lambdas[idx])


def audit(workspace=False):
    check = Checks()
    assert read(OUT / "STATE.json")["status"] == "complete", "C results are not complete"
    cfg = read(OUT / "CONFIG.json")
    assert cfg["methods"] == list(METHODS) and cfg["N"] == list(NS)
    assert cfg["profiles"] == {k: list(v) for k, v in PROFILES.items()}
    assert cfg["d"] == 16 and cfg["c"] == .9 and cfg["scales"] == [i/2 for i in range(19)]
    assert cfg["streams"] == 16 and cfg["family"] == 36 and cfg["tie_tolerance"] == TIE
    assert cfg["SE_gate"] == .002 and cfg["alpha"] == .05 and cfg["material"] == .001
    seal, final = read(OUT / "FREEZE.json"), read(OUT / "RESULTS_FREEZE.json")
    check.hashes(seal["files"])
    check.hashes(final["files"])
    assert seal["reused_B_panels_not_fresh_confirmation"]
    if workspace:
        check.hashes(seal["B_local_panel_hashes"])
    smoke = deterministic_smoke()
    assert read(OUT / "SMOKE.json")["passed"]
    lib, sg = load_library(check), search_geometry()
    row_receipt = row_bound(lib, check)
    fit_index = {}
    count_bounds = {}
    argmins = 0
    global_searches = 0
    B_panels = {}
    limitations = []
    local_hashes = []
    for stream in range(16):
        path = BROOT / "local" / f"s{stream:02}.npz"
        key = path.relative_to(ROOT).as_posix()
        receipt = read(BROOT / "streams" / f"s{stream:02}.json")
        assert receipt["array_sha256"] == seal["B_local_panel_hashes"][key]
        assert receipt["stream"] == stream and receipt["numerical_pass"]
        if workspace:
            with np.load(path) as arr:
                B_panels[stream] = arr["mean_pattern_loss"].copy()
            assert B_panels[stream].shape == (361, 2, 81)
            assert np.isfinite(B_panels[stream]).all()
    for pi, profile in enumerate(PROFILES):
        risks = risk_library(lib, profile, check)
        count_bounds[profile] = risks["count_only_bound"]
        for stream in range(16):
            rec = read(OUT / "streams" / f"{profile}_s{stream:02}.json")
            assert rec["profile"] == profile and rec["stream"] == stream and rec["numerical_pass"]
            assert rec["seed"] == cfg["seed"]+1000*pi+stream
            assert rec["initial_RNG"]["bit_generator"] == rec["final_RNG"]["bit_generator"] == "PCG64"
            true_mask = integer(rec["true_mask"])
            assert 0 <= true_mask <= 65535 and true_mask.bit_count() == 8
            M_all = np.array(rec["empirical_spin_matrices"], dtype=float)
            assert M_all.shape == (3, 16, 16)
            hist = None
            if workspace:
                path = ROOT / rec["array_path"]
                assert sha(path) == rec["array_sha256"]
                local_hashes.append(dict(path=rec["array_path"], sha256=rec["array_sha256"]))
                with np.load(path) as arr:
                    hist = arr["pattern_histograms"].copy()
                    check.near("local_saved_M", arr["empirical_spin_matrices"], M_all, 1e-12)
                assert hist.shape == (3, 65536) and np.issubdtype(hist.dtype, np.integer)
                assert (hist >= 0).all() and (np.diff(hist, axis=0) >= 0).all()
                assert np.array_equal(hist.sum(1), NS)
            assert len(rec["diagnostics"]) == 3 and len(rec["fits"]) == 18
            current = {(x["N"], x["method"]): x for x in rec["fits"]}
            assert len(current) == 18
            for ni, N in enumerate(NS):
                M = M_all[ni]
                check.near("M_symmetric", M, M.T, 1e-12)
                check.near("M_unit_diagonal", np.diag(M), np.ones(16), 1e-12)
                assert np.max(abs(M)) <= 1+1e-12
                if workspace:
                    check.near("histogram_to_uncentered_M", M, histogram_moments(hist[ni], sg), 1e-12)
                gmask, gidx, gmin, ggap = exhaustive_pair(M, sg, lib)
                global_searches += 1
                row_scores = -np.array([math.fsum(float(M[i, j]) for j in range(16) if j != i)/15
                                       for i in range(16)])
                row_mask = sum(1 << i for i in sorted(range(16), key=lambda i: (-row_scores[i], i))[:8])
                diag = rec["diagnostics"][ni]
                assert diag["N"] == N
                check.near("ROW_scores", diag["row_scores"], row_scores, 1e-12)
                dindex = {x["method"]: x for x in diag["selected_group_counts"]}
                assert set(dindex) == set(METHODS)
                count_cache = {}
                for method in METHODS:
                    row = current[N, method]
                    assert row["profile"] == profile and row["stream"] == stream
                    assert row["true_mask"] == true_mask
                    mask = true_mask if method.startswith("ORACLE") else (row_mask if method == "ROW_P" else gmask)
                    assert row["selected_mask"] == mask and dindex[method]["mask"] == mask
                    counts = np.asarray(dindex[method]["training_group_counts"])
                    assert counts.shape == (81,) and np.isfinite(counts).all()
                    assert np.array_equal(counts, counts.astype(np.int64)) and (counts >= 0).all()
                    assert counts.sum() == N
                    if workspace:
                        if mask not in count_cache:
                            count_cache[mask] = np.zeros(81, dtype=np.int64)
                            np.add.at(count_cache[mask], codes_for(mask), hist[ni])
                        assert np.array_equal(counts, count_cache[mask])
                    weights = counts.astype(float)/N
                    if method == "GLOBAL_P":
                        idx, minimum, gap = gidx, gmin, ggap
                        value = float(lib["pair"][idx] @ weights)
                        check.near("GLOBAL_affine_direct_value", value, minimum+gap, 2e-11)
                    else:
                        objectives = (lib["joint"] if method == "ORACLE_J" else lib["pair"]) @ weights
                        if method.startswith("GLOBAL_G"):
                            public = dindex[method].get("training_objectives")
                            if public is None and not workspace:
                                limitations.append("G conditional objective vectors absent: those argmins not replayed in public mode.")
                                objectives = None
                            elif public is not None:
                                objectives = np.asarray(public, dtype=float)
                                assert objectives.shape == (361,) and np.isfinite(objectives).all()
                            if workspace:
                                local = B_panels[stream][:, 0 if method == "GLOBAL_G10" else 1] @ weights
                                if public is not None:
                                    check.near("B_panel_to_public_G_objectives", objectives, local, 2e-11)
                                objectives = local
                        if objectives is None:
                            idx = integer(row["candidate_index"])
                            minimum, gap, value = (float(row[k]) for k in
                                                   ("minimum_numerical_objective", "grid_gap", "training_objective"))
                        else:
                            ids = lib["ordered"] if method == "ROW_P" else np.arange(361)
                            idx, minimum, gap = first_minimum(objectives[ids], ids)
                            value = float(objectives[idx])
                    verified = method == "GLOBAL_P" or not method.startswith("GLOBAL_G") or objectives is not None
                    if verified:
                        assert row["candidate_index"] == idx, (profile, stream, N, method, row["candidate_index"], idx)
                        argmins += 1
                    assert 0 <= idx < 361
                    check.near("selected_training_loss", row["training_objective"], value, 2e-11)
                    check.near("numeric_grid_minimum", row["minimum_numerical_objective"], minimum, 2e-11)
                    check.near("numeric_grid_gap", row["grid_gap"], gap, 2e-12)
                    assert float(row["grid_gap"]) <= 1.1e-12
                    u, v = idx//19/2, idx % 19/2
                    check.near("selected_scales", [row["u"], row["v"]], [u, v], 0.)
                    r = 8-(mask & true_mask).bit_count()
                    assert row["overlap_swap_r"] == r
                    loss = float(risks["risk"][r, idx])
                    best = float(risks["risk"][r].min())
                    for name, expected in (("deployed_KL", loss), ("assumed_group_best_grid_KL", best),
                                           ("excess_above_assumed_group_grid_best", loss-best),
                                           ("total_count_KL", float(risks["count"][idx]))):
                        check.near(name, row[name], expected)
                    identifiable = profile == "contrast" and u != v
                    high = mask if u > v else 65535 ^ mask
                    swaps = 8-(high & true_mask).bit_count() if identifiable else None
                    assert row["group_recovery_identifiable"] == identifiable
                    assert row["high_group_swaps"] == swaps
                    assert row["high_group_exact_recovery"] == (swaps == 0 if identifiable else None)
                    key = profile, stream, N, method
                    assert key not in fit_index
                    fit_index[key] = row
            for name, value in (("full_pattern_observations", 65536), ("finite_grid_fits", 18),
                                ("exhaustive_global_pair_fits", 3), ("global_candidate_representations", 2445300)):
                assert rec[name] == value
            if stream % 4 == 3:
                print(json.dumps(dict(audit_progress=profile, streams=stream+1, new_samples=0)), flush=True)
    assert len(fit_index) == 576 and global_searches == 96
    audit_tables(check, fit_index)
    stat_receipt = audit_statistics(check, fit_index, count_bounds)
    limitations += [
        "All public M-based pair searches are replayed; only workspace links M and grouped counts to the saved integer histograms.",
        "Public G selections use saved conditional objective vectors; only workspace verifies their multiplication from frozen old B panels.",
        "Old B panels are reused development evidence, not fresh independent computational confirmation.",
        "Reference PMFs and numerical qualifications are frozen premises; no reference integrals or scientific draws are regenerated.",
        "The 36 risk intervals remain t approximations (paired where both endpoints are fitted); degenerate intervals do not certify population equivalence.",
        "ROW bounds are conditional on the fixed numerical teacher; the numerical allowance is not a rigorous Gaussian integration certificate.",
    ]
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), passed=True,
                auditor_source_sha256=sha(__file__),
                input_seals={name: sha(OUT/name) for name in ("FREEZE.json", "RESULTS_FREEZE.json")},
                public_streams=32, reference_candidates=361, finite_fits=576,
                independently_replayed_argmins=argmins, global_pair_exhaustive_searches=global_searches,
                candidate_representations_per_global_search=2445300, risk_intervals=36,
                unknown_group_count_only_Bayes_bounds=count_bounds,
                statistics=stat_receipt, ROW=row_receipt, deterministic_smoke=smoke,
                workspace=workspace, local_histograms_verified=32 if workspace else 0,
                old_B_panels_verified=16 if workspace else 0, local_array_receipts=local_hashes,
                hash_checks=check.hash_count, numerical_maxima=check.maxima,
                numerical_tolerance=TOL, tie_tolerance=TIE,
                independent_of_runner_selection_and_statistics=True,
                new_scientific_samples=0, new_reference_integrations=0, new_fits=0,
                limitations=list(dict.fromkeys(limitations)))


def audit_tables(check, index):
    rows = csvrows(OUT / "FITS.csv")
    assert len(rows) == 576
    seen = set()
    integer_fields = ("stream", "N", "true_mask", "selected_mask", "candidate_index", "overlap_swap_r")
    float_fields = ("u", "v", "training_objective", "minimum_numerical_objective", "grid_gap", "deployed_KL",
                    "assumed_group_best_grid_KL", "excess_above_assumed_group_grid_best", "total_count_KL")
    for row in rows:
        key = row["profile"], integer(row["stream"]), integer(row["N"]), row["method"]
        assert key not in seen
        seen.add(key)
        expected = index[key]
        for field in integer_fields:
            assert integer(row[field]) == expected[field]
        for field in float_fields:
            check.near("FITS_csv_"+field, float(row[field]), expected[field])
        assert as_bool(row["group_recovery_identifiable"]) == expected["group_recovery_identifiable"]
        if expected["high_group_swaps"] is None:
            assert row["high_group_swaps"] == row["high_group_exact_recovery"] == ""
        else:
            assert integer(row["high_group_swaps"]) == expected["high_group_swaps"]
            assert as_bool(row["high_group_exact_recovery"]) == expected["high_group_exact_recovery"]


def audit_statistics(check, index, count_bounds):
    comparisons = (("GLOBAL_P", "ORACLE_J"), ("GLOBAL_P", "ORACLE_P"),
                   ("ROW_P", "GLOBAL_P"), ("GLOBAL_G10", "GLOBAL_P"), ("GLOBAL_G100", "GLOBAL_P"),
                   ("GLOBAL_P", "COUNT_ONLY_BOUND"))
    rows = csvrows(OUT / "RISK_INTERVALS.csv")
    assert len(rows) == 36
    rows = {(r["profile"], integer(r["N"]), r["comparison"]): r for r in rows}
    assert len(rows) == 36
    passed, primaries = 0, []
    for profile in PROFILES:
        for N in NS:
            for left, right in comparisons:
                values = [index[profile, s, N, left]["deployed_KL"]-
                          (count_bounds[profile] if right == "COUNT_ONLY_BOUND" else index[profile, s, N, right]["deployed_KL"])
                          for s in range(16)]
                expected = interval(values)
                rec = rows[profile, N, left+"-"+right]
                for name in ("mean", "SE", "lower", "upper"):
                    check.near("CI_"+name, float(rec[name]), expected[name], 2e-11)
                assert as_bool(rec["precision_pass"]) == expected["precision_pass"]
                passed += int(expected["precision_pass"])
                if N == 65536 and left == "GLOBAL_G10":
                    primaries.append(dict(kind="finite_particle_bias_after_learned_group", profile=profile,
                                          passed=expected["precision_pass"] and expected["lower"] > .001,
                                          **{k: expected[k] for k in ("mean", "lower", "upper")}))
                if N == 65536 and profile == "contrast" and right == "COUNT_ONLY_BOUND":
                    primaries.append(dict(kind="labelled_pair_below_count_only_bound", profile=profile,
                                          passed=expected["precision_pass"] and expected["upper"] < 0,
                                          **{k: expected[k] for k in ("mean", "lower", "upper")}))
    summary = csvrows(OUT / "SUMMARY.csv")
    assert len(summary) == 36
    seen = set()
    for rec in summary:
        key = rec["profile"], integer(rec["N"]), rec["method"]
        assert key not in seen
        seen.add(key)
        profile, N, method = key
        selected = [index[profile, s, N, method] for s in range(16)]
        for name, field in (("mean_KL", "deployed_KL"), ("mean_u", "u"), ("mean_v", "v")):
            check.near("summary_"+name, float(rec[name]), math.fsum(r[field] for r in selected)/16)
        good = [r for r in selected if r["group_recovery_identifiable"]]
        assert integer(rec["identifiable_groups"]) == len(good)
        assert integer(rec["exact_high_groups"]) == sum(r["high_group_exact_recovery"] is True for r in selected)
        if good:
            check.near("summary_swaps", float(rec["mean_swaps_if_identifiable"]),
                       math.fsum(r["high_group_swaps"] for r in good)/len(good))
        else:
            assert rec["mean_swaps_if_identifiable"] == ""
    decision = read(OUT / "DECISION.json")
    for name, value in (("family", 36), ("finite_fits", 576), ("global_exhaustive_fits", 96),
                        ("new_pattern_observations", 2097152), ("new_prior_particles", 0), ("precision_pass", passed)):
        assert decision[name] == value
    for field in ("existing_B_panels_reused", "unknown_group_joint_likelihood_not_exhaustively_optimized",
                  "no_strict_equivalence_claim_from_degenerate_intervals"):
        assert decision[field]
    assert len(decision["primary"]) == 3
    for expected in primaries:
        rec = next(r for r in decision["primary"] if r["profile"] == expected["profile"] and r["kind"] == expected["kind"])
        assert rec["passed"] == expected["passed"]
        for name in ("mean", "lower", "upper"):
            check.near("primary_"+name, rec[name], expected[name], 2e-11)
    return dict(family=36, streams_per_interval=16, df=15,
                precision_pass=passed, primary_checks=primaries, summary_rows=36)


def main():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    parser.add_argument("--workspace", action="store_true")
    args = parser.parse_args()
    target = OUT / "AUDIT.json"
    if args.write:
        assert not target.exists(), "AUDIT already exists; use --check"
        assert not (OUT / "MANIFEST.json").exists(), "already sealed; use --check"
    result = audit(workspace=args.workspace)
    if args.write:
        with target.open("x", encoding="utf-8", newline="\n") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
    else:
        saved = read(target)
        assert saved["passed"] and saved["auditor_source_sha256"] == sha(__file__)
        assert saved["input_seals"] == result["input_seals"]
        for name in ("public_streams", "reference_candidates", "finite_fits", "global_pair_exhaustive_searches",
                     "candidate_representations_per_global_search", "risk_intervals", "statistics", "ROW"):
            assert saved[name] == result[name], name
        if args.workspace and saved["workspace"]:
            assert saved["local_array_receipts"] == result["local_array_receipts"]
    print(json.dumps(dict(audit_pass=True, workspace=args.workspace,
                         argmins=result["independently_replayed_argmins"],
                         global_searches=result["global_pair_exhaustive_searches"],
                         intervals=result["risk_intervals"], new_scientific_samples=0)))


if __name__ == "__main__":
    main()
