"""Independent B evidence audit: no B/core statistical or selection imports."""
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
import traceback
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln, stdtr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/structure_learning_v2/B"
A = ROOT / "artifacts/structure_learning_v2/A"
GRID = [(i/2, j/2) for i in range(19) for j in range(19)]
RS = [0, 1, 2, 4]
NS = [4096, 65536]
OBJECTIVES = ["J", "P", "T", "G10", "G100"]
TOL = 3e-11


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csvrows(name):
    with (OUT / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def boolean(value):
    assert value in (True, False, "True", "False")
    return value is True or value == "True"


class AuditChecks:
    def __init__(self):
        self.maxima = {}
        self.hash_checks = 0

    def near(self, label, value, expected, tol=TOL):
        x, y = np.asarray(value, dtype=float), np.asarray(expected, dtype=float)
        assert x.shape == y.shape, (label, x.shape, y.shape)
        assert np.isfinite(x).all() and np.isfinite(y).all(), label
        error = float(np.max(abs(x-y))) if x.size else 0.
        self.maxima[label] = max(self.maxima.get(label, 0.), error)
        assert error <= tol, (label, error, tol)

    def hashes(self, entries):
        for name, expected in entries.items():
            assert name != "HANDOFF.md", "live HANDOFF not an audit input"
            assert sha(ROOT / name) == expected, f"hash mismatch: {name}"
            self.hash_checks += 1

    def seal(self, path):
        obj = read(path)
        self.hashes(obj.get("files", obj.get("scientific_files", {})))
        return obj


def geometry():
    na = np.repeat(np.arange(9), 9)
    nb = np.tile(np.arange(9), 9)
    total = na+nb
    mult = np.array([math.comb(8, int(a))*math.comb(8, int(b)) for a, b in zip(na, nb)])
    # The lgamma expression matches the declared floating representation; its
    # integer-combination identity is independently checked below.
    lg = (gammaln(9)-gammaln(na+1)-gammaln(9-na)
          +gammaln(9)-gammaln(nb+1)-gammaln(9-nb))
    indicator = (total[:, None] == np.arange(17)[None, :]).astype(float)
    counts = np.zeros((81, 3, 4), dtype=np.int64)
    for cell, (a, b) in enumerate(zip(na, nb)):
        pattern = [1]*int(a)+[0]*(8-int(a))+[1]*int(b)+[0]*(8-int(b))
        for i, j in itertools.combinations(range(16), 2):
            kind = 0 if j < 8 else 1 if i >= 8 else 2
            counts[cell, kind, 2*pattern[i]+pattern[j]] += 1
    assert np.array_equal(counts.sum(axis=2), np.broadcast_to([28, 28, 64], (81, 3)))
    sa, sb = 2*na-8, 2*nb-8
    return dict(na=na, nb=nb, total=total, mult=mult, lg=lg, indicator=indicator,
                pair_counts=counts, sa=sa, sb=sb)


def independent_kernel(r):
    """Alternative combinatorics: count moved-slot assignments / orbit size."""
    W = np.zeros((81, 81))
    for a in range(9):
        for b in range(9):
            denominator = math.comb(8, a)*math.comb(8, b)
            for i in range(r+1):
                if not 0 <= a-i <= 8-r:
                    continue
                for j in range(r+1):
                    if not 0 <= b-j <= 8-r:
                        continue
                    numerator = (math.comb(r, i)*math.comb(8-r, a-i)
                                 *math.comb(r, j)*math.comb(8-r, b-j))
                    W[9*a+b, 9*(a-i+j)+b+i-j] += numerator/denominator
    return W


def entropy(p, g):
    return -math.fsum(float(w)*math.log(float(w/m)) for w, m in zip(p, g["mult"]))


def reference_check(ref, g, check):
    p = np.array(ref["mean"])
    variants = np.array(ref["variants"])
    raw = np.array(ref["raw_variants"])
    assert ref["qualified"] and ref["numerical_pass"]
    assert p.shape == (81,) and variants.shape == raw.shape == (3, 81)
    assert np.isfinite(raw).all() and np.all(raw > 0)
    assert np.max(abs(raw.sum(axis=1)-1)) < 1e-12
    check.near("reference_primary", p, variants[2], tol=0.)
    check.near("reference_roundoff_normalization", variants, raw/raw.sum(axis=1, keepdims=True))
    assert max(abs(variants[i]-variants[j]).sum() for i in range(3) for j in range(i)) <= 1e-8
    H = entropy(p, g)
    check.near("reference_H", ref["H"], H)
    totals = p @ g["indicator"]
    logchoose = np.log(np.array([math.comb(16, int(n)) for n in g["total"]], dtype=float))
    sym = totals[g["total"]]*np.exp(g["lg"]-logchoose)
    check.near("reference_total_p", ref["total_p"], totals)
    check.near("reference_symmetry", ref["symmetrized_group_p"], sym)
    check.near("reference_self_KL", ref["model_KL_single"], float(p @ np.log(p/sym)))
    moment = ref["independent_Hermite_moments"]
    if ref["analytic_independence"]:
        corr = np.zeros(3)
        check.near("zero_scale_exact_PMF", p, g["mult"]/65536, tol=0.)
    else:
        source = {float(item["scale"]): item["coefficients"] for item in moment["coefficient_sources"]}
        u, v = ref["scales"]
        corr = []
        for x, y in ((u, u), (v, v), (u, v)):
            if x == 0 or y == 0:
                corr.append(0.)
            else:
                corr.append(-math.fsum(float(a)*float(b)*(.9/15)**k
                    for a, b, k in zip(source[x], source[y], range(1, 14, 2))))
        corr = np.array(corr)
    check.near("independent_Hermite_pair_coefficients", corr, -np.array(moment["pair_magnitudes_AA_BB_AB"]))
    observed = np.array([(p @ (g["sa"]**2)-8)/56, (p @ (g["sb"]**2)-8)/56,
                         p @ (g["sa"]*g["sb"])/64])
    check.near("reference_pair_moments", observed, corr, tol=1e-8)
    weights = np.column_stack([g["sa"]**2, g["sb"]**2, g["sa"]*g["sb"]])
    check.near("reference_spin_moment_receipt", ref["variant_spin_moments"], variants @ weights)
    assert np.max(abs(variants @ weights-np.array(moment["expected_spin_moments"]))) <= 1e-8
    pair_pmf = np.array([[(1+r)/4, (1-r)/4, (1-r)/4, (1+r)/4] for r in corr])
    assert np.all(pair_pmf > 0)
    # Explicit enumeration counts validate the independent compact log1p formula.
    explicit_pair = -np.sum(g["pair_counts"]*np.log(pair_pmf)[None], axis=(1, 2))/120
    same = g["pair_counts"][:, :, 0]+g["pair_counts"][:, :, 3]
    different = np.array([28, 28, 64])-same
    pair = (120*math.log(4)-same @ np.log1p(corr)-different @ np.log1p(-corr))/120
    check.near("explicit_120_pair_loss", pair, explicit_pair)
    return p, variants, pair, -corr


def load_references(g, check):
    teacher = read(A / "TEACHER.json")
    pstar = np.array(teacher["mean"])
    mapped = np.array(teacher["mapped_counts_by_r"])
    mappedv = np.array(teacher["mapped_variants_by_r"])
    H = entropy(pstar, g)
    check.near("original_teacher_H", H, teacher["H_original"])
    for ri, r in enumerate(RS):
        W = independent_kernel(r)
        check.near("A_mapping_independent_combinatorics", pstar @ W, mapped[ri])
        check.near("A_total_invariance", mapped[ri] @ g["indicator"], pstar @ g["indicator"])
    grid_rows = csvrows("GRID.csv")
    assert len(grid_rows) == 361
    refs, pgrid, variants, pairs, magnitudes = [], [], [], [], []
    direct, derived = 0, 0
    for i, (u, v) in enumerate(GRID):
        assert int(grid_rows[i]["index"]) == i
        assert float(grid_rows[i]["u"]) == u and float(grid_rows[i]["v"]) == v
        mirror = (i % 19)*19+i//19
        assert int(grid_rows[i]["mirror_index"]) == mirror
        ref = read(OUT / f"reference/g{i:03}.json")
        assert ref["index"] == i and ref["scales"] == [u, v]
        assert ref["u"] == u and ref["v"] == v
        assert boolean(grid_rows[i]["independent_reference"]) == (u <= v)
        pp, vv, pair, mags = reference_check(ref, g, check)
        if u <= v:
            direct += 1
            assert ref["independent_reference"] and not ref["derived_by_exact_transpose"]
        else:
            derived += 1
            source_path = OUT / f"reference/g{mirror:03}.json"
            source = read(source_path)
            assert ref["source_index"] == mirror and ref["source_sha256"] == sha(source_path)
            assert ref["derived_by_exact_transpose"] and not ref["independent_reference"]
            for name in ("mean", "variants", "raw_variants", "symmetrized_group_p"):
                source_array = np.array(source[name])
                transposed = source_array.reshape(source_array.shape[:-1]+(9, 9)).swapaxes(-2, -1).reshape(source_array.shape)
                check.near("exact_transpose_"+name, ref[name], transposed, tol=0.)
            check.near("transpose_H", ref["H"], source["H"], tol=0.)
            check.near("transpose_total", ref["total_p"], source["total_p"], tol=0.)
            check.near("transpose_AA_BB", ref["independent_Hermite_moments"]["pair_magnitudes_AA_BB_AB"],
                       np.array(source["independent_Hermite_moments"]["pair_magnitudes_AA_BB_AB"])[[1, 0, 2]], tol=0.)
        refs.append(ref)
        pgrid.append(pp)
        variants.append(vv)
        pairs.append(pair)
        magnitudes.append(mags)
    assert (direct, derived) == (190, 171)
    assert len(list((OUT / "reference").glob("g*.json"))) == 361
    pgrid, variants, P, mags = map(np.array, (pgrid, variants, pairs, magnitudes))
    shared_total = variants @ g["indicator"]
    for i in range(361):
        j = (i % 19)*19+i//19
        if i <= j:
            common = .5*(shared_total[i]+shared_total[j])
            shared_total[i], shared_total[j] = common, common
        assert np.array_equal(shared_total[i], shared_total[j])
    J = -np.log(pgrid)+g["lg"]
    T = -np.log(shared_total[:, 2])
    risk = mapped @ J.T-H
    exactJ, exactP = mapped @ J.T, mapped @ P.T
    Tpop = T @ (pstar @ g["indicator"])
    anchor = read(OUT / "REFERENCE_ANCHOR.json")
    error = float(abs(pgrid[12*19+4]-pstar).sum())
    assert error <= 1e-8 and anchor["passed"]
    check.near("teacher_anchor", anchor["L1"], error)
    hv = np.array([entropy(p, g) for p in teacher["variants"]])
    differences = []
    for ri in range(4):
        risks = [mappedv[ri, va] @ (-np.log(variants[:, vb])+g["lg"]).T-hv[va]
                 for va in range(3) for vb in range(3)]
        differences.append(float(np.ptp(risks, axis=0).max()))
    totals = [(-np.log(shared_total[:, vb])) @ (np.array(teacher["variants"])[va] @ g["indicator"])
              for va in range(3) for vb in range(3)]
    sensitivity = max(differences+[float(np.ptp(totals, axis=0).max())])
    assert sensitivity <= 1e-7
    return dict(refs=refs, pgrid=pgrid, variants=variants, P=P, mags=mags, J=J, T=T,
                shared_total=shared_total, pstar=pstar, mapped=mapped, mappedv=mappedv,
                teacher=teacher, H=H, risk=risk, exactJ=exactJ, exactP=exactP,
                exactT=Tpop, best=risk.min(axis=1), sensitivity=sensitivity)


def oracle_and_js(info, g, check):
    diag = read(OUT / "ORACLE_DIAGNOSTIC.json")
    ref = read(OUT / "ORACLE_REFERENCE.json")
    pp, vv, pair, mags = reference_check(ref, g, check)
    w = float(diag["continuous_scales"][0])
    assert diag["continuous_scales"] == [w, w] and 2 < w < 6
    assert ref["scales"] == [w, w]
    assert (w, w) not in GRID and diag["not_added_to_fitting_grid"]
    assert diag["not_a_finite_grid_diagonal_theorem"]
    teacher_mags = info["mags"][12*19+4]
    rA, rB, rAB = teacher_mags
    alpha = (6*(rA+rB)+16*rAB)/28
    beta = (rA+rB+2*rAB)/4
    rbar = (28*rA+28*rB+64*rAB)/120
    totalspin = g["sa"]+g["sb"]
    check.near("oracle_global_pair_average", rbar, (16-info["pstar"] @ (totalspin**2))/240)
    check.near("oracle_rbar", diag["rbar"], rbar)
    check.near("oracle_root_residual", diag["root_residual"], abs(float(mags[0])-rbar))
    assert abs(float(mags[0])-rbar) < 1e-11
    assert beta > alpha and alpha < rbar < beta
    target = info["mapped"][3]
    mapped_magnitudes = np.array([(8-target @ (g["sa"]**2))/56,
                                 (8-target @ (g["sb"]**2))/56,
                                 -target @ (g["sa"]*g["sb"])/64])
    check.near("half_mixing_pair_parameters", mapped_magnitudes, [alpha, alpha, beta])
    # KKT for g(x,y,z)=2z-x-y <=0 and unnormalized weights 28,28,64.
    gradient = np.array([28*(rbar-alpha), 28*(rbar-alpha), 64*(rbar-beta)])/(1-rbar*rbar)
    multiplier = float(gradient[0])
    assert multiplier > 0
    stationarity = gradient+multiplier*np.array([-1., -1., 2.])
    check.near("oracle_KKT_stationarity", stationarity, np.zeros(3), tol=1e-10)
    hessian_diag = np.array([28*(1+rbar*rbar-2*alpha*rbar),
                            28*(1+rbar*rbar-2*alpha*rbar),
                            64*(1+rbar*rbar-2*beta*rbar)])/(1-rbar*rbar)**2
    assert np.all(hessian_diag > 0)
    assert np.min(info["mags"][:, 0]+info["mags"][:, 1]-2*info["mags"][:, 2]) > -1e-10
    def ell(real, proposed):
        return math.log(4)-.5*(1-real)*math.log1p(-proposed)-.5*(1+real)*math.log1p(proposed)
    lower = (56*ell(alpha, rbar)+64*ell(beta, rbar))/120
    check.near("oracle_halfspace_bound", diag["pair_halfspace_lower_bound"], lower)
    check.near("oracle_pair_loss", diag["pair_loss"], float(pair @ target))
    check.near("oracle_attains_halfspace_bound", float(pair @ target), lower, tol=1e-10)
    joint = float((-np.log(pp)+g["lg"]) @ target-info["H"])
    check.near("oracle_original_teacher_joint_KL", diag["continuous_oracle_joint_KL"], joint)
    i = min(range(361), key=lambda j: float(info["exactP"][3, j]))
    gridmin = float(info["exactP"][3, i])
    check.near("oracle_grid_gap", diag["grid_minus_continuous_pair_gap"], gridmin-lower)
    assert gridmin-lower >= -1e-10
    assert diag["grid_selected_index"] == i and diag["grid_selected_scales"] == list(GRID[i])
    assert diag["grid_selected_diagonal"] == (GRID[i][0] == GRID[i][1])
    obstruction = read(OUT / "INFO_OBSTRUCTION.json")
    metrics = []
    count_errors, direction_errors = [], []
    plus, minus = 12*19+4, 4*19+12
    for vi in range(3):
        p, q = info["variants"][plus, vi], info["variants"][minus, vi]
        mid = (p+q)/2
        a = math.fsum(float(x)*math.log(float(x/y)) for x, y in zip(p, mid))
        b = math.fsum(float(x)*math.log(float(x/y)) for x, y in zip(q, mid))
        forward = math.fsum(float(x)*math.log(float(x/y)) for x, y in zip(p, q))
        reverse = math.fsum(float(x)*math.log(float(x/y)) for x, y in zip(q, p))
        values = [(.5*(a+b)), a, b, forward, reverse]
        for key, value in zip(("JS", "KL_plus_mid", "KL_minus_mid", "KL_plus_minus", "KL_minus_plus"), values):
            check.near("JS_"+key, obstruction["variants"][vi][key], value)
        metrics.append(values)
        count_errors.append(float(np.max(abs(p @ g["indicator"]-q @ g["indicator"]))))
        direction_errors.extend([abs(a-b), abs(forward-reverse)])
    check.near("JS_primary", obstruction["primary_JS"], metrics[2][0])
    check.near("JS_minimax_readout", obstruction["count_only_minimax_lower_bound_for_any_training_size"], metrics[2][0])
    check.near("JS_count_identity", obstruction["raw_total_count_max_difference"], max(count_errors))
    assert np.array_equal(info["shared_total"][plus], info["shared_total"][minus])
    assert obstruction["shared_T_total_count_difference"] == 0
    check.near("JS_reference_sensitivity", obstruction["reference_sensitivity"], np.ptp(metrics, axis=0).max())
    assert obstruction["numerical_pass"] and obstruction["additional_references"] == 0
    return dict(KKT_multiplier=multiplier, KKT_stationarity_max=float(abs(stationarity).max()),
                continuous_scale=w, pair_halfspace_lower_bound=lower, grid_pair_gap=gridmin-lower,
                pair_loss_unit="nats per pair", oracle_full_joint_KL=joint,
                JS=metrics[2][0], direction_KL=metrics[2][3],
                original_count_identity_max=max(count_errors), shared_T_count_identity_exact=True,
                continuous_oracle_not_added_to_grid=True)


@lru_cache(maxsize=None)
def critical(family):
    return brentq(lambda z: stdtr(15, z)-(1-.05/(2*family)), 0., 100., xtol=1e-13, rtol=9e-16)


def interval(values, family, gate):
    values = [float(x) for x in values]
    assert len(values) == 16
    avg = math.fsum(values)/16
    se = math.sqrt(math.fsum((x-avg)**2 for x in values)/(15*16))
    radius = critical(family)*se
    return dict(mean=avg, SE=se, lower=avg-radius, upper=avg+radius,
                precision_pass=se <= gate)


def csv_index(name, keys):
    result = {}
    for row in csvrows(name):
        key = tuple(row[k] for k in keys)
        assert key not in result
        result[key] = row
    return result


def streams(info, g, check, workspace):
    records, counts_list = [], []
    receipts = {}
    ids = np.arange(65536, dtype=np.int64) if workspace else None
    bits = ((ids[:, None] >> np.arange(16)) & 1).astype(np.int8) if workspace else None
    for s in range(16):
        arec = read(A / f"streams/s{s:02}.json")
        rec = read(OUT / f"streams/s{s:02}.json")
        assert rec["stream"] == s and rec["seed"] == 20340101+s
        assert rec["initial_RNG"] != rec["final_RNG"]
        assert rec["bank_count"] == 64 and rec["unique_prior_particles"] == 6400
        assert rec["candidate_particle_evaluations"] == 2310400
        assert rec["A_stream_sha256"] == sha(A / f"streams/s{s:02}.json")
        assert rec["numerical_pass"] and rec["normalization_error"] < 1e-12
        assert rec["count_identity_error"] < 1e-10 and rec["propagated_reference_sensitivity"] <= 1e-7
        counts = np.array(arec["training_counts_by_r"])
        assert counts.shape == (4, 2, 81)
        assert np.array_equal(counts, counts.astype(np.int64)) and np.all(counts >= 0)
        assert np.array_equal(counts.sum(axis=-1), np.tile(NS, (4, 1)))
        assert np.all(counts[:, 1] >= counts[:, 0])
        ncounts = counts @ g["indicator"].astype(np.int64)
        assert np.array_equal(ncounts, np.repeat(ncounts[:1], 4, axis=0))
        pop, finite = np.array(rec["population_objectives"]), np.array(rec["finite_objectives"])
        assert pop.shape == (4, 361, 2) and finite.shape == (4, 2, 361, 2)
        assert np.isfinite(pop).all() and np.isfinite(finite).all()
        receipts[rec["array_path"]] = rec["array_sha256"]
        receipts[arec["array_path"]] = arec["array_sha256"]
        if workspace:
            check.hashes({rec["array_path"]: rec["array_sha256"], arec["array_path"]: arec["array_sha256"]})
            with np.load(ROOT / arec["array_path"], allow_pickle=False) as aa:
                hist = aa["pattern_histograms"]
                assert hist.shape == (2, 65536) and np.issubdtype(hist.dtype, np.integer)
                assert np.array_equal(hist.sum(axis=1), NS) and np.all(hist[1] >= hist[0])
                for ri, r in enumerate(RS):
                    coordinates = list(range(r, 8))+list(range(8, 8+r))
                    nA = bits[:, coordinates].sum(axis=1)
                    nB = bits.sum(axis=1)-nA
                    cells = 9*nA+nB
                    for ni in range(2):
                        rebuilt = np.bincount(cells, weights=hist[ni], minlength=81)
                        assert np.array_equal(rebuilt, counts[ri, ni])
                assert np.array_equal(aa["training_counts_by_r"], counts)
            with np.load(ROOT / rec["array_path"], allow_pickle=False) as bb:
                assert set(bb.files) == {"mean_pattern_loss", "population_objectives", "finite_objectives"}
                logs = bb["mean_pattern_loss"]
                assert logs.shape == (361, 2, 81) and np.isfinite(logs).all()
                check.near("B_public_population_array", bb["population_objectives"], pop, tol=0.)
                check.near("B_public_finite_array", bb["finite_objectives"], finite, tol=0.)
                for ri in range(4):
                    check.near("B_population_from_pattern_logs", pop[ri], np.tensordot(logs, info["mapped"][ri], axes=([-1], [0])))
                    for ni, n in enumerate(NS):
                        check.near("B_finite_from_pattern_logs", finite[ri, ni],
                                   np.tensordot(logs, counts[ri, ni]/n, axes=([-1], [0])))
                sensitivity = 0.
                for ri in range(4):
                    values = np.einsum("gka,va->vgk", logs, info["mappedv"][ri])
                    difference = values[:, 6*19+2]-values[:, 12*19+4]
                    sensitivity = max(sensitivity, float(np.ptp(values, axis=0).max()),
                                      float(np.ptp(difference, axis=0).max()))
                assert sensitivity <= 1e-7
                check.near("B_reference_sensitivity_receipt", rec["propagated_reference_sensitivity"], sensitivity)
        records.append(rec)
        counts_list.append(counts)
    return (np.array([r["population_objectives"] for r in records]),
            np.array([r["finite_objectives"] for r in records]),
            np.array(counts_list), receipts)


def check_interval(check, row, expected, name):
    for field in ("mean", "SE", "lower", "upper"):
        check.near(name+"_"+field, row[field], expected[field])
    assert boolean(row["precision_pass"]) == expected["precision_pass"]


def audit(workspace=False):
    ck = AuditChecks()
    ck.seal(OUT / "FREEZE.json")
    reference_seal = ck.seal(OUT / "REFERENCE_FREEZE.json")
    assert reference_seal["independent_grid_references"] == 190
    assert reference_seal["transpose_grid_references"] == 171
    assert reference_seal["diagnostic_full_references"] == reference_seal["continuous_population_solutions"] == 1
    ck.seal(OUT / "TRAINING_INPUTS_FREEZE.json")
    ck.seal(OUT / "RESULTS_FREEZE.json")
    assert read(OUT / "STATE.json")["status"] == "complete"
    for name in ("FREEZE.json", "DATA_FREEZE.json", "RESULTS_FREEZE.json", "MANIFEST.json"):
        ck.seal(A / name)
    cfg = read(OUT / "CONFIG.json")
    assert cfg["objectives"] == OBJECTIVES and cfg["r_values"] == RS and cfg["train_N"] == NS
    assert cfg["finite_family"] == 32 and cfg["population_family"] == 8
    g = geometry()
    ck.near("integer_vs_lgamma_multiplicity", g["lg"], np.log(g["mult"]), tol=1e-12)
    info = load_references(g, ck)
    oracle = oracle_and_js(info, g, ck)
    pop, finite, counts, local_receipts = streams(info, g, ck, workspace)
    reference_rows = csv_index("REFERENCE_CURVES.csv", ["r", "index"])
    population_fits = csv_index("POPULATION_FITS.csv", ["r", "unit", "objective"])
    finite_fits = csv_index("FINITE_FITS.csv", ["r", "N", "stream", "objective"])
    intervals = csv_index("FINITE_RISK_INTERVALS.csv", ["r", "N", "comparison"])
    fixed = csv_index("POPULATION_FIXED_COMPARISONS.csv", ["r", "K"])
    assert [len(x) for x in (reference_rows, population_fits, finite_fits, intervals, fixed)] == [1444, 148, 640, 32, 8]
    tie_rows = read(OUT / "T_EQUIVALENCE_CLASSES.json")
    ties = {(r["r"], str(r["stream"]), r["N"]): r for r in tie_rows}
    assert len(ties) == len(tie_rows) == 132
    risks = np.zeros((4, 2, 16, 5))
    expected_primary, finite_failures, population_failures = [], [], []
    counts_checked = dict(finite=0, aggregate=0, bank_only=0, ties=0)

    def endpoint(row, curve, ri, category):
        chosen = min(range(361), key=lambda j: float(curve[j]))
        assert int(row["index"]) == chosen, (category, row, chosen)
        assert float(row["u"]) == GRID[chosen][0] and float(row["v"]) == GRID[chosen][1]
        field = "train_objective" if category == "finite" else "objective_value"
        ck.near(category+"_selected_objective", row[field], curve[chosen])
        ck.near(category+"_full_Q_KL", row["joint_KL"], info["risk"][ri, chosen])
        ck.near(category+"_excess_over_best_J_grid", row["excess_over_grid_J"], info["risk"][ri, chosen]-info["best"][ri])
        assert boolean(row["grid_selected_diagonal"]) == (GRID[chosen][0] == GRID[chosen][1])
        counts_checked[category] += 1
        return chosen

    def tie_check(r, stream, n, curve, ri):
        row = ties[(r, str(stream), n)]
        best = min(float(v) for v in curve)
        exact = [i for i, v in enumerate(curve) if float(v) == best]
        near = [i for i, v in enumerate(curve) if float(v) <= best+1e-7]
        assert row["exact_indices"] == exact and row["near_indices"] == near
        assert row["near_allowance"] == 1e-7 and row["orientation_success_not_claimed"]
        assert row["tie_break"] == "first lexicographic index, smaller u"
        ck.near("T_exact_minimum", row["exact_minimum"], best)
        assert len(row["exact_candidates"]) == len(exact)
        for item, i in zip(row["exact_candidates"], exact):
            mirror = (i % 19)*19+i//19
            assert mirror in exact and item["index"] == i and item["mirror_index"] == mirror
            assert [item["u"], item["v"]] == list(GRID[i])
            ck.near("T_equivalent_Q_risk", item["joint_KL"], info["risk"][ri, i])
            ck.near("T_mirror_Q_risk", item["mirror_joint_KL"], info["risk"][ri, mirror])
        counts_checked["ties"] += 1

    for ri, r in enumerate(RS):
        for i in range(361):
            row = reference_rows[(str(r), str(i))]
            for key, expected in (
                ("joint_KL", info["risk"][ri, i]),
                ("excess_over_grid_J", info["risk"][ri, i]-info["best"][ri]),
                ("J", info["exactJ"][ri, i]), ("P", info["exactP"][ri, i]),
                ("T", info["exactT"][i]), ("reference_sensitivity", info["sensitivity"])):
                ck.near("reference_curve_"+key, row[key], expected)
        deterministic = {"J": info["exactJ"][ri], "P": info["exactP"][ri], "T": info["exactT"]}
        for objective in OBJECTIVES:
            curve = deterministic[objective] if objective in deterministic else pop[:, ri, :, [10, 100].index(int(objective[1:]))].mean(axis=0)
            row = population_fits[(str(r), "aggregate_or_deterministic", objective)]
            endpoint(row, curve, ri, "aggregate")
            assert not boolean(row["exact_continuous_optimum"])
            if objective == "T":
                tie_check(r, "population", None, curve, ri)
        for ki, k in enumerate((10, 100)):
            expected = interval(pop[:, ri, 6*19+2, ki]-pop[:, ri, 12*19+4, ki], 8, [.01, .005][ki])
            check_interval(ck, fixed[(str(r), str(k))], expected, "population_CI")
            if not expected["precision_pass"]:
                population_failures.append(dict(r=r, K=k, SE=expected["SE"]))
            if r == 0 and k == 10:
                expected_primary.append(dict(kind="population_fixed_witness", r=0,
                    passed=expected["precision_pass"] and expected["upper"] < -.001))
            for s in range(16):
                endpoint(population_fits[(str(r), str(s), "G"+str(k))],
                         pop[s, ri, :, ki], ri, "bank_only")
        for s in range(16):
            totalcounts = counts[s, 0] @ g["indicator"].astype(np.int64)
            for ni, n in enumerate(NS):
                empirical = counts[s, ri, ni]/n
                curves = {"J": info["J"] @ empirical, "P": info["P"] @ empirical,
                          "T": info["T"] @ (totalcounts[ni]/n),
                          "G10": finite[s, ri, ni, :, 0], "G100": finite[s, ri, ni, :, 1]}
                for oi, objective in enumerate(OBJECTIVES):
                    row = finite_fits[(str(r), str(n), str(s), objective)]
                    i = endpoint(row, curves[objective], ri, "finite")
                    assert float(row["grid_minimum_gap"]) == 0.
                    risks[ri, ni, s, oi] = info["risk"][ri, i]
                    if objective == "T":
                        tie_check(r, s, n, curves[objective], ri)
        for ni, n in enumerate(NS):
            for oi, objective in enumerate(OBJECTIVES[1:], start=1):
                expected = interval(risks[ri, ni, :, oi]-risks[ri, ni, :, 0], 32, .001)
                check_interval(ck, intervals[(str(r), str(n), objective+"-J")], expected, "finite_CI")
                if not expected["precision_pass"]:
                    finite_failures.append(dict(r=r, N=n, comparison=objective+"-J", SE=expected["SE"]))
                if r == 0 and n == 65536 and objective == "G10":
                    expected_primary.append(dict(kind="finite_data_excess", r=0,
                        passed=expected["precision_pass"] and expected["lower"] > .001))
    assert counts_checked == dict(finite=640, aggregate=20, bank_only=128, ties=132)
    decision = read(OUT / "DECISION.json")
    assert decision["finite_precision_pass"] == 32-len(finite_failures)
    assert decision["population_precision_pass"] == 8-len(population_failures)
    for key, expected in (("finite_grid_fits", 640), ("bank_only_population_grid_fits", 128),
                          ("aggregate_or_deterministic_population_endpoints", 20),
                          ("continuous_population_oracles", 1), ("independent_grid_references", 190),
                          ("derived_transpose_grid_references", 171), ("diagnostic_full_references", 1),
                          ("unique_prior_particles", 102400), ("candidate_particle_evaluations", 36966400),
                          ("new_teacher_observations", 0)):
        assert decision[key] == expected
    assert len(decision["primary"]) == len(expected_primary) == 2
    for item in expected_primary:
        recorded = next(r for r in decision["primary"] if r["kind"] == item["kind"])
        assert recorded["passed"] == item["passed"]
    assert decision["T_orientation_success_not_claimed"] and decision["r4_continuous_pair_theorem_not_a_grid_diagonal_guarantee"]
    return dict(created_utc=now(), passed=True, auditor_source_sha256=sha(__file__),
        seals={name: sha(OUT / name) for name in ("FREEZE.json", "REFERENCE_FREEZE.json",
                "TRAINING_INPUTS_FREEZE.json", "RESULTS_FREEZE.json")},
        independent_implementation_no_B_or_core_imports=True,
        grid_references=361, direct_grid_references=190, mirror_grid_references=171,
        diagnostic_reference=1, continuous_oracle_solutions=1, oracle_and_JS=oracle,
        argmin_counts=counts_checked, finite_CI_rows=32, population_CI_rows=8,
        finite_precision_pass=32-len(finite_failures), population_precision_pass=8-len(population_failures),
        finite_precision_failures=finite_failures, population_precision_failures=population_failures,
        primary=expected_primary, public_full_selection_and_statistics_verified=True,
        workspace=workspace, B_arrays_verified=16 if workspace else 0,
        A_pattern_arrays_verified=16 if workspace else 0, local_receipts=local_receipts,
        numerical_maxima=ck.maxima, hash_checks=ck.hash_checks,
        A_original_check_scope="implementation self-check; not whole-science independent-person validation",
        limitations=[
            "Independent arithmetic replay of saved evidence; no new reference integration or scientific sampling.",
            "Root residual is verified from the saved qualified oracle coefficients and PMF, not by rerunning the root or quadrature.",
            "Public evidence proves saved objectives' argmins and statistics; workspace also links the public curves to saved pattern-loss arrays.",
            "Confidence families remain separate 32 and 8 t-approximation families; reference qualification is conditional.",
        ], new_scientific_samples=0, new_reference_integrations=0, new_fits=0)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--workspace", action="store_true")
    args = parser.parse_args()
    if args.write:
        assert not (OUT.parent / "MANIFEST.json").exists(), "global package sealed; check only"
    try:
        result = audit(args.workspace)
    except Exception as error:
        if args.write:
            fail = OUT / ("AUDIT_FAILURE_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")+".json")
            fail.write_text(json.dumps(dict(created_utc=now(), audit_only=True, error=repr(error),
                traceback=traceback.format_exc(), auditor_source_sha256=sha(__file__),
                scientific_inputs_not_modified=True), ensure_ascii=False, indent=2)+"\n",
                encoding="utf-8", newline="\n")
        raise
    path = OUT / "AUDIT.json"
    if args.write:
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
                        encoding="utf-8", newline="\n")
    else:
        saved = read(path)
        assert saved["passed"] and saved["auditor_source_sha256"] == sha(__file__)
        for key in ("seals", "argmin_counts", "finite_precision_pass", "population_precision_pass",
                    "primary", "local_receipts"):
            assert saved[key] == result[key], key
    print(json.dumps(dict(audit_pass=True, workspace=args.workspace,
        finite_fits=640, population_fits=148, T_classes=132,
        finite_precision_pass=result["finite_precision_pass"],
        population_precision_pass=result["population_precision_pass"],
        new_scientific_samples=0), ensure_ascii=False))


if __name__ == "__main__":
    main()
