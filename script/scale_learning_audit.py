"""Independent replay of saved scale-learning evidence; no scientific reruns."""
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
OUT = ROOT / "artifacts/scale_learning_v2"
GRID = [i/20 for i in range(31)]
PROFILES = {"null": [4., 4.], "contrast": [6., 2.]}
OBJECTIVES = ["J", "P", "G10", "G100", "G1000", "S10", "S100", "S1000"]
KS = [10, 100, 1000]
NS = [4096, 65536]
TOL = 2e-11


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def boolean(value):
    assert value in (True, False, "True", "False")
    return value is True or value == "True"


class Checks:
    def __init__(self):
        self.maxima = {}
        self.hash_count = 0

    def near(self, label, actual, expected, tol=TOL):
        aa, bb = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        assert aa.shape == bb.shape, (label, aa.shape, bb.shape)
        assert np.isfinite(aa).all() and np.isfinite(bb).all(), label
        error = float(np.max(abs(aa-bb))) if aa.size else 0.
        self.maxima[label] = max(self.maxima.get(label, 0.), error)
        assert error <= tol, (label, error, tol)

    def hashes(self, mapping):
        for name, expected in mapping.items():
            assert name != "HANDOFF.md", "live HANDOFF is not a scientific input"
            path = ROOT / name
            assert path.is_file(), f"missing frozen file {name}"
            assert sha(path) == expected, f"hash mismatch {name}"
            self.hash_count += 1

    def manifest(self, path):
        obj = read(path)
        mapping = obj.get("files", obj.get("scientific_files"))
        assert isinstance(mapping, dict), path
        self.hashes(mapping)
        return obj


@lru_cache(maxsize=None)
def critical(family):
    target = 1-.05/(2*family)
    return brentq(lambda x: stdtr(15, x)-target, 0., 100.,
                  xtol=1e-13, rtol=9e-16)


def interval(values, family, gate):
    values = [float(v) for v in values]
    assert len(values) == 16 and all(math.isfinite(v) for v in values)
    mean = math.fsum(values)/16
    se = math.sqrt(math.fsum((v-mean)**2 for v in values)/(15*16))
    radius = critical(family)*se
    return dict(mean=mean, SE=se, lower=mean-radius, upper=mean+radius,
                precision_pass=se <= gate)


def grid_geometry():
    """Independent integer-combination geometry; no core/runner import."""
    cells = [(a, b) for a in range(9) for b in range(9)]
    na = np.array([a for a, _ in cells], dtype=int)
    nb = np.array([b for _, b in cells], dtype=int)
    total = na+nb
    multiplicity = np.array([math.comb(8, a)*math.comb(8, b) for a, b in cells], dtype=float)
    totalmult = np.array([math.comb(16, int(n)) for n in total], dtype=float)
    indicator = (total[:, None] == np.arange(17)[None, :]).astype(float)
    # Explicitly visit all 120 pairs of every canonical count pattern.
    pair_counts = np.zeros((81, 3, 4), dtype=int)
    for cell, (a, b) in enumerate(cells):
        pattern = [1]*a+[0]*(8-a)+[1]*b+[0]*(8-b)
        for i, j in itertools.combinations(range(16), 2):
            kind = 0 if j < 8 else 1 if i >= 8 else 2
            outcome = 2*pattern[i]+pattern[j]
            pair_counts[cell, kind, outcome] += 1
    assert np.array_equal(pair_counts.sum(axis=2), np.broadcast_to([28, 28, 64], (81, 3)))
    assert np.all(pair_counts.sum(axis=(1, 2)) == 120)
    sa, sb = 2*na-8, 2*nb-8
    pair_spins = np.column_stack([(sa*sa-8)//2, (sb*sb-8)//2, sa*sb])
    same = pair_counts[:, :, 0]+pair_counts[:, :, 3]
    assert np.array_equal(2*same-np.array([28, 28, 64]), pair_spins)
    return dict(na=na, nb=nb, total=total, multiplicity=multiplicity,
                totalmult=totalmult, indicator=indicator, pair_counts=pair_counts,
                sa=sa, sb=sb)


def own_functionals(p, geo):
    count = p @ geo["indicator"]
    sym = count[geo["total"]]*geo["multiplicity"]/geo["totalmult"]
    h = -math.fsum(float(w)*math.log(float(w/m))
                   for w, m in zip(p, geo["multiplicity"]))
    kl = math.fsum(float(w)*math.log(float(w/q)) for w, q in zip(p, sym))
    return h, kl, count, sym


def pair_vector(ref, geo, check):
    """Use saved independent Hermite coefficients and explicit 120-pair counts."""
    if ref["analytic_independence"]:
        corr = np.zeros(3)
    else:
        sources = {float(x["scale"]): x for x in ref["independent_Hermite_moments"]["coefficient_sources"]}
        scales = ref["scales"]
        rho = ref["c"]/(ref["d"]-1)
        corr = []
        for left, right in ((scales[0], scales[0]), (scales[1], scales[1]), (scales[0], scales[1])):
            if left == 0 or right == 0:
                corr.append(0.)
            else:
                aa, bb = sources[left]["coefficients"], sources[right]["coefficients"]
                assert len(aa) == len(bb) == 7
                corr.append(-math.fsum(float(a)*float(b)*rho**k
                                       for a, b, k in zip(aa, bb, range(1, 14, 2))))
        corr = np.asarray(corr)
    receipt = ref["independent_Hermite_moments"]
    check.near("pair_coefficient_replay", corr, -np.asarray(receipt["pair_magnitudes_AA_BB_AB"]))
    pmf = np.array([[(1+r)/4, (1-r)/4, (1-r)/4, (1+r)/4] for r in corr])
    assert np.all(pmf > 0) and np.isfinite(pmf).all()
    # Explicit count-weighted four-outcome likelihood, unlike the core log1p formula.
    vector = -np.sum(geo["pair_counts"]*np.log(pmf)[None, :, :], axis=(1, 2))/120
    p = np.asarray(ref["mean"])
    sa, sb = geo["sa"], geo["sb"]
    pmf_corr = np.array([(p @ (sa*sa)-8)/56, (p @ (sb*sb)-8)/56, p @ (sa*sb)/64])
    check.near("pair_vs_observed_moments", pmf_corr, corr, tol=1e-8)
    return vector


def references(profile, geo, check):
    refs, pair = [], []
    for index, amplitude in enumerate(GRID):
        path = OUT / f"reference/{profile}_t{index:02}.json"
        ref = read(path)
        assert ref["profile"] == profile and ref["amplitude"] == amplitude
        assert ref["d"] == 16 and ref["c"] == .9
        check.near("candidate_scales", ref["scales"], np.asarray(PROFILES[profile])*amplitude)
        assert ref["qualified"] and ref["numerical_pass"]
        p, variants, raw = [np.asarray(ref[k], dtype=float) for k in ("mean", "variants", "raw_variants")]
        assert p.shape == (81,) and variants.shape == raw.shape == (3, 81)
        assert np.isfinite(raw).all() and np.all(raw > 0)
        assert np.max(abs(raw.sum(axis=1)-1)) < 1e-12
        check.near("variant_primary", p, variants[2])
        check.near("roundoff_normalization", variants, raw/raw.sum(axis=1, keepdims=True))
        assert max(np.abs(variants[i]-variants[j]).sum() for i in range(3) for j in range(i)) <= 1e-8
        h, kl, count, sym = own_functionals(p, geo)
        check.near("reference_H", ref["H"], h)
        check.near("reference_self_KL", ref["model_KL_single"], kl)
        check.near("reference_total", ref["total_p"], count)
        check.near("reference_symmetrized", ref["symmetrized_group_p"], sym)
        sa, sb = geo["sa"], geo["sb"]
        empirical_moments = variants @ np.column_stack([sa*sa, sb*sb, sa*sb])
        expected_moments = np.asarray(ref["independent_Hermite_moments"]["expected_spin_moments"])
        assert np.max(abs(empirical_moments-expected_moments)) <= 1e-8
        check.near("stored_reference_moments", ref["variant_spin_moments"], empirical_moments)
        check.near("fair_group_marginals", p @ np.column_stack([sa, sb]), [0., 0.])
        if index == 0:
            check.near("analytic_t0", p, geo["multiplicity"]/65536, tol=0.)
            assert ref["analytic_independence"]
        if index == 20:
            old = read(ROOT / f"artifacts/structure_tradeoff_v2/R2/reference/{profile}_d16.json")
            error = float(np.abs(p-old["mean"]).sum())
            assert error <= 1e-8
            anchor = read(OUT / f"reference/{profile}_OLD_ANCHOR.json")
            assert anchor["passed"]
            check.near("old_reference_anchor", anchor["L1"], error)
        refs.append(ref)
        pair.append(pair_vector(ref, geo, check))
    q = np.array([ref["mean"] for ref in refs])
    variants = np.array([ref["variants"] for ref in refs])
    p = q[20]
    teacher_H, _, pn, _ = own_functionals(p, geo)
    qn = q @ geo["indicator"]
    sym = qn[:, geo["total"]]*geo["multiplicity"]/geo["totalmult"]
    joint_loss = -np.log(q/geo["multiplicity"])
    single_loss = -np.log(sym/geo["multiplicity"])
    pair_loss = np.array(pair)
    true_kl = np.sum(p[None, :]*np.log(p[None, :]/q), axis=1)
    deployed_kl = np.sum(p[None, :]*np.log(p[None, :]/sym), axis=1)
    count_kl = np.sum(pn[None, :]*np.log(pn[None, :]/qn), axis=1)
    risks = []
    for pi in range(3):
        pv = variants[20, pi]
        for qi in range(3):
            qq = variants[:, qi]
            nn = qq @ geo["indicator"]
            ss = nn[:, geo["total"]]*geo["multiplicity"]/geo["totalmult"]
            risks.append([np.sum(pv*np.log(pv/qq), axis=1),
                          np.sum(pv*np.log(pv/ss), axis=1)])
    sensitivity = float(np.ptp(np.array(risks), axis=0).max())
    assert sensitivity <= 1e-7
    return dict(refs=refs, p=p, variants=variants, H=teacher_H, joint=joint_loss,
                single=single_loss, pair=pair_loss, qkl=true_kl, skl=deployed_kl,
                countkl=count_kl, sensitivity=sensitivity)


def indexed_csv(name, keys):
    source = rows(OUT / name)
    result = {}
    for row in source:
        key = tuple(row[k] for k in keys)
        assert key not in result, (name, key)
        result[key] = row
    return result


def stream_surfaces(profile, pi, info, check, workspace):
    populations, particles, counts_list = [], [], []
    array_receipts = {}
    for stream in range(16):
        rec = read(OUT / f"streams/{profile}_s{stream:02}.json")
        assert rec["profile"] == profile and rec["stream"] == stream
        assert rec["numerical_pass"]
        assert rec["bank_seed"] == 20310101+1000*pi+stream
        assert rec["data_seed"] == 20320101+1000*pi+stream
        assert rec["bank_initial_RNG"] != rec["bank_final_RNG"]
        assert rec["data_initial_RNG"] != rec["data_final_RNG"]
        assert rec["bank_count"] == 64 and rec["unique_prior_particles"] == 64000
        assert rec["candidate_particle_evaluations"] == 1984000
        assert rec["count_observations"] == 65536
        assert rec["normalization_error"] < 1e-12 and rec["count_identity_error"] < 1e-10
        assert rec["propagated_reference_sensitivity"] <= 1e-7
        counts = np.asarray(rec["training_counts"])
        assert counts.shape == (2, 81) and np.all(counts >= 0)
        assert np.array_equal(counts, counts.astype(np.int64))
        assert np.array_equal(counts.sum(axis=1), NS)
        assert np.all(counts[1] >= counts[0])
        if "population_objectives" not in rec or "finite_particle_objectives" not in rec:
            raise AssertionError("public objective surfaces absent: cannot claim full public fit/statistics audit")
        pop = np.asarray(rec["population_objectives"], dtype=float)
        finite = np.asarray(rec["finite_particle_objectives"], dtype=float)
        assert pop.shape == (31, 3, 2) and finite.shape == (2, 31, 3, 2)
        assert np.isfinite(pop).all() and np.isfinite(finite).all()
        check.near("t0_population_particle_loss", pop[0], np.full((3, 2), 16*math.log(2)))
        check.near("t0_finite_particle_loss", finite[:, 0], np.full((2, 3, 2), 16*math.log(2)))
        path = ROOT / rec["array_path"]
        expected_path = OUT / f"local/{profile}_s{stream:02}.npz"
        assert path.resolve() == expected_path.resolve()
        array_receipts[rec["array_path"]] = rec["array_sha256"]
        if workspace:
            check.hashes({rec["array_path"]: rec["array_sha256"]})
            with np.load(path, allow_pickle=False) as array:
                assert set(array.files) == {"mean_pattern_loss", "bank_population_scores", "training_counts"}
                logs = array["mean_pattern_loss"]
                scores = array["bank_population_scores"]
                saved_counts = array["training_counts"]
                assert logs.shape == (31, 3, 2, 81) and scores.shape == (64, 31, 3, 2)
                assert np.isfinite(logs).all() and np.isfinite(scores).all()
                assert np.array_equal(saved_counts, counts)
                recomputed_pop = np.tensordot(logs, info["p"], axes=([-1], [0]))
                check.near("public_population_from_local_pattern_losses", pop, recomputed_pop)
                check.near("per_bank_population_mean", scores.mean(axis=0), recomputed_pop)
                for ni, n in enumerate(NS):
                    expected = np.tensordot(logs, counts[ni]/n, axes=([-1], [0]))
                    check.near("public_finite_from_local_pattern_losses", finite[ni], expected)
                variants = np.einsum("va,tkma->vtkm", info["variants"][20], logs)
                diff = variants-variants[:, 20:21]
                sensitivity = max(float(np.ptp(variants, axis=0).max()),
                                  float(np.ptp(diff, axis=0).max()))
                assert sensitivity <= 1e-7
                check.near("bank_reference_sensitivity", rec["propagated_reference_sensitivity"], sensitivity)
        populations.append(pop)
        particles.append(finite)
        counts_list.append(counts)
    return np.array(populations), np.array(particles), np.array(counts_list), array_receipts


def check_stat(check, row, expected, prefix):
    for key in ("mean", "SE", "lower", "upper"):
        check.near(prefix+"_"+key, row[key], expected[key])
    assert boolean(row["precision_pass"]) == expected["precision_pass"]


def risk_at(info, objective, index):
    return float((info["skl"] if objective.startswith("S") else info["qkl"])[index])


def historical(check):
    v1_path = ROOT / "artifacts/paper_v1_final/MANIFEST.json"
    v1 = check.manifest(v1_path)
    lock = read(ROOT / "artifacts/paper_v1_final/VERSION_LOCK.json")
    assert lock["status"] == "frozen" and lock["manifest_sha256"] == sha(v1_path)
    previous = check.manifest(ROOT / "artifacts/structure_tradeoff_v2/MANIFEST.json")
    assert previous.get("live_HANDOFF_excluded", True)
    if "old_science_anchors" in previous:
        check.hashes(previous["old_science_anchors"])
    return dict(v1_scientific_files=len(v1["files"]),
                prior_structure_scientific_files=len(previous["scientific_files"]),
                live_navigation_bound=False, live_HANDOFF_bound=False)


def audit(workspace=False):
    check = Checks()
    freeze = check.manifest(OUT / "FREEZE.json")
    check.manifest(OUT / "TRAINING_INPUTS_FREEZE.json")
    check.manifest(OUT / "RESULTS_FREEZE.json")
    assert read(OUT / "STATE.json")["status"] == "complete"
    cfg = read(OUT / "CONFIG.json")
    assert cfg["grid"] == GRID and cfg["profiles"] == PROFILES and cfg["objectives"] == OBJECTIVES
    assert cfg["d"] == 16 and cfg["c"] == .9
    assert cfg["K"] == KS and cfg["streams"] == 16 and cfg["banks"] == 64
    assert cfg["population_family"] == 360 and cfg["fit_family"] == 20
    old = historical(check)
    geo = grid_geometry()
    curves = indexed_csv("REFERENCE_CURVES.csv", ["profile", "t"])
    bands = indexed_csv("POPULATION_BANDS.csv", ["profile", "method", "K", "t"])
    popfits = indexed_csv("POPULATION_FITS.csv", ["profile", "unit", "objective"])
    fits = indexed_csv("FINITE_FITS.csv", ["profile", "stream", "N", "objective"])
    risk_intervals = indexed_csv("FINITE_RISK_INTERVALS.csv", ["profile", "N", "comparison"])
    assert [len(x) for x in (curves, bands, popfits, fits, risk_intervals)] == [62, 360, 208, 512, 20]
    set_records = read(OUT / "SELECTION_SETS.json")
    selections = {(r["profile"], r["method"], r["K"]): r for r in set_records}
    assert len(selections) == len(set_records) == 12
    raw_records = read(OUT / "RAW_LOCAL_DIAGNOSTIC.json")
    raw = {r["profile"]: r for r in raw_records}
    assert len(raw) == len(raw_records) == 2
    population_pass, finite_pass = 0, 0
    population_failures, finite_failures, primaries = [], [], []
    local_receipts, matched_fits, matched_popfits = {}, 0, 0
    for pi, profile in enumerate(PROFILES):
        info = references(profile, geo, check)
        p = info["p"]
        pop, particle, counts, receipts = stream_surfaces(profile, pi, info, check, workspace)
        local_receipts.update(receipts)
        for ti, amplitude in enumerate(GRID):
            row = curves[(profile, str(amplitude))]
            for field, expected in (
                ("true_Q_KL", info["qkl"][ti]), ("deployed_S_KL", info["skl"][ti]),
                ("total_count_KL", info["countkl"][ti]),
                ("exact_joint_loss", float(info["joint"][ti] @ p)),
                ("pair_loss", float(info["pair"][ti] @ p)),
                ("exact_single_loss", float(info["single"][ti] @ p)),
                ("reference_sensitivity", info["sensitivity"])):
                check.near("curve_"+field, row[field], expected)
        aggregate_curves = {
            "J": info["joint"] @ p, "P": info["pair"] @ p,
            **{method+str(k): pop[:, :, ki, mi].mean(axis=0)
               for mi, method in enumerate(("G", "S")) for ki, k in enumerate(KS)}}
        for objective, curve in aggregate_curves.items():
            index = min(range(31), key=lambda j: float(curve[j]))
            row = popfits[(profile, "aggregate", objective)]
            assert int(row["selected_index"]) == index and float(row["t"]) == GRID[index]
            check.near("aggregate_objective_value", row["objective_value"], curve[index])
            check.near("aggregate_Q_recovery", row["true_Q_KL"], info["qkl"][index])
            check.near("aggregate_deployed_risk", row["deployed_KL"], risk_at(info, objective, index))
            matched_popfits += 1
        for mi, method in enumerate(("G", "S")):
            for ki, k in enumerate(KS):
                allbands = [None]*31
                allbands[20] = dict(lower=0., upper=0., precision_pass=True)
                for ti, amplitude in enumerate(GRID):
                    if ti == 20:
                        continue
                    values = pop[:, ti, ki, mi]-pop[:, 20, ki, mi]
                    expected = interval(values, 360, [.01, .005, .002][ki])
                    row = bands[(profile, method, str(k), str(amplitude))]
                    check_stat(check, row, expected, "population_band")
                    allbands[ti] = expected
                    population_pass += int(expected["precision_pass"])
                    if not expected["precision_pass"]:
                        population_failures.append(dict(profile=profile, method=method, K=k,
                                                        t=amplitude, SE=expected["SE"]))
                    if method == "G" and k == 10 and ti == 10:
                        passed = expected["precision_pass"] and expected["upper"] < -.001 and info["qkl"][10] > .001
                        primaries.append(dict(kind="population_fixed_witness", profile=profile,
                                              passed=bool(passed)))
                cap = min(row["upper"] for row in allbands)
                selected = [j for j, row in enumerate(allbands) if row["lower"] <= cap]
                recorded = selections[(profile, method, k)]
                assert recorded["indices"] == selected and recorded["grid"] == [GRID[j] for j in selected]
                assert recorded["all_precision_pass"] == all(row["precision_pass"] for row in allbands)
                assert recorded["truth_excluded"] == (20 not in selected)
                assert recorded["confidence_family"] == 360
                assert recorded["t_approximation_not_finite_sample_certificate"]
                check.near("selection_min_upper", recorded["min_upper"], cap)
                check.near("selection_min_Q_KL", recorded["min_true_Q_KL"], min(info["qkl"][j] for j in selected))
                check.near("selection_max_Q_KL", recorded["max_true_Q_KL"], max(info["qkl"][j] for j in selected))
                assert recorded["conditional_on_qualified_reference"] is True
                check.near("selection_numerical_allowance", recorded["numerical_allowance"], 1e-7, tol=0.)
                check.near("selection_min_Q_KL_minus_allowance",
                           recorded["min_true_Q_KL_minus_qualification_allowance"],
                           max(0., min(info["qkl"][j] for j in selected)-1e-7))
                for stream in range(16):
                    curve = pop[stream, :, ki, mi]
                    index = min(range(31), key=lambda j: float(curve[j]))
                    objective = method+str(k)
                    row = popfits[(profile, str(stream), objective)]
                    assert int(row["selected_index"]) == index and float(row["t"]) == GRID[index]
                    check.near("single_stream_objective", row["objective_value"], curve[index])
                    check.near("single_stream_Q_recovery", row["true_Q_KL"], info["qkl"][index])
                    check.near("single_stream_deployed", row["deployed_KL"], risk_at(info, objective, index))
                    matched_popfits += 1
        risks = np.zeros((16, 2, 8))
        for stream in range(16):
            for ni, N in enumerate(NS):
                empirical = counts[stream, ni]/N
                objective_curves = {
                    "J": info["joint"] @ empirical, "P": info["pair"] @ empirical,
                    **{method+str(k): particle[stream, ni, :, ki, mi]
                       for mi, method in enumerate(("G", "S")) for ki, k in enumerate(KS)}}
                for oi, objective in enumerate(OBJECTIVES):
                    curve = objective_curves[objective]
                    index = min(range(31), key=lambda j: float(curve[j]))
                    row = fits[(profile, str(stream), str(N), objective)]
                    assert int(row["selected_index"]) == index and float(row["t"]) == GRID[index]
                    check.near("finite_argmin_objective", row["train_objective"], curve[index])
                    check.near("finite_Q_recovery", row["true_Q_KL"], info["qkl"][index])
                    deployed = risk_at(info, objective, index)
                    check.near("finite_deployed_KL", row["deployed_KL"], deployed)
                    check.near("finite_count_KL", row["total_count_KL"], info["countkl"][index])
                    assert float(row["grid_minimum_gap"]) == 0.
                    risks[stream, ni, oi] = deployed
                    matched_fits += 1
        comparisons = [("G10", "J"), ("G100", "J"), ("G1000", "J"), ("S10", "G10"), ("P", "J")]
        for ni, N in enumerate(NS):
            for left, right in comparisons:
                values = risks[:, ni, OBJECTIVES.index(left)]-risks[:, ni, OBJECTIVES.index(right)]
                expected = interval(values, 20, .001)
                key = (profile, str(N), left+"-"+right)
                check_stat(check, risk_intervals[key], expected, "finite_paired_CI")
                finite_pass += int(expected["precision_pass"])
                if not expected["precision_pass"]:
                    finite_failures.append(dict(profile=profile, N=N, comparison=left+"-"+right, SE=expected["SE"]))
                if N == 65536 and left in ("G10", "P"):
                    passed = expected["precision_pass"] and (
                        expected["lower"] > .001 if left == "G10" else
                        expected["lower"] >= -.001 and expected["upper"] <= .001)
                    primaries.append(dict(kind="finite_data_excess" if left == "G10" else "pair_equivalence",
                                          profile=profile, passed=bool(passed)))
        # Matrix expression independent of the runner's aggregated block formula.
        corr = np.zeros((16, 16))
        sa, sb = geo["sa"], geo["sb"]
        values = [(p @ (sa*sa)-8)/56, (p @ (sb*sb)-8)/56, p @ (sa*sb)/64]
        for i in range(16):
            for j in range(16):
                corr[i, j] = 1. if i == j else values[0 if i < 8 and j < 8 else 1 if i >= 8 and j >= 8 else 2]
        scales = np.repeat(PROFILES[profile], 8)
        R = np.full((16, 16), -.9/15)
        np.fill_diagonal(R, 1.)
        S = scales[:, None]*R*scales[None, :]
        trace = float(np.trace(S))
        B = math.fsum(float(S[i, j]*corr[j, i]) for i in range(16) for j in range(16))-trace
        check.near("raw_local_trace", raw[profile]["trace_S"], trace)
        check.near("raw_local_B", raw[profile]["B"], B)
        check.near("raw_local_Kcrit", raw[profile]["critical_K"], 1+trace/B)
        assert raw[profile]["raw_fits"] == 0
        assert raw[profile]["parameter"] == "right derivative in v=t^2 at zero"
        for k in KS:
            derivative = float(np.trace(S @ (np.eye(16)-(1-1/k)*corr)))/8
            check.near("raw_local_derivative", raw[profile]["derivatives"][str(k)], derivative)
    assert matched_fits == 512 and matched_popfits == 208
    decision = read(OUT / "DECISION.json")
    assert decision["population_precision_pass"] == population_pass
    assert decision["finite_precision_pass"] == finite_pass
    assert len(decision["primary"]) == len(primaries) == 6
    recorded_primary = {(r["kind"], r["profile"]): r for r in decision["primary"]}
    for item in primaries:
        assert recorded_primary[(item["kind"], item["profile"])]["pass_primary"] == item["passed"]
    for key, expected in (("finite_grid_fits", 512), ("bank_only_grid_fits", 192),
                          ("aggregate_population_fits", 16), ("unique_prior_particles", 2048000),
                          ("candidate_particle_evaluations", 63488000),
                          ("count_observations", 2097152), ("new_neural_models", 0)):
        assert decision[key] == expected
    assert decision["finite_grid_not_continuous_global_optimum"] and decision["separate_confidence_families"]
    assert len(list((OUT / "reference").glob("*_t??.json"))) == 62
    assert len(list((OUT / "streams").glob("*.json"))) == 32
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), passed=True,
                auditor_source_sha256=sha(__file__),
                input_seals={name: sha(OUT / name) for name in
                             ("FREEZE.json", "TRAINING_INPUTS_FREEZE.json", "RESULTS_FREEZE.json")},
                independent_of_core_and_runner_statistical_functions=True,
                reference_count=62, public_stream_count=32, finite_argmins=512,
                bank_only_argmins=192, aggregate_argmins=16, population_t_bands=360,
                population_precision_pass=population_pass, population_precision_failures=population_failures,
                selection_sets=12, finite_paired_intervals=20,
                finite_precision_pass=finite_pass, finite_precision_failures=finite_failures,
                separate_confidence_families={"population": 360, "finite_risk": 20},
                primary_checks=primaries, raw_local_matrix_checks=2, raw_fits=0,
                pair_count_patterns=81, explicit_pairs_per_pattern=120,
                public_all_objective_selection_and_statistical_replay=True,
                workspace=workspace, local_arrays_verified=(32 if workspace else 0),
                local_array_receipts=local_receipts, old_science=old,
                hash_checks=check.hash_count, numerical_maxima=check.maxima,
                arithmetic_tolerance=TOL,
                limitations=[
                    "Public audit verifies saved reference evidence, objective surfaces, argmins, risks and statistics; it does not regenerate particle likelihoods.",
                    "Workspace additionally verifies local array hashes and links public surfaces to saved mean pattern losses and per-bank population scores.",
                    "No reference integrals or scientific RNG draws were rerun; numerical reference qualification remains the frozen conditional premise.",
                    "The 360 and 20 families are separate t-approximation families, not one joint finite-sample 95 percent certificate.",
                ],
                new_scientific_samples=0, new_reference_integrations=0, new_fits=0)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", "--write", dest="run", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--workspace", action="store_true")
    args = parser.parse_args()
    if args.run:
        assert not (OUT / "MANIFEST.json").exists(), "already sealed; use --check"
    result = audit(workspace=args.workspace)
    path = OUT / "AUDIT.json"
    if args.run:
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
                        encoding="utf-8", newline="\n")
    else:
        saved = read(path)
        assert saved["passed"] and saved["auditor_source_sha256"] == sha(__file__)
        assert saved["input_seals"] == result["input_seals"]
        for key in ("reference_count", "public_stream_count", "finite_argmins", "bank_only_argmins",
                    "aggregate_argmins", "population_t_bands", "population_precision_pass",
                    "selection_sets", "finite_paired_intervals", "finite_precision_pass",
                    "primary_checks", "local_array_receipts"):
            assert saved[key] == result[key], key
    print(json.dumps(dict(audit_pass=True, mode="run" if args.run else "check",
                         workspace=args.workspace, finite_argmins=result["finite_argmins"],
                         population_bands=result["population_t_bands"],
                         population_precision_pass=result["population_precision_pass"],
                         finite_intervals=result["finite_paired_intervals"],
                         finite_precision_pass=result["finite_precision_pass"],
                         new_scientific_samples=0), ensure_ascii=False))


if __name__ == "__main__":
    main()
