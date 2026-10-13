"""B: fixed-K group objectives with nested bank counts and independent confirmation.

No scientific execution at import. This file may be frozen only after root
review of PLAN_B; freeze additionally requires its exact approved SHA256.
"""
from __future__ import annotations
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "2"
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback
import numpy as np
import scipy
from scipy.stats import t as student_t
import scale_learning_core as core

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"artifacts/objective_strengthening/B"
DOC = ROOT/"docs/08_research_研究工作/总体目标塌缩与同模型修复_20261001"
OLD_B = ROOT/"artifacts/structure_learning_v2/B"
OLD_A = ROOT/"artifacts/structure_learning_v2/A"
BASE = "27fad91bc2c15e2ca4aeabbc8c6710f171090fcc"
HANDOFF_SHA = "265476d955d1f5a13a04f45dcaa2ac63575716f9739bf313eb8bc908755a18dd"
CFG = {
    "d": 16, "c": 0.9, "teacher": [6.0, 2.0], "K": 10,
    "grid": [j/2 for j in range(19)], "streams": 16,
    "training_banks": [64, 256, 1024], "confirmation_banks": 4096,
    "batch_banks": 16, "train_seed_base": 20601001,
    "confirmation_seed_base": 30601001, "old_label_N": 65536,
    "old_label_r": 0, "anchors": [[6.0, 2.0], [1.0, 1.0], [3.0, 1.0]],
    "neighbor_index_offsets": [-1, 0, 1], "near_tolerance": 0.005,
    "tie_tolerance": 1e-12, "mass_gate": 1e-12,
    "identity_gate": 1e-10, "reference_sensitivity_gate": 1e-7,
    "contrast_family": 35, "contrast_alpha": 0.025,
    "bands_family": 1080, "bands_alpha": 0.025,
    "confirmation_F_SE_gate": 0.005, "risk_SE_gate": 0.001,
    "band_SE_gate": 0.01, "material": 0.001,
    "new_teacher_observations": 0, "new_reference_integrations": 0,
    "new_neural_networks": 0, "continuous_optimality_claim": False,
    "floor_anchor_is_not_grid_lower_bound": True,
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def serial(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return relative(x)
    raise TypeError(type(x).__name__)


def write(path, obj, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if replace else "x", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, default=serial, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def csvwrite(path, rows):
    if not rows:
        raise ValueError("empty table")
    with Path(path).open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def grid():
    return [(u, v) for u in CFG["grid"] for v in CFG["grid"]]


def index(pair):
    return grid().index(tuple(float(v) for v in pair))


def neighborhood(center):
    i, j = divmod(int(center), 19)
    return sorted({19*(i+di)+j+dj for di in CFG["neighbor_index_offsets"]
                   for dj in CFG["neighbor_index_offsets"]
                   if 0 <= i+di < 19 and 0 <= j+dj < 19})


def fixed_confirmation_indices():
    return sorted({j for pair in CFG["anchors"] for j in neighborhood(index(pair))})


def check_mapping(mapping):
    for name, expected in mapping.items():
        require(sha(ROOT/name) == expected, "input hash mismatch: "+name)


def input_paths():
    paths = [
        ROOT/"script/scale_learning_core.py",
        ROOT/"script/structure_learning_b.py",
        OLD_B/"FREEZE.json", OLD_B/"REFERENCE_FREEZE.json",
        OLD_B/"RESULTS_FREEZE.json", OLD_B/"GRID.csv", OLD_B/"FINITE_FITS.csv",
        OLD_A/"DATA_FREEZE.json", OLD_A/"RESULTS_FREEZE.json", OLD_A/"TEACHER.json",
    ]
    paths += [OLD_B/"reference"/f"g{i:03}.json" for i in range(361)]
    paths += [OLD_A/"streams"/f"s{s:02}.json" for s in range(16)]
    return paths


def validate_old_receipts():
    # No local old pattern arrays or old Monte Carlo curves are used.
    seals = [read(OLD_B/"REFERENCE_FREEZE.json")["files"],
             read(OLD_B/"RESULTS_FREEZE.json")["files"],
             read(OLD_A/"DATA_FREEZE.json")["files"]]
    for path in input_paths():
        name = relative(path)
        claims = [seal[name] for seal in seals if name in seal]
        for expected in claims:
            require(sha(path) == expected, "old science receipt mismatch: "+name)
    old = read(OLD_B/"FREEZE.json")["files"]
    for path in (ROOT/"script/scale_learning_core.py", ROOT/"script/structure_learning_b.py"):
        require(sha(path) == old[relative(path)], "old kernel/source changed")


def load_inputs():
    teacher = read(OLD_A/"TEACHER.json")
    pvar = np.asarray(teacher["variants"], dtype=np.float64)
    require(pvar.shape == (3, 81) and np.all(pvar > 0), "teacher reference shape")
    p = pvar[2]
    lg = core.geometry(16)[3]
    indicator = core.geometry(16)[6]
    hvar = -np.sum(pvar*(np.log(pvar)-lg), axis=1)
    require(abs(hvar[2]-teacher["H_original"]) <= CFG["identity_gate"], "teacher entropy")
    references = [read(OLD_B/"reference"/f"g{i:03}.json") for i in range(361)]
    for i, ref in enumerate(references):
        require(ref["qualified"] and ref["index"] == i, "unqualified reference")
        require(list(ref["scales"]) == list(grid()[i]), "reference candidate ordering")
    qvar = np.asarray([ref["variants"] for ref in references], dtype=np.float64)
    require(qvar.shape == (361, 3, 81) and np.all(qvar > 0), "candidate references")
    require(float(np.max(abs(qvar.sum(-1)-1))) <= CFG["mass_gate"], "reference normalization")
    exact_losses = -np.log(qvar[:, 2])+lg
    risk = exact_losses @ p-hvar[2]
    cross = []
    for a in range(3):
        for b in range(3):
            cross.append((-np.log(qvar[:, b])+lg) @ pvar[a]-hvar[a])
    cross = np.asarray(cross)
    sensitivity = float(np.ptp(cross, axis=0).max())
    require(sensitivity <= CFG["reference_sensitivity_gate"], "reference cross-risk sensitivity")
    anchor = float(abs(qvar[index(CFG["teacher"]), 2]-p).sum())
    require(anchor <= 1e-8, "teacher reference anchor")
    labels = []
    label_sources = []
    for s in range(16):
        path = OLD_A/"streams"/f"s{s:02}.json"
        item = read(path)
        ni = item["N"].index(CFG["old_label_N"])
        ri = item["r_values"].index(CFG["old_label_r"])
        counts = np.asarray(item["training_counts_by_r"], dtype=np.int64)[ri, ni]
        require(counts.shape == (81,) and np.all(counts >= 0)
                and int(counts.sum()) == CFG["old_label_N"], "old integer training counts")
        labels.append(counts/CFG["old_label_N"])
        label_sources.append({"path": relative(path), "sha256": sha(path),
                              "JSON_key": f"training_counts_by_r[{ri}][{ni}]"})
    with (OLD_B/"FINITE_FITS.csv").open(encoding="utf-8", newline="") as f:
        old_fits = list(csv.DictReader(f))
    old_j = []
    for s in range(16):
        row = next(r for r in old_fits if r["r"] == "0" and r["N"] == "65536"
                   and r["stream"] == str(s) and r["objective"] == "J")
        ii = int(row["index"])
        require(ii == int(np.argmin(exact_losses @ labels[s])), "old J endpoint replay")
        old_j.append(ii)
    return {
        "p": p, "pvar": pvar, "h": float(hvar[2]), "lg": lg,
        "indicator": indicator, "risk": risk, "exact_losses": exact_losses,
        "labels": np.asarray(labels), "label_sources": label_sources,
        "old_J_indices": old_j, "risk_cross_variant_sensitivity": sensitivity,
        "teacher_anchor_L1": anchor,
    }


def freeze(approved_plan_sha):
    require(not OUT.exists(), "B output directory already exists")
    plan = DOC/"PLAN_B.md"
    require(approved_plan_sha == sha(plan), "explicit reviewed plan SHA required")
    require(sha(ROOT/"HANDOFF.md") == HANDOFF_SHA, "HANDOFF version changed")
    validate_old_receipts()
    # Reading old qualified references is not a new integration.
    info = load_inputs()
    OUT.mkdir(parents=True)
    snapshot = OUT/"source_inputs/HANDOFF.md"
    snapshot.parent.mkdir()
    snapshot.write_bytes((ROOT/"HANDOFF.md").read_bytes())
    write(OUT/"CONFIG.json", CFG)
    write(OUT/"ENVIRONMENT.json", {
        "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(), "threads": 2, "dtype": "float64", "RNG": "PCG64"})
    write(OUT/"INPUT_RECEIPT.json", {
        "files": {relative(p): sha(p) for p in input_paths()},
        "teacher_reference_anchor_L1": info["teacher_anchor_L1"],
        "risk_cross_variant_sensitivity": info["risk_cross_variant_sensitivity"],
        "label_sources": info["label_sources"],
        "only_old_qualified_references_no_new_integrator_calls": True})
    files = [Path(__file__).resolve(), plan, snapshot,
             OUT/"CONFIG.json", OUT/"ENVIRONMENT.json", OUT/"INPUT_RECEIPT.json"]
    write(OUT/"SOURCE_SEAL.json", {
        "utc": now(), "baseline_commit": BASE, "approved_plan_sha256": approved_plan_sha,
        "before_scientific_RNG": True,
        "files": {relative(p): sha(p) for p in files}})
    write(OUT/"STATE.json", {"status": "frozen", "utc": now(), "active_jobs": []})
    print(json.dumps({"SOURCE_SEAL_sha256": sha(OUT/"SOURCE_SEAL.json")}), flush=True)


def check_source():
    check_mapping(read(OUT/"SOURCE_SEAL.json")["files"])
    check_mapping(read(OUT/"INPUT_RECEIPT.json")["files"])
    require(read(OUT/"CONFIG.json") == CFG, "configuration changed")


def smoke():
    # Fixed d4 algebra fixture, no Gaussian RNG and no scientific teacher.
    eta = np.asarray([[-2., -1., -2., -1.], [-1.5, -.5, -1.5, -.5],
                      [2., 1., 2., 1.], [1.5, .5, 1.5, .5]])
    mass = core.joint_mass(eta)
    indicator = core.geometry(4)[6]
    identity = float(abs(mass @ indicator-core.pb(eta)).max())
    q = mass.reshape(2, 2, 9).mean(axis=1)
    average_log = -np.log(q).mean(axis=0)
    pooled_log = -np.log(q.mean(axis=0))
    require(identity <= CFG["identity_gate"], "d4 group-count identity")
    require(float(np.max(average_log-pooled_log)) > 0.01, "B must remain outside log")
    require(len(fixed_confirmation_indices()) == 27, "fixed confirmation pool")
    write(OUT/"SMOKE.json", {
        "passed": True, "group_total_identity_error": identity,
        "average_log_minus_pooled_log_max": float(np.max(average_log-pooled_log)),
        "no_scientific_samples": True, "RNG_calls": 0})


def bank_panel(seed, n_banks, indices, info, label_weights, checkpoints=()):
    """Average log of each K-particle mass. Never pool B*K inside one log."""
    indices = list(indices)
    rng = np.random.Generator(np.random.PCG64(seed))
    initial = rng.bit_generator.state
    pop = np.empty((n_banks, len(indices)), dtype=np.float64)
    finite = np.empty_like(pop)
    sums = np.zeros((len(indices), 81))
    prefix = []
    norm_error = identity_error = reference_error = 0.0
    minimum_probability = 1.0
    count_updates = 0
    start = time.perf_counter()
    for begin in range(0, n_banks, CFG["batch_banks"]):
        nb = min(CFG["batch_banks"], n_banks-begin)
        z = rng.standard_normal((nb*CFG["K"], CFG["d"]))
        for column, candidate in enumerate(indices):
            eta = core.eta_from_z(z, grid()[candidate], CFG["c"])
            mass = core.joint_mass(eta)
            total = core.pb(eta)
            norm_error = max(norm_error, float(abs(mass.sum(1)-1).max()))
            identity_error = max(identity_error, float(abs(mass @ info["indicator"]-total).max()))
            q = mass.reshape(nb, CFG["K"], 81).mean(axis=1)
            require(np.isfinite(q).all() and np.all(q > 0), "nonpositive finite-K likelihood")
            minimum_probability = min(minimum_probability, float(q.min()))
            norm_error = max(norm_error, float(abs(q.sum(1)-1).max()))
            losses = -np.log(q)+info["lg"]
            require(np.isfinite(losses).all(), "nonfinite score")
            pop[begin:begin+nb, column] = losses @ info["p"]
            finite[begin:begin+nb, column] = losses @ label_weights
            sums[column] += losses.sum(axis=0)
            # Propagate the old teacher-reference variants, no new integration.
            sensitivity = losses @ info["pvar"].T
            reference_error = max(reference_error, float(np.ptp(sensitivity, axis=1).max()))
            count_updates += nb*CFG["K"]
        finished = begin+nb
        if finished in checkpoints:
            prefix.append(sums.copy()/finished)
    require(norm_error <= CFG["mass_gate"], "probability mass gate")
    require(identity_error <= CFG["identity_gate"], "group/total identity gate")
    require(reference_error <= CFG["reference_sensitivity_gate"], "bank reference sensitivity gate")
    return {
        "population": pop, "finite": finite,
        "prefix_pattern_loss": np.asarray(prefix),
        "mean_pattern_loss": sums/n_banks,
        "receipt": {
            "seed": seed, "initial_RNG": initial, "final_RNG": rng.bit_generator.state,
            "banks": n_banks, "K": CFG["K"], "candidates": len(indices),
            "unique_prior_particles": n_banks*CFG["K"],
            "candidate_particle_evaluations": count_updates,
            "Gaussian_variates_requested": n_banks*CFG["K"]*CFG["d"],
            "mass_error": norm_error, "count_identity_error": identity_error,
            "max_per_bank_reference_sensitivity": reference_error,
            "minimum_count_probability": minimum_probability,
            "all_losses_finite": True, "seconds": time.perf_counter()-start,
            "B_average_is_outside_log": True,
        },
    }


def select_curve(curve):
    minimum = float(np.min(curve))
    ties = np.flatnonzero(curve <= minimum+CFG["tie_tolerance"])
    selected = int(ties[0])
    near = np.flatnonzero(curve <= minimum+CFG["near_tolerance"])
    return {"index": selected, "scales": list(grid()[selected]),
            "objective": float(curve[selected]), "minimum": minimum,
            "tie_indices": ties.tolist(), "near_indices": near.tolist(),
            "near_tolerance": CFG["near_tolerance"],
            "tie_rule": "first lexicographic index within 1e-12 of minimum"}


def train_stream(s, info):
    panel = bank_panel(CFG["train_seed_base"]+s, max(CFG["training_banks"]),
                       range(361), info, info["labels"][s], CFG["training_banks"])
    path = OUT/"local"/f"train_s{s:02}.npz"
    np.savez_compressed(path, population=panel["population"], finite=panel["finite"],
                        prefix_pattern_loss=panel["prefix_pattern_loss"])
    selections = []
    for B in CFG["training_banks"]:
        for mode in ("population", "finite"):
            selection = select_curve(panel[mode][:B].mean(axis=0))
            selections.append({"stream": s, "B": B, "mode": mode, **selection})
    pool = set(fixed_confirmation_indices())
    for selection in selections:
        pool.update(neighborhood(selection["index"]))
    require(len(pool) <= 81, "confirmation pool bound")
    chosen = {
        "stream": s, "selections": selections,
        "confirmation_indices": sorted(pool),
        "fixed_confirmation_indices": fixed_confirmation_indices(),
        "selection_inputs": "training bank objective curves only",
        "selected_neighborhoods_are_not_new_fitting_grid": True,
        "new_reference_calls": 0,
    }
    write(OUT/"selections"/f"s{s:02}.json", chosen)
    write(OUT/"training"/f"s{s:02}.json", {
        "stream": s, **panel["receipt"], "array_path": relative(path),
        "array_sha256": sha(path), "old_label_source": info["label_sources"][s],
        "prefixes": CFG["training_banks"]})
    print("B train", s, "pool", len(pool), flush=True)


def confirm_stream(s, info):
    chosen = read(OUT/"selections"/f"s{s:02}.json")
    indices = chosen["confirmation_indices"]
    panel = bank_panel(CFG["confirmation_seed_base"]+s, CFG["confirmation_banks"],
                       indices, info, info["labels"][s])
    path = OUT/"local"/f"confirmation_s{s:02}.npz"
    np.savez_compressed(path, indices=indices, population=panel["population"],
                        finite=panel["finite"], mean_pattern_loss=panel["mean_pattern_loss"])
    write(OUT/"confirmation"/f"s{s:02}.json", {
        "stream": s, **panel["receipt"], "indices": indices,
        "population_means": panel["population"].mean(axis=0),
        "finite_means": panel["finite"].mean(axis=0),
        "array_path": relative(path), "array_sha256": sha(path),
        "selection_sha256": sha(OUT/"selections"/f"s{s:02}.json"),
        "independent_of_all_training_banks": True,
        "confirmation_is_evaluation_only_no_reselection": True})
    print("B confirm", s, len(indices), flush=True)


def interval(values, family, alpha, gate):
    values = np.asarray(values, dtype=np.float64)
    require(values.shape == (16,) and np.isfinite(values).all(), "statistical unit count")
    mean = float(values.mean())
    se = float(values.std(ddof=1)/4)
    critical = float(student_t.ppf(1-alpha/(2*family), 15))
    return {"mean": mean, "SE": se, "lower": mean-critical*se,
            "upper": mean+critical*se, "precision_pass": bool(se <= gate),
            "SE_gate": gate, "family": family, "alpha": alpha,
            "df": 15, "tcrit": critical, "t_is_approximate": True}


def summarize_region(indices, risk):
    ids = np.asarray(indices, dtype=int)
    if len(ids) == 0:
        return {"count": 0, "minimum_KL": None, "maximum_KL": None,
                "median_KL": None, "diagonal_count": 0}
    return {"count": len(ids), "minimum_KL": float(risk[ids].min()),
            "maximum_KL": float(risk[ids].max()), "median_KL": float(np.median(risk[ids])),
            "diagonal_count": sum(grid()[int(i)][0] == grid()[int(i)][1] for i in ids)}


def collect(info):
    training = [read(OUT/"training"/f"s{s:02}.json") for s in range(16)]
    selections = [read(OUT/"selections"/f"s{s:02}.json") for s in range(16)]
    confirms = [read(OUT/"confirmation"/f"s{s:02}.json") for s in range(16)]
    pop = np.empty((16, 3, 361))
    finite = np.empty_like(pop)
    for s, record in enumerate(training):
        require(sha(ROOT/record["array_path"]) == record["array_sha256"], "training array hash")
        with np.load(ROOT/record["array_path"], allow_pickle=False) as arrays:
            for bi, B in enumerate(CFG["training_banks"]):
                pop[s, bi] = arrays["population"][:B].mean(axis=0)
                finite[s, bi] = arrays["finite"][:B].mean(axis=0)
    cmaps = [dict(zip(r["indices"], r["population_means"])) for r in confirms]
    truth = index(CFG["teacher"])
    curves, endpoints, near_regions = [], [], []
    selected = {}
    for s in range(16):
        for row in selections[s]["selections"]:
            bi = CFG["training_banks"].index(row["B"])
            curve = pop[s, bi] if row["mode"] == "population" else finite[s, bi]
            require(select_curve(curve) == {k:row[k] for k in select_curve(curve)}, "selection replay")
            ii = row["index"]
            selected[s, row["B"], row["mode"]] = ii
            endpoints.append({
                "stream": s, "B": row["B"], "mode": row["mode"], "index": ii,
                "u": grid()[ii][0], "v": grid()[ii][1],
                "training_objective": row["objective"],
                "confirmation_population_F": cmaps[s][ii],
                "confirmation_F_minus_truth": cmaps[s][ii]-cmaps[s][truth],
                "true_joint_KL": float(info["risk"][ii]),
                "old_exact_J_index": info["old_J_indices"][s] if row["mode"] == "finite" else truth,
                "grid_selected_diagonal": grid()[ii][0] == grid()[ii][1],
                "continuous_optimum_claim": False})
            near_regions.append({
                "stream": s, "B": row["B"], "mode": row["mode"],
                "indices": row["near_indices"],
                "kind": "empirical_training_near_set_not_confidence_set",
                **summarize_region(row["near_indices"], info["risk"])})
        for bi, B in enumerate(CFG["training_banks"]):
            for mode, values in (("population", pop[s, bi]), ("finite", finite[s, bi])):
                for i, (u, v) in enumerate(grid()):
                    curves.append({"stream": s, "B": B, "mode": mode, "index": i,
                                   "u": u, "v": v, "objective": float(values[i]),
                                   "objective_minus_truth": float(values[i]-values[truth])})
    stats = []
    for ii in fixed_confirmation_indices():
        if ii == truth:
            continue
        values = [c[ii]-c[truth] for c in cmaps]
        stats.append({"comparison": f"fixed_{ii:03}_F_minus_truth", "kind": "fixed_confirmation_F",
                      "B": 0, "index": ii, "u": grid()[ii][0], "v": grid()[ii][1],
                      **interval(values, 35, .025, CFG["confirmation_F_SE_gate"])})
    for B in CFG["training_banks"]:
        ids = [selected[s, B, "population"] for s in range(16)]
        values = [cmaps[s][ii]-cmaps[s][truth] for s, ii in enumerate(ids)]
        stats.append({"comparison": f"population_selected_B{B}_confirm_F_minus_truth",
                      "kind": "selected_procedure_confirmation_F", "B": B, "index": -1,
                      "u": "", "v": "", **interval(values, 35, .025, CFG["confirmation_F_SE_gate"])})
        stats.append({"comparison": f"population_selected_B{B}_true_KL",
                      "kind": "selected_procedure_true_KL", "B": B, "index": -1, "u": "", "v": "",
                      **interval(info["risk"][ids], 35, .025, CFG["risk_SE_gate"])})
        values = [info["risk"][selected[s, B, "finite"]]-info["risk"][info["old_J_indices"][s]]
                  for s in range(16)]
        stats.append({"comparison": f"finite_selected_B{B}_KL_minus_old_J",
                      "kind": "old_labels_selected_procedure_risk", "B": B,
                      "index": -1, "u": "", "v": "",
                      **interval(values, 35, .025, CFG["risk_SE_gate"])})
    require(len(stats) == 35, "confirmation/risk family size")
    bands, regions = [], []
    for bi, B in enumerate(CFG["training_banks"]):
        lower = np.zeros(361)
        upper = np.zeros(361)
        precision = True
        for i, (u, v) in enumerate(grid()):
            if i == truth:
                continue
            record = interval(pop[:, bi, i]-pop[:, bi, truth], 1080, .025, CFG["band_SE_gate"])
            lower[i], upper[i] = record["lower"], record["upper"]
            precision = precision and record["precision_pass"]
            bands.append({"B": B, "index": i, "u": u, "v": v, **record})
        best_upper = float(upper.min())
        possible = np.flatnonzero(lower <= best_upper)
        near = np.flatnonzero(lower <= best_upper+CFG["near_tolerance"])
        regions.append({
            "B": B, "all_band_precision_pass": bool(precision),
            "minimum_upper": best_upper, "possible_grid_optimal_indices": possible.tolist(),
            "possible_near_optimal_indices": near.tolist(),
            "truth_excluded": bool(lower[truth] > best_upper),
            "possible_optimal_KL": summarize_region(possible, info["risk"]),
            "possible_near_optimal_KL": summarize_region(near, info["risk"]),
            "coverage_is_approximate_t_and_conditional_on_old_reference_qualification": True,
            "not_continuous_global_certificate": True})
    require(len(bands) == 1080 and len(endpoints) == 96, "ledger counts")
    confirm_rows = []
    for s, rec in enumerate(confirms):
        for ii, pv, fv in zip(rec["indices"], rec["population_means"], rec["finite_means"]):
            confirm_rows.append({
                "stream": s, "index": ii, "u": grid()[ii][0], "v": grid()[ii][1],
                "fixed_pool": ii in fixed_confirmation_indices(),
                "population_F": pv, "population_F_minus_truth": pv-cmaps[s][truth],
                "finite_F": fv, "true_joint_KL": float(info["risk"][ii]),
                "selected_neighborhood_values_are_descriptive_only": True})
    risk_rows = [{"index": i, "u": u, "v": v, "true_joint_KL": float(info["risk"][i])}
                 for i, (u, v) in enumerate(grid())]
    csvwrite(OUT/"TRAINING_CURVES.csv", curves)
    csvwrite(OUT/"ENDPOINTS.csv", endpoints)
    csvwrite(OUT/"CONTRASTS_35.csv", stats)
    csvwrite(OUT/"POPULATION_BANDS_1080.csv", bands)
    csvwrite(OUT/"CONFIRMATION_POINTS.csv", confirm_rows)
    csvwrite(OUT/"REFERENCE_RISK.csv", risk_rows)
    write(OUT/"EMPIRICAL_NEAR_REGIONS.json", near_regions)
    write(OUT/"POSSIBLE_GRID_OPTIMAL_SETS.json", regions)
    witness = next(s for s in stats if s["comparison"] == f"fixed_{index([3,1]):03}_F_minus_truth")
    high_f = next(s for s in stats if s["comparison"] == "population_selected_B1024_confirm_F_minus_truth")
    high_risk = next(s for s in stats if s["comparison"] == "population_selected_B1024_true_KL")
    decision = {
        "fixed_witness_supported": witness["precision_pass"] and witness["upper"] < -.001,
        "selected_population_highB_confirmed_better_F_than_truth": high_f["precision_pass"] and high_f["upper"] < -.001,
        "selected_population_highB_material_true_KL": high_risk["precision_pass"] and high_risk["lower"] > .001,
        "separate_primary_readouts_not_a_new_general_algorithm_success": True,
        "contrasts_qualified": sum(s["precision_pass"] for s in stats),
        "bands_qualified": sum(s["precision_pass"] for s in bands),
        "all_failures_retained_no_budget_expansion": True,
        "finite_old_labels_are_secondary_development_evidence": True,
        "K_fixed": 10, "B_does_not_change_K": True,
        "new_raw_floor_theory_not_tested_here": True,
    }
    write(OUT/"DECISION.json", decision)
    write(OUT/"COUNTS.json", {
        "new_scientific_prior_particles": sum(r["unique_prior_particles"] for r in training+confirms),
        "candidate_particle_evaluations": sum(r["candidate_particle_evaluations"] for r in training+confirms),
        "training_banks": 16*1024, "confirmation_banks": 16*4096,
        "streams": 16, "new_teacher_observations": 0, "old_label_vectors_reused": 16*65536,
        "new_reference_integrations": 0, "grid_selection_endpoints": 96,
        "reused_exact_J_fits": 16, "new_neural_networks": 0,
        "same_bank_particles_shared_across_candidates_not_counted_as_independent": True})
    write(OUT/"COST.json", {"training": training, "confirmation": confirms})
    return decision


def run():
    check_source()
    require(read(OUT/"STATE.json")["status"] == "frozen", "no silent science restart")
    (OUT/"local").mkdir()
    (OUT/".gitignore").write_text("/local/\n", encoding="utf-8", newline="\n")
    info = load_inputs()
    smoke()
    write(OUT/"START.json", {"first_scientific_RNG_utc": now(),
                             "SOURCE_SEAL_sha256": sha(OUT/"SOURCE_SEAL.json")})
    for s in range(16):
        write(OUT/"STATE.json", {"status": "training", "stream": s, "pid": os.getpid(), "utc": now()}, True)
        train_stream(s, info)
    files = list((OUT/"training").glob("*.json"))+list((OUT/"selections").glob("*.json"))
    write(OUT/"SELECTION_SEAL.json", {
        "utc": now(), "before_first_confirmation_RNG": True,
        "files": {relative(p): sha(p) for p in files},
        "training_arrays": {relative(p): sha(p) for p in (OUT/"local").glob("train_*.npz")}})
    for s in range(16):
        write(OUT/"STATE.json", {"status": "confirmation", "stream": s, "pid": os.getpid(), "utc": now()}, True)
        confirm_stream(s, info)
    decision = collect(info)
    public = [p for p in OUT.rglob("*") if p.is_file()
              and "local" not in p.relative_to(OUT).parts
              and p.name not in ("STATE.json", "RESULTS_SEAL.json")]
    write(OUT/"RESULTS_SEAL.json", {
        "utc": now(), "SOURCE_SEAL_sha256": sha(OUT/"SOURCE_SEAL.json"),
        "files": {relative(p): sha(p) for p in public},
        "local_arrays": {relative(p): sha(p) for p in (OUT/"local").glob("*.npz")}})
    write(OUT/"STATE.json", {"status": "complete", "active_jobs": [], "utc": now()}, True)
    print(json.dumps({"complete": True, **decision}), flush=True)


def check(workspace=False):
    check_source()
    require(read(OUT/"STATE.json")["status"] == "complete", "not complete")
    seal = read(OUT/"RESULTS_SEAL.json")
    require(seal["SOURCE_SEAL_sha256"] == sha(OUT/"SOURCE_SEAL.json"), "seal chain")
    check_mapping(seal["files"])
    if workspace:
        check_mapping(seal["local_arrays"])
    print(json.dumps({"passed": True, "workspace_arrays": workspace,
                      "science_not_reexecuted": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("freeze", "run", "check"))
    parser.add_argument("--approved-plan-sha")
    parser.add_argument("--workspace", action="store_true")
    args = parser.parse_args()
    try:
        if args.phase == "freeze":
            freeze(args.approved_plan_sha)
        elif args.phase == "run":
            run()
        else:
            check(args.workspace)
    except Exception as exc:
        if OUT.exists() and (OUT/"SOURCE_SEAL.json").exists() and args.phase == "run":
            write(OUT/f"FAILURE_{time.time_ns()}.json",
                  {"utc": now(), "error": repr(exc), "traceback": traceback.format_exc()})
            write(OUT/"STATE.json", {"status": "stopped_failure", "active_jobs": [], "utc": now()}, True)
        raise
