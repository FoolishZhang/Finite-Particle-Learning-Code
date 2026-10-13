"""Independent R audit from sealed arrays; no producer-function imports.

Full saved-array selection/statistic reconstruction plus first-32-row score
replay per dimension/stream and phase. No new scientific fits or observations.
"""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "2"
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
from datetime import datetime, timezone
import numpy as np
from scipy.special import betaincinv, log_expit
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
R = ROOT/"artifacts/objective_strengthening/R"
AUDIT = R/"INDEPENDENT_AUDIT.json"
REPLAY_ROWS = 32
ATOL = 1e-10


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_hashes(mapping):
    for name, value in mapping.items():
        assert sha(ROOT/name) == value, name


def within(actual, expected, name, errors, tolerance=ATOL):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
    assert np.isfinite(actual).all() and np.isfinite(expected).all(), name
    error = float(np.max(np.abs(actual-expected))) if actual.size else 0.0
    errors[name] = max(errors.get(name, 0.0), error)
    assert error <= tolerance, (name, error)
    return error


def infer_selection(params, losses, tie, near):
    means = np.asarray([math.fsum(losses[:, j].tolist())/len(losses)
                        for j in range(losses.shape[1])])
    best = float(means.min())
    selected = int(np.flatnonzero(means <= best+tie)[0])
    near_indices = np.flatnonzero(means <= best+near).tolist()
    return {
        "scales": list(params[selected]), "index": selected,
        "objective": float(means[selected]), "near_indices": near_indices,
        "curve": means.tolist(),
    }


def validate_choice(calculated, stored, name, errors):
    for key in ("scales", "index", "near_indices"):
        assert calculated[key] == stored[key], (name, key)
    within(calculated["objective"], stored["objective"], "choice_objective", errors)
    within(calculated["curve"], stored["curve"], "full_curve", errors)


def local_grid(center, cfg):
    # Use integer eighths, not producer np.arange/round floating construction.
    axes = []
    for x in center:
        midpoint = int(x*8)
        assert x*8 == midpoint
        low = max(int(cfg["a"]*8), midpoint-4)
        high = min(int(cfg["b"]*8), midpoint+4)
        axes.append([v/8 for v in range(low, high+1)])
    assert cfg["refine_radius"] == .5 and cfg["refine_step"] == .125
    return list(itertools.product(*axes))


def matrix_root(d, c):
    rho = c/(d-1)
    covariance = (1+rho)*np.eye(d)-rho*np.ones((d, d))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    assert eigenvalues.min() > 0
    square_root = (eigenvectors*np.sqrt(eigenvalues)) @ eigenvectors.T
    error = float(np.max(abs(square_root @ square_root.T-covariance)))
    assert error <= 1e-11
    return square_root, error


def replay_label_prefix(d, seed, cfg, square_root):
    gen = np.random.Generator(np.random.PCG64(seed))
    # Generate the WHOLE first label chunk before its uniforms.
    normal = gen.standard_normal((cfg["label_chunk"], d))
    eta = (normal @ square_root.T)*np.repeat(np.asarray(cfg["truth"]), d//2)
    uniforms = gen.random((cfg["label_chunk"], d))
    probability = 1/(1+np.exp(-eta))
    y = (2*(uniforms < probability).astype(np.int8)-1)[:REPLAY_ROWS]
    return y


def replay_prior_prefix(d, seed, cfg, square_root):
    Ks = cfg["K_by_d"][str(d)]
    gen = np.random.Generator(np.random.PCG64(seed))
    normal = gen.standard_normal((cfg["prior_chunk"], max(Ks), d))
    return (normal @ square_root.T)[:REPLAY_ROWS]


def direct_scores(y, prior, params, Ks):
    answers = np.empty((len(Ks), len(y), len(params)))
    d = y.shape[1]
    for pi, pair in enumerate(params):
        scale = np.repeat(np.asarray(pair, dtype=float), d//2)
        loglikelihood = log_expit(prior*scale*y[:, None, :]).sum(axis=2)
        for ki, K in enumerate(Ks):
            values = loglikelihood[:, :K]
            maximum = values.max(axis=1)
            answers[ki, :, pi] = -maximum-np.log(np.exp(values-maximum[:, None]).mean(axis=1))
    return answers


def cp_interval(successes, n, cfg):
    tail = cfg["event_alpha"]/(2*cfg["event_family"])
    low = 0.0 if successes == 0 else float(betaincinv(successes, n-successes+1, tail))
    high = 1.0 if successes == n else float(betaincinv(successes+1, n-successes, 1-tail))
    return [low, high]


def bernoulli_kl(p, q):
    value = 0.0
    if p > 0:
        value += p*(math.log(p)-math.log(q))
    if p < 1:
        value += (1-p)*(math.log1p(-p)-math.log1p(-q))
    return value


def event_rectangle(p, q):
    dlow, dhigh = p[0]-q[1], p[1]-q[0]
    if dlow <= 0 <= dhigh:
        brier_low = kl_low = 0.0
    elif dlow > 0:
        brier_low = dlow*dlow
        kl_low = bernoulli_kl(p[0], q[1])
    else:
        brier_low = dhigh*dhigh
        kl_low = bernoulli_kl(p[1], q[0])
    return {
        "difference": [dlow, dhigh],
        "brier": [brier_low, max(dlow*dlow, dhigh*dhigh)],
        "logloss": [kl_low, max(bernoulli_kl(x, y) for x in p for y in q)],
    }


def run_audit():
    started = time.perf_counter()
    freeze, seal, cfg = read(R/"FREEZE.json"), read(R/"RESULTS_FREEZE.json"), read(R/"CONFIG.json")
    check_hashes(freeze["files"])
    check_hashes(seal["files"])
    check_hashes(seal["local_arrays"])
    assert seal["source_freeze_sha256"] == sha(R/"FREEZE.json")
    assert read(R/"STATE.json")["status"] == "complete"
    selection_seal = read(R/"SELECTION_SEAL.json")
    check_hashes(selection_seal["files"])
    assert selection_seal["before_confirmation"]
    event_design = read(R/"EVENT_DESIGN_SEAL.json")
    assert event_design["before_event_RNG"]
    assert event_design["selection_seal_sha256"] == sha(R/"SELECTION_SEAL.json")
    assert cfg["dimensions"] == [16, 64, 256] and cfg["streams"] == 8
    assert cfg["K_by_d"] == {"16": [10], "64": [10, 100], "256": [10]}
    assert cfg["label_chunk"] == 256 and cfg["prior_chunk"] == 128
    errors, choices, confirmations, replay = {}, {}, {}, []
    label_equal = 0
    sampled_scalar_losses = 0
    coarse_checks = fine_checks = 0
    covariance_errors = {}
    for di, d in enumerate(cfg["dimensions"]):
        square_root, error = matrix_root(d, cfg["c"])
        covariance_errors[str(d)] = error
        for stream in range(8):
            seed = cfg["seed"]+100000*di+1000*stream
            sel = read(R/"selection"/f"d{d}_s{stream}.json")
            conf = read(R/"confirmation"/f"d{d}_s{stream}.json")
            Ks = cfg["K_by_d"][str(d)]
            assert sel["d"] == conf["d"] == d and sel["stream"] == conf["stream"] == stream
            assert sel["Ks"] == conf["Ks"] == Ks
            assert sel["receipt"]["prior_seed"] == seed+1
            assert sel["refinement_replay"]["prior_seed"] == seed+1
            assert conf["receipt"]["prior_seed"] == seed+3
            assert sel["refinement_uses_same_random_inputs"]
            coarse_params = list(itertools.product(cfg["coarse"], repeat=2))
            assert sel["coarse_parameters"] == [list(x) for x in coarse_params]
            with np.load(ROOT/sel["array"], allow_pickle=False) as arr:
                labels = arr["Y"].copy()
                coarse_loss = arr["coarse_losses"].copy()
                fine_loss = arr["fine_losses"].copy()
            assert labels.shape == (cfg["n_select"], d)
            assert set(np.unique(labels)).issubset({-1, 1})
            assert coarse_loss.shape == (len(Ks), cfg["n_select"], 49)
            assert np.isfinite(coarse_loss).all() and np.isfinite(fine_loss).all()
            local_sets = []
            for ki, K in enumerate(Ks):
                coarse = infer_selection(coarse_params, coarse_loss[ki], cfg["tie"], cfg["near"])
                validate_choice(coarse, sel["coarse"][ki], f"{d}/{stream}/{K}/coarse", errors)
                local_sets.append(local_grid(coarse["scales"], cfg))
                coarse_checks += 1
            union = sorted({x for subset in local_sets for x in subset})
            assert sel["fine_union"] == [list(x) for x in union]
            assert fine_loss.shape == (len(Ks), cfg["n_select"], len(union))
            for ki, K in enumerate(Ks):
                ids = [union.index(x) for x in local_sets[ki]]
                fine = infer_selection(local_sets[ki], fine_loss[ki][:, ids], cfg["tie"], cfg["near"])
                validate_choice(fine, sel["fine"][ki], f"{d}/{stream}/{K}/fine", errors)
                choices[d, stream, K] = fine
                fine_checks += 1
            shared = sorted(set(union) & set(coarse_params))
            for point in shared:
                within(fine_loss[:, :, union.index(point)], coarse_loss[:, :, coarse_params.index(point)],
                       "all_rows_shared_coarse_fine_replay", errors)
            y = replay_label_prefix(d, seed, cfg, square_root)
            assert np.array_equal(y, labels[:REPLAY_ROWS]), ("label replay", d, stream)
            label_equal += REPLAY_ROWS
            prior = replay_prior_prefix(d, seed+1, cfg, square_root)
            anchors = [(6., 2.), (1., 1.), (3., 1.)]
            target = direct_scores(y, prior, anchors, Ks)
            expected = coarse_loss[:, :REPLAY_ROWS, [coarse_params.index(p) for p in anchors]]
            within(target, expected, "sample_selection_anchor_direct_score", errors)
            sampled_scalar_losses += target.size
            fine_points = {tuple(f["scales"]) for f in sel["fine"]}
            for point in list(fine_points):
                u, v = point
                fine_points.update(p for p in ((u+.125, v), (u, v+.125)) if p in union)
            fine_points = sorted(fine_points)
            target = direct_scores(y, prior, fine_points, Ks)
            within(target, fine_loss[:, :REPLAY_ROWS, [union.index(p) for p in fine_points]],
                   "sample_selection_fine_direct_score", errors)
            sampled_scalar_losses += target.size
            planned_confirmation = set(anchors)
            for co, fi in zip(sel["coarse"], sel["fine"]):
                planned_confirmation.add(tuple(co["scales"]))
                u, v = fi["scales"]
                planned_confirmation.add((u, v))
                for du, dv in ((-.125, 0), (.125, 0), (0, -.125), (0, .125)):
                    planned_confirmation.add((min(8, max(1, u+du)), min(8, max(1, v+dv))))
            conf_params = sorted(planned_confirmation)
            assert conf["parameters"] == [list(p) for p in conf_params]
            with np.load(ROOT/conf["array"], allow_pickle=False) as arr:
                conf_loss = arr["losses"].copy()
            assert conf_loss.shape == (len(Ks), cfg["n_confirm"], len(conf_params))
            assert np.isfinite(conf_loss).all()
            means = np.asarray([[math.fsum(conf_loss[ki, :, i].tolist())/cfg["n_confirm"]
                                 for i in range(len(conf_params))] for ki in range(len(Ks))])
            within(means, conf["means"], "all_confirmation_means", errors)
            for ki, K in enumerate(Ks):
                truth_col = conf_params.index((6., 2.))
                selected_col = conf_params.index(tuple(choices[d, stream, K]["scales"]))
                confirmations[d, stream, K] = math.fsum(
                    (conf_loss[ki, :, truth_col]-conf_loss[ki, :, selected_col]).tolist())/cfg["n_confirm"]
            yconf = replay_label_prefix(d, seed+2, cfg, square_root)
            priorconf = replay_prior_prefix(d, seed+3, cfg, square_root)
            points = sorted(set(anchors) | set(fine_points))
            assert all(p in conf_params for p in points)
            target = direct_scores(yconf, priorconf, points, Ks)
            within(target, conf_loss[:, :REPLAY_ROWS, [conf_params.index(p) for p in points]],
                   "sample_confirmation_direct_score", errors)
            sampled_scalar_losses += target.size
            replay.append({
                "d": d, "stream": stream, "Ks": Ks, "rows_per_phase": REPLAY_ROWS,
                "selection_label_rows_exactly_equal": REPLAY_ROWS,
                "label_generation_rows_per_phase": cfg["label_chunk"],
                "prior_generation_rows_per_phase": cfg["prior_chunk"],
                "selection_anchors": anchors, "selection_fine_points": fine_points,
                "confirmation_points": points, "K_prefix_shared_maxK": max(Ks),
                "matrix_square_root_eigendecomposition": True})
    assert coarse_checks == fine_checks == 32
    events = read(R/"EVENTS.json")
    expected_states = {(d, 6., 2.) for d in cfg["dimensions"]} | {(d, 1., 1.) for d in cfg["dimensions"]}
    expected_states |= {(d, *choice["scales"]) for (d, s, K), choice in choices.items()}
    assert event_design["states"] == [list(x) for x in sorted(expected_states)]
    assert len(events) == len(expected_states) <= cfg["event_family"]
    lookup = {}
    total_events = 0
    for i, event in enumerate(events):
        state = tuple(event["state"])
        assert state == sorted(expected_states)[i]
        assert event["seed"] == cfg["seed"]+9000000+i
        with np.load(ROOT/event["array"], allow_pickle=False) as arr:
            counts = arr["counts"].astype(np.int64)
        d = int(state[0])
        assert counts.shape == (cfg["event_N"],) and np.all((counts >= 0) & (counts <= d))
        success = int(np.count_nonzero((2*counts-d)**2 <= d))
        assert event["N"] == cfg["event_N"] and event["event_count"] == success
        probability = success/len(counts)
        ci = cp_interval(success, len(counts), cfg)
        within(probability, event["probability"], "event_probability", errors)
        within(ci, event["interval"], "event_CP_interval", errors, 1e-12)
        lookup[state] = {"probability": probability, "interval": ci}
        total_events += len(counts)
    results = read(R/"RESULTS.json")
    assert len(results) == 4
    tcritical = float(student_t.ppf(1-cfg["score_alpha"]/(2*cfg["score_family"]), 7))
    cases = []
    for result in results:
        d, K = result["d"], result["K"]
        models = [choices[d, s, K]["scales"] for s in range(8)]
        assert result["models"] == models
        assert result["floor_choices"] == sum(pair == [1., 1.] for pair in models)
        x = np.asarray([confirmations[d, s, K] for s in range(8)])
        mean = math.fsum(x.tolist())/8
        se = math.sqrt(math.fsum(((x-mean)**2).tolist())/7/8)
        interval = [mean-tcritical*se, mean+tcritical*se]
        recorded = result["confirmation_truth_minus_selected"]
        within(x, recorded["values"], "confirmation_contrast_values", errors)
        within([mean, se]+interval, [recorded["mean"], recorded["SE"]]+recorded["interval"],
               "four_t_intervals", errors)
        qualified = se/d <= cfg["SE_per_dim"]
        score_pass = qualified and interval[0]/d > cfg["improvement_per_dim"]
        assert recorded["precision_pass"] == qualified and result["primary_score_pass"] == score_pass
        risk_values = []
        teacher = lookup[d, 6., 2.]
        for pair, recorded_risk in zip(models, result["realized_model_event_risks"]):
            predicted = lookup[(d, *pair)]
            bound = event_rectangle(teacher["interval"], predicted["interval"])
            assert recorded_risk["scales"] == pair
            within(predicted["probability"], recorded_risk["probability"], "selected_event_probability", errors)
            plugin = (teacher["probability"]-predicted["probability"])**2
            within(plugin, recorded_risk["brier_plugin"], "event_brier_plugin", errors)
            for key in ("difference", "brier", "logloss"):
                within(bound[key], recorded_risk[key], "event_rectangle_"+key, errors, 1e-12)
            risk_values.append((plugin, bound["brier"]))
        average_plugin = math.fsum(v[0] for v in risk_values)/8
        average_interval = [math.fsum(v[1][j] for v in risk_values)/8 for j in (0, 1)]
        within(average_plugin, result["mean_event_brier_plugin"], "mean_event_brier_plugin", errors)
        within(average_interval, result["mean_event_brier_interval"], "mean_event_brier_interval", errors)
        event_pass = average_interval[0] > cfg["brier_gate"]
        assert result["primary_event_pass"] == event_pass
        cases.append({"d": d, "K": K, "score_precision_pass": qualified,
                      "primary_score_pass": score_pass, "primary_event_pass": event_pass,
                      "score_mean": mean, "score_interval": interval,
                      "mean_event_brier": average_plugin,
                      "mean_event_brier_interval": average_interval})
    count = read(R/"COUNTS.json")
    selection_particles = sum(8*cfg["n_select"]*max(cfg["K_by_d"][str(d)]) for d in cfg["dimensions"])
    confirmation_particles = sum(8*cfg["n_confirm"]*max(cfg["K_by_d"][str(d)]) for d in cfg["dimensions"])
    candidate_evaluations = 0
    for d in cfg["dimensions"]:
        for s in range(8):
            sel = read(R/"selection"/f"d{d}_s{s}.json")
            conf = read(R/"confirmation"/f"d{d}_s{s}.json")
            maxk = max(cfg["K_by_d"][str(d)])
            candidate_evaluations += cfg["n_select"]*maxk*(49+len(sel["fine_union"]))
            candidate_evaluations += cfg["n_confirm"]*maxk*len(conf["parameters"])
    computed = {
        "new_teacher_vectors": 24*(cfg["n_select"]+cfg["n_confirm"]),
        "unique_prior_particles": selection_particles+confirmation_particles,
        "selection_prior_particles": selection_particles,
        "confirmation_prior_particles": confirmation_particles,
        "fitted_procedures": 32, "coarse_intermediate_selections": 32,
        "grid_search_stages": 64, "event_models": len(events),
        "event_observations": total_events, "new_neural_models": 0,
        "primary_score_pass": sum(c["primary_score_pass"] for c in cases),
        "primary_event_pass": sum(c["primary_event_pass"] for c in cases),
        "particle_candidate_evaluations": candidate_evaluations,
        "covariance_structure_known": True, "continuous_global_optimality_certified": False}
    assert count == computed, (count, computed)
    return {
        "passed": True, "utc": datetime.now(timezone.utc).isoformat(),
        "audit_source_sha256": sha(__file__),
        "source_freeze_sha256": sha(R/"FREEZE.json"),
        "results_freeze_sha256": sha(R/"RESULTS_FREEZE.json"),
        "producer_functions_imported": False,
        "selection_NPZ": 24, "confirmation_NPZ": 24, "event_NPZ": len(events),
        "complete_saved_array_coarse_argmins_checked": coarse_checks,
        "complete_saved_array_fine_argmins_checked": fine_checks,
        "complete_saved_array_near_sets_checked": coarse_checks+fine_checks,
        "fine_endpoints_checked": 32, "confirmation_t_intervals_checked": 4,
        "event_CP_probabilities_checked": len(events), "event_primary_decisions_checked": 4,
        "event_family_retained_at_38_not_actual_state_count": True,
        "all_source_and_result_hashes_match": True,
        "maximum_errors": errors, "matrix_covariance_errors": covariance_errors,
        "sampled_replay": {
            "rows_per_phase_per_underlying_stream": REPLAY_ROWS,
            "underlying_dimension_streams": 24, "phases": ["selection", "confirmation"],
            "stored_selection_label_rows_exactly_matched": label_equal,
            "scalar_score_comparisons": sampled_scalar_losses,
            "records": replay,
            "all_original_raw_scoring_replayed": False,
            "confirmation_Y_not_stored_score_matches_are_indirect_label_checks": True,
            "generated_full_first_label_and_prior_chunks_to_preserve_RNG_contract": True},
        "cases": cases, "reconstructed_ledger": computed,
        "new_scientific_samples": 0, "new_scientific_fits": 0,
        "audit_RNG_is_replay_of_old_stream_prefixes_not_new_independent_data": True,
        "elapsed_seconds": time.perf_counter()-started,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("write", "check"))
    args = parser.parse_args()
    if args.mode == "check":
        item = read(AUDIT)
        assert item["passed"] and sha(__file__) == item["audit_source_sha256"]
        assert sha(R/"FREEZE.json") == item["source_freeze_sha256"]
        assert sha(R/"RESULTS_FREEZE.json") == item["results_freeze_sha256"]
        check_hashes(read(R/"FREEZE.json")["files"])
        seal = read(R/"RESULTS_FREEZE.json")
        check_hashes(seal["files"])
        check_hashes(seal["local_arrays"])
        print(json.dumps({"passed": True, "mode": "hash_readback",
                          "arithmetic_or_sampled_replay_not_repeated": True}))
    else:
        assert not AUDIT.exists(), "do not replace an independent receipt"
        result = run_audit()
        with AUDIT.open("x", encoding="utf-8", newline="\n") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
        print(json.dumps({k:v for k,v in result.items() if k != "sampled_replay"},
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
