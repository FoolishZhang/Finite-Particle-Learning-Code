"""Independent bounded audit of the sealed finite-learning RUN.

No producer-module imports, RNG, fits, or quadrature.  Initial --write requires
--workspace; subsequent --check validates the saved evidence hashes only.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import brentq
from scipy.special import log_expit, logsumexp, rel_entr
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/finite_learning/RUN"
REF = ROOT / "artifacts/finite_learning/REF"
DOC = ROOT / "docs/08_research_研究工作/有限样本学习与均值自由度_20261001"
RECEIPT = OUT / "INDEPENDENT_AUDIT.json"
DIRECT_BANKS = 4
DIRECT_LABEL_CHUNK = 128
TOL = 1e-10
SCHEMA = "finite-learning-independent-audit-v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def verify(files):
    for name, expected in files.items():
        if sha(ROOT / name) != expected:
            raise AssertionError("hash mismatch: " + name)


class Audit:
    def __init__(self):
        self.checks = 0
        self.errors = {}

    def require(self, condition, name):
        self.checks += 1
        if not condition:
            raise AssertionError(name)

    def near(self, actual, expected, name, tolerance=TOL, category="public"):
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        self.require(a.shape == b.shape, name + ": shape")
        self.require(bool(np.isfinite(a).all() and np.isfinite(b).all()), name + ": finite")
        error = float(np.max(np.abs(a-b))) if a.size else 0.0
        self.errors[category] = max(self.errors.get(category, 0.0), error)
        self.require(error <= tolerance, name + ": error=" + repr(error))


def interval_values(values, family, alpha, gate):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(np.sum(values)/n)
    se = float(np.sqrt(np.sum((values-mean)**2)/(n*(n-1))))
    radius = float(student_t.isf(alpha/(2*family), n-1))*se
    return dict(values=values, mean=mean, SE=se, lower=mean-radius,
                upper=mean+radius, precision_pass=se <= gate, SE_gate=gate)


def compare_interval(audit, record, values, family, alpha, gate, name):
    expected = interval_values(values, family, alpha, gate)
    for key in ("values", "mean", "SE", "lower", "upper", "SE_gate"):
        audit.near(record[key], expected[key], name + "/" + key, category="interval")
    audit.require(record["precision_pass"] == bool(expected["precision_pass"]), name + "/precision")
    audit.require(record["t_is_approximation"] is True, name + "/approximate t")
    return expected


def cp_lower(k, n, alpha):
    """Invert the binomial upper tail, independent of producer beta.ppf."""
    if not k:
        return 0.0
    def tail(p):
        return math.fsum(math.comb(n, j)*p**j*(1-p)**(n-j) for j in range(k, n+1))
    return float(brentq(lambda p: tail(p)-alpha, 0.0, 1.0,
                        xtol=5e-15, rtol=5e-15))


def pmf_risks(reference, d, truth_index, audit):
    p = np.asarray(reference["pmf"], dtype=float)
    m = d//2
    audit.require(p.shape == (49, m+1, m+1), "reference PMF shape")
    audit.require(bool(np.all(p > 0)), "reference positivity")
    audit.near(p.sum(axis=(1, 2)), np.ones(49), "reference normalization")
    truth = p[truth_index]
    risk = np.sum(rel_entr(truth[None], p), axis=(1, 2))
    n = np.arange(m+1)
    spin = 2*(n[:, None]+n[None, :])-d
    masks = (np.abs(spin) <= math.sqrt(d), spin > math.sqrt(d), spin < -math.sqrt(d))
    probabilities = np.column_stack([p[:, mask].sum(axis=1) for mask in masks])
    teacher_events = probabilities[truth_index]
    brier = (probabilities-teacher_events)**2
    binary_log = rel_entr(teacher_events, probabilities)+rel_entr(1-teacher_events, 1-probabilities)
    multiplicity_logs = np.array([math.log(math.comb(m, j)) for j in n])
    log_pattern = np.log(p)-multiplicity_logs[:, None]-multiplicity_logs[None, :]
    audit.near(reference["log_pattern"], log_pattern, "independent logchoose conversion")
    return dict(joint_KL=risk, event_probabilities=probabilities, event_brier=brier,
                event_logloss=binary_log, teacher_events=teacher_events,
                log_pattern=log_pattern)


def select_key(row):
    return row["method"], row["N"], row["B"], row["K"]


def find_selected(record, method, N, B=None, K=None):
    return next(row for row in record["selected"]
                if select_key(row) == (method, N, B, K))


def check_choices(record, params, cfg, audit):
    seen = set()
    for row in record["selected"]:
        key = select_key(row)
        audit.require(key not in seen, "duplicate training contract")
        seen.add(key)
        curve = np.asarray(row["curve"], dtype=float)
        audit.require(curve.shape == (49,) and bool(np.isfinite(curve).all()), "choice curve")
        minimum = min(curve.tolist())
        index = next(i for i, value in enumerate(curve) if value <= minimum+cfg["tie"])
        near = [i for i, value in enumerate(curve) if value <= minimum+cfg["near"]]
        audit.require(row["index"] == index, "argmin index")
        audit.require(row["scales"] == list(params[index]), "argmin scales")
        audit.require(row["near_indices"] == near, "near-optimal set")
        audit.near(row["objective"], curve[index], "selected objective")
    expected = {("MLE", N, None, None) for N in cfg["N"]}
    for N in cfg["N"]:
        for B in cfg["B"]:
            expected.add(("raw", N, B, record["d"]))
            expected.update(("col", N, B, k*record["d"]) for k in cfg["col_K_multipliers"])
    audit.require(seen == expected and len(seen) == 14, "14 declared contracts")


def direct_scores(Y, common, private, d, params, anchors, ns, lookup, cfg):
    """Full-coordinate Bernoulli log probabilities; independent spline basis."""
    b = DIRECT_BANKS
    rho = cfg["c"]/(d-1)
    base = math.sqrt(rho)*common[:b, :d, None]+math.sqrt(1-rho)*private[:b]
    raw = np.zeros((len(ns), b, len(anchors)))
    col = np.zeros((2, len(ns), b, len(anchors)))
    grid, nodes, primes = lookup
    if np.any(common[:b] < grid[0]) or np.any(common[:b] > grid[-1]):
        raise AssertionError("audit common values outside sealed interpolation domain")
    spline = CubicHermiteSpline(grid, nodes.T, primes.T, axis=0, extrapolate=False)
    h = np.moveaxis(spline(common[:b]), -1, 0)
    if not (np.isfinite(h).all() and np.all((h > 0) & (h < 1))):
        raise AssertionError("audit spline invalid")
    lp, lq = np.log(h), np.log1p(-h)
    for aj, index in enumerate(anchors):
        u, v = params[index]
        eta = base*np.array([u]*(d//2)+[v]*(d//2))
        log_yes, log_no = log_expit(eta), log_expit(-eta)
        ui, vi = cfg["axis"].index(u), cfg["axis"].index(v)
        for first in range(0, len(Y), DIRECT_LABEL_CHUNK):
            y = Y[first:first+DIRECT_LABEL_CHUNK]
            bit = (y > 0).astype(float)
            full_log = (np.einsum("ni,bki->nbk", bit, log_yes, optimize=True)
                        +np.einsum("ni,bki->nbk", 1-bit, log_no, optimize=True))
            raw_loss = math.log(d)-logsumexp(full_log, axis=2)
            na = bit[:, :d//2].sum(axis=1)[:, None, None]
            nb = bit[:, d//2:].sum(axis=1)[:, None, None]
            kernel_log = (na*lp[ui]+(d//2-na)*lq[ui]
                          +nb*lp[vi]+(d//2-nb)*lq[vi])
            col_losses = [math.log(K)-logsumexp(kernel_log[:, :, :K], axis=2)
                          for K in (d, 4*d)]
            for ni, N in enumerate(ns):
                length = min(len(y), max(0, N-first))
                if length:
                    raw[ni, :, aj] += raw_loss[:length].sum(axis=0)
                    for ki in range(2):
                        col[ki, ni, :, aj] += col_losses[ki][:length].sum(axis=0)
    raw /= np.asarray(ns)[:, None, None]
    col /= np.asarray(ns)[None, :, None, None]
    return raw, col


def workspace_record(record, phase, cfg, params, table, lookup, anchors, audit):
    d = record["d"]
    with np.load(ROOT/record["array"], allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    Y, common, private = data["Y"], data["common"], data["private"]
    ns = cfg["N"] if phase == "training" else [cfg["confirm_N"]]
    n, b = max(ns), max(cfg["B"]) if phase == "training" else cfg["confirm_B"]
    audit.require(Y.shape == (n, d) and np.isin(Y, [-1, 1]).all(), "label shape/support")
    audit.require(common.shape == (b, 4*d), "common shape")
    audit.require(private.shape == (b, d, d), "private shape")
    audit.require(np.isfinite(common).all() and np.isfinite(private).all(), "finite input arrays")
    indices = list(range(49)) if phase == "training" else record["indices"]
    audit.require(data["raw"].shape == (len(ns), b, len(indices)), "stored raw shape")
    audit.require(data["col"].shape == (2, len(ns), b, len(indices)), "stored col shape")
    counts = np.column_stack(((Y[:, :d//2] > 0).sum(axis=1), (Y[:, d//2:] > 0).sum(axis=1)))
    unique_states = len({tuple(row) for row in counts.tolist()})
    audit.require(record["raw_receipt"]["prior_vectors"] == b*d, "raw prior ledger")
    audit.require(record["raw_receipt"]["likelihood_particle_candidate_terms"] == n*b*d*len(indices),
                  "raw term ledger")
    col_expected = dict(common_particles=b*4*d, unique_count_states=unique_states,
                        kernel_count_particle_candidate_terms=unique_states*b*4*d*len(indices),
                        logsumexp_terms=unique_states*b*5*d*len(indices))
    for key, value in col_expected.items():
        audit.require(record["col_receipt"][key] == value, "col ledger/" + key)
    if phase == "training":
        audit.require(data["mle"].shape == (len(ns), 49), "stored MLE shape")
        for ni, N in enumerate(ns):
            hist = np.bincount(counts[:N, 0]*(d//2+1)+counts[:N, 1],
                               minlength=(d//2+1)**2)
            exact_mle = -(table["log_pattern"].reshape(49, -1) @ hist)/N
            audit.near(data["mle"][ni], exact_mle, "all MLE objectives", category="workspace_MLE")
        for row in record["selected"]:
            ni = ns.index(row["N"])
            if row["method"] == "MLE":
                curve = data["mle"][ni]
            elif row["method"] == "raw":
                curve = data["raw"][ni, :row["B"]].sum(axis=0)/row["B"]
            else:
                ki = cfg["col_K_multipliers"].index(row["K"]//d)
                curve = data["col"][ki, ni, :row["B"]].sum(axis=0)/row["B"]
            audit.near(row["curve"], curve, "public curve/local means", category="workspace_curves")
    else:
        audit.near(record["raw_means"], data["raw"][0].mean(axis=0),
                   "public confirmation raw", category="workspace_curves")
        audit.near(record["col_means"], data["col"][:, 0].mean(axis=1),
                   "public confirmation col", category="workspace_curves")
    raw, col = direct_scores(Y, common, private, d, params, anchors, ns, lookup, cfg)
    local_columns = [indices.index(index) for index in anchors]
    stored_raw = np.take(data["raw"][:, :DIRECT_BANKS], local_columns, axis=-1)
    stored_col = np.take(data["col"][:, :, :DIRECT_BANKS], local_columns, axis=-1)
    audit.near(raw, stored_raw, "direct full-eta raw", category="direct_raw")
    audit.near(col, stored_col, "independent-spline col", category="direct_col")
    return dict(d=d, stream=record["stream"], phase=phase, full_label_N=n, prefixes=ns,
                banks=DIRECT_BANKS, anchor_indices=anchors,
                raw_label_bank_particle_candidate_terms=n*DIRECT_BANKS*d*len(anchors),
                col_label_bank_particle_candidate_terms=n*DIRECT_BANKS*4*d*len(anchors),
                col_logsumexp_terms=n*DIRECT_BANKS*5*d*len(anchors))


def execute_audit(workspace):
    if not workspace:
        raise AssertionError("initial audit requires --workspace; later --check is public by default")
    audit = Audit()
    audit.require(read(OUT/"STATE.json")["status"] == "complete", "RUN must be complete")
    freeze, results = read(OUT/"FREEZE.json"), read(OUT/"RESULTS_FREEZE.json")
    audit.require(results["source_freeze_sha256"] == sha(OUT/"FREEZE.json"), "RUN seal chain")
    cfg, ref_seal = read(OUT/"CONFIG.json"), read(REF/"RESULTS_FREEZE.json")
    files = {}
    for mapping in (freeze["files"], freeze["reference_files"], results["files"], ref_seal["files"]):
        for name, digest in mapping.items():
            audit.require(name not in files or files[name] == digest, "hash map consistency")
            files[name] = digest
    for p in (OUT/"FREEZE.json", OUT/"RESULTS_FREEZE.json", REF/"RESULTS_FREEZE.json",
              Path(__file__), DOC/"RUN_AUDIT.md"):
        files[rel(p)] = sha(p)
    verify(files)
    local_files = dict(ref_seal["local_arrays"])
    local_files.update(results["local_arrays"])
    verify(local_files)
    selection_seal = read(OUT/"SELECTION_SEAL.json")
    audit.require(selection_seal["before_confirmation_RNG"] is True, "selection before confirmation")
    verify(selection_seal["files"])
    verify(selection_seal["local_arrays"])
    params = [(u, v) for u in cfg["axis"] for v in cfg["axis"]]
    truth_index, anchors = params.index(tuple(cfg["truth"])), [0, params.index(tuple(cfg["truth"]))]
    tables, training, confirmation = {}, {}, {}
    published_tables = read(OUT/"RISK_TABLES.json")
    for d in cfg["dimensions"]:
        reference = read(REF/f"REFERENCE_d{d}.json")
        audit.require([tuple(x) for x in reference["parameters"]] == params, "reference parameter order")
        table = pmf_risks(reference, d, truth_index, audit)
        tables[d] = table
        for key in ("joint_KL", "event_probabilities", "event_brier", "event_logloss", "teacher_events"):
            audit.near(published_tables[str(d)][key], table[key], "reference risk/" + key, category="risk")
        for s in range(cfg["streams"]):
            tr, cf = read(OUT/"training"/f"d{d}_s{s}.json"), read(OUT/"confirmation"/f"d{d}_s{s}.json")
            audit.require((tr["d"], tr["stream"], cf["d"], cf["stream"]) == (d, s, d, s), "record IDs")
            audit.require(tr["array_sha256"] == sha(ROOT/tr["array"]), "training local bytes")
            audit.require(cf["array_sha256"] == sha(ROOT/cf["array"]), "confirmation local bytes")
            check_choices(tr, params, cfg, audit)
            expected_indices = sorted({r["index"] for r in tr["selected"]} | set(anchors))
            audit.require(cf["indices"] == expected_indices, "confirmation candidate union")
            training[d, s], confirmation[d, s] = tr, cf
    models = read(OUT/"MODEL_RESULTS.json")
    model_map = {(row["d"], row["stream"], *select_key(row)): row for row in models}
    audit.require(len(models) == len(model_map) == 224, "224 model records")
    for (d, s), tr in training.items():
        for row in tr["selected"]:
            model = model_map[d, s, *select_key(row)]
            for key, value in row.items():
                audit.require(model[key] == value, "public model selection/" + key)
            j = row["index"]
            for dest, source in (("reference_joint_KL", "joint_KL"), ("event_probabilities", "event_probabilities"),
                                 ("event_brier", "event_brier"), ("event_logloss", "event_logloss")):
                audit.near(model[dest], tables[d][source][j], "model risk/" + dest, category="risk")
    risks = read(OUT/"RISK_CONTRASTS.json")
    scores = read(OUT/"SCORE_CONTRASTS.json")
    risk_map = {(r["d"], r["N"], r["B"], r["comparison"]): r for r in risks}
    score_map = {(r["d"], r["N"], r["B"], r["comparison"]): r for r in scores}
    audit.require(len(risks) == len(risk_map) == 32, "32 risk contrasts")
    audit.require(len(scores) == len(score_map) == 24, "24 score contrasts")
    expected_primaries = {}
    for d in cfg["dimensions"]:
        for N in cfg["N"]:
            for B in cfg["B"]:
                rv = {name: [] for name in ("raw_minus_col_d", "col_d_minus_MLE", "col_4d_minus_MLE", "col_d_minus_col_4d")}
                sv = {name: [] for name in ("raw_truth_minus_selected", "col_d_selected_minus_truth", "col_4d_selected_minus_truth")}
                for s in range(cfg["streams"]):
                    tr, cf = training[d, s], confirmation[d, s]
                    ix = [find_selected(tr, "raw", N, B, d)["index"],
                          find_selected(tr, "col", N, B, d)["index"],
                          find_selected(tr, "col", N, B, 4*d)["index"],
                          find_selected(tr, "MLE", N)["index"]]
                    a, b, c, m = tables[d]["joint_KL"][ix]
                    for name, value in zip(rv, (a-b, b-m, c-m, b-c)):
                        rv[name].append(float(value))
                    it = cf["indices"].index(truth_index)
                    ir, ic, ic4 = [cf["indices"].index(j) for j in ix[:3]]
                    sv["raw_truth_minus_selected"].append(cf["raw_means"][it]-cf["raw_means"][ir])
                    sv["col_d_selected_minus_truth"].append(cf["col_means"][0][ic]-cf["col_means"][0][it])
                    sv["col_4d_selected_minus_truth"].append(cf["col_means"][1][ic4]-cf["col_means"][1][it])
                for name, values in rv.items():
                    expected = compare_interval(audit, risk_map[d, N, B, name], values,
                                                cfg["risk_family"], cfg["risk_alpha"], cfg["risk_SE"], name)
                    if N == 4096 and B == 128 and name != "col_d_minus_col_4d":
                        passed = expected["precision_pass"] and (
                            expected["lower"] > cfg["material_improvement"] if name == "raw_minus_col_d"
                            else expected["upper"] < cfg["noninferiority_margin"])
                        expected_primaries[d, name] = bool(passed)
                for name, values in sv.items():
                    gate = d*cfg["raw_score_SE_per_d"] if name.startswith("raw") else cfg["col_score_SE"]
                    compare_interval(audit, score_map[d, N, B, name], values,
                                     cfg["score_family"], cfg["score_alpha"], gate, name)
    decision = read(OUT/"DECISION.json")
    audit.require(len(decision["primary_readouts"]) == 6, "six original readouts")
    for row in decision["primary_readouts"]:
        audit.require(row["passed"] == expected_primaries[row["d"], row["comparison"]], "original primary decision")
    audit.require(decision["risk_precision_pass"] == sum(r["precision_pass"] for r in risks), "risk precision count")
    audit.require(decision["score_precision_pass"] == sum(r["precision_pass"] for r in scores), "score precision count")
    floors = read(OUT/"FLOOR_PROBABILITY_BOUNDS.json")
    audit.require(len(floors) == 2 and floors == decision["additional_preregistered_floor_readouts"], "two floor readouts")
    for row in floors:
        d = row["d"]
        k = sum(find_selected(training[d, s], "raw", 4096, 128, d)["index"] == 0
                for s in range(cfg["streams"]))
        p = cp_lower(k, cfg["streams"], cfg["floor_alpha"]/cfg["floor_family"])
        r = tables[d]["joint_KL"]
        lower = float(r.min()+p*(r[0]-r.min()))
        audit.require(row["floor_selections"] == k and row["n"] == 8, "floor successes")
        for key, value in (("probability_lower", p), ("reference_minimum", r.min()),
                           ("floor_reference_risk", r[0]), ("expected_fixed_reference_risk_lower", lower)):
            audit.near(row[key], value, "CP/" + key, category="CP")
        audit.require(row["passed"] == bool(lower > cfg["material_improvement"]), "floor primary")
        audit.require(row["preregistered"] and not row["post_hoc"], "floor preregistration")
        audit.require(row["binomial_family_size"] == 2 and row["binomial_family_alpha"] == .05, "floor family")
        audit.require(row["strict_Gaussian_quadrature_error_certificate"] is False, "floor reference limitation")
    work_rows = []
    for d in cfg["dimensions"]:
        meta = read(REF/f"LOOKUP_d{d}.json")
        with np.load(ROOT/meta["array"], allow_pickle=False) as archive:
            lookup = (archive["z"], archive["h"][1], archive["derivative"][1])
        for s in range(cfg["streams"]):
            for phase, records in (("training", training), ("confirmation", confirmation)):
                work_rows.append(workspace_record(records[d, s], phase, cfg, params, tables[d], lookup, anchors, audit))
                print(json.dumps(dict(audit="direct subset", d=d, stream=s, phase=phase)), flush=True)
    counts = read(OUT/"COUNTS.json")
    dimensions, streams = cfg["dimensions"], cfg["streams"]
    train_n, confirm_n = max(cfg["N"]), cfg["confirm_N"]
    train_b, confirm_b = max(cfg["B"]), cfg["confirm_B"]
    expected_counts = dict(
        new_training_teacher_vectors=len(dimensions)*streams*train_n,
        new_confirmation_teacher_vectors=len(dimensions)*streams*confirm_n,
        new_teacher_vectors=len(dimensions)*streams*(train_n+confirm_n),
        teacher_bit_outputs=streams*(train_n+confirm_n)*sum(dimensions),
        training_raw_prior_vectors=streams*train_b*sum(dimensions),
        confirmation_raw_prior_vectors=streams*confirm_b*sum(dimensions),
        training_common_factor_draws=streams*train_b*4*sum(dimensions),
        confirmation_common_factor_draws=streams*confirm_b*4*sum(dimensions),
        common_factor_prefix_shared_with_raw_per_phase=streams*train_b*sum(dimensions),
        final_selection_endpoints=len(dimensions)*streams*len(cfg["N"])*(1+len(cfg["B"])*(1+len(cfg["col_K_multipliers"]))),
        streams=len(dimensions)*streams, new_neural_models=0, new_deployment_samples=0)
    all_records = list(training.values())+list(confirmation.values())
    expected_counts.update(
        raw_likelihood_particle_candidate_terms=sum(r["raw_receipt"]["likelihood_particle_candidate_terms"] for r in all_records),
        col_count_particle_candidate_terms=sum(r["col_receipt"]["kernel_count_particle_candidate_terms"] for r in all_records),
        col_logsumexp_terms=sum(r["col_receipt"]["logsumexp_terms"] for r in all_records))
    audit.require(counts == expected_counts, "all ledger entries")
    return dict(
        schema=SCHEMA, passed=True, checks=audit.checks, maximum_absolute_differences=audit.errors,
        producer_results_freeze_sha256=sha(OUT/"RESULTS_FREEZE.json"),
        source_files=files, local_files=local_files,
        public_recalculation=dict(selections=224, risk_contrasts=32, score_contrasts=24, floor_CP_bounds=2,
                                  original_primary_readouts=6, reference_PMFS=98, events_per_model=3),
        workspace_recalculation=dict(full_local_curve_links=224, full_MLE_curves=32, phase_records=32,
                                     direct_subset=work_rows, all_raw_scores_replayed=False,
                                     spline_expression="scipy CubicHermiteSpline, sealed primary values/derivatives",
                                     distinct_quadrature_not_run=True),
        scientific_ledger_verified=expected_counts,
        audit_new_scientific_samples=0, audit_RNG_calls=0, audit_new_fits=0, audit_new_integrals=0,
        inference_scope="Original approximate t families and fixed-reference CP interpretation are unchanged.")


def check(workspace):
    receipt = read(RECEIPT)
    if receipt["schema"] != SCHEMA or receipt["passed"] is not True:
        raise AssertionError("audit has not passed")
    verify(receipt["source_files"])
    if workspace:
        verify(receipt["local_files"])
    if sha(OUT/"RESULTS_FREEZE.json") != receipt["producer_results_freeze_sha256"]:
        raise AssertionError("producer seal changed")
    print(json.dumps(dict(passed=True, workspace=workspace, receipt_sha256=sha(RECEIPT),
                          arithmetic_not_reexecuted=True, science_not_reexecuted=True)))


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    parser.add_argument("--workspace", action="store_true")
    args = parser.parse_args()
    if not args.write:
        check(args.workspace)
        return
    if RECEIPT.exists():
        raise AssertionError("audit receipt already exists; no overwrite")
    if read(OUT/"STATE.json")["status"] != "complete":
        raise AssertionError("do not audit an unfinished RUN")
    started = time.perf_counter()
    try:
        result = execute_audit(args.workspace)
        result["utc"] = datetime.now(timezone.utc).isoformat()
        result["elapsed_seconds"] = time.perf_counter()-started
        write_new(RECEIPT, result)
        print(json.dumps(dict(passed=True, checks=result["checks"], receipt_sha256=sha(RECEIPT))), flush=True)
    except Exception as exc:
        write_new(RECEIPT, dict(schema=SCHEMA, passed=False, utc=datetime.now(timezone.utc).isoformat(),
                               audit_source_sha256=sha(Path(__file__)), error=repr(exc),
                               traceback=traceback.format_exc(), no_producer_files_modified=True))
        raise


if __name__ == "__main__":
    main()
