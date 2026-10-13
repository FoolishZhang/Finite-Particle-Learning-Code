"""Independent PROFILE arithmetic/input audit; no producer computation imports.

--write --workspace replays recorded RNG consumption and bounded score prefixes.
--check is a public hash/receipt check; --workspace additionally checks arrays.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
import time
import traceback

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"

import numpy as np
from scipy.special import log_expit, logsumexp, rel_entr
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/covariance_boundary/PROFILE"
NUM = ROOT / "artifacts/covariance_boundary/NUMERIC"
MOM = ROOT / "artifacts/objective_strengthening/N/MOMENTS.json"
DOC = ROOT / "docs/08_research_研究工作/相关结构补偿与发射噪声_20261001"
RECEIPT = OUT / "INDEPENDENT_AUDIT.json"
NAMES = ["teacher", "fixedRfloor", "comp", "half", "over", "independent"]
SCHEMA = "covariance-boundary-profile-independent-audit-v1"
TOL = 1e-10


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def verify(files):
    for name, digest in files.items():
        if sha(ROOT/name) != digest:
            raise AssertionError("hash mismatch: " + name)


class Audit:
    def __init__(self):
        self.checks = 0
        self.maximum = {}

    def require(self, condition, message):
        self.checks += 1
        if not condition:
            raise AssertionError(message)

    def near(self, actual, expected, message, category="public", tol=TOL):
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        self.require(a.shape == b.shape, message + ": shape")
        self.require(np.isfinite(a).all() and np.isfinite(b).all(), message + ": finite")
        error = float(np.max(np.abs(a-b))) if a.size else 0.0
        self.maximum[category] = max(self.maximum.get(category, 0.0), error)
        self.require(error <= tol, message + ": difference " + repr(error))


def stable_means(array):
    """Independent summation order for all per-bank means."""
    array = np.asarray(array)
    result = np.empty(array.shape[1:])
    for index in np.ndindex(result.shape):
        result[index] = math.fsum(float(x) for x in array[(slice(None),)+index])/len(array)
    return result


def canonical_patterns(d):
    m = d//2
    pairs = [(a, b) for a in range(m+1) for b in range(m+1)]
    bits = np.array([[int(i < a) if i < m else int(i-m < b)
                      for i in range(d)] for a, b in pairs], dtype=float)
    return bits, pairs


def direct_prefix_scores(common, private, params, probability, psis, Ks):
    """Explicit canonical full-coordinate Bernoulli log probabilities.

    No group logit prefix sums and no producer score function are used.
    """
    banks, kmax, d = private.shape
    bits, pairs = canonical_patterns(d)
    weights = np.array([probability[a, b] for a, b in pairs])
    raw = np.empty((banks, len(Ks), len(params)))
    baseline = np.empty_like(raw)
    corrected = np.empty_like(raw)
    for j, param in enumerate(params):
        scales = np.array([param["scales"][0]]*(d//2)+[param["scales"][1]]*(d//2))
        loading = np.array([param["lambda"][0]]*(d//2)+[param["lambda"][1]]*(d//2))
        eta = scales*(common[:, :, None]*loading+private*np.sqrt(1-loading**2))
        lp, lq = log_expit(eta), log_expit(-eta)
        full = (np.einsum("ti,bki->tbk", bits, lp, optimize=True)
                +np.einsum("ti,bki->tbk", 1-bits, lq, optimize=True))
        for ki, K in enumerate(Ks):
            loss = math.log(K)-logsumexp(full[:, :, :K], axis=2)
            single = -np.sum(full[:, :, :K], axis=2)/K
            raw[:, ki, j] = weights @ loss
            baseline[:, ki, j] = weights @ single
            corrected[:, ki, j] = psis[j] + weights @ (loss-single)
    return corrected, raw, baseline


def replay_rng(record, common_prefix, private_prefix, cfg, audit):
    """Replay only the previously specified streams; no score recomputation."""
    d, s = record["d"], record["stream"]
    sd = cfg["seed"]+100000*d+1000*s
    audit.require((record["seed_common"], record["seed_private"]) == (sd, sd+1), "RNG seed namespace")
    generators = [np.random.Generator(np.random.PCG64(sd)),
                  np.random.Generator(np.random.PCG64(sd+1))]
    audit.require([g.bit_generator.state for g in generators] == record["RNG_initial"], "RNG initial states")
    calls = 0
    for first in range(0, cfg["banks"], cfg["bank_chunk"]):
        size = min(cfg["bank_chunk"], cfg["banks"]-first)
        common = generators[0].standard_normal((size, d))
        private = generators[1].standard_normal((size, d, d))
        calls += 2
        if first == 0:
            n = cfg["prefix_banks_saved"]
            audit.require(np.array_equal(common[:n], common_prefix), "RNG common prefix exact bytes")
            audit.require(np.array_equal(private[:n], private_prefix), "RNG private prefix exact bytes")
    audit.require([g.bit_generator.state for g in generators] == record["RNG_final"], "RNG final states")
    return dict(normal_calls=calls, normal_scalars=cfg["banks"]*d*(d+1),
                seed_common=sd, seed_private=sd+1, full_consumption_replayed=True,
                post_prefix_scores_replayed=False)


def interval(values, alpha, family, gate):
    x = np.asarray(values, dtype=float)
    n = len(x)
    mean = math.fsum(float(v) for v in x)/n
    se = math.sqrt(math.fsum((float(v)-mean)**2 for v in x)/(n*(n-1)))
    radius = float(student_t.isf(alpha/(2*family), n-1))*se
    return dict(values=x, mean=mean, SE=se, lower=mean-radius, upper=mean+radius,
                precision_pass=se <= gate)


def verify_reference(d, cfg, moment_values, audit):
    reference = read(NUM/f"REFERENCE_d{d}.json")
    audit.require(reference["names"] == NAMES, "reference names")
    params = reference["params"]
    audit.require(len(params) == 6, "six parameters")
    numeric_cfg = read(NUM/"CONFIG.json")
    audit.require(numeric_cfg["c"] == .9 and numeric_cfg["a"] == 1.0
                  and numeric_cfg["truth"] == [6.0, 2.0], "fixed numerical teacher contract")
    shared = math.sqrt(numeric_cfg["c"]/(d-1))
    comp = np.array([shared*moment_values[str(float(s))]["alpha"]/moment_values["1.0"]["alpha"]
                     for s in numeric_cfg["truth"]])
    expected_loadings = [[shared, shared], [shared, shared], comp, .5*comp, 1.5*comp, [0., 0.]]
    audit.require(params[0]["scales"] == numeric_cfg["truth"], "teacher scales")
    for j, param in enumerate(params):
        audit.require(param["name"] == NAMES[j] and param["mean"] == [0.0, 0.0], "parameter role/mean")
        audit.near(param["lambda"], expected_loadings[j], "independent loading construction",
                   category="parameters")
        scale = np.repeat(param["scales"], d//2)
        loading = np.repeat(param["lambda"], d//2)
        audit.require(scale.shape == loading.shape == (d,) and np.all(abs(loading) < 1), "valid loadings")
        covariance = np.outer(scale*loading, scale*loading)
        covariance.flat[::d+1] += scale**2*(1-loading**2)
        eigenvalues = np.linalg.eigvalsh(covariance)
        audit.require(eigenvalues[0] >= cfg["nu"]-TOL and eigenvalues[-1] <= cfg["Lambda"]+TOL,
                      "common spectral class")
        audit.near(np.diag(covariance), scale**2, "marginal variances", category="covariance")
        audit.near(reference["spectra"][j]["eigenvalues"], eigenvalues, "reference eigenvalues",
                   category="covariance")
        if j:
            audit.require(param["scales"] == [1.0, 1.0], "all five floor candidates")
    probability = np.asarray(reference["pmf"], dtype=float)
    audit.require(probability.shape == (6, d//2+1, d//2+1) and np.all(probability > 0),
                  "full count PMFs")
    audit.near(probability.sum(axis=(1, 2)), np.ones(6), "PMF normalization")
    risks = np.sum(rel_entr(probability[0][None], probability), axis=(1, 2))
    audit.near(risks, reference["risk_to_teacher"], "independent forward count KL",
               category="reference_risk")
    return params, probability[0], risks


def execute(workspace):
    if not workspace:
        raise AssertionError("initial --write requires --workspace")
    audit = Audit()
    audit.require(read(OUT/"STATE.json")["status"] == "complete", "PROFILE complete")
    freeze, result = read(OUT/"FREEZE.json"), read(OUT/"RESULTS_FREEZE.json")
    audit.require(result["source_freeze_sha256"] == sha(OUT/"FREEZE.json"), "seal chain")
    files = {}
    for mapping in (freeze["files"], freeze["reference_files"], result["files"]):
        for name, digest in mapping.items():
            audit.require(name not in files or files[name] == digest, "hash map agreement")
            files[name] = digest
    for path in (OUT/"FREEZE.json", OUT/"RESULTS_FREEZE.json", Path(__file__), DOC/"PROFILE_AUDIT.md"):
        files[rel(path)] = sha(path)
    verify(files)
    verify(result["local_arrays"])
    cfg = read(OUT/"CONFIG.json")
    audit.require(cfg["dimensions"] == [16, 64] and cfg["names"] == NAMES, "fixed dimension/candidate scope")
    audit.require(cfg["banks"] == 2048 and cfg["streams"] == 8 and cfg["prefix_banks_saved"] == 4,
                  "fixed sampling scope")
    moments = read(MOM)
    audit.require(moments["qualified"] is True, "old moments qualified")
    ell = moments["gh"][-1]["values"]
    published_risks = read(OUT/"REFERENCE_RISKS.json")
    all_rows, recomputed_means, references, replay_rows, direct_rows = [], {}, {}, [], []
    ledger_fields = ("banks", "unique_base_prior_vectors", "base_gaussian_scalars",
                     "transformed_candidate_prior_vectors", "count_particle_candidate_terms",
                     "logsumexp_terms")
    expected_ledger = {name: 0 for name in ledger_fields}
    seeds = []
    for d in cfg["dimensions"]:
        params, weights, risks = verify_reference(d, cfg, ell, audit)
        references[d] = risks
        audit.require(published_risks[str(d)]["names"] == NAMES, "published risk names")
        audit.near(published_risks[str(d)]["risk_to_teacher"], risks, "published risks",
                   category="reference_risk")
        psis = np.array([math.fsum((d//2)*float(ell[str(float(s))]["ell"])
                                   for s in param["scales"]) for param in params])
        audit.require(np.all(psis[1:] == psis[1]), "identical floor Psi")
        Ks = [1, 10, d]
        for s in range(cfg["streams"]):
            record = read(OUT/"streams"/f"d{d}_s{s}.json")
            audit.require((record["d"], record["stream"]) == (d, s), "stream ID")
            audit.require(record["Ks"] == Ks and record["names"] == NAMES, "stream axes")
            audit.near(record["psi"], psis, "analytic Psi from old moments")
            audit.require(record["array_sha256"] == sha(ROOT/record["array"]), "stream local hash")
            with np.load(ROOT/record["array"], allow_pickle=False) as archive:
                values = {key: archive[key] for key in archive.files}
            for key in ("scores", "raw", "single"):
                array = values[key]
                audit.require(array.shape == (cfg["banks"], 3, 6) and np.isfinite(array).all(),
                              "array shape/finite " + key)
                public_key = dict(scores="means", raw="raw_means", single="single_means")[key]
                means = stable_means(array)
                audit.near(record[public_key], means, "all-bank " + public_key, category="stream_means")
                if key == "scores":
                    recomputed_means[d, s] = means
            audit.near(values["scores"], psis[None, None, :]+values["raw"]-values["single"],
                       "all-bank control variate", category="control_variate")
            audit.near(values["raw"][:, 0], values["single"][:, 0], "all-bank K1 raw/single",
                       category="K1", tol=cfg["K1_gate"])
            audit.near(values["scores"][:, 0], np.broadcast_to(psis, (cfg["banks"], 6)),
                       "all-bank exact K1 Psi", category="K1", tol=cfg["K1_gate"])
            audit.require(np.max(values["scores"]-psis[None, None]) <= cfg["K1_gate"], "Jensen gap nonpositive")
            common, private = values["common_prefix"], values["private_prefix"]
            audit.require(common.shape == (4, d) and private.shape == (4, d, d), "stored prefix shape")
            replay = replay_rng(record, common, private, cfg, audit)
            replay_rows.append(dict(d=d, stream=s, **replay))
            seeds.extend([replay["seed_common"], replay["seed_private"]])
            corrected, raw, single = direct_prefix_scores(common, private, params, weights, psis, Ks)
            for key, direct in (("scores", corrected), ("raw", raw), ("single", single)):
                audit.near(values[key][:4], direct, "all-count full-coordinate prefix " + key,
                           category="direct_" + key, tol=cfg["formula_gate"])
            direct_rows.append(dict(d=d, stream=s, banks=4, candidates=6, Ks=Ks,
                                    count_states=(d//2+1)**2,
                                    count_particle_candidate_terms=6*4*d*(d//2+1)**2,
                                    logsumexp_terms=6*4*sum(Ks)*(d//2+1)**2))
            expected = dict(banks=cfg["banks"], unique_base_prior_vectors=cfg["banks"]*d,
                            base_gaussian_scalars=cfg["banks"]*d*(d+1),
                            transformed_candidate_prior_vectors=6*cfg["banks"]*d,
                            count_particle_candidate_terms=6*cfg["banks"]*d*(d//2+1)**2,
                            logsumexp_terms=6*cfg["banks"]*sum(Ks)*(d//2+1)**2)
            for key, value in expected.items():
                audit.require(record[key] == value, "stream ledger/" + key)
                expected_ledger[key] += value
            audit.require(record["prefix_prior_inputs_saved"] == 4*d and record["all_prior_inputs_saved"] is False,
                          "saved-input scope")
            audit.require(record["control_variate_preserves_population_target"] is True, "CV contract")
            all_rows.append(record)
            print(json.dumps(dict(audit="all-bank means and fixed prefix", d=d, stream=s)), flush=True)
    audit.require(len(set(seeds)) == len(seeds) == 32, "32 distinct specified RNG seeds")
    contrasts = read(OUT/"CONTRASTS.json")
    c_map = {(r["d"], r["K"], r["comparison"]): r for r in contrasts}
    audit.require(len(contrasts) == len(c_map) == 16, "16 contrasts")
    comparisons = [("comp_minus_fixedRfloor", 2, 1), ("over_minus_comp", 4, 2),
                   ("half_minus_comp", 3, 2), ("independent_minus_comp", 5, 2)]
    expected_main = {}
    precision_pass = 0
    for d in cfg["dimensions"]:
        means = np.array([recomputed_means[d, s] for s in range(cfg["streams"])])
        for ki, K in enumerate([1, 10, d]):
            if K == 1:
                continue
            for name, left, right in comparisons:
                expected = interval(means[:, ki, left]-means[:, ki, right],
                                    cfg["alpha"], cfg["family"], cfg["SE_gate"])
                row = c_map[d, K, name]
                for key in ("values", "mean", "SE", "lower", "upper"):
                    audit.near(row[key], expected[key], name + "/" + key, category="interval")
                audit.require(row["precision_pass"] == expected["precision_pass"], name + "/precision")
                audit.require(row["t_is_approximation"] is True, name + "/t interpretation")
                risk_difference = float(references[d][left]-references[d][right])
                audit.near(row["reference_risk_difference"], risk_difference, "contrast risk",
                           category="reference_risk")
                precision_pass += int(expected["precision_pass"])
                if name == "over_minus_comp":
                    expected_main[d, K] = dict(
                        passed=bool(expected["precision_pass"] and expected["upper"] < -cfg["score_margin"]
                                    and risk_difference > cfg["risk_margin"]),
                        score_upper=expected["upper"], risk_difference=risk_difference)
    decision = read(OUT/"DECISION.json")
    audit.require(len(decision["primary"]) == 4, "four primary readouts")
    for row in decision["primary"]:
        expected = expected_main[row["d"], row["K"]]
        audit.require(row["passed"] == expected["passed"], "primary decision")
        for key in ("score_upper", "risk_difference"):
            audit.near(row[key], expected[key], "primary/" + key, category="interval")
        audit.require(row["claim"] == "only the prespecified pair, not all covariance optimizers", "claim scope")
    audit.require(decision["precision_pass"] == precision_pass and decision["total"] == 16, "precision ledger")
    audit.require(decision["continuous_global_minimum_claimed"] is False
                  and decision["optimizer_selected_candidate_claimed"] is False
                  and decision["numerical_reference_conditional"] is True
                  and decision["all_failures_preserved"] is True, "decision scope")
    expected_ledger.update(independent_streams=16, new_teacher_labels=0, new_fits=0,
                           new_neural_models=0, new_integrals=0, candidates=6, analytic_K1_scores=True)
    audit.require(read(OUT/"COUNTS.json") == expected_ledger, "complete count ledger")
    return dict(schema=SCHEMA, passed=True, checks=audit.checks, maximum_absolute_differences=audit.maximum,
                producer_results_freeze_sha256=sha(OUT/"RESULTS_FREEZE.json"),
                source_files=files, local_files=result["local_arrays"],
                all_bank_recalculation=dict(streams=16, arrays=16, banks=32768, score_cells=32768*3*6,
                                            kinds=["corrected", "uncontrolled", "single"], contrasts=16, primary_readouts=4),
                bounded_direct_score_replay=direct_rows,
                full_RNG_consumption_replay=replay_rows,
                audit_RNG_replay_calls=sum(r["normal_calls"] for r in replay_rows),
                audit_RNG_replayed_normal_scalars=sum(r["normal_scalars"] for r in replay_rows),
                audit_new_scientific_random_draws=0, audit_new_teacher_labels=0,
                audit_new_fits=0, audit_new_integrals=0,
                all_original_likelihood_terms_replayed=False,
                scientific_ledger_verified=expected_ledger,
                verified_outcomes=dict(precision_pass=precision_pass, total=16,
                                       primary_pass=sum(r["passed"] for r in expected_main.values()), primary_total=4),
                limits="Fixed six-candidate profiles; numerical-reference conditional and approximate t family unchanged.")


def check(workspace):
    receipt = read(RECEIPT)
    if receipt["schema"] != SCHEMA or receipt["passed"] is not True:
        raise AssertionError("audit receipt has not passed")
    verify(receipt["source_files"])
    if workspace:
        verify(receipt["local_files"])
    if sha(OUT/"RESULTS_FREEZE.json") != receipt["producer_results_freeze_sha256"]:
        raise AssertionError("producer result seal changed")
    print(json.dumps(dict(passed=True, workspace=workspace, receipt_sha256=sha(RECEIPT),
                          arithmetic_and_RNG_not_reexecuted=True)))


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
        raise AssertionError("receipt exists; no overwrite")
    if read(OUT/"STATE.json")["status"] != "complete":
        raise AssertionError("PROFILE must be complete before audit")
    started = time.perf_counter()
    try:
        result = execute(args.workspace)
        result["utc"] = datetime.now(timezone.utc).isoformat()
        result["elapsed_seconds"] = time.perf_counter()-started
        write_new(RECEIPT, result)
        print(json.dumps(dict(passed=True, checks=result["checks"], receipt_sha256=sha(RECEIPT))), flush=True)
    except Exception as exc:
        write_new(RECEIPT, dict(schema=SCHEMA, passed=False, utc=datetime.now(timezone.utc).isoformat(),
                               audit_source_sha256=sha(Path(__file__)), error=repr(exc),
                               traceback=traceback.format_exc(), producer_files_unchanged=True))
        raise


if __name__ == "__main__":
    main()
