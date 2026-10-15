"""Frozen J1 same-finite-model proposal comparison; only --run samples banks."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import platform
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import norm

import selection_finite_core as finite
import signal_proposal_core as legacy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/gaussian_remainder_v1_20261014/j1_attempt1"
PLAN = ROOT / "docs/08_research_研究工作/GaussianRemainder_v1_20261014/PLAN.md"
OLD_H3 = ROOT / "artifacts/selection_limit_v1_20261014/h3_attempt1"
MS, ARMS = (32, 128), ("beta1", "critical")
B, K, CHUNK, SEED = 16384, 8, 256, 202610146101
GAMMA, TAU, PATH_END = 7.241913488981693, 3., .45
PATH_N, CHECK_N, CHECK_B = 8, 12, 1024
MAX_NORMALS, MAX_CALLS, MAX_SECONDS = 300000000, 704512, 1800
MEAN_GATE, MAX_GATE = 2e-7, 2e-5
ROLES = {"teacher_z": 0, "z": 1, "counts": 2,
         "private_beta1": 3, "private_critical": 4}
METRICS = ("loss", "controlled_loss", "price", "raw_rp", "cv_rp", "overlap", "ess")
LOCAL_MODULES = ("signal_proposal_core", "competition_core", "selection_finite_core")


def native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: native(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [native(item) for item in value]
    return value


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(native(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def save_new_npz(path, arrays):
    with Path(path).open("xb") as stream:
        np.savez(stream, **arrays)
    return {"path": Path(path).name, "bytes": Path(path).stat().st_size,
            "sha256": digest(path)}


def stats(values):
    values = np.asarray(values)
    return {"banks": len(values), "mean": values.mean(axis=0),
            "variance": values.var(axis=0, ddof=1),
            "se": values.std(axis=0, ddof=1) / math.sqrt(len(values)),
            "second_moment": np.mean(values * values, axis=0)}


def new_ledger():
    return {"normal_scalars": 0, "shared_teacher_active_normal_scalars": 0,
            "binomial_variates": 0, "explicit_rejection_uniforms": 0,
            "bank_score_calls": 0, "independent_target_observations": 0,
            "proposal_banks": 0, "new_risk_observations": 0,
            **{arm + "_private_" + key: 0 for arm in ARMS
               for key in ("proposed", "accepted", "used", "discarded_accepted")}}


def budget_check(ledger, started):
    elapsed = time.monotonic() - started
    if (ledger["normal_scalars"] > MAX_NORMALS or
            ledger["bank_score_calls"] > MAX_CALLS or elapsed > MAX_SECONDS):
        raise RuntimeError(f"Frozen J1 budget exceeded: ledger={ledger}, seconds={elapsed}")
    return elapsed


class CountedRNG:
    """Count explicit draws; NumPy binomial internal uniforms remain unknown."""
    def __init__(self, generator, ledger, role):
        self.generator, self.ledger, self.role = generator, ledger, role

    def standard_normal(self, shape):
        number = int(shape) if np.isscalar(shape) else math.prod(shape)
        if self.ledger["normal_scalars"] + number > MAX_NORMALS:
            raise RuntimeError("J1 Gaussian scalar budget would be exceeded")
        value = self.generator.standard_normal(shape)
        self.ledger["normal_scalars"] += number
        if self.role.startswith("private_"):
            arm = self.role.removeprefix("private_")
            self.ledger[arm + "_private_proposed"] += number
        else:
            self.ledger["shared_teacher_active_normal_scalars"] += number
        return value

    def random(self, shape):
        value = self.generator.random(shape)
        self.ledger["explicit_rejection_uniforms"] += int(shape) if np.isscalar(shape) else math.prod(shape)
        return value

    def binomial(self, n, p):
        value = self.generator.binomial(n, p)
        self.ledger["binomial_variates"] += np.size(value)
        return value


def role_rngs(M, ledger):
    return {name: CountedRNG(np.random.Generator(np.random.PCG64DXSM(
        np.random.SeedSequence([SEED, M, role]))), ledger, name)
        for name, role in ROLES.items()}


def path_rule(order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return PATH_END * (nodes + 1) / 2, PATH_END * weights / 2


def score_block(inputs, parameters, ledger, *, check=False, scorer=None, started=None):
    scorer = finite.full_banks if scorer is None else scorer
    size = len(inputs["z"])
    requested = size * (2 + PATH_N + (CHECK_N if check else 0))
    if ledger["bank_score_calls"] + requested > MAX_CALLS:
        raise RuntimeError("J1 bank-score budget would be exceeded")

    def evaluate(C):
        if started is not None:
            budget_check(ledger, started)
        ledger["bank_score_calls"] += size
        return scorer(inputs["z"], inputs["T"], inputs["counts"], C,
                      label_probs=inputs["q"], **parameters)

    identity = evaluate(np.eye(3))
    equal = evaluate(legacy.correlation_matrix([PATH_END] * 3))
    out = {"I_" + key: identity[key] for key in METRICS}
    out.update({"equal_" + key: equal[key] for key in METRICS})
    out["direct_loss_gap"] = equal["shifted_loss"] - identity["shifted_loss"]
    out["direct_controlled_gap"] = equal["shifted_controlled_loss"] - identity["shifted_controlled_loss"]

    def integrate(order):
        sums = {key: np.zeros(size) for key in ("price", "raw", "cv")}
        for u, weight in zip(*path_rule(order)):
            values = evaluate(legacy.correlation_matrix([u] * 3))
            for name, key in (("price", "price"), ("raw", "raw_rp"), ("cv", "cv_rp")):
                sums[name] += weight * values[key].sum(axis=1)
        # This arm's own I field has zero population mean, not bankwise zero.
        sums["centered_price"] = sums["price"] - PATH_END * identity["price"].sum(axis=1)
        return sums

    key_pairs = (("price", "path_price_uncentered"), ("centered_price", "path_price_gap"),
                 ("raw", "path_raw_integral"), ("cv", "path_cv_integral"))
    sums = integrate(PATH_N)
    for name, key in key_pairs:
        out[key] = sums[name]
    out["path_raw_minus_direct"] = sums["raw"] - out["direct_loss_gap"]
    out["path_cv_minus_direct"] = sums["cv"] - out["direct_controlled_gap"]
    if check:
        refined = integrate(CHECK_N)
        for name, key in key_pairs:
            out[key + "_12"] = refined[name]
            out[key + "_12_minus_8"] = refined[name] - sums[name]
        out["path_raw_12_minus_direct"] = refined["raw"] - out["direct_loss_gap"]
        out["path_cv_12_minus_direct"] = refined["cv"] - out["direct_controlled_gap"]
    return out


def order_diagnostic(difference):
    difference = np.asarray(difference)
    mean, maximum = float(difference.mean()), float(np.max(abs(difference)))
    return {"mean_difference": mean, "absolute_mean_difference": abs(mean),
            "maximum_absolute_bank_difference": maximum,
            "mean_gate": MEAN_GATE, "max_gate": MAX_GATE,
            "passed": abs(mean) <= MEAN_GATE and maximum <= MAX_GATE,
            "qualification": "ordinary_order_diagnostic_not_error_certificate"}


def paired_statistics(beta1, critical):
    return stats(np.asarray(critical) - np.asarray(beta1))


def family_row(M, arm, summary, numerical_passed):
    zcrit = float(norm.ppf(1 - .05 / (2 * 4)))
    lo, hi = float(summary["mean"] - zcrit * summary["se"]), float(summary["mean"] + zcrit * summary["se"])
    expected = "positive" if arm == "beta1" else "negative"
    sign_passed = lo > 0 if arm == "beta1" else hi < 0
    return {"M": M, "arm": arm, "expected_sign": expected, "statistics": summary,
            "interval": [lo, hi], "normal_critical": zcrit, "sign_supported": sign_passed,
            "numerical_qualification_passed": numerical_passed,
            "qualified_sign_supported": sign_passed and numerical_passed}


def freeze(out):
    paths = [Path(__file__), PLAN]
    paths.extend(ROOT / "script" / (name + ".py") for name in LOCAL_MODULES)
    paths.extend(ROOT / "script" / name for name in (
        "selection_finite_test.py", "competition_core_test.py",
        "signal_proposal_core_test.py", "informative_grouped_core.py",
        "informative_grouped_core_test.py"))
    paths.extend(OLD_H3 / f"M{M}" / "RESULTS.json" for M in MS)
    paths.extend(ROOT / "docs/08_research_研究工作/SelectionLimit_v1_20261014" / name
                 for name in ("ORDER_REVERSAL_COROLLARY.md", "RISK_AND_PATH_AUDIT.md"))
    source = out / "source"
    source.mkdir()
    hashes = {}
    for path in paths:
        relative = path.relative_to(ROOT)
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        before = digest(path)
        shutil.copyfile(path, target)
        if before != digest(target) or before != digest(path):
            raise RuntimeError("Source changed during freeze: " + str(relative))
        hashes[relative.as_posix()] = before
    binding = {}
    for M in MS:
        relative = (OLD_H3 / f"M{M}" / "RESULTS.json").relative_to(ROOT)
        old = read(source / relative)
        p = old["parameters"]
        if (p["M"] != M or p["tau"] != TAU or p["a"] != [1.] * 3 or
                p["gamma"] != GAMMA or old["teacher"] != "independent" or
                old["candidate"] != "equal_.45"):
            raise ValueError("Old risk binding is not the specified same model")
        binding[str(M)] = {"source_path": relative.as_posix(), "sha256": hashes[relative.as_posix()],
            "statistics": old["risk"]["statistics"], "banks": old["risk_banks"],
            "numerical_diagnostic": old["risk"]["final"],
            "observation": old["risk"]["observation"], "role": "old_risk_scale_only_not_new_validation"}
    contract = {"schema": "remainder_finite_reversal_J1_v1", "source_sha256": hashes,
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "M": MS, "arms": ARMS, "B_per_M": B, "K": K, "bank_chunk": CHUNK,
        "a": [1.] * 3, "tau": TAU, "gamma": GAMMA, "path_end": PATH_END,
        "seed_family": SEED, "role_ids": ROLES,
        "rng": "PCG64DXSM; SeedSequence([seed_family,M,role]); continuous per-M role streams",
        "shared_inputs": ["teacher_z", "Xstar", "counts", "q", "z"],
        "private_policy": "independent proposal-arm streams; independently drawn from each q_beta,0",
        "path_orders": [PATH_N, CHECK_N], "path_check_banks": CHECK_B,
        "path_rules": {str(order): {"u": path_rule(order)[0], "weights_du": path_rule(order)[1]}
                       for order in (PATH_N, CHECK_N)},
        "h_orders": [128, 192], "h_check_teacher_banks": CHUNK,
        "numerical_gates": {"absolute_mean": MEAN_GATE, "maximum_bank": MAX_GATE},
        "family_size": 4, "family_alpha": .05,
        "primary": "each M: beta1 gap positive and critical gap negative; normal Bonferroni approximations",
        "paired_shift_role": "descriptive; variance computed from bankwise critical-minus-beta1",
        "planned_bank_score_calls": MAX_CALLS, "max_Gaussian_scalars": MAX_NORMALS,
        "max_seconds": MAX_SECONDS, "independent_target_observations": len(MS) * B,
        "proposal_banks": len(MS) * len(ARMS) * B, "new_risk_observations": 0,
        "label_role": "conditionally_enumerated_Y_given_Xstar_population_evaluation_only",
        "teacher_latent_learning_visible": False, "binomial_internal_uniforms": "unknown_not_counted",
        "zero_optimization_updates": True, "old_H3_role": "development_and_existing_risk_scale_only",
        "strict_positive_joint_risk": "analytic: differing Y marginal; joint KL dominates marginal KL",
        "external_library_modules": {"numpy": np.__file__, "scipy": scipy.__file__}}
    write_new_json(out / "OLD_RISK_BINDING.json", binding)
    write_new_json(out / "FREEZE.json", contract)
    return source, hashes, binding


def load_frozen_dependencies(source, hashes):
    """Score with copied local modules, not changing live workspace sources."""
    global finite, legacy
    for relative, expected in hashes.items():
        if digest(source / relative) != expected:
            raise RuntimeError("Frozen source digest mismatch: " + relative)
    script_dir = source / "script"
    sys.path.insert(0, str(script_dir))
    for name in LOCAL_MODULES:
        sys.modules.pop(name, None)
    for name in LOCAL_MODULES:
        module = importlib.import_module(name)
        if Path(module.__file__).resolve() != (script_dir / (name + ".py")).resolve():
            raise RuntimeError("Dependency did not load from frozen source: " + name)
    finite = sys.modules["selection_finite_core"]
    legacy = sys.modules["signal_proposal_core"]


def create_output(out):
    out.mkdir(parents=True, exist_ok=False)


def run():
    create_output(OUT)  # Refuse overwrite before constructing any scientific RNG.
    started, ledger = time.monotonic(), new_ledger()
    contexts, primary, completed = [], [], []
    try:
        source, hashes, binding = freeze(OUT)
        load_frozen_dependencies(source, hashes)
        for M in MS:
            context_start = ledger.copy()
            folder = OUT / f"M{M}"
            folder.mkdir()
            critical = finite.critical_parameters(M, a=[1.] * 3, tau=TAU, gamma=GAMMA)
            parameters = {}
            for arm in ARMS:
                beta = 1. if arm == "beta1" else critical["beta"]
                normalizer = finite.proposal_log_normalizer(beta, tau=TAU, order=192)
                parameters[arm] = {"M": M, "epsilon": critical["epsilon"], "tau": TAU,
                    "beta": beta, "nuisance_log_constant": 3 * M * normalizer["log_Z_beta"]}
                write_new_json(folder / (arm + "_PARAMETERS.json"), {
                    **parameters[arm], "a": [1.] * 3, "gamma": 0. if arm == "beta1" else GAMMA,
                    "proposal_normalizer": normalizer, "full_absolute_ordered_observation_loss": True})
            rng = role_rngs(M, ledger)
            pieces = {arm: {} for arm in ARMS}
            checks = {arm: {} for arm in ARMS}
            paired_pieces, events = [], []
            h_check = None
            for block, start in enumerate(range(0, B, CHUNK)):
                budget_check(ledger, started)
                size = min(CHUNK, B - start)
                blockdir = folder / f"block{block:04d}"
                blockdir.mkdir()
                teacher_z = rng["teacher_z"].standard_normal((size, 3))
                observation = finite.sample_teacher_observations(rng["counts"], teacher_z,
                    M=M, epsilon=critical["epsilon"], tau=TAU, inner_order=192)
                shared = {"teacher_z": teacher_z, "Xstar": teacher_z.copy(),
                    "z": rng["z"].standard_normal((size, K, 3)),
                    "counts": observation["counts"], "q": observation["label_probs"]}
                ledger["independent_target_observations"] += size
                shared_record = save_new_npz(blockdir / "SHARED_INPUTS.npz", shared)
                write_new_json(blockdir / "OBSERVATION_SAMPLING.json", {
                    "sampling": observation["sampling"], "inputs": shared_record,
                    "shared_by_both_arms": True, "ledger_after_shared_sampling": ledger.copy(),
                    "binomial_internal_uniforms": "unknown_not_counted"})
                if block == 0:
                    low_h = finite.h_probabilities(teacher_z * critical["epsilon"], tau=TAU, inner_order=128)
                    h_check = {key: order_diagnostic(low_h[key] - observation[
                        "count_success_probability" if key == "h" else "count_failure_probability"])
                        for key in ("h", "complement")}
                    write_new_json(folder / "H_128_192_CHECK.json", {"teacher_banks": size, "fields": h_check})
                private_arrays, private_records = {}, {}
                for arm in ARMS:
                    before = ledger.copy()
                    try:
                        draws, receipt = finite.sample_oriented_qbeta(rng["private_" + arm],
                            (size, K, 3, M), tau=TAU, beta=parameters[arm]["beta"],
                            proposal_batch_size=1000000, max_proposals=MAX_NORMALS - ledger["normal_scalars"])
                    except BaseException as exc:
                        match = re.search(r"proposed=(\d+), accepted=(\d+), used=(\d+)", str(exc))
                        write_new_json(blockdir / (arm + "_PRIVATE_SAMPLING_FAILURE.json"), {
                            "error": repr(exc), "before": before, "after": ledger.copy(),
                            "shared_inputs": shared_record, "sampler_partial_counts":
                            dict(zip(("proposed", "accepted", "used"), map(int, match.groups()))) if match else None})
                        raise
                    for key in ("accepted", "used", "discarded_accepted"):
                        ledger[arm + "_private_" + key] += receipt[key]
                    ledger["proposal_banks"] += size
                    write_new_json(blockdir / (arm + "_PRIVATE_SAMPLING.json"), {
                        "M": M, "arm": arm, "start": start, "banks": size, "sampler": receipt,
                        "before_any_particle_scoring": True, "ledger_after_sampling": ledger.copy()})
                    private_records[arm] = save_new_npz(blockdir / (arm + "_PRIVATE_INPUTS.npz"), {"T": draws})
                    private_arrays[arm] = draws
                values, score_records = {}, {}
                for arm in ARMS:
                    values[arm] = score_block({**shared, "T": private_arrays[arm]}, parameters[arm],
                        ledger, check=start < CHECK_B, started=started)
                    score_records[arm] = save_new_npz(blockdir / (arm + "_SCORES.npz"), values[arm])
                    for key, value in values[arm].items():
                        (checks[arm] if "_12" in key else pieces[arm]).setdefault(key, []).append(value)
                paired = values["critical"]["path_price_gap"] - values["beta1"]["path_price_gap"]
                paired_pieces.append(paired)
                pair_record = save_new_npz(blockdir / "PAIRED_SCORES.npz", {"critical_minus_beta1": paired})
                event = {"block": block, "start": start, "banks": size, "shared_inputs": shared_record,
                    "private_inputs": private_records, "scores": score_records, "paired_scores": pair_record,
                    "ledger": ledger.copy(), "seconds": budget_check(ledger, started)}
                write_new_json(blockdir / "COMPLETE.json", event)
                events.append(event)
                print(json.dumps({"M": M, "block": block, "seconds": event["seconds"],
                    "normal_scalars": ledger["normal_scalars"], "bank_score_calls": ledger["bank_score_calls"]}), flush=True)
            arms = {}
            for arm in ARMS:
                merged = {key: np.concatenate(value) for key, value in pieces[arm].items()}
                checked = {key: np.concatenate(value) for key, value in checks[arm].items()}
                diagnostic = order_diagnostic(checked["path_price_gap_12_minus_8"])
                write_new_json(folder / (arm + "_PATH_8_12_CHECK.json"), {
                    "banks": CHECK_B, "price_gap": diagnostic,
                    "all_check_statistics": {key: stats(value) for key, value in checked.items()},
                    "all_check_max_absolute": {key: float(np.max(abs(value))) for key, value in checked.items() if "minus" in key}})
                summaries = {key: stats(value) for key, value in merged.items()}
                numerical_passed = diagnostic["passed"] and all(item["passed"] for item in h_check.values())
                row = family_row(M, arm, summaries["path_price_gap"], numerical_passed)
                primary.append(row)
                arms[arm] = {"parameters": parameters[arm], "statistics": summaries,
                    "path_check": diagnostic, "primary": row}
            paired_summary = stats(np.concatenate(paired_pieces))
            context = {"M": M, "independent_target_observations": B, "proposal_banks": 2 * B,
                "arms": arms, "paired_critical_minus_beta1": paired_summary, "h_check": h_check,
                "old_same_model_risk_binding": binding[str(M)], "new_risk_observations": 0,
                "actual_context_counts": {key: ledger[key] - context_start[key] for key in ledger},
                "input_blocks": events, "seconds_cumulative": budget_check(ledger, started)}
            write_new_json(folder / "RESULTS.json", context)
            contexts.append(context)
            completed.append(M)
            write_new_json(OUT / f"STATUS_M{M}.json", {"completed_M": completed,
                "ledger": ledger.copy(), "seconds": budget_check(ledger, started)})
        if (ledger["bank_score_calls"] != MAX_CALLS or
                ledger["independent_target_observations"] != len(MS) * B or
                ledger["proposal_banks"] != len(MS) * len(ARMS) * B):
            raise RuntimeError("Completed counts disagree with frozen J1 contract")
        write_new_json(OUT / "RESULTS.json", {"schema": "remainder_finite_reversal_J1_results_v1",
            "contexts": contexts, "primary": primary, "family_size": 4, "family_alpha": .05,
            "all_four_signs_supported": all(row["qualified_sign_supported"] for row in primary),
            "ledger": ledger, "seconds": budget_check(ledger, started), "zero_optimization_updates": True,
            "qualification": "normal Bonferroni approximations; ordinary GL/GH diagnostics; fixed two-M panel",
            "paired_shift": "descriptive; bankwise paired variance, no added primary tests",
            "old_risk_role": "existing same-model risk scale only; no new risk evaluation"})
    except BaseException as exc:
        write_new_json(OUT / "FAILURE.json", {"error": repr(exc), "completed_M": completed,
            "ledger": ledger, "seconds": time.monotonic() - started, "failed_evidence_retained": True})
        raise


def selftest():
    """Pure arithmetic/fake generators only; no random generator is constructed."""
    size = 4
    inputs = {"z": np.sin(np.arange(size * K * 3)).reshape(size, K, 3),
        "T": np.zeros((size, K, 3, 32)), "counts": np.full((size, 3), 16, dtype=int),
        "q": np.full((size, 8), 1 / 8)}
    params = {"M": 32, "epsilon": [1 / math.sqrt(32)] * 3, "tau": TAU,
        "beta": .2, "nuisance_log_constant": -5.}

    def scalar_scorer(z, T, counts, C, **kwargs):
        u = C[0, 1]
        field = np.full((len(z), 3), 2 * u)
        return {"loss": np.full(len(z), 3 * u * u), "shifted_loss": np.full(len(z), 3 * u * u),
            "controlled_loss": np.full(len(z), 3 * u ** 3), "shifted_controlled_loss": np.full(len(z), 3 * u ** 3),
            "price": field + .1, "raw_rp": field, "cv_rp": np.full_like(field, 3 * u * u),
            "overlap": np.ones(len(z)), "ess": np.ones(len(z))}

    ledger = new_ledger()
    result = score_block(inputs, params, ledger, check=True, scorer=scalar_scorer)
    np.testing.assert_allclose(result["path_price_gap"], 3 * PATH_END ** 2, atol=4e-16, rtol=0)
    for key in ("path_raw_minus_direct", "path_cv_minus_direct"):
        np.testing.assert_allclose(result[key], 0, atol=4e-16, rtol=0)
    assert ledger["bank_score_calls"] == size * 22
    assert order_diagnostic(result["path_price_gap_12_minus_8"])["passed"]
    # Both arms vary, but their paired shift is a constant: independent-SE addition would be wrong.
    beta1 = np.asarray([1., 2., 4., 8.])
    paired = paired_statistics(beta1, beta1 + 3.)
    assert paired["mean"] == 3. and paired["variance"] == 0. and paired["se"] == 0.
    assert stats(beta1)["variance"] > 0
    assert family_row(32, "beta1", stats(np.ones(size)), True)["sign_supported"]
    assert family_row(32, "critical", stats(-np.ones(size)), True)["sign_supported"]

    class FakeGenerator:
        def standard_normal(self, shape):
            return np.zeros(shape)

        def random(self, shape):
            return np.full(shape, .25)

        def binomial(self, n, p):
            return np.full(np.shape(p), n // 2, dtype=int)

    sampling = new_ledger()
    teacher_rng = CountedRNG(FakeGenerator(), sampling, "teacher_z")
    teacher_rng.standard_normal((2, 3))
    teacher_rng.binomial(32, np.full((2, 3), .5))
    private_rng = CountedRNG(FakeGenerator(), sampling, "private_beta1")
    draw, receipt = finite.sample_oriented_qbeta(private_rng, (2, 3), tau=TAU,
        beta=1., proposal_batch_size=32, max_proposals=100)
    for key in ("accepted", "used", "discarded_accepted"):
        sampling["beta1_private_" + key] += receipt[key]
    assert sampling["normal_scalars"] == 6 + receipt["proposed"]
    assert sampling["beta1_private_proposed"] == receipt["proposed"]
    assert sampling["explicit_rejection_uniforms"] == receipt["uniforms_generated"]
    assert sampling["binomial_variates"] == 6 and receipt["used"] == draw.size
    assert sampling["critical_private_proposed"] == 0
    assert sampling["shared_teacher_active_normal_scalars"] == 6
    for key, limit in (("normal_scalars", MAX_NORMALS), ("bank_score_calls", MAX_CALLS)):
        exhausted = new_ledger()
        exhausted[key] = limit + 1
        try:
            budget_check(exhausted, time.monotonic())
        except RuntimeError:
            pass
        else:
            raise AssertionError("Budget was not enforced")
    sampling["normal_scalars"] = MAX_NORMALS
    try:
        private_rng.standard_normal(1)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Gaussian budget precheck was not enforced")
    score_exhausted = new_ledger()
    score_exhausted["bank_score_calls"] = MAX_CALLS
    try:
        score_block(inputs, params, score_exhausted, scorer=scalar_scorer)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Score budget precheck was not enforced")
    try:
        budget_check(new_ledger(), time.monotonic() - MAX_SECONDS - 1)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Wall budget was not enforced")
    with tempfile.TemporaryDirectory(prefix="remainder_j1_selftest_") as name:
        folder = Path(name) / "attempt"
        create_output(folder)
        write_new_json(folder / "receipt.json", {"test": True})
        save_new_npz(folder / "inputs.npz", inputs)
        for action in (lambda: create_output(folder), lambda: write_new_json(folder / "receipt.json", {}),
                       lambda: save_new_npz(folder / "inputs.npz", {})):
            try:
                action()
            except FileExistsError:
                pass
            else:
                raise AssertionError("Overwrite was permitted")
    errors = {}
    for arm, beta in (("beta1", 1.), ("critical", finite.critical_parameters(32, a=[1.] * 3, tau=TAU, gamma=GAMMA)["beta"])):
        actual = score_block(inputs, {**params, "beta": beta}, new_ledger(), check=True)
        for key in ("path_raw_minus_direct", "path_cv_minus_direct"):
            np.testing.assert_allclose(actual[key], 0, atol=2e-7, rtol=0)
            errors[arm + "_" + key] = float(np.max(abs(actual[key])))
    with tempfile.TemporaryDirectory(prefix="remainder_j1_freeze_selftest_") as name:
        folder = Path(name)
        source, hashes, binding = freeze(folder)
        load_frozen_dependencies(source, hashes)
        assert set(binding) == {"32", "128"}
        assert all(item["role"] == "old_risk_scale_only_not_new_validation" for item in binding.values())
        assert read(folder / "FREEZE.json")["new_risk_observations"] == 0
        frozen_score = finite.full_banks(inputs["z"], inputs["T"], inputs["counts"], np.eye(3),
            label_probs=inputs["q"], **params)
        assert np.isfinite(frozen_score["price"]).all()
    return {"passed": True, "new_scientific_RNG": 0, "paired_constant_shift_statistics": native(paired),
        "budgets_and_overwrite_refusal_checked": True, "analytic_scorer_calls": size * 22,
        "finite_actual_two_arm_path_errors": errors,
        "frozen_dependency_loading_and_old_risk_binding_checked": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--run", action="store_true", help="Execute only frozen J1 attempt1")
    modes.add_argument("--selftest", action="store_true", help="Arithmetic fixtures; no scientific RNG")
    args = parser.parse_args()
    if args.selftest:
        print(json.dumps(selftest(), indent=2))
    else:
        run()


if __name__ == "__main__":
    main()
