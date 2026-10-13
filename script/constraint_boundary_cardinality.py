"""Descriptive GoEmotions cardinality and centered/uncentered residual moments.

No test inputs, parameter fits, scientific effect gates, or causal claims.
The first JSON write freezes the plan and input/source hashes before arithmetic;
the final JSON retains that frozen record and its exact initial-file SHA256.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import importlib.metadata
import json
import sys

import numpy as np
from scipy.special import expit


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/constraint_boundary_v2"
OUT = BASE / "MOTIVATION_CARDINALITY.json"
CSV = BASE / "MOTIVATION_CARDINALITY.csv"
GDATA = ROOT / "artifacts/dependence_boundary_v1/GDATA"
ROLES = ["train", "association_fit"]
HEADS = ["word", "word_char"]
QUANTILES = [0., .05, .25, .5, .75, .95, 1.]
PLAN = dict(
    purpose="Describe full 27-label cardinality and residual aggregation; motivate only whether pair scoring omits global label-count structure.",
    roles=ROLES, heads=HEADS, raters=[0, 1, 2], d=27,
    labels="Existing three hash-selected clear-reader binary emotion vectors, excluding neutral; no rater-style causal interpretation.",
    label_statistics="Count frequencies 0..27, empirical mean/variance(ddof0), sum marginal variances, full label covariance/correlation and correlation spectrum, every valid off-diagonal absolute correlation and fixed quantiles.",
    residual_statistics="For p=sigmoid(existing unscaled logits), r=Y-p: mean r, M=mean rrT, C=mean((r-mean r)(r-mean r)T); total residual mean/variance/second moment, trace M, trace C, predicted count variance and label-count/predicted-count covariance.",
    matrix_contract="Covariance and variance use ddof0 throughout. Undefined correlations for zero-variance columns are null; their columns are excluded from correlation eigenvalues and pair summaries. M is never called centered covariance.",
    retention="All 6 role/rater label descriptions and all 12 head/role/rater residual descriptions retained; no effects tested or selected.",
    validation="Algebraic moment identities are checked numerically at absolute tolerance 1e-10; this is a software check, not an empirical effect gate.",
    forbidden="No test read, scoring, training, hyperparameter search, scientific effect threshold, or inference that count is caused by annotator style.",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def encoded(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def input_paths():
    paths = [GDATA / "DATA_AUDIT.json", GDATA / "DATA_SEAL.json", GDATA / "PREDICTION_SEAL.json"]
    paths += [GDATA / "cache" / f"{role}.npz" for role in ROLES]
    paths += [GDATA / "cache" / f"{head}_{role}_means.npz" for head in HEADS for role in ROLES]
    return paths


def verify_prior_seals():
    for name in ["DATA_SEAL.json", "PREDICTION_SEAL.json"]:
        for path, digest in read(GDATA / name)["files"].items():
            assert sha(ROOT / path) == digest, path


def freeze():
    assert not OUT.exists() and not CSV.exists(), "Do not overwrite an existing description"
    verify_prior_seals()
    frozen = dict(
        time_utc=now(), plan=PLAN,
        input_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in input_paths()},
        source_sha256={Path(__file__).relative_to(ROOT).as_posix(): sha(__file__)},
    )
    BASE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(encoded(dict(status="frozen_before_descriptive_arithmetic", freeze=frozen)), encoding="utf-8", newline="\n")


def verify_freeze(frozen):
    assert frozen["plan"] == PLAN
    for mapping in [frozen["input_sha256"], frozen["source_sha256"]]:
        for path, digest in mapping.items():
            assert sha(ROOT / path) == digest, path


def correlation_description(covariance):
    variances = np.diag(covariance)
    active = np.flatnonzero(variances > 0)
    inactive = np.flatnonzero(variances <= 0)
    correlation = np.full_like(covariance, np.nan)
    inverse = 1 / np.sqrt(variances[active])
    subset = covariance[np.ix_(active, active)] * inverse[:, None] * inverse[None, :]
    correlation[np.ix_(active, active)] = subset
    eigenvalues = np.linalg.eigvalsh(subset)
    i, j = np.triu_indices(len(active), 1)
    pairs = subset[i, j]
    absolute = np.abs(pairs)
    quantiles = {str(q): float(v) for q, v in zip(QUANTILES, np.quantile(absolute, QUANTILES))} if len(pairs) else {}
    return dict(
        active_indices=active.tolist(), undefined_correlation_indices=inactive.tolist(),
        correlation=[[float(v) if np.isfinite(v) else None for v in row] for row in correlation],
        eigenvalues_ascending=eigenvalues.tolist(),
        pair_indices=[[int(active[a]), int(active[b])] for a, b in zip(i, j)],
        signed_pair_correlations=pairs.tolist(), absolute_pair_correlations_sorted=np.sort(absolute).tolist(),
        pair_absolute_quantiles=quantiles, valid_pairs=len(pairs),
        pair_absolute_mean=float(absolute.mean()) if len(pairs) else None,
        pair_absolute_population_sd=float(absolute.std()) if len(pairs) else None,
    )


def label_description(labels):
    assert labels.ndim == 2 and labels.shape[1] == 27 and np.isin(labels, [0., 1.]).all()
    n = len(labels); mean = labels.mean(0); count = labels.sum(1)
    centered = labels - mean; covariance = centered.T @ centered / n
    marginal_variances = mean * (1 - mean)
    histogram = np.bincount(count.astype(int), minlength=28)
    variance = float(count.var())
    errors = dict(
        Bernoulli_marginal_variance_max_error=float(np.max(abs(marginal_variances - np.diag(covariance)))),
        count_variance_covariance_sum_error=float(abs(variance - covariance.sum())),
        count_mean_marginal_sum_error=float(abs(count.mean() - mean.sum())),
    )
    assert max(errors.values()) < 1e-10 and histogram.sum() == n
    return dict(
        n=n, count_frequency_0_to_27=histogram.tolist(), count_probability_0_to_27=(histogram / n).tolist(),
        count_mean=float(count.mean()), count_population_variance=variance,
        observed_count_min=int(count.min()), observed_count_max=int(count.max()),
        marginal_means=mean.tolist(), marginal_variances=marginal_variances.tolist(),
        sum_marginal_variances=float(marginal_variances.sum()), covariance=covariance.tolist(),
        sum_offdiagonal_covariances=float(covariance.sum() - np.trace(covariance)),
        correlation=correlation_description(covariance), numerical_checks=errors,
    )


def residual_description(labels, predictions):
    residual = labels - predictions; mean = residual.mean(0)
    M = residual.T @ residual / len(residual)
    centered = residual - mean; C = centered.T @ centered / len(residual)
    residual_count = residual.sum(1); label_count = labels.sum(1); prediction_count = predictions.sum(1)
    cross_count_covariance = float(np.mean((label_count - label_count.mean()) * (prediction_count - prediction_count.mean())))
    trace_M = float(np.trace(M)); trace_C = float(np.trace(C))
    count_variance = float(residual_count.var()); count_second = float(np.mean(residual_count**2))
    errors = dict(
        uncentered_minus_centered_plus_mean_outer_max_error=float(np.max(abs(M - C - np.outer(mean, mean)))),
        residual_count_variance_covariance_sum_error=float(abs(count_variance - C.sum())),
        residual_count_second_moment_M_sum_error=float(abs(count_second - M.sum())),
        sum_squared_residuals_trace_M_error=float(abs(np.mean(np.sum(residual**2, axis=1)) - trace_M)),
        residual_count_variance_difference_identity_error=float(abs(count_variance - label_count.var() - prediction_count.var() + 2 * cross_count_covariance)),
    )
    assert max(errors.values()) < 1e-10
    return dict(
        n=len(residual), mean_residual_vector=mean.tolist(), uncentered_second_moment_M=M.tolist(),
        centered_covariance_C=C.tolist(), M_eigenvalues_ascending=np.linalg.eigvalsh(M).tolist(),
        C_eigenvalues_ascending=np.linalg.eigvalsh(C).tolist(),
        label_count_mean=float(label_count.mean()), label_count_variance=float(label_count.var()),
        prediction_count_mean=float(prediction_count.mean()), prediction_count_variance=float(prediction_count.var()),
        label_count_prediction_count_covariance=cross_count_covariance,
        residual_count_mean=float(residual_count.mean()), residual_count_variance=count_variance,
        residual_count_second_moment=count_second, mean_sum_squared_residuals=trace_M,
        sum_residual_marginal_variances=trace_C, squared_norm_mean_residual=float(mean @ mean),
        sum_M_offdiagonal_entries=float(M.sum() - trace_M), sum_C_offdiagonal_entries=float(C.sum() - trace_C),
        numerical_checks=errors,
    )


def calculate():
    document = read(OUT); assert document["status"] == "frozen_before_descriptive_arithmetic" and not CSV.exists()
    frozen = document["freeze"]; verify_freeze(frozen); initial_sha = sha(OUT)
    label_results = []; residual_results = []; rows = []
    def row(domain, role, rater, head, metric, index, value):
        rows.append(dict(domain=domain, role=role, rater=rater, head=head, metric=metric, index=index, value=value))
    for role in ROLES:
        with np.load(GDATA / "cache" / f"{role}.npz") as archive:
            ids = archive["id"]; labels = archive["labels"]
        assert labels.shape == (len(ids), 3, 27)
        for rater in range(3):
            description = label_description(labels[:, rater]); label_results.append(dict(role=role, rater=rater, **description))
            for metric, value in description.items():
                if isinstance(value, (float, int)): row("labels", role, rater, "", metric, "", value)
            for index, value in enumerate(description["count_frequency_0_to_27"]): row("labels", role, rater, "", "count_frequency", index, value)
            for index, value in enumerate(description["count_probability_0_to_27"]): row("labels", role, rater, "", "count_probability", index, value)
            corr = description["correlation"]
            for index, value in enumerate(corr["eigenvalues_ascending"]): row("labels", role, rater, "", "correlation_eigenvalue", index, value)
            for index, value in corr["pair_absolute_quantiles"].items(): row("labels", role, rater, "", "pair_absolute_correlation_quantile", index, value)
            for index, value in enumerate(corr["absolute_pair_correlations_sorted"]): row("labels", role, rater, "", "absolute_pair_correlation_order_statistic", index, value)
        for head in HEADS:
            with np.load(GDATA / "cache" / f"{head}_{role}_means.npz") as archive:
                assert np.array_equal(archive["ids"], ids); predictions = expit(archive["means"])
            assert predictions.shape == (len(ids), 27) and np.isfinite(predictions).all()
            for rater in range(3):
                description = residual_description(labels[:, rater], predictions)
                residual_results.append(dict(head=head, role=role, rater=rater, **description))
                for metric, value in description.items():
                    if isinstance(value, (float, int)): row("residuals", role, rater, head, metric, "", value)
    assert len(label_results) == 6 and len(residual_results) == 12
    with CSV.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    verify_freeze(frozen)
    original_document = dict(status="frozen_before_descriptive_arithmetic", freeze=frozen)
    assert hashlib.sha256(encoded(original_document).encode()).hexdigest() == initial_sha
    final = dict(
        status="complete_descriptive_cardinality_and_residual_moments", freeze=frozen,
        original_freeze_document_sha256=initial_sha, completed_at_utc=now(),
        concepts=read(GDATA / "DATA_AUDIT.json")["concepts"], label_results=label_results, residual_results=residual_results,
        CSV_sha256=sha(CSV), CSV_rows=len(rows),
        numerical_audit_max_error=max(max(x["numerical_checks"].values()) for x in label_results + residual_results),
        environment=dict(python=sys.version, packages={k: importlib.metadata.version(k) for k in ["numpy", "scipy"]}),
        new_fits=0, test_access=False, scientific_effect_thresholds=[], annotator_style_causal_claim=False,
    )
    OUT.write_text(encoded(final), encoding="utf-8", newline="\n")
    print(dict(status=final["status"], label_combinations=6, residual_combinations=12, CSV_rows=len(rows), maximum_numerical_error=final["numerical_audit_max_error"]))


def check():
    document = read(OUT); verify_freeze(document["freeze"])
    assert document["status"].startswith("complete_") and sha(CSV) == document["CSV_sha256"]
    original = dict(status="frozen_before_descriptive_arithmetic", freeze=document["freeze"])
    assert hashlib.sha256(encoded(original).encode()).hexdigest() == document["original_freeze_document_sha256"]
    assert len(document["label_results"]) == 6 and len(document["residual_results"]) == 12
    print("pass_descriptive_cardinality_plan_inputs_source_moment_checks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("phase", choices=["freeze", "calculate", "check"])
    args = parser.parse_args()
    if args.phase == "calculate":
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2): calculate()
    else: globals()[args.phase]()
