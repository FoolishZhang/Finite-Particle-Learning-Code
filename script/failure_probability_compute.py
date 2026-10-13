"""Eight-label n256 point differences; manufactured qualification / development only.

No RNG, new confidence family, optimizer, or write to an old scientific package.
Reading old saved banks requires a separate explicit CLI stage flag.
"""
from pathlib import Path
from datetime import datetime, timezone
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import hashlib
import json
import math
import sys
import time
import traceback
import numpy as np
from scipy.special import logsumexp
import nuisance_contrast_compute as legacy
from flint import arb, ctx

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'artifacts/failure_probability_v1_20261006'
CONTRACT = ROOT / 'docs/08_research_研究工作/FailureProbability_v1_20261006/DEVELOPMENT_CONTRACT.md'
QUESTIONS = ROOT / 'docs/00_project_项目管理/RESEARCH_QUESTIONS_1_2_20261006.md'
DEFAULT_SOURCE = ROOT / 'artifacts/nuisance_contrast_v1_20261006/EVALUATION_BANKS.npz'
DEFAULT_REFERENCE = ROOT / 'artifacts/nuisance_contrast_v1_20261006/RESULTS.json'
YS = legacy.YS.copy()
TOLERANCE = 1e-11
BLOCK = 256
N_DATA = 65536
B_SELECT = 16384
MAX_SECONDS = 1800
DEVELOPMENT_THRESHOLDS = (5.5, 6., 6.5, 8.)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def native(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [native(v) for v in value]
    return value


def save_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(native(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def save_npz_new(path, **arrays):
    with Path(path).open('xb') as handle:
        np.savez(handle, **arrays)


def check_output(path):
    path = Path(path).resolve()
    assert path == DEFAULT_OUT or DEFAULT_OUT in path.parents, 'Output must stay inside the new development package'
    path.mkdir(parents=True, exist_ok=True)
    return path


def point_differences(z, s):
    """Return B x 8 score(I)-score(Cb), n256 only, in original label order.

    Preserve the old matrix root, active log-likelihood identity and centered
    logweights.  The same Z and S enter both anchors, with no symmetrization.
    """
    z = np.asarray(z, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    assert z.ndim == 3 and z.shape[2] == 3 and s.shape == z.shape[:2]
    assert len(z) and z.shape[1] and np.isfinite(z).all() and np.isfinite(s).all()
    w = legacy.logweights(s)
    xi = z @ legacy.old.root(np.eye(3))
    xb = z @ legacy.old.root(legacy.CB)
    base_i = -np.logaddexp(xi / 2, -xi / 2).sum(axis=2)
    base_b = -np.logaddexp(xb / 2, -xb / 2).sum(axis=2)
    out = np.empty((len(z), 8), dtype=np.float64)
    for yi, y in enumerate(YS):
        logs_i = base_i + sum(float(y[j]) * .5 * xi[:, :, j] for j in range(3))
        logs_b = base_b + sum(float(y[j]) * .5 * xb[:, :, j] for j in range(3))
        out[:, yi] = logsumexp(w + logs_b, axis=1) - logsumexp(w + logs_i, axis=1)
    assert np.isfinite(out).all()
    return out


def audit_single_bank(z, s, values=None):
    """128-bit one-hot Arb audit of all 8 labels at one given real bank.

    This reusable interface reads no files and writes no artifacts.  Inputs are
    exact saved binary64 values; old interval_pair adds root residual allowance.
    """
    ctx.prec = 128
    z, s = np.asarray(z, dtype=np.float64), np.asarray(s, dtype=np.float64)
    assert z.ndim == 2 and z.shape[1] == 3 and s.shape == (len(z),)
    if values is None:
        values = point_differences(z[None], s[None])[0]
    assert np.shape(values) == (8,)
    rows = []
    for yi, y in enumerate(YS):
        counts = np.zeros(8, dtype=np.int64)
        counts[yi] = 1
        vi = legacy.interval_pair((0, 0, 0), z[None], s[None], counts)[0][1]
        vb = legacy.interval_pair((9, 9, 9), z[None], s[None], counts)[0][1]
        difference = vi - vb
        error = abs(arb(float(values[yi])) - difference)
        upper = legacy.old.rounded_upper(error)
        rows.append(dict(label_index=yi, label=y.tolist(), binary64=float(values[yi]),
                         interval=legacy.old.ball(difference), absolute_error_upper=upper,
                         passed=upper <= TOLERANCE))
    assert all(row['passed'] for row in rows), rows
    return dict(bits=128, rows=rows, max_error_upper=max(row['absolute_error_upper'] for row in rows),
                scope='given binary64 Z/S plus matrix-root allowance; not RNG or whole-pipeline certification')


def prepare(out, source, reference):
    assert not (out / 'DEVELOPMENT_INPUTS.json').exists(), 'No silent prepare overwrite'
    assert source.resolve() == DEFAULT_SOURCE.resolve(), 'Current contract allows only the named old evaluation bank'
    assert reference.resolve() == DEFAULT_REFERENCE.resolve()
    snapshot = out / 'HANDOFF_INPUT.md'
    if snapshot.exists():
        assert sha(snapshot) == sha(ROOT / 'HANDOFF.md'), 'Existing HANDOFF snapshot differs; do not overwrite'
    else:
        with snapshot.open('xb') as handle:
            handle.write((ROOT / 'HANDOFF.md').read_bytes())
    paths = [Path(__file__), CONTRACT, QUESTIONS, Path(legacy.__file__), Path(legacy.old.__file__), reference, snapshot]
    source_record = read(reference)['arrays'][rel(source)]
    save_new(out / 'DEVELOPMENT_INPUTS.json', dict(
        created_utc=now(), stage='manufactured_qualification_before_any_scientific_bank_read',
        sources={rel(p): sha(p) for p in paths}, source_array=rel(source),
        expected_source_array_sha256=source_record['sha256'], expected_source_array_bytes=source_record['bytes'],
        reference_results=rel(reference), handoff_sha256=sha(snapshot),
        reviewed_commit='b06ee9b039236a20734cf8a86dc2bd4a09feae7e',
        feedback_use='reference; implementation and post-hoc development diagnostics only',
        label_order=YS, tolerance=TOLERANCE, block=BLOCK, N_data=N_DATA, B_selection=B_SELECT,
        development_truncation_thresholds=DEVELOPMENT_THRESHOLDS,
        old_scientific_array_opened=False, new_rng=0, new_confidence_family=False,
        probability_guarantee_claimed=False))
    print(json.dumps(dict(prepared=True, output=rel(out), scientific_array_opened=False,
                          code_sha256=sha(__file__), contract_sha256=sha(CONTRACT))), flush=True)


def verify_inputs(out):
    inputs = read(out / 'DEVELOPMENT_INPUTS.json')
    for path, digest in inputs['sources'].items():
        assert sha(ROOT / path) == digest, ('Development input changed', path)
    assert inputs['label_order'] == YS.tolist()
    assert inputs['new_rng'] == 0 and inputs['new_confidence_family'] is False
    return inputs


def manufactured_fixture(k):
    index = np.arange(6 * k * 3, dtype=np.float64)
    z = (np.sin(.37 * index) + .15 * np.cos(.11 * index)).reshape(6, k, 3)
    common = np.array([-1000., 1000., .3, -.4, 0., 55.])[:, None]
    relative = np.linspace(-7., 3., 6 * k).reshape(6, k)
    return z, relative + common, common


def qualify(out):
    verify_inputs(out)
    assert not (out / 'MANUFACTURED_QUALIFICATION.json').exists(), 'No silent qualification repeat'
    started = time.monotonic()
    q_cases = [('uniform', np.ones(8) / 8), ('increasing', np.arange(1, 9) / 36),
               ('asymmetric', np.array([2, 3, 1, 5, 4, 2, 7, 6]) / 30)]
    q_cases += [(f'onehot_{i}', np.eye(8)[i]) for i in range(8)]
    comparisons, audits, arrays = [], [], {'label_order': YS}
    for k in (2, 5, 100):
        z, s, common = manufactured_fixture(k)
        differences = point_differences(z, s)
        arrays.update({f'z_k{k}': z, f's_k{k}': s, f'differences_k{k}': differences})
        for name, q in q_cases:
            actual = differences @ q
            score_pair = legacy.score_pair(np.eye(3), z, s, q)[:, 1] - legacy.score_pair(legacy.CB, z, s, q)[:, 1]
            direct = legacy.direct(np.eye(3), z, s, q)[:, 1] - legacy.direct(legacy.CB, z, s, q)[:, 1]
            row = dict(K=k, q_name=name, q=q, max_score_pair_error=float(np.max(np.abs(actual - score_pair))),
                       max_direct_error=float(np.max(np.abs(actual - direct))))
            assert max(row['max_score_pair_error'], row['max_direct_error']) <= TOLERANCE, row
            comparisons.append(row)
        shift_error = float(np.max(np.abs(differences - point_differences(z, s - common))))
        assert shift_error <= TOLERANCE
        for bi in (0, 1):
            audit = audit_single_bank(z[bi], s[bi], differences[bi])
            audits.append(dict(K=k, bank=bi, **audit))
        arrays[f'common_shift_error_k{k}'] = np.array(shift_error)
        print(json.dumps(dict(stage='manufactured', K=k, banks=6, label_differences=48,
                              arb_banks=2, common_shift_error=shift_error)), flush=True)
    array = out / 'MANUFACTURED_POINT_DIFFERENCES.npz'
    save_npz_new(array, **arrays)
    result = dict(created_utc=now(), inputs_sha256=sha(out / 'DEVELOPMENT_INPUTS.json'),
                  code_sha256=sha(__file__), contract_sha256=sha(CONTRACT), manufactured_only=True,
                  passed=True, tolerance=TOLERANCE, K_cases=[2, 5, 100], banks_per_case=6,
                  q_cases=len(q_cases), comparisons=comparisons, arb_audits=audits,
                  max_score_pair_error=max(x['max_score_pair_error'] for x in comparisons),
                  max_direct_error=max(x['max_direct_error'] for x in comparisons),
                  max_arb_error_upper=max(x['max_error_upper'] for x in audits),
                  arb_bits=128, arb_bank_label_differences=len(audits) * 8,
                  array=dict(path=rel(array), sha256=sha(array), bytes=array.stat().st_size),
                  new_rng=0, scientific_array_opened=False, new_observations=0,
                  new_prior_vectors=0, confidence_statement=False, algorithm_probability_bound=False,
                  wall_seconds=time.monotonic() - started)
    save_new(out / 'MANUFACTURED_QUALIFICATION.json', result)
    print(json.dumps(dict(qualified=True, manufactured_only=True,
                          max_score_pair_error=result['max_score_pair_error'],
                          max_direct_error=result['max_direct_error'], max_arb_error_upper=result['max_arb_error_upper'],
                          wall_seconds=result['wall_seconds'], scientific_array_opened=False)), flush=True)


def descriptive_statistics(differences, g_lower, n_data=N_DATA, b_selection=B_SELECT):
    """Empirical two-way ANOVA, explicitly not true-variance upper bounds."""
    d = np.asarray(differences, dtype=np.float64)
    assert d.ndim == 2 and d.shape[1] == 8 and len(d) > 1 and np.isfinite(d).all()
    bank_means = d.mean(axis=1)
    label_means = d.mean(axis=0)
    mean = float(bank_means.mean())
    a = label_means - mean
    b = bank_means - mean
    residual = d - bank_means[:, None] - label_means[None, :] + mean
    data_term = float(np.mean(a * a))
    bank_empirical = float(np.mean(b * b))
    bank_sample_variance = float(bank_means.var(ddof=1))
    interaction = float(np.mean(residual * residual))
    total_variance = float(np.mean((d - mean) ** 2))
    decomposition_error = abs(total_variance - data_term - bank_empirical - interaction)
    assert decomposition_error <= 1e-10 * max(1., total_variance)
    same = np.all(YS == YS[:, :1], axis=1)
    assert int(same.sum()) == 2
    same_mean = float(label_means[same].mean())
    split_mean = float(label_means[~same].mean())
    symmetry_data = 3 / 16 * (same_mean - split_mean) ** 2
    assert abs(mean - (.25 * same_mean + .75 * split_mean)) <= TOLERANCE
    variance_parts = dict(data=data_term / n_data, bank=bank_sample_variance / b_selection,
                          interaction=interaction / (n_data * b_selection))
    variance_illustration = sum(variance_parts.values())
    def cantelli(g, variance):
        margin = max(float(g) - 1e-12, 0.)
        return margin * margin / (variance + margin * margin) if variance + margin * margin else 0.
    return dict(banks=len(d), label_order=YS, point_difference_mean=mean, label_means=label_means,
                same_label_indices=np.flatnonzero(same), split_label_indices=np.flatnonzero(~same),
                same_mean=same_mean, split_mean=split_mean,
                bank_mean_sample_variance=bank_sample_variance,
                empirical_E_y_d_squared=float(np.mean(d * d)),
                empirical_ANOVA=dict(data=data_term, bank=bank_empirical, interaction=interaction,
                                    total=total_variance, decomposition_error=decomposition_error,
                                    denominator='empirical product measure: 1/B and 1/8'),
                symmetry_projection=dict(data_variance=symmetry_data,
                                         within_label_class_asymmetry=data_term - symmetry_data,
                                         modifies_bank_inputs=False),
                Cantelli_illustration=dict(N=n_data, B=b_selection, tolerance=1e-12,
                    variance_parts=variance_parts, empirical_variance_sum=variance_illustration,
                    old_g_lower=g_lower, sample_mean=mean,
                    with_old_g_lower=cantelli(g_lower, variance_illustration),
                    with_sample_mean=cantelli(mean, variance_illustration),
                    bank_only_with_old_g_lower=cantelli(g_lower, variance_parts['bank']),
                    bank_only_with_sample_mean=cantelli(mean, variance_parts['bank']),
                    valid_probability_bound=False, audit_confidence=None,
                    qualification='All variance inputs are empirical estimates, not certified true upper bounds; numerical illustration only'))


def truncation_diagnostics(differences, max_abs_z):
    """Raw sample summaries for fixed development T; no EB or tail guarantees."""
    d = np.asarray(differences, dtype=np.float64)
    maxima = np.asarray(max_abs_z, dtype=np.float64)
    assert maxima.shape == (len(d),)
    same = np.all(YS == YS[:, :1], axis=1)
    x = d.mean(axis=1)
    delta = d[:, same].mean(axis=1) - d[:, ~same].mean(axis=1)
    rows = []
    for threshold in DEVELOPMENT_THRESHOLDS:
        outside = maxima > threshold
        truncated_x = np.where(outside, 0., x)
        truncated_delta = np.where(outside, 0., delta)
        quantities = {'X': truncated_x, 'D_T': truncated_delta, 'X_squared': truncated_x ** 2}
        rows.append(dict(T=threshold, banks=len(d), truncated_banks=int(outside.sum()),
                         summaries={name: dict(mean=float(values.mean()), sample_variance=float(values.var(ddof=1)),
                                              observed_min=float(values.min()), observed_max=float(values.max()))
                                    for name, values in quantities.items()},
                         new_confidence_statement=False,
                         data_role='post-hoc descriptive threshold comparison; not an independent audit'))
    return rows


def run_saved(out, source, reference, authorized=False):
    assert authorized, 'Read saved science banks only after the separate authorized stage notification'
    inputs = verify_inputs(out)
    qualification = read(out / 'MANUFACTURED_QUALIFICATION.json')
    assert qualification['passed'] and qualification['inputs_sha256'] == sha(out / 'DEVELOPMENT_INPUTS.json')
    assert qualification['code_sha256'] == sha(__file__)
    assert rel(source) == inputs['source_array'] and rel(reference) == inputs['reference_results']
    assert not (out / 'DEVELOPMENT_RUN_START.json').exists(), 'No silent scientific-data replay'
    started = time.monotonic()
    source_hash = sha(source)
    assert source_hash == inputs['expected_source_array_sha256']
    assert source.stat().st_size == inputs['expected_source_array_bytes']
    save_new(out / 'DEVELOPMENT_RUN_START.json', dict(utc=now(), input_binding_sha256=sha(out / 'DEVELOPMENT_INPUTS.json'),
             qualification_sha256=sha(out / 'MANUFACTURED_QUALIFICATION.json'), code_sha256=sha(__file__),
             source_array=rel(source), source_sha256=source_hash, new_rng=0,
             data_role='post-hoc development from complete old saved evaluation banks'))
    reference_result = read(reference)
    reference_row = next(row for row in reference_result['population']['results'] if row['contrast'] == 'g256')
    assert reference_result['population']['truncated_banks'] == 0
    with np.load(source, allow_pickle=False) as saved:
        z, s, original = saved['z'], saved['s'], saved['difference']
    banks, k, dimensions = z.shape
    assert (banks, k, dimensions) == (262144, 100, 3)
    assert s.shape == (banks, k) and original.shape == (banks, 2)
    differences = np.empty((banks, 8), dtype=np.float64)
    max_abs_z = np.empty(banks, dtype=np.float64)
    max_error, block_calls = 0., 0
    for lo in range(0, banks, BLOCK):
        hi = min(lo + BLOCK, banks)
        differences[lo:hi] = point_differences(z[lo:hi], s[lo:hi])
        max_abs_z[lo:hi] = np.max(np.abs(z[lo:hi]), axis=(1, 2))
        error = float(np.max(np.abs(differences[lo:hi].mean(axis=1) - original[lo:hi, 1])))
        max_error = max(max_error, error)
        assert error <= TOLERANCE, (lo, hi, error)
        block_calls += 1
        assert time.monotonic() - started <= MAX_SECONDS, 'Development wall-time limit'
        if hi % (BLOCK * 64) == 0 or hi == banks:
            print(json.dumps(dict(stage='saved_bank_development', banks=hi, total=banks,
                                  max_original_bank_difference_error=max_error,
                                  wall_seconds=time.monotonic() - started)), flush=True)
    statistics = descriptive_statistics(differences, reference_row['interval'][0])
    truncation = truncation_diagnostics(differences, max_abs_z)
    mean_error = abs(statistics['point_difference_mean'] - reference_row['mean'])
    assert mean_error <= TOLERANCE
    variance_error = abs(statistics['bank_mean_sample_variance'] - reference_row['variance'])
    assert variance_error <= TOLERANCE
    array = out / 'DEVELOPMENT_POINT_DIFFERENCES.npz'
    save_npz_new(array, difference_by_label=differences, label_order=YS,
                 bank_mean=differences.mean(axis=1), bank_E_y_d_squared=np.mean(differences * differences, axis=1),
                 max_abs_active_z=max_abs_z,
                 bank_same_minus_split=differences[:, [0, 7]].mean(axis=1) - differences[:, 1:7].mean(axis=1))
    assert sha(source) == source_hash, 'Read-only scientific source changed during development'
    resources = dict(new_rng=0, new_observations=0, new_prior_vectors=0, new_Gaussians=0,
                     old_banks_processed=banks, K=k, reused_prior_vectors=banks * k,
                     reused_active_Gaussian_values=banks * k * 3, reused_nuisance_log_sums=banks * k,
                     coupled_kernel_block_calls=block_calls,
                     candidate_bank_label_logscore_evaluations=2 * banks * 8,
                     candidate_particle_logit_transformations=2 * banks * k,
                     scientific_Arb_banks=0, optimizer_updates=0, new_statistical_contrasts=0)
    result = dict(finished_utc=now(), wall_seconds=time.monotonic() - started, development_only=True,
                  post_hoc=True, source_array=dict(path=rel(source), sha256=source_hash, bytes=source.stat().st_size),
                  output_array=dict(path=rel(array), sha256=sha(array), bytes=array.stat().st_size),
                  inputs_sha256=sha(out / 'DEVELOPMENT_INPUTS.json'), code_sha256=sha(__file__),
                  contract_sha256=sha(CONTRACT), statistics=statistics, truncation_diagnostics=truncation,
                  verification=dict(max_original_bank_difference_error=max_error,
                                    old_population_mean_error=mean_error, old_sample_variance_error=variance_error,
                                    tolerance=TOLERANCE, original_source_unchanged=True),
                  resources=resources, claims=dict(new_confidence_family=False, true_variance_upper_bound=False,
                      algorithm_failure_probability_lower_bound=False, whole_pipeline_numerical_certification=False,
                      scientific_all_node_Arb_replay=False, original_learner_modified=False))
    save_new(out / 'DEVELOPMENT_RESULTS.json', result)
    print(json.dumps(native(dict(development_completed=True, statistics=statistics,
                                 verification=result['verification'], resources=resources,
                                 wall_seconds=result['wall_seconds'])), ensure_ascii=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'qualify', 'run-saved'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--reference-results', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--saved-bank-authorized', action='store_true')
    args = parser.parse_args()
    out = check_output(args.output)
    try:
        if args.mode == 'prepare':
            prepare(out, args.source, args.reference_results)
        elif args.mode == 'qualify':
            qualify(out)
        else:
            run_saved(out, args.source, args.reference_results, authorized=args.saved_bank_authorized)
    except BaseException as error:
        save_new(out / f'DEVELOPMENT_FAILURE_{args.mode}_{time.time_ns()}.json',
                 dict(utc=now(), code_sha256=sha(__file__), error=repr(error), traceback=traceback.format_exc(),
                      new_rng=0, scientific_confidence_claim=False))
        raise


if __name__ == '__main__':
    main()
