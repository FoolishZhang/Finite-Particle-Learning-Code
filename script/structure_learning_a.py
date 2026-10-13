"""A: fixed wrong-group assignment, old numerical inputs, new shared pattern data."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import sys
import time
import traceback

os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
import scipy
from scipy.special import stdtr
from scipy.optimize import brentq
from scipy.stats import t as student_t
import structure_learning_mapping as mapping

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "artifacts/scale_learning_v2"
OUT = ROOT / "artifacts/structure_learning_v2/A"
DOC = ROOT / "docs/08_research_研究工作/结构先验释放与分组学习_v2_20260930"
BASE = "5cfbd418a48732ce33c5cde04b370fa5e098cd5f"
CFG = dict(d=16, c=.9, teacher_scales=[6., 2.], profile="contrast",
    grid=[i/20 for i in range(31)], truth_index=20, r_values=[0, 1, 2, 4],
    train_N=[4096, 65536], streams=16, data_seed=20330101,
    objectives=["J", "P", "G10", "G100", "G1000", "S10", "S100", "S1000"],
    K=[10, 100, 1000], family=40, alpha=.05, SE_gate=.001, material=.001,
    primary_N=65536, primary_comparison="G10-J",
    comparisons=[["P", "J"], ["G10", "J"], ["G100", "J"], ["G1000", "J"], ["S10", "G10"]],
    reference_gate=1e-7, identity_gate=1e-10, mass_gate=1e-12,
    new_prior_banks=0, new_reference_integrations=0,
    reused_bank_streams=16, reused_banks_per_stream=64,
    candidate_semantics="scales assigned to assumed groups; only teacher weights map",
    grouping_rule="true A bits0..7; exchange the first r bits of each half",
    finite_data_rule="full-pattern multinomial prefixes; exact integer mask regrouping",
    all_comparisons_are_developmental=True, pair_equivalence_gate=False)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def convert(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return relative(obj)
    raise TypeError(type(obj).__name__)


def write(path, obj, exclusive=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, default=convert, allow_nan=False)
        stream.write("\n")


def csvwrite(path, records):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def hashes(paths):
    return {relative(p): sha(p) for p in paths}


def verify_hashes(files):
    for name, digest in files.items():
        assert name != "HANDOFF.md", "scientific checks must bind the snapshot"
        assert sha(ROOT / name) == digest, f"hash mismatch: {name}"


def freeze():
    assert not OUT.exists(), "A directory already exists; no silent overwrite"
    old_manifest = read(OLD / "MANIFEST.json")
    verify_hashes(old_manifest["scientific_files"])
    refs = [OLD / f"reference/contrast_t{i:02}.json" for i in range(31)]
    receipts = [OLD / f"streams/contrast_s{s:02}.json" for s in range(16)]
    arrays = {read(p)["array_path"]: read(p)["array_sha256"] for p in receipts}
    verify_hashes(arrays)
    assert read(OLD / "STATE.json")["status"] == "complete"
    OUT.mkdir(parents=True)
    (OUT / "local").mkdir()
    (OUT / "source_inputs").mkdir()
    (OUT / ".gitignore").write_text("/local/\n", encoding="utf-8", newline="\n")
    (OUT / "source_inputs/HANDOFF.md").write_bytes((OLD / "source_inputs/HANDOFF.md").read_bytes())
    write(OUT / "CONFIG.json", CFG)
    write(OUT / "ENVIRONMENT.json", dict(python=sys.version, numpy=np.__version__,
        scipy=scipy.__version__, platform=platform.platform(), threads=2))
    write(OUT / "INPUTS.json", dict(old_manifest_sha256=sha(OLD / "MANIFEST.json"),
        reference_files=hashes(refs), stream_receipts=hashes(receipts),
        bank_arrays=arrays, new_banks=0, new_integrations=0,
        old_mean_pattern_loss_shape=[31, 3, 2, 81]))
    paths = [Path(__file__), ROOT / "script/structure_learning_mapping.py",
        DOC / "PLAN_A.md", DOC / "MAPPING.md", OUT / "CONFIG.json",
        OUT / "ENVIRONMENT.json", OUT / "INPUTS.json", OUT / "source_inputs/HANDOFF.md",
        OLD / "MANIFEST.json", OLD / "FREEZE.json", OLD / "RESULTS_FREEZE.json",
        ROOT / "artifacts/paper_v1_final/MANIFEST.json",
        ROOT / "artifacts/paper_v1_final/VERSION_LOCK.json",
        ROOT / "artifacts/structure_tradeoff_v2/MANIFEST.json"] + refs + receipts
    write(OUT / "FREEZE.json", dict(time_utc=now(), baseline_commit=BASE,
        files=hashes(paths), local_inputs=arrays, no_scientific_computation_yet=True))
    write(OUT / "STATE.json", dict(status="frozen", time_utc=now(), active_jobs=[]))
    print(json.dumps(dict(frozen=True, freeze_sha256=sha(OUT / "FREEZE.json"))), flush=True)


def checkfreeze(workspace=False):
    fr = read(OUT / "FREEZE.json")
    assert read(OUT / "CONFIG.json") == CFG
    verify_hashes(fr["files"])
    if workspace:
        verify_hashes(fr["local_inputs"])


def pair_vectors(refs, geo):
    sa, sb = 2*geo["na"]-8, 2*geo["nb"]-8
    pair_n = np.array([28, 28, 64])
    products = np.column_stack([(sa*sa-8)//2, (sb*sb-8)//2, sa*sb])
    same = (pair_n+products)//2
    different = pair_n-same
    result = []
    for ref in refs:
        corr = -np.asarray(ref["independent_Hermite_moments"]["pair_magnitudes_AA_BB_AB"])
        assert np.all(abs(corr) < 1)
        result.append(math.log(4)-(same @ np.log1p(corr)+different @ np.log1p(-corr))/120)
    return np.array(result)


def mapping_smoke():
    maxima = []
    # Direct enumeration of all d4 patterns verifies hypergeometric numerators.
    pg = mapping.pattern_geometry(4, (0, 1, 2))
    geo = mapping.count_geometry(4)
    for ri, r in enumerate((0, 1, 2)):
        observed = np.zeros((9, 9), dtype=np.int64)
        for source, target in zip(pg["true_count_indices"], pg["count_indices_by_r"][ri]):
            observed[source, target] += 1
        kernel = mapping.swap_kernel(r, 2)
        assert np.array_equal(observed*kernel["denominator"],
                              geo["multiplicity"][:, None]*kernel["numerator"])
    # All m8 r values are algebraic kernel checks, not additional science cases.
    for r in range(9):
        kernel = mapping.swap_kernel(r)
        maxima.append(float(np.max(abs(kernel["kernel"].sum(axis=1)-1))))
    assert max(maxima) < 1e-12
    return dict(passed=True, direct_patterns=16, direct_d=4,
                exact_integer_detailed_balance=True, m8_r0_through_r8_algebra=True,
                maximum_float_row_mass_error=max(maxima), scientific_cases=CFG["r_values"])


def prepare_inputs():
    geo = mapping.count_geometry(16)
    pg = mapping.pattern_geometry(16, CFG["r_values"])
    refs = [read(OLD / f"reference/contrast_t{i:02}.json") for i in range(31)]
    assert all(r["qualified"] for r in refs)
    q = np.array([r["mean"] for r in refs])
    variants = np.array([r["variants"] for r in refs])
    assert q.shape == (31, 81) and variants.shape == (31, 3, 81)
    p = q[20]
    teacher_variants = variants[20]
    H = mapping.orbit_entropy(p)
    assert abs(H-refs[20]["H"]) < 1e-12
    kernels = [mapping.swap_kernel(r) for r in CFG["r_values"]]
    W = np.array([k["kernel"] for k in kernels])
    mapped = np.array([p @ w for w in W])
    mapped_variants = np.array([teacher_variants @ w for w in W])
    pattern_prob, pattern_note = mapping.pattern_probabilities(p, pg)
    direct = np.array([np.bincount(cells, weights=pattern_prob, minlength=81)
                       for cells in pg["count_indices_by_r"]])
    map_error = float(np.max(abs(direct-mapped)))
    assert map_error < 1e-12
    assert np.max(abs(mapped @ geo["indicator"]-p @ geo["indicator"])) < 1e-12
    entropy = np.array([mapping.orbit_entropy(row) for row in mapped])
    floor = entropy-H
    assert floor.min() > -1e-12
    qtotal = q @ geo["indicator"]
    joint_logs = -np.log(q)+geo["log_multiplicity"]
    single_logs = -np.log(qtotal[:, geo["total"]])+geo["log_total_multiplicity"]
    joint_log_variants = -np.log(variants)+geo["log_multiplicity"]
    total_variants = np.einsum("tva,an->tvn", variants, geo["indicator"])
    single_log_variants = -np.log(total_variants[:, :, geo["total"]])+geo["log_total_multiplicity"]
    pair_logs = pair_vectors(refs, geo)
    pair_errors = np.array([ref["independent_Hermite_moments"]["pair_error_estimates_plus_analytic_tail"] for ref in refs])
    pair_magnitudes = np.array([ref["independent_Hermite_moments"]["pair_magnitudes_AA_BB_AB"] for ref in refs])
    assert np.all(1-np.abs(pair_magnitudes)-pair_errors > 0)
    pair_error_estimate = float(np.max(pair_errors/(1-np.abs(pair_magnitudes)-pair_errors)))
    assert pair_error_estimate <= CFG["reference_gate"]
    qkl = np.array([joint_logs @ row-H for row in mapped])
    skl = np.array([single_logs @ row-H for row in mapped])
    qsym = qtotal[:, geo["total"]]*geo["multiplicity"]/geo["total_multiplicity"]
    q_residual = np.array([np.sum(row*np.log(row/q), axis=1) for row in mapped])
    s_residual = np.array([np.sum(row*np.log(row/qsym), axis=1) for row in mapped])
    identity = max(float(np.max(abs(qkl-floor[:, None]-q_residual))),
                   float(np.max(abs(skl-floor[:, None]-s_residual))),
                   float(np.max(abs(skl-skl[0]))))
    assert identity < 1e-10
    assert min(float(q_residual.min()), float(s_residual.min())) > -1e-12
    risk_variants = []
    for vi in range(3):
        pv = variants[20, vi]
        hv = mapping.orbit_entropy(pv)
        for vj in range(3):
            qq = variants[:, vj]
            qn = qq @ geo["indicator"]
            jlog = -np.log(qq)+geo["log_multiplicity"]
            slog = -np.log(qn[:, geo["total"]])+geo["log_total_multiplicity"]
            risk_variants.append(np.array([[jlog @ (pv @ w)-hv, slog @ (pv @ w)-hv] for w in W]))
    sensitivity = float(np.ptp(risk_variants, axis=0).max())
    assert sensitivity <= CFG["reference_gate"]
    best = qkl.min(axis=1)
    teacher = dict(d=16, c=.9, scales=[6., 2.], mean=p, variants=teacher_variants,
        H_original=H, r_values=CFG["r_values"], N=CFG["train_N"],
        mapped_counts_by_r=mapped, mapped_variants_by_r=mapped_variants,
        projection_floor_by_r=floor, H_orbit_by_r=entropy,
        group_multiplicities=geo["multiplicity"],
        pattern_indexing=pg["pattern_convention"], cell_indexing=pg["cell_convention"],
        assumed_A_masks=pg["assumed_A_masks"], pattern_sampling=pattern_note,
        direct_pattern_vs_hypergeometric_max_error=map_error,
        numerical_reference_is_inherited=True, new_reference_integrations=0)
    reference = dict(grid=CFG["grid"], r_values=CFG["r_values"],
        group_count_pmf=q, group_total_pmf=qtotal,
        joint_pattern_loss=joint_logs, single_pattern_loss=single_logs,
        joint_pattern_loss_variants=joint_log_variants,
        single_pattern_loss_variants=single_log_variants,
        pair_mean_loss=pair_logs, pair_loss_uniform_error_estimate=pair_error_estimate,
        Q_assumed_group_KL=qkl, S_deployment_KL=skl,
        Q_count_residual_KL=q_residual, S_count_residual_KL=s_residual,
        projection_floor_by_r=floor, J_grid_best_KL=best,
        reference_sensitivity=sensitivity, decomposition_max_error=identity,
        wrong_groups_change_candidate_scale_assignment=True,
        teacher_only_mapped=True, candidate_pmf_not_mapped=True)
    write(OUT / "MAPPING.json", dict(r_values=CFG["r_values"], kernels=kernels,
        convention="row vector: mapped_teacher = teacher @ W",
        candidate_group_assignment_rule=CFG["grouping_rule"]))
    write(OUT / "TEACHER.json", teacher)
    write(OUT / "CANDIDATE_REFERENCE.json", reference)
    return pg, pattern_prob, teacher, reference


def make_data_stream(stream, pg, pattern_prob, teacher, reference):
    old = read(OLD / f"streams/contrast_s{stream:02}.json")
    oldpath = ROOT / old["array_path"]
    assert sha(oldpath) == old["array_sha256"]
    with np.load(oldpath, allow_pickle=False) as archive:
        logs = archive["mean_pattern_loss"].copy()
    assert logs.shape == (31, 3, 2, 81)
    seed = CFG["data_seed"]+stream
    rng = np.random.Generator(np.random.PCG64(seed))
    initial = rng.bit_generator.state
    small = rng.multinomial(4096, pattern_prob)
    large = small+rng.multinomial(61440, pattern_prob)
    hist = np.stack([small, large])
    counts = mapping.integer_group_histograms(hist, pg)
    assert np.all(counts[:, 1] >= counts[:, 0])
    p = np.asarray(teacher["mapped_counts_by_r"])
    pop = np.einsum("tkma,ra->rtkm", logs, p)
    finite = np.stack([np.einsum("tkma,na->ntkm", logs, counts[ri]/np.array(CFG["train_N"])[:, None])
                       for ri in range(4)])
    # S depends on total count, invariant to the same fixed pattern regrouping.
    assert np.max(abs(pop[:, :, :, 1]-pop[0, :, :, 1])) < 1e-10
    assert np.max(abs(finite[:, :, :, :, 1]-finite[0, :, :, :, 1])) < 1e-10
    tv = np.asarray(teacher["mapped_variants_by_r"])
    pv = np.einsum("tkma,rva->vrtkm", logs, tv)
    sensitivity = float(np.ptp(pv, axis=0).max())
    assert sensitivity <= CFG["reference_gate"]
    empirical = counts/np.array(CFG["train_N"])[None, :, None]
    j_variants = np.einsum("tva,rna->vrnt", reference["joint_pattern_loss_variants"], empirical)
    s_variants = np.einsum("tva,rna->vrnt", reference["single_pattern_loss_variants"], empirical)
    empirical_sensitivity = max(float(np.ptp(j_variants, axis=0).max()),
                                float(np.ptp(s_variants, axis=0).max()))
    assert empirical_sensitivity <= CFG["reference_gate"]
    target = OUT / "local" / f"training_s{stream:02}.npz"
    assert not target.exists()
    np.savez_compressed(target, pattern_histograms=hist, training_counts_by_r=counts)
    record = dict(stream=stream, seed=seed, initial_RNG=initial,
        final_RNG=rng.bit_generator.state, N=CFG["train_N"], r_values=CFG["r_values"],
        training_counts_by_r=counts, pattern_histograms_shape=list(hist.shape),
        pattern_indexing=pg["pattern_convention"], array_path=target, array_sha256=sha(target),
        original_bank_path=oldpath, original_bank_sha256=old["array_sha256"],
        population_objectives=pop, finite_particle_objectives=finite,
        propagated_reference_sensitivity=sensitivity, count_observations=65536,
        empirical_reference_sensitivity=empirical_sensitivity,
        pair_loss_error_estimate=reference["pair_loss_uniform_error_estimate"],
        numerical_pass=True,
        new_prior_banks=0, old_bank_stream=stream, old_bank_count=64,
        true_integer_regrouping=True, fractional_hypergeometric_training_counts=False)
    write(OUT / "streams" / f"s{stream:02}.json", record)
    print(json.dumps(dict(phase="data", stream=stream, new_observations=65536, new_banks=0)), flush=True)


def interval(values, independent=False):
    values = np.asarray(values, dtype=float)
    if independent:
        avg = math.fsum(float(v) for v in values)/16
        se = math.sqrt(math.fsum((float(v)-avg)**2 for v in values)/(15*16))
        critical = brentq(lambda z: stdtr(15, z)-(1-.05/80), 0., 100., xtol=1e-13)
    else:
        avg = float(values.mean())
        se = float(values.std(ddof=1)/4)
        critical = float(student_t.ppf(1-.05/80, 15))
    return dict(mean=avg, SE=se, lower=avg-critical*se, upper=avg+critical*se,
                precision_pass=bool(se <= .001))


def public_results():
    teacher = read(OUT / "TEACHER.json")
    reference = read(OUT / "CANDIDATE_REFERENCE.json")
    logsJ, logsP = [np.asarray(reference[name]) for name in ("joint_pattern_loss", "pair_mean_loss")]
    mapped = np.asarray(teacher["mapped_counts_by_r"])
    qkl, skl = [np.asarray(reference[name]) for name in ("Q_assumed_group_KL", "S_deployment_KL")]
    best = np.asarray(reference["J_grid_best_KL"])
    floors = np.asarray(teacher["projection_floor_by_r"])
    qres, sres = [np.asarray(reference[name]) for name in ("Q_count_residual_KL", "S_count_residual_KL")]
    records = [read(OUT / f"streams/s{s:02}.json") for s in range(16)]
    population = np.array([r["population_objectives"] for r in records])
    particle = np.array([r["finite_particle_objectives"] for r in records])
    counts = np.array([r["training_counts_by_r"] for r in records])
    finite_rows, pop_rows, curves, comparisons, primary = [], [], [], [], []
    risks = np.zeros((4, 16, 2, 8))

    def fields(ri, ti, objective):
        deployed = skl[ri, ti] if objective.startswith("S") else qkl[ri, ti]
        return dict(Q_assumed_group_KL=float(qkl[ri, ti]), S_deployment_KL=float(skl[ri, ti]),
            deployed_KL=float(deployed), J_grid_best_KL=float(best[ri]),
            deployed_excess_over_J_grid=float(deployed-best[ri]),
            projection_floor=float(floors[ri]), Q_count_residual_KL=float(qres[ri, ti]),
            S_count_residual_KL=float(sres[ri, ti]))

    for ri, r in enumerate(CFG["r_values"]):
        fixed = {"J": logsJ @ mapped[ri], "P": logsP @ mapped[ri]}
        populations = dict(fixed)
        for mi, method in enumerate(("G", "S")):
            for ki, k in enumerate(CFG["K"]):
                populations[method+str(k)] = population[:, ri, :, ki, mi].mean(axis=0)
        for objective in CFG["objectives"]:
            curve = populations[objective]
            ti = int(np.argmin(curve))
            pop_rows.append(dict(r=r, objective=objective, selected_index=ti,
                t=CFG["grid"][ti], objective_mean=float(curve[ti]),
                grid_estimate_not_known_population_argmin=objective not in ("J", "P"),
                **fields(ri, ti, objective)))
            for index, amplitude in enumerate(CFG["grid"]):
                if objective in ("J", "P"):
                    se = 0.
                else:
                    mi = 0 if objective[0] == "G" else 1
                    ki = CFG["K"].index(int(objective[1:]))
                    se = float(population[:, ri, index, ki, mi].std(ddof=1)/4)
                curves.append(dict(r=r, objective=objective, t=amplitude,
                    objective_mean=float(curve[index]), descriptive_MC_SE=se,
                    objective_units="nats_per_pair" if objective == "P" else "nats_per_full_pattern",
                    old_bank_reuse_developmental=objective not in ("J", "P"),
                    **fields(ri, index, objective)))
        for s in range(16):
            for ni, N in enumerate(CFG["train_N"]):
                empirical = counts[s, ri, ni]/N
                targets = {"J": logsJ @ empirical, "P": logsP @ empirical}
                targets.update({method+str(k): particle[s, ri, ni, :, ki, mi]
                    for mi, method in enumerate(("G", "S")) for ki, k in enumerate(CFG["K"])})
                for oi, objective in enumerate(CFG["objectives"]):
                    target = targets[objective]
                    ti = int(np.argmin(target))
                    f = fields(ri, ti, objective)
                    risks[ri, s, ni, oi] = f["deployed_KL"]
                    finite_rows.append(dict(r=r, stream=s, N=N, objective=objective,
                        selected_index=ti, t=CFG["grid"][ti], train_objective=float(target[ti]),
                        grid_minimum_gap=0., **f))
        for ni, N in enumerate(CFG["train_N"]):
            for left, right in CFG["comparisons"]:
                diff = risks[ri, :, ni, CFG["objectives"].index(left)]-risks[ri, :, ni, CFG["objectives"].index(right)]
                stat = interval(diff)
                comparisons.append(dict(r=r, N=N, comparison=left+"-"+right, **stat,
                    confidence_family=40, alpha=.05, df=15,
                    paired_on_stream=True, development_with_reused_banks=True))
                if N == 65536 and left == "G10" and right == "J":
                    primary.append(dict(r=r, comparison="G10-J", N=N,
                        primary_pass=stat["precision_pass"] and stat["lower"] > .001, **stat))
    assert len(finite_rows) == 1024 and len(pop_rows) == 32 and len(curves) == 992 and len(comparisons) == 40
    return dict(finite=finite_rows, population=pop_rows, curves=curves,
        intervals=comparisons, primary=primary, risks=risks)


def collect():
    result = public_results()
    for key, name in (("finite", "FINITE_FITS.csv"), ("population", "POPULATION_FITS.csv"),
                      ("curves", "POPULATION_CURVES.csv"), ("intervals", "FINITE_RISK_INTERVALS.csv")):
        csvwrite(OUT / name, result[key])
    write(OUT / "DECISION.json", dict(primary=result["primary"],
        all_primary_pass=all(r["primary_pass"] for r in result["primary"]),
        precision_pass=sum(r["precision_pass"] for r in result["intervals"]),
        interval_rows=40, finite_fits=1024, population_grid_estimates=32,
        new_count_observations=1048576, new_prior_banks=0, new_reference_integrations=0,
        reused_prior_banks=1024, reused_unique_prior_particles=1024000,
        pair_equivalence_claim=False, development_not_independent_new_confirmation=True))


def audit(workspace=False):
    checkfreeze(workspace)
    verify_hashes(read(OUT / "DATA_FREEZE.json")["files"])
    pg = mapping.pattern_geometry(16, CFG["r_values"])
    teacher, reference = read(OUT / "TEACHER.json"), read(OUT / "CANDIDATE_REFERENCE.json")
    geo = mapping.count_geometry(16)
    p = np.asarray(teacher["mean"])
    maxima = []
    for ri, r in enumerate(CFG["r_values"]):
        kernel = mapping.swap_kernel(r)["kernel"]
        maxima.append(float(np.max(abs(p @ kernel-np.asarray(teacher["mapped_counts_by_r"])[ri]))))
    array_count = 0
    for s in range(16):
        rec = read(OUT / f"streams/s{s:02}.json")
        counts = np.asarray(rec["training_counts_by_r"])
        assert counts.shape == (4, 2, 81)
        assert np.array_equal(counts.sum(axis=-1), np.tile(CFG["train_N"], (4, 1)))
        assert np.all(counts[:, 1] >= counts[:, 0])
        assert np.array_equal(counts, counts.astype(np.int64))
        if workspace:
            verify_hashes({rec["array_path"]: rec["array_sha256"],
                           rec["original_bank_path"]: rec["original_bank_sha256"]})
            with np.load(ROOT / rec["array_path"], allow_pickle=False) as arr:
                assert arr["pattern_histograms"].shape == (2, 65536)
                assert np.array_equal(mapping.integer_group_histograms(arr["pattern_histograms"], pg), counts)
                assert np.array_equal(arr["training_counts_by_r"], counts)
            with np.load(ROOT / rec["original_bank_path"], allow_pickle=False) as oldarr:
                logs = oldarr["mean_pattern_loss"]
                pop = np.array([logs @ row for row in teacher["mapped_counts_by_r"]])
                finite = np.array([[logs @ (counts[ri, ni]/N) for ni, N in enumerate(CFG["train_N"])]
                                   for ri in range(4)])
                maxima.extend([float(np.max(abs(pop-rec["population_objectives"]))),
                               float(np.max(abs(finite-rec["finite_particle_objectives"])))])
            array_count += 1
    expected = public_results()
    for key, name in (("finite", "FINITE_FITS.csv"), ("population", "POPULATION_FITS.csv"),
                      ("curves", "POPULATION_CURVES.csv"), ("intervals", "FINITE_RISK_INTERVALS.csv")):
        with (OUT / name).open(encoding="utf-8", newline="") as stream:
            saved = list(csv.DictReader(stream))
        assert len(saved) == len(expected[key])
        for a, b in zip(saved, expected[key]):
            for field, value in b.items():
                if isinstance(value, (float, np.floating)):
                    maxima.append(abs(float(a[field])-float(value)))
                elif isinstance(value, (bool, np.bool_)):
                    assert a[field] == str(bool(value))
                else:
                    assert a[field] == str(value)
    # Independently recompute the 40 interval summaries from selected deployment risks.
    rows = expected["intervals"]
    for ri, r in enumerate(CFG["r_values"]):
        for ni, N in enumerate(CFG["train_N"]):
            for left, right in CFG["comparisons"]:
                values = (expected["risks"][ri, :, ni, CFG["objectives"].index(left)]
                          -expected["risks"][ri, :, ni, CFG["objectives"].index(right)])
                stat = interval(values, independent=True)
                row = next(x for x in rows if (x["r"], x["N"], x["comparison"]) == (r, N, left+"-"+right))
                for field in ("mean", "SE", "lower", "upper"):
                    maxima.append(abs(stat[field]-row[field]))
                assert stat["precision_pass"] == row["precision_pass"]
    assert max(maxima) < 2e-11
    return dict(passed=True, time_utc=now(), maximum_replay_error=max(maxima),
        finite_argmins=1024, population_grid_estimates=32, paired_intervals=40,
        new_local_pattern_arrays_verified=array_count, old_bank_arrays_verified=array_count,
        workspace=workspace, public_objective_replay=True,
        independent_t_quantile_and_interval_replay=True,
        limitation="argmin replay shares the declared objective assembly; no scientific resampling or new reference integrals",
        no_new_scientific_samples_in_audit=True)


def report():
    decision = read(OUT / "DECISION.json")
    teacher, reference = read(OUT / "TEACHER.json"), read(OUT / "CANDIDATE_REFERENCE.json")
    lines = ["# A：尺度赋错组的一维学习结果", "",
        "本包复用旧contrast31个合格参考及16个bank流，0新bank、0新参考积分。新产生16个完整pattern数据流，各65536观察，N4096为前缀；全部错误组count由实际模式整数重分组获得。", "",
        "Q风险指该r假定组下未再全对称化的完整候选；S风险指其完整对称化。不能把Q风险误读成把尺度放回真组的参数恢复风险。", "",
        "|交换r|组轨道投影floor|J-grid最佳KL|", "|---:|---:|---:|"]
    for ri, r in enumerate(CFG["r_values"]):
        lines.append(f"|{r}|{teacher['projection_floor_by_r'][ri]:.10g}|{reference['J_grid_best_KL'][ri]:.10g}|")
    lines += ["", f"40项精度通过{decision['precision_pass']}/40；4个预定G10−J主门通过"
              f"{sum(x['primary_pass'] for x in decision['primary'])}/4。P−J不作等效判定。", "",
        "|r|N65536 G10−J均值|同时下界|上界|主门|", "|---:|---:|---:|---:|---|"]
    for row in decision["primary"]:
        lines.append(f"|{row['r']}|{row['mean']:.10g}|{row['lower']:.10g}|{row['upper']:.10g}|{'通过' if row['primary_pass'] else '未通过'}|")
    lines += ["", "完整1024拟合、32聚合网格估计、992总体曲线读数及40配对区间见CSV。总体MC曲线仅为旧16流的开发性重加权，不能称新独立确认；其SE只作描述，不新增总体成功门。", "",
        "所有部署风险都另减同r的J-grid最佳风险列出。S模型类与Q假定分组类不同，其基线扣除值可能为负，原样保留。不可表示floor来自组内轨道投影；full KL=floor+count residual，不能仅用mapped-count KL冒充完整风险。", "",
        "统计单位为16个流，40项Bonferroni同时95% t近似，参考精度单列；不是严格全误差覆盖。新数据并未把旧bank重新变成新确认。", ""]
    (OUT / "READABLE.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def run():
    checkfreeze(True)
    assert read(OUT / "STATE.json")["status"] == "frozen", "no silent restart"
    write(OUT / "STATE.json", dict(status="running_inputs", pid=os.getpid(), time_utc=now()), False)
    write(OUT / "SMOKE.json", mapping_smoke())
    pg, prob, teacher, reference = prepare_inputs()
    for s in range(16):
        write(OUT / "STATE.json", dict(status="running_data", stream=s, pid=os.getpid(), time_utc=now()), False)
        make_data_stream(s, pg, prob, teacher, reference)
    datafiles = [OUT / name for name in ("TEACHER.json", "MAPPING.json", "CANDIDATE_REFERENCE.json", "SMOKE.json")]
    datafiles += sorted((OUT / "streams").glob("*.json"))
    write(OUT / "DATA_FREEZE.json", dict(time_utc=now(), files=hashes(datafiles),
        local_arrays={read(p)["array_path"]: read(p)["array_sha256"]
                      for p in sorted((OUT / "streams").glob("*.json"))},
        data_complete=True, new_count_observations=1048576, new_prior_banks=0))
    collect()
    write(OUT / "AUDIT.json", audit(True))
    report()
    resultfiles = sorted(OUT.glob("*.csv"))+[OUT / name for name in
        ("DECISION.json", "AUDIT.json", "READABLE.md", "DATA_FREEZE.json")]
    write(OUT / "RESULTS_FREEZE.json", dict(time_utc=now(), files=hashes(resultfiles),
        data_freeze_sha256=sha(OUT / "DATA_FREEZE.json")))
    write(OUT / "STATE.json", dict(status="complete", active_jobs=[], time_utc=now()), False)
    public = [p for p in OUT.rglob("*") if p.is_file() and "local" not in p.relative_to(OUT).parts
              and p.name != "MANIFEST.json"]
    write(OUT / "MANIFEST.json", dict(time_utc=now(), files=hashes(public),
        local_arrays_excluded=True, independent_new_banks=0))
    print(json.dumps(read(OUT / "DECISION.json")), flush=True)


def check(workspace):
    checkfreeze(workspace)
    verify_hashes(read(OUT / "MANIFEST.json")["files"])
    verify_hashes(read(OUT / "RESULTS_FREEZE.json")["files"])
    assert read(OUT / "RESULTS_FREEZE.json")["data_freeze_sha256"] == sha(OUT / "DATA_FREEZE.json")
    assert read(OUT / "STATE.json")["status"] == "complete"
    print(json.dumps(audit(workspace)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["freeze", "run", "check"])
    parser.add_argument("--workspace", action="store_true")
    args = parser.parse_args()
    try:
        if args.phase == "freeze":
            freeze()
        elif args.phase == "run":
            run()
        else:
            check(args.workspace)
    except Exception as error:
        if args.phase == "run" and not (OUT / "RESULTS_FREEZE.json").exists():
            write(OUT / f"FAILURE_{time.time_ns()}.json",
                  dict(time_utc=now(), error=repr(error), traceback=traceback.format_exc()))
            write(OUT / "STATE.json", dict(status="stopped_failure", active_jobs=[], time_utc=now()), False)
        raise
