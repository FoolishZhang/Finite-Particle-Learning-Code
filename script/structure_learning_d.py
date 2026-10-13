"""D: training-fixed GP/H0 contrasts and independent empirical-Bernstein validation."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/structure_learning_v2/D"
B = ROOT / "artifacts/structure_learning_v2/B"
C = ROOT / "artifacts/structure_learning_v2/C"
DOC = ROOT / "docs/08_research_研究工作/结构先验释放与分组学习_v2_20260930"
CFG = dict(d=16, c=.9, profiles=["null", "contrast"], streams=16, training_N=65536,
    H0_scales=[i/2 for i in range(19)], H0_tie_tolerance=1e-12,
    validation_n=262144, validation_seed=20360101, cases=32, alpha=.05,
    range_rule="max over all 81 frozen contrast values minus min over all 81",
    variance_denominator="n-1", confidence_log="log(4*32/.05)",
    keep_GP_rule="lower > 0; otherwise H0", material_threshold=None,
    theorem="Maurer-Pontil 2009 Theorem 4, PDF page 2",
    theorem_url="https://www.cs.mcgill.ca/~colt2009/papers/012.pdf",
    mass_gate=1e-12, arithmetic_gate=2e-11, reference_gate=1e-7,
    new_H0_grid_fits=32, reused_GP_models=32, new_GP_fits=0,
    new_reference_integrations=0, new_prior_banks=0, new_validation_observations=8388608,
    selection_certifies_pair_gain_only=True, no_homogeneity_test=True,
    no_joint_risk_selection=True)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write(path, obj, replace=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if replace else "x", encoding="utf-8", newline="\n") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, default=native, allow_nan=False)
        stream.write("\n")


def csvwrite(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def hashes(paths):
    return {rel(path): sha(path) for path in paths}


def verify_hashes(files):
    for name, expected in files.items():
        assert name != "HANDOFF.md", "do not bind live external feedback"
        assert sha(ROOT / name) == expected, f"hash mismatch: {name}"


def GP_training_endpoints():
    """Expose only training-selected model fields; no teacher or joint risk."""
    with (C / "FITS.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = [row for row in rows if row["method"] == "GLOBAL_P" and int(row["N"]) == 65536]
    result = {}
    for row in selected:
        key = (row["profile"], int(row["stream"]))
        assert key not in result
        model = dict(candidate_index=int(row["candidate_index"]),
            selected_mask=int(row["selected_mask"]), u=float(row["u"]), v=float(row["v"]),
            training_objective=float(row["training_objective"]),
            minimum_numerical_objective=float(row["minimum_numerical_objective"]),
            grid_gap=float(row["grid_gap"]))
        assert model["selected_mask"].bit_count() == 8
        assert [model["u"], model["v"]] == [model["candidate_index"]//19/2, model["candidate_index"]%19/2]
        result[key] = model
    assert set(result) == {(p, s) for p in CFG["profiles"] for s in range(16)}
    return result


def freeze():
    assert not OUT.exists(), "D already exists; no silent restart"
    assert read(C / "STATE.json")["status"] == "complete"
    for path in (B / "FREEZE.json", B / "REFERENCE_FREEZE.json", B / "RESULTS_FREEZE.json",
                 C / "FREEZE.json", C / "RESULTS_FREEZE.json"):
        verify_hashes(read(path)["files"])
    gp = GP_training_endpoints()
    indices = sorted({20*i for i in range(19)} | {r["candidate_index"] for r in gp.values()})
    OUT.mkdir(parents=True)
    write(OUT / "CONFIG.json", CFG)
    write(OUT / "ENVIRONMENT.json", dict(python=sys.version, numpy=np.__version__,
        scipy=scipy.__version__, platform=platform.platform(), threads=2))
    refs = [B / f"reference/g{i:03}.json" for i in indices]
    sources = [C / f"streams/{p}_s{s:02}.json" for p in CFG["profiles"] for s in range(16)]
    write(OUT / "INPUTS.json", dict(GP_training_endpoints=32,
        C_fit_ledger_sha256=sha(C / "FITS.csv"), B_reference_indices=indices,
        B_reference_files=hashes(refs), C_public_training_streams=hashes(sources),
        joint_risk_library_role="validation generator PMF and post-selection evaluation only",
        new_validation_rng_started=False))
    paths = [Path(__file__), DOC / "PLAN_D.md", DOC / "VALIDATION_THEORY.md",
        OUT / "CONFIG.json", OUT / "ENVIRONMENT.json", OUT / "INPUTS.json",
        B / "FREEZE.json", B / "REFERENCE_FREEZE.json", B / "RESULTS_FREEZE.json", B / "AUDIT.json",
        C / "FREEZE.json", C / "RESULTS_FREEZE.json", C / "AUDIT.json",
        C / "FITS.csv", C / "RISK_LIBRARY.json"] + refs + sources
    write(OUT / "FREEZE.json", dict(time_utc=now(), files=hashes(paths),
        before_H0_fits=True, before_validation=True,
        live_navigation_and_reports_excluded=True))
    write(OUT / "STATE.json", dict(status="frozen", time_utc=now(), active_jobs=[]))
    print(json.dumps(dict(frozen=True, sha256=sha(OUT / "FREEZE.json"))), flush=True)


def checkfreeze():
    assert read(OUT / "CONFIG.json") == CFG
    verify_hashes(read(OUT / "FREEZE.json")["files"])


def pair_loss_vector(ref):
    """A bounded, fixed scalar loss per whole validation observation."""
    assert ref["qualified"]
    a = np.repeat(np.arange(9), 9)
    b = np.tile(np.arange(9), 9)
    sa, sb = 2*a-8, 2*b-8
    totals = np.array([28, 28, 64])
    products = np.column_stack([(sa*sa-8)//2, (sb*sb-8)//2, sa*sb])
    same = (totals+products)//2
    different = totals-same
    magnitudes = np.asarray(ref["independent_Hermite_moments"]["pair_magnitudes_AA_BB_AB"])
    assert np.all(np.abs(magnitudes) < 1)
    losses = (120*math.log(4)-same @ np.log1p(-magnitudes)
              -different @ np.log1p(magnitudes))/120
    assert losses.shape == (81,) and np.isfinite(losses).all()
    return losses


def homogeneous_fit(M, homogeneous_refs):
    """No profile, true mask, teacher, or risk library enters this selector."""
    M = np.asarray(M, dtype=float)
    assert M.shape == (16, 16) and np.isfinite(M).all()
    assert np.max(abs(M-M.T)) < 1e-12 and np.max(abs(np.diag(M)-1)) < 1e-12
    total_pair_spin = math.fsum(float(M[i, j]) for i in range(16) for j in range(i+1, 16))
    values = []
    for ref in homogeneous_refs:
        mag = np.array(ref["independent_Hermite_moments"]["pair_magnitudes_AA_BB_AB"])
        assert np.max(abs(mag-mag[0])) < 1e-12
        r = float(mag[0])
        values.append(math.log(4)-.5*math.log1p(-r*r)+math.atanh(r)*total_pair_spin/120)
    minimum = min(values)
    selected = next(i for i, value in enumerate(values) if value <= minimum+1e-12)
    return dict(homogeneous_scale_index=selected, candidate_index=20*selected,
        u=selected/2, v=selected/2, training_objectives=values,
        training_objective=values[selected], minimum_numerical_objective=minimum,
        grid_gap=values[selected]-minimum, total_training_pair_spin=total_pair_spin,
        tie_tolerance=1e-12, tie_break="ascending homogeneous scale")


def candidate_objects():
    """Reconstruct the training-only stage without accessing RISK_LIBRARY."""
    gp = GP_training_endpoints()
    inputs = read(OUT / "INPUTS.json")
    refs = {i: read(B / f"reference/g{i:03}.json") for i in inputs["B_reference_indices"]}
    homogeneous = [refs[20*i] for i in range(19)]
    losses = {i: pair_loss_vector(ref) for i, ref in refs.items()}
    cases = []
    for profile in CFG["profiles"]:
        for stream in range(16):
            training = read(C / f"streams/{profile}_s{stream:02}.json")
            M = np.asarray(training["empirical_spin_matrices"][2])
            h0 = homogeneous_fit(M, homogeneous)
            model = gp[(profile, stream)]
            assert model["minimum_numerical_objective"] <= h0["minimum_numerical_objective"]+1e-10
            # Same training data, independently represented as selected-group counts.
            # These checks do not change the matrix-only H0 selection.
            diagnosis = training["diagnostics"][2]
            assert diagnosis["N"] == 65536
            grouped = next(row for row in diagnosis["selected_group_counts"] if row["method"] == "GLOBAL_P")
            assert int(grouped["mask"]) == model["selected_mask"]
            empirical = np.asarray(grouped["training_group_counts"], dtype=float)/65536
            assert empirical.shape == (81,) and abs(float(empirical.sum())-1) < 1e-12
            check_values = [float(losses[20*i] @ empirical) for i in range(19)]
            training_identity_error = max(abs(a-b) for a, b in zip(check_values, h0["training_objectives"]))
            assert training_identity_error < CFG["arithmetic_gate"]
            loss_GP = losses[model["candidate_index"]]
            loss_H0 = losses[h0["candidate_index"]]
            assert abs(float(loss_GP @ empirical)-model["training_objective"]) < 1e-10
            X = loss_H0-loss_GP
            low, high = float(X.min()), float(X.max())
            cases.append(dict(profile=profile, stream=stream, training_N=65536,
                GP=model, H0=h0, pair_loss_GP=loss_GP, pair_loss_H0=loss_H0,
                contrast_X=X, full_support_min=low, full_support_max=high,
                full_support_range=high-low, range_support_count=81,
                GP_and_H0_parameters_identical=model["candidate_index"] == h0["candidate_index"],
                matrix_vs_integer_count_training_identity_error=training_identity_error,
                empirical_spin_matrix_index=2, C_stream_sha256=sha(C / f"streams/{profile}_s{stream:02}.json"),
                no_teacher_or_joint_risk_in_selection=True))
    assert len(cases) == 32
    return cases


def prepare_candidates():
    cases = candidate_objects()
    write(OUT / "CANDIDATES.json", dict(cases=cases, n_cases=32,
        new_H0_fits=32, new_GP_fits=0, validation_rng_started=False,
        contrast_units="nats per pair, averaged over all 120 pairs of one observation"))
    rows = []
    for case in cases:
        h0, gp = case["H0"], case["GP"]
        rows.append(dict(profile=case["profile"], stream=case["stream"],
            GP_candidate_index=gp["candidate_index"], GP_mask=gp["selected_mask"],
            GP_u=gp["u"], GP_v=gp["v"], H0_candidate_index=h0["candidate_index"],
            H0_scale=h0["u"], H0_training_objective=h0["training_objective"],
            H0_minimum=h0["minimum_numerical_objective"], H0_grid_gap=h0["grid_gap"],
            full_support_min=case["full_support_min"], full_support_max=case["full_support_max"],
            full_support_range=case["full_support_range"],
            parameters_identical=case["GP_and_H0_parameters_identical"]))
    csvwrite(OUT / "H0_FITS.csv", rows)
    write(OUT / "CANDIDATES_FREEZE.json", dict(time_utc=now(),
        source_freeze_sha256=sha(OUT / "FREEZE.json"),
        files=hashes([OUT / "CANDIDATES.json", OUT / "H0_FITS.csv"]),
        candidates_fixed_before_any_validation_rng=True, cases=32))
    return cases


def validation_selector(case, hist):
    """Learner boundary: this function sees no teacher parameters or joint risks."""
    X = np.asarray(case["contrast_X"], dtype=float)
    hist = np.asarray(hist, dtype=np.int64)
    assert X.shape == hist.shape == (81,) and np.isfinite(X).all()
    assert np.all(hist >= 0) and int(hist.sum()) == CFG["validation_n"]
    low, high = float(X.min()), float(X.max())
    R = high-low
    assert low == case["full_support_min"] and high == case["full_support_max"]
    assert R == case["full_support_range"]
    n, L = CFG["validation_n"], math.log(4*32/.05)
    if R == 0.:
        mean = float(X[0])
        variance = 0.
        radius = 0.
    else:
        mean = float(hist @ X/n)
        variance = float(hist @ ((X-mean)**2)/(n-1))
        assert variance >= 0
        radius = math.sqrt(2*variance*L/n)+7*R*L/(3*(n-1))
    lower, upper = mean-radius, mean+radius
    keep = lower > 0.
    return dict(mean=mean, unbiased_sample_variance=variance, n=n, L=L,
        full_support_range=R, radius=radius, lower=lower, upper=upper,
        keep_GP=keep, selected_model="GP" if keep else "H0",
        constant_full_support=(R == 0.),
        criterion="strict lower > 0", joint_oracle_used=False)


def generate_validation(case, profile_index):
    """Simulation service: only a count histogram is handed to the selector."""
    seal = read(OUT / "CANDIDATES_FREEZE.json")
    verify_hashes(seal["files"])
    assert seal["candidates_fixed_before_any_validation_rng"]
    profile, stream = case["profile"], case["stream"]
    teacher_receipt = read(C / f"streams/{profile}_s{stream:02}.json")
    true_mask = int(teacher_receipt["true_mask"])
    selected_mask = int(case["GP"]["selected_mask"])
    r = 8-(true_mask & selected_mask).bit_count()
    # This reads only the simulation PMF field; joint-risk fields are not used.
    probability = np.asarray(read(C / "RISK_LIBRARY.json")[profile]["mapped_teacher_p"][r], dtype=float)
    assert probability.shape == (81,) and np.all(probability > 0) and np.isfinite(probability).all()
    mass = float(probability.sum())
    assert abs(mass-1) <= CFG["mass_gate"]
    probability = probability/mass
    seed = CFG["validation_seed"]+1000*profile_index+stream
    rng = np.random.Generator(np.random.PCG64(seed))
    initial = rng.bit_generator.state
    hist = rng.multinomial(CFG["validation_n"], probability)
    selection = validation_selector(case, hist)
    return dict(profile=profile, stream=stream, seed=seed,
        initial_RNG=initial, final_RNG=rng.bit_generator.state,
        count_histogram=hist, n=CFG["validation_n"],
        candidates_freeze_sha256=sha(OUT / "CANDIDATES_FREEZE.json"),
        sampling_swap_r=r, raw_probability_mass_error=abs(mass-1),
        roundoff_normalization_factor=mass,
        validation_iid_conditional_on_training_and_teacher=True,
        selection=selection, joint_risk_not_used_for_selection=True)


def evaluate_after_selection(case, validation):
    """Post-selection oracle evaluation; its output is never passed to selector."""
    profile, r = case["profile"], validation["sampling_swap_r"]
    risks = np.asarray(read(C / "RISK_LIBRARY.json")[profile]["candidate_joint_KL_by_swap"])
    assert risks.shape == (9, 361)
    gp_index, h0_index = case["GP"]["candidate_index"], case["H0"]["candidate_index"]
    gp_risk, h0_risk = float(risks[r, gp_index]), float(risks[r, h0_index])
    assert np.max(abs(risks[:, h0_index]-risks[0, h0_index])) < 1e-10
    selected = gp_risk if validation["selection"]["keep_GP"] else h0_risk
    return dict(GP_joint_KL=gp_risk, H0_joint_KL=h0_risk, selected_joint_KL=selected,
        selected_minus_GP_joint_KL=selected-gp_risk,
        selected_minus_H0_joint_KL=selected-h0_risk,
        evaluation_only_not_selection=True)


def collect(cases):
    rows = []
    for case in cases:
        profile, stream = case["profile"], case["stream"]
        val = read(OUT / f"validation/{profile}_s{stream:02}.json")
        ev, sel = val["evaluation"], val["selection"]
        rows.append(dict(profile=profile, stream=stream,
            GP_u=case["GP"]["u"], GP_v=case["GP"]["v"],
            H0_scale=case["H0"]["u"],
            parameters_identical=case["GP_and_H0_parameters_identical"],
            mean=sel["mean"], sample_variance=sel["unbiased_sample_variance"],
            full_support_range=sel["full_support_range"], radius=sel["radius"],
            lower=sel["lower"], upper=sel["upper"], keep_GP=sel["keep_GP"],
            selected_model=sel["selected_model"], n=sel["n"], **ev))
    csvwrite(OUT / "VALIDATION_RESULTS.csv", rows)
    summary = []
    for profile in CFG["profiles"]:
        subset = [row for row in rows if row["profile"] == profile]
        summary.append(dict(profile=profile, cases=16,
            keep_GP=sum(row["keep_GP"] for row in subset),
            choose_H0=sum(not row["keep_GP"] for row in subset),
            identical_parameter_cases=sum(row["parameters_identical"] for row in subset),
            mean_GP_joint_KL=float(np.mean([row["GP_joint_KL"] for row in subset])),
            mean_H0_joint_KL=float(np.mean([row["H0_joint_KL"] for row in subset])),
            mean_selected_joint_KL=float(np.mean([row["selected_joint_KL"] for row in subset])),
            joint_KL_summary_is_descriptive=True))
    csvwrite(OUT / "SUMMARY.csv", summary)
    write(OUT / "DECISION.json", dict(cases=32, simultaneous_two_sided_confidence=.95,
        one_sided_events=64, log_factor=math.log(4*32/.05), summary=summary,
        new_validation_observations=8388608, validation_n_each=262144,
        new_H0_grid_fits=32, new_GP_fits=0, reused_GP_models=32,
        new_reference_integrations=0, new_prior_banks=0, new_neural_models=0,
        no_required_number_of_GP_retained=True, pair_gain_not_joint_KL_certificate=True,
        not_a_test_of_the_whole_homogeneous_class=True))


def audit():
    checkfreeze()
    seal = read(OUT / "CANDIDATES_FREEZE.json")
    verify_hashes(seal["files"])
    assert seal["source_freeze_sha256"] == sha(OUT / "FREEZE.json")
    saved = read(OUT / "CANDIDATES.json")["cases"]
    reconstructed = candidate_objects()
    assert len(saved) == len(reconstructed) == 32
    maximum = 0.
    keep = {profile: 0 for profile in CFG["profiles"]}
    for case, ref in zip(saved, reconstructed):
        assert (case["profile"], case["stream"]) == (ref["profile"], ref["stream"])
        assert case["GP"] == ref["GP"] and case["H0"]["candidate_index"] == ref["H0"]["candidate_index"]
        for key in ("training_objectives",):
            maximum = max(maximum, float(np.max(abs(np.array(case["H0"][key])-ref["H0"][key]))))
        for key in ("pair_loss_GP", "pair_loss_H0", "contrast_X"):
            maximum = max(maximum, float(np.max(abs(np.array(case[key])-ref[key]))))
        profile, stream = case["profile"], case["stream"]
        val = read(OUT / f"validation/{profile}_s{stream:02}.json")
        assert val["candidates_freeze_sha256"] == sha(OUT / "CANDIDATES_FREEZE.json")
        assert val["seed"] == 20360101+1000*CFG["profiles"].index(profile)+stream
        hist = np.array(val["count_histogram"])
        assert hist.shape == (81,) and np.issubdtype(hist.dtype, np.integer)
        assert np.all(hist >= 0) and int(hist.sum()) == 262144
        X = np.array(case["contrast_X"])
        R = float(X.max()-X.min())
        n, L = 262144, math.log(2560.)
        if R == 0:
            mean, var, radius = float(X[0]), 0., 0.
        else:
            mean = math.fsum(int(h)*float(x) for h, x in zip(hist, X))/n
            # A second variance identity over 81 occupied/support categories.
            pair_sum = math.fsum(int(hist[i])*int(hist[j])*float(X[i]-X[j])**2
                                for i in range(81) for j in range(i+1, 81))
            var = pair_sum/(n*(n-1))
            radius = math.sqrt(2*var*L/n)+7*R*L/(3*(n-1))
        chosen = mean-radius > 0
        sel = val["selection"]
        for key, value in (("mean", mean), ("unbiased_sample_variance", var),
                           ("full_support_range", R), ("radius", radius),
                           ("lower", mean-radius), ("upper", mean+radius)):
            maximum = max(maximum, abs(float(sel[key])-value))
        assert sel["keep_GP"] == chosen and sel["selected_model"] == ("GP" if chosen else "H0")
        assert sel["joint_oracle_used"] is False
        evaluation = evaluate_after_selection(case, val)
        for key, value in evaluation.items():
            if isinstance(value, bool):
                assert val["evaluation"][key] == value
            else:
                maximum = max(maximum, abs(val["evaluation"][key]-value))
        keep[profile] += int(chosen)
    assert maximum < CFG["arithmetic_gate"]
    decision = read(OUT / "DECISION.json")
    for row in decision["summary"]:
        assert row["keep_GP"] == keep[row["profile"]] and row["choose_H0"] == 16-keep[row["profile"]]
    assert decision["new_validation_observations"] == 32*262144
    return dict(passed=True, time_utc=now(), cases=32, H0_grid_argmins=32,
        independent_weighted_pair_variance_replay=True,
        maximum_replay_error=maximum, no_validation_resampling=True,
        no_new_reference_integrations=True,
        scope="implementation self-check plus alternate EB arithmetic; not an independent full-study audit")


def report():
    decision = read(OUT / "DECISION.json")
    lines = ["# D：独立验证选择结果", "",
        "32个GP模型固定复用，新增32个训练同质H0端点；全部loss差和全81支持range在validation前封存。每例262144个新验证观察，总8,388,608；0新GP拟合、0新参考积分、0新先验粒子。", "",
        "只有32项双侧同时EB区间的lower>0才保留GP，否则回退本次训练H0。它认证pair风险相对这个H0端点的改善，不能认证joint KL或真实异质性，不是整个同质类检验。", "",
        "|profile|保留GP|回退H0|同参数案例|原GP平均joint KL|H0平均joint KL|选择后平均joint KL|",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in decision["summary"]:
        lines.append(f"|{row['profile']}|{row['keep_GP']}/16|{row['choose_H0']}/16|"
            f"{row['identical_parameter_cases']}|{row['mean_GP_joint_KL']:.10g}|"
            f"{row['mean_H0_joint_KL']:.10g}|{row['mean_selected_joint_KL']:.10g}|")
    lines += ["", "全部32区间及个例选择见VALIDATION_RESULTS.csv；完整KL仅为选择后的qualified风险库评价，未参与选择，也未新增joint风险置信区间。次数和均值为本固定实验的描述，不外推普遍false-positive率或自然任务收益。", "",
        "理论保证条件于训练及真实教师；推送count可以随训练选中的GP组改变。实际验证分布为冻结数值PMF，未把reference精度门当作原解析Gaussian分布的全误差证书。", "",
        "原A/B/C结果和所有原失败保持。审计用另一加权两两差方差表达复算，不重采样；这是自带校验而非整包独立审查。", ""]
    (OUT / "READABLE.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def run():
    checkfreeze()
    assert read(OUT / "STATE.json")["status"] == "frozen", "no silent restart"
    write(OUT / "STATE.json", dict(status="fitting_H0", time_utc=now(), pid=os.getpid()), replace=True)
    cases = prepare_candidates()
    # No validation Generator has been created before this immutable receipt.
    assert read(OUT / "CANDIDATES_FREEZE.json")["candidates_fixed_before_any_validation_rng"]
    for case in cases:
        pi = CFG["profiles"].index(case["profile"])
        write(OUT / "STATE.json", dict(status="validating", profile=case["profile"],
            stream=case["stream"], time_utc=now(), pid=os.getpid()), replace=True)
        val = generate_validation(case, pi)
        val["evaluation"] = evaluate_after_selection(case, val)
        val["time_utc"] = now()
        write(OUT / f"validation/{case['profile']}_s{case['stream']:02}.json", val)
        print(json.dumps(dict(profile=case["profile"], stream=case["stream"],
            keep_GP=val["selection"]["keep_GP"], validation_n=262144)), flush=True)
    collect(cases)
    write(OUT / "AUDIT.json", audit())
    report()
    files = list((OUT / "validation").glob("*.json"))
    files += [OUT / n for n in ("CANDIDATES_FREEZE.json", "VALIDATION_RESULTS.csv",
                               "SUMMARY.csv", "DECISION.json", "AUDIT.json", "READABLE.md")]
    write(OUT / "RESULTS_FREEZE.json", dict(time_utc=now(), files=hashes(files),
        candidate_seal_sha256=sha(OUT / "CANDIDATES_FREEZE.json")))
    write(OUT / "STATE.json", dict(status="complete", active_jobs=[], time_utc=now()), replace=True)
    public = [p for p in OUT.rglob("*") if p.is_file() and p.name != "MANIFEST.json"]
    write(OUT / "MANIFEST.json", dict(time_utc=now(), files=hashes(public),
        new_full_references=0, new_banks=0, all_validation_counts_public=True))
    print(json.dumps(read(OUT / "DECISION.json")), flush=True)


def check():
    checkfreeze()
    for name in ("CANDIDATES_FREEZE.json", "RESULTS_FREEZE.json", "MANIFEST.json"):
        verify_hashes(read(OUT / name)["files"])
    assert read(OUT / "STATE.json")["status"] == "complete"
    print(json.dumps(audit()), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["freeze", "run", "check"])
    args = parser.parse_args()
    try:
        freeze() if args.phase == "freeze" else run() if args.phase == "run" else check()
    except Exception as error:
        if args.phase == "run" and not (OUT / "RESULTS_FREEZE.json").exists():
            write(OUT / f"FAILURE_{time.time_ns()}.json", dict(time_utc=now(),
                error=repr(error), traceback=traceback.format_exc()))
            write(OUT / "STATE.json", dict(status="stopped_failure", active_jobs=[], time_utc=now()), replace=True)
        raise
