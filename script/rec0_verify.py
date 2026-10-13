"""REC0: independently verify an analytic collision bound on frozen FREE TAIL scales.

No RNG, model integration, scoring, baseline fitting, or training.  --freeze must
precede --run; neither overwrites an existing scientific receipt.  --check only
replays the deterministic arithmetic and compares the saved result.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve()
OUT = ROOT / "artifacts/rec_v1_20261004/REC0"
DOC = ROOT / "docs/08_research_研究工作/REC_v1_20261004"
OLD = ROOT / "artifacts/joint_scale_learning"
TEACHER = ROOT / "artifacts/active_scale_boundary/REFERENCE/RESULTS.json"
EXPECTED_HEAD = "70fa7532651fc062c90d53f15735851b20df7840"
GUARD = 1e-12
ENVELOPE = Decimal("0.053")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def bound_decimal(tau: Decimal, n: int, k: int, precision: int) -> Decimal:
    assert tau >= 0 and n >= 0 and k >= 2
    with localcontext() as context:
        context.prec = precision
        square = tau * tau
        factor = (1 + square / 4) * (1 + 2 * (square / 2).exp() + (2 * square).exp()) / 4
        return +(Decimal(k) / Decimal(k - 1) ** 2 * factor ** n)


def bound_float(tau: float, n: int, k: int) -> float:
    square = tau * tau
    logfactor = math.log1p(square / 4) + math.log((1 + 2 * math.exp(square / 2) + math.exp(2 * square)) / 4)
    return math.exp(math.log(k) - 2 * math.log(k - 1) + n * logfactor)


def qualified_bound(tau: Decimal, n: int, k: int):
    d50, d80 = (bound_decimal(tau, n, k, precision) for precision in (50, 80))
    f64 = bound_float(float(tau), n, k)
    error = abs(f64 - float(d80))
    assert abs(d50 - d80) < Decimal("1e-45")
    assert error < GUARD
    return dict(tau_decimal=str(tau), upper_bound_decimal50=str(d50),
                upper_bound_decimal80=str(d80), upper_bound=float(d80),
                float64_upper_bound=f64, decimal_50_80_difference=str(abs(d50 - d80)),
                float64_decimal_difference=error)


def manufactured_checks():
    # Exhaustive, nonrandom finite support verifies the general inequality on
    # two examples.  The proof, not these examples, establishes the theorem.
    values = [Fraction(1), Fraction(2), Fraction(4)]
    ew2 = sum(w * w for w in values) / 3
    ewinv2 = sum(1 / (w * w) for w in values) / 3
    rows = []
    for k in (2, 3):
        chi = sum(sum(w * w for w in row) / sum(row) ** 2
                  for row in itertools.product(values, repeat=k)) / 3 ** k
        bound = Fraction(k, (k - 1) ** 2) * ew2 * ewinv2
        assert chi <= bound
        rows.append(dict(K=k, exact_chi=str(chi), exact_bound=str(bound)))
    zeros = qualified_bound(Decimal(0), 256, 100)
    assert abs(zeros["upper_bound"] - 100 / 99 ** 2) < GUARD
    return dict(passed=True, discrete_positive_iid_examples=rows,
                zero_scale_bound=zeros, exact_zero_scale_collision=0.01,
                note="The inverse-moment bound is deliberately loose at tau=0; chi=1/K exactly.")


def input_paths():
    return [OLD / "MANIFEST.json", OLD / "SPEC.json", TEACHER,
            ROOT / "artifacts/active_scale_boundary/MANIFEST.json",
            ROOT / "artifacts/paper_v91_20261004/HANDOFF_INPUT.md",
            *[OLD / "TRAIN" / f"stream_{stream:02}.json" for stream in range(16)]]


def freeze():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert head == EXPECTED_HEAD, head
    inputs = {rel(path): sha(path) for path in input_paths()}
    implementation = {rel(path): sha(path) for path in
                      [SOURCE, DOC / "CONTRACT_REC0.md", DOC / "REC0.md"]}
    save(OUT / "FREEZE.json", dict(schema_version="1", created_utc=now(),
        source_head=head, input_sha256=inputs, implementation_sha256=implementation,
        scope="REC0 deterministic bound and frozen panel only",
        budget=dict(new_random_draws=0, new_model_integrals=0, scoring_calls=0,
                    baseline_fits=0, training_updates=0, panel_outputs=32,
                    analytic_panel_bounds=32, envelope_bounds=1,
                    teacher_moment_eigendecompositions=3, teacher_orders=[24, 32, 40],
                    decimal_precisions=[50, 80], arithmetic_guard=GUARD,
                    manufactured_discrete_examples=2),
        command=".venv/Scripts/python.exe script/rec0_verify.py --freeze"))
    print(json.dumps(dict(frozen=True, path=rel(OUT / "FREEZE.json"))))


def verify_freeze():
    frozen = read(OUT / "FREEZE.json")
    for group in ("input_sha256", "implementation_sha256"):
        for name, digest in frozen[group].items():
            assert sha(ROOT / name) == digest, (group, name)
    return frozen


def moments(probability, patterns, delta):
    # Independent scalar sums: no call to the old moments_and_direction routine.
    p = np.array([math.fsum(float(probability[j]) * int(patterns[j, i])
                           for j in range(16)) for i in range(4)])
    gamma = np.array([[math.fsum(float(probability[j]) *
                       (int(patterns[j, i]) - p[i]) * (int(patterns[j, h]) - p[h])
                       for j in range(16)) for h in range(4)] for i in range(4)])
    d = p * (1 - p)
    ry = gamma / np.sqrt(np.outer(d, d))
    eigenvalues, eigenvectors = np.linalg.eigh(ry)
    lam, vector = float(eigenvalues[-1]), eigenvectors[:, -1]
    b = delta + (1 - delta) * lam
    n_squared = float(np.sum(vector ** 2 / d))
    residual = float(np.linalg.norm(ry @ vector - lam * vector))
    assert residual < GUARD
    return dict(p=p.tolist(), Gamma=gamma.tolist(), R_Y=ry.tolist(),
                lambda_max_R_Y=lam, B=b, collision_threshold=1 - 1 / b,
                eigen_residual_l2=residual, normalization_squared=n_squared)


def calculate():
    verify_freeze()
    spec, manifest = read(OLD / "SPEC.json"), read(OLD / "MANIFEST.json")
    config = spec["config"]
    assert (config["K"], config["nuisance"], config["eigen_floor"]) == (100, 256, 0.1)
    assert (config["streams"], config["tail_postupdates"], config["primary_output"]) == (16, 64, "TAIL")
    assert manifest["tracked_files"][rel(OLD / "SPEC.json")] == sha(OLD / "SPEC.json")
    assert spec["teacher_reference_path"] == rel(TEACHER)
    assert spec["teacher_reference_sha256"] == sha(TEACHER)
    old_teacher_manifest = read(ROOT / "artifacts/active_scale_boundary/MANIFEST.json")
    assert old_teacher_manifest["tracked_files"][rel(TEACHER)] == sha(TEACHER)
    reference = read(TEACHER)
    assert reference["qualification_passed"]
    mixed = next(row for row in reference["records"] if row["name"] == "mixed")
    patterns = np.asarray(reference["patterns"], dtype=int)
    assert patterns.shape == (16, 4)
    assert set(map(tuple, patterns)) == set(itertools.product((0, 1), repeat=4))
    teacher_rows = []
    for variant in mixed["teacher_variants"]:
        probability = variant["pmf"]
        assert len(probability) == 16 and min(probability) > 0
        assert abs(math.fsum(probability) - 1) < GUARD
        teacher_rows.append(dict(order=variant["order"], **moments(probability, patterns, .1)))
    assert [row["order"] for row in teacher_rows] == [24, 32, 40]
    main = teacher_rows[-1]
    old_threshold_error = abs(main["collision_threshold"] - mixed["derived"]["collision_threshold"])
    assert old_threshold_error < GUARD
    minimum_threshold = min(row["collision_threshold"] for row in teacher_rows)
    envelope = qualified_bound(ENVELOPE, 256, 100)
    rows = []
    for stream in range(16):
        path = OLD / "TRAIN" / f"stream_{stream:02}.json"
        assert manifest["tracked_files"][rel(path)] == sha(path)
        row = read(path)
        assert row["stream"] == stream
        assert row["source_sha256"] == spec["source_sha256"]
        # The old seal binds the source output tape; this stage reads the already
        # qualified TAIL JSON, and does not reconstruct TAIL or rerun training.
        assert manifest["local_arrays"][row["output_file"]] == row["output_sha256"]
        decimal_row = json.loads(path.read_text("utf-8"), parse_float=Decimal)
        for arm in ("NEAR_FREE", "UNIT_FREE"):
            tau = decimal_row["outputs"][arm]["TAIL"]["tau"]
            assert isinstance(tau, Decimal) and tau >= 0
            item = qualified_bound(tau, 256, 100)
            margin = minimum_threshold - item["upper_bound"]
            rows.append(dict(stream=stream, arm=arm, source=rel(path),
                output="TAIL", **item, envelope_covers=bool(tau <= ENVELOPE),
                conservative_threshold_gap=margin,
                bound_excludes_uniform_stability=bool(margin > 2 * GUARD)))
    assert len(rows) == 32
    # The claim can fail without failing execution; no added MC or larger budget.
    envelope_covers = all(row["envelope_covers"] for row in rows)
    envelope_gap = minimum_threshold - envelope["upper_bound"]
    conclusion = (envelope_covers and envelope_gap > 2 * GUARD and
                  all(row["bound_excludes_uniform_stability"] for row in rows))
    by_arm = {}
    for arm in ("NEAR_FREE", "UNIT_FREE"):
        selected = [row for row in rows if row["arm"] == arm]
        by_arm[arm] = dict(outputs=16, tau_min=min(float(r["tau_decimal"]) for r in selected),
                          tau_max=max(float(r["tau_decimal"]) for r in selected),
                          bound_min=min(r["upper_bound"] for r in selected),
                          bound_max=max(r["upper_bound"] for r in selected))
    return dict(schema_version="1", input_and_implementation_freeze_sha256=sha(OUT / "FREEZE.json"),
        analytic_statement="chi <= K/(K-1)^2 * [(1+tau^2/4)*(1+2exp(tau^2/2)+exp(2tau^2))/4]^n",
        parameters=dict(K=100, nuisance=256, delta=.1, panel=32, independent_data_units=16),
        manufactured_checks=manufactured_checks(), teacher_moments=teacher_rows,
        numerical_qualification=dict(arithmetic_guard=GUARD,
            saved_GH40_threshold_difference=old_threshold_error,
            threshold_max_order_change=max(abs(row["collision_threshold"] - main["collision_threshold"])
                                           for row in teacher_rows),
            scope="Decimal/eigen residual/order agreement are numerical qualifications, not interval or integration remainder certificates"),
        envelope=dict(**envelope, covers_all_outputs=envelope_covers,
                      conservative_threshold_gap=envelope_gap),
        panel_summary=by_arm, panel=rows,
        conclusion=dict(excludes_uniform_C_stability_at_virtual_true_product=conclusion,
            conditional_on="frozen numerical mixed teacher and ideal iid Gaussian-logistic prior law",
            limitations=["No statement about actual learned C, mu, nonzero active scales or finite-amplitude steps",
                         "No diagnosis of Adam or training-history causes",
                         "No empirical estimate of chi; bound is sufficient and may be loose",
                         "No rigorous error certificate for the exact continuous teacher's quadrature PMF"]),
        accounting=read(OUT / "FREEZE.json")["budget"])


def run():
    verify_freeze()
    save(OUT / "EXECUTION.json", dict(started_utc=now(), interpreter=sys.executable,
        source_sha256=sha(SOURCE), freeze_sha256=sha(OUT / "FREEZE.json"),
        command=".venv/Scripts/python.exe script/rec0_verify.py --run"))
    try:
        result = calculate()
        save(OUT / "RESULTS.json", result)
        save(OUT / "COMPLETION.json", dict(completed_utc=now(), passed=True,
            result_sha256=sha(OUT / "RESULTS.json"), freeze_sha256=sha(OUT / "FREEZE.json")))
        print(json.dumps(dict(completed=True, conclusion=result["conclusion"],
                             envelope=result["envelope"], panel_summary=result["panel_summary"]), indent=2))
    except Exception:
        save(OUT / "FAILURE.json", dict(failed_utc=now(), traceback=traceback.format_exc()))
        raise


def check():
    saved, completion = read(OUT / "RESULTS.json"), read(OUT / "COMPLETION.json")
    assert sha(OUT / "RESULTS.json") == completion["result_sha256"]
    assert calculate() == saved, "Deterministic arithmetic replay differs"
    print(json.dumps(dict(passed=True, scope="read-only deterministic replay, no new statistical units",
                          result_sha256=sha(OUT / "RESULTS.json"))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    freeze() if args.freeze else run() if args.run else check()
