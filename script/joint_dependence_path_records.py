"""Append-only correction of aliased theta fields in the path-check summary.

The original driver, RESULTS, immediate PATH records and training tapes are
preserved. No loss, gradient, qualification, optimizer state or sample changes.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"artifacts/joint_dependence_v1_20261007"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text("utf-8"))


def save(p, value):
    with Path(p).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def collect():
    records = []
    for rep in range(4):
        folder = OUT/f"rep_{rep:02d}"
        original = read(folder/"RESULTS.json")
        with np.load(folder/"TRAINING.npz", allow_pickle=False) as tape:
            theta = tape["theta"]
            source_hashes = {}
            corrected, affected = [], []
            for row in original["training"]["path_checks"]:
                step = row["step"]
                path = folder/f"PATH_{step:04d}.json"
                immediate = read(path)
                assert np.array_equal(np.asarray(immediate["theta"]), theta[step,2])
                diff = [k for k in row if row[k] != immediate[k]]
                if step < 1200:
                    assert diff == ["theta"], (rep, step, diff)
                    assert np.array_equal(np.asarray(row["theta"]), theta[-1,2])
                    affected.append(step)
                else:
                    assert diff == [], (rep, step, diff)
                corrected.append(immediate)
                source_hashes[path.relative_to(ROOT).as_posix()] = sha(path)
            assert [x["step"] for x in corrected] == [0,300,600,900,1200]
        for path in [folder/"RESULTS.json", folder/"TRAINING.npz",
                     OUT/"TRAIN_FREEZE.json", ROOT/"script/joint_dependence_train.py"]:
            source_hashes[path.relative_to(ROOT).as_posix()] = sha(path)
        assert sha(ROOT/"script/joint_dependence_train.py") == \
            "8bf07577a36f44ecc96634d29da671078d47a1da50cd16cc72ec049f6622ec34"
        records.append(dict(rep=rep, sources=source_hashes, corrected_path_checks=corrected,
                            affected_summary_steps=affected, only_field_changed="theta",
                            original_records_preserved=True,
                            scientific_outputs_or_qualifications_changed=False))
    # Regression fixture for the one-line forward fix: explicitly own the
    # snapshot instead of retaining a view into a mutable parameter array.
    live = np.arange(12., dtype=float).reshape(3,4)
    view_record = dict(theta=live[2])
    safe_record = dict(theta=np.array(live[2], copy=True))
    before = safe_record["theta"].copy()
    live[2] += 100
    assert not np.array_equal(view_record["theta"], before)
    assert np.array_equal(safe_record["theta"], before)
    return records


def main(mode):
    rows = collect()
    paths = [OUT/f"rep_{r['rep']:02d}/PATH_METADATA_CORRECTED.json" for r in rows]
    if mode == "write":
        for path,row in zip(paths,rows):
            save(path,row)
        save(OUT/"PATH_RECORDS_ERRATUM.json", dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            source_sha256=sha(__file__), status="corrected_by_original_immediate_records_and_tape",
            affected_streams=4, affected_theta_summary_fields=16,
            authoritative_records="Immediate PATH_*.json and matching TRAINING.npz GH states",
            cause="path_check kept a theta view; subsequent in-place Adam updates moved summary views to LAST",
            correction={p.relative_to(ROOT).as_posix():sha(p) for p in paths},
            loss_gradient_gate_or_training_changed=False, new_scientific_execution=False,
            frozen_driver_modified=False, original_RESULTS_modified=False,
            future_fix="Store theta=np.array(theta,copy=True) in a new, separately frozen driver version",
            forward_copy_regression_fixture_passed=True))
    else:
        for path,row in zip(paths,rows):
            assert read(path)==row
        receipt=read(OUT/"PATH_RECORDS_ERRATUM.json")
        assert receipt["source_sha256"]==sha(__file__)
        assert receipt["correction"]=={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
        assert receipt["affected_theta_summary_fields"]==16
        assert not receipt["loss_gradient_gate_or_training_changed"]
    print(json.dumps(dict(passed=True,mode=mode,affected_summary_fields=16,
                          original_scientific_outputs_unchanged=True)))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["write","check"])
    main(parser.parse_args().mode)
