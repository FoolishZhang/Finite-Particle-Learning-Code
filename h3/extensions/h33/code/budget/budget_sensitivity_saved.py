"""Frozen, descriptive decomposition of stored diagnostics; no model calls."""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import json
import math
import shutil
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.contract.read_text(encoding="utf-8"))
    assert cfg["schema"] == "budget_sensitivity_saved_diagnostics_v1"
    assert cfg["frozen"] and cfg["authorized_to_execute"] and __debug__
    assert sha(ROOT / "HANDOFF.md") == cfg["handoff_sha256"]
    assert sha(Path(__file__)) == cfg["source_sha256"]
    for name, digest in cfg["inputs"].items():
        assert sha(ROOT / name) == digest, name
    source = ROOT / cfg["diagnostics_csv"]
    source_manifest = json.loads((ROOT / cfg["qualification_manifest"]).read_text(encoding="utf-8"))
    original = next(row for row in source_manifest["files"] if row["path"] == source.name)
    assert sha(source) == original["sha256"] and source.stat().st_size == original["bytes"]
    assert source.stat().st_size <= cfg["max_source_bytes"]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.contract, out / "CONTRACT.json")
    shutil.copyfile(__file__, out / "SOURCE.py")
    save(out / "FREEZE.json", {"contract_sha256": sha(args.contract), "before_saved_arithmetic": True})
    started = time.perf_counter()
    with source.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == cfg["expected_rows"] == 576
    assert set(row["method"] for row in rows) == {"integrated", "group8", "ridge"}
    group = [row for row in rows if row["method"] == "group8"]
    assert len(group) == 192
    cells, enriched = {}, []
    for row in group:
        own = float(row["fit_mapping"])
        mapping, se = float(row["independent_mapping"]), float(row["max_gradient_SE"])
        assert all(math.isfinite(v) for v in (own, mapping, se))
        flags = (row["fit_status"] == "mapping_tolerance", mapping <= cfg["mapping_limit"],
                 se <= cfg["gradient_SE_limit"])
        assert all(flags) == (row["reference_diagnostic_passed"] == "True")
        failed = [name for name, ok in zip(("fit_status", "mapping", "SE"), flags) if not ok]
        category = "+".join(failed) or "pass"
        enriched.append({**row, "failure_components": category,
                         "numeric_own_mapping_pass": own <= cfg["own_mapping_limit"]})
    for cell in sorted({row["cell_id"] for row in group}):
        part = [row for row in enriched if row["cell_id"] == cell]
        assert len(part) == 48 and len({row["case_id"] for row in part}) == 24
        categories = collections.Counter(row["failure_components"] for row in part)
        metrics = {}
        for name in ("fit_mapping", "independent_mapping", "max_gradient_SE"):
            values = [float(row[name]) for row in part]
            metrics[name] = {"minimum": min(values), "median": statistics.median(values), "maximum": max(values)}
        cells[cell] = {"outputs": 48, "datasets": 24, "categories": dict(categories),
            "mapping_failed": sum("mapping" in row["failure_components"] for row in part),
            "SE_failed": sum("SE" in row["failure_components"] for row in part),
            "fit_status_failed": sum("fit_status" in row["failure_components"] for row in part),
            "numeric_own_mapping_failed": sum(not row["numeric_own_mapping_pass"] for row in part),
            "metrics": metrics}
    strong = [row for row in enriched if row["cell_id"] == "strong_N2048"]
    assert {int(row["case_id"]) for row in strong} == set(range(2000, 2024))
    with (out / "GROUP_COMPONENTS.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(enriched[0]))
        writer.writeheader()
        writer.writerows(enriched)
    elapsed = time.perf_counter() - started
    assert elapsed <= cfg["max_seconds"]
    summary = {"schema": cfg["schema"], "status": "completed", "all_fit_rows": len(rows),
        "all_fit_status_counts": dict(collections.Counter(row["fit_status"] for row in rows)),
        "all_fit_own_mapping_maximum": max(float(row["fit_mapping"]) for row in rows),
        "group_rows": 192, "cells": cells, "seconds": elapsed,
        "scope": "New descriptive decomposition of saved diagnostics only; output counts are not independent dataset counts.",
        "new_observations": 0, "new_particles": 0, "new_model_integrations": 0, "new_fits": 0,
        "source": {"path": cfg["diagnostics_csv"], "sha256": sha(source)}}
    save(out / "SUMMARY.json", summary)
    save(out / "MANIFEST.json", {"files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
         for p in sorted(out.iterdir()) if p.is_file()}})
    print(json.dumps({"passed": True, "strong": cells["strong_N2048"], "other_categories":
        {cell: value["categories"] for cell, value in cells.items() if cell != "strong_N2048"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
