"""Call archived plotters on compact public saved values; no model calls."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args(); out = a.output_dir.resolve()
    if out.exists() or out.is_relative_to(ROOT):
        raise ValueError("new output directory outside source checkout required")
    out.mkdir(parents=True)
    env = os.environ.copy(); env["MPLCONFIGDIR"] = str(out / "matplotlib-config")
    commands = [
        [sys.executable, "-I", "-B", str(HERE / "code/budget/budget_sensitivity_plot.py"), "--analysis", str(HERE / "data/budget"), "--out", str(out / "budget")],
        [sys.executable, "-I", "-B", str(HERE / "code/learned/learned_precision_plot.py"), "--analysis", str(HERE / "data/precision/ANALYSIS_001.json"), "--out", str(out / "precision")],
    ]
    for command in commands:
        subprocess.run(command, env=env, check=True, capture_output=True, text=True)
    comparisons = []
    for p in sorted((HERE / "figures").rglob("*.png")):
        q = out / p.relative_to(HERE / "figures")
        digest = lambda x: hashlib.sha256(x.read_bytes()).hexdigest()
        comparisons.append({"figure": p.relative_to(HERE / "figures").as_posix(), "archived_sha256": digest(p), "redrawn_sha256": digest(q), "byte_equal": digest(p) == digest(q)})
    report = {"scope": "Original plot functions on saved aggregates; font and library versions may change bytes.", "new_model_calls": 0, "scientific_RNG": 0, "PNG_comparisons": comparisons}
    (out / "REDRAW.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
