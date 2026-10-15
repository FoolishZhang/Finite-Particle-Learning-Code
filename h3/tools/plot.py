"""Reuse the original figure functions on public saved aggregates; never fit."""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import os

ROOT = Path(__file__).resolve().parents[2]
H3 = ROOT / "h3"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.is_relative_to(ROOT):
        raise ValueError("Output must be outside the source checkout")
    out.mkdir(parents=True, exist_ok=False)
    os.environ["MPLCONFIGDIR"] = str(out / "matplotlib-config")
    import numpy as np
    source = H3 / "code/figures/h32_figures_source.py"
    spec = importlib.util.spec_from_file_location("original_h32_figures", source)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    j1 = read(H3 / "data/j1/SUMMARY.json")
    local = read(H3 / "data/local_risk/SUMMARY.json")
    first = [{"M": r["M"], "arm": r["arm"], "mean": r["statistics"]["mean"],
              "interval": r["interval"]} for r in j1["primary"]]
    with (H3 / "data/local_risk/PLOT_POINTS.csv").open(encoding="utf-8", newline="") as f:
        points = [{"cell_id": r["cell_id"], "N": int(r["N"]), "S": float(r["S"]),
            "method": r["method"], "scaled_mean": float(r["scaled_mean"]),
            "scaled_SE": float(r["scaled_SE"])} for r in csv.DictReader(f)]
    r = local["theory_coefficients"]["r_K8"]
    limits = [3 * r / (2 - r), 3 * (1 + r) / (1 - r)]
    grid = np.linspace(0, 12.5, 501)
    theory = {"group8": .5 * ((1-r)**2 * grid + 3*r*r),
              "integrated": np.full_like(grid, 1.5), "I": .5 * grid}
    mod.style()
    mod.figure1(first, out)
    mod.figure2(points, grid, theory, limits, out)
    mod.figure3(points, grid, theory, limits, out)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    names = ["proposal_reversal.png", "local_risk.png", "local_risk_n256.png"]
    result = {"schema": "h3_public_plot_v1", "original_figure_source_sha256": sha(source),
              "figures": {n: {"sha256": sha(out / n),
                  "matches_archived_png_bytes": sha(out / n) == sha(H3 / "figures" / n)} for n in names},
              "permitted_arithmetic": "Saved means/SE/CI endpoints and existing local-risk formula on a display grid.",
              "new_random_samples": 0, "new_integrations": 0, "new_fits": 0,
              "new_sample_statistics_or_intervals": 0,
              "pixel_identity_is_platform_dependent": True}
    (out / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figures": names, "output_dir": str(out)}))


if __name__ == "__main__":
    main()
