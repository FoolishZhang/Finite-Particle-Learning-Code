"""Independent checks of published saved values; no sampling, integration or fits."""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
H3 = ROOT / "h3"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def close(actual, expected, tolerance=1e-12):
    if not math.isclose(float(actual), float(expected), rel_tol=1e-11, abs_tol=tolerance):
        raise AssertionError((actual, expected))


def identity():
    manifest = read(H3 / "MANIFEST.json")
    for name, info in manifest["files"].items():
        p = (ROOT / name).resolve()
        if not p.is_relative_to(H3) or p.is_symlink():
            raise ValueError("Unsafe manifest path")
        data = p.read_bytes()
        if len(data) != info["bytes"] or hashlib.sha256(data).hexdigest() != info["sha256"]:
            raise ValueError("Changed public input: " + name)
    return {"files_checked": len(manifest["files"])}


def saved_values():
    from scipy.stats import norm, t
    import numpy as np
    j1 = read(H3 / "data/j1/SUMMARY.json")
    data = rows(H3 / "data/j1/BANK_DIFFERENCES.csv")
    assert len(data) == 32768
    j1_checks = []
    for expected in j1["primary"]:
        selected = [r for r in data if int(r["M"]) == expected["M"]]
        assert len(selected) == 16384
        assert [int(r["replicate"]) for r in selected] == list(range(16384))
        key = "beta1_gap" if expected["arm"] == "beta1" else "tempered_gap"
        x = [float(r[key]) for r in selected]
        mean = statistics.mean(x)
        se = statistics.stdev(x) / math.sqrt(len(x))
        crit = float(norm.ppf(1 - .05 / 8))
        interval = [mean - crit * se, mean + crit * se]
        close(mean, expected["statistics"]["mean"])
        close(se, expected["statistics"]["se"])
        for a, b in zip(interval, expected["interval"]):
            close(a, b)
        j1_checks.append({"M": expected["M"], "proposal": expected["arm"],
                          "n": len(x), "mean": mean, "SE": se, "interval": interval})
    for r in data:
        assert float(r["tempered_minus_beta1"]) == float(r["tempered_gap"]) - float(r["beta1_gap"])
    local = H3 / "data/local_risk"
    summary = read(local / "SUMMARY.json")
    cases = rows(local / "CASE_METHOD_RISKS.csv")
    assert len(cases) == 576
    indexed = {}
    for r in cases:
        risks = json.loads(r["start_risks"])
        close(statistics.mean(risks), r["risk"])
        close(int(r["N"]) * float(r["risk"]), r["scaled_risk"])
        indexed[(int(r["case_id"]), r["method"])] = r
    primary = rows(local / "PRIMARY_CASE_VALUES.csv")
    assert len(primary) == 96
    checks = []
    for e in summary["primary_family"]:
        selected = [r for r in primary if r["comparison"] == e["name"]]
        assert len(selected) == 24 and len({r["case_id"] for r in selected}) == 24
        for r in selected:
            left = float(indexed[(int(r["case_id"]), e["left"])]["risk"])
            right = float(indexed[(int(r["case_id"]), e["right"])]["risk"])
            close(left - right, r["raw_difference"])
            close(2048 * (left - right), r["scaled_difference"])
        x = [float(r["scaled_difference"]) for r in selected]
        mean, se = statistics.mean(x), statistics.stdev(x) / math.sqrt(24)
        crit = float(t.ppf(1 - .05 / 8, 23))
        interval = [mean - crit * se, mean + crit * se]
        close(mean, e["scaled"]["mean"])
        close(se, e["scaled"]["SE"])
        for a, b in zip(interval, e["scaled"]["CI"]):
            close(a, b)
        checks.append({"comparison": e["name"], "n": 24, "mean": mean,
                       "SE": se, "interval": interval})
    points = rows(local / "PLOT_POINTS.csv")
    assert len(points) == 16
    for p in points:
        selected = [float(r["scaled_risk"]) for r in cases
                    if r["cell_id"] == p["cell_id"] and r["method"] == p["method"]]
        assert len(selected) == 24
        close(statistics.mean(selected), p["scaled_mean"])
        close(statistics.stdev(selected) / math.sqrt(24), p["scaled_SE"])
    diagnostics = rows(local / "FIT_DIAGNOSTICS.csv")
    assert len(diagnostics) == 576
    passed = collections.Counter(r["cell_id"] for r in diagnostics
        if r["method"] == "group8" and r["reference_diagnostic_passed"] == "True")
    assert dict(passed) == {"weak_N2048": 48, "mid_N2048": 44, "strong_N2048": 12, "mid_N256": 6}
    assert sum(r["reference_diagnostic_passed"] == "True" for r in diagnostics) == 494
    precision = rows(local / "RISK_PRECISION.csv")
    assert len(precision) == 864 and all(r["risk_qualified"] == "True" for r in precision)
    obs_index = read(H3 / "data/OBSERVATION_INDEX.json")
    assert len(obs_index) == 96 and sum(r["N"] for r in obs_index) == 153600
    for r in obs_index:
        p = (ROOT / r["path"]).resolve()
        assert p.is_relative_to(H3)
        x = np.loadtxt(p, delimiter=",", skiprows=1, dtype=np.int64)
        assert x.shape == (r["N"], 6)
        assert np.all((x[:, :3] >= 0) & (x[:, :3] <= 8))
        assert np.isin(x[:, 3:], [-1, 1]).all()
    outputs = read(local / "FIT_OUTPUTS.json")
    assert len(outputs) == 576
    assert set(collections.Counter((r["case_id"], r["method"]) for r in outputs).values()) == {2}
    return {"j1_comparisons": j1_checks, "local_comparisons": checks,
            "group_diagnostic_pass_counts": dict(passed), "risk_records": 864,
            "observations": 153600, "observation_cases": 96, "fit_outputs": 576,
            "scope": "Recomputed stored-value arithmetic, not risk integrals, sampling coverage, exact stationarity or fits."}


def imports():
    status = read(H3 / "CODE_STATUS.json")
    results = []
    for name, modules in status["import_sets"].items():
        source = H3 / "code" / name
        program = ("import sys,importlib,pathlib,json; "
            "p=pathlib.Path(sys.argv[1]).resolve();sys.path.insert(0,str(p));"
            "names=json.loads(sys.argv[2]);"
            "loaded=[importlib.import_module(n) for n in names];"
            "assert all(pathlib.Path(m.__file__).resolve().parent==p for m in loaded);"
            "print(json.dumps({'modules':names,'imported':len(loaded)}))")
        result = subprocess.run([sys.executable, "-I", "-B", "-c", program, str(source),
                                 json.dumps(modules)], check=True, capture_output=True, text=True)
        results.append({"namespace": name, **json.loads(result.stdout)})
    return {"sets": results, "scope": "Import-only; no original run/freeze/fit or model-integration function called."}


def main():
    if not __debug__:
        raise RuntimeError("Do not use -O or -OO")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["identity", "saved", "imports", "all"], default="all")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(ROOT) or output.exists():
        raise ValueError("Output must be new and outside this checkout")
    result = {"schema": "h3_public_saved_check_v1", "passed": False, "mode": args.mode,
              "identity": identity(), "new_scientific_rng": 0, "new_integrations": 0, "new_fits": 0}
    if args.mode in ("saved", "all"):
        result["saved_values"] = saved_values()
    if args.mode in ("imports", "all"):
        result["imports"] = imports()
    result["passed"] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    print(json.dumps({"passed": True, "mode": args.mode, "output": str(output)}))


if __name__ == "__main__":
    main()
