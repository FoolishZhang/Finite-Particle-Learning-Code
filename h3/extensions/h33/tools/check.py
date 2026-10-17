"""Independent saved-replicate arithmetic; no model/RNG/integration imports."""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import json
import subprocess
import sys
import traceback
import numpy as np
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
COUNT = 0
ALPHA = .3201223434446316
V = np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0.])


def require(ok, label):
    global COUNT
    COUNT += 1
    if not ok:
        raise ValueError(label)


def close(actual, expected, label, atol=2e-11):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            and np.allclose(a, b, rtol=2e-10, atol=atol), label)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def csv_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def public_path(name):
    p = (ROOT / name).resolve()
    require(p.is_relative_to(ROOT) and not p.is_symlink(), "unsafe public path")
    return p


def identity():
    m = read(HERE / "MANIFEST.json")
    for name, row in m["files"].items():
        p = public_path(name)
        require(p.is_relative_to(HERE), "extension manifest leaves extension")
        blob = p.read_bytes()
        require(len(blob) == row["bytes"] and hashlib.sha256(blob).hexdigest() == row["sha256"], "changed input " + name)
    return {"files": len(m["files"]), "manifest_sha256": hashlib.sha256((HERE / "MANIFEST.json").read_bytes()).hexdigest()}


def moments(x):
    a = np.asarray(x, dtype=float)
    require(len(a) > 1 and np.isfinite(a).all(), "nonfinite or incomplete replicate vector")
    return a.mean(axis=0), a.std(axis=0, ddof=1) / np.sqrt(len(a))


def stat(saved, x, label):
    a = np.asarray(x, dtype=float)
    if "R" in saved:
        require(saved["R"] == len(a), label + ".R")
    m, se = moments(a)
    close(m, saved["mean"], label + ".mean")
    close(se, saved["SE"], label + ".SE")
    if "sample_SD" in saved:
        close(se * np.sqrt(len(a)), saved["sample_SD"], label + ".SD")
    if "replicates" in saved:
        close(a, saved["replicates"], label + ".replicates")


def walk_statistics(obj, label):
    if isinstance(obj, dict):
        if all(k in obj for k in ("replicates", "mean", "SE")):
            stat(obj, obj["replicates"], label)
        for k, v in obj.items():
            walk_statistics(v, label + "/" + k)
    elif isinstance(obj, list):
        for i, x in enumerate(obj):
            walk_statistics(x, label + "/" + str(i))


def interval(saved, x, family, label):
    a = np.asarray(x); m, se = moments(a); df = len(a) - 1
    critical = float(student_t.ppf(1 - .05 / (2 * family), df))
    close([float(m - critical * se), float(m + critical * se)], [saved["lower"], saved["upper"]], label)
    require(saved["df"] == df, label + ".df")
    require(saved.get("family", saved.get("family_size")) == family, label + ".family")
    close(critical, saved.get("critical", saved.get("critical_value")), label + ".critical")


def prediction(statistics, prefix, amplitude):
    g = np.asarray(statistics[prefix + "_g"]["replicates"])
    H = np.asarray(statistics[prefix + "_H"]["replicates"])
    B = np.asarray(statistics[prefix + "_B"]["replicates"])
    require(g.shape == (16, 3) and H.shape == B.shape == (16, 3, 3), "full g/H/B replicate shape")
    linear = (ALPHA - 1) * (g @ V)
    h = .5 * (ALPHA ** 2 - 1) * np.einsum("i,rij,j->r", V, H, V)
    b = -(ALPHA - 1) * np.einsum("i,rij,j->r", V, B, V)
    return amplitude * linear + amplitude ** 2 * (h + b), (linear, h, b)


def budget():
    base = HERE / "data/budget"
    risk = read(base / "RISK_ANALYSIS.json")
    raw = risk["raw_risk_rows"]
    expected = {(c, s, m, i) for c in range(2000, 2024) for s in ("P0", "H") for m in ("integrated", "group8", "ridge") for i in (0, 1)}
    indexed = {(r["case_id"], r["setting"], r["method"], r["initialization_index"]): r["raw"] for r in raw}
    require(len(raw) == len(indexed) == 288 and set(indexed) == expected, "all 288 budget risk slots")
    for key, r in indexed.items():
        require(r["precision_qualified"] and r["included_without_diagnostic_filter"], "risk retained/qualified")
        close(r["joint_kl"] * 2048, r["scaled_joint_kl"], "scaled risk")
        require([x["orders"] for x in r["risk_levels"]] == [[28, 192], [36, 256]], "risk rule identities")
        close(r["joint_kl"], r["risk_levels"][1]["joint_kl"], "common final rule")
        require(abs(r["risk_levels"][0]["joint_kl"] - r["risk_levels"][1]["joint_kl"]) <= 4.8828125e-8, "ordinary KL order gate")
    rows = risk["paired_case_rows"]
    require([r["case_id"] for r in rows] == list(range(2000, 2024)), "all paired cases in fixed order")
    for row in rows:
        c = row["case_id"]
        for s in ("P0", "H"):
            for m in ("integrated", "group8", "ridge"):
                val = 2048 * np.mean([indexed[c, s, m, i]["joint_kl"] for i in (0, 1)])
                close(val, row[f"{s}_{m}_scaled_risk"], "average risks before dataset comparison")
            close(row[f"{s}_group8_scaled_risk"] - row[f"{s}_integrated_scaled_risk"], row["D_" + s], "dataset difference")
        close(row["D_H"] - row["D_P0"], row["G"], "paired budget change")
    for key in ("D_H", "G"):
        x = [r[key] for r in rows]
        interval(risk["primary"][key]["simultaneous_interval"], x, 2, "budget." + key)
    for a, b in zip(rows, csv_rows(base / "PAIRED_CASES.csv")):
        for k in ("case_id", "D_P0", "D_H", "G"):
            close(a[k], b[k], "paired CSV." + k)
    diagnostics = read(base / "DIAGNOSTICS.json")
    require(len(diagnostics["rows"]) == 96, "all new diagnostic rows")
    require({(r["case_id"], r["setting"], r["initialization_index"]) for r in diagnostics["rows"]} ==
            {(c, s, i) for c in range(2000, 2024) for s in ("P0", "H") for i in (0, 1)}, "diagnostic identities")
    counts = collections.Counter()
    for r in diagnostics["rows"]:
        mp, se = r["mapping_norm_inf"] <= .0003, r["maximum_gradient_SE"] <= .0001
        require(mp == r["mapping_pass"] and se == r["SE_pass"], "diagnostic numeric gates")
        passed = mp and se and r["own_status_pass"]
        require(passed == r["composite_pass"], "diagnostic conjunction")
        counts[r["setting"]] += int(passed)
    require(dict(counts) == {"P0": 36, "H": 48}, "new diagnostic counts")
    old = csv_rows(base / "original_diagnostics/GROUP_COMPONENTS.csv")
    require(len(old) == 192, "original diagnostic rows retained")
    categories = collections.Counter()
    for r in old:
        if r["cell_id"] != "strong_N2048":
            continue
        mp, se = float(r["independent_mapping"]) <= .0003, float(r["max_gradient_SE"]) <= .0001
        category = "pass" if mp and se else ("SE" if mp else ("mapping" if se else "mapping+SE"))
        require(category == r["failure_components"], "old failure decomposition")
        categories[category] += 1
    require(dict(categories) == {"pass": 12, "mapping": 2, "SE": 20, "mapping+SE": 14}, "original diagnostic identity")
    return {"datasets": 24, "risk_rows": 288, "new_diagnostic_rows": 96, "family": 2, "df": 23, "diagnostic_passes": [12, 36, 48]}


def training(index):
    freeze = read(HERE / "data/training/FROZEN_PROPOSALS.json")
    require(len(freeze["pairs"]) == 6 and len(index["training"]) == 6, "six nested training pairs")
    pairs = {(r["rep"], r["N"]): r for r in freeze["pairs"]}
    for row in index["training"]:
        saved = pairs[row["rep"], row["N"]]
        selection = read(public_path(row["SELECTION.json"]))
        scores = selection["scores"]
        require([s["epoch"] for s in scores] == [0, 5, 20, 80], "all checkpoint scores")
        for s in scores:
            close((s["raw_mean_NLL"] + s["sym_mean_NLL"]) / 2, s["paired_mean_NLL"], "paired validation score")
        chosen = min(scores, key=lambda x: (x["paired_mean_NLL"], x["epoch"]))["epoch"]
        require(chosen == saved["selected_epoch"] == selection["selected_epoch"], "checkpoint selected only by saved validation scores")
        require(selection["candidate_teacher_or_target_comparison_used"] is False, "no target selection")
        raw = read(public_path(row["PROPOSAL_raw.json"]))
        sym = read(public_path(row["PROPOSAL_sym.json"]))
        for k in ("weights", "means", "stds"):
            close(raw["raw_parameters"][k], saved["raw_parameters"][k], "freeze raw parameter")
            close(selection["raw_parameters"][k], saved["raw_parameters"][k], "selection parameter")
            a = np.asarray(raw["parameters"][k]); require(a.shape == (18, 2), "raw mixture shape")
            other = a[::-1] * (-1 if k == "means" else 1)
            reflected = np.concatenate([a, other], axis=1) * (.5 if k == "weights" else 1)
            # Saved canonical coupling keeps component order when reflecting
            # states 0..8 into 17..9; swapping halves is density-equivalent
            # but is not the actual serialized parameter representation.
            reflected[9:] = reflected[:9][::-1] * (-1 if k == "means" else 1)
            close(reflected, sym["parameters"][k], "reflection mixture construction")
        require(raw["defense"] == sym["defense"] == .05, "fixed defensive mass")
        require(np.all(np.asarray(raw["parameters"]["weights"]) >= .02) and np.all(np.asarray(raw["parameters"]["weights"]) <= .98), "raw weight bounds")
        close(np.asarray(raw["parameters"]["weights"]).sum(axis=1), np.ones(18), "raw weight normalization")
    return {"training_pools": 3, "nested_sizes": [16384, 65536], "raw_sym_pairs": 6, "training_replayed": False}


def old_family():
    a = read(HERE / "data/previous/ANALYSIS.json")
    require(len(a["primary_cells"]) == 12 and len(a["raw_cells"]) == 24 and len(a["coefficient_cells"]) == 12 and len(a["raw_odd_even"]) == 12, "complete historical schedule")
    walk_statistics(a, "historical")
    coef = {(c["rep"], c["N"], c["kind"]): c["statistics"] for c in a["coefficient_cells"]}
    passing = 0
    for c in a["primary_cells"] + a["raw_cells"]:
        require(c["status"] == "available", "historical row retained")
        p, parts = prediction(coef[c["rep"], c["N"], c["kind"]], "cv", c["t"])
        D = np.asarray(c["D"]["replicates"])
        stat(c["prediction"], p, "old full-matrix prediction")
        stat(c["normalized_residual"], (D - p) / c["t"] ** 2, "old paired residual")
        if c["kind"] == "sym":
            interval(c["D_simultaneous_interval"], D, 24, "historical D")
            w = c["normalized_residual_simultaneous_interval"]
            interval(w, (D - p) / c["t"] ** 2, 24, "historical W")
            passing += int(w["lower"] >= -.01 and w["upper"] <= .01)
    require(passing == 1, "historical 1/12 preserved")
    for c in a["raw_odd_even"]:
        s = c["statistics"]; t = c["abs_t"]
        dp, dm = np.asarray(s["D_positive"]["replicates"]), np.asarray(s["D_negative"]["replicates"])
        stat(s["odd_part"], (dp - dm) / 2, "same-stream raw odd")
        stat(s["even_part"], (dp + dm) / 2, "same-stream raw even")
        stat(s["odd_coefficient"], (dp - dm) / (2 * t), "raw odd coefficient")
        stat(s["even_coefficient"], (dp + dm) / (2 * t * t), "raw even coefficient")
    return {"family": 24, "df": 15, "primary_cells": 12, "precision_passes": 1, "raw_descriptive_cells": 24, "new_raw_significance_tests": False}


def precision(index):
    a = read(HERE / "data/precision/ANALYSIS_001.json")
    freeze = read(HERE / "data/precision/COEFFICIENTS_AND_PREDICTIONS_FROZEN_001.json")
    require(len(a["cells"]) == 12 and len(index["precision_statistics"]) == 54, "complete new schedule")
    require(freeze["component_scaling"] == "coefficients_before_t_scaling", "frozen component units")
    config_c = read(HERE / "data/precision/COEFFICIENTS_CONFIG.json")
    config_v = read(HERE / "data/precision/VALIDATION_CONFIG.json")
    for cfg, phase, B in ((config_c, "coefficients", 256), (config_v, "validation", 512)):
        require(cfg["phase"] == phase and cfg["K"] == 8 and cfg["randomization"]["R"] == 16 and cfg["randomization"]["B"] == B, "scientific phase and R/B settings")
        close(cfg["alpha"], ALPHA, "fixed alpha"); close(cfg["direction"], V, "fixed full direction")
    require(config_c["randomization"]["family"] != config_v["randomization"]["family"], "independent numerical families")
    development = read(HERE / "data/precision/DEVELOPMENT_DECISION_001.json")
    walk_statistics(development, "development")
    require(len(development["cases"]) == 6 and development["selected_estimator"] == "q2", "all six control-development cases retained")
    lookup = {(r["phase"], r["rep"], r["relative"]): read(public_path(r["path"])) for r in index["precision_statistics"]}
    for key, d in lookup.items():
        walk_statistics(d, str(key))
        if key[2].endswith("COEFFICIENT_STATISTICS.json"):
            s = d["statistics"]
            close(s["q2_g"]["replicates"], s["cv_g"]["replicates"], "q2 unchanged g")
            close(s["q2_B"]["replicates"], s["cv_B"]["replicates"], "q2 unchanged B")
            close(np.asarray(s["cv_H"]["replicates"]) + np.asarray(d["control_statistics"]["origin"]["replicates"]), s["q2_H"]["replicates"], "q2 H plus C at replicate level")
    frozen = {(r["rep"], r["N"], r["t"]): r for r in freeze["predictions"]}
    expected = {(r, n, t) for r in range(3) for n in (16384, 65536) for t in (.1, .2)}
    require(set(frozen) == expected and {(c["rep"], c["N"], c["t"]) for c in a["cells"]} == expected, "all 12 identities")
    pass_ind = pass_same = negative = nonzero = 0
    for c in a["cells"]:
        rep, n, t = c["rep"], c["N"], c["t"]
        base = f"rep_{rep}_N{n}_sym/"; tag = "t0p1" if t == .1 else "t0p2"
        independent = lookup["coefficients", rep, base + "COEFFICIENT_STATISTICS.json"]
        same = lookup["validation", rep, base + "COEFFICIENT_STATISTICS.json"]
        contrast = lookup["validation", rep, base + "CONTRAST_" + tag + ".json"]
        remainder = lookup["validation", rep, base + "REMAINDER_" + tag + ".json"]
        pi, components = prediction(independent["statistics"], "q2", t)
        ps, _ = prediction(same["statistics"], "q2", t)
        f = frozen[rep, n, t]
        for key in ("bank_id", "reference_identity", "reference_p0_sha256", "reference_dp0_sha256"):
            require(f[key] == independent[key], "frozen coefficient binding " + key)
        for obj in (contrast, remainder):
            require(obj["bank_id"] == same["bank_id"] and obj["reference_identity"] == same["reference_identity"], "same-bank and finite-reference identities")
        close(pi, f["replicates"], "frozen independent prediction")
        for key, values in zip(("linear_replicates", "H_quadratic_replicates", "B_quadratic_replicates"), components):
            close(values, f[key], "frozen unscaled component")
        D = np.asarray(contrast["statistics"]["q2_D"]["replicates"])
        wi, ws = (D - pi) / t ** 2, (D - ps) / t ** 2
        for key, values in (("D", D), ("independent_prediction", pi), ("same_prediction", ps), ("W_ind", wi), ("W_same", ws)):
            stat(c[key], values, "new " + key)
        stat(remainder["statistics"]["q2_prediction"], ps, "same bank prediction")
        stat(remainder["statistics"]["q2_normalized_remainder"], ws, "same bank remainder")
        for key, values in (("D", D), ("W_ind", wi), ("W_same", ws)):
            interval(c[key + "_interval"], values, 36, key + " family36")
        close(np.cov(D, pi, ddof=1)[0, 1], c["cov_D_Pind"], "paired index covariance")
        close(np.cov(D, ps, ddof=1)[0, 1], c["cov_D_Psame"], "same-bank covariance")
        close(np.var(ws, ddof=1) / np.var(wi, ddof=1), c["same_over_independent_residual_variance"], "paired residual variance ratio")
        negative += c["D_interval"]["upper"] < 0
        pass_ind += c["W_ind_interval"]["lower"] >= -.01 and c["W_ind_interval"]["upper"] <= .01
        pass_same += c["W_same_interval"]["lower"] >= -.01 and c["W_same_interval"]["upper"] <= .01
        nonzero += c["W_same_interval"]["upper"] < 0 or c["W_same_interval"]["lower"] > 0
    require((negative, pass_ind, pass_same, nonzero) == (12, 12, 12, 12), "new complete outcome counts")
    # Preselected single proposal, full replicate differences, not six-proposal certification.
    base = "rep_0_N16384_sym/"
    rule = lookup["validation", 0, base + "RULE_ORDER_COMPARISON.json"]
    for t, tag in ((.1, "t0p1"), (.2, "t0p2")):
        d36 = lookup["validation", 0, base + "CONTRAST_" + tag + ".json"]
        d28 = lookup["validation", 0, base + "GH28/CONTRAST_" + tag + ".json"]
        w36 = lookup["validation", 0, base + "REMAINDER_" + tag + ".json"]
        w28 = lookup["validation", 0, base + "GH28/REMAINDER_" + tag + ".json"]
        ds = (np.asarray(d36["statistics"]["q2_D"]["replicates"]) - np.asarray(d28["statistics"]["q2_D"]["replicates"])) / t ** 2
        ws = np.asarray(w36["statistics"]["q2_normalized_remainder"]["replicates"]) - np.asarray(w28["statistics"]["q2_normalized_remainder"]["replicates"])
        saved_rule = next(r for r in rule["rows"] if r["t"] == t)
        stat(saved_rule["D_normalized_rule_difference"], ds, "GH D paired difference")
        stat(saved_rule["W_same_rule_difference"], ws, "GH W paired difference")
        require(np.max(np.abs(ds)) <= 1e-4 and np.max(np.abs(ws)) <= 1e-4, "single-proposal ordinary GH rule comparison")
    return {"family": 36, "df": 15, "cells": 12, "D_negative": int(negative), "W_ind_precision": int(pass_ind), "W_same_precision": int(pass_same), "nonzero_same_remainders": int(nonzero), "source_R_statistic_JSONs": 54,
            "scope": "conditional on six fixed proposals; R-level arithmetic, not per-bank identity or model evaluation"}


def imports():
    results = []
    for name in ("budget", "learned"):
        folder = HERE / "code" / name
        modules = sorted(p.stem for p in folder.glob("*.py"))
        program = "import sys,importlib,pathlib,json;p=pathlib.Path(sys.argv[1]).resolve();sys.path.insert(0,str(p));names=json.loads(sys.argv[2]);m=[importlib.import_module(n) for n in names];assert all(pathlib.Path(x.__file__).resolve().parent==p for x in m);print(json.dumps({'modules':names,'count':len(m)}))"
        p = subprocess.run([sys.executable, "-I", "-B", "-c", program, str(folder), json.dumps(modules)], check=True, capture_output=True, text=True)
        results.append({"namespace": name, **json.loads(p.stdout)})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("identity", "saved", "imports", "all"), default="all")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    require(not out.exists() and not out.is_relative_to(ROOT), "new output outside checkout required")
    report = {"schema": "h33_public_saved_check_v1", "passed": False, "mode": args.mode,
              "new_model_calls": 0, "new_training": 0, "scientific_RNG": 0,
              "scope": "Public file identity, saved R-level arithmetic and optional isolated imports only; not proof, quadrature, optimizer or training replay."}
    try:
        report["identity"] = identity()
        if args.mode in ("saved", "all"):
            index = read(HERE / "data/INDEX.json")
            report.update(budget=budget(), training=training(index), previous=old_family(), precision=precision(index))
        if args.mode in ("imports", "all"):
            report["imports"] = imports()
        report["passed"] = True
    except Exception:
        report["error"] = traceback.format_exc()
    report["checks"] = COUNT
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False); f.write("\n")
    print(json.dumps({"passed": report["passed"], "checks": COUNT, "output": str(out)}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
