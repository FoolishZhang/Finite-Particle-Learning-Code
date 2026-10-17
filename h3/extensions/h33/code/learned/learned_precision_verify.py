"""Independent arithmetic verification of saved learned-precision products.

Reads SHA-bound JSON and derived NPZ only. Never imports the scientific core or
analyzer, and never reads particle samples/local weights or executes a model,
quadrature, optimizer, or RNG. Uses complete-randomization means as SE units.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import traceback

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "script/learned_precision_verify.py"
ANALYZER = "script/learned_precision_analyze.py"
SCHEMA = "learned_precision_saved_verification_v1"
ALPHA = .3201223434446316
V = np.array([1/math.sqrt(2), 1/math.sqrt(2), 0.])
R = 16
CASES = {(r, n) for r in range(3) for n in (16384, 65536)}
TIMES = (.1, .2)
METRICS = tuple(prefix+"_"+name for prefix in ("raw", "cv", "q2") for name in ("g", "H", "B"))
ARRAY_FORBIDDEN = ("SAMPLES", "LOCAL_WEIGHTS", "UNIFORMS", "PHYSICAL_DATA")
ATOL, RTOL = 3e-13, 3e-11


def require(ok, message):
    if not ok:
        raise ValueError(message)


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def path(name):
    rel = Path(name)
    require(not rel.is_absolute() and ".." not in rel.parts, "repository-relative bound path required")
    result = (ROOT/rel).resolve()
    require(result.is_relative_to(ROOT), "path escapes repository")
    return result


def safe_input(name):
    p = path(name)
    if p.suffix.lower() == ".npz":
        require(not any(word in p.name.upper() for word in ARRAY_FORBIDDEN),
                "original samples/local-weight/RNG arrays forbidden to this verifier: "+name)
    return p


def stable_mean(a, axis):
    a = np.moveaxis(np.asarray(a, float), axis, 0)
    n, trailing = a.shape[0], a.shape[1:]
    flat = a.reshape(n, -1)
    return np.array([math.fsum(flat[:, j])/n for j in range(flat.shape[1])]).reshape(trailing)


def moments(replicates):
    a = np.asarray(replicates, float)
    require(a.shape[0] == R and np.isfinite(a).all(), "16 finite complete randomizations required")
    mean = stable_mean(a, 0)
    centered = a-mean
    flat = centered.reshape(R, -1)
    variance = np.array([math.fsum(flat[:, j]*flat[:, j])/(R-1) for j in range(flat.shape[1])]).reshape(a.shape[1:])
    sd = np.sqrt(variance)
    return {"replicates": a, "mean": mean, "sample_SD": sd, "SE": sd/4, "R": R}


def bank_moments(banks):
    a = np.asarray(banks, float)
    require(a.ndim >= 2 and a.shape[0] == R and a.shape[1] >= 2 and np.isfinite(a).all(), "finite [R16,B,...] arrays required")
    return {**moments(stable_mean(a, 1)), "B": a.shape[1]}


def variance(a):
    return float(np.asarray(moments(a)["sample_SD"])**2)


def cov(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    require(a.shape == b.shape == (R,), "two scalar complete-randomization arrays required")
    ac, bc = a-float(stable_mean(a, 0)), b-float(stable_mean(b, 0))
    return float(math.fsum(ac*bc)/(R-1))


def critical():
    return float(student_t.ppf(1-.05/72, 15))


def ci(stat):
    width = critical()*float(stat["SE"])
    return {"lower": float(stat["mean"])-width, "upper": float(stat["mean"])+width,
            "critical": critical(), "df": 15, "family": 36, "alpha": .05}


def classify(interval):
    lo, hi = interval["lower"], interval["upper"]
    return "within_tolerance" if lo >= -.01 and hi <= .01 else (
        "outside_tolerance" if lo > .01 or hi < -.01 else "unresolved")


def components(arrays, prefix, amplitude):
    """Explicit full-matrix contractions, independently of the production einsums."""
    g, h, b = [np.asarray(arrays[prefix+"_"+name], float) for name in ("g", "H", "B")]
    require(g.shape[-1] == 3 and h.shape == b.shape == g.shape[:-1]+(3, 3), "full g/H/B dimensions")
    linear = (ALPHA-1)*(g @ V)
    hpart = .5*(ALPHA**2-1)*((h @ V) @ V)
    bpart = -(ALPHA-1)*((b @ V) @ V)
    return {"linear_coefficient": linear, "H_quadratic_coefficient": hpart,
            "B_quadratic_coefficient": bpart, "quadratic_coefficient": hpart+bpart,
            "prediction": amplitude*linear+amplitude**2*(hpart+bpart)}


class Checks:
    def __init__(self):
        self.rows = []

    def exact(self, label, actual, expected):
        passed = actual == expected
        self.rows.append({"name": label, "passed": bool(passed), "actual": actual, "expected": expected})
        return bool(passed)

    def close(self, label, actual, expected, *, atol=ATOL, rtol=RTOL):
        a, b = np.asarray(actual, float), np.asarray(expected, float)
        finite = a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        error = float(np.max(abs(a-b))) if finite and a.size else 0. if finite else None
        passed = bool(finite and np.all(abs(a-b) <= atol+rtol*abs(b)))
        self.rows.append({"name": label, "passed": passed, "actual_shape": list(a.shape), "expected_shape": list(b.shape),
                          "max_abs_error": error, "atol": atol, "rtol": rtol})
        return passed

    def stat(self, label, saved, expected, *, bank=False):
        for field in ("replicates", "mean", "SE"):
            self.close(label+"."+field, saved[field], expected[field])
        if "sample_SD" in saved:
            self.close(label+".sample_SD", saved["sample_SD"], expected["sample_SD"])
        self.exact(label+".R", saved["R"], R)
        if bank:
            self.exact(label+".B", saved["B"], expected["B"])

    def interval(self, label, saved, expected):
        for key in ("lower", "upper", "critical"):
            self.close(label+"."+key, saved[key], expected[key])
        for key in ("df", "family", "alpha"):
            self.exact(label+"."+key, saved[key], expected[key])


class BoundInputs:
    def __init__(self, cfg, checks):
        require(cfg.get("schema") == "learned_precision_analysis_contract_v1" and cfg.get("frozen") is True,
                "frozen precision analysis contract required")
        self.cfg, self.checks, self.cache, self.audit = cfg, checks, {}, []
        for mapping_name in ("source_sha256", "input_sha256"):
            for name, expected in cfg[mapping_name].items():
                filename = safe_input(name)
                actual = digest(filename)
                require(actual == expected, "bound identity changed: "+name)
                self.audit.append({"path": name, "sha256": actual, "role": mapping_name})
        require(cfg["source_sha256"].get(SOURCE) == digest(Path(__file__)), "verifier source must be bound in the analysis contract")

    def payload(self, name):
        require(name in self.cfg["input_sha256"], "unbound verifier input: "+name)
        filename = safe_input(name)
        payload = filename.read_bytes()
        require(hashlib.sha256(payload).hexdigest() == self.cfg["input_sha256"][name], "input changed while reading: "+name)
        return payload

    def json(self, name):
        if name not in self.cache:
            require(path(name).suffix.lower() == ".json", "JSON input required")
            self.cache[name] = json.loads(self.payload(name).decode("utf-8-sig"))
        return self.cache[name]

    def arrays(self, name):
        if name not in self.cache:
            require(path(name).suffix.lower() == ".npz", "derived NPZ input required")
            with np.load(io.BytesIO(self.payload(name)), allow_pickle=False) as archive:
                self.cache[name] = {key: archive[key].copy() for key in archive.files}
            require(all(value.dtype.kind in "fiu" and np.isfinite(value).all() for value in self.cache[name].values()),
                    "derived arrays must all be finite numeric arrays")
        return self.cache[name]

    def record(self, record, field):
        name = record[field]
        require(self.cfg["input_sha256"].get(name) == record[field+"_sha256"], "record identity is not contract-bound: "+name)
        return name


class Verifier:
    def __init__(self, inputs, checks):
        self.i, self.c = inputs, checks
        self.coef_cache, self.contrast_cache = {}, {}
        self.verified_stages = []
        self.recomputed = {}
        self.frozen_coefficients = {}

    def prefix(self, label, doc, array):
        half = array.shape[1]//2
        full, small = bank_moments(array), bank_moments(array[:, :half])
        diff = moments(full["replicates"]-small["replicates"])
        self.c.stat(label+".full", doc["full"], full, bank=True)
        self.c.stat(label+".prefix", doc["prefix"], small, bank=True)
        self.c.stat(label+".difference", doc["paired_full_minus_prefix"], diff)

    def coefficient(self, record, B, *, order=36, expected_phase=None, expected_case=None):
        name = self.i.record(record, "statistics")
        arrays_name = self.i.record(record, "banks")
        if name in self.coef_cache:
            result = self.coef_cache[name]
            self.c.exact(name+".reused_B", result["B"], B)
            self.c.exact(name+".reused_order", result["doc"]["outer_order"], order)
            if expected_phase is not None:
                self.c.exact(name+".reused_phase", result["doc"]["phase"], expected_phase)
            if expected_case is not None:
                self.c.exact(name+".reused_case", (result["doc"]["rep"], result["doc"]["N"]), expected_case)
            return result
        doc, arrays = self.i.json(name), self.i.arrays(arrays_name)
        expected_keys = set(METRICS) | {"C_origin", "C_t0p1", "C_t0p2"}
        self.c.exact(name+".all_array_keys", sorted(arrays), sorted(expected_keys))
        require(expected_keys <= arrays.keys(), "incomplete saved coefficient/control matrices")
        for key in METRICS:
            shape = (R, B, 3) if key.endswith("g") else (R, B, 3, 3)
            self.c.exact(name+"."+key+".shape", list(arrays[key].shape), list(shape))
            self.c.stat(name+"."+key, doc["statistics"][key], bank_moments(arrays[key]), bank=True)
        for teacher, array_key in (("origin", "C_origin"), ("t0p1", "C_t0p1"), ("t0p2", "C_t0p2")):
            self.c.exact(name+"."+array_key+".shape", list(arrays[array_key].shape), [R, B, 3, 3])
            self.c.stat(name+".control."+teacher, doc["control_statistics"][teacher], bank_moments(arrays[array_key]), bank=True)
        self.c.close(name+".g_unchanged", arrays["q2_g"], arrays["cv_g"], atol=0, rtol=0)
        self.c.close(name+".B_unchanged", arrays["q2_B"], arrays["cv_B"], atol=0, rtol=0)
        self.c.close(name+".H_plus_C0", arrays["q2_H"], arrays["cv_H"]+arrays["C_origin"])
        self.c.exact(name+".outer_order", doc["outer_order"], order)
        self.c.exact(name+".inner_order", doc["inner_order"], 256)
        self.c.exact(name+".teacher_derivative", doc["reference_derivative"], "same_finite_GH_Cholesky_chain_rule")
        self.c.exact(name+".no_projection", doc["no_group_permutation_projection"], True)
        if expected_phase is not None:
            self.c.exact(name+".phase", doc["phase"], expected_phase)
        if expected_case is not None:
            self.c.exact(name+".case", (doc["rep"], doc["N"]), expected_case)
        for field in ("rep", "N", "kind", "bank_id", "reference_identity", "outer_order"):
            if field in record:
                self.c.exact(name+".record_"+field, doc[field], record[field])
        prefix_name = (Path(name).parent/"PREFIX_COMPARISON.json").as_posix()
        prefix = self.i.json(prefix_name)
        self.c.exact(prefix_name+".all_keys", sorted(prefix), sorted(expected_keys))
        for key in expected_keys:
            self.prefix(prefix_name+"."+key, prefix[key], arrays[key])
        result = {"doc": doc, "arrays": arrays, "B": B, "source": name}
        self.coef_cache[name] = result
        return result

    def contrast(self, record, coefficient):
        name = self.i.record(record, "statistics")
        if name in self.contrast_cache:
            return self.contrast_cache[name]
        array_name = self.i.record(record, "banks")
        rname = self.i.record(record, "remainder_statistics")
        rarray_name = self.i.record(record, "remainder_banks")
        doc, arrays = self.i.json(name), self.i.arrays(array_name)
        rdoc, remainder = self.i.json(rname), self.i.arrays(rarray_name)
        amp, cd, ca = record["t"], coefficient["doc"], coefficient["arrays"]
        require(amp in TIMES, "fixed amplitude required")
        self.c.exact(name+".t", doc["t"], amp)
        for label, item in ((name, doc), (rname, rdoc)):
            self.c.exact(label+".alpha", item["alpha"], ALPHA)
            self.c.close(label+".direction", item["direction"], V, atol=2e-15, rtol=0)
            self.c.exact(label+".bank", item["bank_id"], cd["bank_id"])
            self.c.exact(label+".reference", item["reference_identity"], cd["reference_identity"])
            self.c.exact(label+".teacher_key", item["teacher_key"], "t"+str(amp).replace(".", "p"))
        tag = doc["teacher_key"]
        self.c.exact(name+".teacher_weights", doc["teacher_weight_sha256"], cd["teacher_weight_sha256"][tag])
        for key, value in arrays.items():
            self.c.exact(name+"."+key+".shape", list(value.shape), [R, coefficient["B"]])
            self.c.stat(name+"."+key, doc["statistics"][key], bank_moments(value), bank=True)
            self.prefix(name+".prefix."+key, doc["prefix_comparison"][key], value)
        increment = .5*(ALPHA**2-1)*amp**2*((ca["C_"+tag] @ V) @ V)
        self.c.close(name+".D_increment", arrays["D_control_increment"], increment)
        self.c.close(name+".q2D_minus_cvD", arrays["q2_D"]-arrays["cv_D"], increment)
        all_expected = {}
        for kind in ("cv", "q2"):
            part = components(ca, kind, amp)
            expected = {**{kind+"_"+key: value for key, value in part.items()},
                        kind+"_D": arrays[kind+"_D"], kind+"_remainder": arrays[kind+"_D"]-part["prediction"],
                        kind+"_normalized_remainder": (arrays[kind+"_D"]-part["prediction"])/amp**2}
            all_expected.update(expected)
        self.c.exact(rname+".all_keys", sorted(remainder), sorted(all_expected))
        for key, expected in all_expected.items():
            self.c.close(rname+"."+key+".bank_identity", remainder[key], expected)
            self.c.stat(rname+"."+key, rdoc["statistics"][key], bank_moments(expected), bank=True)
            self.prefix(rname+".prefix."+key, rdoc["prefix_comparison"][key], expected)
        cincrement = .5*(ALPHA**2-1)*amp**2*(((ca["C_"+tag]-ca["C_origin"]) @ V) @ V)
        self.c.close(rname+".Ct_minus_C0_increment", remainder["q2_remainder"]-remainder["cv_remainder"], cincrement)
        observed_error = float(np.max(abs((remainder["q2_remainder"]-remainder["cv_remainder"])-cincrement)))
        self.c.close(rname+".recorded_identity_error", rdoc["control_remainder_increment_identity_max_error"], observed_error, atol=3e-13, rtol=0)
        result = {"doc": doc, "arrays": arrays, "remainder_doc": rdoc, "remainder": remainder,
                  "coefficient": coefficient, "source": name, "record": record}
        self.contrast_cache[name] = result
        return result

    def cases(self, summary_paths, producer_phase):
        require(len(summary_paths) == 3 and len(set(summary_paths)) == 3, "three distinct shard summaries required")
        rows = []
        for name in summary_paths:
            summary = self.i.json(name)
            self.c.exact(name+".status", summary["status"], "completed")
            self.c.exact(name+".phase", summary["phase"], producer_phase)
            rows.extend(summary["outputs"])
        require(len(rows) == 6 and {(row["rep"], row["N"]) for row in rows} == CASES, "six complete distinct original cases required")
        require(all(row["kind"] == "sym" for row in rows), "sym-only precision study")
        return sorted(rows, key=lambda row: (row["rep"], row["N"]))

    def phase_paths(self, saved):
        return [row["path"] for row in saved["phases"]]

    def development(self, saved, summary_paths=None):
        paths = self.phase_paths(saved) if summary_paths is None else summary_paths
        rows = self.cases(paths, "development")
        actual = {(row["rep"], row["N"]): row for row in saved["cases"]}
        remrows = {(row["rep"], row["N"], row["t"]): row for row in saved["same_bank_development"]}
        require(len(actual) == 6 and len(remrows) == 12, "complete development analysis slots")
        ratios, recomputed = [], []
        for row in rows:
            key = row["rep"], row["N"]
            coef = self.coefficient(row["coefficient_bank"], 256, expected_phase="development", expected_case=key)
            part = {kind: components(coef["arrays"], kind, .1) for kind in ("cv", "q2")}
            base, new = (bank_moments(part[k]["quadratic_coefficient"])["replicates"] for k in ("cv", "q2"))
            var0, var1 = variance(base), variance(new)
            require(var0 > 0 and np.isfinite([var0, var1]).all(), "valid positive baseline Q variance")
            ratio = var1/var0
            ratios.append(ratio)
            label = "development."+repr(key)
            self.c.stat(label+".base_Q", actual[key]["baseline_Q"], moments(base))
            self.c.stat(label+".q2_Q", actual[key]["q2_Q"], moments(new))
            self.c.close(label+".variance_ratio", actual[key]["variance_ratio"], ratio)
            for kind, field in (("cv", "baseline_cov_H_B"), ("q2", "q2_cov_H_B")):
                hh, bb = (bank_moments(part[kind][name])["replicates"] for name in ("H_quadratic_coefficient", "B_quadratic_coefficient"))
                self.c.close(label+"."+field, actual[key][field], cov(hh, bb))
                self.c.close(label+"."+kind+"_Q_variance_decomposition", variance(hh+bb), variance(hh)+variance(bb)+2*cov(hh, bb))
            val = row["validation_bank"]
            vc = self.coefficient(val["coefficient"], 512, expected_phase="development", expected_case=key)
            require({item["t"] for item in val["contrasts"]} == set(TIMES), "both development amplitudes required")
            for item in val["contrasts"]:
                got = self.contrast(item, vc)
                for kind in ("cv", "q2"):
                    rr = remrows[(*key, item["t"])]["controls"][kind]
                    d, p, w = [bank_moments(got["remainder"][kind+"_"+field])["replicates"]
                               for field in ("D", "prediction", "normalized_remainder")]
                    for field, values in (("D", d), ("same_P", p), ("same_W", w)):
                        self.c.stat(label+"."+str(item["t"])+"."+kind+"."+field, rr[field], moments(values))
                    self.c.close(label+"."+kind+".cov_D_P."+str(item["t"]), rr["cov_D_P"], cov(d, p))
            recomputed.append({"rep": key[0], "N": key[1], "variance_ratio": ratio, "base_variance": var0, "q2_variance": var1})
        ordered = sorted(ratios)
        median, maximum = (ordered[2]+ordered[3])/2, ordered[-1]
        selection = self.i.json(self.i.cfg["policy"])["development_selection"]
        passed = median <= selection["median_ratio_max"] and maximum <= selection["maximum_ratio_max"]
        self.c.close("development.median", saved["median_ratio"], median)
        self.c.close("development.maximum", saved["maximum_ratio"], maximum)
        self.c.exact("development.gate", saved["passed"], passed)
        self.c.exact("development.selection", saved["selected_estimator"], "q2" if passed else "none_no_fresh_study")
        self.c.exact("development.all_six", saved["all_six_cases_complete"], True)
        self.c.exact("development.old_primary_unchanged", saved["old_primary_results_reinterpreted"], False)
        self.recomputed["development"] = {"cases": recomputed, "median_ratio": median, "maximum_ratio": maximum, "scientific_gate_passed": passed}
        self.verified_stages.append("development")
        return passed

    def freeze_predictions(self, frozen, summary_paths=None):
        require(frozen["all_predictions_frozen"] and frozen["before_validation_randomization"], "prevalidation freeze required")
        self.c.exact("freeze.component_scaling", frozen["component_scaling"], "coefficients_before_t_scaling")
        decision_name = frozen["development_decision"]
        self.c.exact("freeze.decision_identity", frozen["development_decision_sha256"], self.i.cfg["input_sha256"][decision_name])
        decision = self.i.json(decision_name)
        if "development" not in self.verified_stages:
            self.development(decision)
        self.c.exact("freeze.development_passed", decision["passed"], True)
        rows = self.cases(self.phase_paths(frozen) if summary_paths is None else summary_paths, "coefficients")
        records = {(r["rep"], r["N"]): r for r in frozen["coefficient_outputs"]}
        predictions = {(r["rep"], r["N"], r["t"]): r for r in frozen["predictions"]}
        require(len(records) == 6 and len(predictions) == 12, "all6 coefficients and12 predictions frozen")
        rebuilt = {}
        for row in rows:
            key = row["rep"], row["N"]
            record = row["coefficient"]
            for field in ("statistics", "statistics_sha256", "banks", "banks_sha256"):
                self.c.exact("freeze.record."+repr(key)+"."+field, records[key][field], record[field])
            coef = self.coefficient(record, 256, expected_phase="coefficients", expected_case=key)
            self.frozen_coefficients[key] = coef
            for amp in TIMES:
                pkey = (*key, amp)
                saved = predictions[pkey]
                self.c.exact("freeze.alpha."+repr(pkey), saved["alpha"], ALPHA)
                self.c.exact("freeze.kind."+repr(pkey), saved["kind"], "sym")
                self.c.exact("freeze.component_semantics."+repr(pkey), saved["component_scaling"], "coefficients_before_t_scaling")
                self.c.exact("freeze.source."+repr(pkey), saved["coefficient_statistics"], record["statistics"])
                self.c.exact("freeze.source_hash."+repr(pkey), saved["coefficient_statistics_sha256"], record["statistics_sha256"])
                for identity in ("bank_id", "reference_identity", "reference_p0_sha256", "reference_dp0_sha256"):
                    self.c.exact("freeze."+repr(pkey)+"."+identity, saved[identity], coef["doc"][identity])
                parts = components(coef["arrays"], "q2", amp)
                quantities = {name: bank_moments(value)["replicates"] for name, value in parts.items()}
                for field, comp in (("replicates", "prediction"), ("linear_replicates", "linear_coefficient"),
                                    ("H_quadratic_replicates", "H_quadratic_coefficient"), ("B_quadratic_replicates", "B_quadratic_coefficient")):
                    self.c.close("freeze."+repr(pkey)+"."+field, saved[field], quantities[comp])
                self.c.close("freeze.component_scaling."+repr(pkey), saved["replicates"],
                             amp*np.asarray(saved["linear_replicates"])+amp**2*(np.asarray(saved["H_quadratic_replicates"])+np.asarray(saved["B_quadratic_replicates"])))
                rebuilt[pkey] = quantities["prediction"]
        self.recomputed["frozen_predictions"] = [{"rep": k[0], "N": k[1], "t": k[2], "statistics": moments(value)} for k, value in rebuilt.items()]
        self.verified_stages.append("freeze_predictions")
        return rebuilt

    def rule_comparison(self, row, primary):
        require((row["rep"], row["N"]) == (0, 16384), "only prespecified rule diagnostic case")
        block = row["GH28_check"]
        lowcoef = self.coefficient(block["coefficient"], 512, order=28, expected_phase="validation", expected_case=(0, 16384))
        low = {item["t"]: self.contrast(item, lowcoef) for item in block["contrasts"]}
        record = block["comparison"]
        name, arrays_name = self.i.record(record, "statistics"), self.i.record(record, "banks")
        doc, arrays = self.i.json(name), self.i.arrays(arrays_name)
        saved = {r["t"]: r for r in doc["rows"]}
        expected_keys, qualified, recomputed = [], [], []
        for amp in TIMES:
            hi, lo = primary[amp], low[amp]
            d = (hi["arrays"]["q2_D"]-lo["arrays"]["q2_D"])/amp**2
            w = hi["remainder"]["q2_normalized_remainder"]-lo["remainder"]["q2_normalized_remainder"]
            label = "t"+str(amp).replace(".", "p")
            for field, value in (("D_normalized_rule_difference", d), ("W_same_rule_difference", w)):
                key = label+"_"+field
                expected_keys.append(key)
                self.c.close(name+"."+key+".banks", arrays[key], value)
                self.c.stat(name+"."+key, saved[amp][field], bank_moments(value), bank=True)
            dmax = float(np.max(abs(stable_mean(d, 1))))
            wmax = float(np.max(abs(stable_mean(w, 1))))
            self.c.close(name+"."+label+".Dmax", saved[amp]["max_abs_complete_replicate_D_normalized_difference"], dmax)
            self.c.close(name+"."+label+".Wmax", saved[amp]["max_abs_complete_replicate_W_same_difference"], wmax)
            ok = dmax <= 1e-4 and wmax <= 1e-4
            qualified.append(ok)
            self.c.exact(name+"."+label+".qualified", saved[amp]["qualified"], ok)
            recomputed.append({"t": amp, "max_D": dmax, "max_W_same": wmax, "qualified": ok})
        self.c.exact(name+".keys", sorted(arrays), sorted(expected_keys))
        self.c.exact(name+".all_qualified", doc["all_qualified"], all(qualified))
        self.recomputed["rule_comparison"] = recomputed
        return doc

    def final(self, saved, summary_paths, frozen_name):
        frozen = self.i.json(frozen_name)
        predictions = self.freeze_predictions(frozen)
        rows = self.cases(summary_paths, "validation")
        analysis_cells = {(row["rep"], row["N"], row["t"]): row for row in saved["cells"]}
        require(len(analysis_cells) == 12, "all12 final panels required")
        counts = {kind: {label: 0 for label in ("within_tolerance", "outside_tolerance", "unresolved")}
                  for kind in ("independent_precision", "same_precision")}
        negatives, exclusions, records, rule_count = 0, 0, [], 0
        for row in rows:
            key = row["rep"], row["N"]
            coef = self.coefficient(row["coefficient"], 512, expected_phase="validation", expected_case=key)
            earlier = self.frozen_coefficients[key]
            self.c.exact("final.independent_bank."+repr(key), coef["doc"]["bank_id"] != earlier["doc"]["bank_id"], True)
            for identity in ("reference_identity", "reference_p0_sha256", "reference_dp0_sha256", "teacher_weight_sha256"):
                self.c.exact("final.common_reference."+repr(key)+"."+identity, coef["doc"][identity], earlier["doc"][identity])
            require({item["t"] for item in row["contrasts"]} == set(TIMES), "two final amplitudes required")
            primary = {}
            for item in row["contrasts"]:
                amp = item["t"]
                pkey = (*key, amp)
                got = self.contrast(item, coef)
                primary[amp] = got
                d = bank_moments(got["arrays"]["q2_D"])["replicates"]
                psame = bank_moments(got["remainder"]["q2_prediction"])["replicates"]
                pind = predictions[pkey]
                wi, ws = (d-pind)/amp**2, (d-psame)/amp**2
                values = {"D": d, "independent_prediction": pind, "same_prediction": psame, "W_ind": wi, "W_same": ws}
                target, label = analysis_cells[pkey], "final."+repr(pkey)
                stats = {name: moments(value) for name, value in values.items()}
                for name, stat in stats.items():
                    self.c.stat(label+"."+name, target[name], stat)
                cis = {name: ci(stats[name]) for name in ("D", "W_ind", "W_same")}
                for name, interval in cis.items():
                    self.c.interval(label+"."+name+"_interval", target[name+"_interval"], interval)
                direction = "negative" if cis["D"]["upper"] < 0 else "positive" if cis["D"]["lower"] > 0 else "unresolved"
                labels = {"independent_precision": classify(cis["W_ind"]), "same_precision": classify(cis["W_same"])}
                self.c.exact(label+".direction", target["direction"], direction)
                for name, classification in labels.items():
                    self.c.exact(label+"."+name, target[name], classification)
                    counts[name][classification] += 1
                excludes = cis["W_same"]["upper"] < 0 or cis["W_same"]["lower"] > 0
                self.c.exact(label+".same_excludes_zero", target["same_remainder_excludes_zero"], excludes)
                negatives += direction == "negative"
                exclusions += excludes
                self.c.close(label+".cov_D_Pind", target["cov_D_Pind"], cov(d, pind))
                self.c.close(label+".cov_D_Psame", target["cov_D_Psame"], cov(d, psame))
                self.c.close(label+".ind_variance", variance(d-pind), variance(d)+variance(pind)-2*cov(d, pind))
                self.c.close(label+".same_variance", variance(d-psame), variance(d)+variance(psame)-2*cov(d, psame))
                varwi = variance(wi)
                if varwi > 0:
                    self.c.close(label+".same_independent_variance_ratio", target["same_over_independent_residual_variance"], variance(ws)/varwi)
                else:
                    self.c.exact(label+".undefined_variance_ratio", target["same_over_independent_residual_variance"], None)
                self.c.close(label+".residual_decomposition", (wi-ws)*amp**2, psame-pind)
                ind_name = self.i.record(item, "independent_statistics")
                independent = self.i.json(ind_name)
                self.c.exact(ind_name+".freeze", independent["prediction_freeze"], frozen_name)
                self.c.exact(ind_name+".freeze_sha", independent["prediction_freeze_sha256"], self.i.cfg["input_sha256"][frozen_name])
                for name, values in (("P_ind", pind), ("D_q2", d), ("remainder_ind", d-pind), ("W_ind", wi)):
                    self.c.stat(ind_name+"."+name, independent["statistics"][name], moments(values))
                records.append({"rep": key[0], "N": key[1], "t": amp, "statistics": stats,
                                "intervals": cis, "direction": direction, **labels, "same_remainder_excludes_zero": excludes})
            if "GH28_check" in row:
                rule_doc = self.rule_comparison(row, primary)
                self.c.exact("final.saved_rule_record", saved["rule_order_comparison"], [rule_doc])
                rule_count += 1
        self.c.exact("final.rule_check_count", rule_count, 1)
        summary = saved["summary"]
        self.c.exact("final.family", summary["family_size"], 36)
        if "expected_cells" in summary:
            self.c.exact("final.expected_cells", summary["expected_cells"], 12)
            self.c.exact("final.available_cells", summary["available_cells"], 12)
            self.c.exact("final.unresolved_cells", summary["unresolved_cells"], 0)
        self.c.close("final.critical", summary["critical_value"], critical())
        self.c.exact("final.negative_count", summary["direction_negative"], negatives)
        self.c.exact("final.same_nonzero_count", summary["same_remainder_excludes_zero"], exclusions)
        for name, count in counts.items():
            self.c.exact("final."+name+"_counts", summary[name], count)
        self.c.exact("final.old_primary_unchanged", saved["old_1_of_12_unchanged"], True)
        self.recomputed["final"] = {"cells": records, "negative_count": negatives, "same_nonzero_count": exclusions, **counts}
        self.verified_stages.append("final")


def run(contract_path, analysis_path, out):
    require(not out.exists(), "new verification JSON required; do not overwrite")
    cfg = json.loads(contract_path.read_text(encoding="utf-8-sig"))
    checks = Checks()
    report = {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
              "contract_sha256": digest(contract_path), "analysis_sha256": digest(analysis_path),
              "verifier_source_sha256": digest(Path(__file__)), "phase": cfg.get("phase"),
              "scope": "independent reconstruction from derived saved arrays/statistics only; not recomputation of model or original particle targets",
              "original_particle_or_local_weight_arrays_read": 0, "model_RNG_quadrature_optimization_calls": 0,
              "old_results_modified": False, "all_new_precision_gates_must_pass": False,
              "tolerances": {"absolute": ATOL, "relative": RTOL}, "passed": False}
    try:
        inputs = BoundInputs(cfg, checks)
        policy = inputs.json(cfg["policy"])
        require(policy["primary_family"]["family_size"] == 36 and policy["primary_family"]["R"] == R
                and policy["primary_family"]["df"] == 15, "unchanged fixed statistical policy required")
        saved = json.loads(analysis_path.read_text(encoding="utf-8-sig"))
        checks.exact("analysis.contract_identity", saved["analysis_contract_sha256"], digest(contract_path))
        checks.exact("analysis.source_identity", saved["analysis_source_sha256"], cfg["source_sha256"][ANALYZER])
        checks.exact("analysis.no_model_calls", saved["no_scientific_kernels_or_RNG"], True)
        verifier = Verifier(inputs, checks)
        phase = cfg["phase"]
        if phase == "development":
            verifier.development(saved, cfg["shard_summaries"])
        elif phase == "freeze_predictions":
            verifier.freeze_predictions(saved, cfg["shard_summaries"])
        elif phase == "final":
            verifier.final(saved, cfg["shard_summaries"], cfg["prediction_freeze"])
        else:
            raise ValueError("unknown analysis phase")
        report.update(input_identities=inputs.audit, verified_stages=verifier.verified_stages,
                      derived_coefficient_bundles=len(verifier.coef_cache), derived_contrast_remainder_pairs=len(verifier.contrast_cache),
                      recomputed=verifier.recomputed, passed=all(row["passed"] for row in checks.rows))
    except Exception:
        report["error"] = traceback.format_exc()
    report["checks"] = checks.rows
    report["check_count"] = len(checks.rows)
    report["failed_checks"] = [row["name"] for row in checks.rows if not row["passed"]]
    with out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, default=serial, allow_nan=False)
        stream.write("\n")
    return report


def manufactured_bundle():
    """Full derived-output schema made from arbitrary deterministic tensors."""
    r, z = np.arange(R)[:, None], np.arange(8)[None, :]
    u = r/100+z/1000
    g = np.stack([u+.1, .2-u/2, np.zeros_like(u)], axis=-1)
    h = np.broadcast_to(np.array([[1., .4, 0], [.4, 1., 0], [0, 0, 1.]]), (R, 8, 3, 3)).copy()
    bb = np.broadcast_to(np.array([[.2, .3, 0], [.7, .4, 0], [0, 0, .5]]), h.shape).copy()
    h[..., 0, 0] += u
    bb[..., 1, 1] += u/4
    c0 = np.broadcast_to(np.diag([.01, -.02, .03]), h.shape).copy()
    c0[..., 0, 0] += u/3
    arrays = {"raw_g": g+.01, "cv_g": g, "q2_g": g.copy(), "raw_H": h+.02, "cv_H": h, "q2_H": h+c0,
              "raw_B": bb+.03, "cv_B": bb, "q2_B": bb.copy(), "C_origin": c0,
              "C_t0p1": c0+.01*np.diag([.1, .2, .3]), "C_t0p2": c0+.04*np.diag([.1, .2, .3])}

    def saved_stat(a):
        rep = a.mean(axis=1)
        return {"replicates": rep, "mean": rep.mean(axis=0), "SE": rep.std(axis=0, ddof=1)/4, "R": R, "B": a.shape[1]}

    def prefix(a):
        full, half = saved_stat(a), saved_stat(a[:, :a.shape[1]//2])
        delta = full["replicates"]-half["replicates"]
        return {"full": full, "prefix": half, "paired_full_minus_prefix": {
            "replicates": delta, "mean": delta.mean(axis=0), "SE": delta.std(axis=0, ddof=1)/4, "R": R}}

    def hand_components(data, tag, tval):
        gg, hh, bmat = (data[tag+"_"+field] for field in ("g", "H", "B"))
        linear = (ALPHA-1)*(gg[..., 0]+gg[..., 1])/math.sqrt(2)
        hs = hh[..., 0, 0]+hh[..., 0, 1]+hh[..., 1, 0]+hh[..., 1, 1]
        bs = bmat[..., 0, 0]+bmat[..., 0, 1]+bmat[..., 1, 0]+bmat[..., 1, 1]
        hv, bv = (ALPHA**2-1)*hs/4, -(ALPHA-1)*bs/2
        return {"linear_coefficient": linear, "H_quadratic_coefficient": hv, "B_quadratic_coefficient": bv,
                "quadratic_coefficient": hv+bv, "prediction": tval*linear+tval*tval*(hv+bv)}

    docs, packs = {}, {}
    coef_path, bank_path = "manufactured/COEFFICIENT_STATISTICS.json", "manufactured/COEFFICIENT_BANKS.npz"
    common = {"bank_id": "manufactured-fixed-bank", "reference_identity": "manufactured-same-rule",
              "outer_order": 36, "inner_order": 256, "rep": 0, "N": 16384, "kind": "sym", "phase": "development",
              "reference_derivative": "same_finite_GH_Cholesky_chain_rule", "no_group_permutation_projection": True,
              "reference_p0_sha256": "a"*64, "reference_dp0_sha256": "b"*64,
              "teacher_weight_sha256": {tag: "c"*64 for tag in ("origin", "t0p1", "t0p2")}}
    docs[coef_path] = {**common, "statistics": {key: saved_stat(arrays[key]) for key in METRICS},
                       "control_statistics": {tag: saved_stat(arrays["C_"+tag]) for tag in ("origin", "t0p1", "t0p2")}}
    packs[bank_path] = arrays
    docs["manufactured/PREFIX_COMPARISON.json"] = {key: prefix(value) for key, value in arrays.items()}
    coefficient = {"statistics": coef_path, "statistics_sha256": "d"*64, "banks": bank_path, "banks_sha256": "d"*64,
                   "rep": 0, "N": 16384, "kind": "sym", "bank_id": common["bank_id"],
                   "reference_identity": common["reference_identity"], "outer_order": 36}
    contrasts = []
    for amp, tag in ((.1, "t0p1"), (.2, "t0p2")):
        cp = {kind: hand_components(arrays, kind, amp) for kind in ("cv", "q2")}
        d0 = cp["cv"]["prediction"]+amp**4*(1+r/10+z/100)
        ct = arrays["C_"+tag]
        increment = (ALPHA**2-1)*amp**2*(ct[..., 0, 0]+ct[..., 0, 1]+ct[..., 1, 0]+ct[..., 1, 1])/4
        dd = {"cv_D": d0, "q2_D": d0+increment, "D_control_increment": increment, "raw_D": d0+.003}
        rem = {}
        for kind in ("cv", "q2"):
            rem.update({kind+"_"+name: value for name, value in cp[kind].items()})
            rem[kind+"_D"] = dd[kind+"_D"]
            rem[kind+"_remainder"] = dd[kind+"_D"]-cp[kind]["prediction"]
            rem[kind+"_normalized_remainder"] = rem[kind+"_remainder"]/amp**2
        info = {"bank_id": common["bank_id"], "reference_identity": common["reference_identity"], "t": amp,
                "alpha": ALPHA, "direction": V.copy(), "teacher_key": tag, "teacher_weight_sha256": "c"*64}
        sp, bp = "manufactured/CONTRAST_"+tag+".json", "manufactured/CONTRAST_"+tag+".npz"
        rp, rb = "manufactured/REMAINDER_"+tag+".json", "manufactured/REMAINDER_"+tag+".npz"
        docs[sp] = {**info, "statistics": {key: saved_stat(value) for key, value in dd.items()},
                    "prefix_comparison": {key: prefix(value) for key, value in dd.items()}}
        docs[rp] = {**info, "statistics": {key: saved_stat(value) for key, value in rem.items()},
                    "prefix_comparison": {key: prefix(value) for key, value in rem.items()},
                    "control_remainder_increment_identity_max_error": 0.}
        packs[bp], packs[rb] = dd, rem
        record = {"t": amp}
        for field, value in (("statistics", sp), ("banks", bp), ("remainder_statistics", rp), ("remainder_banks", rb)):
            record[field], record[field+"_sha256"] = value, "d"*64
        contrasts.append(record)

    class InMemory:
        def __init__(self):
            self.cfg = {"input_sha256": {name: "d"*64 for name in [*docs, *packs]}}

        def json(self, name):
            return docs[name]

        def arrays(self, name):
            return packs[name]

        def record(self, record, field):
            require(self.cfg["input_sha256"][record[field]] == record[field+"_sha256"], "manufactured binding mismatch")
            return record[field]

    return InMemory(), coefficient, contrasts


def self_test():
    """Deterministic manufactured arrays only; no scientific file reads."""
    checks = Checks()
    r = np.arange(R, dtype=float)[:, None]
    b = np.arange(8, dtype=float)[None, :]
    values = 2*r + .1*b
    stat = bank_moments(values)
    checks.close("within_B_before_R", stat["replicates"], 2*np.arange(R)+.35)
    checks.close("SE_unit_R_not_RB", stat["SE"], np.std(2*np.arange(R), ddof=1)/4)
    g = np.zeros((R, 8, 3)); g[..., 0] = values; g[..., 1] = 1
    h = np.broadcast_to(np.eye(3), (R, 8, 3, 3)).copy()
    h[..., 0, 1] = h[..., 1, 0] = .4
    bb = np.broadcast_to(np.array([[.2, .3, 0], [.7, .4, 0], [0, 0, .5]]), h.shape).copy()
    c0 = np.zeros_like(h); c0[..., 0, 0] = .02+values/1000; c0[..., 1, 1] = -.01
    ct = c0+.005*np.eye(3)
    arrays = {"cv_g": g, "q2_g": g.copy(), "cv_H": h, "q2_H": h+c0,
              "cv_B": bb, "q2_B": bb.copy(), "C_origin": c0}
    amp = .2
    base, new = components(arrays, "cv", amp), components(arrays, "q2", amp)
    expected_base = (ALPHA-1)*amp*(values+1)/math.sqrt(2) + amp**2*(.5*(ALPHA**2-1)*1.4-(ALPHA-1)*.8)
    checks.close("full_matrix_prediction", base["prediction"], expected_base)
    delta_h = .5*(ALPHA**2-1)*amp**2*(c0[..., 0, 0]+c0[..., 1, 1])/2
    checks.close("H_control_prediction_increment", new["prediction"]-base["prediction"], delta_h)
    d0 = base["prediction"]+.01*amp**2+values/50000
    dinc = .5*(ALPHA**2-1)*amp**2*((ct[..., 0, 0]+ct[..., 1, 1])/2)
    d1 = d0+dinc
    checks.close("Ct_minus_C0_remainder", (d1-new["prediction"])-(d0-base["prediction"]),
                 .5*(ALPHA**2-1)*amp**2*((ct[..., 0, 0]-c0[..., 0, 0]+ct[..., 1, 1]-c0[..., 1, 1])/2))
    a = np.arange(R, dtype=float)
    z = a[::-1]**2
    checks.close("variance_covariance", variance(a-z), variance(a)+variance(z)-2*cov(a, z))
    checks.close("paired_cancellation", moments(a-(a+.002))["SE"], 0., atol=1e-15, rtol=0)
    checks.close("family36_tail", student_t.sf(critical(), 15), .05/72, atol=2e-14, rtol=0)
    checks.exact("inclusive_precision_boundary", classify({"lower": -.01, "upper": .01}), "within_tolerance")
    checks.exact("overlap_is_unresolved", classify({"lower": -.02, "upper": 0}), "unresolved")
    checks.exact("outside_tolerance", classify({"lower": .011, "upper": .02}), "outside_tolerance")
    rejected = 0
    for forbidden in ("a/SAMPLES.npz", "a/LOCAL_WEIGHTS.npz", "a/UNIFORMS.npz", "a/PHYSICAL_DATA.npz"):
        try:
            safe_input(forbidden)
        except ValueError:
            rejected += 1
    checks.exact("original_arrays_refused", rejected, 4)
    altered = dinc.copy(); altered[2, 4] += 1e-4
    guard = Checks(); guard.close("bad_control", altered, dinc)
    checks.exact("corrupted_derived_control_detected", guard.rows[0]["passed"], False)
    fixture, coef_record, contrast_records = manufactured_bundle()
    bundle_checks = Checks()
    verifier = Verifier(fixture, bundle_checks)
    coef = verifier.coefficient(coef_record, 8, expected_phase="development", expected_case=(0, 16384))
    for record in contrast_records:
        verifier.contrast(record, coef)
    failed = [row["name"] for row in bundle_checks.rows if not row["passed"]]
    checks.exact("full_derived_bundle_statistics_prefix_control_remainder", failed, [])
    checks.exact("full_bundle_checks_nontrivial", len(bundle_checks.rows) > 500, True)
    # Mutate one saved derived point, without changing expected metadata, to
    # demonstrate that bundle checks do not merely trust reported summaries.
    fixture, coef_record, contrast_records = manufactured_bundle()
    fixture.arrays(coef_record["banks"])["q2_H"][4, 3, 0, 1] += .001
    corrupted_checks = Checks()
    Verifier(fixture, corrupted_checks).coefficient(coef_record, 8)
    checks.exact("corrupt_saved_H_bundle_rejected", any(not row["passed"] for row in corrupted_checks.rows), True)
    return {"schema": SCHEMA, "passed": all(row["passed"] for row in checks.rows), "checks": checks.rows,
            "check_count": len(checks.rows), "manufactured_bundle_check_count": len(bundle_checks.rows),
            "scientific_files_read": 0, "RNG_or_model_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
        print(json.dumps(result, ensure_ascii=False, indent=2, default=serial, allow_nan=False))
    else:
        require(args.contract is not None and args.analysis is not None and args.out is not None,
                "--contract, --analysis, --out required")
        result = run(args.contract.resolve(), args.analysis.resolve(), args.out.resolve())
        print(json.dumps({"passed": result["passed"], "phase": result["phase"], "check_count": result["check_count"],
                          "failed_checks": result["failed_checks"], "error": result.get("error")}, ensure_ascii=False))
    raise SystemExit(0 if result["passed"] else 1)
