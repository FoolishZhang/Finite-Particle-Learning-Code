"""Independent JSON arithmetic verification of the learned-local saved study.

No analyzer/scientific-target import, NPZ access, RNG, quadrature, optimization,
or training replay. Scalar/tensor summaries use Python statistics; chain
quadratic forms explicitly retain both off-diagonal entries.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import unittest
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
R, DF, FAMILY = 16, 15, 24
ALPHA = .3201223434446316
SIZES, KINDS = (16384, 65536), ("raw", "sym")
ALL_KEYS = ("raw_g", "cv_g", "raw_H", "cv_H", "raw_B", "cv_B")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_evidence": repr(value)}
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return [clean(v) for v in sorted(value, key=repr)]
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    return value


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def resolve(name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("repository-relative binding required")
    path = (ROOT/path).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("binding leaves repository")
    return path


def keys():
    return [(rep, n, kind) for rep in range(3) for n in SIZES for kind in KINDS]


def critical():
    return float(student_t.ppf(1-.05/(2*FAMILY), DF))


class Audit:
    def __init__(self):
        self.checks, self.errors, self.unresolved = 0, [], []
        self.cache, self.identities = {}, {}

    def eq(self, actual, expected, label):
        self.checks += 1
        if actual != expected:
            self.errors.append({"label": label, "actual": actual, "expected": expected})

    def close(self, actual, expected, label):
        self.checks += 1
        if expected is None:
            okay = actual is None
        else:
            okay = finite(actual) and math.isclose(actual, expected, rel_tol=2e-11, abs_tol=2e-12)
        if not okay:
            self.errors.append({"label": label, "actual": actual, "expected": expected})

    def array(self, actual, expected, label):
        if isinstance(expected, list):
            self.eq(isinstance(actual, list) and len(actual) == len(expected), True, label+"/shape")
            if isinstance(actual, list) and len(actual) == len(expected):
                for index, value in enumerate(expected):
                    self.array(actual[index], value, label+f"/{index}")
        else:
            self.close(actual, expected, label)

    def pending(self, label, reason):
        entry = {"label": label, "reason": reason}
        if entry not in self.unresolved:
            self.unresolved.append(entry)

    def load(self, binding, label):
        if not isinstance(binding, dict) or not binding.get("path"):
            self.pending(label, "Missing explicit JSON binding.")
            return None
        try:
            path = resolve(binding["path"])
            if path.suffix.lower() != ".json":
                raise ValueError("only JSON inputs permitted")
            wanted = binding["sha256"].lower()
            key = str(path).casefold()
            if key in self.identities and self.identities[key] != wanted:
                self.eq(wanted, self.identities[key], label+"/conflicting_binding")
                return None
            self.identities[key] = wanted
            if (key, wanted) in self.cache:
                return self.cache[key, wanted]
            actual = sha(path)
            self.eq(actual, wanted, label+"/sha256")
            if actual != wanted:
                self.pending(label, "Hash mismatch; input not used in arithmetic.")
                return None
            document = read(path)
            if not isinstance(document, dict):
                raise ValueError("JSON object required")
            self.cache[key, wanted] = document
            return document
        except FileNotFoundError:
            self.pending(label, "Bound JSON file missing.")
            return None
        except (ValueError, TypeError, KeyError, OSError) as error:
            self.errors.append({"label": label, "input_error": str(error)})
            return None


def shape(value):
    if isinstance(value, list):
        if not value:
            return (0,)
        tail = shape(value[0])
        if any(shape(v) != tail for v in value):
            raise ValueError("ragged tensor")
        return (len(value),)+tail
    if not finite(value):
        raise ValueError("nonfinite/non-numeric tensor")
    return ()


def flatten(value):
    if isinstance(value, list):
        return [x for v in value for x in flatten(v)]
    return [value]


def rebuilt(values, dimensions):
    iterator = iter(values)
    def make(dims):
        return [make(dims[1:]) for _ in range(dims[0])] if dims else next(iterator)
    return make(dimensions)


def summarize(values):
    dimensions = shape(values)
    if not dimensions or dimensions[0] != R:
        raise ValueError("16 complete replicate values required")
    columns = list(zip(*(flatten(v) for v in values)))
    means = [statistics.mean(column) for column in columns]
    sds = [statistics.stdev(column) for column in columns]
    return {"R": R, "replicates": values, "mean": rebuilt(means, dimensions[1:]),
            "sample_SD": rebuilt(sds, dimensions[1:]), "SE": rebuilt([s/4 for s in sds], dimensions[1:])}


def check_summary(audit, saved, expected, label, *, sample_sd=True):
    audit.eq(isinstance(saved, dict), True, label+"/object")
    if not isinstance(saved, dict):
        return
    audit.eq(saved.get("R"), R, label+"/R")
    for key in ("replicates", "mean", "SE") + (("sample_SD",) if sample_sd else ()):
        audit.array(saved.get(key), expected[key], label+"/"+key)


def saved_replicates(document, dimensions, B, audit, label):
    if not isinstance(document, dict):
        raise ValueError("missing statistic object")
    if document.get("R") != R or document.get("B") != B:
        raise ValueError("wrong R/B")
    values = document["replicates"]
    if shape(values) != (R,)+dimensions:
        raise ValueError("wrong replicate shape")
    result = summarize(values)
    check_summary(audit, document, result, label, sample_sd=False)
    return result


def dot_chain(vector):
    return (vector[0]+vector[1])/math.sqrt(2)


def quadratic_chain(matrix):
    return (matrix[0][0]+matrix[0][1]+matrix[1][0]+matrix[1][1])/2


def prediction_vectors(g, H, B):
    linear = [(ALPHA-1)*dot_chain(row) for row in g]
    quadratic = [.5*(ALPHA*ALPHA-1)*quadratic_chain(h)-(ALPHA-1)*quadratic_chain(b)
                 for h, b in zip(H, B)]
    return linear, quadratic


def response_ratio(h, b, kind):
    hmean, bmean = statistics.mean(h), statistics.mean(b)
    if kind != "sym" or hmean <= 0:
        return None
    estimate = bmean/hmean
    delta = [(bv-estimate*hv)/hmean for hv, bv in zip(h, b)]
    return {"estimate": estimate, "delta_SE": statistics.stdev(delta)/4, "delta_replicates": delta}


def interval(summary):
    cv = critical()
    return {"lower": summary["mean"]-cv*summary["SE"], "upper": summary["mean"]+cv*summary["SE"],
            "critical_value": cv, "df": DF, "family_size": FAMILY, "family_alpha": .05,
            "method": "approximate_Bonferroni_two_sided_RQMC_replicate_t"}


def compare_interval(audit, saved, expected, label):
    audit.eq(isinstance(saved, dict), True, label+"/object")
    if isinstance(saved, dict):
        for key, value in expected.items():
            if finite(value):
                audit.close(saved.get(key), value, label+"/"+key)
            else:
                audit.eq(saved.get(key), value, label+"/"+key)


def covariance(columns, labels):
    means = [statistics.mean(column) for column in columns]
    matrix = [[math.fsum((a-ma)*(b-mb) for a, b in zip(left, right))/DF
               for right, mb in zip(columns, means)] for left, ma in zip(columns, means)]
    return {"labels": labels, "sample_covariance": matrix,
            "covariance_of_mean": [[v/R for v in row] for row in matrix]}


def compare_covariance(audit, saved, expected, label):
    audit.eq(isinstance(saved, dict), True, label+"/object")
    if isinstance(saved, dict):
        audit.eq(saved.get("labels"), expected["labels"], label+"/labels")
        for key in ("sample_covariance", "covariance_of_mean"):
            audit.array(saved.get(key), expected[key], label+"/"+key)


def unique_index(rows, fields, audit, label):
    result = {}
    for row in rows:
        key = tuple(row.get(field) for field in fields)
        if key in result:
            audit.eq(False, True, label+"/duplicate/"+repr(key))
            result[key] = None
        else:
            result[key] = row
    return result


def source_indices(cfg, audit):
    coefficients, contrasts, models = {}, {}, {}
    pfreeze = audit.load(cfg.get("proposal_freeze"), "proposal_freeze")
    for pair in (pfreeze or {}).get("pairs", []):
        for kind in KINDS:
            key = (pair["rep"], pair["N"], kind)
            models[key] = audit.load(pair.get("proposals", {}).get(kind), "proposal/"+repr(key))
    cfreeze = audit.load(cfg.get("coefficient_freeze"), "coefficient_freeze")
    for row in (cfreeze or {}).get("outputs", []):
        key = (row["rep"], row["N"], row["kind"])
        if key in coefficients:
            audit.eq(False, True, "coefficient/duplicate/"+repr(key))
            coefficients[key] = None
            continue
        coefficients[key] = audit.load({"path": row["statistics"], "sha256": row["statistics_sha256"]}, "coefficient/"+repr(key))
    for binding in cfg.get("validation_summaries", []):
        document = audit.load(binding, "validation_summary")
        for row in (document or {}).get("outputs", []):
            for contrast in row.get("contrasts", []):
                key = (row["rep"], row["N"], row["kind"], float(contrast["t"]))
                if key in contrasts:
                    audit.eq(False, True, "contrast/duplicate/"+repr(key))
                    contrasts[key] = None
                    continue
                contrasts[key] = audit.load({"path": contrast["statistics"], "sha256": contrast["statistics_sha256"]}, "contrast/"+repr(key))
    return models, coefficients, contrasts


def reconstruct_coefficients(source, models, saved_rows, audit):
    saved = unique_index(saved_rows, ("rep", "N", "kind"), audit, "saved_coefficients")
    audit.eq(set(saved), set(keys()), "saved_coefficients/12_slots")
    result = {}
    for key in keys():
        label = "coefficient/"+repr(key)
        actual = saved.get(key) or {}
        document = source.get(key)
        try:
            if document is None or models.get(key) is None:
                raise ValueError("missing coefficient or proposal input")
            if document.get("kind") != key[2] or document.get("control") != "uniform_logratio_actual_mixture_analytic_moments":
                raise ValueError("wrong coefficient kind/control")
            summaries = {name: saved_replicates(document["statistics"][name], (3,) if name.endswith("g") else (3, 3), 256,
                                               audit, label+"/source/"+name) for name in ALL_KEYS}
            if actual.get("status") != "available":
                if actual.get("errors"):
                    audit.pending(label, "Saved analysis reports producer-flow/input errors; numerical eligibility not promoted by this arithmetic verifier.")
                    result[key] = None
                    continue
                audit.eq(actual.get("status"), "available", label+"/status")
            for name in ALL_KEYS:
                check_summary(audit, actual.get("statistics", {}).get(name), summaries[name], label+"/analysis/"+name)
            H, B = (summaries[name]["replicates"] for name in ("cv_H", "cv_B"))
            hs = summarize([math.fsum(matrix[i][i] for i in range(3))/3 for matrix in H])
            bs = summarize([math.fsum(matrix[i][i] for i in range(3))/3 for matrix in B])
            check_summary(audit, actual.get("h"), hs, label+"/h")
            check_summary(audit, actual.get("b"), bs, label+"/b")
            compare_covariance(audit, actual.get("h_b_covariance"), covariance([hs["replicates"], bs["replicates"]], ["h_trace_over3", "b_trace_over3"]), label+"/h_b_covariance")
            ratio = response_ratio(hs["replicates"], bs["replicates"], key[2])
            if ratio is not None:
                observed = actual.get("response_ratio") or {}
                for name, value in ratio.items():
                    audit.array(observed.get(name), value, label+"/ratio/"+name)
                audit.eq(observed.get("certified_h_positive"), False, label+"/ratio/no_h_certificate")
                audit.eq(observed.get("certified_shrinkage_or_optimum"), False, label+"/ratio/no_optimum_certificate")
            else:
                audit.eq(actual.get("response_ratio"), None, label+"/no_inapplicable_ratio")
            linear, quadratic = prediction_vectors(*(summaries[name]["replicates"] for name in ("cv_g", "cv_H", "cv_B")))
            result[key] = {"statistics": summaries, "h": hs, "b": bs, "ratio": ratio, "linear": linear, "quadratic": quadratic}
        except (ValueError, TypeError, KeyError, IndexError) as error:
            audit.pending(label, str(error))
            audit.eq(actual.get("status"), "unresolved", label+"/unresolved_status")
            result[key] = None
    return result


def contrast_arithmetic(coefficient, D, rawD, tval):
    prediction = [tval*l+tval*tval*q for l, q in zip(coefficient["linear"], coefficient["quadratic"])]
    residual = [d-p for d, p in zip(D, prediction)]
    normalized = [r/(tval*tval) for r in residual]
    return {"D": summarize(D), "raw_uncontrolled_D": summarize(rawD), "prediction": summarize(prediction),
            "residual": summarize(residual), "normalized_residual": summarize(normalized),
            "linear_coefficient": summarize(coefficient["linear"]), "quadratic_coefficient": summarize(coefficient["quadratic"])}


def reconstruct_contrasts(coefficients, source, primary_rows, raw_rows, audit):
    fields = ("rep", "N", "kind", "t")
    saved = unique_index([*primary_rows, *raw_rows], fields, audit, "saved_contrasts")
    expected = [(*key, tv) for key in keys() for tv in ((.1, .2) if key[2] == "sym" else (-.2, -.1, .1, .2))]
    audit.eq(set(saved), set(expected), "saved_contrasts/36_slots")
    audit.eq(len(primary_rows), 12, "saved_primary/12_slots")
    audit.eq(len(raw_rows), 24, "saved_raw/24_slots")
    result = {}
    for key in expected:
        label = "contrast/"+repr(key)
        actual, document = saved.get(key) or {}, source.get(key)
        coefficient, tval = coefficients.get(key[:3]), key[3]
        try:
            if coefficient is None or document is None:
                raise ValueError("coefficient/contrast unavailable")
            if document.get("t") != tval or document.get("alpha") != ALPHA:
                raise ValueError("wrong fixed t/alpha")
            audit.array(document.get("candidate_coordinates"), [[ALPHA*tval/math.sqrt(2), ALPHA*tval/math.sqrt(2), 0.],
                         [tval/math.sqrt(2), tval/math.sqrt(2), 0.]], label+"/fixed_candidates")
            audit.eq(document.get("pairing"), "same_q_particles_and_same_teacher_PMF; shrink_candidate_minus_teacher_candidate", label+"/difference_direction")
            D = saved_replicates(document["statistics"], (), 512, audit, label+"/source/D")
            rawD = saved_replicates(document["raw_statistics"], (), 512, audit, label+"/source/rawD")
            if actual.get("status") != "available":
                if actual.get("errors"):
                    audit.pending(label, "Saved producer-flow/input gate unresolved; no interval promoted by arithmetic alone.")
                    result[key] = None
                    continue
                audit.eq(actual.get("status"), "available", label+"/status")
            value = contrast_arithmetic(coefficient, D["replicates"], rawD["replicates"], tval)
            for name, summary in value.items():
                check_summary(audit, actual.get(name), summary, label+"/"+name)
            compare_covariance(audit, actual.get("covariance"), covariance([value[name]["replicates"] for name in
                ("D", "prediction", "residual", "normalized_residual")], ["D", "P", "D_minus_P", "(D_minus_P)/t_squared"]), label+"/covariance")
            if key[2] == "sym":
                ci, wi = interval(value["D"]), interval(value["normalized_residual"])
                compare_interval(audit, actual.get("D_simultaneous_interval"), ci, label+"/D_interval")
                compare_interval(audit, actual.get("normalized_residual_simultaneous_interval"), wi, label+"/W_interval")
                direction = "negative_supports_fixed_shrink_candidate" if ci["upper"] < 0 else "positive_supports_teacher_candidate" if ci["lower"] > 0 else "unresolved_direction"
                precision = "within_predefined_tolerance" if wi["lower"] >= -.01 and wi["upper"] <= .01 else "not_within_predefined_tolerance"
                value.update(D_interval=ci, W_interval=wi)
            else:
                direction = precision = "descriptive_only"
                audit.eq(actual.get("simultaneous_interval"), None, label+"/no_raw_inferential_interval")
            audit.eq(actual.get("D_direction"), direction, label+"/direction")
            audit.eq(actual.get("prediction_precision"), precision, label+"/precision_gate")
            value.update(direction=direction, precision=precision)
            result[key] = value
        except (ValueError, TypeError, KeyError, IndexError) as error:
            audit.pending(label, str(error))
            audit.eq(actual.get("status"), "unresolved", label+"/unresolved_status")
            result[key] = None
    return result


def odd_even_arithmetic(plus, minus, linear, quadratic, tval):
    odd_part = [(a-b)/2 for a, b in zip(plus, minus)]
    even_part = [(a+b)/2 for a, b in zip(plus, minus)]
    odd, even = [a/tval for a in odd_part], [a/(tval*tval) for a in even_part]
    return {"D_positive": plus, "D_negative": minus, "odd_part": odd_part, "even_part": even_part,
            "odd_coefficient": odd, "even_coefficient": even, "predicted_odd_coefficient": linear,
            "predicted_even_coefficient": quadratic, "odd_coefficient_residual": [a-b for a, b in zip(odd, linear)],
            "even_coefficient_residual": [a-b for a, b in zip(even, quadratic)]}


def verify_odd_even(coefficients, contrasts, saved_rows, audit):
    saved = unique_index(saved_rows, ("rep", "N", "abs_t"), audit, "saved_odd_even")
    expected = [(rep, n, tval) for rep in range(3) for n in SIZES for tval in (.1, .2)]
    audit.eq(set(saved), set(expected), "saved_odd_even/12_slots")
    available = 0
    for rep, n, tval in expected:
        key = (rep, n, tval)
        label = "odd_even/"+repr(key)
        actual = saved.get(key) or {}
        plus, minus, coeff = contrasts.get((rep, n, "raw", tval)), contrasts.get((rep, n, "raw", -tval)), coefficients.get((rep, n, "raw"))
        if plus is None or minus is None or coeff is None:
            audit.eq(actual.get("status"), "unresolved", label+"/unresolved_status")
            audit.pending(label, "Both signed raw cells and coefficients are required.")
            continue
        available += 1
        audit.eq(actual.get("status"), "available", label+"/status")
        values = odd_even_arithmetic(plus["D"]["replicates"], minus["D"]["replicates"], coeff["linear"], coeff["quadratic"], tval)
        for name, vector in values.items():
            check_summary(audit, actual.get("statistics", {}).get(name), summarize(vector), label+"/"+name)
        compare_covariance(audit, actual.get("covariance"), covariance(list(values.values()), list(values)), label+"/covariance")
    return available


def csv_index(path, fields, audit):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for field in fields:
            if field in ("rep", "N"):
                row[field] = int(row[field])
            elif field == "t":
                row[field] = float(row[field])
    return unique_index(rows, fields, audit, str(path.name))


def csv_number(audit, row, key, expected, where):
    raw = row.get(key)
    actual = None if raw in (None, "") else float(raw)
    audit.close(actual, expected, where+"/"+key)


def verify_csv(directory, coefficients, contrasts, audit):
    coeff_rows = csv_index(directory/"COEFFICIENT_CELLS.csv", ("rep", "N", "kind"), audit)
    audit.eq(set(coeff_rows), set(keys()), "CSV/coefficient_slots")
    for key, value in coefficients.items():
        row = coeff_rows.get(key) or {}
        audit.eq(row.get("status"), "available" if value else "unresolved", "CSV/coefficient/"+repr(key)+"/status")
        if value is None:
            continue
        for name, summary in value["statistics"].items():
            for statistic in ("mean", "SE"):
                tensor = summary[statistic]
                for i in range(3):
                    for j in (range(3) if isinstance(tensor[i], list) else [None]):
                        suffix = str(i)+(str(j) if j is not None else "")
                        number = tensor[i][j] if j is not None else tensor[i]
                        csv_number(audit, row, name+"_"+statistic+"_"+suffix, number, "CSV/coefficient/"+repr(key))
        for name in ("h", "b"):
            for statistic in ("mean", "SE"):
                csv_number(audit, row, name+"_"+statistic, value[name][statistic], "CSV/coefficient/"+repr(key))
        for name, field in (("sym_response_ratio", "estimate"), ("sym_response_ratio_delta_SE", "delta_SE")):
            csv_number(audit, row, name, value["ratio"][field] if value["ratio"] else None, "CSV/coefficient/"+repr(key))
    for kind, filename in (("sym", "PRIMARY_CELLS.csv"), ("raw", "RAW_CELLS.csv")):
        rows = csv_index(directory/filename, ("rep", "N", "kind", "t"), audit)
        wanted = {key for key in contrasts if key[2] == kind}
        audit.eq(set(rows), wanted, "CSV/"+filename+"/slots")
        for key in wanted:
            row, value = rows.get(key) or {}, contrasts[key]
            audit.eq(row.get("status"), "available" if value else "unresolved", "CSV/"+repr(key)+"/status")
            if value is None:
                continue
            for name in ("D", "raw_uncontrolled_D", "prediction", "residual", "normalized_residual"):
                for statistic in ("mean", "SE"):
                    csv_number(audit, row, name+"_"+statistic, value[name][statistic], "CSV/"+repr(key))
            audit.eq(row.get("D_direction"), value["direction"], "CSV/"+repr(key)+"/direction")
            audit.eq(row.get("prediction_precision"), value["precision"], "CSV/"+repr(key)+"/precision")
            if kind == "sym":
                csv_number(audit, row, "family_size", FAMILY, "CSV/"+repr(key))
                csv_number(audit, row, "df", DF, "CSV/"+repr(key))
                for name, field in (("D", "D_interval"), ("normalized_residual", "W_interval")):
                    for bound in ("lower", "upper"):
                        csv_number(audit, row, name+"_"+bound, value[field][bound], "CSV/"+repr(key))


def verify_counts(summary, coefficients, contrasts, odd_available, audit):
    primary = [value for key, value in contrasts.items() if key[2] == "sym"]
    raw = [value for key, value in contrasts.items() if key[2] == "raw"]
    count = sum(value is not None for value in primary)
    fixed = {"expected_primary_cells": 12, "expected_primary_quantities": FAMILY, "available_primary_cells": count,
             "unresolved_primary_quantities": 2*(12-count), "primary_family_size_never_reduced": FAMILY,
             "family_alpha": .05, "df": DF}
    for key, expected in fixed.items():
        audit.eq(summary.get(key), expected, "SUMMARY/"+key)
    audit.close(summary.get("critical_value"), critical(), "SUMMARY/fixed_critical")
    for key, field in (("candidate_direction_counts", "direction"), ("prediction_precision_counts", "precision")):
        audit.eq(summary.get(key), dict(Counter(value[field] if value else "unresolved" for value in primary)), "SUMMARY/"+key)
    for field, values in (("coefficient_status_counts", coefficients.values()), ("raw_contrast_status_counts", raw)):
        audit.eq(summary.get(field), dict(Counter("available" if value else "unresolved" for value in values)), "SUMMARY/"+field)
    odd_counts = {"available": odd_available} if odd_available else {}
    if odd_available < 12:
        odd_counts["unresolved"] = 12-odd_available
    audit.eq(summary.get("raw_odd_even_status_counts"), odd_counts, "SUMMARY/raw_odd_even_status_counts")
    audit.eq(summary.get("sym_positive_estimated_h_ratio_count"), sum(bool(value and value["ratio"]) for value in coefficients.values()), "SUMMARY/sym_ratio_count")


def hash_tree(value, audit, label="contract"):
    """Identity-only check of explicitly nested JSON bindings, without NPZ."""
    if isinstance(value, dict):
        if "path" in value and "sha256" in value and isinstance(value["path"], str):
            suffix = Path(value["path"]).suffix.lower()
            if suffix == ".json":
                audit.load(value, label)
            elif suffix != ".npz":
                try:
                    audit.eq(sha(resolve(value["path"])), value["sha256"], label+"/text_identity")
                except (OSError, ValueError) as error:
                    audit.errors.append({"label": label, "identity_error": str(error)})
        for key, item in value.items():
            hash_tree(item, audit, label+"/"+key)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            hash_tree(item, audit, label+f"/{i}")


def verify_manifest(contract_path, directory, cfg, audit):
    audit.eq(sha(directory/"CONTRACT.json"), sha(contract_path), "analysis/contract_copy")
    audit.eq(sha(directory/"SOURCE.py"), cfg["source_sha256"]["script/learned_local_analyze.py"], "analysis/source_copy")
    for name, expected in cfg.get("source_sha256", {}).items():
        audit.eq(sha(resolve(name)), expected, "analysis/live_bound_source/"+name)
    manifest = read(directory/"MANIFEST.json")
    rows = manifest.get("files", [])
    names = [row["path"] for row in rows]
    audit.eq(len(names), len(set(names)), "analysis/manifest_unique")
    audit.eq(set(names), {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file() and p.name != "MANIFEST.json"}, "analysis/manifest_coverage")
    audit.eq(manifest.get("manifest_self_excluded"), True, "analysis/manifest_self_exclusion")
    for row in rows:
        path = (directory/row["path"]).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("analysis manifest path escape")
        audit.eq(sha(path), row["sha256"], "analysis/manifest_hash/"+row["path"])
        audit.eq(path.stat().st_size, row["bytes"], "analysis/manifest_bytes/"+row["path"])


def verify_costs(cfg, saved, audit):
    producers = {row["id"]: row for row in cfg.get("producer_runs", [])}
    counts, roots, durations = Counter(), set(), []
    counted = 0
    for record in cfg.get("phase_receipts", []):
        producer = producers.get(record.get("producer_id"))
        if producer is None:
            audit.pending("cost/"+str(record.get("phase_id")), "Producer root not declared.")
            continue
        path, root = resolve(record["path"]), resolve(producer["root"])
        if path.parent != root or path.name not in ("SUMMARY.json", "FAILED.json"):
            audit.eq(False, True, "cost/root_terminal_only/"+record["path"])
            continue
        canonical = str(root).casefold()
        if canonical in roots:
            audit.eq(False, True, "cost/duplicate_root/"+canonical)
            continue
        document = audit.load(record, "cost/"+record["phase_id"])
        if document is None:
            continue
        if document.get("phase") != producer["phase"]:
            audit.eq(document.get("phase"), producer["phase"], "cost/phase")
            continue
        values = document.get("budget", {}).get("counts", {})
        if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in values.values()):
            audit.eq(False, True, "cost/nonnegative_integer_counters")
            continue
        counts.update(values)
        roots.add(canonical)
        durations.append(document["budget"]["seconds"])
        counted += 1
        if path.name == "FAILED.json" or document.get("status") != "completed":
            audit.pending("cost/"+record["phase_id"], "Failed phase retained in actual counted work.")
    audit.eq(saved.get("counts"), dict(counts), "cost/independent_sum")
    audit.eq(saved.get("receipt_count"), len(cfg.get("phase_receipts", [])), "cost/receipt_count")
    audit.eq(saved.get("unique_phase_receipts_counted"), counted, "cost/unique_root_count")
    audit.close(saved.get("sum_process_phase_duration_seconds"), math.fsum(durations), "cost/process_seconds_sum")
    audit.eq(saved.get("overall_elapsed_wall_seconds"), None, "cost/not_parallel_elapsed_wall")
    return {"counted_phase_roots": counted, "work_counts": dict(counts), "sum_process_seconds": math.fsum(durations)}


def verify_predictions(cfg, coefficients, audit):
    """Recompute all36 pre-validation predictions from full saved matrices."""
    document = audit.load(cfg.get("prediction_freeze"), "prediction_freeze")
    if document is None:
        return {"available": False}
    audit.eq(document.get("frozen_before_validation"), True, "prediction_freeze/pre_validation_flag")
    for field, target in (("coefficient_freeze", "coefficient_freeze"), ("analysis_policy", "policy")):
        source = document.get(field)
        binding = source if isinstance(source, dict) else {"path": source, "sha256": document.get(field+"_sha256")}
        audit.eq(str(resolve(binding["path"])).casefold(), str(resolve(cfg[target]["path"])).casefold(), "prediction_freeze/"+field+"/path")
        audit.eq(binding.get("sha256"), cfg[target]["sha256"], "prediction_freeze/"+field+"/sha256")
    for name, digest in document.get("input_sha256", {}).items():
        if Path(name).suffix.lower() == ".json":
            audit.load({"path": name, "sha256": digest}, "prediction_freeze/input/"+name)
    cfreeze = audit.load(cfg.get("coefficient_freeze"), "prediction_freeze/coefficient_index") or {}
    references = {(row["rep"], row["N"], row["kind"]): row for row in cfreeze.get("outputs", [])}
    rows = unique_index(document.get("predictions", []), ("rep", "N", "kind", "t"), audit, "frozen_predictions")
    expected = {(*key, tv) for key in keys() for tv in ((.1, .2) if key[2] == "sym" else (-.2, -.1, .1, .2))}
    audit.eq(set(rows), expected, "frozen_predictions/36_fixed_slots")
    verified = 0
    for key in sorted(expected):
        row, coefficient = rows.get(key), coefficients.get(key[:3])
        label = "frozen_prediction/"+repr(key)
        if row is None or coefficient is None:
            audit.pending(label, "Pre-validation prediction or complete coefficient matrix unavailable.")
            continue
        reference = references.get(key[:3])
        if reference is None:
            audit.pending(label, "Coefficient freeze identity unavailable.")
            continue
        audit.eq(row.get("alpha"), ALPHA, label+"/alpha")
        audit.array(row.get("v"), [1/math.sqrt(2), 1/math.sqrt(2), 0.], label+"/v")
        audit.eq(row.get("R"), R, label+"/R")
        audit.eq(row.get("coefficient_B"), 256, label+"/coefficient_B")
        audit.eq(row.get("randomization_ids"), list(range(R)), label+"/randomization_ids")
        audit.eq(row.get("full_matrix_quadratic_forms_used"), True, label+"/full_matrices")
        audit.eq(row.get("diagonal_or_trace_substitution"), False, label+"/no_trace_substitution")
        audit.eq(row.get("coefficient_statistics"), reference["statistics"], label+"/coefficient_source")
        audit.eq(row.get("coefficient_statistics_sha256"), reference["statistics_sha256"], label+"/coefficient_sha256")
        tval = key[3]
        stat = coefficient["statistics"]
        terms = {
            "linear_g": [(ALPHA-1)*tval*dot_chain(g) for g in stat["cv_g"]["replicates"]],
            "quadratic_H": [.5*(ALPHA*ALPHA-1)*tval*tval*quadratic_chain(H) for H in stat["cv_H"]["replicates"]],
            "teacher_cross_B": [-(ALPHA-1)*tval*tval*quadratic_chain(B) for B in stat["cv_B"]["replicates"]]}
        prediction = [a+b+c for a, b, c in zip(terms["linear_g"], terms["quadratic_H"], terms["teacher_cross_B"])]
        audit.eq(set(row.get("components_replicates", {})), set(terms), label+"/component_names")
        for name, values in terms.items():
            audit.array(row.get("components_replicates", {}).get(name), values, label+"/"+name)
        audit.array(row.get("prediction_replicates"), prediction, label+"/replicates")
        audit.close(row.get("prediction_mean"), statistics.mean(prediction), label+"/mean")
        audit.close(row.get("prediction_SE"), statistics.stdev(prediction)/4, label+"/SE")
        verified += 1
    return {"available": verified == 36, "expected_rows": 36, "verified_rows": verified,
            "row_components_use_full_g_H_B": True, "creation_time_is_recorded_provenance_not_external_timestamp_certificate": True}


def verify(contract_path, directory):
    audit = Audit()
    report = {"schema": "independent_learned_local_saved_verification_v1", "verifier_sha256": sha(__file__),
              "contract": str(contract_path), "analysis": str(directory), "scope": {
                  "JSON_identity_and_saved_arithmetic_only": True, "analyzer_or_scientific_target_imported": False,
                  "NPZ_read": False, "new_GH_RNG_fits_training": 0,
                  "not_certified": "Detailed producer-flow semantic audit, probabilistic seed independence, true integrals, learned-density population quality, global optima, or the separate 6597-check training chain.",
                  "comparison_absolute_tolerance": 2e-12, "comparison_relative_tolerance": 2e-11}}
    try:
        cfg = read(contract_path)
        audit.eq(cfg.get("schema"), "learned_local_analysis_contract_v1", "contract/schema")
        audit.eq(cfg.get("frozen"), True, "contract/frozen")
        hash_tree(cfg, audit)
        verify_manifest(contract_path, directory, cfg, audit)
        policy = audit.load(cfg.get("policy"), "policy")
        if policy is None:
            raise ValueError("fixed policy unavailable")
        for key, expected in (("training_repeats", [0, 1, 2]), ("training_N", list(SIZES)), ("proposal_kinds", list(KINDS)),
                              ("alpha", ALPHA), ("sym_t", [.1, .2]), ("raw_t", [-.2, -.1, .1, .2])):
            audit.eq(policy.get(key), expected, "policy/"+key)
        for key, expected in (("size", FAMILY), ("df", DF), ("alpha", .05)):
            audit.eq(policy.get("primary_family", {}).get(key), expected, "policy/family/"+key)
        analysis, summary = read(directory/"ANALYSIS.json"), read(directory/"SUMMARY.json")
        audit.eq(analysis.get("summary"), summary, "analysis/summary_copy")
        audit.eq(analysis.get("policy"), policy, "analysis/policy_copy")
        models, coefficient_source, contrast_source = source_indices(cfg, audit)
        coefficients = reconstruct_coefficients(coefficient_source, models, analysis.get("coefficient_cells", []), audit)
        report["prediction_freeze"] = verify_predictions(cfg, coefficients, audit)
        contrasts = reconstruct_contrasts(coefficients, contrast_source, analysis.get("primary_cells", []), analysis.get("raw_cells", []), audit)
        odd_available = verify_odd_even(coefficients, contrasts, analysis.get("raw_odd_even", []), audit)
        verify_csv(directory, coefficients, contrasts, audit)
        verify_counts(summary, coefficients, contrasts, odd_available, audit)
        report["costs"] = verify_costs(cfg, summary.get("costs", {}), audit)
        report["available_slots"] = {"coefficients": sum(v is not None for v in coefficients.values()),
            "primary": sum(v is not None for k, v in contrasts.items() if k[2] == "sym"),
            "raw": sum(v is not None for k, v in contrasts.items() if k[2] == "raw"), "odd_even": odd_available,
            "fixed_primary_family": FAMILY, "critical_value": critical()}
    except Exception as error:
        audit.errors.append({"label": "verification_exception", "type": type(error).__name__, "message": str(error)})
    status = "failed" if audit.errors else "consistent_but_unresolved" if audit.unresolved else "passed_complete"
    report.update(status=status, checks=audit.checks, mismatches=audit.errors, unresolved=audit.unresolved,
                  bound_JSON_files=len(audit.cache), complete_verification_passed=status == "passed_complete")
    return report


class ManufacturedTests(unittest.TestCase):
    def test_full_offdiagonal_prediction_and_signed_linear_term(self):
        g = [[.1, .2, .3] for _ in range(R)]
        H = [[[1., .4, 0.], [.6, 2., 0.], [0., 0., 3.]] for _ in range(R)]
        B = [[[.2, .1, 0.], [.3, .4, 0.], [0., 0., .5]] for _ in range(R)]
        linear, quadratic = prediction_vectors(g, H, B)
        expected = .5*(ALPHA**2-1)*2-(ALPHA-1)*.5
        self.assertAlmostEqual(quadratic[0], expected)
        diagonal_only = .5*(ALPHA**2-1)*1.5-(ALPHA-1)*.3
        self.assertGreater(abs(quadratic[0]-diagonal_only), .01)
        self.assertAlmostEqual(linear[0], (ALPHA-1)*.3/math.sqrt(2))

    def test_index_paired_residual_SE_differs_from_separate_variance(self):
        coefficient = {"linear": [.2+.01*i for i in range(R)], "quadratic": [.1]*R}
        prediction = [.1*l+.01*q for l, q in zip(coefficient["linear"], coefficient["quadratic"])]
        D = [p+.0001*(-1)**i for i, p in enumerate(prediction)]
        got = contrast_arithmetic(coefficient, D, D, .1)
        expected = [(d-p)/.01 for d, p in zip(D, prediction)]
        self.assertAlmostEqual(got["normalized_residual"]["SE"], statistics.stdev(expected)/4)
        separate = math.sqrt(got["D"]["SE"]**2+got["prediction"]["SE"]**2)/.01
        self.assertGreater(abs(separate-got["normalized_residual"]["SE"]), .1)

    def test_fixed_family_critical_and_missing_slot(self):
        self.assertAlmostEqual(float(student_t.sf(critical(), DF)), .05/48, places=14)
        coefficient = {key: None for key in keys()}
        source = {}
        primary = [{"rep": rep, "N": n, "kind": "sym", "t": tval, "status": "unresolved"}
                   for rep in range(3) for n in SIZES for tval in (.1, .2)]
        raw = [{"rep": rep, "N": n, "kind": "raw", "t": tval, "status": "unresolved"}
               for rep in range(3) for n in SIZES for tval in (-.2, -.1, .1, .2)]
        audit = Audit()
        got = reconstruct_contrasts(coefficient, source, primary, raw, audit)
        self.assertEqual(len(got), 36)
        self.assertEqual(sum(k[2] == "sym" for k in got), 12)
        self.assertFalse(audit.errors)
        self.assertEqual(len(audit.unresolved), 36)
        self.assertEqual(interval(summarize([float(i) for i in range(R)]))["family_size"], 24)

    def test_raw_odd_even_and_scalar_summary(self):
        linear = [.3+.001*i for i in range(R)]
        quadratic = [-.2+.002*i for i in range(R)]
        plus = [.2*l+.04*q for l, q in zip(linear, quadratic)]
        minus = [-.2*l+.04*q for l, q in zip(linear, quadratic)]
        got = odd_even_arithmetic(plus, minus, linear, quadratic, .2)
        for actual, expected in zip(got["odd_coefficient"], linear):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(got["even_coefficient"], quadratic):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(summarize(linear)["SE"], statistics.stdev(linear)/4)

    def test_response_ratio_uses_delta_method_and_only_sym(self):
        h = [1+.01*i for i in range(R)]
        b = [.3+.004*i+.001*(-1)**i for i in range(R)]
        ratio = response_ratio(h, b, "sym")
        expected = statistics.mean(b)/statistics.mean(h)
        influence = [(bv-expected*hv)/statistics.mean(h) for hv, bv in zip(h, b)]
        self.assertAlmostEqual(ratio["estimate"], expected)
        self.assertAlmostEqual(ratio["delta_SE"], statistics.stdev(influence)/4)
        self.assertIsNone(response_ratio(h, b, "raw"))
        self.assertIsNone(response_ratio([-x for x in h], b, "sym"))

    def test_costs_count_only_unique_explicit_phase_roots(self):
        cfg = {"producer_runs": [{"id": "qa", "root": "artifacts/MANUFACTURED_ONLY/qa", "phase": "qa"},
                                 {"id": "training", "root": "artifacts/MANUFACTURED_ONLY/training", "phase": "training"}],
               "phase_receipts": [{"producer_id": key, "phase_id": key, "path": f"artifacts/MANUFACTURED_ONLY/{key}/SUMMARY.json", "sha256": "unused"}
                                  for key in ("qa", "training")]}
        docs = {cfg["phase_receipts"][0]["path"]: {"phase": "qa", "status": "completed", "budget": {"counts": {"work": 7}, "seconds": 1.5}},
                cfg["phase_receipts"][1]["path"]: {"phase": "training", "status": "completed", "budget": {"counts": {"work": 11}, "seconds": 2.5}}}
        audit = Audit()
        audit.load = lambda binding, label: docs[binding["path"]]
        saved = {"counts": {"work": 18}, "receipt_count": 2, "unique_phase_receipts_counted": 2,
                 "sum_process_phase_duration_seconds": 4., "overall_elapsed_wall_seconds": None}
        result = verify_costs(cfg, saved, audit)
        self.assertFalse(audit.errors)
        self.assertEqual(result["work_counts"]["work"], 18)
        cfg["phase_receipts"].append(dict(cfg["phase_receipts"][0]))
        saved["receipt_count"] = 3
        audit = Audit()
        audit.load = lambda binding, label: docs[binding["path"]]
        result = verify_costs(cfg, saved, audit)
        self.assertTrue(any("duplicate_root" in e["label"] for e in audit.errors))
        self.assertEqual(result["work_counts"]["work"], 18)

    def test_pre_validation_prediction_binding_components_and_tampering(self):
        cfg = {"prediction_freeze": {"path": "artifacts/MANUFACTURED_ONLY/predictions.json", "sha256": "a"*64},
               "coefficient_freeze": {"path": "artifacts/MANUFACTURED_ONLY/coefficients.json", "sha256": "b"*64},
               "policy": {"path": "artifacts/MANUFACTURED_ONLY/policy.json", "sha256": "c"*64}}
        coefficients, references, rows = {}, [], []
        for key in keys():
            g = [[.1+.001*i, .2, .3] for i in range(R)]
            H = [[[1+.01*i, .4, 0.], [.6, 2., 0.], [0., 0., 3.]] for i in range(R)]
            B = [[[.2, .1, 0.], [.3, .4+.002*i, 0.], [0., 0., .5]] for i in range(R)]
            coefficients[key] = {"statistics": {name: summarize(value) for name, value in (("cv_g", g), ("cv_H", H), ("cv_B", B))}}
            source = f"artifacts/MANUFACTURED_ONLY/coeff_{key[0]}_{key[1]}_{key[2]}.json"
            references.append({"rep": key[0], "N": key[1], "kind": key[2], "statistics": source, "statistics_sha256": "d"*64})
            for tval in ((.1, .2) if key[2] == "sym" else (-.2, -.1, .1, .2)):
                components = {"linear_g": [(ALPHA-1)*tval*(.3+.001*i)/math.sqrt(2) for i in range(R)],
                              "quadratic_H": [.5*(ALPHA**2-1)*tval*tval*(4+.01*i)/2 for i in range(R)],
                              "teacher_cross_B": [-(ALPHA-1)*tval*tval*(1+.002*i)/2 for i in range(R)]}
                p = [sum(items) for items in zip(*components.values())]
                rows.append({"rep": key[0], "N": key[1], "kind": key[2], "t": tval, "alpha": ALPHA,
                    "v": [2**-.5, 2**-.5, 0.], "R": R, "coefficient_B": 256, "randomization_ids": list(range(R)),
                    "full_matrix_quadratic_forms_used": True, "diagonal_or_trace_substitution": False,
                    "coefficient_statistics": source, "coefficient_statistics_sha256": "d"*64,
                    "components_replicates": components, "prediction_replicates": p,
                    "prediction_mean": statistics.mean(p), "prediction_SE": statistics.stdev(p)/4})
        prediction = {"frozen_before_validation": True, "coefficient_freeze": cfg["coefficient_freeze"],
                      "analysis_policy": cfg["policy"], "input_sha256": {}, "predictions": rows}
        documents = {cfg["prediction_freeze"]["path"]: prediction,
                     cfg["coefficient_freeze"]["path"]: {"outputs": references}}
        audit = Audit()
        audit.load = lambda binding, label: documents[binding["path"]]
        result = verify_predictions(cfg, coefficients, audit)
        self.assertFalse(audit.errors)
        self.assertEqual(result["verified_rows"], 36)
        rows[0]["components_replicates"]["quadratic_H"][0] += .001
        audit = Audit()
        audit.load = lambda binding, label: documents[binding["path"]]
        verify_predictions(cfg, coefficients, audit)
        self.assertTrue(audit.errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ManufacturedTests))
        return 0 if result.wasSuccessful() else 1
    if not args.contract or not args.analysis or not args.out:
        parser.error("--contract --analysis --out are required")
    out, directory = args.out.resolve(), args.analysis.resolve()
    if out.exists() or out.is_relative_to(directory):
        parser.error("--out must be new and outside the sealed analysis directory")
    report = verify(args.contract.resolve(), directory)
    with out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(clean(report), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "checks": report["checks"], "mismatches": len(report["mismatches"]),
                      "unresolved": len(report["unresolved"]), "out": str(out)}))
    return 1 if report["mismatches"] else 2 if report["unresolved"] else 0


if __name__ == "__main__":
    sys.exit(main())
