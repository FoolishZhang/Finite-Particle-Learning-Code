"""Read-only execution audit of the frozen RankOneStability-v1 outputs.

No study kernel/driver is imported. All saved per-label quantities are reweighted,
and fixed-label numerical checks use numpy Legendre nodes and adaptive quad_vec.
--run exclusively creates AUDIT.json; --check repeats without writing anything.
Numerical qualification failures in the study are retained, not audit failures.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import scipy
from scipy.integrate import quad_vec

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/rankone_stability_v1_20261007"
SOURCE = ROOT / "artifacts/joint_dependence_v1_20261007"
RECEIPT = ART / "AUDIT.json"
KS = (8, 64, "inf")
TARGETS = ("empirical", "population")
ANCHORS = ("b8_s8", "b8_sREF", "bREF_s8", "bREF_sREF", "b8_lambda025", "b8_lambda050", "b8_lambda075")
RULES = {
    "A": ("hermite", 96, None, -24., 48., .25),
    "B": ("legendre", 256, 10., -24., 48., .25),
    "C": ("legendre", 384, 10., -24., 56., .125),
    "D": ("legendre", 512, 11., -28., 64., .125),
}
GATES = dict(matrix_difference=2e-7, eigenvalue_difference=2e-6, ward_residual=2e-7,
             normal_mass_error=2e-12, d_mass_error=2e-7, c_mass_error=2e-7, sign_resolution=1e-5)
SAMPLE_ANCHORS = ("b8_s8", "b8_sREF")
LABELS = (0, 85, 170, 255)
QUAD_CONFIG = dict(inner_rule="numpy.polynomial.legendre.leggauss", inner_order=512,
    normal_limit=11., outer_rule="scipy.integrate.quad_vec", interval=[-28., 64.],
    breakpoints=[-10., -5., 0., 5., 10., 20., 32.], epsabs=1e-10, epsrel=1e-9,
    norm="max", comparison_absolute_tolerance=5e-9)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def native(x):
    if isinstance(x, np.ndarray):
        return native(x.tolist())
    if isinstance(x, np.generic):
        return native(x.item())
    if isinstance(x, dict):
        return {str(k): native(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [native(v) for v in x]
    return x


class Audit:
    def __init__(self):
        self.assertions = 0
        self.failures = []
        self.maxima = {}
        self.inputs = {}
        self.numeric_elements = 0

    def require(self, ok, context, detail=None):
        self.assertions += 1
        if not bool(ok):
            self.failures.append(dict(context=context, detail=native(detail)))

    def close(self, actual, expected, field, context, tol=2e-12):
        a, b = np.asarray(actual), np.asarray(expected)
        same_shape = a.shape == b.shape
        finite = same_shape and np.isfinite(a).all() and np.isfinite(b).all()
        error = float(np.max(np.abs(a - b), initial=0.)) if finite else None
        self.numeric_elements += int(a.size)
        if error is not None and error > self.maxima.get(field, {}).get("error", -1.):
            self.maxima[field] = dict(error=error, context=context)
        self.require(finite and error <= tol, context + ":" + field,
                     dict(max_abs_error=error, tolerance=tol, actual_shape=a.shape, expected_shape=b.shape))
        return error

    def exact(self, actual, expected, context):
        self.require(native(actual) == native(expected), context)

    def bind(self, path, expected=None, size=None):
        path = Path(path)
        name = rel(path)
        digest = sha(path)
        if name in self.inputs:
            self.exact(digest, self.inputs[name], name + ":unchanged_during_audit")
        self.inputs[name] = digest
        if expected is not None:
            self.exact(digest, expected, name + ":sha256")
        if size is not None:
            self.exact(path.stat().st_size, size, name + ":bytes")
        return digest


def helmert():
    Q = np.zeros((8, 7))
    for k in range(1, 8):
        Q[:k, k - 1] = 1 / math.sqrt(k * (k + 1))
        Q[k, k - 1] = -k / math.sqrt(k * (k + 1))
    return Q


def expected_anchors(candidates):
    byname = {row["name"]: row for row in candidates}
    bs = {name: np.asarray(byname[name]["b"]) for name in ("K8", "REF")}
    ss = {name: np.sqrt(np.sum(np.asarray(byname[name]["W"]) ** 2, axis=1)) for name in bs}
    out = []
    for name, bias, lam in zip(ANCHORS, ("K8", "K8", "REF", "REF", "K8", "K8", "K8"), (0., 1., 0., 1., .25, .5, .75)):
        a = (1 - lam) * ss["K8"] + lam * ss["REF"]
        out.append(dict(name=name, b=bs[bias], a=a, bias_source=bias, lambda_value=lam,
                        W=np.column_stack([a, np.zeros(8)]), canonical_common_line=True,
                        original_training_endpoint=False))
    return native(out)


def reconstruct(raw, q, a, audit, context):
    """Sum each label directly, then assemble pairwise edges explicitly."""
    Q = helmert()
    out = {key: raw[key] for key in ("label_M", "label_bias_gradient", "label_radial_gradient",
        "label_sigma_prime_average", "label_d_mass", "label_c_mass", "c_mass_upper", "normal_quadrature_mass")}
    shapes = dict(laplacian=(8, 8), edge_weights=(8, 8), Hperp=(8, 8),
        ward_angle_hessian=(8, 8), ward_residual=(8, 8), bias_gradient=(8,), radial_gradient=(8,),
        ward_residual_max=(), common_angle_residual=(), eigenvalues=(8,), projected_hessian=(7, 7),
        projected_eigenvalues=(7,), target_d_mass=(), target_c_mass=())
    for key, shape in shapes.items():
        out[key] = np.empty((2, 3) + shape)
    audit.close(raw["helmert_basis"], Q, "helmert", context, 0.)
    audit.close(Q.T @ Q, np.eye(7), "helmert_orthogonality", context)
    for t, weights in enumerate(q):
        for ki in range(3):
            M = np.sum(weights[:, None, None] * raw["label_M"][ki], axis=0)
            gb = np.sum(weights[:, None] * raw["label_bias_gradient"][ki], axis=0)
            ga = np.sum(weights[:, None] * raw["label_radial_gradient"][ki], axis=0)
            sp = np.sum(weights[:, None] * raw["label_sigma_prime_average"][ki], axis=0)
            edges = np.zeros((8, 8)); L = np.zeros((8, 8))
            for i in range(8):
                for j in range(8):
                    if i != j:
                        e = a[i] * a[j] * M[i, j]
                        edges[i, j] = e
                        L[i, j] = -e
                        L[i, i] += e
            H = np.diag(sp) - M
            direct = np.diag(a) @ H @ np.diag(a) - np.diag(a * ga)
            ward = direct - L
            P = Q.T @ L @ Q
            cells = dict(laplacian=L, edge_weights=edges, Hperp=H, ward_angle_hessian=direct,
                ward_residual=ward, bias_gradient=gb, radial_gradient=ga,
                ward_residual_max=np.max(abs(ward)), common_angle_residual=np.max(abs(L.sum(axis=1))),
                eigenvalues=np.linalg.eigvalsh(L), projected_hessian=P, projected_eigenvalues=np.linalg.eigvalsh(P),
                target_d_mass=weights @ raw["label_d_mass"][ki], target_c_mass=weights @ raw["label_c_mass"][ki])
            for key, value in cells.items():
                out[key][t, ki] = value
            V, values = raw["projected_eigenvectors"][t, ki], raw["projected_eigenvalues"][t, ki]
            audit.close(P @ V, V * values, "projected_eigenpair_residual", context)
            audit.close(V.T @ V, np.eye(7), "eigenvector_orthogonality", context)
            audit.close(raw["angle_eigenvectors"][t, ki], Q @ V, "angle_eigenvectors", context)
    for key in shapes:
        audit.close(raw[key], out[key], "reconstruction." + key, context)
    audit.close(raw["label_d_mass_error"], raw["label_d_mass"] - 1, "label_d_mass_error", context, 0.)
    audit.close(raw["label_c_mass_upper_excess"], raw["label_c_mass"] - raw["c_mass_upper"][:, None], "label_c_mass_upper_excess", context, 0.)
    return out


def mass(r, ki):
    d, c = r["label_d_mass"][ki], r["label_c_mass"][ki]
    up = float(r["c_mass_upper"][ki]); idx = int(np.argmax(abs(d - 1)))
    nm = float(r["normal_quadrature_mass"])
    return dict(normal_mass=nm, normal_mass_error=abs(nm - 1), d_mass_max_error=float(max(abs(d - 1))),
        d_mass_worst_label=idx, d_mass_at_worst_label=float(d[idx]), d_mass_min=float(min(d)), d_mass_max=float(max(d)),
        c_mass_min=float(min(c)), c_mass_max=float(max(c)), c_mass_exact_upper=up,
        c_mass_lower_violation=max(0., float(-min(c))), c_mass_upper_violation=max(0., float(max(c) - up)),
        c_mass_min_label=int(np.argmin(c)), c_mass_max_label=int(np.argmax(c)),
        scope="all labels, including labels with zero empirical frequency")


def intrinsic(r, ti, ki):
    m = mass(r, ki)
    return dict(independent_Ward=float(r["ward_residual_max"][ti, ki]) <= 2e-7,
        normal_mass_error=m["normal_mass_error"] <= 2e-12, d_mass_max_error=m["d_mass_max_error"] <= 2e-7,
        c_mass_lower_violation=m["c_mass_lower_violation"] <= 2e-7,
        c_mass_upper_violation=m["c_mass_upper_violation"] <= 2e-7)


def comparison(left, right, ti, ki):
    fields = dict(laplacian_max_difference="laplacian", projected_hessian_max_difference="projected_hessian",
        projected_eigenvalue_max_difference="projected_eigenvalues", Hperp_max_difference="Hperp",
        ward_angle_hessian_max_difference="ward_angle_hessian", radial_gradient_max_difference="radial_gradient",
        bias_gradient_max_difference="bias_gradient")
    result = {k: float(np.max(abs(left[v][ti, ki] - right[v][ti, ki]))) for k, v in fields.items()}
    for key in ("label_d_mass", "label_c_mass", "label_M", "label_radial_gradient", "label_bias_gradient"):
        result[key + "_max_difference"] = float(np.max(abs(left[key][ki] - right[key][ki])))
    return result


def classification(value, passed):
    if not passed:
        return "unresolved_numerical_qualification"
    if value > 1e-5:
        return "numerically_qualified_angular_local_minimum"
    if value < -1e-5:
        return "numerically_qualified_negative_angular_curvature"
    return "unresolved_at_fixed_sign_resolution"


def check_summary(summary, raw, rebuilt, a, audit, context):
    audit.exact(len(summary["matrices"]), 6, context + ":summary_matrix_count")
    for index, row in enumerate(summary["matrices"]):
        ti, ki = divmod(index, 3)
        audit.exact((row["target"], row["K"]), (TARGETS[ti], KS[ki]), context + ":matrix_identity")
        for key in ("laplacian", "edge_weights", "Hperp", "ward_angle_hessian", "ward_residual", "eigenvalues",
                    "projected_hessian", "projected_eigenvalues", "projected_eigenvectors", "angle_eigenvectors",
                    "radial_gradient", "bias_gradient", "target_d_mass", "target_c_mass"):
            audit.close(row[key], raw[key][ti, ki], "json_npz." + key, context, 0.)
        direct = rebuilt["ward_angle_hessian"][ti, ki]
        ga, gb = rebuilt["radial_gradient"][ti, ki], rebuilt["bias_gradient"][ti, ki]
        extra = dict(radial_gradient_L2=np.linalg.norm(ga), radial_gradient_Linf=max(abs(ga)),
            bias_gradient_L2=np.linalg.norm(gb), bias_gradient_Linf=max(abs(gb)),
            independent_Ward_max=rebuilt["ward_residual_max"][ti, ki],
            direct_common_angle_residual=max(abs(direct.sum(axis=-1))),
            Cartesian_Ward_max=max(abs(rebuilt["Hperp"][ti, ki] @ a - ga)),
            structural_laplacian_rowsum_max=rebuilt["common_angle_residual"][ti, ki])
        for key, value in extra.items():
            audit.close(row[key], value, "json_derived." + key, context)
        gates = intrinsic(rebuilt, ti, ki)
        audit.exact(row["intrinsic_diagnostic_gates"], gates, context + ":intrinsic_gates")
        audit.exact(row["failed_intrinsic_diagnostic_gates"], [k for k, v in gates.items() if not v], context + ":failed_intrinsic_gates")
        audit.exact(row["scalar_diagnostics"], mass(rebuilt, ki), context + ":mass_diagnostics")


def check_qualification(saved, rules, audit, context):
    records = []
    audit.exact(len(saved["targets"]), 6, context + ":qualification_matrix_count")
    for idx, row in enumerate(saved["targets"]):
        ti, ki = divmod(idx, 3)
        audit.exact((row["target"], row["K"]), (TARGETS[ti], KS[ki]), context + ":qualification_identity")
        comp = comparison(rules["C"], rules["D"], ti, ki)
        gates = dict(matrix_difference=comp["laplacian_max_difference"] <= 2e-7,
                     projected_eigenvalue_difference=comp["projected_eigenvalue_max_difference"] <= 2e-6)
        for name in ("C", "D"):
            gates.update({name + "_" + k: v for k, v in intrinsic(rules[name], ti, ki).items()})
            audit.exact(row["scalar_diagnostics"][name], mass(rules[name], ki), context + ":qualification_mass")
        passed = all(gates.values())
        value = float(rules["C"]["projected_eigenvalues"][ti, ki, 0])
        audit.exact(row["gates"], gates, context + ":qualification_gates")
        audit.exact(row["qualified"], passed, context + ":qualified")
        audit.exact(row["failed_gates"], [k for k, v in gates.items() if not v], context + ":failed_gates")
        audit.exact(row["classification"], classification(value, passed), context + ":classification")
        audit.exact(row["raw_C_sign_classification_ignoring_qualification"], classification(value, True), context + ":raw_classification")
        for key, val in comp.items():
            audit.close(row["C_D"][key], val, "qualification_comparison." + key, context)
        for rn in ("C", "D"):
            audit.close(row[rn + "_minimum_projected_eigenvalue"], rules[rn]["projected_eigenvalues"][ti, ki, 0], "qualification_minimum", context)
        for l, r in itertools.combinations(RULES, 2):
            for key, val in comparison(rules[l], rules[r], ti, ki).items():
                audit.close(row["all_rule_pair_comparisons"][l + "_" + r][key], val, "all_pair_comparison." + key, context)
        records.append(dict(target=TARGETS[ti], K=KS[ki], qualified=passed, classification=classification(value, passed),
                            minimum=value, failed_gates=[k for k, v in gates.items() if not v]))
    audit.exact(saved["all_primary_matrices_qualified"], all(r["qualified"] for r in records), context + ":all_primary_qualified")
    return records


def independent_label(b, a, yi, z, w):
    """Independent ordinary-probability RS4 integration; no kernel log helpers."""
    y = np.array([(yi >> j) & 1 for j in range(7, -1, -1)])
    logits = b[None, :] + z[:, None] * a[None, :]
    p = np.exp(-np.logaddexp(0., -logits))
    A = np.prod(np.where(y[None, :] == 1, p, np.exp(-np.logaddexp(0., logits))), axis=1)
    evidence = w @ A
    x = A / evidence
    r = y - p
    rr = (r[:, :, None] * r[:, None, :]).reshape(len(z), 64)
    d_features = np.column_stack((-r, -z[:, None] * r, p * (1 - p), np.ones(len(z))))
    c_features = np.column_stack((rr, np.ones(len(z))))

    def integrand(v):
        u = math.exp(v)
        laplace = np.exp(-u * x)
        psi = float(w @ laplace)
        wx = w * x * laplace
        b0 = float(wx.sum())
        dm = wx @ d_features
        cm = wx @ c_features
        rows = []
        for K in (8, 64):
            rows.append(np.r_[K * (K - 1) * u * u * b0 * psi ** (K - 2) * cm,
                              K * u * psi ** (K - 1) * dm])
        return np.asarray(rows)

    values, error, info = quad_vec(integrand, -28., 64., points=QUAD_CONFIG["breakpoints"],
        epsabs=1e-10, epsrel=1e-9, norm="max", full_output=True)
    posterior = w * A / evidence
    post = np.r_[posterior @ c_features, posterior @ d_features]
    all_values = np.vstack([values, post])
    # Packing: M[64], c_mass, g_b[8], g_a[8], sigma_prime[8], d_mass.
    result = dict(label_M=all_values[:, :64].reshape(3, 8, 8), label_c_mass=all_values[:, 64],
        label_bias_gradient=all_values[:, 65:73], label_radial_gradient=all_values[:, 73:81],
        label_sigma_prime_average=all_values[:, 81:89], label_d_mass=all_values[:, 89])
    return result, dict(label_index=yi, estimated_outer_error=float(error), quad_success=bool(info.success),
        quad_status=int(info.status), quad_message=str(info.message), integrand_evaluations=int(info.neval),
        adaptive_intervals=int(len(info.intervals)), evidence=float(evidence))


def check_directional(audit):
    path = ART / "DIRECTIONAL_ANALYSIS.json"
    audit.bind(path)
    data = read(path)["validation"]
    audit.bind(ROOT / "script/rankone_stability_analysis.py", data["source_sha256"])
    audit.bind(ROOT / "docs/08_research_研究工作/RankOneStability_v1_20261007/ANALYSIS_RULES.md", data["rule_sha256"])
    for name, digest in data["sources"].items():
        audit.bind(ROOT / name, digest)
    for key in ("new_quadrature_evaluations", "new_random_draws"):
        audit.exact(data[key], 0, "directional:" + key)
    audit.exact(data["classification_threshold"], 1e-5, "directional:sign_threshold")
    audit.exact(len(data["directional_rows"]), 28, "directional:rows")
    audit.exact(len(data["projected_spectra"]), 168, "directional:spectra")
    audit.exact(len(data["K8_length_interpolation"]), 8, "directional:interpolation_rows")
    audit.exact(sorted((r["rep"], r["anchor"]) for r in data["directional_rows"]),
                sorted(itertools.product(range(4), ANCHORS)), "directional:all_anchor_pairs")
    spectral = {(r["rep"], r["anchor"], r["target"], r["K"]): r for r in data["projected_spectra"]}
    rebuilt_spectra = {}
    Q = helmert()

    def sign(x, ok):
        return "unqualified" if not ok else ("positive" if x > 1e-5 else "negative" if x < -1e-5 else "unresolved")

    for row in data["directional_rows"]:
        rep, name = row["rep"], row["anchor"]
        context = f"directional/rep{rep}/{name}"
        folder = ART / f"rep_{rep:02d}"
        qual = read(folder / (name + "_QUALIFICATION.json"))
        matrices = []
        for rn in ("C", "D"):
            matrices.append({(r["target"], str(r["K"])): r for r in read(folder / (name + "_" + rn + ".json"))["matrices"]})
        C, D = matrices
        qualifies = {(r["target"], str(r["K"])): r["qualified"] for r in qual["targets"]}
        R = np.asarray(C[("population", "inf")]["laplacian"])
        ev, V = np.linalg.eigh(Q.T @ R @ Q)
        psi = Q @ V[:, 0]
        if psi[np.argmax(abs(psi))] < 0:
            psi *= -1
        stored_psi = np.asarray(row["psi"])
        audit.close(stored_psi, psi, "directional.selected_vector", context)
        audit.close(stored_psi @ stored_psi, 1., "directional.unit_norm", context)
        audit.close(stored_psi.sum(), 0., "directional.rotation_orthogonality", context)
        audit.require(stored_psi[np.argmax(abs(stored_psi))] > 0, context + ":canonical_sign")
        for key, value in dict(selection_eigenvalue=ev[0], selection_eigengap=ev[1]-ev[0],
                               normalization=psi@psi, common_rotation_component=psi.sum()).items():
            audit.close(row[key], value, "directional." + key, context)
        audit.exact(row["selection"], "C population integrated minimum projected eigenvector", context + ":selection_rule")
        audit.exact(row["risk_direction_is_oracle_diagnostic_not_training_input"], True, context + ":scope")
        audit.exact(len(row["curvatures"]), 6, context + ":curvature_count")
        signs = {}
        for cell in row["curvatures"]:
            key = (cell["target"], cell["K"])
            H, HD = np.asarray(C[key]["laplacian"]), np.asarray(D[key]["laplacian"])
            value = float(psi @ H @ psi); check = float(psi @ HD @ psi)
            ok = qualifies[key] and qualifies[("population", "inf")]
            for field, expected in dict(C=value, D=check, C_D_difference=abs(value-check)).items():
                audit.close(cell[field], expected, "directional.curvature_" + field, context)
            audit.exact(cell["qualified"], ok, context + ":curvature_qualification")
            signs[key] = sign(value, ok)
            audit.exact(cell["sign"], signs[key], context + ":curvature_sign")
            vals = np.linalg.eigvalsh(Q.T @ H @ Q)
            skey = (rep, name, *key)
            spec = spectral[skey]
            for field, expected in dict(minimum=vals[0], maximum=vals[-1],
                bias_gradient_Linf=np.max(abs(np.asarray(C[key]["bias_gradient"]))),
                radial_gradient_Linf=np.max(abs(np.asarray(C[key]["radial_gradient"])))).items():
                audit.close(spec[field], expected, "directional.spectrum_" + field, context)
            audit.exact(spec["qualified"], qualifies[key], context + ":spectrum_qualification")
            audit.exact(spec["sign"], sign(vals[0], qualifies[key]), context + ":spectrum_sign")
            for field in ("bias_source", "lambda_value"):
                audit.exact(spec[field], qual["anchor"][field], context + ":spectrum_" + field)
            rebuilt_spectra[skey] = (float(vals[0]), sign(vals[0], qualifies[key]))
        risk_down = signs[("population", "inf")] == "negative"
        for field, expected in dict(risk_decreasing=risk_down,
            empirical_F8_rejects_risk_direction=risk_down and signs[("empirical", "8")] == "positive",
            population_F8_rejects_risk_direction=risk_down and signs[("population", "8")] == "positive",
            F64_population_accepts_risk_direction=risk_down and signs[("population", "64")] == "negative").items():
            audit.exact(row[field], expected, context + ":" + field)
    seq_names = ("b8_s8", "b8_lambda025", "b8_lambda050", "b8_lambda075", "b8_sREF")
    lambdas = [0., .25, .5, .75, 1.]
    for row in data["K8_length_interpolation"]:
        context = f"interpolation/rep{row['rep']}/{row['target']}"
        values, signs = zip(*(rebuilt_spectra[(row["rep"], n, row["target"], "8")] for n in seq_names))
        expected = [dict(left=lambdas[i], right=lambdas[i+1]) for i in range(4)
                    if {signs[i], signs[i+1]} == {"positive", "negative"}]
        audit.exact(row["lambdas"], lambdas, context + ":lambdas")
        audit.close(row["minimum_eigenvalues"], values, "interpolation.minima", context)
        audit.exact(row["signs"], signs, context + ":signs")
        audit.exact(row["sign_change_intervals"], expected, context + ":intervals")
        audit.exact(row["uniqueness_or_monotonicity_proved"], False, context + ":scope")
    return dict(directions=28, same_direction_curvatures=168, projected_spectra=168,
                interpolation_sequences=8, newly_integrated_points=0)


def perform():
    tick = time.perf_counter(); audit = Audit()
    # Refuse partial scientific output; this check precedes every raw-array read.
    for rep in range(4):
        result_path = ART / f"rep_{rep:02d}/RESULTS.json"
        if not result_path.exists() or read(result_path).get("status") != "completed":
            raise RuntimeError("All four RESULTS must be completed before this audit: " + rel(result_path))
    freeze_path = ART / "FREEZE.json"
    freeze_hash = audit.bind(freeze_path)
    frozen = read(freeze_path)
    audit.exact(frozen["config"]["thresholds"], GATES, "frozen_thresholds")
    audit.exact(frozen["config"]["K_values"], KS, "frozen_K")
    audit.exact(frozen["config"]["targets"], TARGETS, "frozen_targets")
    audit.exact(frozen["config"]["target_weights_renormalized"], False, "frozen_no_renormalization")
    for path, digest in frozen["sources"].items():
        audit.bind(ROOT / path, digest)
    labels = np.array(list(itertools.product((0, 1), repeat=8)), dtype=np.int8)
    ln, lw = np.polynomial.legendre.leggauss(512)
    z = 11. * ln
    w = 11. * lw * np.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    records, sampled, coarse_failures = [], [], []
    raw_count = 0
    for rep, stream in enumerate(frozen["streams"]):
        folder = ART / f"rep_{rep:02d}"
        source_folder = SOURCE / f"rep_{rep:02d}"
        for path, digest in stream["sources"].items():
            audit.bind(ROOT / path, digest)
        original = read(source_folder / "CANDIDATES_LOCK.json")
        old_results = read(source_folder / "RESULTS.json")
        qualification = read(source_folder / "ENDPOINT_QUALIFICATION.json")
        audit.exact(old_results["candidates"], original["candidates"], f"rep{rep}:old_candidates")
        audit.exact(old_results["frequency"], original["frequency"], f"rep{rep}:old_frequency")
        expected = expected_anchors(original["candidates"])
        audit.exact(stream["anchors"], expected, f"rep{rep}:frozen_anchor_fields")
        q = np.asarray([original["frequency"], qualification["teacher_pmf"]])
        audit.exact(stream["frequency"], q[0], f"rep{rep}:frozen_empirical_original")
        audit.exact(stream["teacher_pmf"], q[1], f"rep{rep}:frozen_population_original")
        audit.exact(stream["target_masses"], q.sum(axis=1), f"rep{rep}:frozen_target_mass")
        audit.require(np.all(q >= 0) and np.max(abs(q.sum(axis=1) - 1)) < 1e-12, f"rep{rep}:target_valid")
        lock = read(folder / "ANCHORS_LOCK.json")
        result = read(folder / "RESULTS.json")
        audit.exact(len(result["anchors"]), 7, f"rep{rep}:anchor_count")
        audit.exact(len(result["rule_results"]), 28, f"rep{rep}:rule_result_count")
        audit.exact(sorted((r["anchor_name"], r["rule"]) for r in result["rule_results"]),
                    sorted(itertools.product(ANCHORS, RULES)), f"rep{rep}:all_anchor_rule_pairs")
        audit.bind(folder / "RESULTS.json")
        audit.bind(folder / "ANCHORS_LOCK.json", result["anchors_lock_sha256"])
        audit.exact(lock["anchors"], expected, f"rep{rep}:locked_anchor_fields")
        audit.exact(lock["targets"], dict(zip(TARGETS, q)), f"rep{rep}:locked_unmodified_targets")
        audit.exact(lock["target_weights_renormalized"], False, f"rep{rep}:no_target_renormalization")
        audit.exact(result["target_weights_renormalized"], False, f"rep{rep}:result_no_target_renormalization")
        audit.exact(lock["locked_before_rankone_quadrature"], True, f"rep{rep}:anchor_lock_order")
        for obj in (result, lock, read(folder / "RUN_START.json")):
            audit.exact(obj["freeze_sha256"], freeze_hash, f"rep{rep}:freeze_reference")
        for item in result["artifacts"].values():
            audit.bind(ROOT / item["path"], item["sha256"], item["bytes"])
        audit.exact(result["source_hashes"], stream["sources"], f"rep{rep}:source_hash_references")
        audit.exact(lock["source_hashes"], stream["sources"], f"rep{rep}:lock_source_hashes")
        for anchor_idx, anchor in enumerate(expected):
            name = anchor["name"]; b, a = np.asarray(anchor["b"]), np.asarray(anchor["a"])
            audit.require(np.all(a > 0), f"rep{rep}/{name}:positive_amplitudes")
            rules, raw_rules = {}, {}
            for rn, (inner, n, lim, vlo, vhi, step) in RULES.items():
                context = f"rep_{rep:02d}/{name}/{rn}"
                path = folder / (name + "_" + rn + ".npz")
                with np.load(path, allow_pickle=False) as loaded:
                    raw = {key: loaded[key] for key in loaded.files}
                raw_count += 1
                summary = read(path.with_suffix(".json"))
                audit.exact((summary["anchor_name"], summary["rule"]), (name, rn), context + ":rule_identity")
                audit.bind(path, summary["array_artifact"]["sha256"], summary["array_artifact"]["bytes"])
                rr = next(r for r in result["rule_results"] if r["anchor_name"] == name and r["rule"] == rn)
                audit.bind(path.with_suffix(".json"), rr["summary_artifact"]["sha256"], rr["summary_artifact"]["bytes"])
                audit.exact(rr["array_artifact"], summary["array_artifact"], context + ":array_reference")
                for key, value in dict(b=b, a=a, patterns=labels, target_weights=q, target_names=TARGETS, K_names=[str(k) for k in KS]).items():
                    audit.exact(raw[key], value, context + ":input_" + key)
                audit.close(raw["c_mass_upper"], [1-1/8, 1-1/64, 1.], "c_mass_upper", context, 0.)
                for key, value in dict(normal_rule=inner, normal_order=n, normal_limit=lim, logu_min=vlo,
                    logu_max=vhi, logu_step=step, outer_rule="trapezoid", outer_order=None,
                    likelihood_scale_source="inner_quadrature_evidence", target_masses=q.sum(axis=1)).items():
                    audit.exact(summary["config"][key], value, context + ":config_" + key)
                v = np.linspace(vlo, vhi, round((vhi-vlo)/step) + 1)
                vw = np.full(len(v), step); vw[[0, -1]] /= 2
                audit.close(raw["outer_logu_nodes"], v, "outer_nodes", context, 0.)
                audit.close(raw["outer_weights"], vw, "outer_weights", context, 0.)
                audit.close(raw["normal_quadrature_mass"], raw["normal_weights"].sum(), "normal_mass", context, 0.)
                audit.close(raw["log_likelihood_scale"], raw["log_evidence"], "likelihood_scale", context, 0.)
                for key, value in raw.items():
                    if value.dtype.kind in "fc":
                        audit.require(np.isfinite(value).all(), context + ":finite_" + key)
                rebuilt = reconstruct(raw, q, a, audit, context)
                check_summary(summary, raw, rebuilt, a, audit, context)
                if rn in ("A", "B"):
                    for ti, target in enumerate(TARGETS):
                        for ki, K in enumerate(KS):
                            failed = [key for key, passed in intrinsic(rebuilt, ti, ki).items() if not passed]
                            if failed:
                                coarse_failures.append(dict(rep=rep, anchor=name, rule=rn, target=target, K=K, failed_gates=failed))
                raw_rules[rn], rules[rn] = raw, rebuilt
            saved_qual = read(folder / (name + "_QUALIFICATION.json"))
            audit.exact(saved_qual, result["anchors"][anchor_idx], f"rep{rep}/{name}:qualification_record")
            audit.exact(saved_qual["anchor"], anchor, f"rep{rep}/{name}:qualification_anchor")
            rows = check_qualification(saved_qual, rules, audit, f"rep{rep}/{name}")
            records.append(dict(rep=rep, anchor=name, target_K=rows))
            if name in SAMPLE_ANCHORS:
                for yi in LABELS:
                    independent, info = independent_label(b, a, yi, z, w)
                    info.update(rep=rep, anchor=name, errors={}, errors_by_K={}, independent_values=independent)
                    audit.require(info["quad_success"], f"rep{rep}/{name}/{yi}:adaptive_integral", info)
                    for key, value in independent.items():
                        err = audit.close(value, raw_rules["D"][key][:, yi], "independent." + key,
                                          f"rep{rep}/{name}/{yi}", QUAD_CONFIG["comparison_absolute_tolerance"])
                        info["errors"][key] = err
                        info["errors_by_K"][key] = [float(np.max(abs(value[k] - raw_rules["D"][key][k, yi]))) for k in range(3)]
                    audit.close(math.log(info["evidence"]), raw_rules["D"]["log_evidence"][yi],
                                "independent.log_evidence", f"rep{rep}/{name}/{yi}", 2e-11)
                    sampled.append(info)
            print(json.dumps(dict(audited_rep=rep, anchor=name, qualified=sum(r["qualified"] for r in rows))), flush=True)
        rep_rows = [cell for rec in records if rec["rep"] == rep for cell in rec["target_K"]]
        qualified_count = sum(r["qualified"] for r in rep_rows)
        audit.exact(result["qualified_primary_matrices"], qualified_count, f"rep{rep}:qualified_total")
        audit.exact(result["unresolved_primary_matrices"], 42-qualified_count, f"rep{rep}:unresolved_total")
        audit.exact(result["all_primary_matrices_qualified"], qualified_count == 42, f"rep{rep}:all_qualified")
        audit.exact(result["primary_matrices"], 42, f"rep{rep}:primary_count")
        audit.exact(result["all_rule_matrices"], 168, f"rep{rep}:all_rule_count")
        for field in ("new_random_draws", "new_observations", "optimizer_updates", "statistical_alpha_spent", "failed_kernel_calls"):
            audit.exact(result["resources"][field], 0, f"rep{rep}:resources_{field}")
        audit.exact(result["resources"]["successful_kernel_calls"], 28, f"rep{rep}:kernel_calls")
    audit.exact(raw_count, 112, "all_raw_files_checked")
    directional = check_directional(audit)
    for path, digest in list(audit.inputs.items()):
        audit.bind(ROOT / path, digest)
    return dict(passed=not audit.failures, source_sha256=sha(__file__), freeze_sha256=freeze_hash,
        scope=dict(role="independent numerical and execution verification, not scientific peer review",
            all_saved_raw_files=raw_count, anchor_count=len(records), all_rule_matrices=672, primary_matrices=168,
            per_label_K_cells=raw_count*3*256, new_random_draws=0, new_scientific_anchors=0,
            independent_selected_anchor_label_cells=len(sampled), independent_finite_integral_K_cells=len(sampled)*2,
            independent_posterior_label_cells=len(sampled), full_inner_outer_replay=False,
            saved_per_label_reweighting_not_full_independent_integration=True,
            rigorous_inner_or_outer_remainder_certificate=False, coarse_rule_failures_preserved=True),
        configuration=QUAD_CONFIG, assertions=audit.assertions, numeric_elements_compared=audit.numeric_elements,
        failures=audit.failures, numerical_maxima=audit.maxima, input_hashes=audit.inputs,
        reconstructed_classifications=records, coarse_rule_qualification_failures=coarse_failures,
        independent_integral_checks=sampled, directional_checks=directional, seconds=time.perf_counter()-tick,
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
            threads={n: os.environ[n] for n in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.run and RECEIPT.exists():
        raise FileExistsError("AUDIT.json exists; use --check for a read-only repeat")
    if args.check:
        previous = read(RECEIPT)
        if previous["source_sha256"] != sha(__file__):
            raise ValueError("audit source changed after receipt; do not relabel the saved audit")
    try:
        outcome = perform()
    except BaseException as error:
        outcome = dict(passed=False, source_sha256=sha(__file__), exception=str(error), traceback=traceback.format_exc())
    if args.run:
        outcome["created_utc"] = datetime.now(timezone.utc).isoformat()
        with RECEIPT.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(native(outcome), handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
    elif outcome["passed"]:
        if outcome["input_hashes"] != previous["input_hashes"]:
            raise ValueError("audit inputs differ from the saved receipt")
        if outcome["reconstructed_classifications"] != previous["reconstructed_classifications"]:
            raise ValueError("repeated classifications differ")
    print(json.dumps(native({key: outcome[key] for key in ("passed", "seconds", "failures", "exception") if key in outcome})), flush=True)
    return 0 if outcome["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
