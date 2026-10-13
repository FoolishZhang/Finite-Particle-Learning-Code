"""Frozen empirical zero-covariance spectrum; only I/O rows enter calculations.

For the expected finite-K negative log Monte Carlo likelihood, the right
derivative at covariance t*S, t=0, is
    .5 * trace(S @ (D - (1 - 1/K)*M)).
This is an empirical fixed-data objective diagnostic, not a training optimum
certificate or an inferential statement about a natural population.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import zipfile

import numpy as np
from scipy.special import expit


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/dependence_boundary_v1/SPECTRUM"
OLD = ROOT / "artifacts/feedback_uncertainty_v3"
INPUTS = [
    OLD / "cache/cub_means.npz",
    ROOT / "artifacts/dependence_controls_v4/G0/means.npz",
    OLD / "D/IMAGE_ROLES.json",
    OLD / "D/LABEL_DEFINITION.json",
    OLD / "R/DATA_MATCH.json",
]
CONFIG = dict(
    scales=[.5, 1., 2.], K=[1, 2, 4, 16, 100], roles=["I", "O"],
    old_mean_replicates=6, linear_mean_replicates=1,
    expected_combinations=42, expected_derivative_rows=210,
    matrix_absolute_tolerance=1e-11, relative_tolerance=1e-11,
    derivative_absolute_tolerance=1e-11,
)
PLAN = {
    "question": "Which fixed empirical mean/data/scale combinations have a PSD covariance descent direction at zero, and how does finite K alter that boundary?",
    "scope": "All 6 old MLP means and the single existing linear mean; two 915-row I/O roles; scales .5,1,2; no model fitting or selection.",
    "data_contract": "Only I/O row indices and train-defined class labels enter arithmetic. No C/T labels, predictions or outcomes are analyzed. Compressed mean archives contain all roles; unselected NPY bytes are skipped without materializing their values.",
    "matrices": "p=sigmoid(scale*mu), D=diag(mean(p*(1-p))), M=mean((Y-p)(Y-p)^T), R=D^(-1/2) M D^(-1/2).",
    "direction": "Top unit eigenvector v of R; S=D^(-1/2) v v^T D^(-1/2)/trace(D^(-1/2) v v^T D^(-1/2)). This maximizes the generalized ratio, not necessarily the magnitude of trace-normalized descent.",
    "finite_K": "Expected finite-K source likelihood covariance right derivative: .5 tr S[D-(1-1/K)M]. At lambda_max>1, Kcrit=lambda_max/(lambda_max-1), with negative direction for strict K>Kcrit; otherwise no finite-K instability in any PSD direction.",
    "audit": "Independent per-example outer-product loops reconstruct D/M/R; residual projection formula checks each derivative; pair-count algebra K(K-1)/K^2 checks finite-K coefficient; eigen residual, PSD, trace and spectral-sign identities checked.",
    "retention": "All 42 combinations and 210 K-specific derivatives retained. No threshold-based selection and no training.",
    "interpretation": "Empirical expected-MC objective local boundary only; not realized finite-draw path derivative, population inference, finite-t improvement, or global optimum certificate.",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def csvwrite(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def freeze():
    assert not OUT.exists(), "Refuse to overwrite an existing spectrum package"
    write(OUT / "FREEZE.json", dict(
        time_utc=now(), config=CONFIG, plan=PLAN,
        sources={Path(__file__).relative_to(ROOT).as_posix(): sha(__file__)},
        inputs={p.relative_to(ROOT).as_posix(): sha(p) for p in INPUTS},
    ))


def check_freeze():
    frozen = read(OUT / "FREEZE.json")
    assert frozen["config"] == CONFIG and frozen["plan"] == PLAN
    for collection in ("sources", "inputs"):
        for name, digest in frozen[collection].items():
            assert sha(ROOT / name) == digest, name


def selected_means(path, indices):
    """Decode selected rows only; skip intervening NPY bytes without decoding."""
    indices = np.asarray(indices, dtype=np.int64)
    assert np.all(np.diff(indices) > 0)
    with zipfile.ZipFile(path) as archive, archive.open("means.npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise AssertionError(version)
        assert not fortran and dtype == np.dtype("float32")
        assert shape[-2:] == (11788, 24) and len(shape) in (2, 3)
        reps = 1 if len(shape) == 2 else shape[0]
        result = np.empty((reps, len(indices), 24), dtype=np.float64)
        width = 24 * dtype.itemsize
        cursor = 0
        for rep in range(reps):
            for local, index in enumerate(indices):
                offset = (rep * 11788 + int(index)) * width
                skip = offset - cursor
                assert skip >= 0
                if skip:
                    assert len(stream.read(skip)) == skip
                raw = stream.read(width)
                assert len(raw) == width
                result[rep, local] = np.frombuffer(raw, dtype=dtype)
                cursor = offset + width
    return result, list(shape)


def arrays():
    rows = read(INPUTS[2])
    prototypes = np.asarray(read(INPUTS[3])["class_labels"], dtype=float)
    roles = {
        "I": np.asarray(read(INPUTS[4])["indices"], dtype=int),
        "O": np.asarray([i for i, r in enumerate(rows) if r["role"] == "model_validation"], dtype=int),
    }
    assert all(len(ix) == 915 for ix in roles.values())
    assert len(set(roles["I"]) & set(roles["O"])) == 0
    assert all(rows[i]["role"] == "train" for i in roles["I"])
    selected = np.unique(np.concatenate(list(roles.values())))
    positions = {int(index): local for local, index in enumerate(selected)}
    local = {role: np.asarray([positions[int(i)] for i in ix]) for role, ix in roles.items()}
    labels = {role: prototypes[[rows[i]["class_id"] - 1 for i in ix]] for role, ix in roles.items()}
    assert np.array_equal(labels["I"], labels["O"])
    old, old_shape = selected_means(INPUTS[0], selected)
    linear, linear_shape = selected_means(INPUTS[1], selected)
    assert old.shape == (6, 1830, 24) and linear.shape == (1, 1830, 24)
    write(OUT / "DATA_RECEIPT.json", dict(
        old_archive_shape=old_shape, linear_archive_shape=linear_shape,
        decoded_rows_per_mean=1830, analyzed_roles=["I", "O"],
        role_rows={role: ix.tolist() for role, ix in roles.items()},
        matched_label_sequences=True, C_T_prediction_values_decoded=False,
        C_T_labels_or_outcomes_analyzed=False,
    ))
    return {"old": old, "linear": linear}, local, labels


def diagnose(mu, labels):
    p = expit(mu)
    residual = labels - p
    diagonal = np.mean(p * (1 - p), axis=0)
    assert np.all(diagonal > 0)
    D = np.diag(diagonal)
    M = residual.T @ residual / len(labels)
    inverse_root = 1 / np.sqrt(diagonal)
    R = inverse_root[:, None] * M * inverse_root[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    v = eigenvectors[:, -1]
    if v[np.argmax(np.abs(v))] < 0:
        v = -v
    direction = inverse_root * v
    direction /= np.linalg.norm(direction)
    S = np.outer(direction, direction)
    lam = float(eigenvalues[-1])
    critical = lam / (lam - 1) if lam > 1 else None

    # Independently accumulate each example, without matrix multiplication.
    loop_diagonal = np.zeros(24)
    loop_M = np.zeros((24, 24))
    for prob, target in zip(p, labels):
        loop_diagonal += prob * (1 - prob)
        error = target - prob
        loop_M += np.outer(error, error)
    loop_diagonal /= len(labels)
    loop_M /= len(labels)
    loop_R = np.empty((24, 24))
    for i in range(24):
        for j in range(24):
            loop_R[i, j] = loop_M[i, j] / np.sqrt(loop_diagonal[i] * loop_diagonal[j])
    checks = dict(
        D_loop_max_abs=float(np.max(abs(loop_diagonal - diagonal))),
        M_loop_max_abs=float(np.max(abs(loop_M - M))),
        R_loop_max_abs=float(np.max(abs(loop_R - R))),
        eigen_residual_max_abs=float(np.max(abs(R @ v - lam * v))),
        S_trace_error=float(abs(np.trace(S) - 1)),
        S_min_eigenvalue=float(np.linalg.eigvalsh(S)[0]),
    )
    assert np.allclose(np.diag(loop_diagonal), D, atol=1e-11, rtol=1e-11)
    assert np.allclose(loop_M, M, atol=1e-11, rtol=1e-11)
    assert np.allclose(loop_R, R, atol=1e-11, rtol=1e-11)
    assert checks["eigen_residual_max_abs"] < 1e-10
    assert checks["S_trace_error"] < 1e-12 and checks["S_min_eigenvalue"] > -1e-12
    derivatives = []
    for K in CONFIG["K"]:
        coefficient = 1 - 1 / K
        derivative = float(.5 * np.trace(S @ (D - coefficient * M)))
        # Expand the K diagonal and K(K-1) ordered off-diagonal sample pairs.
        pair_coefficient = sum(1 for i in range(K) for j in range(K) if i != j) / K**2
        projected_variance = float(np.mean((residual @ direction)**2))
        projected_curvature = float(np.mean(np.sum(p * (1 - p) * direction**2, axis=1)))
        independent = .5 * (projected_curvature - pair_coefficient * projected_variance)
        spectral = .5 * projected_curvature * (1 - coefficient * lam)
        error = max(abs(derivative - independent), abs(derivative - spectral))
        assert error <= CONFIG["derivative_absolute_tolerance"]
        assert abs(coefficient - pair_coefficient) < 1e-15
        derivatives.append(dict(
            K=K, derivative=derivative, independent_projection_derivative=independent,
            spectral_derivative=spectral, absolute_audit_error=error,
            finite_K_max_generalized_eigenvalue=coefficient * lam,
            negative_top_direction=bool(derivative < 0),
            PSD_descent_exists_analytic=bool(coefficient * lam > 1),
        ))
        assert (derivative < 0) == (coefficient * lam > 1)
    checks["derivative_max_abs_error"] = max(d["absolute_audit_error"] for d in derivatives)
    return dict(
        D=D.tolist(), M=M.tolist(), R=R.tolist(),
        eigenvalues_ascending=eigenvalues.tolist(), top_eigenvector=v.tolist(),
        top_direction_unit_vector=direction.tolist(), S=S.tolist(),
        lambda_max=lam, Kcrit=critical,
        trace_M=float(np.trace(M)), trace_D=float(np.trace(D)),
        infinite_K_derivative=float(.5 * np.trace(S @ (D - M))),
        marginal_NLL=float(np.mean(np.logaddexp(0, mu) - labels * mu)),
        marginal_Brier=float(np.mean(residual**2)),
        derivatives=derivatives, audit=checks,
    )


def calculate():
    check_freeze()
    assert not (OUT / "COMPLETE.json").exists()
    means, local, labels = arrays()
    summaries, derivative_rows, audits = [], [], []
    for kind, predictions in means.items():
        for rep in range(len(predictions)):
            for role in CONFIG["roles"]:
                for scale in CONFIG["scales"]:
                    identity = dict(kind=kind, rep=rep, role=role, scale=scale, n=915)
                    result = diagnose(scale * predictions[rep, local[role]], labels[role])
                    name = f"{kind}_{rep}_{role}_scale{scale:g}"
                    write(OUT / "matrices" / (name + ".json"), dict(**identity, **result))
                    summaries.append(dict(**identity, **{key: result[key] for key in [
                        "lambda_max", "Kcrit", "trace_M", "trace_D", "infinite_K_derivative", "marginal_NLL", "marginal_Brier",
                    ]}))
                    derivative_rows.extend(dict(**identity, **d) for d in result["derivatives"])
                    audits.append(dict(**identity, **result["audit"]))
    assert len(summaries) == CONFIG["expected_combinations"]
    assert len(derivative_rows) == CONFIG["expected_derivative_rows"]
    csvwrite(OUT / "SUMMARY.csv", summaries)
    csvwrite(OUT / "DERIVATIVES.csv", derivative_rows)
    csvwrite(OUT / "AUDIT_DETAILS.csv", audits)
    write(OUT / "AUDIT.json", dict(
        status="pass_independent_loop_matrices_and_finite_K_algebra",
        combinations=len(summaries), derivatives=len(derivative_rows),
        max_errors={key: max(a[key] for a in audits) for key in [
            "D_loop_max_abs", "M_loop_max_abs", "R_loop_max_abs", "eigen_residual_max_abs", "S_trace_error", "derivative_max_abs_error",
        ]},
        new_fits=0, population_inference=False, global_optimality_certificate=False,
        finite_MC_formula_is_expectation_over_iid_normal_draws=True,
        no_C_T_prediction_values_decoded=True,
    ))
    check_freeze()
    files = sorted(p for p in OUT.rglob("*") if p.is_file())
    write(OUT / "COMPLETE.json", dict(time_utc=now(), status="complete_frozen_spectrum_diagnostic", files={p.relative_to(ROOT).as_posix(): sha(p) for p in files}))
    print(json.dumps(dict(combinations=len(summaries), derivatives=len(derivative_rows), audit=read(OUT / "AUDIT.json")), ensure_ascii=False))


def verify():
    check_freeze()
    for path, digest in read(OUT / "COMPLETE.json")["files"].items():
        assert sha(ROOT / path) == digest, path
    assert read(OUT / "AUDIT.json")["status"].startswith("pass_")
    print("pass_frozen_spectrum_hashes_and_completed_audit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["freeze", "calculate", "check"])
    args = parser.parse_args()
    {"freeze": freeze, "calculate": calculate, "check": verify}[args.phase]()
