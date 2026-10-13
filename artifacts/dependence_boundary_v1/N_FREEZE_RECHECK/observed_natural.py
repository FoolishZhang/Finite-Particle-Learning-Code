"""Frozen same-budget iid/antithetic covariance training; I/O only, no C/T."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import itertools
import json
import os
import subprocess
import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F
from script.dependence_boundary_spectrum import selected_means
from script.shared_uncertainty_holdout_train import source_loss, factor, SRC

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/dependence_boundary_v1/N"
OLD = ROOT / "artifacts/feedback_uncertainty_v3"
SPECTRUM = ROOT / "artifacts/dependence_boundary_v1/SPECTRUM"
PLAN = ROOT / "docs/08_research_研究工作/边际依赖接口_20260929/PLAN_N.md"
SEEDS = list(range(20266001, 20266007))
CONFIG = dict(
    roles=["I", "O"], K=[4, 100], samplers=["iid", "antithetic"],
    initial_sd=[.01, 1.], seeds=SEEDS, d=24, batch=128,
    phase_steps=[1500, 500], phase_rng_offsets=[2000, 3000],
    learning_rate_first_1000=.002, learning_rate_remaining=.0005,
    optimizer="torch.optim.Adam defaults; reset between phases",
    precision_regularization=None, CPU_threads=2, dtype="float32",
    repeats=6, configuration_count=16, terminal_models=96,
    fit_phases=192, new_means=0, new_task_heads=0,
    checkpoint_steps=250, factor_diag_floor=1e-6,
    noise_contract="Generate full iid (128,K,24), then anti concatenates first half and negatives; consume final unused integer",
    data_contract="I/O only; no C/T prediction or label arithmetic",
    float_boundary="Count source float32 sigmoid 0/1; real-valued spectrum is not a float32 gradient certificate",
)
SOURCES = [
    Path(__file__), ROOT / "script/dependence_boundary_spectrum.py",
    ROOT / "script/dependence_v4_covariance.py",
    ROOT / "script/shared_uncertainty_holdout_train.py",
    ROOT / "script/shared_uncertainty_core.py",
    SRC / "models/losses.py", PLAN,
]
INPUTS = [
    OLD / "cache/cub_means.npz", OLD / "D/IMAGE_ROLES.json",
    OLD / "D/LABEL_DEFINITION.json", OLD / "R/DATA_MATCH.json",
    OLD / "N/MEANS_FREEZE.json",
    SPECTRUM / "FREEZE.json", SPECTRUM / "COMPLETE.json",
    SPECTRUM / "SUMMARY.csv", SPECTRUM / "DERIVATIVES.csv",
]

def now():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

def write(path, obj, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

def csvwrite(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def cells():
    for K, role, sd, sampler in itertools.product(
        CONFIG["K"], CONFIG["roles"], CONFIG["initial_sd"], CONFIG["samplers"]
    ):
        yield dict(K=K, role=role, initial_sd=sd, sampler=sampler)

def cell_id(cell):
    sd = "001" if cell["initial_sd"] == .01 else "1"
    return f'{cell["role"]}_K{cell["K"]}_{cell["sampler"]}_sd{sd}'

def freeze():
    assert not OUT.exists(), "Refuse to overwrite N package"
    write(OUT / "CONFIG.json", CONFIG)
    write(OUT / "FREEZE.json", dict(
        time_utc=now(), git_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        files={p.relative_to(ROOT).as_posix(): sha(p)
               for p in SOURCES + INPUTS + [OUT / "CONFIG.json"]},
        scope="96 terminal covariance models, no C/T evaluation"))

def check_freeze():
    assert read(OUT / "CONFIG.json") == CONFIG
    for name, expected in read(OUT / "FREEZE.json")["files"].items():
        assert sha(ROOT / name) == expected, name

def data():
    rows = read(OLD / "D/IMAGE_ROLES.json")
    prototypes = np.asarray(read(OLD / "D/LABEL_DEFINITION.json")["class_labels"])
    indices = dict(
        I=np.asarray(read(OLD / "R/DATA_MATCH.json")["indices"], dtype=int),
        O=np.asarray([i for i, r in enumerate(rows) if r["role"] == "model_validation"], dtype=int))
    assert all(len(ix) == 915 for ix in indices.values())
    assert set(indices["I"]).isdisjoint(indices["O"])
    assert all(rows[i]["role"] == "train" for i in indices["I"])
    selected = np.unique(np.concatenate(list(indices.values())))
    position = {int(index): k for k, index in enumerate(selected)}
    mu, archive_shape = selected_means(OLD / "cache/cub_means.npz", selected)
    assert mu.shape == (6, 1830, 24)
    labels = {role: prototypes[[rows[i]["class_id"] - 1 for i in ix]].astype(np.float32)
              for role, ix in indices.items()}
    assert np.array_equal(labels["I"], labels["O"])
    result, saturated = {}, []
    for role, ix in indices.items():
        local = np.asarray([position[int(i)] for i in ix])
        m = torch.tensor(mu[:, local], dtype=torch.float32)
        result[role] = (m, torch.tensor(labels[role], dtype=torch.float32))
        pp = torch.sigmoid(m)
        for rep in range(6):
            saturated.append(dict(role=role, rep=rep,
                sigmoid_zero=int((pp[rep] == 0).sum()),
                sigmoid_one=int((pp[rep] == 1).sum()), values=915 * 24))
    receipt = dict(
        time_utc=now(), decoded_means_rows=1830, archive_shape=archive_shape,
        analyzed_roles=["I", "O"], role_indices={r: v.tolist() for r, v in indices.items()},
        C_T_predictions_decoded=False, C_T_labels_used=False,
        labels_same_sequence=True, row_count=915,
        original_mean_float32_saturation=saturated)
    path = OUT / "DATA_RECEIPT.json"
    if path.exists():
        previous = read(path)
        assert {k: v for k, v in previous.items() if k != "time_utc"} == {
            k: v for k, v in receipt.items() if k != "time_utc"}
    else:
        write(path, receipt)
    return result

def spectrum_rows(cell):
    result = []
    with (SPECTRUM / "SUMMARY.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["kind"] == "old" and row["role"] == cell["role"] and float(row["scale"]) == 1.:
                lam = float(row["lambda_max"])
                result.append(dict(
                    rep=int(row["rep"]), lambda_max=lam, Kcrit=float(row["Kcrit"]),
                    iid_PSD_descent_exists=lam > cell["K"] / (cell["K"] - 1),
                    antithetic_PSD_descent_exists=lam > 1,
                    interpretation="empirical expected-MC at zero; not float32 or optimization certificate"))
    result.sort(key=lambda x: x["rep"])
    assert [row["rep"] for row in result] == list(range(6))
    return result

def snapshot(raw):
    raw_np = raw.detach().cpu().numpy().astype(float)
    L = factor(raw.detach()).cpu().numpy().astype(float)
    cov = L @ L.transpose(0, 2, 1)
    return dict(raw=raw_np.tolist(), factors=L.tolist(), covariance=cov.tolist())

def train_cell(cell, arrays):
    check_freeze()
    ident = cell_id(cell)
    path = OUT / "models" / f"{ident}.json"
    assert not path.exists(), ident
    assert not (OUT / "initial" / f"{ident}.json").exists(), f"Partial configuration exists: {ident}"
    K = cell["K"]
    mu, labels = arrays[cell["role"]]
    raw = torch.zeros(6, 24, 24, dtype=torch.float32)
    raw[:, torch.arange(24), torch.arange(24)] = float(
        np.log(np.expm1(cell["initial_sd"] - 1e-6)))
    raw = torch.nn.Parameter(raw)
    write(OUT / "initial" / f"{ident}.json", dict(
        time_utc=now(), id=ident, config=cell, seeds=SEEDS, **snapshot(raw)))
    lossfun = source_loss()
    reps = torch.arange(6)[:, None].expand(6, 128)
    trace, rng_final = [], []
    start = time.perf_counter()
    for phase, length in enumerate(CONFIG["phase_steps"]):
        opt = torch.optim.Adam([raw], lr=.002 if phase == 0 else .0005)
        rng = [np.random.default_rng(seed + CONFIG["phase_rng_offsets"][phase]) for seed in SEEDS]
        for step in range(1, length + 1):
            indices, noise = [], []
            for r in rng:
                indices.append(r.integers(0, 915, 128))
                full_noise = r.normal(size=(128, K, 24))
                r.integers(0, 24)
                if cell["sampler"] == "antithetic":
                    z = full_noise[:, :K // 2]
                    full_noise = np.concatenate((z, -z), axis=1)
                noise.append(full_noise)
            index = torch.tensor(np.asarray(indices))
            z = torch.tensor(np.asarray(noise), dtype=torch.float32)
            m, y = mu[reps, index], labels[index]
            L = factor(raw)
            eta = m[:, :, None] + torch.einsum("sbkj,sij->sbki", z, L)
            probs = torch.sigmoid(eta).permute(0, 1, 3, 2)
            vals = lossfun(probs.reshape(-1, 24, K), y.reshape(-1, 24),
                           torch.zeros(768, 1), torch.zeros(768), L)
            value = vals[-1] * 6
            opt.zero_grad()
            value.backward()
            assert torch.isfinite(value) and torch.isfinite(raw.grad).all(), (ident, phase, step)
            opt.param_groups[0]["lr"] = .002 if phase == 0 and step <= 1000 else .0005
            opt.step()
            assert torch.isfinite(raw).all(), (ident, phase, step, "updated raw")
            if step % 250 == 0:
                with torch.no_grad():
                    bce = F.binary_cross_entropy(probs, y[..., None].expand_as(probs), reduction="none")
                    losses = -torch.logsumexp(-bce.sum(2), dim=2).mean(1) + np.log(K)
                    final_L = factor(raw.detach())
                    variance_trace = (final_L * final_L).sum((1, 2))
                    trace.append(dict(
                        phase=phase, step=step, total_step=phase * 1500 + step,
                        pre_update_source_losses=[float(v) for v in vals],
                        pre_update_concept_NLL=losses.tolist(),
                        pre_update_sigmoid_zero=(probs == 0).sum((1, 2, 3)).tolist(),
                        pre_update_sigmoid_one=(probs == 1).sum((1, 2, 3)).tolist(),
                        probability_values_per_rep=128 * 24 * K,
                        post_update_tracevariance=variance_trace.tolist(),
                        learning_rate=opt.param_groups[0]["lr"]))
                print(json.dumps(dict(
                    id=ident, phase=phase, step=step,
                    seconds=round(time.perf_counter() - start, 1),
                    mean_concept_NLL=float(losses.mean()),
                    mean_tracevariance=float(variance_trace.mean()))), flush=True)
            del eta, probs, vals, value
        phase_rng = [digest(r.bit_generator.state) for r in rng]
        rng_final.append(phase_rng)
        write(OUT / "checkpoints" / f"{ident}_phase{phase}.json", dict(
            time_utc=now(), id=ident, config=cell, phase=phase,
            steps=length, rng_final_hashes=phase_rng, **snapshot(raw)))
    write(path, dict(
        time_utc=now(), id=ident, config=cell, seeds=SEEDS,
        phases=CONFIG["phase_steps"], total_steps=2000,
        elapsed_seconds=time.perf_counter() - start,
        spectrum=spectrum_rows(cell), trace=trace,
        rng_final_hashes=rng_final, **snapshot(raw),
        finite_loss_gradient_raw_every_step=True))

def train_all():
    check_freeze()
    torch.set_num_threads(2)
    arrays = data()
    for cell in cells():
        ident = cell_id(cell)
        if (OUT / "models" / f"{ident}.json").exists():
            continue
        write(OUT / "RUN_STATE.json", dict(
            status="running", pid=os.getpid(), current_configuration=ident, time_utc=now()),
            exclusive=False)
        train_cell(cell, arrays)
    write(OUT / "RUN_STATE.json", dict(
        status="training_finished_unsealed", pid=os.getpid(), time_utc=now()), exclusive=False)

def audit():
    check_freeze()
    detail, trace_rows, eigen_rows = [], [], []
    for cell in cells():
        ident = cell_id(cell)
        model = read(OUT / "models" / f"{ident}.json")
        initial = read(OUT / "initial" / f"{ident}.json")
        assert model["config"] == initial["config"] == cell
        assert model["total_steps"] == 2000 and len(model["trace"]) == 8
        assert model["finite_loss_gradient_raw_every_step"]
        paired = dict(cell, sampler="antithetic" if cell["sampler"] == "iid" else "iid")
        other = read(OUT / "models" / f"{cell_id(paired)}.json")
        assert model["rng_final_hashes"] == other["rng_final_hashes"]
        raw, saved_factor = np.asarray(model["raw"]), np.asarray(model["factors"])
        factors = np.tril(raw, -1)
        factors[:, np.arange(24), np.arange(24)] = np.logaddexp(
            0, np.diagonal(raw, axis1=1, axis2=2)) + 1e-6
        assert raw.shape == saved_factor.shape == (6, 24, 24)
        assert np.isfinite(raw).all() and np.isfinite(saved_factor).all()
        assert np.max(np.abs(factors - saved_factor)) < 5e-7
        cov = saved_factor @ saved_factor.transpose(0, 2, 1)
        np.testing.assert_allclose(cov, model["covariance"], rtol=1e-13, atol=1e-13)
        initial_L, initial_cov = np.asarray(initial["factors"]), np.asarray(initial["covariance"])
        np.testing.assert_allclose(initial_cov,
            np.broadcast_to(np.eye(24) * cell["initial_sd"] ** 2, (6, 24, 24)),
            rtol=2e-6, atol=1e-12)
        assert np.max(np.abs(np.tril(initial_L, -1))) == 0
        for rep in range(6):
            eigen = np.linalg.eigvalsh(cov[rep])
            assert np.isfinite(eigen).all()
            assert np.all(np.diag(saved_factor[rep]) > 0)
            assert eigen[0] >= -1e-10 * max(1., eigen[-1])
            diag = np.diag(cov[rep])
            detail.append(dict(
                id=ident, **cell, rep=rep, seed=SEEDS[rep],
                initial_tracevariance=float(np.trace(initial_cov[rep])),
                final_tracevariance=float(np.trace(cov[rep])),
                diag_min=float(diag.min()), diag_max=float(diag.max()),
                covariance_frobenius=float(np.linalg.norm(cov[rep])),
                offdiagonal_frobenius=float(np.linalg.norm(cov[rep] - np.diag(diag))),
                eigenvalue_min=float(eigen.min()), eigenvalue_max=float(eigen.max()),
                spectral_condition=float(eigen.max() / eigen.min()) if eigen.min() > 0 else None,
                factor_numpy_max_error=float(np.max(np.abs(factors[rep] - saved_factor[rep]))),
                raw_absmax=float(np.max(np.abs(raw[rep]))), finite=True,
                positive_definite_from_factor=True, numerical_min_eigenvalue_positive=bool(eigen.min() > 0),
                iid_PSD_descent_exists=model["spectrum"][rep]["iid_PSD_descent_exists"],
                antithetic_PSD_descent_exists=model["spectrum"][rep]["antithetic_PSD_descent_exists"],
                lambda_max=model["spectrum"][rep]["lambda_max"]))
            eigen_rows.append(dict(id=ident, rep=rep, **{
                f"eigenvalue_{j}": float(v) for j, v in enumerate(eigen)}))
            for record in model["trace"]:
                trace_rows.append(dict(
                    id=ident, **cell, rep=rep, seed=SEEDS[rep],
                    phase=record["phase"], step=record["step"], total_step=record["total_step"],
                    pre_update_concept_NLL=record["pre_update_concept_NLL"][rep],
                    post_update_tracevariance=record["post_update_tracevariance"][rep],
                    sigmoid_zero=record["pre_update_sigmoid_zero"][rep],
                    sigmoid_one=record["pre_update_sigmoid_one"][rep],
                    probability_values=record["probability_values_per_rep"]))
        for phase in range(2):
            cp = read(OUT / "checkpoints" / f"{ident}_phase{phase}.json")
            assert cp["phase"] == phase and cp["steps"] == CONFIG["phase_steps"][phase]
            assert cp["rng_final_hashes"] == model["rng_final_hashes"][phase]
        final_cp = read(OUT / "checkpoints" / f"{ident}_phase1.json")
        assert final_cp["raw"] == model["raw"] and final_cp["factors"] == model["factors"]
    assert len(detail) == 96 and len(trace_rows) == 768
    csvwrite(OUT / "MODEL_AUDIT.csv", detail)
    csvwrite(OUT / "EIGENSPECTRA.csv", eigen_rows)
    csvwrite(OUT / "TRAJECTORIES.csv", trace_rows)
    write(OUT / "AUDIT.json", dict(
        time_utc=now(), passed=True, models=96, configurations=16,
        matched_iid_antithetic_RNG_states=True, all_finite=True,
        all_covariances_positive_definite_from_factor=True,
        numerical_min_eigenvalue_nonpositive_count=sum(not r["numerical_min_eigenvalue_positive"] for r in detail),
        initializations_checked=True,
        source_factor_numpy_max_error=max(r["factor_numpy_max_error"] for r in detail),
        C_T_labels_or_predictions_analyzed=False))

def seal():
    check_freeze()
    assert read(OUT / "AUDIT.json")["passed"]
    paths = sorted((OUT / "models").glob("*.json"))
    assert len(paths) == 16
    write(OUT / "MODELS_FREEZE.json", dict(
        time_utc=now(), files={p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
        mean_models=6, new_mean_models=0, terminal_covariance_models=96,
        covariance_fit_phases=192, C_T_evaluation_started=False,
        model_schema=dict(
            raw="[6,24,24] float32 values serialized as native floats",
            factors="[6,24,24] source softplus triangular factors",
            covariance="[6,24,24] float64 reconstruction factors @ factors.T",
            config="role, K, sampler, initial_sd",
            ordering="rep=0..5 corresponds SEEDS 20266001..20266006")))
    write(OUT / "RUN_STATE.json", dict(
        status="complete_models_sealed", time_utc=now(), active_jobs=[]), exclusive=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["freeze", "train", "audit", "seal", "run"])
    args = parser.parse_args()
    try:
        if args.phase == "run":
            train_all()
            audit()
            seal()
        else:
            {"freeze": freeze, "train": train_all, "audit": audit, "seal": seal}[args.phase]()
    except Exception:
        if OUT.exists():
            write(OUT / f"FAILURE_{time.time_ns()}.json", dict(
                time_utc=now(), phase=args.phase, traceback=traceback.format_exc()))
        raise
