"""Joint active and nuisance physical-scale optimization kernel.

The constructor knows neither a teacher covariance nor a direction of dependence.
It accepts an explicit initial mean; experiment sampling and initializations are
owned by a separately frozen runner.  The official32 path calls the untouched
SCBM API.  The math64 path is a stable mathematical control, not that API.
"""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

from dataclasses import dataclass
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

import scbminv_kernel as official_adapter

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve()
ADAPTER_SHA = "c6712abbb4e05ef0c25c688a05dad106034c41888040592424f50362f04129ad"
PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
METHODS = ("FIXED", "FREE", "ZERO")
PATHS = ("official32", "math64")
MEAN_BOUNDS = (-2., 2.)
SCALE_BOUNDS = (.001, 2.)
TAU_BOUNDS = (0., 2.)
EIGEN_FLOOR = .1
REPAIR_TARGET = .1001
torch.set_num_threads(1)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def basis(dtype):
    result = torch.zeros((6, 4, 4), dtype=dtype)
    for k, (i, j) in enumerate(PAIRS):
        result[k, i, j] = result[k, j, i] = 1.
    return result


def correlation_numpy(rho):
    rho = np.asarray(rho, dtype=np.float64)
    assert rho.shape == (6,) and np.isfinite(rho).all()
    result = np.eye(4)
    for value, (i, j) in zip(rho, PAIRS):
        result[i, j] = result[j, i] = value
    return result


def provenance():
    assert sha(official_adapter.__file__) == ADAPTER_SHA
    return dict(source_sha256=sha(SOURCE), adapter_sha256=ADAPTER_SHA,
        official=official_adapter.provenance(), methods=list(METHODS),
        paths=dict(official32="Unmodified official concept-loss API, float32",
                   math64="Stable-logit mathematical expression, float64; not official API"),
        physical_parameter_order=["mu[4]", "s[4]", "rho[6]", "tau[1]"],
        covariance="diag(s) C(rho) diag(s)", pair_order=[list(p) for p in PAIRS],
        mean_bounds=list(MEAN_BOUNDS), scale_bounds=list(SCALE_BOUNDS),
        tau_bounds=list(TAU_BOUNDS), eigen_floor=EIGEN_FLOOR,
        correlation_repair_target=REPAIR_TARGET,
        initial_correlation="Identity; caller cannot supply a teacher direction to constructor",
        repair="Box clipping plus radial shrinkage of C toward I; Adam moments retained",
        ZERO_scope="Active finite-K score plus n log 2; not exact likelihood or MLE",
        scope="Controlled concept-score optimization only; no head, precision penalty, full SCBM network or optimizer convergence guarantee")


@dataclass(frozen=True)
class Batch:
    active_eps: torch.Tensor
    nuisance_eps: torch.Tensor
    active_labels: torch.Tensor
    nuisance_labels: torch.Tensor


def prepare_batch(active_eps, nuisance_eps, active_labels, nuisance_labels, path="official32"):
    assert path in PATHS
    dtype = torch.float32 if path == "official32" else torch.float64
    e = torch.as_tensor(active_eps, dtype=dtype).contiguous()
    nu = torch.as_tensor(nuisance_eps, dtype=dtype).contiguous()
    y = torch.as_tensor(active_labels)
    yn = torch.as_tensor(nuisance_labels)
    assert e.ndim == nu.ndim == 3 and e.shape[2] == 4 and e.shape[:2] == nu.shape[:2]
    assert y.shape == (e.shape[1], 4) and yn.shape == (e.shape[1], nu.shape[2])
    assert bool(torch.isfinite(e).all()) and bool(torch.isfinite(nu).all())
    assert bool(((y == 0) | (y == 1)).all()) and bool(((yn == 0) | (yn == 1)).all())
    return Batch(e, nu, y.to(torch.int64).contiguous(), yn.to(torch.int64).contiguous())


class ScoreProgram(torch.nn.Module):
    def __init__(self, method, mu0, scale0=1., K=100, n=256, tau0=1., path="official32"):
        super().__init__()
        assert sha(official_adapter.__file__) == ADAPTER_SHA
        assert method in METHODS and path in PATHS and K >= 1 and n >= 1
        mu0 = np.asarray(mu0, dtype=float)
        scales = np.broadcast_to(np.asarray(scale0, dtype=float), (4,)).copy()
        assert mu0.shape == (4,) and np.isfinite(mu0).all() and np.isfinite(scales).all()
        assert np.all((MEAN_BOUNDS[0] <= mu0) & (mu0 <= MEAN_BOUNDS[1]))
        assert np.all((SCALE_BOUNDS[0] <= scales) & (scales <= SCALE_BOUNDS[1]))
        assert math.isfinite(tau0) and TAU_BOUNDS[0] <= tau0 <= TAU_BOUNDS[1]
        self.method, self.K, self.n, self.path = method, int(K), int(n), path
        self.dtype = torch.float32 if path == "official32" else torch.float64
        self.mu = torch.nn.Parameter(torch.tensor(mu0, dtype=self.dtype))
        self.s = torch.nn.Parameter(torch.tensor(scales, dtype=self.dtype))
        self.rho = torch.nn.Parameter(torch.zeros(6, dtype=self.dtype))
        self.register_buffer("identity", torch.eye(4, dtype=self.dtype))
        self.register_buffer("symmetric_basis", basis(self.dtype))
        if method == "FREE":
            self.tau = torch.nn.Parameter(torch.tensor(tau0, dtype=self.dtype))
        else:
            self.register_parameter("tau", None)
        self.register_buffer("fixed_tau", torch.tensor(0. if method == "ZERO" else tau0, dtype=self.dtype))

    def correlation(self):
        return self.identity + torch.einsum("p,pij->ij", self.rho, self.symmetric_basis)

    def covariance(self):
        return self.s[:, None] * self.correlation() * self.s[None, :]

    def nuisance_scale(self):
        return self.tau if self.tau is not None else self.fixed_tau

    def physical_state(self):
        return dict(mu=self.mu.detach().numpy().tolist(), s=self.s.detach().numpy().tolist(),
                    rho=self.rho.detach().numpy().tolist(), tau=float(self.nuisance_scale().detach()))

    def physical_vector(self):
        state = self.physical_state()
        return np.asarray(state["mu"] + state["s"] + state["rho"] + [state["tau"]], dtype=np.float64)

    @torch.no_grad()
    def load_physical_state(self, state):
        """Explicit diagnostic/checkpoint restoration; never repairs silently."""
        for name, count in (("mu", 4), ("s", 4), ("rho", 6)):
            values = np.asarray(state[name], dtype=float)
            assert values.shape == (count,) and np.isfinite(values).all()
            getattr(self, name).copy_(torch.as_tensor(values, dtype=self.dtype))
        value = float(state["tau"])
        assert math.isfinite(value)
        if self.tau is not None:
            self.tau.copy_(torch.tensor(value, dtype=self.dtype))
        else:
            assert abs(value - float(self.fixed_tau)) <= 1e-7
        self.assert_feasible()

    def assert_feasible(self):
        assert all(bool(torch.isfinite(p).all()) for p in self.parameters())
        # Float32 can represent .001 just below its Python decimal value.
        assert bool((self.mu >= self.mu.new_tensor(MEAN_BOUNDS[0])).all())
        assert bool((self.mu <= self.mu.new_tensor(MEAN_BOUNDS[1])).all())
        assert bool((self.s >= self.s.new_tensor(SCALE_BOUNDS[0])).all())
        assert bool((self.s <= self.s.new_tensor(SCALE_BOUNDS[1])).all())
        assert TAU_BOUNDS[0] <= float(self.nuisance_scale()) <= TAU_BOUNDS[1]
        eigmin = float(np.linalg.eigvalsh(correlation_numpy(self.rho.detach().numpy())).min())
        assert eigmin >= EIGEN_FLOOR, eigmin
        _, info = torch.linalg.cholesky_ex(self.correlation())
        assert int(info) == 0, "Post-repair Cholesky failed; no retry input"

    def forward(self, batch, diagnostics=None):
        assert batch.active_eps.shape[0] == self.K and batch.nuisance_eps.shape[2] == self.n
        assert batch.active_eps.dtype == batch.nuisance_eps.dtype == self.dtype
        lower = torch.linalg.cholesky(self.correlation())
        normal = torch.matmul(lower, batch.active_eps.unsqueeze(-1)).squeeze(-1)
        active = self.mu + self.s * normal
        if self.method == "ZERO":
            logits, labels = active, batch.active_labels
        else:
            logits = torch.cat((active, batch.nuisance_eps * self.nuisance_scale()), dim=-1)
            labels = torch.cat((batch.active_labels, batch.nuisance_labels), dim=1)
        by_concept = logits.movedim(0, -1)
        probabilities = None
        if self.path == "official32":
            probabilities = torch.sigmoid(by_concept)
            loss = official_adapter.official_loss(self.K).compute_concept_loss(probabilities, labels)
        else:
            targets = labels.to(self.dtype).unsqueeze(-1).expand_as(by_concept)
            log_likelihood = -F.binary_cross_entropy_with_logits(by_concept, targets, reduction="none").sum(dim=1)
            loss = -torch.logsumexp(log_likelihood, dim=1).mean() + math.log(self.K)
        if self.method == "ZERO":
            loss = loss + self.n * math.log(2.)
        if diagnostics is not None:
            diagnostics.update(path=self.path, official_API_calls=int(self.path == "official32"),
                logit_evaluations=logits.numel(), maximum_absolute_logit=float(logits.detach().abs().max()))
            if probabilities is not None:
                p = probabilities.detach()
                diagnostics.update(probability_evaluations=p.numel(), sigmoid_zero_count=int((p == 0).sum()),
                    sigmoid_one_count=int((p == 1).sum()), probability_min=float(p.min()), probability_max=float(p.max()))
        return loss

    @torch.no_grad()
    def restore_feasibility(self):
        proposal = self.physical_state()
        assert all(np.isfinite(proposal[k]).all() for k in ("mu", "s", "rho", "tau"))
        self.mu.clamp_(*MEAN_BOUNDS)
        self.s.clamp_(*SCALE_BOUNDS)
        if self.tau is not None:
            self.tau.clamp_(*TAU_BOUNDS)
        proposed_rho = np.asarray(proposal["rho"], dtype=np.float64)
        before = float(np.linalg.eigvalsh(correlation_numpy(proposed_rho)).min())
        alpha = 1.
        if before < EIGEN_FLOOR:
            alpha = (1. - REPAIR_TARGET) / (1. - before)
            assert 0. < alpha < 1.
            self.rho.copy_(torch.as_tensor(alpha * proposed_rho, dtype=self.dtype))
        self.assert_feasible()
        after = self.physical_state()
        return dict(proposal=proposal, projected={k:bool(np.any(np.asarray(proposal[k]) != np.asarray(after[k])))
            for k in ("mu", "s", "rho", "tau")}, projection_alpha=alpha,
            proposal_eigmin=before, eigmin=float(np.linalg.eigvalsh(correlation_numpy(after["rho"])).min()))


def make_optimizer(program, lr=.02):
    assert math.isfinite(lr) and lr > 0
    return torch.optim.Adam(program.parameters(), lr=lr, betas=(.9, .999), eps=1e-8,
        weight_decay=0., amsgrad=False, foreach=False, fused=False)


def make_programs(mu0, scale0=1., K=100, n=256, tau0=1., path="official32", lr=.02):
    programs = {m: ScoreProgram(m, mu0, scale0, K, n, tau0, path) for m in METHODS}
    return programs, {m:make_optimizer(p, lr) for m,p in programs.items()}


def gradients(program, batch):
    loss = program(batch)
    named = tuple(program.named_parameters())
    values = torch.autograd.grad(loss, tuple(p for _,p in named))
    return loss, {name:value for (name,_),value in zip(named, values)}


def optimizer_step(program, optimizer, batch, diagnostics=False):
    optimizer.zero_grad(set_to_none=True)
    before = program.physical_state()
    detail = {} if diagnostics else None
    loss = program(batch, detail)
    assert bool(torch.isfinite(loss)), "Nonfinite loss; no scientific input retry"
    loss.backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in program.parameters())
    gradient = {name:p.grad.detach().numpy().tolist() for name,p in program.named_parameters()}
    optimizer.step()
    repair = program.restore_feasibility()
    return dict(loss=float(loss.detach()), before=before, gradient=gradient,
                after=program.physical_state(), repair=repair, diagnostics=detail)


def clone_to_path(program, path):
    state = program.physical_state()
    result = ScoreProgram(program.method, state["mu"], state["s"], program.K, program.n,
                          state["tau"], path)
    result.load_physical_state(state)
    return result


def deterministic_batch(K=100, B=7, n=256, path="official32"):
    arrays = official_adapter.deterministic_inputs(K, B, n)
    return prepare_batch(*arrays, path=path), arrays


def finite_difference(program, batch, name, index=None, step=1e-3):
    parameter = getattr(program, name)
    original = parameter.detach().clone()
    values = []
    try:
        for multiple in (-2., -1., 1., 2.):
            with torch.no_grad():
                parameter.copy_(original)
                if index is None:
                    parameter.add_(multiple * step)
                else:
                    parameter[index] += multiple * step
                values.append(float(program(batch).detach()))
    finally:
        with torch.no_grad():
            parameter.copy_(original)
    return (values[0] - 8 * values[1] + 8 * values[2] - values[3]) / (12 * step)


def check():
    """Deterministic qualification only; returns data, never writes artifacts."""
    rows = []
    for path in PATHS:
        batch, arrays = deterministic_batch(n=5, path=path)
        for method in METHODS:
            program = ScoreProgram(method, [-.35, -.1, .1, .35], [.4, .7, .9, 1.1], n=5, path=path)
            state = program.physical_state(); state["rho"] = [.11, -.07, .05, .09, -.06, .13]
            program.load_physical_state(state)
            loss, grad = gradients(program, batch)
            step = .02 if path == "official32" else 2e-4
            tolerance = 4e-4 if path == "official32" else 4e-8
            errors = []
            for name, parameter in program.named_parameters():
                for index in range(parameter.numel()):
                    scalar = parameter.ndim == 0
                    numeric = finite_difference(program, batch, name, None if scalar else index, step)
                    analytic = float(grad[name] if scalar else grad[name][index])
                    errors.append(abs(numeric - analytic))
            assert max(errors) < tolerance, (path, method, max(errors))
            control = clone_to_path(program, "math64")
            control_batch = prepare_batch(*arrays, path="math64")
            closs, cg = gradients(control, control_batch)
            value_error = abs(float(loss.detach()) - float(closs.detach()))
            gradient_error = max(float(np.max(abs(grad[k].detach().numpy() - cg[k].detach().numpy()))) for k in grad)
            assert value_error < 1e-4 and gradient_error < 4e-5
            rows.append(dict(path=path, method=method, coordinates=len(errors), finite_difference_max_error=max(errors),
                finite_difference_step=step, tolerance=tolerance, vs_math64_value_error=value_error,
                vs_math64_gradient_max_error=gradient_error))
    identities=[]
    for path in PATHS:
        batch, _ = deterministic_batch(path=path)
        fixed = ScoreProgram("FIXED", [-.6,-.2,.2,.6], K=100, tau0=0., path=path)
        zero = ScoreProgram("ZERO", [-.6,-.2,.2,.6], K=100, path=path)
        fl, fg = gradients(fixed, batch); zl, zg = gradients(zero, batch)
        value_error = abs(float((fl-zl).detach()))
        gradient_error = max(float((fg[k]-zg[k]).abs().max()) for k in fg)
        assert value_error < (4e-5 if path == "official32" else 1e-12)
        assert gradient_error < (4e-6 if path == "official32" else 1e-12)
        identities.append(dict(path=path, value_error=value_error, gradient_error=gradient_error))
    geometry = []
    for signs in itertools.product((-1.,1.), repeat=6):
        program = ScoreProgram("FREE", [0.]*4)
        with torch.no_grad():
            program.rho.copy_(torch.tensor(.95*np.asarray(signs),dtype=program.dtype))
            program.mu.copy_(torch.tensor([-3.,-1.,1.,3.],dtype=program.dtype))
            program.s.copy_(torch.tensor([-.1,.01,1.,3.],dtype=program.dtype))
            program.tau.copy_(torch.tensor(-.1,dtype=program.dtype))
        repair = program.restore_feasibility()
        assert program.physical_state()["mu"] == [-2.,-1.,1.,2.]
        assert float(program.tau) == 0.
        geometry.append(repair["eigmin"])
    adam=[]
    batch, _ = deterministic_batch()
    models, optimizers = make_programs([-.6,-.2,.2,.6])
    for method,program in models.items():
        _, g = gradients(program, batch)
        expected = {name:p.detach().numpy().astype(float)-.02*g[name].detach().numpy().astype(float)/(abs(g[name].detach().numpy().astype(float))+1e-8)
                    for name,p in program.named_parameters()}
        result = optimizer_step(program,optimizers[method],batch,True)
        assert not any(result["repair"]["projected"].values())
        error=max(float(np.max(abs(p.detach().numpy()-expected[name]))) for name,p in program.named_parameters())
        assert error < 2e-7
        adam.append(dict(method=method, maximum_parameter_error=error))
    assert len({id(o) for o in optimizers.values()}) == 3
    assert len({id(p) for model in models.values() for p in model.parameters()}) == 10
    forced=ScoreProgram("FREE",[0.]*4)
    with torch.no_grad():
        forced.rho.fill_(.89998); forced.tau.zero_(); forced.s.fill_(.001)
    opt=make_optimizer(forced)
    for name,p in forced.named_parameters():
        p.grad=torch.full_like(p,-1. if name=="rho" else 1.)
    opt.step()
    moments={name:{k:opt.state[p][k].clone() for k in ("exp_avg","exp_avg_sq")} for name,p in forced.named_parameters()}
    repaired=forced.restore_feasibility()
    assert repaired["projected"]["rho"] and repaired["projected"]["s"] and repaired["projected"]["tau"]
    assert all(torch.equal(opt.state[p][k],moments[name][k]) for name,p in forced.named_parameters() for k in moments[name])
    return dict(passed=True,**provenance(),finite_differences=rows,zero_identity=identities,
        geometry_vertices=64,minimum_repaired_eigenvalue=min(geometry),first_Adam_step=adam,
        private_optimizers=True,post_repair_moments_unchanged=True,
        new_scientific_RNG_draws=0,new_model_integrals=0,new_scientific_fits=0,
        test_scope="Deterministic tensors, finite differences, geometric repair, and isolated first Adam updates; no trajectory pilot")


def bench():
    batch,_=deterministic_batch(B=64)
    times={m:[] for m in METHODS};checksums={m:[] for m in METHODS}
    for repeat in range(5):
        models,optimizers=make_programs([-.6,-.2,.2,.6])
        for method,program in models.items():
            start=time.perf_counter();result=optimizer_step(program,optimizers[method],batch,True)
            elapsed=time.perf_counter()-start
            if repeat>=2:times[method].append(elapsed);checksums[method].append(result["loss"])
    assert all(len(set(values))==1 for values in checksums.values())
    medians={m:float(np.median(values)) for m,values in times.items()}
    return dict(**provenance(),K=100,batch=64,n=256,method_seconds=times,method_median_seconds=medians,
        projected_16_streams_2_starts_512_steps_compute_minutes=sum(medians.values())*16*2*512/60,
        timing_scope="Isolated first updates on deterministic inputs, includes full diagnostics, excludes sampling/storage/reference scoring/multiprocessing contention",
        new_scientific_RNG_draws=0,new_model_integrals=0,new_scientific_fits=0)


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=("check","bench","provenance"))
    args=parser.parse_args()
    print(json.dumps(globals()[args.phase](),ensure_ascii=False,indent=2,allow_nan=False))
