"""Differentiable controlled optimization through the genuine SCBM score API.

Only deterministic self-tests and fixed-shape timing are executable here. No
scientific sampler, training runner, pilot, or frozen experiment is supplied.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
import scbminv_kernel as legacy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/scbm_scale_optimization/KERNEL'
SOURCE = Path(__file__).resolve()
LEGACY_SHA = 'c6712abbb4e05ef0c25c688a05dad106034c41888040592424f50362f04129ad'
FLOOR = 1e-6
T_BOUNDS = (-.75, .75)
METHODS = ('FIXED', 'FREE', 'FACT')
DTYPE = torch.float32


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def provenance():
    assert sha(legacy.__file__) == LEGACY_SHA, 'Frozen adapter changed'
    return dict(source_sha256=sha(SOURCE), legacy_source=legacy.rel(Path(legacy.__file__)),
                legacy_source_sha256=LEGACY_SHA, official=legacy.provenance(),
                fixture=legacy.FIXTURE, floor=FLOOR, physical_t_bounds=list(T_BOUNDS),
                fixed_mu_and_diagonal=True, parameter_dtype='float32',
                layout='Explicit standard normals [K,B,C]; logits movedim(0,-1) to [B,C,K]',
                scope='Actual unmodified concept-loss API only; no encoder, task head, precision penalty, official optimizer or checkpoint reproduction')


def save_new(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open('x', encoding='utf8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


@dataclass(frozen=True)
class Batch:
    active_eps: torch.Tensor
    nuisance_eps: torch.Tensor
    active_labels: torch.Tensor
    nuisance_labels: torch.Tensor


def prepare_batch(active_eps, nuisance_eps, active_labels, nuisance_labels):
    """One CPU conversion shared by all methods, without draws or mutation."""
    eps = torch.as_tensor(active_eps, dtype=DTYPE, device='cpu').contiguous()
    nu = torch.as_tensor(nuisance_eps, dtype=DTYPE, device='cpu').contiguous()
    y_source = torch.as_tensor(active_labels, device='cpu')
    yn_source = torch.as_tensor(nuisance_labels, device='cpu')
    assert eps.ndim == nu.ndim == 3 and eps.shape[2] == 4
    assert eps.shape[:2] == nu.shape[:2]
    assert y_source.shape == (eps.shape[1], 4) and yn_source.shape == (eps.shape[1], nu.shape[2])
    assert bool(torch.all((y_source == 0) | (y_source == 1)))
    assert bool(torch.all((yn_source == 0) | (yn_source == 1)))
    assert bool(torch.isfinite(eps).all()) and bool(torch.isfinite(nu).all())
    return Batch(eps, nu, y_source.to(torch.int64).contiguous(), yn_source.to(torch.int64).contiguous())


class ScoreProgram(torch.nn.Module):
    """Physical t and, for FREE only, one raw scale per nuisance coordinate.

    Optimizers are external and private to each program. The finite-particle
    score is float32 because the original API casts labels to float32.
    """
    def __init__(self, method, K=100, n=256, t0=0., scale0=1.):
        super().__init__()
        assert sha(legacy.__file__) == LEGACY_SHA
        assert method in METHODS and isinstance(K, int) and K >= 1
        assert isinstance(n, int) and n >= 1
        assert math.isfinite(t0) and T_BOUNDS[0] <= t0 <= T_BOUNDS[1]
        assert math.isfinite(scale0) and scale0 > FLOOR
        self.method, self.K, self.n = method, K, n
        self.t = torch.nn.Parameter(torch.tensor(t0, dtype=DTYPE))
        self.register_buffer('mu', torch.tensor(legacy.FIXTURE['mu'], dtype=DTYPE))
        self.register_buffer('identity', torch.eye(4, dtype=DTYPE))
        self.register_buffer('direction', torch.tensor(legacy.FIXTURE['C0'], dtype=DTYPE) - self.identity)
        self.register_buffer('fixed_scales', torch.full((n,), scale0, dtype=DTYPE))
        if method == 'FREE':
            # Numerically stable inverse softplus, including larger scales.
            target = scale0 - FLOOR
            raw0 = target + math.log(-math.expm1(-target))
            self.raw_scale = torch.nn.Parameter(torch.full((n,), raw0, dtype=DTYPE))
        else:
            self.register_parameter('raw_scale', None)

    def scales(self):
        return F.softplus(self.raw_scale) + FLOOR if self.raw_scale is not None else self.fixed_scales

    def covariance(self):
        # Never convert t to a Python float: its Cholesky derivative is required.
        return self.identity + (1. + self.t) * self.direction

    def forward(self, batch, diagnostics=None):
        assert batch.active_eps.shape[0] == self.K and batch.nuisance_eps.shape[2] == self.n
        lower = torch.linalg.cholesky(self.covariance())
        active = torch.matmul(lower, batch.active_eps.unsqueeze(-1)).squeeze(-1) + self.mu
        if self.method == 'FACT':
            logits, labels = active, batch.active_labels
        else:
            nuisance = batch.nuisance_eps * self.scales()
            logits = torch.cat((active, nuisance), dim=-1)
            labels = torch.cat((batch.active_labels, batch.nuisance_labels), dim=1)
        probabilities = torch.sigmoid(logits.movedim(0, -1))
        # Actual bound method from the frozen public source, not an equivalent
        # hand-written loss. This scalar retains the complete autograd graph.
        loss = legacy.official_loss(self.K).compute_concept_loss(probabilities, labels)
        if self.method == 'FACT':
            loss = loss + self.n * math.log(2.)
        if diagnostics is not None:
            p = probabilities.detach()
            diagnostics.update(official_API_calls=1, probability_evaluations=p.numel(),
                sigmoid_zero_count=int((p == 0).sum()), sigmoid_one_count=int((p == 1).sum()),
                maximum_absolute_logit=float(logits.detach().abs().max()),
                probability_min=float(p.min()), probability_max=float(p.max()))
        return loss

    @torch.no_grad()
    def project_t(self):
        self.t.clamp_(*T_BOUNDS)


def make_optimizer(program, lr_t=.02, lr_scale=.02):
    assert math.isfinite(lr_t) and math.isfinite(lr_scale) and lr_t > 0 and lr_scale > 0
    groups = [dict(params=[program.t], lr=lr_t)]
    if program.raw_scale is not None:
        groups.append(dict(params=[program.raw_scale], lr=lr_scale))
    return torch.optim.Adam(groups, betas=(.9, .999), eps=1e-8,
                            weight_decay=0., amsgrad=False, foreach=False, fused=False)


def make_programs(K=100, n=256, t0=0., scale0=1., lr_t=.02, lr_scale=.02):
    models = {method: ScoreProgram(method, K, n, t0, scale0) for method in METHODS}
    optimizers = {method: make_optimizer(model, lr_t, lr_scale) for method, model in models.items()}
    return models, optimizers


def gradients(program, batch):
    loss = program(batch)
    params = tuple(program.parameters())
    g = torch.autograd.grad(loss, params)
    return loss, {name: value for (name, _), value in zip(program.named_parameters(), g)}


def optimizer_step(program, optimizer, batch, diagnostics=False):
    """One actual update, projecting physical t only; raw scales never clamped."""
    optimizer.zero_grad(set_to_none=True)
    detail = {} if diagnostics else None
    loss = program(batch, detail)
    assert bool(torch.isfinite(loss)), 'Nonfinite loss; do not retry this input'
    loss.backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in program.parameters())
    before = float(program.t.detach())
    grad_t = float(program.t.grad.detach())
    optimizer.step()
    attempted = float(program.t.detach())
    program.project_t()
    scales = program.scales().detach()
    assert all(bool(torch.isfinite(p).all()) for p in program.parameters())
    assert bool(torch.isfinite(scales).all()) and bool((scales >= FLOOR).all())
    return dict(loss=float(loss.detach()), t_before=before, t_gradient=grad_t,
                t_unprojected=attempted, t_after=float(program.t.detach()),
                projected=attempted != float(program.t.detach()),
                scale_min=float(scales.min()), scale_mean=float(scales.mean()),
                scale_max=float(scales.max()), diagnostics=detail)


def deterministic_batch(K=100, B=7, n=256):
    arrays = legacy.deterministic_inputs(K, B, n)
    return prepare_batch(*arrays), arrays


def finite_difference(program, batch, name, index=None, step=.02):
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
                values.append(float(program(batch)))
    finally:
        with torch.no_grad():
            parameter.copy_(original)
    return (values[0] - 8 * values[1] + 8 * values[2] - values[3]) / (12 * step)


def check():
    prov = provenance()
    batch, arrays = deterministic_batch()
    compatible = []
    for t in (-.75, 0., .75):
        old = legacy.evaluate(*arrays, t=t, padding=256, need_gradient=True)
        for method in METHODS:
            program = ScoreProgram(method, t0=t)
            loss, grads = gradients(program, batch)
            comparison = legacy.evaluate(*arrays, t=t, padding=256, factorized=True, need_gradient=True) if method == 'FACT' else old
            value_error = abs(float(loss.detach()) - comparison['value'])
            t_error = abs(float(grads['t']) - comparison['gradient']['correlation_direction'])
            assert value_error <= 4e-5 and t_error <= 3e-6, (method, t, value_error, t_error)
            row = dict(method=method, t=t, value_error=value_error, t_gradient_error=t_error,
                       scale_initial_min=float(program.scales().detach().min()), scale_initial_max=float(program.scales().detach().max()))
            if method == 'FREE':
                # Common physical scale derivative equals sum_i raw-gradient_i
                # divided by d softplus(raw0)/d raw0 at equal initialization.
                common = float((grads['raw_scale'] / torch.sigmoid(program.raw_scale.detach())).sum())
                error = abs(common - old['gradient']['nuisance_scale'])
                assert error <= 2e-5, error
                row['common_physical_scale_gradient_error'] = error
            compatible.append(row)
    small, _ = deterministic_batch(n=5)
    fd = []
    for method in METHODS:
        program = ScoreProgram(method, n=5, t0=.13)
        _, grad = gradients(program, small)
        analytic = [float(grad['t'])]
        numeric = [finite_difference(program, small, 't')]
        if method == 'FREE':
            for i in range(5):
                analytic.append(float(grad['raw_scale'][i]))
                numeric.append(finite_difference(program, small, 'raw_scale', i))
        error = float(np.max(np.abs(np.asarray(analytic) - numeric)))
        assert error < 2e-4, (method, error)
        fd.append(dict(method=method, coordinates=['t'] + ([f'raw_{i}' for i in range(5)] if method == 'FREE' else []),
                       analytic=analytic, finite_difference=numeric, step=.02, maximum_error=error, tolerance=2e-4))
    free = ScoreProgram('FREE')
    _, g = gradients(free, batch)
    summed = float(g['raw_scale'].sum())
    directional = finite_difference(free, batch, 'raw_scale', step=.02)
    assert abs(summed - directional) < 1e-3
    base = legacy.evaluate(*arrays, t=.13, padding=0, need_gradient=True)
    fact = ScoreProgram('FACT', t0=.13)
    fact_loss, fact_g = gradients(fact, batch)
    factor_value_error = abs(float(fact_loss.detach()) - base['value'] - 256 * math.log(2.))
    factor_gradient_error = abs(float(fact_g['t']) - base['gradient']['correlation_direction'])
    assert factor_value_error < 3e-5 and factor_gradient_error < 1e-7
    # Independent hand calculation of the first Adam update. Reset models for
    # each check; no trajectory or convergence pilot is performed.
    adam_checks = []
    models, optimizers = make_programs()
    for method, program in models.items():
        _, grad = gradients(program, batch)
        expected = {}
        for name, parameter in program.named_parameters():
            p = parameter.detach().numpy().astype(np.float64)
            g0 = grad[name].detach().numpy().astype(np.float64)
            candidate = p - .02 * g0 / (np.abs(g0) + 1e-8)
            expected[name] = np.clip(candidate, *T_BOUNDS) if name == 't' else candidate
        result = optimizer_step(program, optimizers[method], batch, diagnostics=True)
        errors = {name: float(np.max(np.abs(parameter.detach().numpy() - expected[name]))) for name, parameter in program.named_parameters()}
        assert max(errors.values()) < 2e-7, (method, errors)
        assert len(optimizers[method].state) == (2 if method == 'FREE' else 1)
        adam_checks.append(dict(method=method, maximum_parameter_errors=errors, step=result))
    assert len({id(p) for m in models.values() for p in m.parameters()}) == 4
    assert len({id(o) for o in optimizers.values()}) == 3
    free = ScoreProgram('FREE', n=3)
    with torch.no_grad():
        free.raw_scale.copy_(torch.tensor([-100., 0., 100.]))
        free.t.copy_(torch.tensor(2.))
        raw_before = free.raw_scale.clone()
        free.project_t()
    assert float(free.t.detach()) == .75 and torch.equal(raw_before, free.raw_scale)
    scales = free.scales().detach()
    assert bool(torch.isfinite(scales).all()) and bool((scales >= FLOOR).all())
    with torch.no_grad():
        free.t.copy_(torch.tensor(-2.))
        free.project_t()
    assert float(free.t.detach()) == -.75 and torch.equal(raw_before, free.raw_scale)
    # Explicitly force an Adam update across the allowed t endpoint and verify
    # that only physical t is projected, with Adam's moments retained.
    boundary = ScoreProgram('FREE', n=3, t0=.75)
    opt = make_optimizer(boundary)
    boundary.t.grad = torch.tensor(-1.)
    boundary.raw_scale.grad = torch.full((3,), 1.)
    raw_before = boundary.raw_scale.detach().clone()
    opt.step()
    attempted = float(boundary.t.detach())
    boundary.project_t()
    assert attempted > .75 and float(boundary.t.detach()) == .75
    assert bool((boundary.raw_scale.detach() < raw_before).all())
    output = dict(passed=True, **prov, compatible_with_frozen_kernel=compatible,
        finite_differences=fd, full256_common_raw_direction=dict(analytic=summed, finite_difference=directional,
            absolute_error=abs(summed-directional), step=.02, tolerance=1e-3),
        factorized_identity=dict(value_error=factor_value_error, t_gradient_error=factor_gradient_error,
            nuisance_independent=True), adam_first_step_checks=adam_checks,
        optimizer_states_private=True, scale_floor_check=dict(raw_values=[-100.,0.,100.],scales=scales.tolist(),
            note='Mathematically softplus(raw)+floor is strictly above floor; float32 can round to floor for very negative raw.'),
        projection_check=dict(attempted_t=attempted, final_t=float(boundary.t.detach()), raw_not_projected=True),
        new_scientific_RNG_draws=0, new_model_integrals=0, new_data_fits=0,
        test_scope='Deterministic tensor self-test and isolated one-step optimizer checks; no scientific sample or convergence pilot')
    save_new('SELFTEST.json', output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


def bench():
    prov = provenance()
    cases = []
    for B in (64, 128):
        batch, _ = deterministic_batch(B=B)
        elapsed = {m: [] for m in METHODS}
        checksums = {m: [] for m in METHODS}
        for repeat in range(9):
            # Initialization occurs outside the timer. Every measured update is
            # the first Adam step from the same parameters on the same tensors.
            models, optimizers = make_programs()
            for method in METHODS:
                start = time.perf_counter()
                result = optimizer_step(models[method], optimizers[method], batch)
                seconds = time.perf_counter() - start
                if repeat >= 2:
                    elapsed[method].append(seconds)
                    checksums[method].append(result['loss'])
        assert all(len(set(v)) == 1 for v in checksums.values())
        timings = {method: dict(seconds=elapsed[method], median_seconds=float(np.median(elapsed[method])),
                               mean_seconds=float(np.mean(elapsed[method])), loss_checksum=checksums[method][0]) for method in METHODS}
        total = sum(r['median_seconds'] for r in timings.values())
        cases.append(dict(K=100, batch=B, nuisance=256, methods=timings,
                          three_programs_median_step_seconds=total,
                          projected_16_streams_1024_steps_seconds=total*16*1024,
                          projected_minutes=total*16*1024/60,
                          projected_16_streams_minutes_by_steps={str(steps):total*16*steps/60 for steps in (512,1024,2048)}))
    output = dict(**prov, cases=cases, warmup_repeats=2, measured_repeats=7,
        repeated_inputs='Deterministic sine/cosine tensors, every step reset to same t0=0/scales1 and empty optimizer state',
        optimizer='Private Adam per method; t/raw lr=.02, beta=(.9,.999), eps=1e-8, no weight decay; physical t clamped after step',
        timing_scope='Forward + backward + Adam + finite checks + t projection; preconversion/model construction excluded. Projection excludes RNG, storage, checkpoint scoring and thread contention.',
        new_scientific_RNG_draws=0, new_model_integrals=0, new_data_fits=0)
    save_new('BENCHMARK.json', output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['check', 'bench', 'provenance'])
    phase = parser.parse_args().phase
    if phase == 'provenance':
        print(json.dumps(provenance(), ensure_ascii=False, indent=2))
    else:
        globals()[phase]()
