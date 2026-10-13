"""Manufactured-only qualification; never reads saved scientific banks."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import asymptotic_bridge_core as core
import numpy as np
from scipy.special import expit, logsumexp


def scalar_reference(z, prefixes, u, m):
    """Separate explicit scalar likelihood implementation, no production score call."""
    result = np.zeros((len(z), 4, 5))
    def transform(bank, rho):
        C = (1 - rho) * np.eye(3) + rho * np.ones((3, 3))
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        root = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
        return bank @ root
    for b, bank in enumerate(z):
        xs = [transform(bank, r) for r in (.45, float(u[b]), .9)]
        for ni in range(4):
            ss = prefixes[b, ni]
            for y in core.YS:
                for arm, x in enumerate(xs[:2]):
                    ll = np.asarray([float(ss[k]) - sum(float(np.logaddexp(0., -int(y[j]) * x[k, j])) for j in range(3)) for k in range(len(bank))])
                    norm = logsumexp(ll)
                    pi = np.exp(ll - norm)
                    gamma = np.asarray([pi[k] * math.fsum(float(pi[l]) for l in range(len(bank)) if l != k) for k in range(len(bank))])
                    r = np.asarray([[int(y[j]) * float(expit(-int(y[j]) * x[k, j])) for j in range(3)] for k in range(len(bank))])
                    h = r[:, 0] * r[:, 1]
                    if arm == 0:
                        result[b, ni, 1] -= math.fsum(float(gamma[k] * h[k]) for k in range(len(bank))) / 8
                        result[b, ni, 2] += math.fsum(float(v) for v in gamma) / 8
                        result[b, ni, 3] += math.fsum(float(gamma[k] * (h[k] - m)) for k in range(len(bank))) / 8
                    else:
                        result[b, ni, 0] += .9 / 8 * math.fsum(float(gamma[k] * (r[k, 0]*r[k, 1]+r[k, 0]*r[k, 2]+r[k, 1]*r[k, 2])) for k in range(len(bank)))
                l0 = np.asarray([ss[k] - sum(float(np.logaddexp(0., -int(y[j])*bank[k, j])) for j in range(3)) for k in range(len(bank))])
                lb = np.asarray([ss[k] - sum(float(np.logaddexp(0., -int(y[j])*xs[2][k, j])) for j in range(3)) for k in range(len(bank))])
                result[b, ni, 4] += (logsumexp(lb)-logsumexp(l0))/8
        if np.max(np.abs(bank)) > 8:
            result[b, :, 4] = 0
    return result


def mixed_hessian_fd(x, s):
    """Sum same-particle logit mixed Hessians, not a covariance path derivative."""
    h = 2e-4
    def objective(a, y):
        return -logsumexp(s - np.logaddexp(0., -a*y).sum(axis=1))
    out = 0.
    for y in core.YS:
        for k in range(len(x)):
            ei, ej = np.zeros_like(x), np.zeros_like(x)
            ei[k, 0], ej[k, 1] = h, h
            out += (objective(x+ei+ej, y)-objective(x+ei-ej, y)
                    -objective(x-ei+ej, y)+objective(x-ei-ej, y))/(4*h*h*8)
    return out


def qualification():
    start = time.monotonic()
    max_scalar = max_identity = max_root = 0.
    fixtures = []
    m = .0123
    for k in (1, 2, 3, 100):
        z = 2.1*np.sin(np.arange(3*k*3).reshape(3, k, 3)*.71)
        s = np.stack([np.cos(np.arange(3*k).reshape(3,k)*.3)*v for v in (0., 2., 30., 300.)], axis=1)
        u = np.asarray([0., .45, .9])
        got, clips = core.score_block(z, s, u, m)
        want = scalar_reference(z, s, u, m)
        error = float(np.max(np.abs(got-want)))
        identity = float(np.max(np.abs(got[:,:,1]+m*got[:,:,2]+got[:,:,3])))
        assert error < 2e-12, (k, error)
        assert identity < 5e-15, (k, identity)
        max_scalar, max_identity = max(max_scalar,error), max(max_identity,identity)
        assert not clips.any()
        support = 1-1/k
        assert np.all(got[:,:,0] >= -.675*support-1e-13)
        assert np.all(got[:,:,0] <= 1.35*support+1e-13)
        assert np.all(abs(got[:,:,1]) <= support/2+1e-13)
        assert np.all(got[:,:,2] >= -1e-13) and np.all(got[:,:,2] <= support+1e-13)
        assert np.all(got[:,:,3] >= -(.5+m)*support-1e-13)
        assert np.all(got[:,:,3] <= .5*(1-m)*support+1e-13)
        if k == 1:
            assert np.array_equal(got[:,:,:4], np.zeros((3,4,4)))
        fixtures.append(dict(K=k,banks=3,max_scalar_error=error,max_decomposition_error=identity))
    for rho in (0., .45, .9):
        root = core.equicorrelated(np.eye(3)[None],rho)[0]
        residual = float(np.max(abs(root@root.T-((1-rho)*np.eye(3)+rho*np.ones((3,3))))))
        max_root=max(max_root,residual)
        assert residual < 2e-15
    x = np.asarray([[-.7,.4,1.2],[1.1,-.8,.2],[.3,.2,-1.]])
    s = np.asarray([-.2,-2.,-.8])
    fd = mixed_hessian_fd(x,s)
    exact = float(core.fixed_price(x[None],s[None],m)[0,0])
    fd_error = abs(fd-exact)
    assert fd_error < 2e-7, fd_error
    pi,gamma = core.posterior_gamma(np.asarray([[0.,-40.]]))
    assert pi[0,0] == 1. and gamma[0,0] > 0.
    assert abs(gamma[0,0]-gamma[0,1]) < 1e-32
    z = np.asarray([[[9.,0.,0.],[0.,1.,-1.]]])
    got,mask = core.score_block(z,np.zeros((1,4,2)),np.asarray([.3]),m)
    assert mask.tolist()==[True] and np.array_equal(got[:,:,4],np.zeros((1,4)))
    assert np.any(got[:,:,0] != 0), "primary must not be clipped"
    # A small distinct manufactured seed checks the primitive-generation contract.
    dummy_seed = 202001010000
    zg,sg,ug=core.generate_block(0,seed_family=dummy_seed,banks=3,particles=3)
    rng=np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([dummy_seed,0,1])))
    raw=rng.standard_normal((3,3,4096))
    want=np.stack([(-np.logaddexp(0.,-raw[:,:,:n])).sum(axis=2) for n in core.N_VALUES],axis=1)
    generation_error=float(np.max(abs(sg-want)))
    assert generation_error < 5e-12,generation_error
    root=Path(__file__).resolve().parents[1]
    sources={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
             for p in (Path(__file__),root/"script/asymptotic_bridge_core.py")}
    return dict(schema_version="1.0",passed=True,manufactured_only=True,scientific_banks=0,
                scientific_seed_calls=0,fixture_banks=12,label_count=8,fixtures=fixtures,
                max_scalar_score_error=max_scalar,max_decomposition_error=max_identity,
                max_root_residual=max_root,mixed_logit_hessian_fd_error=fd_error,
                mixed_logit_hessian_fd_step=2e-4,dominant_complement_preserved=True,
                primary_not_clipped=True,K1_price_zero=True,primitive_generation_error=generation_error,
                manufactured_rng_seed=dummy_seed,manufactured_normal_scalars=3*3*(4096+3),
                sources=sources,
                elapsed_seconds=time.monotonic()-start,
                scope="manufactured finite inputs and scalar/logit-Hessian checks; not a population or floating-point certificate")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path)
    parser.add_argument("--replace-manufactured",action="store_true",help="retain prior manufactured receipt inside the replacement")
    args=parser.parse_args()
    result=qualification()
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        if args.output.exists() and args.replace_manufactured:
            previous=json.loads(args.output.read_text("utf-8"))
            assert previous.get("manufactured_only") and previous.get("scientific_banks")==0
            result["previous_manufactured_receipt"]=previous
        with args.output.open("w" if args.replace_manufactured else "x",encoding="utf-8",newline="\n") as f:
            json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
