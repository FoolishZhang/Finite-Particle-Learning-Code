"""Different-expression Hermite audit of the frozen limit calculations.

The same implementer wrote this and reglearn_numeric.py: this is not a blind
independent review.  Exact Hermite-series tails are bounded using Var(f)<=1;
quadrature errors in computed coefficients are qualified, not certified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import roots_hermitenorm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/regularization_learning/NUMERIC"
M = 96
R = .75


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


class HermiteAudit:
    def __init__(self, a, q):
        z, w = roots_hermitenorm(q)
        w = w/np.sqrt(2*np.pi)
        f = np.tanh(a*z/2)
        polys = np.empty((M+1, q))
        polys[0], polys[1] = 1.0, z
        for m in range(1, M):
            polys[m+1] = (z*polys[m]-np.sqrt(m)*polys[m-1])/np.sqrt(m+1)
        b = polys @ (w*f)
        b[::2] = 0.0  # exact odd symmetry, not fitted coefficient truncation
        self.b, self.c, self.a, self.q = b, b*b, a, q
        self.d = float(self.c[1])
        self.A = a*a/4
        self.varf = float(w @ (f*f))
        self.d_price_check = float(w @ ((a/2)*(1-f*f)))**2
        logsigmoid = -np.logaddexp(0, -a*z)
        self.logsigmoid_mean = float(w @ logsigmoid)
        self.logsigmoid_var = float(w @ ((logsigmoid-self.logsigmoid_mean)**2))
        self.root_calls = 0

    def r(self, rho):
        return float(np.polynomial.polynomial.polyval(rho, self.c))

    def rp(self, rho):
        return float(np.polynomial.polynomial.polyval(rho, np.arange(1, M+1)*self.c[1:]))

    def H(self, rho):
        return float(np.polynomial.polynomial.polyval(rho, np.r_[0.0, self.c/np.arange(1, M+2)]))

    def root(self, f, lo, hi):
        self.root_calls += 1
        return float(brentq(f, lo, hi, xtol=1e-13, rtol=1e-13))

    def minimize(self, theta, rule):
        t = self.r(theta)
        if rule == "raw":
            candidates = [-R, R]
            potential = lambda x: -self.A*(t*x+self.H(x))
        elif rule == "zero":
            candidates = [0.0]
            potential = lambda x: 0.0
        elif rule == "informed":
            candidates = [-R, R]
            derivative = lambda x: self.r(x)-t
            if abs(derivative(-R)) < 1e-14:
                candidates.append(-R)
            elif abs(derivative(R)) < 1e-14:
                candidates.append(R)
            else:
                candidates.append(self.root(derivative, -R, R))
            potential = lambda x: self.A*(self.H(x)-t*x)
        else:
            critical = []
            if self.rp(R) > 2*self.d:
                z = self.root(lambda x: self.rp(x)-2*self.d, 0.0, R)
                critical = [-z, z]
            cuts = sorted([-R, 0.0, R]+critical)
            candidates = [-R, R]
            deriv = lambda x: 2*self.d*x-t-self.r(x)
            for left, right in zip(cuts[:-1], cuts[1:]):
                if abs(deriv(left)) < 1e-14:
                    candidates.append(left)
                if abs(deriv(right)) < 1e-14:
                    candidates.append(right)
                if deriv(left)*deriv(right) < 0:
                    candidates.append(self.root(deriv, left, right))
            potential = lambda x: self.A*(self.d*x*x-t*x-self.H(x))
        best = min(potential(x) for x in candidates)
        rho = min((x for x in candidates if potential(x) <= best+1e-12), key=lambda x: (abs(x), x))
        rr = self.r(rho)
        # Alternate risk expression via the two observed Bernoulli probabilities.
        p, q = (1+t)/2, (1+rr)/2
        kl = float(max(0.0, p*np.log(p/q)+(1-p)*np.log((1-p)/(1-q))))
        return {"rho": rho, "r": rr, "H": self.H(rho), "joint_KL": kl}


def run():
    prod = json.loads((OUT / "RUN.json").read_text(encoding="utf-8"))
    if (OUT / "AUDIT.json").exists():
        raise RuntimeError("Refuse to overwrite audit output")
    cells, checks = [], []
    def check(kind, a, q, field, x, y, tol):
        diff = abs(float(x)-float(y))
        checks.append({"kind": kind, "a": a, "order": q, "field": field,
                       "absolute_difference": diff, "tolerance": tol, "pass": diff <= tol})
    for source in prod["cells"]:
        a, q = source["a"], source["order"]
        eng = HermiteAudit(a, q)
        constants = {"d": eng.d, "gamma0": eng.A*eng.d,
                     "gamma_local": eng.A*eng.d/2, "gamma_global": eng.A*eng.H(R)/R**2,
                     "var_f": eng.varf, "r_R": eng.r(R), "r_prime_R": eng.rp(R), "H_R": eng.H(R),
                     "mean_logsigmoid": eng.logsigmoid_mean, "var_logsigmoid": eng.logsigmoid_var}
        for key, v in constants.items():
            check("different_expression", a, q, "constants/"+key, v, source["constants"][key], 1e-9)
        check("integration_by_parts", a, q, "d", eng.d, eng.d_price_check, 1e-9)
        probes = []
        for i, p in enumerate(source["probes"]):
            vals = {"rho": p["rho"], "r": eng.r(p["rho"]), "r_prime": eng.rp(p["rho"]), "H": eng.H(p["rho"])}
            probes.append(vals)
            for key in ("r", "r_prime", "H"):
                check("different_expression", a, q, f"probes/{i}/{key}", vals[key], p[key], 1e-9)
        rows = []
        for i, p in enumerate(source["rows"]):
            rules = {rule: eng.minimize(p["theta"], rule) for rule in ("raw", "quadratic", "informed", "zero")}
            rows.append({"theta": p["theta"], "rules": rules})
            for rule in rules:
                for key, tol in (("rho", 1e-6), ("r", 1e-9), ("joint_KL", 1e-8)):
                    check("different_expression", a, q, f"rows/{i}/{rule}/{key}", rules[rule][key], p["rules"][rule][key], tol)
        cells.append({"a": a, "order": q, "constants": constants, "b_m": eng.b.tolist(),
                      "c_m": eng.c.tolist(), "probes": probes, "rows": rows,
                      "coefficient_integrals": M+1, "additional_scalar_integrals": 4,
                      "root_calls": eng.root_calls})
    for a in (.5, 1, 2, 4):
        main = next(c for c in cells if c["a"] == a and c["order"] == 64)
        for q in (48, 96):
            other = next(c for c in cells if c["a"] == a and c["order"] == q)
            for i, v in enumerate(main["c_m"]):
                check("coefficient_order_agreement", a, q, f"c_{i}", v, other["c_m"][i], 1e-9)
    result = {"schema": 1, "source_sha256": sha(__file__), "production_sha256": sha(OUT / "RUN.json"),
              "M": M, "method": "Normalized probabilists' Hermite coefficients; Mehler series, polynomial antiderivative, alternate Bernoulli risk expression",
              "independence": "Different mathematical expression; same implementer. Not a blind independent audit.",
              "exact_series_tail_bounds_using_Var_f_le_1": {"r_uniform_R": R**(M+1),
                    "H_uniform_R": R**(M+2)/(M+2), "r_prime_uniform_R": (M+1)*R**M},
              "tail_scope": "Bounds apply to exact Hermite coefficients, NOT to quadrature error in computed coefficients; no strict total integration certificate.",
              "cells": cells, "checks": checks, "check_count": len(checks),
              "passed": sum(x["pass"] for x in checks), "failed": [x for x in checks if not x["pass"]],
              "accounting": {"one_dimensional_scalar_integrals": len(cells)*(M+5),
                   "root_calls": sum(c["root_calls"] for c in cells), "random_draws": 0, "new_labels": 0, "banks": 0}}
    dump(OUT / "AUDIT.json", result)
    print(json.dumps({"checks": result["check_count"], "passed": result["passed"],
                      "failed": len(result["failed"]), "accounting": result["accounting"]}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="store_true", required=True)
    p.parse_args()
    run()
