"""Frozen deterministic regularization limit calculations (no random draws).

This is independent of the external review's inaccessible sandbox scripts.
Run only after the package's FREEZE.json has been written.  Quadrature order
agreement is a numerical qualification, not a rigorous integration certificate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import roots_hermitenorm, roots_legendre

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/regularization_learning/NUMERIC"
PLAN = ROOT / "docs/08_research_研究工作/统一正则与真实相关学习_20261002/PLAN.md"
PLAN_SHA = "b1f1363b7238e34fa7acafd0be6772f4c2cddc313352e1e31a1161e09657e04a"
ORDERS = (48, 64, 96)
SIX = (0.0, .15, .30, .45, .60, .70)
R = .75


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def observed_kl(t, r):
    # Bernoulli-pair reduction of the four-state observed joint distribution.
    ans = .5 * ((1+t)*np.log1p((t-r)/(1+r)) + (1-t)*np.log1p((r-t)/(1-r)))
    return float(max(0.0, ans))


class DirectGH:
    def __init__(self, a, order):
        self.a, self.order = float(a), int(order)
        self.x, self.w = roots_hermitenorm(order)
        self.w = self.w / np.sqrt(2*np.pi)
        self.ww = self.w[:, None] * self.w[None, :]
        self.fx = np.tanh(a*self.x/2)
        self.dx = (a/2)*(1-self.fx**2)
        self.A = a*a/4
        self.m = float(self.w @ self.dx)
        self.d = self.m*self.m
        self.varf = float(self.w @ self.fx**2)
        log_sigmoid = -np.logaddexp(0.0, -a*self.x)
        self.mean_logsigmoid = float(self.w @ log_sigmoid)
        self.var_logsigmoid = float(self.w @ (log_sigmoid-self.mean_logsigmoid)**2)
        self.cache, self.hcache = {}, {0.0: 0.0}
        self.counts = {"one_dimensional_scalar_integrals": 4, "two_dimensional_scalar_integrals": 0,
                       "two_dimensional_node_evaluations": 0, "H_outer_integrals": 0,
                       "root_calls": 0, "root_function_calls": 0, "root_iterations": 0}
        self.glx, self.glw = roots_legendre(order)

    def pair(self, rho):
        key = float(abs(rho))
        if key not in self.cache:
            if key == 0:
                self.cache[key] = (0.0, self.d)
            else:
                y = key*self.x[:, None] + np.sqrt(1-key*key)*self.x[None, :]
                fy = np.tanh(self.a*y/2)
                rr = float(np.sum(self.ww * self.fx[:, None] * fy))
                dr = float(np.sum(self.ww * self.dx[:, None] * (self.a/2)*(1-fy*fy)))
                self.cache[key] = (rr, dr)
                self.counts["two_dimensional_scalar_integrals"] += 2
                self.counts["two_dimensional_node_evaluations"] += 2*self.order*self.order
        rr, dr = self.cache[key]
        return (float(np.copysign(rr, rho)) if rho else 0.0), dr

    def r(self, rho):
        return self.pair(rho)[0]

    def rp(self, rho):
        return self.pair(rho)[1]

    def H(self, rho):
        key = float(abs(rho))
        if key not in self.hcache:
            nodes = key*(self.glx+1)/2
            self.hcache[key] = float(key/2 * np.dot(self.glw, [self.r(x) for x in nodes]))
            self.counts["H_outer_integrals"] += 1
        return self.hcache[key]

    def root(self, fun, lo, hi):
        self.counts["root_calls"] += 1
        def wrapped(x):
            self.counts["root_function_calls"] += 1
            return fun(x)
        ans, info = brentq(wrapped, lo, hi, xtol=1e-13, rtol=1e-13, full_output=True)
        if not info.converged:
            raise RuntimeError("brentq did not converge")
        self.counts["root_iterations"] += int(info.iterations)
        return float(ans)


def candidate_minima(engine, theta, rule):
    A, d, t = engine.A, engine.d, engine.r(theta)
    if rule == "raw":
        derivative = lambda x: -A*(t+engine.r(x))
        potential = lambda x: -A*(t*x+engine.H(x))
        cuts = [-R, R]
    elif rule == "quadratic":
        derivative = lambda x: A*(2*d*x-t-engine.r(x))
        potential = lambda x: -A*(t*x+engine.H(x))+A*d*x*x
        # Mehler coefficients are nonnegative and odd: r' is even and
        # nondecreasing on [0,R].  Hence these are all monotonicity intervals
        # of the objective's first derivative, including a possible inflection.
        cuts = [-R, 0.0, R]
        if engine.rp(R) > 2*d:
            critical = engine.root(lambda x: engine.rp(x)-2*d, 0.0, R)
            cuts = [-R, -critical, 0.0, critical, R]
    elif rule == "informed":
        derivative = lambda x: A*(engine.r(x)-t)
        potential = lambda x: A*(engine.H(x)-t*x)
        cuts = [-R, R]
    else:
        raise ValueError(rule)
    roots = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        fl, fh = derivative(lo), derivative(hi)
        if abs(fl) <= 1e-14:
            roots.append(float(lo))
        if abs(fh) <= 1e-14:
            roots.append(float(hi))
        if fl*fh < 0:
            roots.append(engine.root(derivative, lo, hi))
    roots = sorted(set(round(x, 13) for x in roots))
    candidates = sorted(set([-R, R]+roots))
    entries = [{"rho": x, "potential": float(potential(x)), "derivative": float(derivative(x)),
                "r": engine.r(x), "H": engine.H(x), "joint_KL": observed_kl(t, engine.r(x))}
               for x in candidates]
    best = min(x["potential"] for x in entries)
    tied = [x for x in entries if x["potential"] <= best+1e-12]
    # Explicit deterministic choice; all numerical ties are also exported.
    chosen = min(tied, key=lambda x: (abs(x["rho"]), x["rho"]))
    return {"rho": chosen["rho"], "joint_KL": chosen["joint_KL"], "r": chosen["r"],
            "potential": chosen["potential"], "H": chosen["H"],
            "stationary_roots": roots, "derivative_monotonicity_cuts": cuts,
            "all_candidates": entries, "minimizers_within_1e-12": [x["rho"] for x in tied]}


def run_cell(a, order):
    eng = DirectGH(a, order)
    theta_values = [float(round(x, 12)) for x in np.linspace(0, R, 61)] if a == 1 else list(SIX)
    rows = []
    for theta in theta_values:
        t = eng.r(theta)
        rules = {rule: candidate_minima(eng, theta, rule) for rule in ("raw", "quadratic", "informed")}
        rules["zero"] = {"rho": 0.0, "r": 0.0, "joint_KL": observed_kl(t, 0.0)}
        rows.append({"theta": theta, "r_teacher": t, "H_teacher": eng.H(theta),
                     "teacher_exact_quadratic_stationarity_gamma": eng.A*(t/theta if theta else eng.d),
                     "N1024_pair_signal_z": float(t*np.sqrt(1024/(1-t*t))),
                     "N1024_pair_sd": float(np.sqrt((1-t*t)/1024)), "rules": rules})
    probes = [{"rho": float(rho), "r": eng.r(rho), "r_prime": eng.rp(rho), "H": eng.H(rho)}
              for rho in np.linspace(-R, R, 61)]
    constants = {"A": eng.A, "E_f_prime": eng.m, "d": eng.d, "var_f": eng.varf,
                 "mean_logsigmoid": eng.mean_logsigmoid, "var_logsigmoid": eng.var_logsigmoid,
                 "gamma0": eng.A*eng.d, "gamma_local": eng.A*eng.d/2,
                 "gamma_global": eng.A*eng.H(R)/(R*R), "r_R": eng.r(R),
                 "r_prime_R": eng.rp(R), "H_R": eng.H(R),
                 "quadratic_curvature_min": eng.A*(2*eng.d-eng.rp(R))}
    return {"a": a, "order": order, "constants": constants, "rows": rows, "probes": probes,
            "accounting": eng.counts}


def qualify(cells):
    comparisons = []
    for a in (.5, 1, 2, 4):
        main = next(c for c in cells if c["a"] == a and c["order"] == 64)
        for order in (48, 96):
            alt = next(c for c in cells if c["a"] == a and c["order"] == order)
            diffs = []
            def add(path, v, w, tol):
                diff = abs(v-w)
                diffs.append({"field": path, "absolute_difference": diff, "tolerance": tol, "pass": diff <= tol})
            for k, v in main["constants"].items():
                add("constants/"+k, v, alt["constants"][k], 1e-9)
            for i, p in enumerate(main["probes"]):
                for field in ("r", "r_prime", "H"):
                    add(f"probes/{i}/{field}", p[field], alt["probes"][i][field], 1e-9)
            for i, row in enumerate(main["rows"]):
                other = alt["rows"][i]
                for field in ("r_teacher", "H_teacher", "teacher_exact_quadratic_stationarity_gamma"):
                    add(f"rows/{i}/{field}", row[field], other[field], 1e-9)
                for rule in ("raw", "quadratic", "informed", "zero"):
                    for field, tol in (("rho", 1e-6), ("joint_KL", 1e-8), ("r", 1e-9)):
                        add(f"rows/{i}/{rule}/{field}", row["rules"][rule][field], other["rules"][rule][field], tol)
            comparisons.append({"a": a, "main_order": 64, "comparison_order": order,
                                "checks": len(diffs), "passed": sum(d["pass"] for d in diffs),
                                "failed": [d for d in diffs if not d["pass"]], "all_differences": diffs,
                                "pass": all(d["pass"] for d in diffs)})
    return {"comparisons": comparisons, "checks": sum(x["checks"] for x in comparisons),
            "passed": sum(x["passed"] for x in comparisons), "all_pass": all(x["pass"] for x in comparisons),
            "warning": "Fixed-order convergence qualification only; not a rigorous quadrature remainder bound."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", required=True)
    args = parser.parse_args()
    if sha(PLAN) != PLAN_SHA:
        raise RuntimeError("PLAN bytes changed")
    freeze = ROOT / "artifacts/regularization_learning/FREEZE.json"
    if not freeze.exists():
        raise RuntimeError("Wait for root's FREEZE and explicit run authorization")
    if (OUT / "RUN.json").exists():
        raise RuntimeError("Refuse to overwrite existing production RUN.json")
    cells = []
    for a in (.5, 1, 2, 4):
        for order in ORDERS:
            cells.append(run_cell(a, order))
            print(f"completed a={a} order={order}", flush=True)
    data = {"schema": 1, "plan_sha256": sha(PLAN), "freeze_sha256": sha(freeze),
            "source_sha256": sha(__file__), "settings": {"R": R, "orders": ORDERS, "main_order": 64,
                "six_teachers": SIX, "main_a": 1, "main_teacher_grid_points": 61,
                "probes_per_cell": 61, "random_draws": 0, "new_labels": 0, "banks": 0},
            "definitions": {"r": "E[tanh(a X/2)tanh(a Y/2)], corr(X,Y)=rho",
                "d": "(E[(a/2)sech^2(a Z/2)])^2 = r_prime(0)",
                "H": "integral_0^rho r(u) du", "raw": "-A*(r(theta)*rho+H(rho))",
                "quadratic": "raw+A*d*rho^2", "informed": "A*(H(rho)-r(theta)*rho)"},
            "root_method": "All roots on derivative-monotonicity intervals plus both endpoints; Mehler positivity supplies interval count. Numerical roots, not interval certificates.",
            "cells": cells,
            "accounting": {k: sum(c["accounting"][k] for c in cells) for k in cells[0]["accounting"]}}
    write(OUT / "RUN.json", data)
    write(OUT / "QUALIFICATION.json", qualify(cells))
    growth_orders = []
    for order in ORDERS:
        cell = next(c for c in cells if c["a"] == 1 and c["order"] == order)
        probes = cell["probes"][::3]
        teachers = []
        for theta in (0.0, .3, .6):
            row = next(row for row in cell["rows"] if abs(row["theta"]-theta) < 1e-12)
            teachers.append({"theta": theta, "r_theta": row["r_teacher"],
                             "risk_grid21": [observed_kl(row["r_teacher"], p["r"]) for p in probes]})
        growth_orders.append({"order": order, "a": 1, "R": R, "d_r0": cell["constants"]["d"],
            "A": cell["constants"]["A"], "gamma0": cell["constants"]["gamma0"],
            "var_logsigmoid": cell["constants"]["var_logsigmoid"],
            "rho_grid21": [p["rho"] for p in probes], "r_grid21": [p["r"] for p in probes],
            "H_grid21": [p["H"] for p in probes], "teachers": teachers})
    growth_main = dict(next(c for c in growth_orders if c["order"] == 64))
    growth_main.update({"source_run_sha256": sha(OUT / "RUN.json"), "all_orders": growth_orders,
                       "qualification": "See QUALIFICATION.json a=1 comparisons; all arrays reuse RUN.json integrals, zero additional integration calls."})
    write(OUT / "GROWTH_REFERENCE.json", growth_main)
    print(json.dumps(data["accounting"]))


if __name__ == "__main__":
    main()
