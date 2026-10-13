"""Eight independent row lengths: strict finite F_K gradient/Hessian blocks.

No file I/O, historical producer import, study input, optimizer, RNG, or NPZ.
Only --self-test is executable. A caller must freeze its own scientific inputs.
Requires the public python-flint package (tested with 0.8.0).
"""
from __future__ import annotations

from fractions import Fraction
import argparse
import hashlib
import json
import time

import flint
from flint import arb, arb_mat, ctx

M = 8
BLOCKS = ("Hbb", "Haa", "Hperp")
MOMENT_BASIS = "p_i_minus_one_half"
LABEL_COEFFICIENT_BASIS = "y_i_minus_one_half"
DEFAULT_CONFIG = dict(K=8, precision_bits=192, hz="1/40", L="11", dz="1/8",
                      hv="1/8", vmin="-32", vmax="64", dv="1/2", T="4",
                      saturation_threshold=128, label_mode="complement")


def exact(value):
    if isinstance(value, (float, bool)):
        raise ValueError("exact rational string/int/Fraction required")
    return Fraction(value)


def aq(value):
    value = exact(value)
    return arb(value.numerator) / value.denominator


def endpoint(value):
    mantissa, exponent = value.man_exp()
    return dict(mantissa=str(mantissa), exponent_of_two=int(exponent))


def endpoint_fraction(value):
    exponent = value["exponent_of_two"]
    if type(exponent) is not int:
        raise ValueError("dyadic exponent must be an actual integer")
    return Fraction(int(value["mantissa"])) * Fraction(2) ** exponent


def record(value):
    if not value.is_finite():
        raise ArithmeticError("nonfinite Arb enclosure")
    return dict(arb_display_only=str(value), lower_exact_dyadic=endpoint(value.lower()),
                upper_exact_dyadic=endpoint(value.upper()))


def as_arb(value):
    if isinstance(value, arb):
        if not value.is_finite():
            raise ValueError("finite Arb input required")
        return value
    if isinstance(value, dict):
        lo = endpoint_fraction(value["lower_exact_dyadic"])
        hi = endpoint_fraction(value["upper_exact_dyadic"])
        if lo > hi:
            raise ValueError("reversed interval endpoints")
        return aq(lo).union(aq(hi))
    return aq(value)


def input_audit(value):
    if isinstance(value, dict):
        # Preserve all supplied record metadata, not just a recreated midpoint.
        return json.loads(json.dumps(value, allow_nan=False))
    if isinstance(value, arb):
        return record(value)
    return dict(exact_rational=str(exact(value)))


def configuration(config=None):
    values = dict(DEFAULT_CONFIG)
    if config is not None:
        if set(config) - set(values):
            raise ValueError("unknown configuration field")
        values.update(config)
    if type(values["K"]) is not int or values["K"] < 2:
        raise ValueError("actual integer K>=2 required")
    if type(values["precision_bits"]) is not int or values["precision_bits"] < 64:
        raise ValueError("actual integer precision_bits>=64 required")
    for key in ("hz", "L", "dz", "hv", "vmin", "vmax", "dv", "T"):
        values[key] = str(exact(values[key]))
    q = {key: exact(values[key]) for key in ("hz", "L", "dz", "hv", "vmin", "vmax", "dv", "T")}
    if any(q[key] <= 0 for key in ("hz", "L", "dz", "hv", "dv", "T")):
        raise ValueError("positive steps, strip widths, L and T required")
    if q["vmin"] >= q["vmax"]:
        raise ValueError("vmin<vmax required")
    if any(value.denominator != 1 for value in
           (q["L"] / q["hz"], q["vmin"] / q["hv"], q["vmax"] / q["hv"])):
        raise ValueError("exact integer grid endpoints required")
    if values["saturation_threshold"] is not None and (type(values["saturation_threshold"]) is not int or values["saturation_threshold"] != 128):
        raise ValueError("saturation_threshold must be None or 128")
    if values["label_mode"] not in ("complement", "all_labels"):
        raise ValueError("label_mode must be complement or all_labels")
    return values


def lengths(a_values):
    if len(a_values) != M:
        raise ValueError("exactly eight independent row lengths required")
    values = [as_arb(value) for value in a_values]
    if any(not value > 0 or not value <= 1 for value in values):
        raise ValueError("each entire row-length enclosure must lie in (0,1]")
    return values


def label_bits(label):
    if type(label) is not int or not 0 <= label < 256:
        raise ValueError("actual integer label in [0,255] required")
    return tuple((label >> (7-i)) & 1 for i in range(M))


def matrix_records(matrix):
    return [[record(value) for value in row] for row in matrix]


def zero_matrix():
    return [[arb(0) for _ in range(M)] for _ in range(M)]


def precompute(a_records_or_rationals, config=None):
    """Build one Gaussian/99-column moment cache; no label or target evaluation.

    Exact rational points, native Arb intervals and dyadic endpoint records are
    accepted. The cached objects retain input interval dependency inflation.
    """
    cfg = configuration(config)
    previous = ctx.prec
    ctx.prec = cfg["precision_bits"]
    try:
        a = lengths(a_records_or_rationals)
        audit = [input_audit(value) for value in a_records_or_rationals]
        h, L = exact(cfg["hz"]), exact(cfg["L"])
        n = int(L/h)
        z = [aq(h)*j for j in range(-n, n+1)]
        w = [aq(h)*(-x*x/2).exp()/(2*arb.pi()).sqrt() for x in z]
        probabilities = [[(1/(1+(ai*x).exp()), 1/(1+(-ai*x).exp())) for ai in a] for x in z]
        p = [[pair[1] for pair in row] for row in probabilities]
        columns = []
        for x, weight, ps in zip(z, w, p):
            centered = [pi-arb(1)/2 for pi in ps]
            row = [weight, weight*x, weight*x*x]
            for power in (0, 1, 2):
                row.extend(weight*x**power*pi for pi in centered)
            for power in (0, 2):
                row.extend(weight*x**power*centered[i]*centered[j] for i in range(M) for j in range(i, M))
            columns.append(row)
        signature = hashlib.sha256(json.dumps(dict(a=audit, config=cfg, moment_basis=MOMENT_BASIS,
                                                   label_coefficient_basis=LABEL_COEFFICIENT_BASIS), sort_keys=True,
                                               separators=(",", ":")).encode()).hexdigest()
        return dict(config=cfg, a=a, a_input_audit=audit, a_enclosures=[record(value) for value in a],
                    signature=signature, z=z, w=w, probabilities=probabilities, p=p,
                    moment_matrix=arb_mat(columns), moment_columns=99,
                    moment_basis=MOMENT_BASIS, label_coefficient_basis=LABEL_COEFFICIENT_BASIS,
                    scalar_moments_including_Psi=100, normal_mass=sum(w, arb(0)),
                    normal_nodes=len(z), precompute_logistic_exponentials=2*M*len(z),
                    precompute_gaussian_exponentials=len(z))
    finally:
        ctx.prec = previous


def unpack_moments(values):
    if len(values) != 99:
        raise ValueError("99 moment columns required")
    T = values[:3]
    P = [values[3+M*k:3+M*(k+1)] for k in range(3)]
    offset, PP = 27, {}
    for power in (0, 2):
        matrix = zero_matrix()
        for i in range(M):
            for j in range(i, M):
                matrix[i][j] = matrix[j][i] = values[offset]
                offset += 1
        PP[power] = matrix
    return T, P, PP


def conditional_integrand(label, psi, values, K):
    """Native g_a and three matrices, after dv change of variables.

    P/PP are moments of p-1/2; labels use y-1/2. This exactly preserves
    each mathematical finite sum. Hbb uses Cov(y-p)=Cov(p-1/2). Haa
    retains all label products. Hperp has no cross-particle mean term.
    """
    y = tuple(arb(bit)-arb(1)/2 for bit in label_bits(label))
    T, P, PP = unpack_moments(values)
    aK, bK = K*psi**(K-1), K*(K-1)*psi**(K-2)
    gradient = [-aK*(T[1]*y[i]-P[1][i]) for i in range(M)]
    matrices = {name: zero_matrix() for name in BLOCKS}
    for i in range(M):
        for j in range(i, M):
            gamma0 = T[0]*y[i]*y[j]-y[i]*P[0][j]-P[0][i]*y[j]+PP[0][i][j]
            gamma2 = T[2]*y[i]*y[j]-y[i]*P[2][j]-P[2][i]*y[j]+PP[2][i][j]
            mean1i, mean1j = T[1]*y[i]-P[1][i], T[1]*y[j]-P[1][j]
            jbb = T[0]*PP[0][i][j]-P[0][i]*P[0][j]
            jaa = T[0]*gamma2-mean1i*mean1j
            jp = T[0]*gamma0
            d0 = T[0]/4-PP[0][i][i] if i == j else arb(0)
            d2 = T[2]/4-PP[2][i][i] if i == j else arb(0)
            for name, value in (("Hbb", aK*d0-bK*jbb), ("Haa", aK*d2-bK*jaa),
                                ("Hperp", aK*d0-bK*jp)):
                matrices[name][i][j] = matrices[name][j][i] = value
    return gradient, matrices


class CandidateBudgetStop(RuntimeError):
    def __init__(self, partial):
        super().__init__("cooperative wall budget exhausted; partial result retained; no retry")
        self.partial = partial


def finite_label_block(cache, indices, *, hard_seconds=None, deadline=None):
    """Compute the requested labels, returning all native conditional endpoints.

    Deadline is an absolute time.perf_counter timestamp. Timeout raises
    CandidateBudgetStop whose partial is a serializable block: completed labels
    plus one incomplete outer prefix. Incomplete prefixes cannot be aggregated.
    No producer file I/O or automatic retry/resume is performed.
    """
    labels = list(indices)
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("nonempty distinct label indices required")
    cfg = cache["config"]
    for label in labels:
        label_bits(label)
        if cfg["label_mode"] == "complement" and label >= 128:
            raise ValueError("complement representative indices must be 0..127")
    if hard_seconds is not None and (isinstance(hard_seconds, bool) or not hard_seconds > 0):
        raise ValueError("positive hard_seconds required")
    tick = time.perf_counter()
    if hard_seconds is not None:
        deadline = min(deadline, tick+hard_seconds) if deadline is not None else tick+hard_seconds
    previous = ctx.prec
    ctx.prec = cfg["precision_bits"]
    rows = []
    try:
        h = aq(cfg["hv"])
        outer = range(int(exact(cfg["vmin"])/exact(cfg["hv"])), int(exact(cfg["vmax"])/exact(cfg["hv"]))+1)
        outer_u = [(h*index).exp() for index in outer]
        threshold = cfg["saturation_threshold"]
        if threshold is not None:
            E, S = arb(2)**-184, arb(2)**-177
            if not (-arb(128)).exp() < E or not 128*(-arb(128)).exp() < S:
                raise ArithmeticError("saturation upper caps not proved")
            ecap, tcap = arb(E/2, E/2), arb(S/2, S/2)
        block_factors = 0

        def output(complete, partial=None):
            return dict(schema="r03_candidate_finite_label_block_v1", status="completed" if complete else "budget_stopped",
                        model_signature=cache["signature"], config=cfg, a_input_audit=cache["a_input_audit"],
                        a_enclosures=cache["a_enclosures"], labels_requested=labels, conditional_rows=rows,
                        partial_conditional_row=partial, raw_normal_grid_mass=record(cache["normal_mass"]),
                        moment_columns=99, scalar_moments_including_Psi=100,
                        moment_basis=MOMENT_BASIS, label_coefficient_basis=LABEL_COEFFICIENT_BASIS,
                        counts=dict(completed_labels=len(rows), laplace_factors_evaluated=block_factors,
                                    inner_nodes=cache["normal_nodes"], outer_nodes_per_label=len(outer),
                                    outer_u_exponential_calls=len(outer), new_random_draws=0,
                                    new_optimizer_updates=0, old_npz_reads=0),
                        label_bits_order="big endian: y_i=(label>>(7-i))&1",
                        complement_identity="A_(255-y)(z)=A_y(-z); g_a and Hbb/Haa/Hperp invariant",
                        paired_row_assumption_used=False, Gaussian_or_teacher_mass_renormalized=False,
                        finite_sum_arithmetic_enclosed=True, analytic_errors_added=False,
                        true_stationary_point_certified=False, automatic_retry=False,
                        wall_seconds=time.perf_counter()-tick)

        for label in labels:
            y = label_bits(label)
            g = [arb(0) for _ in range(M)]
            matrices = {name: zero_matrix() for name in BLOCKS}
            completed = factors = saturated = current_inner = 0

            def row_output(complete):
                return dict(label_index=label, label_bits=list(y), status="completed" if complete else "budget_stopped",
                            conditional_gradient_finite=[record(value) for value in g],
                            matrix_blocks={name: matrix_records(matrix) for name, matrix in matrices.items()},
                            outer_nodes_completed=completed, last_completed_outer_index=None if not completed else outer[completed-1],
                            partial_current_inner_factors=current_inner,
                            counts=dict(laplace_factors_evaluated=factors, saturated_factor_enclosures=saturated,
                                        ordinary_exponential_calls=factors-saturated))

            likelihoods = []
            for probs in cache["probabilities"]:
                likelihood = arb(1)
                for bit, pair in zip(y, probs):
                    likelihood *= pair[bit]
                if not likelihood > 0:
                    raise ArithmeticError("likelihood enclosure failed strict positivity")
                likelihoods.append(likelihood)
            for u in outer_u:
                if deadline is not None and time.perf_counter() >= deadline:
                    raise CandidateBudgetStop(output(False, row_output(False)))
                psi, coefficients, current_inner = arb(0), [], 0
                for j, (likelihood, w) in enumerate(zip(likelihoods, cache["w"])):
                    if deadline is not None and j % 32 == 0 and time.perf_counter() >= deadline:
                        raise CandidateBudgetStop(output(False, row_output(False)))
                    t = u*likelihood
                    if threshold is not None and t.lower() >= threshold:
                        exp_value, texp_value = ecap, tcap
                        saturated += 1
                    else:
                        exp_value = (-t).exp()
                        texp_value = t*exp_value
                    psi += w*exp_value
                    coefficients.append(texp_value)
                    factors += 1
                    block_factors += 1
                    current_inner += 1
                dot = arb_mat([coefficients])*cache["moment_matrix"]
                dg, dh = conditional_integrand(label, psi, [dot[0,j] for j in range(99)], cfg["K"])
                for i in range(M):
                    g[i] += h*dg[i]
                    for j in range(i, M):
                        for name in BLOCKS:
                            matrices[name][i][j] += h*dh[name][i][j]
                            matrices[name][j][i] = matrices[name][i][j]
                completed += 1
                current_inner = 0
            rows.append(row_output(True))
        return output(True)
    finally:
        ctx.prec = previous


def aggregate(blocks, q_values, *, teacher_complement_symmetry):
    """Aggregate complete conditional blocks against all original 256 weights.

    teacher_complement_symmetry is a nonempty caller-owned analytic statement,
    not a numeric overlap test. It is needed to assert the omitted exact mixed
    b/a and bias-gradient terms vanish; no pair-exchange symmetry is asserted.
    Saved finite q values need not have bit-identical complement enclosures.
    """
    if not isinstance(teacher_complement_symmetry, str) or not teacher_complement_symmetry.strip():
        raise ValueError("explicit analytic teacher-complement declaration required")
    if len(q_values) != 256 or not blocks:
        raise ValueError("all 256 teacher weights and nonempty blocks required")
    cfg = blocks[0]["config"]
    previous = ctx.prec
    ctx.prec = cfg["precision_bits"]
    try:
        weights = [as_arb(value) for value in q_values]
        if any(not value >= 0 for value in weights):
            raise ValueError("nonnegative original weight enclosures required")
        expected = list(range(128 if cfg["label_mode"] == "complement" else 256))
        rows = {}
        for block in blocks:
            if block["status"] != "completed" or block["partial_conditional_row"] is not None:
                raise ValueError("partial label block cannot certify an aggregate")
            if block["model_signature"] != blocks[0]["model_signature"] or block["config"] != cfg:
                raise ValueError("candidate/configuration mismatch")
            if block.get("moment_basis") != MOMENT_BASIS or block.get("label_coefficient_basis") != LABEL_COEFFICIENT_BASIS:
                raise ValueError("centered moment/label basis mismatch")
            for row in block["conditional_rows"]:
                index = row["label_index"]
                if index in rows or row["status"] != "completed":
                    raise ValueError("duplicate or incomplete conditional label")
                prescribed = int((exact(cfg["vmax"])-exact(cfg["vmin"]))/exact(cfg["hv"]))+1
                if row["outer_nodes_completed"] != prescribed or row["partial_current_inner_factors"] != 0:
                    raise ValueError("incomplete outer-grid coverage")
                if row["label_bits"] != list(label_bits(index)) or len(row["conditional_gradient_finite"]) != M:
                    raise ValueError("label or gradient shape mismatch")
                if any(len(row["matrix_blocks"][name]) != M or any(len(r) != M for r in row["matrix_blocks"][name]) for name in BLOCKS):
                    raise ValueError("Hessian block shape mismatch")
                rows[index] = row
        if sorted(rows) != expected:
            raise ValueError("exact complete label coverage required")
        g = [arb(0) for _ in range(M)]
        matrices = {name: zero_matrix() for name in BLOCKS}
        folded = []
        for label in expected:
            q = weights[label]+weights[255-label] if cfg["label_mode"] == "complement" else weights[label]
            folded.append(record(q))
            row = rows[label]
            for i in range(M):
                g[i] += q*as_arb(row["conditional_gradient_finite"][i])
                for j in range(M):
                    for name in BLOCKS:
                        matrices[name][i][j] += q*as_arb(row["matrix_blocks"][name][i][j])
        return dict(schema="r03_candidate_finite_aggregate_v1", model_signature=blocks[0]["model_signature"],
                    config=cfg, a_input_audit=blocks[0]["a_input_audit"],
                    moment_basis=MOMENT_BASIS, label_coefficient_basis=LABEL_COEFFICIENT_BASIS,
                    original_q_input_audit=[input_audit(value) for value in q_values],
                    original_q_256=[record(value) for value in weights], q_mass=record(sum(weights,arb(0))),
                    folded_q=folded, conditional_labels=expected,
                    teacher_complement_symmetry_analytic_declaration=teacher_complement_symmetry,
                    teacher_symmetry_proved_by_numeric_overlap=False,
                    gradient_finite=[record(value) for value in g],
                    matrix_blocks={name:matrix_records(matrix) for name,matrix in matrices.items()},
                    teacher_or_Gaussian_mass_renormalized=False, paired_row_assumption_used=False,
                    conditional_rows_retained_in_input_blocks=True, analytic_errors_added=False,
                    full_true_gradient_or_Hessian_certified=False)
    finally:
        ctx.prec = previous


def analytic_error_bounds(a_records_or_rationals, config=None, *,
                          teacher_error_L1_upper, teacher_mass_upper=None):
    """Uniform analytic budgets, evaluated only as scalar Arb expressions.

    Covers every exact a in the supplied box. Uses upper endpoints in monotone
    sector constants. The finite-sum interval already encloses arithmetic and
    dependence inflation; these budgets do not replace it. Teacher errors refer
    to the exact finite coefficients enclosed by the caller's q input balls.
    """
    cfg = configuration(config)
    te = exact(teacher_error_L1_upper)
    tm = 1+te if teacher_mass_upper is None else exact(teacher_mass_upper)
    if te < 0 or tm < 0:
        raise ValueError("nonnegative teacher error/mass bounds required")
    previous = ctx.prec
    ctx.prec = cfg["precision_bits"]
    try:
        aa = lengths(a_records_or_rationals)
        uppers = [endpoint_fraction(endpoint(value.upper())) for value in aa]
        a = [aq(value) for value in uppers]
        x = {key:aq(cfg[key]) for key in ("hz","L","dz","hv","vmin","vmax","dv","T")}
        if exact(cfg["L"]) < 2:
            raise ValueError("analytic Hessian tail proof requires L>=2")
        K, m = cfg["K"], M
        e, quarter = arb(1).exp(), arb(1)/4
        asum = sum(a,arb(0))
        theta = asum*x["dz"]
        if not theta < arb.pi()/2 or not x["dv"] < arb.pi()/2:
            raise ValueError("inner likelihood sector / outer right-half-plane strip not established")
        cz, cv = theta.cos(), x["dv"].cos()
        if not cz > 0 or not cv > 0:
            raise ValueError("positive cosines required")
        gauss = (x["dz"]**2/2).exp()
        cr2 = sum((1/(ai*x["dz"]/2).cos()**2 for ai in a),arb(0))
        secmax2 = 1/(aq(max(uppers))*x["dz"]/2).cos()**2
        qz = (-2*arb.pi()*x["dz"]/x["hz"]).exp()
        qv = (-2*arb.pi()*x["dv"]/x["hv"]).exp()
        phi = lambda z: (-z*z/2).exp()/(2*arb.pi()).sqrt()
        pL = 2*phi(x["L"])/x["L"]
        Wv = x["vmax"]-x["vmin"]+x["hv"]
        eta = (-x["T"]*asum).exp()/(arb(2)**m)
        Ueta = x["vmax"].exp()*eta
        if not Ueta > 2:
            raise ValueError("U*eta>2 needed for gradient and both Hessian good-event tails")
        pT = 2*phi(x["T"])/x["T"]
        CG = arb(m).sqrt()
        eg = CG*(2/arb.pi()).sqrt()
        MD = gauss*(cr2*(1+x["dz"]**2)).sqrt()/(e*cz)
        dpsi = 2*gauss*qz/(1-qz)+pL
        dD = 2*MD*qz/(1-qz)+2*CG*phi(x["L"])/e
        Dmax = eg/e
        gradient_inner = K*Wv*((1+dpsi)**(K-1)*dD+(K-1)*(1+dpsi)**(K-2)*dpsi*Dmax)
        gradient_alias = 2*K*eg/cv*qv/(1-qv)
        low1 = x["hv"]*x["vmin"].exp()/x["hv"].expm1()
        low2 = x["hv"]*(2*x["vmin"]).exp()/(2*x["hv"]).expm1()
        gradient_low = K*eg*low1
        gradient_good = CG*arb(K).sqrt()*(-Ueta).exp()
        gradient_bad = CG*(1+2*x["hv"]/e)*2*K*phi(x["T"])*pT**(K-1)
        gradient_uniform = gradient_inner+gradient_alias+gradient_low+gradient_good+gradient_bad
        moment = 2+x["dz"]**2
        base = gauss/(e*cz)
        boundary = dict(Psi=gauss,T0=base,R=base*(cr2*moment).sqrt(),Q=base*cr2*moment,
                        B=base*secmax2*moment/4,Qp=base*cr2,Bp=base*secmax2/4)
        real = dict(T0=1/e,R=arb(2*m).sqrt()/e,Q=2*m/e,B=arb(1)/2/e,Qp=m/e,Bp=quarter/e)
        momtail = 2*phi(x["L"])*(x["L"]+2/x["L"])
        tails = dict(Psi=pL,T0=pL/e,R=2*arb(2*m).sqrt()*phi(x["L"])/e,
                     Q=m*momtail/e,B=momtail/4/e,Qp=m*pL/e,Bp=pL/4/e)
        aliases = {key:2*value*qz/(1-qz) for key,value in boundary.items()}
        delta = {key:aliases[key]+tails[key] for key in boundary}
        def power_product(n,b,db):
            perturb = arb(0) if n == 0 else n*(1+delta["Psi"])**(n-1)*delta["Psi"]*b
            return (1+delta["Psi"])**n*db+perturb
        Cmax = real["T0"]*real["Q"]+real["R"]**2
        dC = delta["T0"]*real["Q"]+(real["T0"]+delta["T0"])*delta["Q"]+2*real["R"]*delta["R"]+delta["R"]**2
        Cpmax = real["T0"]*real["Qp"]
        dCp = delta["T0"]*real["Qp"]+(real["T0"]+delta["T0"])*delta["Qp"]
        inner_ba = Wv*(K*power_product(K-1,real["B"],delta["B"])+K*(K-1)*power_product(K-2,Cmax,dC))
        inner_p = Wv*(K*power_product(K-1,real["Bp"],delta["Bp"])+K*(K-1)*power_product(K-2,Cpmax,dCp))
        Mvp = quarter/cv+m/cv**2
        alias_ba, alias_p = 2*(1+K)*Mvp*qv/(1-qv), 2*Mvp*qv/(1-qv)
        lower_p = K*low1/4+m*K*K*low2
        lower_ba = (1+K)*lower_p
        t2T = 2*phi(x["T"])*(x["T"]+1/x["T"])
        bad_coefficient = quarter*(1+2*x["hv"]/e)+m*(1+8*x["hv"]/e**2)
        good_p = (quarter+m*(1+Ueta))*(-Ueta).exp()
        good_ba = (1+K)*good_p
        bad_ba = bad_coefficient*(pT**K+K*t2T*pT**(K-1))
        bad_p = bad_coefficient*pT**K
        uniform_ba = inner_ba+alias_ba+lower_ba+good_ba+bad_ba
        uniform_p = inner_p+alias_p+lower_p+good_p+bad_p
        global_g, global_ba, global_p = arb(m*K).sqrt(),(2*m+quarter)*(1+K),m+quarter
        terms = dict(inner_gradient_L2=gradient_inner,outer_alias_gradient_L2=gradient_alias,
                     outer_lower_gradient_L2=gradient_low,outer_upper_good_gradient_L2=gradient_good,
                     outer_upper_bad_gradient_L2=gradient_bad,uniform_gradient_L2=gradient_uniform,
                     inner_Hba_operator=inner_ba,inner_Hperp_operator=inner_p,
                     outer_alias_Hba_operator=alias_ba,outer_alias_Hperp_operator=alias_p,
                     outer_lower_Hba_operator=lower_ba,outer_lower_Hperp_operator=lower_p,
                     outer_upper_good_Hba_operator=good_ba,outer_upper_good_Hperp_operator=good_p,
                     outer_upper_bad_Hba_operator=bad_ba,outer_upper_bad_Hperp_operator=bad_p,
                     uniform_Hba_operator=uniform_ba,uniform_Hperp_operator=uniform_p,
                     teacher_global_gradient_L2=global_g,teacher_global_Hba_operator=global_ba,
                     teacher_global_Hperp_operator=global_p,teacher_transfer_gradient_L2=aq(te)*global_g,
                     teacher_transfer_Hba_operator=aq(te)*global_ba,teacher_transfer_Hperp_operator=aq(te)*global_p,
                     total_gradient_L2=aq(tm)*gradient_uniform+aq(te)*global_g,
                     total_Hba_operator=aq(tm)*uniform_ba+aq(te)*global_ba,
                     total_Hperp_operator=aq(tm)*uniform_p+aq(te)*global_p)
        return dict(schema="r03_candidate_analytic_bounds_v1",config=cfg,m=8,Hba_dimension=16,Hperp_dimension=8,
                    a_upper_exact=[str(value) for value in uppers],full_l1_asum=record(asum),
                    inner_CR_squared=record(cr2),inner_phase_theta=record(theta),upper_U_eta=record(Ueta),
                    teacher_error_L1_upper=str(te),teacher_mass_upper=str(tm),
                    bounds={key:record(value) for key,value in terms.items()},
                    inner_moments={key:dict(boundary=record(boundary[key]),alias=record(aliases[key]),
                                           grid_tail=record(tails[key]),total_error=record(delta[key])) for key in boundary},
                    gradient_inner_moments=dict(Psi_boundary=record(gauss),D_boundary_L2=record(MD),
                                                Psi_error=record(dpsi),D_error_L2=record(dD)),
                    uniform_all_labels_and_input_box=True,paired_dimension_doubling_used=False,
                    compute_arithmetic_included=False,interval_dependency_inflation_already_in_finite_balls=True,
                    Hessian_errors_are_operator_not_entrywise=True,
                    teacher_decomposition="qfinite*(true-grid)+(ptrue-qfinite)*true",
                    scientific_grid_evaluations=0,random_draws=0,old_npz_reads=0)
    finally:
        ctx.prec = previous


def _direct_manufactured(cache, label):
    """Independent finite same/cross expression; self-test small grids only."""
    y, K = label_bits(label), cache["config"]["K"]
    h = aq(cache["config"]["hv"])
    lo,hi = (int(exact(cache["config"][key])/exact(cache["config"]["hv"])) for key in ("vmin","vmax"))
    g = [arb(0) for _ in range(M)]
    matrices = {name:zero_matrix() for name in BLOCKS}
    for index in range(lo,hi+1):
        u = (h*index).exp()
        atoms = []
        psi = arb(0)
        for z,w,ps in zip(cache["z"],cache["w"],cache["p"]):
            A = arb(1)
            for bit,p in zip(y,ps):
                A *= p if bit else 1-p
            exp_value = (-u*A).exp()
            psi += w*exp_value
            atoms.append((w*u*A*exp_value,z,[bit-p for bit,p in zip(y,ps)],[p*(1-p) for p in ps]))
        T0 = sum((row[0] for row in atoms),arb(0))
        mean_b = [sum((t*r[i] for t,z,r,s in atoms),arb(0)) for i in range(M)]
        mean_a = [sum((t*z*r[i] for t,z,r,s in atoms),arb(0)) for i in range(M)]
        for i in range(M):
            g[i] -= h*K*psi**(K-1)*mean_a[i]
            for j in range(M):
                rr = sum((t*r[i]*r[j] for t,z,r,s in atoms),arb(0))
                zzrr = sum((t*z*z*r[i]*r[j] for t,z,r,s in atoms),arb(0))
                d0 = sum((t*s[i] for t,z,r,s in atoms),arb(0)) if i==j else arb(0)
                d2 = sum((t*z*z*s[i] for t,z,r,s in atoms),arb(0)) if i==j else arb(0)
                fac1,fac2 = h*K*psi**(K-1),h*K*(K-1)*psi**(K-2)
                matrices["Hbb"][i][j] += fac1*d0-fac2*(T0*rr-mean_b[i]*mean_b[j])
                matrices["Haa"][i][j] += fac1*d2-fac2*(T0*zzrr-mean_a[i]*mean_a[j])
                matrices["Hperp"][i][j] += fac1*d0-fac2*T0*rr
    return g,matrices


def self_test():
    """Artificial 3x3 grids; no stored teacher/candidate or old data read."""
    previous = ctx.prec
    ctx.prec = 192
    try:
        a = [Fraction(i,16) for i in range(1,9)]
        cfg = dict(hz="1",L="1",hv="1",vmin="-1",vmax="1",K=3,
                   label_mode="all_labels",saturation_threshold=None)
        cache = precompute(a,cfg)
        assert cache["moment_basis"]==MOMENT_BASIS and cache["label_coefficient_basis"]==LABEL_COEFFICIENT_BASIS
        chosen = [0,37,128,218,255]
        finite = finite_label_block(cache,chosen,hard_seconds=20)
        comparisons = 0
        for row in finite["conditional_rows"]:
            direct_g,direct_h = _direct_manufactured(cache,row["label_index"])
            for i in range(M):
                assert as_arb(row["conditional_gradient_finite"][i]).overlaps(direct_g[i])
                comparisons += 1
                for j in range(M):
                    for name in BLOCKS:
                        assert as_arb(row["matrix_blocks"][name][i][j]).overlaps(direct_h[name][i][j])
                        comparisons += 1
        by_label = {row["label_index"]:row for row in finite["conditional_rows"]}
        for left,right in ((0,255),(37,218)):
            for i in range(M):
                assert as_arb(by_label[left]["conditional_gradient_finite"][i]).overlaps(as_arb(by_label[right]["conditional_gradient_finite"][i]))
                for j in range(M):
                    for name in BLOCKS:
                        assert as_arb(by_label[left]["matrix_blocks"][name][i][j]).overlaps(as_arb(by_label[right]["matrix_blocks"][name][i][j]))
        intervals = [aq(value).union(aq(value+Fraction(1,10000))) for value in a]
        iblock = finite_label_block(precompute(intervals,cfg),[37],hard_seconds=20)
        exact_row = by_label[37]
        for i in range(M):
            assert as_arb(iblock["conditional_rows"][0]["conditional_gradient_finite"][i]).contains(as_arb(exact_row["conditional_gradient_finite"][i]))
            for j in range(M):
                for name in BLOCKS:
                    assert as_arb(iblock["conditional_rows"][0]["matrix_blocks"][name][i][j]).contains(as_arb(exact_row["matrix_blocks"][name][i][j]))
        # Complete artificial 128-label coverage; independent reference uses
        # the four nonzero original teacher weights and reflection identity.
        complement_cfg = dict(cfg,label_mode="complement")
        complete = finite_label_block(precompute(a,complement_cfg),range(128),hard_seconds=20)
        q = [Fraction(0) for _ in range(256)]
        q[0]=q[255]=Fraction(1,8)
        q[37]=q[218]=Fraction(3,8)
        aggregated = aggregate([complete],q,teacher_complement_symmetry="Manufactured exact four-atom weights q_y=q_(255-y).")
        d0g,d0h = _direct_manufactured(cache,0)
        d1g,d1h = _direct_manufactured(cache,37)
        for i in range(M):
            assert as_arb(aggregated["gradient_finite"][i]).overlaps(d0g[i]/4+3*d1g[i]/4)
            comparisons += 1
            for j in range(M):
                for name in BLOCKS:
                    assert as_arb(aggregated["matrix_blocks"][name][i][j]).overlaps(d0h[name][i][j]/4+3*d1h[name][i][j]/4)
                    comparisons += 1
        assert len(aggregated["original_q_256"])==256 and len(aggregated["folded_q"])==128
        assert aggregated["moment_basis"]==MOMENT_BASIS and aggregated["label_coefficient_basis"]==LABEL_COEFFICIENT_BASIS
        # A distinct rational box supplies scalar analytic constants only.
        budget = analytic_error_bounds(intervals,teacher_error_L1_upper="1/1000000000000")
        expected_asum = sum((value.upper() for value in intervals),arb(0))
        assert budget["m"]==8 and as_arb(budget["full_l1_asum"]).overlaps(expected_asum)
        assert as_arb(budget["bounds"]["teacher_global_Hba_operator"]).contains(aq("585/4"))
        stopped = False
        try:
            finite_label_block(cache,[0],deadline=time.perf_counter()-1)
        except CandidateBudgetStop as error:
            stopped = error.partial["status"]=="budget_stopped" and error.partial["counts"]["completed_labels"]==0
        assert stopped
        rejected = 0
        for bad in (["1/2"]*7,["0"]*8,["1001/1000"]*8):
            try:
                precompute(bad,cfg)
            except ValueError:
                rejected += 1
        assert rejected==3
        return dict(status="passed",manufactured_labels=134,manufactured_grid="3 Gaussian x 3 Laplace nodes",
                    manufactured_laplace_factors=1206,independent_direct_comparisons=comparisons,
                    independent_direct_reference_labels=7,independent_direct_reference_factors=63,
                    complement_reflection_pairs=2,interval_box_contains_point_gradient=True,
                    interval_box_contains_point_Hessian_blocks=True,full_128_label_aggregation_checked=True,
                    centered_moments_against_unchanged_uncentered_direct_reference=True,
                    scalar_budget_m=8,timeout_partial_checked=True,rejected_models=rejected,
                    scientific_target_evaluations=0,old_npz_reads=0,random_draws=0,
                    python_flint_version=flint.__version__)
    finally:
        ctx.prec = previous


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true",required=True)
    parser.parse_args()
    print(json.dumps(self_test(),ensure_ascii=False,indent=2))
