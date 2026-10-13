"""R03 continuous-box certificate from saved interval quantities only.

No file I/O, finite-grid evaluation, optimization, RNG, or historical module
imports. The caller supplies source-bound inputs. Command-line execution runs
manufactured algebra fixtures only. python-flint is the public dependency.
"""
from __future__ import annotations

from fractions import Fraction
import copy
import hashlib
import json

from flint import arb, arb_mat, ctx


M = 8
BLOCKS = ("Hbb", "Haa", "Hperp")
PHI = ["-7/10", "-1/2", "-3/10", "-1/10", "1/10", "3/10", "1/2", "7/10"]
FIXED_U = [-8440215, -6758298, -4585241, -1673686, 1673686, 4585241, 6758298, 8440215]
FIXED_S = 281474961915412


def exact(value):
    if isinstance(value, (float, bool)):
        raise ValueError("explicit exact rational input required")
    if isinstance(value, dict) and "numerator" in value:
        return Fraction(int(value["numerator"]), int(value["denominator"]))
    return Fraction(value)


def aq(value):
    value = exact(value)
    return arb(value.numerator) / value.denominator


def endpoint_fraction(value):
    if type(value["exponent_of_two"]) is not int:
        raise ValueError("actual integer dyadic exponent required")
    return Fraction(int(value["mantissa"])) * Fraction(2) ** value["exponent_of_two"]


def endpoint(value):
    m, e = value.man_exp()
    return dict(mantissa=str(m), exponent_of_two=int(e))


def record(value):
    if not value.is_finite():
        raise ArithmeticError("nonfinite certificate enclosure")
    return dict(arb_display_only=str(value), lower_exact_dyadic=endpoint(value.lower()),
                upper_exact_dyadic=endpoint(value.upper()))


def interval(value):
    if isinstance(value, arb):
        if not value.is_finite():
            raise ValueError("finite Arb input required")
        return value
    if isinstance(value, dict) and "lower_exact_dyadic" in value:
        lo, hi = [endpoint_fraction(value[k]) for k in ("lower_exact_dyadic", "upper_exact_dyadic")]
        if lo > hi:
            raise ValueError("reversed endpoints")
        return aq(lo).union(aq(hi))
    return aq(value)


def _positive_upper(value, name):
    value = interval(value)
    if not value >= 0:
        raise ValueError(name+" must be a nonnegative enclosure")
    return value.upper()


def _symball(radius):
    radius = radius.upper()
    return (-radius).union(radius)


def _matrix(values, rows=M, cols=M):
    if len(values) != rows or any(len(row) != cols for row in values):
        raise ValueError("matrix shape mismatch")
    return [[interval(value) for value in row] for row in values]


def _zeros(n=M, m=M):
    return [[arb(0) for _ in range(m)] for _ in range(n)]


def _matmul(A, B):
    return [[sum((A[i][k]*B[k][j] for k in range(len(B))), arb(0))
             for j in range(len(B[0]))] for i in range(len(A))]


def _matvec(A, x):
    return [sum((ai*xi for ai, xi in zip(row, x)), arb(0)) for row in A]


def _transpose(A):
    return [list(row) for row in zip(*A)]


def _records(A):
    return [[record(x) for x in row] for row in A]


def _symmetrize_enclosures(A):
    """Intersect duplicated symmetric-entry enclosures, never average a bound."""
    result = [list(row) for row in A]
    for i in range(len(A)):
        for j in range(i):
            if not A[i][j].overlaps(A[j][i]):
                raise ValueError("inconsistent enclosures for a symmetric matrix")
            result[i][j] = result[j][i] = A[i][j].intersection(A[j][i])
    return result


def ldl_positive(matrix, shift=arb(0)):
    """Outward interval LDL; failure is unresolved, not a negative-eigenvalue claim."""
    A = _symmetrize_enclosures(matrix)
    n = len(A)
    L, D = _zeros(n, n), []
    for i in range(n):
        L[i][i] = arb(1)
        pivot = A[i][i]-shift-sum((L[i][k]*L[i][k]*D[k] for k in range(i)), arb(0))
        D.append(pivot)
        if not pivot > 0:
            return dict(passed=False, status="positive_lower_bound_unresolved", failed_pivot=i,
                        pivots=[record(x) for x in D], shift=record(shift))
        for j in range(i+1, n):
            L[j][i] = (A[j][i]-sum((L[j][k]*L[i][k]*D[k] for k in range(i)), arb(0)))/pivot
    return dict(passed=True, status="strictly_positive", failed_pivot=None,
                pivots=[record(x) for x in D], L=_records(L), shift=record(shift))


def householder(a0):
    norm = sum((x*x for x in a0), arb(0)).sqrt()
    u = [x/norm for x in a0]
    v = list(u)
    v[0] += 1  # All a0 coordinates are strictly positive.
    denominator = sum((x*x for x in v), arb(0))
    return [[arb(int(i == j))-2*v[i]*v[j]/denominator for j in range(1, M)] for i in range(M)]


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _range_center(values):
    """One fixed exact center for a collection of saved conditional intervals."""
    lo = min(endpoint_fraction(endpoint(value.lower())) for value in values)
    hi = max(endpoint_fraction(endpoint(value.upper())) for value in values)
    centre = (lo+hi)/2
    return centre, aq(centre)


def _gradient_transfer_bounds(A, gradients, epsg, rownorm):
    """Zero-mass transfer caps; conditional finite moments stay unchanged."""
    transformed = [_matvec(A,g) for g in gradients]
    selected, row_records = [], []
    for i in range(M):
        centre_exact, centre = _range_center([row[i] for row in transformed])
        finite = max(abs(row[i]-centre).upper() for row in transformed)
        centered = (finite+rownorm[i]*epsg).upper()
        legacy = (8*rownorm[i]).upper()
        cap = min(centered,legacy)
        selected.append(cap)
        row_records.append(dict(coordinate=i+1,centre_exact=str(centre_exact),centre=record(centre),
            finite_centered_range_upper=record(finite),centered_true_cap=record(centered),
            legacy_true_cap=record(legacy),selected_true_cap=record(cap),
            selected_method="centered_zero_mass" if centered<=legacy else "legacy_global_cap",
            selected_not_above_legacy=bool(cap<=legacy)))
    centers_exact, centers = zip(*[_range_center([row[i] for row in gradients]) for i in range(M)])
    finite_norm = max(sum((abs(row[i]-centers[i]).upper()**2 for i in range(M)),arb(0)).sqrt().upper()
                      for row in gradients)
    centered_norm = (finite_norm+epsg).upper()
    norm_cap = min(centered_norm,arb(8))
    vector_record = dict(centres_exact=[str(x) for x in centers_exact],
        centres=[record(x) for x in centers],finite_centered_L2_cap=record(finite_norm),
        centered_true_L2_cap=record(centered_norm),legacy_true_L2_cap=record(arb(8)),
        selected_true_L2_cap=record(norm_cap),selected_not_above_legacy=bool(norm_cap<=8),
        selected_method="centered_zero_mass" if centered_norm<=8 else "legacy_global_cap")
    return selected,norm_cap,dict(rows=row_records,gradient_vector=vector_record,
        analytic_zero_mass_required=True,numerical_zero_sum_used_as_proof=False,
        conditional_finite_moments_replaced=False,scope="teacher Taylor remainders only")


def _covers_exact_box(saved, lo, hi):
    return (endpoint_fraction(saved["lower_exact_dyadic"]) <= lo and
            endpoint_fraction(saved["upper_exact_dyadic"]) >= hi)


def _read_rows(blocks, exact_box, analytic):
    if not blocks:
        raise ValueError("nonempty candidate blocks required")
    first = blocks[0]
    cfg = first["config"]
    if cfg["K"] != 8 or cfg["label_mode"] != "complement":
        raise ValueError("K8 complete 128-label complement data required")
    if analytic["schema"] != "r03_candidate_analytic_bounds_v1" or analytic["config"] != cfg:
        raise ValueError("candidate analytic-bound configuration mismatch")
    if exact(analytic["teacher_error_L1_upper"]) != 0 or exact(analytic["teacher_mass_upper"]) != 1:
        raise ValueError("uniform conditional alias bounds require teacher_error=0 and mass=1")
    if not analytic["uniform_all_labels_and_input_box"] or not analytic["Hessian_errors_are_operator_not_entrywise"]:
        raise ValueError("uniform box/operator-bound declarations required")
    saved = [interval(x) for x in first["a_enclosures"]]
    if len(saved) != M:
        raise ValueError("eight candidate box coordinates required")
    for i, (lo, hi) in enumerate(exact_box):
        if not _covers_exact_box(first["a_enclosures"][i], lo, hi):
            raise ValueError("candidate finite box does not contain requested X")
        if exact(analytic["a_upper_exact"][i]) < endpoint_fraction(first["a_enclosures"][i]["upper_exact_dyadic"]):
            raise ValueError("analytic bound is smaller than candidate box upper endpoint")
    expected_outer = int((exact(cfg["vmax"])-exact(cfg["vmin"]))/exact(cfg["hv"]))+1
    rows = {}
    for block in blocks:
        if (block["schema"] != "r03_candidate_finite_label_block_v1" or block["status"] != "completed"
                or block["partial_conditional_row"] is not None):
            raise ValueError("partial candidate blocks cannot certify")
        if (block["model_signature"] != first["model_signature"] or block["config"] != cfg
                or block["a_enclosures"] != first["a_enclosures"]):
            raise ValueError("inconsistent model/box/configuration among blocks")
        for row in block["conditional_rows"]:
            j = row["label_index"]
            if (type(j) is not int or j in rows or not 0 <= j < 128 or row["status"] != "completed"
                    or row["outer_nodes_completed"] != expected_outer or row["partial_current_inner_factors"] != 0
                    or row["label_bits"] != [(j >> (7-i)) & 1 for i in range(M)]):
                raise ValueError("incomplete, duplicate or invalid conditional row")
            g = [interval(x) for x in row["conditional_gradient_finite"]]
            if len(g) != M:
                raise ValueError("eight conditional radial-gradient coordinates required")
            rows[j] = dict(g=g, **{name:_symmetrize_enclosures(_matrix(row["matrix_blocks"][name])) for name in BLOCKS})
    if sorted(rows) != list(range(128)):
        raise ValueError("all 128 complement representatives required")
    return rows, saved


def _teacher(teacher):
    if teacher["schema"] != "r03_teacher_derivative_certificate_v1" or teacher["renormalized"]:
        raise ValueError("true unnormalized teacher certificate required")
    family = teacher["family"]
    if (exact(family["lambda_centre_exact"]) != 0
            or [exact(x) for x in family["effective_angles_exact"]] != [exact(x) for x in PHI]
            or exact(family["config"]["amplitude"]) != Fraction(4, 5)
            or [exact(x) for x in family["bias_exact"]] != [0]*8):
        raise ValueError("teacher must be the frozen eight-row zero-bias lambda0 family")
    if exact(teacher["sum_qprime_exact"]) != 0 or not teacher["taylor_remainder_L1"]["valid_for_all_real_lambda"]:
        raise ValueError("zero derivative mass and uniform Taylor remainder declarations required")
    q, p = [[interval(x) for x in teacher[key]] for key in
            ("q_true_component_intervals", "prime_true_component_intervals")]
    if len(q) != 256 or len(p) != 256:
        raise ValueError("all 256 original teacher labels required")
    if not sum(q, arb(0)).contains(1) or not sum(p, arb(0)).contains(0):
        raise ValueError("teacher enclosures contradict mass identities")
    if any(not q[j].overlaps(q[255-j]) or not p[j].overlaps(p[255-j]) for j in range(128)):
        raise ValueError("teacher enclosures contradict analytic complement symmetry")
    q1 = _positive_upper(teacher["prime_true_L1"], "prime L1")
    q2 = _positive_upper(teacher["qsecond_global_L1_cap"], "qsecond L1")
    if not q2 < aq(".442323"):
        raise ValueError("prescribed global second derivative cap not established")
    return q, p, q1


def compute(compactinputs, candidateblocks, teacher, risk=None):
    """Compose one prespecified h box; no numerical search or new integrals."""
    bits = compactinputs.get("precision_bits", 192)
    if type(bits) is not int or bits < 128:
        raise ValueError("integer precision_bits>=128 required")
    previous = ctx.prec
    ctx.prec = bits
    try:
        hq = exact(compactinputs["h_exact"])
        if hq not in (Fraction(1, 100), Fraction(1, 1000), Fraction(1, 10000)):
            raise ValueError("h must be one of .01, .001, .0001")
        rq = [exact(x) for x in compactinputs["radii_exact"]]
        if rq != [Fraction(3, 4)*hq] + [Fraction(3, 20)*hq]*7:
            raise ValueError("frozen radii (.75,.15,...,.15)*h required")
        a0q = [exact(x) for x in compactinputs["a0_exact"]]
        if len(a0q) != M or any(a-r <= 0 or a+r > 1 for a,r in zip(a0q,rq)):
            raise ValueError("entire eight-coordinate X must be in (0,1]")
        Aexact = [[exact(x) for x in row] for row in compactinputs["A_exact"]]
        if len(Aexact) != M or any(len(row) != M for row in Aexact):
            raise ValueError("A is an exact rational 8x8 matrix")
        if any(Aexact[i][j] != Aexact[j][i] for i in range(M) for j in range(M)):
            raise ValueError("frozen A must be exactly symmetric")
        h, r, a0, A = aq(hq), [aq(x) for x in rq], [aq(x) for x in a0q], _matrix(Aexact)
        box = [aq(a-b).union(aq(a+b)) for a,b in zip(a0q,rq)]
        analytic = compactinputs["candidate_analytic_bounds"]
        exact_box = [(a-b,a+b) for a,b in zip(a0q,rq)]
        rows, saved_box = _read_rows(candidateblocks, exact_box, analytic)
        q, p, q1 = _teacher(teacher)
        epsg, epsH, epsP = [_positive_upper(analytic["bounds"][key], key) for key in
                           ("uniform_gradient_L2", "uniform_Hba_operator", "uniform_Hperp_operator")]
        oldg = aq(compactinputs.get("old_gradient_L2_cap_exact", ".0000000007572"))
        olddist = aq(compactinputs.get("old_root_distance_cap_exact", ".00000001805"))
        if not oldg >= 0 or not olddist >= 0:
            raise ValueError("nonnegative old certificate caps required")
        tail = aq(".2211615")*h*h
        EH, EP = epsH*(1+h*q1)+aq("146.25")*tail, epsP*(1+h*q1)+aq("8.25")*tail
        G1 = [arb(0) for _ in range(M)]
        H0, H1 = {name:_zeros() for name in BLOCKS}, {name:_zeros() for name in BLOCKS}
        for label in range(128):
            w0, w1 = q[label]+q[255-label], p[label]+p[255-label]
            for i in range(M):
                G1[i] += w1*rows[label]["g"][i]
                for j in range(M):
                    for name in BLOCKS:
                        H0[name][i][j] += w0*rows[label][name][i][j]
                        H1[name][i][j] += w1*rows[label][name][i][j]
        hball = _symball(h)
        H = {name:_symmetrize_enclosures([[H0[name][i][j]+hball*H1[name][i][j]
                 for j in range(M)] for i in range(M)]) for name in BLOCKS}
        Aproof = ldl_positive(A)
        AG1 = _matvec(A, G1)
        AH = _matmul(A, H["Haa"])
        rownorm = [sum((x*x for x in row), arb(0)).sqrt().upper() for row in A]
        rnorm = sum((x*x for x in r), arb(0)).sqrt().upper()
        row_transfer_caps,gradient_transfer_cap,transfer_records = _gradient_transfer_bounds(
            A,[rows[label]["g"] for label in range(128)],epsg,rownorm)
        Y, Z, margin, ratios, radial_rows = [], [], [], [], []
        for i in range(M):
            y_base = oldg*rownorm[i]+h*abs(AG1[i]).upper()+h*rownorm[i]*epsg*q1
            y = y_base+tail*row_transfer_caps[i]
            legacy_remainder = tail*8*rownorm[i]
            finite = sum((abs(arb(int(i == j))-AH[i][j]).upper()*r[j] for j in range(M)), arb(0))
            z = finite+rownorm[i]*EH*rnorm
            Y.append(y); Z.append(z); margin.append(r[i]-y-z); ratios.append(z/r[i])
            radial_rows.append(dict(coordinate=i+1,Y=record(y),Z=record(z),
                Y_without_teacher_remainder=record(y_base),
                Y_teacher_remainder_selected=record(tail*row_transfer_caps[i]),
                Y_teacher_remainder_legacy=record(legacy_remainder),
                Y_legacy=record(y_base+legacy_remainder),
                Z_finite=record(finite),Z_error=record(rownorm[i]*EH*rnorm),
                radius=record(r[i]),inclusion_margin=record(margin[-1]),
                contraction_ratio=record(ratios[-1]),strict_inclusion=bool(margin[-1]>0),
                contraction_less_than_one=bool(ratios[-1]<1)))
        inclusion = all(value > 0 for value in margin)
        contraction = all(value < 1 for value in ratios)
        oldroot_inside = all(olddist < ri for ri in r)
        U = householder(a0)
        projected = _symmetrize_enclosures(_matmul(_matmul(_transpose(U), H["Hperp"]), U))
        stability = dict(Hbb=ldl_positive(H["Hbb"], aq(".1")+EH),
                         Haa=ldl_positive(H["Haa"], aq(".02")+EH),
                         transverse_slice=ldl_positive(projected, aq(".005")+EP))
        slice_dot = sum((a0[i]*box[i] for i in range(M)), arb(0))
        rootpass = Aproof["passed"] and inclusion and contraction and oldroot_inside
        stablepass = rootpass and all(x["passed"] for x in stability.values()) and slice_dot > 0
        result = dict(schema="r03_continuous_box_certificate_v1",precision_bits=bits,
            inputs_sha256={"compactinputs":_hash(compactinputs),"candidateblocks":_hash(candidateblocks),
                           "teacher":_hash(teacher),"risk":None if risk is None else _hash(risk)},
            h_exact=str(hq),radii_exact=[str(x) for x in rq],a0_exact=[str(x) for x in a0q],
            A_exact=[[str(x) for x in row] for row in Aexact],a_box=[record(x) for x in box],
            saved_candidate_box=[record(x) for x in saved_box],A_positive_definite=Aproof,
            teacher_derivative_L1_upper=record(q1),teacher_remainder_L1=record(tail),
            teacher_gradient_zero_mass_transfer=transfer_records,
            analytic_errors=dict(gradient_per_label=record(epsg),Hba_per_label=record(epsH),
                Hperp_per_label=record(epsP),Hba_total_operator=record(EH),Hperp_total_operator=record(EP)),
            signed_combinations=dict(G1_box=[record(x) for x in G1],A_G1_box=[record(x) for x in AG1],
                H0={name:_records(H0[name]) for name in BLOCKS},H1={name:_records(H1[name]) for name in BLOCKS},
                Hlambda_finite={name:_records(H[name]) for name in BLOCKS}),
            contraction=dict(rows=radial_rows,strict_self_mapping=inclusion,uniform_contraction=contraction,
                maximum_ratio_upper=record(max(x.upper() for x in ratios)),old_gradient_L2_cap=record(oldg),
                old_root_distance_cap=record(olddist),old_root_inside=oldroot_inside,passed=bool(rootpass)),
            stability=stability,transverse_slice_basis=_records(U),transverse_slice_matrix=_records(projected),
            slice_dot_a0_a=record(slice_dot),strict_local_minimum_mod_rotation=bool(stablepass),
            root_exists_unique_in_box_for_every_lambda=bool(rootpass),same_continuous_old_branch=bool(rootpass),
            new_target_integrals=0,new_teacher_integrals=0,new_random_draws=0,new_optimizer_updates=0,
            numerical_failure_is_not_negative_scientific_result=True,
            analytic_identities=dict(teacher_mass_one=True,teacher_derivative_mass_zero=True,
                mixed_bias_radial_block_zero="teacher complement and z reflection at b=0",
                bias_gradient_zero="same exact complement identity",
                full_transverse_null_at_root="Hperp*a=0 by the rotational Ward identity",
                exact_householder_orthonormal=True,householder_columns_orthogonal_to_exact_a0=True))
        # Optional implicit derivative. Entrywise enlargement is justified by
        # the full operator error, although it loses correlations conservatively.
        try:
            derivative_error = epsg*q1+gradient_transfer_cap*aq(".442323")*h
            Htrue = arb_mat([[H["Haa"][i][j]+_symball(EH) for j in range(M)] for i in range(M)])
            rhs = arb_mat([[-G1[i]+_symball(derivative_error)] for i in range(M)])
            solved = Htrue.solve(rhs)
            deriv = [solved[i,0] for i in range(M)]
            delta = deriv[0]-deriv[7]
            result["reversal_derivative"] = dict(status="enclosed",a_prime=[record(x) for x in deriv],
                a1_minus_a8_prime=record(delta),strictly_positive=bool(delta>0),
                gradient_lambda_error_L2=record(derivative_error),
                legacy_gradient_lambda_error_L2=record(epsg*q1+8*aq(".442323")*h),
                core_certificate_depends_on_this=False)
        except (ArithmeticError, ValueError, ZeroDivisionError) as error:
            result["reversal_derivative"] = dict(status="unresolved",strictly_positive=False,
                reason=str(error),core_certificate_depends_on_this=False)
        result["risk"] = _risk(risk, q, p, tail, exact_box, h)
        result["passed"] = bool(stablepass and result["risk"]["passed"])
        result["status"] = ("certified" if result["passed"] else
                            "candidate_partial" if risk is None else "unresolved")
        return result
    finally:
        ctx.prec = previous


def _risk(risk, q, p, tail, exact_box, h):
    if risk is None:
        return dict(status="not_supplied",passed=False,true_risk_curvature_negative=False)
    if risk["schema"] != "r03_risk_box_ratios_v1" or risk.get("true_ratios_enclosed") is not True:
        raise ValueError("risk must explicitly enclose true N_y/Q_y ratios")
    if (risk["direction"]["u_integer"] != FIXED_U or risk["direction"]["S_integer"] != FIXED_S):
        raise ValueError("risk must use the previously frozen angular direction u/sqrt(S)")
    rho = [interval(x) for x in risk["rho_true_intervals"]]
    if len(rho) != 256 or len(risk["a_enclosures"]) != M:
        raise ValueError("risk requires all 256 labels and eight box coordinates")
    if any(not _covers_exact_box(saved,lo,hi) for saved,(lo,hi) in zip(risk["a_enclosures"],exact_box)):
        raise ValueError("risk box does not cover requested X")
    lo, hi = min(x.lower() for x in rho), max(x.upper() for x in rho)
    B0 = (lo+hi)/2
    residual = max(abs(x-B0).upper() for x in rho)
    R0_original = sum((q[i]*rho[i] for i in range(256)),arb(0))
    R0, R0_centered = R0_original, None
    own_target = dict(enabled=False,ratio_bindings_checked=0,
                      analytic_numerator_sum_zero_used=False)
    if risk.get("own_target_numerator_sum_zero") is True:
        if len(risk["Q_true_intervals"]) != 256 or len(risk["N_true_intervals"]) != 256:
            raise ValueError("own-target centering requires all 256 Q and N intervals")
        Q = [interval(x) for x in risk["Q_true_intervals"]]
        N = [interval(x) for x in risk["N_true_intervals"]]
        if any(not x > 0 for x in Q):
            raise ValueError("every own-target Q interval must be strictly positive")
        if not sum(Q,arb(0)).contains(1) or not sum(N,arb(0)).contains(0):
            raise ValueError("own-target Q/N enclosures contradict analytic mass identities")
        centered_rho = []
        for i in range(256):
            independent_ratio = N[i]/Q[i]
            if not rho[i].overlaps(independent_ratio):
                raise ValueError("saved rho and independent N/Q are inconsistent at label "+str(i))
            # Both are already qualified true enclosures from source-bound
            # producers. Overlap is merely a contradiction check, never the
            # proof that an arbitrary saved rho is a valid true enclosure.
            centered_rho.append(rho[i].intersection(independent_ratio))
        R0_centered = sum(((q[i]-Q[i])*centered_rho[i] for i in range(256)),arb(0))
        if not R0_original.overlaps(R0_centered):
            raise ValueError("original and own-target-centered R0 enclosures are disjoint")
        R0 = R0_original.intersection(R0_centered)
        own_target = dict(enabled=True,ratio_bindings_checked=256,
            independent_ratio_overlap_consistency=True,overlap_is_mathematical_certificate=False,
            centered_ratio_source="intersection of source-qualified true rho and independently reconstructed N/Q",
            true_enclosure_qualification_source="source-bound producer proofs; not numerical overlap",
            analytic_numerator_sum_zero_used=True,
            Q_mass=record(sum(Q,arb(0))),N_sum=record(sum(N,arb(0))),
            intersection_used=True,
            identity="sum_y N_y(a)=0 for each fixed a; R0=sum_y(q0_y-Q_y(a))*N_y(a)/Q_y(a)")
    R1 = sum((p[i]*rho[i] for i in range(256)),arb(0))
    true = R0+_symball(h)*R1+_symball(tail*residual)
    return dict(status="enclosed",R0=record(R0),R0_original=record(R0_original),
        R0_centered=None if R0_centered is None else record(R0_centered),
        own_target_centering=own_target,R1=record(R1),centering_B0=record(B0),
        max_centered_ratio_absolute=record(residual),Taylor_remainder=record(tail*residual),
        true_curvature=record(true),true_risk_curvature_negative=bool(true<0),passed=bool(true<0),
        uses_teacher_at_each_lambda=True,remainder_mass_zero_centering=True)


def manufactured_checks():
    """Artificial saved 8D affine-root/constant-Hessian data; zero grid calls."""
    previous = ctx.prec
    ctx.prec = 192
    try:
        rec=lambda x:record(aq(x))
        h=Fraction(1,1000);r=[Fraction(3,4)*h]+[Fraction(3,20)*h]*7
        box=[record(aq(Fraction(1,2)-v).union(aq(Fraction(1,2)+v))) for v in r]
        cfg=dict(K=8,label_mode="complement",precision_bits=192,vmin="-1",vmax="1",hv="1")
        matrices={name:[[rec(1 if i==j else 0) for j in range(8)] for i in range(8)] for name in BLOCKS}
        rows=[dict(label_index=j,label_bits=[(j>>(7-i))&1 for i in range(8)],status="completed",
            outer_nodes_completed=3,partial_current_inner_factors=0,
            conditional_gradient_finite=[record(_symball(aq(v))) for v in r],
            matrix_blocks=copy.deepcopy(matrices)) for j in range(128)]
        block=dict(schema="r03_candidate_finite_label_block_v1",status="completed",partial_conditional_row=None,
                   model_signature="manufactured",config=cfg,a_enclosures=box,conditional_rows=rows)
        bounds=dict(schema="r03_candidate_analytic_bounds_v1",config=cfg,teacher_error_L1_upper="0",
            teacher_mass_upper="1",uniform_all_labels_and_input_box=True,Hessian_errors_are_operator_not_entrywise=True,
            a_upper_exact=[str(endpoint_fraction(x["upper_exact_dyadic"])) for x in box],
            bounds={key:rec(0) for key in ("uniform_gradient_L2","uniform_Hba_operator","uniform_Hperp_operator")})
        inp=dict(h_exact=str(h),radii_exact=[str(x) for x in r],a0_exact=["1/2"]*8,
                 A_exact=[[str(int(i==j)) for j in range(8)] for i in range(8)],candidate_analytic_bounds=bounds,
                 old_gradient_L2_cap_exact="0",old_root_distance_cap_exact="0")
        teacher=dict(schema="r03_teacher_derivative_certificate_v1",renormalized=False,
            family=dict(lambda_centre_exact="0",effective_angles_exact=PHI,bias_exact=["0"]*8,
                        config=dict(amplitude="4/5")),sum_qprime_exact="0",
            taylor_remainder_L1=dict(valid_for_all_real_lambda=True),
            q_true_component_intervals=[rec("1/256")]*256,prime_true_component_intervals=[rec(0)]*256,
            prime_true_L1=rec(0),qsecond_global_L1_cap=rec(0))
        risk=dict(schema="r03_risk_box_ratios_v1",true_ratios_enclosed=True,rho_true_intervals=[rec(-1)]*256,
                  a_enclosures=box,direction=dict(u_integer=FIXED_U,S_integer=FIXED_S))
        result=compute(inp,[block],teacher,risk)
        checks=dict(affine_eight_dimensional_root=result["passed"],
                    constant_negative_risk=result["risk"]["passed"],
                    centered_constant_risk_remainder_zero=interval(result["risk"]["Taylor_remainder"]).contains(0),
                    strict_json_roundtrip=bool(json.loads(json.dumps(result,allow_nan=False))["passed"]))
        no_risk=compute(inp,[block],teacher)
        checks["missing_risk_not_a_complete_certificate"]=not no_risk["passed"] and no_risk["status"]=="candidate_partial"
        signed=copy.deepcopy(teacher)
        signed["prime_true_L1"]=rec(1)
        for label,value in ((0,"1/4"),(255,"1/4"),(1,"-1/4"),(254,"-1/4")):
            signed["prime_true_component_intervals"][label]=rec(value)
        linear=copy.deepcopy(block)
        for label,value in ((0,"-1/5"),(1,"1/5")):
            linear["conditional_rows"][label]["conditional_gradient_finite"][0]=record(aq(value)+_symball(aq(r[0])))
        signed_result=compute(inp,[linear],signed,risk)
        checks["signed_teacher_derivative_preserved"]=(
            signed_result["passed"] and interval(signed_result["signed_combinations"]["G1_box"][0])<0 and
            signed_result["reversal_derivative"]["strictly_positive"])
        wide=copy.deepcopy(block)
        for row in wide["conditional_rows"]:
            row["matrix_blocks"]["Haa"]=[[record(aq(0).union(aq(2))) if i==j else rec(0) for j in range(8)] for i in range(8)]
        failed=compute(inp,[wide],teacher,risk)
        checks["broad_hessian_is_unresolved"]=not failed["passed"] and not failed["contraction"]["passed"]
        badA=copy.deepcopy(inp);badA["A_exact"][0][0]="-1"
        failed=compute(badA,[block],teacher,risk)
        checks["nonpositive_preconditioner_fails"]=not failed["passed"] and not failed["A_positive_definite"]["passed"]
        partial=copy.deepcopy(block);partial["status"]="budget_stopped"
        try:compute(inp,[partial],teacher,risk)
        except ValueError:checks["partial_grid_rejected"]=True
        else:checks["partial_grid_rejected"]=False
        bad_direction=copy.deepcopy(risk);bad_direction["direction"]["S_integer"]+=1
        try:compute(inp,[block],teacher,bad_direction)
        except ValueError:checks["wrong_risk_direction_rejected"]=True
        else:checks["wrong_risk_direction_rejected"]=False
        centered=copy.deepcopy(risk)
        centered["own_target_numerator_sum_zero"]=True
        centered["Q_true_intervals"]=[rec("1/256")]*256
        centered["N_true_intervals"]=[rec(0)]*256
        for j,v in ((0,"1/512"),(255,"1/512"),(1,"-1/512"),(254,"-1/512")):
            centered["N_true_intervals"][j]=rec(v)
        centered["rho_true_intervals"]=[record(interval(n)/interval(qq)) for n,qq in
                                       zip(centered["N_true_intervals"],centered["Q_true_intervals"])]
        qfake=[aq("1/256") for _ in range(256)]
        for j,v in ((0,"1/1024"),(255,"1/1024"),(1,"-1/1024"),(254,"-1/1024")):
            qfake[j]+=aq(v)
        exact_box=[(Fraction(1,2)-v,Fraction(1,2)+v) for v in r]
        algebra=_risk(centered,qfake,[arb(0)]*256,arb(0),exact_box,aq(h))
        checks["own_target_both_forms_same_nonzero_value"]=(
            interval(algebra["R0_original"]).contains(aq("1/512")) and
            interval(algebra["R0_centered"]).contains(aq("1/512")) and
            algebra["own_target_centering"]["ratio_bindings_checked"]==256)
        wide_risk=copy.deepcopy(centered)
        for j in (0,255):
            wide_risk["N_true_intervals"][j]=record(aq("49/25600").union(aq("51/25600")))
        for j in (1,254):
            wide_risk["N_true_intervals"][j]=record(aq("-51/25600").union(aq("-49/25600")))
        wide_risk["rho_true_intervals"]=[record(interval(n)/interval(qq)) for n,qq in
                                       zip(wide_risk["N_true_intervals"],wide_risk["Q_true_intervals"])]
        tightened=_risk(wide_risk,[aq("1/256")]*256,[arb(0)]*256,arb(0),exact_box,aq(h))
        checks["own_target_intersection_reduces_common_uncertainty"]=(
            interval(tightened["R0"]).is_zero() and not interval(tightened["R0_original"]).is_zero())
        incorrect=copy.deepcopy(centered);incorrect["rho_true_intervals"][0]=rec(0)
        try:_risk(incorrect,qfake,[arb(0)]*256,arb(0),exact_box,aq(h))
        except ValueError:checks["wrong_own_target_ratio_rejected"]=True
        else:checks["wrong_own_target_ratio_rejected"]=False
        eye=[[arb(int(i==j)) for j in range(8)] for i in range(8)]
        narrow=[[aq("1/64" if j%2 else "-1/64") for i in range(8)] for j in range(128)]
        c0,n0,d0=_gradient_transfer_bounds(eye,narrow,arb(0),[arb(1)]*8)
        shifted=[[x+aq("1/4") for x in row] for row in narrow]
        c1,n1,d1=_gradient_transfer_bounds(eye,shifted,arb(0),[arb(1)]*8)
        checks["zero_mass_caps_invariant_under_common_shift"]=(
            all(x==y for x,y in zip(c0,c1)) and n0==n1 and
            all(exact(d1["rows"][i]["centre_exact"])-exact(d0["rows"][i]["centre_exact"])==Fraction(1,4)
                for i in range(8)))
        checks["centered_transfer_caps_do_not_exceed_legacy"]=(
            all(x<=8 for x in c0) and n0<=8 and all(x<8 for x in c0))
        broad=[[arb(16 if j%2 else -16)]+[arb(0)]*7 for j in range(128)]
        cb,nb,db=_gradient_transfer_bounds(eye,broad,arb(0),[arb(1)]*8)
        checks["legacy_transfer_bound_independently_retained"]=(
            cb[0]==8 and nb==8 and db["rows"][0]["selected_method"]=="legacy_global_cap" and
            interval(db["rows"][0]["centered_true_cap"])==16)
        if not all(checks.values()):raise AssertionError(checks)
        return dict(schema="r03_continuous_manufactured_checks_v1",checks=checks,passed=True,
                    scientific_integrals=0,scientific_input_reads=0)
    finally:ctx.prec=previous


if __name__ == "__main__":
    print(json.dumps(manufactured_checks(),indent=2,allow_nan=False))
