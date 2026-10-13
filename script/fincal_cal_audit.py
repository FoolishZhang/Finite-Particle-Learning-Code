"""Independent fixed-scope audit of finite calibration and validation.

No producer is imported. First-prefix Price scores use direct probabilities;
full RNG hashes, stored-array statistics, splines, exact rational envelopes and
all validation spline selections have separately declared scopes.
"""
from pathlib import Path
from fractions import Fraction
import argparse
import hashlib
import json
import math
import os
import time
import traceback

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/finite_calibration/CAL"
PROTOCOL = OUT / "AUDIT_PROTOCOL.json"
DEST = OUT / "AUDIT.json"
TOL, CTOL, RTOL = 1e-11, 5e-9, 2e-8


def read(p):
    return json.loads(Path(p).read_text("utf-8"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_new(p, obj):
    with Path(p).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def fmean(values):
    return math.fsum(values) / len(values)


def mean_axis0(values):
    x = np.asarray(values)
    flat = x.reshape(len(x), -1)
    return np.asarray([fmean(flat[:, j]) for j in range(flat.shape[1])]).reshape(x.shape[1:])


def stats_axis0(values):
    x = np.asarray(values)
    mu = mean_axis0(x)
    square = ((x - mu) ** 2).reshape(len(x), -1)
    se = [math.sqrt(math.fsum(square[:, j]) / (len(x)*(len(x)-1))) for j in range(square.shape[1])]
    return mu, np.asarray(se).reshape(x.shape[1:])


def direct_price(z, rho, ns, ks, signs):
    out = np.empty((len(z), len(ns), len(rho), 4))
    sig = lambda t: 1.0 / (1.0 + np.exp(-t))
    lowest = 1.0
    for ni, (n, k) in enumerate(zip(ns, ks)):
        nuisance = np.prod(sig(z[:, :k, 2:n+2]), axis=2)
        for ri, r in enumerate(rho):
            second = r*z[:, :k, 0] + math.sqrt(1-r*r)*z[:, :k, 1]
            for si, (y1, y2) in enumerate(signs):
                raw_probability = nuisance*sig(y1*z[:, :k, 0])*sig(y2*second)
                normalizer = raw_probability.sum(axis=1, keepdims=True)
                assert np.all(normalizer > 0), "Probability normalization underflow"
                lowest = min(lowest, float(normalizer.min()))
                weights = raw_probability/normalizer
                marks = sig(-y1*z[:, :k, 0])*sig(-y2*second)
                out[:, ni, ri, si] = -y1*y2*np.sum(weights*(1-weights)*marks, axis=1)
    return out, lowest


def projection(mode_means, r_values):
    x = np.asarray(mode_means)
    u = np.empty(x.shape[:-1]); v = np.empty_like(u)
    for b in range(x.shape[0]):
        for ni in range(x.shape[1]):
            for ri in range(x.shape[2]):
                q = x[b, ni, ri]
                u[b, ni, ri] = math.fsum(q)/4
                v[b, ni, ri] = math.fsum([q[0], -q[1], -q[2], q[3]])/4
    odd_u = np.empty_like(u); even_v = np.empty_like(v)
    last = x.shape[2]-1
    for ri in range(last+1):
        odd_u[..., ri] = .5*(u[..., ri]-u[..., last-ri])
        even_v[..., ri] = .5*(v[..., ri]+v[..., last-ri])
    return dict(U_prime_unsym=u, V_prime_unsym=v, U_prime=odd_u,
                V_prime=even_v, g_prime=-odd_u-r_values*even_v)


def polynomial_real_roots(ascending, width):
    coefficients = np.trim_zeros(np.asarray(ascending, dtype=float), "b")
    if len(coefficients) <= 1:
        return []
    roots = np.polynomial.polynomial.polyroots(coefficients)
    return [float(np.clip(z.real, 0., width)) for z in roots
            if abs(z.imag) <= 1e-10 and -1e-12 <= z.real <= width+1e-12]


def rational_second_sup(c, knots):
    largest = Fraction(0)
    location = None
    for i in range(len(knots)-1):
        width = Fraction.from_float(float(knots[i+1]))-Fraction.from_float(float(knots[i]))
        cc = Fraction.from_float(float(c[2, i]))
        bb = 2*Fraction.from_float(float(c[1, i]))
        aa = 3*Fraction.from_float(float(c[0, i]))
        candidates = [(Fraction(0), cc), (width, cc+bb*width+aa*width*width)]
        if aa:
            vertex = -bb/(2*aa)
            if 0 < vertex < width:
                candidates.append((vertex, cc-bb*bb/(4*aa)))
        for point, value in candidates:
            if abs(value) > largest:
                largest, location = abs(value), [i, str(point)]
    rounded = float(largest)
    if Fraction.from_float(rounded) < largest:
        rounded = math.nextafter(rounded, math.inf)
    return largest, rounded, location


def freeze():
    assert not PROTOCOL.exists() and not DEST.exists()
    names = ["PROTOCOL.json", "REFERENCE.json", "TRAIN_EXECUTION.json", "TRAIN.json", "MODEL.json",
             "VALIDATION_PROTOCOL.json", "VALIDATION_STREAMS.json", "VALIDATION.json"]
    files = [OUT/x for x in names] + [ROOT/"script/fincal_calibrate.py"]
    pools = [read(OUT/"TRAIN.json"), read(OUT/"VALIDATION_STREAMS.json")]
    write_new(PROTOCOL, dict(
        schema="fincal-cal-independent-audit-v1", source_sha256=sha(__file__),
        inputs={p.relative_to(ROOT).as_posix(): sha(p) for p in files},
        expected_npz_sha256={r["file"]: r["file_sha256"] for pool in pools for r in pool["records"]},
        D_modes_absolute_tolerance=TOL, function_statistics_absolute_tolerance=TOL,
        stored_spline_coefficient_absolute_tolerance=CTOL,
        spline_root_and_selected_point_absolute_tolerance=RTOL,
        coefficient_tolerance_reason="Cubic coefficients amplify tiny independent summation changes by knot-spacing^-3; function values have the tighter 1e-11 check",
        price_scope="Each of 20 streams first 64 banks; both n/K, all 101 rho, four modes; direct sigmoid probability multiplication and normalization, independent Price expression",
        RNG_scope="Every original Gaussian in 20 streams x 2048 banks x 256 particles x 256 coordinates; exact original call shape (64,256,256)",
        aggregation="math.fsum all saved D_modes; unsymmetrized U/V, odd/even projection, g_prime, stream SE, antiderivative values",
        spline_scope="Reconstruct not-a-knot splines and 51/101 fixed-1001-point gates; diagnostic piecewise extrema by local polynomial roots",
        exact_envelope="Fraction extrema of every derivative quadratic from stored binary spline coefficients and knots; check receipt and minimal outward float",
        validation_scope="All residual/ratio/key-point arithmetic; all 32 pooled root/endpoints selections and 128 per-validation-stream selections",
        spline_tie_tolerance=1e-12, spline_tie_rule="smallest absolute rho, then signed rho",
        reference_scope="Recompute stored 64/96 differences, H spline identities, case scalings; selected moment consistency through stored reference interpolation and KL formula; no new Gaussian integrals",
        chronology_scope="Frozen source path plus MODEL hash in pre-validation receipt, validation stream record and validation summary; not external timestamp attestation",
        preserve_failures=True, automatic_retry=False,
        new_scientific_data=0, new_Gaussian_integrals=0))
    print(json.dumps({"frozen": True, "protocol_sha256": sha(PROTOCOL)}), flush=True)


def run():
    assert not DEST.exists(), "Preserve previous audit result"
    protocol = read(PROTOCOL)
    assert protocol["source_sha256"] == sha(__file__)
    for p, digest in protocol["inputs"].items():
        assert sha(ROOT/p) == digest, p
    started = time.perf_counter()
    comparisons, exacts, streams, envelopes, validations, resolution_rows = [], [], [], [], [], []
    counters = dict(replayed_Gaussian_scalars=0, independently_rescored_banks=0,
                    independently_rescored_D_modes=0, saved_D_modes_reaggregated=0,
                    repeated_spline_constructions=0, pooled_validation_selections=0,
                    per_stream_validation_selections=0)
    failure = None

    def compare(name, actual, expected, tolerance=TOL):
        a, e = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        assert a.shape == e.shape, (name, a.shape, e.shape)
        difference = np.abs(a-e)
        comparisons.append(dict(name=name, count=int(difference.size), tolerance=tolerance,
            max_absolute_error=float(difference.max()) if difference.size else 0.,
            failures=int(np.count_nonzero(difference > tolerance))))

    def exact(name, actual, expected):
        exacts.append(dict(name=name, passed=bool(actual == expected)))

    def spline(x, y):
        counters["repeated_spline_constructions"] += 1
        return CubicSpline(x, y, bc_type="not-a-knot")

    def select_minimum(x, derivatives):
        s = spline(x, derivatives); anti = s.antiderivative(); shift = float(anti(0.))
        if np.all(np.asarray(derivatives) == 0):
            return dict(rho=0., flat_entire_interval=True, candidates=[], tied_candidates=[-.75, 0., .75])
        candidates = [float(x[0]), float(x[-1])]
        for j in range(len(x)-1):
            for root in polynomial_real_roots(s.c[:, j][::-1], x[j+1]-x[j]):
                point = float(np.clip(x[j]+root, x[0], x[-1]))
                if all(abs(point-old) > 1e-10 for old in candidates):
                    candidates.append(point)
        candidates.sort()
        values = [float(anti(point)-shift) for point in candidates]
        smallest = min(values)
        tied = [point for point, value in zip(candidates, values) if value-smallest <= 1e-12]
        selected = min(tied, key=lambda point: (abs(point), point))
        return dict(rho=selected, flat_entire_interval=False,
                    candidates=[dict(rho=r, potential=v) for r,v in zip(candidates,values)],
                    tied_candidates=tied, selected_derivative=float(s(selected)))

    try:
        p, ref, model = read(OUT/"PROTOCOL.json"), read(OUT/"REFERENCE.json"), read(OUT/"MODEL.json")
        train, val_streams, validation = read(OUT/"TRAIN.json"), read(OUT/"VALIDATION_STREAMS.json"), read(OUT/"VALIDATION.json")
        val_receipt = read(OUT/"VALIDATION_PROTOCOL.json")
        source_hash, model_hash = sha(ROOT/"script/fincal_calibrate.py"), sha(OUT/"MODEL.json")
        for name, obj in [("protocol",p),("reference",ref),("model",model),("train",train),("validation-streams",val_streams),("validation",validation),("validation-receipt",val_receipt)]:
            exact(name+" source binding",obj["source_sha256"],source_hash)
        for name, obj in [("receipt",val_receipt),("validation",validation)]:
            exact(name+" frozen MODEL",obj["model_sha256"],model_hash)
        exact("stream start MODEL",val_streams["model_sha256_at_start"],model_hash)
        exact("validation execution receipt",val_streams["execution_receipt_sha256"],sha(OUT/"VALIDATION_PROTOCOL.json"))
        exact("train execution receipt",train["execution_receipt_sha256"],sha(OUT/"TRAIN_EXECUTION.json"))
        exact("model training binding",model["training_sha256"],sha(OUT/"TRAIN.json"))
        exact("model reference binding",model["reference_sha256"],sha(OUT/"REFERENCE.json"))
        exact("no teacher-specific penalty",model["teacher_free"] and model["shared_across_all_teachers"] and validation["no_teacher_specific_penalty_or_postvalidation_refit"],True)
        rho = np.array(p["rho101"]); ns, ks, signs = p["ns"], p["ks"], p["active_signs"]
        assert len(rho)==101 and ns==[62,254] and ks==[64,256]
        assert p["streams"]==dict(train=16,validation=4) and p["banks_per_stream"]==2048
        assert p["max_prior_shape_per_block"]==[64,256,256]
        for path,digest in p["inputs"].items():
            exact("producer input "+path,sha(ROOT/path),digest)
        for field in ["r801","rp801","H801","d","var_logsigmoid"]:
            d=float(np.max(np.abs(np.asarray(ref["orders"][0][field])-np.asarray(ref["orders"][1][field]))))
            compare("reference saved difference "+field,d,ref["absolute_differences"][field])
            exact("reference gate "+field,d<=p["qualification"]["reference64_vs96_absolute"],ref["qualification_by_field"][field])
        x801=np.array(ref["rho801"])
        for table in ref["orders"]:
            rr=spline(x801,table["r801"]); hh=rr.antiderivative()
            compare("reference H "+str(table["order"]),hh(x801)-hh(0),table["H801"])
        for name in ["r801","rp801","H801"]:
            exact("MODEL reference "+name,model["reference"][name],ref["orders"][0][name])
        for name in ["d","A","var_logsigmoid"]:
            exact("MODEL inherited "+name,model["reference"][name],ref["inherited"][name])
        for case in model["cases"]:
            compare("case w "+str(case["n"]),case["J"]/math.sqrt(case["n"]*ref["inherited"]["var_logsigmoid"]),case["w"])

        means_by_pool={}
        for pool, info in [("train",train),("validation",val_streams)]:
            pool_means=[]
            for rec in info["records"]:
                filepath=ROOT/rec["file"]
                file_hash=sha(filepath)
                exact(pool+str(rec["stream"])+" NPZ hash",file_hash,rec["file_sha256"])
                with np.load(filepath,allow_pickle=False) as data:
                    derivatives=data["D_modes"]
                    assert derivatives.shape==(2048,2,101,4) and np.isfinite(derivatives).all()
                    assert np.array_equal(data["rho"],rho) and np.array_equal(data["ns"],ns)
                    assert np.array_equal(data["ks"],ks) and np.array_equal(data["signs"],signs)
                expected_seed=p["seed_bases"][pool]+rec["stream"]
                exact(pool+str(rec["stream"])+" seed",rec["seed"],expected_seed)
                rng=np.random.Generator(np.random.PCG64(expected_seed))
                full=hashlib.sha256();first_hash=None;score_error=None;lowest=None
                for offset in range(0,2048,64):
                    z=rng.standard_normal((64,256,256))
                    full.update(z.tobytes(order="C"));counters["replayed_Gaussian_scalars"]+=int(z.size)
                    if offset==0:
                        first_hash=hashlib.sha256(z.tobytes(order="C")).hexdigest()
                        independent,lowest=direct_price(z,rho,ns,ks,signs)
                        compare(pool+str(rec["stream"])+" independent Price",independent,derivatives[:64])
                        score_error=float(np.max(np.abs(independent-derivatives[:64])))
                        counters["independently_rescored_banks"]+=64
                        counters["independently_rescored_D_modes"]+=int(independent.size)
                exact(pool+str(rec["stream"])+" full RNG",full.hexdigest(),rec["full_prior_sha256"])
                exact(pool+str(rec["stream"])+" prefix RNG",first_hash,rec["first64_prior_sha256"])
                independent_mean=mean_axis0(derivatives)
                counters["saved_D_modes_reaggregated"]+=int(derivatives.size)
                compare(pool+str(rec["stream"])+" mean",independent_mean,rec["D_modes_mean"])
                pool_means.append(independent_mean)
                streams.append(dict(pool=pool,stream=rec["stream"],file=rec["file"],file_sha256=file_hash,
                    full_prior_sha256=full.hexdigest(),first64_prior_sha256=first_hash,
                    max_D_mode_error=score_error,smallest_probability_normalizer=lowest))
                print(json.dumps({"pool":pool,"stream":rec["stream"],"seconds":round(time.perf_counter()-started,2)}),flush=True)
            means_by_pool[pool]=np.asarray(pool_means)

        r101=np.array(ref["orders"][0]["r801"])[::8]
        train_proj=projection(means_by_pool["train"],r101)
        val_proj=projection(means_by_pool["validation"],r101)
        train_stats={name:stats_axis0(array) for name,array in train_proj.items()}
        val_stats={name:stats_axis0(array) for name,array in val_proj.items()}
        full_g=np.empty((16,2,101))
        for b in range(16):
            for ni in range(2):
                gp=spline(rho,train_proj["g_prime"][b,ni]);anti=gp.antiderivative()
                full_g[b,ni]=anti(rho)-anti(0.)
        train_stats["g_values"]=stats_axis0(full_g)
        gate_count=gate_passed=0;gres_pass=gpres_pass=0
        for ni, case in enumerate(model["cases"]):
            for field,array in train_proj.items():
                compare("train per-stream "+field+str(ni),array[:,ni],case["per_stream"][field])
            for field in ["U_prime","V_prime","g_prime"]:
                compare("train pooled "+field+str(ni),train_stats[field][0][ni],case[field])
            for field,(mu,se) in train_stats.items():
                compare("train SE "+field+str(ni),se[ni],case["calibration_SE"][field])
            compare("per-stream g "+str(ni),full_g[:,ni],case["per_stream_g_values"])
            gp=spline(rho,train_stats["g_prime"][0][ni]);anti=gp.antiderivative();shift=float(anti(0.))
            compare("pooled g "+str(ni),anti(rho)-shift,case["g_values"])
            compare("origin shift "+str(ni),shift,case["g_antiderivative_origin_shift"])
            compare("spline coefficients "+str(ni),gp.c,case["gprime_spline_c"],CTOL)
            exact("spline knot axis "+str(ni),gp.x.tolist(),case["gprime_spline_x"])
            compare("linear aggregation "+str(ni),float(np.max(np.abs((anti(rho)-shift)-train_stats["g_values"][0][ni]))),case["linear_aggregation_check_max"])
            node_pass=(train_stats["g_prime"][1][ni]<=p["qualification"]["training_gprime_SE_absolute"]).tolist()
            exact("node SE gates "+str(ni),node_pass,case["gprime_SE_pass_by_node"])
            gate_count+=len(node_pass);gate_passed+=sum(node_pass)
            stored_c=np.array(case["gprime_spline_c"]);stored_x=np.array(case["gprime_spline_x"])
            rat,up,loc=rational_second_sup(stored_c,stored_x)
            receipt=case["g_second_sup_receipt"]
            exact("exact rational sup numerator "+str(ni),str(rat.numerator),receipt["exact_numerator"])
            exact("exact rational sup denominator "+str(ni),str(rat.denominator),receipt["exact_denominator"])
            exact("minimal outward float "+str(ni),up,case["g_second_sup"])
            exact("outward float containment "+str(ni),Fraction.from_float(up)>=rat,True)
            exact("preceding float not upper "+str(ni),Fraction.from_float(math.nextafter(up,-math.inf))<rat,True)
            envelopes.append(dict(n=case["n"],exact_numerator=str(rat.numerator),exact_denominator=str(rat.denominator),upper_float=up,maximizer_segment_and_offset=loc))
            coarse=spline(rho[::2],train_stats["g_prime"][0][ni][::2]);ca=coarse.antiderivative();cs=float(ca(0.))
            dense=np.linspace(-.75,.75,1001)
            gdiff=float(np.max(np.abs((anti(dense)-shift)-(ca(dense)-cs))))
            gpdiff=float(np.max(np.abs(gp(dense)-coarse(dense))))
            expected=case["resolution_51_vs101"]
            compare("resolution g "+str(ni),gdiff,expected["g_max_abs_difference"])
            compare("resolution gp "+str(ni),gpdiff,expected["gprime_max_abs_difference"])
            gpass=gdiff<=p["qualification"]["g51_vs101_absolute"]
            gppass=gpdiff<=p["qualification"]["gprime51_vs101_absolute"]
            exact("resolution g gate "+str(ni),gpass,case["g_resolution_pass"])
            exact("resolution gp gate "+str(ni),gppass,case["gprime_resolution_pass"])
            gres_pass+=gpass;gpres_pass+=gppass
            points_g=list(dense)+list(rho);points_d=list(dense)+list(rho)
            for j in range(len(rho)-1):
                left=float(rho[j]);width=float(rho[j+1]-rho[j])
                coeff=np.array([(float(gp.derivative(k)(left))-float(coarse.derivative(k)(left)))/math.factorial(k) for k in range(4)])
                points_g.extend(left+root for root in polynomial_real_roots(coeff,width))
                points_d.extend(left+root for root in polynomial_real_roots([coeff[1],2*coeff[2],3*coeff[3]],width))
            diag_g=float(np.max(np.abs((anti(points_g)-shift)-(ca(points_g)-cs))))
            diag_d=float(np.max(np.abs(gp(points_d)-coarse(points_d))))
            compare("diagnostic piecewise g max "+str(ni),diag_g,expected["g_piecewise_search_max_abs_difference"])
            compare("diagnostic piecewise gp max "+str(ni),diag_d,expected["gprime_piecewise_search_max_abs_difference"])
            resolution_rows.append(dict(n=case["n"],g1001_max=gdiff,gprime1001_max=gpdiff,
                g_piecewise_max=diag_g,gprime_piecewise_max=diag_d,
                note="Diagnostic extremum location/count can differ under tiny coefficient roundoff; no gate is based on those locations"))
        exact("202 node count",gate_count,model["gprime_SE_gate_count"])
        exact("202 node pass count",gate_passed,model["gprime_SE_gate_passed"])
        exact("g resolution pass count",gres_pass,model["g_resolution_gates_passed"])
        exact("gp resolution pass count",gpres_pass,model["gprime_resolution_gates_passed"])

        r_splines=[spline(x801,t["r801"]) for t in ref["orders"]]
        for ni, case in enumerate(validation["cases"]):
            trained=model["cases"][ni]
            u,use=val_stats["U_prime"][0][ni],val_stats["U_prime"][1][ni]
            v,vse=val_stats["V_prime"][0][ni],val_stats["V_prime"][1][ni]
            for field,actual in [("U_prime",u),("V_prime",v),("U_prime_SE",use),("V_prime_SE",vse),("minus_V_prime",-v)]:
                compare("validation "+field+str(ni),actual,case[field])
            exact("validation V negative "+str(ni),bool(np.all(v<0)),True)
            penalties=dict(raw=np.zeros(101),finite=np.array(trained["g_prime"]),
                asymptotic=.5*trained["w"]*r101,
                quadratic=.5*trained["w"]*model["reference"]["d"]*rho)
            residuals={}
            for method,penalty in penalties.items():
                vals=penalty-val_proj["g_prime"][:,ni]
                mu,se=stats_axis0(vals)
                ratios=vals/(-val_proj["V_prime"][:,ni]);ratio_mu,ratio_se=stats_axis0(ratios)
                tse=np.array(trained["calibration_SE"]["g_prime"]) if method=="finite" else np.zeros(101)
                current=dict(mean=mu,validation_conditional_SE=se,
                    combined_train_validation_SE=np.hypot(se,tse),per_validation_stream=vals,
                    delta_over_minus_Vprime_ratio_of_means=mu/(-v),implied_signed_moment_error_at_same_rho=-mu/(-v),
                    per_stream_ratio_mean=ratio_mu,per_stream_ratio_SE=ratio_se)
                for field,value in current.items():
                    compare("residual "+str(ni)+method+field,value,case["residuals"][method][field])
                residuals[method]=current
            for point in case["key_points"]:
                index=int(np.argmin(np.abs(rho-point["rho"])))
                assert abs(rho[index]-point["rho"])<1e-12
                compare("key V "+str(ni)+str(point["rho"]),[-v[index],vse[index]],[point["minus_V_prime"],point["minus_V_prime_SE"]])
                for method, row in point["residuals"].items():
                    for field,value in row.items():
                        compare("key residual "+str(ni)+str(point["rho"])+method+field,residuals[method][field][index],value)
            for target in case["spline_optima"]:
                theta,method=target["theta"],target["method"]
                teacher_index=int(np.argmin(np.abs(x801-theta)))
                assert abs(x801[teacher_index]-theta)<1e-12
                rt=float(ref["orders"][0]["r801"][teacher_index])
                compare("teacher moment "+str(ni)+str(theta),rt,target["r_teacher"])
                chosen=select_minimum(rho,u+rt*v+penalties[method]);counters["pooled_validation_selections"]+=1
                tag="optimum "+str(ni)+str(theta)+method
                compare(tag+" rho",chosen["rho"],target["rho"],RTOL)
                exact(tag+" flat flag",chosen["flat_entire_interval"],target["flat_entire_interval"])
                compare(tag+" tied points",chosen["tied_candidates"],target["tied_candidates"],RTOL)
                compare(tag+" candidates",[x["rho"] for x in chosen["candidates"]],[x["rho"] for x in target["candidates"]],RTOL)
                compare(tag+" candidate values",[x["potential"] for x in chosen["candidates"]],[x["potential"] for x in target["candidates"]])
                if not chosen["flat_entire_interval"]:
                    compare(tag+" derivative",chosen["selected_derivative"],target["selected_derivative"])
                per=[]
                for si in range(4):
                    q=select_minimum(rho,val_proj["U_prime"][si,ni]+rt*val_proj["V_prime"][si,ni]+penalties[method])
                    per.append(q["rho"]);counters["per_stream_validation_selections"]+=1
                compare(tag+" per-stream rhos",per,target["per_validation_stream_rhos"],RTOL)
                rs=float(r_splines[0](chosen["rho"]));rs96=float(r_splines[1](chosen["rho"]))
                compare(tag+" selected reference interpolation",rs,target["r_selected"])
                saved_rt,saved_rs=target["r_teacher"],target["r_selected"]
                kl=(1+saved_rt)/2*math.log((1+saved_rt)/(1+saved_rs))+(1-saved_rt)/2*math.log((1-saved_rt)/(1-saved_rs))
                compare(tag+" KL algebra",max(0.,kl),target["reference_joint_KL"])
                exact(tag+" selected reference gate",target["selected_r_GH64_vs96"]<=p["qualification"]["r64_vs96_absolute"],target["selected_r_qualification_pass"])
                validations.append(dict(n=case["n"],theta=theta,method=method,independent=chosen,
                    per_validation_stream_rhos=per,selected_reference_interpolation64=rs,
                    selected_reference_interpolation96=rs96,
                    note="No selected-point Gaussian integral was repeated; r is a consistency comparison to the fixed reference spline"))
        exact("pooled selections count",counters["pooled_validation_selections"],32)
        exact("per-stream selections count",counters["per_stream_validation_selections"],128)
    except Exception:
        failure=traceback.format_exc()
    nf=sum(row["failures"] for row in comparisons);ef=sum(not row["passed"] for row in exacts)
    ledger=dict(counters)
    ledger.update(new_scientific_samples=0,new_observations=0,new_data_training_fits=0,new_Gaussian_integrals=0,
        replayed_prior_vectors=counters["replayed_Gaussian_scalars"]//256,
        replayed_banks=counters["replayed_Gaussian_scalars"]//65536,
        first64_replayed_Gaussian_prefix_values=counters["independently_rescored_banks"]*65536,
        remaining_banks_not_independently_rescored=counters["replayed_Gaussian_scalars"]//65536-counters["independently_rescored_banks"])
    result=dict(status="passed" if failure is None and nf==0 and ef==0 else "failed",
        code_sha256=sha(__file__),protocol_sha256=sha(PROTOCOL),input_sha256=protocol["inputs"],
        elapsed_seconds=time.perf_counter()-started,numerical_comparisons=comparisons,exact_checks=exacts,
        numerical_failures=nf,exact_failures=ef,stream_replays=streams,
        rational_g_second_sup=envelopes,resolution_checks=resolution_rows,
        validation_selections=validations,ledger=ledger,software_error=failure,
        scope="Full original RNG replay; independent Price only each first64; all saved-array statistics and model/validation spline arithmetic; no independent Gaussian reference integration or uniform MC certification")
    write_new(DEST,result)
    print(json.dumps({"status":result["status"],"numeric_failures":nf,"exact_failures":ef,"ledger":ledger,"sha256":sha(DEST)}),flush=True)
    if result["status"]!="passed":
        raise SystemExit(1)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=["freeze","run"])
    args=parser.parse_args();freeze() if args.mode=="freeze" else run()
