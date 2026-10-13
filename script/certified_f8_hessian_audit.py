"""Two freeze-gated independent coefficient-first finite-grid Hessian replays.

Likelihoods use cosh, probabilities tanh; teacher labels are enumerated and
count-folded here. No producer W/M/S, moment-expansion or matrix helper is used.
--check only reads saved JSON and rebuilds scalar/matrix comparisons, not grids.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import operator
from pathlib import Path
import time
import certified_teacher_compute as common
from flint import arb, arb_mat, ctx

ROOT=common.ROOT
OUT=ROOT/"artifacts/certified_f8_hessian_v1_20261008"
INPUTS=OUT/"INPUTS.json";FREEZE=OUT/"FREEZE.json";SUMMARY=OUT/"FINITE_HESSIANS.json"
NOMINAL="artifacts/local_witness_feasibility_v1_20261008/NOMINAL_SUMMARY.json"
TEACHER="artifacts/certified_teacher_v1_20261008/FINITE_GRID_SUM.json"
SOURCES=("script/certified_f8_hessian_audit.py","script/certified_f8_hessian_audit_test.py")
SELECTED=((0,0,0,0),(1,1,1,1))
BITS=192;SECONDS=600
BLOCKS=("Hbb","Haa","Hperp")


def integer(value,name,minimum=1):
    if isinstance(value,bool):raise ValueError(f"actual integer {name} required")
    try:value=operator.index(value)
    except TypeError as error:raise ValueError(f"actual integer {name} required") from error
    if value<minimum:raise ValueError(f"invalid integer {name}")
    return value


def bits(label,m):
    label=integer(label,"label",0)
    if label>=1<<m:raise ValueError("label out of range")
    return tuple((label>>(m-1-j))&1 for j in range(m))


def counts(y):
    m=len(y)
    return tuple(y[j]+y[m-1-j] for j in range(m//2))


def ball(value):
    if isinstance(value,arb):
        if not value.is_finite():raise ValueError("finite weight required")
        return value
    return common.aq(common.exact(value))


def fold_class(weights,c):
    """Count-only complement folding; self-complement counts retain every y."""
    c=tuple(integer(v,"count",0) for v in c)
    if not c or any(v>2 for v in c):raise ValueError("nonempty ternary counts")
    comp=tuple(2-v for v in c)
    if c!=min(c,comp):raise ValueError("canonical count representative required")
    m=2*len(c);weights=list(weights)
    if len(weights)!=1<<m:raise ValueError("complete original teacher weights required")
    accum={};originals={};selected=[]
    for label,weight in enumerate(weights):
        y=bits(label,m);cc=counts(y)
        if cc not in (c,comp):continue
        q=ball(weight)
        if q.upper()<0:raise ValueError("negative teacher weight")
        # Only the noncanonical count side is complemented. For c==comp
        # this condition is false, including y lexicographically above 1-y.
        folded=tuple(1-v for v in y) if cc!=min(cc,tuple(2-v for v in cc)) else y
        if counts(folded)!=c:raise ArithmeticError("independent folding failed")
        accum[folded]=accum.get(folded,arb(0))+q
        originals.setdefault(folded,[]).append(label);selected.append(label)
    labels=[dict(y=y,q=accum[y],original_labels=originals[y]) for y in sorted(accum)]
    return labels,selected


def zero(m):return [[arb(0) for _ in range(m)] for _ in range(m)]


def matrix_records(matrix):return [[common.record(value) for value in row] for row in matrix]


def saturation_caps(threshold=128):
    threshold=integer(threshold,"saturation threshold")
    if threshold!=128:raise ValueError("fixed threshold128 required")
    ecap=arb(2)**-184;tecap=arb(2)**-177
    if not (-arb(128)).exp()<ecap or not 128*(-arb(128)).exp()<tecap:
        raise ArithmeticError("independent saturation upper inequalities failed")
    return arb(0).union(ecap),arb(0).union(tecap)


def exponential_pair(t,threshold,caps):
    if threshold is not None and t.lower()>=threshold:
        return caps[0],caps[1],True
    e=(-t).exp()
    return e,t*e,False


class AuditBudgetStop(RuntimeError):
    def __init__(self,partial):
        super().__init__("fixed independent replay wall budget reached; partial saved, no retry")
        self.partial=partial


def direct_weighted_class(a,c,weights,hz,L,hv,lo,hi,*,K=8,precision_bits=192,
                          deadline=None,saturation_threshold=128):
    """Generic manufactured/authorized replay; coefficient-first, explicit y."""
    a=tuple(common.exact(value) for value in a)
    c=tuple(integer(v,"count",0) for v in c)
    hz,L,hv,lo,hi=map(common.exact,(hz,L,hv,lo,hi))
    K=integer(K,"K",2);precision_bits=integer(precision_bits,"precision",64)
    if precision_bits!=BITS or int(ctx.prec)!=precision_bits:raise ValueError("fixed192bit caller precision required")
    if not a or len(a)!=len(c) or any(v>2 for v in c):raise ValueError("paired lengths/count shape")
    if min(hz,L,hv)<=0 or lo>=hi or any(q.denominator!=1 for q in (L/hz,lo/hv,hi/hv)):
        raise ValueError("positive exact integer grids required")
    if saturation_threshold is not None:saturation_threshold=integer(saturation_threshold,"threshold")
    labels,selected=fold_class(weights,c)
    p=len(a);m=2*p;aa=list(map(common.aq,a));nz=(L/hz).numerator
    z=[common.aq(hz)*i for i in range(-nz,nz+1)]
    w=[common.aq(hz)*(-x*x/2).exp()/(2*arb.pi()).sqrt() for x in z]
    slope=sum((common.aq(aj)*(cj-1) for aj,cj in zip(a,c)),arb(0))
    A=[];prob=[];sp=[];features=[]
    for x,weight in zip(z,w):
        denom=arb(1);pp=[];ss=[]
        for aj in aa:
            ch=(aj*x/2).cosh()
            denom*=(2*ch)**2
            pp.append((1+(aj*x/2).tanh())/2)
            ss.append(1/(4*ch*ch))
        A.append((x*slope).exp()/denom);prob.append(pp);sp.append(ss)
        features.append([weight,weight*x]+[weight*v for v in pp]+[weight*x*v for v in pp])
    moment_matrix=arb_mat(features)
    if moment_matrix.ncols()!=2+2*p:raise ArithmeticError("independent moment columns")
    caps=None if saturation_threshold is None else saturation_caps(saturation_threshold)
    dc=[arb(0) for _ in z];cc=[arb(0) for _ in z]
    crossbb=zero(m);crossaa=zero(m)
    samebb=zero(m);sameaa=zero(m);Bbb=[arb(0) for _ in range(m)];Baa=[arb(0) for _ in range(m)]
    same_labels_completed=same_nodes_current=sigma_nodes_completed=0
    completed=factors=saturated=0;last=None;partial_inner=partial_coeff=0;stage="outer"
    start=time.perf_counter();classweight=sum((entry["q"] for entry in labels),arb(0))
    def collect(complete=False,blocks=None):
        data=dict(status="completed" if complete else "budget_stopped",pair_counts=list(c),K=K,
            precision_bits=precision_bits,original_labels=selected,
            original_label_count=len(selected),folded_label_count=len(labels),
            folded_labels=[dict(y=list(e["y"]),original_labels=e["original_labels"],
                weight=common.record(e["q"])) for e in labels],class_weight=common.record(classweight),
            inner_grid=dict(h_exact=str(hz),L_exact=str(L),nodes=len(z),full_weights=True),
            outer_grid=dict(h_exact=str(hv),vmin_exact=str(lo),vmax_exact=str(hi),
                nodes_prescribed=int((hi-lo)/hv)+1,nodes_completed=completed,full_weights=True),
            implementation="cosh likelihood/tanh probabilities; accumulated d/c; explicit weighted y same/cross",
            producer_W_M_S_helpers_used=False,moment_columns=2+2*p,
            saturation_threshold=saturation_threshold,saturation_caps=None if caps is None else
                dict(exp_upper_exact="2^-184",t_exp_upper_exact="2^-177",
                     Arb_outward_rounding_retained=True,numerical_negative_lower_not_clipped=True),
            resources=dict(laplace_factors_evaluated=factors,saturated_factor_enclosures=saturated,
                ordinary_exponential_calls=factors-saturated,outer_nodes_completed=completed,
                matrix_entries_returned=0 if blocks is None else 3*m*m,
                new_random_draws=0,new_optimizer_updates=0,scientific_NPZ_reads=0),
            stage=stage,last_completed_outer_index=last,current_partial_inner_factors=partial_inner,
            current_partial_coefficient_nodes=partial_coeff,wall_seconds=time.perf_counter()-start,
            same_labels_completed=same_labels_completed,same_nodes_current=same_nodes_current,
            sigma_nodes_completed=sigma_nodes_completed,
            teacher_or_normal_mass_renormalized=False,class_weight_division_used=False,
            finite_sum_arithmetic_enclosed=True,analytic_errors_added=False,
            true_Hessian_certified=False)
        if blocks is not None:data["matrix_blocks"]={name:matrix_records(value) for name,value in blocks.items()}
        if not complete:
            data["partial_state"]=dict(d_coefficients=[common.record(x) for x in dc],
                c_coefficients=[common.record(x) for x in cc],
                crossbb=matrix_records(crossbb),crossaa=matrix_records(crossaa),
                samebb=matrix_records(samebb),sameaa=matrix_records(sameaa),
                Bbb=[common.record(x) for x in Bbb],Baa=[common.record(x) for x in Baa])
        return data
    def check_time():
        if deadline is not None and time.perf_counter()>=deadline:raise AuditBudgetStop(collect())
    step=common.aq(hv)
    for vi in range((lo/hv).numerator,(hi/hv).numerator+1):
        check_time();u=(step*vi).exp();psi=arb(0);tevalues=[];partial_inner=partial_coeff=0
        stage="Laplace_factors"
        for j,(likelihood,weight) in enumerate(zip(A,w)):
            if j%32==0:check_time()
            ee,te,used=exponential_pair(u*likelihood,saturation_threshold,caps)
            psi+=weight*ee;tevalues.append(te);factors+=1;saturated+=used;partial_inner+=1
        dot=arb_mat([tevalues])*moment_matrix
        T0,T1=dot[0,0],dot[0,1]
        P0=[dot[0,2+j] for j in range(p)];P1=[dot[0,2+p+j] for j in range(p)]
        dpre=K*step*psi**(K-1);xpre=K*(K-1)*step*psi**(K-2);cpre=xpre*T0
        stage="coefficient_accumulation"
        for j,(weight,te) in enumerate(zip(w,tevalues)):
            if j%32==0:check_time()
            wt=weight*te;dc[j]+=dpre*wt;cc[j]+=cpre*wt;partial_coeff+=1
        stage="explicit_label_cross"
        for entry in labels:
            check_time();y=entry["y"];q=entry["q"]
            Rb=[T0*y[j]-P0[j if j<p else m-1-j] for j in range(m)]
            Ra=[T1*y[j]-P1[j if j<p else m-1-j] for j in range(m)]
            pref=q*xpre
            for i in range(m):
                for j in range(i,m):
                    crossbb[i][j]+=pref*Rb[i]*Rb[j]
                    crossaa[i][j]+=pref*Ra[i]*Ra[j]
        completed+=1;last=vi;partial_inner=partial_coeff=0
    stage="direct_label_same"
    for entry in labels:
        y=entry["y"];q=entry["q"];same_nodes_current=0
        for k,(x,pi) in enumerate(zip(z,prob)):
            if k%32==0:check_time()
            rr=[y[j]-pi[j if j<p else m-1-j] for j in range(m)]
            coef=q*cc[k];coefaa=coef*x*x
            for i in range(m):
                for j in range(i,m):
                    product=rr[i]*rr[j];samebb[i][j]+=coef*product;sameaa[i][j]+=coefaa*product
            same_nodes_current+=1
        same_labels_completed+=1;same_nodes_current=0
    stage="bias_sigma_diagonal"
    for k,x in enumerate(z):
        if k%32==0:check_time()
        for i in range(m):
            term=classweight*dc[k]*sp[k][i if i<p else m-1-i]
            Bbb[i]+=term;Baa[i]+=term*x*x
        sigma_nodes_completed+=1
    blocks={name:zero(m) for name in BLOCKS}
    for i in range(m):
        for j in range(i,m):
            b=Bbb[i] if i==j else arb(0);ba=Baa[i] if i==j else arb(0)
            blocks["Hbb"][i][j]=blocks["Hbb"][j][i]=b-samebb[i][j]+crossbb[i][j]
            blocks["Haa"][i][j]=blocks["Haa"][j][i]=ba-sameaa[i][j]+crossaa[i][j]
            blocks["Hperp"][i][j]=blocks["Hperp"][j][i]=b-samebb[i][j]
    stage="complete"
    return collect(True,blocks)


def compare(row,saved):
    if row["pair_counts"]!=saved["pair_counts"] or row["status"]!="completed" or saved["status"]!="completed":
        raise ValueError("completed matching count rows required")
    output={}
    for name in BLOCKS:
        left,right=row["matrix_blocks"][name],saved["matrix_blocks"][name]
        if len(left)!=len(right) or not left:raise ValueError("matrix shape mismatch")
        n=len(left);comparisons=[]
        for i in range(n):
            if len(left[i])!=n or len(right[i])!=n:raise ValueError("square matrix required")
            line=[]
            for j in range(n):
                a,b=common.from_record(left[i][j]),common.from_record(right[i][j])
                if not a.overlaps(b):raise ValueError(f"independent matrix nonoverlap {name}[{i},{j}]")
                line.append(dict(overlap=True,difference=common.record(a-b)))
            comparisons.append(line)
        output[name]=comparisons
    return output


def fixed_configuration():
    return dict(selected_counts=[list(c) for c in SELECTED],precision_bits=BITS,K=8,
        hz="1/40",L="11",hv="1/8",vmin="-32",vmax="64",hard_wall_seconds=SECONDS,
        full_endpoint_weights=True,saturation_threshold=128)


def verify_freeze():
    freeze=common.read(FREEZE);sources=freeze.get("sources")
    required=(*SOURCES,"script/certified_teacher_compute.py",INPUTS.relative_to(ROOT).as_posix(),NOMINAL,TEACHER)
    if not isinstance(sources,dict) or any(name not in sources for name in required):
        raise ValueError("FREEZE omits independent audit dependencies/inputs")
    for name,digest in sources.items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or common.sha(path)!=digest:
            raise ValueError("frozen source drift: "+name)
    return freeze


def producer_entries():
    summary=common.read(SUMMARY);entries=[]
    if len(summary["representative_files"])!=41:raise ValueError("fixed all41 producer summary required")
    for entry in summary["representative_files"]:
        path=(ROOT/entry["path"]).resolve()
        if not path.is_relative_to(ROOT) or common.sha(path)!=entry["sha256"] or path.stat().st_size!=entry["bytes"]:
            raise ValueError("producer representative seal drift")
        row=common.read(path)
        if tuple(row["pair_counts"]) in SELECTED:
            if row["status"]!="completed" or row["K"]!=8 or row["precision_bits"]!=BITS:
                raise ValueError("fixed completed producer row required")
            if row["inner_grid"]!=dict(h_exact="1/40",L_exact="11",nodes=881,full_weights=True):
                raise ValueError("producer audit inner grid drift")
            if row["outer_grid"]!=dict(h_exact="1/8",vmin_exact="-32",vmax_exact="64",
                    nodes_prescribed=769,nodes_completed=769,full_weights=True):
                raise ValueError("producer audit outer grid drift")
            if any(len(row["matrix_blocks"][name])!=8 or
                    any(len(line)!=8 for line in row["matrix_blocks"][name]) for name in BLOCKS):
                raise ValueError("producer full8 matrices required")
            entries.append((tuple(row["pair_counts"]),entry,row))
    if len(entries)!=2 or {c for c,_,_ in entries}!=set(SELECTED):raise ValueError("two fixed producer counts required")
    return {c:(entry,row) for c,entry,row in entries}


def bindings(entries):
    paths=(*SOURCES,"script/certified_teacher_compute.py",INPUTS.relative_to(ROOT).as_posix(),
           FREEZE.relative_to(ROOT).as_posix(),SUMMARY.relative_to(ROOT).as_posix(),NOMINAL,TEACHER)
    result={name:common.sha(ROOT/name) for name in paths}
    for entry,_ in entries.values():result[entry["path"]]=common.sha(ROOT/entry["path"])
    return result


def run():
    verify_freeze();entries=producer_entries();locks=bindings(entries)
    # A start lock makes an interrupted/partial audit nonresumable and prevents
    # silently replacing prior output. All scientific files use exclusive x.
    common.write_new(OUT/"AUDIT_START.json",dict(created_utc=common.now(),bindings=locks,
        configuration=fixed_configuration()))
    previous=ctx.prec;ctx.prec=BITS;tick=time.perf_counter();results=[]
    try:
        source=common.read(ROOT/NOMINAL)["center"]["exact_center_dyadic"]
        theta=[Fraction(int(v["numerator"]),int(v["denominator"])) for v in source]
        if len(theta)!=16 or any(theta[:8]) or theta[8:]!=theta[8:][::-1]:raise ValueError("exact paired zero-bias center required")
        a=theta[8:12]
        teacher=common.read(ROOT/TEACHER)
        if teacher["label_mode"]!="all_labels" or len(teacher["pmf_finite_sum"])!=256 or teacher["precision_bits"]!=BITS:
            raise ValueError("original all256 teacher records required")
        q=[common.from_record(v) for v in teacher["pmf_finite_sum"]]
        for count in SELECTED:
            try:
                row=direct_weighted_class(a,count,q,"1/40","11","1/8","-32","64",
                    K=8,precision_bits=BITS,deadline=tick+SECONDS)
            except AuditBudgetStop as error:
                common.write_new(OUT/"AUDIT_PARTIAL.json",dict(created_utc=common.now(),bindings=locks,
                    configuration=fixed_configuration(),completed_replays=results,partial=error.partial))
                raise
            entry,saved=entries[count]
            try:comparisons=compare(row,saved)
            except BaseException as error:
                common.write_new(OUT/"AUDIT_FAILURE.json",dict(created_utc=common.now(),bindings=locks,
                    configuration=fixed_configuration(),completed_replays=results,
                    failed_replay=row,producer_rep=entry,error_type=type(error).__name__,error=str(error)))
                raise
            row.update(producer_rep=entry,comparisons=comparisons);results.append(row)
        result=dict(schema="certified_f8_hessian_independent_replay_v1",created_utc=common.now(),
            bindings=locks,configuration=fixed_configuration(),replays=results,
            replayed_representatives=2,total_producer_representatives=len(common.read(SUMMARY)["representative_files"]),
            all_384_matrix_entries_overlap=True,new_factors=sum(r["resources"]["laplace_factors_evaluated"] for r in results),
            matrix_entries_compared=sum(r["resources"]["matrix_entries_returned"] for r in results),
            wall_seconds=time.perf_counter()-tick,new_random_draws=0,new_optimizer_updates=0,
            scientific_NPZ_reads=0,scope="two fixed representatives only, full same grid, no independent replay of all41")
        if result["new_factors"]!=1354978 or result["matrix_entries_compared"]!=384:raise ArithmeticError("fixed replay resources drift")
        common.write_new(OUT/"AUDIT.json",result)
        return dict(passed=True,new_factors=result["new_factors"],matrix_entries_compared=384,
                    wall_seconds=result["wall_seconds"])
    finally:ctx.prec=previous


def check():
    verify_freeze();entries=producer_entries();saved=common.read(OUT/"AUDIT.json")
    if saved["bindings"]!=bindings(entries) or saved["configuration"]!=fixed_configuration():
        raise ValueError("audit source/input bindings drift")
    if (saved["replayed_representatives"]!=2 or len(saved["replays"])!=2
            or saved["new_factors"]!=1354978 or saved["matrix_entries_compared"]!=384
            or saved["total_producer_representatives"]!=41 or not saved["all_384_matrix_entries_overlap"]):
        raise ValueError("audit accounting drift")
    previous=ctx.prec;ctx.prec=BITS
    try:
        for count,row in zip(SELECTED,saved["replays"]):
            if row["pair_counts"]!=list(count) or row["precision_bits"]!=BITS or row["K"]!=8:
                raise ValueError("audit fixed count/precision drift")
            if (row["stage"]!="complete" or row["saturation_threshold"]!=128
                    or row["producer_W_M_S_helpers_used"] or row["class_weight_division_used"]
                    or row["teacher_or_normal_mass_renormalized"] or row["moment_columns"]!=10
                    or not row["finite_sum_arithmetic_enclosed"]):
                raise ValueError("audit independent backend qualifications drift")
            if row["inner_grid"]!=dict(h_exact="1/40",L_exact="11",nodes=881,full_weights=True):
                raise ValueError("audit inner grid drift")
            if row["outer_grid"]!=dict(h_exact="1/8",vmin_exact="-32",vmax_exact="64",
                                     nodes_prescribed=769,nodes_completed=769,full_weights=True):
                raise ValueError("audit outer grid drift")
            expectedlabels,selected=fold_class([common.from_record(v) for v in common.read(ROOT/TEACHER)["pmf_finite_sum"]],count)
            expected=[dict(y=list(e["y"]),original_labels=e["original_labels"],weight=common.record(e["q"])) for e in expectedlabels]
            if row["original_labels"]!=selected or row["folded_labels"]!=expected:
                raise ValueError("saved original teacher folding drift")
            entry,source=entries[count]
            if row["producer_rep"]!=entry or row["comparisons"]!=compare(row,source):
                raise ValueError("saved overlap comparisons drift")
            if row["resources"]["laplace_factors_evaluated"]!=677489 or row["resources"]["matrix_entries_returned"]!=192:
                raise ValueError("row resource accounting drift")
        return dict(passed=True,new_grid_evaluations=0,new_integrals=0,new_random_draws=0)
    finally:ctx.prec=previous


if __name__=="__main__":
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run",action="store_true");mode.add_argument("--check",action="store_true")
    args=parser.parse_args()
    print(common.json.dumps(run() if args.run else check(),ensure_ascii=False,indent=2))
