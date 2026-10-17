"""Saved-JSON precision analysis; no particle arrays, model calls or RNG."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
import numpy as np
from scipy.stats import t as student

ROOT = Path(__file__).resolve().parents[1]
ALPHA = .3201223434446316
V = np.array([1., 1., 0.]) / math.sqrt(2)
EXPECTED = {(r, n) for r in range(3) for n in (16384, 65536)}
AMPLITUDES = (.1, .2)
SOURCE = "script/learned_precision_analyze.py"


class MissingInput(ValueError):
    """Missing outputs remain scientific slots, never exclusion decisions."""


class InvalidInput(ValueError):
    """Identity conflict invalidates inference, not the scientific hypothesis."""


def expected_cells(reason=None):
    return [{"rep":r,"N":n,"kind":"sym","t":t,"status":"unresolved",
             "errors":[] if reason is None else [str(reason)],
             "direction":"unresolved","independent_precision":"unresolved","same_precision":"unresolved"}
            for r,n in sorted(EXPECTED) for t in AMPLITUDES]


def failed_report(phase, error, *, invalid=False):
    return {"status":"invalid_input" if invalid else "completed_with_unresolved_slots", "phase":phase,
            "errors":[str(error)], "cells":expected_cells(error),
            "summary":{"family_size":36,"expected_cells":12,"available_cells":0,"unresolved_cells":12},
            "passed":False,"all_six_cases_complete":False,"all_coefficients_complete":False,
            "all_predictions_frozen":False,"before_validation_randomization":False,
            "selected_estimator":"none_no_fresh_study","source_failure_is_scientific_counterexample":False,
            "old_1_of_12_unchanged":True}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(type(obj).__name__)


def write(path, obj):
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=jsonable, allow_nan=False)
        f.write("\n")


def summarize(values):
    a = np.asarray(values, float)
    require(a.shape == (16,) and np.isfinite(a).all(), "16 finite complete randomizations required")
    with np.errstate(over="ignore",invalid="ignore"):
        mean,sd=float(a.mean()),float(a.std(ddof=1))
    require(math.isfinite(mean) and math.isfinite(sd), "finite derived mean/SD required")
    return {"replicates": a, "mean":mean,"sample_SD":sd,"SE":sd/4,"R":16}


def covariance(a, b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    require(a.shape==b.shape==(16,) and np.isfinite(a).all() and np.isfinite(b).all(),"paired finite covariance inputs")
    result=float(np.cov(a,b,ddof=1)[0,1]);require(math.isfinite(result),"finite covariance required")
    return result


def interval(stat, family=36):
    require(family==36,"fixed family36 cannot be reduced")
    critical = float(student.ppf(1-.05/(2*family), 15))
    return {"lower": stat["mean"]-critical*stat["SE"], "upper": stat["mean"]+critical*stat["SE"],
            "critical": critical, "df": 15, "family": family, "alpha": .05,
            "method": "approximate_Bonferroni_two_sided_randomization_t"}


def precision(ci):
    if ci["lower"] >= -.01 and ci["upper"] <= .01:
        return "within_tolerance"
    if ci["lower"] > .01 or ci["upper"] < -.01:
        return "outside_tolerance"
    return "unresolved"


def pred_components(coef, kind):
    s = coef["statistics"]
    g, h, b = [np.asarray(s[kind+"_"+name]["replicates"], float) for name in ("g", "H", "B")]
    require(g.shape == (16, 3) and h.shape == b.shape == (16, 3, 3), "full coefficient shapes")
    require(all(np.isfinite(a).all() for a in (g, h, b)), "finite full coefficients")
    linear = (ALPHA-1)*np.einsum("ri,i->r", g, V)
    hh = .5*(ALPHA*ALPHA-1)*np.einsum("i,rij,j->r", V, h, V)
    bb = -(ALPHA-1)*np.einsum("i,rij,j->r", V, b, V)
    return {"linear": linear, "H_quadratic": hh, "B_quadratic": bb, "quadratic": hh+bb}


def predict(coef, kind, amplitude):
    p = pred_components(coef, kind)
    return amplitude*p["linear"]+amplitude**2*p["quadratic"]


def policy_check(policy):
    require(policy.get("schema")=="learned_precision_policy_v1" and policy.get("frozen_before_scientific_development") is True,"frozen precision policy")
    f=policy["primary_family"];s=policy["development_selection"];q=policy["teacher_numerics"]
    require((f["D"],f["independent_normalized_residual"],f["same_bank_normalized_remainder"],f["family_size"],f["alpha"],f["R"],f["df"])
            ==(12,12,12,36,.05,16,15) and f["residual_tolerance"]==[-.01,.01],"fixed complete family36 policy")
    require((s["median_ratio_max"],s["maximum_ratio_max"],s["fixed_control_coefficient"])==(.5,1.,1.)
            and s["all_six_sym_required"] is True and s["no_per_proposal_selection"] is True,"fixed all-six Q variance gate")
    require((q["primary_outer"],q["diagnostic_outer"],q["private_inner"])==(36,28,256)
            and q["normalized_rule_difference_tolerance"]==1e-4,"fixed teacher rules")
    require(policy["maximum_fresh_cycles"]==1 and policy["all_outputs_retained"] is True,"one cycle/all outputs policy")


class Inputs:
    def __init__(self, cfg):
        require(cfg["schema"] == "learned_precision_analysis_contract_v1" and cfg["frozen"] is True, "frozen analysis contract schema")
        self.cfg,self.cache,self.producers,self.common={}, {}, {}, {}
        self.cfg=cfg
        self.missing=set()
        for mapping in (cfg["source_sha256"], cfg["input_sha256"]):
            for name, digest in mapping.items():
                path = self.path(name)
                if not path.is_file():
                    self.missing.add(name);continue
                if sha(path)!=digest:raise InvalidInput("bound file changed: "+name)
        require(cfg["source_sha256"].get(SOURCE) == sha(Path(__file__)), "analyzer must be bound")
        self.policy = self.read(cfg["policy"])
        policy_check(self.policy)
        plan=self.policy["plan"]
        require(cfg["input_sha256"].get(plan)==self.policy["plan_sha256"] and sha(self.path(plan))==self.policy["plan_sha256"],"frozen policy/actual PLAN identity")

    def path(self, name):
        require(isinstance(name,str) and name and not Path(name).is_absolute() and ".." not in Path(name).parts,"repository-relative path required")
        p = (ROOT / name).resolve()
        require(p.is_relative_to(ROOT), "path outside repository")
        return p

    def read(self, name):
        if name not in self.cfg["input_sha256"]:raise InvalidInput("unbound JSON: "+name)
        if name in self.missing or not self.path(name).is_file():raise MissingInput("missing bound JSON: "+name)
        if not name.endswith(".json"):raise InvalidInput("only JSON statistics may be loaded")
        if name not in self.cache:
            self.cache[name] = json.loads(self.path(name).read_text(encoding="utf-8-sig"))
        return self.cache[name]

    def binding(self,name,digest):
        if self.cfg["input_sha256"].get(name)!=digest:raise InvalidInput("input identity mismatch: "+name)

    def producer(self,name,expected_phase=None):
        path=self.path(name)
        root=None
        for parent in path.parents:
            if parent==ROOT:break
            rel=parent.relative_to(ROOT).as_posix()
            if rel+"/CONTRACT.json" in self.cfg["input_sha256"]:
                root=rel;break
        if root is None:raise InvalidInput("no bound producer CONTRACT for "+name)
        if root not in self.producers:
            contract,freeze,manifest=[self.read(root+"/"+basename) for basename in ("CONTRACT.json","FREEZE.json","MANIFEST.json")]
            def need(ok,msg):
                if not ok:raise InvalidInput(root+": "+msg)
            phase=contract.get("phase")
            need(contract.get("schema")=="learned_precision_execution_v1" and phase in ("development","coefficients","validation"),"wrong producer schema/phase")
            need(contract.get("frozen") is True and contract.get("authorized_to_execute") is True,"producer not frozen/authorized")
            need(freeze.get("schema")==contract["schema"] and freeze.get("phase")==phase and freeze.get("before_scientific_calls") is True,"FREEZE identity")
            need(freeze.get("contract_sha256")==self.cfg["input_sha256"][root+"/CONTRACT.json"],"FREEZE contract SHA")
            need(freeze.get("source_sha256")==contract.get("source_sha256") and freeze.get("input_sha256")==contract.get("input_sha256"),"FREEZE source/input closure")
            need(contract.get("baseline_commit")=="1e9ca3374665194c11d2d4a85ffcf945718f24d4" and
                 contract.get("handoff_sha256")=="43fe4a37bf2cfad61f0c584330c4d99c1ae27101bd67a260719867c04366fbe2","current-round provenance")
            need(contract.get("case_ids") in ([0],[1],[2]) and contract.get("training_N")==[16384,65536] and contract.get("proposal_kind")=="sym","fixed producer cases")
            need(contract.get("K")==8 and contract.get("alpha")==ALPHA and contract.get("t_values")==[.1,.2]
                 and np.array_equal(contract.get("direction"),V),"fixed candidates")
            need(contract.get("teacher_policy")=="same_saved_GH36_Cholesky_p0_dp0_and_pt","finite teacher definition")
            if phase=="development":need(contract.get("randomization") is None,"development must reuse old banks")
            else:
                family,role,B=(202610166301,60,256) if phase=="coefficients" else (202610166401,70,512)
                need(contract.get("randomization")=={"family":family,"role":role,"R":16,"B":B,"bits":52},"new numerical stream identity")
            need(manifest.get("manifest_self_excluded") is True,"producer manifest semantics")
            index={}
            for row in manifest["files"]:
                rel=Path(row["path"])
                need(not rel.is_absolute() and ".." not in rel.parts,"unsafe manifest entry")
                full=(Path(root)/rel).as_posix()
                need(full not in index,"duplicate manifest entry");index[full]=row["sha256"]
            for basename in ("CONTRACT.json","FREEZE.json"):
                full=root+"/"+basename;need(index.get(full)==self.cfg["input_sha256"][full],"producer manifest binding "+basename)
            sources=contract["source_sha256"]
            need({"script/learned_precision_run.py","script/learned_precision_core.py","script/learned_local_objective.py"}<=set(sources),"scientific source closure incomplete")
            for source,digest in sources.items():need(index.get(root+"/SEALED_SOURCE/"+source)==digest,"sealed source identity "+source)
            if "producer_source_sha256" in self.cfg:need(sources==self.cfg["producer_source_sha256"],"analysis-declared scientific source closure")
            for field in ("proposal_freeze","qa_index"):
                location=contract[field];digest=contract["input_sha256"].get(location)
                need(self.cfg["input_sha256"].get(location)==digest and digest is not None,"common prerequisite hash "+field)
                pair=(location,digest)
                if field in self.common:need(self.common[field]==pair,"different common "+field)
                else:self.common[field]=pair
            if phase=="validation":
                pred=contract["prediction_freeze"]
                need(pred==self.cfg.get("prediction_freeze") and contract["input_sha256"].get(pred)==self.cfg["input_sha256"].get(pred),"validation did not bind this prediction freeze")
                frozen=self.read(pred)
                for record in frozen.get("coefficient_outputs",[]):
                    for field in ("statistics","banks"):
                        need(contract["input_sha256"].get(record[field])==record[field+"_sha256"],"validation omitted frozen coefficient input")
            self.producers[root]={"root":root,"phase":phase,"contract":contract,"manifest":index}
        info=self.producers[root]
        if expected_phase is not None and info["phase"]!=expected_phase:raise InvalidInput("wrong producer phase for "+name)
        self.manifest_binding(info,name)
        return info

    def manifest_binding(self,info,name,digest=None):
        digest=self.cfg["input_sha256"].get(name) if digest is None else digest
        if digest is None or info["manifest"].get(name)!=digest:raise InvalidInput("file differs from producer manifest: "+name)

    def require_row(self,row):
        if row.get("_error"):raise MissingInput(row["_error"])

    @staticmethod
    def contrast_record(records,t):
        matches=[row for row in records if row.get("t")==t]
        if len(matches)!=1:raise MissingInput("one output required at fixed t="+str(t))
        return matches[0]

    def pairs(self):
        indexed, phases = {}, []
        expected_phase={"development":"development","freeze_predictions":"coefficients","final":"validation"}[self.cfg["phase"]]
        seen_roots=set()
        for name in self.cfg["shard_summaries"]:
            try:
                info=self.producer(name,expected_phase)
                if name!=info["root"]+"/SUMMARY.json" or info["root"] in seen_roots:raise InvalidInput("unique root terminal SUMMARY required")
                seen_roots.add(info["root"])
                summary=self.read(name)
                if summary.get("status")!="completed":raise MissingInput("incomplete shard: "+name)
                if summary.get("phase")!=expected_phase or summary.get("schema")!="learned_precision_execution_v1":raise InvalidInput("summary phase/schema mismatch")
                phases.append({"path":name,"producer_root":info["root"],"budget":summary["budget"]})
                for row in summary["outputs"]:
                    key=(row.get("rep"),row.get("N"))
                    if key not in EXPECTED or row.get("kind")!="sym" or key[0] not in info["contract"]["case_ids"]:raise InvalidInput("unexpected producer case")
                    if key in indexed:raise InvalidInput("duplicate case across summaries")
                    indexed[key]={**row,"_producer_root":info["root"]}
            except MissingInput:
                continue
        rows=[indexed.get((r,n),{"rep":r,"N":n,"kind":"sym","_error":"missing/incomplete producer output for fixed case"}) for r,n in sorted(EXPECTED)]
        return rows,phases

    def coefficient(self,record,*,row=None,bank_role=None,expected_phase=None):
        name=record["statistics"];self.binding(name,record["statistics_sha256"])
        info=self.producer(name,expected_phase)
        doc=self.read(name);phase=info["phase"]
        def need(ok,msg):
            if not ok:raise InvalidInput(name+": "+msg)
        rep,n=doc.get("rep"),doc.get("N")
        need((rep,n) in EXPECTED and doc.get("kind")=="sym" and doc.get("phase")==phase,"coefficient rep/N/kind/phase")
        if row is not None:need((rep,n)==(row["rep"],row["N"]),"coefficient row identity")
        folder=Path(name).parent.as_posix()
        base=info["root"]+f"/rep_{rep}_N{n}_sym"
        if phase=="development":
            if bank_role is None:bank_role="coefficient" if folder.endswith("/COEFFICIENT_BANK") else "validation"
            need(folder==base+"/"+bank_role.upper()+"_BANK","development bank directory")
        else:need(folder==base,"primary coefficient directory")
        need(name==folder+"/COEFFICIENT_STATISTICS.json","coefficient filename")
        need(doc.get("outer_order")==36 and doc.get("inner_order")==256 and doc.get("reference_derivative")=="same_finite_GH_Cholesky_chain_rule","same finite teacher derivative")
        B=256 if (phase=="coefficients" or bank_role=="coefficient") else 512
        for stat in doc["statistics"].values():need(stat.get("R")==16 and stat.get("B")==B,"coefficient R/B metadata")
        for field in ("bank_id","reference_identity","outer_order"):
            if field in record:need(record[field]==doc[field],"coefficient record "+field)
        if "banks" in record:self.binding(record["banks"],record["banks_sha256"]);self.manifest_binding(info,record["banks"])
        ref=self.read(info["root"]+"/REFERENCE_GH36.json");self.manifest_binding(info,info["root"]+"/REFERENCE_GH36.json")
        for field,ref_field in (("reference_identity","identity"),("reference_p0_sha256","p0_sha256"),("reference_dp0_sha256","dp0_sha256")):
            need(doc[field]==ref[ref_field],"teacher reference "+field)
        qa=self.read(info["contract"]["qa_index"])
        for key,t in (("origin",0.),("t0p1",.1),("t0p2",.2)):
            expected=qa["teachers"][str(t)]["36"]
            need(doc["reference_bindings"][key]==expected and ref["source_laws"][key]==expected,"teacher source PMF binding")
            for ext in ("json","npz"):need(info["contract"]["input_sha256"].get(expected[ext])==expected[ext+"_sha256"],"teacher producer input SHA")
        need(doc["reference_identity"]==qa["teachers"]["0.0"]["36"]["npz"]+"#"+qa["teachers"]["0.0"]["36"]["npz_sha256"],"teacher reference identity construction")
        self.verify_bank(folder,doc,info,rep,n,B,bank_role)
        return doc

    def verify_bank(self,folder,coef,info,rep,n,B,bank_role):
        def need(ok,msg):
            if not ok:raise InvalidInput(folder+": "+msg)
        name=folder+"/BANK_INPUT_IDENTITY.json";identity=self.read(name);self.manifest_binding(info,name)
        inputs=identity["inputs"];saved=info["phase"]=="development"
        prefix="saved:" if saved else "new:"
        expected=prefix+hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()
        need(identity.get("reused_old_bank") is saved and identity["bank_id"]==coef["bank_id"]==expected,"actual bank identity digest")
        frozen=self.read(info["contract"]["proposal_freeze"])
        pairs=[p for p in frozen["pairs"] if (p["rep"],p["N"])==(rep,n)]
        need(len(pairs)==1 and frozen.get("all_training_complete") is True,"unique frozen proposal pair")
        pair=pairs[0];need(inputs["proposal"]==pair["proposals"]["sym"],"actual frozen proposal identity")
        proposal=self.read(inputs["proposal"]["path"]);self.binding(inputs["proposal"]["path"],inputs["proposal"]["sha256"])
        for field in ("samples","local","moments","complete","rng"):
            binding=inputs[field]
            if saved:need(info["contract"]["input_sha256"].get(binding["path"])==binding["sha256"],"saved bank binding "+field)
            else:self.manifest_binding(info,binding["path"],binding["sha256"])
        complete=self.read(inputs["complete"]["path"]);rng=self.read(inputs["rng"]["path"])
        self.binding(inputs["complete"]["path"],inputs["complete"]["sha256"]);self.binding(inputs["rng"]["path"],inputs["rng"]["sha256"])
        actual=complete["proposal"]
        for field in ("schema","kind","defense","raw_parameters","parameters","construction","shared_homogeneous_groups"):
            need(actual.get(field)==proposal.get(field),"bank proposal differs: "+field)
        need((actual["source"].get("rep"),actual["source"].get("N"),actual["source"].get("selected_epoch"))==(rep,n,pair["selected_epoch"]),"bank training source")
        need((proposal["source"].get("rep"),proposal["source"].get("N"),proposal["source"].get("epoch"))==(rep,n,pair["selected_epoch"]),"frozen training source")
        if saved:family,role=(202610165301,30) if bank_role=="coefficient" else (202610165401,40)
        else:family,role=(202610166301,60) if info["phase"]=="coefficients" else (202610166401,70)
        need((rng["family"],rng["role"],rng["R"],rng["B"],rng["K"],rng["kind"])==(family,role,16,B,8,"sym"),"actual bank randomization")
        need(len(rng["states"])==16,"complete numerical randomizations")
        for r,state in enumerate(rng["states"]):
            need(state["randomization"]==r and state["entropy"]==[family,role,rep,n,1,r] and state["rng_after"]["bit_generator"]=="PCG64DXSM","actual RNG entropy/index")

    def contrast(self,record,*,row,coefficient):
        name=record["statistics"];self.binding(name,record["statistics_sha256"])
        info=self.producer(name);doc=self.read(name);amp=record["t"]
        def need(ok,msg):
            if not ok:raise InvalidInput(name+": "+msg)
        need(amp in AMPLITUDES and doc.get("t")==amp and doc.get("alpha")==ALPHA and np.array_equal(doc.get("direction"),V),"candidate definition")
        need(doc["bank_id"]==coefficient["bank_id"] and doc["reference_identity"]==coefficient["reference_identity"],"same-bank/same-teacher scalar identity")
        key="t"+str(amp).replace(".","p")
        need(doc["teacher_key"]==key and doc["teacher_weight_sha256"]==coefficient["teacher_weight_sha256"][key],"candidate teacher PMF hash")
        for tag in ("cv_D","q2_D"):
            need(doc["statistics"][tag].get("R")==16 and doc["statistics"][tag].get("B")==512,"candidate R/B")
        rname=record["remainder_statistics"];self.binding(rname,record["remainder_statistics_sha256"]);self.manifest_binding(info,rname)
        rem=self.read(rname)
        need(rem["bank_id"]==doc["bank_id"] and rem["reference_identity"]==doc["reference_identity"]
             and rem["teacher_key"]==key and rem["t"]==amp and rem["alpha"]==ALPHA and np.array_equal(rem["direction"],V),"saved same-bank remainder identity")
        for tag in ("cv","q2"):
            d=np.asarray(doc["statistics"][tag+"_D"]["replicates"],float);p=predict(coefficient,tag,amp)
            need(np.allclose(rem["statistics"][tag+"_prediction"]["replicates"],p,atol=2e-14,rtol=2e-12),"saved same-bank prediction differs")
            need(np.allclose(rem["statistics"][tag+"_normalized_remainder"]["replicates"],(d-p)/amp**2,atol=2e-12,rtol=2e-10),"saved same-bank remainder differs")
        return doc

    def verify_development_decision(self,decision):
        if decision.get("phase")!="development" or decision.get("policy")!=self.cfg["policy"] or decision.get("policy_sha256")!=self.cfg["input_sha256"][self.cfg["policy"]]:
            raise InvalidInput("development decision policy/source mismatch")
        contract_name=decision.get("analysis_contract_path")
        self.binding(contract_name,decision["analysis_contract_sha256"])
        source_contract=self.read(contract_name)
        if (source_contract.get("phase")!="development" or source_contract.get("policy")!=self.cfg["policy"]
                or source_contract["source_sha256"].get(SOURCE)!=decision.get("analysis_source_sha256")):
            raise InvalidInput("development decision originating analysis contract/source")
        original_rows={}
        for name in source_contract["shard_summaries"]:
            self.binding(name,source_contract["input_sha256"][name]);self.producer(name,"development")
            summary=self.read(name)
            if summary.get("status")!="completed":raise MissingInput("development phase unfinished")
            for row in summary["outputs"]:
                key=(row["rep"],row["N"])
                if key in original_rows:raise InvalidInput("duplicate development output")
                original_rows[key]=row
        if set(original_rows)!=EXPECTED:raise MissingInput("development producer schedule incomplete")
        cases=decision.get("cases",[])
        if len(cases)!=6 or {(r.get("rep"),r.get("N")) for r in cases}!=EXPECTED:raise MissingInput("development decision lacks six fixed cases")
        ratios=[]
        for case in cases:
            if case.get("status")!="available":raise MissingInput("development case unresolved; no fresh cycle")
            original=original_rows[case["rep"],case["N"]]
            if case["coefficient_source"]!=original["coefficient_bank"]:raise InvalidInput("development decision coefficient differs from producer")
            coef=self.coefficient(case["coefficient_source"],row=case,bank_role="coefficient",expected_phase="development")
            base,new=pred_components(coef,"cv"),pred_components(coef,"q2")
            v0,v1=np.var(base["quadratic"],ddof=1),np.var(new["quadratic"],ddof=1)
            require(v0>0 and np.isfinite([v0,v1]).all(),"valid development variance")
            ratio=float(v1/v0);ratios.append(ratio)
            if not np.isclose(ratio,case["variance_ratio"],rtol=2e-12,atol=2e-14):raise InvalidInput("development saved variance ratio mismatch")
            for name,values in (("baseline_Q",base["quadratic"]),("q2_Q",new["quadratic"])):
                if not np.allclose(case[name]["replicates"],values,rtol=2e-12,atol=2e-14):raise InvalidInput("development Q input mismatch")
            val=original["validation_bank"]
            vcoef=self.coefficient(val["coefficient"],row=case,bank_role="validation",expected_phase="development")
            for amp in AMPLITUDES:self.contrast(self.contrast_record(val["contrasts"],amp),row=case,coefficient=vcoef)
        remnants=decision.get("same_bank_development",[])
        if (len(remnants)!=12 or {(r.get("rep"),r.get("N"),r.get("t")) for r in remnants}!={(r,n,t) for r,n in EXPECTED for t in AMPLITUDES}
                or any(r.get("status")!="available" for r in remnants)):raise MissingInput("incomplete development output slots")
        passed=np.median(ratios)<=.5 and max(ratios)<=1.
        if not passed:raise MissingInput("recomputed development variance gate did not pass")
        if not np.isclose(decision["median_ratio"],np.median(ratios),rtol=2e-12,atol=2e-14) or not np.isclose(decision["maximum_ratio"],max(ratios),rtol=2e-12,atol=2e-14):raise InvalidInput("development saved aggregate ratio mismatch")
        if decision.get("passed") is not True or decision.get("all_six_cases_complete") is not True or decision.get("selected_estimator")!="q2":raise InvalidInput("development declaration conflicts with recalculated policy")


def development(inputs):
    rows, phases = inputs.pairs()
    ratios, details, remainder_rows = [], [], []
    for row in rows:
        key={"rep":row["rep"],"N":row["N"]}
        try:
            inputs.require_row(row)
            coef = inputs.coefficient(row["coefficient_bank"], row=row, bank_role="coefficient")
            base, new = pred_components(coef,"cv"), pred_components(coef,"q2")
            v0, v1 = float(np.var(base["quadratic"],ddof=1)), float(np.var(new["quadratic"],ddof=1))
            require(v0 > 0 and np.isfinite([v0,v1]).all(), "valid development variance")
            ratio = v1/v0; ratios.append(ratio)
            details.append({**key,"status":"available","coefficient_source":row["coefficient_bank"],
                            "baseline_Q":summarize(base["quadratic"]),"q2_Q":summarize(new["quadratic"]),
                            "variance_ratio":ratio,"baseline_cov_H_B":covariance(base["H_quadratic"],base["B_quadratic"]),
                            "q2_cov_H_B":covariance(new["H_quadratic"],new["B_quadratic"])})
        except InvalidInput:
            raise
        except (MissingInput,KeyError,ValueError,TypeError) as error:
            details.append({**key,"status":"unresolved","errors":[str(error)]})
        for amp in AMPLITUDES:
            cell={**key,"t":amp,"status":"unresolved","inference":"posthoc_development_only_no_confirmatory_gate"}
            try:
                inputs.require_row(row)
                val=row["validation_bank"]
                vcoef=inputs.coefficient(val["coefficient"],row=row,bank_role="validation")
                record=inputs.contrast_record(val["contrasts"],amp)
                d=inputs.contrast(record,row=row,coefficient=vcoef)["statistics"]
                part={}
                for kind in ("cv","q2"):
                    dd=np.asarray(d[kind+"_D"]["replicates"],float);pp=predict(vcoef,kind,amp)
                    part[kind]={"D":summarize(dd),"same_P":summarize(pp),
                                "same_W":summarize((dd-pp)/amp**2),"cov_D_P":covariance(dd,pp)}
                cell.update(status="available",controls=part)
            except InvalidInput:
                raise
            except (MissingInput,KeyError,ValueError,TypeError) as error:
                cell["errors"]=[str(error)]
            remainder_rows.append(cell)
    selection=inputs.policy["development_selection"]
    complete=len(ratios)==6 and all(row["status"]=="available" for row in remainder_rows)
    passed=bool(complete and np.median(ratios)<=selection["median_ratio_max"] and max(ratios)<=selection["maximum_ratio_max"])
    return {"status":"completed" if complete else "completed_with_unresolved_slots","phase":"development","passed":passed,
            "selected_estimator":"q2" if passed else "none_no_fresh_study",
            "all_six_cases_complete":complete,"median_ratio":float(np.median(ratios)) if len(ratios)==6 else None,
            "maximum_ratio":max(ratios) if len(ratios)==6 else None,
            "cases":details,"same_bank_development":remainder_rows,"phases":phases,
            "policy":inputs.cfg["policy"],"policy_sha256":inputs.cfg["input_sha256"][inputs.cfg["policy"]],
            "old_primary_results_reinterpreted":False,"new_scientific_samples":0}


def freeze_predictions(inputs):
    rows, phases = inputs.pairs()
    decision=inputs.read(inputs.cfg["development_decision"])
    inputs.verify_development_decision(decision)
    outputs, predictions=[],[]
    for row in rows:
        key={"rep":row["rep"],"N":row["N"],"kind":"sym"}
        try:
            inputs.require_row(row)
            record=row["coefficient"];coef=inputs.coefficient(record,row=row,bank_role="coefficient");part=pred_components(coef,"q2")
            outputs.append({**key,**record})
            for amplitude in AMPLITUDES:
                predictions.append({**key,"t":amplitude,"alpha":ALPHA,"direction":V.tolist(),"status":"available",
                    "replicates":predict(coef,"q2",amplitude),"bank_id":coef["bank_id"],
                    "reference_identity":coef["reference_identity"],"reference_p0_sha256":coef["reference_p0_sha256"],
                    "reference_dp0_sha256":coef["reference_dp0_sha256"],
                    "linear_replicates":part["linear"],"H_quadratic_replicates":part["H_quadratic"],
                    "B_quadratic_replicates":part["B_quadratic"],"component_scaling":"coefficients_before_t_scaling",
                    "coefficient_statistics":record["statistics"],
                    "coefficient_statistics_sha256":record["statistics_sha256"]})
        except InvalidInput:
            raise
        except (MissingInput,KeyError,ValueError,TypeError) as error:
            predictions.extend({**key,"t":amplitude,"direction":V.tolist(),"status":"unresolved","errors":[str(error)]}
                               for amplitude in AMPLITUDES)
    complete=len(outputs)==6 and len(predictions)==12 and all(p["status"]=="available" for p in predictions)
    return {"schema":"learned_precision_prediction_freeze_v1","status":"completed" if complete else "completed_with_unresolved_slots",
            "all_coefficients_complete":complete,"all_predictions_frozen":complete,"selected_estimator":"q2",
            "component_scaling":"coefficients_before_t_scaling",
            "coefficient_outputs":outputs,"predictions":predictions,"before_validation_randomization":complete,
            "development_decision":inputs.cfg["development_decision"],
            "development_decision_sha256":inputs.cfg["input_sha256"][inputs.cfg["development_decision"]],
            "policy":inputs.cfg["policy"],"policy_sha256":inputs.cfg["input_sha256"][inputs.cfg["policy"]],
            "created_utc":datetime.now(timezone.utc).isoformat(),"phases":phases}


def final_analysis(inputs):
    rows, phases=inputs.pairs();frozen=inputs.read(inputs.cfg["prediction_freeze"])
    require(frozen.get("schema")=="learned_precision_prediction_freeze_v1" and frozen["all_predictions_frozen"]
            and frozen["all_coefficients_complete"] and frozen["before_validation_randomization"]
            and frozen["selected_estimator"]=="q2" and frozen["component_scaling"]=="coefficients_before_t_scaling","prevalidation predictions required")
    if frozen.get("policy")!=inputs.cfg["policy"] or frozen.get("policy_sha256")!=inputs.cfg["input_sha256"][inputs.cfg["policy"]]:raise InvalidInput("prediction policy binding")
    prediction={(p["rep"],p["N"],p["t"]):p for p in frozen["predictions"]}
    require(len(frozen["predictions"])==len(prediction)==12 and set(prediction)=={(r,n,t) for r,n in EXPECTED for t in AMPLITUDES},"all12 unique fixed prediction slots")
    outputs={(r["rep"],r["N"]):r for r in frozen["coefficient_outputs"]}
    require(len(frozen["coefficient_outputs"])==len(outputs)==6 and set(outputs)==EXPECTED,"six unique frozen coefficient outputs")
    cells=[];order=[];aux_errors=[]
    for row in rows:
        for amp in AMPLITUDES:
            cell={"rep":row["rep"],"N":row["N"],"kind":"sym","t":amp,"status":"unresolved",
                  "direction":"unresolved","independent_precision":"unresolved","same_precision":"unresolved"}
            try:
                inputs.require_row(row)
                coef=inputs.coefficient(row["coefficient"],row=row,bank_role="validation",expected_phase="validation")
                r=inputs.contrast_record(row["contrasts"],amp)
                key=(row["rep"],row["N"],amp);pr=prediction[key];source=outputs[key[:2]]
                if (pr.get("kind")!="sym" or pr.get("alpha")!=ALPHA or not np.array_equal(pr.get("direction"),V)
                        or source["statistics"]!=pr["coefficient_statistics"] or source["statistics_sha256"]!=pr["coefficient_statistics_sha256"]):
                    raise InvalidInput("frozen prediction case/candidate/source binding")
                old_coef=inputs.coefficient(source,row=row,bank_role="coefficient",expected_phase="coefficients")
                if old_coef["bank_id"]==coef["bank_id"]:raise InvalidInput("independent prediction unexpectedly reuses validation bank")
                for field in ("reference_identity","reference_p0_sha256","reference_dp0_sha256"):
                    if old_coef[field]!=coef[field] or pr.get(field)!=old_coef[field]:raise InvalidInput("independent/same prediction teacher scalar differs")
                if pr.get("bank_id")!=old_coef["bank_id"]:raise InvalidInput("frozen independent bank identity")
                part=pred_components(old_coef,"q2");pind=predict(old_coef,"q2",amp)
                for field,values in (("replicates",pind),("linear_replicates",part["linear"]),
                                     ("H_quadratic_replicates",part["H_quadratic"]),("B_quadratic_replicates",part["B_quadratic"])):
                    if not np.allclose(pr[field],values,atol=1e-15,rtol=1e-13):raise InvalidInput("frozen prediction arithmetic: "+field)
                drec=inputs.contrast(r,row=row,coefficient=coef)
                d=np.asarray(drec["statistics"]["q2_D"]["replicates"],float)
                psame=predict(coef,"q2",amp)
                wi=(d-pind)/amp**2;ws=(d-psame)/amp**2
                dstat,wistat,wsstat=map(summarize,(d,wi,ws))
                dci,wici,wsci=map(interval,(dstat,wistat,wsstat))
                v_ind=float(np.var(wi,ddof=1));ratio=float(np.var(ws,ddof=1)/v_ind) if v_ind>0 else None
                cell.update({"status":"available",
                          "D":dstat,"independent_prediction":summarize(pind),"same_prediction":summarize(psame),
                          "W_ind":wistat,"W_same":wsstat,"D_interval":dci,"W_ind_interval":wici,"W_same_interval":wsci,
                          "direction":"negative" if dci["upper"]<0 else "positive" if dci["lower"]>0 else "unresolved",
                          "independent_precision":precision(wici),"same_precision":precision(wsci),
                          "same_remainder_excludes_zero":bool(wsci["upper"]<0 or wsci["lower"]>0),
                          "cov_D_Pind":covariance(d,pind),"cov_D_Psame":covariance(d,psame),
                          "same_over_independent_residual_variance":ratio,
                          "variance_ratio_undefined_if_independent_variance_zero":v_ind==0,"source":r})
            except InvalidInput:
                raise
            except (MissingInput,KeyError,ValueError,TypeError) as error:
                cell["errors"]=[str(error)]
            cells.append(cell)
        if (row["rep"],row["N"])==(0,16384):
            try:
                inputs.require_row(row);item=row["GH28_check"]["comparison"]
                inputs.binding(item["statistics"],item["statistics_sha256"])
                info=inputs.producer(item["statistics"],"validation")
                diagnostic=inputs.read(item["statistics"])
                require(len(diagnostic["rows"])==2 and {r["t"] for r in diagnostic["rows"]}==set(AMPLITUDES),"complete two-amplitude rule comparison")
                passed=[]
                for record in diagnostic["rows"]:
                    maxima=[]
                    for field in ("D_normalized_rule_difference","W_same_rule_difference"):
                        stat=record[field];require(stat["R"]==16 and stat["B"]==512,"rule comparison complete bank units")
                        values=summarize(stat["replicates"]);maxima.append(float(np.max(abs(values["replicates"]))))
                    flag=max(maxima)<=1e-4;passed.append(flag)
                    if flag!=record["qualified"] or record["threshold"]!=1e-4:raise InvalidInput("rule qualification arithmetic")
                if all(passed)!=diagnostic["all_qualified"]:raise InvalidInput("rule overall qualification")
                order.append(diagnostic)
            except InvalidInput:
                raise
            except (MissingInput,KeyError,ValueError,TypeError) as error:
                aux_errors.append("GH28 ordinary diagnostic unresolved: "+str(error))
    require(len(cells)==12,"retain whole family")
    counts={key:{value:sum(c[key]==value for c in cells) for value in ("within_tolerance","outside_tolerance","unresolved")}
            for key in ("independent_precision","same_precision")}
    complete=all(cell["status"]=="available" for cell in cells) and not aux_errors
    return {"status":"completed" if complete else "completed_with_unresolved_slots","phase":"final","cells":cells,
            "summary":{"family_size":36,"critical_value":float(student.ppf(1-.05/72,15)),
                       "expected_cells":12,"available_cells":sum(c["status"]=="available" for c in cells),
                       "unresolved_cells":sum(c["status"]!="available" for c in cells),
                       "direction_negative":sum(c["direction"]=="negative" for c in cells),
                       **counts,"same_remainder_excludes_zero":sum(c.get("same_remainder_excludes_zero",False) for c in cells)},
            "rule_order_comparison":order,"auxiliary_errors":aux_errors,"phases":phases,
            "scope":"conditional numerical uncertainty given six frozen proposals; no training-population or risk conclusion",
            "old_1_of_12_unchanged":True}


def selftest():
    import unittest
    import copy

    class MissingFixture:
        def __init__(self,phase):
            self.cfg={"phase":phase,"policy":"policy.json","input_sha256":{"policy.json":"p"},"prediction_freeze":"freeze.json"}
            self.policy={"development_selection":{"median_ratio_max":.5,"maximum_ratio_max":1.}}
            self.frozen={"schema":"learned_precision_prediction_freeze_v1","all_predictions_frozen":True,
                "all_coefficients_complete":True,"before_validation_randomization":True,"selected_estimator":"q2",
                "component_scaling":"coefficients_before_t_scaling","policy":"policy.json","policy_sha256":"p",
                "predictions":[{"rep":r,"N":n,"t":t} for r,n in sorted(EXPECTED) for t in AMPLITUDES],
                "coefficient_outputs":[{"rep":r,"N":n} for r,n in sorted(EXPECTED)]}
        def pairs(self):return ([{"rep":r,"N":n,"kind":"sym","_error":"manufactured missing phase"} for r,n in sorted(EXPECTED)],[])
        def require_row(self,row):raise MissingInput(row["_error"])
        def read(self,name):return self.frozen
        def verify_development_decision(self,decision):pass

    def producer_fixture():
        root="artifacts/_manufactured_precision_only/coefficients_rep0"
        contract={"schema":"learned_precision_execution_v1","phase":"coefficients","frozen":True,"authorized_to_execute":True,
            "baseline_commit":"1e9ca3374665194c11d2d4a85ffcf945718f24d4",
            "handoff_sha256":"43fe4a37bf2cfad61f0c584330c4d99c1ae27101bd67a260719867c04366fbe2",
            "case_ids":[0],"training_N":[16384,65536],"proposal_kind":"sym","K":8,"alpha":ALPHA,"t_values":[.1,.2],
            "direction":V.tolist(),"teacher_policy":"same_saved_GH36_Cholesky_p0_dp0_and_pt",
            "randomization":{"family":202610166301,"role":60,"R":16,"B":256,"bits":52},
            "source_sha256":{name:"a"*64 for name in ("script/learned_precision_run.py","script/learned_precision_core.py","script/learned_local_objective.py")},
            "proposal_freeze":"artifacts/_manufactured_precision_only/proposals.json","qa_index":"artifacts/_manufactured_precision_only/qa.json"}
        contract["input_sha256"]={contract["proposal_freeze"]:"b"*64,contract["qa_index"]:"c"*64}
        digest=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
        freeze={"schema":contract["schema"],"phase":contract["phase"],"before_scientific_calls":True,
                "contract_sha256":digest(contract),"source_sha256":contract["source_sha256"],"input_sha256":contract["input_sha256"]}
        names={root+"/CONTRACT.json":digest(contract),root+"/FREEZE.json":digest(freeze),root+"/STATISTICS.json":"d"*64}
        files=[{"path":Path(k).relative_to(root).as_posix(),"sha256":v} for k,v in names.items()]
        files += [{"path":"SEALED_SOURCE/"+k,"sha256":v} for k,v in contract["source_sha256"].items()]
        manifest={"manifest_self_excluded":True,"files":files}
        docs={root+"/CONTRACT.json":contract,root+"/FREEZE.json":freeze,root+"/MANIFEST.json":manifest}
        inputs=Inputs.__new__(Inputs)
        inputs.cfg={"input_sha256":{**names,root+"/MANIFEST.json":digest(manifest),**contract["input_sha256"]}}
        inputs.common={};inputs.producers={};inputs.cache=docs;inputs.missing=set()
        inputs.read=lambda name:docs[name]
        return inputs,root,docs

    def bank_fixture():
        folder="artifacts/_manufactured_precision_only/validation/rep_0_N16384_sym"
        proposal={"schema":"manufactured_proposal","kind":"sym","defense":.05,"raw_parameters":{"fixed":1},
            "parameters":{"fixed":2},"construction":"reflection_average","shared_homogeneous_groups":3,
            "source":{"rep":0,"N":16384,"epoch":20}}
        complete={"proposal":{**proposal,"source":{"rep":0,"N":16384,"selected_epoch":20}}}
        rng={"family":202610166401,"role":70,"R":16,"B":512,"K":8,"kind":"sym",
             "states":[{"randomization":r,"entropy":[202610166401,70,0,16384,1,r],"rng_after":{"bit_generator":"PCG64DXSM"}} for r in range(16)]}
        bindings={k:{"path":folder+"/BANK/"+n,"sha256":"a"*64} for k,n in (("samples","SAMPLES.npz"),("local","LOCAL_WEIGHTS.npz"),("moments","ANALYTIC_PROPOSAL_MOMENTS.json"),("complete","COMPLETE.json"),("rng","RNG.json"))}
        bindings["proposal"]={"path":"artifacts/_manufactured_precision_only/proposal.json","sha256":"b"*64}
        identity={"inputs":bindings,"reused_old_bank":False,
                  "bank_id":"new:"+hashlib.sha256(json.dumps(bindings,sort_keys=True).encode()).hexdigest()}
        freeze_name="artifacts/_manufactured_precision_only/proposals.json"
        docs={folder+"/BANK_INPUT_IDENTITY.json":identity,bindings["complete"]["path"]:complete,bindings["rng"]["path"]:rng,
              bindings["proposal"]["path"]:proposal,freeze_name:{"all_training_complete":True,"pairs":[{"rep":0,"N":16384,"selected_epoch":20,"proposals":{"sym":bindings["proposal"]}}]}}
        inputs=Inputs.__new__(Inputs);inputs.read=lambda name:docs[name]
        inputs.binding=lambda name,digest:None;inputs.manifest_binding=lambda *args:None
        info={"phase":"validation","contract":{"proposal_freeze":freeze_name}}
        return inputs,folder,{"bank_id":identity["bank_id"]},info,docs,rng,complete

    def decision_fixture(scale=.5):
        fake=Inputs.__new__(Inputs)
        fake.cfg={"policy":"policy.json","input_sha256":{"policy.json":"policy_sha"}}
        docs={};coefs={};cases=[];rows=[]
        u=np.arange(16,dtype=float)/20+1
        for rep,n in sorted(EXPECTED):
            name=f"coef_{rep}_{n}.json";record={"statistics":name,"statistics_sha256":"fixture"}
            stats={}
            for tag,factor in (("cv",1.),("q2",scale)):
                for field,values in (("g",np.zeros((16,3))),("H",factor*u[:,None,None]*np.eye(3)),("B",np.zeros((16,3,3)))):
                    stats[tag+"_"+field]={"replicates":values}
            coef={"statistics":stats};coefs[name]=coef
            b,q=pred_components(coef,"cv")["quadratic"],pred_components(coef,"q2")["quadratic"]
            cases.append({"rep":rep,"N":n,"status":"available","coefficient_source":record,
                          "baseline_Q":summarize(b),"q2_Q":summarize(q),"variance_ratio":scale**2})
            rows.append({"rep":rep,"N":n,"coefficient_bank":record,
                         "validation_bank":{"coefficient":record,"contrasts":[{"t":t} for t in AMPLITUDES]}})
        source={"phase":"development","policy":"policy.json","source_sha256":{SOURCE:"source_sha"},
                "shard_summaries":["summary.json"],"input_sha256":{"summary.json":"summary_sha"}}
        docs["contract.json"]=source;docs["summary.json"]={"status":"completed","outputs":rows}
        fake.read=lambda name:docs[name];fake.binding=lambda *args:None;fake.producer=lambda *args:None
        fake.coefficient=lambda record,**kwargs:coefs[record["statistics"]];fake.contrast=lambda *args,**kwargs:None
        decision={"phase":"development","policy":"policy.json","policy_sha256":"policy_sha",
            "analysis_contract_path":"contract.json","analysis_contract_sha256":"contract_sha","analysis_source_sha256":"source_sha",
            "cases":cases,"same_bank_development":[{"rep":r,"N":n,"t":t,"status":"available"} for r,n in sorted(EXPECTED) for t in AMPLITUDES],
            "passed":True,"all_six_cases_complete":True,"selected_estimator":"q2","median_ratio":scale**2,"maximum_ratio":scale**2}
        return fake,decision

    class Tests(unittest.TestCase):
        def test_paired_cancellation(self):
            a=np.arange(16,dtype=float);b=a+.002
            self.assertLess(summarize(a-b)["SE"],1e-15)
            self.assertGreater(summarize(a)["SE"],1)
        def test_variance_covariance(self):
            a=np.arange(16,dtype=float);b=a[::-1]**2
            self.assertAlmostEqual(np.var(a-b,ddof=1),np.var(a,ddof=1)+np.var(b,ddof=1)-2*covariance(a,b))
        def test_precision_boundaries(self):
            self.assertEqual(precision({"lower":-.01,"upper":.01}),"within_tolerance")
            self.assertEqual(precision({"lower":-.02,"upper":.0}),"unresolved")
            self.assertEqual(precision({"lower":.011,"upper":.02}),"outside_tolerance")
        def test_full_matrix_orientation(self):
            g=np.tile([1.,2.,3.],(16,1));h=np.tile(np.diag([4.,5.,6.]),(16,1,1));b=np.tile([[1.,2.,0.],[3.,4.,0.],[0.,0.,5.]],(16,1,1))
            r={"statistics":{"q2_"+n:{"replicates":v} for n,v in zip(("g","H","B"),(g,h,b))}}
            amp=.2;expected=(ALPHA-1)*amp*3/math.sqrt(2)+amp**2*(.5*(ALPHA**2-1)*4.5-(ALPHA-1)*5)
            np.testing.assert_allclose(predict(r,"q2",amp),expected,rtol=1e-14)
        def test_family_never_unadjusted(self):
            ci=interval(summarize(np.arange(16,dtype=float)))
            self.assertEqual(ci["family"],36);self.assertGreater(ci["critical"],3.7)
        def test_reject_wrong_units(self):
            with self.assertRaises(ValueError):summarize(np.arange(512))
            with self.assertRaises(ValueError):summarize(np.full(16,np.nan))
        def test_missing_development_retains_six_and_twelve_and_never_passes(self):
            result=development(MissingFixture("development"))
            self.assertFalse(result["passed"]);self.assertFalse(result["all_six_cases_complete"])
            self.assertEqual(len(result["cases"]),6);self.assertEqual(len(result["same_bank_development"]),12)
            self.assertIsNone(result["median_ratio"])
        def test_missing_freeze_retains_twelve_and_never_declares_frozen(self):
            fixture=MissingFixture("freeze_predictions");fixture.cfg["development_decision"]="decision.json"
            fixture.cfg["input_sha256"]["decision.json"]="d"
            result=freeze_predictions(fixture)
            self.assertFalse(result["all_predictions_frozen"]);self.assertFalse(result["before_validation_randomization"])
            self.assertEqual(len(result["predictions"]),12)
            self.assertTrue(all(row["direction"]==V.tolist() for row in result["predictions"]))
        def test_missing_final_retains_twelve_family36_and_unresolved_auxiliary(self):
            result=final_analysis(MissingFixture("final"))
            self.assertEqual(len(result["cells"]),12);self.assertEqual(result["summary"]["family_size"],36)
            self.assertEqual(result["summary"]["unresolved_cells"],12)
            self.assertTrue(result["auxiliary_errors"])
        def test_identity_failure_is_not_scientific_counterexample(self):
            result=failed_report("final","SHA mismatch",invalid=True)
            self.assertEqual(len(result["cells"]),12);self.assertEqual(result["summary"]["family_size"],36)
            self.assertFalse(result["source_failure_is_scientific_counterexample"])
        def test_producer_full_contract_freeze_and_source_manifest_chain(self):
            inputs,root,docs=producer_fixture()
            self.assertEqual(inputs.producer(root+"/STATISTICS.json","coefficients")["phase"],"coefficients")
            inputs,root,docs=producer_fixture();docs[root+"/FREEZE.json"]["contract_sha256"]="0"*64
            with self.assertRaises(InvalidInput):inputs.producer(root+"/STATISTICS.json")
        def test_wrong_producer_stream_rejected_even_with_bound_metadata(self):
            inputs,root,docs=producer_fixture();docs[root+"/CONTRACT.json"]["randomization"]["family"]+=1
            with self.assertRaises(InvalidInput):inputs.producer(root+"/STATISTICS.json")
        def test_actual_bank_seed_and_training_epoch_semantics(self):
            inputs,folder,coef,info,docs,rng,complete=bank_fixture()
            inputs.verify_bank(folder,coef,info,0,16384,512,"validation")
            rng["states"][9]["entropy"][0]+=1
            with self.assertRaises(InvalidInput):inputs.verify_bank(folder,coef,info,0,16384,512,"validation")
        def test_actual_bank_proposal_change_rejected(self):
            inputs,folder,coef,info,docs,rng,complete=bank_fixture()
            complete["proposal"]=copy.deepcopy(complete["proposal"]);complete["proposal"]["parameters"]["fixed"]=3
            with self.assertRaises(InvalidInput):inputs.verify_bank(folder,coef,info,0,16384,512,"validation")
        def test_policy_gate_and_family_cannot_silently_change(self):
            # Policy checks deliberately inspect actual values rather than a
            # caller's assertion that a decision passed.
            policy={"schema":"learned_precision_policy_v1","frozen_before_scientific_development":True,
                "primary_family":{"D":12,"independent_normalized_residual":12,"same_bank_normalized_remainder":12,
                    "family_size":36,"alpha":.05,"R":16,"df":15,"residual_tolerance":[-.01,.01]},
                "development_selection":{"median_ratio_max":.5,"maximum_ratio_max":1.,"fixed_control_coefficient":1.,
                    "all_six_sym_required":True,"no_per_proposal_selection":True},
                "teacher_numerics":{"primary_outer":36,"diagnostic_outer":28,"private_inner":256,"normalized_rule_difference_tolerance":1e-4},
                "maximum_fresh_cycles":1,"all_outputs_retained":True}
            policy_check(policy);policy["development_selection"]["median_ratio_max"]=.8
            with self.assertRaises(ValueError):policy_check(policy)
            with self.assertRaises(ValueError):interval(summarize(np.arange(16)),family=12)
        def test_development_decision_recomputed_not_trusted(self):
            fake,decision=decision_fixture(.5);fake.verify_development_decision(decision)
            fake,decision=decision_fixture(2.)
            with self.assertRaises(MissingInput):fake.verify_development_decision(decision)
            fake,decision=decision_fixture(.5);decision["cases"][0]["variance_ratio"]=.1
            with self.assertRaises(InvalidInput):fake.verify_development_decision(decision)
            fake,decision=decision_fixture(.5);decision["same_bank_development"].pop()
            with self.assertRaises(MissingInput):fake.verify_development_decision(decision)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


def main():
    p=argparse.ArgumentParser();p.add_argument("--contract",type=Path);p.add_argument("--out",type=Path)
    p.add_argument("--check",action="store_true");p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if args.self_test:return selftest()
    require(args.contract is not None,"contract required")
    if not args.check:require(args.out is not None and not args.out.exists(),"fresh output JSON required")
    cfg={};result=None
    try:
        cfg=json.loads(args.contract.read_text(encoding="utf-8-sig"));inputs=Inputs(cfg)
        fn={"development":development,"freeze_predictions":freeze_predictions,"final":final_analysis}[cfg["phase"]]
        result=fn(inputs)
    except InvalidInput as error:
        result=failed_report(cfg.get("phase","unknown"),error,invalid=True)
    except (MissingInput,OSError,KeyError,ValueError,TypeError) as error:
        result=failed_report(cfg.get("phase","unknown"),error,invalid=not isinstance(error,MissingInput))
    result["analysis_contract_sha256"]=sha(args.contract)
    result["analysis_contract_path"]=args.contract.resolve().relative_to(ROOT).as_posix()
    result["analysis_source_sha256"]=sha(Path(__file__));result["no_scientific_kernels_or_RNG"]=True
    if args.check:
        print(json.dumps({"passed":result.get("status")=="completed","status":result.get("status"),
            "errors":result.get("errors",[]),"summary":result.get("summary"),
            "full_provenance_and_saved_arithmetic_checked":True,"no_scientific_kernels_or_RNG":True},default=jsonable))
        return 0 if result.get("status")=="completed" else 1
    write(args.out,result)
    print(json.dumps({"output":str(args.out),"sha256":sha(args.out),"status":result.get("status","frozen"),
                      "passed":result.get("passed"),"summary":result.get("summary")},default=jsonable))
    return 0 if result.get("status")=="completed" else 1


if __name__=="__main__":
    raise SystemExit(main())
