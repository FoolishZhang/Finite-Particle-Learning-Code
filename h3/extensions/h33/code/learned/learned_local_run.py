"""Frozen, fresh-directory phases for the learned local proposal study.

Imports do not execute science. --check validates schemas and file identities.
No historical run/data generator/gate is called. Caller contracts own every
random role, case, numerical level, work counter and stopping limit.
"""
from __future__ import annotations
import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
import scipy
from scipy.stats import qmc
import learned_local_objective as objective
from learned_local_proposal import LocalProposal, TrainingFailure, fit_path, select_checkpoint, simulate_local

SCHEMA = "learned_local_execution_v1"
REPO = Path(__file__).resolve().parents[1]
PHASES = ("qa", "training", "profile", "coefficients", "validation")
BASE_SOURCES = {"script/learned_local_run.py", "script/learned_local_proposal.py", "script/learned_local_objective.py",
                "script/competition_core.py", "script/active_reference_likelihood.py", "script/group_posterior_objective.py",
                "script/signal_proposal_core.py", "script/informative_projection.py"}
LOCAL_COUNTERS = {"local_likelihood_x","local_private_node_terms"}
BANK_COUNTERS = LOCAL_COUNTERS | {"proposal_density_values","qmc_uniform_scalars","qmc_nets","posterior_coordinates"}
REQUIRED_COUNTERS = {
    "qa": LOCAL_COUNTERS | {"pmf_law_calls","pmf_private_node_terms"},
    "training": {"simulated_local_rows","active_normal_scalars","private_normal_scalars","label_uniform_scalars","item_uniform_scalars",
                 "em_state_updates","em_row_visits","training_loglikelihood_values","validation_density_values"},
    "profile": BANK_COUNTERS | {"anchor_calls","anchor_particle_pattern_terms"},
    "coefficients": BANK_COUNTERS | {"anchor_calls","anchor_particle_pattern_terms","density_diagnostic_rules"},
    "validation": BANK_COUNTERS | {"candidate_calls","candidate_particle_pattern_terms"},
}


def require(test, message):
    if not test:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=serial, allow_nan=False)
        stream.write("\n")


def npz(path, values):
    with Path(path).open("xb") as stream:
        np.savez(stream, **values)


def resolve(name):
    path = Path(name)
    require(not path.is_absolute() and ".." not in path.parts, "safe repository-relative identity required")
    return REPO/path


def relative(path):
    return Path(path).resolve().relative_to(REPO).as_posix()


def bundle(prefix, values):
    prefix = Path(prefix)
    arrays = {k:v for k,v in values.items() if isinstance(v, np.ndarray)}
    metadata = {k:v for k,v in values.items() if not isinstance(v, np.ndarray)}
    npz(prefix.with_suffix(".npz"), arrays)
    write(prefix.with_suffix(".json"), metadata)
    return {"json": relative(prefix.with_suffix(".json")), "npz": relative(prefix.with_suffix(".npz")),
            "json_sha256": sha(prefix.with_suffix(".json")), "npz_sha256": sha(prefix.with_suffix(".npz"))}


def load_bundle(record, cfg):
    for name in ("json", "npz"):
        require(cfg["input_sha256"].get(record[name]) == record[name+"_sha256"], "unbound bundle input")
    result = read(resolve(record["json"]))
    with np.load(resolve(record["npz"]), allow_pickle=False) as arrays:
        result.update({k:arrays[k] for k in arrays.files})
    return result


class Budget:
    def __init__(self, cfg, out):
        self.caps = cfg["caps"]
        self.resources = cfg.get("resources", {})
        self.counts = {k:0 for k in self.caps if k != "seconds"}
        self.started = time.perf_counter()
        self.out = out
        self.maximum_rss = 0
        self.last_disk_check = 0.
        self.output_bytes = 0

    def reserve(self, name, amount):
        require(name in self.counts, "counter not declared by frozen contract: "+name)
        require(isinstance(amount, (int,np.integer)) and amount >= 0, "nonnegative integer work amount required")
        require(self.counts[name]+int(amount) <= self.caps[name], "frozen counter cap reached: "+name)

    def add(self, name, amount):
        self.reserve(name, amount)
        self.counts[name] += int(amount)

    def check(self):
        require(time.perf_counter()-self.started < self.caps["seconds"], "frozen wall cap reached")
        if self.resources.get("max_rss_bytes"):
            import psutil
            rss=psutil.Process().memory_info().rss
            self.maximum_rss=max(self.maximum_rss,rss)
            require(rss <= self.resources["max_rss_bytes"], "frozen RSS monitor limit reached")
        if self.resources.get("max_output_bytes") and time.perf_counter()-self.last_disk_check >= 2:
            self.output_bytes=sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file())
            self.last_disk_check=time.perf_counter()
            require(self.output_bytes <= self.resources["max_output_bytes"], "frozen output monitor limit reached")

    def snapshot(self):
        return {"counts":dict(self.counts), "caps":self.caps, "seconds":time.perf_counter()-self.started,
            "maximum_observed_rss_bytes":self.maximum_rss,"last_observed_output_bytes":self.output_bytes,
            "monitor_scope":"checks at phase/kernel boundaries; output checks no more often than2seconds; notOS hard quotas"}


def validate(cfg):
    require(cfg["schema"] == SCHEMA and cfg["phase"] in PHASES, "declared learned-local phase required")
    require(cfg.get("frozen") is True and cfg.get("authorized_to_execute") is True, "frozen authorized contract required")
    require(isinstance(cfg["baseline_commit"],str) and len(cfg["baseline_commit"])==40, "baseline commit identity required")
    require(sha(REPO/"HANDOFF.md") == cfg["handoff_sha256"], "current feedback differs")
    require(BASE_SOURCES <= set(cfg["source_sha256"]), "producer/proposal/objective source identities required")
    for field in ("source_sha256","input_sha256"):
        for name,digest in cfg[field].items():
            require(sha(resolve(name)) == digest, "bound file changed: "+name)
    runtime=cfg["runtime"]
    require(runtime["python"]==sys.version.split()[0] and runtime["numpy"]==np.__version__ and runtime["scipy"]==scipy.__version__, "runtime identity differs")
    require(all(os.environ.get(k)=="1" for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS")), "single-thread numeric libraries required")
    require(cfg["caps"].get("seconds",0)>0 and all(isinstance(v,(int,float)) and not isinstance(v,bool) and np.isfinite(v) and v>0 for v in cfg["caps"].values()), "positive finite explicit caps required")
    require(cfg["case_ids"] and len(set(cfg["case_ids"]))==len(cfg["case_ids"]) and set(cfg["case_ids"])<=set(range(3)), "unique training repeat IDs required")
    phase=cfg["phase"]
    require(REQUIRED_COUNTERS[phase] <= set(cfg["caps"]), "complete phase work-counter caps required")
    if phase=="qa":
        require(cfg["qa"]["outer_orders"]==[192,256,384] and cfg["qa"]["teacher_orders"]==[28,36] and cfg["qa"]["teacher_t"]==[0.,.1,.2], "fixed QA rules required")
        require(set(cfg["qa"]["tolerances"])=={"mass","moments","local_ll","pmf","origin_derivative","identity"}, "complete QA tolerance declarations required")
    if phase=="training":
        require(cfg["case_ids"]==[0,1,2], "all3 training repeats in this phase required")
        require(cfg["training"]=={"highN":65536,"lowN":16384,"validationN":32768,"epochs":80,"checkpoints":[0,5,20,80],"defense":.05,"seed_family":202610165201}, "fixed finite training/selection policy required")
        require(cfg["qa_index"] in cfg["input_sha256"],"successful QA must be bound before physical training generation")
        require(read(resolve(cfg["qa_index"])).get("passed") is True,"QA must pass before training")
    if phase in ("profile","coefficients","validation"):
        for key in ("qa_index","proposal_freeze"):
            require(cfg[key] in cfg["input_sha256"], "unbound stage prerequisite: "+key)
        frozen=read(resolve(cfg["proposal_freeze"]))
        require(frozen.get("all_training_complete") is True and len(frozen["pairs"])==6 and
                {(p["rep"],p["N"]) for p in frozen["pairs"]}=={(r,n) for r in range(3) for n in (16384,65536)}, "all six proposals must freeze before evaluation")
        qa_index=read(resolve(cfg["qa_index"]))
        require(qa_index.get("passed") is True,"successful QA required")
        bundles=[*qa_index["levels"].values(),qa_index["private_diagnostic"],qa_index["population_origin"]]
        bundles += [record for orders in qa_index["teachers"].values() for record in orders.values()]
        for record in bundles:
            for kind in ("json","npz"):
                require(cfg["input_sha256"].get(record[kind])==record[kind+"_sha256"],"all QA numerical inputs must be bound before RNG")
        for pair in frozen["pairs"]:
            require(pair["defense"]==.05 and pair["selected_epoch"] in (0,5,20,80),"declared defense and allowed checkpoint required")
            require(set(pair["proposals"])=={"raw","sym"},"paired raw/sym records required")
            for kind,record in pair["proposals"].items():
                require(cfg["input_sha256"].get(record["path"])==record["sha256"],"all frozen proposal records must be bound before RNG")
                proposal_record=read(resolve(record["path"]))
                require(proposal_record["raw_parameters"]==pair["raw_parameters"] and proposal_record["kind"]==kind and proposal_record["defense"]==pair["defense"],"proposal freeze and saved record disagree")
        expected={"profile":(202610165299,29,2,64),"coefficients":(202610165301,30,16,256),"validation":(202610165401,40,16,512)}[phase]
        q=cfg["randomization"]
        require((q["family"],q["role"],q["R"],q["B"])==expected and q["bits"]==52 and cfg["K"]==8, "fixed phase numerical stream/budget required")
        require(isinstance(cfg["pattern_chunk"],int) and 1<=cfg["pattern_chunk"]<=64, "bounded explicit pattern_chunk required")
        if phase=="profile": require(cfg["case_ids"]==[0], "resource profile fixed to repeat0/low/sym")
        if phase=="validation":
            require(cfg["coefficient_freeze"] in cfg["input_sha256"], "full coefficient freeze required")
            coefficient=read(resolve(cfg["coefficient_freeze"]))
            require(coefficient.get("all_coefficients_complete") is True and len(coefficient["outputs"])==12 and
                    {(p["rep"],p["N"],p["kind"]) for p in coefficient["outputs"]}=={(r,n,k) for r in range(3) for n in (16384,65536) for k in ("raw","sym")}, "all12 coefficient outputs must freeze before validation")
            for record in coefficient["outputs"]:
                for name in ("statistics","banks"):
                    require(cfg["input_sha256"].get(record[name])==record[name+"_sha256"],"coefficient file not bound before validation RNG")
            require(cfg["alpha"]==objective.ALPHA and cfg["sym_t"]==[.1,.2] and cfg["raw_t"]==[-.2,-.1,.1,.2], "predeclared candidates only")
    return cfg


def freeze(path,out):
    cfg=validate(read(path))
    require(out.resolve().is_relative_to(REPO),"phase output must be under repository for immutable relative bundle references")
    out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(path,out/"CONTRACT.json")
    for name,digest in cfg["source_sha256"].items():
        dest=out/"SEALED_SOURCE"/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(resolve(name),dest);require(sha(dest)==digest,"source changed during copy")
    write(out/"FREEZE.json",{"schema":SCHEMA,"phase":cfg["phase"],"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "contract_sha256":sha(path),"source_sha256":cfg["source_sha256"],"input_sha256":cfg["input_sha256"],"before_scientific_calls":True})
    return cfg,Budget(cfg,out)


def manifest(out):
    write(out/"MANIFEST.json",{"files":[{"path":p.relative_to(out).as_posix(),"sha256":sha(p),"bytes":p.stat().st_size}
        for p in sorted(out.rglob("*")) if p.is_file()],"manifest_self_excluded":True})


def qa(cfg,budget,out):
    levels={};laws={};index={"levels":{},"teachers":{}};checks=[];tol=cfg["qa"]["tolerances"]
    def gate(label,value,kind):
        checks.append({"name":label,"value":float(value),"tolerance":tol[kind],"passed":bool(value<=tol[kind])})
    for order in cfg["qa"]["outer_orders"]:
        level=objective.channel_level(order,inner_order=256,budget=budget);levels[order]=level
        index["levels"][str(order)]=bundle(out/f"CHANNEL_outer{order}_inner256",level)
        gate(f"channel_mass_{order}",abs(level["mass"]-1),"mass")
        gate(f"reflection_mass_{order}",level["reflection_mass_error"],"identity")
        gate(f"reflection_mean_{order}",level["reflection_mean_error"],"identity")
        gate(f"reflection_m2_{order}",np.max(abs(level["second_moment"]-level["second_moment"][::-1])),"identity")
    private=objective.channel_level(256,inner_order=192,budget=budget)
    index["private_diagnostic"]=bundle(out/"CHANNEL_outer256_inner192",private)
    for name in ("normalizer","mean","second_moment"):
        for lo,hi in ((192,256),(256,384)):
            gate(f"outer_{name}_{lo}_{hi}",np.max(abs(levels[hi][name]-levels[lo][name])),"moments")
        gate("private_"+name,np.max(abs(levels[256][name]-private[name])),"moments")
    anchors=np.array([-8.,-4.,-2.,0.,2.,4.,8.]);ll={}
    for inner in (192,256):
        ll[inner]=objective.channel_loglike(anchors[:,None],np.arange(18)[None,:],inner_order=inner,budget=budget)
    bundle(out/"LOCAL_LIKELIHOOD_CHECK",{"anchors":anchors,"inner192":ll[192],"inner256":ll[256]})
    gate("local_likelihood_private_order",np.max(abs(ll[192]-ll[256])),"local_ll")
    gate("local_likelihood_reflection",np.max(abs(ll[256]-ll[256][::-1,::-1])),"identity")
    origin=objective.population_origin(levels[384]);index["population_origin"]=bundle(out/"POPULATION_ORIGIN",origin)
    gate("population_origin_raw_mass",abs(float(origin["weights"].sum())-1),"mass")
    for t in cfg["qa"]["teacher_t"]:
        index["teachers"][str(t)]={}
        for order in cfg["qa"]["teacher_orders"]:
            budget.reserve("pmf_private_node_terms",(order+order**2+order**3)*256)
            law=objective.teacher_law(t,outer_order=order,budget=budget);laws[t,order]=law
            index["teachers"][str(t)][str(order)]=bundle(out/f"TEACHER_t{str(t).replace('.','p')}_outer{order}",law)
            gate(f"teacher_mass_{t}_{order}",abs(law["mass"]-1),"mass")
        gate(f"teacher_PMF_order_{t}",np.max(abs(laws[t,36]["joint_pmf"]-laws[t,28]["joint_pmf"])),"pmf")
    p0=origin["weights"].reshape(18,18,18)
    grad=(origin["weights"][:,None]*origin["teacher_score"]).T.reshape(3,18,18,18)
    for order in (28,36):
        gate(f"origin_product_PMF_{order}",np.max(abs(laws[0.,order]["joint_pmf"]-p0)),"pmf")
        gate(f"origin_derivative_paths_{order}",np.max(abs(laws[0.,order]["pmf_gradient"]-grad)),"origin_derivative")
    bundle(out/"ORIGIN_PATH_COMPARISON",{"channel_product_PMF":p0,"channel_product_gradient":grad})
    write(out/"QA_CHECKS.json",{"checks":checks,"passed":all(r["passed"] for r in checks),"ordinary_order_checks_not_integral_certificates":True})
    require(all(r["passed"] for r in checks),"numerical QA gate failed; stop without tolerance change")
    index["passed"]=True;write(out/"QA_INDEX.json",index)
    return {"checks":checks,"qa_index":relative(out/"QA_INDEX.json")}


def simulated_dataset(cfg,budget,rep,role,N,destination):
    destination.mkdir(exist_ok=False);family=cfg["training"]["seed_family"]
    rngs=[np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([family,role,rep,c]))) for c in range(4)]
    before=[copy.deepcopy(r.bit_generator.state) for r in rngs]
    costs={"simulated_local_rows":N,"active_normal_scalars":N,"private_normal_scalars":8*N,"label_uniform_scalars":N,"item_uniform_scalars":8*N}
    for name,value in costs.items():budget.reserve(name,value)
    write(destination/"RNG_BEFORE.json",{"family":family,"role":role,"rep":rep,"states":before,
        "entropy":[[family,role,rep,c] for c in range(4)]})
    partial={}
    class RecordedStream:
        def __init__(self,rng,name,counter):self.rng,self.name,self.counter=rng,name,counter
        def draw(self,kind,shape):
            value=getattr(self.rng,kind)(shape);partial[self.name]=value
            budget.add(self.counter,value.size);return value
        def standard_normal(self,shape):return self.draw("standard_normal",shape)
        def random(self,shape):return self.draw("random",shape)
    streams=[RecordedStream(rng,name,counter) for rng,name,counter in zip(rngs,
        ("x","private_z","label_u","item_u"),("active_normal_scalars","private_normal_scalars","label_uniform_scalars","item_uniform_scalars"))]
    try:
        data=simulate_local(N,active_rng=streams[0],private_rng=streams[1],label_rng=streams[2],item_rng=streams[3])
        npz(destination/"PHYSICAL_DATA.npz",data);budget.add("simulated_local_rows",N)
        write(destination/"RNG.json",{"family":family,"role":role,"rep":rep,"component_names":["active","private","label","item"],
            "entropy":[[family,role,rep,c] for c in range(4)],"before":before,"after":[r.bit_generator.state for r in rngs],"counts":costs})
        budget.check();return data
    except Exception:
        npz(destination/"FAILED_PARTIAL_PHYSICAL_DRAWS.npz",partial)
        write(destination/"FAILED_GENERATION.json",{"traceback":traceback.format_exc(),"before":before,
            "after":[r.bit_generator.state for r in rngs],"budget":budget.snapshot(),"automatic_retry":False})
        raise


def training(cfg,budget,out):
    spec=cfg["training"];pairs=[]
    for rep in cfg["case_ids"]:
        folder=out/f"rep_{rep}";folder.mkdir()
        train=simulated_dataset(cfg,budget,rep,10,spec["highN"],folder/"TRAIN")
        valid=simulated_dataset(cfg,budget,rep,20,spec["validationN"],folder/"VALIDATION")
        for N in (spec["lowN"],spec["highN"]):
            dest=folder/f"N_{N}";dest.mkdir()
            prefix={k:v[:N] for k,v in train.items()}
            write(dest/"TRAIN_PREFIX.json",{"N":N,"source":relative(folder/"TRAIN/PHYSICAL_DATA.npz"),"source_sha256":sha(folder/"TRAIN/PHYSICAL_DATA.npz"),
                "array_prefix_sha256":{k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in prefix.items()},
                "low_is_same_high_prefix":True,"validation_shared_with_other_scale_in_same_repeat":True})
            costs={"em_state_updates":18*spec["epochs"],"em_row_visits":N*spec["epochs"],"training_loglikelihood_values":N*(spec["epochs"]+1)}
            for name,value in costs.items():budget.reserve(name,value)
            try:path=fit_path(prefix["x"],prefix["state"],epochs=spec["epochs"],checkpoints=spec["checkpoints"],defense=spec["defense"])
            except TrainingFailure as error:
                partial=error.partial;write(dest/"FAILED_TRAINING_PATH.json",partial)
                updates=partial["completed_state_updates"];full,remainder=divmod(updates,18)
                budget.add("em_state_updates",updates);budget.add("em_row_visits",full*N+sum(partial["state_sample_counts"][:remainder]))
                budget.add("training_loglikelihood_values",len(partial["loglikelihood_trace"])*N)
                raise
            write(dest/"TRAINING_PATH.json",path)
            for name,value in costs.items():budget.add(name,value)
            budget.check()
            budget.add("validation_density_values",2*len(spec["checkpoints"])*spec["validationN"])
            selection=select_checkpoint(path,valid["x"],valid["state"]);write(dest/"SELECTION.json",selection)
            pair={"rep":rep,"N":N,"selected_epoch":selection["selected_epoch"],"defense":selection["defense"],"raw_parameters":selection["raw_parameters"],"proposals":{}}
            for kind in ("raw","sym"):
                proposal=LocalProposal(selection["raw_parameters"],kind=kind,defense=selection["defense"],source={"rep":rep,"N":N,"epoch":selection["selected_epoch"],"path":relative(dest/"SELECTION.json"),"sha256":sha(dest/"SELECTION.json")})
                p=dest/f"PROPOSAL_{kind}.json";write(p,proposal.to_record());pair["proposals"][kind]={"path":relative(p),"sha256":sha(p)}
            write(dest/"COMPLETE.json",pair);pairs.append(pair);budget.check()
        write(folder/"COMPLETE.json",{"rep":rep,"pairs":[p for p in pairs if p["rep"]==rep],"budget":budget.snapshot()})
    write(out/"FROZEN_PROPOSALS.json",{"schema":SCHEMA,"all_training_complete":True,"pairs":pairs,"before_any_coefficients_or_candidate_comparisons":True})
    return {"pairs":pairs,"new_local_labeled_rows":3*(spec["highN"]+spec["validationN"])}


def proposal_bank(cfg,budget,pair,kind,dest):
    dest.mkdir(exist_ok=False);q=cfg["randomization"];R,B,K=q["R"],q["B"],cfg["K"]
    budget.reserve("qmc_uniform_scalars",R*B*K*6);budget.reserve("qmc_nets",R)
    budget.reserve("posterior_coordinates",R*B*K*3*18)
    u=np.full((R,B,K,3,2),np.nan);states=[]
    try:
        for randomization in range(R):
            entropy=[q["family"],q["role"],pair["rep"],pair["N"],0 if kind=="raw" else 1,randomization]
            rng=np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(entropy)))
            net=qmc.Sobol(d=6*K,scramble=True,bits=q["bits"],rng=rng)
            u[randomization]=net.random_base2(int(np.log2(B))).reshape(B,K,3,2)
            budget.add("qmc_uniform_scalars",B*K*6);budget.add("qmc_nets",1)
            states.append({"randomization":randomization,"entropy":entropy,"rng_after":rng.bit_generator.state})
            require(np.all(u[randomization]>0) and np.all(u[randomization]<1),"noninterior Sobol uniform; no clipping or reseed")
        npz(dest/"UNIFORMS.npz",{"uniforms":u});write(dest/"RNG.json",{"states":states,"R":R,"B":B,"K":K,"kind":kind,"family":q["family"],"role":q["role"]})
        proposal=LocalProposal(pair["raw_parameters"],kind=kind,defense=pair["defense"],source={"rep":pair["rep"],"N":pair["N"],"selected_epoch":pair["selected_epoch"]})
        x=proposal.sample_from_uniforms(u);npz(dest/"SAMPLES.npz",{"x":x});budget.add("posterior_coordinates",x.size);budget.check()
        for name,value in {"proposal_density_values":x.size,"local_likelihood_x":x.size,"local_private_node_terms":x.size*256}.items():
            budget.reserve(name,value)
        local=objective.local_weights(proposal,x,inner_order=256,budget=budget);npz(dest/"LOCAL_WEIGHTS.npz",{"local":local})
        moments=proposal.moments();write(dest/"ANALYTIC_PROPOSAL_MOMENTS.json",moments)
        qm,qs=(np.tile(moments[name],(3,1)) for name in ("mean","second_moment"))
        write(dest/"COMPLETE.json",{"proposal":proposal.to_record(),"budget":budget.snapshot()})
        budget.check();return x,local,qm,qs
    except Exception:
        if not (dest/"UNIFORMS.npz").exists():npz(dest/"FAILED_PARTIAL_UNIFORMS.npz",{"uniforms":u})
        write(dest/"FAILED.json",{"traceback":traceback.format_exc(),"states":states,"budget":budget.snapshot(),"automatic_retry":False})
        raise


def paired_prefix(banks):
    B=banks.shape[1];prefix=objective.stats(banks[:,:B//2]);full=objective.stats(banks)
    delta=full["replicates"]-prefix["replicates"]
    return {"full":full,"prefix":prefix,"paired_full_minus_prefix":{"replicates":delta,"mean":delta.mean(axis=0),
        "SE":delta.std(axis=0,ddof=1)/np.sqrt(len(delta)),"R":len(delta),"same_complete_randomizations":True}}


def scored_phase(cfg,budget,out):
    qa_index=read(resolve(cfg["qa_index"]));require(qa_index.get("passed") is True,"successful QA required")
    origin=load_bundle(qa_index["population_origin"],cfg)
    reference_level=load_bundle(qa_index["levels"]["384"],cfg)
    frozen=read(resolve(cfg["proposal_freeze"]));outputs=[]
    for pair in frozen["pairs"]:
        if pair["rep"] not in cfg["case_ids"] or cfg["phase"]=="profile" and pair["N"]!=16384:continue
        for kind in (("sym",) if cfg["phase"]=="profile" else ("raw","sym")):
            folder=out/f"rep_{pair['rep']}_N{pair['N']}_{kind}";folder.mkdir()
            if cfg["phase"]=="coefficients":
                proposal=LocalProposal(pair["raw_parameters"],kind=kind,defense=pair["defense"])
                diagnostic=objective.density_metrics(proposal,reference_level["normalizer"],orders=(256,384),budget=budget)
                write(folder/"DENSITY_DIAGNOSTICS.json",diagnostic)
            x,local,qm,qs=proposal_bank(cfg,budget,pair,kind,folder/"BANK")
            if cfg["phase"] in ("profile","coefficients"):
                started=time.perf_counter()
                result=objective.origin_coefficients(x,local,origin["state_indices"],origin["weights"],origin["teacher_score"],qm,qs,pattern_chunk=cfg["pattern_chunk"],budget=budget)
                npz(folder/"COEFFICIENT_BANKS.npz",result["banks"])
                coefficient_metadata={k:v for k,v in result.items() if k!="banks"}
                coefficient_metadata.update(kind=kind,
                    symmetry_structure={"mirror_state_antithetic_samples":kind=="sym",
                        "complete_observation_state_enumeration":18**3,
                        "origin_zero_terms_not_independent_numerical_validation":kind=="sym",
                        "explanation":"For sym, x_J=-x_s with complete18^3 weighted state enumeration imposes bankwise reflection cancellation of g and off-diagonal H/B in exact arithmetic; floating-point and QA errors remain. These zero terms are structural identities, not independent validation." if kind=="sym" else "Raw density and samples do not impose state reflection; all matrix/vector entries are evaluated.",
                        "h_b_and_candidate_contrasts_not_assigned_old_theory_values":True})
                write(folder/"COEFFICIENT_STATISTICS.json",coefficient_metadata)
                write(folder/"PREFIX_COMPARISON.json",{k:paired_prefix(v) for k,v in result["banks"].items()})
                record={"rep":pair["rep"],"N":pair["N"],"kind":kind,"statistics":relative(folder/"COEFFICIENT_STATISTICS.json"),"statistics_sha256":sha(folder/"COEFFICIENT_STATISTICS.json"),
                    "banks":relative(folder/"COEFFICIENT_BANKS.npz"),"banks_sha256":sha(folder/"COEFFICIENT_BANKS.npz"),"score_seconds":time.perf_counter()-started,
                    "role":"resource_development_not_primary_statistics" if cfg["phase"]=="profile" else "independent_coefficient_evaluation"}
            else:
                contrasts=[]
                for t in cfg["sym_t"] if kind=="sym" else cfg["raw_t"]:
                    teacher=load_bundle(qa_index["teachers"][str(abs(t))]["36"],cfg)
                    pmf=teacher["joint_pmf"] if t>0 else teacher["joint_pmf"][::-1,:,:]
                    result=objective.candidate_difference(x,local,origin["state_indices"],pmf.ravel(),qm,qs,t=t,alpha=cfg["alpha"],pattern_chunk=cfg["pattern_chunk"],budget=budget)
                    prefix=f"CONTRAST_t{str(t).replace('-','m').replace('.','p')}"
                    npz(folder/(prefix+".npz"),{"banks":result["banks"],"raw_banks":result["raw_banks"]})
                    saved={k:v for k,v in result.items() if k not in ("banks","raw_banks")}
                    saved["prefix_comparison"]=paired_prefix(result["banks"])
                    if kind=="sym":
                        reflection_x=float(np.max(abs(x+x[...,::-1])))
                        reflection_local=float(np.max(abs(local-local[...,::-1])))
                        require(reflection_x<=cfg["reflection_identity_tolerance"] and reflection_local<=cfg["reflection_identity_tolerance"],"sym reflection identity failed")
                        saved["negative_chain_identity"]={"negative_t":-t,"constructed_from_positive_contrast":True,"additional_model_calls":0,"additional_environment":False,"sample_reflection_error":reflection_x,"local_weight_reflection_error":reflection_local}
                    write(folder/(prefix+".json"),saved)
                    contrasts.append({"t":t,"statistics":relative(folder/(prefix+".json")),"statistics_sha256":sha(folder/(prefix+".json")),"banks":relative(folder/(prefix+".npz")),"banks_sha256":sha(folder/(prefix+".npz"))})
                record={"rep":pair["rep"],"N":pair["N"],"kind":kind,"contrasts":contrasts}
            write(folder/"COMPLETE.json",{**record,"budget":budget.snapshot()});outputs.append(record);budget.check()
            print(json.dumps({"phase":cfg["phase"],"rep":pair["rep"],"N":pair["N"],"kind":kind,"status":"completed"}),flush=True)
    expected=1 if cfg["phase"]=="profile" else 4*len(cfg["case_ids"])
    require(len(outputs)==expected,"complete fixed proposal schedule required")
    if cfg["phase"]=="coefficients":write(out/"FROZEN_COEFFICIENT_OUTPUTS.json",{"outputs":outputs,"all_shard_outputs_complete":True,"not_global_freeze":True,"before_validation_randomization":True})
    return {"outputs":outputs}


def run(contract_path,out):
    cfg,budget=freeze(contract_path,out)
    try:
        result=qa(cfg,budget,out) if cfg["phase"]=="qa" else training(cfg,budget,out) if cfg["phase"]=="training" else scored_phase(cfg,budget,out)
        report={"schema":SCHEMA,"phase":cfg["phase"],"status":"completed",**result,"budget":budget.snapshot(),"automatic_followup":False}
        write(out/"SUMMARY.json",report);manifest(out);return report
    except Exception:
        write(out/"FAILED.json",{"schema":SCHEMA,"phase":cfg["phase"],"traceback":traceback.format_exc(),"budget":budget.snapshot(),"automatic_retry":False})
        manifest(out);raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract",type=Path,required=True);parser.add_argument("--out",type=Path);parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    if args.check:
        validate(read(args.contract));print("source/input/schema identities pass; no science executed")
    else:
        if args.out is None:parser.error("--out required for execution")
        run(args.contract.resolve(),args.out.resolve())
