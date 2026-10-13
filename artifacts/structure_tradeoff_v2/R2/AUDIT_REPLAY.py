"""Post-execution read-only audit and readable tables; no RNG draws or model evaluation."""
from pathlib import Path
import csv, hashlib, json, math
from datetime import datetime, timezone
import numpy as np
from scipy.stats import t
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
def read(p): return json.loads(p.read_text(encoding="utf-8"))
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p, x):
    with p.open("x", encoding="utf-8", newline="\n") as f: f.write(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
def check(ok, msg):
    if not ok: raise AssertionError(msg)
cfg = read(OUT/"CONFIG.json")
hash_count = 0
for receipt in ["FREEZE.json", "RESULTS_FREEZE.json", "ORIGINAL_R_RECEIPT.json"]:
    for name, expected in read(OUT/receipt)["files"].items():
        check(digest(ROOT/name)==expected, name); hash_count += 1
check(read(OUT/"STATE.json")["status"]=="complete", "state")
check(len(list((OUT/"reference").glob("*.json")))==6, "reference count")
check(len(list((OUT/"streams").glob("*.json")))==48, "stream count")
check(len(list((OUT/"local").glob("*.npz")))==48, "array count")
check(not list(OUT.glob("FAILURE_*.json")), "R2 failures")
references = {}; stream_data = {}; table = []; maxima = dict(array_mean_error=0., weighted_mean_error=0., pair_identity_error=0., reference_F_error=0., statistical_replay_error=0.)
for d in cfg["d"]:
    m = d//2
    pairs = [(a,b) for a in range(m+1) for b in range(m+1)]
    logmultiplicity = np.log([math.comb(m,a)*math.comb(m,b) for a,b in pairs])
    for profile in cfg["profiles"]:
        r = read(OUT/"reference"/f"{profile}_d{d}.json")
        check(r["numerical_pass"], "reference qualification")
        p = np.array(r["mean"])
        H = -float(np.dot(p,np.log(p)-logmultiplicity))
        pn = np.bincount([a+b for a,b in pairs],weights=p,minlength=d+1)
        ps = np.array([pn[a+b]*math.comb(m,a)*math.comb(m,b)/math.comb(d,a+b) for a,b in pairs])
        D = float(np.dot(p,np.log(p/ps)))
        check(abs(H-r["H"])<2e-12 and abs(D-r["model_KL_single"])<2e-12, "reference functional replay")
        references[(d,profile)] = r
        table.append(dict(d=d,profile=profile,H=r["H"],model_KL_single=r["model_KL_single"],pairwise_L1=r["pairwise_L1"],
                          independent_moment_max_error=r["independent_moment_max_error"],elapsed_seconds=r["elapsed_seconds"]))
        for stream in range(8):
            record = read(OUT/"streams"/f"{profile}_d{d}_s{stream}.json")
            npzpath = OUT/"local"/f"{profile}_d{d}_s{stream}.npz"
            check(digest(npzpath)==record["array_sha256"],"array hash")
            check(record["numerical_pass"],"stream qualification")
            z=np.load(npzpath,allow_pickle=False)
            losses=z["losses"];weights=z["mean_log_pattern_loss"]
            check(losses.shape==(256,3,4) and weights.shape==(3,3,(m+1)**2),"array shape")
            for key in z.files: check(np.isfinite(z[key]).all(),"nonfinite array")
            avg=losses.mean(0)
            e=float(np.max(abs(avg-np.array(record["means"]))))
            maxima["array_mean_error"]=max(maxima["array_mean_error"],e)
            recomputed=np.einsum("kqm,m->kq",weights,p)-H
            maxima["weighted_mean_error"]=max(maxima["weighted_mean_error"],float(np.max(abs(recomputed-avg[:,:3]))))
            maxima["pair_identity_error"]=max(maxima["pair_identity_error"],float(np.max(abs(losses[:,:,3]-(losses[:,:,2]-losses[:,:,1])))))
            refF=np.stack([np.einsum("kqm,m->kq",weights,v) for v in np.array(r["variants"])])
            maxima["reference_F_error"]=max(maxima["reference_F_error"],float(np.max(abs(refF-z["reference_F"]))))
            check(record["seed"]==cfg["bank_seed"]+10000*d+100*list(cfg["profiles"]).index(profile)+stream,"seed")
            expected_initial=np.random.PCG64(record["seed"]).state
            check(record["initial_RNG"]==expected_initial,"initial RNG state")
            check(record["final_RNG"]!=record["initial_RNG"],"RNG did not advance")
            check(record["bank_count"]==256 and record["prior_particles"]==25600,"bank counts")
            stream_data[(d,profile,stream)]=avg
rows=list(csv.DictReader((OUT/"PRIMARY.csv").open(encoding="utf-8")))
streams=list(csv.DictReader((OUT/"STREAM_SCORES.csv").open(encoding="utf-8")))
check(len(rows)==72 and len(streams)==576,"public row counts")
names=["raw_excess","group_excess","single_excess","single_minus_group"]
critical=float(t.ppf(1-.05/(2*72),7)); failed=[]
for row in rows:
    d=int(row["d"]);k=int(row["K"]);q=names.index(row["reading"]);ki=cfg["K"].index(k);profile=row["profile"]
    x=[float(stream_data[(d,profile,s)][ki,q]) for s in range(8)]
    mean=math.fsum(x)/8
    se=math.sqrt(math.fsum((v-mean)**2 for v in x)/7/8)
    want=[mean,se,mean-critical*se,mean+critical*se]
    err=max(abs(float(row[key])-value) for key,value in zip(["mean","SE","lower","upper"],want))
    maxima["statistical_replay_error"]=max(maxima["statistical_replay_error"],err)
    gate=1. if q==0 else (.005 if q==3 else .01)
    check((row["precision_pass"]=="True")==(se<=gate),"precision rule")
    if se>gate: failed.append({k:row[k] for k in ["d","profile","K","reading","SE"]})
for row in streams:
    actual=stream_data[(int(row["d"]),row["profile"],int(row["stream"]))][cfg["K"].index(int(row["K"])),names.index(row["reading"])]
    check(abs(actual-float(row["value"]))<2e-12,"stream score CSV")
check(max(maxima.values())<2e-12,"replay tolerance")
decision=read(OUT/"DECISION.json")
check(decision["precision_pass"]==72-len(failed),"decision count")
for endpoint in decision["primary"]:
    row=next(r for r in rows if r["d"]=="16" and r["K"]=="10" and r["profile"]==endpoint["profile"] and r["reading"]=="single_minus_group")
    check(endpoint["wrong_symmetry_preferred"]==((row["precision_pass"]=="True") and float(row["upper"])<-.001),"decision replay")
audit=dict(time_utc=datetime.now(timezone.utc).isoformat(),passed=True,hash_checks=hash_count,array_hash_checks=48,
           reference_count=6,stream_count=48,statistical_rows=72,maxima=maxima,precision_pass=72-len(failed),
           precision_failures=failed,no_new_random_draws=True,no_new_scientific_observations_or_fits=True,
           audit_source_sha256=digest(Path(__file__)),immutable_experiment_freeze_sha256=digest(OUT/"FREEZE.json"),
           results_freeze_sha256=digest(OUT/"RESULTS_FREEZE.json"))
dump(OUT/"POSTHOC_AUDIT.json",audit)
lines=["# R2 完整可读结果","",
"R2 更换未收敛的参考积分算法，原 R 失败和 0 bank 完整保留。新包按原科学设计完成 48 流、12,288 bank、1,228,800 prior 粒子；0 拟合、0 新观察标签。","",
"6/6 参考和 48/48 bank 数值资格门通过；72 项统计精度门中 60 通过，12 失败。两个事前主终点 d16/K10 的 mild 与 contrast 均通过 −.001 门。区间是 72 项 Bonferroni 同时95% t bank积分区间；参考敏感性另列，不宣称全误差严格95%。","",
"| 主终点 | single−group | 同时区间 | 模型 D(P‖S(P)) |","|---|---:|---:|---:|"]
for e in decision["primary"]:
    r=next(x for x in rows if x["d"]=="16" and x["K"]=="10" and x["profile"]==e["profile"] and x["reading"]=="single_minus_group")
    lines.append(f"| {e['profile']} | {float(r['mean']):.9f} | [{float(r['lower']):.9f}, {float(r['upper']):.9f}] | {e['model_KL_reference']:.10f} |")
lines += ["","这表明此固定点比较中，单count错误对称化的模型损失小于其有限评分收益。mild的模型错配约8.04e−5，contrast约.00192；不能把约.128评分差称为语义收益或全部来自错误对称化。null同样有大幅降方差收益。不能由此推出训练偏好、全局最优、一般非交换模型算法或自然任务改善。","",
"null模型D理论上为0，下面保留浮点正负残差，未裁剪为零。12个统计精度失败全部是两维度、三profile的K1 group_excess及single_minus_group，不将这些格列为方向性成功。","",
"## 参考积分全表","",
"| d | profile | H(P) | D(P‖S(P)) | 三法最大L1差 | 独立矩最大差 | 秒 |","|---:|---|---:|---:|---:|---:|---:|"]
for r in table:
    lines.append(f"| {r['d']} | {r['profile']} | {r['H']:.10f} | {r['model_KL_single']:.12g} | {r['pairwise_L1']:.3g} | {r['independent_moment_max_error']:.3g} | {r['elapsed_seconds']:.3f} |")
lines += ["","GH256/GH512与adaptive为三种确定性算法，不能充当随机重复。各参考完整PMF、原始质量、尾界、quad误差估计、Hermite系数与二阶矩见reference/*.json；主参考预先固定为adaptive。所有标记只表示数值qualified，非全求积/舍入严格认证。","",
"## 全部72项","",
"| d | profile | K | 读数 | 均值 | SE | 下界 | 上界 | 精度 |","|---:|---|---:|---|---:|---:|---:|---:|---|"]
for r in rows:
    lines.append(f"| {r['d']} | {r['profile']} | {r['K']} | {r['reading']} | {float(r['mean']):.9f} | {float(r['SE']):.9f} | {float(r['lower']):.9f} | {float(r['upper']):.9f} | {'通过' if r['precision_pass']=='True' else '失败'} |")
lines += ["","## 复核与来源","",
f"事前FREEZE SHA256：{digest(OUT/'FREEZE.json')}。结果RESULTS_FREEZE SHA256：{digest(OUT/'RESULTS_FREEZE.json')}。","",
f"事后只读审计检查 {hash_count} 项收据哈希和48个本地数组哈希，独立重算全部流均值、模式加权均值、配对身份和72项均值/SE/区间，最大差 {max(maxima.values()):.3g}。未重启科学运行或生成新随机样本。详见POSTHOC_AUDIT.json和公开AUDIT_REPLAY.py。","",
"逐bank数组保存在local/并由各流JSON记录哈希；不提交巨大本地数组。参考方法变体不是16个scramble；不会给其虚构SE。原R的首个L1失败、0bank和冻结哈希见ORIGINAL_R_RECEIPT.json。"]
with (OUT/"READABLE.md").open("x",encoding="utf-8",newline="\n") as f:f.write("\n".join(lines)+"\n")
public=[p for p in OUT.rglob("*") if p.is_file() and "local" not in p.relative_to(OUT).parts and p.name!="MANIFEST.json" and "__pycache__" not in p.parts]
dump(OUT/"MANIFEST.json",dict(time_utc=datetime.now(timezone.utc).isoformat(),kind="post_execution_public_delivery_not_new_science_freeze",
                            files={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(public)},
                            local_arrays_excluded=True,local_hash_receipts_in_streams=True))
print(json.dumps(dict(passed=True,precision_pass=72-len(failed),failures=len(failed),maxima=maxima,manifest_sha256=digest(OUT/"MANIFEST.json")),ensure_ascii=False))

